# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf

AudioLike = Union[
    str,                      # wav path / URL / base64
    Tuple[np.ndarray, int],   # (waveform, sr)
]
MaybeList = Union[Any, List[Any]]

SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200
MAX_FORCE_ALIGN_INPUT_SECONDS = 180
MIN_ASR_INPUT_SECONDS = 0.5
SUPPORTED_LANGUAGES: List[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian"
]
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "
_TURN_STATE_TAG = "<turn_state>"
_LANG_RE = re.compile(r"language\s+([^\n<]+)", re.IGNORECASE)
_TAIL_LABEL_RE = re.compile(
    r"^(.*?)(<(?:complete|incomplete|backchannel|wait)>)\s*$",
    re.IGNORECASE | re.DOTALL,
)
TURN_LABELS: Tuple[str, ...] = ("complete", "incomplete", "backchannel", "wait")
TURN_LABEL_TOKENS = {f"<{label}>": label for label in TURN_LABELS}
TURN_LABEL_TOKENS_UPPER = {token.upper(): label for token, label in TURN_LABEL_TOKENS.items()}
LABEL_POSITION_MODES: Tuple[str, ...] = ("label_first", "label_last", "auto")


@dataclass
class ParsedJointOutput:
    raw_text: str
    language: str
    turn_label: Optional[str]
    transcript: str
    position_used: str
    has_language_prefix: bool
    has_asr_tag: bool
    has_turn_state_tag: bool
    has_label_token: bool
    strict_schema_valid: bool
    soft_parse_success: bool


def normalize_language_name(language: str) -> str:
    """
    Normalize language name to the canonical format used by Qwen3-ASR:
    first letter uppercase, the rest lowercase (e.g., 'cHINese' -> 'Chinese').

    Args:
        language (str): Input language name.

    Returns:
        str: Normalized language name.

    Raises:
        ValueError: If language is empty.
    """
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def validate_language(language: str) -> None:
    """
    Validate the language is supported.

    Args:
        language (str): Canonical language name.

    Raises:
        ValueError: If unsupported.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


def ensure_list(x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]


def is_url(s: str) -> bool:
    """
    粗略判断一个字符串是否像 HTTP/HTTPS URL。

    这个函数的目的不是做严格的 URL 校验，而是给音频输入分流：
    - 如果像 URL，就走“先下载再解码音频”的路径
    - 否则继续尝试把它当 base64 或本地路径处理

    为什么这里只检查 scheme 和 netloc：
    - 对当前场景来说已经足够
    - 可以避免引入过重的校验逻辑
    - 即使误判，后续加载失败也会在真正读取音频时暴露出来
    """
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def is_probably_base64(s: str) -> bool:
    """
    粗略判断一个字符串是否“看起来像”一段 base64 音频。

    当前采用的是启发式规则，不是严格校验：
    1. 如果是 `data:audio...` 这种 data URI，直接认为是 base64 音频
    2. 如果字符串非常长，而且不像路径（不含 `/` 或 `\\`），也倾向认为它是 base64

    这样设计的原因：
    - 推理入口既支持本地路径，也支持 URL，也支持 base64
    - 需要一个便宜的分流器，先决定大概走哪条读取逻辑
    - 真正的正确性由后面的 `base64.b64decode` 和音频解码保证
    """
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def decode_base64_bytes(b64: str) -> bytes:
    """
    把 base64 字符串解码成原始字节。

    兼容两类输入：
    - 纯 base64 内容
    - `data:audio/...;base64,...` 这种 data URI

    对 data URI，会先把前缀元信息去掉，只保留逗号后面的真正 base64 部分。
    """
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_any(x: str) -> Tuple[np.ndarray, int]:
    """
    统一读取字符串形式的音频输入，返回 `(audio, sr)`。

    支持三类字符串输入：
    - URL
    - base64 音频字符串
    - 本地文件路径

    返回值里的 `audio` 还不保证已经是：
    - 单声道
    - 16kHz
    - [-1, 1] 范围

    这些后处理会在 `normalize_audio_input()` 里继续完成。
    这里的职责只是一件事：把“各种字符串来源”读成波形和采样率。
    """
    if is_url(x):
        # URL 路径：先下载字节，再用 soundfile 从内存里解码。
        # 这里不用先落地临时文件，能减少一次磁盘 IO。
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif is_probably_base64(x):
        # base64 路径：先还原字节，再从内存字节流里解码。
        audio_bytes = decode_base64_bytes(x)
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        # 本地文件路径：直接用 librosa 读取。
        # 这里显式设置 `sr=None`，表示先保留原始采样率，不在这一步重采样。
        # `mono=False` 表示先保留声道信息，后面再统一做单声道转换。
        audio, sr = librosa.load(x, sr=None, mono=False)

    # 统一转成 float32，避免后续不同音频后端返回不同 dtype。
    audio = np.asarray(audio, dtype=np.float32)
    sr = int(sr)
    return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    把音频统一转成单声道 waveform。

    为什么这里要单独做单声道：
    - 模型后续统一按单通道输入处理
    - 不同音频解码后，声道维可能排列方式不同

    支持的情况：
    - `(T,)`：本来就是单声道，直接返回
    - `(T, C)`：常见的 soundfile 输出格式
    - `(C, T)`：某些管线里会采用这种布局

    对二维输入，这里最终使用“多声道求平均”的方式合成单声道。
    这是语音推理里最稳妥、最常见的默认做法。
    """
    if audio.ndim == 1:
        return audio
    # soundfile can return shape (T, C); some pipelines use (C, T)
    if audio.ndim == 2:
        # 如果第一维很小、第二维明显更大，通常更像 `(C, T)`，
        # 这里转置成 `(T, C)` 后再按最后一维求平均。
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    """
    把波形整理成 float32，并尽量规范到 [-1, 1]。

    这一步不是做响度归一化，而是做“数值范围安全化”：
    - 如果本来就是合理 float 波形，基本不会改动
    - 如果解码后数值超出 [-1, 1]，则按峰值做保守缩放
    - 最后再做一次 clip，避免极少数异常值继续泄漏到模型

    为什么只在 `peak > 1.0` 时缩放：
    - 有些输入本来就是标准 float32 波形，范围已在 [-1, 1]
    - 没必要对正常样本重复缩放
    - 这里只处理“明显超范围”的情况
    """
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    # If decoded audio is int-like scaled or out-of-range, normalize conservatively.
    if peak > 1.0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def normalize_audio_input(a: AudioLike) -> np.ndarray:
    """
    把一条音频输入统一规范成：

    - 单声道
    - 16kHz
    - `float32`
    - 幅值范围尽量落在 `[-1, 1]`

    这是推理前最核心的一步“输入标准化”。

    为什么需要统一到这个格式：
    1. 仓库内部默认按 16kHz 处理音频
    2. 多声道音频如果不先合并，会导致后续特征提取接口不一致
    3. 不同数据源可能返回不同 dtype、不同数值范围，不先规整容易让模型输入分布漂移

    Supported inputs:
        - str: local file path / https URL / base64 audio string
        - (np.ndarray, sr): waveform and sampling rate

    Returns:
        np.ndarray:
            Mono 16k float32 waveform in [-1, 1].
    """
    if isinstance(a, str):
        # 字符串输入统一交给 `load_audio_any()`。
        # 它会根据内容自动分流：
        # - URL
        # - base64
        # - 本地路径
        audio, sr = load_audio_any(a)
    elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
        # 直接传 `(waveform, sr)` 时，不再做额外 I/O，
        # 直接取出数组和采样率进入后续标准化流程。
        audio, sr = a[0], int(a[1])
    else:
        raise TypeError(f"Unsupported audio input type: {type(a)}")

    # 第一步：统一转单声道。
    # 这样后续重采样和特征提取只需要处理一维 waveform。
    audio = to_mono(np.asarray(audio))

    # 第二步：如果采样率不是目标采样率 16k，则重采样到 16k。
    # 这里使用 librosa.resample，是因为它对常见语音推理场景足够稳定。
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)

    # 第三步：做 dtype 和数值范围规整。
    # 注意这一步不是“让每条音频听起来一样响”，而是把数值整理到模型期望的安全范围。
    audio = float_range_normalize(audio)

    # 返回的一定是一维、16k、float32 的 waveform。
    return audio


def normalize_audios(audios: Union[AudioLike, List[AudioLike]]) -> List[np.ndarray]:
    """
    批量版本的 `normalize_audio_input()`。

    这里先用 `ensure_list()` 把单条输入也包装成列表，
    然后逐条做标准化，最终统一返回 `List[np.ndarray]`。
    """
    items = ensure_list(audios)
    return [normalize_audio_input(a) for a in items]


def chunk_list(xs: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    """
    Yield chunks of a list.

    Args:
        xs (List[Any]): Input list.
        chunk_size (int): Chunk size.

    Yields:
        List[Any]: Slices of xs.
    """
    if chunk_size <= 0:
        yield xs
        return
    for i in range(0, len(xs), chunk_size):
        yield xs[i : i + chunk_size]


@dataclass(frozen=True)
class AudioChunk:
    """
    One chunk cut from an original audio.

    Attributes:
        orig_index: Index of the original sample in the input batch.
        chunk_index: Index of this chunk within the original sample.
        wav: Mono float32 waveform.
        sr: Sampling rate.
        offset_sec: Start offset of this chunk in the original audio, in seconds.
    """
    orig_index: int
    chunk_index: int
    wav: np.ndarray
    sr: int
    offset_sec: float


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    max_chunk_sec: float,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Split a long audio into chunks close to max_chunk_sec, using a low-energy boundary.

    This implementation guarantees:
      - Concatenating all returned chunks reproduces the original audio exactly
        (total number of samples is identical, no overlaps, no gaps).

    Args:
        wav: Mono waveform float32.
        sr: Sampling rate.
        max_chunk_sec: Target max chunk duration in seconds.
        search_expand_sec: Boundary search half-window in seconds.
        min_window_ms: Sliding window in milliseconds for energy estimation.

    Returns:
        List[Tuple[np.ndarray, float]]: List of (chunk_wav, offset_sec).
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)

    total_len = int(wav.shape[0])
    total_sec = total_len / float(sr)
    if total_sec <= max_chunk_sec:
        return [(wav, 0.0)]

    max_len = int(max_chunk_sec * sr)
    expand = int(search_expand_sec * sr)
    win = max(4, int((min_window_ms / 1000.0) * sr))

    chunks: List[Tuple[np.ndarray, float]] = []

    start = 0
    offset_sec = 0.0

    while (total_len - start) > max_len:
        cut = start + max_len

        left = max(start, cut - expand)
        right = min(total_len, cut + expand)

        if right - left <= win:
            boundary = cut
        else:
            seg = wav[left:right]
            seg_abs = np.abs(seg)

            window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")

            min_pos = int(np.argmin(window_sums))

            wstart = min_pos
            wend = min_pos + win
            local = seg_abs[wstart:wend]
            inner = int(np.argmin(local))
            boundary = left + wstart + inner

        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))

        chunk = wav[start:boundary]
        chunks.append((chunk, offset_sec))

        offset_sec += (boundary - start) / float(sr)
        start = boundary

    tail = wav[start:total_len]
    chunks.append((tail, offset_sec))

    # Pad too-short chunks to at least MIN_ASR_INPUT_SECONDS (zero-padding at tail)
    min_len = int(MIN_ASR_INPUT_SECONDS * sr)
    padded: List[Tuple[np.ndarray, float]] = []
    for c, off in chunks:
        if c.shape[0] < min_len:
            pad = min_len - int(c.shape[0])
            c = np.pad(c, (0, pad), mode="constant", constant_values=0.0).astype(np.float32)
        padded.append((c, off))
    chunks = padded

    return chunks


def detect_and_fix_repetitions(text, threshold=20):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
            
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                    
                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break
                
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            
            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)
    
    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def parse_asr_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Parse Qwen3-ASR raw output into (language, text).

    Cases:
      - With tag: "language Chinese<asr_text>...."
      - With newlines: "language Chinese\\n...\\n<asr_text>...."
      - No tag: treat whole string as text.
      - "language None<asr_text>": treat as empty audio -> ("", "")

    If user_language is provided, language is forced to user_language and raw is treated as text-only
    (the model is expected to output plain transcription without metadata).

    Args:
        raw: Raw decoded string.
        user_language: Canonical language name if user forced language.

    Returns:
        Tuple[str, str]: (language, text)
    """
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = detect_and_fix_repetitions(s)

    if user_language:
        # user explicitly forced language => model output is treated as pure text
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        # no tag => pure text
        return "", s.strip()

    meta_lower = meta_part.lower()

    # empty audio heuristic
    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        # if model still returned something, keep it but language unknown
        return "", t

    # extract "language xxx" from meta
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()


def build_vllm_transcription_prompt(
    audio_placeholder: str,
    request_prompt: Optional[str] = None,
    to_language: Optional[str] = None,
) -> str:
    """
    Build the text prompt used by vLLM transcription requests.

    Behavior:
      - If request_prompt is provided, use the same system/user/assistant chat
        layout as the training pipeline so task-specific prompts can steer the
        output format.
      - Otherwise, preserve the original ASR prompt behavior for compatibility.
    """
    request_prompt = (request_prompt or "").strip()
    if request_prompt:
        return (
            f"<|im_start|>system\n{request_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    if to_language is None:
        return (
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    return (
        f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
        f"<|im_start|>assistant\nlanguage {to_language}<asr_text>"
    )


def normalize_turn_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if not s:
        return None
    if s in TURN_LABELS:
        return s
    if s in TURN_LABEL_TOKENS:
        return TURN_LABEL_TOKENS[s]
    if s.upper() in TURN_LABEL_TOKENS_UPPER:
        return TURN_LABEL_TOKENS_UPPER[s.upper()]
    return None


def extract_joint_language(meta_text: str) -> Tuple[str, bool]:
    m = _LANG_RE.search(meta_text or "")
    if not m:
        return "", False
    raw = (m.group(1) or "").strip()
    if not raw:
        return "", False
    try:
        return normalize_language_name(raw), True
    except Exception:
        return raw, True


def find_turn_label_token(text: str) -> Optional[str]:
    lower = str(text or "").lower()
    for token, label in TURN_LABEL_TOKENS.items():
        if token in lower:
            return label
    return None


def _parse_label_first_joint_output(text: str) -> ParsedJointOutput:
    s = detect_and_fix_repetitions(str(text or "").strip())
    s_lower = s.lower()
    has_asr_tag = _ASR_TEXT_TAG in s_lower
    meta_part = s
    transcript = ""
    if has_asr_tag:
        idx = s_lower.find(_ASR_TEXT_TAG)
        meta_part = s[:idx]
        transcript = s[idx + len(_ASR_TEXT_TAG) :].strip()

    language, has_language_prefix = extract_joint_language(meta_part)
    has_turn_state_tag = _TURN_STATE_TAG in meta_part.lower()
    turn_label = find_turn_label_token(meta_part)
    has_label_token = turn_label is not None
    strict_schema_valid = bool(has_language_prefix and has_asr_tag and has_turn_state_tag and has_label_token)
    soft_parse_success = bool(has_asr_tag and has_label_token)
    return ParsedJointOutput(
        raw_text=s,
        language=language,
        turn_label=turn_label,
        transcript=transcript,
        position_used="label_first",
        has_language_prefix=has_language_prefix,
        has_asr_tag=has_asr_tag,
        has_turn_state_tag=has_turn_state_tag,
        has_label_token=has_label_token,
        strict_schema_valid=strict_schema_valid,
        soft_parse_success=soft_parse_success,
    )


def _parse_label_last_joint_output(text: str) -> ParsedJointOutput:
    s = detect_and_fix_repetitions(str(text or "").strip())
    s_lower = s.lower()
    has_asr_tag = _ASR_TEXT_TAG in s_lower
    meta_part = s
    text_part = ""
    if has_asr_tag:
        idx = s_lower.find(_ASR_TEXT_TAG)
        meta_part = s[:idx]
        text_part = s[idx + len(_ASR_TEXT_TAG) :].strip()

    language, has_language_prefix = extract_joint_language(meta_part)
    has_turn_state_tag = _TURN_STATE_TAG in meta_part.lower()
    turn_label = None
    transcript = text_part
    m = _TAIL_LABEL_RE.match(text_part)
    if m:
        transcript = m.group(1).strip()
        turn_label = normalize_turn_label(m.group(2))
    has_label_token = turn_label is not None
    strict_schema_valid = bool(has_language_prefix and has_asr_tag and has_label_token)
    soft_parse_success = bool(has_asr_tag and has_label_token)
    return ParsedJointOutput(
        raw_text=s,
        language=language,
        turn_label=turn_label,
        transcript=transcript,
        position_used="label_last",
        has_language_prefix=has_language_prefix,
        has_asr_tag=has_asr_tag,
        has_turn_state_tag=has_turn_state_tag,
        has_label_token=has_label_token,
        strict_schema_valid=strict_schema_valid,
        soft_parse_success=soft_parse_success,
    )


def parse_joint_output(text: Any, label_position: str = "label_first") -> ParsedJointOutput:
    label_position = str(label_position).strip().lower()
    if label_position == "label_first":
        return _parse_label_first_joint_output(text)
    if label_position == "label_last":
        return _parse_label_last_joint_output(text)

    first = _parse_label_first_joint_output(text)
    last = _parse_label_last_joint_output(text)

    def _score(parsed: ParsedJointOutput) -> Tuple[int, int, int, int]:
        return (
            1 if parsed.strict_schema_valid else 0,
            1 if parsed.soft_parse_success else 0,
            1 if parsed.has_turn_state_tag else 0,
            len(parsed.transcript),
        )

    return first if _score(first) >= _score(last) else last


def post_process_vllm_transcription_output(text: str) -> str:
    """
    Post-process transcription output conservatively.

    Default ASR outputs in the form "language X<asr_text>..." should still be
    trimmed to plain transcript. But if the model emits a structured joint
    output such as "language X<turn_state><label><asr_text>...", keep the full
    string so downstream code can parse the protocol fields.
    """
    if not text:
        return ""

    text = str(text).strip()
    if _ASR_TEXT_TAG not in text:
        return text

    parsed = parse_joint_output(text, label_position="auto")
    if parsed.has_turn_state_tag and parsed.has_label_token:
        return parsed.raw_text

    _, text_part = text.rsplit(_ASR_TEXT_TAG, 1)
    return text_part.strip()


def merge_languages(langs: List[str]) -> str:
    """
    Merge per-chunk languages into a compact comma-separated string,
    keeping order and removing consecutive duplicates and empty entries.

    Example:
      ["Chinese", "English", "English"] -> "Chinese,English"

    Args:
        langs: List of canonical language names.

    Returns:
        str: Merged language string.
    """
    out: List[str] = []
    prev = None
    for x in langs:
        x = (x or "").strip()
        if not x:
            continue
        if x == prev:
            continue
        out.append(x)
        prev = x
    return ",".join(out)
