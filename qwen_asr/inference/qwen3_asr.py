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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

from .qwen3_forced_aligner import Qwen3ForcedAligner
from .utils import (
    MAX_ASR_INPUT_SECONDS,
    MAX_FORCE_ALIGN_INPUT_SECONDS,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    AudioChunk,
    AudioLike,
    chunk_list,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)

try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", Qwen3ASRForConditionalGeneration)
except:
    pass


@dataclass
class ASRTranscription:
    """
    一条 ASR 结果。

    Attributes:
        language (str):
            该音频的语种结果。
            如果整段音频只有一种语言，可能是 `"Chinese"`；
            如果不同 chunk 被识别成多种语言，可能是 `"Chinese,English"`。
            如果无法判断语种或音频基本无有效语音，则可能为空字符串。
        text (str):
            转写文本。
        time_stamps (Optional[Any]):
            时间戳结果，来自 forced aligner。
            只有在 `return_time_stamps=True` 时才会有值。
    """
    language: str
    text: str
    time_stamps: Optional[Any] = None


@dataclass
class ASRStreamingState:
    """
    单条流式语音识别任务的状态对象。

    它保存了流式识别过程中需要持续维护的上下文，
    比如已经缓存了多少音频、之前识别出的文本、当前 prompt 等。
    调用方通常这样使用：

    1. 先用 `init_streaming_state()` 初始化
    2. 多次调用 `streaming_transcribe()` 持续喂入音频
    3. 最后用 `finish_streaming_transcribe()` 冲刷尾部音频

    Attributes:
        unfixed_chunk_num (int):
            前 N 个 chunk 不使用历史识别结果做 prefix。
            这样可以减少刚开始时错误前缀对后续生成的干扰。
        unfixed_token_num (int):
            当 chunk 数已经超过 `unfixed_chunk_num` 后，
            在把历史文本作为 prefix 继续提示模型之前，会先回退最后 K 个 token。
            这样可以减少 chunk 边界处的抖动和重复。
        chunk_size_sec (float):
            每个流式 chunk 的时长，单位秒。
        chunk_size_samples (int):
            按 16kHz 采样率换算后的 chunk 采样点数。
        chunk_id (int):
            当前已经处理到第几个 chunk，从 0 开始。
        buffer (np.ndarray):
            当前缓冲区里尚未凑满一个 chunk 的 PCM 音频。
        audio_accum (np.ndarray):
            从流开始到当前时刻为止累计的所有音频，不做 padding。
        prompt_raw (str):
            由 chat template 生成的基础 prompt。
            这里只包含系统提示和生成起点，不包含后续追加的 prefix 文本。
        context (str):
            用户传入的上下文文本。
        force_language (Optional[str]):
            如果指定了语言，会在 prompt 中追加
            `"language X<asr_text>"`，要求模型只输出该语言的转写文本。
        language (str):
            最近一次解析出的语种结果。
        text (str):
            最近一次解析出的转写文本。
        _raw_decoded (str):
            内部维护的原始生成文本。
            这里还没有经过 `parse_asr_output()` 的格式化解析，
            主要用于回退 token 和构造下一轮 prefix。
    """
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int

    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray

    prompt_raw: str
    context: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str


class Qwen3ASRModel:
    """
    Qwen3-ASR 的统一推理封装。

    这个类把两种后端统一成同一套接口：
    - Transformers 后端
    - vLLM 后端

    另外它还可以在识别后接上 `Qwen3-ForcedAligner`，
    从而返回时间戳结果。

    你可以把它理解成“用户真正调用的外层 API”：
    - 加载模型
    - 处理批量输入
    - 做长音频切块
    - 调后端执行识别
    - 解析输出格式
    - 可选地做时间戳对齐

    说明：
    - 每条请求本质上都是“上下文文本 + 一段音频”
    - 如果显式指定 `language`，prompt 会被强制改成只输出该语种文本
    """

    def __init__(
        self,
        backend: str,
        model: Any,
        processor: Any,
        sampling_params: Optional[Any] = None,
        forced_aligner: Optional[Qwen3ForcedAligner] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: int = 512,
    ):
        self.backend = backend  # "transformers" | "vllm"
        self.model = model
        self.processor = processor
        self.sampling_params = sampling_params
        self.forced_aligner = forced_aligner
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = max_new_tokens

        if backend == "transformers":
            self.device = getattr(model, "device", None)
            if self.device is None:
                try:
                    self.device = next(model.parameters()).device
                except StopIteration:
                    self.device = torch.device("cpu")
            self.dtype = getattr(model, "dtype", torch.float32)
        else:
            self.device = None
            self.dtype = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: Optional[int] = 512,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        使用 Transformers 后端初始化 `Qwen3ASRModel`。

        Args:
            pretrained_model_name_or_path:
                Hugging Face 模型名，或者本地模型目录。
            forced_aligner:
                可选的 forced aligner 模型路径或 repo id。
            forced_aligner_kwargs:
                传给 `Qwen3ForcedAligner.from_pretrained(...)` 的额外参数。
            max_inference_batch_size:
                推理批大小上限。
                `-1` 表示不手动分 batch。
                显存紧张时，可以把它调小以避免 OOM。
            max_new_tokens:
                单次生成最多输出多少个 token。
            **kwargs:
                其他参数会继续传给 `AutoModel.from_pretrained(...)`。

        Returns:
            初始化好的 `Qwen3ASRModel` 实例。
        """

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)

        forced_aligner_model = None
        if forced_aligner is not None:
            forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                forced_aligner, **(forced_aligner_kwargs or {})
            )

        return cls(
            backend="transformers",
            model=model,
            processor=processor,
            sampling_params=None,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    @classmethod
    def LLM(
        cls,
        model: str,
        forced_aligner: Optional[str] = None,
        forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: Optional[int] = 4096,
        **kwargs,
    ) -> "Qwen3ASRModel":
        """
        使用 vLLM 后端初始化 `Qwen3ASRModel`。

        这里把 vLLM 的 import 放在函数内部，
        这样没安装 vLLM 时，普通 Transformers 路径仍然可用。

        Args:
            model:
                vLLM 使用的模型路径或 repo id。
            forced_aligner:
                可选的 forced aligner 模型路径或 repo id。
            forced_aligner_kwargs:
                传给 `Qwen3ForcedAligner.from_pretrained(...)` 的额外参数。
            max_inference_batch_size:
                推理批大小上限。
            max_new_tokens:
                单次生成最多输出多少个 token。
            **kwargs:
                其他参数会继续传给 `vllm.LLM(...)`。

        Returns:
            初始化好的 `Qwen3ASRModel` 实例。

        Raises:
            ImportError:
                如果当前环境没有安装 vLLM。
        """
        try:
            from vllm import LLM as vLLM
            from vllm import SamplingParams
        except Exception as e:
            raise ImportError(
                "vLLM is not available. Install with: pip install qwen-asr[vllm]"
            ) from e

        llm = vLLM(model=model, **kwargs)

        processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
        sampling_params = SamplingParams(**({"temperature": 0.0, "max_tokens": max_new_tokens}))

        forced_aligner_model = None
        if forced_aligner is not None:
            forced_aligner_model = Qwen3ForcedAligner.from_pretrained(
                forced_aligner, **(forced_aligner_kwargs or {})
            )

        return cls(
            backend="vllm",
            model=llm,
            processor=processor,
            sampling_params=sampling_params,
            forced_aligner=forced_aligner_model,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=None,
        )

    def get_supported_languages(self) -> List[str]:
        """
        返回当前模型支持的语种列表。

        Returns:
            List[str]:
                标准化后的语种名列表。
        """
        return list(SUPPORTED_LANGUAGES)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscription]:
        """
        对输入音频做转写，可选上下文和时间戳。

        这是最常用的离线识别入口。
        它内部会做这些事：
        1. 把输入音频统一规范化
        2. 如果音频太长，就自动切成多个 chunk
        3. 逐 chunk 调后端做 ASR
        4. 把模型原始输出解析成 `language` 和 `text`
        5. 如果要求时间戳，再调用 forced aligner
        6. 最后把各个 chunk 的结果重新拼回原音频级别

        Args:
            audio:
                音频输入。支持：
                - 本地路径
                - URL
                - base64
                - `(np.ndarray, sr)`
                - 上述格式组成的列表
            context:
                上下文文本。
                如果只传一个字符串而音频是 batch，会自动广播到每条样本。
            language:
                可选的强制语种。
                如果传入，则必须在支持语种列表中。
                如果只传一个值而音频是 batch，会自动广播到每条样本。
                指定后，prompt 会要求模型输出该语种下的纯转写文本。
            return_time_stamps:
                是否返回时间戳。
                如果为 True，需要初始化时已经传入 `forced_aligner`。

        Returns:
            List[ASRTranscription]:
                每条输入音频对应一条结果。

        Raises:
            ValueError:
                - 要求时间戳但没有初始化 `forced_aligner`
                - 指定了不支持的语言
                - `audio / context / language` 的 batch 数量对不上
        """
        if return_time_stamps and self.forced_aligner is None:
            raise ValueError("return_time_stamps=True requires `forced_aligner` to be provided at initialization.")

        wavs = normalize_audios(audio)
        n = len(wavs)

        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")

        langs_in: List[Optional[str]]
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(f"Batch size mismatch: audio={n}, language={len(langs_in)}")

        langs_norm: List[Optional[str]] = []
        for l in langs_in:
            if l is None or str(l).strip() == "":
                langs_norm.append(None)
            else:
                ln = normalize_language_name(str(l))
                validate_language(ln)
                langs_norm.append(ln)

        max_chunk_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS

        # chunk audios and record mapping
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(
                wav=wav,
                sr=SAMPLE_RATE,
                max_chunk_sec=max_chunk_sec,
            )
            for j, (cwav, offset_sec) in enumerate(parts):
                chunks.append(AudioChunk(orig_index=i, chunk_index=j, wav=cwav, sr=SAMPLE_RATE, offset_sec=offset_sec))

        # run ASR on chunks
        chunk_ctx: List[str] = [ctxs[c.orig_index] for c in chunks]
        chunk_lang: List[Optional[str]] = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs: List[np.ndarray] = [c.wav for c in chunks]
        raw_outputs = self._infer_asr(chunk_ctx, chunk_wavs, chunk_lang)

        # parse outputs, prepare for optional alignment
        per_chunk_lang: List[str] = []
        per_chunk_text: List[str] = []
        for out, forced_lang in zip(raw_outputs, chunk_lang):
            lang, txt = parse_asr_output(out, user_language=forced_lang)
            per_chunk_lang.append(lang)
            per_chunk_text.append(txt)

        # forced alignment (optional)
        per_chunk_align: List[Optional[Any]] = [None] * len(chunks)
        if return_time_stamps:
            to_align_audio = []
            to_align_text = []
            to_align_lang = []
            to_align_idx = []

            for idx, (c, txt, lang_pred) in enumerate(zip(chunks, per_chunk_text, per_chunk_lang)):
                if txt.strip() == "":
                    continue
                to_align_audio.append((c.wav, c.sr))
                to_align_text.append(txt)
                to_align_lang.append(lang_pred)
                to_align_idx.append(idx)

            # batch align with max_inference_batch_size
            aligned_results: List[Any] = []
            for a_chunk, t_chunk, l_chunk in zip(
                chunk_list(to_align_audio, self.max_inference_batch_size),
                chunk_list(to_align_text, self.max_inference_batch_size),
                chunk_list(to_align_lang, self.max_inference_batch_size),
            ):
                aligned_results.extend(
                    self.forced_aligner.align(audio=a_chunk, text=t_chunk, language=l_chunk)
                )

            # offset fix
            for k, idx in enumerate(to_align_idx):
                c = chunks[idx]
                r = aligned_results[k]
                per_chunk_align[idx] = self._offset_align_result(r, c.offset_sec)

        # merge chunks back to original samples
        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]
        out_aligns: List[List[Any]] = [[] for _ in range(n)]

        for c, lang, txt, al in zip(chunks, per_chunk_lang, per_chunk_text, per_chunk_align):
            out_langs[c.orig_index].append(lang)
            out_texts[c.orig_index].append(txt)
            if return_time_stamps and al is not None:
                out_aligns[c.orig_index].append(al)

        results: List[ASRTranscription] = []
        for i in range(n):
            merged_text = "".join([t for t in out_texts[i] if t is not None])
            merged_language = merge_languages(out_langs[i])
            merged_align = None
            if return_time_stamps:
                merged_align = self._merge_align_results(out_aligns[i])
            results.append(ASRTranscription(language=merged_language, text=merged_text, time_stamps=merged_align))

        return results

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        """
        为单条请求构造文本 prompt。

        这里会先用 chat template 把 system/user/assistant 格式排好，
        然后在需要时追加：

            language X<asr_text>

        这样模型就会被约束成输出指定语种的纯转写文本。
        """
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            base = base + f"language {force_language}{'<asr_text>'}"
        return base

    def _infer_asr(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        """
        对已经切好的 chunk 执行后端推理。

        Args:
            contexts:
                每个 chunk 对应的上下文文本列表。
            wavs:
                每个 chunk 对应的单声道 waveform 列表。
            languages:
                每个 chunk 对应的强制语种，或 `None`。

        Returns:
            List[str]:
                每个 chunk 对应一条模型原始输出文本。
        """
        if self.backend == "transformers":
            return self._infer_asr_transformers(contexts, wavs, languages)
        if self.backend == "vllm":
            return self._infer_asr_vllm(contexts, wavs, languages)
        raise RuntimeError(f"Unknown backend: {self.backend}")

    def _infer_asr_transformers(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        outs: List[str] = []

        texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]

        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(texts)

        for i in range(0, len(texts), batch_size):
            sub_text = texts[i : i + batch_size]
            sub_wavs = wavs[i : i + batch_size]
            inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            decoded = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(list(decoded))

        return outs

    def _infer_asr_vllm(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
    ) -> List[str]:
        inputs: List[Dict[str, Any]] = []
        for c, w, fl in zip(contexts, wavs, languages):
            prompt = self._build_text_prompt(context=c, force_language=fl)
            inputs.append({"prompt": prompt, "multi_modal_data": {"audio": [w]}})

        outs: List[str] = []
        for batch in chunk_list(inputs, self.max_inference_batch_size):
            outputs = self.model.generate(batch, sampling_params=self.sampling_params, use_tqdm=False)
            for o in outputs:
                outs.append(o.outputs[0].text)
        return outs

    def _offset_align_result(self, result: Any, offset_sec: float) -> Any:
        """
        给对齐结果整体加上时间偏移量。

        场景：
        长音频先被切成多个 chunk 后，每个 chunk 单独对齐得到的时间戳
        都是“相对该 chunk 起点”的局部时间。
        这个函数会把它们改成“相对整段原始音频”的全局时间。

        注意：
        上游的 dataclass 可能是 frozen 的，所以这里不是原地改值，
        而是按原类型重新构造一个新对象。

        Args:
            result:
                ForcedAlignResult 风格的对象。
            offset_sec:
                需要加上的偏移量，单位秒。

        Returns:
            新的对齐结果对象，里面的时间戳已经平移过。
        """
        if result is None:
            return None
        items = []
        for it in result.items:
            items.append(type(it)(text=it.text, 
                                  start_time=round(it.start_time + offset_sec, 3), 
                                  end_time=round(it.end_time + offset_sec, 3)))
        return type(result)(items=items)

    def _merge_align_results(self, results: List[Any]) -> Optional[Any]:
        """
        把多个 chunk 的对齐结果拼接成一个整体结果。

        Args:
            results:
                ForcedAlignResult 列表。

        Returns:
            合并后的 ForcedAlignResult；如果没有有效结果则返回 `None`。
        """
        if not results:
            return None
        all_items = []
        for r in results:
            if r is None:
                continue
            all_items.extend(list(r.items))
        if not all_items:
            return None
        return type(results[0])(items=all_items)

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> ASRStreamingState:
        """
        初始化单条流式识别任务的状态。

        说明：
        - 流式识别只支持 vLLM 后端
        - 流式模式不支持时间戳
        - 流式模式不支持 batch

        Args:
            context:
                上下文文本。
            language:
                可选的强制语种。
                行为和 `transcribe()` 一致，会通过 prompt 后缀强制模型输出纯文本转写。
            unfixed_chunk_num:
                前 N 个 chunk 不使用历史输出做 prefix。
            unfixed_token_num:
                之后每次把历史文本当 prefix 时，先回退最后 K 个 token。
            chunk_size_sec:
                每个流式 chunk 的时长，单位秒。

        Returns:
            `ASRStreamingState`：
                后续会传给 `streaming_transcribe()` 和 `finish_streaming_transcribe()`。

        Raises:
            ValueError:
                - 当前后端不是 vLLM
                - `chunk_size_sec <= 0`
                - `language` 非法
        """
        if self.backend != "vllm":
            raise ValueError("Streaming ASR is supported only for vLLM backend (backend='vllm').")
        if chunk_size_sec is None or float(chunk_size_sec) <= 0:
            raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

        force_language = None
        if language is not None and str(language).strip() != "":
            ln = normalize_language_name(str(language))
            validate_language(ln)
            force_language = ln

        chunk_size_samples = int(round(float(chunk_size_sec) * SAMPLE_RATE))
        chunk_size_samples = max(1, chunk_size_samples)

        prompt_raw = self._build_text_prompt(context=context, force_language=force_language)

        return ASRStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=int(chunk_size_samples),
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            prompt_raw=prompt_raw,
            context=context or "",
            force_language=force_language,
            language="",
            text="",
            _raw_decoded="",
        )

    def streaming_transcribe(self, pcm16k: np.ndarray, state: ASRStreamingState) -> ASRStreamingState:
        """
        执行一次流式识别增量解码。

        这个函数接收任意长度的 16k 单声道 PCM 音频。
        它会先把音频追加到缓冲区里；
        当缓冲区里的音频足够凑成一个完整 chunk 时，就触发一次解码，并更新：

        - `state.language`
        - `state.text`

        调用方只需要不断喂入新音频，并从 state 中读取当前结果即可。

        实现细节：
        - 每次 chunk 准备好后，会把它追加到 `audio_accum`
        - 当前实现会把“从流开始到当前为止的全部音频”重新喂给模型
        - prompt 会动态更新为：`state.prompt_raw + prefix_text`
        - prefix 会按回退策略裁掉最后若干 token，减少边界抖动

        说明：
        - 仅支持 vLLM
        - 不支持时间戳
        - 只支持单流，不支持 batch

        Args:
            pcm16k:
                16kHz 单声道 PCM 音频。
                长度可以任意，dtype 可以是 float32 / float64 / int16，
                内部会统一转成 float32。
            state:
                由 `init_streaming_state()` 返回的状态对象。

        Returns:
            更新后的状态对象。

        Raises:
            ValueError:
                如果当前后端不是 vLLM，或者状态非法。
        """
        if self.backend != "vllm":
            raise ValueError("streaming_transcribe() is supported only for vLLM backend (backend='vllm').")
        if state is None:
            raise ValueError("state must not be None. Call init_streaming_state() first.")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        # Ensure 1D mono
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)

        # Convert to float32 PCM in [-1, 1] if int16 provided
        if x.dtype == np.int16:
            x = (x.astype(np.float32) / 32768.0)
        else:
            x = x.astype(np.float32, copy=False)

        # Append to buffer
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        # Consume full chunks
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]

            # Accumulate audio (re-feed from start, no padding)
            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

            # Build prefix with rollback strategy
            prefix = ""
            if state.chunk_id < state.unfixed_chunk_num:
                prefix = ""
            else:
                cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
                k = int(state.unfixed_token_num)
                while True:
                    end_idx = max(0, len(cur_ids) - k)
                    prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                    if '\ufffd' not in prefix:
                        break
                    else:
                        if end_idx == 0:
                            prefix = ""
                            break
                        k += 1

            prompt = state.prompt_raw + prefix

            # vLLM input: single item
            inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

            outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
            gen_text = outputs[0].outputs[0].text

            # Accumulate raw decoded (then parse to lang/text)
            state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text

            lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
            state.language = lang
            state.text = txt

            state.chunk_id += 1

        return state

    def finish_streaming_transcribe(self, state: ASRStreamingState) -> ASRStreamingState:
        """
        结束流式识别并冲刷尾部缓冲音频。

        它会把 `state.buffer` 里剩下的尾部音频也送进模型，
        即便这段尾音频还不足一个完整 chunk，也会进行最后一次识别。
        然后再更新一次 `state.language` 和 `state.text`。

        说明：
        - 仅支持 vLLM
        - 不支持时间戳
        - 只支持单流

        Args:
            state:
                流式状态对象。

        Returns:
            更新后的状态对象。

        Raises:
            ValueError:
                如果当前后端不是 vLLM，或者状态非法。
        """
        if self.backend != "vllm":
            raise ValueError("finish_streaming_transcribe() is supported only for vLLM backend (backend='vllm').")
        if state is None:
            raise ValueError("state must not be None.")

        # If no remaining buffer, still return state as-is.
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        # Append tail to accumulated audio
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

        # Prefix rollback strategy (same as per-chunk)
        prefix = ""
        if state.chunk_id < state.unfixed_chunk_num:
            prefix = ""
        else:
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
            prefix = self.processor.tokenizer.decode(cur_ids[:end_idx])

        prompt = state.prompt_raw + prefix
        inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

        outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
        gen_text = outputs[0].outputs[0].text

        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt

        state.chunk_id += 1
        return state
