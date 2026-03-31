# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight turn detector built on top of Qwen3-ASR hidden states.
#
# The model intentionally solves only the speech-completion problem:
#   - complete
#   - incomplete
# "waiting" is left to the policy/LLM layer after ASR, because it is usually
# a response-control intent rather than an end-of-utterance boundary problem.
"""
这个文件实现了一个“复用 Qwen3-ASR 底座的二分类器”。

可以把它理解成：
    Qwen3-ASR backbone + classifier head

背后的设计思路是：
1. 不重新训练一个音频模型；
2. 直接复用 Qwen3-ASR 已经学到的声学 / 韵律 / 多模态表示；
3. 在顶部接一个轻量分类头，预测 complete / incomplete。

对新手来说，最值得理解的是 `_encode()`：
- 它并没有走文本生成；
- 而是复用了 Qwen3-ASR 的“文本 embedding + 音频 embedding 融合 -> thinker”这条路径；
- 最后从隐藏状态里池化出一个向量做分类。
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from qwen_asr.inference.utils import AudioLike, normalize_audio_input
from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

TURN_LABELS = ("incomplete", "complete")

DEFAULT_TURN_DETECTION_PROMPT = (
    "Decide whether the user's utterance is complete at the end of this audio clip. "
    "Focus only on speech completion, not on whether the assistant should reply."
)


def _slice_candidate_window(
    wav: np.ndarray,
    sr: int,
    cut_time_ms: Optional[float],
    left_context_ms: float,
    right_context_ms: float,
) -> np.ndarray:
    """围绕候选切点裁出一个短窗口，供 turn detector 判断。"""
    if cut_time_ms is None:
        return wav

    cut_sample = int(round(float(cut_time_ms) * sr / 1000.0))
    left_samples = int(round(float(left_context_ms) * sr / 1000.0))
    right_samples = int(round(float(right_context_ms) * sr / 1000.0))

    start = max(0, cut_sample - max(0, left_samples))
    end = min(len(wav), cut_sample + max(0, right_samples))
    if end <= start:
        return wav
    return wav[start:end]


def _build_turn_detection_messages(prompt: str, audio_payload: Any) -> List[Dict[str, Any]]:
    """沿用 Qwen3-ASR 的 chat-style 输入协议。"""
    return [
        {"role": "system", "content": prompt or DEFAULT_TURN_DETECTION_PROMPT},
        {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
    ]


def build_turn_detection_prompt_text(processor, prompt: Optional[str] = None) -> str:
    """
    生成分类任务的 prefix_text。

    注意这里虽然最后不是“生成文本”，但仍然复用了 Qwen3-ASR 原本的输入模板，
    这样可以最大程度贴近底座的预训练 / 推理分布。
    """
    messages = _build_turn_detection_messages(prompt or DEFAULT_TURN_DETECTION_PROMPT, None)
    prompt_text = processor.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(prompt_text, list):
        return prompt_text[0]
    return prompt_text


@dataclass
class TurnDetectorPrediction:
    label: str
    complete_prob: float
    incomplete_prob: float
    logits: Optional[List[float]] = None


class Qwen3TurnDetector(nn.Module):
    """
    Two-class turn detector on top of Qwen3-ASR.

    The backbone is reused as a feature extractor. We keep the model separate
    from the ASR generation path so that:
      - training objective stays classification-focused;
      - inference is cheaper than text generation;
      - "waiting" can remain a policy-layer decision.
    """

    def __init__(
        self,
        backbone: nn.Module,
        processor: Any,
        base_model_path: str,
        prompt: str = DEFAULT_TURN_DETECTION_PROMPT,
        pooling: str = "last_token",
        num_labels: int = 2,
    ):
        super().__init__()
        if pooling not in {"last_token", "mean"}:
            raise ValueError(f"Unsupported pooling={pooling}")

        # backbone 就是完整的 Qwen3-ASR 模型；
        # classifier 是这个文件新增的小头。
        self.backbone = backbone
        self.processor = processor
        self.base_model_path = base_model_path
        self.prompt = prompt or DEFAULT_TURN_DETECTION_PROMPT
        self.pooling = pooling
        self.num_labels = int(num_labels)
        hidden_size = int(self.backbone.thinker.config.text_config.hidden_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.id2label = {0: "incomplete", 1: "complete"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.backbone.parameters()).dtype
        except StopIteration:
            return torch.float32

    @classmethod
    def from_qwen3_asr_pretrained(
        cls,
        base_model_path: str,
        prompt: str = DEFAULT_TURN_DETECTION_PROMPT,
        pooling: str = "last_token",
        num_labels: int = 2,
        **kwargs,
    ) -> "Qwen3TurnDetector":
        """
        先按普通 ASR 模型加载底座，再包装成 turn detector。

        这是“复用预训练权重”的真正入口。
        """
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            base_model_path,
            max_inference_batch_size=1,
            **kwargs,
        )
        return cls(
            backbone=asr_wrapper.model,
            processor=asr_wrapper.processor,
            base_model_path=base_model_path,
            prompt=prompt,
            pooling=pooling,
            num_labels=num_labels,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        base_model_path: Optional[str] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "Qwen3TurnDetector":
        """从 turn detector checkpoint 恢复模型。"""
        config_path = os.path.join(checkpoint_path, "turn_detector_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config file: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = cls.from_qwen3_asr_pretrained(
            base_model_path=base_model_path or cfg["base_model_path"],
            prompt=cfg.get("prompt", DEFAULT_TURN_DETECTION_PROMPT),
            pooling=cfg.get("pooling", "last_token"),
            num_labels=int(cfg.get("num_labels", 2)),
            **kwargs,
        )

        bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        safe_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location=map_location or "cpu")
        elif os.path.exists(safe_path):
            from safetensors.torch import load_file

            state_dict = load_file(safe_path, device="cpu")
        else:
            raise FileNotFoundError(
                f"Could not find checkpoint weights under {checkpoint_path}. "
                f"Expected {bin_path} or {safe_path}."
            )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading turn detector: {unexpected}")
        if missing:
            # Missing keys are okay only when loading a classifier-only checkpoint.
            allowed_missing = {k for k in missing if not k.startswith("classifier")}
            if len(allowed_missing) != len(missing):
                raise RuntimeError(f"Missing classifier weights when loading turn detector: {missing}")
        return model

    def save_pretrained(
        self,
        save_directory: str,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        safe_serialization: bool = False,
        **kwargs,
    ):
        """
        保存 detector 自己的权重和配置。

        这里不依赖 Hugging Face AutoModel 的默认保存逻辑，而是显式写出：
        - pytorch_model.bin
        - turn_detector_config.json
        """
        os.makedirs(save_directory, exist_ok=True)
        payload = state_dict if state_dict is not None else self.state_dict()
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(payload, weights_path)
        config = {
            "base_model_path": self.base_model_path,
            "prompt": self.prompt,
            "pooling": self.pooling,
            "num_labels": self.num_labels,
            "id2label": self.id2label,
        }
        with open(os.path.join(save_directory, "turn_detector_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def set_trainable_parts(
        self,
        freeze_audio_tower: bool = True,
        freeze_text_model: bool = True,
        unfreeze_last_n_layers: int = 0,
    ):
        """
        控制训练时哪些参数参与更新。

        默认推荐：
        - 冻结 audio_tower
        - 冻结 thinker 文本部分
        - 只训练 classifier

        如果任务数据更多、效果不够，再逐步解冻高层。
        """
        for p in self.backbone.parameters():
            p.requires_grad = False

        if not freeze_audio_tower:
            for p in self.backbone.thinker.audio_tower.parameters():
                p.requires_grad = True

        if not freeze_text_model:
            for p in self.backbone.thinker.model.parameters():
                p.requires_grad = True

        if unfreeze_last_n_layers > 0:
            for p in self.backbone.thinker.model.norm.parameters():
                p.requires_grad = True
            for layer in self.backbone.thinker.model.layers[-int(unfreeze_last_n_layers):]:
                for p in layer.parameters():
                    p.requires_grad = True

        for p in self.classifier.parameters():
            p.requires_grad = True

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        复用 Qwen3-ASR 的多模态编码路径，产出最后一层隐藏状态。

        对理解模型结构最关键的一步是：
        1. 先把文本 prompt 变成 token embedding；
        2. 再把音频波形编码成连续 audio_features；
        3. 用 audio_features 覆盖 prompt 里的 `<audio>` 占位 embedding；
        4. 把整条融合后的 embedding 序列送入 thinker。

        这也是为什么 turn detector 能“吃到底座已经学到的音频知识”。
        """
        thinker = self.backbone.thinker

        inputs_embeds = thinker.get_input_embeddings()(input_ids)

        if input_features is not None:
            # `get_audio_features` 把 mel 特征编码成连续向量；
            # `get_placeholder_mask` 找出 prompt 中 `<audio>` 占位的位置；
            # `masked_scatter` 则把占位 token 的 embedding 替换成真实音频表示。
            audio_features = thinker.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        position_ids = None
        if attention_mask is not None:
            # 这里沿用底座原本的 RoPE 位置编码逻辑，确保分类任务看到的序列位置定义
            # 与 ASR 主任务保持一致。
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = thinker.get_rope_index(attention_mask)
            thinker.rope_deltas = rope_deltas - delta0

        outputs = thinker.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        把整条序列的隐藏状态压成一个向量。

        - `last_token` 更接近“取 assistant 起始位的最终表征”；
        - `mean` 更像把整条序列做平均。
        """
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (hidden_states * mask).sum(dim=1) / denom

        seq_len = attention_mask.shape[1]
        positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand_as(attention_mask)
        last_token_idx = (positions * attention_mask.long()).max(dim=1).values
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_idx, last_token_idx]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """
        标准二分类前向：
        hidden_states -> pooling -> classifier -> cross entropy
        """
        hidden_states = self._encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        pooled = self._pool_hidden_states(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def _prepare_audio_batch(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        cut_time_ms: Optional[Union[float, List[Optional[float]]]] = None,
        left_context_ms: Optional[Union[float, List[float]]] = None,
        right_context_ms: Optional[Union[float, List[float]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        预测阶段的数据整理逻辑。

        和训练时的 collator 类似，但这里直接支持：
        - 单条音频；
        - 多条音频；
        - 路径 / ndarray / (waveform, sr) 等输入格式。
        """
        audios = audio if isinstance(audio, list) else [audio]
        cut_list = cut_time_ms if isinstance(cut_time_ms, list) else [cut_time_ms] * len(audios)
        left_list = left_context_ms if isinstance(left_context_ms, list) else [left_context_ms] * len(audios)
        right_list = right_context_ms if isinstance(right_context_ms, list) else [right_context_ms] * len(audios)
        prompt_list = prompt if isinstance(prompt, list) else [prompt] * len(audios)

        if not (len(audios) == len(cut_list) == len(left_list) == len(right_list) == len(prompt_list)):
            raise ValueError("Batch size mismatch in audio/cut/prompt inputs")

        wavs = []
        prompt_texts = []
        for audio_item, cut_ms, left_ms, right_ms, prompt_item in zip(
            audios, cut_list, left_list, right_list, prompt_list
        ):
            wav = normalize_audio_input(audio_item)
            wav = _slice_candidate_window(
                wav=wav,
                sr=16000,
                cut_time_ms=cut_ms,
                left_context_ms=2000.0 if left_ms is None else float(left_ms),
                right_context_ms=600.0 if right_ms is None else float(right_ms),
            )
            wavs.append(wav)
            prompt_texts.append(build_turn_detection_prompt_text(self.processor, prompt_item or self.prompt))

        inputs = self.processor(
            text=prompt_texts,
            audio=wavs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        return inputs

    @torch.no_grad()
    def predict(
        self,
        audio: AudioLike,
        cut_time_ms: Optional[float] = None,
        left_context_ms: float = 2000.0,
        right_context_ms: float = 600.0,
        prompt: Optional[str] = None,
    ) -> TurnDetectorPrediction:
        """
        方便直接推理单条样本。

        返回值里显式给出两类概率，方便业务层做阈值控制：
        - 若 complete_prob 很高，可以认为用户说完了；
        - 若两类接近，可以交给上层策略决定是否继续等待。
        """
        was_training = self.training
        self.eval()
        inputs = self._prepare_audio_batch(
            audio=audio,
            cut_time_ms=cut_time_ms,
            left_context_ms=left_context_ms,
            right_context_ms=right_context_ms,
            prompt=prompt,
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        outputs = self(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
        pred_id = int(torch.argmax(outputs.logits, dim=-1)[0].item())
        if was_training:
            self.train()
        return TurnDetectorPrediction(
            label=self.id2label[pred_id],
            complete_prob=float(probs[self.label2id["complete"]]),
            incomplete_prob=float(probs[self.label2id["incomplete"]]),
            logits=outputs.logits[0].tolist(),
        )
