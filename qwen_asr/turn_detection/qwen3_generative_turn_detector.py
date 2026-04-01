# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.inference.utils import AudioLike, normalize_audio_input

from .metrics import canonical_prediction_label, log_softmax_np
from .qwen3_turn_detector import (
    DEFAULT_TURN_DETECTION_PROMPT,
    TURN_LABELS,
    TurnDetectorPrediction,
    _slice_candidate_window,
    build_turn_detection_prompt_text,
)


@dataclass
class GenerativeTurnDetectorPrediction(TurnDetectorPrediction):
    raw_text: str = ""
    complete_logprob: Optional[float] = None
    incomplete_logprob: Optional[float] = None
    first_token_margin: Optional[float] = None
    exact_match: Optional[bool] = None


class _TrieNode:
    __slots__ = ("children", "terminal")

    def __init__(self):
        self.children: Dict[int, "_TrieNode"] = {}
        self.terminal = False


def _build_trie(sequences: Sequence[Sequence[int]]) -> _TrieNode:
    root = _TrieNode()
    for seq in sequences:
        node = root
        for token_id in seq:
            node = node.children.setdefault(int(token_id), _TrieNode())
        node.terminal = True
    return root


class Qwen3GenerativeTurnDetector:
    """
    Generative turn detector baseline on top of Qwen3-ASR.

    The model is still reused as an audio-text backbone, but the task output is
    forced to be a short label string instead of a classifier head.
    """

    def __init__(
        self,
        asr_wrapper: Qwen3ASRModel,
        prompt: str = DEFAULT_TURN_DETECTION_PROMPT,
        max_new_tokens: int = 4,
    ):
        if asr_wrapper.backend != "transformers":
            raise ValueError("Qwen3GenerativeTurnDetector currently supports only the transformers backend.")
        self.asr_wrapper = asr_wrapper
        self.model = asr_wrapper.model
        self.processor = asr_wrapper.processor
        self.prompt = prompt or DEFAULT_TURN_DETECTION_PROMPT
        self.max_new_tokens = int(max_new_tokens)

    @property
    def device(self) -> torch.device:
        return self.asr_wrapper.device

    @property
    def dtype(self) -> torch.dtype:
        return self.asr_wrapper.dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        prompt: str = DEFAULT_TURN_DETECTION_PROMPT,
        max_new_tokens: int = 4,
        **kwargs,
    ) -> "Qwen3GenerativeTurnDetector":
        config_path = os.path.join(pretrained_model_name_or_path, "turn_detection_generative_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if prompt == DEFAULT_TURN_DETECTION_PROMPT:
                prompt = cfg.get("prompt", prompt)
        wrapper = Qwen3ASRModel.from_pretrained(
            pretrained_model_name_or_path,
            max_inference_batch_size=1,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        return cls(
            asr_wrapper=wrapper,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    def set_trainable_parts(
        self,
        freeze_audio_tower: bool = True,
        freeze_text_model: bool = True,
        unfreeze_last_n_layers: int = 0,
        train_lm_head: bool = True,
    ) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

        if not freeze_audio_tower:
            for p in self.model.thinker.audio_tower.parameters():
                p.requires_grad = True

        if not freeze_text_model:
            for p in self.model.thinker.model.parameters():
                p.requires_grad = True

        if int(unfreeze_last_n_layers) > 0:
            for p in self.model.thinker.model.norm.parameters():
                p.requires_grad = True
            for layer in self.model.thinker.model.layers[-int(unfreeze_last_n_layers):]:
                for p in layer.parameters():
                    p.requires_grad = True

        if train_lm_head:
            for p in self.model.thinker.lm_head.parameters():
                p.requires_grad = True

    def _prepare_audio_batch(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        cut_time_ms: Optional[Union[float, List[Optional[float]]]] = None,
        left_context_ms: Optional[Union[float, List[float]]] = None,
        right_context_ms: Optional[Union[float, List[float]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        audios = audio if isinstance(audio, list) else [audio]
        cut_list = cut_time_ms if isinstance(cut_time_ms, list) else [cut_time_ms] * len(audios)
        left_list = left_context_ms if isinstance(left_context_ms, list) else [left_context_ms] * len(audios)
        right_list = right_context_ms if isinstance(right_context_ms, list) else [right_context_ms] * len(audios)
        prompt_list = prompt if isinstance(prompt, list) else [prompt] * len(audios)

        if not (len(audios) == len(cut_list) == len(left_list) == len(right_list) == len(prompt_list)):
            raise ValueError("Batch size mismatch in audio/cut/prompt inputs")

        wavs: List[np.ndarray] = []
        prompt_texts: List[str] = []
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
        return wavs, prompt_texts

    def _move_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved: Dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(self.device)
                if value.is_floating_point():
                    value = value.to(self.dtype)
            moved[key] = value
        return moved

    def _single_sample_allowed_sequences(self, wav: np.ndarray, prompt_text: str) -> Tuple[List[List[int]], Dict[str, List[int]]]:
        eos_id = self.processor.tokenizer.eos_token_id
        variants_by_label: Dict[str, List[List[int]]] = {label: [] for label in TURN_LABELS}
        first_tokens: Dict[str, List[int]] = {label: [] for label in TURN_LABELS}

        base = self.processor(text=[prompt_text], audio=[wav], return_tensors="pt", padding=True)
        base_len = base["input_ids"].shape[1]

        text_variants = {
            label: [label, f" {label}"]
            for label in TURN_LABELS
        }
        allowed_sequences: List[List[int]] = []
        for label, variants in text_variants.items():
            for variant in variants:
                full = self.processor(text=[prompt_text + variant], audio=[wav], return_tensors="pt", padding=True)
                full_ids = full["input_ids"][0].tolist()
                continuation = full_ids[base_len:]
                if not continuation:
                    continue
                if eos_id is not None and (not continuation or continuation[-1] != int(eos_id)):
                    continuation = continuation + [int(eos_id)]
                if continuation not in variants_by_label[label]:
                    variants_by_label[label].append(continuation)
                    allowed_sequences.append(continuation)
                    first_tokens[label].append(int(continuation[0]))
        dedup_first_tokens = {k: sorted(set(v)) for k, v in first_tokens.items()}
        return allowed_sequences, dedup_first_tokens

    def _build_prefix_allowed_tokens_fn(
        self,
        wavs: List[np.ndarray],
        prompt_texts: List[str],
        base_input_len: int,
    ):
        tries: List[_TrieNode] = []
        first_token_ids: List[Dict[str, List[int]]] = []
        eos_id = self.processor.tokenizer.eos_token_id

        for wav, prompt_text in zip(wavs, prompt_texts):
            sequences, first_tokens = self._single_sample_allowed_sequences(wav, prompt_text)
            tries.append(_build_trie(sequences))
            first_token_ids.append(first_tokens)

        def _allowed(batch_id: int, input_ids: torch.LongTensor) -> List[int]:
            prefix = input_ids.tolist()[base_input_len:]
            node = tries[int(batch_id)]
            for token_id in prefix:
                child = node.children.get(int(token_id))
                if child is None:
                    return [int(eos_id)] if eos_id is not None else list(node.children.keys())
                node = child

            allowed = list(node.children.keys())
            if node.terminal and eos_id is not None:
                allowed.append(int(eos_id))
            if not allowed and eos_id is not None:
                allowed = [int(eos_id)]
            return sorted(set(int(x) for x in allowed))

        return _allowed, first_token_ids

    def score_label_logprobs(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        cut_time_ms: Optional[Union[float, List[Optional[float]]]] = None,
        left_context_ms: Optional[Union[float, List[float]]] = None,
        right_context_ms: Optional[Union[float, List[float]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, List[float]]:
        wavs, prompt_texts = self._prepare_audio_batch(
            audio=audio,
            cut_time_ms=cut_time_ms,
            left_context_ms=left_context_ms,
            right_context_ms=right_context_ms,
            prompt=prompt,
        )

        prompt_inputs = self.processor(
            text=prompt_texts,
            audio=wavs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prompt_inputs = self._move_inputs(prompt_inputs)
        prefix_lens = prompt_inputs["attention_mask"].sum(dim=1).tolist()

        label_logprobs: Dict[str, List[float]] = {}
        for label in TURN_LABELS:
            full_texts = [prompt_text + label for prompt_text in prompt_texts]
            full_inputs = self.processor(
                text=full_texts,
                audio=wavs,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            labels = full_inputs["input_ids"].clone()
            for i, pl in enumerate(prefix_lens):
                labels[i, : int(pl)] = -100
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
            full_inputs["labels"] = labels
            full_inputs = self._move_inputs(full_inputs)

            outputs = self.model.thinker.forward(
                input_ids=full_inputs["input_ids"],
                attention_mask=full_inputs["attention_mask"],
                input_features=full_inputs.get("input_features"),
                feature_attention_mask=full_inputs.get("feature_attention_mask"),
                labels=full_inputs["labels"],
                use_cache=False,
            )

            shift_logits = outputs.logits[:, :-1, :].detach().float().cpu().numpy()
            shift_labels = full_inputs["labels"][:, 1:].detach().cpu().numpy()
            token_logprobs = log_softmax_np(shift_logits, axis=-1)
            sample_scores: List[float] = []
            for row_idx in range(shift_labels.shape[0]):
                valid_mask = shift_labels[row_idx] != -100
                if not np.any(valid_mask):
                    sample_scores.append(float("-inf"))
                    continue
                valid_labels = shift_labels[row_idx][valid_mask].astype(np.int64)
                valid_logprobs = token_logprobs[row_idx][valid_mask]
                gathered = valid_logprobs[np.arange(len(valid_labels)), valid_labels]
                sample_scores.append(float(gathered.sum()))
            label_logprobs[label] = sample_scores

        incomplete_lp = np.asarray(label_logprobs["incomplete"], dtype=np.float64)
        complete_lp = np.asarray(label_logprobs["complete"], dtype=np.float64)
        stacked = np.stack([incomplete_lp, complete_lp], axis=-1)
        norm = log_softmax_np(stacked, axis=-1)
        complete_prob = np.exp(norm[:, 1]).tolist()
        incomplete_prob = np.exp(norm[:, 0]).tolist()
        margin = (stacked[:, 1] - stacked[:, 0]).tolist()

        return {
            "complete_logprob": complete_lp.tolist(),
            "incomplete_logprob": incomplete_lp.tolist(),
            "complete_prob": complete_prob,
            "incomplete_prob": incomplete_prob,
            "logprob_margin": margin,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        cut_time_ms: Optional[Union[float, List[Optional[float]]]] = None,
        left_context_ms: Optional[Union[float, List[float]]] = None,
        right_context_ms: Optional[Union[float, List[float]]] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        constrained_decode: bool = True,
    ) -> List[GenerativeTurnDetectorPrediction]:
        wavs, prompt_texts = self._prepare_audio_batch(
            audio=audio,
            cut_time_ms=cut_time_ms,
            left_context_ms=left_context_ms,
            right_context_ms=right_context_ms,
            prompt=prompt,
        )
        inputs = self.processor(
            text=prompt_texts,
            audio=wavs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = self._move_inputs(inputs)
        base_input_len = inputs["input_ids"].shape[1]

        prefix_allowed_tokens_fn = None
        first_token_ids: List[Dict[str, List[int]]] = [{label: [] for label in TURN_LABELS} for _ in wavs]
        if constrained_decode:
            prefix_allowed_tokens_fn, first_token_ids = self._build_prefix_allowed_tokens_fn(
                wavs,
                prompt_texts,
                base_input_len=base_input_len,
            )

        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            return_dict_in_generate=True,
            output_scores=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        generated_ids = outputs.sequences[:, base_input_len:]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        first_step_scores = None
        if getattr(outputs, "scores", None):
            first_step_scores = outputs.scores[0].detach().float().cpu().numpy()

        label_scores = self.score_label_logprobs(
            audio=audio,
            cut_time_ms=cut_time_ms,
            left_context_ms=left_context_ms,
            right_context_ms=right_context_ms,
            prompt=prompt,
        )

        predictions: List[GenerativeTurnDetectorPrediction] = []
        for idx, raw_text in enumerate(decoded):
            raw_normalized = " ".join(str(raw_text).strip().lower().split())
            canonical = canonical_prediction_label(raw_text)
            pred_label = canonical if canonical in TURN_LABELS else ("complete" if label_scores["complete_prob"][idx] >= 0.5 else "incomplete")
            margin = None
            if first_step_scores is not None:
                complete_first = first_token_ids[idx].get("complete", [])
                incomplete_first = first_token_ids[idx].get("incomplete", [])
                if complete_first and incomplete_first:
                    complete_score = max(float(first_step_scores[idx, tok]) for tok in complete_first)
                    incomplete_score = max(float(first_step_scores[idx, tok]) for tok in incomplete_first)
                    margin = complete_score - incomplete_score
            predictions.append(
                GenerativeTurnDetectorPrediction(
                    label=pred_label,
                    complete_prob=float(label_scores["complete_prob"][idx]),
                    incomplete_prob=float(label_scores["incomplete_prob"][idx]),
                    logits=None,
                    latency_ms=None,
                    raw_text=str(raw_text),
                    complete_logprob=float(label_scores["complete_logprob"][idx]),
                    incomplete_logprob=float(label_scores["incomplete_logprob"][idx]),
                    first_token_margin=margin,
                    exact_match=(raw_normalized in TURN_LABELS),
                )
            )

        if predictions:
            per_item_ms = elapsed_ms / len(predictions)
            for pred in predictions:
                pred.latency_ms = per_item_ms
        return predictions

    @torch.no_grad()
    def predict(
        self,
        audio: AudioLike,
        cut_time_ms: Optional[float] = None,
        left_context_ms: float = 2000.0,
        right_context_ms: float = 600.0,
        prompt: Optional[str] = None,
        constrained_decode: bool = True,
    ) -> GenerativeTurnDetectorPrediction:
        return self.predict_batch(
            audio=[audio],
            cut_time_ms=[cut_time_ms],
            left_context_ms=[left_context_ms],
            right_context_ms=[right_context_ms],
            prompt=[prompt],
            constrained_decode=constrained_decode,
        )[0]
