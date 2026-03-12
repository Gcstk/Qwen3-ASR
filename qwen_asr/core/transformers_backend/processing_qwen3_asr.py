# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
import re

import numpy as np

from transformers.audio_utils import AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import TextInput


class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    # processor 的默认参数分两类：
    # 1. text_kwargs: 传给 tokenizer
    # 2. audio_kwargs: 传给 feature_extractor
    # 这样调用 processor(text=..., audio=...) 时，文本和音频都能拿到各自默认配置。
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
        },
    }


def _get_feat_extract_output_lengths(input_lengths):
    """
    根据原始音频长度，估算音频前端处理后的输出长度。

    更直白地说：
    原始 waveform 经过特征提取、卷积和下采样后，时间维长度会变短。
    这个函数就是用来推导“最终会剩下多少个音频位置”。

    参数：
        input_lengths:
            原始音频长度，通常来自音频 attention mask 的有效长度统计。

    返回：
        output_lengths:
            音频进入模型时对应的特征长度，也就是需要占用多少个音频 token 位置。
    """
    # 原始音频长度不会直接等于模型里“音频 token”的长度。
    # 音频经过前端特征提取和下采样后，会缩短成更小的时序长度。
    # 这里的作用就是：给定原始音频长度，推导出最终要占多少个音频位置。

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3ASRProcessor(ProcessorMixin):
    r"""
    Qwen3-ASR 的统一预处理器。

    这个类不是模型本体，也不是分类头或解码头。
    它的职责是把“文本 + 音频”整理成模型能直接使用的输入张量。

    你可以把它理解成一个总接线员：
    - 文本部分交给 tokenizer
    - 音频部分交给 feature extractor
    - 最后把两边结果打包成同一个输入对象返回给模型

    从能力上看，它同时具备：
    - [`WhisperFeatureExtractor`] 的音频特征提取能力
    - [`Qwen2TokenizerFast`] 的文本分词能力

    如果你想看主流程，最重要的方法是 [`~Qwen3ASRProcessor.__call__`]。

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            音频特征提取器。负责把原始音频 waveform 转成模型使用的声学特征。
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            文本分词器。负责把 prompt 或聊天模板文本转成 token ids。
        chat_template (`Optional[str]`, *optional*):
            聊天模板。用于把 `system/user/assistant` 这样的结构化消息排版成模型可读的文本格式。
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self, feature_extractor=None, tokenizer=None, chat_template=None
    ):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        # tokenizer 里定义了多模态输入需要用到的音频特殊 token。
        # 它们不是“模型层”，而是输入文本里的占位标记。
        self.audio_token = self.tokenizer.audio_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def __call__(
        self,
        text: TextInput = None,
        audio: AudioInput = None,
        **kwargs,
    ) -> BatchFeature:
        """
        处理一条或多条“文本 + 音频”输入，并返回模型可直接使用的输入包。

        它的大致流程是：
        1. 如果有音频，先用 `feature_extractor` 处理音频，得到 `input_features`
        2. 根据音频处理后的长度，扩展文本里的音频占位符
        3. 用 `tokenizer` 处理文本，得到 `input_ids`、`attention_mask`
        4. 把文本侧和音频侧结果合并成一个 `BatchFeature`

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                要编码的文本，既可以是一条字符串，也可以是一个 batch 的字符串列表。
                如果你传入的是“已经分好词”的列表，需要额外设置 `is_split_into_words=True`。
            audio (`np.ndarray`, `List[np.ndarray]`):
                要处理的音频，既可以是一条音频，也可以是一个 batch 的音频列表。
                每条音频通常是 `np.ndarray`。

        返回：
            `BatchFeature`:
                一个类似字典的输入包，通常包含：
                - 文本侧：`input_ids`、`attention_mask`
                - 音频侧：`input_features`、`feature_attention_mask`
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        # 把默认参数、tokenizer 初始化参数以及这次调用传入的参数合并起来。
        # 最终会分别形成 text_kwargs 和 audio_kwargs。
        output_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            # 对音频侧强制开启 padding，关闭 truncation。
            # 这样 batch 内不同长度的音频都能对齐，而且不会在 processor 阶段截断。
            output_kwargs["audio_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["truncation"] = False
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            # 文本侧已经会产出 attention_mask，所以音频侧改名成 feature_attention_mask，
            # 避免后面拼接输入字典时发生字段名冲突。
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            # 计算每条音频在进入模型后实际会占多少个“音频 token 位置”。
            # 后面替换文本里的音频占位符时会用到这个长度。
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if not isinstance(text, list):
            # 统一按 batch 形式处理，哪怕只传入了一条文本也包装成 list。
            text = [text]

        # 文本里原本只有一个音频特殊 token，但真实音频会展开成很多位置。
        # 这里会根据 audio_lengths，把文本中的音频占位符扩展到正确长度。
        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
        )

        # 文本部分交给 tokenizer，转成 input_ids / attention_mask 等张量。
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # 最终把文本侧输入和音频侧输入放到同一个 BatchFeature 里返回。
        # 这就是模型 forward 真正会接收的“输入包”。
        return BatchFeature(
            data={**texts_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
    ):
        # 这个函数的核心目标：
        # 把文本中的一个 audio_token，扩展成“与该音频实际长度匹配”的多个 audio_token。
        # 否则文本序列里的音频占位长度和真实音频特征长度对不上。
        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    # 先替换成临时 placeholder，避免直接重复替换同一个 audio_token
                    # 时把已经展开的部分再次错误处理。
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)

            # 最后再把临时 placeholder 统一换回正式的 audio_token。
            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            processed_text.append(sample)
        return processed_text

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        """
        按 token 编号范围，把一串已排序的索引切成多个 chunk。

        给定一串递增的 token 索引，这个函数会返回若干 `(start, end)`，
        表示这些索引在原数组里应如何切片。

        例如当 `tokens_per_chunk=1000` 时：
        - 第一个 chunk 包含编号 < 1000 的 token
        - 第二个 chunk 包含编号 >= 1000 且 < 2000 的 token
        - 以此类推

        Parameters:
            token_indices (`np.ndarray`): A monotonically increasing list of token index values.
                一个递增的 token 索引数组。
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).
                每个 chunk 对应的 token 范围宽度。

        Returns:
            `list[tuple[int, int]]`:
                返回若干 `(start, end)` 元组。
                其中 `start` 是包含位置，`end` 是不包含位置。
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        # 这里直接复用 ProcessorMixin 的 chat template 能力。
        # 它会把 system/user/assistant 这样的结构化消息，排版成模型可读的文本格式。
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        # 告诉外部：这个 processor 最终可能返回哪些模型输入字段。
        # 文本侧通常有 input_ids / attention_mask，
        # 音频侧通常有 input_features，
        # 另外这里额外暴露 feature_attention_mask 供模型区分音频 padding。
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + ["feature_attention_mask"]
            )
        )


__all__ = ["Qwen3ASRProcessor"]
