# Qwen3ASRProcessor 注释版

这份文档是给没接触过 `transformers`、也没做过多模态模型的人看的。

目标只有一个：

- 把 `qwen_asr/core/transformers_backend/processing_qwen3_asr.py` 里 `Qwen3ASRProcessor` 的作用讲清楚

你可以把 `Qwen3ASRProcessor` 先理解成一句话：

`它不是模型本体，而是模型前面的总预处理器。`

它负责把：

- 人类能理解的输入
  - 文本 prompt
  - 音频 waveform

变成：

- 模型能直接吃的张量
  - `input_ids`
  - `attention_mask`
  - `input_features`
  - `feature_attention_mask`

---

## 1. 原文件做什么

对应源码文件：

- [processing_qwen3_asr.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/core/transformers_backend/processing_qwen3_asr.py)

它主要做 5 件事：

1. 定义默认的文本和音频处理参数
2. 计算音频经过特征提取后会变成多长
3. 同时处理文本和音频
4. 把文本里的音频占位符扩展到正确长度
5. 告诉外部，这个 processor 最终会返回哪些输入字段

---

## 2. 逐段注释版

下面不是逐字复制源码，而是按原始代码结构改写成“初学者注释版”。

```python
# 正则库，用来找特殊 token
import re

# NumPy 主要用来处理数组
import numpy as np

# AudioInput: Transformers 里定义的“音频输入类型”
from transformers.audio_utils import AudioInput

# BatchFeature: 一个“输入包”，本质上像 dict
# 里面装的是模型要用的各种张量
from transformers.feature_extraction_utils import BatchFeature

# ProcessingKwargs / ProcessorMixin:
# 这两个是 Hugging Face 里 processor 相关的基础类
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin

# TextInput: Transformers 里定义的“文本输入类型”
from transformers.tokenization_utils_base import TextInput
```

这一段只是导入依赖。

如果你是初学者，只要记住：

- `tokenizer` 处理文本
- `feature_extractor` 处理音频
- `processor` 把两者打包到一起

---

### 2.1 `Qwen3ASRProcessorKwargs`

```python
class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
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
```

这段是在定义“默认参数”。

可以把它理解成：

- 文本默认怎么补齐
- 音频默认用什么采样率
- 音频处理后要不要返回 mask

这里最重要的是：

- 文本默认左侧 padding
- 音频默认按 `16000 Hz`
- 音频默认返回 `attention_mask`

为什么要 `attention_mask`？

因为一个 batch 里的音频长度通常不一样，短音频会被补零。
模型需要知道：

- 哪些位置是真实音频
- 哪些位置只是补齐出来的 padding

---

### 2.2 `_get_feat_extract_output_lengths`

```python
def _get_feat_extract_output_lengths(input_lengths):
    """
    计算音频经过前端特征提取和下采样以后，还剩多少“特征帧”
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths
```

这一段最容易看懵。

简单讲，它是在回答一个问题：

`原始音频长度是 N，经过模型前面的音频特征提取后，会变成多少个音频 token 位置？`

为什么这个长度重要？

因为文本里会有一个“音频占位符”，但真实音频不是 1 个 token 就能表示的。
processor 必须知道：

- 这一段音频最终会对应多少个位置

这样才能把文本里的音频占位符扩展成正确长度。

如果你现在看不懂公式本身，没有关系。
你只需要记住它的职责：

- 输入：音频原始长度
- 输出：音频在模型输入里将占多少个位置

---

### 2.3 `Qwen3ASRProcessor` 类本体

```python
class Qwen3ASRProcessor(ProcessorMixin):
```

这个类是整个文件的核心。

它继承自 `ProcessorMixin`，说明它本质上是 Hugging Face 风格的 processor。

也就是：

- 不是模型层
- 不是分类头
- 不是解码头
- 只是预处理组件

---

### 2.4 类说明

```python
    r"""
    Constructs a Qwen3ASR processor.
    [`Qwen3ASRProcessor`] offers all the functionalities of
    [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`].
    """
```

这段官方说明的意思是：

这个 processor 同时具备两种能力：

- `WhisperFeatureExtractor` 的音频处理能力
- `Qwen2TokenizerFast` 的文本分词能力

所以你可以把它理解为：

`音频前端 + 文本 tokenizer + 多模态打包器`

---

### 2.5 `attributes`

```python
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
```

这几行是在声明：

- 这个 processor 内部包含两个核心部件
  - `feature_extractor`
  - `tokenizer`

也就是说：

- 音频交给 `feature_extractor`
- 文本交给 `tokenizer`

这就是为什么 `AutoProcessor.from_pretrained(...)` 加载出来后，既能处理音频，也能处理文本。

---

### 2.6 `__init__`

```python
    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.audio_token = self.tokenizer.audio_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token
```

这里做了两件事。

第一件事：

- 把传进来的 `feature_extractor` 和 `tokenizer` 保存起来

第二件事：

- 从 tokenizer 里取出和音频相关的特殊 token

这些特殊 token 是什么？

你可以把它们理解成文本里的“标记位”：

- 这里开始放音频
- 这里是音频内容
- 这里音频结束

在多模态模型里，音频不会凭空出现，通常需要一些特殊 token 作为占位和边界标记。

---

### 2.7 `__call__`

这是最重要的方法。

```python
    def __call__(self, text: TextInput = None, audio: AudioInput = None, **kwargs) -> BatchFeature:
```

这表示：

- 你可以像调用函数一样调用 processor

例如：

```python
inputs = processor(
    text=prompt_texts,
    audio=wavs,
    return_tensors="pt",
    padding=True,
)
```

#### 第一步：检查文本

```python
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
```

这里说明当前实现要求必须有文本。

为什么？

因为这个模型的输入不是“只有音频”，而是：

- 一段聊天模板格式的文本
- 加上一段音频

也就是多模态对话输入。

---

#### 第二步：整理参数

```python
        output_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
```

这一步是在把：

- 默认参数
- tokenizer 初始化参数
- 你这次调用传进来的参数

合并成最终使用的参数。

通俗理解：

`把系统默认设置和你手动传入的设置，合并成最终版本。`

---

#### 第三步：如果有音频，先处理音频

```python
        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["truncation"] = False
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
```

这里会调用音频特征提取器。

输入是：

- 原始 waveform

输出通常是：

- `input_features`
- `attention_mask`

你可以把 `input_features` 理解成：

`音频经过前端处理后得到的特征图`

而不是原始波形。

---

#### 第四步：重命名音频字段

```python
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_inputs["input_features"] = audio_inputs.pop("input_features")
```

这里看起来第二行像“没改名”，其实作者是在明确统一字段名。

重点是第一行：

- 把音频侧的 `attention_mask`
- 改名成 `feature_attention_mask`

为什么？

因为文本侧也会有一个 `attention_mask`。

如果不改名，后面把文本输入和音频输入放进同一个字典时，就会冲突。

所以这里是在区分：

- 文本 mask：`attention_mask`
- 音频 mask：`feature_attention_mask`

---

#### 第五步：计算每条音频最终会占多少位置

```python
            audio_lengths = iter(
                _get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1))
            )
```

这一句的作用是：

- 先统计每条音频真实长度
- 再算出经过特征提取后，对应多少个音频 token 位置

后面扩展音频占位符时就要用这个长度。

如果没有音频：

```python
        else:
            audio_inputs = {}
            audio_lengths = iter([])
```

就说明只处理文本，不处理音频。

---

#### 第六步：把单条文本统一成列表

```python
        if not isinstance(text, list):
            text = [text]
```

这是一个很常见的小技巧。

因为后面的处理逻辑想统一按 batch 来写。

哪怕你只传一条文本，也先包装成列表，后面处理更简单。

---

#### 第七步：替换多模态特殊 token

```python
        text = self.replace_multimodal_special_tokens(text, audio_lengths)
```

这是这个 processor 最关键的地方之一。

为什么关键？

因为文本里原本只写了一个音频占位符，但真实音频会展开成很多位置。

这个函数会把：

- 一个音频占位符

扩展成：

- 和该音频实际长度匹配的很多占位位置

这样文本和音频才能在模型输入里对齐。

---

#### 第八步：处理文本

```python
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
```

这一步就是普通 tokenizer 的工作：

- 把文本转成 token ids
- 生成文本 attention mask

结果里一般会有：

- `input_ids`
- `attention_mask`

---

#### 第九步：把文本输入和音频输入打包返回

```python
        return BatchFeature(
            data={**texts_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )
```

这一句就是把前面两边结果拼起来。

最终得到一个统一输入包。

这个包里可能包含：

- `input_ids`
- `attention_mask`
- `input_features`
- `feature_attention_mask`

模型前向时直接吃这个包里的字段即可。

---

### 2.8 `replace_multimodal_special_tokens`

```python
    def replace_multimodal_special_tokens(self, text, audio_lengths):
```

这个函数很重要。

它解决的问题是：

`文本里一个音频 token，不足以表示真实音频长度。`

来看逻辑。

```python
        processed_text = []
        for sample in text:
```

逐条文本处理。

```python
            special_tokens = [re.escape(tok) for tok in [self.audio_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
```

这一步是在文本里查找音频特殊 token 的位置。

你可以理解成：

- 找到“这里有一个音频占位符”

```python
            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(
                        self.audio_token,
                        "<|audio_placeholder|>" * next(audio_lengths),
                        1,
                    )
```

这里是真正的扩展逻辑。

意思是：

- 原来只有 1 个音频 token
- 现在把它替换成很多个临时占位符
- 占位符个数由该音频的长度决定

为什么先替换成临时占位符，再换回去？

因为直接边找边替换同一个 token，容易把已经替换过的内容再次误处理。

所以作者用了一个临时名字：

- `<|audio_placeholder|>`

最后再统一换回正式的 `audio_token`：

```python
            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            processed_text.append(sample)
```

这样最终文本表面上还是音频 token，
但数量已经变成了“和真实音频长度匹配”的数量。

---

### 2.9 `get_chunked_index`

```python
    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
```

这个函数和主流程关系没那么强。

它的作用是：

- 根据 token 的编号范围，把它们切成多个 chunk

例如：

- 小于 1000 的是一块
- 1000 到 1999 的是一块
- 2000 到 2999 的是一块

你可以把它理解成：

`给一串已经排好序的位置编号，按固定区间分段。`

它更像辅助工具函数，不是 processor 的主逻辑核心。

---

### 2.10 `apply_chat_template`

```python
    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        return super().apply_chat_template(conversations, chat_template, **kwargs)
```

这一段看起来什么都没做，其实它保留了 chat template 能力。

Qwen3-ASR 的输入不是裸文本，而是聊天格式：

- `system`
- `user`
- `assistant`

processor 会把这类结构化对话，转成模型真正认识的字符串格式。

通俗理解：

- 先把“角色对话”排版成模型习惯的模板
- 再做 tokenizer

---

### 2.11 `model_input_names`

```python
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + ["feature_attention_mask"]
            )
        )
```

这一段是在告诉外部：

`这个 processor 最终会返回哪些输入字段。`

通常包括：

- 文本侧输入名
- 音频侧输入名
- 额外的 `feature_attention_mask`

为什么这有用？

因为训练器、推理器、模型封装代码经常会问：

`这个 processor 会生成哪些字段？`

这里就给了统一答案。

---

## 3. 用一句最通俗的话总结每个组件

### `feature_extractor`

把原始音频波形变成模型可用的声学特征。

### `tokenizer`

把文本 prompt 变成 token id。

### `chat_template`

把 `system/user/assistant` 这种对话结构排成模型熟悉的文本格式。

### `Qwen3ASRProcessor`

把上面三件事串起来，最后返回模型输入包。

---

## 4. 一个最小调用流程

```python
inputs = processor(
    text="请判断这段音频里的话是否说完",
    audio=wav,
    return_tensors="pt",
    padding=True,
)
```

背后发生的事可以理解成：

1. 先处理音频，得到 `input_features`
2. 计算这段音频会占多少输入位置
3. 处理文本，把文本转成 `input_ids`
4. 把文本里的音频占位符扩展到正确长度
5. 返回一个统一字典给模型

---

## 5. 为什么它不是“音频头”

很多初学者会把 processor 和模型层搞混。

这里明确一下：

- `Qwen3ASRProcessor` 不是模型层
- 不是 encoder
- 不是 decoder
- 不是 classification head
- 不是 ASR head

它只是在模型前面做准备工作。

更像：

- 数据整理器
- 输入转换器
- 多模态打包器

---

## 6. 最后再压缩成一句话

如果你只记一句，就记这句：

`Qwen3ASRProcessor 的职责，是把“文本 + 音频”整理成 Qwen3-ASR 能直接吃的标准输入。`

如果你还想继续往下看，下一步最适合读的是：

- [modeling_qwen3_asr.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

因为 processor 负责“喂进去什么”，而 model 负责“进去之后怎么计算”。
