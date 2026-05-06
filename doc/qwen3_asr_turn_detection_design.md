# Qwen3-ASR 对话轮次检测设计文档

## 1. 文档目标

这份文档用于指导在 **纯净 `Qwen3-ASR` 主干模型** 上实现一个数据效率高、推理效率高、便于后续扩展的对话轮次检测模型。

本文档刻意 **不把当前仓库中已有的 turn detection 代码当作设计前提**。后续编码应从 `Qwen3-ASR` 的基础结构出发，重新设计一条最适合小数据场景的路线。

本文档重点回答 5 个问题：

1. 为什么要从纯净 `Qwen3-ASR` 结构出发，而不是直接套一个普通分类器
2. 怎样最大化复用 `Qwen3-ASR` 的预训练能力
3. 在数据不充足时，推荐怎样组织训练目标
4. 后续编码时应该新增哪些模块，接口如何定义
5. 第一版先做什么，第二版再扩什么

---

## 2. 版本范围

### 2.1 V1 目标

第一版只做二分类：

- `complete`
- `incomplete`

V1 的目标是先把“用户当前话语是否已经说完”做稳，形成可上线、可调阈值、可接入 agent 路由的强基线。

### 2.2 V2 目标

第二版在 V1 稳定后再扩展：

- `complete`
- `incomplete`
- `backchannel`

`backchannel` 是否纳入正式训练任务，取决于：

- V1 二分类指标是否达到要求
- `backchannel` 数据量和标注一致性是否足够
- 与 `complete` 的边界是否能在当前输入条件下稳定区分

如果 `backchannel` 在当前输入条件下很难做，V2 可以先做离线分析和单独评估，不急于并入正式线上模型。

---

## 3. 当前 Qwen3-ASR 结构与可复用能力

后续设计必须基于当前项目真实代码，而不是抽象想象。

### 3.1 主干结构

`Qwen3-ASR` 的核心链路在：

- `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`

关键结构如下：

1. `audio_tower` 先把音频编码成连续音频特征
2. 文本侧 `thinker` 使用 decoder-only Transformer
3. 文本 prompt 中预留音频占位 token
4. 前向时把音频特征覆盖到这些占位 token 的 embedding 上
5. 整个多模态序列再统一送入 thinker decoder

对应代码锚点：

- `Qwen3ASRThinkerForConditionalGeneration`：
  `../qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`
- `get_audio_features(...)`：
  `../qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`
- `get_placeholder_mask(...)`：
  `../qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`
- thinker 前向中音频注入逻辑：
  `../qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`

### 3.2 输入组织方式

`Qwen3-ASR` 的输入不是“裸音频 + 裸标签”，而是：

- system prompt
- user audio
- assistant generation target

相关代码在：

- `qwen_asr/inference/qwen3_asr.py`
- `qwen_asr/core/transformers_backend/processing_qwen3_asr.py`
- `finetuning/qwen3_asr_sft.py`

尤其要注意：

- `processor.apply_chat_template(...)` 负责把消息组织成模型熟悉的文本前缀
- `processor(text=..., audio=...)` 负责同时处理文本和音频
- `qwen3_asr_sft.py` 的训练数据期望主结构是 `audio + text + prompt`

这决定了后续 turn detection 设计应尽量保持同样的数据协议。

### 3.3 预训练能力里最值得复用的部分

在 turn detection 任务中，最有价值的不是单纯的 ASR 文本输出，而是下面三类能力：

1. 音频边界和韵律建模能力
2. 语音内容的隐式语义理解能力
3. 在指令条件下，围绕音频输出短文本判断的能力

如果训练数据不多，最稳妥的办法不是大改结构，而是：

- 尽量保留原始输入分布
- 尽量保留原始生成式任务形式
- 在此基础上增加一个轻量、可部署的判别读出层

---

## 4. 设计原则

### 4.1 原则一：最终模型数据结构尽量和 Qwen3-ASR SFT 保持一致

最终训练/验证/测试 manifest 统一采用：

- `audio`
- `text`
- `prompt`

同时允许额外保留：

- `label`
- `sample_id`
- `session_id`
- `source_audio`
- `clip_start_ms`
- `clip_end_ms`
- `split`

其中：

- `audio + text + prompt` 与 `qwen3_asr_sft.py` 完全同构
- `label` 给未来分类头训练、评估和分析使用

### 4.2 原则二：保持原始多模态输入路径不变

后续 turn detection 模型必须继续使用：

- `processor.apply_chat_template(...)`
- `processor(text=..., audio=...)`
- thinker 中的音频占位注入路径

不要为了“更像分类任务”而把模型改成另一套输入管线。

### 4.3 原则三：优先利用生成式预训练语义，再追求高效推理

小数据场景下，直接新增一个随机初始化分类头，虽然推理快，但任务语义迁移不一定稳。

更合理的路线是分两步：

1. 先利用生成式 verbalizer 让模型学会任务语义
2. 再把这个能力蒸馏/迁移到轻量分类头

### 4.4 原则四：第一版只解决一个核心问题

V1 只做：

- “用户当前话语是否已经说完”

不要在第一版同时解决：

- `complete / incomplete`
- `backchannel`
- 多说话人重叠抢话
- 强策略控制语义

这些问题拆开做，数据和工程都会更稳。

---

## 5. 推荐模型路线

## 5.1 总体路线

推荐采用 **两阶段方案**：

### 阶段 A：生成式语义适配

目标：

- 让纯净 `Qwen3-ASR` 学会“对候选端点窗口做 complete/incomplete 判断”
- 最大化复用其原有生成式预训练能力

输入格式：

- `audio`: 候选端点附近裁好的音频 clip
- `prompt`: 任务提示词
- `text`: 固定 verbalizer 目标句

推荐 verbalizer：

- `The user's utterance is complete.`
- `The user's utterance is incomplete.`

推荐 prompt：

`Listen to the audio clip and decide whether the user's utterance is complete at the end of the clip. Answer with exactly one sentence.`

说明：

- prompt 和 verbalizer 在一个实验中必须统一，不要混用中英文模板
- 若业务语言主要是中文，也可以改成统一中文模板，但同一批训练中不要混写

训练策略：

- 冻结 `audio_tower`
- 优先只解冻 `lm_head` 和 thinker 顶部少量层
- 初始建议解冻 thinker 顶部 `2` 到 `4` 层
- 不建议一开始全参微调

阶段 A 的产物：

- 一个经过轻量任务适配的 `Qwen3-ASR` 检查点
- 它可以直接做生成式推理，作为强语义教师模型

### 阶段 B：高效判别式部署模型

目标：

- 在阶段 A 的 backbone 上增加轻量分类头
- 将阶段 A 学到的语义迁移成高效推理模型

输入格式：

- 与阶段 A 保持相同的 `audio + prompt`
- 使用统一 manifest 中的 `label`

推荐结构：

1. 使用 prompt-only 输入
2. 保持音频和文本仍经原始 processor 与 thinker 融合
3. 读取最后一个有效 token 的 hidden state
4. 经过轻量 MLP / Linear head 输出二分类 logits

推荐 loss：

- 主损失：交叉熵分类损失
- 可选辅助损失：对阶段 A 教师模型输出分布做 KL distillation

阶段 B 的产物：

- 一个可部署的 turn detection 模型
- 推理时不再生成整句文本，只输出 `label + probs + confidence`

---

## 6. 为什么不推荐直接采用以下方案

### 6.1 不推荐直接做平面三分类

理由：

- `backchannel` 与 `complete` 语义并不天然同层
- 小数据时三分类边界更容易塌陷
- 第一版先做稳定二分类，更容易快速形成可靠基线

### 6.2 不推荐 V1 直接做纯分类头、完全不经过 verbalizer 适配

理由：

- 数据太少时，随机初始化头很容易只学到浅层时长、停顿等偏差
- 不能充分利用 `Qwen3-ASR` 已有的“音频到短文本判断”能力

### 6.3 不推荐 V1 直接上线生成式推理

理由：

- 推理慢
- 线上阈值、校准、批量化部署都不如分类头方便
- 输出字符串也更容易受模板波动影响

因此推荐：

- 阶段 A 用生成式适配任务语义
- 阶段 B 用判别式读出满足效率要求

---

## 7. V1 模型输入与输出协议

### 7.1 输入

V1 模型输入为：

- 单条候选端点音频 clip
- 对应任务 prompt

注意：

- clip 必须是已经裁好的窗口音频
- 模型训练与推理都不依赖 `cut_time_ms` 这类在线裁剪字段
- 这些字段只保留在原始数据准备阶段

### 7.2 输出

部署模型统一输出：

- `label`
- `complete_prob`
- `incomplete_prob`
- `confidence`

推荐 dataclass 设计：

```python
@dataclass
class TurnDetectionPrediction:
    label: str
    complete_prob: float
    incomplete_prob: float
    confidence: float
    logits: Optional[List[float]] = None
    latency_ms: Optional[float] = None
```

其中：

- `confidence = max(complete_prob, incomplete_prob)`

### 7.3 推理接口建议

后续编码建议提供：

- `predict(audio: AudioLike, prompt: Optional[str] = None) -> TurnDetectionPrediction`
- `predict_batch(audio: List[AudioLike], prompt: Optional[List[str]] = None) -> List[TurnDetectionPrediction]`

不要要求调用方再传 `cut_time_ms` 来做在线裁切。裁切工作应该在数据准备和服务入口层完成。

---

## 8. V2 backchannel 扩展方案

V2 在数据结构不变的前提下扩展标签：

- `complete`
- `incomplete`
- `backchannel`

但推荐仍采用 **层级建模**，而不是简单平面三分类：

1. 先判 `complete / incomplete`
2. 只在 `complete` 中再判 `backchannel / non-backchannel`

原因：

- `backchannel` 往往本身是“说完了”的
- 它和 `complete` 的区别在于“是否接管轮次”，不是单纯边界问题

如果后续实践证明 `backchannel` 特别难做，可以保留为：

- V2 离线分析任务
- 不立即纳入线上正式路由

---

## 9. 推荐的后续编码拆分

为了避免直接污染现有旧实验代码，建议后续编码按新文件落地。

### 9.1 数据准备

- `finetuning/prepare_turn_detection_dataset.py`

职责：

- 从原始会话数据和候选端点清单中裁 clip
- 导出最终 `train/dev/test` JSONL
- 生成分析 slice manifest

### 9.2 阶段 A 训练脚本

- `finetuning/qwen3_turn_detection_sft_v1.py`

职责：

- 复用 `qwen3_asr_sft.py` 的训练逻辑
- 使用 turn detection verbalizer 数据训练纯净 `Qwen3-ASR`

### 9.3 阶段 B 模型定义

- `qwen_asr/turn_detection/qwen3_turn_detector_v1.py`

职责：

- 加载阶段 A backbone
- 定义轻量分类头
- 实现 `predict / predict_batch`

### 9.4 阶段 B 训练脚本

- `finetuning/qwen3_turn_detection_distill_v1.py`

职责：

- 使用统一 manifest 中的 `label`
- 训练分类头
- 可选接入教师蒸馏

### 9.5 评估脚本

- `finetuning/eval_qwen3_turn_detection_v1.py`

职责：

- 计算分类指标
- 输出阈值分析
- 输出 slice 指标

---

## 10. V1 推荐实验矩阵

### 10.1 阶段 A

- A1: 只训 `lm_head`
- A2: `lm_head + thinker top-2`
- A3: `lm_head + thinker top-4`

### 10.2 阶段 B

- B1: 只训分类头
- B2: 分类头 + thinker top-2
- B3: 分类头 + 蒸馏

推荐先跑：

- A2
- A3
- B1
- B3

---

## 11. 验收标准

V1 完成后，至少应满足：

1. 在主测试集上形成稳定的 `complete/incomplete` 判别能力
2. 在短句、犹豫停顿、噪声场景上不过度退化
3. 能输出稳定概率分布，支持阈值调节
4. 推理效率明显优于生成式上线方案

如果这些目标还未达到，不应进入 V2 的 `backchannel` 扩展。

---

## 12. 设计结论

最终设计结论如下：

1. 后续 turn detection 编码应从纯净 `Qwen3-ASR` 主干出发
2. 最终数据结构与 `Qwen3-ASR` SFT 一致，主字段为 `audio + text + prompt`
3. V1 先做 `complete / incomplete`
4. 推荐采用“两阶段方案”：
   - 阶段 A：生成式 verbalizer 适配
   - 阶段 B：高效分类头部署
5. `backchannel` 放到 V2，在 V1 稳定后再决定是否并入正式模型

如果后续编码遵守以上约束，就能在最大复用 `Qwen3-ASR` 预训练能力的前提下，以较少数据训练出一个高效、可用、可扩展的轮次检测模型。
