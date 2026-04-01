# Qwen3-ASR 四分类 Turn-Taking 训练方案规划（Label-First 优先）

## 1. 文档目的

本文档用于规划当前想到的 Qwen3-ASR 四分类 turn-taking 训练方案。

当前结论是：

- 优先推进 `label-first` 版本；
- 先尽量复用当前 Qwen3-ASR 的生成式 SFT 路径，不改模型结构；
- 主要通过 **任务协议、目标文本格式、prompt 设计、数据构造、结果解析与评估口径** 来完成任务对齐；
- 其余两个方案先作为后续对照/扩展方向记录，暂不展开实现细节。

这里的四分类标签定义为：

- `<complete>`：语义完整，当前用户发言可以视作结束
- `<incomplete>`：语义未完整，用户仍可能继续说
- `<backchannel>`：简短附和、回应性反馈，通常不应触发完整主响应
- `<wait>`：请求暂停、先别回应、终止当前交流或要求继续等待


## 2. 当前总体推进分三阶段

### 阶段 1：`label-first` 离线联合生成基线

主目标：

- 输入音频
- 输出 `language + turn_state + transcript`
- 先生成 turn label，再生成 ASR 文本

推荐输出协议：

```text
language Chinese<turn_state><complete><asr_text>你好
language English<turn_state><incomplete><asr_text>I want to ask...
```

这是当前优先落地的主方案，本文档将详细展开。

### 阶段 2：`label-last` 离线联合生成对照

主目标：

- 保持相同训练数据、相同模型、相同训练脚本
- 仅把输出顺序改成 transcript 在前，label 在后

示例：

```text
language Chinese<asr_text>你好<complete>
```

这个方案后续主要用于和 `label-first` 做对照实验，验证：

- turn label 放前面是否更稳
- turn label 是否更不容易被长 transcript 稀释
- transcript-first 是否对某些类别更有帮助

当前先不详细展开。

### 阶段 3：Prefix-Supervision 流式版本

主目标：

- 面向实时流式输入
- 对同一条 utterance 的多个前缀版本做联合监督
- 允许 turn label 随音频前缀推进而变化

示例：

```text
t1: language Chinese<turn_state><incomplete><asr_text>我想
t2: language Chinese<turn_state><incomplete><asr_text>我想问一下
t3: language Chinese<turn_state><complete><asr_text>我想问一下今天上海天气
```

这个方案后续再详细规划。当前先把离线 `label-first` 跑通，再决定是否进入流式专项版本。


## 3. 为什么当前优先 `label-first`

当前优先 `label-first`，不是因为它“绕开了 LLM”，而是因为它更适合当前这个联合生成任务。

核心理由如下：

1. Qwen3-ASR 在生成第一个标签 token 时，已经看到了整段音频和 prompt，不是“还没理解语音就盲猜标签”。
2. 把标签放在 transcript 前面，可以减少“长 transcript 把少量标签 token 淹没”的问题。
3. 把标签放在前面，可以避免推理时标签强依赖模型自己先生成出的 transcript，从而减小 exposure bias。
4. 如果未来要走流式版本，`label-first` 更容易扩展成“先输出当前状态，再持续刷新转写”的交互形态。

因此，当前推荐的理解方式不是：

> 先有文字，再靠文字能力判断标签

而是：

> 模型先基于音频隐表示做 turn-state 判断，再继续生成 transcript。


## 4. 与当前仓库训练路径的兼容性结论

基于当前仓库实现，`label-first` 方案可以尽量沿用现有训练路径。

### 4.1 可直接复用的部分

- `finetuning/qwen3_asr_sft.py`
- 当前 JSONL 数据读取方式
- 现有 chat-template 前缀构造逻辑
- 现有生成式 SFT 训练方式

### 4.2 当前不需要改模型结构

当前版本不计划新增：

- 分类头
- 边界头
- 额外 encoder
- 多任务 loss head

当前版本先只做：

- 输出协议改造
- prompt 改造
- 数据准备与标签规范
- 解析器扩展
- 评估脚本扩展

### 4.3 当前需要修改的主要位置

后续实现时，预计主要会改下面几类内容：

1. 训练数据格式与数据转换脚本
2. prompt 默认文案与 prompt 采样池
3. 推理输出解析器
4. 评估脚本


## 5. Label-First 主方案定义

## 5.1 任务定义

输入：

- 单条音频

输出：

- 语种信息
- turn-taking 四分类标签
- 对应 transcript

推荐统一输出格式：

```text
language {LANG}<turn_state><LABEL><asr_text>{TRANSCRIPT}
```

其中：

- `{LANG}` 取当前仓库已有的语言格式，例如 `Chinese`、`English`
- `<turn_state>` 作为 turn 标签的固定锚点
- `<LABEL>` 只能从以下四类中选择：
  - `<complete>`
  - `<incomplete>`
  - `<backchannel>`
  - `<wait>`
- `{TRANSCRIPT}` 是标准 ASR 文本

### 5.2 为什么推荐加 `<turn_state>` 锚点

不建议只写成：

```text
language Chinese<complete><asr_text>你好
```

更推荐写成：

```text
language Chinese<turn_state><complete><asr_text>你好
```

原因：

1. `<turn_state>` 可以把 turn label 和普通文本明确分隔开，减少 parser 歧义。
2. 后续如果扩展更多状态或附加字段，协议更稳定。
3. 训练时它给模型一个明确的“接下来要输出的是状态标签”的信号。
4. 对离线和未来流式版本都更统一。


## 6. 数据准备方案

## 6.1 推荐训练样本字段

当前推荐先准备如下字段：

```json
{
  "audio": "/path/to/audio.wav",
  "language": "Chinese",
  "transcript": "你好",
  "turn_label": "complete",
  "text": "language Chinese<turn_state><complete><asr_text>你好"
}
```

其中：

- `audio`：音频路径
- `language`：规范化后的语言名
- `transcript`：目标转写文本
- `turn_label`：四分类标签的字符串形式
- `text`：最终喂给当前 SFT 脚本的目标文本

注意：

- 当前 SFT 脚本真正强依赖的是 `audio + text`
- 其余字段主要用于数据检查、评估和后续分析

### 6.2 标签规范

建议统一使用小写标签名，严格固定为：

- `complete`
- `incomplete`
- `backchannel`
- `wait`

映射到目标文本时，固定转成：

- `<complete>`
- `<incomplete>`
- `<backchannel>`
- `<wait>`

不建议在训练集里混用大小写或同义词。

### 6.3 transcript 规范

建议：

1. 尽量保留原始转写风格，不额外做过重文本清洗。
2. 同一批数据内统一中文标点或英文标点风格。
3. 对明显非语音噪声标记、人工注释说明、质量标签等内容做剔除。
4. 不把 turn label 写进 transcript 字段本体，避免重复污染。

### 6.4 类别分布与采样建议

四分类任务里，`backchannel` 和 `wait` 往往显著少于 `complete / incomplete`。

因此当前版本推荐：

1. 先统计四类样本数与总时长。
2. 训练时对尾类做重采样或 batch-level 平衡采样。
3. 开发集、测试集保持尽量贴近真实分布，不要过度人工平衡。

建议至少输出下列数据统计：

- 四类样本数
- 四类总时长
- 四类平均时长
- 不同语言的类分布
- 不同说话人/场景/设备的类分布

### 6.5 数据切分建议

推荐按以下优先级切分训练/验证/测试：

1. 先按 session 或会话切分，避免泄漏
2. 再按 speaker 切分，避免说话人过拟合
3. 最后检查四类标签分布和语言分布

不建议简单随机切分整表。

### 6.6 当前版本建议先准备的 hard cases

即使当前主方案是生成式联合输出，仍建议在开发/测试集中显式覆盖这些样本：

- 短暂停顿但未说完
- 句尾拖音
- 自我修正
- 重启
- 列举中断
- 简短附和语
- “等一下 / 先别回 / 稍后再说”这类 wait 类语义


## 7. Label-First 的 prompt 设计

## 7.1 设计原则

prompt 的目标不是让模型“靠提示词记答案”，而是把任务格式说清楚，同时减少 prompt 依赖。

当前建议：

1. 输出协议固定
2. prompt 用小规模模板池随机采样
3. 不要无限制地写很多风格差异极大的 prompt
4. 先保证格式稳定，再逐步增加表述多样性

### 7.2 输出协议要求

无论 prompt 如何变化，输出格式都固定为：

```text
language {LANG}<turn_state><LABEL><asr_text>{TRANSCRIPT}
```

重点是：

- 输出顺序固定
- 标签集合固定
- 协议 token 固定

### 7.3 推荐的 prompt 模板池

下面这些模板参考了当前已有的“转写 + 末尾分类”提示风格，但统一改写为 `label-first` 任务形式。

#### 模板 A

请先判断这段音频在结尾处的打断状态，再输出转写文本。输出格式必须为：`language 语种<turn_state><标签><asr_text>转写文本`。标签只能是 `<complete>`、`<incomplete>`、`<backchannel>`、`<wait>` 之一。

#### 模板 B

请先给出当前音频对应的打断类型，再给出完整转写。使用固定格式输出：`language 语种<turn_state><标签><asr_text>文本`。其中 `<complete>` 表示语义完整，`<incomplete>` 表示语义不完整，`<backchannel>` 表示附和语句，`<wait>` 表示请求暂停或先不要继续对话。

#### 模板 C

请基于音频内容先判断用户当前发言的 turn-taking 状态，再输出识别文本。请严格使用：`language 语种<turn_state><标签><asr_text>文本`。标签四选一：`<complete>`、`<incomplete>`、`<backchannel>`、`<wait>`。

#### 模板 D

请先输出当前语音片段的交互状态标签，再输出文字转写。输出时先写语种，再写 `<turn_state>` 和状态标签，最后写 `<asr_text>` 与识别文本。`<complete>` 为语义完整，`<incomplete>` 为语义未完整，`<backchannel>` 为简短附和，`<wait>` 为请求暂停、终止或先别回应。

#### 模板 E

请识别音频内容，并在转写前先标注当前片段的打断状态。固定输出协议为：`language 语种<turn_state><标签><asr_text>文本`。请勿输出额外解释，标签只能从 `<complete>`、`<incomplete>`、`<backchannel>`、`<wait>` 中选择。

### 7.4 prompt 使用策略

当前版本建议：

- 先使用 4 到 8 条模板即可
- 训练阶段对模板随机采样
- 验证和测试阶段固定使用 1 到 2 条主模板做稳定评估

不建议一开始就把 prompt 变体做得非常多。

原因：

1. 当前主要目标是先把任务协议训稳
2. prompt 变化过多容易引入额外噪声
3. 当前我们更需要验证 `label-first` 协议本身，而不是 prompt 泛化上限


## 8. Label-First 的训练思路

## 8.1 核心训练目标

当前版本采用单一生成式目标：

- 模型输入音频
- 模型生成 `language + turn_state + transcript`
- 全部目标 token 统一使用 CausalLM loss

当前版本先不引入：

- 单独标签 loss
- 多任务加权
- 额外分类头
- 蒸馏项

这样做的目的不是证明“复杂方案没用”，而是先验证：

> 在 Qwen3-ASR 上，仅靠任务协议重对齐，能否把四分类 turn-taking 和 transcript 联合学出来。

### 8.2 为什么当前版本先不额外改结构

原因有三点：

1. 当前仓库已经有成熟的生成式 SFT 路径，改动最小。
2. 先跑结构不变版本，最容易判断任务协议本身是否成立。
3. 如果这个版本已经有效，后续所有增强版都可以在它上面继续迭代。

### 8.3 当前推荐的训练策略

当前推荐按下面顺序推进。

#### 步骤 1：先做最小可运行版

只做：

- 固定输出协议
- 小规模 prompt 池
- 标准生成式 SFT
- 标准离线评估

先回答一个最核心问题：

> Qwen3-ASR 在不改结构的前提下，是否能稳定输出合法协议，并在四分类上学到可用结果。

#### 步骤 2：做尾类采样增强

如果发现：

- `backchannel`
- `wait`

明显偏弱，则优先做：

- 重采样
- batch 平衡
- 数据配比调节

而不是立刻改模型结构。

#### 步骤 3：控制 transcript 能力漂移

如果发现 turn label 提升但 transcript 退化明显，则再考虑：

- 混入一小部分纯 ASR 数据
- 或者在同一四分类数据上做更严格的文本清洗与语言约束

当前版本先不默认引入额外 ASR 数据，但把这条作为第一优先级补救手段记录下来。


## 9. 推荐训练样本示例

### 9.1 中文 complete

```json
{
  "audio": "/data/utt001.wav",
  "prompt": "请先判断这段音频在结尾处的打断状态，再输出转写文本。输出格式必须为：language 语种<turn_state><标签><asr_text>转写文本。标签只能是 <complete>、<incomplete>、<backchannel>、<wait> 之一。",
  "text": "language Chinese<turn_state><complete><asr_text>我想问一下今天上海的天气。"
}
```

### 9.2 中文 incomplete

```json
{
  "audio": "/data/utt002.wav",
  "prompt": "请先给出当前音频对应的打断类型，再给出完整转写。使用固定格式输出：language 语种<turn_state><标签><asr_text>文本。",
  "text": "language Chinese<turn_state><incomplete><asr_text>我想问一下今天上海"
}
```

### 9.3 中文 backchannel

```json
{
  "audio": "/data/utt003.wav",
  "prompt": "请基于音频内容先判断用户当前发言的 turn-taking 状态，再输出识别文本。请严格使用：language 语种<turn_state><标签><asr_text>文本。",
  "text": "language Chinese<turn_state><backchannel><asr_text>嗯嗯。"
}
```

### 9.4 中文 wait

```json
{
  "audio": "/data/utt004.wav",
  "prompt": "请先输出当前语音片段的交互状态标签，再输出文字转写。",
  "text": "language Chinese<turn_state><wait><asr_text>你先别回我，我还没说完。"
}
```


## 10. 训练实现建议

## 10.1 训练入口

当前建议直接基于现有生成式训练脚本实现：

- `finetuning/qwen3_asr_sft.py`

当前版本先不新建第二套完全独立的训练框架。

### 10.2 训练输入不变

仍然使用：

- `audio`
- `text`
- 可选 `prompt`

这意味着当前版本的数据转换成本较低。

### 10.3 推理解析需要扩展

当前仓库默认解析的是：

```text
language Chinese<asr_text>你好
```

后续需要扩展成能解析：

```text
language Chinese<turn_state><complete><asr_text>你好
```

解析结果建议至少返回：

- `language`
- `turn_label`
- `text`
- `raw_text`
- `is_valid_schema`

### 10.4 结果合法性校验

建议在推理后做协议校验：

1. 是否包含 `language`
2. 是否包含 `<turn_state>`
3. 是否包含且仅包含一个合法标签
4. 是否包含 `<asr_text>`
5. transcript 是否为空

这套校验会成为后续评估的一部分。


## 11. 对齐效果怎么评判

当前版本的“对齐效果”不能只看 WER/CER，也不能只看四分类 accuracy。

必须同时看下面三层。

## 11.1 协议对齐

这层评估模型有没有学会我们定义的输出协议。

建议指标：

- schema valid rate
- turn tag parse success rate
- 非法输出率
- 多标签冲突率
- 漏标签率

只要协议不稳，后续 transcript 和 turn label 的结果都不可靠。

## 11.2 Turn label 对齐

这层评估模型有没有学会 turn-taking 四分类。

建议主指标：

- overall accuracy
- macro F1
- per-class precision / recall / F1

建议重点盯：

- `backchannel` F1
- `wait` F1

因为这两类通常最难，也是最容易被主流类掩盖的。

## 11.3 Transcript 对齐

这层评估联合训练后 transcript 有没有保持住。

建议指标：

- CER
- WER
- 按类别分组的 CER / WER
- label 正确条件下的 CER / WER
- label 错误条件下的 CER / WER

特别建议看：

- 当 turn label 正确时，文本质量如何
- 当 turn label 错误时，文本是否仍稳定

这样可以判断模型错误到底是“状态没学会”，还是“整体识别能力都漂了”。

## 11.4 端到端联合指标

建议再加一组联合指标，评估真正业务可用性。

推荐：

- `joint_exact_match`
  - 语言、turn label、transcript 同时完全正确
- `label_correct_text_close`
  - 标签正确且文本 CER / WER 在阈值内
- `text_correct_label_wrong`
  - 文本正确但标签错误

这一层最能反映“联合输出协议”到底值不值得继续做。


## 12. 当前版本的风险与注意事项

### 12.1 不要把标签语义和策略语义混淆

虽然四分类里有 `wait`，但标签定义仍应保持清晰，避免把过多上层策略因素揉进训练数据。

如果 `wait` 的定义过宽，模型会学成模糊的“系统该怎么办”分类器，而不是稳定的 speech + intent 边界模型。

### 12.2 不要让 prompt 风格过度发散

prompt 变体过多时，模型可能先学 prompt 风格，再学任务协议。

当前版本先以小规模模板池为宜。

### 12.3 不要忽视类不平衡

如果训练数据中 `backchannel` 和 `wait` 极少，单纯跑一遍 SFT 往往会得到：

- transcript 还行
- `complete / incomplete` 还行
- 少数类很差

因此尾类采样是当前版本的首要增强方向。

### 12.4 不要只看 overall accuracy

四分类如果分布偏斜，overall accuracy 很容易虚高。

当前版本必须同时看：

- macro F1
- per-class F1
- 协议合法率
- transcript 指标


## 13. 当前推荐实施顺序

### 第一步：数据清洗与协议定稿

完成：

- turn label 规范化
- 语言字段规范化
- transcript 清洗
- 目标文本协议统一

### 第二步：准备 prompt 模板池

完成：

- 固定输出协议
- 选定 4 到 8 条主模板
- 确定训练时的 prompt 采样方式

### 第三步：跑最小可运行版

完成：

- 直接基于 `qwen3_asr_sft.py` 训练
- 扩展推理 parser
- 做第一轮离线评估

### 第四步：针对尾类和协议稳定性做增强

视结果决定是否追加：

- 尾类重采样
- schema 约束增强
- 小比例纯 ASR 数据混入


## 14. 阶段 2 与阶段 3 的简要备注

## 14.1 阶段 2：Label-Last 对照

后续只需要回答一个核心问题：

> transcript-first 是否比 label-first 更适合当前四分类任务

它的价值主要在于做对照，而不是当前主线。

## 14.2 阶段 3：Prefix-Supervision 流式版本

后续需要重点回答：

1. 同一条 utterance 的不同前缀标签如何标注
2. turn label 如何随时间变化
3. 流式输出如何做平滑与回滚

这是未来真正走实时系统时的重要方向，但当前先不展开。


## 15. 当前最终建议

当前最值得优先实现的版本是：

> **不改 Qwen3-ASR 模型结构，沿用现有生成式 SFT 路径，把任务协议改成 `language + turn_state + transcript` 的 label-first 联合生成方案。**

更具体地说：

- 先做 `label-first`
- 先不改结构
- 先不加额外 loss
- 先验证协议是否能训稳
- 重点盯协议合法率、四分类 macro F1、尾类 F1 和 transcript 质量

如果这个版本表现成立，再进入：

- `label-last` 对照
- `prefix-supervision` 流式扩展

