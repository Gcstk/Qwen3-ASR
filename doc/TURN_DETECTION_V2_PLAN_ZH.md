# Qwen3-ASR Turn Detection V2 规划

## 1. 文档目的

这份文档的目标不是直接给出实现代码，而是把 **Turn Detection V2** 的设计思路、训练目标、数据要求、验证方法和取舍理由说明清楚，作为后续实现与评审的依据。

V2 的核心目标是：

- 在线推理时仍然只输出 `complete / incomplete` 二分类结果；
- 尽量继承和放大 Qwen3-ASR 底座已经具备的音频理解、语义建模、指令跟随和多模态融合能力；
- 在只有几百小时标注音频的前提下，不把方案做成“只训练一个浅分类头”的弱迁移，而是把 backbone 有控制地往 turn detection 任务上拉；
- 兼顾低延迟上线需求，不把线上推理退化成慢速生成式方案。


## 2. 任务背景与当前 V1 的真实定义

### 2.1 当前目标

这里的 turn detection 不是传统离线句子切分，也不是外部 VAD 子模块，而是：

- 输入一段用户语音；
- 判断这段语音本身是否已经构成一段完整用户发言；
- 如果已经说完，就可以交给 agent 做下一步动作；
- 如果没有说完，就继续等待更多语音。

从业务上，它更接近：

- `speech completion detection`
- `endpoint verification`
- `end-of-utterance classification`

而不是完整的对话策略决策。

### 2.2 对输入接口的纠正

V2 的外部接口定义需要进一步收紧为：

- 输入一段已经给定的用户音频；
- 模型直接判断这段音频是否已经表达完整；
- 不把“候选结束点”“切分点”“外部 VAD 活动段”写入模型任务定义；
- 不把音频切分本身当成这个模型要解决的问题。

同时需要明确：

- 如果模型内部后续显式利用 pause、静音、边界感知或类似 VAD 的表征辅助判断，这是允许的；
- 但这些能力只能作为内部辅助信号，不能改变模型的任务边界；
- 模型对外始终定义为“输入一段音频，输出这段音频是否已经说完”。

这点非常重要，因为它意味着当前样本已经天然包含：

- 整段发声的韵律演化；
- 尾部拖音、犹豫、重启、自我修正；
- 句内停顿和最终收束之间的区别。

因此，V2 应该定义成：

> **对一段完整输入音频做 utterance-level 完整性判断，同时在训练中显式学习边界相关的局部时间结构。**


## 3. 当前仓库 V1 的结构回顾

基于仓库代码，当前 V1 的 turn detection 方案本质上是：

`Qwen3-ASR backbone + pooling + classifier`

关键事实如下：

1. 它确实复用了 Qwen3-ASR 的音频 encoder 和 thinker，而不是只借了 processor。
2. 它不是生成文本标签，而是直接从 hidden states 池化后做二分类。
3. 默认冻结大部分 backbone，只训练分类头；必要时再解冻高层。
4. 当前 V1 代码层既支持整段音频，也支持围绕 `cut_time_ms` 裁子窗口；但 V2 的任务规划不再把切分或候选点作为前提。

换句话说，V1 已经证明：

- 当前仓库能够把 Qwen3-ASR 的多模态 backbone 直接改造成判别式模型；
- 这条路是成立的；
- 但它目前还更像一个 **feature extractor + probe head**，而不是充分利用底座全部能力的 turn-detect 专项模型。


## 4. 为什么 V1 不够，V2 需要解决什么

当前 V1 最大的问题不在于“能不能做”，而在于“能做到什么程度”。

### 4.1 当前不足

#### 1. 只有 utterance-level 二分类，没有稠密边界监督

V1 只知道一段 clip 最终是 `complete` 还是 `incomplete`，但它并不知道：

- 这段语音的哪里开始显现 turn-yielding cue；
- 哪些 token/时间区域最像真正的结束边界；
- 为什么两个都带停顿的片段，一个是“已经说完”，另一个只是“还在想”。

这会导致模型只能从整体 embedding 里“猜”，而不能显式学习局部边界模式。

#### 2. 没有显式利用底座的 ASR / 标点 / 语义能力

Qwen3-ASR 底座本来已经学到：

- 语音到文本的对齐和识别；
- 语义收束与句法闭合；
- 标点、停顿、韵律和语义之间的对应关系；
- 多轮指令风格下的音频理解。

但 V1 的训练目标只有一个交叉熵分类 loss。这样做有两个问题：

- 它没有显式告诉模型哪些 ASR/语义能力对 turn detection 有帮助；
- 它也没有限制模型在小数据上为了优化分类而破坏底座已有能力。

#### 3. 默认冻结太多时，只能“读特征”，很难“拉主干”

只训练分类头时，模型更像是做 linear probe：

- 好处是稳；
- 代价是上限受限。

如果样本只有几百小时，这个策略适合当 baseline，但不适合当最终版本，因为：

- turn detection 不是和原始 ASR 完全同构的任务；
- 它需要模型更偏向 utterance completion、句尾韵律和语义闭合；
- 这些偏好仅靠顶层线性头很难完全学出来。

#### 4. 语音边界和对话策略还没有明确分层

用户是否“说完了”和 agent 是否“应该马上回复”，并不完全是同一个问题。

例如：

- “你等一下，我还没说完” -> 往往是 `incomplete`
- “你先别回我，我说完了” -> 往往是 `complete`

V1 文档已经意识到了这个问题，但还没有把边界分类、语义完整、策略等待三者的边界写得足够清楚。

#### 5. 评估维度还不够贴近上线

当前指标主要是：

- accuracy
- complete precision / recall / f1

这不足以回答真正的上线问题：

- 相同“不要抢答”的约束下，能多早交给 agent？
- 对犹豫停顿、半句自修正、尾部拖音、列举句的鲁棒性如何？
- 不同语言、不同说话风格、不同麦克风条件下是否稳定？


## 5. V2 的核心设计原则

V2 的设计不追求“结构上最花哨”，而追求在当前仓库和数据约束下，最大化复用底座能力并保住上线速度。

### 原则 1：线上保持判别式，训练阶段允许多任务

这是 V2 最重要的原则。

你的线上需求很明确：

- 推理必须快；
- 最终输出必须是二分类；
- 暂时不把线上接口做成生成式标签或自由文本解释。

因此，V2 的线上形态必须仍然是：

- 单次前向；
- 得到二分类 logits；
- 输出 `complete_prob` / `incomplete_prob`。

但训练阶段不应该被这个限制绑死。为了最大化利用 Qwen3-ASR 底座，训练时应允许引入：

- ASR 辅助监督；
- 标点/语义收束辅助监督；
- 边界定位辅助监督；
- teacher 保持项或蒸馏项。

也就是说：

> **V2 不是“生成式推理”，但应当是“多任务训练、判别式推理”。**

### 原则 2：主任务定义必须贴合真实输入

V2 的主任务不是“做音频切分”或“从一个很短的局部裁窗判断 pause 类型”，而是：

- 对一段完整输入音频判断，用户在这段音频里是否已经完成了这次 utterance。

因此主标签必须围绕“整段音频是否完整”定义，而不是围绕外部候选切分点定义。

### 原则 3：利用底座已有能力，不和底座对着干

Qwen3-ASR 的优势不只是声学编码器，还包括：

- 大规模语音训练数据学到的跨语言稳健性；
- Qwen3-Omni 继承来的音频理解能力；
- 统一多模态输入协议；
- thinker 对音频和文本条件联合建模的能力。

V2 不应抛弃这些能力，而应让 turn detection 目标顺着这些能力去长。

### 原则 4：小数据条件下优先做“强监督构造”，不是盲目解冻

几百小时是珍贵但仍然有限的数据量。V2 不应一上来做全量解冻，而应依赖：

- 更强的数据构造；
- 多任务辅助；
- 受控分阶段解冻；
- 蒸馏/保持项减少遗忘。


## 6. 公开资料对 V2 方向的支撑

下面这些资料并不直接给出“Qwen3-ASR 做 turn detection”的现成答案，但它们对 V2 的关键设计提供了清晰支撑。

### 6.1 Qwen3-ASR 官方模型卡

Qwen3-ASR 官方模型卡明确指出：

- 它建立在大规模语音训练数据之上；
- 并且继承了 Qwen3-Omni 的强音频理解能力。

这说明把它用作单独的 end-of-utterance / turn-completion backbone 在前提上是合理的，不是在拿一个“纯转写器”硬拗做分类。

来源：

- https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- https://huggingface.co/Qwen/Qwen3-ASR-1.7B

### 6.2 Google 2022：联合 ASR 与 endpointing

Google 在 *Unified End-to-End Speech Recognition and Endpointing for Fast and Efficient Speech Systems* 中指出：

- 外挂 endpointer 往往受计算预算限制，质量不稳定；
- 联合训练 ASR 和 endpointing 可以显著降低 latency；
- endpointing 可以直接消费 ASR encoder 的 latent representations；
- 在降低端点延迟的同时不回退 WER。

这直接支持两件事：

1. endpointing / turn completion 应该尽量消费 ASR 主干的内部表示；
2. 单独二分类头可以存在，但更好的做法是和 ASR 能力一起训练，而不是彻底切断。

来源：

- https://research.google/pubs/unified-end-to-end-speech-recognition-and-endpointing-for-fast-and-efficient-speech-systems/

### 6.3 Google 2019：joint endpointing and decoding

在 *Joint Endpointing and Decoding with End-to-End Models* 中，Google 进一步说明：

- 把 endpointing 作为 ASR 的独立外部模块，会在 WER 和 latency 间形成额外折中；
- 和 E2E ASR 联合优化后，可以明显降低 latency，且不伤质量。

这支持 V2 的一个核心立场：

> turn detection 虽然线上是二分类，但训练上不应和底座完全切割。

来源：

- https://research.google/pubs/joint-endpointing-and-decoding-with-end-to-end-models/

### 6.4 Semantic VAD：语义监督比纯静音规则更适合低延迟交互

*Semantic VAD: Low-Latency Voice Activity Detection for Speech Interaction* 提出：

- 传统 VAD 往往需要等待固定尾静音，导致延迟大；
- 引入 frame-level punctuation prediction 和 ASR-related semantic loss 后，可以在不明显恶化后端 CER 的情况下显著降低延迟。

这和 V2 非常一致：

- 仅看“有没有停顿”是不够的；
- 语义闭合、标点倾向、句尾结构都能帮助更早判断“是不是说完了”；
- 因此 V2 应该把 ASR/标点/语义作为辅助训练信号，而不是只做裸分类。

这里引用 Semantic VAD 的重点不是要把 V2 定义成 VAD 模型，而是说明：

- 即使最终接口是“输入一段音频，判断是否完整”，
- 模型内部仍然可以利用 pause、边界和语义闭合相关信号辅助判断。

来源：

- https://www.isca-archive.org/interspeech_2023/shi23c_interspeech.html

### 6.5 长期 turn-taking 研究：韵律、时长和语义需要联合使用

长期的 turn-taking / prosody 研究反复表明，turn 完结信号不只来自静音，也来自：

- final-word lengthening
- pitch movement
- intensity 变化
- speech rate
- lexical / syntactic / semantic cue

这类结果说明，V2 不应过于依赖单一的 pause pattern，而应让模型同时使用：

- 声学韵律
- ASR 语义
- 句法/标点闭合

来源示例：

- https://www.sciencedirect.com/science/article/pii/S0167639320302727
- https://www.isca-archive.org/interspeech_2019/razavi19_interspeech.html
- https://pubmed.ncbi.nlm.nih.gov/19122857/


## 7. V2 的总体方案

### 7.1 一句话定义

V2 采用：

> **二分类线上推理 + 多任务训练 + 分阶段解冻 + 强 hard-negative 数据构造**

### 7.2 方案目标

V2 想解决的不是“能不能把 ASR 模型换成分类模型”，而是：

- 如何在几百小时数据下，尽可能保住底座已有能力；
- 如何让模型学到“说完了”的语音、韵律和语义证据；
- 如何在不增加明显推理延迟的前提下提升完成度判断质量。


## 8. V2 推荐的模型形态

### 8.1 线上模型输出

线上只保留一个主输出：

- `complete`
- `incomplete`

输出字段建议固定为：

- `label`
- `complete_prob`
- `incomplete_prob`
- `confidence`

其中 `confidence` 可以是 `max(prob)` 或校准后的 margin，但不要求线上输出解释文本。

### 8.2 训练时的内部头部设计

V2 建议保留 backbone 主体不变：

- `audio_tower`
- `thinker`
- 原始 `lm_head`

在其上增加三个训练相关头：

#### 1. 主分类头：Utterance Completion Head

输入：

- 多模态融合后的最后层 hidden states

作用：

- 直接预测整段 clip 在末尾时是否已经完成 utterance

这是线上真正使用的头。

#### 2. 稠密边界头：Boundary Head

输入：

- 音频对应位置的 hidden states，或者音频末尾一段 token 的 hidden states

作用：

- 预测哪些时间区域接近“真正完整表达的结束边界”

这个头主要服务训练，不强制上线使用。

#### 3. 辅助语义头：ASR / punctuation / semantic closure head

作用：

- 让模型保留并利用原始 ASR 与句尾语义能力；
- 防止模型为了二分类而只学简单的尾静音模式。

这个头不进入线上主推理路径，但其损失应参与训练。


## 9. 为什么 V2 不能只做“更强的二分类头”

一个看似简单的方案是：

- 不改训练目标；
- 只把 `last_token` pooling 换成 attention pooling；
- 再多解冻几层。

这个方案有一定提升空间，但仍然不够，原因如下：

### 9.1 它没有告诉模型“完整性”背后的结构

`complete / incomplete` 是结果标签，不是机制标签。

模型需要区分的其实是：

- 句尾韵律收束；
- 语义是否闭合；
- 是否只是犹豫停顿；
- 是否准备继续列举；
- 是否处于自我修正或重启中。

单个二分类标签不足以把这些结构信息都稳定学出来。

### 9.2 它无法充分利用底座已有的 ASR 能力

Qwen3-ASR 已经非常擅长把语音映射到文本和语义结构。若训练时彻底丢掉这些监督，相当于放弃了最值钱的先验。

### 9.3 小数据条件下更容易过拟合表面模式

只做二分类时，模型很可能学到：

- 尾部静音长短；
- 某些说话人习惯；
- 某些录音条件下的 pause 形态；
- 数据采集流程里的伪特征。

而不是更本质的 utterance completion 规律。


## 10. V2 推荐的训练目标

V2 推荐采用“主任务 + 三类辅助任务”的结构。

### 10.1 主任务：utterance-level binary classification

目标：

- 对整段输入音频输出 `complete / incomplete`

loss：

- 标准交叉熵即可

这是主优化目标，也是线上唯一必须保留的输出。

### 10.2 辅助任务 A：边界定位监督

目的：

- 让模型学习“接近真正完整结束边界的局部形态”
- 减少只依赖整段全局 embedding 的问题

实现建议：

- 在音频 token 维度上做二值或软标签预测
- `complete` 样本：音频末尾附近一个容忍区间标为高分
- `incomplete` 样本：整段都不应出现强结束边界

这里不要求帧级人工精标到极致，建议优先使用：

- forced aligner
- word timestamp
- 音频末尾规则

构造一个“足够稳定”的 boundary supervision。

这里的 boundary supervision 只是帮助模型学习“完整表达通常如何收束”，不意味着要让该模型承担音频切分职责。

### 10.3 辅助任务 B：ASR 保持 / transcript 辅助

目的：

- 显式复用 Qwen3-ASR 的原始转写能力；
- 让模型的隐藏空间仍然对“已说出的内容”敏感；
- 避免模型退化成只盯尾静音的分类器。

建议做法：

- 对 `complete` 样本，用完整 transcript 做 teacher-forced loss
- 对 `incomplete` 样本，用截至 clip 末尾已说出的 transcript prefix 做 teacher-forced loss

这不会改变线上接口，但会让训练目标和底座能力更一致。

### 10.4 辅助任务 C：标点 / 语义闭合监督

目的：

- turn 完成经常对应句法/语义闭合；
- 尾部是否可能接句号、问号、感叹号或 EOS，本身就是强信号。

建议做法：

- 如果 transcript 带标点，则显式监督句尾标点类别；
- 如果 transcript 不稳定，可退化成 “是否语义闭合” 的二分类辅助标签；
- 也可以从原始 `lm_head` 的末尾 token 预测里抽取句尾 closure 相关损失。

### 10.5 可选辅助：teacher 保持项 / 蒸馏

如果后续实验发现 backbone 解冻后 ASR 能力明显漂移，可加入：

- 冻结 teacher（原始 Qwen3-ASR）
- student 学习 turn detection
- 同时对 teacher 的转写分布或隐藏状态做保持项

这个机制的目的不是“完全不变”，而是减少灾难性遗忘。


## 11. 为什么这些训练目标合理

### 11.1 它们与线上路径不冲突

线上仍然只需要一次前向和一个二分类头，不会因为训练用了辅助损失就被迫上线生成式推理。

### 11.2 它们与公开研究方向一致

联合 ASR + endpointing、语义 VAD、turn-taking 语义特征研究，都支持：

- 完整性判断不应只靠 pause；
- 深层语义和韵律信号可以显著改进低延迟交互场景。

### 11.3 它们最符合 Qwen3-ASR 的基础能力形态

Qwen3-ASR 的强项不是独立小音频分类器，而是：

- 音频编码；
- 多模态融合；
- 语义理解；
- ASR 条件生成。

V2 的训练目标就是要把这些强项转化成 turn detection 优势。


## 12. 数据设计建议

### 12.1 样本基本结构

V2 的推荐样本字段如下。

### 必需字段

- `audio`
- `label`

其中：

- `audio` 是一段完整输入音频
- `label` 是 `complete` 或 `incomplete`

### 强烈建议字段

- `transcript`
- `session_id`
- `speaker_id`

### 有条件时建议字段

- `word_timestamps`
- `true_end_ms`
- `prev_agent_action`
- `dialog_context_text`

### 字段含义建议

- `transcript`
  - 当前 clip 到末尾时，对应的已说出文本
- `true_end_ms`
  - 对 `complete` 样本而言，真实 utterance 结束点
- `word_timestamps`
  - 供边界辅助监督和截断样本自动构造使用
- `prev_agent_action`
  - 上一轮 agent 的动作，如 `replying`, `waiting`, `tool_calling`
- `dialog_context_text`
  - 轻量文本上下文，不是长历史缓存

### 12.2 样本来源建议

建议把训练样本分成三类。

### A. 真实完整样本

特点：

- 一整段用户发声自然结束；
- 末尾确实可以交给 agent。

作用：

- 提供真实的 positive 完结模式。

### B. 合成截断样本

从完整样本自动截出中间版本，构造成 `incomplete`：

- 在词边界后截断；
- 在短停顿后截断；
- 在逗号、连接词、列举项后截断；
- 在“我想… / 就是… / 然后… / 等下…”等犹豫位置后截断。

这是 V2 数据设计里最重要的一块，因为它能把有限标注扩充成大量高价值 hard negatives。

### C. 边界模糊样本

这类样本要单独提权：

- 长拖音
- 自我修正
- 中途重启
- 连续列举
- 说半句后短暂停顿
- 语义上结束但策略上不该立即回复

这类样本对最终实用效果的影响远大于普通 easy case。

### 12.3 为什么必须做合成截断

如果只收真实 `complete/incomplete` 样本，通常会出现两个问题：

1. `incomplete` 的难负样本比例不够高；
2. 模型容易把“末尾静音长短”当成捷径。

合成截断能强迫模型学会：

- 并不是所有接近句尾的停顿都代表结束；
- 句法未闭合、语义未闭合、列举未完成时，即使有停顿也不应交付。


## 13. 上下文该怎么接入

### 13.1 当前结论

基于当前任务定义，V2 不需要把外部切分信息作为主路径输入。模型面对的对象就是一段完整给定音频。

### 13.2 仍然建议加入的轻量上下文

如果后续数据里能提供上下文，V2 推荐只加入轻量文本上下文：

- 上一轮 agent 的动作
- 上一轮简短系统回复摘要
- 对话状态标签

推荐原因：

- 这类上下文能帮助区分“用户已经说完，但语义上要求等待”的边界情况；
- 它们走文本侧 prompt，不会显著增加音频推理成本；
- 与当前 Qwen3-ASR 的 chat-template 兼容。

### 13.3 不建议当前阶段引入的复杂机制

当前不建议一上来做：

- 长会话记忆模块
- 跨轮音频缓存建模
- 单独 speaker memory encoder
- 多阶段 online RL 式策略训练

原因很简单：

- 实现成本高；
- 数据要求高；
- 会把问题从“turn detection”扩散成“全对话策略建模”。


## 14. 解冻与训练策略建议

V2 不建议再沿用“默认只训分类头”的最终方案。

推荐三阶段训练。

### 阶段 A：稳定热启动

做法：

- 先冻结大部分 backbone；
- 训练主分类头、边界头和少量顶层模块；
- 打通多任务 loss。

目的：

- 保证训练稳定；
- 先让新增头部学会读取 backbone 表征；
- 避免一开始就破坏底座。

### 阶段 B：受控解冻

做法：

- 逐步解冻 thinker 顶层若干层；
- 再视数据规模和显存情况，解冻 audio_tower 的高层部分；
- 主任务和辅助任务共同训练。

目的：

- 真正把 backbone 向 turn detection 偏置；
- 学到更贴近任务的韵律和语义空间。

### 阶段 C：收敛与校准

做法：

- 降低 backbone 学习率；
- 提高主分类稳定性；
- 单独做阈值和概率校准。

目的：

- 为上线准备稳定、可控的打分输出。

### 为什么这样分阶段

因为你的数据量属于“足够做专项迁移，但不足以随便全量乱训”的区间。

分阶段策略的优势是：

- 比只训头部上限更高；
- 比全量解冻更稳；
- 更符合“保住底座能力 + 拉向 turn detection”的目标。


## 15. 推理路径建议

V2 的线上推理应保持极简。

### 输入

- 一段完整给定音频
- 可选轻量文本上下文

### 输出

- `complete_prob`
- `incomplete_prob`
- `label`

### 不建议的线上行为

- 不在线上调用 `generate()`
- 不让模型生成 `<complete>` / `<incomplete>` 文本标签
- 不在线上同时解码 transcript 再做二次判别，除非确实证明额外延迟可接受

### 原因

1. 二分类头更快；
2. 阈值更容易调；
3. 更适合接业务层 fallback 策略；
4. 与当前仓库的推理封装更容易兼容。

补充说明：

- 如果后续发现模型内部引入轻量 pause/VAD 风格辅助特征能提升判断质量，这属于内部实现优化；
- 但对外接口和任务定义仍然不应改成“先切分再判别”。


## 16. 为什么 V2 有望优于 V1

### 16.1 更强的任务对齐

V1 更像“读出 backbone 的已有特征”。

V2 更像“告诉 backbone 什么是 utterance completion，并用多个辅助任务把这个定义钉住”。

### 16.2 更好地利用了 Qwen3-ASR 的原始能力

V2 不只复用：

- 音频 encoder
- thinker hidden states

还复用：

- 原始转写能力
- 语义闭合能力
- 标点/句尾倾向
- 对话提示格式适配能力

### 16.3 对小样本更稳

多任务训练和 hard-negative 合成会让模型更难走表面捷径，从而提升泛化。

### 16.4 更符合低延迟交互系统实践

公开研究已经反复表明：

- 纯尾静音规则不够；
- 语义/句尾结构能显著改善低延迟 endpointer。

V2 正是沿着这个方向设计的。


## 17. 风险与注意事项

### 17.1 不要把任务定义漂移成策略判断

V2 的主标签应该始终围绕：

- 用户是否说完

而不是：

- agent 是否现在就回复

策略相关因素可以进辅助上下文，但不能混淆主标签定义。

### 17.2 transcript 质量会直接影响辅助监督价值

如果 transcript 噪声很大：

- ASR 辅助 loss 可能引入噪声；
- 标点/闭合监督可能不稳定。

因此需要对 transcript 质量做分层使用。

### 17.3 解冻过快可能破坏底座

如果一开始就全量解冻：

- 小数据下容易过拟合；
- 原始 ASR 表征可能被破坏；
- 训练不稳定。

### 17.4 数据采样偏差会造成“伪好结果”

例如：

- `complete` 样本都尾静音更长；
- `incomplete` 样本都来自固定场景；
- 某些设备/说话人只出现在一类里。

这样会导致离线指标虚高、上线效果差。


## 18. 验证与评估建议

V2 评估不能只看 accuracy。

### 18.1 主指标

- `complete_precision`
- `complete_recall`
- `complete_f1`
- AUROC / AUPRC

其中最关键的是：

- 在固定 precision 约束下的 recall

因为这更接近“少抢答前提下，能多早放行多少真正结束的 utterance”。

### 18.2 边界相关指标

如果引入边界头，建议评估：

- boundary hit rate
- boundary localization error
- 末尾容忍窗口命中率

### 18.3 业务型评估集

建议专门建 hard set：

- 犹豫停顿
- 中途自修正
- 列举句
- 句尾拖音
- 口语填充词
- “说完了但要求你先别回”

这些场景对真正上线价值最高。

### 18.4 分组评估

至少按以下维度切分：

- 语言
- 说话人性别或音色风格
- 设备/噪声条件
- clip 长度
- 是否带上下文


## 19. 推荐的实施顺序

后续真正编码实现时，建议按下面顺序推进，而不是一次性全上。

### 第一步：把 V2 数据定义和标注规范定清楚

优先明确：

- 完整样本定义
- 截断样本构造规则
- transcript 与 timestamp 的质量标准

### 第二步：先做主分类 + ASR 辅助的最小版

这是最值得先验证的一版，因为它最能回答：

- 多任务训练是否真的优于只训分类头

### 第三步：加入边界头

当主分类 + ASR 辅助跑通后，再加边界监督，验证是否能进一步改善 hard case。

### 第四步：加入轻量上下文

如果业务里确实存在“语音上说完，但策略上要等”的大量边界情况，再加上下文字段。

### 第五步：阈值和校准

最后再做：

- 不同业务目标下的阈值选择
- 高 precision 模式 / 平衡模式


## 20. 最终推荐结论

综合当前仓库结构、你的业务诉求和公开研究，推荐的 V2 方向是：

> **保持线上二分类推理不变，但把训练改成多任务、强监督、分阶段解冻的 turn detection 适配方案。**

更具体地说：

- 不建议停留在“只换更强 pooling + 多解冻几层”的弱升级；
- 不建议走“线上生成标签”的慢路径；
- 推荐把 Qwen3-ASR 当成一个可继续适配的音频-语义 backbone；
- 通过 `utterance classification + boundary supervision + ASR auxiliary + semantic closure` 的组合，把模型真正往 turn detection 上拉。

这条路线的主要优势是：

- 保留低延迟二分类接口；
- 最大化继承 Qwen3-ASR 原始能力；
- 更符合 turn completion 的本质；
- 在几百小时数据条件下，比“只训分类头”更有上限。


## 21. 参考资料

### 官方与仓库

- Qwen3-ASR 模型卡（0.6B）
  - https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- Qwen3-ASR 模型卡（1.7B）
  - https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- 当前仓库 turn detection 文档
  - `/mnt/e/Work/AI/AUDIO/Qwen3-ASR/finetuning/TURN_DETECTION.md`
- 当前仓库 turn detection 训练脚本
  - `/mnt/e/Work/AI/AUDIO/Qwen3-ASR/finetuning/qwen3_turn_detection.py`
- 当前仓库 turn detector 模型
  - `/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/turn_detection/qwen3_turn_detector.py`

### 研究资料

- Unified End-to-End Speech Recognition and Endpointing for Fast and Efficient Speech Systems
  - https://research.google/pubs/unified-end-to-end-speech-recognition-and-endpointing-for-fast-and-efficient-speech-systems/
- Joint Endpointing and Decoding with End-to-End Models
  - https://research.google/pubs/joint-endpointing-and-decoding-with-end-to-end-models/
- Semantic VAD: Low-Latency Voice Activity Detection for Speech Interaction
  - https://www.isca-archive.org/interspeech_2023/shi23c_interspeech.html
- A cross-linguistic analysis of the temporal dynamics of turn-taking cues using machine learning as a descriptive tool
  - https://www.sciencedirect.com/science/article/pii/S0167639320302727
- Investigating Linguistic and Semantic Features for Turn-Taking Prediction in Open-Domain Human-Computer Conversation
  - https://www.isca-archive.org/interspeech_2019/razavi19_interspeech.html
- Automatic Prosodic Event Detection Using Acoustic, Lexical, and Syntactic Evidence
  - https://pubmed.ncbi.nlm.nih.gov/19122857/
