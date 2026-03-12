# Qwen3-ASR 训练上手与实战指南

这份文档面向这样的读者：

- 了解 NLP 和 LLM 的基本原理
- 做过或看过 LLM SFT / instruction tuning
- 还没有系统接触过音频模型、ASR 训练和语音数据处理

目标不是重复仓库里的快速命令，而是把这件事讲清楚：

- 这个项目的训练入口在哪里
- 训练脚本到底在做什么
- 为什么训练文本要写成 `language English<asr_text>...`
- 训练数据应该怎么处理
- 第一次做 Qwen3-ASR 微调时，最稳妥的入手方案是什么

## 1. 先建立全局认知

这个仓库可以粗分成三部分：

- 推理接口：`qwen_asr/inference/`
- 模型实现和处理器：`qwen_asr/core/`
- 训练入口：`finetuning/qwen3_asr_sft.py`

如果你只是想做训练，最重要的文件是：

- `finetuning/qwen3_asr_sft.py`
- `finetuning/README.md`

当前仓库提供的是一种很明确的训练路径：

- 基于 Hugging Face `Trainer`
- 使用 JSONL 音频-文本对
- 做全参 SFT 风格训练

它没有内置这些能力：

- LoRA / QLoRA
- 数据加权采样
- 自动清洗数据
- 内置 WER / CER 评测脚本
- 分阶段训练或蒸馏流水线

所以阅读和使用这个仓库时，最好先接受一个事实：它现在是一套“能直接跑起来的 Qwen3-ASR SFT 基线”，不是完整的语音训练平台。

## 2. 如果你做过 LLM SFT，可以先这样理解

对 LLM 读者来说，这套训练流程最容易理解的方式，是先和文本 SFT 做一个映射。

### 2.1 相同点

- 都是“给定输入条件，预测目标输出”
- 都会构造 prompt
- 都会对输入前缀做 label masking，只监督真正希望模型生成的部分
- 都可以通过 `batch_size`、`grad_acc`、`lr`、`epochs` 这些参数控制训练

### 2.2 不同点

Qwen3-ASR 不是“把音频转成文本再当普通文本模型训练”，而是：

- 音频先经 processor 转成 `input_features`
- 文本经 tokenizer 转成 `input_ids`
- 两者一起送入同一个多模态生成模型

你可以把它理解为：

- 文本 prompt 负责定义任务格式
- 音频负责提供实际语音内容
- 模型最后仍然以“生成文本”的方式输出识别结果

### 2.3 为什么音频训练比纯文本训练更容易踩坑

文本训练时，坏样本往往能直接看出来。音频训练不是。

常见问题包括：

- 音频路径存在，但文件损坏
- 采样率混乱
- 立体声、多通道、静音段过多
- 切分不对，文本只对应中间一小段语音
- 文本标注看起来没错，但对应错音频

这些问题不会因为 JSONL 格式合法就自动消失。训练脚本也不会替你检查。

## 3. 当前训练脚本到底做了什么

训练入口是 `finetuning/qwen3_asr_sft.py`。

把它拆开看，核心流程如下：

1. 加载 `Qwen3ASRModel.from_pretrained(...)`
2. 取出底层 `model` 和 `processor`
3. 用 `datasets.load_dataset("json", ...)` 读取训练 JSONL
4. 预处理每条样本，构造 `prefix_text`
5. 在 collator 里实时加载音频并做 batch 编码
6. 构造 `labels`，把 prefix 和 padding 位置设为 `-100`
7. 用 Hugging Face `Trainer` 开始训练

### 3.1 `patch_outer_forward()` 在做什么

训练脚本里有一个 `patch_outer_forward(model)`。

它的作用不是改变模型能力，而是把外层对象补成一个符合 `Trainer` 预期的 `forward(...)` 入口。否则 `Trainer` 不一定知道该怎么把 `input_ids`、`input_features`、`labels` 这些字段正确喂给模型。

你可以把它理解为一个“训练兼容层”。

### 3.2 `prompt` 和 `prefix_text` 是怎么来的

脚本里有这样一段逻辑：

- `prompt` 被放进 system message
- 音频被放进 user message
- `processor.apply_chat_template(..., add_generation_prompt=True)` 生成文本前缀

也就是说，训练并不是直接拿 `text` 去拟合，而是先构造出一段“模型应该如何开始回答”的前缀，再把目标文本接到后面。

### 3.3 为什么只监督 target，不监督 prefix

这是这份脚本最关键的设计之一。

在 collator 里，脚本会分别编码：

- `full_texts = prefix_text + target + eos`
- `prefix_texts = prefix_text`

然后计算 `prefix_lens`，把 `labels` 前半段对应 prefix 的 token 全部设成 `-100`。

原因很简单：

- prefix 是输入条件，不是训练目标
- 我们不需要模型“学会把系统提示词背出来”
- 我们需要模型在看到这种输入结构时，生成正确的 ASR 输出

这和常见 LLM instruction tuning 的 masking 思路是一致的。

## 4. 为什么训练文本推荐写成 `language X<asr_text>...`

这是当前项目里最容易被忽略、但最重要的约束之一。

推理侧的解析逻辑在 `qwen_asr/inference/utils.py` 的 `parse_asr_output(...)` 中。它默认期望模型输出类似：

```text
language English<asr_text>This is a test sentence.
```

或者：

```text
language Chinese<asr_text>今天天气很好。
```

### 4.1 这样做的原因

这样设计的好处是把两件事统一到一条生成结果里：

- 语言识别
- 语音转写

推理时模型输出一段文本，解析器再把它拆成：

- `language`
- `text`

### 4.2 如果不写语言前缀会怎么样

可以训练，但要知道代价。

如果目标文本里没有 `language ... <asr_text>` 这层协议：

- 模型仍可能学会转写
- 但语言识别能力可能弱化
- 推理输出格式可能和当前解析逻辑不一致
- 部分结果会更依赖后处理猜测

### 4.3 `language None<asr_text>` 什么时候用

当你没有可靠语言标签时，可以写：

```text
language None<asr_text>hello world
```

这适合“先把转写学会”的场景，但它不会帮助模型学习稳定的语言识别头部。

所以一个实用建议是：

- 有语言标签时，尽量显式写语言
- 没有语言标签时，再退到 `language None`

## 5. 数据格式：脚本真正接受什么

当前脚本按 JSON/JSONL 读取数据。每行至少要有两个字段：

- `audio`
- `text`

可选字段：

- `prompt`

### 5.1 最小示例

```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language Chinese<asr_text>这是一个测试句子。"}
```

### 5.2 没有语言标签的示例

```jsonl
{"audio":"/data/wavs/utt0003.wav","text":"language None<asr_text>Hello, this is still a valid training sample."}
```

### 5.3 带 `prompt` 的示例

```jsonl
{"audio":"/data/wavs/meeting001.wav","prompt":"Please transcribe the speech accurately and keep technical terms unchanged.","text":"language English<asr_text>Today we discuss the encoder latency issue."}
```

### 5.4 这三个字段分别有什么作用

`audio`

- 是音频文件路径
- 在 collator 里通过 `librosa.load(path, sr=16000, mono=True)` 读取
- 这意味着训练时会实时从磁盘加载和重采样

`text`

- 是最终监督目标
- 会和 `prefix_text` 拼接成完整训练目标
- 它的格式最好和推理期希望得到的格式一致

`prompt`

- 进入 system message
- 用来补充任务约束或上下文
- 如果你只是做标准 ASR，完全可以先留空

## 6. 音频数据应该怎么处理

如果你过去只做过文本数据，最需要更新的习惯是：音频数据必须先做离线检查。

### 6.1 推荐的基础规范

- 尽量统一为单声道
- 尽量统一为 16kHz
- 保证音频能正常播放
- 保证文本只对应这条音频里的有效语音
- 去掉明显错误、纯静音、空文件、破损文件

虽然脚本会在加载时强制 `sr=16000, mono=True`，但这只是“兼容输入”，不是“高质量数据处理”。

为什么仍然建议离线统一：

- 可以提前发现坏样本
- 降低训练时动态重采样带来的不确定性
- 让数据统计更可靠
- 让 DataLoader 更稳定

### 6.2 训练前至少做这几类检查

#### 1. 文件可用性检查

- 路径是否真实存在
- 是否能被 `librosa` 或 `soundfile` 正常读取
- 是否有空文件、零字节文件、损坏文件

#### 2. 音频时长检查

- 极短音频要重点确认
- 超长音频要重点确认
- 文本很短但音频很长，或者相反，通常要复查

#### 3. 文本内容检查

- 去掉控制字符
- 统一空格和标点风格
- 统一数字书写规范
- 统一英文大小写策略

#### 4. 对齐关系检查

最常见也最致命的问题不是文本错，而是“文本和音频没对齐”。例如：

- 文本只标了前半句，但音频有整段会议
- 一条音频里有两个说话人，文本只写了一个
- 标注漏掉口头语或重复词，导致训练目标不稳定

#### 5. 分布统计检查

至少要知道这些数据分布：

- 总样本数
- 总时长
- 平均时长
- 语言分布
- 场景分布
- 说话人分布

如果你连这些统计都没有，就很难判断训练结果到底是模型问题还是数据问题。

## 7. 训练集和验证集怎么切

当前脚本支持：

- `--train_file`
- `--eval_file`

如果你提供了 `eval_file`，脚本会在保存 checkpoint 的相同步数上做验证；如果没有提供，就不会真的得到有效验证结果。

### 7.1 推荐原则

- 验证集要代表你的真实目标场景
- 不要把同一条长音频切片后随意分到 train 和 val
- 不要让同一说话人的高度相似录音同时出现在 train 和 val

### 7.2 为什么这点对语音尤其重要

语音数据的泄漏比文本更隐蔽。

如果同一个人、同一个房间、同一个录音源的大量相似片段同时在训练集和验证集里，指标会非常乐观，但上线后不一定成立。

## 8. 第一次训练，建议怎么开始

不要一上来就全量训练。最稳妥的路径是分三步。

### 8.1 第一步：先跑通

目标不是效果最好，而是验证链路没问题。

建议：

- 先抽 100 到 500 条高质量样本
- 先不加复杂 `prompt`
- 文本统一成 `language X<asr_text>...`
- 单卡先跑一个短实验

示例：

```bash
python finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train_small.jsonl \
  --eval_file ./eval_small.jsonl \
  --output_dir ./outputs/qwen3-asr-sanity \
  --batch_size 2 \
  --grad_acc 8 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 50 \
  --save_total_limit 2
```

先确认这些事情：

- 训练能正常启动
- 不会立刻 OOM
- checkpoint 能正常保存
- checkpoint 能正常回载推理
- 输出格式没有崩

### 8.2 第二步：做小规模领域适配

如果你已经确认链路没问题，再扩大到一批高质量领域数据，例如：

- 医疗问诊
- 会议纪要
- 客服通话
- 某种口音或方言场景

为什么建议先做“高质量、小规模、单领域”：

- 更容易观察模型是否学到了术语和场景特征
- 更容易判断数据值不值得继续扩充
- 更容易看出过拟合迹象

### 8.3 第三步：再考虑混合数据

如果你的目标不是“只在一个很窄的领域里识别得更好”，而是“尽量保留通用能力的同时增强某领域”，通常要做混合训练：

- 一部分通用数据
- 一部分目标领域数据

原因是纯小领域数据容易造成灾难性遗忘，表现为：

- 术语更准了
- 通用音频反而更差了
- 多语言或语言识别能力下降

## 9. 常见训练路线：怎么做、为什么

### 9.1 跑通型路线

怎么做：

- 用极小子集
- 只验证格式、脚本、显存、checkpoint、回载推理

为什么：

- 先解决工程问题
- 不要拿大数据查格式错误

### 9.2 小规模领域微调路线

怎么做：

- 只用高质量目标场景数据
- 控制轮数
- 重点看人工试听结果

为什么：

- 最容易确认“这批数据是否能给业务带来增益”

### 9.3 混合数据稳态路线

怎么做：

- 把通用数据和领域数据按比例混合
- 验证集同时覆盖通用场景和目标场景

为什么：

- 只靠窄域数据容易牺牲原始模型的通用性

### 9.4 错例回流路线

怎么做：

- 先训一个版本
- 收集失败案例
- 把错例按类别回流补数

为什么：

- ASR 提升通常更依赖“错误分布驱动的数据迭代”，而不是盲目地调学习率和 epoch

## 10. 参数怎么理解

当前训练脚本暴露了这些关键参数。

### `--batch_size`

每张卡上的 batch size。

怎么调：

- 先从小值开始
- OOM 就继续减小

为什么：

- 音频长度差异大，显存波动比纯文本训练更明显

### `--grad_acc`

梯度累积步数。

怎么调：

- 显存不够时，先减小 `batch_size`，再用 `grad_acc` 补回有效 batch

为什么：

- 它能在不直接增大单步显存占用的前提下，维持更大的有效 batch

### `--lr`

学习率，默认 `2e-5`。

怎么调：

- 第一次训练先别激进
- 先用默认值或附近量级

为什么：

- 小数据、全参微调、语音任务，这三者叠加时，过大的学习率很容易破坏原有能力

### `--epochs`

训练轮数。

怎么调：

- 小数据场景不要机械拉高

为什么：

- 语音数据通常重复模式强，小数据更容易过拟合

### `--save_steps`

每多少步保存一次 checkpoint。

为什么要注意：

- 这个脚本里 `eval_steps` 也绑定到了 `save_steps`
- 所以它同时影响保存频率和验证频率

### `--num_workers`

DataLoader worker 数。

怎么调：

- 先从保守值开始
- 结合机器 CPU 和磁盘吞吐调整

为什么：

- 当前 collator 是实时读音频，worker 太少可能卡 IO，太多也可能造成系统抖动

### `--resume_from` 和 `--resume`

用于断点恢复。

- `--resume_from`：显式指定 checkpoint
- `--resume 1`：自动找 `output_dir` 下最新 checkpoint

### `--sr`

音频采样率，默认 16000。

为什么通常不要改：

- 当前脚本和处理流程都默认围绕 16kHz 组织
- 随意改采样率会让数据侧和模型预期不一致

## 11. 多卡训练怎么理解

脚本支持用 `torchrun` 做多卡训练，例如：

```bash
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --eval_file ./eval.jsonl \
  --output_dir ./outputs/qwen3-asr-exp \
  --batch_size 4 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5
```

需要知道的现实问题是：

- 多卡不是为了替代数据清洗
- 多卡只解决吞吐，不解决标注质量
- 如果单卡都没跑通，不要先上多卡

## 12. 为什么每个 checkpoint 都能直接拿来推理

训练脚本里有一个 callback：`MakeEveryCheckpointInferableCallback`。

它会在保存 checkpoint 时，把推理需要的几个 Hugging Face 配置文件一起复制进去，例如：

- `config.json`
- `generation_config.json`
- `processor_config.json`
- `tokenizer.json`
- `chat_template.json`

这样做的原因是：让每个 checkpoint 目录都能直接被 `Qwen3ASRModel.from_pretrained(...)` 加载，而不只是保存一堆权重分片。

这很实用，因为你可以在训练中途随时拿任意 checkpoint 回放测试。

## 13. 训练后怎么验证

当前仓库没有内置完整评测流水线，所以建议至少做两层验证。

### 13.1 人工试听验证

准备一小组固定样本，覆盖：

- 清晰普通话 / 英语
- 目标领域样本
- 噪声样本
- 长音频和短音频

每次都用同一组样本回载 checkpoint 做对比，重点看：

- 文本是否更准
- 术语是否更稳
- 语言标签是否正常
- 幻觉是否增加

### 13.2 指标验证

如果条件允许，自己补一套 WER / CER 统计脚本。

为什么要补：

- 人工试听适合发现大问题
- 指标更适合比较多个实验版本

但要注意：

- 指标只在验证集足够干净、切分合理时才有意义

## 14. 常见问题与排查

### 14.1 输出格式错乱

现象：

- 模型不再输出 `language X<asr_text>...`

常见原因：

- 训练文本格式不统一
- 部分样本没有保留目标协议
- `prompt` 风格差异过大

处理建议：

- 先统一 `text` 格式
- 先不要给过多 prompt 变化
- 用小集重训确认是数据问题还是训练问题

### 14.2 语言识别变差

现象：

- 文本大致对，但语言字段空、错或不稳定

常见原因：

- 大量样本使用 `language None`
- 训练文本中语言前缀不一致
- 领域数据太单一

处理建议：

- 尽量补语言标签
- 严格统一语言名称写法
- 在验证集里单独检查语言识别

### 14.3 通用能力下降

现象：

- 目标领域更好了
- 通用语音更差了

常见原因：

- 纯窄域数据微调
- 训练轮数过多
- 学习率过高

处理建议：

- 混入一定比例通用数据
- 降低训练轮数
- 保守调学习率

### 14.4 DataLoader 很慢或训练抖动

现象：

- GPU 利用率低
- step 时间不稳定

常见原因：

- 音频实时读取和重采样开销大
- 磁盘吞吐不足
- `num_workers` 不合适

处理建议：

- 先检查磁盘和 CPU
- 调整 `num_workers`
- 尽量离线统一音频格式

### 14.5 checkpoint 保存了但推理失败

先检查：

- 是否直接加载某个 `checkpoint-*` 目录
- 目录里是否包含 tokenizer / processor / config 相关文件

正常情况下，这个脚本的 callback 会自动复制这些文件；如果没有，优先怀疑保存流程异常或 checkpoint 目录不完整。

## 15. 一个最稳的第一次训练方案

如果你现在就要开始，推荐按这个顺序做。

### 第 1 步：准备一个干净的小数据子集

- 100 到 500 条
- 高质量优先，不求量大
- 路径可读
- 音频可播放
- 文本格式统一

### 第 2 步：统一目标文本协议

统一成类似：

```text
language English<asr_text>...
language Chinese<asr_text>...
```

没有语言标签时再用：

```text
language None<asr_text>...
```

### 第 3 步：单卡短跑

- 用保守 batch
- 保留 eval 集
- 频繁保存 checkpoint

### 第 4 步：回载 checkpoint 做试听

确认这些问题：

- 格式对不对
- 文本准不准
- 语言字段稳不稳
- 有没有明显幻觉

### 第 5 步：再扩大规模

只有当“小样本、短实验、可回载、效果方向正确”都成立时，再扩大到更多数据、更多卡和更复杂的实验设计。

## 16. 结论

对第一次做 Qwen3-ASR 训练的人来说，最重要的不是先把参数调到极致，而是先把下面三件事做对：

- 明确当前训练目标协议：`language X<asr_text>...`
- 把音频数据质量和对齐关系处理干净
- 用小规模实验先验证训练链路和收益方向

在这个仓库里，训练本身并不复杂，复杂的是数据和实验纪律。只要这两件事处理得足够扎实，这个脚本已经足够作为一个可靠的起点。
