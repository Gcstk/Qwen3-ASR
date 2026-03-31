# Qwen3-ASR 训练代码导读

这份文档配合代码里的中文注释一起看，适合：

- 刚接触 `transformers.Trainer`
- 对 `deepspeed` 还不熟
- 想知道这个仓库“训练相关代码到底是怎么串起来的”

## 1. 训练相关代码在哪些文件

最值得读的文件只有 4 个：

- [qwen3_asr_sft.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/finetuning/qwen3_asr_sft.py)
- [qwen3_turn_detection.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/finetuning/qwen3_turn_detection.py)
- [qwen3_turn_detector.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/turn_detection/qwen3_turn_detector.py)
- [modeling_qwen3_asr.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

可以把它们分成两层理解：

- 训练脚本层
  - 负责读数据、构造 batch、设置 `TrainingArguments`、启动 `Trainer`
- 模型结构层
  - 负责把文本和音频真正融合到同一个前向过程里

## 2. 先理解 Qwen3-ASR 的模型结构

顶层类是 [Qwen3ASRForConditionalGeneration](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L1482)。

它里面最关键的是：

- `audio_tower`
- `thinker`

可以把它理解成：

1. `audio_tower` 负责把音频特征编码成连续向量
2. 文本 prompt 先变成 token embedding
3. prompt 里的 `<audio>` 占位 token 会被真实音频向量覆盖
4. 覆盖后的整条 embedding 序列送进 `thinker`
5. `lm_head` 再把隐藏状态映射成词表 logits

这就是为什么它是“多模态生成模型”，不是传统 CTC ASR。

## 3. ASR SFT 脚本怎么训练

[qwen3_asr_sft.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/finetuning/qwen3_asr_sft.py) 的关键路径是：

1. `Qwen3ASRModel.from_pretrained(...)`
2. `patch_outer_forward(model)`
3. `datasets.load_dataset(...)`
4. `make_preprocess_fn_prefix_only(...)`
5. `DataCollatorForQwen3ASRFinetuning`
6. `Trainer.train()`

最重要的设计点是 **label masking**。

训练时不会让模型去“背 prompt”，而是：

- `prefix_text` 只作为输入条件
- `target` 才是监督目标
- collator 里把 prefix 对应 token 的 label 全部置成 `-100`

这样 loss 只会落在你真正想让模型学会输出的那部分文本上。

## 4. Turn Detection 脚本怎么训练

[qwen3_turn_detection.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/finetuning/qwen3_turn_detection.py) 的思路完全不同：

- 不训练文本生成
- 只训练 `complete / incomplete` 二分类

它的关键路径是：

1. 加载 Qwen3-ASR 底座
2. 用 [Qwen3TurnDetector](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/turn_detection/qwen3_turn_detector.py) 包一层分类头
3. 在 collator 里把候选切点附近的音频裁成窗口
4. 前向得到 `logits`
5. 用交叉熵训练

这个设计比“让模型直接生成 `<complete>` 文本标签”更务实，因为：

- 推理更轻
- 阈值更好调
- 更适合和前置 VAD 结合

## 5. `transformers.Trainer` 在这里到底干了什么

如果你是 Trainer 新手，可以把它理解成“通用训练循环框架”。

它负责：

- 取 batch
- 调 `model(**batch)`
- 从返回值里拿 `loss`
- 反向传播
- 更新优化器
- 保存 checkpoint
- 跑验证

你真正需要自己写的只有三块：

- 数据怎么变成 batch
- 模型 `forward` 怎么定义
- 指标怎么算

在这个仓库里分别对应：

- `DataCollatorForQwen3ASRFinetuning`
- `DataCollatorForQwen3TurnDetection`
- `Qwen3ASRThinkerForConditionalGeneration.forward`
- `Qwen3TurnDetector.forward`
- `compute_metrics`

## 6. 当前代码里比较好的设计

### 1. 训练脚本和推理脚本分层明确

训练脚本不把模型结构逻辑写死在里面，而是：

- 模型结构在 `qwen_asr/core/...`
- 训练组织在 `finetuning/...`

这让你后续替换训练策略时，不必重写模型实现。

### 2. prefix 预处理和音频读取分离

`datasets.map()` 只生成 `prefix_text`，不提前读音频。

好处：

- map 阶段更轻
- 不会在预处理时就把大批音频拉进内存
- collator 里更接近真实 batch

### 3. checkpoint 被补成可推理格式

ASR 脚本里会把 tokenizer / processor / chat template 一起拷到 checkpoint。

这很实用，因为中间 checkpoint 可以直接推理验证。

### 4. turn detection 默认只训分类头

这对小数据任务很重要：

- 风险更小
- 更省显存
- 最大程度复用底座已经学到的音频知识

## 7. 当前代码里还可以怎么优化

### 1. 把训练参数改成 dataclass + HfArgumentParser

现在两个训练脚本都用的是手写 `argparse`。

这够用，但如果你以后想加：

- `--deepspeed`
- `--gradient_checkpointing`
- `--report_to`
- `--save_strategy epoch`

会越来越乱。

更推荐改成：

- dataclass 定义参数
- `HfArgumentParser`
- 直接复用 Hugging Face 生态标准 CLI

### 2. 直接接入 DeepSpeed

当前这两个脚本都没有直接暴露 `--deepspeed`。

如果你以后要接 DeepSpeed，建议顺序是：

1. 先保留 `Trainer`
2. 改用 `HfArgumentParser`
3. 把 deepspeed config 路径接到 `TrainingArguments(deepspeed=...)`

这样你不用自己重写训练循环。

### 3. 音频预处理可以缓存

现在训练时音频是实时 `librosa.load()`。

优点是简单。
缺点是：

- IO 压力大
- 重采样开销大
- 多 worker 下磁盘容易抖动

如果数据量大，可以考虑离线统一：

- 采样率
- 声道
- 切片窗口

### 4. turn detection 可以增加更细的池化策略

当前只支持：

- `last_token`
- `mean`

以后可以尝试：

- attention pooling
- 多层 hidden state 融合
- 专门读 assistant 起始位附近 token

## 8. 当前仓库和 DeepSpeed 的关系

一句话：**底层兼容 Hugging Face 训练范式，但训练脚本还没把 DeepSpeed 入口接出来。**

也就是说：

- 模型本身不是不能配合 DeepSpeed
- 而是训练脚本当前没有把它做成开箱即用

如果你是 DeepSpeed 小白，最稳的顺序是：

1. 先用当前脚本把单卡 / 多卡训练跑通
2. 理解 `TrainingArguments`
3. 再把脚本改造成支持 `deepspeed`

## 9. 你读代码时建议的顺序

推荐顺序：

1. [qwen3_asr_sft.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/finetuning/qwen3_asr_sft.py)
2. [qwen3_turn_detection.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/finetuning/qwen3_turn_detection.py)
3. [qwen3_turn_detector.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/turn_detection/qwen3_turn_detector.py)
4. [modeling_qwen3_asr.py](/mnt/e/Work/AI/AUDIO/Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

这样你会先理解“训练怎么组织”，再回头看“模型内部怎么融合音频和文本”。
