# Qwen3-ASR 对话轮次检测数据与操作文档

## 1. 文档目标

这份文档定义后续 turn detection 项目的 **数据结构、数据处理流程、实验组织方式和编码落地边界**。

本文档与 `doc/qwen3_asr_turn_detection_design.md` 配套使用：

- 设计文档回答“模型为什么这样设计”
- 本文档回答“数据怎么准备、怎么组织、怎么训练、怎么评估、怎么指导后续编码”

---

## 2. 版本策略

### 2.1 V1

只做：

- `complete`
- `incomplete`

### 2.2 V2

在 V1 达标后，再考虑：

- `backchannel`

V2 仍沿用与 V1 相同的数据主结构，只扩展 `label` 和 `text` 的目标模板。

---

## 3. 总体数据流

整个项目的数据流分为三层：

### 3.1 原始会话层

保存完整会话音频与原始元数据。

这一层不直接喂模型。

### 3.2 候选端点层

保存从原始会话中挖出的候选切点，以及围绕这些切点的标注信息。

这一层是中间加工层，不直接喂模型。

### 3.3 最终模型层

把候选端点裁成固定音频 clip，并导出最终 JSONL manifest。

这一层才是训练、验证、测试真正使用的数据。

---

## 4. 最终数据结构

## 4.1 设计目标

最终模型层的数据结构要尽量和 `Qwen3-ASR` 原始 SFT 训练保持一致。

当前 `qwen3_asr_sft.py` 训练数据的核心结构是：

- `audio`
- `text`
- `prompt`

因此 turn detection 的最终 JSONL 也统一采用同样主结构，并额外保留少量元数据字段。

## 4.2 最终 JSONL 字段定义

### 必需字段

- `audio`
  - 裁剪后的单条音频 clip 路径
  - 必须是模型实际读取的最终音频

- `text`
  - 训练目标文本
  - 用固定 verbalizer 模板

- `prompt`
  - 任务说明
  - 作为 system prompt 写入

### 强烈建议保留字段

- `label`
  - 规范化标签名
  - V1 取值为 `complete` 或 `incomplete`

- `sample_id`
  - 全局唯一样本 ID

- `session_id`
  - 原始会话 ID

- `source_audio`
  - 原始长音频路径

- `clip_start_ms`
  - 在原始长音频中的起始毫秒

- `clip_end_ms`
  - 在原始长音频中的结束毫秒

- `language`
  - 可选语言标记

- `candidate_source`
  - 候选端点来源，如 `vad`、`manual`、`rule`

- `split`
  - `train` / `dev` / `test`

### 可选分析字段

- `pause_ms`
- `duration_ms`
- `noise_level`
- `snr_db`
- `overlap_speech`
- `notes`

这些字段不要求直接参与模型前向，但后续评估和切片分析很有价值。

---

## 5. V1 最终 manifest 规范

## 5.1 标签定义

### `complete`

表示在当前 clip 末尾，用户这一段话已经说完。

### `incomplete`

表示在当前 clip 末尾，用户这一段话还没说完，当前停顿更像：

- 犹豫
- 呼吸
- 半句停顿
- 从句过渡
- 被截断的延续

## 5.2 推荐 prompt

推荐统一使用：

```text
Listen to the audio clip and decide whether the user's utterance is complete at the end of the clip. Answer with exactly one sentence.
```

在同一个实验中不要混用不同 prompt。

## 5.3 推荐 `text` verbalizer

V1 统一使用：

- `The user's utterance is complete.`
- `The user's utterance is incomplete.`

不要出现：

- `complete`
- `incomplete`
- `Yes, it is complete.`
- `The utterance seems complete.`

必须严格统一模板，避免引入不必要的文本变体噪声。

## 5.4 V1 最终 JSONL 示例

```jsonl
{"audio":"data/turn_detection_v1/clips/train/td_v1_train_000001.wav","text":"The user's utterance is incomplete.","prompt":"Listen to the audio clip and decide whether the user's utterance is complete at the end of the clip. Answer with exactly one sentence.","label":"incomplete","sample_id":"td_v1_train_000001","session_id":"sess_018","source_audio":"data/raw_sessions/sess_018/user.wav","clip_start_ms":1320,"clip_end_ms":3920,"language":"zh","candidate_source":"vad","split":"train"}
{"audio":"data/turn_detection_v1/clips/train/td_v1_train_000002.wav","text":"The user's utterance is complete.","prompt":"Listen to the audio clip and decide whether the user's utterance is complete at the end of the clip. Answer with exactly one sentence.","label":"complete","sample_id":"td_v1_train_000002","session_id":"sess_018","source_audio":"data/raw_sessions/sess_018/user.wav","clip_start_ms":4180,"clip_end_ms":6780,"language":"zh","candidate_source":"vad","split":"train"}
```

说明：

- 阶段 A 生成式训练主要使用 `audio + text + prompt`
- 阶段 B 分类训练可以读取同一份 JSONL 中的 `label`

---

## 6. V2 最终 manifest 扩展规范

在 V2 中，数据主结构保持不变，仅扩展：

- `label`
- `text`
- `prompt`

### V2 标签

- `complete`
- `incomplete`
- `backchannel`

### V2 verbalizer

- `The user's utterance is complete.`
- `The user's utterance is incomplete.`
- `The user's utterance is a backchannel.`

### V2 prompt

```text
Listen to the audio clip and decide whether the user's utterance is complete, incomplete, or only a backchannel. Answer with exactly one sentence.
```

V2 是否正式启用，取决于 `backchannel` 样本质量和 V1 的稳定性。

---

## 7. 原始数据组织方式

## 7.1 原始层目录建议

```text
data/
  raw_sessions/
    sess_001/
      user.wav
      meta.json
    sess_002/
      user.wav
      meta.json
```

其中 `meta.json` 推荐包含：

- `session_id`
- `speaker_id`
- `language`
- `scene`
- `sample_rate`
- 可选转写
- 可选 VAD 信息

## 7.2 中间候选端点层建议

```text
data/
  manifests/
    candidate_endpoints_v1.jsonl
```

这一层每行描述一个候选端点，不直接喂模型。推荐字段：

- `session_id`
- `source_audio`
- `cut_time_ms`
- `left_context_ms`
- `right_context_ms`
- `label`
- `language`
- `candidate_source`
- `notes`

示例：

```jsonl
{"session_id":"sess_018","source_audio":"data/raw_sessions/sess_018/user.wav","cut_time_ms":3320,"left_context_ms":2000,"right_context_ms":600,"label":"incomplete","language":"zh","candidate_source":"vad","notes":"hesitation pause"}
{"session_id":"sess_018","source_audio":"data/raw_sessions/sess_018/user.wav","cut_time_ms":6180,"left_context_ms":2000,"right_context_ms":600,"label":"complete","language":"zh","candidate_source":"vad","notes":"utterance finished"}
```

这一层的作用是：

- 保存原始切点定义
- 便于重复裁切和规则回溯
- 便于后续重新生成 clip

---

## 8. 最终模型层目录建议

```text
data/
  turn_detection_v1/
    clips/
      train/
      dev/
      test/
    manifests/
      train.jsonl
      dev.jsonl
      test.jsonl
      test_short_complete.jsonl
      test_hesitation.jsonl
      test_noisy.jsonl
```

说明：

- `clips/` 存真正训练和评估使用的裁剪音频
- `manifests/` 存最终模型 JSONL
- 额外 `test_*.jsonl` 用于 slice 分析

---

## 9. 数据处理流程

## 9.1 步骤 1：准备原始会话级数据

原始数据以完整会话为单位保存。

要求：

- 音频可正常读取
- 统一单声道
- 原始采样率可不同，但后续裁切导出时统一到 `16k`
- 每个 session 具备稳定的唯一 ID

## 9.2 步骤 2：生成候选端点

候选端点可以来自：

- VAD 停顿检测
- 规则切点
- 手工标注
- 旧系统日志

V1 推荐优先使用：

- VAD 候选端点

每个候选端点用以下窗口规则生成 clip：

- `left_context_ms = 2000`
- `right_context_ms = 600`

即：

- clip 起点 = `cut_time_ms - 2000`
- clip 终点 = `cut_time_ms + 600`

如果越界，则裁到合法范围。

这是 V1 的默认窗口，不建议一开始频繁改动。

## 9.3 步骤 3：人工标注

V1 标注流程建议：

1. 先听 clip
2. 判断 clip 末尾是否已经说完
3. 标成 `complete` 或 `incomplete`
4. 对无法判断或上下文严重缺失的样本直接丢弃

### `complete` 的典型场景

- 一个问题已经问完
- 一个命令已经说完
- 一个短回答已经收尾
- 语句虽然短，但意图已经完整闭合

### `incomplete` 的典型场景

- “我想问一下那个...”
- “就是你刚刚说的那个我觉得...”
- 明显还有后续但被切掉
- 中间长呼吸或犹豫停顿

### 直接丢弃的场景

- 音频损坏
- 纯噪声
- 语音极短且无可判定内容
- 重叠说话导致无法一致标注

## 9.4 步骤 4：裁切最终 clip

从原始长音频按候选端点层 manifest 裁切，得到最终模型层 clip。

导出规则：

- WAV
- 单声道
- 16kHz
- `float32` 或标准 PCM 均可，但建议统一
- 文件名使用 `sample_id`

例如：

- `td_v1_train_000001.wav`
- `td_v1_dev_000031.wav`
- `td_v1_test_000104.wav`

## 9.5 步骤 5：切分 train / dev / test

最重要的规则：

- **必须按 `session_id` 切分**

严禁：

- 按 clip 随机切分
- 同一 session 同时出现在 train 和 test

推荐比例：

- `train = 80%`
- `dev = 10%`
- `test = 10%`

如果 session 数量较少，可以按会话数人工分配，但必须保持会话隔离。

## 9.6 步骤 6：导出最终 JSONL

最终分别导出：

- `train.jsonl`
- `dev.jsonl`
- `test.jsonl`

并额外导出若干 slice：

- `test_short_complete.jsonl`
- `test_hesitation.jsonl`
- `test_noisy.jsonl`

---

## 10. 训练数据组织建议

## 10.1 类别平衡

V1 不要求完全 1:1，但要避免极端失衡。

推荐：

- `complete` 与 `incomplete` 比例控制在 `0.7:1` 到 `1.4:1` 之间

如果真实数据严重偏斜，文档里允许训练时做：

- 重采样
- loss reweight

但测试集不做重采样。

## 10.2 难例覆盖

训练集中必须刻意覆盖以下场景：

- 短完整句
- 长完整句
- 犹豫停顿
- 半句延续
- 尾音拖长
- 中低噪声
- 语速快慢变化

原因：

- 避免模型学成“短句就 complete”或“停顿就 incomplete”

## 10.3 不建议的数据增强

V1 不建议强行使用：

- 大幅 speed perturb
- 大幅 time stretch

因为这些变换会直接改变停顿和韵律，对 turn detection 边界有副作用。

## 10.4 推荐的数据增强

如需增强，优先使用：

- 轻微加噪
- 轻微混响
- 候选切点附近的小范围窗口扰动

其中窗口扰动最重要：

- 对同一原始切点，生成少量相邻窗口
- 帮助模型学习边界附近的鲁棒性

---

## 11. 测试数据组织建议

## 11.1 主测试集

主测试集用于正式报告核心指标。

要求：

- 按 session 隔离
- 不参与任何调参
- 不做重采样

## 11.2 Slice 测试集

建议额外维护以下切片：

- `short_complete`
  - 很短但已完成的请求或回答

- `hesitation`
  - 犹豫停顿类 incomplete

- `noisy`
  - 噪声较高样本

- `tail_dragging`
  - 尾音拉长样本

这些切片可以是 `test.jsonl` 的子集 manifest，不必复制音频文件。

## 11.3 V2 额外测试切片

如果进入 V2，再新增：

- `backchannel_short`
- `backchannel_laughter`
- `short_complete_vs_backchannel`

---

## 12. 实验组织建议

## 12.1 阶段 A：生成式适配

输入：

- V1 最终 manifest

模型：

- 纯净 `Qwen3-ASR`

训练目标：

- 学会输出统一 verbalizer 句子

建议实验：

- A1: `lm_head only`
- A2: `lm_head + thinker top-2`
- A3: `lm_head + thinker top-4`

## 12.2 阶段 B：判别式部署

输入：

- 与阶段 A 相同 manifest

模型：

- 阶段 A backbone + 轻量分类头

建议实验：

- B1: head only
- B2: head + thinker top-2
- B3: head + distillation

## 12.3 阈值策略

部署时不要只用默认 `0.5`。

至少保存三套阈值：

- 默认 `0.5`
- 高精度阈值
- 平衡 F1 阈值

dev 集只用于阈值选择，test 集只用于最终报告。

---

## 13. 评估指标建议

V1 至少报告：

- Accuracy
- Precision / Recall / F1
- AUROC
- AUPRC
- Brier score
- ECE

并按以下切片输出：

- 时长
- pause 时长
- 语言
- 噪声
- 是否重叠

V2 再增加：

- 三分类混淆矩阵
- `complete` 与 `backchannel` 的相互混淆分析

---

## 14. 后续编码指导

## 14.1 数据准备脚本

建议新增：

- `finetuning/prepare_turn_detection_dataset.py`

输入：

- 原始 session 目录
- 候选端点 manifest

输出：

- 裁好的 clip
- `train/dev/test` JSONL
- slice JSONL

## 14.2 阶段 A 训练脚本

建议新增：

- `finetuning/qwen3_turn_detection_sft_v1.py`

设计要求：

- 输入 manifest 兼容 `audio + text + prompt`
- 复用 `qwen3_asr_sft.py` 的 processor 和训练数据协议

## 14.3 阶段 B 模型文件

建议新增：

- `qwen_asr/turn_detection/qwen3_turn_detector_v1.py`

设计要求：

- 加载阶段 A backbone
- 提供 `predict / predict_batch`
- 输出规范化 dataclass

## 14.4 评估脚本

建议新增：

- `finetuning/eval_qwen3_turn_detection_v1.py`

设计要求：

- 能读主测试集和 slice 测试集
- 输出 JSON 报告
- 输出逐条预测 JSONL

---

## 15. 操作结论

后续实现时，数据层必须坚持以下约束：

1. 原始层、候选端点层、最终模型层三层分离
2. 最终模型 manifest 与 `Qwen3-ASR` SFT 主结构一致，核心字段固定为 `audio + text + prompt`
3. V1 只做 `complete / incomplete`
4. 训练、验证、测试必须按 `session_id` 切分
5. 所有 prompt 与 verbalizer 必须模板统一
6. 后续编码必须优先围绕这套统一 manifest 设计，而不是为不同训练阶段造不同数据协议

只要按本文档准备数据，后续无论实现生成式适配、分类头蒸馏，还是未来扩展到 `backchannel`，都可以在同一套数据主结构上平滑推进。
