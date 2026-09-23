# 机器翻译实验要求

## 任务目标

根据课程作业要求，本实验需要：

1. 了解 Word Embedding 的相关知识。
2. 了解 Transformer 的结构与工作原理。
3. 了解机器翻译的基本过程。
4. 了解机器翻译的评价标准。

## 入门学习资料

如果你对 NLP 和深度学习基础不熟悉，建议先按 [`learn/README.md`](./learn/README.md) 的顺序阅读。资料覆盖词元与词向量、Transformer、训练/推理、mask 和 BLEU，并包含实现前的自测题。


实现和说明中应重点关注以下内容：

- Encoder 和 Decoder 的基本结构。
- Self-Attention 与 Multi-Head Attention。
- Feed-Forward Network（FFN）。
- Residual Connection 与 Layer Normalization。
- Positional Encoding。
- Seq2Seq 机器翻译流程。

## 评价要求

使用 BLEU（Bilingual Evaluation Understudy）评价机器翻译结果。报告中应说明 BLEU 的计算配置，包括使用的 n-gram 设置、参考译文配置以及长度惩罚（brevity penalty）等相关信息。

## 报告中应包含

- 实验背景和问题介绍。
- 当前研究现状。
- 模型与算法设计。
- 实验结果与分析。
- 总结。
- 组员具体分工。

## 依据

本要求整理自根目录的 `自然语言处理大作业.pdf`，具体数据集、代码实现方式和实验参数需根据实际实验方案确定并如实记录。
