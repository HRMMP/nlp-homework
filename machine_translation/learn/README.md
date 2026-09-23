# 机器翻译学习路线

本目录面向几乎没有 NLP 基础的读者，目标是帮助你理解并逐步实现一个基于 Transformer 的机器翻译实验。这里讲的是完成实验所需要的知识，不包含任何未经运行的实验结果。

## 推荐顺序

1. [NLP、词元和词向量](./01_nlp_and_embeddings.md)：先弄清楚文本如何变成模型可以处理的数字。
2. [Transformer 结构](./02_transformer.md)：理解 Encoder、Decoder、Attention 和训练时的数据流。
3. [训练、推理与 BLEU](./03_training_and_bleu.md)：把数据、损失函数、推理过程和评价指标串起来。

## 你需要先会什么

- 会使用终端进入目录、运行 Python 文件。
- 知道 Python 的变量、函数、列表、字典、循环和类。
- 能阅读矩阵的形状，例如 `(batch_size, sequence_length, hidden_size)`。
- 知道神经网络训练的大致循环：前向计算、计算损失、反向传播、更新参数。

不会这些也可以边学边补。重点是每学一个概念，都能回答“它在翻译程序的哪一步使用”。

## 完成实验前的知识检查

你应该能够用自己的话解释：

- 为什么需要词元、词表、特殊符号和 padding。
- Encoder 和 Decoder 分别接收什么、输出什么。
- 为什么 Decoder 训练时不能看到目标句子的未来词。
- Self-Attention 如何根据上下文混合信息。
- 训练损失为什么使用目标词的下一个词，而不是直接比较整句字符串。
- BLEU 比较什么，为什么还需要长度惩罚。

## 和后续代码的对应关系

| 学习概念 | 实现时通常对应 |
| --- | --- |
| 词元化、词表、特殊 token | 数据预处理和 batch 构造 |
| Embedding、位置编码 | 模型输入层 |
| Encoder、Decoder | Transformer 主体 |
| causal mask、padding mask | Attention 的掩码 |
| teacher forcing | 训练时的 Decoder 输入 |
| greedy/beam search | 推理时生成译文 |
| cross entropy | 训练损失 |
| BLEU | 独立评估脚本 |

具体源语言、目标语言、数据集、词元化方法、模型规模和训练轮数需要根据实际方案选择并记录，不能从本学习资料中直接假定。

## 建议的学习方式

每读完一篇，做文末的自测题；然后用纸或 Markdown 画出一个 batch 从原句到 BLEU 的流程。实现代码时优先先做一个很小的 toy 数据集，确认模型可以过拟合几条样本，再扩大数据规模。
