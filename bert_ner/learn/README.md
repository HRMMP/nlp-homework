# BERT 命名实体识别学习路线

本目录面向几乎没有 NLP 基础的读者，目标是帮助你理解使用 BERT 做 NER 的完整链路：文本切分、标签对齐、Token Classification、训练和实体级评估。

## 推荐顺序

1. [NER、BIO 标注和标签对齐](./01_ner_and_labels.md)：先理解任务数据长什么样，以及 subword 为什么会造成麻烦。
2. [BERT 和 Transformer Encoder](./02_bert.md)：理解预训练模型读入文本后产生什么表示。
3. [微调、预测与实体级指标](./03_finetune_and_metrics.md)：把分类头、训练流程和 Precision/Recall/F1 串起来。

## 你需要先会什么

- Python 的字符串、列表、字典和文件读写。
- 分类问题的基本概念：输入、标签、预测和损失。
- 张量形状的基本读法，例如 `(batch_size, sequence_length, hidden_size)`。
- 训练循环的基本步骤：前向计算、损失、反向传播、参数更新。

## 完成实验前的知识检查

你应该能够解释：

- NER 和普通文本分类有什么不同。
- `B-PER`、`I-PER`、`O` 分别表示什么。
- 为什么一个原始词可能被 tokenizer 切成多个 subword。
- 特殊 token、padding token 为什么通常不应该参与实体标签损失。
- BERT 的输出如何经过一个线性层变成每个位置的标签概率。
- 为什么 token accuracy 不能替代实体级 Precision、Recall 和 F1。

## 和后续代码的对应关系

| 学习概念 | 实现时通常对应 |
| --- | --- |
| BIO/BIOES 标签 | 标签表和数据集读取 |
| subword 对齐 | tokenizer 的 `word_ids()` 或等价信息 |
| attention mask | 告诉 BERT 哪些位置是真实输入 |
| BERT hidden states | 每个 token 的上下文表示 |
| token classification head | hidden state 到标签 logits 的线性层 |
| `-100` 等忽略标签 | loss 中排除特殊 token/padding |
| seqeval 等实体级评估 | Precision、Recall、F1 计算 |

数据集、实体类别、预训练模型、BIO/BIOES 选择和评估库需要根据实际实验确认并记录。本资料不虚构这些信息。

## 建议的学习方式

先拿一条短句，手动写出“词、subword、标签”的三列表格。然后只用少量样本做标签对齐测试，最后才开始微调。任何时候如果预测结果长度和原始标签长度对不上，应先检查对齐和 mask，而不是盲目调学习率。
