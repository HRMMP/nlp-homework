# 01. NLP、词元和词向量

## 1. 文本为什么不能直接输入神经网络

计算机里的神经网络主要处理数字。句子“我喜欢机器学习”需要先转换成一串整数，例如：

```text
原句:      我    喜欢    机器    学习
词元:      我    喜欢    机器    学习
词元 ID:   17    203     91     408
```

这里的整数本身没有大小关系。ID 为 408 的词并不比 ID 为 17 的词“更大”或“更重要”。这些 ID 只是词表中的索引。

## 2. 词元、词表和分词

- **词元（token）**：模型实际处理的最小文本单位，可以是字、词、子词或字符。
- **词表（vocabulary）**：从词元到整数 ID 的映射。
- **词元化（tokenization）**：把原始字符串拆成词元。

不同语言和不同 tokenizer 的结果可能不同。例如英文的一个长词可能被拆为多个子词；中文可以按字、词或子词切分。实验必须记录使用的 tokenizer 和词表构建方式，因为它们会影响序列长度和未知词数量。

常见特殊 token：

| token | 用途 |
| --- | --- |
| `<pad>` | 把同一 batch 的序列补到相同长度 |
| `<unk>` | 词表中没有对应词元时使用 |
| `<bos>` | 目标句子的开始 |
| `<eos>` | 目标句子的结束 |
|

机器翻译至少要明确源语言和目标语言各自的词表，以及 Decoder 使用的开始和结束标记。

## 3. 为什么需要 Embedding

Embedding 是一个可训练矩阵：

```text
embedding_table.shape = (vocab_size, embedding_dim)
```

词元 ID 只是查表的行号。查表后，每个词元得到一个向量：

```text
ids.shape       = (batch_size, sequence_length)
embeddings.shape = (batch_size, sequence_length, embedding_dim)
```

训练过程中，Embedding 矩阵会和其他模型参数一起更新，因此相似上下文中的词元可能逐渐获得有用的表示。Embedding 向量的每一维通常没有单独、固定的人类含义，不要把它当成手工设计的特征。

## 4. 位置为什么重要

只使用词元 Embedding 时，模型很难区分：

```text
“猫追狗” 和 “狗追猫”
```

两句话包含相同的词元集合，但顺序不同。Transformer 没有循环结构，所以需要额外加入位置信息。位置编码可以是固定的正弦/余弦编码，也可以是可学习的位置 Embedding；具体方法要在实验配置中记录。

通常输入表示类似：

```text
x = token_embedding + positional_embedding
```

两者形状必须一致，才能逐元素相加。

## 5. Padding 和 mask

同一个 batch 中的句子长度可能不同：

```text
[我, 喜欢, 猫, <eos>]
[他, 学习, <eos>, <pad>]
```

`<pad>` 只是为了对齐长度，不是真实内容。Attention 和损失函数都应该知道哪些位置是 padding，否则模型可能学习到无意义的信息。

因此通常需要：

- `padding_mask`：阻止注意力读取 padding 位置。
- `loss mask` 或 `ignore_index`：不对目标中的 padding 位置计算损失。

## 6. 一个 batch 中的数据

机器翻译的一个样本通常是源句和目标句：

```text
source: [src_bos, 我, 喜欢, 猫, src_eos]
target: [tgt_bos, I, like, cats, tgt_eos]
```

训练 Decoder 时，输入和监督目标会错开一位：

```text
Decoder 输入: [tgt_bos, I, like, cats]
监督目标:    [I, like, cats, tgt_eos]
```

这样模型在看到前面的目标词后，学习预测下一个词。

## 7. 最小自测

1. 词元 ID 为什么不能直接当作词向量？
2. 如果 batch 中最长句子长度为 10，较短句子为什么需要 padding？
3. `<bos>` 和 `<eos>` 分别解决什么问题？
4. 为什么“我喜欢猫”和“猫喜欢我”不能只用词袋表示？
5. 写出 Decoder 输入和监督目标的错位关系。

如果你能回答这些问题，再阅读 Transformer 结构会容易很多。

## 进一步阅读

- [PyTorch Embedding 文档](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
