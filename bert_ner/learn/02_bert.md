# 02. BERT 和 Transformer Encoder

## 1. BERT 解决什么问题

BERT 是一个基于 Transformer Encoder 的预训练语言模型。它先在大量无标注文本上学习通用的上下文表示，再针对 NER 这类下游任务微调。

“上下文表示”意味着同一个词在不同句子中可以根据周围词得到不同向量。例如“苹果”可能是水果，也可能是公司名称，模型需要结合上下文判断。

## 2. BERT 的输入表示

BERT 的输入表示通常由三部分相加：

```text
input_representation
  = token_embedding
  + position_embedding
  + segment_embedding
```

- Token Embedding：当前 token 的语义表示。
- Position Embedding：序列位置。
- Segment Embedding：区分句子 A 和句子 B；单句 NER 中通常是同一个 segment。

输入还会带有 `attention_mask`，用来区分真实 token 和 padding。

## 3. BERT 的输出

对于输入长度为 `L` 的序列，BERT 会产生：

```text
last_hidden_state.shape = (batch_size, L, hidden_size)
```

其中每一个位置都有一个 `hidden_size` 维向量。NER 正好需要每个位置的表示，所以可以在每个位置接一个分类层。

## 4. 从 BERT 到 Token Classification

最简单的 NER 分类头是一个线性层：

```text
logits = hidden_states W + b
```

如果标签数是 `num_labels`，则：

```text
logits.shape = (batch_size, sequence_length, num_labels)
```

对每个有效位置取最大 logit 对应的标签，就得到预测标签。训练时对有效位置计算交叉熵，对特殊 token 和 padding 位置忽略。

## 5. 预训练和微调

预训练阶段，模型先通过自监督任务从大量文本学习，例如遮住部分 token，让模型根据上下文预测它们。下游微调阶段：

1. 加载预训练 BERT。
2. 增加 NER 分类头。
3. 使用有标签的 NER 训练集计算损失。
4. 更新 BERT 和分类头（除非显式采用冻结策略）。
5. 在验证集和测试集上预测。

预训练模型的名字、语言、大小和 tokenizer 必须与实验实际使用的模型一致。不要只写“使用了 BERT”而不记录具体模型标识。

## 6. Encoder 与 Decoder 的区别

BERT 使用 Encoder。Encoder 的 self-attention 可以看到序列中所有非 padding 位置，因此适合理解整句文本。

这和机器翻译 Decoder 的 causal attention 不同：Decoder 在生成位置 `t` 时不能读取未来位置。不要因为两者都基于 Transformer，就把 mask 规则混为一谈。

## 7. 需要检查的张量

训练第一批数据时建议打印或断言：

```text
input_ids:      (B, L)
attention_mask: (B, L)
labels:         (B, L)
logits:         (B, L, C)
```

这里 `B` 是 batch size，`L` 是 tokenizer 后长度，`C` 是标签数量。`labels` 中忽略位置应和 `attention_mask`、subword 策略相符。

## 8. 最小自测

1. BERT 的每个输入位置最终得到什么？
2. 为什么 NER 可以直接使用每个 token 的 hidden state？
3. 分类头的输出维度为什么是标签数，而不是词表大小？
4. BERT Encoder 的注意力和生成模型的 causal attention 有什么区别？
5. 为什么必须固定 tokenizer 和预训练模型的对应关系？

## 进一步阅读

- [BERT 原论文](https://aclanthology.org/N19-1423/)
- [Hugging Face Token Classification 任务指南](https://huggingface.co/docs/transformers/tasks/token_classification)
- [BERT 模型文档](https://huggingface.co/docs/transformers/model_doc/bert)
