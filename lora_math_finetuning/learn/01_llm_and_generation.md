# 01. 语言模型、提示词和生成

## 1. 语言模型在做什么

自回归语言模型接收一段 token 前缀，预测下一个 token 的概率：

```text
输入:  The answer is
输出: answer 后面每个候选 token 的概率
```

生成时，模型把选出的 token 接回输入，再预测下一步：

```text
prompt -> token_1 -> token_2 -> token_3 -> ...
```

GPT 类模型使用 Transformer Decoder 风格的 causal self-attention，因此当前位置只能读取当前位置及之前的 token。

## 2. Token、logits 和概率

模型最后通常输出 logits，形状大致为：

```text
(batch_size, sequence_length, vocabulary_size)
```

logits 经过 softmax 才变成概率。生成策略可以选择概率最高的 token，也可以按概率随机采样。常见参数包括 temperature、top-k、top-p 和最大新 token 数；正式评估必须记录这些参数，因为它们会影响答案。

## 3. 什么是 prompt

Prompt 是交给模型的输入模板。数学任务可能把题目、解题要求和输出格式组合起来：

```text
问题：{question}
请给出推理，并在最后写出最终答案。
答案：
```

模板不是无关紧要的字符串，它决定模型看到的任务形式。训练和评估应使用一致的模板规则，除非实验明确比较不同模板。

## 4. 预训练和指令微调

- **预训练**：通常使用大量文本做下一个 token 预测，学习语言和知识。
- **指令微调**：使用“指令/问题 -> 期望回答”样本，让模型更倾向于遵循任务要求。

数学微调通常属于有监督微调：给模型题目和参考回答，让它学习生成参考回答的 token。基础模型是否为 instruction-tuned、训练数据的回答格式，都要在实验记录中说明。

## 5. 训练时的 labels

在 causal language modeling 中，模型学习预测序列中每个位置的下一个 token。实现通常会在内部或数据整理阶段完成 shift：

```text
输入 token:  A  B  C
目标 token:     B  C  D
```

如果只希望模型学习回答而不学习 prompt，可以对 prompt 部分使用忽略标签，只在回答部分计算 loss。这是一个实验设计选择，必须明确记录，不应不加说明地混用。

## 6. 生成答案不等于评估答案

模型可能输出：

```text
先计算得到 ... 因此最终答案是 42。
```

评估脚本还需要从这段文本中抽取 `42`，再和参考答案比较。因此生成、答案抽取、答案归一化和判分是四个不同步骤。

## 7. 最小自测

1. 为什么自回归模型要逐步生成？
2. prompt 模板改变后，为什么模型输出可能改变？
3. logits 和概率有什么区别？
4. 为什么模型生成了一段看似合理的解释，仍然可能在 Accuracy 中判错？
5. 训练时只对回答部分计算 loss 有什么可能的好处和代价？

## 进一步阅读

- [Hugging Face Causal Language Modeling 指南](https://huggingface.co/docs/transformers/tasks/language_modeling)
- [GPT-3 论文](https://arxiv.org/abs/2005.14165)
