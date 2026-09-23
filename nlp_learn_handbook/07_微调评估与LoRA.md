# 07. 微调、评估与 LoRA

## 1. 什么是微调

微调是在预训练模型参数基础上，使用特定任务数据继续训练。最简单的全量微调会更新模型的大量甚至全部参数，但这对大模型可能需要较多显存、时间和存储空间。

微调实验至少要记录：

- 基础模型的准确标识和版本。
- tokenizer 和最大长度。
- 训练数据、划分和预处理。
- 学习率、batch size、epoch、梯度累积和随机种子。
- 哪些参数可训练。
- checkpoint 的选择规则。

## 2. LoRA 的核心公式

设基础线性层为：

```text
y = W x
```

LoRA 冻结 `W`，学习一个低秩增量：

```text
y = W x + (alpha / r) B A x
```

其中：

- `W`：预训练权重，通常冻结。
- `A`、`B`：新增可训练矩阵。
- `r`：低秩 rank。
- `alpha`：缩放因子。
- `alpha / r`：控制增量分支的尺度。

若 `W` 是 `d_out x d_in`，全量更新约有 `d_out * d_in` 个参数；LoRA 增量约有：

```text
r * d_in + d_out * r
```

当 `r` 比两个原始维度小很多时，可训练参数明显减少。

## 3. 为什么必须检查可训练参数

“使用了 PEFT 库”不等于实验一定是 LoRA。配置后应该统计：

```text
总参数量
可训练参数量
可训练参数占比
```

并检查基础模型参数的 `requires_grad` 是否为 `False`、LoRA 参数是否为 `True`。还要确认 `target_modules` 真的存在于当前基础模型架构中。

如果所有参数都在训练，或目标模块没有任何可训练参数，实验都可能不符合原定设计。

## 4. LoRA 配置项

- `r`：适配器容量和可训练参数量的主要控制项。
- `lora_alpha`：缩放增量分支。
- `lora_dropout`：LoRA 分支的正则化。
- `target_modules`：要插入适配器的线性层名称，例如某些注意力投影；名称依赖模型架构。
- `bias`：是否训练 bias，需要如实记录。

这些不是可以脱离模型和数据集背诵的“最佳答案”。应使用配置文件保存，先通过小实验比较，再在报告中说明选择依据。

## 5. 数学任务的训练目标

数学题生成任务通常是 causal language modeling。可以有两类常见 loss 设计：

1. 对 prompt 和回答的完整拼接序列都计算 loss。
2. 对 prompt 部分使用忽略标签，只对回答部分计算 loss。

第二种方式更直接地让模型学习回答，但可能需要更仔细地构造标签边界；两种方案不能不说明地混用。

## 6. 生成、抽取和判分是三件事

模型可能生成：

```text
先计算 ... 因此最终答案是 42。
```

评估还要：

1. 从原始文本中抽取答案。
2. 按明确规则归一化格式。
3. 与参考答案比较。
4. 保存抽取失败或解析异常的原因。

建议每道题保存：

```json
{
  "question_id": "...",
  "reference_answer": "42",
  "generation": "原始模型输出",
  "extracted_answer": "42",
  "correct": true,
  "error_reason": null
}
```

只保存最终 Accuracy 会丢失关键证据，无法区分模型算错、模型格式不对和解析器出错。

## 7. 指标基础

### 7.1 Precision、Recall 和 F1

对实体集合而言：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

NER 中 TP 必须同时满足实体边界和实体类别正确。预测少了会降低 Recall，预测了很多错误实体会降低 Precision。

### 7.2 BLEU

BLEU 通过候选译文和参考译文的 n-gram 重叠计算分数，并加入 brevity penalty 防止过短输出获得不合理优势。报告必须说明实现、tokenization、大小写、n-gram 阶数、平滑和参考译文数量。

### 7.3 Accuracy

数学题准确率是：

```text
Accuracy = 判对题数 / 测试题总数
```

必须先明确什么叫“判对”。答案抽取、数字归一化、分数和单位处理都会影响结果。

## 8. 数据泄漏

常见泄漏包括：

- 用测试集反复选择模型和超参数。
- 把测试答案或测试标签写进 prompt。
- 词表、归一化规则或去重规则使用了测试集信息。
- 训练和测试包含重复题目或同源改写，却未检查。

发现测试集规则需要修改时，应记录这一事实，并重新划分或重新定义评估，不能悄悄替换规则后只保留较高分数。

## 9. 本章自测

1. LoRA 和全量微调分别更新什么？
2. 如何用参数统计证明基础模型被冻结？
3. `r`、`alpha` 和 `target_modules` 各控制什么？
4. 为什么生成文本、抽取答案和 Accuracy 不能混成一个函数？
5. NER 为什么不能只报告 token accuracy？

## 10. 建议的小作业

设计一个小型数学答案抽取器的测试表，至少包括正整数、负数、分数、带单位、没有最终答案标记和输出多个数字的情况。先写判分规则，再实现解析器；不要为了让示例通过而临时增加没有记录的特殊规则。

## 进一步阅读

- Hu et al., [LoRA 原论文](https://arxiv.org/abs/2106.09685)。
- [Hugging Face PEFT 文档](https://huggingface.co/docs/peft/index)。
- [Hugging Face Token Classification 指南](https://huggingface.co/docs/transformers/tasks/token_classification)。
- [SacreBLEU](https://github.com/mjpost/sacrebleu)。
