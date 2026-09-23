# 09. BERT 命名实体识别实验

## 1. 先定义 NER 输出

NER 不是判断整句话属于哪个类别，而是找出实体的起止位置并判断类别。例如：

```text
句子：张三在北京大学学习。
实体：张三 -> PER
      北京大学 -> ORG（具体类别以数据集定义为准）
```

如果使用 BIO：

```text
词：    张    三    在    北京    大学    学习
标签：  B-PER I-PER O     B-ORG   I-ORG O
```

BIOES 还会使用 `E-X` 表示结束、`S-X` 表示单 token 实体。BIO 和 BIOES 不能不加说明地混用。

## 2. 数据格式和标签表

开始训练前确认：

- 原始句子是按字符、词还是其他单位标注。
- 每条句子的 token 数和标签数是否相等。
- 实体类别名称和大小写。
- 标签是否满足 BIO/BIOES 约束。
- 训练、验证和测试如何划分，是否有重复句子。

建立显式映射，例如：

```text
O      -> 0
B-PER  -> 1
I-PER  -> 2
B-ORG  -> 3
I-ORG  -> 4
```

真实标签表应从数据集和配置产生，不要手写一套与数据不一致的类别。

## 3. BERT + Token Classification

BERT 输出每个输入位置的隐藏向量：

```text
hidden_states: (B, L, H)
```

接一个线性分类头：

```text
logits: (B, L, C)
```

对每个有效位置取类别分数最大的标签。特殊 token、padding 和不参与损失的 subword 通常使用忽略标签，例如 `-100`；实际值要与 loss 配置一致。

## 4. Subword 对齐

原始标注可能按词提供，但 BERT tokenizer 可能把一个词拆开。建议打印一条完整的对齐表：

| subword | 原始词索引 | 原始标签 | 训练标签 |
| --- | ---: | --- | --- |
| `[CLS]` | - | - | `-100` |
| 张 | 0 | `B-PER` | `B-PER` |
| 三 | 1 | `I-PER` | `I-PER` |
| `[SEP]` | - | - | `-100` |

当一个词被拆成多个 subword 时，需要选择策略：

1. 只给第一个 subword 计算损失，其余忽略。
2. 首个 subword 使用 `B-X`，后续使用 `I-X`。
3. 复制原始标签到所有 subword，但必须确认评估逻辑接受这种形式。

不同策略不应混合使用。训练标签、预测还原和实体级评估必须相互一致。

## 5. 训练流程

```text
原始 token + 标签
  -> tokenizer
  -> word_ids() 或等价映射
  -> labels 对齐
  -> BERT
  -> 分类头
  -> 忽略特殊位置计算 loss
  -> 验证集选择 checkpoint
  -> 测试集预测
  -> 解析实体并计算指标
```

第一批数据至少打印：

```text
input_ids:      (B, L)
attention_mask: (B, L)
labels:         (B, L)
logits:         (B, L, C)
```

并人工查看真实标签和预测标签的长度是否对应。

## 6. 实体级 Precision、Recall、F1

不能只逐 token 比较。先把 BIO 标签解析成实体集合，例如：

```text
(B-PER, I-PER, O) -> 实体区间 [0, 2), 类别 PER
```

只有边界和类别都正确，才算一个 TP。预测了不存在的实体是 FP，漏掉真实实体是 FN。然后计算：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

报告要写清楚评估库、标签解析方式、micro/macro 选择（如果适用）以及是否忽略 `O`。`O` 通常不是实体类别，但不能只因为它数量多就用 token accuracy 代替实体指标。

## 7. 错误分析

从保存的真实预测中挑选样例，按现象分析：

- 边界多一个或少一个 token。
- 地点、机构、人名等类别混淆。
- 真实实体漏检。
- 普通词被错误识别为实体。
- 训练集中未出现的新实体。
- 长实体或 subword 切分导致的错误。

每条结论都应引用真实样本。不能只写“模型可能受到数据量影响”而不说明哪个样本体现了什么现象。

## 8. 评估文件建议

```json
{
  "tokens": ["张", "三", "在", "北京"],
  "gold_labels": ["B-PER", "I-PER", "O", "B-LOC"],
  "pred_labels": ["B-PER", "I-PER", "O", "B-ORG"]
}
```

评估脚本应能够从保存的预测文件独立重新计算 Precision、Recall、F1。训练脚本打印的指标不能替代独立评估。

## 9. 学习任务

1. 手工把三条 BIO 标签序列解析成实体区间。
2. 构造一个“边界错一位”的例子，说明为什么可能同时产生 FP 和 FN。
3. 对一条被拆成多个 subword 的样本制作对齐表。
4. 写一个极小的实体解析器测试表，包括单 token、相邻实体和非法标签。
5. 用极小样本检查 loss 能否下降。

## 10. 对应的专项材料

- [NER、BIO 标注和标签对齐](../bert_ner/learn/01_ner_and_labels.md)
- [BERT 和 Transformer Encoder](../bert_ner/learn/02_bert.md)
- [微调、预测与实体级指标](../bert_ner/learn/03_finetune_and_metrics.md)
- [BERT NER 实验 README](../bert_ner/README.md)

## 11. 实验完成标准

需要有数据与标签说明、tokenizer 和对齐测试、训练配置、checkpoint 选择规则、保存的 gold/pred 标签、独立实体级指标和真实错误样例。没有运行指标时，不应在报告中填写看起来完整但未经验证的数值。
