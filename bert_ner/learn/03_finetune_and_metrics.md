# 03. 微调、预测与实体级指标

## 1. 一个清晰的微调流程

建议把流程拆成可独立检查的步骤：

1. 读取原始样本和标签。
2. 划分 train、validation、test，并记录划分规则和随机种子。
3. tokenizer 编码文本。
4. 按 `word_ids()` 等信息完成标签对齐。
5. 对齐后的 batch 输入 BERT 和分类头。
6. 只在有效标签位置计算 loss。
7. 在验证集选择 checkpoint。
8. 在测试集生成预测，并独立计算指标。

测试集只在最终评估使用。不能用测试集反复选择模型、调整标签对齐规则或调超参数后仍把它当作未使用的测试集。

## 2. Precision、Recall 和 F1

把完整实体当作预测对象时：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

- TP：预测的实体在真实数据中存在，并且边界与类别都正确。
- FP：预测了错误或不存在的实体。
- FN：真实实体没有被正确预测。

Precision 高表示预测出来的实体较少出错；Recall 高表示真实实体中找回的比例较高。F1 是二者的调和平均。

## 3. 为什么不能只用 token accuracy

NER 中 `O` 往往占大多数。一个模型把所有位置都预测为 `O`，token accuracy 可能看起来不低，但它找不到任何实体，实体 Recall 会很差。

实体级评估通常需要：

1. 把 BIO/BIOES 标签解析为实体区间和类别。
2. 比较预测实体集合与真实实体集合。
3. 统计 TP、FP、FN。
4. 计算 Precision、Recall、F1。

报告应说明使用了哪一个评估库、哪种平均方式（例如 micro 或 macro，若库提供）、是否忽略 `O`，以及标签非法时如何处理。

## 4. 预测保存格式

不要只保存一个 F1 数字。建议保存 JSON 或 JSONL，包含：

```json
{
  "tokens": ["张", "三", "在", "北京"],
  "gold_labels": ["B-PER", "I-PER", "O", "B-LOC"],
  "pred_labels": ["B-PER", "I-PER", "O", "B-ORG"]
}
```

评估脚本应能从这个文件重新计算指标。这样可以发现“训练时的指标”和“独立评估脚本的指标”是否一致，也方便错误分析。

## 5. 常见错误分析

至少查看这些类型：

- 边界错误：实体多包含或少包含一个 token。
- 类别混淆：例如地点被预测为组织。
- 漏检：真实实体没有被识别。
- 过检：普通词被错误识别为实体。
- 未知或新实体：训练数据中没有见过的名称。
- 长实体和 subword：切分或对齐导致类别错误。

每个现象都应引用真实样本，不要只写抽象结论。

## 6. 一个最小验证清单

- [ ] 随机抽取样本，人工检查 token 和 label 对齐。
- [ ] 检查 `[CLS]`、`[SEP]`、padding 和非首 subword 是否被正确忽略或标注。
- [ ] 检查 logits 的最后一维等于标签数量。
- [ ] 用一个极小数据集验证 loss 可以下降。
- [ ] 对一组手写 BIO 标签验证实体解析器。
- [ ] 评估脚本能从保存的预测独立复算 Precision、Recall、F1。

## 7. 最小自测

1. 一个实体边界错一位时，为什么可能同时增加 FP 和 FN？
2. Precision 很高但 Recall 很低，模型可能有什么行为？
3. 为什么保存 gold/pred labels 比只保存 F1 更有用？
4. 训练集、验证集和测试集在 NER 实验中分别承担什么作用？
5. 你会如何验证 subword 对齐逻辑，而不是等训练后才发现错误？

## 进一步阅读

- [seqeval 文档](https://github.com/chakki-works/seqeval)
- [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
