# BERT 命名实体识别实验

## 数据集

本实验使用本地 `data/cluener2020/` 中的 CLUENER 2020 中文细粒度 NER 数据。标注给出实体类别和原文字符的闭区间位置；程序会校验区间能否还原实体文本，并将字符序列转换为 BIO。类别共 10 种，BIO 标签共 21 种（含 `O`）。

本仓库中的 `train.json` 有 10,748 条、`dev.json` 有 1,343 条、`test.json` 有 1,345 条。当前 test 文件不包含标签，不能在其上计算 P/R/F1；训练期间以 dev 集选择最佳模型，test 只能用于无标签预测。train 中有 1 条样本包含 2 个嵌套实体，扁平 BIO 无法同时表达；转换时保留较长的外层实体并计数。正式报告应披露这一处理。

数据说明及上游项目：<https://github.com/CLUEbenchmark/CLUENER>。本地数据包没有明确提供许可证文件，使用与再分发许可需向数据上游核实，不应据此推断许可证。

## 环境

建议使用 Python 3.10 或 3.11。先根据机器的 CPU/CUDA 情况从 PyTorch 官方页面安装 PyTorch，再安装本实验依赖：

```bash
python -m pip install -r bert_ner/requirements.txt
```

第一次训练需要能访问模型仓库并下载 `bert-base-chinese` 权重；也可以在配置中将 `model_name` 改为本地模型路径，并记录实际来源与版本。

## 数据审计和测试

在仓库根目录运行：

```bash
python bert_ner/run_ner.py audit
cd bert_ner && python -m unittest discover -s tests -v
```

## 训练

默认配置在 `bert_ner/config.json`，使用 `bert-base-chinese`、最大长度 256、随机种子 42、3 个 epoch。本机已完成一次训练，dev 实体级 Precision / Recall / F1 为 77.16% / 80.18% / 78.64%。dev 同时用于选模，不能把这些数字描述为独立测试成绩。

```bash
python bert_ner/run_ner.py train --config bert_ner/config.json
```

```bash
python bert_ner/run_ner.py evaluate bert_ner/outputs/bert-base-chinese/dev_predictions.json
```

训练过程每个 epoch 在 dev 上评估并保存检查点，按实体级 F1 选择最佳模型。产物写入 `bert_ner/outputs/bert-base-chinese/`，包含实际配置、环境版本、独立可复算的 dev 预测、结果摘要和 `best_model/`。`outputs/` 被 Git 忽略，不要提交模型权重。

## 实现与评估约定

- 原始字符区间为闭区间，BIO 标签遵循 IOB2。
- 通过 fast tokenizer 的 `word_ids()` 对齐原始字符和 subword；特殊 token 与同一字符的后续 subword 使用 `-100` 忽略损失及评估。
- 实体指标由 `seqeval` 按 strict IOB2 计算 micro Precision、Recall、F1；`O` 不作为实体类别计分。
- `test.json` 是无标签数据，当前流程不对其产生测试指标。

## 保存预测的错误审计

无需 GPU 或重新训练：

```bash
CUDA_VISIBLE_DEVICES='' python bert_ner/analyze.py \
  --output reports/evidence/ner_analysis.json
```

工具用保存的 fast tokenizer 回查 gold/字符位置，用 seqeval strict IOB2 提取实体，并验证跨度 TP/FP/FN 与实体指标一致。当前 TP=2,463、FN=609、FP=729；FN 中同类别错边界 257、同边界错类别 152、边界和类别均不同且有重叠 51、无重叠预测 149。FP 独立按相同优先规则分类，各侧每个错误实体只归一类，不宣称完成一对一错误匹配。

按“实体原文＋类别是否在 train 出现”分组，seen / unseen 的 F1 为 87.02% / 65.17%，对应 gold 支持数 1,908 / 1,164；这不是分词器 OOV 或独立泛化测量。JSON 包含完整 10 类指标、同边界类别混淆、输入哈希和固定 seed 的候选错误实例。实例是自动抽样，仍须学生核验解释。当前静态检查问题已修正，未改变有效输入的标签转换或模型预测。

本机已有训练检查点、dev 预测和报告审阅稿。本次使用已保存的 tokenizer 回查 1,343 条 dev，gold 标签全部一致，dev 无字符因当前分词/截断失去监督位置；train/dev 原始文本精确交集为 0，不代表已排除近重复。仍须学生核验错误案例、确认数据许可，并决定是否另行设计有标签独立评测；不能将当前模型已训练过的样本临时划成 test。实验完善范围见 [`../reports/experiment_plan.md`](../reports/experiment_plan.md)。
