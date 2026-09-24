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

默认配置在 `bert_ner/config.json`，使用 `bert-base-chinese`、最大长度 256、随机种子 42、3 个 epoch。配置是待验证的起始参数，不代表已经运行或取得任何结果。

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

完整训练、正式结果、错误分析及许可证确认仍待实际执行和核验；本仓库不会预填实验数值。
