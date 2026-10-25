# LoRA 数学大模型微调实验

本实验使用 `openai/gsm8k` 数学数据集，在 `Qwen/Qwen2.5-1.5B-Instruct` 上进行 LoRA 参数高效微调，并与未微调的同一基础模型比较测试集 Accuracy。训练和推理均使用模型自带的 Chat Template；prompt 区域不计入训练 loss，只监督 assistant 的解题过程和最终答案。

数据集与模型均锁定 revision，避免上游默认分支变化导致无法复现：GSM8K revision 为 `740312add88f781978c0658806c59bc2815b9866`，模型 revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。GSM8K `main` 配置包含 7,473 条训练题和 1,319 条测试题，数据卡标注 MIT 许可证；正式运行时审计脚本会再次记录样本数和答案格式。

LoRA 正式训练与完整测试集评估已完成。实际结果及其限制见文末“正式运行结果”，不可将 smoke test 指标作为正式结果。

## 实验方案

- 数据集：`openai/gsm8k`，使用官方 `train` 和 `test` 划分。
- 验证集：从官方训练集按照配置中的 `validation_ratio` 和随机种子划分，不使用官方测试集调参。
- 基础模型：`Qwen/Qwen2.5-1.5B-Instruct`。
- 微调：冻结基础模型，只训练 PEFT LoRA adapter。
- `r=8`、`alpha=16`、`dropout=0.05`，目标模块为 `q_proj` 和 `v_proj`。
- prompt 区域标签设为 `-100`，因此不参与 causal language modeling loss；assistant 区域使用 GSM8K 的逐步解答及 `####` 最终答案训练。
- 指标：优先抽取模型输出中最后一个 `####` 标记后的内容；没有该标记时，取最后一个 `\boxed{...}`；再计算测试集 Accuracy。
- 基线：同一基础模型、同一 prompt、同一解码参数和同一答案抽取规则，但不加载 LoRA。

所有超参数和本次运行配置见 [`config.json`](./config.json)。改变配置后请使用新的输出目录，保留各次实验的配置与日志。基础概念和代码对应关系见 [`learn/README.md`](./learn/README.md)。

## 环境安装

建议在仓库根目录创建独立环境。先根据 CUDA 版本从 PyTorch 官方页面安装适配的 PyTorch，再安装本实验依赖：

```bash
python -m venv .venv-lora
source .venv-lora/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r lora_math_finetuning/requirements.txt
python -m pip install 'peft==0.21.0'
```

首次运行需要下载数据集和模型。请记录 Python、PyTorch、Transformers、PEFT、CUDA 和 GPU 版本；不要把模型权重、数据缓存或输出文件提交到仓库。

## 数据审计

先检查题目字段、答案字段和 `####` 最终答案格式：

```bash
python -m lora_math_finetuning.audit_data
```

程序应报告官方训练集和测试集的样本数，以及无法抽取最终答案的样本数。若出现异常，应先检查数据版本，不要直接修改答案抽取规则来追求更高分数。

## 单元测试

这些测试不需要下载模型：

```bash
python -m unittest discover -s lora_math_finetuning/tests -v
```

测试覆盖：

- `####` 最终答案抽取及 `\boxed{...}` 回退；
- 多个答案标记时选择最后一个；
- 缺少标记时记录抽取失败；
- 整数、小数、分数和单位的保守归一化；
- Chat Template 下的 prompt/assistant token 边界和 prompt loss mask；
- LoRA 总参数量和可训练参数量统计。

## 基线评估

先运行未微调基础模型基线。为了先验证完整流程，可以只评估少量测试题；正式评估时去掉 `--limit`：

为了保留已有证据，下面重新生成的示例使用 `reproduce-*` 前缀；重复运行时仍应另选新路径。GPU 任务可用 `scripts/run_on_idle_gpu.py` 包裹，动态检查空闲卡，具体用法见翻译 README。

```bash
python -m lora_math_finetuning.evaluate generate \
  --config lora_math_finetuning/config.json \
  --output lora_math_finetuning/outputs/reproduce-baseline-test.jsonl \
  --limit 20

python -m lora_math_finetuning.evaluate score \
  --predictions lora_math_finetuning/outputs/reproduce-baseline-test.jsonl \
  --predictions-output lora_math_finetuning/outputs/reproduce-baseline-test.scored.v2.jsonl \
  --output lora_math_finetuning/outputs/reproduce-baseline-test.metrics.v2.json \
  --analysis-output lora_math_finetuning/outputs/reproduce-baseline-test.errors.v2.jsonl
```

`--limit 20` 只用于流程检查，不能作为正式测试结果。

## LoRA smoke test

先用极小数据检查前向、反向、保存和加载 adapter：

```bash
python -m lora_math_finetuning.train \
  --config lora_math_finetuning/config.json \
  --max-train-samples 20 \
  --max-eval-samples 5 \
  --max-steps 5
```

训练输出目录会保存配置、环境信息、数据切分清单、参数统计、Trainer 日志和 `final_adapter/`。重点检查：

- 基础模型参数被冻结；
- LoRA 参数可训练；
- 可训练参数量明显少于总参数量；
- adapter 能够保存并重新加载。

smoke test 结果不能写成正式实验结果。

## 学习率对照实验

本次完成了一轮仅改变学习率的对照实验：已有 `lr=2e-4` adapter，加上新训的 `lr=1e-4`、`5e-5`。其余设置固定为 `r=8, alpha=16, dropout=0.05`、1 epoch、seed 42、有效 batch size 16。候选配置保存在 [`configs/sweep_lr1e-4.json`](./configs/sweep_lr1e-4.json) 和 [`configs/sweep_lr5e-5.json`](./configs/sweep_lr5e-5.json)，两组训练分别输出到独立目录。

当前 test 集及错误案例此前已经查看过，因此没有用 test 分数选参。配置选择只参考固定 seed 的完整 validation split；现有 test 结果应视为探索性结果。若需要严格的后续确认，应另用一个未查看过的独立评测集。

本次只使用 GPU 0 顺序运行；基线和三个 adapter 都使用相同 greedy 解码，在完整 validation split（748 题）上评估。按严格 Accuracy，`lr=1e-4` 本次最高，比 `2e-4` 多答对 7 题（0.94 个百分点）；差距较小且每组只有一个 seed，不能据此认定它稳定更优。

本轮 sweep 未在 official test 上评估新 adapter，避免把已经查看过的 test 集继续用于配置选择。`1e-4` 目前只是 validation 上的候选值，不应替换原 test 结果或被描述为已验证的最终最优模型。

训练命令（从仓库根目录运行）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m lora_math_finetuning.train \
  --config lora_math_finetuning/configs/sweep_lr1e-4.json

CUDA_VISIBLE_DEVICES=0 python -m lora_math_finetuning.train \
  --config lora_math_finetuning/configs/sweep_lr5e-5.json
```

验证集生成示例（以下以 `lr=1e-4` 为例；其余 adapter 使用对应的 config 和输出目录）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m lora_math_finetuning.evaluate generate \
  --config lora_math_finetuning/configs/sweep_lr1e-4.json \
  --adapter-path lora_math_finetuning/outputs/sweep-lr1e-4/final_adapter \
  --split validation \
  --output lora_math_finetuning/outputs/reproduce-lr1e-4-validation.jsonl

python -m lora_math_finetuning.evaluate score \
  --predictions lora_math_finetuning/outputs/reproduce-lr1e-4-validation.jsonl \
  --predictions-output lora_math_finetuning/outputs/reproduce-lr1e-4-validation.scored.v2.jsonl \
  --output lora_math_finetuning/outputs/reproduce-lr1e-4-validation.metrics.v2.json \
  --analysis-output lora_math_finetuning/outputs/reproduce-lr1e-4-validation.errors.v2.jsonl
```

本次实际结果（答案抽取失败计入 Accuracy 错误分母）：

| 模型/学习率 | Validation Accuracy | 格式可抽取率 | 答案不匹配 | 抽取失败 | Validation loss |
|---|---:|---:|---:|---:|---:|
| 未微调基线 | 38.64% (289/748) | 51.74% | 98 | 361 | 未计算 |
| LoRA `2e-4`（已有运行） | 61.50% (460/748) | 99.87% | 287 | 1 | 0.33051 |
| LoRA `1e-4` | 62.43% (467/748) | 99.47% | 277 | 4 | 0.33697 |
| LoRA `5e-5` | 60.56% (453/748) | 98.80% | 286 | 9 | 0.34580 |

`1e-4` 是这次固定 validation 上的名义最高值；与 `2e-4` 相差 7/748 题，差距较小且每组只有一个 seed，不能据此认定它稳定优于其他学习率。建议把 `1e-4` 作为后续候选配置，而不要覆盖原始 `2e-4` 实验。实验环境为 Python 3.10.12、PyTorch 2.14.0+cu130、Transformers 4.57.6、PEFT 0.21.0、Datasets 4.8.5、Accelerate 1.15.0、NVIDIA RTX A6000。各组的原始预测、重算指标、逐题错误和训练日志保存在 `outputs/sweep-baseline/`、`outputs/sweep-lr2e-4/`、`outputs/sweep-lr1e-4/`、`outputs/sweep-lr5e-5/`。

## LoRA 正式训练和评估

确认 smoke test 通过后再开始正式训练：

```bash
python -m lora_math_finetuning.train \
  --config lora_math_finetuning/config.json
```

训练完成后，使用保存的 adapter 在官方测试集生成：

```bash
python -m lora_math_finetuning.evaluate generate \
  --config lora_math_finetuning/config.json \
  --adapter-path lora_math_finetuning/outputs/qwen2.5-1.5b-gsm8k/final_adapter \
  --output lora_math_finetuning/outputs/reproduce-lora-test.jsonl

python -m lora_math_finetuning.evaluate score \
  --predictions lora_math_finetuning/outputs/reproduce-lora-test.jsonl \
  --predictions-output lora_math_finetuning/outputs/reproduce-lora-test.scored.v2.jsonl \
  --output lora_math_finetuning/outputs/reproduce-lora-test.metrics.v2.json \
  --analysis-output lora_math_finetuning/outputs/reproduce-lora-test.errors.v2.jsonl
```

基线和 LoRA 预测文件都保存原始生成文本、抽取答案、参考答案和判分结果。评估脚本可以从预测 JSONL 独立重算指标，并将答错/无法抽取的样本写入 `.errors.jsonl` 供人工分析。

## 答案判定规则

当前规则是：

1. 优先寻找最后一个 `####` 标记；
2. 若没有 `####`，抽取最后一个 `\boxed{...}` 中的内容（支持嵌套花括号）；
3. 两种格式都没有则记为 `extraction_failed`；
4. 整数和小数去除数字内部千位逗号；
5. 分数使用精确分数比较；
6. 若提取结果被 Markdown 反引号或粗体包裹，会在比较前去除这些包裹符；
7. 单位和其他文本不会被静默删除；
8. 抽取失败仍然计入测试集分母。

当前评分版本为 `numeric-v2-leading-decimal`：保留 `.25` 等小数的前导点，不再误归一化成 `25`。旧版在 test 题号 `493` 上因此误判一题正确；已保存旧指标，不修改原始预测。v2 严格分数由 655/1,319 校正为 654/1,319，学习率验证集四份分数未变。重新评分应写到 `.metrics.v2.json` / `.scored.v2.jsonl` / `.errors.v2.jsonl`，不要覆盖旧文件。

例如模型输出：

```text
The calculation gives 15 - 6 = 9.
#### 9
```

没有 `####` 但输出 `The answer is \boxed{9}.` 时，也会抽取 `9`。当前汇总指标包括：

- `accuracy`：答对数除以全部测试题数；答案抽取失败计为答错。
- `format_compliance_rate`：能按约定从 `####` 或 `\boxed{...}` 抽取答案的题目比例。这里的“格式遵从”指可被当前规则识别，不等同于完整符合要求的输出格式。
- `accuracy_when_extracted`：可抽取答案样本中的正确率，只作诊断，不能代替全测试集 Accuracy，因为可抽取样本可能存在选择偏差。
- `answer_mismatch` 和 `extraction_failed`：分别统计抽取到答案但判错、未抽取到答案的数量。

`.errors.jsonl` 每行保存一个错误样本及分类：`answer_mismatch` 或 `format_noncompliant`。抽样人工阅读时，还应进一步区分运算错误、题意理解错误、推理步骤问题和单纯答案解析/格式问题；这些细分类不由程序自动推断。

## 抽取规则敏感性诊断

现有严格分数不能单独说明推理能力提升。新增工具对同一批已保存预测做事后诊断，不加载模型、不修改原始预测或历史指标文件，另存当前评分：

```bash
python -m lora_math_finetuning.analyze_scoring \
  --baseline lora_math_finetuning/outputs/baseline_test.jsonl \
  --lora lora_math_finetuning/outputs/lora_test.jsonl \
  --output reports/evidence/scoring_sensitivity.json \
  --disagreements-output reports/evidence/lora_disagreements.jsonl
```

工具借用固定 commit 的 lm-evaluation-harness GSM8K `flexible-extract` 正则，取最后一个匹配数字，并沿用本项目的数值比较器。本机结果如下：

| 口径 | 基础模型 | LoRA |
|---|---:|---:|
| v2 严格规则 | 36.54% (482/1,319) | 49.58% (654/1,319) |
| v2 最后数字诊断 | 64.97% (857/1,319) | 49.66% (655/1,319) |

第二行**不是官方 harness 成绩**：没有重跑其 5-shot prompt、停止规则和完整 exact-match 流程。最后数字也可能取到中间值、尾注或标点，不能作为无偏的推理能力测量；基线有 408 题从严格错变为诊断对，同时有 33 题反向变化。该结果说明结论对判分口径高度敏感，不能只保留对 LoRA 有利的一行，也不能据此精确量化真实推理能力下降。

JSON 记录源文件 SHA-256、上游 commit、完整口径差异和配对统计；工具校验题目 ID、原题、参考答案和唯一性。固定 seed=42、每层最多 5 条的分歧抽样包含基线两种变化方向及 LoRA 的 1 条变化，共 11 条自动候选，不代表学生已复核。旧版汇总留在 `reports/evidence/scoring_sensitivity.v1.json`，当前汇总明确标记 v2。相关测试位于 `tests/test_scoring_sensitivity.py`。

保存的 adapter 可不加载模型权重而独立审计：

```bash
CUDA_VISIBLE_DEVICES='' python -m lora_math_finetuning.audit_adapter \
  --output reports/evidence/adapter_audit.json
```

借助 safetensors 头部 API 检查 28 层 q/v 模块、rank 和训练参数名，共 112 个张量、1,089,536 个参数，与记录一致。该文件检查不能单独证明历史优化器行为或冻结策略；训练实现仍须结合 PEFT 配置和参数审计解释。`safetensors` 已显式加入依赖清单。具体完善范围见 [`../reports/experiment_plan.md`](../reports/experiment_plan.md)。

## 报告记录

正式实验至少保存：

- 模型名称和版本；
- 数据集名称、来源、划分和许可证信息；
- prompt 模板；
- 最大输入长度和最大生成长度；
- LoRA 的 `r`、`alpha`、`dropout`、目标模块和 bias 策略；
- 学习率、batch size、梯度累积、epoch 和随机种子；
- 基线与 LoRA 的测试样本数、判对数、Accuracy、抽取失败数；
- 若干原始生成结果及错误分析。

训练输出目录中的关键文件：

- `config.json`、`environment.json`、`data_manifest.json`：运行配置、软件/硬件版本、固定数据 revision 与训练/验证/测试行数。
- `parameter_stats.json`、`checkpoint-421/trainer_state.json`、`final_adapter/`：可训练参数审计、训练记录与 LoRA 权重。

## 参考资料

- Cobbe et al. (2021), [Training Verifiers to Solve Math Word Problems (GSM8K)](https://arxiv.org/abs/2110.14168).
- Hu et al. (2022), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- [GSM8K 数据卡](https://huggingface.co/datasets/openai/gsm8k)；[Qwen2.5-1.5B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

本机已完成一次正式运行，v2 重新计分后的结果如下：基础模型在 1,319 条测试题上答对 482 题（Accuracy 36.54%），可抽取答案 685 题（格式可抽取率 51.93%），抽取后答案不匹配 203 题、抽取失败 634 题；LoRA 模型答对 654 题（Accuracy 49.58%），可抽取答案 1,314 题（99.62%），答案不匹配 660 题、抽取失败 5 题。可抽取样本上的正确率分别为 70.36% 和 49.77%，但由于两种模型的可抽取样本不同，这两个条件正确率不能直接作为模型推理能力的公平比较。整体 Accuracy 的提升与格式遵从率大幅提升同时发生，且另一抽取口径改变模型优劣顺序，不能将其单独归因于数学推理能力。

同一测试题的预测也显示了格式与解题正确性并非一回事：测试集 question_id `1` 的题目问“蓝色纤维 2 bolt，白色为一半，总共多少”。基线算出 3，但没有 `####` 或 `\boxed{...}`，因此当前评估无法抽取；LoRA 输出 `#### 5`，格式符合解析规则但把蓝色用量误算为 4，答案错误。这个例子说明需要结合原始生成检查运算/题意错误和格式问题。程序生成的逐题错误记录见 `outputs/baseline_test.errors.jsonl` 与 `outputs/lora_test.errors.v2.jsonl`；报告应从中抽取多个案例人工核查，不应只引用这一题。

- 基线预测和 v2 指标：`outputs/baseline_test.jsonl`、`outputs/baseline_test.metrics.v2.json`。
- LoRA 原始预测：`outputs/lora_test.jsonl`；v2 判分和错误：`outputs/lora_test.scored.v2.jsonl`、`outputs/lora_test.metrics.v2.json`、`outputs/lora_test.errors.v2.jsonl`。旧版 `.metrics.json` / `.errors.jsonl` 作为历史记录保留。
- 完整训练配置、日志和 adapter：`outputs/qwen2.5-1.5b-gsm8k/`。

训练与完整测试集评估已实际运行。报告中的结果应以这些结构化文件为依据，并由学生核查错误案例后再作分析。
