# 机器翻译实验

本实验使用 PyTorch Encoder-Decoder Transformer 完成英译法，数据集为 OPUS Books。目录中提供数据准备、训练、贪心解码、SacreBLEU 独立计分和最小测试入口。旧运行的 6,355 条测试预测 BLEU 为 13.78，但存在语向、疑似错配和跨集合精确重复。本轮已完成语言筛查和分组去重后的 12 轮重训：5,008 条新测试预测 BLEU 13.69、chrF++ 38.48；新旧测试集不同，不能按分差声称模型改进。自动筛查不保证语义对齐，抽样记录仍须学生核验。

## 语言筛查与分组去重版

`configs/clean_data.json` 固定清洗规则和 fastText `lid.176.ftz` 模型 SHA-256：NFKC/空白归一化，两侧至少 10 个字母、最多 500 字符，字符长度比不超过 4，预测语向须为 en/fr 且两侧概率至少 0.8；排除完全相同文本，在切分前精确句对去重，并将同源或同目标的连通组放在同一集合。不依据模型预测得分清洗，不自动交换反向句对。模型来源及 CC-BY-SA-3.0 许可见 [fastText 官方页面](https://fasttext.cc/docs/en/language-identification.html)。

127,085 对中保留 100,141 对，train/validation/test=90,126/5,007/5,008；跨集合精确句对、源句、目标句交集均为 0。完整规则排除计数、固定随机抽样和各文件 SHA-256 保存在 `outputs/en-fr-clean-data-v1/manifest.json`，重建结果逐文件一致。没有按书籍隔离，也没有完成近重复或全量人工语义审计。

从根目录复现数据准备时使用新的输出目录和训练配置文件（命令拒绝覆盖）：

```bash
mkdir -p machine_translation/outputs/resources
curl -fL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz \
  -o machine_translation/outputs/resources/lid.176.ftz
python -m machine_translation.prepare \
  --config machine_translation/configs/clean_data.json \
  --output-dir machine_translation/outputs/reproduce-clean-data \
  --training-config-output machine_translation/outputs/reproduce-clean-config.json
```

`fasttext-wheel` 仅用于预处理；批量预测兼容本机 NumPy 2。训练与评价使用生成的、固定 manifest 哈希的配置，从已准备文件读取，不重新联网切分。

```bash
python scripts/run_on_idle_gpu.py \
  --record machine_translation/outputs/reproduce-clean-train.job.json \
  --log machine_translation/outputs/reproduce-clean-train.log -- \
  python -m machine_translation.train \
  --config machine_translation/outputs/reproduce-clean-config.json \
  --device cuda:0 --output-dir machine_translation/outputs/reproduce-clean-run
python scripts/run_on_idle_gpu.py \
  --record machine_translation/outputs/reproduce-clean-test.job.json \
  --log machine_translation/outputs/reproduce-clean-test.log -- \
  python -m machine_translation.evaluate \
  --run-dir machine_translation/outputs/reproduce-clean-run \
  --split test --batch-size 64 --device cuda:0
```

选择器要求显存不超过 256 MiB、利用率不超过 5%、无计算进程，并间隔两秒复查；按物理 UUID 设置 `CUDA_VISIBLE_DEVICES`，进程内统一用 `cuda:0`。锁只协调本用户的启动，不替代集群调度或保证其他用户不会随后使用该卡。四次本轮 GPU 任务均正常退出，记录在 `outputs/clean-v1-*.job.json`，没有停止其他用户的进程。

本机正式产物为 `outputs/en-fr-clean-v1-seed42/`：两步 smoke、完整 12 轮训练、完整测试和独立评分均已执行。参数与旧全量模型一致（11,682,624 参数）；最佳 epoch=12，验证 loss=2.4438。`code_snapshot.json` 记录相关源码 SHA-256。新测试有 7 条 source、25 条 target 按当前 tokenizer/长度规则会截断；参考译文仍保留完整文本，未删去这些记录。

```bash
CUDA_VISIBLE_DEVICES='' python reports/verify_results.py --audit-cached-mt \
  --clean-mt-run machine_translation/outputs/en-fr-clean-v1-seed42
```

新结果与训练曲线分别在 `reports/evidence/results.json` 的 `translation_clean` 和 `reports/evidence/translation_clean_loss.csv`；JSON 含前 10 条系统抽样译例，尚未经过学生人工核验。旧运行与旧数据保持不变。以下保留原始加载入口，勿将原始配置与清洗版混同。

## 实验方案

- **任务**：英语到法语的句子级机器翻译，源/目标语言代码分别为 `en` 和 `fr`。
- **数据**：Hugging Face `Helsinki-NLP/opus_books` 的 `en-fr` 配置。当前锁定 revision 为 `1f9f6191d0e91a3c539c2595e2fe48fc1420de9b`；该 revision 的数据卡记录 127,085 对句子、下载文件约 21 MB。默认按 seed 42 固定随机打乱后取 20,000 对，按 90%/5%/5% 划分训练、验证、测试集。切分前去除空句，词表只使用训练集拟合。
- **数据许可**：Hugging Face 数据卡的 license 字段为 `other`，卡片说明数据来自双语书籍并描述为版权已过期/可自由使用。原始书籍的授权和再分发条件仍需逐本核对；报告中应准确引用数据卡与原始来源，不应把 `other` 改写成未经核实的标准许可证。
- **分词**：使用一个由 `sentencepiece` 训练的共享 Unigram 子词词表，源/目标共用，默认词表上限 8,000。特殊 ID 固定为 PAD=0、UNK=1、BOS=2、EOS=3；源句以 EOS 结束，目标句以 BOS 开始并以 EOS 结束。
- **模型**：`torch.nn.Transformer`，batch-first，3 层 Encoder、3 层 Decoder、隐藏维度 256、4 个注意力头、FFN 维度 1024、dropout 0.1，固定正弦位置编码。Decoder 使用 causal mask；源/目标 padding mask 中 `True` 表示忽略该位置。
- **训练目标**：Teacher forcing。完整目标序列只右移一次，Decoder 输入为 `target[:-1]`、监督标签为 `target[1:]`；交叉熵使用原始 logits 并忽略目标 PAD。使用 Adam、梯度裁剪、验证集 loss 选最佳 checkpoint 和 early stopping。
- **解码与评价**：贪心解码，遇到 EOS 停止。测试时保存 source/reference/prediction JSONL，可独立重算 SacreBLEU。BLEU 配置为单参考、4-gram、13a tokenizer、大小写敏感、exp smoothing、默认 brevity penalty；分词和参数签名写入结构化结果。

## 环境

建议使用独立 Python 3.10+ 虚拟环境。PyTorch wheel 需与实际 CPU/CUDA 环境匹配；如需指定 PyTorch 官方 CUDA wheel 源，先按 PyTorch 安装页面选择版本，再安装其余依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r machine_translation/requirements-dev.txt
```

依赖用途：PyTorch 实现 Transformer 与训练；`datasets` 下载固定 revision 的公开语料；SentencePiece 训练子词模型；SacreBLEU 计算可复现 BLEU；pytest/ruff 用于测试和静态检查。

## 运行

从仓库根目录执行。

完整训练使用 [`config.json`](./config.json)，训练结果会写到带时间戳的 `machine_translation/outputs/` 子目录（该目录已在 `.gitignore` 中排除）：

```bash
python -m machine_translation.train
```

也可显式指定配置、设备和输出目录。输出目录必须为空或不存在：

```bash
python -m machine_translation.train \
  --config machine_translation/config.json \
  --device auto \
  --output-dir machine_translation/outputs/en-fr-seed42
```

第一次运行前先完成小规模 smoke test。它会从固定数据集 revision 取最多 64 对句子，只训练最多 2 个 batch；需要能访问 Hugging Face，产物仍不代表正式实验结果。每次选用新的输出目录：

```bash
python -m machine_translation.train \
  --smoke \
  --max-steps 2 \
  --output-dir machine_translation/outputs/smoke-01
```

在最佳 checkpoint 上评估独立测试集并保存预测、指标和最多 20 条样例：

```bash
python -m machine_translation.evaluate \
  --run-dir machine_translation/outputs/en-fr-seed42 \
  --split test
```

`--limit 20` 可用于快速检查解码和评估链路，不是正式测试集分数。验证集可用 `--split validation`。

从导出的 JSONL 独立复算 BLEU，不需要加载模型或重新访问数据集：

```bash
python -m machine_translation.score \
  --predictions machine_translation/outputs/en-fr-seed42/predictions.test.jsonl
```

## 产物与可追溯性

每次训练运行目录包含：

- `run_config.json`、`runtime.json`：实验配置、Python/PyTorch/关键包版本、设备和 CUDA 信息。
- `data_manifest.json`：数据 revision、seed、划分策略和样本数。
- `model_summary.json`：模型结构、特殊 token ID、总参数和可训练参数量。
- `tokenizer.model`、`best.pt`：训练集拟合的 SentencePiece 模型和验证 loss 最佳检查点。
- `training.jsonl`、`history.json`：每轮训练/验证 loss 和耗时。
- `predictions.test.jsonl`、`metrics.test.json`、`examples.test.json`：测试预测、SacreBLEU 指标签名及错误分析样例（评估后产生）。

SacreBLEU 输出记录 precision 1-4、brevity penalty、系统/参考长度、样本数和完整签名。BLEU 必须和数据 revision、测试划分、分词及大小写配置一起解释；不得只抄单个分数。错误分析应从实际预测中选择漏译、错译、重复或专名问题，不能把示例错误类别当成已观测结果。

## 检查

```bash
pytest -q machine_translation/tests
python -m compileall -q machine_translation
ruff check machine_translation
```

没有安装 PyTorch 时，纯数据切分、预测文件和部分 tokenizer 测试仍可运行；涉及 batch padding 或模型的测试会明确跳过。早期开发已完成 2 个 optimizer batch 的 smoke 链路；本机后续正式运行产物位于 `outputs/en-fr-full-seed42/`，与 smoke 结果分开保存。

可用本地缓存重建原切分并审计精确重复，无须加载模型：

```bash
CUDA_VISIBLE_DEVICES='' python reports/verify_results.py --audit-cached-mt
```

`data.audit_split_overlap()` 只计数，不修改文本或切分。本次发现 45 个 train/test 共有句对键，在 test 中出现 52 次；对应结构化结果见 `reports/evidence/results.json`。这是精确重复审计，不是完整语言识别、近重复或语义对齐检查；新清洗规则需要在重新切分和训练前确认，不能通过事后只删除测试难例改善原分数。

## 文件结构

- `data.py`：数据集读取、固定切分、SentencePiece 和 padding/collation。
- `model.py`：位置编码、Encoder-Decoder Transformer、causal mask 和贪心解码。
- `train.py`：训练、验证、checkpoint、运行元数据和日志。
- `evaluate.py`：加载 checkpoint，对验证或测试划分生成并保存预测。
- `score.py`：只依赖预测 JSONL 的独立 BLEU 复算入口。
- `tests/`：数据切分、目标右移、mask、输出形状、未来信息隔离和贪心停止条件测试。

## 参考资料

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Papineni et al. (2002), [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/).
- Post (2018), [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/).
- Kudo and Richardson (2018), [SentencePiece](https://aclanthology.org/D18-2012/).
- [OPUS Books 数据卡](https://huggingface.co/datasets/Helsinki-NLP/opus_books)；[OPUS 语料库](https://opus.nlpl.eu/).
- [PyTorch Transformer 文档](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)；[SacreBLEU 文档](https://github.com/mjpost/sacrebleu).

开始学习前可按 [`learn/README.md`](./learn/README.md) 阅读基础概念与实现对应关系。
