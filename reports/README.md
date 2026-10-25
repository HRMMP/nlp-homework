# 实验记录与报告审阅稿

打开 [实验报告模板.pdf](./实验报告模板.pdf)。[LaTeX 源文件](./实验报告模板.tex) 与 [build 预览](./build/实验报告模板.pdf) 对应同一版本。事实表、训练曲线、判分修复及复现入口已同步；封面明确标为审阅稿，不代表组员已核验全部解释或可以直接提交。

## 已执行的技术补强

| 实验 | 当前可复算结果 | 解释边界 |
|---|---|---|
| Transformer 翻译 | 语言筛查、分组去重后保留 100,141 对；12 轮重训；完整 5,008 条测试 BLEU 13.69、chrF++ 38.48 | 精确跨集合重复为 0，不等于无近重复或语义错配；新旧测试集不同，不能按旧 BLEU 13.78 的差值声称提升 |
| BERT-NER | dev 1,343 条，实体级 P/R/F1=77.16%/80.18%/78.64%；TP/FN/FP=2,463/609/729 | dev 参与选模，官方 test 无 gold；错误和未见实体分析仍是开发集诊断 |
| 数学 LoRA | v2 严格 Accuracy=36.54%/49.58%；最后数字诊断=64.97%/49.66%；保存的 112 个 adapter 张量与参数统计一致 | 诊断非官方 harness 成绩，不能据严格分数单独证明推理改善；文件审计不单独证明历史冻结行为 |

用户已授权本轮范围及使用空闲卡。翻译 smoke、smoke 解码、正式训练及完整测试四个任务均经两次空闲状态/计算进程检查后启动，物理 GPU 0（RTX A6000）通过检查，均正常退出；没有停止其他用户的进程。启动命令、物理 UUID、日志及退出状态在 `machine_translation/outputs/clean-v1-*.job.json`。NER 和 LoRA 本轮没有新增训练或用 test 继续调参。

## 复算与证据

在本机已有环境中运行，不需要 GPU：

```bash
source /home/yifeiwang/project/.venv/bin/activate
CUDA_VISIBLE_DEVICES='' python reports/verify_results.py --audit-cached-mt \
  --clean-mt-run machine_translation/outputs/en-fr-clean-v1-seed42
CUDA_VISIBLE_DEVICES='' python bert_ner/analyze.py \
  --output reports/evidence/ner_analysis.json
CUDA_VISIBLE_DEVICES='' python -m lora_math_finetuning.analyze_scoring \
  --baseline lora_math_finetuning/outputs/baseline_test.jsonl \
  --lora lora_math_finetuning/outputs/lora_test.jsonl \
  --output reports/evidence/scoring_sensitivity.json \
  --disagreements-output reports/evidence/lora_disagreements.jsonl
CUDA_VISIBLE_DEVICES='' python -m lora_math_finetuning.audit_adapter \
  --output reports/evidence/adapter_audit.json
```

`--audit-cached-mt` 只使用本地缓存，不联网下载原始语料。没有缓存时可去掉它；新翻译运行仍需要带哈希的准备文件。各实验 README 给出环境、数据准备、训练和评价入口。

- [results.json](./evidence/results.json)：新旧翻译结果、NER 指标、LoRA v2 与历史 v1 对照、唯一修正题号、源文件哈希。新翻译有前 10 条系统抽样候选，未宣称学生已核验。
- [translation_clean_loss.csv](./evidence/translation_clean_loss.csv) 与 [translation_loss.csv](./evidence/translation_loss.csv)：新旧运行的 12 轮曲线。
- [ner_analysis.json](./evidence/ner_analysis.json)：标签回查、完整类别指标、互斥的 FN/FP 分类、同边界类型混淆、seen/unseen 诊断及自动候选。
- [scoring_sensitivity.json](./evidence/scoring_sensitivity.json) 与 [lora_disagreements.jsonl](./evidence/lora_disagreements.jsonl)：两种口径和 11 条固定种子的分歧候选，含原始生成。
- [adapter_audit.json](./evidence/adapter_audit.json)：保存张量的名字、形状、参数量、来源哈希和审计限制。
- [results.v1.json](./evidence/results.v1.json)、[scoring_sensitivity.v1.json](./evidence/scoring_sensitivity.v1.json)：修复前的历史汇总，不是当前评分。

评分器旧版把 `.25` 归一化成 `25`，导致 LoRA test 题号 `493` 误判正确。v2 保留前导小数点，严格正确数从 655 改为 654，最后数字诊断从 656 改为 655；其他五份预测的严格分数不变。原始预测和旧 `.metrics.json` 均保留，新 LoRA 判定/错误/指标另存 `.scored.v2.jsonl`、`.errors.v2.jsonl`、`.metrics.v2.json`，不得将历史字段当成当前正确性。

新翻译准备文件完整重建后，manifest、三个切分、排除记录和抽样的六个哈希一致。权重、准备文件、原始预测和日志都被 Git 忽略，仅克隆仓库不能获得它们，须另行备份或取得对应产物；证据 JSON 不代替原始大文件。

## 检查与构建

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=bert_ner python -m pytest -q \
  machine_translation/tests bert_ner/tests lora_math_finetuning/tests tests
ruff check machine_translation bert_ner lora_math_finetuning scripts tests reports/verify_results.py
bash reports/build_report.sh
```

本轮 70 项测试通过，三个实验、选择器、测试和复算脚本静态检查通过，变更文件格式检查通过。PyTorch 嵌套张量原型提示未造成失败。NER 原有 5 项静态问题已修正，未改变有效输入的标签转换或模型预测。

构建脚本使用 XeLaTeX/latexmk，成功后同步两份 PDF；需要 ctex、TikZ、pgfplots、fvextra、xurl 和 Noto CJK 字体。编辑器显示旧内容时重新打开 PDF。历史旧 PDF 位于被忽略的 build 目录，不作为当前稿。

## 仍须组员核验

- 解释模型与指标，核验清洗样本、译例、NER 错误和 LoRA 分歧；自动候选不是已完成的人工误差分析。
- 填写三人的姓名、学号、组号、真实分工和提交日期，按实际理解修订文字及 AI 使用披露。
- 确认许可和来源，尤其 CLUENER、本次未固定的 BERT revision 及历史训练没有精确代码快照的限制。新翻译已另存源码哈希。
- 本轮没有新增 NER 独立有标签评测、多随机种子或进一步 LoRA 训练，不把这些计划写成已完成。

完整范围、Web 依据和执行记录见 [experiment_plan.md](./experiment_plan.md)。17 篇论文 PDF 仍在 [references/](./references/)；本轮另引用 fastText 官方语言识别文档和固定版本 GSM8K 评测配置，没有将它们说成本地新增论文。引用和许可仍须组员核对。

课程要求每组 3 人，提交邮箱为 `nlp_course_ustc@163.com`，最终文件名为 `26年秋实验报告_第x组_组长姓名_组长学号`。本轮未发送、上传或提交任何内容，也未创建 Git 提交。
