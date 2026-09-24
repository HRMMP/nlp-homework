# NLP 课程大作业

本项目用于完成自然语言处理课程大作业，包含三个相互独立的实验：机器翻译、命名实体识别，以及大模型参数高效微调。

如果你没有学过 NLP，建议先阅读面向零基础的[《NLP 学习手册》](./nlp_learn_handbook/README.md)，再进入具体实验目录。

课程要求以根目录的 [`自然语言处理大作业.pdf`](./自然语言处理大作业.pdf) 为准。各实验的具体要求分别记录在对应目录的 README 中。

## 实验目录

| Lab | 目录 | 实验内容 | 当前进度 |
| --- | --- | --- | --- |
| Lab 1 | [`machine_translation/`](./machine_translation/) | 使用 Transformer 实现机器翻译，并使用 BLEU 评价翻译结果。 | 已实现 OPUS Books 英译法数据处理、Transformer 训练/解码/独立 BLEU 评估入口、配置与基础测试；仓库默认环境未安装依赖，已在临时环境通过测试和小规模 smoke，正式训练/测试集评估尚未运行。 |
| Lab 2 | [`bert_ner/`](./bert_ner/) | 使用 BERT 完成命名实体识别，并报告 Precision、Recall 和 F1 Score。 | 已准备 CLUENER 2020 数据，加入数据审计、BIO/subword 对齐、实体级评估、训练入口和单元测试；尚未安装训练依赖或运行训练。 |
| Lab 3 | [`lora_math_finetuning/`](./lora_math_finetuning/) | 在数学数据集上使用 LoRA 进行大模型参数高效微调，并在测试集上统计准确率。 | 已创建目录、要求文档和入门学习资料；尚未开始数据准备、代码实现和实验运行。 |

## 当前项目状态

### 已完成

- 创建三个实验目录。
- 根据课程作业 PDF 整理三个实验的要求文档。
- 明确每个实验的主要任务和评价指标。
- 为三个实验补充面向零基础的分阶段学习资料。
- 新增面向零基础的 [NLP 学习手册](./nlp_learn_handbook/README.md)，覆盖先修知识、经典方法、Transformer、BERT、GPT、LoRA、三个实验和可复现实验流程。

### 尚未完成

- 确认并准备机器翻译实验数据；运行小规模 smoke test 和正式训练，保存日志、检查点、测试集预测及 BLEU 结果。
- 确认并准备 NER 实验的 CLUENER 2020 数据集；LoRA 实验仍需选择和准备数据集。
- 完成各实验的小规模 smoke test。
- 运行正式实验并保存日志、检查点和结构化结果。
- 撰写实验结果分析和最终实验报告。

实验报告初稿及 XeLaTeX 编译说明见 [`reports/README.md`](./reports/README.md)。

## 共同报告要求

根据课程作业要求：

- 每组 3 人。
- 报告需要覆盖机器翻译、命名实体识别和大模型参数高效微调三个实验。
- 报告提纲至少包括：
  1. 背景和问题介绍
  2. 目前的研究现状
  3. 模型与算法设计
  4. 实验结果与分析
  5. 总结
- 报告中需要介绍每位组员的具体工作。
- 提交邮箱：`nlp_course_ustc@163.com`
- 报告命名格式：`26年秋实验报告_第x组_组长姓名_组长学号`

## 实验要求概览

### Lab 1：机器翻译

使用 Transformer 实现机器翻译。实现和报告应覆盖 Transformer 的 Encoder、Decoder、注意力机制、FFN、位置编码等相关内容，并使用 BLEU 评价翻译质量。

详细要求见 [`machine_translation/README.md`](./machine_translation/README.md)。

### Lab 2：命名实体识别

使用基于 BERT 的方法完成命名实体识别，说明实体标注方式和模型微调过程，并报告 Precision、Recall 和 F1 Score。

详细要求见 [`bert_ner/README.md`](./bert_ner/README.md)。

### Lab 3：大模型参数高效微调

在数学数据集上使用 LoRA 对预训练模型进行参数高效微调，并在测试集上统计 Accuracy，同时记录答案抽取和判分规则。

详细要求见 [`lora_math_finetuning/README.md`](./lora_math_finetuning/README.md)。

## 实施记录原则

- 实验结果、指标、图表和结论必须来自实际运行记录，不预先填写未经验证的结果。
- 记录数据集名称、来源、划分方式、预处理流程和许可证信息。
- 记录 Python、深度学习框架、CUDA 和关键依赖版本。
- 保存训练配置、日志、模型检查点路径和最终评估结果。
- 不将大型原始数据集、模型权重、缓存文件、密钥或个人凭据提交到仓库。
