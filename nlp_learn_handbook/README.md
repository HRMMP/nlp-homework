# NLP 学习手册

这是一份面向 NLP 初学者的学习手册，目标是让你能够从“文本是什么”开始，逐步理解神经网络、Transformer、预训练模型和 LoRA，并最终看懂、运行和解释本仓库中的三个课程实验：

1. 使用 Transformer 做机器翻译。
2. 使用 BERT 做命名实体识别。
3. 在数学数据集上使用 LoRA 做大语言模型参数高效微调。

## 先说清楚这份手册是什么

本手册根据高校 NLP 课程中常见的知识链路整理，并参考公开教材、课程和原始论文。它是学习材料，不是某一所学校的官方课程大纲，也不替代任课教师的具体要求。

本手册不会填写未经实际运行得到的 BLEU、Precision、Recall、F1、Accuracy、训练时间或显存占用。文中的数字主要用于解释形状和计算过程；正式实验结果必须来自你保存的日志和预测文件。

## 推荐路线

| 阶段 | 阅读内容 | 学完后应能回答 |
| --- | --- | --- |
| 0. 先修 | [00 课程路线与学习方法](./00_课程路线与学习方法.md)、[01 Python、数学与机器学习先修](./01_Python、数学与机器学习先修.md) | Python 代码、向量矩阵、训练循环分别是什么 |
| 1. NLP 基础 | [02 NLP 问题与文本表示](./02_NLP问题与文本表示.md)、[03 经典 NLP 方法](./03_经典NLP方法.md) | 文本如何变成特征，分类、序列标注、生成有什么区别 |
| 2. 神经网络 | [04 神经网络与序列模型](./04_神经网络与序列模型.md) | Embedding、RNN、Seq2Seq、Attention 解决什么问题 |
| 3. Transformer | [05 Attention 与 Transformer](./05_Attention与Transformer.md) | Encoder、Decoder、mask、训练和生成如何连接 |
| 4. 预训练 | [06 预训练语言模型](./06_预训练语言模型.md) | BERT 和 GPT 的结构、目标和适用任务有什么差别 |
| 5. 微调与评估 | [07 微调、评估与 LoRA](./07_微调评估与LoRA.md) | 如何划分数据、训练参数、计算指标并防止泄漏 |
| 6. 作业实验 | [08 机器翻译](./08_机器翻译实验.md)、[09 NER](./09_NER实验.md)、[10 数学 LLM](./10_数学LLM与LoRA实验.md) | 能把通用概念对应到作业中的具体数据流 |
| 7. 实验实践 | [11 实验工程与报告写作](./11_实验工程与报告写作.md)、[12 综合练习与检查清单](./12_综合练习与检查清单.md) | 能做可复现的小实验并用证据写分析 |

不建议跳过前两阶段直接开始调大模型。你可以先读每章正文和“自测”，再根据实验需要深入阅读。

## 各知识章节的学习方法

第 00 节是路线总览，不要求立即完成自测或作业。从第 01 节开始，建议每个知识章节完成以下四步：

1. 用自己的话写出本章最重要的三个概念。
2. 把示例中的每个张量写出形状，尤其是 batch、序列长度和隐藏维度。
3. 完成自测题，不要只看答案；不会时回到对应小节。
4. 把概念映射到仓库中的一个文件、一个命令或一个待实现模块。

“能看懂名词”不等于“理解模型”。至少要能解释输入是什么、输出是什么、参数在哪里更新、指标如何计算，以及一个错误可能发生在哪一步。

## 与现有实验资料的关系

本目录提供共同基础和统一学习路线。三个实验目录中的 `learn/` 文件则提供更短的实验专项材料：

- [机器翻译专项学习](../machine_translation/learn/README.md)
- [BERT NER 专项学习](../bert_ner/learn/README.md)
- [LoRA 数学微调专项学习](../lora_math_finetuning/learn/README.md)

建议先读本手册对应的通用章节，再读专项材料，最后阅读实验 README 和代码。专项材料中的数据集、模型、超参数仍需要根据最终实验方案确认，不能直接当作已经确定的实验配置。

## 基础术语速查

- **样本 sample**：一次训练或评估的输入及其标签。
- **token**：模型实际处理的文本单位，可能是字、词、字符或子词。
- **token id**：token 在词表中的整数编号。
- **embedding**：把离散编号映射为连续向量的可训练表。
- **logit**：模型输出的、尚未经过 softmax 的分数。
- **loss**：训练时衡量预测与目标差异的数值。
- **epoch**：完整遍历一次训练集。
- **batch**：一次并行处理的一小组样本。
- **checkpoint**：某个训练时刻保存的模型状态。
- **inference / generation**：使用模型预测或生成结果。
- **metric**：从预测和真实答案计算出的评价指标。

## 公开参考资料

以下资料用于支撑知识路线和进一步阅读。阅读时以资料本身的版本和许可证说明为准：

1. Daniel Jurafsky and James H. Martin, *Speech and Language Processing*, 3rd ed. draft: [Stanford online draft](https://web.stanford.edu/~jurafsky/slp3/)。适合查阅语言模型、文本分类、序列标注、机器翻译等基础主题。
2. Yoav Goldberg, *Neural Network Methods for Natural Language Processing*。适合系统学习神经网络 NLP，出版社页面：[Morgan & Claypool](https://www.cs.bgu.ac.il/~yoavg/)。
3. Stanford CS224N: [Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)。适合配合讲义和作业学习神经 NLP。
4. Harvard NLP 的 [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)。适合对照代码理解 Transformer。
5. Hugging Face [NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)。适合学习 tokenizer、Transformers、Datasets 和训练工具。
6. Ashish Vaswani et al., “Attention Is All You Need”, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)。Transformer 原始论文。
7. Jacob Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, [ACL Anthology](https://aclanthology.org/N19-1423/)。BERT 原始论文。
8. Edward J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models”, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)。LoRA 原始论文。
9. Kishore Papineni et al., “BLEU”, [ACL Anthology](https://aclanthology.org/P02-1040/)。机器翻译 BLEU 指标原始论文。

## 诚实记录原则

本手册中出现的“应当”“通常”“可以”是方法说明，不是你的实验事实。完成作业时，至少保存：

- 实际使用的数据集名称、来源、许可证和划分方式。
- Python、PyTorch、Transformers、CUDA 等版本。
- 配置文件、随机种子、训练日志和 checkpoint 路径。
- 测试集的逐样本预测、原始生成文本和指标复算结果。
- 真实的失败案例、限制和未完成项目。

不能运行验证的内容标记为“待确认”，不能用教程里的示例数字替代实验结果。
