# 02. Transformer 结构

## 1. 整体数据流

一个 Encoder-Decoder Transformer 可以抽象为：

```text
源句 token IDs
  -> 源 Embedding + 位置表示
  -> Encoder 多层
  -> 编码后的源序列表示

目标前缀 token IDs
  -> 目标 Embedding + 位置表示
  -> Decoder 多层，并读取 Encoder 输出
  -> 线性层
  -> 每个位置的目标词概率
```

Encoder 负责理解源句；Decoder 根据已经生成的目标前缀和源句表示，逐步生成译文。

## 2. Attention 在做什么

对某个位置，模型会计算它应该从其他位置读取多少信息。输入先经过三个线性变换，得到 Query、Key 和 Value：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Scaled Dot-Product Attention 的公式是：

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V
```

直观理解：

1. Query 和每个 Key 做相似度计算。
2. Softmax 把相似度变成权重。
3. 用这些权重对 Value 加权求和。

`mask` 会把不允许读取的位置设为极小值，使其 Softmax 后的权重接近 0。

## 3. Self-Attention 和 Cross-Attention

- **Self-Attention**：Q、K、V 都来自同一个序列。Encoder 用它理解源句内部关系，Decoder 也用它理解已经出现的目标词。
- **Cross-Attention**：Q 来自 Decoder，K 和 V 来自 Encoder 输出。它让 Decoder 在翻译某个目标词时查看源句相关位置。

## 4. Multi-Head Attention

一个注意力头可能只学到一种关系。Multi-Head Attention 把隐藏向量拆成多个头，每个头独立计算注意力，最后拼接并投影：

```text
head_i = Attention(X W_Q_i, X W_K_i, X W_V_i)
MHA(X) = Concat(head_1, ..., head_h) W_O
```

实现时要特别检查形状。常见排列包括：

```text
(batch, length, hidden)
(batch, heads, length, head_dim)
```

## 5. Encoder 层

一个典型 Encoder layer 包含：

```text
x -> Self-Attention -> 残差连接和 LayerNorm
  -> FFN            -> 残差连接和 LayerNorm
```

Feed-Forward Network 对每个位置独立应用相同的两层网络：

```text
FFN(x) = activation(x W_1 + b_1) W_2 + b_2
```

它提供逐位置的非线性变换。残差连接帮助信息和梯度跨层传播，LayerNorm 稳定训练。不同实现的 LayerNorm 顺序可能不同，应以代码和配置为准。

## 6. Decoder 层和两个 mask

Decoder layer 通常包含：

1. 带 **causal mask** 的目标 Self-Attention。
2. 读取 Encoder 输出的 Cross-Attention。
3. FFN。
4. 每个子层附近的残差连接和 LayerNorm。

训练时，位置 `t` 不能看到位置 `t+1` 以及更后面的真实目标词。causal mask 通常是一个下三角可见区域。

还要处理 padding：

- source padding mask：Encoder 和 Cross-Attention 不读取源句 padding。
- target padding mask：目标 Self-Attention 不读取目标 padding。
- target causal mask：目标位置不能看到未来位置。

两个 mask 的作用不同，不能只保留一个。

## 7. 训练和推理的差异

训练时可以一次性把完整的目标序列送入 Decoder，因为 causal mask 会遮挡未来位置，并且每个位置都有监督标签。这通常称为 teacher forcing。

推理时没有真实目标句：

```text
从 <bos> 开始
-> 预测下一个词
-> 把预测词接到前缀
-> 再预测下一个词
-> 直到 <eos> 或达到最大长度
```

所以推理是自回归的，容易出现训练和推理行为不同的问题。

## 8. 常见实现错误

- 把目标输入和目标标签错位方向写反。
- 忘记 target causal mask，导致训练时偷看答案。
- 只 mask 了源句 padding，却在 loss 中计算了目标 padding。
- Multi-Head Attention reshape 后没有正确 transpose 或 contiguous。
- 位置编码长度小于实际序列长度。
- 推理时忘记切换 `eval()` 或没有关闭梯度。

## 9. 最小自测

1. Q、K、V 各自表达什么作用？
2. Cross-Attention 为什么需要同时使用 Encoder 和 Decoder 的表示？
3. causal mask 防止了什么信息泄漏？
4. Padding mask 和 causal mask 的区别是什么？
5. 为什么训练可以并行处理目标序列，而推理通常要逐步生成？

## 进一步阅读

- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [PyTorch MultiheadAttention 文档](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
