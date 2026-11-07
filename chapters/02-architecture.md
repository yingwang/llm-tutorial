[← 上一章](01-tokenizer.md) | [目录](../README.md) | [下一章 →](03-pretraining.md)

# 第二章：模型架构

## 2.1 Transformer 基础

Transformer（[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)）是现代大语言模型的主干架构，主要由自注意力层、前馈网络和归一化层交替堆叠而成。

### 2.1.1 Self-Attention

**直觉**：处理序列中的每个 Token 时，模型会计算它与上下文里其他 Token 的相关程度，再按权重把相关信息汇总过来。

**计算过程**：
```
输入: X ∈ ℝ^(n×d)  # n个token, d维

Q = X @ W_Q  # Query  ∈ ℝ^(n×d_k)
K = X @ W_K  # Key    ∈ ℝ^(n×d_k)
V = X @ W_V  # Value  ∈ ℝ^(n×d_v)

Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**为什么要除以 $\sqrt{d_k}$**：当向量维度 $d_k$ 较大时，两个向量点积的方差会随 $d_k$ 线性增大。如果不做缩放，点积结果很容易变得过大，把 Softmax 推入梯度接近于 0 的饱和区，导致反向传播时梯度消失。

**Multi-Head Attention (MHA)**：
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

把注意力拆成多个并行的头（Head），可以让模型在不同的投影子空间里同时关注不同类型的关系，比如有的头负责就近的语法搭配，有的头负责跨句的指代消解。

### 2.1.2 FFN (Feed-Forward Network)

每个 Transformer 层在完成自注意力汇总后，都会经过一个逐位置独立计算的前馈网络（Position-wise FFN）：
```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

- **原始 Transformer**：使用 ReLU 激活函数；
- **GPT-2 / GPT-3**：改用平滑的 GELU 激活函数；
- **LLaMA 等现代模型**：普遍采用带门控机制的 SwiGLU（[Shazeer, 2020](https://arxiv.org/abs/2002.05202)）：
```
SwiGLU(x) = (x @ W_1 · σ(x @ W_gate)) @ W_2
```
SwiGLU 比普通 FFN 多了一路门控投影矩阵 $W_{\text{gate}}$（线性变换矩阵从两个变成三个，通常通过把中间层维度缩小到原来的 $\frac{2}{3}$ 来保持总参数量不变），实际训练收敛更快，下游效果也更好。

### 2.1.3 Layer Normalization

归一化层放在残差连接的哪个位置，直接影响深层网络的训练稳定性：
- **Post-Norm**（原始 Transformer）：`x + LayerNorm(SubLayer(x))`。归一化放在残差相加之后，主干上的信号方差会随着层数叠加而放大，深层模型很容易训练崩溃，必须配合精细的学习率预热（Warmup）。
- **Pre-Norm**（GPT-2 起成为主流标配）：`x + SubLayer(LayerNorm(x))`。把归一化移到子层输入端，残差主干保持畅通，梯度可以直接从顶层传回底层，大幅改善了大模型训练的稳定性。
- **RMSNorm**（[Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)，LLaMA、Gemma 采用）：在 Pre-Norm 架构下用 RMSNorm 替代标准 LayerNorm：
  ```
  RMSNorm(x) = x / RMS(x) * γ
  RMS(x) = √(mean(x²))
  ```
  它省去了减均值的中心化步骤，只保留均方根缩放，计算开销比 LayerNorm 少约 7%–10%，而模型效果几乎没有差别。

### 2.1.4 Decoder-Only vs Encoder-Decoder

| 架构 | 代表模型 | 训练目标 | 适用场景 |
|------|---------|---------|---------|
| Encoder-only | BERT, RoBERTa | Masked LM | 文本分类、命名实体识别、向量检索 |
| Encoder-Decoder | T5, BART | Seq2Seq | 机器翻译、文本摘要 |
| **Decoder-only** | **GPT, LLaMA, Claude** | **Causal LM** | **文本生成、多轮对话、通用任务** |

**为什么 Decoder-only 成了主流**：
1. **扩展效率更高**：在固定算力预算下，自回归 Decoder-only 架构随着规模扩大展现出更好的 Scaling Law 表现（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)）；
2. **训练信号更密**：采用 Next-Token Prediction 目标时，序列里的每一个 Token 都能提供监督信号，不像掩码语言模型（MLM）那样每次只预测 15% 被遮盖的位置；
3. **天然支持上下文学习（In-Context Learning）**：单向自回归生成与 Prompt 输入无缝衔接，不需要微调就能直接在上下文中给示例做少样本推理；
4. **结构统一**：理解与生成任务全都统一成条件文本生成，工程实现和推理优化都更简单。

## 2.2 位置编码 (Positional Encoding)

自注意力计算本身对输入顺序不敏感（置换不变），打乱 Token 顺序并不会改变点积关系，因此必须在模型里显式加入位置信息。

### 2.2.1 绝对位置编码

**正弦位置编码**（原始 Transformer）：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- **优势**：不需要额外训练参数，公式本身可以算出任意长度的位置向量；
- **劣势**：直接加在输入 Embedding 上，模型在超出训练长度时外推效果并不好。

**可学习位置编码**（GPT-1/2）：
- 给每个绝对位置分配一个可学习的向量 `P[i] ∈ ℝ^d`，在输入层与 Token Embedding 相加；
- **劣势**：受限于预设的最大位置数，推理时无法处理超过训练窗口长度的序列。

### 2.2.2 RoPE (Rotary Position Embedding)

旋转位置编码（[Su et al., 2021](https://arxiv.org/abs/2104.09864)）是目前大语言模型的标准方案，LLaMA、Qwen、Mistral、Gemma 等主流模型都在使用。

**核心思想**：不在输入 Embedding 上直接加位置向量，而是在每一层算注意力之前，按每个 Token 所在的位置对 Query 和 Key 向量做旋转变换：

```
# 把 d 维向量看成 d/2 个二维平面
# 在每个平面上按位置旋转角度 pos · θ_i

R(pos, i) = [[cos(pos·θ_i), -sin(pos·θ_i)],
              [sin(pos·θ_i),  cos(pos·θ_i)]]

# θ_i = 10000^(-2i/d)

q_rotated = R(m) @ q   # position m
k_rotated = R(n) @ k   # position n

# 关键性质: q_rotated^T @ k_rotated 只依赖于相对位置 (m-n)
```

**优势**：
- **天然编码相对距离**：位置 $m$ 的 Query 和位置 $n$ 的 Key 旋转后再做内积，结果只取决于两者的相对距离 $m - n$；
- **远程自然衰减**：两个 Token 距离越远，旋转后内积的期望幅值自然越小，符合语言里上下文就近相关的规律；
- **方便扩展上下文窗口**：后续只需调整基频常数 $\theta$，就能以较低成本把上下文窗口扩到更长（见 2.2.3）。

### 2.2.3 长上下文扩展

预训练阶段为了控制算力开销，通常只用 4K–8K 的上下文长度；到了后训练或推理阶段，往往需要把窗口扩展到 128K 甚至更长。

**NTK-aware Interpolation**（[Reddit/bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)，Code LLaMA 采用）：
```python
# 不直接线性压缩位置坐标（否则会把相邻 token 挤得太近，破坏局部高频信息），
# 而是放大 RoPE 的 base 频率：
base_new = base_old * (scale_factor ** (d / (d - 2)))
# d = dimension, scale_factor = target_length / trained_length
```

**YaRN**（[Peng et al., 2023](https://arxiv.org/abs/2309.00071)）：
- 把 RoPE 的各个维度按波长分成三组：高频维度保持不变（保住相邻词的局部关系），低频维度做位置插值，中频维度在两者之间平滑过渡；
- 同时在 Softmax 前乘一个温度系数，补偿序列变长后注意力分布变平的问题；
- LLaMA 3.1 扩展到 128K 上下文时就采用了类似的分频缩放策略。

**ALiBi**（[Press et al., 2022](https://arxiv.org/abs/2108.12409)）：
- 完全不加位置向量，而是在算出的注意力分数上直接减去一个与距离成正比的惩罚项：`score -= m * |i - j|`；
- 每个注意力头使用不同的预设斜率 $m$；
- 在短序列上训练后直接外推长序列的效果很好，早期的 BLOOM 和 MPT 采用过这一方案。

## 2.3 现代架构改进

### 2.3.1 Grouped-Query Attention (GQA)

（[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)）

**问题**：在标准多头注意力（MHA）中，每个 Query 头都配有一套独立的 Key 和 Value 头。到了长文本或高并发推理时，缓存历史 Key/Value 的 KV Cache 会占满显存，每次解码步从显存搬运 KV Cache 也会成为带宽瓶颈。

**解决方案**：让一组（多个）Query 头共享同一套 Key 和 Value 头。

```
MHA:  h个Q head, h个K head, h个V head  (如 h=32)
MQA:  h个Q head, 1个K head, 1个V head  (极端共享)
GQA:  h个Q head, g个K head, g个V head  (如 h=32, g=8)
```

- **折中权衡**：GQA 介于全独立的 MHA 和只留单头 KV 的 MQA 之间；
- **节省显存**：KV Cache 直接缩小为原来的 $g/h$（比如 $8/32 = 1/4$），同时模型效果几乎不掉点；
- **主流标配**：LLaMA 2 70B、LLaMA 3、Mistral、Qwen2、Gemma 2 等主流模型全部采用了 GQA。

```python
# GQA 伪代码
def grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads):
    # Q: [batch, seq, num_q_heads, head_dim]
    # K, V: [batch, seq, num_kv_heads, head_dim]
    
    group_size = num_q_heads // num_kv_heads  # e.g., 4
    
    # 扩展 K, V 到 Q 的 head 数（每组共享）
    K = K.repeat_interleave(group_size, dim=2)
    V = V.repeat_interleave(group_size, dim=2)
    
    return standard_attention(Q, K, V)
```

### 2.3.2 Mixture of Experts (MoE)

**核心思想**：把原本单一的密集 FFN 层替换成多个并列的“专家”前馈网络（Expert FFN），每个 Token 到来时，先由一个轻量的门控网络（Router）打分，只挑出得分最高的 Top-$k$ 个专家参与计算（[Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)）。

```
# 标准 FFN:
output = FFN(x)

# MoE:
gate_scores = softmax(x @ W_gate)  # [n_experts]
top_k_experts = topk(gate_scores, k=2)

output = Σ(gate_scores[i] * Expert_i(x))  for i in top_k_experts
```

**优势**：可以在不增加单 Token 计算量（FLOPs）的前提下大幅扩充模型总参数量。比如 [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) 总参数量有 47B，但处理每个 Token 时只激活其中 2 个专家，实际参与计算的参数只有约 13B。

**工程挑战**：
- **负载均衡（Load Balancing）**：训练时如果不加约束，门控网络很容易把绝大多数 Token 都路由给某两三个起跑稍快的专家，导致其他专家得不到训练。为此通常需要加一项负载均衡辅助损失：
  ```
  L_balance = α * n_experts * Σ(f_i * P_i)
  # f_i = 实际分配给专家 i 的 token 比例
  # P_i = 门控网络给专家 i 的平均概率
  ```
- **跨卡通信开销**：当专家分散在不同 GPU 上（Expert Parallelism）时，每个注意力层之后都要通过 All-to-All 通信把 Token 发送到对应的专家卡上，算完再收回来；
- **训练稳定性**：路由器打分对数值波动较敏感，容易出现部分专家坍缩（Expert Collapse）或梯度尖峰。

**代表模型**：[Mixtral 8x7B/8x22B](https://arxiv.org/abs/2401.04088)、[DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)、Grok-1、[DeepSeek-V2/V3](https://arxiv.org/abs/2412.19437)、Qwen2-MoE。

**DeepSeek-V2/V3 的 MoE 改进**（[DeepSeek-V2](https://arxiv.org/abs/2405.04434)）：
- **细粒度专家（Fine-grained Experts）**：不再用传统的“8 个大专家选 2 个”，而是把专家切小、数量扩到 160 或 256 个，每次激活 Top-6 或 Top-8，让知识组合更灵活；
- **共享专家（Shared Experts）**：固定留出 1–2 个不参与路由竞争、对所有 Token 始终激活的共享专家来存放通用常识，让其余路由专家专注于细分领域；
- **无辅助损失负载均衡（Auxiliary-loss-free Load Balancing）**：DeepSeek-V3 去掉了会干扰语言模型主损失的负载均衡辅助 Loss，改为在路由分数上直接加一个随各专家负载动态微调的偏置项 $b_i$。

### 2.3.3 Multi-head Latent Attention (MLA)

由 [DeepSeek-V2](https://arxiv.org/abs/2405.04434) 提出，并在 DeepSeek-V3 和 R1 中继续沿用。

**问题**：GQA 通过减少 KV 头数来省显存，但砍掉太多头数会影响模型的表达能力。

**方案**：先把输入特征下投影压缩成一个低维的潜向量 $c_{KV}$，推理时 KV Cache 只存这个短短的潜向量；等需要计算注意力时，再通过上投影矩阵把 Key 和 Value 恢复出来（或把上投影矩阵直接吸收到 Query 的权重矩阵里）：

```
# 训练时:
c_kv = x @ W_DKV           # 压缩: d_model → d_latent (如 4096 → 512)
K = c_kv @ W_UK             # 解压 Key
V = c_kv @ W_UV             # 解压 Value

# 推理 KV cache 只存 c_kv (512维) 而非 K+V (4096*2维)
```

这样一来，推理时的 KV Cache 体积可以压到标准 MHA 的 1/16 左右，而多头注意力的表达效果比同显存占用的 GQA 更好。

### 2.3.4 Sliding Window Attention

[Mistral 7B](https://arxiv.org/abs/2310.06825) 引入的局部注意力机制，现代模型（如 Gemma 2）常将滑动窗口注意力与全局注意力在不同层之间交替排布。

```
# 滑动窗口: 每个 token 只看它前面 W 个 token
# 但因为多层堆叠，信息可以逐层往后传递
# L 层滑动窗口的理论感受野 = L * W

# Mistral 7B: W=4096, 32层 → 理论感受野 131072
```

**优势**：浅层用固定窗口 $W$ 的局部注意力抓上下文细节，深层穿插全局注意力汇总全篇信息，在把单层计算量和 KV Cache 限制在窗口大小 $W$ 内的同时，依然保持了长文本建模能力。

### 2.3.5 其他现代技巧

**Parallel Attention + FFN**（GPT-J、[PaLM](https://arxiv.org/abs/2204.02311) 风格）：
```python
# 标准: 串行
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# 并行: 计算更快 (attention 和 FFN 的矩阵乘可以合并并行执行)
x = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))
```

**QK-Norm**：在把 Query 和 Key 送去算点积之前，先分别过一层 RMSNorm 或 LayerNorm，防止深层或大规模训练时注意力 Logits 数值失控。已在 [Gemma 2](https://arxiv.org/abs/2408.00118)、Cohere Command R+ 等模型中使用。

**Logit Soft-Capping**（[Gemma 2](https://arxiv.org/abs/2408.00118)）：通过 `logits = soft_cap * tanh(logits / soft_cap)` 把注意力分数和输出层 Logits 软截断在固定区间内，避免极端数值引发训练不稳定。

## 2.4 模型规模设计

### Scaling Laws (Chinchilla)

给定固定的总算力预算 $C$（FLOPs），应该训一个参数更大但喂数据更少的模型，还是训一个参数较小但喂更多数据的模型？[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) 通过大量对比实验给出了 Chinchilla 定律：

```
给定计算预算 C (FLOPs):
最优模型大小 N ∝ C^0.5
最优数据量 D ∝ C^0.5
即: N 和 D 应该等比例同步增长

经验公式:
训练算力最优的 tokens 数 ≈ 20 × 参数量
```

| 参数量 | Chinchilla 最优 tokens | 代表模型 |
|--------|----------------------|---------|
| 1B | 20B | [TinyLlama](https://github.com/jzhang38/TinyLlama) |
| 7B | 140B | LLaMA 2 7B (实际训了 2T) |
| 13B | 260B | LLaMA 2 13B |
| 70B | 1.4T | LLaMA 2 70B |

**实际工程趋势**：Chinchilla 定律只考虑了“训练阶段怎么花算力最划算”，没把模型上线后的推理成本算进去。实际业务中，预训练只跑一次，而线上推理要跑千百万次。因此现在的开源和商用模型普遍选择**超额训练（Over-training）**，也就是拿远超 Chinchilla 建议的数据量去把中小尺寸模型“喂饱”。比如 LLaMA 3 8B 喂了整整 15T Token（接近参数量的 2000 倍），虽然预训练多花了算力，但换来了一个部署成本极低、效果接近旧版几十亿大模型的小巧基座。

### 常见架构参数

| 模型 | 参数量 | Layers | Hidden | Heads | KV Heads | Vocab | Context |
|------|--------|--------|--------|-------|----------|-------|---------|
| [LLaMA 2 7B](https://arxiv.org/abs/2307.09288) | 6.7B | 32 | 4096 | 32 | 32 (MHA) | 32K | 4K |
| [LLaMA 3 8B](https://arxiv.org/abs/2407.21783) | 8.0B | 32 | 4096 | 32 | 8 (GQA) | 128K | 128K |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 7.3B | 32 | 4096 | 32 | 8 (GQA) | 32K | 32K |
| [Qwen2 72B](https://arxiv.org/abs/2407.10671) | 72.7B | 80 | 8192 | 64 | 8 (GQA) | 152K | 128K |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B | 61 | 7168 | 128 | MLA | 128K | 128K |

## 关键论文

- [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762)：提出 Transformer 架构的奠基论文
- [Su et al. (2021): RoFormer / RoPE](https://arxiv.org/abs/2104.09864)：旋转位置编码
- [Ainslie et al. (2023): GQA](https://arxiv.org/abs/2305.13245)：分组查询注意力，大幅节省推理 KV Cache
- [Fedus et al. (2021): Switch Transformer](https://arxiv.org/abs/2101.03961)：简化稀疏路由的现代 MoE 代表工作
- [DeepSeek-AI (2024): DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434)：提出多头潜变量注意力（MLA）与 DeepSeekMoE

## 进一步阅读

- Karpathy：[nanoGPT](https://github.com/karpathy/nanoGPT)，约 300 行 PyTorch 代码完整实现 GPT 训练与推理
- Harvard NLP：[The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)，逐行配合代码讲解 Transformer 原论文
- Lilian Weng：[The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)，系统梳理各类 Transformer 变体

## 练习题

1. **手写 Attention**：用 NumPy 或原生 PyTorch 从头实现带掩码的 Scaled Dot-Product Attention，再扩展成 Multi-Head Attention。
2. **MHA → GQA 改造**：把 nanoGPT 中的标准 MHA 改造成 GQA（设 KV 头数为 Query 头数的 1/4），对比长序列下 KV Cache 的显存占用变化。
3. **RoPE 外推实验**：在一个小型 Transformer 上分别实现正弦位置编码、ALiBi 和 RoPE，在长度 512 的序列上训练，再测试在长度 1024 上的困惑度（Perplexity）变化。

---

[← 上一章](01-tokenizer.md) | [目录](../README.md) | [下一章 →](03-pretraining.md)
