[← 上一章](01-tokenizer.md) | [目录](../README.md) | [下一章 →](03-pretraining.md)

# 第二章：模型架构

## 2.1 Transformer 基础

Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) 是现代 LLM 的骨架。核心组件：

### 2.1.1 Self-Attention

**直觉**: 对于序列中的每个 token，计算它应该"关注"其他哪些 token。

**计算过程**:
```
输入: X ∈ ℝ^(n×d)  # n个token, d维

Q = X @ W_Q  # Query  ∈ ℝ^(n×d_k)
K = X @ W_K  # Key    ∈ ℝ^(n×d_k)
V = X @ W_V  # Value  ∈ ℝ^(n×d_v)

Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**为什么除以 √d_k**: 点积的方差随维度增长，不缩放的话 softmax 会饱和（梯度消失）。

**Multi-Head Attention (MHA)**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

多个 head 让模型在不同子空间学习不同类型的注意力模式（如语法关系、语义关系、位置关系）。

### 2.1.2 FFN (Feed-Forward Network)

每个 Transformer 层里，attention 之后是一个 position-wise FFN：
```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

- 原始 Transformer 用 ReLU
- GPT-2 用 GELU
- LLaMA/现代模型用 SwiGLU ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)):
```
SwiGLU(x) = (x @ W_1 · σ(x @ W_gate)) @ W_2
```
SwiGLU 用了 gating mechanism，实际效果更好（但参数多 50%）。

### 2.1.3 Layer Normalization

**位置很重要**:
- **Post-Norm** (原始 Transformer): `x + LayerNorm(SubLayer(x))`
  - 训练不稳定，需要 warmup
- **Pre-Norm** (GPT-2+, 现代 LLM 标配): `x + SubLayer(LayerNorm(x))`
  - 训练更稳定，但最终性能可能略差
- **RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) (LLaMA, Gemma): Pre-Norm 用 RMSNorm 替代 LayerNorm
  ```
  RMSNorm(x) = x / RMS(x) * γ
  RMS(x) = √(mean(x²))
  ```
  去掉了均值中心化（减去 mean），计算更快，效果不输 LayerNorm

### 2.1.4 Decoder-Only vs Encoder-Decoder

| 架构 | 代表模型 | 训练目标 | 适用场景 |
|------|---------|---------|---------|
| Encoder-only | BERT, RoBERTa | Masked LM | 分类、NER、检索 |
| Encoder-Decoder | T5, BART | Seq2Seq | 翻译、摘要 |
| **Decoder-only** | **GPT, LLaMA, Claude** | **Causal LM** | **生成、对话、通用** |

**为什么 Decoder-only 赢了**:
- Scaling law 表现最好（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)）
- 训练简单：一个 loss (next token prediction)
- 天然支持 in-context learning
- 统一了理解和生成

## 2.2 位置编码 (Positional Encoding)

Attention 是置换不变的（permutation invariant），所以必须注入位置信息。

### 2.2.1 绝对位置编码

**正弦位置编码** (原始 Transformer):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- 优势：理论上可外推到任意长度
- 劣势：实践中外推效果差

**可学习位置编码** (GPT-1/2):
- 每个位置一个可学习向量 `P[i] ∈ ℝ^d`
- 直接加到 token embedding 上
- 劣势：固定最大长度，不能外推

### 2.2.2 RoPE (Rotary Position Embedding)

**现代 LLM 的标准选择** ([Su et al., 2021](https://arxiv.org/abs/2104.09864))。LLaMA, Qwen, Mistral, Gemma 等全部使用。

**核心思想**: 不加位置编码到 embedding 上，而是在 attention 计算时旋转 Q 和 K。

```
# 把 d 维向量看成 d/2 个二维平面
# 在每个平面上按位置旋转角度 θ

R(pos, i) = [[cos(pos·θ_i), -sin(pos·θ_i)],
              [sin(pos·θ_i),  cos(pos·θ_i)]]

# θ_i = 10000^(-2i/d)

q_rotated = R(m) @ q   # position m
k_rotated = R(n) @ k   # position n

# 关键性质: q_rotated^T @ k_rotated 只依赖于相对位置 (m-n)
```

**为什么好**:
- 天然编码相对位置
- 远距离 token 间的 attention 自然衰减
- 可以通过调整 θ 的 base 来扩展上下文窗口（见 2.2.3）

### 2.2.3 长上下文扩展

预训练通常只用 2K-8K context，但推理时想用 128K+。

**NTK-aware Interpolation** ([Reddit/bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), Code LLaMA 风格):
```python
# 不直接做位置插值（会模糊近距离信息），
# 而是修改 RoPE 的 base frequency
base_new = base_old * (scale_factor ** (d / (d - 2)))
# d = dimension, scale_factor = target_length / trained_length
```

**YaRN** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071)):
- 把 RoPE 的频率维度分为三组：低频（不改）、中频（插值）、高频（NTK 缩放）
- 加一个 temperature 修正 attention 分布
- LLaMA 3.1 的 128K context 就用了 YaRN 变体

**ALiBi** ([Press et al., 2022](https://arxiv.org/abs/2108.12409)):
- 不用位置编码，直接在 attention score 上加线性偏置：`score -= m * |i - j|`
- m 是 head-specific 的斜率
- 训练短、推理长，外推效果好
- 用于 BLOOM、MPT

## 2.3 现代架构改进

### 2.3.1 GQA (Grouped-Query Attention)

([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))

**问题**: MHA 中每个 head 有独立的 K、V，KV cache 随 head 数线性增长，是长上下文推理的瓶颈。

**解决方案**: 多个 Q head 共享一组 K、V head。

```
MHA:  h个Q head, h个K head, h个V head  (如 h=32)
MQA:  h个Q head, 1个K head, 1个V head  (极端共享)
GQA:  h个Q head, g个K head, g个V head  (如 h=32, g=8)
```

- **GQA** 是 MHA 和 MQA 的折中
- KV cache 减少 h/g 倍（如 32/8 = 4倍）
- 质量损失很小
- **LLaMA 2 70B, LLaMA 3, Mistral, Gemma 2 全部使用 GQA**

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

### 2.3.2 MoE (Mixture of Experts)

**核心思想**: FFN 层变成多个 "expert" FFN，每个 token 只激活其中 top-k 个。([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538))

```
# 标准 FFN:
output = FFN(x)

# MoE:
gate_scores = softmax(x @ W_gate)  # [n_experts]
top_k_experts = topk(gate_scores, k=2)

output = Σ(gate_scores[i] * Expert_i(x))  for i in top_k_experts
```

**优势**: 总参数量大但每个 token 的计算量小。[Mixtral 8x7B](https://arxiv.org/abs/2401.04088) 有 47B 参数但推理只激活 ~13B。

**挑战**:
- **Load balancing**: 需要 auxiliary loss 防止所有 token 涌入同一个 expert
  ```
  L_balance = α * n_experts * Σ(f_i * P_i)
  # f_i = fraction of tokens routed to expert i
  # P_i = average gate probability for expert i
  ```
- **通信开销**: expert 分布在不同 GPU 上，token 路由需要 all-to-all 通信
- **训练不稳定**: 容易出现 expert collapse（某些 expert 学废了，永远不被选）

**SOTA MoE 模型**: [Mixtral 8x7B/8x22B](https://arxiv.org/abs/2401.04088), [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), Grok-1, [DeepSeek-V2/V3](https://arxiv.org/abs/2412.19437), Qwen2-MoE

**DeepSeek-V2/V3 的 MoE 改进** ([DeepSeek-V2](https://arxiv.org/abs/2405.04434)):
- **DeepSeekMoE**: 更多更小的 expert (如 160 个 expert, top-6 激活)，替代传统的 8 expert top-2
- **Shared expert**: 保留 1-2 个 expert 始终激活（处理通用知识），其余 expert 竞争路由
- **Auxiliary-loss-free load balancing**: 用 bias term 而非 loss 来平衡负载

### 2.3.3 MLA (Multi-head Latent Attention)

[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 引入, DeepSeek-V3/R1 沿用。

**问题**: GQA 虽然减少了 KV cache，但仍有信息损失。

**方案**: 把 KV 压缩到一个低维 latent 向量，推理时从 latent 恢复 KV。

```
# 训练时:
c_kv = x @ W_DKV           # 压缩: d_model → d_latent (如 4096 → 512)
K = c_kv @ W_UK             # 解压 Key
V = c_kv @ W_UV             # 解压 Value

# 推理 KV cache 只存 c_kv (512维) 而非 K+V (4096*2维)
```

KV cache 压缩到原来的 ~1/16，同时质量不输 MHA。

### 2.3.4 Sliding Window Attention

[Mistral 7B](https://arxiv.org/abs/2310.06825) 引入: 不同层交替使用全局 attention 和滑动窗口 attention。

```
# 滑动窗口: 每个 token 只能看前后 W 个 token
# 但因为层叠效果，信息可以跨层传播到更远
# L 层滑动窗口的有效感受野 = L * W

# Mistral 7B: W=4096, 32层 → 理论感受野 131072
```

**优势**: 前面的层用局部 attention（捕捉局部模式），后面的层用全局 attention（整合全局信息）。减少计算量但保持性能。

### 2.3.5 其他现代技巧

**Parallel Attention + FFN** (GPT-J, [PaLM](https://arxiv.org/abs/2204.02311) 风格):
```python
# 标准: 串行
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# 并行: 计算更快 (attention 和 FFN 可以并行)
x = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))
```

**QK-Norm**: 对 Q 和 K 做 LayerNorm/RMSNorm，防止注意力 logits 爆炸。[Gemma 2](https://arxiv.org/abs/2408.00118), Cohere 等使用。

**Logit Soft-Capping** ([Gemma 2](https://arxiv.org/abs/2408.00118)): `logits = soft_cap * tanh(logits / soft_cap)`，限制 attention logits 和 final logits 的范围。

## 2.4 模型规模设计

### Scaling Laws (Chinchilla)

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) 发现：

```
给定计算预算 C (FLOPs):
最优模型大小 N ∝ C^0.5
最优数据量 D ∝ C^0.5
即: N 和 D 应该同步增长

经验公式:
最优 tokens 数 ≈ 20 × 参数量
```

| 参数量 | 最优 tokens | 代表模型 |
|--------|------------|---------|
| 1B | 20B | [TinyLlama](https://github.com/jzhang38/TinyLlama) |
| 7B | 140B | LLaMA 2 7B (2T 实际用了更多) |
| 13B | 260B | LLaMA 2 13B |
| 70B | 1.4T | LLaMA 2 70B |

**实际趋势**: 后来的模型普遍 over-train（用远超 Chinchilla 最优的数据量），因为推理成本 > 训练成本，smaller model + more data 在推理时更划算。LLaMA 3 8B 训练了 15T tokens。

### 常见架构参数

| 模型 | 参数量 | Layers | Hidden | Heads | KV Heads | Vocab | Context |
|------|--------|--------|--------|-------|----------|-------|---------|
| [LLaMA 2 7B](https://arxiv.org/abs/2307.09288) | 6.7B | 32 | 4096 | 32 | 32 (MHA) | 32K | 4K |
| [LLaMA 3 8B](https://arxiv.org/abs/2407.21783) | 8.0B | 32 | 4096 | 32 | 8 (GQA) | 128K | 128K |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 7.3B | 32 | 4096 | 32 | 8 (GQA) | 32K | 32K |
| [Qwen2 72B](https://arxiv.org/abs/2407.10671) | 72.7B | 80 | 8192 | 64 | 8 (GQA) | 152K | 128K |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B | 61 | 7168 | 128 | MLA | 128K | 128K |

---

[← 上一章](01-tokenizer.md) | [目录](../README.md) | [下一章 →](03-pretraining.md)
