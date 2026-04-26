[← Previous Chapter](01-tokenizer.md) | [Table of Contents](README.md) | [Next Chapter →](03-pretraining.md)

# Chapter 2: Model Architecture

## 2.1 Transformer Basics

The Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) is the backbone of modern LLMs. Core components:

### 2.1.1 Self-Attention

**Intuition**: For each token in the sequence, compute how much it should "attend to" every other token.

**Computation**:
```
Input: X ∈ ℝ^(n×d)  # n tokens, d dimensions

Q = X @ W_Q  # Query  ∈ ℝ^(n×d_k)
K = X @ W_K  # Key    ∈ ℝ^(n×d_k)
V = X @ W_V  # Value  ∈ ℝ^(n×d_v)

Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**Why divide by √d_k**: The variance of dot products grows with dimension; without scaling, softmax saturates (vanishing gradients).

**Multi-Head Attention (MHA)**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

Multiple heads let the model learn different types of attention patterns in different subspaces (e.g., syntactic relations, semantic relations, positional relations).

### 2.1.2 FFN (Feed-Forward Network)

In each Transformer layer, after attention comes a position-wise FFN:
```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

- Original Transformer uses ReLU
- GPT-2 uses GELU
- LLaMA / modern models use SwiGLU ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)):
```
SwiGLU(x) = (x @ W_1 · σ(x @ W_gate)) @ W_2
```
SwiGLU uses a gating mechanism and works better in practice (but has 50% more parameters).

### 2.1.3 Layer Normalization

**Placement matters**:
- **Post-Norm** (original Transformer): `x + LayerNorm(SubLayer(x))`
  - Unstable training, requires warmup
- **Pre-Norm** (GPT-2+, standard in modern LLMs): `x + SubLayer(LayerNorm(x))`
  - More stable training, but may have slightly worse final performance
- **RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) (LLaMA, Gemma): Replaces LayerNorm with RMSNorm in Pre-Norm
  ```
  RMSNorm(x) = x / RMS(x) * γ
  RMS(x) = √(mean(x²))
  ```
  Removes mean centering (subtracting the mean), faster computation, performance on par with LayerNorm

### 2.1.4 Decoder-Only vs Encoder-Decoder

| Architecture | Representative models | Training objective | Use cases |
|-------------|----------------------|-------------------|-----------|
| Encoder-only | BERT, RoBERTa | Masked LM | Classification, NER, retrieval |
| Encoder-Decoder | T5, BART | Seq2Seq | Translation, summarization |
| **Decoder-only** | **GPT, LLaMA, Claude** | **Causal LM** | **Generation, dialogue, general-purpose** |

**Why Decoder-only won**:
- Best scaling law performance ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556))
- Simple training: a single loss (next token prediction)
- Natively supports in-context learning
- Unifies understanding and generation

## 2.2 Positional Encoding

Attention is permutation invariant, so positional information must be injected.

### 2.2.1 Absolute Positional Encoding

**Sinusoidal positional encoding** (original Transformer):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- Advantage: theoretically can extrapolate to any length
- Disadvantage: poor extrapolation in practice

**Learned positional encoding** (GPT-1/2):
- One learnable vector `P[i] ∈ ℝ^d` per position
- Added directly to token embeddings
- Disadvantage: fixed maximum length, cannot extrapolate

### 2.2.2 RoPE (Rotary Position Embedding)

**The standard choice for modern LLMs** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)). Used by LLaMA, Qwen, Mistral, Gemma, and more.

**Core idea**: Instead of adding positional encoding to embeddings, rotate Q and K during attention computation.

```
# Treat the d-dimensional vector as d/2 two-dimensional planes
# Rotate by an angle proportional to position θ in each plane

R(pos, i) = [[cos(pos·θ_i), -sin(pos·θ_i)],
              [sin(pos·θ_i),  cos(pos·θ_i)]]

# θ_i = 10000^(-2i/d)

q_rotated = R(m) @ q   # position m
k_rotated = R(n) @ k   # position n

# Key property: q_rotated^T @ k_rotated depends only on relative position (m-n)
```

**Why it works well**:
- Naturally encodes relative position
- Attention between distant tokens decays naturally
- Context window can be extended by adjusting the base of θ (see 2.2.3)

### 2.2.3 Long Context Extension

Pretraining typically uses 2K-8K context, but inference may require 128K+.

**NTK-aware Interpolation** ([Reddit/bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), Code LLaMA style):
```python
# Instead of direct position interpolation (which blurs short-range info),
# modify RoPE's base frequency
base_new = base_old * (scale_factor ** (d / (d - 2)))
# d = dimension, scale_factor = target_length / trained_length
```

**YaRN** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071)):
- Divides RoPE's frequency dimensions into three groups: low-frequency (unchanged), mid-frequency (interpolated), high-frequency (NTK-scaled)
- Adds a temperature correction to the attention distribution
- LLaMA 3.1's 128K context uses a YaRN variant

**ALiBi** ([Press et al., 2022](https://arxiv.org/abs/2108.12409)):
- No positional encoding — directly adds a linear bias to attention scores: `score -= m * |i - j|`
- m is a head-specific slope
- Train short, infer long — good extrapolation
- Used by BLOOM, MPT

## 2.3 Modern Architecture Improvements

### 2.3.1 GQA (Grouped-Query Attention)

([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))

**Problem**: In MHA, each head has independent K and V. KV cache grows linearly with the number of heads, becoming a bottleneck for long-context inference.

**Solution**: Multiple Q heads share a single group of K and V heads.

```
MHA:  h Q heads, h K heads, h V heads  (e.g., h=32)
MQA:  h Q heads, 1 K head, 1 V head  (extreme sharing)
GQA:  h Q heads, g K heads, g V heads  (e.g., h=32, g=8)
```

- **GQA** is a middle ground between MHA and MQA
- KV cache reduced by h/g times (e.g., 32/8 = 4x)
- Minimal quality loss
- **LLaMA 2 70B, LLaMA 3, Mistral, Gemma 2 all use GQA**

```python
# GQA pseudocode
def grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads):
    # Q: [batch, seq, num_q_heads, head_dim]
    # K, V: [batch, seq, num_kv_heads, head_dim]
    
    group_size = num_q_heads // num_kv_heads  # e.g., 4
    
    # Expand K, V to match Q's head count (shared within each group)
    K = K.repeat_interleave(group_size, dim=2)
    V = V.repeat_interleave(group_size, dim=2)
    
    return standard_attention(Q, K, V)
```

### 2.3.2 MoE (Mixture of Experts)

**Core idea**: The FFN layer becomes multiple "expert" FFNs, and each token only activates the top-k of them. ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538))

```
# Standard FFN:
output = FFN(x)

# MoE:
gate_scores = softmax(x @ W_gate)  # [n_experts]
top_k_experts = topk(gate_scores, k=2)

output = Σ(gate_scores[i] * Expert_i(x))  for i in top_k_experts
```

**Advantage**: Large total parameter count but low per-token compute. [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) has 47B parameters but only activates ~13B during inference.

**Challenges**:
- **Load balancing**: Requires an auxiliary loss to prevent all tokens from routing to the same expert
  ```
  L_balance = α * n_experts * Σ(f_i * P_i)
  # f_i = fraction of tokens routed to expert i
  # P_i = average gate probability for expert i
  ```
- **Communication overhead**: Experts are distributed across different GPUs; token routing requires all-to-all communication
- **Training instability**: Prone to expert collapse (some experts degrade and are never selected)

**SOTA MoE models**: [Mixtral 8x7B/8x22B](https://arxiv.org/abs/2401.04088), [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), Grok-1, [DeepSeek-V2/V3](https://arxiv.org/abs/2412.19437), Qwen2-MoE

**DeepSeek-V2/V3 MoE improvements** ([DeepSeek-V2](https://arxiv.org/abs/2405.04434)):
- **DeepSeekMoE**: More and smaller experts (e.g., 160 experts, top-6 activation), replacing the traditional 8 experts with top-2
- **Shared expert**: 1-2 experts are always active (handling general knowledge); the rest compete for routing
- **Auxiliary-loss-free load balancing**: Uses a bias term instead of a loss to balance load

### 2.3.3 MLA (Multi-head Latent Attention)

Introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434), continued in DeepSeek-V3/R1.

**Problem**: GQA reduces KV cache but still incurs information loss.

**Approach**: Compress KV into a low-dimensional latent vector; recover KV from the latent during inference.

```
# During training:
c_kv = x @ W_DKV           # Compress: d_model → d_latent (e.g., 4096 → 512)
K = c_kv @ W_UK             # Decompress Key
V = c_kv @ W_UV             # Decompress Value

# Inference KV cache only stores c_kv (512-dim) instead of K+V (4096*2-dim)
```

KV cache compressed to ~1/16 of original size, with quality on par with MHA.

### 2.3.4 Sliding Window Attention

Introduced in [Mistral 7B](https://arxiv.org/abs/2310.06825): different layers alternate between global attention and sliding window attention.

```
# Sliding window: each token can only attend to W tokens before and after it
# But due to layer stacking, information propagates further across layers
# Effective receptive field with L layers of sliding window = L * W

# Mistral 7B: W=4096, 32 layers → theoretical receptive field 131072
```

**Advantage**: Earlier layers use local attention (capturing local patterns), later layers use global attention (integrating global information). Reduces computation while maintaining performance.

### 2.3.5 Other Modern Techniques

**Parallel Attention + FFN** (GPT-J, [PaLM](https://arxiv.org/abs/2204.02311) style):
```python
# Standard: sequential
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Parallel: faster computation (attention and FFN can run in parallel)
x = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))
```

**QK-Norm**: Apply LayerNorm/RMSNorm to Q and K to prevent attention logits from exploding. Used by [Gemma 2](https://arxiv.org/abs/2408.00118), Cohere, etc.

**Logit Soft-Capping** ([Gemma 2](https://arxiv.org/abs/2408.00118)): `logits = soft_cap * tanh(logits / soft_cap)` — bounds the range of attention logits and final logits.

## 2.4 Model Scale Design

### Scaling Laws (Chinchilla)

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) found:

```
Given a compute budget C (FLOPs):
Optimal model size N ∝ C^0.5
Optimal data size D ∝ C^0.5
i.e., N and D should grow in proportion

Rule of thumb:
Optimal tokens ≈ 20 × parameter count
```

| Parameters | Optimal tokens | Representative model |
|-----------|---------------|---------------------|
| 1B | 20B | [TinyLlama](https://github.com/jzhang38/TinyLlama) |
| 7B | 140B | LLaMA 2 7B (actually used 2T, more than optimal) |
| 13B | 260B | LLaMA 2 13B |
| 70B | 1.4T | LLaMA 2 70B |

**Actual trend**: Later models generally over-train (using far more data than the Chinchilla optimum), because inference cost > training cost — a smaller model + more data is more cost-effective at inference time. LLaMA 3 8B was trained on 15T tokens.

### Common Architecture Parameters

| Model | Parameters | Layers | Hidden | Heads | KV Heads | Vocab | Context |
|-------|-----------|--------|--------|-------|----------|-------|---------|
| [LLaMA 2 7B](https://arxiv.org/abs/2307.09288) | 6.7B | 32 | 4096 | 32 | 32 (MHA) | 32K | 4K |
| [LLaMA 3 8B](https://arxiv.org/abs/2407.21783) | 8.0B | 32 | 4096 | 32 | 8 (GQA) | 128K | 128K |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 7.3B | 32 | 4096 | 32 | 8 (GQA) | 32K | 32K |
| [Qwen2 72B](https://arxiv.org/abs/2407.10671) | 72.7B | 80 | 8192 | 64 | 8 (GQA) | 152K | 128K |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B | 61 | 7168 | 128 | MLA | 128K | 128K |

## Key Papers

- [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original Transformer paper, required reading
- [Su et al. (2021) — RoFormer / RoPE](https://arxiv.org/abs/2104.09864) — the now-standard positional encoding
- [Ainslie et al. (2023) — GQA](https://arxiv.org/abs/2305.13245) — KV-cache savings used by Llama 2/3
- [Fedus et al. (2021) — Switch Transformer](https://arxiv.org/abs/2101.03961) — modern MoE formulation
- [DeepSeek-AI (2024) — DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434) — Multi-head Latent Attention

## Further Reading

- Karpathy — [nanoGPT](https://github.com/karpathy/nanoGPT) — GPT in ~300 lines of PyTorch
- Harvard NLP — [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — line-by-line annotations of the original paper
- Lilian Weng — [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)

## Exercises

1. **Implement attention from scratch**: build scaled dot-product attention in numpy, then extend to multi-head.
2. **MHA → GQA refactor**: convert nanoGPT's MHA to GQA (kv heads = q heads / 4) and observe the KV-cache reduction.
3. **RoPE experiment**: on a small transformer, compare sinusoidal, ALiBi, and RoPE on length extrapolation (train at 512, test at 1024).

---

[← Previous Chapter](01-tokenizer.md) | [Table of Contents](README.md) | [Next Chapter →](03-pretraining.md)
