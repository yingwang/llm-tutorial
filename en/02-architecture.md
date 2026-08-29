[← Previous Chapter](01-tokenizer.md) | [Table of Contents](README.md) | [Next Chapter →](03-pretraining.md)

# Chapter 2: Model Architecture

## 2.1 Transformer Foundations

The Transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) forms the structural foundation of modern Large Language Models. Its primary computational building blocks include:

### 2.1.1 Self-Attention

**Intuition**: For every token representation in a sequence, compute dynamic routing weights (attention coefficients) that determine how information is aggregated from all other positions.

**Mathematical Formulation**:
```
Input: X ∈ ℝ^(n×d)  # n tokens, d hidden dimensions

Q = X @ W_Q  # Query matrix ∈ ℝ^(n×d_k)
K = X @ W_K  # Key matrix   ∈ ℝ^(n×d_k)
V = X @ W_V  # Value matrix ∈ ℝ^(n×d_v)

Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**The Scaling Factor $\frac{1}{\sqrt{d_k}}$**: Under the assumption of independent components with zero mean and unit variance, the variance of the inner product $q \cdot k$ grows linearly with dimension $d_k$. Without the scaling factor $\frac{1}{\sqrt{d_k}}$, large values push the softmax function into regions of vanishingly small gradients.

**Multi-Head Attention (MHA)**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

Partitioning projection channels into $h$ independent heads allows the network to jointly attend to information across diverse representation subspaces at different positions (capturing syntactic structures, semantic co-references, and long-range dependencies concurrently).

### 2.1.2 Feed-Forward Networks (FFN)

Following the attention sub-layer, each position is processed identically and independently by a Feed-Forward Network:
```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

- Original Transformer: Standard ReLU activation.
- GPT-2 / GPT-3: Smooth Gaussian Error Linear Unit (GELU).
- Modern Architectures (LLaMA, Mistral, Gemma): Swish-Gated Linear Unit (SwiGLU) ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)):
```
SwiGLU(x) = (x @ W_gate · Swish(x @ W_1)) @ W_2
```
SwiGLU introduces a bilinear gating mechanism that delivers consistent empirical gains in perplexity and downstream task accuracy, typically adopting an intermediate hidden dimension of $\frac{8}{3}d_{\text{model}}$ to maintain FLOP parity.

### 2.1.3 Layer Normalization and Stability

**Normalization Topology**:
- **Post-Norm** (Original Transformer): `x + LayerNorm(SubLayer(x))`
  - Suffers from vanishing/exploding gradients near input layers during initial training iterations, requiring aggressive learning rate warmup schemes.
- **Pre-Norm** (GPT-2, modern LLMs): `x + SubLayer(LayerNorm(x))`
  - Places normalization directly on the residual branch, creating an unhindered gradient highway throughout the backbone and stabilizing deep network optimization.
- **RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) (LLaMA, Gemma, Qwen):
  ```
  RMSNorm(x) = (x / RMS(x)) ⊙ γ
  RMS(x) = √( (1/d) * Σ(x_i²) + ε )
  ```
  Eliminates the mean-centering step of standard LayerNorm, reducing GPU memory traffic and compute overhead while matching representation quality.

### 2.1.4 Decoder-Only vs. Encoder-Decoder Paradigms

| Architecture | Representative Models | Training Objective | Primary Modality |
|-------------|----------------------|-------------------|------------------|
| Encoder-Only | BERT, RoBERTa | Masked LM (MLM) | Semantic extraction, classification, retrieval |
| Encoder-Decoder | T5, BART | Span corruption / Seq2Seq | Conditional generation, machine translation |
| **Decoder-Only** | **GPT-4, LLaMA, Claude, Qwen** | **Causal Autoregressive LM** | **General-purpose generation, reasoning, dialogue** |

**Why Decoder-Only Dominates Modern Scaling**:
- Optimal empirical scaling efficiency: Exhibits superior scaling exponents for generative capacity per compute FLOP ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)).
- Unified objective: Autoregressive next-token prediction simultaneously trains comprehension, context reasoning, and synthesis.
- Native Zero-Shot and In-Context Learning: Prompts and responses share a continuous, homogeneous causal attention space.

## 2.2 Positional Encodings

Because self-attention is mathematically permutation-equivariant, positional information must be explicitly integrated into token representations.

### 2.2.1 Absolute Positional Encoding

**Sinusoidal Encodings** (Vaswani et al., 2017):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- Characteristic: Deterministic geometric progressions that theoretically capture relative distances via trigonometric addition formulas.
- Limitation: Degrades significantly when extrapolating to context lengths beyond the pretraining window.

**Learned Positional Embeddings** (GPT-2):
- Assigns a dedicated trainable parameter vector $P_{\text{pos}} \in \mathbb{R}^d$ to each discrete index.
- Limitation: Hard upper-bound on sequence length with zero capability for zero-shot length extrapolation.

### 2.2.2 RoPE (Rotary Position Embedding)

**The Modern Industry Standard** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)), adopted by LLaMA, Mistral, Qwen, DeepSeek, and Gemma.

**Core Formulation**: Rather than adding additive positional biases to token embeddings, RoPE applies a position-dependent orthogonal rotation matrix to Query and Key vectors in 2D coordinate slices:

```
# Treat d-dimensional projection vectors as d/2 orthogonal 2D subspaces
# Rotate by an angle proportional to position index m in each 2D plane:

R(pos, i) = [[ cos(pos * θ_i), -sin(pos * θ_i)],
             [ sin(pos * θ_i),  cos(pos * θ_i)]]

θ_i = 10000^(-2i/d)

q_rot = R(m) @ q   # Position index m
k_rot = R(n) @ k   # Position index n

# Fundamental Property: Inner product depends purely on relative displacement (m - n):
# <q_rot, k_rot> = q^T @ R(m)^T @ R(n) @ k = q^T @ R(n - m) @ k
```

**Key Advantages**:
- Preserves relative distance decay: Attention weights naturally attenuate as token distance $|m - n|$ increases.
- Seamless extrapolation: Readily extensible to ultra-long contexts via base frequency manipulation and interpolation.

### 2.2.3 Context Window Expansion Techniques

While base pretraining is frequently conducted over 4K-8K token sequences, production inference demands 128K to 1M+ context horizons.

**NTK-Aware RoPE Interpolation** ([bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)):
Instead of uniform linear position downscaling (which distorts high-frequency local positional resolution), NTK-aware scaling modifies the base frequency $\theta$:
```python
base_scaled = base_orig * (scale_factor ** (d / (d - 2)))
# where scale_factor = target_context_length / pretrain_context_length
```

**YaRN (Yet another RoPE extensioN)** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071)):
- Partitions frequency bands into three distinct regimes: high frequencies (retained without interpolation to preserve local syntax), low frequencies (pure linear interpolation), and intermediate frequencies (smooth blending).
- Injects an attention temperature multiplier $\sqrt{t}$ to counteract entropy dispersion across long sequences.
- Powers the 128K context capabilities of LLaMA 3.1.

**ALiBi (Attention with Linear Biases)** ([Press et al., 2022](https://arxiv.org/abs/2108.12409)):
- Eliminates explicit position embeddings entirely, subtracting a static linear penalty from pre-softmax attention logits:
$$\text{Attention\_Score}_{ij} = q_i k_j^T - m \cdot |i - j|$$
- $m$: Head-specific geometric slopes. Provides strong zero-shot length extrapolation without positional retraining (adopted in BLOOM and MPT).

## 2.3 Frontier Architectural Innovations

### 2.3.1 Grouped-Query Attention (GQA)

([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))

**The Memory Bandwidth Bottleneck**: In standard Multi-Head Attention (MHA), each Query head possesses a distinct Key-Value head pair. Autoregressive KV caching scales linearly with head count, creating severe memory capacity and IO bandwidth bottlenecks during multi-turn generation.

**The Solution**: Multiple Query heads share a single Key-Value head group.

```
MHA:  h Q heads, h K heads, h V heads  (e.g., h=32, g=32)
MQA:  h Q heads, 1 K head, 1 V head   (single shared KV head; maximal memory saving, marginal quality drop)
GQA:  h Q heads, g K heads, g V heads  (e.g., h=32, g=8; 4 queries share 1 KV pair)
```

- Reduces KV cache memory footprint by a factor of $h/g$ (e.g., 4x to 8x).
- Preserves downstream task quality comparable to full MHA.
- Standardized across LLaMA 2 (70B), LLaMA 3, Mistral, and Gemma 2.

```python
# Grouped-Query Attention Implementation Logic
def grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads):
    # Q: [batch, seq, num_q_heads, head_dim]
    # K, V: [batch, seq, num_kv_heads, head_dim]
    
    group_size = num_q_heads // num_kv_heads  # e.g., 4
    
    # Broadcast KV heads across their corresponding Query groups
    K_expanded = K.repeat_interleave(group_size, dim=2)
    V_expanded = V.repeat_interleave(group_size, dim=2)
    
    return scaled_dot_product_attention(Q, K_expanded, V_expanded)
```

### 2.3.2 Mixture of Experts (MoE)

**Architectural Principle**: Replaces standard dense FFN layers with a collection of specialized expert sub-networks, dynamically routing each token to a sparse subset (top-$k$) of experts ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)).

```
# Dense FFN:
output = FFN(x)

# Sparse Mixture of Experts:
gate_logits = x @ W_gate                       # [num_experts]
gate_weights, indices = top_k(softmax(gate_logits), k=2)

output = Σ(gate_weights[i] * Expert_indices[i](x))
```

**Core Benefit**: Decouples total parameter capacity from FLOP compute costs. For instance, [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) maintains 47B total parameters while activating only ~13B parameters per token during the forward pass.

**Systems and Training Challenges**:
- **Expert Load Balancing**: Without regularizing constraints, routing gates collapse into degenerate equilibria where a few experts handle all tokens while others starve. Mitigated via auxiliary balancing losses:
  $$\mathcal{L}_{\text{balance}} = \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i$$
  where $f_i$ denotes the fraction of tokens routed to expert $i$, and $P_i$ denotes the mean gating probability.
- **Inter-Node All-to-All Communication**: Distributed expert parallelism requires high-bandwidth All-to-All collective operations to route token activations across GPU ranks.
- **DeepSeek Innovations (DeepSeekMoE)** ([DeepSeek-V2](https://arxiv.org/abs/2405.04434), [DeepSeek-V3](https://arxiv.org/abs/2412.19437)):
  - **Fine-Grained Expert Granularity**: Replaces coarse 8-expert configurations with 160+ micro-experts, activating top-6 or top-8 for precise specialization.
  - **Isolated Shared Experts**: Dedicated experts always remain active to capture universal invariant knowledge, allowing routed experts to specialize strictly in distinct domains.
  - **Auxiliary-Loss-Free Balancing**: Employs adaptive dynamic bias offsets in gating scores rather than loss penalties, preserving pure language modeling loss gradients.

### 2.3.3 Multi-Head Latent Attention (MLA)

Introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434) and refined in DeepSeek-V3 and DeepSeek-R1.

**Motivation**: GQA reduces KV cache size by pooling heads, which inevitably constrains representational capacity. MLA achieves extreme KV cache compression without sacrificing multi-head modeling flexibility by projecting Key and Value representations into a low-rank latent subspace.

```
# Compression during forward pass:
c_kv = x @ W_DKV           # Down-projection: d_model (e.g. 4096) -> d_latent (e.g. 512)
k_c  = c_kv @ W_UK         # Up-projection to Key space
v_c  = c_kv @ W_UV         # Up-projection to Value space

# Inference KV Cache Optimization:
# Only the compressed latent state c_kv (512 dimensions) is stored in the KV cache,
# achieving a ~93% memory footprint reduction compared to standard MHA.
```

### 2.3.4 Sliding Window Attention

Introduced in [Mistral 7B](https://arxiv.org/abs/2310.06825): alternates local sliding window attention with global attention layers.

```
# Sliding Window Attention:
# Token at index i attends strictly to tokens within the local window [i - W, i]
# Layer stacking propagates context transitively across depth:
# Effective receptive field across L layers = L * W
# (e.g., W = 4096, 32 layers yields an effective receptive span of 131,072 tokens)
```

### 2.3.5 Numerical Stabilization and Normalization Variants

- **QK-Norm**: Normalizes Query and Key vectors (using RMSNorm) prior to computing dot-product attention matrices, preventing logit divergence and loss spikes during extended training runs ([Gemma 2](https://arxiv.org/abs/2408.00118)).
- **Logit Soft-Capping**: Applies smooth hyperbolic tangent bounding to prevent extreme logit expansion:
  $$\text{logits} = \text{cap\_val} \cdot \tanh\left(\frac{\text{logits}}{\text{cap\_val}}\right)$$

## 2.4 Compute Scaling Laws and Model Sizing

### The Chinchilla Scaling Laws

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) established the compute-optimal allocation between parameter count $N$ and pretraining token volume $D$:

$$\text{For a fixed compute budget } C \approx 6ND \text{ FLOPs: } N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}$$

**Compute-Optimal Rule of Thumb**:
$$D_{\text{optimal}} \approx 20 \times N_{\text{parameters}}$$

| Model Scale | Chinchilla Optimal Tokens | Historical Practice | Modern Paradigm (Inference-Optimized) |
|-------------|---------------------------|---------------------|---------------------------------------|
| 1B | 20B | [TinyLlama](https://github.com/jzhang38/TinyLlama) (3T) | 150x Over-trained |
| 7B / 8B | 140B–160B | LLaMA 1 (1T), LLaMA 2 (2T) | LLaMA 3 (15T; 90x Over-trained) |
| 70B | 1.4T | LLaMA 2 70B (2T) | LLaMA 3 70B (15T) |
| 405B | 8.1T | LLaMA 3 405B (15.5T) | Approaching Compute-Optimal |

**The Over-Training Shift**: Frontier models deliberately exceed Chinchilla optimality by orders of magnitude. Because deployment inference costs scale strictly with parameter size $N$ rather than training token count $D$, expending additional pretraining compute on a compact model dramatically lowers total lifetime inference expenditure.

### Architectural Parameter Configurations of Frontier Models

| Model | Parameters | Layers | Hidden Dim | Attn Heads | KV Heads | Vocab Size | Native Context |
|-------|-----------|--------|------------|------------|----------|------------|----------------|
| [LLaMA 2 7B](https://arxiv.org/abs/2307.09288) | 6.7B | 32 | 4096 | 32 | 32 (MHA) | 32K | 4K |
| [LLaMA 3 8B](https://arxiv.org/abs/2407.21783) | 8.0B | 32 | 4096 | 32 | 8 (GQA) | 128K | 128K |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 7.3B | 32 | 4096 | 32 | 8 (GQA) | 32K | 32K |
| [Qwen2 72B](https://arxiv.org/abs/2407.10671) | 72.7B | 80 | 8192 | 64 | 8 (GQA) | 152K | 128K |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B (37B active) | 61 | 7168 | 128 | MLA | 128K | 128K |

## Key Papers

- [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762): Foundational Transformer paper.
- [Su et al. (2021): RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864): Standard relative positional framework.
- [Ainslie et al. (2023): GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245): KV cache compression standard.
- [Shazeer et al. (2017): Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538): Modern sparse routing foundations.
- [DeepSeek-AI (2024): DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434): Introduction of MLA and fine-grained MoE routing.

## Further Reading

- Andrej Karpathy: [nanoGPT Repository](https://github.com/karpathy/nanoGPT) (Clean, readable GPT implementation in PyTorch).
- Harvard NLP: [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) (Line-by-line pedagogical walkthrough).
- Lilian Weng: [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) (Comprehensive survey of architectural variants).

## Exercises

1. **First-Principles Attention**: Implement scaled dot-product attention and Grouped-Query Attention (GQA) in pure NumPy or PyTorch without high-level library abstractions.
2. **KV Cache Profile**: Refactor a standard MHA block to GQA ($h_q = 32, h_{kv} = 8$), measure memory savings during generation, and verify numerical equivalence when weights are duplicated.
3. **RoPE Extrapolation Analysis**: Train a compact Transformer on sequence length $L = 512$ using Sinusoidal, ALiBi, and RoPE embeddings; evaluate perplexity degradation when evaluating on sequence lengths up to $L = 2048$.

---

[← Previous Chapter](01-tokenizer.md) | [Table of Contents](README.md) | [Next Chapter →](03-pretraining.md)
