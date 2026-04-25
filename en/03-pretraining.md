[← Previous Chapter](02-architecture.md) | [Table of Contents](README.md) | [Next Chapter →](04-post-training.md)

# Chapter 3: Pretraining

## 3.1 Training Objectives

### 3.1.1 Causal Language Modeling (CLM)

**The core objective of all modern LLMs**: predict the next token.

```
Given sequence x = (x_1, x_2, ..., x_n)
Loss = -Σ log P(x_t | x_1, ..., x_{t-1})
```

**Why next-token prediction is so powerful**:
- Compression is understanding: to accurately predict the next token, the model must understand grammar, semantics, world knowledge, and reasoning
- Self-supervised: no human annotation needed — internet text is the training data
- Scaling: a simple objective + more data + a larger model = continuous capability improvement

### 3.1.2 Fill-in-the-Middle (FIM)

Adds infilling capability on top of CLM (especially important for code completion) ([Bavarian et al., 2022](https://arxiv.org/abs/2207.14255)):

```
Original: A B C D E
Transformed: <fim_prefix> A B <fim_suffix> D E <fim_middle> C

# The model learns to fill in the middle given surrounding context
# PSM (prefix-suffix-middle) format
```

- GPT-4, [StarCoder](https://arxiv.org/abs/2305.06161), [CodeLlama](https://arxiv.org/abs/2308.12950), and other code models all use FIM
- Typically only a fraction of data undergoes FIM transformation (e.g., 50%); the rest stays CLM

### 3.1.3 Masked Language Modeling (MLM)

[BERT](https://arxiv.org/abs/1810.04805) style — randomly mask 15% of tokens and predict the masked ones:
```
Input: The [MASK] sat on the [MASK]
Target: cat, mat
```
- Bidirectional attention, good for understanding tasks
- Not suitable for generation (no masks at inference time)
- Now mainly used for encoder models (embeddings, retrieval)

## 3.2 Data (The Most Critical Part)

### 3.2.1 Data Sources

| Source | Scale | Quality | Representative datasets |
|--------|-------|---------|------------------------|
| Web crawl | PB-scale | Low | [Common Crawl](https://commoncrawl.org/), [C4](https://huggingface.co/datasets/allenai/c4), [OSCAR](https://oscar-project.github.io/documentation/) |
| Encyclopedia/Knowledge | TB-scale | High | [Wikipedia](https://dumps.wikimedia.org/), Wikidata |
| Books | TB-scale | High | Books3, [Project Gutenberg](https://www.gutenberg.org/) |
| Code | TB-scale | Medium | GitHub, [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| Scientific papers | TB-scale | High | [arXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [S2ORC](https://github.com/allenai/s2orc) |
| Conversations/Forums | TB-scale | Medium | Reddit, StackOverflow |
| Math | GB-scale | High | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [ProofPile](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |

### 3.2.2 Data Processing Pipeline

```
Raw Web Crawl
    │
    ▼
① URL filtering (remove adult/spam/policy-violating sites)
    │
    ▼
② Language identification (fastText lid.176.bin)
    │
    ▼
③ Text extraction (trafilatura, resiliparse)
    │  - Strip HTML boilerplate
    │  - Extract body text
    │
    ▼
④ Quality filtering
    │  - Length filtering (remove too-short documents)
    │  - Perplexity filtering (KenLM, remove documents with excessively high perplexity)
    │  - Classifier filtering (train a fastText classifier to distinguish high/low quality)
    │  - Rule-based filtering (remove documents with high repeated-word ratio, high special-character ratio)
    │
    ▼
⑤ Deduplication
    │  - Exact dedup: hash-based (SHA-256)
    │  - Fuzzy dedup: MinHash + LSH (jaccard similarity > 0.8 → deduplicate)
    │  - Cross-document dedup: global MinHash
    │
    ▼
⑥ PII removal
    │  - Regex matching for emails, phone numbers, ID numbers
    │  - NER models to identify names, addresses
    │
    ▼
⑦ Tokenize → binary format (stored as memory-mapped files)
```

> Tools: [HuggingFace datatrove](https://github.com/huggingface/datatrove) — complete data processing framework | [dolma](https://github.com/allenai/dolma) — Allen AI's data toolkit

### 3.2.3 Data Mix

**Key decision**: what proportions to mix data from different sources.

LLaMA 3's approximate mix:
```
Web (English): ~50%
Web (multilingual): ~15%
Code: ~17%
Math/Science: ~5%
Books: ~5%
Encyclopedia/Knowledge: ~5%
Conversations: ~3%
```

**Mix tuning methods**:
1. **Small model proxy experiments**: Run multiple mix ratios with a 1B model, pick the one that performs best on downstream tasks
2. **DoReMi** ([Xie et al., 2023](https://arxiv.org/abs/2305.10429)): Automatically learn the optimal mix using DRO (Distributionally Robust Optimization)
3. **Rules of thumb**: More code improves reasoning, more math improves math ability, more conversations improve instruction following

### 3.2.4 Data Curriculum

**Data is not fed all at once — it is adjusted in phases**:

1. **Phase 1** (majority of training): General web data — large volume, high diversity
2. **Phase 2** (later stage): Increase the proportion of high-quality data (code, math, knowledge)
3. **Annealing** (final few % of training):
   - Dramatically increase the high-quality data proportion
   - Decay learning rate toward 0
   - LLaMA 3's annealing phase raised the high-quality data proportion to ~50%

### 3.2.5 Synthetic Data

**The biggest current trend**: use strong models to generate data for training weaker models.

| Type | Method | Use case |
|------|--------|----------|
| Instruction data | Strong model (GPT-4) generates QA pairs | SFT |
| Code data | Model generates code + execution verification | Code pretraining |
| Math data | Model generates proof steps + verification | Math reasoning |
| Textbook-quality | ["Textbooks Are All You Need"](https://arxiv.org/abs/2306.11644) (Phi) | Pretraining |
| Rephrased data | Strong model rewrites web text in textbook style | Pretraining quality boost |

**[Phi series](https://arxiv.org/abs/2404.14219) (Microsoft)**: Demonstrated that high-quality synthetic data + small model can match large model performance. Phi-3 3.8B ≈ LLaMA 3 8B.

## 3.3 Training Process

### 3.3.1 Optimizer

**AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) (the standard choice for nearly all LLMs):
```
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t          # first moment (momentum)
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t²         # second moment (adaptive LR)
m̂_t = m_t / (1 - β_1^t)                         # bias correction
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + wd * θ_{t-1})  # weight decay
```

**Typical hyperparameters**:
- β_1 = 0.9, β_2 = 0.95 (LLaMA), some use 0.999
- ε = 1e-8
- weight_decay = 0.1
- Peak learning rate: 3e-4 (small models) ~ 1.5e-4 (large models)

**Adam's memory overhead**: Each parameter requires storing m (fp32) + v (fp32) + params (fp16/bf16). Optimizer states alone for a 70B model require ~560GB.

**Alternative optimizers**:
- **[Adafactor](https://arxiv.org/abs/1804.04235)**: Approximates v via matrix factorization, halving memory. Used by T5
- **[LION](https://arxiv.org/abs/2302.06675)**: From Google, uses only sign(momentum), smaller memory footprint. Requires more careful tuning
- **[Sophia](https://arxiv.org/abs/2305.14342)**: Uses second-order information (Hessian diagonal estimates) for adaptive LR. Faster convergence but more compute per step
- **[MUON](https://arxiv.org/abs/2502.16982)**: Newest optimizer, uses SVD of momentum for update direction. Dramatically accelerates convergence in some settings

### 3.3.2 Learning Rate Schedule

**Standard approach**: Linear Warmup + Cosine Decay

```
# Warmup: linearly ramp to peak over the first 2000 steps
if step < warmup_steps:
    lr = peak_lr * step / warmup_steps

# Cosine decay: slowly decay to minimum (typically 10% of peak)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
```

**WSD (Warmup-Stable-Decay)** ([MiniCPM](https://arxiv.org/abs/2404.06395)):
```
Warmup → constant learning rate (majority of training) → rapid decay
```
- Advantage: can "branch" a fast-decaying checkpoint at any point during the stable phase
- Used by MiniCPM, DeepSeek, etc.
- Convenient for continual pretraining

### 3.3.3 Training Stability

**Loss spikes**: sudden surges in loss during training, common with large models.

**Causes and mitigations**:
| Problem | Symptom | Solution |
|---------|---------|----------|
| Gradient explosion | Loss suddenly goes NaN/Inf | Gradient clipping (max_norm=1.0) |
| Data issues | Loss spikes then recovers | Skip bad batches (skip if loss > threshold) |
| Learning rate too high | Repeated spikes | Lower peak LR |
| Attention divergence | QK dot products explode | QK-Norm, Logit soft-capping |
| BF16 overflow | Spikes in embedding layer | Use FP32 for embedding layer |
| z-loss | Logit absolute values too large | Add z-loss regularization: α * log²(Σexp(logits)) |

**PaLM's experience**: Google encountered ~20 loss spikes training [PaLM 540B](https://arxiv.org/abs/2204.02311). The fix was to roll back to the pre-spike checkpoint and skip the data that caused the spike.

### 3.3.4 Batch Size Strategy

```
Effective batch size = micro_batch_size × gradient_accumulation × dp_world_size

Typical values:
- LLaMA 2: 4M tokens per batch
- GPT-4 (estimated): 60M+ tokens per batch
```

**Batch size warmup**: Use small batches early in training (better generalization), increase later (more stable, faster).

**Critical batch size** ([McCandlish et al., 2018](https://arxiv.org/abs/1812.06162)): Below this value, increasing batch size gives near-linear speedup; above it, returns diminish. Rule of thumb:
```
B_crit ≈ B_noise / L   # B_noise is gradient noise scale, L is current loss
```

### 3.3.5 Numerical Precision

| Precision | Bit width | Range | Use case |
|-----------|-----------|-------|----------|
| FP32 | 32-bit | ±3.4e38 | Optimizer states, master weights |
| TF32 | 19-bit | Same as FP32 | Default matmul on NVIDIA Ampere+ |
| BF16 | 16-bit | ±3.4e38 | **Mainstream training precision** (same range as FP32) |
| FP16 | 16-bit | ±65504 | Previous-gen training precision (requires loss scaling) |
| FP8 | 8-bit | E4M3/E5M2 | Frontier training on H100+ |

**Mixed Precision Training** ([Micikevicius et al., 2018](https://arxiv.org/abs/1710.03740)):
```python
# PyTorch AMP
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# Gradient computation in BF16, optimizer update in FP32
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**FP8 Training** (H100/B200):
- Forward pass uses E4M3 (higher precision)
- Backward pass uses E5M2 (wider range)
- Requires per-tensor scaling
- DeepSeek-V3 successfully trained a 671B MoE in FP8

## 3.4 Context Length Training

### 3.4.1 Short-to-Long Training Strategy

1. **Phase 1**: Pretrain on 4K/8K context for the majority of tokens
2. **Phase 2**: Continue training on long sequences (32K-128K) for a small number of tokens
   - Adjust RoPE base frequency
   - Only ~1-5% of total training tokens needed

**Why not train directly on long sequences**:
- Attention's O(n²) compute cost — 128K is 1024x the cost of 4K
- Long documents are scarce; short documents are more diverse
- Learn language fundamentals first, then long-range dependencies

### 3.4.2 Ring Attention / Context Parallelism

([Liu et al., 2023](https://arxiv.org/abs/2310.01889)) When ultra-long sequences cannot fit on a single GPU, split the sequence into chunks distributed across multiple GPUs, using ring communication to pass KV blocks.

```
GPU 0: tokens 0-32K      → compute attention chunk
GPU 1: tokens 32K-64K    → compute attention chunk
GPU 2: tokens 64K-96K    → compute attention chunk
GPU 3: tokens 96K-128K   → compute attention chunk
           ↻ KV blocks passed between GPUs in a ring
```

---

[← Previous Chapter](02-architecture.md) | [Table of Contents](README.md) | [Next Chapter →](04-post-training.md)
