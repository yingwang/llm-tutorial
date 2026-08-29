[← Previous Chapter](02-architecture.md) | [Table of Contents](README.md) | [Next Chapter →](04-post-training.md)

# Chapter 3: Pretraining

## 3.1 Training Objectives

### 3.1.1 Causal Language Modeling (CLM)

**The Foundational Objective of Modern LLMs**: Autoregressive next-token prediction conditioned on preceding sequence history.

$$\mathcal{L}_{\text{CLM}}(\theta) = - \sum_{t=1}^n \log P_\theta(x_t \mid x_1, x_2, \dots, x_{t-1})$$

**Why Next-Token Prediction Powers Generalized Intelligence**:
- **Compression as Comprehension**: Accurate next-token prediction over vast natural distributions forces the internal representations to encode syntax, semantic structures, world knowledge, and deductive logic.
- **Self-Supervised Data Abundance**: Eliminates the bottleneck of manual labeling; arbitrary text corpora on the internet serve directly as supervised training signals.
- **Monotonic Scaling Trajectories**: A stationary, mathematically simple objective paired with expanded compute budgets and model parameters reliably yields predictable capability gains.

### 3.1.2 Fill-in-the-Middle (FIM)

Augments standard causal generation with bi-directional context infilling, which is indispensable for code completion and iterative editing ([Bavarian et al., 2022](https://arxiv.org/abs/2207.14255)):

```
Original Document:      [Prefix: A B] [Middle: C] [Suffix: D E]
Transformed Sequence:   <fim_prefix> A B <fim_suffix> D E <fim_middle> C

# The model learns to generate the middle segment conditioned on both prefix and suffix
# Structure: Prefix-Suffix-Middle (PSM) or Suffix-Prefix-Middle (SPM)
```

- Standardized in code intelligence backbones including [StarCoder](https://arxiv.org/abs/2305.06161), [CodeLlama](https://arxiv.org/abs/2308.12950), and DeepSeek-Coder.
- Typically applied stochastically to 50% of code documents during pretraining while preserving standard CLM ordering on the remainder.

### 3.1.3 Masked Language Modeling (MLM)

Pioneered by [BERT](https://arxiv.org/abs/1810.04805): stochastically corrupts a fraction (typically 15%) of input tokens with special `[MASK]` tokens or random replacements, optimizing reconstruction cross-entropy:
```
Input:  The [MASK] sat on the [MASK]
Target: cat, mat
```
- Provides unconstrained bidirectional contextual attention, ideal for discriminative encoders and embedding extractors.
- Incurable mismatch with sequential autoregressive generation, relegating MLM primarily to embedding models and retrieval rerankers.

## 3.2 Data Engineering (The Decisive Differentiator)

### 3.2.1 Data Sourcing and Modalities

| Data Modality | Typical Volume | Quality Density | Representative Repositories |
|---------------|----------------|-----------------|-----------------------------|
| Raw Web Crawls | Multi-Petabyte | Variable / Low | [Common Crawl](https://commoncrawl.org/), [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [C4](https://huggingface.co/datasets/allenai/c4) |
| Encyclopedias & Knowledge Bases | Terabyte | High | [Wikipedia](https://dumps.wikimedia.org/), Wikidata |
| Curated Books & Literature | Terabyte | High | Books3, [Project Gutenberg](https://www.gutenberg.org/) |
| Source Code & Version Control | Terabyte | High | GitHub public repositories, [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| Scientific Literature | Terabyte | High | [arXiv](https://arxiv.org/), [PubMed Central](https://pubmed.ncbi.nlm.nih.gov/), [S2ORC](https://github.com/allenai/s2orc) |
| Technical Forums & Discussions | Terabyte | Medium | StackOverflow, technical Reddit communities |
| Formal Mathematics | Gigabyte | Very High | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [ProofPile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |

### 3.2.2 End-to-End Data Processing Pipeline

```
Raw Web Crawl Corpus (WARC format)
    │
    ▼
① URL & Domain Filtering (blocklists, spam domains, adult content)
    │
    ▼
② Language Identification (fastText lid.176.bin; classify and partition by script)
    │
    ▼
③ Structural Text Extraction (Trafilatura, Resiliparse)
    │  - Strip boilerplate headers, footers, CSS, and navigation markup
    │  - Retain structural paragraphs and markdown tables
    │
    ▼
④ Multi-Tier Quality Filtering
    │  - Document length thresholds (discard degenerate fragments)
    │  - Perplexity filtering (KenLM n-gram models trained on Wikipedia)
    │  - High-precision classifier filtering (fastText or logistic regression on curated anchors)
    │  - Heuristic rule filtering (symbol-to-word ratios, bullet-point repetition, ellipsis density)
    │
    ▼
⑤ Multi-Stage Deduplication
    │  - Exact string dedup: Document-level SHA-256 hash sets
    │  - Near-dedup: MinHash + Locality Sensitive Hashing (LSH) with Jaccard threshold > 0.8
    │  - Sub-document dedup: Suffix Array exact substring elimination
    │
    ▼
⑥ Privacy & Compliance Sanitization (PII Masking)
    │  - Regex tokenizers for emails, phone numbers, IP addresses, credit cards
    │  - High-throughput NER models for name and address anonymization
    │
    ▼
⑦ Tokenization & Memory-Mapped Binary Serialization (e.g., uint16/uint32 bin files)
```

> Production toolchains: [Hugging Face Datatrove](https://github.com/huggingface/datatrove) (scalable distributed data processing) | [Allen AI Dolma](https://github.com/allenai/dolma) (curated pipeline toolkit)

### 3.2.3 Data Mixture Optimization

**The Core Optimization Problem**: Finding the optimal convex combination of data domains that maximizes downstream model capabilities across all evaluations.

Approximate domain distribution in LLaMA 3:
```
Web Text (English):        ~50%
Web Text (Multilingual):   ~15%
Source Code:               ~17%
Mathematics & STEM:        ~5%
Curated Books:             ~5%
Knowledge & Encyclopedias: ~5%
Dialogue & Forums:         ~3%
```

**Mixture Optimization Strategies**:
1. **Small-Scale Proxy Sweeps**: Train families of 1B-parameter probe models across varying domain blends, evaluating downstream benchmark transfer to identify the Pareto frontier.
2. **Distributionally Robust Optimization (DoReMi)** ([Xie et al., 2023](https://arxiv.org/abs/2305.10429)): Employs reference models and excess loss tracking to automatically optimize domain weights via online game dynamics.
3. **Domain Transfer Empirical Heuristics**: Increasing code ratios directly elevates multi-step algorithmic reasoning; increasing math ratios sharpens symbolic logic; dialogue data primes instruction responsiveness.

### 3.2.4 Curriculum Learning and Annealing

Pretraining data is not presented as a static, homogeneous mixture throughout training; modern training recipes employ multi-phase curricula:

1. **Phase 1: Broad Foundational Pretraining** (80-90% of token budget): Broad-spectrum web corpora providing massive linguistic diversity and factual coverage.
2. **Phase 2: Domain Enrichment** (5-10% of token budget): Uprating high-value specialized domains (formal proofs, advanced coding repositories, synthetic logic problems).
3. **Phase 3: Learning Rate Annealing & High-Quality Wash** (Final 1-5% of token budget):
   - Decaying the learning rate towards zero over a short horizon.
   - Adjusting data composition to contain up to 50% premium curated data (synthetic reasoning, textbooks, vetted instruction sets), yielding substantial performance surges before checkpoint finalization.

### 3.2.5 Synthetic Data Engineering

**The Paradigm Shift**: Replacing unstructured web crawls with verified, high-density synthetic data synthesized by frontier teacher models.

| Data Category | Synthesis Methodology | Target Capability |
|---------------|----------------------|-------------------|
| Instruction & Reasoning | Multi-turn QA generated by frontier models with self-correction | SFT and Pretraining Alignment |
| Code Synthesis | LLM-generated solutions paired with automated unit test execution | Code Reasoning & Execution |
| Mathematical Proofs | Formal Lean/Isabelle steps and step-by-step synthetic solutions | Symbolic & Math Reasoning |
| Synthetic Textbooks | Synthetic pedagogical expositions ([Phi Series](https://arxiv.org/abs/2306.11644)) | Foundational Knowledge |
| Web Rewriting | Converting noisy web pages into clear, structured pedagogical essays | Knowledge Density Amplification |

**The Phi Paradigm** ([Abdin et al., 2024](https://arxiv.org/abs/2404.14219)): Demonstrated that filtering for "textbook quality" via aggressive synthetic data generation enables a 3.8B model (Phi-3) to match the benchmark performance of standard 8B-14B models.

## 3.3 Optimization and Training Dynamics

### 3.3.1 Optimization Algorithms

**AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)): The universal optimizer across production LLMs.

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t && \text{(First moment / momentum)} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 && \text{(Second uncentered moment)} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} && \text{(Bias correction)} \\
\theta_t &= \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \theta_{t-1} \right) && \text{(Decoupled weight decay update)}
\end{aligned}$$

**Hyperparameter Conventions**:
- $\beta_1 = 0.9$, $\beta_2 = 0.95$ (LLaMA family) or $\beta_2 = 0.98$ / $0.999$.
- $\varepsilon = 10^{-8}$ or $10^{-6}$ for numerical stability under reduced precision.
- Weight decay $\lambda = 0.1$.
- Peak Learning Rate: $\eta_{\text{max}} \approx 3 \times 10^{-4}$ for small models (1B-7B), scaling down to $1.0 \times 10^{-4}$ for 70B+ scales.

**Memory Overhead of Optimizer States**: For each parameter stored in BF16 (2 bytes), AdamW maintains FP32 master weights (4 bytes), FP32 first moments (4 bytes), and FP32 second moments (4 bytes), requiring 16 bytes of optimizer memory per parameter (560 GB for a 70B model before activations).

**Next-Generation Optimizers**:
- **Adafactor** ([Shazeer & Stern, 2018](https://arxiv.org/abs/1804.04235)): Factorizes second-moment matrices into row and column sums, reducing optimizer state memory from 8 bytes to $\approx 4$ bytes per parameter.
- **Lion** ([Chen et al., 2023](https://arxiv.org/abs/2302.06675)): Relies entirely on the sign of the momentum buffer, tracking only first moments in 8-bit/16-bit representations.
- **Muon** ([Jordan et al., 2025](https://arxiv.org/abs/2502.16982)): Orthogonalized momentum optimizer applying matrix Newton-Schulz iterations; exhibits rapid convergence on 2D linear projection parameters.

### 3.3.2 Learning Rate Scheduling

**Standard Cosine Decay**:
```python
# Linear warmup phase
if step < warmup_steps:
    lr = peak_lr * (step / warmup_steps)
# Cosine annealing phase
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```

**Warmup-Stable-Decay (WSD)** ([MiniCPM](https://arxiv.org/abs/2404.06395)):
- **Warmup**: Rapid ramp to peak learning rate.
- **Stable**: Sustained constant learning rate for the overwhelming majority of pretraining compute.
- **Decay**: Rapid 10% step cooldown.
- **Strategic Advantage**: Eliminates the requirement to fix total pretraining token budgets beforehand. Checkpoints can be branched off the stable trajectory and annealed on specialized domain mixtures at any point.

### 3.3.3 Training Stability and Loss Spikes

Loss spikes represent sudden, catastrophic divergence in pretraining loss metrics during large-scale cluster execution.

| Failure Mode | Root Cause | Engineering Remediation |
|--------------|------------|-------------------------|
| Gradient Explosion | Pathological token batches or sudden gradient co-alignment | Global gradient norm clipping ($\|\mathbf{g}\| \le 1.0$) |
| Corrupt Data Batches | Malformed binary files, runaway formatting strings | Real-time batch loss monitoring and outlier batch dropping |
| Attention Logit Divergence | Large dot-product values in deep layers saturated by softmax | QK-Norm (RMSNorm on Query/Key), Logit Soft-Capping |
| BF16 Dynamic Range Overflow | Cumulative embedding gradients exceeding exponent limits | FP32 residual accumulation, embedding normalization |
| Logit Drift / Entropy Collapse | Model logits drifting to extreme unconstrained values | Auxiliary $z$-loss regularization: $\mathcal{L}_z = \alpha \log^2 \sum e^{z_i}$ |

**PaLM Incident Remediation** ([Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311)): Google encountered ~20 unexplained loss spikes during the training of PaLM 540B; the production mitigation strategy relied on rolling back 100 steps prior to the divergence and skipping the exact data shards implicated in the spike.

### 3.3.4 Batch Sizing and Gradient Accumulation

$$\text{Effective Global Batch Size (Tokens)} = B_{\text{micro}} \times T_{\text{seq\_len}} \times N_{\text{grad\_accum}} \times N_{\text{data\_parallel}}$$

**Typical Scale Conventions**:
- LLaMA 2: 4 Million tokens per batch.
- LLaMA 3 (405B): Scaled dynamically from 4M up to 16M tokens per batch.

**Critical Batch Size Dynamics** ([McCandlish et al., 2018](https://arxiv.org/abs/1812.06162)): Characterizes the transition where further batch size expansion ceases to yield linear reductions in step count due to diminishing signal-to-noise ratios in empirical gradients.

### 3.3.5 Numerical Precision Frameworks

| Precision Format | Total Bits | Exponent Bits | Mantissa Bits | Dynamic Range | Primary Workload |
|------------------|------------|---------------|---------------|---------------|------------------|
| FP32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | Master weights, optimizer states |
| TF32 (NVIDIA) | 19 | 8 | 10 | $\pm 3.4 \times 10^{38}$ | Hardware matrix multiplications |
| BF16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | Standard forward and backward passes |
| FP16 | 16 | 5 | 10 | $\pm 65,504$ | Legacy mixed precision (demands loss scaling) |
| FP8 (E4M3 / E5M2) | 8 | 4 / 5 | 3 / 2 | Format dependent | Frontier training (Hopper / Blackwell) |

**FP8 Low-Precision Training at Scale** ([DeepSeek-V3](https://arxiv.org/abs/2412.19437)):
- Forward activations and weights quantized to E4M3 for higher mantissa precision.
- Backward gradients represented in E5M2 for wider dynamic range.
- Fine-grained per-tensor and per-tile dynamic scaling factors to eliminate underflow.

## 3.4 Context Window Expansion Strategies

### 3.4.1 Two-Stage Context Staging

1. **Stage 1: Short-Context Foundation**: Execute 95-98% of total pretraining FLOPs at standard 4K or 8K context windows, capturing general grammatical and domain foundations efficiently.
2. **Stage 2: Long-Context Extension**: Continue pretraining for the remaining 2-5% of tokens at expanded lengths (32K to 128K+):
   - Scale RoPE base frequencies $\theta_{\text{base}}$ upward (e.g., from $10^4$ to $5 \times 10^5$).
   - Re-balance data mixtures to include concatenated long-form documents, synthetic repository completions, and book collections.

**Computational Rationale**: The quadratic $O(N^2)$ complexity of standard attention makes training exclusively at 128K context horizons 1,024 times more computationally expensive per token step than at 4K context.

### 3.4.2 Ring Attention and Context Parallelism

([Liu et al., 2023](https://arxiv.org/abs/2310.01889)) When individual sequence lengths exceed the High Bandwidth Memory (HBM) capacity of a single accelerator, Context Parallelism shards the sequence across a ring of GPUs:

```
GPU 0: Tokens [0, 32K)    ──> Compute Attention Chunk
GPU 1: Tokens [32K, 64K)  ──> Compute Attention Chunk
GPU 2: Tokens [64K, 96K)  ──> Compute Attention Chunk
GPU 3: Tokens [96K, 128K) ──> Compute Attention Chunk
        │                        ▲
        └────── Pass KV Blocks ──┘ (Overlapped Ring P2P Communication)
```

## Key Papers

- [Brown et al. (2020): Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165): Milestone paper establishing modern in-context learning.
- [Hoffmann et al. (2022): Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556): Empirical compute-optimal scaling laws.
- [Touvron et al. (2023): Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288): Landmark blueprint for open-weight foundation model engineering.
- [DeepSeek-AI (2024): DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): Frontier demonstration of FP8 mixed-precision MoE pretraining.
- [Liu et al. (2023): Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889): Foundational Context Parallelism method.

## Further Reading

- Andrej Karpathy: [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (Comprehensive walkthrough of pretraining mechanics and GPU optimization).
- Meta AI: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Comprehensive architecture, training infrastructure, and data recipe report).
- EleutherAI: [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) (Open-source foundational corpus methodology).

## Exercises

1. **Compute-Optimal Scaling Calculations**: Given a dedicated hardware allocation of 8x NVIDIA H100 GPUs for two weeks ($C \approx 2 \times 10^{24}$ FLOPs), calculate the compute-optimal parameter count $N$ and token volume $D$ according to Chinchilla scaling equations.
2. **Pretraining from Scratch with nanoGPT**: Configure a small causal Transformer (124M parameters) using `nanoGPT`, execute pretraining over the OpenWebText dataset with BF16 mixed precision, and log training loss trajectories against validation perplexity.
3. **Data Mixture Ablation**: Implement a small-scale data mixture experiment on a 100M parameter model; compare validation perplexity when the baseline corpus is augmented with 10% high-quality synthetic math data versus raw web scrapes.

---

[← Previous Chapter](02-architecture.md) | [Table of Contents](README.md) | [Next Chapter →](04-post-training.md)
