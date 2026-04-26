[← 上一章](02-architecture.md) | [目录](../README.md) | [下一章 →](04-post-training.md)

# 第三章：预训练 (Pretraining)

## 3.1 训练目标

### 3.1.1 Causal Language Modeling (CLM)

**所有现代 LLM 的核心目标**: 预测下一个 token。

```
给定序列 x = (x_1, x_2, ..., x_n)
Loss = -Σ log P(x_t | x_1, ..., x_{t-1})
```

**为什么 next-token prediction 这么强大**:
- 压缩即理解：要准确预测下一个 token，模型必须理解语法、语义、世界知识、推理
- Self-supervised：不需要人工标注，互联网文本就是训练数据
- Scaling：简单的目标 + 更多数据 + 更大模型 = 持续的能力提升

### 3.1.2 Fill-in-the-Middle (FIM)

在 CLM 基础上加入中间填充能力（对代码补全尤其重要）([Bavarian et al., 2022](https://arxiv.org/abs/2207.14255)):

```
原始: A B C D E
变换: <fim_prefix> A B <fim_suffix> D E <fim_middle> C

# 模型学会了在知道前后文的情况下填充中间
# PSM (prefix-suffix-middle) 格式
```

- GPT-4, [StarCoder](https://arxiv.org/abs/2305.06161), [CodeLlama](https://arxiv.org/abs/2308.12950) 等代码模型都用 FIM
- 通常只对一部分数据做 FIM 变换（如 50%），其余保持 CLM

### 3.1.3 Masked Language Modeling (MLM)

[BERT](https://arxiv.org/abs/1810.04805) 风格，随机 mask 15% 的 token，预测被 mask 的 token：
```
输入: The [MASK] sat on the [MASK]
目标: cat, mat
```
- 双向注意力，适合理解任务
- 不适合生成（推理时没有 mask）
- 现在主要用于 encoder 模型（embedding、检索）

## 3.2 数据 (最关键的部分)

### 3.2.1 数据来源

| 来源 | 规模 | 质量 | 代表数据集 |
|------|------|------|-----------|
| Web crawl | PB级 | 低 | [Common Crawl](https://commoncrawl.org/), [C4](https://huggingface.co/datasets/allenai/c4), [OSCAR](https://oscar-project.github.io/documentation/) |
| 百科/知识 | TB级 | 高 | [Wikipedia](https://dumps.wikimedia.org/), Wikidata |
| 书籍 | TB级 | 高 | Books3, [Project Gutenberg](https://www.gutenberg.org/) |
| 代码 | TB级 | 中 | GitHub, [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| 科学论文 | TB级 | 高 | [arXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [S2ORC](https://github.com/allenai/s2orc) |
| 对话/论坛 | TB级 | 中 | Reddit, StackOverflow |
| 数学 | GB级 | 高 | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [ProofPile](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |

### 3.2.2 数据处理 Pipeline

```
Raw Web Crawl
    │
    ▼
① URL 过滤 (去掉成人/垃圾/违规网站)
    │
    ▼
② 语言识别 (fastText lid.176.bin)
    │
    ▼
③ 文本提取 (trafilatura, resiliparse)
    │  - 去 HTML boilerplate
    │  - 提取正文
    │
    ▼
④ 质量过滤
    │  - 长度过滤 (去掉太短的)
    │  - 困惑度过滤 (KenLM, 去掉 perplexity 太高的)
    │  - 分类器过滤 (训练一个 fastText 分类器区分高质量/低质量)
    │  - 规则过滤 (去掉重复词比例高、特殊字符比例高的)
    │
    ▼
⑤ 去重 (Deduplication)
    │  - Exact dedup: hash-based (SHA-256)
    │  - Fuzzy dedup: MinHash + LSH (jaccard similarity > 0.8 → 去重)
    │  - 跨文档去重: 全局 MinHash
    │
    ▼
⑥ PII 去除
    │  - 正则匹配邮箱、电话、身份证号
    │  - NER 模型识别人名、地址
    │
    ▼
⑦ Tokenize → 二进制格式 (存为 memory-mapped 文件)
```

> 工具: [HuggingFace datatrove](https://github.com/huggingface/datatrove) — 完整的数据处理框架 | [dolma](https://github.com/allenai/dolma) — Allen AI 的数据 toolkit

### 3.2.3 数据配比 (Data Mix)

**关键决策**：不同来源的数据按什么比例混合。

LLaMA 3 的大致配比:
```
Web (英文): ~50%
Web (多语言): ~15%
代码: ~17%
数学/科学: ~5%
书籍: ~5%
百科/知识: ~5%
对话: ~3%
```

**配比调优方法**:
1. **小模型代理实验**: 用 1B 模型跑多组不同配比，选下游任务最好的
2. **DoReMi** ([Xie et al., 2023](https://arxiv.org/abs/2305.10429)): 用 DRO (Distributionally Robust Optimization) 自动学习最优配比
3. **经验法则**: 代码比例↑提升推理能力，数学比例↑提升数学能力，对话比例↑提升指令遵循

### 3.2.4 Data Curriculum

**不是一股脑把所有数据灌进去，而是按阶段调整数据**:

1. **Phase 1** (大部分训练): 通用 web 数据，大量、多样
2. **Phase 2** (后期): 增加高质量数据比例（代码、数学、知识）
3. **Annealing** (最后几%的训练): 
   - 大幅提升高质量数据比例
   - 降低学习率到接近 0
   - LLaMA 3 的 annealing 阶段把高质量数据比例提到 ~50%

### 3.2.5 Synthetic Data

**当前最大趋势**: 用强模型生成数据训练弱模型。

| 类型 | 方法 | 用途 |
|------|------|------|
| Instruction data | 强模型 (GPT-4) 生成 QA pairs | SFT |
| Code data | 模型生成代码 + 执行验证 | Code pretraining |
| Math data | 模型生成证明步骤 + 验证 | Math reasoning |
| Textbook-quality | ["Textbooks Are All You Need"](https://arxiv.org/abs/2306.11644) (Phi) | 预训练 |
| Rephrased data | 用强模型重写 web 文本为教科书风格 | 预训练质量提升 |

**[Phi 系列](https://arxiv.org/abs/2404.14219) (Microsoft)**: 证明高质量 synthetic data + 小模型可以达到大模型级别的性能。Phi-3 3.8B ≈ LLaMA 3 8B。

## 3.3 训练过程

### 3.3.1 优化器

**AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) (几乎所有 LLM 的标准选择):
```
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t          # 一阶矩 (momentum)
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t²         # 二阶矩 (adaptive LR)
m̂_t = m_t / (1 - β_1^t)                         # bias correction
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + wd * θ_{t-1})  # weight decay
```

**典型超参**:
- β_1 = 0.9, β_2 = 0.95 (LLaMA), 有的用 0.999
- ε = 1e-8
- weight_decay = 0.1
- 峰值学习率: 3e-4 (小模型) ~ 1.5e-4 (大模型)

**Adam 的内存开销**: 每个参数需要存 m (fp32) + v (fp32) + params (fp16/bf16)。一个 70B 模型光优化器状态就要 ~560GB。

**替代优化器**:
- **[Adafactor](https://arxiv.org/abs/1804.04235)**: 用矩阵分解近似 v，内存减半。T5 用
- **[LION](https://arxiv.org/abs/2302.06675)**: Google 提出，只用 sign(momentum)，内存更小。但需要更仔细的调参
- **[Sophia](https://arxiv.org/abs/2305.14342)**: 用二阶信息（Hessian diagonal 估计）做 adaptive LR。收敛更快但计算更贵
- **[MUON](https://arxiv.org/abs/2502.16982)**: 最新的优化器，用 momentum 的 SVD 来做更新方向。某些设置下大幅加速收敛

### 3.3.2 学习率调度

**标准方案**: Linear Warmup + Cosine Decay

```
# Warmup: 前 2000 步线性增到峰值
if step < warmup_steps:
    lr = peak_lr * step / warmup_steps

# Cosine decay: 缓慢降到最小值 (通常是峰值的 10%)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
```

**WSD (Warmup-Stable-Decay)** ([MiniCPM](https://arxiv.org/abs/2404.06395)):
```
Warmup → 恒定学习率（大部分训练时间）→ 快速 decay
```
- 优势：可以随时在 stable 阶段"分支"出一个快速 decay 的 checkpoint
- MiniCPM, DeepSeek 等使用
- 方便做 continual pretraining

### 3.3.3 训练稳定性

**Loss spike**: 训练中 loss 突然飙升的现象，常见于大模型。

**原因和对策**:
| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 梯度爆炸 | loss 突然 NaN/Inf | gradient clipping (max_norm=1.0) |
| 数据问题 | loss spike 后可恢复 | 跳过坏 batch (loss > threshold 就跳过) |
| 学习率过大 | 反复 spike | 降低 peak LR |
| 注意力发散 | QK 点积爆炸 | QK-Norm, Logit soft-capping |
| BF16 溢出 | embedding 层 spike | embedding 层用 FP32 |
| z-loss | logits 绝对值过大 | 加 z-loss 正则化: α * log²(Σexp(logits)) |

**PaLM 的经验**: Google 训练 [PaLM 540B](https://arxiv.org/abs/2204.02311) 时遇到 ~20 次 loss spike，处理方式是回滚到 spike 前的 checkpoint，跳过导致 spike 的数据。

### 3.3.4 Batch Size 策略

```
有效 batch size = micro_batch_size × gradient_accumulation × dp_world_size

典型值:
- LLaMA 2: 4M tokens per batch
- GPT-4 (推测): 60M+ tokens per batch
```

**Batch size warmup**: 训练初期用小 batch（更好的泛化），后期增大（更稳定、更快）。

**Critical batch size** ([McCandlish et al., 2018](https://arxiv.org/abs/1812.06162)): 低于此值，增大 batch 几乎线性加速；高于此值，收益递减。经验公式:
```
B_crit ≈ B_noise / L   # B_noise 是梯度噪声规模，L 是当前 loss
```

### 3.3.5 数值精度

| 精度 | 位宽 | 范围 | 用途 |
|------|------|------|------|
| FP32 | 32位 | ±3.4e38 | 优化器状态、master weights |
| TF32 | 19位 | 同 FP32 | NVIDIA Ampere+ 默认 matmul |
| BF16 | 16位 | ±3.4e38 | **主流训练精度** (范围同 FP32) |
| FP16 | 16位 | ±65504 | 旧一代训练精度 (需要 loss scaling) |
| FP8 | 8位 | E4M3/E5M2 | H100+ 前沿训练 |

**Mixed Precision Training** ([Micikevicius et al., 2018](https://arxiv.org/abs/1710.03740)):
```python
# PyTorch AMP
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# 梯度计算用 BF16，优化器更新用 FP32
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**FP8 Training** (H100/B200):
- 前向用 E4M3 (更高精度)
- 反向用 E5M2 (更大范围)
- 需要 per-tensor scaling
- DeepSeek-V3 成功在 FP8 下训练了 671B MoE

## 3.4 上下文长度训练

### 3.4.1 短→长训练策略

1. **Phase 1**: 在 4K/8K context 上预训练大部分 token
2. **Phase 2**: 在长序列 (32K-128K) 上继续训练少量 token
   - 调整 RoPE base frequency
   - 只需要 ~1-5% 的总训练 token 数

**为什么不直接在长序列上训练**:
- Attention 的 O(n²) 计算量，128K 是 4K 的 1024 倍
- 长文档数据稀缺，短文档更多样
- 先学好语言基础，再学长距离依赖

### 3.4.2 Ring Attention / Context Parallelism

([Liu et al., 2023](https://arxiv.org/abs/2310.01889)) 超长序列不能放到一张卡上时，把序列切成 chunks 分布到多张卡，用 ring 通信传递 KV。

```
GPU 0: tokens 0-32K      → 计算 attention chunk
GPU 1: tokens 32K-64K    → 计算 attention chunk
GPU 2: tokens 64K-96K    → 计算 attention chunk
GPU 3: tokens 96K-128K   → 计算 attention chunk
           ↻ KV blocks 在 GPU 间 ring 传递
```

## 关键论文

- [Brown et al. (2020) — GPT-3](https://arxiv.org/abs/2005.14165) — few-shot 学习，开启大模型时代
- [Hoffmann et al. (2022) — Chinchilla](https://arxiv.org/abs/2203.15556) — Scaling laws，token / 参数比 ≈ 20
- [Touvron et al. (2023) — Llama 2](https://arxiv.org/abs/2307.09288) — 开源大模型工程细节
- [DeepSeek-AI (2024) — DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MoE + FP8 + MTP
- [Liu et al. (2023) — Ring Attention](https://arxiv.org/abs/2310.01889) — 超长上下文训练

## 进一步阅读

- [Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)：4 小时手写训练 GPT-2
- [Llama 3 — Model Card & Tech Report](https://arxiv.org/abs/2407.21783)：405B 训练全貌
- [Eleuther — The Pile](https://arxiv.org/abs/2101.00027)：开源大规模语料构建

## 练习题

1. **Chinchilla 推算**：给定 2x H100（约 2000 PFLOP/s）训练 1 周的算力预算，按 Chinchilla 最优算出能训多大的模型 + 多少 token。
2. **跑 nanoGPT 复现 GPT-2**：用 OpenWebText 训一个 124M 模型，记录 loss 曲线和 perplexity。
3. **数据消融**：把训练集中 5% 替换为高质量代码（StarCoder）或数学（MATH），观察 HumanEval / GSM8K 是否提升。

---

[← 上一章](02-architecture.md) | [目录](../README.md) | [下一章 →](04-post-training.md)
