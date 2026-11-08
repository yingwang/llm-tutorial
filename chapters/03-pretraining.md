[← 上一章](02-architecture.md) | [目录](../README.md) | [下一章 →](04-post-training.md)

# 第三章：预训练 (Pretraining)

## 3.1 训练目标

### 3.1.1 Causal Language Modeling (CLM)

现代生成式大语言模型的核心预训练目标其实只有一件事：根据前文预测下一个 Token（Next-Token Prediction）。

```
给定序列 x = (x_1, x_2, ..., x_n)
Loss = -Σ log P(x_t | x_1, ..., x_{t-1})
```

**为什么预测下一个词这么有效**：
- **压缩即理解**：想要准确预测下一处该填什么词，模型不能只靠死记硬背，还必须顺带学会语法结构、事实知识、上下文指代甚至逻辑推理；
- **纯自监督学习**：不需要人工打标签，互联网上的海量文本直接就能拿来当训练样本；
- **可持续扩展**：这个简单的目标配合更大的模型、更多的数据和算力，能力可以沿着 Scaling Law 持续提升。

### 3.1.2 Fill-in-the-Middle (FIM)

标准自回归模型只能从左往右接着写，但在写代码或改文档时，我们经常需要“看着光标前后的内容，补全中间空缺的一段”。Fill-in-the-Middle（[Bavarian et al., 2022](https://arxiv.org/abs/2207.14255)）通过简单改造训练数据的排布顺序，让自回归模型直接具备中间补全能力：

```
原始: A B C D E
变换: <fim_prefix> A B <fim_suffix> D E <fim_middle> C

# 模型学会了在知道前后文的情况下填充中间
# PSM (prefix-suffix-middle) 格式
```

- GPT-4、[StarCoder](https://arxiv.org/abs/2305.06161)、[CodeLlama](https://arxiv.org/abs/2308.12950)、DeepSeek-Coder 等代码与通用模型都采用了 FIM；
- 训练时通常只把一部分样本（例如 50%）随机切开做 FIM 变换，其余数据仍按正常从左到右的顺序训练，不会影响常规续写能力。

### 3.1.3 Masked Language Modeling (MLM)

[BERT](https://arxiv.org/abs/1810.04805) 代表的训练方式：随机遮盖（Mask）输入序列里 15% 的 Token，让模型结合前后双向上下文把被遮住的词还原出来：
```
输入: The [MASK] sat on the [MASK]
目标: cat, mat
```
- **优势**：每一层都能同时看到左右两侧的上下文，适合做文本分类、命名实体识别、Embedding 向量提取与检索重排；
- **劣势**：训练时有 `[MASK]` 标记而推理时没有，且每次只算 15% 位置的损失，不适合直接做长文本生成。如今主要用于训练专门的编码器（Encoder）模型。

## 3.2 数据 (最关键的部分)

模型架构演进到今天已经高度趋同，决定基座模型能力上限的最关键因素往往是预训练数据的质量、覆盖面与配比。

### 3.2.1 数据来源

| 来源 | 规模 | 质量 | 代表数据集 |
|------|------|------|-----------|
| Web crawl | PB级 | 参差不齐 | [Common Crawl](https://commoncrawl.org/), [C4](https://huggingface.co/datasets/allenai/c4), [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| 百科/知识 | TB级 | 高 | [Wikipedia](https://dumps.wikimedia.org/), Wikidata |
| 书籍 | TB级 | 高 | Books3, [Project Gutenberg](https://www.gutenberg.org/) |
| 代码 | TB级 | 中高 | GitHub, [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| 科学论文 | TB级 | 高 | [arXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [S2ORC](https://github.com/allenai/s2orc) |
| 对话/论坛 | TB级 | 中 | Reddit, StackOverflow |
| 数学 | GB级 | 高 | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [ProofPile](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |

### 3.2.2 数据处理 Pipeline

从原始网页抓取（Common Crawl）到真正送进 GPU 训练，中间要经过一条严密的清洗流水线，通常会筛掉 80%–90% 的低质与重复内容：

```
Raw Web Crawl
    │
    ▼
① URL 过滤 (去掉成人/垃圾/广告/违规网站)
    │
    ▼
② 语言识别 (fastText lid.176.bin)
    │
    ▼
③ 文本提取 (trafilatura, resiliparse)
    │  - 剥离 HTML 标签、导航栏与页脚模板 (boilerplate)
    │  - 提取正文并保留基础段落与公式格式
    │
    ▼
④ 质量过滤
    │  - 长度过滤 (去掉过短或无完整句子的碎片)
    │  - 困惑度过滤 (用 KenLM 语言模型筛掉 perplexity 异常高的乱码文本)
    │  - 分类器过滤 (用 fastText 或小模型区分百科/教材级高质文本与垃圾网页)
    │  - 启发式规则 (去掉重复词比例过高、特殊符号异常堆砌的页面)
    │
    ▼
⑤ 去重 (Deduplication)
    │  - 精确去重: 基于 SHA-256 哈希去掉完全相同的文档或段落
    │  - 模糊去重: MinHash + LSH (Jaccard 相似度 > 0.8 即视为重复)
    │  - 跨文档子串去重: 后缀数组 (Suffix Array) 剔除大段重复模板
    │
    ▼
⑥ PII 去除 (隐私脱敏)
    │  - 正则匹配替换邮箱、电话、身份证号、IP 地址
    │  - NER 模型识别并隐去敏感个人信息
    │
    ▼
⑦ Tokenize → 打包存为二进制 memory-mapped 文件
```

> 常用开源清洗工具：[HuggingFace datatrove](https://github.com/huggingface/datatrove)（大规模分布式数据处理框架） | [dolma](https://github.com/allenai/dolma)（Allen AI 开源的数据处理工具链）

### 3.2.3 数据配比 (Data Mix)

不同来源的语料按什么比例混合喂给模型，会直接影响模型在各项能力上的长短板。以 LLaMA 3 为例，其预训练语料的大致构成为：
```
Web (英文): ~50%
Web (多语言): ~15%
代码: ~17%
数学/科学: ~5%
书籍: ~5%
百科/知识: ~5%
对话: ~3%
```

**怎么找最优配比**：
1. **小模型代理实验**：先在 1B 规模的小模型上跑多组不同配比，看哪组在下游评测集上涨分最快，再把验证过的配比放大到主模型；
2. **DoReMi**（[Xie et al., 2023](https://arxiv.org/abs/2305.10429)）：利用分布鲁棒优化（DRO），根据小模型在各个数据域上的损失下降难易程度自动调整各域权重；
3. **常见经验**：适当提高代码占比能显著增强长链条逻辑推理能力，提高数学与论文占比有助符号推导，而书籍数据则对长文本连贯性至关重要。

### 3.2.4 Data Curriculum

现代预训练很少从头到尾使用同一套数据分布，而是按训练阶段动态调整数据内容（课程学习）：

1. **第一阶段（常规预训练，占绝大部分步数）**：使用覆盖面广、多样性高的通用网页与基础语料，打好语言与世界常识底子；
2. **第二阶段（中期能力强化）**：逐步调高代码、数学、科学文献与长文本的比例；
3. **退火阶段（Annealing，最后 1%–5% 的训练步数）**：
   - 把极高质量的精选数据（百科、教材、精选问答、合成数学与代码）比例大幅拉高到 50% 甚至更高；
   - 同时把学习率快速衰减到接近 0。LLaMA 3 和 DeepSeek-V3 都在退火阶段通过这种做法获得了明显的评测涨点。

### 3.2.5 Synthetic Data

高质量的自然人类文本（尤其是带详细解题步骤的数学、严谨的代码和教材级解释）正在被逐渐耗尽，用强模型生成合成数据（Synthetic Data）已经成为预训练和后训练的重要补充：

| 类型 | 方法 | 用途 |
|------|------|------|
| Instruction data | 用强模型生成多样化指令与回答对 | SFT 指令微调 |
| Code data | 模型生成代码与测试用例 + 沙盒运行验证 | 代码预训练与强化 |
| Math data | 模型生成分步推导过程 + 符号求解器/答案校验 | 数学推理能力 |
| Textbook-quality | 仿照教材风格生成概念讲解与例题（["Textbooks Are All You Need"](https://arxiv.org/abs/2306.11644)） | 高密度预训练 |
| Rephrased data | 用模型把行文杂乱的网页改写成条理清晰的知识段落 | 提升网页语料信噪比 |

**[Phi 系列](https://arxiv.org/abs/2404.14219)（Microsoft）**展示了一个典型案例：只要把合成的“教科书级”高密度语料做精，3.8B 参数的 Phi-3 在推理和代码基准上就能追平甚至超过许多用普通网页语料训出来的 7B–8B 模型。

## 3.3 训练过程

### 3.3.1 优化器

**AdamW**（[Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)）目前仍是几乎所有大模型预训练的默认选择：
```
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t          # 一阶矩 (momentum)
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t²         # 二阶矩 (adaptive LR)
m̂_t = m_t / (1 - β_1^t)                         # bias correction
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + wd * θ_{t-1})  # 解耦权重衰减
```

**典型超参设置**：
- $\beta_1 = 0.9, \beta_2 = 0.95$（LLaMA、DeepSeek 常用 0.95，比默认的 0.999 对梯度突变反应更快，能减少 Loss Spike）；
- $\epsilon = 1\times 10^{-8}$，`weight_decay = 0.1`；
- 峰值学习率：小模型常用 `3e-4` 到 `6e-4`，百亿到千亿大模型通常降到 `1e-4`–`1.5e-4`。

**AdamW 的显存开销**：为了保证数值精度，混合精度训练时每个参数除了存 BF16 权重（2 字节）和 BF16 梯度（2 字节），优化器里还得存一份 FP32 主权重（4 字节）、FP32 一阶矩 $m$（4 字节）和 FP32 二阶矩 $v$（4 字节），加起来优化器状态就要占 12 字节/参数。对于一个 70B 模型，仅优化器状态就要占约 840 GB 显存。

**其他替代优化器**：
- **[Adafactor](https://arxiv.org/abs/1804.04235)**：用行列向量的外积低秩近似二阶矩 $v$，省下一半优化器显存，T5 曾使用；
- **[LION](https://arxiv.org/abs/2302.06675)**：只追踪一阶动量并用符号函数 $\text{sign}()$ 更新，显存开销更小，但对学习率和权重衰减超参更敏感；
- **[Sophia](https://arxiv.org/abs/2305.14342)**：引入轻量的对角海森矩阵（Hessian diagonal）近似二阶曲率，自适应调整各维度步长；
- **[MUON](https://arxiv.org/abs/2502.16982)**：对矩阵参数的动量做 Newton-Schulz 正交化更新，在不少中大规模预训练实验里展现出比 AdamW 更快的收敛速度。

### 3.3.2 学习率调度

**经典方案**：Linear Warmup + Cosine Decay（线性预热 + 余弦衰减）

```python
# Warmup: 前 2000 步从 0 线性升到峰值学习率
if step < warmup_steps:
    lr = peak_lr * step / warmup_steps

# Cosine decay: 按余弦曲线平滑降到最小值 (通常是峰值的 10%)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
```

**WSD (Warmup-Stable-Decay)**（[MiniCPM](https://arxiv.org/abs/2404.06395)）：
```
Warmup (预热) → 保持恒定学习率（占主体训练时间）→ 最后阶段快速 Decay 退火
```
- **优势**：余弦调度必须事先定死总训练步数 `total_steps`，中途想加数据很不方便；而 WSD 在恒定学习率阶段可以一直训下去，随时可以从任意 Checkpoint 拉出一条分支做快速退火拿到可用模型；
- MiniCPM、DeepSeek-V3 等模型都采用了这种或类似的调度策略。

### 3.3.3 训练稳定性

大规模预训练动辄跑几周甚至几个月，途中经常会遇到 Loss 突然急剧上升（Loss Spike）甚至变成 `NaN` 的情况。

**常见原因与应对手段**：

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 梯度爆炸 | 梯度范数急剧增大，Loss 突然变 NaN/Inf | 开启梯度裁剪 `gradient clipping (max_norm=1.0)` |
| 脏数据批次 | 遇到某个 Batch 后 Loss 骤升，随后慢慢恢复 | 检测并自动跳过异常 Batch，排查对应数据分片 |
| 学习率偏高 | Loss 频繁出现尖峰震荡 | 调低峰值学习率，或把 AdamW 的 $\beta_2$ 从 0.999 降到 0.95 |
| 注意力数值发散 | 深层 QK 点积过大导致 Softmax 饱和 | 给 Q 和 K 加 RMSNorm（QK-Norm），或做 Logit Soft-Capping |
| 低精度溢出 | Embedding 层或输出层数值越界 | 保持 Embedding 与 LM Head 的计算在 FP32 下进行 |
| Logits 整体漂移 | 输出层 Logits 绝对值越训越大 | 加入 z-loss 正则项：$\alpha \cdot \log^2(\sum \exp(\text{logits}))$ |

**工程经验**：Google 在训练 [PaLM 540B](https://arxiv.org/abs/2204.02311) 时遇到了大约 20 次明显的 Loss Spike。最直接管用的工程恢复办法就是：从尖峰出现前约 100 步的 Checkpoint 重新拉起训练，并跳过触发尖峰的那几批数据。

### 3.3.4 Batch Size 策略

分布式训练中的全局有效批次大小（Global Batch Size，常按总 Token 数计量）由三项相乘决定：
```
有效 batch size = micro_batch_size × gradient_accumulation × dp_world_size

典型值:
- LLaMA 2: 4M tokens per batch
- LLaMA 3 405B / DeepSeek-V3: 16M ~ 60M tokens per batch
```

**Batch Size Warmup**：训练刚起步时梯度方向变化快，先用较小的全局 Batch Size（如 4M Tokens）跑完早期阶段，中后期再逐步放大到 16M–60M Tokens，兼顾早期收敛速度与后期的多卡并行吞吐。

**临界批次大小（Critical Batch Size）**（[McCandlish et al., 2018](https://arxiv.org/abs/1812.06162)）：当 Batch Size 小于临界值时，加倍 Batch Size 差不多能把训练步数减半（近乎线性加速）；一旦超过临界值，继续靠增加 GPU 数量增大 Batch Size 只会带来梯度信息冗余，算力利用率显著下降：
```
B_crit ≈ B_simple = tr(Σ) / |G|²  # McCandlish et al. 的简化梯度噪声尺度：Σ 为单样本梯度协方差，G 为真实梯度
B_crit(L) ≈ B* / L^(1/α_B)        # Kaplan et al. (2020) 的经验拟合：B* ≈ 2e8 tokens，α_B ≈ 0.21
```
随着训练推进、Loss $L$ 不断下降，临界批次大小 $B_{\text{crit}}$ 会自然变大，这也解释了为什么训练中后期可以把 Batch Size 调得更大。

### 3.3.5 数值精度

| 精度 | 位宽 | 动态范围 | 典型用途 |
|------|------|---------|---------|
| FP32 | 32位 (1+8+23) | $\pm 3.4\times 10^{38}$ | 优化器状态、Master Weights、归一化累加 |
| TF32 | 19位 (1+8+10) | 同 FP32 | NVIDIA Ampere+ 架构默认矩阵乘内部格式 |
| BF16 | 16位 (1+8+7) | 同 FP32 | **主流训练默认精度**（指数位同 FP32，不易溢出） |
| FP16 | 16位 (1+5+10) | $\pm 65504$ | 早期半精度训练（动态范围窄，必须配 Loss Scaling） |
| FP8 | 8位 | E4M3 / E5M2 | H100 / B200 等新架构上的高吞吐训练 |

**混合精度训练（Mixed Precision Training）**（[Micikevicius et al., 2018](https://arxiv.org/abs/1710.03740)）：
```python
# PyTorch AMP 示例（BF16 指数位与 FP32 相同，不需要 GradScaler）
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# 前向与反向用 BF16 加速，优化器用 FP32 主权重更新
loss.backward()
optimizer.step()
optimizer.zero_grad()

# 若改用 FP16（dtype=torch.float16），则需要损失缩放：
# scaler = torch.amp.GradScaler('cuda')
# scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
```

**FP8 训练**（H100 / B200）：
- 前向传播对精度更敏感，使用尾数多一位的 **E4M3**（4 位指数、3 位尾数）；
- 反向传播梯度变化范围大，使用指数多一位的 **E5M2**（5 位指数、2 位尾数）；
- 配合细粒度的按块缩放（Tile/Block-wise Scaling）防止数值下溢；
- DeepSeek-V3 在 671B MoE 模型上跑通了全流程 FP8 预训练，在几乎不损精度的前提下大幅提升了计算吞吐。

## 3.4 上下文长度训练

### 3.4.1 短→长两阶段训练策略

几乎没有模型会从第 1 个 Token 起就直接用 128K 的序列长度做预训练，标准做法都是“先短后长”：

1. **第一阶段（常规短序列预训练）**：在 4K 或 8K 上下文窗口下完成 90%–95% 以上的训练量；
2. **第二阶段（长上下文持续训练）**：调大 RoPE 的基频（如从 10,000 调到 500,000 或更高），换用包含长文档的语料，在 32K–128K 长度上继续训练 1%–5% 的 Token。

**为什么不从一开始就用长序列**：
- 自注意力的计算量随序列长度呈 $\mathcal{O}(n^2)$ 增长，直接训 128K 序列的单步算力开销是 4K 的上千倍；
- 互联网上真正具备长程依赖的高质量长文档（长篇书籍、完整代码库）占比并不高；
- 模型先在短窗口里把词法、句法和基础推理学扎实，后期只需要少量长文本就能快速学会跨长距离检索信息。

### 3.4.2 Ring Attention / Context Parallelism

（[Liu et al., 2023](https://arxiv.org/abs/2310.01889)）当序列长度扩到 128K 甚至 1M 时，单条样本的中间激活值和 KV 张量在一张 GPU 上根本装不下。这时需要把同一条长序列按位置切成多段，分给多张 GPU 协同计算（即上下文并行 Context Parallelism）：

```
GPU 0: tokens 0-32K      → 计算本段 attention
GPU 1: tokens 32K-64K    → 计算本段 attention
GPU 2: tokens 64K-96K    → 计算本段 attention
GPU 3: tokens 96K-128K   → 计算本段 attention
           ↻ KV blocks 在 GPU 之间排成环状边传边算
```
每张卡拿着自己那一段的 Query 不动，一边计算当前手头 KV 块的注意力，一边通过环形网络把 KV 块传给下一张卡，让通信时间完美隐藏在矩阵计算背后。

## 关键论文

- [Brown et al. (2020): GPT-3 (Language Models are Few-Shot Learners)](https://arxiv.org/abs/2005.14165)：展示大规模预训练带来的上下文少样本学习能力
- [Hoffmann et al. (2022): Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)：明确模型参数量与训练 Token 数的算力最优配比
- [Touvron et al. (2023): LLaMA 2](https://arxiv.org/abs/2307.09288)：详尽公开预训练与后训练流程的开源基座报告
- [DeepSeek-AI (2024): DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：涵盖细粒度 MoE、FP8 混合精度训练与多 Token 预测的工程实践
- [Liu et al. (2023): Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)：突破单卡显存限制的超长序列并行训练方案

## 进一步阅读

- [Karpathy: Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)：4 小时手把手从零写出并训练 GPT-2
- [LLaMA 3 Herd of Models 技术报告](https://arxiv.org/abs/2407.21783)：详解 15T 语料清洗、退火策略与 405B 分布式训练细节
- [Penedo et al. (2024): The FineWeb Datasets](https://arxiv.org/abs/2406.17557)：HuggingFace 开源的 15T 高质量网页清洗流程与消融实验

## 练习题

1. **Chinchilla 算力推算**：假设手头有 8 张 H100（按 45% MFU 估算有效算力），计划连续训练 2 周，请按 Chinchilla 最优比例算出大概适合训练多大参数的模型、需要准备多少亿 Token 的数据。
2. **动手复现 GPT-2**：使用 nanoGPT 在 OpenWebText 子集上训练一个 124M 模型，记录训练 Loss 曲线与验证集 Perplexity，并对比开启与关闭学习率 Warmup 的差异。
3. **数据配比消融**：在一个小型语言模型训练中，把 5% 的普通网页语料替换成高质量代码（如 StarCoder 数据）或数学推导语料（如 OpenWebMath），观察模型在 GSM8K 或代码补全上的变化。

---

[← 上一章](02-architecture.md) | [目录](../README.md) | [下一章 →](04-post-training.md)
