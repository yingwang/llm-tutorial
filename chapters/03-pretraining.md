[← 上一章](02-architecture.md) | [目录](../README.md) | [下一章 →](04-post-training.md)

# 第三章：预训练 (Pretraining)

## 3.1 训练目标

### 3.1.1 因果语言建模 (Causal Language Modeling, CLM)

**自回归生成基石**：自回归因果语言建模构成了现代大语言模型预训练的核心优化目标。其数学形式定义为最大化前向序列的联合条件似然概率：

```
给定离散 Token 序列 x = (x_1, x_2, ..., x_n)
损失函数 Loss = -Σ_{t=1}^n log P(x_t | x_1, ..., x_{t-1}; θ)
```

**因果语言建模的核心动力学**：
- **压缩即表征**：信息论视角下，准确预测下一个 Token 要求模型隐式构建对世界运行规律、句法结构、语义逻辑与多步推理的高维压缩流形；
- **纯粹的自监督范式**：无需任何人工标注，海量自然语料即天然的监督信号源；
- **标度律的可预测性**：简单的自回归目标结合模型规模与计算量的持续扩展，能够催生跨领域的通用泛化与少样本学习能力。

### 3.1.2 中间填充 (Fill-in-the-Middle, FIM)

在标准因果自回归基础上赋予模型双向局部补全能力，对代码补全与文档编辑场景至关重要 ([Bavarian et al., 2022](https://arxiv.org/abs/2207.14255))：

```
原始文本序列: A B C D E
变换为 PSM 模式: <fim_prefix> A B <fim_suffix> D E <fim_middle> C

# 机制：将中间缺失片段搬移至末尾，使因果自回归模型学会在已知前后文的前提下补全空缺
```

- **工业实践**：GPT-4、[StarCoder](https://arxiv.org/abs/2305.06161) 与 [CodeLlama](https://arxiv.org/abs/2308.12950) 等代码模型普遍在预训练中按 50% 比例混入 FIM 变换数据，其余保留标准 CLM；
- **推理零开销**：无需更改底层网络架构，仅通过数据表示变换即实现端到端局部插入补全。

### 3.1.3 掩码语言建模 (Masked Language Modeling, MLM)

[BERT](https://arxiv.org/abs/1810.04805) 范式：随机对输入序列中 15% 的 Token 进行掩码，依托全连接双向注意力重构被掩盖的真实词：
```
输入: The [MASK] sat on the [MASK]
目标: cat, mat
```
- **特性**：具备天然的双向上下文感知能力，在判别式理解任务（语义检索、向量表示、命名实体识别）中表现优异；
- **局限**：缺乏因果自回归生成的自洽性，难以直接扩展至长程开放式生成，现代生成式基座已全面转向 CLM。

## 3.2 数据工程：大模型的知识基座

预训练数据的质量与多样性直接决定了基座模型的能力上限与安全边界。

### 3.2.1 多源语料生态

| 语料范畴 | 规模体量 | 质量密度 | 典型代表数据集 |
|---------|---------|---------|--------------|
| 网络抓取 (Web Crawl) | PB 级 | 异质低密 | [Common Crawl](https://commoncrawl.org/), [C4](https://huggingface.co/datasets/allenai/c4), FineWeb |
| 百科与结构化知识 | TB 级 | 高密严谨 | [Wikipedia](https://dumps.wikimedia.org/), Wikidata |
| 专业书籍 | TB 级 | 连贯深层 | Books3, [Project Gutenberg](https://www.gutenberg.org/) |
| 开源代码 | TB 级 | 强逻辑性 | GitHub, [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| 学术论文与期刊 | TB 级 | 高维严谨 | [arXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [S2ORC](https://github.com/allenai/s2orc) |
| 论坛问答与社群对话 | TB 级 | 多样交互 | Reddit, StackOverflow |
| 形式化数学与推理 | GB 级 | 极致严密 | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [ProofPile](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |

### 3.2.2 工业级数据清洗流水线

```
原始 Web 网页抓取 (Raw Web Crawl)
    │
    ▼
① URL/域名过滤 (屏蔽成人、低质农场、黑产与合规违规站点)
    │
    ▼
② 语言判别与语系分流 (fastText lid.176.bin)
    │
    ▼
③ 正文提取与噪声剥离 (Trafilatura, Resiliparse)
    │  - 剥离 HTML 模板、广告脚本与导航栏冗余
    │  - 提取主体正文并保留段落层级
    │
    ▼
④ 多维度质量过滤 (Quality Filtering)
    │  - 启发式规则过滤 (剔除异常长度、字符重复率过高、标点异常文本)
    │  - 困惑度过滤 (KenLM N-gram 语言模型，剔除 Perplexity 异常的乱码)
    │  - 神经分类器过滤 (训练 fastText / BERT 分类器对齐高质量文本分布)
    │
    ▼
⑤ 层次化去重 (Deduplication)
    │  - 精确去重 (SHA-256 哈希匹配行/段级重复)
    │  - 模糊去重 (MinHash + LSH，在 Jaccard 相似度阈值 > 0.8 实施聚类剔除)
    │  - 跨文档全局去重 (后缀数组 Suffix Array 去除长公共子串)
    │
    ▼
⑥ 隐私敏感信息脱敏 (PII Masking)
    │  - 正则模式识别脱敏邮箱、电话、身份证与 IP 地址
    │  - NER 实体识别脱敏姓名与敏感地理坐标
    │
    ▼
⑦ Tokenization 编码与二进制固化 (写入 Memory-mapped 文件，如 Arrow / Binary Shards)
```

> 工业级核心工具：[HuggingFace datatrove](https://github.com/huggingface/datatrove)（大规模分布式数据清洗框架）与 [dolma](https://github.com/allenai/dolma)（Allen AI 数据工具链）。

### 3.2.3 数据混合配比 (Data Mixture)

**全局知识拓扑配比**：针对多源异质数据确定最优采样比例，直接影响模型在不同下游维度的能力平衡。

LLaMA 3 的典型配比结构：
```
英文通用网页语料: ~50%
多语言网页语料: ~15%
高质量源代码: ~17%
形式化数学与科技文献: ~5%
高质专业书籍: ~5%
百科与权威知识库: ~5%
高质量对话交互: ~3%
```

**配比搜索与优化方法**：
1. **代理模型消融法**：利用 1B 参数规模的小型代理模型，在不同配比下训练并评测下游基准，以低成本拟合泛化曲面；
2. **分布稳健优化 (DoReMi)** ([Xie et al., 2023](https://arxiv.org/abs/2305.10429))：利用小模型在各领域的超额损失自适应计算最优重采样权重；
3. **能力经验规律**：提高代码占比可显著激发因果推理与指令逻辑；提升数学占比强化符号推演；注入适量对话语料改善对话对齐。

### 3.2.4 课程学习与退火策略 (Curriculum & Annealing)

**动态数据课程调度**：预训练并非静态混合全量语料，而是依据优化动力学实施渐进式阶段调度：

1. **基础泛化阶段 (Major Training)**：输入广阔多样的通用网页语料，建立宽广的语言规律与常识底座；
2. **能力强化阶段 (Mid-to-Late Phase)**：上调高质量代码、数学与学术文献的采样权重，强化逻辑推理深度；
3. **学习率退火阶段 (Annealing)**：
   - 在训练末期（最后 2%–5% Token），将高质量合成数据与权威知识库比例大幅提升至 50% 以上；
   - 学习率配合余弦衰减逼近零，快速将模型参数沉淀收敛于高泛化低熵低谷。

### 3.2.5 合成数据范式 (Synthetic Data)

**前沿发展方向**：利用高能力前沿模型（如 GPT-4）作为数据引擎，构建高结构化、高信息密度的合成语料，弥补自然文本中高质量知识与推理步骤的短缺。

| 数据范畴 | 生成与构建方法 | 核心应用阶段 |
|---------|--------------|------------|
| 复杂指令与回答 | 强模型生成多轮 QA 与复杂约束提示词 | SFT / RLHF |
| 算法与可执行代码 | 模型生成候选代码 + 沙盒单元测试执行过滤 | 代码预训练与代码微调 |
| 形式化数学与思维链 | 模型生成细粒度推导步骤 + Lean/Python 求解器验证 | 数学逻辑与推理强化 |
| 教科书式优质读物 | ["Textbooks Are All You Need"](https://arxiv.org/abs/2306.11644) (Phi) | 极小参数模型预训练 |
| 网页文本重写与提炼 | 强模型将口语化/杂乱网页重写为结构化专业论述 | 预训练质量提纯 |

**[Phi 系列](https://arxiv.org/abs/2404.14219) 启示**：微软 Phi-3 证实，依托极致清洗的高质量合成语料，3.8B 模型能够在常识与推理评测上比肩传统海量自然语料训练的 8B 基座。

## 3.3 训练过程与优化动力学

### 3.3.1 优化器选型与状态显存分析

**AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) 构成了现代大模型预训练的通用基准：
```
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t          # 一阶动量 (Momentum)
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t²         # 二阶矩 (自适应方差估计)
m̂_t = m_t / (1 - β_1^t)                         # 偏差修正
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + wd * θ_{t-1})  # 显式权重衰减解耦
```

**典型超参数配置**：
- $\beta_1 = 0.9, \beta_2 = 0.95$（LLaMA/DeepSeek 配置，较标准 0.999 具备更强抗噪性）；
- $\epsilon = 1\times 10^{-8}$，$\text{weight\_decay} = 0.1$；
- 峰值学习率：$3\times 10^{-4}$（1B–7B 规模）至 $1.5\times 10^{-4}$（70B+ 规模）。

**优化器显存占用分析**：每个模型参数在 FP32 Master Weight、FP32 一阶动量 $m$ 与 FP32 二阶矩 $v$ 下共需消耗 12 字节显存。对于 70B 模型，仅优化器状态即需占用约 840 GB 显存。

**前沿替代优化器探索**：
- **[Adafactor](https://arxiv.org/abs/1804.04235)**：通过行/列低秩矩阵分解近似二阶矩，优化器显存消耗减半，曾应用于 T5；
- **[LION](https://arxiv.org/abs/2302.06675)**：利用符号函数 $\text{sign}(\text{momentum})$ 指导更新，仅需维护一阶动量，显存更省；
- **[Sophia](https://arxiv.org/abs/2305.14342)**：引入对角 Hessian 矩阵估计曲率，具备更敏捷的逃离鞍点能力；
- **[MUON](https://arxiv.org/abs/2502.16982)**：在矩阵参数空间利用动量的极分解（Polar Decomposition）生成正交更新方向，显著提升超大矩阵运算的收敛效率。

### 3.3.2 学习率调度策略

**标准余弦衰减 (Linear Warmup + Cosine Decay)**：
```
# 预热阶段: 前 2000 步线性爬升至峰值
if step < warmup_steps:
    lr = peak_lr * step / warmup_steps

# 余弦衰减阶段: 平滑下行至终点最小值 (通常为峰值的 10%)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
```

**WSD 调度策略 (Warmup-Stable-Decay)** ([MiniCPM](https://arxiv.org/abs/2404.06395))：
```
Warmup (预热) → Stable (长周期恒定学习率) → Decay (急剧退火)
```
- **核心优势**：在 Stable 阶段任意时刻均可低成本派生（Branch out）退火分支，评估当前状态能力；
- **工程价值**：大幅简化持续预训练（Continual Pretraining）与领域扩展的流程。

### 3.3.3 训练稳定性与异常处理

**Loss 突刺 (Loss Spike)**：预训练过程中损失值突然异常飙升，为千卡集群训练中最常见的风险。

**常见成因与防御方案**：
| 异常诱因 | 典型表征 | 工程防御对策 |
|---------|---------|------------|
| 梯度爆炸 | 梯度范数突增，损失变为 NaN/Inf | 梯度全局截断（Gradient Clipping，阈值设为 1.0） |
| 低质恶性数据 | 单个 Batch 后损失骤增但可缓慢恢复 | 动态跳过损失超过阈值的异常异常 Batch |
| 学习率超标 | 损失呈现周期性高频振荡 | 调低峰值学习率或延长 Warmup 周期 |
| 注意力数值发散 | QK 点积极大导致 Softmax 溢出 | 引入 QK-Norm 或 Logit 软截断机制 |
| BF16 精度下溢/溢出 | Embedding 层或顶层 Head 梯度数值越界 | 将 Embedding 与 Logits 计算提升至 FP32 精度 |
| 偏置漂移 | 预测 Logits 绝对值整体失控飘高 | 引入 z-loss 辅助正则：$\mathcal{L}_z = \alpha \log^2(\sum \exp(\text{logits}))$ |

### 3.3.4 批次规模与临界批次策略

```
全局有效 Batch Size = 微批次 (micro_batch_size) × 梯度累积步数 (grad_accum) × 数据并行规模 (DP_size)
```

**典型工业配置**：
- LLaMA 2: ~4M Tokens/Batch
- LLaMA 3 405B: ~16M Tokens/Batch

**临界批次规模 (Critical Batch Size)** ([McCandlish et al., 2018](https://arxiv.org/abs/1812.06162))：在临界批次以下，增大 Batch Size 能带来几乎线性的硬件吞吐加速；一旦超越临界批次，样本信息冗余导致优化效率边际收益递减。

### 3.3.5 混合精度计算体系

| 精度格式 | 位宽构成 | 动态范围 | 工业典型用途 |
|---------|---------|---------|-------------|
| FP32 | 1 位符号 + 8 位指数 + 23 位尾数 | $\pm 3.4\times 10^{38}$ | 优化器状态主权重更新、损失缩放 |
| TF32 | 1 位符号 + 8 位指数 + 10 位尾数 | $\pm 3.4\times 10^{38}$ | NVIDIA Ampere 架构默认 GEMM 运算 |
| BF16 | 1 位符号 + 8 位指数 + 7 位尾数 | $\pm 3.4\times 10^{38}$ | **主流基座预训练默认精度**（防溢出能力同 FP32） |
| FP16 | 1 位符号 + 5 位指数 + 10 位尾数 | $\pm 65504$ | 早期训练格式（依赖复杂的 Loss Scaling 防止下溢） |
| FP8 | E4M3 (4指数3尾数) / E5M2 (5指数2尾数) | 视格式而定 | Hopper/Blackwell 架构高吞吐矩阵乘法 |

**FP8 混合精度训练前沿**（DeepSeek-V3 实践）：
- 前向计算采用 E4M3 保证尾数精度；
- 反向传播采用 E5M2 保证梯度动态范围；
- 引入细粒度逐张量/分块缩放因子（Per-tensor / Block Scaling），成功完成 671B 规模 MoE 预训练。

## 3.4 长序列训练演进

### 3.4.1 两阶段短到长训练范式

1. **第一阶段 (标准预训练)**：在 4K/8K 窗口下吞吐 90% 以上的数据，低成本夯实通用语法与知识表征；
2. **第二阶段 (长上下文持续训练)**：通过 RoPE 基频调优将窗口扩展至 32K–128K，仅需消耗约 1%–5% 的总 Token 预算即可完成长距离注意力对齐。

### 3.4.2 序列并行与环形注意力 (Ring Attention)

([Liu et al., 2023](https://arxiv.org/abs/2310.01889)) 当单序列超长超出单卡显存承载时，将序列切分至多张 GPU，通过环形拓扑流动传递 KV 分块：

```
GPU 0: 持有序列分块 0–32K   → 局部 Attention 计算
GPU 1: 持有序列分块 32K–64K → 局部 Attention 计算
GPU 2: 持有序列分块 64K–96K → 局部 Attention 计算
GPU 3: 持有序列分块 96K–128K→ 局部 Attention 计算
       ↻ KV Blocks 沿设备环异步重叠传输与计算
```

## 关键论文

- [Brown et al. (2020) — GPT-3](https://arxiv.org/abs/2005.14165): Few-Shot 上下文学习奠基之作
- [Hoffmann et al. (2022) — Chinchilla](https://arxiv.org/abs/2203.15556): 计算最优标度律
- [Touvron et al. (2023) — LLaMA 2](https://arxiv.org/abs/2307.09288): 开源大模型工程实现细节
- [DeepSeek-AI (2024) — DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): MoE 细粒度路由与全流程 FP8 训练
- [Liu et al. (2023) — Ring Attention](https://arxiv.org/abs/2310.01889): 跨节点超长序列分布式注意力

## 进阶参考

- Karpathy: [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)（4 小时手写从零训练 GPT-2 完整实战）
- [LLaMA 3 技术报告](https://arxiv.org/abs/2407.21783)（405B 万卡规模预训练全景拆解）
- EleutherAI: [The Pile 论文](https://arxiv.org/abs/2101.00027)（大规模多源预训练语料构建典范）

## 实践训练

1. **算力与标度律估算**：给定 8 卡 H100 训练 2 周的算力预算，依据 Chinchilla 计算最优法则推导最优参数量与训练 Token 规划。
2. **微型 GPT 训练与 Loss 分析**：基于精简语料训练 124M 模型，绘制完整 Loss 曲线与 Perplexity 走势，观察余弦退火阶段的学习动态。
3. **领域数据消融评测**：在预训练混合语料中将 5% 通用语料替换为高质量形式化数学语料，对比验证基座在下游 GSM8K 上的少样本推演收益。

---

[← 上一章](02-architecture.md) | [目录](../README.md) | [下一章 →](04-post-training.md)
