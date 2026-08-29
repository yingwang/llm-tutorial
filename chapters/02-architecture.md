[← 上一章](01-tokenizer.md) | [目录](../README.md) | [下一章 →](03-pretraining.md)

# 第二章：模型架构

## 2.1 Transformer 核心机制

Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) 构筑了现代大语言模型的计算骨干。其核心组件与设计机理如下：

### 2.1.1 自注意力机制 (Self-Attention)

**数学机理**：自注意力机制为序列中的每个 Token 动态计算其对全序列上下文的关联权重，实现基于内容寻址的信息路由。

**计算流程**：
```
输入: X ∈ ℝ^(n×d)  # n 个 token, d 维特征

Q = X @ W_Q  # Query  ∈ ℝ^(n×d_k)
K = X @ W_K  # Key    ∈ ℝ^(n×d_k)
V = X @ W_V  # Value  ∈ ℝ^(n×d_v)

Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**缩放因子 $\sqrt{d_k}$ 的物理意义**：当维度 $d_k$ 较大时，点积结果的方差随维度线性增长。若不引入缩放因子，内积绝对值将急剧增大，使 Softmax 函数落入梯度极小的饱和区，引发梯度消失。

**多头注意力 (Multi-Head Attention, MHA)**：
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

多头机制允许模型将特征投影至多个正交子空间，并行捕捉语法依存、长程指代与局部搭配等异构关联模式。

### 2.1.2 前馈网络 (Feed-Forward Network)

在每个 Transformer 模块中，注意力层后均紧跟一个逐位置前馈网络（Position-wise FFN）：
```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

- **经典 Transformer**：采用 ReLU 激活函数；
- **GPT-2/3**：采用平滑近似的 GELU 激活函数；
- **现代大模型 (LLaMA, Gemma 等)**：普遍采用 SwiGLU 变体 ([Shazeer, 2020](https://arxiv.org/abs/2002.05202))：
```
SwiGLU(x) = (x @ W_1 · σ(x @ W_gate)) @ W_2
```
SwiGLU 引入门控线性机制，增强非线性特征筛选能力。尽管增加了约 50% 的线性变换参数，但在收敛速率与下游泛化表现上均展现出明显优势。

### 2.1.3 归一化层 (Layer Normalization)

**归一化拓扑演进**：
- **Post-Norm**（原始 Transformer）：`x + LayerNorm(SubLayer(x))`。主干残差路径上的方差随层数累积，易引发深层梯度爆炸或消失，高度依赖严苛的学习率预热（Warmup）；
- **Pre-Norm**（GPT-2+，现代 LLM 标配）：`x + SubLayer(LayerNorm(x))`。梯度流可直接经由残差分支无阻畅通回传，训练稳定性显著提升；
- **RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467))（LLaMA, Gemma）：
  ```
  RMSNorm(x) = x / RMS(x) * γ
  RMS(x) = √(mean(x²))
  ```
  摒弃均值中心化（Zero-centering）操作，仅保留均方根缩放，在削减约 7% 归一化计算开销的同时保持了完全相当的表征性能。

### 2.1.4 架构选型：Decoder-Only 为何成为主流

| 架构 | 代表模型 | 训练目标 | 核心优势与适用场景 |
|------|---------|---------|------------------|
| Encoder-only | BERT, RoBERTa | Masked LM (掩码重构) | 双向上下文感知，适用于文本分类、NER 与密集检索 |
| Encoder-Decoder | T5, BART | Seq2Seq (序列到序列重构) | 结构清晰，适用于机器翻译、抽象式摘要 |
| **Decoder-only** | **GPT, LLaMA, Claude** | **Causal LM (因果自回归)** | **自回归连续生成、上下文少样本学习与多任务泛化** |

**Decoder-only 胜出的核心动因**：
1. **标度律扩展优势**：在同等计算预算与数据规模下，自回归 Decoder 展现出更平滑、优异的泛化前沿（Pareto Frontier）；
2. **统一的因果监督目标**：以统一的 Next-Token Prediction 充分利用序列中每一个 Token 的监督信号，训练效率远高于低掩码率的 MLM；
3. **上下文学习（In-Context Learning）天然涌现**：单向因果流与 Prompt 交互范式高度契合，无需微调即可激发极强的少样本推理能力；
4. **统一生成与理解**：将所有自然语言任务均建模为条件文本生成问题，消除了架构割裂。

## 2.2 位置编码 (Positional Encoding)

自注意力算子具备天然的置换不变性（Permutation Invariance），必须显式或隐式引入位置信息以感知序列的时序拓扑。

### 2.2.1 绝对位置编码

**正弦绝对位置编码** (原始 Transformer)：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- 优势：无需额外可学习参数，理论上涵盖任意位置；
- 劣势：缺乏对相对距离的直接建模，长序列外推能力有限。

**可学习绝对位置编码** (GPT-1/2)：
- 为每个绝对位置分配独立的可学习嵌入向量 `P[i] ∈ ℝ^d`，直接叠加至 Token Embedding；
- 劣势：强绑定于预设的最大序列长度，完全无法直接外推超出训练长度的上下文。

### 2.2.2 旋转位置编码 (RoPE)

**现代大模型的基准方案** ([Su et al., 2021](https://arxiv.org/abs/2104.09864))。LLaMA, Qwen, Mistral, Gemma 等前沿模型普遍采用。

**核心思想**：不直接在 Embedding 空间叠加位置向量，而是在自注意力计算时通过正交旋转矩阵对 Query 和 Key 实施坐标变换：

```
# 将 d 维向量视作 d/2 个二维子平面的直和
# 在每个子平面上依据绝对位置 m 旋转对应角度 θ_i

R(pos, i) = [[cos(pos·θ_i), -sin(pos·θ_i)],
              [sin(pos·θ_i),  cos(pos·θ_i)]]

θ_i = 10000^(-2i/d)

q_rotated = R(m) @ q   # 位置 m
k_rotated = R(n) @ k   # 位置 n

# 核心数学性质: q_rotated^T @ k_rotated = q^T @ R(n - m) @ k
# 内积结果严格仅取决于相对位移 (m - n)
```

**理论与工程优势**：
- **相对位置天然内嵌**：通过复数旋转将绝对坐标映射为内积中的相对距离；
- **长程注意力自然衰减**：随着相对距离拉大，内积期望值自然衰减，契合语言局部性先验；
- **灵活的长度外推与扩展性**：可通过调整基频 $\theta$ 实现平滑的上下文窗口扩展。

### 2.2.3 长上下文扩展技术

受限于二次方计算与显存开销，基座预训练阶段序列长度通常设定在 2K–8K。在后训练或微调阶段扩展至 128K 乃至 1M 上下文时，需借助位置编码外推与插值技术：

**NTK-aware 插值** ([Reddit/bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), Code LLaMA 风格)：
```python
# 避免直接线性压缩位置序号导致近邻高频信息丢失，
# 而是依据神经切切核（NTK）理论非线性缩放 RoPE 的基频常数：
base_new = base_old * (scale_factor ** (d / (d - 2)))
# d 为 head 维度, scale_factor = 目标扩展长度 / 原始训练长度
```

**YaRN** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071))：
- 将 RoPE 的频率维度划分为三组：高频波段（保持原频以保真局部细节）、中频波段（插值过渡）、低频波段（纯 NTK 缩放）；
- 引入温度因子修正长序列下的注意力分布熵；
- LLaMA 3.1 的 128K 上下文窗口即深度借鉴了 YaRN 的波段分治思想。

**ALiBi** ([Press et al., 2022](https://arxiv.org/abs/2108.12409))：
- 完全摒弃显式位置向量，在注意力得分矩阵上直接施加线性距离惩罚：`score -= m * |i - j|`；
- 各注意力头赋予预设的几何级数斜率 $m$；
- 具备优异的短训长推特性，曾应用于 BLOOM 与 MPT。

## 2.3 现代架构改进

### 2.3.1 分组查询注意力 (Grouped-Query Attention, GQA)

([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))

**痛点分析**：标准 MHA 中每个 Query 头均拥有独立的 Key 和 Value 头，使得自回归推理期的 KV Cache 显存占用随并发批次和序列长度线性膨胀，构成主要显存墙。

**拓扑演进**：
```
MHA:  h 个 Q 头, h 个 K 头, h 个 V 头  (如 h=32)
MQA:  h 个 Q 头, 1 个 K 头, 1 个 V 头  (极端单头共享)
GQA:  h 个 Q 头, g 个 K 头, g 个 V 头  (如 h=32, g=8，4:1 分组共享)
```

- **折中设计**：GQA 在 MHA 的表达容量与 MQA 的极致轻量之间取得优异平衡；
- **显存压缩**：KV Cache 占用直接压缩 $h/g$ 倍（如 32/8 = 4 倍），大幅提升单卡推理并发上限；
- **工业采纳**：LLaMA 2 70B、LLaMA 3、Mistral、Gemma 2 等已全线将 GQA 作为标准配置。

```python
# GQA 计算逻辑示意
def grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads):
    # Q: [batch, seq, num_q_heads, head_dim]
    # K, V: [batch, seq, num_kv_heads, head_dim]
    
    group_size = num_q_heads // num_kv_heads  # 例如 32 // 8 = 4
    
    # 将 K, V 在 head 维度上重复广播以对齐 Q 的头数
    K = K.repeat_interleave(group_size, dim=2)
    V = V.repeat_interleave(group_size, dim=2)
    
    return standard_attention(Q, K, V)
```

### 2.3.2 混合专家模型 (Mixture of Experts, MoE)

**核心机理**：将密集 FFN 层拆解为多个参数对称的专家网络（Expert FFN），每个 Token 由门控路由网络动态分派至 Top-$k$ 个专家进行前向激活。([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538))

```
# 标准密集 FFN:
output = FFN(x)

# MoE 稀疏路由:
gate_scores = softmax(x @ W_gate)  # [n_experts]
top_k_experts = topk(gate_scores, k=2)

output = Σ(gate_scores[i] * Expert_i(x))  for i in top_k_experts
```

**核心优势**：在总参数量呈倍数级扩展的同时，保持恒定的单 Token 计算复杂度（FLOPs）。例如 [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) 具备 47B 总参数，但每次前向仅激活约 13B 参数。

**关键挑战与解决方案**：
- **负载均衡（Load Balancing）**：引入辅助损失（Auxiliary Loss）防止路由网络陷入马太效应（所有 Token 涌向少数专家）：
  ```
  L_balance = α * n_experts * Σ(f_i * P_i)
  # f_i: 路由至专家 i 的 Token 比例
  # P_i: 专家 i 的平均门控概率
  ```
- **专家并行通信开销**：当专家分布于不同 GPU 节点时，Token 分发与聚合需要全对全（All-to-All）通信；
- **专家坍缩防范**：防止部分专家在训练早期梯度停滞而失去表达能力。

**前沿 MoE 演进（DeepSeek 范式）**：
- **细粒度专家分割 (DeepSeekMoE)**：将传统 8 个大专家切分为 160 个细粒度小专家，每次激活 Top-6，提升专家知识解耦精度；
- **共享专家隔离 (Shared Expert)**：设置固定激活的共享专家承载通用世界知识，专用专家专注于细分领域，提升路由特化效率；
- **无辅助损失负载均衡**：通过动态偏置项（Bias Term）调控专家被选概率，避免辅助损失对语言建模主目标的潜在干扰。

### 2.3.3 多头潜变量注意力 (Multi-head Latent Attention, MLA)

[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 首创，并在 DeepSeek-V3/R1 中延续。

**设计痛点**：GQA 在极低 KV 头数下仍会损失多头多样性，而在长上下文场景下显存占用依然显著。

**机理突破**：通过低秩投影将 Key 和 Value 联合压缩至低维潜空间（Latent Space），在推理期仅需缓存极低维度的潜变量，解码时再按需投影解压：

```
# 训练与投影:
c_kv = x @ W_DKV           # 压缩: d_model → d_latent (如 4096 → 512)
K = c_kv @ W_UK             # 上投影解压 Key
V = c_kv @ W_UV             # 上投影解压 Value

# 推理期 KV Cache 仅存储低维压缩张量 c_kv (512 维)，而非全量 K + V (4096 × 2 维)
```

MLA 将推理期 KV Cache 压缩至传统 MHA 的约 1/16，且在模型表达容量上完全比肩全量 MHA。

### 2.3.4 滑动窗口注意力 (Sliding Window Attention)

[Mistral 7B](https://arxiv.org/abs/2310.06825) 引入：在不同网络层交替配置局部滑动窗口注意力与全局注意力。

```
# 滑动窗口机制: 每个 Token 仅对局部窗口 W 内的前序 Token 进行注意力交互
# 借助深层网络的层叠效应，信息可在纵深方向跨层扩散
# L 层堆叠后的理论感受野上限 = L * W

# 以 Mistral 7B 为例: W = 4096, 32 层堆叠 → 理论感受野达 131,072
```

**工程价值**：浅层聚焦局部句法特征，深层整合全局语义，在大幅缩减注意力二次方显存与计算开销的同时维持卓越的长文建模能力。

### 2.3.5 现代网络结构细节优化

**注意力与 FFN 并行化** (GPT-J, [PaLM](https://arxiv.org/abs/2204.02311))：
```python
# 传统串行计算路径:
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# 并行前向路径 (可将 Attention 与 FFN 的 GEMM 算子融合同步分发):
x = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))
```

**QK-Norm**：在计算注意力得分前对 Query 和 Key 显式实施 LayerNorm/RMSNorm，抑制深层注意力 Logits 异常发散，为 [Gemma 2](https://arxiv.org/abs/2408.00118) 与 Command R+ 等模型所采纳。

**Logit 软截断 (Logit Soft-Capping)** ([Gemma 2](https://arxiv.org/abs/2408.00118))：`logits = soft_cap * tanh(logits / soft_cap)`，限制注意力和最终分类 Logits 的数值边界，提升极大模型训练稳定性。

## 2.4 模型标度律与参数规划

### 标度律 (Chinchilla Scaling Laws)

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) 系统揭示了计算最优预算分配准则：

```
在给定计算预算 C (FLOPs) 的约束下:
最优参数规模 N ∝ C^0.5
最优训练 Token 数 D ∝ C^0.5
即: 模型参数量与训练数据量应保持等比例协同增长

经验基准:
最优训练 Token 数 ≈ 20 × 模型参数量
```

| 参数规模 | Chinchilla 最优 Token 预算 | 典型工业级代表模型 |
|---------|--------------------------|------------------|
| 1B | 20B | [TinyLlama](https://github.com/jzhang38/TinyLlama) |
| 7B | 140B | LLaMA 2 7B (2T，超额训练) |
| 13B | 260B | LLaMA 2 13B |
| 70B | 1.4T | LLaMA 2 70B |

**推理经济学驱动的超额训练（Over-training）**：在实际工业落地中，前沿基座普遍大幅超越 Chinchilla 最优数据量。由于模型训练是一次性固定成本，而线上推理是长期边际成本，“较小参数量 + 极充沛语料”的方案能够大幅压缩后续高并发部署时的单 Token 吞吐成本。例如 LLaMA 3 8B 在 15T Token 上进行了超额训练。

### 主流模型架构参数对照

| 模型 | 参数量 | 层数 | 隐层维度 | 注意力头数 | KV 头数 | 词表规模 | 上下文窗口 |
|------|--------|------|---------|-----------|--------|---------|-----------|
| [LLaMA 2 7B](https://arxiv.org/abs/2307.09288) | 6.7B | 32 | 4096 | 32 | 32 (MHA) | 32K | 4K |
| [LLaMA 3 8B](https://arxiv.org/abs/2407.21783) | 8.0B | 32 | 4096 | 32 | 8 (GQA) | 128K | 128K |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 7.3B | 32 | 4096 | 32 | 8 (GQA) | 32K | 32K |
| [Qwen2 72B](https://arxiv.org/abs/2407.10671) | 72.7B | 80 | 8192 | 64 | 8 (GQA) | 152K | 128K |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B | 61 | 7168 | 128 | MLA | 128K | 128K |

## 关键论文

- [Vaswani et al. (2017) — Attention Is All You Need](https://arxiv.org/abs/1706.03762): Transformer 经典奠基论文
- [Su et al. (2021) — RoFormer / RoPE](https://arxiv.org/abs/2104.09864): 现代主流旋转位置编码
- [Ainslie et al. (2023) — GQA](https://arxiv.org/abs/2305.13245): 面向大模型推理显存优化的分组查询注意力
- [Fedus et al. (2021) — Switch Transformer](https://arxiv.org/abs/2101.03961): 稀疏混合专家（MoE）现代化范式
- [DeepSeek-AI (2024) — DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434): 低秩潜变量注意力机制

## 进阶参考

- Karpathy: [nanoGPT](https://github.com/karpathy/nanoGPT)（约 300 行 PyTorch 极简 GPT 复现）
- Harvard NLP: [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)（逐行注解 Transformer 原论文）
- Lilian Weng: [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)

## 实践训练

1. **手写注意力算子**：从 NumPy 或原生 PyTorch 出发，实现 Scaled Dot-Product Attention，并逐步扩展为 Multi-Head Attention。
2. **MHA 至 GQA 架构改造**：在微型 GPT 代码库中将 MHA 模块改造为 GQA（例如设置 KV 头数为 Query 头数的 1/4），对比验证 KV Cache 的显存压缩收益。
3. **位置编码外推特性对比**：在小型 Transformer 模型上，对比正弦绝对位置编码、ALiBi 与 RoPE 在序列长度外推测试（训练 512，外推评估 1024）中的困惑度（Perplexity）表现。

---

[← 上一章](01-tokenizer.md) | [目录](../README.md) | [下一章 →](03-pretraining.md)
