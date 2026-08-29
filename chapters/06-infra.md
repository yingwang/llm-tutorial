[← 上一章](05-peft.md) | [目录](../README.md) | [下一章 →](07-inference.md)

---

# 第六章：训练基础设施 (Infra)

超大规模语言模型的预训练与微调已演进为顶尖的高性能计算与分布式系统工程。从千卡互联拓扑、张量与流水线切分，到底层通信算子重叠与容错调度，系统级基础设施直接决定了模型训练的可扩展性上限与算力利用效率。

## 6.1 硬件与集群网络拓扑

### 6.1.1 计算芯片选型与关键物理指标

| 计算卡型号 | 显存规格与类型 | BF16 算力 (TFLOPS) | FP8 算力 (TFLOPS) | 互联拓扑与带宽 | 典型应用定位 |
|-----------|--------------|-------------------|-------------------|---------------|-------------|
| NVIDIA A100 | 80GB HBM2e | 312 | - | NVLink 3.0 (600 GB/s) | 经典基座预训练与微调主力 |
| NVIDIA H100 | 80GB HBM3 | 989 | 1979 | NVLink 4.0 (900 GB/s) | 现代大模型千卡/万卡预训练标准配置 |
| NVIDIA H200 | 141GB HBM3e | 989 | 1979 | NVLink 4.0 (900 GB/s) | 超长序列与超大 MoE 显存密集型场景 |
| NVIDIA B200 | 192GB HBM3e | 2250 | 4500 | NVLink 5.0 (1800 GB/s) | 下一代万亿参数混合架构 |
| AMD MI300X | 192GB HBM3 | 1307 | 2614 | Infinity Fabric (896 GB/s) | 高显存容量开源替代路线 |

**核心性能瓶颈分析**：
- **显存容量（Memory Capacity）**：硬性约束单卡承载的模型参数、优化器状态与中间激活峰值；
- **显存带宽（Memory Bandwidth）**：决定自回归解码等访存受限（Memory-bound）算子的吞吐上限；
- **张量算力（Compute TFLOPS）**：决定高维 GEMM 矩阵乘法等计算受限（Compute-bound）环节的耗时；
- **跨卡互联带宽（Interconnect Bandwidth）**：制约多维分布式并行时的通信同步延迟与集群扩展效率。

### 6.1.2 层次化网络互联体系

```
节点机内互联 (Intra-Node):
  GPU 0 ←── NVLink (900 GB/s 双向) ──→ GPU 1..7 (8 卡全连接 Mesh 拓扑)

跨节点网络互联 (Inter-Node):
  Node 0 ←── InfiniBand NDR 400Gb/s (约 50 GB/s) ──→ Node 1
  
物理带宽鸿沟:
  机内 NVLink 带宽约为跨机 InfiniBand 带宽的 18 倍。
  这一物理约束奠定了分布式切分的基本原则：将通信高密度的张量切分收敛于节点机内，跨节点仅执行通信量较小的流水线或数据并行。
```

**大规模万卡集群网络拓扑**：
- 采用无阻塞胖树（Fat-Tree）或蜻蜓（Dragonfly+）网络拓扑；
- 依赖 NCCL 集合通信库结合网络拓扑感知进行通道编排（Channel Search）；
- 引入 RoCEv2 或 InfiniBand 自适应路由（Adaptive Routing）与硬件遥测拥塞控制，规避网络丢包与长尾拖慢。

### 6.1.3 高性能存储与 Checkpoint 流水线

```
训练数据读取流水线:
  大规模异构清洗语料 → 分布式并行文件系统 (Lustre, GPFS, WekaFS) / 对象存储 (S3)
  → 本地高速 NVMe SSD 内存映射缓存 (Mmap I/O)

Checkpoint 检查点写入:
  70B 级别模型单个全量 Checkpoint 达数百 GB
  → 采用异步持久化机制：前向主进程仅将权重快照非阻塞复制至 Host 内存，后台守护线程异步写入分布式存储；
  → 采用分布式分片写入（Distributed Checkpointing），各 GPU 仅持久化本地分片，避免单点 I/O 汇聚阻塞。
```

## 6.2 分布式训练并行策略

### 6.2.1 数据并行 (Data Parallelism, DP)

**经典数据并行**：每个计算节点维护一份完整的模型参数与优化器状态副本，各节点并行吞吐异构的数据微批次，并在反向传播完成后通过集合通信原语（AllReduce）同步梯度均值。

```
GPU 0: 持有全量模型 + 数据微批次 0 → 计算梯度 g_0
GPU 1: 持有全量模型 + 数据微批次 1 → 计算梯度 g_1
GPU 2: 持有全量模型 + 数据微批次 2 → 计算梯度 g_2
GPU 3: 持有全量模型 + 数据微批次 3 → 计算梯度 g_3
       ↓ 集合通信 AllReduce: 计算全局平均梯度 g_avg = Σ(g_i) / 4 ↓
各 GPU 基于 g_avg 独立执行优化器步进更新
```

**显存瓶颈**：随着模型参数增长，单卡无法容纳模型本身与优化器状态（以 BF16 训练 70B 模型为例，参数需 140GB，AdamW 优化器状态需 840GB，远超单卡显存）。

### 6.2.2 零冗余优化器 (ZeRO) 与 FSDP

([Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)) **[DeepSpeed](https://github.com/microsoft/DeepSpeed) 的核心洞见**：经典数据并行存在极高的显存冗余，每个计算节点均重复持有全量模型参数、梯度与优化器状态。ZeRO（Zero Redundancy Optimizer）通过在数据并行维度渐进式切分状态张量，打破单卡显存墙：

```
ZeRO-Stage 1 (优化器状态切分):
  将 FP32 优化器状态 (一阶动量、二阶矩、Master Weights) 均匀分片至各卡，显存占用降低至原来的 1/4，通信开销保持不变。

ZeRO-Stage 2 (优化器状态 + 梯度切分):
  各卡仅保留本地分片参数对应的梯度张量，反向传播时采用 Reduce-Scatter 同步，显存占用降低至原来的 1/8。

ZeRO-Stage 3 (完全参数分片 / FSDP):
  模型参数亦按卡分片存储。在前向计算某一层时通过 All-Gather 动态拉取全量权重，计算完毕立即释放；反向传播重复此动态收集与释放过程。
```

**PyTorch 原生完全分片数据并行 (FSDP)**：
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 对应 ZeRO-3 级别的完全分片
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    auto_wrap_policy=transformer_auto_wrap_policy,
)
```

**FSDP2 现代化升级**：在 PyTorch 2.x 中引入逐参数（Per-parameter）细粒度分片，显著优化与张量并行的组合兼容性，原生支持 FP8 高性能混合精度。

### 6.2.3 张量并行 (Tensor Parallelism, TP)

([Shoeybi et al., 2019 — Megatron-LM](https://arxiv.org/abs/1909.08053))

**算子内权重矩阵切分**：将单个线性层或注意力矩阵切分到节点内的多张 GPU 并行计算：

```
以前馈网络两层 MLP 为例: Y = GELU(X @ W_1) @ W_2
将 W_1 按列切分 (Column Parallel): W_1 = [W_11, W_12]
将 W_2 按行切分 (Row Parallel):    W_2 = [W_21; W_22]

计算路径:
  1. GPU 0 计算: H_0 = GELU(X @ W_11)
     GPU 1 计算: H_1 = GELU(X @ W_12)  (无需跨卡通信)
  2. GPU 0 计算局部投影: Y_0 = H_0 @ W_21
     GPU 1 计算局部投影: Y_1 = H_1 @ W_22
  3. 执行 AllReduce: Y = Y_0 + Y_1 (完成单层前向合并)
```

- **注意力层切分**：$W_Q, W_K, W_V$ 沿 Head 维度按列切分，$W_O$ 沿 Head 维度按行切分；
- **通信约束**：每个 Transformer 模块需在 Attention 和 FFN 各执行一次 AllReduce（前向 2 次，反向 2 次），通信频次极高，通常严格限制在单机 8 卡 NVLink 高速域内运行。

### 6.2.4 流水线并行 (Pipeline Parallelism, PP)

([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473))

**层间纵向切分**：将模型的不同层序列分发至不同节点，数据按微批次（Micro-batch）在网络层间流水传递。

```
Stage 0 (GPU 0): 执行第 0–7 层
Stage 1 (GPU 1): 执行第 8–15 层
Stage 2 (GPU 2): 执行第 16–23 层
Stage 3 (GPU 3): 执行第 24–31 层
```

**流水线气泡（Bubble）消除机制**：
- **GPipe 调度** ([Huang et al., 2019](https://arxiv.org/abs/1811.06965))：将全局批次拆解为 $m$ 个微批次，但反向传播集中在末尾，气泡率相对较高；
- **1F1B 调度 (One-Forward-One-Backward)**：在稳定阶段前向与反向交替推进，及时释放浅层激活显存，将显存占用与批次深度解耦；
- **交错式 1F1B (Interleaved 1F1B)**：每张 GPU 持有多个非连续的虚构阶段（Virtual Stages，如 GPU 0 持有 Layer 0–3 与 Layer 16–19），进一步将气泡率压缩至数分之一；
- **零气泡流水线 (Zero-Bubble PP)** ([Qi et al., 2024](https://arxiv.org/abs/2401.10241))：将反向传播解耦为计算输入的梯度与计算权重的梯度，通过动态填补实现极低气泡率。

### 6.2.5 序列并行与上下文并行 (Context Parallelism, CP)

针对超长上下文场景，当单序列前向激活超出单卡显存时，将 Token 序列切分至多张 GPU：
- **Megatron 序列并行 (Sequence Parallelism)**：在 TP 内部将 LayerNorm 和 Dropout 沿序列维度切分，消除 TP 中的激活冗余；
- **环形上下文并行 (Ring Attention)**：利用 GPU 环形网络流动传递 Key/Value 分块，实现近乎无限长序列的线性扩展。

### 6.2.6 专家并行 (Expert Parallelism, EP)

针对 MoE 架构的特化并行范式：将不同的专家模块部署于独立的 GPU 设备：
- **Token 动态路由**：门控网络计算 Token 与专家的匹配概率后，通过全对全通信（All-to-All Dispatch）将 Token 张量路由至对应专家卡；
- **局部前向与聚合**：各专家卡完成计算后，再次执行 All-to-All Combine 将特征张量汇总回原始设备。

### 6.2.7 多维混合并行设计实战

在万卡级工程中，需结合硬件拓扑对多种并行策略进行多维正交组合：

```
DeepSeek-V3 预训练并行架构 (2048 块 H800 GPU 集群):
  - TP = 1: 摒弃跨卡张量并行，规避密集的 AllReduce 通信；
  - PP = 16: 跨节点配置 16 级流水线；
  - DP = 128: 结合 ZeRO-1 进行数据并行；
  - EP = 64: 跨 64 张卡配置细粒度专家并行；
  - 核心优势: 将核心计算收敛为高吞吐 GEMM，通信主要由双向重叠的 All-to-All 承担。

LLaMA 3 405B 预训练并行架构 (16384 块 H100 GPU 集群):
  - TP = 8: 严格限制于机内 8 卡 NVLink 域；
  - PP = 16: 跨机柜构建 16 阶段流水线；
  - DP = 128: 全局数据并行；
  - CP: 依据序列长度动态接入上下文并行；
  - 总并行维度: 8 (TP) × 16 (PP) × 128 (DP) = 16,384 GPU。
```

## 6.3 训练框架与高效算子加速

### 6.3.1 主流分布式训练框架生态

| 框架名称 | 主导机构 | 核心技术特色 | 适用场景 |
|---------|---------|------------|---------|
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA | 极致优化的 3D/4D 混合并行与底层 CUDA 算子 | 百亿至万亿级参数超大集群预训练 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft | 完备的 ZeRO-1/2/3 体系、Offload 与 MoE 支持 | 十亿至千亿级模型预训练与微调 |
| [FSDP](https://pytorch.org/docs/stable/fsdp.html) | Meta / PyTorch | 原生深度集成、轻量简洁的完全分片数据并行 | 中大规模集群与标准化工业训练 |
| [Colossal-AI](https://github.com/hpcaitech/ColossalAI) | HPC-AI Tech | 多维异构并行与多任务编排优化 | 通用分布式加速与混合部署 |
| [NanoGPT](https://github.com/karpathy/nanoGPT) | Karpathy | 极简原生 DDP 实现，结构清晰直观 | 教学演练与微型模型算法验证 |

### 6.3.2 显存与计算加速核心技巧

**激活值重计算 (Gradient Checkpointing)** ([Chen et al., 2016](https://arxiv.org/abs/1604.06174))：
- 前向传播阶段丢弃大部分中间层的激活值张量，仅保留关键边界节点；
- 反向传播时根据需要动态局部重新计算前向激活；
- 以约 30%–33% 的额外计算开销为代价，将激活显存从 $\mathcal{O}(L)$ 降低至 $\mathcal{O}(\sqrt{L})$。

**闪速注意力 (FlashAttention)** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135))：
- **IO 感知计算**：传统注意力需将 $N \times N$ 的中间注意力矩阵物化写入高延迟的 HBM 显存；
- FlashAttention 通过分块分治（Tiling）结合在线 Softmax（Online Softmax）算法，在片上高速 SRAM 中一次性完成注意力计算并直接写出结果；
- 显存复杂度降低至 $\mathcal{O}(N)$，端到端吞吐加速 2–4 倍。

**通信与计算异步重叠 (Overlap)**：
- 在反向传播中，某层的梯度一旦计算完毕，即刻异步触发跨卡 AllReduce / ReduceScatter 通信；
- 在通信进行的同时，GPU 继续执行前序层的反向计算，实现通信开销的近乎完全隐藏。

## 6.4 容错机制与硬件利用率评估

### 6.4.1 万卡集群容错体系

在大规模集群训练中，硬件故障是确定性常态事件：
- 万卡规模下，平均每数小时即可能发生 GPU 显存单比特翻转（ECC Error）、光纤网卡丢包或节点掉线；
- **秒级故障诊断与热迁移**：实时监控网络拓扑与心跳，自动隔离故障节点并利用温备节点快速顶替；
- **分级快速 Checkpoint**：将中间状态以 5–10 分钟为周期高频写入本地 NVMe Ramdisk，一旦发生中断可在数分钟内无缝回退恢复。

### 6.4.2 模型算力利用率 (MFU)

**衡量分布式训练效率的黄金指标**：
$$\text{MFU} = \frac{\text{每次迭代实际完成的理论浮点计算量 (FLOPs)}}{\text{集群 GPU 理论峰值算力} \times \text{单步实际耗时}}$$

**工业级 MFU 基准分布**：
- 单机 8 卡环境：50%–60%
- 中等规模集群 (64–256 GPU)：45%–55%
- 超大规模集群 (1K–16K GPU)：35%–45%（LLaMA 3 405B 在 16K 卡集群上达成 38%–43% 的卓越 MFU）

## 关键论文

- [Rajbhandari et al. (2019) — ZeRO](https://arxiv.org/abs/1910.02054): 零冗余优化器内存切分奠基之作
- [Shoeybi et al. (2019) — Megatron-LM](https://arxiv.org/abs/1909.08053): Transformer 张量并行与流水线设计
- [Huang et al. (2018) — GPipe](https://arxiv.org/abs/1811.06965): 经典流水线并行架构
- [Dao et al. (2022) — FlashAttention](https://arxiv.org/abs/2205.14135): 基于 IO 内存层次优化的极速注意力算子
- [Dao (2023) — FlashAttention-2](https://arxiv.org/abs/2307.08691): 极致并行度与线程块工作划分优化

## 进阶参考

- NVIDIA: [Megatron-LM 官方代码库](https://github.com/NVIDIA/Megatron-LM)（前沿 3D 并行工业级参考实现）
- Microsoft: [DeepSpeed 官方文档](https://www.deepspeed.ai/)（ZeRO 体系最佳实践）
- Lilian Weng: [How to Train Really Large Models](https://lilianweng.github.io/posts/2021-09-25-train-large/)（系统级大模型训练全景剖析）

## 实践训练

1. **显存占用理论推导**：针对 70B 模型（BF16 精度，AdamW 优化器，批次 1M Tokens），分别推导在经典 DP、ZeRO-1、ZeRO-2 与 ZeRO-3 架构下的单卡显存占用构成。
2. **PyTorch Profiler 性能瓶颈分析**：使用 PyTorch Profiler 分析单步 Transformer 前向与反向传播的耗时热点，量化开启 FlashAttention 与算子融合后的访存等待改善。
3. **混合并行拓扑规划**：给定 64 块 H100 计算节点，设计训练 70B 密集基座的最佳 (DP, TP, PP) 组合参数，并从物理网络带宽角度阐述为何张量并行（TP）必须收敛于单机内部。

---

[← 上一章](05-peft.md) | [目录](../README.md) | [下一章 →](07-inference.md)
