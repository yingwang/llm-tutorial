[← 上一章](05-peft.md) | [目录](../README.md) | [下一章 →](07-inference.md)

# 第六章：训练基础设施 (Infra)

训练百亿到万亿参数的大模型，本质上是一项分布式系统工程。单张 GPU 的显存和算力远远不够，必须把模型和数据切分到成百上千张卡上协同计算。怎么选硬件、怎么连网络、怎么切分模型与优化器状态、怎么在硬件频繁故障时保住训练进度，直接决定了训练能否跑通以及算力利用率（MFU）有多高。

## 6.1 硬件与集群网络

### 6.1.1 GPU 选型

| GPU | 显存 | BF16 算力 (TFLOPS) | FP8 算力 (TFLOPS) | 互连带宽 | 典型定位 |
|-----|------|-------------------|------------------|----------|----------|
| A100 80GB | 80GB HBM2e | 312 | - | NVLink 600GB/s | 2022–2024 主流训练卡 |
| H100 SXM | 80GB HBM3 | 989 | 1979 | NVLink 900GB/s | 当前大规模预训练主力 |
| H200 | 141GB HBM3e | 989 | 1979 | NVLink 900GB/s | 长序列与超大模型（大显存） |
| B200 | 192GB HBM3e | 2250 | 4500 | NVLink 1800GB/s | 下一代训练与推理主力 |
| MI300X (AMD) | 192GB HBM3 | 1307 | 2614 | Infinity Fabric 896GB/s | 大显存 AMD 替代方案 |

选卡时最常看四个硬件指标：
- **显存容量**：决定单卡能装下多大的模型权重、优化器状态和中间激活值；
- **显存带宽**：决定推理生成和逐元素算子的速度（Decode 阶段主要受显存带宽限制，即 Memory-bound）；
- **算力（TFLOPS）**：决定大矩阵乘法的速度（训练和 Prefill 阶段主要受算力限制，即 Compute-bound）；
- **互连带宽**：决定多卡之间同步梯度、交换激活值的速度，直接限制分布式并行的扩展效率。

### 6.1.2 集群网络

单台服务器内部通常装有 8 张 GPU，通过 NVSwitch 全连接；跨机器之间则通过 InfiniBand（IB）或 RoCE 高速网卡互连。

```
单机内 (Intra-Node):
  GPU ←── NVLink 900GB/s ──→ GPU    (8 卡全连接)

跨机 (Inter-Node):
  Node ←── InfiniBand 400Gb/s ──→ Node
  
  400Gb IB ≈ 50GB/s (单向)
  NVLink   ≈ 900GB/s
  
  跨机通信带宽比机内慢约 18 倍！
  → 并行切分策略必须尽量把高频通信留在单机内，减少跨机通信
```

**大规模集群拓扑**（如 16K GPU）：
```
GPU (8) → Node → Leaf switch (32 nodes) → Spine switch → Fat-tree / Dragonfly
```

在大集群里跑训练，只要有一个慢节点（Straggler）或者一条链路丢包，所有卡都会在集合通信时停下来等它。因此工程上需要配合三件事：
- [NCCL](https://github.com/NVIDIA/nccl) 通信参数与通道调优；
- 拓扑感知的进程调度，把需要频繁通信的进程尽量分在同一台机器或同一个交换机下；
- 自动化容错机制，及时发现掉线或降速的坏节点并用备用节点顶替。

### 6.1.3 存储系统

```
训练数据读取:
  → 分布式文件系统 (Lustre, GPFS, WekaFS)
  → 或对象存储 (S3) + 计算节点本地 NVMe 缓存

Checkpoint 写入:
  → 70B 模型单个全量 checkpoint 约 500GB–1TB
  → 每 1000 步保存一次，一天会产生几十 TB 写入量
  → 必须采用异步 checkpoint（先拷贝到内存，后台线程慢慢写盘，不阻塞 GPU 训练）
  → 配合分布式 checkpoint 存储，每张卡只写自己负责的分片
```

## 6.2 分布式训练策略

### 6.2.1 数据并行 (Data Parallelism, DP)

最直接的并行方式：每张卡上都放一份完整的模型副本，不同卡喂不同的数据批次（Micro-batch），各自独立跑完前向和反向传播后，用一次 `AllReduce` 把所有卡算出的梯度求平均，再用平均梯度同步更新各自的模型权重。

```
GPU 0: Model copy + Data batch 0 → gradient_0
GPU 1: Model copy + Data batch 1 → gradient_1
GPU 2: Model copy + Data batch 2 → gradient_2
GPU 3: Model copy + Data batch 3 → gradient_3
          ↓ AllReduce: average gradients ↓
GPU 0-3: 用平均梯度同步更新模型参数
```

**局限**：每张卡都得装下整套模型参数、梯度和优化器状态。按照第三章算过的显存账，混合精度（BF16 + AdamW）训练每 1B 参数大约要占 16GB 显存。一个 7B 模型光权重、梯度和优化器状态就要占约 112GB（若不计 FP32 主权重副本也超过 70GB），单张 80GB 显卡已经捉襟见肘；到了 70B 模型，单卡更是完全装不下。

### 6.2.2 ZeRO (Zero Redundancy Optimizer) 与 FSDP

（[Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)）既然朴素数据并行里每张卡存的模型状态一模一样，为什么不把这些冗余状态拆开，让每张卡只存其中的 $1/N$？这就是 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 提出的 ZeRO 核心思路：

```
ZeRO Stage 1: 只切分优化器状态 (Optimizer States)
  → 静态显存降为原来的约 1/4，通信量与普通 DP 相同

ZeRO Stage 2: 切分优化器状态 + 梯度 (Gradients)
  → 静态显存降为原来的约 1/8，反向传播用 Reduce-Scatter 代替 AllReduce

ZeRO Stage 3: 切分优化器状态 + 梯度 + 模型参数 (Parameters)
  → 静态显存降为原来的 1/N (N 为 GPU 数量)
  → 前向和反向算到某一层时，临时用 AllGather 从各卡收集该层完整权重，算完立刻释放
  → 相当于 PyTorch 的 FSDP (Fully Sharded Data Parallel)
```

**PyTorch FSDP (Fully Sharded Data Parallel)**：
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 对应 ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    auto_wrap_policy=transformer_auto_wrap_policy,
)
```

**FSDP2**（PyTorch 新版）：
- 改用基于 `DTensor` 的逐参数（per-parameter）细粒度分片，内存管理更规整；
- 更容易和张量并行（TP）、上下文并行（CP）组合使用；
- 原生支持 FP8 混合精度训练。

### 6.2.3 张量并行 (Tensor Parallelism, TP)

（[Shoeybi et al., 2019 — Megatron-LM](https://arxiv.org/abs/1909.08053)）

当单层权重本身就很大，或者需要连同中间激活值一起切分时，张量并行直接把**同一层内部的矩阵乘法**拆到多张卡上算：

```
# 线性层 Y = X @ W (W ∈ ℝ^{d×d})
# 把权重矩阵 W 按列切分到 2 张卡 (Column Parallel):

GPU 0: Y_0 = X @ W_0   (W_0 = W[:, :d/2])
GPU 1: Y_1 = X @ W_1   (W_1 = W[:, d/2:])

# 再通过 AllGather 拼回完整输出，或直接接下一个按行切分的线性层做 AllReduce
```

**Megatron-LM 风格的张量并行**：
- **Self-Attention**：把 $W_Q, W_K, W_V$ 按注意力头（head）维度做列切分，每张卡独立算一部分头；最后的输出投影矩阵 $W_O$ 做行切分，算完用一次 `AllReduce` 把结果加起来。
- **FFN**：第一个线性层（或 SwiGLU 的 `gate_proj` 与 `up_proj`）按列切分，中间激活函数各自在本地算，第二个线性层（`down_proj`）按行切分，最后再用一次 `AllReduce` 汇总。
- **通信代价与适用范围**：每一层 Transformer 在前向和反向中各需要 2 次 `AllReduce`，通信极其频繁，而且在切分点上必须同步等待。因此 TP 通常只在一台机器内部开（如 `TP=8`，走 900GB/s 的 NVLink），极少跨机器使用。

### 6.2.4 流水线并行 (Pipeline Parallelism, PP)

（[Narayanan et al., 2021](https://arxiv.org/abs/2104.04473)）

既然跨机器带宽窄、不能开 TP，对于装不进单机的超大模型，可以把模型**按层切分**到不同机器上，像工厂流水线一样接力处理：

```
GPU 0: Layers 0-7    (Stage 0)
GPU 1: Layers 8-15   (Stage 1)
GPU 2: Layers 16-23  (Stage 2)
GPU 3: Layers 24-31  (Stage 3)

数据从 Stage 0 → 1 → 2 → 3 依次传递，每台机器只传层间边界上的激活值张量
```

**核心问题**：朴素的流水线并行会产生大量“气泡”（Bubble，即下游卡干等上游卡算完的空闲时间）。

**消除气泡的调度改进**：
- **GPipe**（[Huang et al., 2019](https://arxiv.org/abs/1811.06965)）：把一个大 batch 拆成多个小 micro-batch 连续送入流水线，把空闲气泡摊薄；但所有 micro-batch 的前向激活都要攒到反向传播时才能释放，显存占用高。
- **1F1B（One-Forward-One-Backward）**：流水线跑满后，每张卡严格按照“算一个 micro-batch 的前向、紧接着算一个 micro-batch 的反向”交替执行，算完反向立刻释放对应的前向激活，大幅省下激活显存：
  ```
  Stage 0: F0 F1 F2 F3 B0 B1 B2 B3   (朴素 GPipe，激活全攒着)
  Stage 0: F0 F1 F2 F3 B0 F4 B1 F5   (1F1B 交替调度，气泡同而显存省)
  ```
- **Interleaved 1F1B（交错式流水线）**：每张卡不再只管连续的几层，而是分管多段不连续的层（例如 GPU 0 负责第 0–3 层和第 16–19 层），让数据在卡间多转几圈，把气泡比例再压低数倍。
- **Zero-Bubble PP**（[Qi et al., 2024](https://arxiv.org/abs/2401.10241)）：把反向传播进一步拆成“对输入求导（$B$，后续 stage 急需）”和“对本层权重求导（$W$，只需在更新前算完即可）”两步，拿 $W$ 去填补流水线空隙，将气泡压到接近 0。

### 6.2.5 上下文并行 (Context Parallelism, CP)

当训练序列拉长到 128K 甚至 1M token 时，哪怕模型参数切得再细，单条长序列的中间激活值也会超出单卡显存。上下文并行（CP）直接沿着**序列长度维度**把输入切分到多张卡上：

```
序列长度 128K，切分到 4 张卡:
GPU 0: tokens 0-32K
GPU 1: tokens 32K-64K
GPU 2: tokens 64K-96K
GPU 3: tokens 96K-128K

线性层和 FFN 各算各的 token 即可；
唯独 Attention 需要跨卡看全局上下文，通常采用 Ring Attention 计算:
- 每张卡先用本地的 Q 和本地的 K, V 算注意力分块；
- 一边算，一边把本地的 K, V 通过环形网络传给下一张卡，同时接收上一张卡传来的 K, V；
- 循环传递 N-1 轮，直到每张卡的 Q 都和全部 token 的 K, V 算完注意力，通信开销被计算过程遮盖。
```

### 6.2.6 专家并行 (Expert Parallelism, EP)

**MoE 模型专用**：MoE 层里有几十到几百个路由专家（Routed Experts），专家并行把不同的专家放到不同的 GPU 上：

```
GPU 0: Expert 0, 1
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

Token 路由与计算流程:
1. 门控网络 (Gate) 算出每个 token 应该去哪几个专家 → 触发第一次 All-to-All 通信分发 token；
2. 每张卡用本地的专家权重计算收到的 token；
3. 算完后再触发第二次 All-to-All 通信，把结果送回 token 原先所在的卡。
```

**通信瓶颈**：All-to-All 通信量与 `token 数量 × hidden_size × top_k` 成正比，对跨卡和跨机网络带宽要求很高，通常需要专门设计算子让通信与计算重叠（如 DeepSeek-V3 的 DualPipe）。

### 6.2.7 3D / 4D / 5D 混合并行实战

实际训练千亿级大模型时，会把上面几种并行方式按硬件特性组合起来：

```
DeepSeek-V3 (2048 张 H800 GPU):
  TP = 1   (不开张量并行，靠 MLA 和细粒度 MoE 省显存，省掉高频 TP 通信)
  PP = 16  (16 级双向流水线 DualPipe)
  EP = 64  (跨 64 张卡做专家并行)
  DP = 128 (配合 ZeRO-1 切分优化器状态)
  
  2048 = 16 (PP) × 128 (DP)

LLaMA 3 405B (16384 张 H100 GPU):
  TP = 8   (限制在单机 8 卡 NVLink 内)
  PP = 16  (跨机流水线)
  CP = 1 或按长上下文阶段开启
  DP = 128 (配合 FSDP / ZeRO)
  
  16384 = 8 (TP) × 16 (PP) × 128 (DP)
```

## 6.3 训练框架与高效算子

### 6.3.1 框架对比

| 框架 | 维护团队 | 支持特性 | 适用规模 |
|------|----------|----------|----------|
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA | TP + PP + DP + CP + EP，深度 CUDA 优化 | 百亿至万亿参数预训练 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft | ZeRO-1/2/3、Offload、PP、MoE | 十亿至千亿参数预训练与微调 |
| [FSDP](https://pytorch.org/docs/stable/fsdp.html) (PyTorch) | Meta | 原生 ZeRO-3 与混合并行，代码侵入小 | 十亿至百亿参数主力方案 |
| [ColossalAI](https://github.com/hpcaitech/ColossalAI) | HPC-AI Tech | 多种并行策略与异构内存管理 | 十亿至百亿参数训练 |
| [Levanter](https://github.com/stanford-crfm/levanter) | Stanford | 基于 JAX 与 Haliax 命名张量，可复现性好 | 学术研究与中等规模预训练 |
| [NanoGPT](https://github.com/karpathy/nanoGPT) | Karpathy | 极简 PyTorch DDP 实现 | 教学入门与小模型实验 |

### 6.3.2 Megatron-LM 核心配置

```python
# Megatron-LM 的 3D 并行典型配置
# 启动命令: torchrun --nproc_per_node=8 --nnodes=64

args = {
    "tensor_model_parallel_size": 8,     # TP: 单机 8 卡内切分
    "pipeline_model_parallel_size": 16,   # PP: 跨机 16 级流水线
    "data_parallel_size": 4,              # DP: 总卡数 512 / (8 × 16) = 4
    "sequence_parallel": True,            # 序列并行 (配合 TP 切分 LayerNorm/Dropout 激活)
    "use_flash_attn": True,
    "bf16": True,
    "micro_batch_size": 1,
    "global_batch_size": 1024,
}
```

### 6.3.3 高效训练技巧

**激活检查点（Gradient Checkpointing / Activation Recomputation）**（[Chen et al., 2016](https://arxiv.org/abs/1604.06174)）：
```
常规做法: 前向传播保存每一层的所有中间激活值 → 反向传播直接取用求导
问题: 激活值显存占用极大，与 batch_size × seq_len × hidden_dim × layers 成正比

Gradient Checkpointing: 前向时只保存每层入口（或部分关键节点）的激活，其余算完即丢；
反向传播算到该层时，临时从保存点重跑一次局部前向，把丢掉的激活重新算出来
  → 激活显存占用降低到原来的约 1/√L (L 为层数)
  → 代价是多花约 33% 的计算时间

选择性重计算 (Selective Checkpointing): 只丢弃并重算显存体积最大但计算量小的部分（如 Attention 内部激活），保住大矩阵乘法结果
```

**Flash Attention**（[Dao et al., 2022](https://arxiv.org/abs/2205.14135)）：
```
标准 Attention:
  S = Q @ K^T          → 计算量 O(n²d)，先把 n×n 矩阵写回 HBM 显存
  P = softmax(S)       → 再从 HBM 读出 S，算完写回 n×n 的 P
  O = P @ V            → 再从 HBM 读出 P，乘上 V

Flash Attention (Tri Dao):
  绝不在 HBM 显存里完整实例化 n×n 的注意力矩阵
  利用分块 (Tiling) + 在线 Softmax (Online Softmax) 递推更新公式，
  把 Q, K, V 分块加载到 GPU 片上高速 SRAM 中一次性算完输出 O
  → 注意力显存占用从 O(n²) 降为 O(n)
  → 端到端速度提升 2–4 倍（省掉了最耗时的 HBM 反复读写）
  
  Flash Attention 2: 调整线程块循环顺序与并行维度，进一步榨干 GPU 利用率
  Flash Attention 3: 针对 Hopper (H100) 架构异步指令与 FP8 专门优化
```

> 代码实现：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

**算子编译与融合（Compiled / Fused Kernels）**：
```python
# 使用 torch.compile 自动融合逐元素算子，减少显存读写往返
model = torch.compile(model, mode="max-autotune")

# 大模型训练中常手动替换为融合算子 (Fused Kernels) 的环节:
# - Fused Attention (FlashAttention)
# - Fused RMSNorm / LayerNorm + Residual + Dropout
# - Fused SwiGLU
# - Fused RoPE (旋转位置编码原地计算)
# - Fused Cross-Entropy (把 lm_head 投影与交叉熵合并分块算，省掉巨型 logits 显存)
```

**通信与计算重叠（Communication-Computation Overlap）**：
```
不重叠 (串行等待):
  Backward_layer_N → AllReduce_layer_N → Backward_layer_{N-1} → AllReduce_layer_{N-1} ...

重叠 (异步并行):
  Backward_layer_N 算完立刻在后台流发起异步 AllReduce_layer_N，
  GPU 计算流无需等待通信完成，直接往下算 Backward_layer_{N-1}
  → 用算梯度的时间把跨卡传输梯度的延迟藏起来
```

## 6.4 容错与训练效率

### 6.4.1 Checkpoint 保存策略

```python
# 1. 异步 Checkpoint (不阻塞训练主进程)
def async_save_checkpoint(model, optimizer, step):
    # 先非阻塞地把权重快照从 GPU 显存复制到 CPU 内存
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    # 交给后台线程慢慢写入磁盘或对象存储，GPU 继续跑下一步
    threading.Thread(target=torch.save, args=(state, f"ckpt_{step}.pt")).start()

# 2. 分布式 Checkpoint (每张卡只存自己负责的 shard，避免单节点汇总时内存不足)
from torch.distributed.checkpoint import save, load, FileSystemWriter
save(model.state_dict(), storage_writer=FileSystemWriter(path))
```

### 6.4.2 大集群故障恢复

在千卡、万卡集群上训练大模型，硬件故障不是偶发意外，而是家常便饭：
- 16K 张 GPU 同时满载跑，平均每 2–3 小时就会遇上一次硬件异常；
- 常见故障包括 HBM 显存多位 ECC 错误、光模块与交换机抖动、节点掉线、电源或液冷异常。

**工程应对方案（以 DeepSeek-V3 / LLaMA 3 为例）**：
- 每隔 5–10 分钟做一次快速快照，先把状态写到本地高速 NVMe 或内存盘（Ramdisk），再异步同步到远端存储；
- 监控系统实时检测心跳与通信超时，发现坏节点后自动将其隔离，从备用池拉起热备节点顶上；
- 训练作业自动从几分钟前的最近一次快照重启，每次故障只损失几分钟的训练进度。

### 6.4.3 MFU (Model FLOPs Utilization)

衡量分布式训练系统到底把硬件压榨到了什么程度，最核心的指标是 **MFU（模型算力利用率）**：

$$\text{MFU} = \frac{\text{模型单步实际所需的理论浮点运算量 (FLOPs)}}{\text{集群 GPU 标称峰值算力} \times \text{单步实际耗时}}$$

```
经验基准:
  单机 8 卡:          50% - 60%
  中小集群 (64-256卡): 40% - 50%
  千卡集群 (1K+ 卡):   35% - 45%
  万卡集群 (16K 卡):   35% - 43% (如 LLaMA 3 405B 报告 MFU 为 38%–43%)
```

**拉低 MFU 的主要因素**：
- 跨卡与跨机通信等待（未能被计算完全遮盖的 AllReduce、AllGather、All-to-All）；
- 流水线并行中的空闲气泡（Bubble）；
- 数据加载或预处理跟不上 GPU 吞吐；
- 细碎算子过多导致频繁读写显存；
- 激活检查点（Gradient Checkpointing）带来的约 33% 额外重计算开销（计算 MFU 分子时只算单次前向+反向的净理论计算量，不把重计算算进有效产出）。

## 关键论文

- [Rajbhandari et al. (2020) — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)：消除数据并行显存冗余的 ZeRO-1/2/3（DeepSpeed 核心）
- [Shoeybi et al. (2019) — Megatron-LM](https://arxiv.org/abs/1909.08053)：Transformer 张量并行与大规模分布式训练经典设计
- [Huang et al. (2019) — GPipe](https://arxiv.org/abs/1811.06965) 与 [Narayanan et al. (2021) — Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)：流水线并行与 1F1B 交错调度
- [Dao et al. (2022) — FlashAttention](https://arxiv.org/abs/2205.14135)：利用 SRAM 分块与在线 Softmax 消除 $O(n^2)$ 显存瓶颈，提速 2–4 倍
- [Dao (2023) — FlashAttention-2](https://arxiv.org/abs/2307.08691)：并行度与工作划分优化

## 进一步阅读

- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)：3D/4D 混合并行工业级参考实现
- [DeepSpeed 文档](https://www.deepspeed.ai/)：ZeRO-1/2/3 与 Offload 配置指南
- [How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/) — Lilian Weng 的分布式并行综述

## 练习题

1. **显存预算推导**：估算训练一个 70B 模型所需的总显存（BF16 混合精度，AdamW 优化器，全局 batch = 1M tokens），并分别算出在 64 张 80GB H100 上开启普通 DP、ZeRO-1、ZeRO-2、ZeRO-3 时每张卡的静态显存占用。
2. **训练性能剖析**：用 PyTorch Profiler 跑一个 step 的 nanoGPT 训练，找出耗时最长的算子，对比开启 `torch.compile` 和 FlashAttention 前后的单步耗时与显存峰值。
3. **3D 并行切分设计**：给定 8 台 8 卡 H100 服务器（共 64 张卡，机内 NVLink 900GB/s，跨机 IB 50GB/s）训练 70B 稠密模型，你会怎么设置 `(TP, PP, DP)`？请从通信量和带宽差异解释为什么 TP 几乎总是留在单机内。

---

[← 上一章](05-peft.md) | [目录](../README.md) | [下一章 →](07-inference.md)
