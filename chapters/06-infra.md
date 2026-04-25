[← 上一章](05-peft.md) | [目录](../README.md) | [下一章 →](07-inference.md)

---

# 第六章：训练基础设施 (Infra)

## 6.1 硬件

### 6.1.1 GPU 选型

| GPU | 显存 | BF16 TFLOPS | FP8 TFLOPS | 互连 | 用途 |
|-----|------|-------------|------------|------|------|
| A100 80GB | 80GB HBM2e | 312 | - | NVLink 600GB/s | 主流训练 (2022-2024) |
| H100 SXM | 80GB HBM3 | 989 | 1979 | NVLink 900GB/s | 当前主力 |
| H200 | 141GB HBM3e | 989 | 1979 | NVLink 900GB/s | 大模型 (更大显存) |
| B200 | 192GB HBM3e | 2250 | 4500 | NVLink 1800GB/s | 下一代主力 |
| MI300X (AMD) | 192GB HBM3 | 1307 | 2614 | Infinity Fabric | AMD 替代方案 |

**关键指标**:
- **显存容量**: 决定能放多大的模型
- **显存带宽**: 决定推理速度（推理是 memory-bound）
- **算力 (TFLOPS)**: 决定训练速度（训练是 compute-bound）
- **互连带宽**: 决定多卡并行效率

### 6.1.2 集群网络

```
单机内:
  GPU ←NVLink 900GB/s→ GPU    (8卡全连接)

跨机:
  Node ←InfiniBand 400Gb/s→ Node
  
  400Gb IB ≈ 50GB/s (每个方向)
  NVLink ≈ 900GB/s
  
  所以跨机通信比机内慢 ~18倍！
  → 并行策略必须最小化跨机通信
```

**大集群拓扑** (如 16K GPU):
```
GPU (8) → Node → Leaf switch (32 nodes) → Spine switch → Fat-tree/Dragonfly
```

**网络问题**: 训练大模型时，一个慢节点或丢包就会拖慢整个训练。需要：
- [NCCL](https://github.com/NVIDIA/nccl) 调优
- 网络拓扑感知的进程放置
- 容错机制（自动检测并替换故障节点）

### 6.1.3 存储

```
训练数据读取:
  → 分布式文件系统 (Lustre, GPFS, WekaFS)
  → 或对象存储 (S3) + 本地 NVMe 缓存

Checkpoint 写入:
  → 70B 模型一个 checkpoint ~500GB
  → 每 1000 步存一次 → 每天几十TB
  → 异步 checkpoint (不阻塞训练)
  → Nebula/分布式 checkpoint 存储
```

## 6.2 分布式训练策略

### 6.2.1 数据并行 (Data Parallelism, DP)

**最简单的并行方式**: 每张卡存完整模型副本，不同卡处理不同数据。

```
GPU 0: Model copy + Data batch 0 → gradient_0
GPU 1: Model copy + Data batch 1 → gradient_1
GPU 2: Model copy + Data batch 2 → gradient_2
GPU 3: Model copy + Data batch 3 → gradient_3
          ↓ AllReduce: average gradients ↓
GPU 0-3: 用平均梯度更新模型
```

**限制**: 每张卡要放整个模型。7B 模型 (BF16) = 14GB 参数 + 优化器状态 ~56GB → 一张 80GB 卡勉强放下。70B 模型放不下。

### 6.2.2 ZeRO (Zero Redundancy Optimizer)

([Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)) **[DeepSpeed](https://github.com/microsoft/DeepSpeed) 的核心贡献**: 数据并行中的冗余太多——每张卡都存完整模型 + 优化器状态。ZeRO 把它们切分。

```
ZeRO Stage 1: 切分优化器状态
  → 内存减少 4倍

ZeRO Stage 2: 切分优化器状态 + 梯度
  → 内存减少 8倍

ZeRO Stage 3: 切分优化器状态 + 梯度 + 模型参数
  → 内存减少 N倍 (N = GPU数)
  → 相当于 FSDP
```

**PyTorch FSDP (Fully Sharded Data Parallel)**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    auto_wrap_policy=transformer_auto_wrap_policy,
)
```

**FSDP2** (PyTorch 最新):
- 更细粒度的 sharding (per-parameter)
- 更好地配合 tensor parallel
- 支持 FP8

### 6.2.3 张量并行 (Tensor Parallelism, TP)

([Shoeybi et al., 2019 — Megatron-LM](https://arxiv.org/abs/1909.08053))

**把单个层的矩阵乘法切分到多张卡上**:

```
# 线性层 Y = X @ W (W ∈ ℝ^{d×d})
# 把 W 按列切分到 2 张卡:

GPU 0: Y_0 = X @ W_0   (W_0 = W[:, :d/2])
GPU 1: Y_1 = X @ W_1   (W_1 = W[:, d/2:])

# AllGather 或 ReduceScatter 合并结果
```

**Megatron-LM 风格 TP**:
- Self-Attention: Q, K, V 按 head 切分
- FFN: 第一个线性层按列切，第二个按行切
- 每层需要 2 次 AllReduce

**适用场景**: 机内（NVLink 带宽足够），通常 TP=8（一台机器内）

### 6.2.4 流水线并行 (Pipeline Parallelism, PP)

([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473))

**把模型按层切分到不同机器**:

```
GPU 0: Layers 0-7    (Stage 0)
GPU 1: Layers 8-15   (Stage 1)
GPU 2: Layers 16-23  (Stage 2)
GPU 3: Layers 24-31  (Stage 3)

数据从 Stage 0 → 1 → 2 → 3 流水线式处理
```

**问题**: 朴素 PP 有大量"bubble"（GPU 空闲等待）。

**解决方案**:
- **GPipe** ([Huang et al., 2019](https://arxiv.org/abs/1811.06965)): 把 micro-batch 切成更小的 mini-batch，增加流水线并行度
- **1F1B** (one-forward-one-backward): 前向和反向交替调度，减少 bubble
  ```
  Stage 0: F0 F1 F2 F3 B0 B1 B2 B3   (朴素, 大bubble)
  Stage 0: F0 F1 F2 F3 B0 F4 B1 F5   (1F1B, 小bubble)
  ```
- **Interleaved PP**: 每张卡放非连续的层（如 GPU 0 放 layer 0,8,16,24），减少 bubble
- **Zero-bubble PP** ([Qi et al., 2024](https://arxiv.org/abs/2401.10241)): 通过重新调度把 bubble 降到接近 0

### 6.2.5 Context Parallelism (CP)

**长序列并行**: 把序列切分到多张卡，每张卡处理一部分序列。

```
序列长度 128K, 4张卡:
GPU 0: tokens 0-32K
GPU 1: tokens 32K-64K
GPU 2: tokens 64K-96K
GPU 3: tokens 96K-128K

Attention 通过 Ring Attention 计算:
- 每张卡本地计算 QK^T 的一部分
- KV 通过 ring 传递到下一张卡
- 重复直到所有 KV 都遍历过
```

### 6.2.6 Expert Parallelism (EP)

**MoE 模型专用**: 不同 expert 放在不同 GPU 上。

```
GPU 0: Expert 0, 1
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

Token routing:
1. Gate 计算每个 token 要去哪个 expert → All-to-All
2. 每张卡计算自己的 expert → All-to-All
3. 结果送回原来的卡
```

**通信瓶颈**: All-to-All 通信量和 token 数 × hidden_size 成正比。

### 6.2.7 3D/4D/5D 并行组合

实际大模型训练组合多种并行:

```
DeepSeek-V3 (2048 H800 GPUs):
  TP=1 (不用TP，因为用了MLA和MoE)
  PP=16 (16个pipeline stage)
  DP=128 (128路数据并行)
  EP=64 (64路expert并行)
  
  2048 = 16 × 128 = 16 × 2 × 64

LLaMA 3 405B (16384 H100 GPUs):
  TP=8 (机内)
  PP=16 (跨机)
  DP=128 (数据并行)
  CP 用于长上下文训练
  
  16384 = 8 × 16 × 128
```

## 6.3 训练框架

### 6.3.1 框架对比

| 框架 | 公司 | 并行策略 | 适用规模 |
|------|------|---------|---------|
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA | TP+PP+DP, MoE | 百B-万B |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft | ZeRO, PP, MoE | 十B-千B |
| [FSDP](https://pytorch.org/docs/stable/fsdp.html) (PyTorch) | Meta | ZeRO-3 | 十B-百B |
| [ColossalAI](https://github.com/hpcaitech/ColossalAI) | HPC-AI Tech | 多种 | 十B-百B |
| [Levanter](https://github.com/stanford-crfm/levanter) | Stanford | JAX-based | 研究 |
| [NanoGPT](https://github.com/karpathy/nanoGPT) | Karpathy | DDP | 学习/小规模 |

### 6.3.2 Megatron-LM 核心

```python
# Megatron-LM 的 3D 并行配置
# launch: torchrun --nproc_per_node=8 --nnodes=64

args = {
    "tensor_model_parallel_size": 8,     # TP: 机内
    "pipeline_model_parallel_size": 16,   # PP: 跨机
    "data_parallel_size": 32,             # DP: auto = total / TP / PP
    "sequence_parallel": True,            # 序列并行 (和 TP 搭配)
    "use_flash_attn": True,
    "bf16": True,
    "micro_batch_size": 1,
    "global_batch_size": 1024,
}
```

### 6.3.3 高效训练技巧

**Gradient Checkpointing (Activation Recomputation)** ([Chen et al., 2016](https://arxiv.org/abs/1604.06174)):
```
正常: 前向保存所有中间激活 → 反向使用
问题: 激活占内存太多 (正比于 batch_size × seq_len × hidden × layers)

Gradient Checkpointing: 只保存部分层的激活，反向时重算
  → 内存减少 √L 倍 (L = 层数)
  → 计算增加 ~33%

选择性 checkpointing: 只 checkpoint 占内存大的操作 (attention)
```

**Flash Attention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)):
```
标准 Attention:
  S = Q @ K^T          → O(n²d) compute, O(n²) memory
  P = softmax(S)       → 存 n² 矩阵
  O = P @ V            

Flash Attention (Tri Dao):
  不 materialize n² attention matrix
  用 tiling + online softmax 在 SRAM 中分块计算
  → 内存 O(n) 而非 O(n²)
  → 速度快 2-4x (减少 HBM 读写)
  
  Flash Attention 2: 更好的并行化
  Flash Attention 3: H100 优化, FP8 支持
```

> 代码: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

**Compiled/Fused Kernels**:
```python
# torch.compile 自动 fuse 操作
model = torch.compile(model, mode="max-autotune")

# 手动 fuse 的关键 kernel:
# - Fused Attention (FlashAttention)
# - Fused LayerNorm + Dropout
# - Fused SwiGLU
# - Fused RoPE
# - Fused Cross-Entropy (lm_head + CE loss)
```

**Communication-Computation Overlap**:
```
不 overlap:
  Forward → AllReduce → Forward → AllReduce ...

Overlap:
  Forward_layer_N → [AllReduce_layer_{N-1} 同时进行] → ...
  
  异步启动通信，计算的同时传输梯度
```

## 6.4 容错与效率

### 6.4.1 Checkpoint 策略

```python
# 异步 checkpoint (不阻塞训练)
def async_save_checkpoint(model, optimizer, step):
    # 把 state_dict copy 到 CPU (非阻塞)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    # 在后台线程写入存储
    threading.Thread(target=torch.save, args=(state, f"ckpt_{step}.pt")).start()

# 分布式 checkpoint (每张卡只存自己的 shard)
from torch.distributed.checkpoint import save, load
save(model.state_dict(), storage_writer=FileSystemWriter(path))
```

### 6.4.2 故障恢复

大集群训练中故障是常态:
- 16K GPU 训练，平均每 2-3 小时一次硬件故障
- GPU 内存错误、网络抖动、节点宕机

**DeepSeek-V3 的解决方案**:
- 每 10 分钟快速 checkpoint 到本地 NVMe (ramdisk)
- 故障时从最近的快速 checkpoint 恢复
- 坏节点自动被替换节点顶替
- 训练仅丢失 10 分钟内的进度

### 6.4.3 MFU (Model FLOPS Utilization)

**衡量训练效率的核心指标**:
```
MFU = 实际计算量 / (理论峰值算力 × 训练时间)

好的 MFU:
  单机: 50-60%
  小集群 (64-256 GPU): 40-50%
  大集群 (1K+ GPU): 30-45%
  超大集群 (16K GPU): 35-40% (LLaMA 3 报告 38-43%)
```

**影响 MFU 的因素**:
- 通信开销（并行策略选择）
- Bubble 比例（PP 调度）
- 数据加载延迟
- Kernel 效率
- Gradient checkpointing 的重计算开销

---

[← 上一章](05-peft.md) | [目录](../README.md) | [下一章 →](07-inference.md)
