[← Previous Chapter](05-peft.md) | [Table of Contents](README.md) | [Next Chapter →](07-inference.md)

---

# Chapter 6: Training Infrastructure

Scaling large language models to hundreds of billions of parameters requires orchestrating multi-thousand GPU clusters with near-linear scaling efficiency, robust fault tolerance, and tight communication-computation overlap.

## 6.1 Hardware Architecture

### 6.1.1 Accelerator Selection Matrix

| Accelerator | VRAM Capacity | BF16 Tensor TFLOPS | FP8 Tensor TFLOPS | Interconnect Bandwidth | Primary Deployment Realm |
|-------------|---------------|-------------------|-------------------|------------------------|--------------------------|
| NVIDIA A100 | 80GB HBM2e | 312 | - | NVLink (600 GB/s) | Mature baseline clusters (2022-2024) |
| NVIDIA H100 SXM | 80GB HBM3 | 989 | 1,979 | NVLink (900 GB/s) | Industry workhorse for pretraining |
| NVIDIA H200 | 141GB HBM3e | 989 | 1,979 | NVLink (900 GB/s) | Memory-bound long-context pretraining |
| NVIDIA B200 | 192GB HBM3e | 2,250 | 4,500 | NVLink (1,800 GB/s) | Next-generation scaling frontier |
| AMD MI300X | 192GB HBM3 | 1,307 | 2,614 | Infinity Fabric (896 GB/s) | High-capacity open ecosystem alternative |

**Core Hardware Scaling Bottlenecks**:
- **High Bandwidth Memory (HBM) Capacity**: Sets the hard ceiling on maximum parameter and activation storage per device.
- **Memory Bandwidth**: Dictates decoding throughput in inference (which is memory-bandwidth bound).
- **Tensor Core Compute**: Governs pretraining step time (which is compute-bound during large-batch matrix multiplication).
- **Interconnect Bandwidth**: Determines the scaling efficiency of Tensor and Pipeline Parallelism across distributed ranks.

### 6.1.2 Cluster Fabric Networking

```
Intra-Node Domain:
  GPU 0 ←── NVLink 900 GB/s (All-to-All Full Mesh) ──→ GPU 7 (Within 8-GPU Chassis)

Inter-Node Cluster Fabric:
  Node A ←── InfiniBand 400 Gbps / 800 Gbps (RDMA) ──→ Node B
  
  Bandwidth Disparity Analysis:
  400 Gbps InfiniBand ≈ 50 GB/s per rail
  NVLink 4.0          ≈ 900 GB/s bidirectional
  
  Physical Reality: Inter-node transfer is ~18x slower than intra-node NVLink.
  Systems Rule: High-frequency collective operations (Tensor Parallelism) must be
  confined strictly within single nodes, reserving inter-node fabric for DP and PP.
```

**Large Cluster Topology** (16K+ GPUs): Non-blocking Fat-Tree or Rail-Optimized Dragonfly topology organized via leaf, spine, and core InfiniBand switches.

**Network Engineering Requirements**:
- Automated NCCL tree and ring algorithm selection tailored to physical topology.
- Rail-aligned process placement (ensuring rank $i$ on node $A$ communicates with rank $i$ on node $B$ through dedicated NICs).
- Fast failover detection to isolate stragglers, silent data corruption, and flapping optical links.

### 6.1.3 Storage Hierarchies and I/O Bandwidth

```
Training Data Ingestion:
  Distributed Parallel Filesystem (Lustre / GPFS / WekaFS)
  or High-Throughput Object Storage (S3) paired with local NVMe caching.

Checkpoint Ingestion and Egress:
  Saving full model and optimizer states for a 70B parameter model produces ~500 GB per snapshot.
  Periodic writes (e.g. every 1,000 steps) demand multi-terabyte burst bandwidth.
  Mitigation: Asynchronous, non-blocking snapshot streaming to remote storage tiers.
```

## 6.2 Distributed Parallelism Paradigms

### 6.2.1 Data Parallelism (DP) and DDP

**The Fundamental Baseline**: Replicates identical model weights across all GPU devices, partitions the global batch across ranks, and executes a collective `AllReduce` to synchronize gradients prior to the optimizer update:

```
GPU 0: Model Copy + Micro-Batch 0 ──> Gradient_0
GPU 1: Model Copy + Micro-Batch 1 ──> Gradient_1
GPU 2: Model Copy + Micro-Batch 2 ──> Gradient_2
GPU 3: Model Copy + Micro-Batch 3 ──> Gradient_3
        │
        └──> Collective AllReduce: Average Gradients across All Ranks
        │
GPU 0-3: Synchronous Optimizer Step
```

**Capacity Barrier**: Standard DDP breaks down when model states (parameters + gradients + optimizer states + activations) exceed single-GPU VRAM capacity.

### 6.2.2 Zero Redundancy Optimizer (ZeRO) & FSDP

([Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)): Foundational breakthrough of DeepSpeed. Standard data parallelism introduces severe memory redundancy because every rank replicates the entire model, gradients, and optimizer states. ZeRO eliminates this redundancy through partitioned sharding:

```
ZeRO-Stage 1 (Optimizer State Partitioning):
  Shards FP32 AdamW states across DP ranks.
  Memory reduction: 4x savings without added communication overhead.

ZeRO-Stage 2 (Gradient + Optimizer State Partitioning):
  Shards both optimizer states and gradients across DP ranks.
  Memory reduction: 8x savings.

ZeRO-Stage 3 (Full Parameter Sharding / PyTorch FSDP):
  Shards model parameters, gradients, and optimizer states across all DP ranks.
  Memory reduction: Proportional to world size N.
  Communication: Broadcasts parameters on-the-fly via AllGather during forward pass,
  releases them immediately, and performs ReduceScatter during backward pass.
```

**PyTorch Fully Sharded Data Parallel (FSDP)**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    auto_wrap_policy=transformer_auto_wrap_policy,
)
```

### 6.2.3 Tensor Parallelism (TP)

([Shoeybi et al., 2019: Megatron-LM](https://arxiv.org/abs/1909.08053))

**Intra-Layer Tensor Sharding**: Partitions individual weight matrices across GPUs within the same node:

```
# Column-Parallel Linear Layer (e.g. Attention QKV projection):
W = [W_1 | W_2]  (Split along column dimension across GPU 0 and GPU 1)
Y_1 = X @ W_1,   Y_2 = X @ W_2

# Row-Parallel Linear Layer (e.g. Attention Output projection):
W = [W_1]
    [---]        (Split along row dimension across GPU 0 and GPU 1)
    [W_2]
Y = Y_1 @ W_1 + Y_2 @ W_2  ──> Combined via AllReduce collective
```

- In Megatron-LM style Transformer blocks, Column-Parallel and Row-Parallel layers are paired consecutively, requiring only two `AllReduce` operations per Transformer layer.
- Because `AllReduce` must execute on every single forward and backward layer step, TP is confined strictly to high-bandwidth intra-node NVLink domains ($TP \le 8$).

### 6.2.4 Pipeline Parallelism (PP)

([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473))

**Inter-Layer Sequential Sharding**: Partitions the model depth-wise across multiple nodes, streaming micro-batches through a pipeline schedule:

```
Stage 0 (GPU 0): Layers [0, 8)   ──> P2P Send Activation ──>
Stage 1 (GPU 1): Layers [8, 16)  ──> P2P Send Activation ──>
Stage 2 (GPU 2): Layers [16, 24) ──> P2P Send Activation ──>
Stage 3 (GPU 3): Layers [24, 32) ──> Loss Computation
```

**Pipeline Scheduling and Bubble Reduction**:
- **GPipe** ([Huang et al., 2019](https://arxiv.org/abs/1811.06965)): Accumulates all micro-batch activations before executing backward passes; suffers from high activation memory overhead and significant bubble latency ($t_{\text{bubble}} \approx \frac{p-1}{m}$).
- **1F1B (One-Forward-One-Backward)**: Interleaves execution such that each device immediately executes one backward step after one forward step, bounding peak activation memory to the number of pipeline stages $p$.
- **Zero-Bubble Pipeline Parallelism** ([Qi et al., 2024](https://arxiv.org/abs/2401.10241)): Decouples backward computations into gradient computation ($B$) and activation recomputation ($W$), scheduling them into pipeline voids to virtually eliminate idle bubble time.

### 6.2.5 Context Parallelism (CP) and Sequence Parallelism (SP)

- **Megatron Sequence Parallelism (SP)**: Breaks activation memory redundancy in LayerNorm and Dropout layers across the TP group via `ReduceScatter` and `AllGather`.
- **Context Parallelism (Ring Attention)** ([Liu et al., 2023](https://arxiv.org/abs/2310.01889)): For ultra-long sequence horizons (128K to 1M tokens), CP partitions sequence lengths across distributed devices, asynchronously circulating Key-Value blocks in a communication ring while computing local attention blocks.

### 6.2.6 Expert Parallelism (EP) for MoE

Assigns distinct expert sub-networks to different GPU ranks:
1. **Token Dispatch (All-to-All)**: Tokens route to target expert ranks based on gate routing scores.
2. **Local Expert Computation**: Each GPU processes its assigned tokens through its local experts.
3. **Token Combine (All-to-All)**: Output activations are gathered back to the originating rank.

### 6.2.7 Multi-Dimensional Parallelism Compositions (3D/4D/5D)

Production pretraining runs orchestrate multidimensional parallelism hybrids:

```
DeepSeek-V3 Infrastructure Layout (2,048 H800 GPUs):
  Tensor Parallelism:   TP = 1   (Eliminated via MLA and fine-grained MoE design)
  Pipeline Parallelism: PP = 16  (16 pipeline stages across nodes)
  Data Parallelism:     DP = 128 (ZeRO-1 optimizer sharding)
  Expert Parallelism:   EP = 64  (MoE expert dispatch across 64 ranks)
  Topology Formula:     Total GPUs = 16 (PP) × 128 (DP) = 2,048

LLaMA 3 (405B) Infrastructure Layout (16,384 H100 GPUs):
  Tensor Parallelism:   TP = 8   (Intra-node NVLink mesh)
  Pipeline Parallelism: PP = 16  (Inter-node InfiniBand fabric)
  Data Parallelism:     DP = 128 (FSDP / ZeRO-3 data parallel ranks)
  Context Parallelism:  CP enabled for 128K sequence phases
  Topology Formula:     Total GPUs = 8 (TP) × 16 (PP) × 128 (DP) = 16,384
```

## 6.3 Distributed Training Frameworks and Acceleration

### 6.3.1 Framework Landscape

| Framework | Sponsoring Organization | Primary Strengths | Target Scaling Realm |
|-----------|------------------------|-------------------|----------------------|
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA | Industrial 3D Parallelism, custom kernels | 100B-1T+ parameters |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft | ZeRO-1/2/3, 3D Parallelism, MoE | 10B-1T parameters |
| [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) | Meta | Native PyTorch ZeRO-3 integration | 10B-100B parameters |
| [Colossal-AI](https://github.com/hpcaitech/ColossalAI) | HPC-AI Tech | Multi-dimensional auto-parallelism | 10B-100B parameters |

### 6.3.2 Systems Acceleration Techniques

**Selective Activation Checkpointing** ([Chen et al., 2016](https://arxiv.org/abs/1604.06174)): Discards intermediate activations during forward passes and selectively recomputes only memory-intensive operators (such as attention projections) during backward passes, cutting peak activation memory by $\approx 70\%$ at a modest $\approx 30\%$ compute cost.

**FlashAttention IO-Aware Tiling** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)):
- Bypasses materialization of the full $N \times N$ attention matrix in High Bandwidth Memory (HBM).
- Splits queries, keys, and values into sub-blocks loaded into fast on-chip SRAM, computing online softmax incremental scaling dynamically.
- Eliminates memory reads/writes, yielding a 2x-4x wall-clock speedup and reducing memory complexity from $O(N^2)$ to $O(N)$.

**Kernel Fusion**: Fuses multiple sequential memory-bound operations into unified GPU kernels (e.g., Fused RMSNorm + RoPE, Fused SwiGLU, Fused Cross-Entropy), eliminating unnecessary round-trip global memory latency.

**Communication-Computation Overlap**: Launches asynchronous collective communications (such as `ReduceScatter` in DP/FSDP) concurrently with backward pass matrix multiplications on preceding layers.

## 6.4 Cluster Reliability and Operational Efficiency

### 6.4.1 Resilient Checkpointing Mechanisms

```python
# Asynchronous distributed snapshotting logic
from torch.distributed.checkpoint import save, FileSystemWriter

# Each rank writes strictly its local shard directly to NVMe storage in non-blocking threads
storage_writer = FileSystemWriter(checkpoint_dir)
save(
    state_dict={"model": model.state_dict(), "optim": optimizer.state_dict()},
    storage_writer=storage_writer,
)
```

### 6.4.2 Failure Mitigation in 10K-GPU Fleets

In clusters operating at 16,000+ accelerators, Mean Time Between Failures (MTBF) drops to mere hours due to cosmic-ray memory flips, high-voltage transceiver faults, and link degradation:
- **Fast Heartbeat Health Probes**: Real-time diagnostic monitors detecting silent GPU hangs and CUDA error states within seconds.
- **Dynamic Node Eviction & Hot-Spare Slicing**: Automatically draining failing compute nodes, routing traffic to hot-standby nodes, and resuming training from local NVMe checkpoints within minutes.

### 6.4.3 Model FLOPS Utilization (MFU)

**The Universal Metric of Distributed Engineering Health**:

$$\text{MFU} = \frac{\text{Observed Training Throughput (Tokens/sec)} \times \text{Theoretical FLOPs per Token}}{\text{Peak Hardware Cluster Theoretical Floating-Point Capacity (FLOPs/sec)}}$$

$$\text{FLOPs per Token for Standard Decoder LLMs} \approx 6N_{\text{params}} + 12L_{\text{layers}} H_{\text{hidden}} T_{\text{seq\_len}}$$

**Production MFU Baselines**:
- Single-Node Execution: 55-65% MFU.
- Mid-Scale Cluster (64-256 GPUs): 45-55% MFU.
- Frontier 10K-GPU Cluster: 35-43% MFU (Meta LLaMA 3 reported 38-43% on 16K H100s).

## Key Papers

- [Rajbhandari et al. (2020): ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054): Foundational zero-redundancy memory sharding paper.
- [Shoeybi et al. (2019): Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053): Foundational Tensor Parallelism framework.
- [Narayanan et al. (2021): Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473): Milestone 3D parallelism synthesis.
- [Dao et al. (2022): FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135): Foundational IO-aware attention paper.
- [Qi et al. (2024): Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241): Frontier bubble-free pipeline scheduling.

## Further Reading

- NVIDIA: [Megatron-LM Official Repository](https://github.com/NVIDIA/Megatron-LM) (Reference implementation for distributed 3D parallelism).
- DeepSpeed: [DeepSpeed Documentation & Tutorials](https://www.deepspeed.ai/) (Complete ZeRO-1/2/3 configuration guides).
- Lilian Weng: [How to Train Really Large Models](https://lilianweng.github.io/posts/2021-09-25-train-large/) (Comprehensive architectural survey of distributed methods).

## Exercises

1. **Distributed Memory Estimation**: Calculate the exact VRAM footprint (parameters, master weights, gradients, optimizer states) required to train a 70B parameter model in BF16 precision under ZeRO-1, ZeRO-2, and ZeRO-3 across 64 GPUs.
2. **PyTorch Profiler & FlashAttention Benchmark**: Profile a forward-backward pass of a causal Transformer block with and without FlashAttention-2; inspect kernel timelines and memory bandwidth saturation.
3. **3D Parallelism Topology Design**: Given a cluster of 64 NVIDIA H100 SXM GPUs (8 nodes with NVLink within nodes, 400 Gbps InfiniBand across nodes), design the optimal $(TP, PP, DP)$ configuration for pretraining a 70B model and justify your architectural decisions.

---

[← Previous Chapter](05-peft.md) | [Table of Contents](README.md) | [Next Chapter →](07-inference.md)
