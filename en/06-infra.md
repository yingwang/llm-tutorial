[← Previous Chapter](05-peft.md) | [Table of Contents](README.md) | [Next Chapter →](07-inference.md)

---

# Chapter 6: Training Infrastructure

## 6.1 Hardware

### 6.1.1 GPU Selection

| GPU | VRAM | BF16 TFLOPS | FP8 TFLOPS | Interconnect | Use Case |
|-----|------|-------------|------------|------|------|
| A100 80GB | 80GB HBM2e | 312 | - | NVLink 600GB/s | Mainstream training (2022-2024) |
| H100 SXM | 80GB HBM3 | 989 | 1979 | NVLink 900GB/s | Current workhorse |
| H200 | 141GB HBM3e | 989 | 1979 | NVLink 900GB/s | Large models (more VRAM) |
| B200 | 192GB HBM3e | 2250 | 4500 | NVLink 1800GB/s | Next-gen workhorse |
| MI300X (AMD) | 192GB HBM3 | 1307 | 2614 | Infinity Fabric | AMD alternative |

**Key metrics**:
- **VRAM capacity**: Determines the maximum model size that fits
- **Memory bandwidth**: Determines inference speed (inference is memory-bound)
- **Compute (TFLOPS)**: Determines training speed (training is compute-bound)
- **Interconnect bandwidth**: Determines multi-GPU parallelism efficiency

### 6.1.2 Cluster Networking

```
Intra-node:
  GPU ←NVLink 900GB/s→ GPU    (8 GPUs fully connected)

Inter-node:
  Node ←InfiniBand 400Gb/s→ Node
  
  400Gb IB ≈ 50GB/s (per direction)
  NVLink ≈ 900GB/s
  
  So inter-node communication is ~18x slower than intra-node!
  → Parallelism strategies must minimize inter-node communication
```

**Large cluster topology** (e.g., 16K GPUs):
```
GPU (8) → Node → Leaf switch (32 nodes) → Spine switch → Fat-tree/Dragonfly
```

**Network issues**: When training large models, a single slow node or packet loss can slow down the entire job. Requirements:
- [NCCL](https://github.com/NVIDIA/nccl) tuning
- Topology-aware process placement
- Fault tolerance (automatic detection and replacement of failed nodes)

### 6.1.3 Storage

```
Training data reads:
  → Distributed filesystem (Lustre, GPFS, WekaFS)
  → Or object storage (S3) + local NVMe cache

Checkpoint writes:
  → A single 70B model checkpoint is ~500GB
  → Saving every 1000 steps → tens of TB per day
  → Async checkpointing (non-blocking)
  → Nebula/distributed checkpoint storage
```

## 6.2 Distributed Training Strategies

### 6.2.1 Data Parallelism (DP)

**The simplest parallelism strategy**: Each GPU holds a full model replica and processes different data.

```
GPU 0: Model copy + Data batch 0 → gradient_0
GPU 1: Model copy + Data batch 1 → gradient_1
GPU 2: Model copy + Data batch 2 → gradient_2
GPU 3: Model copy + Data batch 3 → gradient_3
          ↓ AllReduce: average gradients ↓
GPU 0-3: Update model with averaged gradients
```

**Limitation**: Each GPU must hold the entire model. A 7B model (BF16) = 14GB parameters + optimizer states ~56GB → barely fits on one 80GB card. A 70B model won't fit.

### 6.2.2 ZeRO (Zero Redundancy Optimizer)

([Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)) **Core contribution of [DeepSpeed](https://github.com/microsoft/DeepSpeed)**: Data parallelism has too much redundancy — every GPU stores the full model + optimizer states. ZeRO shards them.

```
ZeRO Stage 1: Shard optimizer states
  → 4x memory reduction

ZeRO Stage 2: Shard optimizer states + gradients
  → 8x memory reduction

ZeRO Stage 3: Shard optimizer states + gradients + model parameters
  → Nx memory reduction (N = number of GPUs)
  → Equivalent to FSDP
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

**FSDP2** (latest in PyTorch):
- Finer-grained sharding (per-parameter)
- Better integration with tensor parallelism
- FP8 support

### 6.2.3 Tensor Parallelism (TP)

([Shoeybi et al., 2019 — Megatron-LM](https://arxiv.org/abs/1909.08053))

**Splits individual layer matrix multiplications across multiple GPUs**:

```
# Linear layer Y = X @ W (W ∈ ℝ^{d×d})
# Split W by columns across 2 GPUs:

GPU 0: Y_0 = X @ W_0   (W_0 = W[:, :d/2])
GPU 1: Y_1 = X @ W_1   (W_1 = W[:, d/2:])

# AllGather or ReduceScatter to combine results
```

**Megatron-LM style TP**:
- Self-Attention: Q, K, V split by heads
- FFN: First linear layer split by columns, second by rows
- Requires 2 AllReduce operations per layer

**Best suited for**: Intra-node (NVLink bandwidth is sufficient), typically TP=8 (within a single machine)

### 6.2.4 Pipeline Parallelism (PP)

([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473))

**Splits the model by layers across different machines**:

```
GPU 0: Layers 0-7    (Stage 0)
GPU 1: Layers 8-15   (Stage 1)
GPU 2: Layers 16-23  (Stage 2)
GPU 3: Layers 24-31  (Stage 3)

Data flows through the pipeline: Stage 0 → 1 → 2 → 3
```

**Problem**: Naive PP has significant "bubble" time (GPUs idle waiting).

**Solutions**:
- **GPipe** ([Huang et al., 2019](https://arxiv.org/abs/1811.06965)): Split micro-batches into smaller mini-batches to increase pipeline parallelism
- **1F1B** (one-forward-one-backward): Alternate forward and backward scheduling to reduce bubbles
  ```
  Stage 0: F0 F1 F2 F3 B0 B1 B2 B3   (naive, large bubble)
  Stage 0: F0 F1 F2 F3 B0 F4 B1 F5   (1F1B, small bubble)
  ```
- **Interleaved PP**: Each GPU holds non-contiguous layers (e.g., GPU 0 holds layers 0, 8, 16, 24) to reduce bubbles
- **Zero-bubble PP** ([Qi et al., 2024](https://arxiv.org/abs/2401.10241)): Rescheduling to bring bubble time close to 0

### 6.2.5 Context Parallelism (CP)

**Long-sequence parallelism**: Split the sequence across multiple GPUs, with each GPU processing a portion.

```
Sequence length 128K, 4 GPUs:
GPU 0: tokens 0-32K
GPU 1: tokens 32K-64K
GPU 2: tokens 64K-96K
GPU 3: tokens 96K-128K

Attention computed via Ring Attention:
- Each GPU locally computes part of QK^T
- KV is passed to the next GPU via a ring
- Repeat until all KV have been seen
```

### 6.2.6 Expert Parallelism (EP)

**Specific to MoE models**: Different experts are placed on different GPUs.

```
GPU 0: Expert 0, 1
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

Token routing:
1. Gate computes which expert each token goes to → All-to-All
2. Each GPU computes its own experts → All-to-All
3. Results are sent back to the originating GPU
```

**Communication bottleneck**: All-to-All communication scales with token count × hidden_size.

### 6.2.7 3D/4D/5D Parallelism Combinations

Real large-model training combines multiple parallelism strategies:

```
DeepSeek-V3 (2048 H800 GPUs):
  TP=1 (no TP, because MLA and MoE are used)
  PP=16 (16 pipeline stages)
  DP=128 (128-way data parallelism)
  EP=64 (64-way expert parallelism)
  
  2048 = 16 × 128 = 16 × 2 × 64

LLaMA 3 405B (16384 H100 GPUs):
  TP=8 (intra-node)
  PP=16 (inter-node)
  DP=128 (data parallelism)
  CP for long-context training
  
  16384 = 8 × 16 × 128
```

## 6.3 Training Frameworks

### 6.3.1 Framework Comparison

| Framework | Company | Parallelism Strategies | Scale |
|------|------|---------|---------|
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA | TP+PP+DP, MoE | 100B-1T+ |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft | ZeRO, PP, MoE | 10B-1T |
| [FSDP](https://pytorch.org/docs/stable/fsdp.html) (PyTorch) | Meta | ZeRO-3 | 10B-100B |
| [ColossalAI](https://github.com/hpcaitech/ColossalAI) | HPC-AI Tech | Multiple | 10B-100B |
| [Levanter](https://github.com/stanford-crfm/levanter) | Stanford | JAX-based | Research |
| [NanoGPT](https://github.com/karpathy/nanoGPT) | Karpathy | DDP | Learning/small scale |

### 6.3.2 Megatron-LM Core

```python
# Megatron-LM 3D parallelism configuration
# launch: torchrun --nproc_per_node=8 --nnodes=64

args = {
    "tensor_model_parallel_size": 8,     # TP: intra-node
    "pipeline_model_parallel_size": 16,   # PP: inter-node
    "data_parallel_size": 32,             # DP: auto = total / TP / PP
    "sequence_parallel": True,            # Sequence parallelism (paired with TP)
    "use_flash_attn": True,
    "bf16": True,
    "micro_batch_size": 1,
    "global_batch_size": 1024,
}
```

### 6.3.3 Efficient Training Techniques

**Gradient Checkpointing (Activation Recomputation)** ([Chen et al., 2016](https://arxiv.org/abs/1604.06174)):
```
Normal: Save all intermediate activations during forward → use in backward
Problem: Activations consume too much memory (proportional to batch_size × seq_len × hidden × layers)

Gradient Checkpointing: Save activations only at selected layers, recompute during backward
  → Memory reduced by √L (L = number of layers)
  → ~33% compute overhead

Selective checkpointing: Only checkpoint memory-heavy operations (attention)
```

**Flash Attention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)):
```
Standard Attention:
  S = Q @ K^T          → O(n²d) compute, O(n²) memory
  P = softmax(S)       → stores n² matrix
  O = P @ V            

Flash Attention (Tri Dao):
  Does not materialize the n² attention matrix
  Uses tiling + online softmax to compute in SRAM block by block
  → O(n) memory instead of O(n²)
  → 2-4x speedup (fewer HBM reads/writes)
  
  Flash Attention 2: Better parallelization
  Flash Attention 3: H100 optimizations, FP8 support
```

> Code: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

**Compiled/Fused Kernels**:
```python
# torch.compile auto-fuses operations
model = torch.compile(model, mode="max-autotune")

# Key manually fused kernels:
# - Fused Attention (FlashAttention)
# - Fused LayerNorm + Dropout
# - Fused SwiGLU
# - Fused RoPE
# - Fused Cross-Entropy (lm_head + CE loss)
```

**Communication-Computation Overlap**:
```
Without overlap:
  Forward → AllReduce → Forward → AllReduce ...

With overlap:
  Forward_layer_N → [AllReduce_layer_{N-1} running concurrently] → ...
  
  Launch communication asynchronously, transmit gradients while computing
```

## 6.4 Fault Tolerance and Efficiency

### 6.4.1 Checkpoint Strategy

```python
# Async checkpointing (non-blocking)
def async_save_checkpoint(model, optimizer, step):
    # Copy state_dict to CPU (non-blocking)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    # Write to storage in a background thread
    threading.Thread(target=torch.save, args=(state, f"ckpt_{step}.pt")).start()

# Distributed checkpointing (each GPU saves only its own shard)
from torch.distributed.checkpoint import save, load
save(model.state_dict(), storage_writer=FileSystemWriter(path))
```

### 6.4.2 Failure Recovery

Failures are the norm in large cluster training:
- With 16K GPU training, hardware failure occurs on average every 2-3 hours
- GPU memory errors, network jitter, node crashes

**DeepSeek-V3's approach**:
- Fast checkpoint to local NVMe (ramdisk) every 10 minutes
- On failure, recover from the most recent fast checkpoint
- Failed nodes are automatically replaced by spare nodes
- Only ~10 minutes of training progress is lost

### 6.4.3 MFU (Model FLOPS Utilization)

**The key metric for training efficiency**:
```
MFU = Actual computation / (Theoretical peak FLOPS × Training time)

Good MFU:
  Single node: 50-60%
  Small cluster (64-256 GPUs): 40-50%
  Large cluster (1K+ GPUs): 30-45%
  Very large cluster (16K GPUs): 35-40% (LLaMA 3 reported 38-43%)
```

**Factors affecting MFU**:
- Communication overhead (parallelism strategy choices)
- Bubble ratio (PP scheduling)
- Data loading latency
- Kernel efficiency
- Recomputation overhead from gradient checkpointing

## Key Papers

- [Rajbhandari et al. (2019) — ZeRO](https://arxiv.org/abs/1910.02054) — memory optimization under data parallelism (the heart of DeepSpeed)
- [Shoeybi et al. (2019) — Megatron-LM](https://arxiv.org/abs/1909.08053) — Tensor Parallelism for transformers
- [Huang et al. (2018) — GPipe](https://arxiv.org/abs/1811.06965) — the classic Pipeline Parallelism paper
- [Dao et al. (2022) — FlashAttention](https://arxiv.org/abs/2205.14135) — IO-aware attention, 2–4× speedup
- [Dao (2023) — FlashAttention-2](https://arxiv.org/abs/2307.08691) — further parallelism optimizations

## Further Reading

- [Megatron-LM repo](https://github.com/NVIDIA/Megatron-LM) — reference 3D-parallel implementation
- [DeepSpeed docs](https://www.deepspeed.ai/) — ZeRO-1/2/3 configuration
- [How to Train Really Large Models](https://lilianweng.github.io/posts/2021-09-25-train-large/) — Lilian Weng

## Exercises

1. **Memory budget**: estimate GPU count needed to train a 70B model (FP16, AdamW, batch=1M tokens) under ZeRO-1, 2, and 3.
2. **Profile training**: use PyTorch profiler on one nanoGPT step; identify the slowest op and compare before/after FlashAttention.
3. **3D-parallel tradeoff**: with 64 H100s for a 70B model, choose a reasonable (DP, TP, PP) configuration and explain why TP shouldn't cross nodes.

---

[← Previous Chapter](05-peft.md) | [Table of Contents](README.md) | [Next Chapter →](07-inference.md)
