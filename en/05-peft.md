[← Previous Chapter](04-post-training.md) | [Table of Contents](README.md) | [Next Chapter →](06-infra.md)

---

# Chapter 5: Parameter-Efficient Fine-Tuning (PEFT)

Full-parameter fine-tuning of a 70B foundation model demands approximately 1TB of distributed accelerator VRAM across FP16/BF16 weights, FP32 master copies, optimizer momentum states, and intermediate activation buffers. Parameter-Efficient Fine-Tuning (PEFT) freezes the underlying foundation weights and injects small, modular adaptation parameters (typically <1% of the original model footprint), reducing hardware requirements by an order of magnitude.

## 5.1 LoRA (Low-Rank Adaptation)

([Hu et al., 2021](https://arxiv.org/abs/2106.09685)): The industry-standard parameter-efficient adaptation paradigm.

### 5.1.1 Foundational Formulation

LoRA hypothesizes that weight updates $\Delta W$ during task adaptation possess a low "intrinsic dimension." Instead of updating the full $d \times k$ weight matrix $W_0$, LoRA decomposes the update into the product of two low-rank matrices:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

where $W_0 \in \mathbb{R}^{d \times k}$ remains frozen, $A \in \mathbb{R}^{r \times k}$ is initialized via Gaussian random initialization $\mathcal{N}(0, \sigma^2)$, and $B \in \mathbb{R}^{d \times r}$ is initialized to zero, ensuring $\Delta W = 0$ at the start of training.

```
Forward Pass Formulation:
h = W_0 @ x + (α / r) * (B @ (A @ x))

Parameter Scaling Example (d = 4096, r = 16):
Full-Rank Projection:  4096 × 4096 = 16,777,216 parameters
LoRA Low-Rank Path:    4096 × 16 + 16 × 4096 = 131,072 parameters (128x reduction)
```

### 5.1.2 Key Hyperparameter Tuning

| Hyperparameter | Production Range | Engineering Role & Rationale |
|----------------|------------------|------------------------------|
| `r` (Rank) | 8 to 64 | Controls the expressive capacity of the low-rank subspace |
| `lora_alpha` | 16 to 64 | Fixed constant scaling factor; effective multiplier is $\frac{\alpha}{r}$ |
| `target_modules` | All linear layers | Injecting LoRA into both Attention ($q, k, v, o$) and MLP ($gate, up, down$) layers maximizes expressivity |
| `lora_dropout` | 0.05 to 0.1 | Regularization to prevent adapter overfitting on small datasets |

**Empirical Optimization Rules**:
- Setting $\alpha = 2r$ is a robust production default.
- Scaling rank $r > 64$ rarely yields downstream accuracy gains commensurate with the increased memory footprint; full linear layer coverage with modest rank ($r=16$) consistently outperforms high-rank adaptation on attention layers alone.

### 5.1.3 Implementation via Hugging Face PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.52%
```

> Core library: [huggingface/peft](https://github.com/huggingface/peft)

### 5.1.4 Zero-Overhead Inference Weight Merging

Upon completing fine-tuning, the low-rank update $\Delta W = \frac{\alpha}{r} BA$ can be analytically folded directly into the base weights, eliminating runtime latency penalties during inference serving:

```python
# Fold adapter parameters directly into base model weights
merged_model = model.merge_and_unload()
# W_final = W_0 + (alpha / r) * (B @ A)
# Serves with standard native inference kernels without custom adapter routing
```

## 5.2 QLoRA (Quantized Low-Rank Adaptation)

([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)): Democratized large model adaptation by enabling 70B parameter fine-tuning on a single 48GB/80GB GPU.

### 5.2.1 Core Architectural Pillars

1. **4-Bit NormalFloat (NF4) Quantization**: An information-theoretically optimal quantile quantization format designed specifically for normally distributed neural network weights.
2. **Double Quantization (DQ)**: Quantizes the first-stage quantization scaling constants (converting 32-bit floats to 8-bit integers), saving an additional 0.37 bits per parameter.
3. **Paged Optimizers**: Leverages CUDA unified memory to dynamically swap optimizer state allocations between GPU VRAM and CPU system RAM during activation memory spikes.

```
VRAM Footprint Breakdown for LLaMA 65B/70B Fine-Tuning:
Full 16-bit Precision Fine-Tuning: ~780 GB (requires multi-node GPU cluster)
Standard LoRA (16-bit Base Weights): ~130 GB (requires 2x A100 80GB)
QLoRA (NF4 4-bit Base Weights):     ~48 GB  (fits on a single A100/H100 80GB)
```

### 5.2.2 QLoRA Pipeline Configuration

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Attach standard LoRA adapter over the quantized base
model = get_peft_model(model, lora_config)
```

## 5.3 Architectural Taxonomy of PEFT Paradigms

| Method | Architectural Injection Mechanism | Parameter Budget | Reference |
|--------|-----------------------------------|------------------|-----------|
| **Prefix Tuning** | Prepends learnable continuous vectors to Key and Value states | ~0.1% | [Li & Liang, 2021](https://arxiv.org/abs/2101.00190) |
| **Prompt Tuning** | Prepends learnable soft embeddings strictly to input token sequences | ~0.01% | [Lester et al., 2021](https://arxiv.org/abs/2104.08691) |
| **Series Adapters** | Inserts bottleneck MLP sub-layers sequentially after attention and FFN blocks | ~1-3% | [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751) |
| **IA³** | Rescales internal activation channels via learned element-wise vectors | ~0.01% | [Liu et al., 2022](https://arxiv.org/abs/2205.05638) |
| **DoRA** | Decomposes weight matrix updates into directional and magnitude components | ~0.6% | [Liu et al., 2024](https://arxiv.org/abs/2402.09353) |

## 5.4 Production Engineering Selection Matrix

**Decision Framework**:
- **Curated Dataset < 100K Demonstrations**: Deploy LoRA/QLoRA. Full fine-tuning on constrained datasets frequently induces catastrophic forgetting of general world knowledge.
- **Large Dataset > 1M Pairs + Sufficient Compute**: Deploy full-parameter fine-tuning for maximum reasoning plasticity and domain adaptation.
- **Multi-Tenant Serving Infrastructure**: LoRA enables dynamic multi-tenant serving (e.g., via S-LoRA or vLLM adapter hot-swapping), routing requests to distinct task-specific adapters on top of a single shared base model.

**Production Toolchains**:
- [Hugging Face TRL](https://github.com/huggingface/trl): Integrated SFT, DPO, and PPO pipelines with native LoRA bindings.
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl): High-performance YAML-configured training framework for distributed fine-tuning.
- [Unsloth](https://github.com/unslothai/unsloth): Custom Triton kernels delivering 2x-5x faster fine-tuning with 70% lower VRAM overhead.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Production-grade unified fine-tuning suite supporting 100+ open architectures.

## Key Papers

- [Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): Landmark low-rank parameter-efficient adaptation paper.
- [Dettmers et al. (2023): QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314): NF4 quantization and double quantization framework.
- [Houlsby et al. (2019): Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751): Foundational sequential bottleneck adapter architecture.
- [Li & Liang (2021): Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190): Continuous virtual prefix optimization.
- [Liu et al. (2024): DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353): Directional and magnitude decoupled adaptation.

## Further Reading

- Hugging Face: [PEFT Library Documentation](https://huggingface.co/docs/peft) (Comprehensive developer guide and API reference).
- Sebastian Raschka: [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) (In-depth hyperparameter sweep analysis).
- Tim Dettmers: [QLoRA Code Repository and Methodology](https://github.com/artidoro/qlora) (Reference implementation and precision analysis).

## Exercises

1. **LoRA Fine-Tuning Benchmark**: Attach a LoRA adapter ($r = 16, \alpha = 32$) to a 7B base model using Hugging Face PEFT; measure VRAM usage, step throughput, and downstream instruction accuracy against a full fine-tuning run.
2. **Rank Sweep & Pareto Frontier**: Fine-tune a compact 1.5B model across ranks $r \in \{2, 8, 32, 128\}$; plot training loss convergence curves and evaluate where parameter returns saturate.
3. **Multi-Adapter Dynamic Fusion**: Fine-tune two distinct LoRA adapters (one for SQL extraction and one for technical translation); use `mergekit` to fuse their weight matrices via spherical linear interpolation (SLERP), and evaluate cross-task retention.

---

[← Previous Chapter](04-post-training.md) | [Table of Contents](README.md) | [Next Chapter →](06-infra.md)
