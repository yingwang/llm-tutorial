[← Previous Chapter](04-post-training.md) | [Table of Contents](README.md) | [Next Chapter →](06-infra.md)

---

# Chapter 5: Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning a 70B model requires ~1TB of VRAM (parameters + optimizer states + gradients + activations). PEFT methods train only a small subset of parameters, drastically reducing resource requirements.

## 5.1 LoRA (Low-Rank Adaptation)

([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) — **Currently the most widely used PEFT method.**

### 5.1.1 Core Idea

Freeze the pretrained weights and add a trainable low-rank decomposition as a bypass:

```
# Original: Y = X @ W       (W ∈ ℝ^{d×d})
# LoRA: Y = X @ W + X @ A @ B
#       A ∈ ℝ^{d×r}, B ∈ ℝ^{r×d}, r << d

# Parameter count comparison (d=4096, r=16):
# Full: 4096 × 4096 = 16.7M
# LoRA: 4096 × 16 + 16 × 4096 = 131K (128x reduction!)
```

### 5.1.2 Key Hyperparameters

| Hyperparameter | Typical Values | Description |
|------|--------|------|
| `r` (rank) | 8-64 | Higher values approach full fine-tuning but add more parameters |
| `alpha` | 16-64 | Scaling factor. Effective scaling = alpha/r |
| `target_modules` | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj | Which layers to apply LoRA to |
| `dropout` | 0.05-0.1 | Dropout for LoRA layers |

**Rules of thumb**:
- `r=16, alpha=32` is a good starting point
- Applying LoRA to both attention layers (q, k, v, o) and FFN layers (gate, up, down) works best
- Rank too small → underfitting; too large → wasted resources and potential overfitting

### 5.1.3 Code Implementation

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.52%
```

> Library: [huggingface/peft](https://github.com/huggingface/peft)

### 5.1.4 LoRA Merging

After training, LoRA weights can be merged back into the base model for zero overhead at inference:

```python
merged_model = model.merge_and_unload()
# W_new = W + A @ B
# At inference, identical to a fully fine-tuned model
```

## 5.2 QLoRA

([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) — **Enables fine-tuning a 70B model on a single 24GB GPU.**

### 5.2.1 Core Innovations

Three memory optimizations on top of LoRA:

1. **4-bit NormalFloat (NF4) Quantization**: Quantize frozen parameters to 4-bit
2. **Double Quantization**: Quantize the scaling factors themselves (further memory savings)
3. **Paged Optimizers**: Use CPU memory as swap space for GPU memory

```
Memory comparison (LLaMA 65B):
Full fine-tuning: ~780GB (requires multi-node)
LoRA FP16: ~130GB (2× A100 80GB)
QLoRA 4-bit: ~48GB (1× A100 80GB!)
```

### 5.2.2 Usage

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
)

# Then add the adapter just like regular LoRA
model = get_peft_model(model, lora_config)
```

## 5.3 Other PEFT Methods

| Method | Mechanism | Parameter % | Paper |
|------|------|--------|------|
| **Prefix Tuning** | Add trainable prefix tokens before each attention layer | 0.1% | [Li & Liang, 2021](https://arxiv.org/abs/2101.00190) |
| **Prompt Tuning** | Add trainable soft prompts before the input | 0.01% | [Lester et al., 2021](https://arxiv.org/abs/2104.08691) |
| **Adapter** | Insert small MLPs after each Transformer layer | 1-5% | [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751) |
| **IA3** | Learn scaling vectors for K, V, and FFN | 0.01% | [Liu et al., 2022](https://arxiv.org/abs/2205.05638) |
| **DoRA** | LoRA + weight magnitude decomposition | ~LoRA | [Liu et al., 2024](https://arxiv.org/abs/2402.09353) |

**Practical choice**: LoRA or QLoRA covers the vast majority of use cases. Other methods have advantages in specific scenarios but lack the ecosystem support of LoRA.

## 5.4 Practical PEFT Guidelines

**When to use PEFT vs full fine-tuning**:
- Data < 100K samples: PEFT (full fine-tuning tends to overfit)
- Data > 1M samples + large budget: Full fine-tuning
- Limited budget (single/few GPUs): QLoRA
- Need to switch between multiple tasks: LoRA (can load multiple adapters simultaneously)

**Common tools**:
- [huggingface/trl](https://github.com/huggingface/trl) — SFT, DPO, PPO + LoRA
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl) — All-in-one fine-tuning (supports LoRA/QLoRA + various formats)
- [unsloth](https://github.com/unslothai/unsloth) — 2x faster LoRA fine-tuning (hand-written kernels)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — Popular fine-tuning framework in the Chinese community

## Key Papers

- [Hu et al. (2021) — LoRA](https://arxiv.org/abs/2106.09685) — low-rank adaptation, the dominant PEFT method
- [Dettmers et al. (2023) — QLoRA](https://arxiv.org/abs/2305.14314) — 4-bit quantization + LoRA, fits 65B on a single GPU
- [Houlsby et al. (2019) — Adapter](https://arxiv.org/abs/1902.00751) — the original parameter-efficient adapter
- [Li & Liang (2021) — Prefix Tuning](https://arxiv.org/abs/2101.00190) — learnable prefixes injected into attention
- [Liu et al. (2022) — IA³](https://arxiv.org/abs/2205.05638) — even fewer parameters than LoRA

## Further Reading

- HuggingFace — [PEFT library](https://github.com/huggingface/peft) — unified interface for all methods
- [Ahead of AI — Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [QLoRA companion code](https://github.com/artidoro/qlora)

## Exercises

1. **LoRA fine-tune**: use PEFT to attach LoRA (rank=8) to Qwen2-1.5B and fine-tune on Alpaca; compare memory and quality vs full fine-tuning.
2. **Rank sweep**: fix the dataset and sweep LoRA rank ∈ {1, 4, 16, 64}; plot loss vs rank to find the sweet spot.
3. **Multi-LoRA merge**: train two task-specific LoRAs (Chinese-English translation + code generation), merge with mergekit, and check whether both capabilities are retained.

---

[← Previous Chapter](04-post-training.md) | [Table of Contents](README.md) | [Next Chapter →](06-infra.md)
