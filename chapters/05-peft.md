[← 上一章](04-post-training.md) | [目录](../README.md) | [下一章 →](06-infra.md)

---

# 第五章：参数高效微调 (PEFT)

全量微调一个 70B 模型需要 ~1TB 显存（参数 + 优化器状态 + 梯度 + 激活）。PEFT 方法只训练一小部分参数，大幅降低资源需求。

## 5.1 LoRA (Low-Rank Adaptation)

([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) — **目前最主流的 PEFT 方法。**

### 5.1.1 核心思想

预训练权重冻结，旁路加入低秩分解的可训练矩阵：

```
# 原始: Y = X @ W       (W ∈ ℝ^{d×d})
# LoRA: Y = X @ W + X @ A @ B
#       A ∈ ℝ^{d×r}, B ∈ ℝ^{r×d}, r << d

# 参数量对比 (d=4096, r=16):
# 全量: 4096 × 4096 = 16.7M
# LoRA: 4096 × 16 + 16 × 4096 = 131K (减少 128 倍!)
```

### 5.1.2 关键超参

| 超参 | 常用值 | 说明 |
|------|--------|------|
| `r` (rank) | 8-64 | 越大越接近全量微调，但参数越多 |
| `alpha` | 16-64 | 缩放因子。实际 scaling = alpha/r |
| `target_modules` | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj | 对哪些层加 LoRA |
| `dropout` | 0.05-0.1 | LoRA 层的 dropout |

**经验法则**:
- `r=16, alpha=32` 是很好的起点
- 对 attention 层 (q, k, v, o) + FFN 层 (gate, up, down) 都加 LoRA 效果最好
- rank 太小会欠拟合，太大浪费资源且可能过拟合

### 5.1.3 代码实现

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

> 库: [huggingface/peft](https://github.com/huggingface/peft)

### 5.1.4 LoRA 合并

训练完后可以把 LoRA 权重合并回原模型，推理时零额外开销：

```python
merged_model = model.merge_and_unload()
# W_new = W + A @ B
# 推理时和全量微调的模型完全相同
```

## 5.2 QLoRA

([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) — **让单卡 24GB GPU 也能微调 70B 模型。**

### 5.2.1 核心创新

在 LoRA 基础上加入三个内存优化：

1. **4-bit NormalFloat (NF4) 量化**: 把 frozen 参数量化到 4-bit
2. **Double Quantization**: 量化 scaling factor 本身（再省内存）
3. **Paged Optimizers**: 用 CPU 内存作为 GPU 内存的交换空间

```
内存对比 (LLaMA 65B):
全量微调: ~780GB (需要多节点)
LoRA FP16: ~130GB (2× A100 80GB)
QLoRA 4-bit: ~48GB (1× A100 80GB!)
```

### 5.2.2 使用

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

# 然后像普通 LoRA 一样加 adapter
model = get_peft_model(model, lora_config)
```

## 5.3 其他 PEFT 方法

| 方法 | 原理 | 参数量 | 论文 |
|------|------|--------|------|
| **Prefix Tuning** | 在每层 attention 前加可训练 prefix tokens | 0.1% | [Li & Liang, 2021](https://arxiv.org/abs/2101.00190) |
| **Prompt Tuning** | 在输入前加可训练 soft prompts | 0.01% | [Lester et al., 2021](https://arxiv.org/abs/2104.08691) |
| **Adapter** | 在每层 Transformer 后插入小型 MLP | 1-5% | [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751) |
| **IA3** | 学习对 K, V, FFN 的缩放向量 | 0.01% | [Liu et al., 2022](https://arxiv.org/abs/2205.05638) |
| **DoRA** | LoRA + 权重幅度分解 | ~LoRA | [Liu et al., 2024](https://arxiv.org/abs/2402.09353) |

**实际选择**: 绝大多数场景用 LoRA 或 QLoRA 即可。其他方法在特定场景有优势但生态支持不如 LoRA。

## 5.4 PEFT 实战建议

**什么时候用 PEFT vs 全量微调**:
- 数据量 < 100K：PEFT（全量微调容易过拟合）
- 数据量 > 1M + 大预算：全量微调
- 预算有限（单卡/几卡）：QLoRA
- 需要切换多个任务：LoRA（可以同时加载多个 adapter）

**常见工具**:
- [huggingface/trl](https://github.com/huggingface/trl) — SFT, DPO, PPO + LoRA
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl) — 一站式微调（支持 LoRA/QLoRA + 各种格式）
- [unsloth](https://github.com/unslothai/unsloth) — 2x 加速 LoRA 微调（手写 kernel）
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 中文社区常用的微调框架

---

[← 上一章](04-post-training.md) | [目录](../README.md) | [下一章 →](06-infra.md)
