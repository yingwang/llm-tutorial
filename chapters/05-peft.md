[← 上一章](04-post-training.md) | [目录](../README.md) | [下一章 →](06-infra.md)

# 第五章：参数高效微调 (PEFT)

如果对一个 70B 参数的模型做全参数微调（Full Fine-Tuning），光模型权重、梯度、AdamW 优化器状态和中间激活值加起来就要占近 1TB 显存，至少得动用两台 8 卡 80GB A100/H100 服务器。参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）的思路很直接：把原模型 99% 以上的权重冻结住，只训练极小一部分新增或旁路参数，从而用单卡甚至消费级显卡完成大模型微调。

## 5.1 LoRA (Low-Rank Adaptation)

（[Hu et al., 2021](https://arxiv.org/abs/2106.09685)）目前应用最广、生态支持最完善的 PEFT 方法。

### 5.1.1 核心思想

预训练模型已经学到了丰富的通用表示，在适配下游任务时，权重的变化量 $\Delta W$ 其实并不需要很高的自由度（即具有较低的“本征秩” Intrinsic Rank）。因此，LoRA 把原本的权重矩阵 $W_0$ 冻结起来，在旁边并联两个一瘦一扁的低秩矩阵 $A$ 和 $B$：

```
# 原始线性层: Y = X @ W_0                    (W_0 ∈ ℝ^{d×d} 冻结不动)
# 加入 LoRA:  Y = X @ W_0 + (α / r) · X @ A @ B
#             A ∈ ℝ^{d×r}, B ∈ ℝ^{r×d}, 秩 r << d

# 参数量对比 (以 d=4096, r=16 为例):
# 全量单矩阵: 4096 × 4096 = 16.7M 参数
# LoRA 旁路:  4096 × 16 + 16 × 4096 = 131K 参数 (缩小了 128 倍)
```

**为什么 $A$ 用高斯初始化、$B$ 用全 0 初始化**：
开始训练时，矩阵 $A$ 用正态分布 $\mathcal{N}(0, \sigma^2)$ 随机初始化，而矩阵 $B$ 初始化为全 0。这样在第 0 步时，旁路增量 $\Delta W = A B = 0$，整个模型的输出和没加 LoRA 的原模型完全一致，不会在一上来就用随机噪声把预训练特征冲乱。

### 5.1.2 关键超参

| 超参 | 常用值 | 说明 |
|------|--------|------|
| `r` (rank) | 8-64 | 低秩矩阵的秩；越大表达能力越强，但可训练参数和显存也随之增加 |
| `alpha` | 16-64 | 旁路输出缩放系数，实际乘在旁路上的系数是 `alpha / r` |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | 对哪些线性层挂载 LoRA |
| `dropout` | 0.05-0.1 | 加在 LoRA 分支输入上的 Dropout，防止小数据集过拟合 |

**实用经验**：
- `r=16, lora_alpha=32`（即保持 `alpha = 2 * r`）是最稳妥的起点；
- 早期很多教程只给注意力的 `q_proj` 和 `v_proj` 加 LoRA，但大量实验表明，对注意力层（`q, k, v, o`）和 FFN 层（`gate, up, down`）的**所有线性层**都加上 LoRA，效果明显更好，最接近全参数微调；
- 如果微调任务只是调整回复格式或语气，`r=8` 就够了；如果要在微调中灌入大量新领域知识，需要调高 `r`（如 64–128）甚至直接做全量持续预训练。

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

> 常用库：[huggingface/peft](https://github.com/huggingface/peft)

### 5.1.4 LoRA 合并

因为旁路计算 $(\alpha / r) X A B$ 是纯线性的，训练结束后可以直接把 $A @ B$ 算出来加回到原权重矩阵里。合并后的模型在推理时没有任何旁路分支，推理延迟与原模型完全一样：

```python
merged_model = model.merge_and_unload()
# W_new = W_0 + (alpha / r) * (A @ B)
# 推理时和全量微调的模型完全相同，零额外开销
```

## 5.2 QLoRA

（[Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)）普通 LoRA 虽然省掉了绝大多数梯度和优化器状态，但冻结的基座模型本身如果用 FP16/BF16 加载，65B/70B 模型光放权重就要占 130GB–140GB 显存。QLoRA 把冻结基座进一步压成 4-bit，使单张 48GB 甚至 24GB 显卡也能微调数十亿到上百亿参数的模型。

### 5.2.1 三项显存优化

1. **4-bit NormalFloat (NF4) 量化**：预训练好的神经网络权重天然近似服从正态分布。NF4 按标准正态分布的等概率分位数来划分 16 个量化桶，在权重密集的 0 附近刻度更细、两端刻度更粗，比均匀间隔的普通 INT4 保留了更多精度；
2. **双重量化（Double Quantization）**：分块量化本身会产生一批 FP32 的缩放系数（Scaling Factors），QLoRA 把这批缩放系数再做一次 8-bit 量化，每个参数还能再省下约 0.37 bit 显存；
3. **分页优化器（Paged Optimizers）**：借助 NVIDIA 统一内存机制，遇到长序列样本导致激活值显存瞬间冲高时，自动把优化器状态临时换出到 CPU 内存，算完再换回来，避免进程直接因 OOM 崩溃。

> 注意：训练时只有冻结的基座权重是以 4-bit 存储的；每次前向和反向传播算矩阵乘时，会把 4-bit 权重实时反量化成 BF16，与 BF16 的 LoRA 旁路一起计算。

```
显存对比 (以 LLaMA 65B 为例):
全量微调:   ~780GB (需要多机多卡)
LoRA FP16:  ~130GB (需要 2× A100 80GB)
QLoRA 4-bit: ~48GB (单张 A100 80GB 即可跑通)
```

### 5.2.2 使用

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

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

# 然后像普通 LoRA 一样挂载 adapter
model = get_peft_model(model, lora_config)
```

## 5.3 其他 PEFT 方法

| 方法 | 原理 | 可训练参数占比 | 论文 |
|------|------|---------------|------|
| **Prefix Tuning** | 在每层注意力的 K、V 前拼接可学习的虚拟前缀向量 | ~0.1% | [Li & Liang, 2021](https://arxiv.org/abs/2101.00190) |
| **Prompt Tuning** | 只在输入 Embedding 层前面拼几个可学习的 Soft Prompt 向量 | ~0.01% | [Lester et al., 2021](https://arxiv.org/abs/2104.08691) |
| **Adapter** | 在每层 Transformer 子层后面串行插入小型瓶颈 MLP | 1%-5% | [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751) |
| **IA³** | 学习三个通道缩放向量，直接乘在 K、V 和 FFN 激活值上 | ~0.01% | [Liu et al., 2022](https://arxiv.org/abs/2205.05638) |
| **DoRA** | 把权重拆成“幅度（Magnitude）”和“方向（Direction）”，对方向用 LoRA 更新 | ~LoRA | [Liu et al., 2024](https://arxiv.org/abs/2402.09353) |

**实际怎么选**：绝大多数工程场景直接用 **LoRA** 或 **QLoRA** 即可（如果追求极限效果可以试试 `use_dora=True` 开启 DoRA）。Adapter 因为是串行插在层里的，合并不了权重，会增加推理延迟；Prefix/Prompt Tuning 则会占掉一部分上下文窗口长度，而且小规模下不如 LoRA 好训。

## 5.4 PEFT 实战建议

**什么时候用 PEFT，什么时候用全量微调**：
- **数据量较少（< 100K 条指令）**：优先用 LoRA / QLoRA，不容易过拟合，也能较好地保留基座原本的通用能力；
- **注入大量新领域知识或数据量很大（> 1M 条）**：如果算力允许，全量微调（Full Fine-Tuning）的上限更高；
- **硬件预算紧张（单卡或几张消费级显卡）**：用 QLoRA；
- **一个基座服务几十个不同垂直业务**：用 LoRA 最划算。推理引擎（如 vLLM、SGLang）支持在显存里只常驻一份基座模型，同时动态加载几十个只有几十 MB 的 LoRA Adapter，按请求路由到对应的 Adapter。

**常用微调工具**：
- [huggingface/trl](https://github.com/huggingface/trl)：官方后训练库，SFT、DPO、GRPO 都原生支持一键开启 PEFT；
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)：配置驱动的一站式微调工具，支持各种数据格式、Multipacking 与分布式加速；
- [unsloth](https://github.com/unslothai/unsloth)：用 Triton 手写了反向传播与 RoPE/Cross-Entropy 算子，单卡微调 LoRA/QLoRA 速度大约能快 2 倍、显存省 60% 以上；
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：中文社区广泛使用的微调框架，提供命令行和 WebUI 界面。

## 关键论文

- [Hu et al. (2021): LoRA (Low-Rank Adaptation of Large Language Models)](https://arxiv.org/abs/2106.09685)：提出低秩旁路适配，奠定现代 PEFT 标准
- [Dettmers et al. (2023): QLoRA (Efficient Finetuning of Quantized LLMs)](https://arxiv.org/abs/2305.14314)：提出 NF4 量化、双重量化与分页优化器
- [Houlsby et al. (2019): Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)：早期串行 Adapter 结构代表作
- [Li & Liang (2021): Prefix-Tuning](https://arxiv.org/abs/2101.00190)：在注意力层注入可学习的连续前缀向量
- [Liu et al. (2024): DoRA (Weight-Decomposed Low-Rank Adaptation)](https://arxiv.org/abs/2402.09353)：解耦权重幅度与方向的改进版 LoRA

## 进一步阅读

- HuggingFace：[PEFT 官方文档](https://github.com/huggingface/peft)，统一封装了 LoRA、DoRA、IA³、Prefix Tuning 等所有主流方法
- Sebastian Raschka：[Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)，包含大量关于 Rank、Alpha 和目标层选择的消融实验总结
- John Schulman et al. (Thinking Machines Lab, 2025)：*LoRA Without Regrets*，系统分析了在后训练与强化学习中如何配置 LoRA 使其逼近全参数微调效果

## 练习题

1. **LoRA vs 全量微调对比**：使用 `peft` 库给 Qwen2.5-1.5B 加上 LoRA（`r=16, alpha=32`），在 Alpaca 数据集上微调 1 个 Epoch，对比它与全参数微调在峰值显存占用、训练耗时和验证集 Loss 上的差异。
2. **Rank 与挂载层扫描**：固定训练数据，分别测试“只挂 `q_proj, v_proj`”与“挂全部 7 个线性层”，以及扫描 `r ∈ {4, 16, 64}`，画出验证集 Loss 变化曲线。
3. **多 LoRA 权重合并**：分别训练两个不同任务的 LoRA Adapter（例如一个学古文翻译，一个学 JSON 格式抽取），尝试将它们的低秩矩阵按不同权重加权合并回同一个基座，观察两项能力能否共存。

---

[← 上一章](04-post-training.md) | [目录](../README.md) | [下一章 →](06-infra.md)
