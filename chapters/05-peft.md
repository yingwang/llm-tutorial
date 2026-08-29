[← 上一章](04-post-training.md) | [目录](../README.md) | [下一章 →](06-infra.md)

---

# 第五章：参数高效微调 (PEFT)

全参数微调（Full Fine-Tuning）70B 级别大模型需消耗近 1TB 显存用于存储模型参数、AdamW 优化器状态、中间激活值与反向梯度。参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）通过冻结基座主体权重并引入轻量级可训练适配层，将显存与存储开销压缩达数个数量级，成为大模型领域适配与定制的核心范式。

## 5.1 低秩适配 (Low-Rank Adaptation, LoRA)

([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) 构成了目前工业界最成熟、应用最广泛的 PEFT 机制。

### 5.1.1 数学机理与本征低秩假设

**核心假设**：神经网络在向特定下游任务迁移适配时，其权重更新矩阵 $\Delta W$ 具有极低的本征维度（Intrinsic Rank），无需在全量高维空间进行密集更新。

**前向计算与低秩分解**：
```
原始稠密前向:  h = W_0 x             (W_0 ∈ ℝ^(d×k) 保持冻结)
LoRA 适配前向: h = W_0 x + ΔW x = W_0 x + (α / r) · B A x
               其中 A ∈ ℝ^(r×k), B ∈ ℝ^(d×r), 秩 r << min(d, k)
```

**参数量对比**（以 $d = k = 4096, r = 16$ 为例）：
- 全参数微调：$4096 \times 4096 \approx 16.7\text{M}$ 参数；
- LoRA 旁路：$4096 \times 16 + 16 \times 4096 = 131\text{K}$ 参数（参数量压缩逾 120 倍）。

**初始化设计**：矩阵 $A$ 采用高斯分布 $\mathcal{N}(0, \sigma^2)$ 初始化，矩阵 $B$ 初始化为全 0。这确保在训练伊始 $\Delta W = BA = 0$，模型完全等价于未微调的基座输出，避免初始扰动破坏预训练表征。

### 5.1.2 核心超参数配置

| 超参数 | 工业典型值 | 作用机理与调优指导 |
|-------|-----------|------------------|
| 秩 `r` (Rank) | 8–64 | 控制适配子空间的自由度；通常 16–32 即可满足复杂推理与指令对齐 |
| 缩放因子 `alpha` | 16–64 | 决定适配增量的缩放强度，实际缩放倍率为 $\alpha / r$；常用经验为令 $\alpha = 2r$ |
| 目标模块 `target_modules` | 全线性层 (all-linear) | 对 Attention (q, k, v, o) 与 FFN (gate, up, down) 全量施加 LoRA，效果最佳 |
| 丢弃率 `dropout` | 0.05–0.1 | 作用于低秩分支输入，增强模型泛化并抑制小样本过拟合 |

### 5.1.3 工业级代码实现

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

> 官方核心库：[huggingface/peft](https://github.com/huggingface/peft)

### 5.1.4 权重折叠与推理零时延

在训练完成后，可在线性层面上将低秩增量矩阵与基座原始权重直接解析相加，完全消除推理期的额外计算延迟与分支寻址：

```python
# 权重无损折叠
merged_model = model.merge_and_unload()
# W_new = W_0 + (alpha / r) * (B @ A)
# 折叠后的模型在结构与推理延迟上与原生全参数模型完全一致
```

## 5.2 量化低秩适配 (QLoRA)

([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) 通过极限显存压缩技术，使消费级单卡运行 70B 级别大模型微调成为可能。

### 5.2.1 三大核心优化

1. **4-bit NormalFloat (NF4) 数据类型**：针对正态分布权重构建信息论意义上的最优等概率量化分位数，相较传统 INT4 具备更高的信息保真度；
2. **双重量化 (Double Quantization, DQ)**：对量化基座所需的缩放因子（Scaling Factors）再次实施 8-bit 量化，每参数显存占用进一步节省约 0.37 字节；
3. **分页优化器 (Paged Optimizers)**：利用 CUDA 统一内存机制，在显存峰值突发时将优化器状态自动换出至 CPU 内存，彻底规避 OOM 崩溃。

```
显存开销对比 (LLaMA 65B 模型):
全参数微调: ~780 GB (需多节点 8× A100 集群)
标准 LoRA FP16: ~130 GB (需 2× A100 80GB)
QLoRA 4-bit: ~48 GB (仅需单张 A100 80GB 即可全流程微调)
```

### 5.2.2 工程调用范式

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
    device_map="auto"
)

# 挂载 LoRA 适配层并执行微调
model = get_peft_model(model, lora_config)
```

## 5.3 其他主流 PEFT 技术演进

| 微调方法 | 拓扑机制 | 可训练参数比例 | 核心文献 |
|---------|---------|--------------|---------|
| **Prefix Tuning** | 在每层注意力键值张量前拼接可学习虚拟 Prefix 前缀 | ~0.1% | [Li & Liang, 2021](https://arxiv.org/abs/2101.00190) |
| **Prompt Tuning** | 仅在输入 Embedding 层前缀拼接连续软提示向量 | ~0.01% | [Lester et al., 2021](https://arxiv.org/abs/2104.08691) |
| **Adapter** | 在自注意力层与 FFN 层之后串行插入瓶颈 MLP 模块 | 1%–5% | [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751) |
| **IA³** | 引入可学习的逐通道缩放向量对 K、V 及 FFN 激活进行调制 | ~0.01% | [Liu et al., 2022](https://arxiv.org/abs/2205.05638) |
| **DoRA** | 将权重分解为幅值（Magnitude）与方向（Direction）分别适配 | 与 LoRA 相当 | [Liu et al., 2024](https://arxiv.org/abs/2402.09353) |

**工程选型共识**：LoRA 与 QLoRA 凭借优秀的生态兼容性、零推理延迟折叠能力以及极高的表征容量，已确立为现代大模型适配的事实标准。

## 5.4 工业实践与微调策略

**全参数微调 vs PEFT 选型决策**：
- **语料规模 < 100K 样本**：优先采用 LoRA/QLoRA（全参数微调极易陷入过拟合与灾难性遗忘）；
- **领域持续预训练 / 海量数据（> 1M 样本）**：具备充沛算力时推荐全参数微调；
- **多租户业务并发与热切换**：采用单一冻结基座，线上依据请求租户动态挂载不同的轻量 LoRA 权重；
- **硬件资源受限**：单卡或消费级显卡场景下首选 QLoRA。

**成熟工业框架推荐**：
- [huggingface/trl](https://github.com/huggingface/trl)：覆盖 SFT、DPO 与 RLHF 全流程的 PEFT 原生集成；
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl)：高度工程化的全功能大模型微调工具链；
- [unsloth](https://github.com/unslothai/unsloth)：深度重写 Triton 与 CUDA 算子，实现数倍加速与显存节省；
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：功能全面的 WebUI/命令行大模型微调一体化平台。

## 关键论文

- [Hu et al. (2021) — LoRA](https://arxiv.org/abs/2106.09685): 低秩适配基础奠基论文
- [Dettmers et al. (2023) — QLoRA](https://arxiv.org/abs/2305.14314): 4-bit 量化与分页优化器微调
- [Houlsby et al. (2019) — Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751): Adapter 拓扑结构首创
- [Li & Liang (2021) — Prefix-Tuning](https://arxiv.org/abs/2101.00190): 连续提示前缀调优
- [Liu et al. (2024) — DoRA](https://arxiv.org/abs/2402.09353): 权重方向与幅值解耦微调

## 进阶参考

- HuggingFace: [PEFT 官方文档与代码库](https://github.com/huggingface/peft)
- Sebastian Raschka: [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- Tim Dettmers: [QLoRA 官方实现仓库](https://github.com/artidoro/qlora)

## 实践训练

1. **LoRA 适配层实战**：使用 PEFT 库对 1.5B 开源基座在标准指令集上进行微调（设定 $r=16, \alpha=32$），对比其与全参数微调在峰值显存占用与下游准确率上的差异。
2. **秩参数超参扫描实验**：在固定训练数据集上扫描 $r \in \{4, 16, 64, 128\}$，绘制收敛损失曲线与显存开销曲线，探寻最优性能拐点。
3. **多任务 Adapter 动态融合**：分别针对代码生成与中英翻译任务训练两个独立的 LoRA 适配层，借助 mergekit 探索线性加权合并与 TIES 合并后的跨任务综合能力。

---

[← 上一章](04-post-training.md) | [目录](../README.md) | [下一章 →](06-infra.md)
