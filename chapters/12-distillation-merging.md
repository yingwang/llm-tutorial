[← 上一章](11-sota-models.md) | [目录](../README.md) | [下一章 →](13-roadmap.md)

---

# 第十二章：知识蒸馏与模型合并

在算力与显存受限的落地场景下，如何将超大规模旗舰模型的深厚推理能力低成本迁移至轻量级端侧设备，或是将多个领域专精模型的知识在权重空间无缝融合，构成了大模型工程学中的关键命题。知识蒸馏（Knowledge Distillation）与模型权重合并（Model Merging）分别从“行为特征迁移”与“参数流形叠加”两个正交维度，提供了无需巨额重训成本的高效解法。

## 12.1 知识蒸馏 (Knowledge Distillation)

([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) 奠定了通过教师模型软概率分布传递“暗知识”（Dark Knowledge）的基础范式。

### 12.1.1 经典蒸馏目标与数学机理

```python
# 标准知识蒸馏损失函数 (Hinton 经典公式)
# 损失 = α * 真实硬标签交叉熵 + (1 - α) * 软标签 KL 散度 * T²
L_KD = α * CrossEntropy(student_logits, hard_labels) + \
       (1 - α) * (T**2) * KL_Div(
           softmax(student_logits / T),
           softmax(teacher_logits / T)
       )

# 温度超参数 T (通常设为 2.0 ~ 20.0):
# 提升温度 T 能够平滑 Logits 输出分布，显式放大负标签之间的细微概率差异（暗知识）
```

### 12.1.2 大语言模型蒸馏演进路径

| 蒸馏范式 | 技术实现机理 | 核心优势与适用场景 | 代表工作 |
|---------|------------|------------------|---------|
| **白盒 Logit 蒸馏** | 严格对齐 Student 与 Teacher 在词表上的每步输出概率分布 | 信息密度极高，同时学习各 Token 间的相对语义关系 | 经典 KD, [MiniLLM](https://arxiv.org/abs/2306.08543) |
| **在线策略蒸馏 (On-Policy)** | Student 自主生成序列，Teacher 针对其生成的 Token 进行评分重加权 | 规避离线蒸馏下的分布偏移（Distribution Shift） | [GKD](https://arxiv.org/abs/2306.13649) |
| **黑盒合成数据蒸馏** | 仅依赖 Teacher 生成高质量指令与复杂多轮解答，作为 Student 的 SFT 语料 | 无需 Teacher 的内部权重或 Logits，工程实现极其简洁 | Alpaca, UltraChat |
| **思维链轨迹蒸馏 (CoT Distillation)** | 将 Teacher 生成的长链推理、反思与回溯轨迹直接用于 Student 微调 | 显著激活小模型的深层多步推理潜能 | [DeepSeek-R1 蒸馏系列](https://arxiv.org/abs/2501.12948) |

**DeepSeek-R1 蒸馏启示**：直接利用 671B 旗舰模型输出的高质量推理轨迹（Reasoning Traces）对 Qwen 与 LLaMA 小尺寸基座进行 SFT，所获得的 1.5B 至 14B 蒸馏模型在数学与代码基准上大幅超越了同尺寸下从零进行纯 RL 探索的收益，展现出高阶推理知识的高度可压缩性。

## 12.2 模型权重合并 (Model Merging)

无需引入额外的 GPU 训练算力，直接在参数空间通过线性代数变换与几何插值将多个同源或异构模型的权重融合为一个全能模型。

### 12.2.1 核心算法机理

```
基本假设 (Mode Connectivity): 
在同一基座上经过不同任务微调的模型，其参数依然处于相同的低损失盆地（Loss Basin）附近，
参数差异矩阵 (Task Vectors) 可以在向量空间中进行线性或流形叠加。
```

| 合并算法 | 数学与几何原理 | 核心优势与适用场景 | 核心文献 |
|---------|--------------|------------------|---------|
| **线性加权 (Linear)** | $W_{\text{new}} = \sum_i \alpha_i W_i$ | 计算极简，适合同分布模型的加权平滑 | 基础算术基线 |
| **球面线性插值 (SLERP)** | 在高维超球面上沿测地线进行角度插值：$\text{SLERP}(W_A, W_B; t)$ | 保持高维权重向量的几何模长不变，融合效果显著优于简单平面线性插值 | 经典图形学与流形插值 |
| **TIES-Merging** | 1. 剪除微小扰动参数（Trim）；2. 解决参数符号冲突（Elect Sign）；3. 仅对同符号增量求均值（Disjoint Merge） | 彻底消除跨任务合并时的参数方向冲突与干扰抵消 | [Yadav et al., 2023](https://arxiv.org/abs/2306.01708) |
| **DARE (Drop And Rescale)** | 随机丢弃 90% 的微调增量参数（$\Delta W$），并将剩余参数按比例放大重缩放 | 极大增强多任务参数的稀疏性与独立性，可高保真合并数十个异构领域模型 | [Yu et al., 2024](https://arxiv.org/abs/2311.03099) |
| **模型汤 (Model Soups)** | 在微调阶段使用不同超参/数据子集训练多个 Checkpoint，事后对权重取均匀或贪心平均 | 零额外推理开销下稳定提升模型鲁棒性与分布外泛化能力 | [Wortsman et al., 2022](https://arxiv.org/abs/2203.05482) |

### 12.2.2 工业级合并工具与实战配置

> [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) 构成了目前开源社区最主流的模型合并工具链。

```yaml
# mergekit 配置示例: 基于 DARE-TIES 融合通用、数学与代码模型
models:
  - model: meta-llama/Llama-3-8B-Instruct
    # 基准参考基座
  - model: abacusai/Llama-3-Smaug-8B
    parameters:
      weight: 0.4
      density: 0.6
  - model: cognitivecomputations/dolphin-2.9-llama3-8b
    parameters:
      weight: 0.3
      density: 0.6
  - model: NousResearch/Hermes-3-Llama-3-8B
    parameters:
      weight: 0.3
      density: 0.6
merge_method: dare_ties
base_model: meta-llama/Llama-3-8B-Instruct
dtype: bfloat16
```

**工程价值**：通过将通用对话模型、严密数学推理模型与代码辅助模型进行低成本合并，可在零训练卡时投入下打造出多任务综合能力领先的复合模型。

## 关键论文

- [Hinton et al. (2015) — Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531): 知识蒸馏奠基之作
- [Gu et al. (2023) — Knowledge Distillation of Large Language Models (MiniLLM)](https://arxiv.org/abs/2306.08543): 面向自回归生成模型的反向 KL 散度蒸馏
- [Wortsman et al. (2022) — Model Soups: Averaging Weights of Multiple Fine-Tuned Models](https://arxiv.org/abs/2203.05482): 权重平均提升泛化能力
- [Yadav et al. (2023) — Resolving Interference When Merging Models (TIES)](https://arxiv.org/abs/2306.01708): 消除参数符号冲突的结构化合并算法
- [Yu et al. (2024) — DARE: Language Models are Super Mario](https://arxiv.org/abs/2311.03099): 随机丢弃与重缩放的大模型合并范式

## 进阶参考

- [mergekit 官方代码库](https://github.com/arcee-ai/mergekit)（工业级全算法模型合并工具链）
- [DistilBERT 官方技术博客](https://medium.com/huggingface/distilbert-8cf3380435b5)（经典轻量化压缩工程范式）
- [Awesome Knowledge Distillation](https://github.com/dkozlov/awesome-knowledge-distillation)（知识蒸馏前沿论文与资源精选）

## 实践训练

1. **白盒与黑盒蒸馏对比**：以 Qwen2.5-7B-Instruct 为教师模型，分别采用 Logits 散度匹配与 SFT 文本生成两种方式在 0.5B 基座上实施蒸馏微调，评测二者在 GSM8K 上的推理表现。
2. **Model Soups 泛化性验证**：在固定指令微调任务中，使用 3 组不同的学习率和随机种子训练 3 个 Checkpoint，计算其算术平均权重模型并评测跨域鲁棒性。
3. **多任务 DARE-TIES 合并实战**：基于 mergekit 将一个代码微调模型与一个角色扮演微调模型进行 DARE-TIES 融合，测试合并后的新模型在 HumanEval 与开放对话基准上的综合能力保留率。

---

[← 上一章](11-sota-models.md) | [目录](../README.md) | [下一章 →](13-roadmap.md)
