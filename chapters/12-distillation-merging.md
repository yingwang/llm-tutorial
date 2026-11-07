[← 上一章](11-sota-models.md) | [目录](../README.md) | [下一章 →](13-roadmap.md)

# 第十二章：知识蒸馏与模型合并

线上部署时，671B 或 70B 的大模型效果虽好，但延迟高、机器贵；而从头训练一个 7B 小模型又很难达到同样的聪明程度。知识蒸馏（Knowledge Distillation）通过“大模型带小模型”把能力传给小尺寸基座；模型合并（Model Merging）则干脆连显卡训练都省了，直接把几个同源微调模型的权重按数学规则拼到一起。

## 12.1 知识蒸馏（Knowledge Distillation）

知识蒸馏最早由 [Hinton et al. (2015)](https://arxiv.org/abs/1503.02531) 提出。它的直观想法是：拿标准答案（Hard Labels，即独热编码）训小模型，小模型只知道“正确答案是 A”，却不知道“B 虽然不对但沾边，而 C 完全离谱”。而教师模型（Teacher）输出的完整词表概率分布（Soft Labels）恰好包含了这些词汇之间的远近关系。

### 12.1.1 经典蒸馏损失函数

```python
# 标准知识蒸馏损失 (Hinton 经典公式)
# 总损失 = α * 硬标签交叉熵 + (1 - α) * 软标签 KL 散度 * T²
L_KD = α * CrossEntropy(student_logits, hard_labels) + \
       (1 - α) * (T ** 2) * KL_Div(
           softmax(student_logits / T),
           softmax(teacher_logits / T)
       )

# T 是温度参数 (Temperature，通常取 2 ~ 20):
# T 越大，softmax 出来的概率分布越平缓，原本概率很小的次优候选词也会露出来，方便学生模型学习
```

### 12.1.2 大模型常用的四种蒸馏方式

在自回归大模型上，蒸馏按能不能拿到教师模型的内部概率（Logits）以及怎么采样，分为四类：

| 方法 | 做法 | 优缺点与适用场景 | 代表工作 |
|------|------|------|---------|
| **白盒 Logit 蒸馏** | 要求学生模型在每个 token 位置输出的概率分布去逼近教师模型的词表概率分布 | 信息最密、效果最好，但要求师生模型词表一致（或做词表对齐），且需同时把大模型装进显存算前向（Gemma 2 9B/27B 就全程用此法预训练） | 经典 KD, [MiniLLM](https://arxiv.org/abs/2306.08543) |
| **On-Policy 蒸馏** | 先让学生模型自己生成一段话，再让教师模型在这段话的每个位置上给出正确分布指导 | 解决离线蒸馏中“训练时看老师写的句子、上线后自己写错一步就步步错”（Exposure Bias）的问题 | [GKD](https://arxiv.org/abs/2306.13649) |
| **黑盒合成数据蒸馏** | 调教师模型的 API 批量生成问答对，学生模型直接拿这些文本做普通 SFT | 最简单，不需要大模型权重，也不要求同词表；早期的 Alpaca、Vicuna 全靠这招起家 | Alpaca, UltraChat |
| **长思维链蒸馏（CoT Distillation）** | 让推理大模型（如 DeepSeek-R1、o1）生成带完整推导、试错与反思的解题过程，拿来微调小模型 | 把大模型的慢思考习惯直接“手把手”教给小模型 | [DeepSeek-R1-Distill 系列](https://arxiv.org/abs/2501.12948) |

**DeepSeek-R1 蒸馏经验**：DeepSeek 团队做过一个直接对比：拿一个 32B 的基座模型，一条路是让它自己从零跑强化学习摸索解题，另一条路是直接拿 671B 的 R1 生成的 80 万条长思维链样本做简单 SFT 蒸馏。结果发现，直接蒸馏出来的 `DeepSeek-R1-Distill-Qwen-32B` 不仅成本低得多，数学和代码分数还大幅碾压了小模型自己硬跑 RL 的结果。这说明小模型自己从头探索极难，但顺着大模型探好的路去模仿学习却非常快。

## 12.2 模型合并（Model Merging）

如果你手头有三个从同一个基座（比如 `Llama-3-8B`）微调出来的模型：一个擅长写代码，一个擅长算数学，一个擅长写小说。能不能一分钱显卡费都不花，直接把它们的权重加权平均一下，合成一个三样都会的模型？

这听起来很违反直觉（毕竟神经网络是非线性的），但实验证明：只要这些模型是从**同一个预训练基座**微调出来的，它们在参数空间里依然待在同一个低损失盆地（Loss Basin）里，直接对权重做加减法居然真的能把各自的能力拼到一起。

### 12.2.1 常用合并算法

记预训练基座权重为 $W_{\text{base}}$，某个微调模型相对于基座的权重变化量为任务向量（Task Vector）$\Delta W_i = W_i - W_{\text{base}}$。

| 方法 | 原理与步骤 | 解决什么问题 | 论文 |
|------|------|------|------|
| **Linear（线性加权）** | $W_{\text{new}} = \sum_i \alpha_i W_i$ | 最简单的加权平均，合并两三个相近模型时够用 | [Model Soups (2022)](https://arxiv.org/abs/2203.05482) |
| **SLERP（球面线性插值）** | 不走直线，而是沿着高维球面弧线在两个模型权重之间做等角插值 | 避免高维向量直接求平均导致模长缩水，是合并两个模型时最稳的默认方法 | - |
| **TIES-Merging** | 1. **Trim**：把 $\Delta W$ 里变化微小的噪音置零（只留前 20%）；<br>2. **Elect Sign**：统计每个参数位置上正负号哪边总和更大，少数服从多数；<br>3. **Merge**：只把符号一致的模型增量求平均 | 解决多个模型在同一个参数上一正一负互相抵消（符号冲突）的问题 | [Yadav et al. (2023)](https://arxiv.org/abs/2306.01708) |
| **DARE（Drop And Rescale）** | 随机把微调增量 $\Delta W$ 里 90% 甚至 95% 的参数直接丢弃置零，再把剩下的参数乘以 $\frac{1}{1-p}$ 放大回去 | 发现微调增量极度冗余，随机扔掉 90% 后再合并，不同模型之间几乎不会互相打架 | [Yu et al. (2024)](https://arxiv.org/abs/2311.03099) |
| **Model Soups（模型汤）** | 自己微调时用不同随机种子或超参存下几个 Checkpoint，最后把它们直接求平均 | 不增加任何推理耗时，白捡 1-2 个点的泛化准确率与鲁棒性 | [Wortsman et al. (2022)](https://arxiv.org/abs/2203.05482) |

### 12.2.2 使用 mergekit 合并模型

开源社区最常用的合并工具是 [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)，甚至不需要高端 GPU，在普通 CPU 内存上就能跑完。

```yaml
# mergekit 配置文件示例 (config.yaml): 用 DARE-TIES 合并三个 Llama-3-8B 微调模型
models:
  - model: meta-llama/Llama-3-8B-Instruct
    # 作为 base_model 计算差值，自身不设微调权重
  - model: abacusai/Llama-3-Smaug-8B
    parameters:
      weight: 0.4
      density: 0.5   # 保留 50% 的增量参数
  - model: cognitivecomputations/dolphin-2.9-llama3-8b
    parameters:
      weight: 0.3
      density: 0.5
  - model: NousResearch/Hermes-3-Llama-3-8B
    parameters:
      weight: 0.3
      density: 0.5
merge_method: dare_ties
base_model: meta-llama/Llama-3-8B-Instruct
dtype: bfloat16
```

运行合并命令：
```bash
pip install mergekit
mergekit-yaml config.yaml ./merged-llama3-8b --cuda
```

## 关键论文

- [Hinton et al. (2015): Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)：提出用温度参数软化输出概率来做知识蒸馏的开山之作。
- [Sanh et al. (2019): DistilBERT](https://arxiv.org/abs/1910.01108)：把 12 层 BERT 蒸馏成 6 层小模型的经典工程实践。
- [Wortsman et al. (2022): Model Soups](https://arxiv.org/abs/2203.05482)：展示将多个微调 Checkpoint 直接平均即可提升泛化性能。
- [Gu et al. (2023): MiniLLM (Knowledge Distillation of Large Language Models)](https://arxiv.org/abs/2306.08543)：指出自回归生成蒸馏时用反向 KL 散度（Reverse KL）比标准前向 KL 更能防止学生模型往低概率空白区乱猜。
- [Yadav et al. (2023): TIES-Merging](https://arxiv.org/abs/2306.01708)：通过修剪微小增量与投票统一符号来消除多模型合并冲突。
- [Yu et al. (2024): DARE (Language Models are Super Mario)](https://arxiv.org/abs/2311.03099)：发现微调参数增量可被随机丢弃 90% 以上而不伤性能，大幅降低了多模型合并时的干扰。

## 进一步阅读

- [mergekit 仓库与文档](https://github.com/arcee-ai/mergekit)：涵盖 Linear、SLERP、TIES、DARE、Passthrough（Frankenmerging）等所有主流合并方法的配置说明。
- HuggingFace Blog: [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models)：手把手教你用 mergekit 融合模型。

## 练习题

1. **知识蒸馏对比**：拿 `Qwen2.5-7B-Instruct` 在 GSM8K 训练集上生成 2000 条带详细解题步骤的回答，用它微调 `Qwen2.5-0.5B`，对比直接用原始简短答案微调的效果差异。
2. **Model Soup 实验**：在同一个微调任务上，用不同学习率或不同数据打乱顺序训出 3 个 LoRA 权重，把它们按 $1:1:1$ 平均合并，看测试集准确率是否比单拿其中任何一个都高。
3. **DARE-TIES 合并实战**：在 HuggingFace 上找两个同基于 `Qwen2.5-7B`（或 `Llama-3.1-8B`）微调的垂直模型（比如一个数学模型、一个中文写作模型），用 `mergekit` 的 `slerp` 和 `dare_ties` 分别合并，测试合并后模型在两项任务上的保留程度。

---

[← 上一章](11-sota-models.md) | [目录](../README.md) | [下一章 →](13-roadmap.md)
