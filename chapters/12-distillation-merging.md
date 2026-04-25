[← 上一章](11-sota-models.md) | [目录](../README.md) | [下一章 →](13-roadmap.md)

---

# 第十二章：知识蒸馏与模型合并

## 12.1 知识蒸馏 (Knowledge Distillation)

([Hinton et al., 2015](https://arxiv.org/abs/1503.02531))

**核心思想**: 用大模型 (teacher) 的 soft labels 训练小模型 (student)。

```python
# 标准 KD Loss
L = α * CE(student_logits, hard_labels) + (1-α) * KL(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T²

# T = temperature (通常 2-20)，让 soft labels 更"软"
# α = hard/soft label 的权重平衡
```

### LLM 蒸馏方法

| 方法 | 说明 | 例子 |
|------|------|------|
| **Logit distillation** | 学 teacher 的 output distribution | 经典 KD |
| **On-policy distillation** | student 生成 → teacher 打分 → 训练 student | GKD ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)) |
| **Synthetic data** | teacher 生成回答，student 当 SFT 数据 | Alpaca, Vicuna |
| **Reasoning distillation** | teacher 生成 CoT → student 学 CoT | DeepSeek-R1 蒸馏版 |

**DeepSeek-R1 蒸馏**: 用 R1 (671B) 的 reasoning traces 蒸馏出 1.5B-70B 的小模型，效果惊人地好。

## 12.2 模型合并 (Model Merging)

不额外训练，直接在权重空间合并多个模型。

### 12.2.1 方法

| 方法 | 原理 | 论文 |
|------|------|------|
| **Linear** | `W = α * W_A + (1-α) * W_B` | - |
| **SLERP** | 球面线性插值 | - |
| **TIES** | 修剪冲突参数后合并 | [Yadav et al., 2023](https://arxiv.org/abs/2306.01708) |
| **DARE** | 随机 drop 并 rescale delta weights | [Yu et al., 2024](https://arxiv.org/abs/2311.03099) |
| **Model Soups** | 平均多个 fine-tune checkpoint | [Wortsman et al., 2022](https://arxiv.org/abs/2203.05482) |

### 12.2.2 工具

> [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) — 最流行的模型合并工具

```yaml
# mergekit 配置示例 (YAML)
models:
  - model: base_model
    parameters:
      weight: 0.5
  - model: math_model
    parameters:
      weight: 0.3
  - model: code_model
    parameters:
      weight: 0.2
merge_method: linear
dtype: bfloat16
```

**实际用途**: 合并一个通用模型 + 一个数学模型 + 一个代码模型 → 得到一个三者兼备的模型，不需要额外训练。[Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) 上很多顶级模型是合并得到的。

---

[← 上一章](11-sota-models.md) | [目录](../README.md) | [下一章 →](13-roadmap.md)
