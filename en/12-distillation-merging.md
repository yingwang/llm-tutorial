[← Previous Chapter](11-sota-models.md) | [Table of Contents](README.md) | [Next Chapter →](13-roadmap.md)

---

# Chapter 12: Knowledge Distillation and Model Merging

## 12.1 Knowledge Distillation

([Hinton et al., 2015](https://arxiv.org/abs/1503.02531))

**Core idea**: Train a small model (student) using soft labels from a large model (teacher).

```python
# Standard KD Loss
L = α * CE(student_logits, hard_labels) + (1-α) * KL(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T²

# T = temperature (typically 2-20), makes soft labels "softer"
# α = weight balance between hard/soft labels
```

### LLM Distillation Methods

| Method | Description | Example |
|--------|-------------|---------|
| **Logit distillation** | Learn the teacher's output distribution | Classic KD |
| **On-policy distillation** | Student generates → teacher scores → train student | GKD ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)) |
| **Synthetic data** | Teacher generates responses, used as SFT data for student | Alpaca, Vicuna |
| **Reasoning distillation** | Teacher generates CoT → student learns CoT | DeepSeek-R1 distilled versions |

**DeepSeek-R1 Distillation**: Distilled small models (1.5B-70B) from R1 (671B) reasoning traces, with surprisingly strong results.

## 12.2 Model Merging

Merge multiple models directly in weight space, with no additional training.

### 12.2.1 Methods

| Method | Principle | Paper |
|--------|-----------|-------|
| **Linear** | `W = α * W_A + (1-α) * W_B` | - |
| **SLERP** | Spherical linear interpolation | - |
| **TIES** | Prune conflicting parameters, then merge | [Yadav et al., 2023](https://arxiv.org/abs/2306.01708) |
| **DARE** | Randomly drop and rescale delta weights | [Yu et al., 2024](https://arxiv.org/abs/2311.03099) |
| **Model Soups** | Average multiple fine-tune checkpoints | [Wortsman et al., 2022](https://arxiv.org/abs/2203.05482) |

### 12.2.2 Tools

> [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) — The most popular model merging tool

```yaml
# mergekit config example (YAML)
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

**Practical use**: Merge a general model + a math model + a code model to get a model good at all three, without any additional training. Many top models on the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) are produced this way.

---

[← Previous Chapter](11-sota-models.md) | [Table of Contents](README.md) | [Next Chapter →](13-roadmap.md)
