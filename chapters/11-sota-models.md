[← 上一章](10-safety.md) | [目录](../README.md) | [下一章 →](12-distillation-merging.md)

---

# 第十一章：SOTA 模型深度解析

## 11.1 LLaMA 3 / 3.1 / 3.3 (Meta)

> 论文: [LLaMA 3](https://arxiv.org/abs/2407.21783) | 权重: [meta-llama](https://huggingface.co/meta-llama)

```
参数: 8B, 70B, 405B
架构: Dense Transformer, GQA, RoPE, SwiGLU, RMSNorm
词表: 128K (扩大后的 tiktoken BPE)
上下文: 128K (通过 YaRN 扩展)
训练数据: 15T tokens
训练硬件: 16384 H100 GPUs
训练时间: ~54天 (405B)

关键创新:
- 超大词表 (128K) 大幅改善多语言和代码
- Over-training: 8B 模型用 15T tokens (远超 Chinchilla 最优)
- 3阶段 Post-training: SFT → Rejection Sampling → DPO
- 代码执行反馈: 用代码执行结果作为 reward
```

## 11.2 DeepSeek-V3 / R1

> 论文: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 权重: [deepseek-ai](https://huggingface.co/deepseek-ai)

```
DeepSeek-V3:
  参数: 671B (MoE, 37B 激活)
  架构: MLA + DeepSeekMoE (256 experts, top-8 + 1 shared)
  训练: 14.8T tokens, 2048 H800 GPUs
  成本: ~$5.5M (极低!)
  
  关键创新:
  - MLA: KV cache 压缩 93%
  - FP8 混合精度训练 (首次大规模成功)
  - Auxiliary-loss-free load balancing
  - Multi-Token Prediction (预测未来多个 token)

DeepSeek-R1:
  基于 V3，加了推理能力
  
  关键创新:
  - R1-Zero: 纯 RL (GRPO + 可验证 reward) → 涌现 CoT
  - R1: SFT (cold start) → RL → Rejection Sampling → SFT
  - 在 AIME 2024 上接近 OpenAI o1 水平
  - 完全开源 (权重 + 论文)
```

## 11.3 Claude 3/4 (Anthropic)

> 文档: [anthropic.com/claude](https://www.anthropic.com/claude)

```
架构: 未公开 (推测 dense transformer, cross-attention multimodal)
系列: Haiku (小) → Sonnet (中) → Opus (大)

关键特点:
- Constitutional AI: 基于原则的自我对齐
- 超长上下文: 200K tokens
- 强文档/代码理解
- 安全性领先
- Claude 4: 超强 coding, agentic 能力
```

## 11.4 Gemini 2 (Google)

> 论文: [Gemini](https://arxiv.org/abs/2312.11805) | [Gemini 1.5](https://arxiv.org/abs/2403.05530)

```
架构: 原生多模态 Transformer (MoE)
训练: TPU v5p/v6e 集群

关键特点:
- 原生多模态 (文本, 图像, 音频, 视频 一起预训练)
- 超长上下文: Gemini 1.5 支持 2M tokens
- Project Astra: 实时视频 + 音频理解
- Gemini 2 Flash: 极快推理
```

## 11.5 GPT-4 / o1 / o3 (OpenAI)

> 文档: [platform.openai.com](https://platform.openai.com/docs)

```
GPT-4:
  参数: ~1.8T (MoE, 8 experts, 推测)
  训练: ~13T tokens on ~25000 A100s

o1/o3 (reasoning):
  关键创新:
  - 大规模 RL 训练 (推测用了 PRM + MCTS)
  - Test-time compute scaling
  - Hidden chain-of-thought (用户不可见)
  - o3 在 ARC-AGI 上达到 87.5%
```

## 11.6 Qwen 2.5 / QwQ (Alibaba)

> 论文: [Qwen2](https://arxiv.org/abs/2407.10671) | 权重: [Qwen](https://huggingface.co/Qwen)

```
参数: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B + MoE
架构: GQA, RoPE, SwiGLU
词表: 152K
训练: 18T tokens

关键特点:
- 最全的开源模型家族 (覆盖所有规模)
- 强多语言 (中英日韩等)
- Qwen2-VL: 强视觉理解
- QwQ: 推理模型 (类 o1)
- Qwen-Agent: 工具使用框架
```

## 11.7 Mistral / Mixtral

> 论文: [Mistral 7B](https://arxiv.org/abs/2310.06825) | [Mixtral](https://arxiv.org/abs/2401.04088) | 权重: [mistralai](https://huggingface.co/mistralai)

```
Mistral 7B:
  创新: Sliding Window Attention, GQA
  效果: 7B 超过 LLaMA 2 13B

Mixtral 8x7B:
  创新: 开源首个 MoE LLM
  47B 参数, 13B 激活
  效果: 接近 GPT-3.5

Mistral Large 2:
  123B dense model
  强代码和多语言
```

## 11.8 小模型 (Small Language Models)

| 模型 | 参数量 | 训练数据 | 特点 |
|------|--------|---------|------|
| [Phi-3/3.5](https://arxiv.org/abs/2404.14219) | 3.8B/14B | Synthetic重 | "教科书级"数据质量 |
| [Gemma 2](https://arxiv.org/abs/2408.00118) | 2B/9B/27B | 大规模过训练 | Google 开源 |
| [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 135M-1.7B | 高质量 web data | HuggingFace |
| [TinyLlama](https://github.com/jzhang38/TinyLlama) | 1.1B | 3T tokens | 极度过训练 |

**趋势**: 小模型 + 高质量数据 + 过训练 → 性价比极高。端侧部署 (手机、PC) 是重要方向。

---

[← 上一章](10-safety.md) | [目录](../README.md) | [下一章 →](12-distillation-merging.md)
