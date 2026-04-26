[← Previous Chapter](10-safety.md) | [Table of Contents](README.md) | [Next Chapter →](12-distillation-merging.md)

---

# Chapter 11: SOTA Models Deep Dive

## 11.1 LLaMA 3 / 3.1 / 3.3 (Meta)

> Paper: [LLaMA 3](https://arxiv.org/abs/2407.21783) | Weights: [meta-llama](https://huggingface.co/meta-llama)

```
Parameters: 8B, 70B, 405B
Architecture: Dense Transformer, GQA, RoPE, SwiGLU, RMSNorm
Vocabulary: 128K (expanded tiktoken BPE)
Context: 128K (extended via YaRN)
Training data: 15T tokens
Training hardware: 16384 H100 GPUs
Training time: ~54 days (405B)

Key innovations:
- Large vocabulary (128K) significantly improves multilingual and code performance
- Over-training: 8B model trained on 15T tokens (far beyond Chinchilla optimal)
- 3-stage post-training: SFT → Rejection Sampling → DPO
- Code execution feedback: Uses code execution results as reward
```

## 11.2 DeepSeek-V3 / R1

> Paper: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | Weights: [deepseek-ai](https://huggingface.co/deepseek-ai)

```
DeepSeek-V3:
  Parameters: 671B (MoE, 37B activated)
  Architecture: MLA + DeepSeekMoE (256 experts, top-8 + 1 shared)
  Training: 14.8T tokens, 2048 H800 GPUs
  Cost: ~$5.5M (extremely low!)
  
  Key innovations:
  - MLA: 93% KV cache compression
  - FP8 mixed-precision training (first large-scale success)
  - Auxiliary-loss-free load balancing
  - Multi-Token Prediction (predict multiple future tokens)

DeepSeek-R1:
  Based on V3, with added reasoning capability
  
  Key innovations:
  - R1-Zero: Pure RL (GRPO + verifiable rewards) → emergent CoT
  - R1: SFT (cold start) → RL → Rejection Sampling → SFT
  - Near OpenAI o1 level on AIME 2024
  - Fully open-source (weights + paper)
```

## 11.3 Claude 3/4 (Anthropic)

> Documentation: [anthropic.com/claude](https://www.anthropic.com/claude)

```
Architecture: Undisclosed (speculated dense transformer, cross-attention multimodal)
Series: Haiku (small) → Sonnet (medium) → Opus (large)

Key features:
- Constitutional AI: Principle-based self-alignment
- Ultra-long context: 200K tokens
- Strong document/code understanding
- Industry-leading safety
- Claude 4: Exceptional coding and agentic capabilities
```

## 11.4 Gemini 2 (Google)

> Paper: [Gemini](https://arxiv.org/abs/2312.11805) | [Gemini 1.5](https://arxiv.org/abs/2403.05530)

```
Architecture: Native multimodal Transformer (MoE)
Training: TPU v5p/v6e clusters

Key features:
- Native multimodal (text, image, audio, video pretrained together)
- Ultra-long context: Gemini 1.5 supports 2M tokens
- Project Astra: Real-time video + audio understanding
- Gemini 2 Flash: Extremely fast inference
```

## 11.5 GPT-4 / o1 / o3 (OpenAI)

> Documentation: [platform.openai.com](https://platform.openai.com/docs)

```
GPT-4:
  Parameters: ~1.8T (MoE, 8 experts, speculated)
  Training: ~13T tokens on ~25000 A100s

o1/o3 (reasoning):
  Key innovations:
  - Large-scale RL training (speculated PRM + MCTS)
  - Test-time compute scaling
  - Hidden chain-of-thought (not visible to users)
  - o3 achieves 87.5% on ARC-AGI
```

## 11.6 Qwen 2.5 / QwQ (Alibaba)

> Paper: [Qwen2](https://arxiv.org/abs/2407.10671) | Weights: [Qwen](https://huggingface.co/Qwen)

```
Parameters: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B + MoE
Architecture: GQA, RoPE, SwiGLU
Vocabulary: 152K
Training: 18T tokens

Key features:
- Most comprehensive open-source model family (covering all scales)
- Strong multilingual (Chinese, English, Japanese, Korean, etc.)
- Qwen2-VL: Strong visual understanding
- QwQ: Reasoning model (o1-style)
- Qwen-Agent: Tool-use framework
```

## 11.7 Mistral / Mixtral

> Paper: [Mistral 7B](https://arxiv.org/abs/2310.06825) | [Mixtral](https://arxiv.org/abs/2401.04088) | Weights: [mistralai](https://huggingface.co/mistralai)

```
Mistral 7B:
  Innovation: Sliding Window Attention, GQA
  Result: 7B outperforms LLaMA 2 13B

Mixtral 8x7B:
  Innovation: First open-source MoE LLM
  47B parameters, 13B activated
  Result: Approaches GPT-3.5

Mistral Large 2:
  123B dense model
  Strong code and multilingual
```

## 11.8 Small Language Models

| Model | Parameters | Training Data | Highlights |
|-------|-----------|--------------|------------|
| [Phi-3/3.5](https://arxiv.org/abs/2404.14219) | 3.8B/14B | Heavy synthetic | "Textbook-quality" data |
| [Gemma 2](https://arxiv.org/abs/2408.00118) | 2B/9B/27B | Massive over-training | Google open-source |
| [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 135M-1.7B | High-quality web data | HuggingFace |
| [TinyLlama](https://github.com/jzhang38/TinyLlama) | 1.1B | 3T tokens | Extreme over-training |

**Trend**: Small models + high-quality data + over-training → excellent cost-effectiveness. On-device deployment (phones, PCs) is a key direction.

## Key Papers

- [Llama Team (2024) — Llama 3](https://arxiv.org/abs/2407.21783) — full disclosure of 405B training
- [DeepSeek-AI (2024) — DeepSeek-V3](https://arxiv.org/abs/2412.19437) — MoE + FP8 training, extreme cost optimization
- [Qwen Team (2024) — Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) — Chinese-friendly open-weights at all sizes
- [OpenAI (2023) — GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — high-level methodology and evaluation
- [Mistral AI (2023) — Mistral 7B](https://arxiv.org/abs/2310.06825) — sliding-window attention

## Further Reading

- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — head-to-head comparison
- [Chatbot Arena](https://lmarena.ai/) — human preference leaderboard
- [HuggingFace model index](https://huggingface.co/models) — read the source releases

## Exercises

1. **Deep-read one tech report**: pick Llama 3 or DeepSeek-V3 and read end-to-end; produce a "my notes" doc (data, architecture, hyperparams, evals).
2. **Same-size head-to-head**: with the same prompts, run Llama-3-8B, Qwen2.5-7B, Mistral-7B-v0.3 on tasks you care about; do blind scoring.
3. **Reproduce one detail**: pick one technique (YaRN context extension, MTP, MLA) and implement a toy version on a small model; understand why it works.

---

[← Previous Chapter](10-safety.md) | [Table of Contents](README.md) | [Next Chapter →](12-distillation-merging.md)
