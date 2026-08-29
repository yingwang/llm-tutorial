[← Previous Chapter](10-safety-alignment.md) | [Table of Contents](README.md) | [Next Chapter →](12-distillation-merging.md)

---

# Chapter 11: Frontier SOTA Models Deep Dive

A rigorous architectural decomposition of the defining frontier foundation models across open-weight ecosystems and proprietary commercial labs.

## 11.1 Meta LLaMA 3 / 3.1 / 3.3

> Research Report: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | Checkpoints: [meta-llama](https://huggingface.co/meta-llama)

```
Architectural Profile:
Parameters:         8B, 70B, 405B (Dense Decoder-Only Transformers)
Attention:          Grouped-Query Attention (GQA, 8 KV heads across all scales)
Embeddings:         Rotary Position Embeddings (RoPE) with YaRN context scaling (128K horizon)
Activation:         SwiGLU with RMSNorm pre-normalization
Vocabulary:         128K tiktoken BPE tokenizer (optimized for non-Latin fertility)
Pretraining Budget: 15.6 Trillion tokens (405B model)
Cluster Hardware:   16,384 NVIDIA H100 SXM GPUs over RoCEv2 fabric

Key Systems Innovations:
- Heavy Over-Training Regime: The 8B variant was trained on 15T tokens (90x Chinchilla optimal), driving down downstream inference serving costs.
- Iterative Alignment Pipeline: Multi-round SFT, Rejection Sampling, and Direct Preference Optimization (DPO).
- Verifiable Code Feedback: Injects automated code execution telemetry into preference scoring.
```

## 11.2 DeepSeek-V3 / DeepSeek-R1

> Technical Reports: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | Checkpoints: [deepseek-ai](https://huggingface.co/deepseek-ai)

```
DeepSeek-V3 (Frontier Sparse MoE Foundation):
Parameters:       671B Total Parameters (37B Active Parameters per token)
Architecture:     Multi-Head Latent Attention (MLA) + DeepSeekMoE (256 experts, top-8 routed + 1 shared)
Pretraining:      14.8 Trillion tokens on 2,048 NVIDIA H800 GPUs ($5.5M total cluster cost)

Core Innovations:
- MLA Latent KV Compression: Compresses KV cache memory footprint by ~93% relative to standard MHA.
- Native FP8 Mixed-Precision Training: First trillion-token scale validation of tile-quantized FP8 pretraining.
- Auxiliary-Loss-Free Load Balancing: Dynamic gate bias adaptation replacing loss penalties.
- Multi-Token Prediction (MTP): Speculatively decodes multiple consecutive tokens per step.

DeepSeek-R1 (Large-Scale Reasoning Model):
- DeepSeek-R1-Zero: Pure reinforcement learning via GRPO directly on base foundation models without SFT demonstrations, eliciting emergent reasoning chains and self-reflection.
- DeepSeek-R1: Cold-start CoT SFT initialization followed by multi-stage RLVR and preference alignment, matching OpenAI o1 on math and coding benchmarks.
```

## 11.3 Anthropic Claude 3 / 3.5

> Documentation & Safety Philosophy: [anthropic.com/claude](https://www.anthropic.com/claude)

```
Tiered Model Family: Claude 3.5 Haiku (Speed/Edge) → Sonnet (Coding/Analysis) → Opus (Complex Synthesis)
Context Horizon:     200K tokens native (sub-needle retrieval fidelity)

Core Distinctive Strengths:
- Constitutional AI (CAI): Automated self-alignment against codified behavioral principles.
- Code & Tool Orchestration: Industry-standard reasoning on multi-file SWE benchmarks.
- Multi-Modal Cross-Attention: High-resolution visual understanding across dense charts and UI layouts.
```

## 11.4 Google Gemini 1.5 / 2.0

> Technical Reports: [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) | [Gemini 1.5](https://arxiv.org/abs/2403.05530)

```
Infrastructure: Scaled across Google TPU v5p and TPU v6e supercomputer pods.

Key Architectural Capabilities:
- Native Omni-Modal Backbone: Pretrained jointly across interleaved audio, video, image, and text.
- 1M to 2M+ Token Context Horizon: Massive attention window enabling whole-video and repository ingestion.
- Low-Latency Flash Engine: Sub-hundred millisecond interactive real-time multi-modal streaming.
```

## 11.5 OpenAI GPT-4o / o1 / o3

> Documentation: [platform.openai.com/docs](https://platform.openai.com/docs)

```
GPT-4o (Omni-Modal Foundation):
- End-to-end multi-modal Transformer supporting native audio input-to-audio output streaming.
- Sub-300ms real-time voice latency mimicking human conversational cadence.

o1 / o3 Series (Test-Time Compute Reasoning Engines):
- Large-scale reinforcement learning over chain-of-thought generation graphs.
- Decoupled test-time search: Expands compute dynamically via internal deliberation tokens before final output emission.
- Frontier performance on formal competitive programming (Codeforces) and Olympiad math (AIME).
```

## 11.6 Alibaba Qwen 2.5 / QwQ

> Technical Reports: [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) | Checkpoints: [Qwen](https://huggingface.co/Qwen)

```
Model Hierarchy:  Dense parameters (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B) + Sparse MoE variants
Vocabulary:       152K tokens (exceptional Asian script compression)
Training Volume:  18 Trillion tokens

Core Strengths:
- Comprehensive Open Matrix: Spans mobile-edge scales (1.5B) up to datacenter enterprise scales (72B).
- QwQ Reasoning Series: Open-weight reasoning models matching o1-style long-chain deliberation.
- Qwen2-VL: High-resolution dynamic visual perception and spatiotemporal video understanding.
```

## 11.7 Mistral AI: Mistral 7B & Mixtral 8x22B

> Technical Reports: [Mistral 7B](https://arxiv.org/abs/2310.06825) | [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

```
Mistral 7B:
  - Pioneered Sliding Window Attention (SWA) combined with Grouped-Query Attention.
  - Set the open-weight performance standard for sub-10B dense models.

Mixtral 8x7B / 8x22B:
  - Popularized open Sparse MoE architectures: 8 experts per MLP with top-2 dynamic routing.
  - 8x22B achieves 141B total parameters with only 39B active parameters during forward inference.
```

## 11.8 Small Language Models (SLMs)

| Architecture | Parameter Count | Training Token Volume | Core Methodology |
|--------------|-----------------|-----------------------|------------------|
| [Microsoft Phi-3.5](https://arxiv.org/abs/2404.14219) | 3.8B / 14B | ~5T (Heavy Synthetic) | "Textbook Are All You Need" curated synthetic data |
| [Google Gemma 2](https://arxiv.org/abs/2408.00118) | 2.6B / 9B / 27B | Massive Over-Training | QK-Norm, logit soft-capping, knowledge distillation |
| [SmolLM-2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 135M / 360M / 1.7B | Curated Web & Math | Optimized for edge and on-device execution |
| [TinyLlama](https://github.com/jzhang38/TinyLlama) | 1.1B | 3 Trillion tokens | Extreme over-training proof-of-concept |

## Key Papers

- [Meta AI (2024): The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783): Full disclosure of 405B distributed pretraining and alignment.
- [DeepSeek-AI (2024): DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): Frontier FP8 mixed-precision MoE pretraining.
- [DeepSeek-AI (2025): DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948): Open-source reasoning models via RL.
- [Qwen Team (2024): Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115): Multilingual foundation model scaling across all sizes.
- [Jiang et al. (2024): Mixtral of Experts](https://arxiv.org/abs/2401.04088): Foundational open sparse MoE architecture.

## Further Reading

- LMSYS: [Chatbot Arena Leaderboard](https://lmarena.ai/) (Crowdsourced real-world conversational Elo ratings).
- Hugging Face: [Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) (Standardized academic benchmark tracking).
- Artificial Analysis: [LLM Performance and Serving Cost Benchmarks](https://artificialanalysis.ai/) (Real-time latency, throughput, and pricing comparisons).

## Exercises

1. **Architecture Dissection**: Compare the attention mechanisms of LLaMA 3.1 (GQA) and DeepSeek-V3 (MLA); calculate the KV cache memory savings of MLA on a 128K context sequence.
2. **Blind Model Evaluation**: Deploy Qwen2.5-7B, LLaMA-3.1-8B, and Mistral-7B-v0.3 on a standardized prompt set covering code debugging, creative extraction, and JSON schemas; conduct blind human evaluation to calculate pairwise win rates.
3. **Small Model Distillation Analysis**: Examine how Microsoft Phi-3 achieves performance competitive with 8B models using synthetic data; design a toy distillation curriculum on a 500M parameter student model.

---

[← Previous Chapter](10-safety-alignment.md) | [Table of Contents](README.md) | [Next Chapter →](12-distillation-merging.md)
