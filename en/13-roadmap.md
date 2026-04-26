[← Previous Chapter](12-distillation-merging.md) | [Table of Contents](README.md)

# Chapter 13: Practical Roadmap

## 13.1 Learning Path

### Phase 1: Foundations (2-4 weeks)

```
□ Understand the Transformer architecture
  - Implement a simple Attention from scratch
  - Read "Attention Is All You Need"
  - Implement nanoGPT (Karpathy)

□ Understand Tokenizers
  - Train a BPE tokenizer with HuggingFace tokenizers
  - Read the SentencePiece paper

□ PyTorch proficiency
  - Write a training loop from scratch
  - Understand autograd
  - Use DataLoader, Dataset
```

### Phase 2: Pretraining (4-8 weeks)

```
□ Train a small LLM from scratch
  - Prepare data: download a RedPajama-Data-v2 subset
  - Train a tokenizer
  - Implement GPT-2 architecture (124M parameters)
  - Single-GPU training, observe loss curves
  - Recommended: nanoGPT, litGPT

□ Multi-GPU training
  - DDP (torch.nn.parallel.DistributedDataParallel)
  - FSDP
  - Understand all-reduce, ring-allreduce

□ Data processing
  - Implement a simple data processing pipeline
  - Deduplication (MinHash)
  - Quality filtering
```

### Phase 3: Post-Training (2-4 weeks)

```
□ SFT
  - Use HuggingFace TRL or Axolotl
  - Fine-tune LLaMA 3 8B with LoRA
  - Prepare chat format data
  - Understand packing, loss masking

□ DPO
  - Use TRL's DPOTrainer
  - Prepare preference data
  - Compare SFT-only vs SFT+DPO

□ RLHF (advanced)
  - Train a Reward Model
  - Use TRL's PPOTrainer
  - Understand PPO hyperparameter tuning
```

### Phase 4: Advanced (ongoing)

```
□ Efficient inference
  - Deploy vLLM
  - Quantize with GPTQ/AWQ
  - Implement speculative decoding

□ RAG
  - Train or use an embedding model
  - Build a vector retrieval pipeline
  - Optimize: hybrid search + reranker

□ Multimodal
  - Train a simple LLaVA
  - Understand CLIP, SigLIP
  - Visual instruction tuning

□ Infra
  - Use Megatron-LM
  - Configure TP + PP + DP
  - Profile training performance (nsys, tensorboard)
  - Optimize MFU
```

## 13.2 Recommended Resources

### Must-Read Papers (with links)

**Foundations**:
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani 2017) — Transformer
2. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
3. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)

**Scaling**:
4. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan 2020)
5. [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) (Chinchilla, Hoffmann 2022)

**Architecture**:
6. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron 2023)
7. [Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang 2023)
8. [DeepSeek-V2: A Strong, Economical, and Efficient MoE](https://arxiv.org/abs/2405.04434) (2024)

**Post-Training**:
9. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT)
10. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov 2023)
11. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025)

**PEFT**:
12. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu 2021)
13. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers 2023)

**Infra**:
14. [Megatron-LM](https://arxiv.org/abs/1909.08053) (Shoeybi 2019)
15. [ZeRO: Memory Optimizations](https://arxiv.org/abs/1910.02054) (Rajbhandari 2020)
16. [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao 2022)

**Multimodal**:
17. [Learning Transferable Visual Models (CLIP)](https://arxiv.org/abs/2103.00020) (Radford 2021)
18. [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) (Liu 2023)
19. [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) (Google 2023)

**Reasoning**:
20. [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) (Wei 2022)

### Code Projects

**Getting Started**:
| Project | Purpose |
|---------|---------|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Train GPT-2 from scratch |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | GPT-2 training in C |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | Minimal BPE implementation |

**Fine-tuning**:
| Project | Purpose |
|---------|---------|
| [huggingface/trl](https://github.com/huggingface/trl) | SFT, DPO, PPO, RLOO |
| [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | All-in-one fine-tuning |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 2x faster LoRA |
| [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Popular in China; broad model coverage |
| [pytorch/torchtune](https://github.com/pytorch/torchtune) | PyTorch native |

**Pretraining**:
| Project | Purpose |
|---------|---------|
| [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) | Production-grade LLM training |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Large-scale pretraining |
| [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | Distributed training |

**Inference**:
| Project | Purpose |
|---------|---------|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Efficient inference serving |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | Inference + complex prompts |
| [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | Local/CPU inference |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA-optimized inference |

**Multimodal**:
| Project | Purpose |
|---------|---------|
| [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | Vision-language model |
| [openai/whisper](https://github.com/openai/whisper) | Speech recognition |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Unified framework |

**Data & Evaluation**:
| Project | Purpose |
|---------|---------|
| [huggingface/datatrove](https://github.com/huggingface/datatrove) | Data processing pipeline |
| [allenai/dolma](https://github.com/allenai/dolma) | Data toolkit |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Unified evaluation |
| [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) | Software engineering evaluation |

### Courses

| Course | Platform | Notes |
|--------|----------|-------|
| [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | YouTube | Karpathy, best intro |
| [Stanford CS224N](https://web.stanford.edu/class/cs224n/) | Stanford | NLP with Deep Learning |
| [Stanford CS336](https://stanford-cs336.github.io/spring2024/) | Stanford | Language Modeling from Scratch (2024) |
| [CMU 11-868](https://llms-11-868.github.io/) | CMU | Large Language Models (2024) |
| [LLM Course](https://github.com/mlabonne/llm-course) | GitHub | Maxime Labonne, free |
| [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) | Online | Engineering-oriented |

## 13.3 Hardware Recommendations

### Personal Learning
```
Beginner: 1x RTX 4090 (24GB)
  → Enough to train a 1B model
  → LoRA fine-tune 7B
  → QLoRA fine-tune 70B
  → Run quantized 70B locally

Intermediate: 2x RTX 4090
  → Learn multi-GPU training

Cloud GPU: Lambda, RunPod, vast.ai on-demand
  → RTX 4090: ~$0.5/hr
  → A100 80GB: ~$2/hr
```

### Small Team
```
4-8x A100/H100
  → Can pretrain 7B
  → Full fine-tune 70B
  → Cloud options: AWS p5, GCP a3, Azure ND H100
```

### Large Scale
```
256+ GPUs
  → Pretrain 70B+
  → Self-hosted or long-term cloud lease
  → Needs dedicated infra engineers
```

## 13.4 Training Cost Reference

### Pretraining

| Model Size | GPU | Tokens | Approx. Cost | Time |
|------------|-----|--------|---------------|------|
| 124M (GPT-2 small) | 1x RTX 4090 | 10B | ~$20 | ~8h |
| 1B | 1x A100 80GB | 20B | ~$500 | ~2 days |
| 7B | 8x A100 80GB | 1T | ~$50K | ~2 weeks |
| 13B | 32x A100 80GB | 2T | ~$200K | ~3 weeks |
| 70B | 256x H100 | 2T | ~$2M | ~1 month |

### Fine-tuning

| Method | Model | GPU | Approx. Cost | Time |
|--------|-------|-----|---------------|------|
| QLoRA SFT | 7B | 1x RTX 4090 | ~$5 | ~2h |
| QLoRA SFT | 70B | 1x A100 80GB | ~$20 | ~10h |
| LoRA SFT | 7B | 1x A100 80GB | ~$10 | ~5h |
| Full SFT | 7B | 4x A100 80GB | ~$100 | ~12h |
| Full SFT | 70B | 32x A100 80GB | ~$2K | ~2 days |
| DPO | 7B | 2x A100 80GB | ~$50 | ~6h |

> Estimates based on typical cloud rates: A100 ~$2/hr, H100 ~$3.5/hr, RTX 4090 ~$0.5/hr.

---

# Appendix: Complete Paper List

## Tokenizer
- [Sennrich et al. (2016) — BPE for NMT](https://arxiv.org/abs/1508.07909)
- [Kudo & Richardson (2018) — SentencePiece](https://arxiv.org/abs/1808.06226)
- [Kudo (2018) — Subword Regularization (Unigram)](https://arxiv.org/abs/1804.10959)

## Architecture
- [Vaswani et al. (2017) — Transformer](https://arxiv.org/abs/1706.03762)
- [Su et al. (2021) — RoPE](https://arxiv.org/abs/2104.09864)
- [Shazeer (2019) — Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [Ainslie et al. (2023) — GQA](https://arxiv.org/abs/2305.13245)
- [Shazeer (2020) — GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202)
- [Zhang & Sennrich (2019) — RMSNorm](https://arxiv.org/abs/1910.07467)
- [Dao et al. (2022) — FlashAttention](https://arxiv.org/abs/2205.14135)
- [Dao (2023) — FlashAttention-2](https://arxiv.org/abs/2307.08691)

## Pretraining
- [Radford et al. (2019) — GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Brown et al. (2020) — GPT-3](https://arxiv.org/abs/2005.14165)
- [Kaplan et al. (2020) — Scaling Laws](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. (2022) — Chinchilla](https://arxiv.org/abs/2203.15556)
- [Touvron et al. (2023a) — LLaMA](https://arxiv.org/abs/2302.13971)
- [Touvron et al. (2023b) — LLaMA 2](https://arxiv.org/abs/2307.09288)
- [Dubey et al. (2024) — LLaMA 3](https://arxiv.org/abs/2407.21783)
- [DeepSeek (2024) — DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek (2024) — DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek (2025) — DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## Post-Training
- [Ouyang et al. (2022) — InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155)
- [Schulman et al. (2017) — PPO](https://arxiv.org/abs/1707.06347)
- [Rafailov et al. (2023) — DPO](https://arxiv.org/abs/2305.18290)
- [Bai et al. (2022) — Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Shao et al. (2024) — DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
- [Lightman et al. (2023) — Process Reward Models](https://arxiv.org/abs/2305.20050)

## PEFT
- [Hu et al. (2021) — LoRA](https://arxiv.org/abs/2106.09685)
- [Dettmers et al. (2023) — QLoRA](https://arxiv.org/abs/2305.14314)
- [Li & Liang (2021) — Prefix Tuning](https://arxiv.org/abs/2101.00190)
- [Liu et al. (2024) — DoRA](https://arxiv.org/abs/2402.09353)

## MoE
- [Shazeer et al. (2017) — Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)
- [Fedus et al. (2022) — Switch Transformer](https://arxiv.org/abs/2101.03961)
- [Jiang et al. (2024) — Mixtral](https://arxiv.org/abs/2401.04088)

## Multimodal
- [Radford et al. (2021) — CLIP](https://arxiv.org/abs/2103.00020)
- [Liu et al. (2023) — LLaVA](https://arxiv.org/abs/2304.08485)
- [Alayrac et al. (2022) — Flamingo](https://arxiv.org/abs/2204.14198)
- [Team Gemini (2023) — Gemini](https://arxiv.org/abs/2312.11805)
- [Rombach et al. (2022) — Latent Diffusion](https://arxiv.org/abs/2112.10752)
- [Peebles & Xie (2023) — DiT](https://arxiv.org/abs/2212.09748)

## Reasoning
- [Wei et al. (2022) — Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- [Snell et al. (2024) — Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314)

## Infra
- [Shoeybi et al. (2019) — Megatron-LM](https://arxiv.org/abs/1909.08053)
- [Rajbhandari et al. (2020) — ZeRO](https://arxiv.org/abs/1910.02054)
- [Narayanan et al. (2021) — Pipeline Parallelism](https://arxiv.org/abs/2104.04473)
- [Kwon et al. (2023) — PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Leviathan et al. (2023) — Speculative Decoding](https://arxiv.org/abs/2211.17192)

## Further Reading

- Book: Sebastian Raschka — *Build a Large Language Model (From Scratch)* — implement a GPT end-to-end
- Course: [Stanford CS25 — Transformers United](https://web.stanford.edu/class/cs25/)
- Course: [NYU/Yann LeCun — Deep Learning](https://atcold.github.io/NYU-DLSP21/)
- Stay current: [Sebastian Raschka — Ahead of AI](https://magazine.sebastianraschka.com/), [Nathan Lambert — Interconnects](https://www.interconnects.ai/), [Jay Alammar's Blog](https://jalammar.github.io/)

## Exercises (project-scale)

1. **Train a 100M model from scratch**: use nanoGPT on OpenWebText (or a Chinese mini-corpus) to train a 124M GPT; carry it through the full pipeline — pretrain → SFT → DPO → quantize → vLLM serve.
2. **Build an enterprise RAG**: pick a document set you know well (≥1000 docs); implement chunking + embedding + retrieval + generation + evaluation; write up "lessons learned and footguns I hit."
3. **Reproduce a SOTA paper**: pick a recent arxiv paper (e.g. GRPO, MTP, MLA); do a toy reproduction on a small model; write a blog post explaining the mechanism and its limits.

---

> Last updated: 2026-04-25
> The best way to keep learning: read papers, run code, read more papers.

[← Previous Chapter](12-distillation-merging.md) | [Table of Contents](README.md)
