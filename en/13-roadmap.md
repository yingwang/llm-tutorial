[← Previous Chapter](12-distillation-merging.md) | [Table of Contents](README.md)

# Chapter 13: Practical Engineering Roadmap and Reference

An end-to-end curriculum and engineering reference for mastering large language model development, ranging from foundational Transformer kernels to large-scale pretraining, alignment pipelines, and production serving infrastructure.

## 13.1 Step-by-Step Mastery Curriculum

### Phase 1: Architectural Foundations (2–4 Weeks)

```
[ ] Core Attention Mechanics:
    - Implement scaled dot-product attention and multi-head attention from scratch in PyTorch.
    - Study "Attention Is All You Need" (Vaswani et al., 2017).
    - Implement a complete character-level and BPE-level autoregressive Transformer (Karpathy nanoGPT).

[ ] Tokenization Systems:
    - Train Byte-Pair Encoding (BPE) and Unigram tokenizers using Hugging Face Tokenizers and SentencePiece.
    - Inspect token fertility ratios and special token boundary encodings across diverse scripts.

[ ] Deep Learning Systems Engineering:
    - Write explicit manual training and backpropagation loops in raw PyTorch.
    - Master PyTorch Autograd, custom autograd Functions, and DataLoader multi-process pinning.
```

### Phase 2: Pretraining & Scaling Engineering (4–8 Weeks)

```
[ ] Foundation Training from Step Zero:
    - Ingest and process a curated web subset (FineWeb-Edu or RedPajama-v2).
    - Train custom BPE vocabulary (32K to 64K tokens).
    - Implement modern LLaMA-style architectures (RoPE, SwiGLU, RMSNorm, GQA).
    - Execute single-GPU pretraining runs on a 124M-parameter baseline, tracking Chinchilla loss curves.

[ ] Distributed Training Systems:
    - DistributedDataParallel (DDP) gradient synchronization and ring-allreduce communication.
    - Fully Sharded Data Parallel (FSDP) and DeepSpeed ZeRO-1/2/3 state partitioning.
    - Multi-GPU compute scheduling and communication-computation overlap profiling.

[ ] Scalable Data Curation Pipelines:
    - Implement scalable heuristic text filters (Gopher / C4 quality rules).
    - Build MinHash LSH and Bloom filter deduplication pipelines for billion-token datasets.
    - Integrate synthetic data generation and LLM-as-a-judge quality scoring.
```

### Phase 3: Post-Training, Alignment, and Reasoning (2–4 Weeks)

```
[ ] Supervised Instruction Tuning (SFT):
    - Conduct parameter-efficient fine-tuning (LoRA / QLoRA) on 8B models using Hugging Face TRL or Axolotl.
    - Structure conversational datasets with precise system prompt isolation and prompt loss masking.
    - Implement packed sample batching to eliminate padding token compute waste.

[ ] Direct Preference Optimization (DPO & SimPO):
    - Curate pairwise preference datasets (chosen vs. rejected completions).
    - Fine-tune policy models using DPO loss, monitoring implicit reward margins and KL drift.

[ ] Reinforcement Learning from Verifiable Rewards (RLVR & GRPO):
    - Implement Group Relative Policy Optimization (GRPO) over deterministic math and coding tasks.
    - Observe the emergence of long-chain reasoning, self-correction, and backtracking deliberation.
```

### Phase 4: Serving, Systems, and Multimodal (Ongoing)

```
[ ] High-Throughput Inference Deployment:
    - Deploy production OpenAI-compatible endpoints via vLLM and SGLang.
    - Execute post-training quantization (AWQ / GPTQ / FP8) and evaluate perplexity trade-offs.
    - Configure speculative decoding pairing 70B target foundations with 8B draft engines.

[ ] Enterprise Retrieval-Augmented Generation (RAG):
    - Build two-stage hybrid retrieval pipelines fusing BM25 and dense bi-encoder embeddings via RRF.
    - Integrate cross-encoder rerankers and contextual parent document expansion.

[ ] Multimodal System Integration:
    - Align visual encoders (SigLIP) with LLM backbones via MLP projectors (LLaVA architecture).
    - Fine-tune vision-language pipelines on structured visual question answering and OCR datasets.

[ ] Extreme Distributed Infrastructure:
    - Implement 3D parallelism (TP + PP + DP) using Megatron-LM.
    - Profile GPU CUDA kernels using NVIDIA Nsight Systems (nsys), optimizing Model FLOPs Utilization (MFU).
```

## 13.2 Recommended Engineering Resources

### Foundational Paper Reading List

**Architectural Foundations**:
1. [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762): Original Transformer formulation.
2. [Radford et al. (2019): Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): GPT-2 zero-shot task transfer.
3. [Brown et al. (2020): Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): GPT-3 in-context scaling paradigm.

**Empirical Scaling Laws**:
4. [Kaplan et al. (2020): Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361): Empirical compute power-law scaling.
5. [Hoffmann et al. (2022): Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556): Chinchilla compute-optimal parameter-to-data balancing.

**Modern Foundation Architectures**:
6. [Touvron et al. (2023): LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971): Standard open-source architectural template.
7. [Jiang et al. (2023): Mistral 7B](https://arxiv.org/abs/2310.06825): Sliding window attention and GQA.
8. [DeepSeek-AI (2024): DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): Frontier FP8 MoE with Multi-Head Latent Attention.

**Post-Training and Alignment**:
9. [Ouyang et al. (2022): Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155): InstructGPT RLHF methodology.
10. [Rafailov et al. (2023): Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290): Closed-form preference loss.
11. [DeepSeek-AI (2025): DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948): Reasoning emergence through pure RL.

**Parameter-Efficient Fine-Tuning**:
12. [Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685): Low-rank matrix decomposition.
13. [Dettmers et al. (2023): QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314): 4-bit NormalFloat quantized adapters.

**Systems and High-Performance Infrastructure**:
14. [Shoeybi et al. (2019): Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053): Tensor parallel linear decomposition.
15. [Rajbhandari et al. (2020): ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054): Memory sharding across optimizer, gradient, and parameter tiers.
16. [Dao et al. (2022): FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135): Hardware-aware exact attention tiling.

### Open-Source Codebases

| Category | Repository | Engineering Focus |
|----------|------------|-------------------|
| **Foundational Kernels** | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Minimal, clean educational GPT implementation |
| | [karpathy/llm.c](https://github.com/karpathy/llm.c) | Pure C/CUDA GPT-2 training with zero heavy dependencies |
| | [karpathy/minbpe](https://github.com/karpathy/minbpe) | Minimal educational Byte-Pair Encoding engine |
| **Fine-Tuning Engines** | [huggingface/trl](https://github.com/huggingface/trl) | Production SFT, DPO, PPO, and GRPO training loops |
| | [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | Declarative YAML-driven production fine-tuning suite |
| | [unslothai/unsloth](https://github.com/unslothai/unsloth) | Hand-crafted OpenAI Triton kernels for 2x faster LoRA |
| | [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Unified multi-model fine-tuning framework |
| **Distributed Pretraining** | [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) | Modular, hackable LLM pretraining and fine-tuning |
| | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Enterprise 3D parallelism pretraining framework |
| | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | Distributed memory optimization library |
| **Serving & Inference** | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Production serving engine with PagedAttention |
| | [sgl-project/sglang](https://github.com/sgl-project/sglang) | RadixAttention prefix caching and structured runtime |
| | [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | Ultra-fast C++ edge and CPU inference runtime |
| | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Maximum-throughput NVIDIA kernel serving runtime |

### Academic Courses & Industry Guides

| Curriculum | Institution / Author | Core Subject Focus |
|------------|----------------------|--------------------|
| [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | Andrej Karpathy | Deep learning and language modeling from first principles |
| [Stanford CS336](https://stanford-cs336.github.io/spring2024/) | Stanford University | Language Modeling from Scratch (Systems and scaling focus) |
| [Stanford CS224N](https://web.stanford.edu/class/cs224n/) | Stanford University | Natural Language Processing with Deep Learning |
| [CMU 11-868](https://llms-11-868.github.io/) | Carnegie Mellon University | Large Language Models systems and architectures |
| [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) | Full Stack Deep Learning | Production deployment, evaluations, and enterprise RAG |

## 13.3 Hardware Sizing & Infrastructure Guide

### Individual Engineer Tier
```
Local Workstation: 1x NVIDIA RTX 4090 (24GB VRAM)
  - Train 100M-1B foundation models from scratch.
  - Full parameter SFT on 3B models; 16-bit LoRA on 8B models.
  - 4-bit QLoRA on 70B parameter foundations.
  - Serve quantized 70B models locally via llama.cpp or vLLM.

Cloud On-Demand (Lambda / RunPod / vast.ai):
  - RTX 4090: ~$0.50 / GPU-hour
  - A100 SXM 80GB: ~$2.00 / GPU-hour
  - H100 SXM 80GB: ~$3.50 / GPU-hour
```

### Research Team Tier
```
Dedicated Cluster Node: 8x NVIDIA H100 / H200 SXM (640GB - 1128GB Unified VRAM)
  - Full pretraining runs for 7B-8B parameter foundation models over 1T+ tokens.
  - Full-parameter supervised fine-tuning and GRPO reasoning alignment on 70B models.
  - Inter-GPU Bandwidth: 900 GB/s NVLink enables frictionless Tensor and FSDP scaling.
```

### Enterprise Datacenter Tier
```
Supercomputing Cluster: 256 to 16,384+ NVIDIA H100/B200 SXM GPUs
  - Pretrain frontier 70B to 400B+ foundation models over 15T+ tokens.
  - Non-blocking InfiniBand Quantum-2 (3.2 Tbps per node) or 800GbE RoCEv2 fabric.
```

## 13.4 Compute Budget and Training Cost Benchmarks

### Pretraining Cost Formulations

| Model Scale | Target Token Volume | Hardware Configuration | Approximate Cloud Cost | Elapsed Wall-Clock Time |
|-------------|---------------------|------------------------|------------------------|-------------------------|
| 124M (GPT-2 Small) | 10 Billion | 1x RTX 4090 | ~$20 | ~8 Hours |
| 1.1B (TinyLlama) | 100 Billion | 8x A100 80GB | ~$1,200 | ~3 Days |
| 7B | 1 Trillion | 64x A100 80GB | ~$35,000 | ~10 Days |
| 70B | 2 Trillion | 256x H100 SXM | ~$750,000 | ~14 Days |
| 405B | 15 Trillion | 16,384x H100 SXM | ~$120,000,000 | ~54 Days |

### Fine-Tuning & Alignment Cost Formulations

| Adaptation Technique | Base Model Size | GPU Hardware Setup | Approximate Cloud Cost | Elapsed Training Time |
|----------------------|-----------------|--------------------|------------------------|-----------------------|
| 4-bit QLoRA SFT | 8B | 1x RTX 4090 (24GB) | ~$5 | ~2 Hours |
| 4-bit QLoRA SFT | 70B | 1x A100 80GB | ~$20 | ~8 Hours |
| 16-bit LoRA SFT | 8B | 1x A100 80GB | ~$12 | ~4 Hours |
| Full Parameter SFT | 8B | 8x A100 80GB | ~$80 | ~5 Hours |
| Full Parameter SFT | 70B | 64x A100 80GB | ~$1,800 | ~12 Hours |
| Direct Preference Optimization | 8B | 4x A100 80GB | ~$40 | ~4 Hours |

---

## Appendix: Comprehensive Technical Literature Index

### Tokenization & Vocabulary
- [Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909)
- [Kudo & Richardson (2018): SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)
- [Kudo (2018): Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Unigram)](https://arxiv.org/abs/1804.10959)

### Architecture Innovations
- [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Su et al. (2021): RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Shazeer (2019): Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)
- [Ainslie et al. (2023): GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Shazeer (2020): GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202)
- [Zhang & Sennrich (2019): Root Mean Square Layer Normalization (RMSNorm)](https://arxiv.org/abs/1910.07467)
- [Dao et al. (2022): FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Dao (2023): FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

### Pretraining & Scaling Laws
- [Radford et al. (2019): Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Brown et al. (2020): Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Kaplan et al. (2020): Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. (2022): Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Touvron et al. (2023): LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Dubey et al. (2024): The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-AI (2024): DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

### Post-Training, Alignment & Reasoning
- [Ouyang et al. (2022): Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Schulman et al. (2017): Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347)
- [Rafailov et al. (2023): Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)](https://arxiv.org/abs/2305.18290)
- [Bai et al. (2022): Constitutional AI: A Promising Approach for Using AI to Align AI](https://arxiv.org/abs/2212.08073)
- [Shao et al. (2024): DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-AI (2025): DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Wei et al. (2022): Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Snell et al. (2024): Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)

### Parameter-Efficient Adaptation & Merging
- [Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Dettmers et al. (2023): QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Liu et al. (2024): DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [Wortsman et al. (2022): Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time](https://arxiv.org/abs/2203.05482)
- [Yadav et al. (2023): Resolving Interference When Merging Models (TIES-Merging)](https://arxiv.org/abs/2306.01708)
- [Yu et al. (2024): Language Models are Super Mario: Absorbing Abilities from Homologous Models with DARE](https://arxiv.org/abs/2311.03099)

### Multimodal Architectures
- [Radford et al. (2021): Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Liu et al. (2023): Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
- [Alayrac et al. (2022): Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [Rombach et al. (2022): High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Peebles & Xie (2023): Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)

### Systems Infrastructure & Inference Acceleration
- [Shoeybi et al. (2019): Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Rajbhandari et al. (2020): ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Narayanan et al. (2021): Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Kwon et al. (2023): Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Leviathan et al. (2023): Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

## Capstone Engineering Projects

1. **Full-Stack Small Language Model**: Train a 124M-to-500M parameter foundation model from raw web text using `litgpt` or `nanoGPT`; carry the checkpoint through Supervised Fine-Tuning, Direct Preference Optimization, FP8 quantization, and production deployment on `vLLM`.
2. **Production Enterprise RAG System**: Construct an end-to-end RAG service over a 10,000-document technical knowledge base; implement hybrid dense/sparse retrieval with Reciprocal Rank Fusion, cross-encoder reranking, parent document retrieval, and automated citation verification.
3. **Frontier Architecture Kernel Reproduction**: Select an architectural mechanism (such as Multi-Head Latent Attention, Group Relative Policy Optimization, or speculative Multi-Token Prediction) and implement an educational kernel from scratch in PyTorch/Triton, profiling its latency and memory footprint against standard baselines.

---

[← Previous Chapter](12-distillation-merging.md) | [Table of Contents](README.md)
