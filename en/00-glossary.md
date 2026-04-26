[← Table of Contents](README.md)

# Glossary

High-frequency terms in LLM training and deployment, grouped by topic.

> Each entry is kept to 1–2 sentences; for deeper coverage see the corresponding chapter.

---

## Architecture

- **Attention**: weighted sum of values via dot-product of query/key; the core operator of the Transformer.
- **MHA (Multi-Head Attention)**: splits attention into parallel heads that learn different subspaces.
- **GQA (Group-Query Attention)**: multiple q heads share one group of kv heads, dramatically shrinking the KV cache. Used by Llama 2/3.
- **MQA (Multi-Query Attention)**: all q heads share a single kv-head group; the extreme of GQA, with mild quality loss.
- **MLA (Multi-head Latent Attention)**: introduced in DeepSeek-V2 — projects KV to a low-rank latent space, even smaller than GQA.
- **RoPE (Rotary Position Embedding)**: injects positional information by rotating q/k vectors; the modern standard.
- **ALiBi**: replaces explicit positional encodings with a linear bias; strong length extrapolation.
- **SwiGLU**: a gated FFN variant slightly outperforming GeLU; the de facto default in new models.
- **MoE (Mixture of Experts)**: FFN layer with many experts; each token activates only top-k experts — large parameters, low FLOPs.
- **Routing**: the mechanism in MoE that decides which experts each token is sent to.
- **MTP (Multi-Token Prediction)**: predicts several future tokens during training; used by DeepSeek-V3 for speed and quality.

## Training Objectives

- **CLM (Causal LM)**: next-token prediction; the GPT-family objective.
- **MLM (Masked LM)**: predict masked tokens; the BERT objective.
- **PrefixLM**: bidirectional attention on the prefix, causal on the suffix; used in T5 / GLM.
- **Next-Sentence Prediction**: BERT auxiliary task; later shown to add little value.

## Post-Training

- **SFT (Supervised Fine-Tuning)**: train on (prompt, response) pairs to teach instruction-following.
- **RLHF**: Reinforcement Learning from Human Feedback — reward model + PPO/GRPO three-stage pipeline.
- **RLAIF**: RLHF where human feedback is replaced by AI feedback (e.g., Constitutional AI).
- **PPO**: classic policy-gradient algorithm; used by OpenAI InstructGPT.
- **DPO (Direct Preference Optimization)**: skip the reward model and supervise directly on preference pairs; simple and stable.
- **GRPO**: introduced in DeepSeekMath — no critic; estimates advantage from intra-group relative ranking. Used in R1.
- **KTO**: needs only thumbs-up/down per sample, no pairwise preferences.
- **Reward Model (RM)**: scores outputs; usually built on the same backbone with a value head.
- **KL penalty**: keeps the RL policy close to the SFT model, mitigating reward hacking.
- **Constitutional AI**: uses a set of "constitutional" principles to have the model self-critique and self-revise; a basis for RLAIF.

## Parameter-Efficient Fine-Tuning (PEFT)

- **PEFT**: umbrella term for methods that train only a small fraction of parameters (usually <1%).
- **LoRA**: adds low-rank ΔW = BA next to q/k/v/o; only B and A are trained; mergeable at inference time.
- **QLoRA**: 4-bit NF4-quantized backbone + LoRA; fits 65B on a single GPU.
- **Adapter**: small MLP modules inserted between transformer layers.
- **IA³**: three scaling vectors per layer; even fewer parameters than LoRA.
- **Prefix / P-Tuning**: concatenates learnable prefix tokens at attention inputs.

## Training Infrastructure

- **DP (Data Parallelism)**: each GPU holds a full model replica, processes a different batch.
- **TP (Tensor Parallelism)**: splits a single matrix across GPUs; requires high bandwidth.
- **PP (Pipeline Parallelism)**: splits layers across GPUs and overlaps via micro-batches.
- **3D Parallelism**: DP × TP × PP combined; the standard for 10K-GPU training.
- **ZeRO**: the core of DeepSpeed; shards optimizer state / gradient / parameter across DP ranks.
- **FSDP**: PyTorch's equivalent of ZeRO-3.
- **FlashAttention**: IO-aware attention that fuses softmax in SRAM; 2–4× speedup.
- **Gradient Checkpointing**: trade recomputation in the forward pass for memory; standard for large-model training.
- **Mixed Precision**: BF16/FP16 compute with FP32 optimizer states.
- **FP8 Training**: 8-bit training on H100+; DeepSeek-V3 was the first large-scale production use.
- **NCCL**: NVIDIA's multi-GPU/multi-node communication library; the de facto standard for AllReduce / AllGather.
- **NVLink / InfiniBand**: intra-node / inter-node high-speed interconnects.

## Inference

- **KV cache**: cache historical K/V tensors to avoid recomputation; the core autoregressive optimization.
- **PagedAttention**: vLLM's paged KV-cache management — on-demand allocation, low fragmentation.
- **Continuous Batching**: dynamic batch packing where new requests are inserted as they arrive; standard in vLLM/SGLang.
- **Speculative Decoding**: a small model proposes N tokens, the large model verifies them in one forward pass; saves forwards.
- **Medusa**: multiple parallel prediction heads; a simplified variant of speculative decoding.
- **Prefix caching**: compute KV once for a shared system prompt and reuse across requests.
- **Quantization**: convert weights from FP16 → INT8 / INT4 to save memory and accelerate; near-lossless for inference.
- **GPTQ / AWQ**: two mainstream post-training quantization methods; AWQ is more activation-aware.

## Evaluation

- **Perplexity (PPL)**: exponentiated negative log-likelihood on test data; lower is better.
- **MMLU**: 57-subject multiple choice; the general-knowledge benchmark.
- **GSM8K**: grade-school to middle-school math; tests reasoning.
- **HumanEval / MBPP**: code-generation benchmarks.
- **MT-Bench**: GPT-4-as-judge for dialogue quality.
- **Chatbot Arena**: blind human-preference leaderboard — the real ranking.
- **HELM**: Stanford's holistic evaluation framework.

## Data

- **CommonCrawl / C4 / The Pile / FineWeb**: common pretraining corpora.
- **Deduplication**: dedup; MinHash / SuiteSparse / etc.
- **Quality filter**: classifier-based or heuristic filtering of low-quality text.
- **Mixture / Curriculum**: data mix ratios and ordering during training.

## RAG / Retrieval

- **Embedding**: maps text to a dense vector.
- **Bi-encoder**: encodes query and document separately, matches via dot product; fast, lower precision.
- **Cross-encoder**: encodes query and document jointly for an exact similarity score; slow, accurate.
- **Reranker**: first recall with a bi-encoder, then re-rank with a cross-encoder.
- **BM25**: classic sparse-retrieval baseline; still in the top tier for hybrid search.
- **ColBERT / Late Interaction**: token-level matching between bi-encoder and cross-encoder.
- **Hybrid Search**: weighted fusion of BM25 + dense retrieval.

## Multimodal

- **VLM (Vision-Language Model)**: a model that processes images and text jointly.
- **ViT (Vision Transformer)**: a transformer over image patches as tokens.
- **CLIP**: the seminal image-text contrastive learning work.
- **VQA (Visual Question Answering)**: question-answering grounded in an image.
- **OCR-free**: model reads text directly from pixels rather than relying on OCR.

## Inference-time Tricks

- **Temperature**: sampling temperature; higher is more random, 0 = greedy.
- **Top-k / Top-p (nucleus)**: two ways to truncate the sampling distribution.
- **Repetition penalty**: penalizes recently emitted tokens to break loops.
- **Beam Search**: maintain N candidate beams during search; still common in translation.
- **CoT (Chain-of-Thought)**: prompt the model to write reasoning steps before the final answer.
- **Few-shot**: include a few examples in the prompt.
- **System prompt**: leading instruction defining the model's role and constraints.

---

[← Table of Contents](README.md) | [Next Chapter →](01-tokenizer.md)
