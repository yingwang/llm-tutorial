[← Table of Contents](README.md)

# Glossary

High-frequency terms in LLM training and deployment, grouped by topic.

> Each entry is kept to 1–2 sentences; for deeper coverage see the corresponding chapter.

---

## Architecture

- **Attention**: Weighted aggregation of value vectors via query-key dot-product affinity; the foundational operator of Transformer architectures.
- **MHA (Multi-Head Attention)**: Partitions representations into parallel projection subspaces to capture diverse relational patterns simultaneously.
- **GQA (Group-Query Attention)**: Multiple query heads share a single key-value head group, substantially reducing KV cache memory footprint (adopted in Llama 2/3).
- **MQA (Multi-Query Attention)**: All query heads share a single key-value head; the extreme limit of GQA, trading marginal representational capacity for maximum memory efficiency.
- **MLA (Multi-head Latent Attention)**: Introduced in DeepSeek-V2; compresses key-value states into a low-rank latent subspace, achieving smaller memory overhead than GQA.
- **RoPE (Rotary Position Embedding)**: Encodes relative positional information by applying rotation matrices to query and key representations; the standard in modern LLMs.
- **ALiBi**: Injects linear attention biases inversely proportional to token distances, offering strong zero-shot length extrapolation without explicit position embeddings.
- **SwiGLU**: A Swish-gated Feed-Forward Network variant that consistently outperforms standard GELU/ReLU activations; the de facto modern default.
- **MoE (Mixture of Experts)**: Replaces dense FFN layers with multiple expert sub-networks; dynamic routing activates only top-$k$ experts per token, decoupling parameter capacity from compute cost.
- **Routing**: The gating mechanism in MoE architectures that dynamically assigns tokens to candidate expert networks based on learned dispatch scores.
- **MTP (Multi-Token Prediction)**: Simultaneously predicts multiple sequential future tokens during training, improving representations and enabling speculative decoding (as in DeepSeek-V3).

## Training Objectives

- **CLM (Causal Language Modeling)**: Autoregressive next-token prediction conditioned on preceding context; the standard objective for GPT-style generative models.
- **MLM (Masked Language Modeling)**: Bidirectional reconstruction of randomly masked tokens; the foundational objective of BERT-style encoders.
- **PrefixLM**: Applies bidirectional attention over prompt prefixes and causal unidirectional attention over target completions; used in models like T5 and GLM.
- **Next-Sentence Prediction (NSP)**: Binary classification auxiliary objective from original BERT; largely phased out in modern pretraining recipes.

## Post-Training

- **SFT (Supervised Fine-Tuning)**: Supervised instruction tuning on curated prompt-response pairs to instill conversational and task-following behavior.
- **RLHF (Reinforcement Learning from Human Feedback)**: Alignment framework aligning model generations with human intent via reward modeling and policy optimization (e.g., PPO or GRPO).
- **RLAIF (Reinforcement Learning from AI Feedback)**: Alignment paradigm where human evaluators are replaced or augmented by scalable model-generated feedback (e.g., Constitutional AI).
- **PPO (Proximal Policy Optimization)**: Actor-critic policy gradient algorithm with clipped surrogate objectives; foundational to early alignment pipelines such as InstructGPT.
- **DPO (Direct Preference Optimization)**: Derives closed-form policy updates directly from preference pairs, bypassing explicit reward model training and reinforcement learning loops.
- **GRPO (Group Relative Policy Optimization)**: Introduced in DeepSeekMath; removes the value critic network by computing baseline advantages over group-sampled completions (powers DeepSeek-R1).
- **KTO (Kahneman-Tversky Optimization)**: Alignment objective derived from prospect theory that operates on binary (thumbs up / thumbs down) feedback without requiring paired preference data.
- **Reward Model (RM)**: Discriminator network (typically sharing the backbone architecture with a scalar projection head) trained to score completion quality according to preference data.
- **KL Penalty**: Regularization term penalizing divergence between the active RL policy and the reference SFT model, preventing reward hacking and policy collapse.
- **Constitutional AI**: Self-alignment methodology where a model critiques and refines its own completions against an explicit set of behavioral principles.

## Parameter-Efficient Fine-Tuning (PEFT)

- **PEFT**: Paradigm of adaptation techniques that update or insert a small fraction of parameters (typically <1%) while freezing the base model weights.
- **LoRA (Low-Rank Adaptation)**: Injects trainable rank-decomposition matrices ($\Delta W = BA$) into linear projections; can be folded directly into frozen weights at inference.
- **QLoRA**: Combines 4-bit NormalFloat (NF4) base weight quantization with double quantization and paged optimizers, enabling 65B+ model fine-tuning on a single consumer GPU.
- **Adapters**: Lightweight bottleneck feed-forward sub-layers inserted sequentially or in parallel within Transformer blocks.
- **IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)**: Scales inner activations via learned element-wise vectors, using even fewer parameters than LoRA.
- **Prefix / P-Tuning**: Prepends learnable continuous virtual token embeddings to key-value representations across attention layers.

## Training Infrastructure

- **DP (Data Parallelism)**: Replicates the model across multiple devices, processing independent micro-batches and synchronizing gradients.
- **TP (Tensor Parallelism)**: Partitions individual weight matrices (e.g., Megatron-style column/row parallel linear layers) across GPUs, requiring high intra-node interconnect bandwidth.
- **PP (Pipeline Parallelism)**: Partitions model layers sequentially across devices and interleaves execution using pipelined micro-batches.
- **3D Parallelism**: Composes DP, TP, and PP to scale pretraining across thousands of GPUs.
- **ZeRO (Zero Redundancy Optimizer)**: Memory optimization paradigm partitioning optimizer states (Stage 1), gradients (Stage 2), and model parameters (Stage 3) across data-parallel ranks.
- **FSDP (Fully Sharded Data Parallel)**: PyTorch native implementation of ZeRO-3 sharding for distributed parameter, gradient, and optimizer states.
- **FlashAttention**: IO-aware exact attention algorithm tiling computation in GPU SRAM to minimize High Bandwidth Memory (HBM) read/write traffic; provides 2-4x speedups.
- **Gradient Checkpointing (Activation Recomputation)**: Trades redundant forward compute for memory by discarding intermediate activations and recomputing them during the backward pass.
- **Mixed Precision**: Uses BF16/FP16 for compute-heavy forward/backward passes while maintaining FP32 master weights and optimizer accumulators.
- **FP8 Training**: Low-precision training format with dynamic scaling factors on Hopper/Blackwell architectures; pioneered at scale in DeepSeek-V3.
- **NCCL**: NVIDIA Collective Communications Library; high-performance multi-GPU primitive library providing optimized AllReduce, AllGather, and ReduceScatter routines.
- **NVLink / InfiniBand**: Ultra-high-bandwidth physical interconnect technologies for intra-node GPU-to-GPU and inter-node cluster fabric communication, respectively.

## Inference

- **KV Cache**: Stores computed key and value states from previous tokens to eliminate redundant computation during autoregressive generation.
- **PagedAttention**: Memory allocation algorithm inspired by OS virtual memory paging; allocates KV cache dynamically in non-contiguous memory blocks, virtually eliminating fragmentation.
- **Continuous Batching (Iteration-level Scheduling)**: Dynamically injects newly arriving requests and evicts finished requests on each decode step, maximizing GPU utilization.
- **Speculative Decoding**: Uses a lightweight draft model to generate speculative candidate tokens, verified in parallel by the target model in a single forward pass.
- **Medusa**: Multi-head speculative architecture generating multiple token predictions concurrently from a single backbone without an auxiliary draft model.
- **Prefix Caching**: Retains and shares precomputed KV cache states for common prompt prefixes (e.g., system prompts) across concurrent requests.
- **Quantization**: Maps high-precision floating-point weights (and optionally activations) to low-bit integer formats (e.g., INT8/INT4/FP8), reducing memory bandwidth pressure.
- **GPTQ / AWQ**: Prominent post-training quantization (PTQ) techniques; AWQ protects salient weight channels by observing activation magnitudes.

## Evaluation

- **Perplexity (PPL)**: Exponentiated cross-entropy loss over a validation corpus; measures how well the probability distribution predicts test tokens (lower is better).
- **MMLU (Massive Multitask Language Understanding)**: Standardized benchmark testing broad knowledge and problem-solving across 57 academic subjects.
- **GSM8K**: Grade-school math reasoning benchmark evaluating multi-step mathematical problem solving.
- **HumanEval / MBPP**: Standard code generation benchmarks assessing functional correctness against unit tests.
- **MT-Bench**: Multi-turn dialogue evaluation benchmark using strong LLMs (such as GPT-4) as judges.
- **Chatbot Arena**: Crowdsourced, blind A/B evaluation platform tracking conversational capabilities via Elo ratings from real-world human interactions.
- **HELM (Holistic Evaluation of Language Models)**: Comprehensive multi-metric evaluation framework assessing accuracy, robustness, fairness, and toxicity.

## Data

- **Common Crawl / FineWeb / RedPajama**: Large-scale public web crawl datasets forming the backbone of modern open-source pretraining corpora.
- **Deduplication**: Removing redundant text instances via MinHash, SimHash, or exact suffix arrays to prevent memorization and improve sample efficiency.
- **Quality Filtering**: Heuristic and classifier-based pipeline filtering out machine-generated spam, toxic content, and low-quality web pages.
- **Curriculum / Data Mixture**: Dynamic scheduling and domain proportioning (code, math, books, web) across pretraining stages to optimize learning trajectories.

## RAG & Retrieval

- **Embedding Model**: Neural encoder mapping natural language text into dense semantic vector representations.
- **Bi-Encoder**: Independently encodes queries and documents into separate vector embeddings, scoring similarity via dot product; fast and scalable for vector index search.
- **Cross-Encoder**: Jointly processes query and document through full self-attention layers; highly accurate but computationally heavy, standard for re-ranking top candidates.
- **Reranker**: Second-stage retrieval module refining candidates returned by fast initial retrieval passes using cross-attention scoring.
- **BM25**: Classical probabilistic lexical retrieval algorithm; remains a powerful baseline and essential component in hybrid retrieval systems.
- **ColBERT / Late Interaction**: Preserves token-level embeddings and performs max-similarity operations across query and document tokens, balancing bi-encoder efficiency with cross-encoder quality.
- **Hybrid Search**: Fuses lexical keyword search (BM25) and dense semantic vector search via Reciprocal Rank Fusion (RRF) or convex score combinations.

## Multimodal

- **VLM (Vision-Language Model)**: Multimodal architecture capable of understanding, grounding, and reasoning over both visual inputs and natural language.
- **ViT (Vision Transformer)**: Architecture that treats non-overlapping 2D image patches as a sequence of input tokens processed by standard Transformer blocks.
- **CLIP (Contrastive Language-Image Pre-Training)**: Dual-encoder model trained on massive image-text pairs via contrastive loss to map vision and text into a shared embedding space.
- **VQA (Visual Question Answering)**: Benchmark task requiring models to answer open-ended questions conditioned on image context.
- **OCR-Free Document Understanding**: Architecture that parses complex textual documents, layouts, and tables directly from raw image pixels without external OCR tools.

## Inference-Time Techniques

- **Temperature**: Scaling parameter applied to output logits prior to softmax; controls randomness (lower values sharpen probabilities toward greedy argmax).
- **Top-$k$ / Top-$p$ (Nucleus) Sampling**: Truncation strategies restricting sampling candidates to the $k$ most probable tokens or the smallest set whose cumulative probability exceeds $p$.
- **Repetition Penalty**: Logit penalty applied to previously generated tokens to suppress repetitive degeneration loops.
- **Beam Search**: Deterministic decoding heuristic maintaining top-$B$ partial sequence hypotheses at each step; widely used in translation and structured extraction.
- **CoT (Chain-of-Thought)**: Prompting or generation strategy where intermediate reasoning steps are emitted sequentially before stating the final conclusion.
- **Few-Shot Prompting**: Conditioning model generation by providing illustrative demonstration examples directly within the context window.
- **System Prompt**: Foundational leading instruction specifying the model's persona, operational constraints, safety policies, and output formats.

---

[← Table of Contents](README.md) | [Next Chapter →](01-tokenizer.md)
