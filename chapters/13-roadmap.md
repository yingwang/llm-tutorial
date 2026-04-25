[← 上一章](12-distillation-merging.md) | [目录](../README.md)

# 第十三章：实战路线图

## 13.1 学习路径

### Phase 1: 基础 (2-4 周)

```
□ 理解 Transformer 架构
  - 手写一个简单的 Attention
  - 读 "Attention Is All You Need"
  - 实现 nanoGPT (Karpathy)

□ 理解 Tokenizer
  - 用 HuggingFace tokenizers 训练一个 BPE
  - 读 SentencePiece 论文

□ PyTorch 熟练
  - 手写训练循环
  - 理解 autograd
  - 用 DataLoader, Dataset
```

### Phase 2: 预训练 (4-8 周)

```
□ 从头训练一个小 LLM
  - 准备数据: 下载 RedPajama-Data-v2 子集
  - 训练 tokenizer
  - 实现 GPT-2 架构 (124M 参数)
  - 单卡训练, 观察 loss 曲线
  - 推荐: nanoGPT, litGPT

□ 多卡训练
  - DDP (torch.nn.parallel.DistributedDataParallel)
  - FSDP
  - 理解 all-reduce, ring-allreduce

□ 数据处理
  - 实现一个简单的数据处理 pipeline
  - 去重 (MinHash)
  - 质量过滤
```

### Phase 3: Post-Training (2-4 周)

```
□ SFT
  - 用 HuggingFace TRL 或 Axolotl
  - 用 LoRA 微调 LLaMA 3 8B
  - 准备 chat format 数据
  - 理解 packing, loss masking

□ DPO
  - 用 TRL 的 DPOTrainer
  - 准备偏好数据
  - 对比 SFT-only vs SFT+DPO

□ RLHF (进阶)
  - 训练 Reward Model
  - 用 TRL 的 PPOTrainer
  - 理解 PPO 的超参调优
```

### Phase 4: 进阶 (持续)

```
□ 高效推理
  - 部署 vLLM
  - 用 GPTQ/AWQ 做量化
  - 实现 speculative decoding

□ RAG
  - 训练或使用 embedding 模型
  - 搭建向量检索 pipeline
  - 优化: hybrid search + reranker

□ 多模态
  - 训练一个简单的 LLaVA
  - 理解 CLIP, SigLIP
  - 视觉指令微调

□ Infra
  - Megatron-LM 使用
  - TP + PP + DP 配置
  - Profile 训练性能 (nsys, tensorboard)
  - 优化 MFU
```

## 13.2 推荐资源

### 必读论文 (带链接)

**基础**:
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani 2017) — Transformer
2. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
3. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)

**Scaling**:
4. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan 2020)
5. [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) (Chinchilla, Hoffmann 2022)

**架构**:
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

**多模态**:
17. [Learning Transferable Visual Models (CLIP)](https://arxiv.org/abs/2103.00020) (Radford 2021)
18. [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) (Liu 2023)
19. [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) (Google 2023)

**Reasoning**:
20. [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) (Wei 2022)

### 代码项目

**入门**:
| 项目 | 用途 |
|------|------|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 从头训练 GPT-2 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | C 语言实现 GPT-2 训练 |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | 最小化 BPE 实现 |

**微调**:
| 项目 | 用途 |
|------|------|
| [huggingface/trl](https://github.com/huggingface/trl) | SFT, DPO, PPO, RLOO |
| [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | 一站式微调 |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 2x 加速 LoRA |
| [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 中文社区常用 |
| [pytorch/torchtune](https://github.com/pytorch/torchtune) | PyTorch 原生 |

**预训练**:
| 项目 | 用途 |
|------|------|
| [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) | 生产级 LLM 训练 |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 大规模预训练 |
| [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | 分布式训练 |

**推理**:
| 项目 | 用途 |
|------|------|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 高效推理服务 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 推理 + 复杂 prompt |
| [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | 本地/CPU 推理 |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 优化推理 |

**多模态**:
| 项目 | 用途 |
|------|------|
| [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | 视觉语言模型 |
| [openai/whisper](https://github.com/openai/whisper) | 语音识别 |
| [huggingface/transformers](https://github.com/huggingface/transformers) | 统一框架 |

**数据 & 评估**:
| 项目 | 用途 |
|------|------|
| [huggingface/datatrove](https://github.com/huggingface/datatrove) | 数据处理 pipeline |
| [allenai/dolma](https://github.com/allenai/dolma) | 数据 toolkit |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 统一评估 |
| [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) | 软件工程评估 |

### 课程

| 课程 | 平台 | 说明 |
|------|------|------|
| [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | YouTube | Karpathy, 最好的入门 |
| [Stanford CS224N](https://web.stanford.edu/class/cs224n/) | Stanford | NLP with Deep Learning |
| [Stanford CS336](https://stanford-cs336.github.io/spring2024/) | Stanford | Language Modeling from Scratch (2024) |
| [CMU 11-868](https://llms-11-868.github.io/) | CMU | Large Language Models (2024) |
| [LLM Course](https://github.com/mlabonne/llm-course) | GitHub | Maxime Labonne, 免费 |
| [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) | 在线 | 工程导向 |

## 13.3 硬件建议

### 个人学习
```
入门: 1x RTX 4090 (24GB)
  → 够训练 1B 模型
  → LoRA 微调 7B
  → QLoRA 微调 70B
  → 本地跑量化 70B

进阶: 2x RTX 4090
  → 学习多卡训练

云 GPU: Lambda, RunPod, vast.ai 按需租用
  → RTX 4090: ~$0.5/hr
  → A100 80GB: ~$2/hr
```

### 小团队
```
4-8x A100/H100
  → 可以预训练 7B
  → 全量微调 70B
  → 云方案: AWS p5, GCP a3, Azure ND H100
```

### 大规模
```
256+ GPU
  → 预训练 70B+
  → 自建或云上长期租约
  → 需要专职 infra 工程师
```

## 13.4 训练成本参考

### 预训练

| 模型规模 | GPU | Token 数 | 约成本 | 时间 |
|----------|-----|---------|--------|------|
| 124M (GPT-2 small) | 1x RTX 4090 | 10B | ~$20 | ~8h |
| 1B | 1x A100 80GB | 20B | ~$500 | ~2天 |
| 7B | 8x A100 80GB | 1T | ~$50K | ~2周 |
| 13B | 32x A100 80GB | 2T | ~$200K | ~3周 |
| 70B | 256x H100 | 2T | ~$2M | ~1月 |

### 微调

| 方法 | 模型 | GPU | 约成本 | 时间 |
|------|------|-----|--------|------|
| QLoRA SFT | 7B | 1x RTX 4090 | ~$5 | ~2h |
| QLoRA SFT | 70B | 1x A100 80GB | ~$20 | ~10h |
| LoRA SFT | 7B | 1x A100 80GB | ~$10 | ~5h |
| 全量 SFT | 7B | 4x A100 80GB | ~$100 | ~12h |
| 全量 SFT | 70B | 32x A100 80GB | ~$2K | ~2天 |
| DPO | 7B | 2x A100 80GB | ~$50 | ~6h |

> 成本基于 A100 ~$2/hr, H100 ~$3.5/hr, RTX 4090 ~$0.5/hr 的云价格估算。

---

# 附录：完整论文清单

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

---

> 最后更新: 2026-04-25
> 持续学习最好的方式：读论文 → 跑代码 → 读论文。

[← 上一章](12-distillation-merging.md) | [目录](../README.md)
