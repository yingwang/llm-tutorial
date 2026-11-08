[← 上一章](12-distillation-merging.md) | [目录](../README.md)

# 第十三章：实战路线图

大模型涉及的模块很多，光看论文很容易迷失在名词里。最有效的学习路线是边读边动手写代码：先在单卡上把一个一亿参数的小模型从分词、预训练到微调跑通，再去碰多卡并行、强化学习后训练与高并发推理。本章给出一份按阶段拆解的实操清单，并汇总常用的开源工具、课程与论文索引。

## 13.1 分阶段学习路径

### Phase 1：打牢基础（2-4 周）

```
□ 吃透 Transformer 架构
  - 不用现成库，纯 PyTorch 手写 Scaled Dot-Product Attention、Multi-Head Attention、RoPE 和 RMSNorm
  - 精读《Attention Is All You Need》与 GPT-2 论文
  - 跟着 Karpathy 的视频从头敲一遍 nanoGPT 代码

□ 理解 Tokenizer
  - 用 HuggingFace tokenizers 在自己的文本语料上训一个 BPE 分词器
  - 读懂 Byte-level BPE 与 SentencePiece (Unigram) 的区别
  - 观察不同语言和代码的 token 压缩率 (Fertility)

□ 熟练掌握 PyTorch 训练细节
  - 自己手写完整的训练循环 (前向、loss、backward、梯度裁剪、混合精度 AMP、余弦学习率调度)
  - 理解 Autograd 计算图与显存占用分布 (权重、梯度、Adam 优化器状态、激活值各占多少字节)
  - 熟练使用 Dataset、DataLoader 与 `np.memmap` 读取大规模二进制语料
```

### Phase 2：预训练与分布式训练（4-8 周）

```
□ 从头预训练一个小语言模型
  - 数据准备: 下载 FineWeb-Edu 或 RedPajama 的一个小切片 (如 10B tokens)
  - 用训练好的 tokenizer 把文本转成 uint16 二进制分片文件
  - 搭建一个 124M (GPT-2 small 规模) 的 LLaMA 风格模型
  - 在单卡上跑通预训练，盯着训练 Loss 曲线和梯度范数 (Grad Norm) 排查不收敛问题
  - 推荐参考代码: nanoGPT, litgpt

□ 多卡并行训练
  - 从单机多卡 DDP (DistributedDataParallel) 练起，理解 All-Reduce 梯度同步
  - 切换到 FSDP (或 DeepSpeed ZeRO-2 / ZeRO-3)，对比切分优化器状态前后的显存变化
  - 弄明白机内通信 (NVLink) 与跨机通信 (InfiniBand / RoCE) 的带宽差距

□ 数据清洗流水线
  - 写一个小型数据清洗脚本: 语言识别 → 启发式规则过滤 → MinHash LSH 去重 → KenLM 困惑度过滤
```

### Phase 3：后训练与对齐（2-4 周）

```
□ SFT 指令微调
  - 熟悉 HuggingFace TRL、Axolotl 或 LLaMA-Factory
  - 用 LoRA / QLoRA 在单张显卡上微调一个 7B/8B 开源基座 (如 Qwen2.5-7B 或 Llama-3.1-8B)
  - 理解 Chat Template 特殊 token 的拼接格式，确认只对 assistant 回复部分算 Loss (Loss Masking)
  - 理解多样本拼接 (Sequence Packing) 为什么能避免 Padding 浪费

□ DPO 偏好对齐
  - 用 TRL 的 DPOTrainer 在成对偏好数据 (如 UltraFeedback) 上跑通 DPO
  - 盲测对比 SFT 模型和 SFT+DPO 模型的回答风格变化，观察 KL 惩罚系数 β 设太大或太小会发生什么

□ 强化学习与推理模型 (进阶)
  - 在 GSM8K 数学题上用 TRL 跑通 GRPO 训练: 写一个正则表达式提取最终数字答案做 0/1 奖励
  - 观察训练过程中模型回答长度与准确率的变化
```

### Phase 4：推理部署、RAG、多模态与底层 Infra（持续深入）

```
□ 高性能推理与部署
  - 用 vLLM 或 SGLang 把微调好的模型部署成 OpenAI 兼容 API，用压测脚本测并发吞吐
  - 用 AWQ 或 GPTQ 把模型压到 4-bit，对比量化前后的显存占用、生成速度与榜单掉点
  - 配置投机采样 (Speculative Decoding) 测试低并发下的加速比

□ 检索增强生成 (RAG)
  - 不用重型框架，自己用 BGE-M3 + Qdrant/FAISS + LLM 写一个 100 行代码的 RAG 系统
  - 加上 BM25 混合检索 (RRF 融合) 和 Cross-Encoder Reranker，在测试集上量化每一步的召回提升

□ 多模态大模型
  - 读懂 LLaVA 代码，冻结 SigLIP 和小尺寸 LLM，只训两层 MLP Projector 跑通图文对齐
  - 了解动态分辨率切块与视觉 token 压缩方法

□ 大规模训练 Infra
  - 学习使用 Megatron-LM 配置 3D 并行 (TP + PP + DP)
  - 用 PyTorch Profiler 和 Nsight Systems (nsys) 抓出训练过程中的通信空泡与慢算子，优化 MFU
```

## 13.2 推荐资源清单

### 二十篇必读论文

**基础与上下文学习**：
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)：提出 Transformer 架构。
2. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019)：GPT-2，确立自回归预训练解决多任务的路线。
3. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)：GPT-3（175B），展示上下文少样本学习（In-Context Learning）。

**Scaling Laws**：
4. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)：发现损失随算力、参数量和数据的幂律下降关系。
5. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022)：Chinchilla 法则，证明参数量与训练 token 数应按约 $1:20$ 同比例放大。

**现代模型架构**：
6. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023) & [Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024)：现代开源稠密模型的标准模板与全流程工程报告。
7. [Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023)：GQA 与滑动窗口注意力的紧凑实现。
8. [DeepSeek-V2](https://arxiv.org/abs/2405.04434) & [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (2024)：多头潜在注意力（MLA）、细粒度 MoE 与 FP8 训练。

**后训练与推理**：
9. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022)：InstructGPT，确立 `SFT → RM → PPO` 的经典三步法。
10. [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)：绕开独立奖励模型，直接用分类损失做偏好优化。
11. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025)：用 GRPO 与规则可验证奖励激发出长思维链推理能力。

**高效微调（PEFT）**：
12. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)：冻结原权重、只训低秩旁路矩阵的标准方法。
13. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)：NF4 量化加分页优化器，让消费级单卡也能微调大模型。

**分布式系统与底层算子**：
14. [Megatron-LM](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019)：张量并行（TP）与流水线并行（PP）的奠基工作。
15. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2020)：切分优化器状态、梯度和参数的显存优化方案。
16. [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)：利用 GPU SRAM 分块计算与在线 Softmax 避免读写巨型注意力矩阵。

**多模态与思维链**：
17. [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)：图文对比学习基座。
18. [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) (Liu et al., 2023)：用简单 MLP 拼接视觉编码器与 LLM 的视觉指令微调。
19. [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) (Google, 2023)：原生多模态架构。
20. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) (Wei et al., 2022)：证明写出中间推导步骤能大幅提升大模型推理准确率。

### 常用开源代码库

**从零手写与入门**：
| 项目 | 用途 |
|------|------|
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 不到千行 PyTorch 代码从头训练 GPT-2 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | 纯 C/CUDA 实现 GPT-2 训练，帮你看清底层矩阵乘与梯度传递 |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | 极简 BPE 分词器实现 |

**微调与后训练**：
| 项目 | 用途 |
|------|------|
| [huggingface/trl](https://github.com/huggingface/trl) | 官方维护的 SFT、DPO、PPO、GRPO 训练库 |
| [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) | 只需写 YAML 配置文件就能跑各种微调的瑞士军刀 |
| [unslothai/unsloth](https://github.com/unslothai/unsloth) | 手写 Triton 算子加速 LoRA 微调，省显存且速度快一倍 |
| [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 带 WebUI 界面的微调工具，中文社区文档齐全，支持百余种模型 |
| [pytorch/torchtune](https://github.com/pytorch/torchtune) | PyTorch 官方出品的纯原生微调库，代码干净易改 |

**大规模预训练**：
| 项目 | 用途 |
|------|------|
| [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) | 20 多种主流开源模型结构的清晰单文件实现，适合预训练与微调 |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 大集群 3D 并行（TP/PP/DP）预训练工业标准 |
| [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | ZeRO-1/2/3 显存切分与分布式训练加速库 |

**推理部署**：
| 项目 | 用途 |
|------|------|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 基于 PagedAttention 的高吞吐推理服务引擎 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 带 RadixAttention 前缀缓存与极速结构化输出的推理引擎 |
| [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | 无外部依赖的纯 C/C++ 本地与 Apple Silicon 推理工具 |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 针对 NVIDIA 显卡深度定制的推理加速库 |

**多模态、数据处理与评测**：
| 项目 | 用途 |
|------|------|
| [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | 视觉语言模型两阶段微调模板 |
| [openai/whisper](https://github.com/openai/whisper) | 多语言语音识别基座 |
| [huggingface/datatrove](https://github.com/huggingface/datatrove) | HuggingFace 处理 FineWeb 用的大规模分布式语料清洗去重工具 |
| [allenai/dolma](https://github.com/allenai/dolma) | AI2 开源的万亿级预训练语料清洗工具包 |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 标准大模型评测跑分工具 |
| [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) | 真实 GitHub Issue 修复能力评测基准 |

### 推荐公开课

| 课程 | 平台 / 讲师 | 特点 |
|------|------|------|
| [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | YouTube (Andrej Karpathy) | 最好的动手入门课，从自动求导一直讲到手写 GPT-2 和 Tokenizer |
| [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/) | Stanford | 专门讲怎么从零写数据清洗、分词、Transformer、并行训练和对齐的深入工程课 |
| [Stanford CS224N](https://web.stanford.edu/class/cs224n/) | Stanford (Chris Manning) | NLP 与深度学习基础名课 |
| [CMU 11-868: Large Language Models](https://llms-11-868.github.io/) | CMU | 偏系统和最新前沿论文精读 |
| [LLM Course](https://github.com/mlabonne/llm-course) | GitHub (Maxime Labonne) | 偏重开源微调、量化、合并实战的免费教程 |
| [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) | FSDL | 面向应用开发、Prompt 工程与生产部署 |

## 13.3 硬件配置建议

### 个人学习与实验
```
入门单卡: 1× RTX 4090 (24GB 显存) 或 Mac Studio (64GB+ 统一内存)
  → 从头预训练 100M - 500M 的小模型
  → 用 LoRA 微调 7B/8B 模型
  → 用 QLoRA (4-bit) 微调 32B/70B 模型
  → 本地跑 4-bit 量化版 32B/70B 推理

进阶双卡: 2× RTX 3090 / 4090
  → 在本地练习 DDP、FSDP 与张量并行通信

按需租云 GPU (推荐平时用笔记本，跑实验时按小时租卡):
  → 平台: RunPod, Lambda Labs, vast.ai, AutoDL
  → RTX 4090: ~$0.4 - $0.5 / 小时
  → A100 80GB: ~$1.5 - $2.0 / 小时
  → H100 80GB: ~$2.5 - $3.5 / 小时
```

### 小团队与初创公司
```
1 台 8× A100 (80GB) 或 8× H100 节点:
  → 足够做 7B-14B 模型的持续预训练 (Continued Pretraining)
  → 足够做 70B 模型的全参数微调 (Full SFT) 与 DPO/GRPO 训练
```

### 大规模预训练团队
```
32 台以上 8× H100 节点 (256+ GPUs，配 InfiniBand 或 RoCE 高速互联):
  → 从头预训练百亿到千亿参数基座模型
  → 需要专职的分布式系统与集群运维工程师处理节点故障与通信瓶颈
```

## 13.4 常见任务耗时与账单速查

### 预训练成本估算

| 模型规模 | GPU 配置 | 训练 Token 数 | 预估耗时 | 预估云租卡账单 |
|----------|-----|---------|--------|------|
| 124M (GPT-2 small) | 1× RTX 4090 | 10B | ~31 小时 | ~$16 |
| 1B | 1× A100 80GB | 20B | ~11 天 | ~$530 |
| 1B (充分过训练) | 8× A100 80GB | 100B | ~7 天 | ~$2,700 |
| 7B | 8× H100 80GB | 1T | ~154 天 | ~$103,000 |
| 13B | 32× H100 80GB | 2T | ~143 天 | ~$383,000 |
| 70B | 256× H100 80GB | 2T | ~96 天 | ~$2,060,000 |

### 微调成本估算（以 1-5 万条高质量指令数据为例）

| 方法 | 模型规模 | GPU 配置 | 预估耗时 | 预估云租卡账单 |
|------|------|-----|--------|------|
| QLoRA SFT | 7B/8B | 1× RTX 4090 | ~1-2 小时 | ~$1 |
| LoRA SFT | 7B/8B | 1× A100 80GB | ~2-3 小时 | ~$5 |
| 全参数 SFT | 7B/8B | 4× A100 80GB | ~3-5 小时 | ~$30 |
| QLoRA SFT | 70B | 1× A100 80GB | ~8-10 小时 | ~$20 |
| LoRA SFT | 70B | 8× H100 80GB | ~6-10 小时 | ~$250 |
| LoRA DPO | 7B/8B | 2× A100 80GB | ~3-4 小时 | ~$15 |

> 以上按 RTX 4090 ~$0.5/h、A100 ~$2.0/h、H100 ~$3.5/h 的按需租价估算。预训练耗时按 7.6.1 节的 $6ND$ 公式计算，MFU 取 0.4，BF16 峰值算力取 H100 989、A100 312、RTX 4090 约 165 TFLOPS。70B 全参数 SFT 仅权重、梯度与 AdamW 状态就需约 1.1TB 显存，单台 8× H100 装不下，需要多机配合 ZeRO-3 / FSDP。实际耗时取决于序列长度与是否开启 FlashAttention / Sequence Packing。

---

# 附录：按主题分类论文清单

## 分词器（Tokenizer）
- [Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909)
- [Kudo (2018): Subword Regularization (Unigram)](https://arxiv.org/abs/1804.10959)
- [Kudo & Richardson (2018): SentencePiece](https://arxiv.org/abs/1808.06226)

## 模型架构（Architecture）
- [Vaswani et al. (2017): Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [Shazeer (2019): Fast Transformer Decoding (MQA)](https://arxiv.org/abs/1911.02150)
- [Zhang & Sennrich (2019): Root Mean Square Layer Normalization (RMSNorm)](https://arxiv.org/abs/1910.07467)
- [Shazeer (2020): GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202)
- [Su et al. (2021): RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
- [Dao et al. (2022): FlashAttention](https://arxiv.org/abs/2205.14135)
- [Ainslie et al. (2023): GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Dao (2023): FlashAttention-2](https://arxiv.org/abs/2307.08691)

## 预训练与 Scaling Laws
- [Radford et al. (2019): Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Kaplan et al. (2020): Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Brown et al. (2020): Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Hoffmann et al. (2022): Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Touvron et al. (2023a): LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Touvron et al. (2023b): LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Dubey et al. (2024): The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-AI (2024a): DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-AI (2024b): DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-AI (2025): DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## 后训练与对齐（Post-Training）
- [Schulman et al. (2017): Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/abs/1707.06347)
- [Ouyang et al. (2022): Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Bai et al. (2022): Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Rafailov et al. (2023): Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [Lightman et al. (2023): Let's Verify Step by Step (PRM)](https://arxiv.org/abs/2305.20050)
- [Shao et al. (2024): DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)

## 参数高效微调（PEFT）
- [Li & Liang (2021): Prefix-Tuning](https://arxiv.org/abs/2101.00190)
- [Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Dettmers et al. (2023): QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Liu et al. (2024): DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

## 混合专家模型（MoE）
- [Shazeer et al. (2017): Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [Fedus et al. (2022): Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Jiang et al. (2024): Mixtral of Experts](https://arxiv.org/abs/2401.04088)

## 多模态（Multimodal）
- [Radford et al. (2021): Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Rombach et al. (2022): High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Alayrac et al. (2022): Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [Peebles & Xie (2023): Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [Liu et al. (2023): Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
- [Team Gemini (2023): Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805)

## 复杂推理（Reasoning）
- [Wei et al. (2022): Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Snell et al. (2024): Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)

## 训练与推理基础设施（Infra & Inference）
- [Shoeybi et al. (2019): Megatron-LM](https://arxiv.org/abs/1909.08053)
- [Rajbhandari et al. (2020): ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Narayanan et al. (2021): Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Leviathan et al. (2023): Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Kwon et al. (2023): Efficient Memory Management for Large Language Models with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)

## 进一步阅读

- 书籍：Sebastian Raschka 著 *Build a Large Language Model (From Scratch)*（手把手从零实现 GPT）
- 研讨课：[Stanford CS25: Transformers United](https://web.stanford.edu/class/cs25/)
- 课程：[NYU / Yann LeCun: Deep Learning](https://atcold.github.io/NYU-DLSP21/)
- 技术博客与通讯：[Sebastian Raschka: Ahead of AI](https://magazine.sebastianraschka.com/)、[Nathan Lambert: Interconnects](https://www.interconnects.ai/)、[Jay Alammar's Blog](https://jalammar.github.io/)

## 毕业实战项目

1. **端到端训练一个 100M 小模型**：用 `nanoGPT` 在公开语料（如 FineWeb-Edu 或中文小语料）上从零预训练一个 124M 模型，接着带它走完 `SFT 指令微调 → DPO 对齐 → 4-bit 量化 → vLLM 部署` 的全套流程。
2. **搭建一套带评测的垂直领域 RAG**：找一个包含 1000 篇以上文档的专业知识库，自己实现分块、向量入库、BM25+向量混合检索、Reranker 重排与引用生成，并构造 50 道测试题量化每一步改进带来的准确率变化。
3. **复现一篇近期论文的核心模块**：挑一个近一年的代表性设计（如 GRPO、Multi-Token Prediction 或 MLA 潜在注意力），在小模型上写出百行以内的复现代码，对比开关该模块时的显存与收敛曲线差异。

---

> 持续学习最管用的办法：读论文 → 跑代码 → 看报错 → 再读论文。

[← 上一章](12-distillation-merging.md) | [目录](../README.md)
