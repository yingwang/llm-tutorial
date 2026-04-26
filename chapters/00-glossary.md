[← 目录](../README.md)

# 术语表

本文档收录 LLM 训练与部署中的高频术语。按主题分组。

> 为节省空间，每个条目只保留 1–2 句要点；详细解释见对应章节。

---

## 架构 (Architecture)

- **Attention（注意力）**：query/key/value 三元组通过点积加权求和；Transformer 的核心算子。
- **MHA (Multi-Head Attention)**：把 attention 拆成多个并行 head 学习不同子空间。
- **GQA (Group-Query Attention)**：多个 q head 共享一组 kv head，显著缩小 KV cache。Llama 2/3 在用。
- **MQA (Multi-Query Attention)**：所有 q head 共享 1 组 kv head；GQA 的极端形式，质量略损。
- **MLA (Multi-head Latent Attention)**：DeepSeek-V2 引入，把 KV 投影到低维潜空间，KV cache 比 GQA 还小。
- **RoPE (Rotary Position Embedding)**：把位置信息以旋转矩阵形式注入 q/k；现代主流位置编码。
- **ALiBi**：用线性偏置代替显式位置编码，外推能力强。
- **SwiGLU**：FFN 的 gated 变体，比 GeLU 略好；几乎是新模型默认。
- **MoE (Mixture of Experts)**：FFN 层用多个专家，每 token 只激活 top-k 个，参数大但 FLOPs 少。
- **Routing**：MoE 中决定每个 token 走哪几个专家的机制。
- **MTP (Multi-Token Prediction)**：训练时一次预测多个未来 token；DeepSeek-V3 用于加速 + 提升质量。

## 训练目标 (Objectives)

- **CLM (Causal LM)**：next-token prediction；GPT 系列的训练目标。
- **MLM (Masked LM)**：随机遮盖 token 让模型还原；BERT 的训练目标。
- **PrefixLM**：前缀双向 attention，后缀单向；T5 / GLM 用过。
- **Next-Sentence Prediction**：BERT 的辅助任务；后被证明意义不大。

## 后训练 (Post-Training)

- **SFT (Supervised Fine-Tuning)**：用 (prompt, response) 对做有监督训练，教模型遵循指令。
- **RLHF**：Reinforcement Learning from Human Feedback；reward model + PPO/GRPO 的三阶段流程。
- **RLAIF**：把 RLHF 的人类反馈替换成 AI 反馈（如 Constitutional AI）。
- **PPO**：最经典的策略梯度算法；OpenAI InstructGPT 用过。
- **DPO (Direct Preference Optimization)**：跳过 reward model，直接用偏好对监督；简单稳定。
- **GRPO**：DeepSeekMath 提出；不需 critic，靠组内相对排名估计 advantage。R1 用的。
- **KTO**：只需 thumbs-up/down 单点反馈，无需 pairwise 偏好。
- **Reward Model (RM)**：给输出打分的模型，通常从同 backbone 加 value head。
- **KL 惩罚**：RL 阶段约束策略不要偏离 SFT 太远，防止 reward hacking。
- **Constitutional AI**：用一组"宪法"原则让模型自我批评、自我改写，用于 RLAIF。

## 参数高效微调 (PEFT)

- **PEFT**：只训练少量参数（通常 <1%）让大模型适配新任务的统称。
- **LoRA**：在 q/k/v/o 矩阵旁加低秩 ΔW = BA，只训 B、A；推理时可合并。
- **QLoRA**：4-bit NF4 量化 backbone + LoRA，单卡可训 65B。
- **Adapter**：在 transformer 层间插入小 MLP 模块。
- **IA³**：每层引入 3 个缩放向量；参数比 LoRA 还少。
- **Prefix Tuning / P-Tuning**：在 attention 输入前拼接可学习 prefix token。

## 训练基础设施 (Infra)

- **DP (Data Parallelism)**：多 GPU 各持完整模型副本，处理不同 batch。
- **TP (Tensor Parallelism)**：把单个矩阵在多 GPU 间切分计算，需要高带宽。
- **PP (Pipeline Parallelism)**：把模型层切到不同 GPU，用 micro-batch 重叠。
- **3D 并行**：DP × TP × PP 三维组合，万卡训练标配。
- **ZeRO**：DeepSpeed 的核心；分别在 DP 上切分 optimizer state / gradient / param 来省显存。
- **FSDP**：PyTorch 的 ZeRO-3 等价实现。
- **FlashAttention**：IO-aware attention，把 softmax 融合在 SRAM 中算，2-4x 加速。
- **Gradient Checkpointing**：用前向重算换显存；大模型训练标配。
- **Mixed Precision**：BF16/FP16 计算 + FP32 优化器状态。
- **FP8 训练**：H100 起的 8-bit 训练，DeepSeek-V3 是首个大规模实战。
- **NCCL**：NVIDIA 的多卡 / 多机通信库；AllReduce / AllGather 的事实标准。
- **NVLink / InfiniBand**：节点内 / 节点间高速互联。

## 推理 (Inference)

- **KV cache**：缓存历史 token 的 K/V，避免重复计算；自回归推理的核心优化。
- **PagedAttention**：vLLM 的 KV cache 分页管理，按需分配，降低碎片。
- **Continuous Batching**：动态拼 batch，新请求随时插入；vLLM/SGLang 标配。
- **Speculative Decoding**：小模型先猜 N 个 token，大模型一次性验证；省 forward。
- **Medusa**：多个并行预测头加速 decoding；Spec Decoding 的简化变体。
- **Prefix caching**：相同 system prompt 只计算一次 KV，跨请求复用。
- **量化 (Quantization)**：把权重从 FP16 → INT8 / INT4，省显存 + 加速；推理几乎无损。
- **GPTQ / AWQ**：两种主流后训练量化方法；AWQ 对激活敏感权重更友好。

## 评测 (Evaluation)

- **Perplexity (PPL)**：模型在测试集上的指数化负对数似然；越低越好。
- **MMLU**：57 学科多选题；通用知识标杆。
- **GSM8K**：小学到中学数学题；测推理。
- **HumanEval / MBPP**：代码生成基准。
- **MT-Bench**：用 GPT-4 当裁判评对话能力。
- **Chatbot Arena**：人类盲测打分的真榜单。
- **HELM**：Stanford 提出的全面评测框架。

## 数据 (Data)

- **CommonCrawl / C4 / The Pile / FineWeb**：常见预训练语料。
- **Deduplication**：去重；MinHash / SuiteSparse 等。
- **Quality filter**：基于分类器或启发式规则过滤低质内容。
- **Mixture / Curriculum**：训练数据的混合配比与顺序。

## RAG / 检索

- **Embedding**：把文本映射成稠密向量。
- **Bi-encoder（双塔）**：query 和 document 分别 encode，点积匹配；快但精度低。
- **Cross-encoder**：query 和 document 一起喂入，计算精确相似度；慢但准。
- **Reranker**：先用 bi-encoder 召回，再用 cross-encoder 精排。
- **BM25**：经典稀疏检索基线；至今仍是混合检索一档。
- **ColBERT / Late Interaction**：token 级匹配，介于 bi-encoder 和 cross-encoder 之间。
- **Hybrid Search**：BM25 + dense 的加权融合。

## 多模态

- **VLM (Vision-Language Model)**：能同时处理图文的模型。
- **ViT (Vision Transformer)**：把图像切 patch 后当 token 喂 Transformer。
- **CLIP**：图文对比学习的开山之作。
- **VQA (Visual Question Answering)**：看图问答任务。
- **OCR-free**：模型直接从图像读字而非依赖 OCR。

## 推理时技巧

- **Temperature**：采样温度；越高越随机，0 = greedy。
- **Top-k / Top-p (nucleus)**：截断采样的两种方式。
- **Repetition penalty**：惩罚最近出现过的 token，避免循环。
- **Beam Search**：维护 N 个候选 beam 的搜索；翻译任务还在用。
- **CoT (Chain-of-Thought)**：让模型先写思考过程再给答案。
- **Few-shot**：在 prompt 里给几个示例。
- **System prompt**：定义模型角色和约束的开头指令。

---

[← 目录](../README.md) | [下一章 →](01-tokenizer.md)
