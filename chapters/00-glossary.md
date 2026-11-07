[← 目录](../README.md)

# 术语表

这份文档整理了大语言模型训练与部署中的常见术语，按主题分组排列。

> 为了方便快速查阅，每个词条只保留一两句核心解释；具体原理与工程细节见对应章节。

---

## 架构 (Architecture)

- **Attention（注意力）**：用 Query、Key、Value 三组向量做点积加权求和，是 Transformer 汇总上下文信息的核心算子。
- **MHA (Multi-Head Attention)**：把注意力拆成多个并行的头，分别在不同的子空间里捕捉语法和语义关联。
- **GQA (Group-Query Attention)**：让多个 Query 头共享同一组 Key/Value 头，在保持模型效果的同时大幅缩小 KV Cache，是 Llama 2/3 等主流开源模型的标配。
- **MQA (Multi-Query Attention)**：所有 Query 头共享唯一一组 Key/Value 头，可视为 GQA 的极端情况；显存省得最多，但生成质量略有折损。
- **MLA (Multi-head Latent Attention)**：DeepSeek-V2 引入的注意力机制，先把 Key 和 Value 压缩到低维潜空间再缓存，KV Cache 占用比 GQA 更小。
- **RoPE (Rotary Position Embedding)**：通过旋转矩阵把位置信息注入 Query 和 Key，使内积自然反映相对距离，是当前主流的位置编码方案。
- **ALiBi**：不加显式位置向量，而是在注意力分数上直接减去与相对距离成正比的偏置，长度外推能力较好。
- **SwiGLU**：带门控机制（GLU）和 Swish 激活函数的前馈网络变体，效果稳定优于普通 GeLU，基本是现代大模型的默认选择。
- **MoE (Mixture of Experts)**：把每层的前馈网络拆成多个专家模块，每个 Token 只由路由网络激活其中 Top-k 个，从而在总参数量很大时依然保持较低的单步计算量。
- **Routing**：MoE 架构里决定每个 Token 分发给哪几个专家、并兼顾负载均衡的调度机制。
- **MTP (Multi-Token Prediction)**：训练时让模型一次预测未来连续多个 Token，DeepSeek-V3 用它提升表示质量并配合推理阶段的投机采样。

## 训练目标 (Objectives)

- **CLM (Causal Language Modeling)**：采用单向掩码自回归预测下一个 Token，是 GPT 等生成式大模型的标准训练目标。
- **MLM (Masked Language Modeling)**：随机遮盖输入里的部分 Token，让模型利用双向上下文还原，是 BERT 等编码器模型的训练目标。
- **PrefixLM**：输入前缀部分使用双向注意力，生成目标部分使用单向因果注意力，T5 和早期 GLM 采用过这种形式。
- **Next-Sentence Prediction**：BERT 早期用来判断两句话是否相连的辅助二分类任务，后续研究发现收益有限，现已基本弃用。

## 后训练 (Post-Training)

- **SFT (Supervised Fine-Tuning)**：用指令和回答配对的数据做有监督微调，教会基座模型理解人类意图并按格式作答。
- **RLHF (Reinforcement Learning from Human Feedback)**：先用人类偏好数据训练奖励模型，再用 PPO 或 GRPO 等强化学习算法调整策略模型。
- **RLAIF (Reinforcement Learning from AI Feedback)**：用强模型或预定义原则（如 Constitutional AI）代替人工标注偏好数据，降低对齐成本。
- **PPO (Proximal Policy Optimization)**：通过裁剪重要性采样比率来限制单步更新幅度的经典策略梯度算法，曾用于 InstructGPT。
- **DPO (Direct Preference Optimization)**：跳过独立的奖励模型和在线采样，直接把偏好对比数据转化为交叉熵形式的分类损失来训练模型。
- **GRPO (Group Relative Policy Optimization)**：DeepSeekMath 与 DeepSeek-R1 采用的强化学习算法，对同一提示采样一组回答，用组内相对得分代替独立的 Critic 模型来估计优势。
- **KTO (Kahneman-Tversky Optimization)**：只需要对单条回答给出好或坏的二元反馈即可训练，不要求成对的偏好比较数据。
- **Reward Model (RM)**：给模型回答打分的评估模型，通常在预训练基座顶部换上标量输出头训练而成。
- **KL 惩罚**：强化学习阶段在目标函数里加入的 KL 散度约束，防止当前策略偏离 SFT 参考模型太远、出现钻奖励空子（Reward Hacking）的现象。
- **Constitutional AI**：给定一套行为准则，让模型对自己的回答做批评和改写，再拿这些生成的数据做对齐训练。

## 参数高效微调 (PEFT)

- **PEFT (Parameter-Efficient Fine-Tuning)**：冻结原模型绝大多数权重、只训练极小比例（通常不到 1%）新增参数的微调方法统称。
- **LoRA (Low-Rank Adaptation)**：在原始权重矩阵旁并联低秩分解矩阵 $\Delta W = BA$，微调时只更新 $B$ 和 $A$，推理时可直接合并回原权重，不增加推理延迟。
- **QLoRA**：把基座模型量化成 4-bit NF4 格式，再配合双重量化和分页优化器训练旁路 LoRA，使单张消费级显卡也能微调数十亿到上百亿参数的模型。
- **Adapter**：在 Transformer 各层之间串行插入小型瓶颈 MLP 模块来适配下游任务。
- **IA³**：在每层引入三个可学习的缩放向量，直接乘在 Key、Value 和 FFN 激活值上，可训练参数比 LoRA 更少。
- **Prefix Tuning / P-Tuning**：在输入或每层注意力前拼接一段可学习的连续虚拟 Token，引导冻结模型适应特定任务。

## 训练基础设施 (Infra)

- **DP (Data Parallelism)**：每张 GPU 放一份完整模型，各自处理不同的数据批次，反向传播后通过 AllReduce 同步梯度。
- **TP (Tensor Parallelism)**：把单层的权重矩阵按行或按列切分到多张 GPU 上协同计算，要求卡间具备极高通信带宽（如 NVLink）。
- **PP (Pipeline Parallelism)**：按模型层数切分到不同 GPU 或节点上，并把数据批次拆成微批次（Micro-batch）做流水线调度以减少设备空闲。
- **3D 并行**：同时组合数据并行（DP）、张量并行（TP）和流水线并行（PP），是万卡集群训练超大模型的标准方案。
- **ZeRO (Zero Redundancy Optimizer)**：DeepSpeed 提出的显存优化技术，按阶段将优化器状态（ZeRO-1）、梯度（ZeRO-2）和模型参数（ZeRO-3）切分到各个数据并行进程。
- **FSDP (Fully Sharded Data Parallel)**：PyTorch 原生实现的完全分片数据并行方案，机制对应 ZeRO-3。
- **FlashAttention**：利用 GPU 的 SRAM 局部性对注意力计算做分块融合，搭配在线 Softmax 避免读写完整的 $N \times N$ 注意力矩阵，省显存且显著提速。
- **Gradient Checkpointing**：前向传播时不保存全部中间激活值，反向传播需要时再局部重算，用大约 25%–33% 的额外算力换取大幅显存节省。
- **Mixed Precision**：前向和反向计算使用 BF16 或 FP16，同时在优化器里保留一份 FP32 主权重做累加更新，兼顾吞吐量与数值稳定性。
- **FP8 训练**：在 H100 等支持 8-bit 浮点 Tensor Core 的硬件上用 E4M3/E5M2 格式做矩阵乘与通信，DeepSeek-V3 在超大规模预训练中验证了这套方案。
- **NCCL**：NVIDIA 提供的多卡与多机集合通信库，负责底层 AllReduce、AllGather、ReduceScatter 等操作。
- **NVLink / InfiniBand**：分别用于机内 GPU 之间和跨节点服务器之间的高速低延迟互连网络。

## 推理 (Inference)

- **KV Cache**：自回归生成时把历史 Token 算出的 Key 和 Value 张量缓存在显存里，后续每步只需计算最新 Token，避免重复前向计算。
- **PagedAttention**：vLLM 核心机制，借鉴操作系统的虚拟内存分页管理，把 KV Cache 拆成固定大小的块按需分配，基本消除显存碎片。
- **Continuous Batching**：按单步迭代动态管理请求队列，已生成完的请求立刻退出、新到达的请求随时插入，大幅提高推理吞吐。
- **Speculative Decoding（投机采样）**：先用轻量草稿模型快速连猜几个 Token，再交由主模型一次前向并行验证，在不改变输出分布的前提下降低解码延迟。
- **Medusa**：在主模型顶部直接加几个并行的预测头来猜测后续 Token，不需要额外部署独立的草稿模型。
- **Prefix Caching**：跨请求缓存并复用公共前缀（如长 System Prompt 或共享文档）的 KV Cache，减少首字延迟（TTFT）和重复算力开销。
- **量化 (Quantization)**：把模型权重或激活值从 FP16/BF16 压缩到 INT8、INT4 或 FP8，大幅降低显存占用和访存带宽需求。
- **GPTQ / AWQ**：常用的后训练量化（PTQ）算法；GPTQ 利用二阶海森矩阵逐列补偿误差，AWQ 则根据激活值幅度重点保护约 1% 的关键权重通道。

## 评测 (Evaluation)

- **Perplexity (PPL)**：模型在测试语料上的指数化交叉熵损失，数值越低说明模型对文本分布的预测越准。
- **MMLU**：涵盖人文、社科、理工等 57 个科目的多选题集合，常用来衡量基座模型的综合知识储备。
- **GSM8K**：由小学高年级数学应用题组成的评测集，主要考察多步算术与逻辑推导能力。
- **HumanEval / MBPP**：通过运行单元测试（Pass@k）来检验代码生成准确率的基准测试。
- **MT-Bench**：用 GPT-4 等强模型充当裁判，对多轮开放式问答打分的评测集。
- **Chatbot Arena**：由真实用户匿名盲测两个模型的回答并投票，按 Elo 积分排名的公开榜单。
- **HELM**：斯坦福大学推出的多维度语言模型综合评测框架。

## 数据 (Data)

- **CommonCrawl / C4 / The Pile / FineWeb**：预训练阶段常用的大规模公开网页抓取与清洗数据集。
- **Deduplication（去重）**：用 MinHash LSH 或后缀数组等方法在文档级、段落级剔除重复语料，防止模型死记硬背并提高训练效率。
- **Quality Filter**：用规则启发式、困惑度阈值或轻量分类器过滤乱码、广告与低质网页。
- **Mixture / Curriculum**：不同来源语料（网页、代码、数学、书籍）的配比方案，以及随训练阶段调整数据分布的课程学习策略。

## RAG / 检索 (Retrieval-Augmented Generation)

- **Embedding**：用编码器把文本映射到固定维度的稠密向量，使语义相近的文本在向量空间中距离更近。
- **Bi-encoder（双塔）**：问题和文档分别独立编码成向量，检索时直接算内积或余弦相似度，速度极快，适合大规模初筛召回。
- **Cross-encoder（交叉编码器）**：把问题和候选文档拼接在一起送入 Transformer 做全注意力交互，打分更准但计算开销大，无法直接用于全库检索。
- **Reranker（重排器）**：在双塔向量检索召回 Top-K 个候选后，用 Cross-encoder 对这批候选做精细重排。
- **BM25**：基于词频（TF）、逆文档频率（IDF）和文档长度归一化的经典字面匹配检索算法，常与向量检索配合使用。
- **ColBERT / Late Interaction**：保留每个 Token 的向量表示，检索时通过最大相似度求和（MaxSim）做细粒度交互，兼顾双塔的检索速度与交叉编码器的匹配精度。
- **Hybrid Search**：将 BM25 稀疏检索与稠密向量检索的结果加权融合（如通过 RRF 倒数秩融合），兼顾关键词精确匹配和语义召回。

## 多模态 (Multimodality)

- **VLM (Vision-Language Model)**：能同时接收图像和文本输入、进行跨模态理解与回答的视觉语言模型。
- **ViT (Vision Transformer)**：把图像切成固定大小的网格块（Patch），线性映射后当作 Token 序列送入 Transformer 编码。
- **CLIP**：用海量图文对做对比学习，将图像和文本对齐到同一个向量空间的经典双塔模型。
- **VQA (Visual Question Answering)**：给定图像和自然语言问题，让模型结合画面内容作答的任务。
- **OCR-free**：不依赖外部 OCR 工具提取文字，而是让视觉语言模型直接从高分辨率图像中读取排版和文字内容。

## 解码采样技巧 (Decoding Strategies)

- **Temperature**：调整 Softmax 输出分布平滑度的温度参数；值越大采样越随机，设为 0 时等价于每次选最大概率词的贪心解码（Greedy Decoding）。
- **Top-k / Top-p (Nucleus Sampling)**：截断长尾低概率词的采样策略；Top-k 固定保留概率最高的 $k$ 个词，Top-p 则动态保留累计概率达到 $p$ 的最小词集。
- **Repetition Penalty**：对上下文里已经出现过的 Token 打折降权，抑制模型原地复读或陷入死循环。
- **Beam Search**：解码时同时保留概率最高的 $N$ 条候选序列往前搜索，早期机器翻译常用，开放式对话生成中较少使用。
- **CoT (Chain-of-Thought)**：让模型在给出最终答案前先把中间推理步骤写出来，能明显提升数学、逻辑和复杂问题上的表现。
- **Few-shot**：在提示词里直接放几个输入输出示例，引导模型按示范格式和逻辑完成任务（即上下文学习 In-Context Learning）。
- **System Prompt**：放在对话最前面的全局指令，用来设定模型的身份角色、回复风格与安全边界。

---

[← 目录](../README.md) | [下一章 →](01-tokenizer.md)
