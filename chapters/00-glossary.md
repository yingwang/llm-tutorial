[← 目录](../README.md)

# 术语表

本文档系统收录大语言模型全栈训练与部署体系中的核心技术术语。内容按技术范畴结构化分组呈现。

> 为契合快速检索需求，各词条提炼为精准精要的定义与机理概述；深层理论推导与工程实现请参阅对应章节。

---

## 架构 (Architecture)

- **Attention（注意力机制）**：基于 Query-Key-Value 三元组的点积相似度动态分配表征权重，构成 Transformer 架构的核心信息路由算子。
- **MHA (Multi-Head Attention)**：将隐层空间投影至多个正交子空间并行计算注意力，捕捉多粒度的语法、语义与长程依赖。
- **GQA (Group-Query Attention)**：多组 Query 共享同一组 Key/Value 投影头，在维持模型表征容量的同时显著压缩 KV Cache 显存占用，已成为主流开源基座的标准配置。
- **MQA (Multi-Query Attention)**：所有 Query 头共享单一组 Key/Value 头，为 GQA 的极限退化形式，显存压缩收益最大，但表达容量略有折损。
- **MLA (Multi-head Latent Attention)**：DeepSeek 提出的低秩潜空间注意力机制，将 Key 与 Value 联合压缩至低维流形投影，使推理期 KV Cache 占用远低于 GQA。
- **RoPE (Rotary Position Embedding)**：通过复数旋转正交变换将绝对位置信息以相对距离衰减的形式隐式注入 Query 与 Key 向量，为当前大模型位置编码的标准方案。
- **ALiBi**：在注意力得分矩阵上直接引入随相对距离线性递减的偏置惩罚，具备卓越的序列长度外推特性。
- **SwiGLU**：引入门控线性单元（GLU）与 Swish 激活函数增强 FFN 层的非线性特征筛选能力，为现代前沿大模型的主流选择。
- **MoE (Mixture of Experts)**：将前馈网络解耦为多个稀疏专家模块，每个 Token 仅由门控网络动态激活 Top-k 个专家，在拓宽参数容量的同时维持恒定的单 Token 计算复杂度。
- **Routing**：MoE 架构中基于门控概率分布将 Token 动态分发至目标专家并实现负载均衡的核心调度机制。
- **MTP (Multi-Token Prediction)**：在主干网络后并行构建多个预测头预测未来连续 Token，在加速推理推演的同时显著强化训练期的表征泛化能力。

## 训练目标 (Objectives)

- **CLM (Causal Language Modeling)**：自回归单向自注意力下的因果语言建模，通过最大化自左向右的 Next-Token 条件似然概率训练生成式大模型。
- **MLM (Masked Language Modeling)**：随机掩码输入序列的部分 Token 并借助双向上下文进行完形填空重构，多用于编码器表征模型。
- **PrefixLM**：在输入前缀部分采用双向全连接注意力，在目标生成部分采用因果单向注意力，兼顾上下文理解与自回归生成。
- **Next-Sentence Prediction**：早期 BERT 用于判断句子连贯性的辅助二分类目标，后续研究证实其对深度语义表征增益有限，现代基座多已弃用。

## 后训练 (Post-Training)

- **SFT (Supervised Fine-Tuning)**：在高质量指令与回答对构建的样本集上进行有监督微调，引导预训练基座模型掌握对话交互与指令遵循范式。
- **RLHF (Reinforcement Learning from Human Feedback)**：通过人类偏好标注训练奖励模型，并借助 PPO 或 GRPO 等策略梯度算法优化模型输出分布。
- **RLAIF (Reinforcement Learning from AI Feedback)**：利用高阶模型或宪法原则（Constitutional AI）生成的机器反馈替代人工标注，实现低成本、可扩展的模型偏好对齐。
- **PPO (Proximal Policy Optimization)**：基于重要性采样与裁剪目标函数的经典强化学习策略梯度优化算法，用于约束策略更新步幅。
- **DPO (Direct Preference Optimization)**：基于隐式奖励函数将偏好对齐直接解析推导为分类交叉熵损失，摆脱独立奖励模型的训练与在线采样开销。
- **GRPO (Group Relative Policy Optimization)**：DeepSeek 提出的免 Critic 强化学习算法，通过单 Prompt 多样化采样组内的相对优劣评估优势函数，大幅节约显存开销。
- **KTO (Kahneman-Tversky Optimization)**：基于前景理论直接利用单样本正负二元反馈信号进行对齐优化，无需成对偏好数据。
- **Reward Model (RM)**：通常基于同源预训练骨干网络后接标量投影头构建，负责评估文本序列契合人类偏好或客观规范的质量得分。
- **KL 惩罚**：在强化学习目标函数中引入相对熵惩罚项，约束当前策略分布不偏离参考策略过远，防止策略塌陷与奖励作弊（Reward Hacking）。
- **Constitutional AI**：通过一组预定义的行为原则引导模型进行自我反思与迭代修订，构建高质量对齐样本。

## 参数高效微调 (PEFT)

- **PEFT (Parameter-Efficient Fine-Tuning)**：在冻结基座绝大部分参数的前提下，仅微调极小比例（通常低于 1%）附加参数的适配技术总称。
- **LoRA (Low-Rank Adaptation)**：在原始密集权重矩阵旁路引入低秩分解矩阵（$\Delta W = BA$），推理期可将增量无缝折叠回原始权重，实现零额外时延。
- **QLoRA**：结合 4-bit NormalFloat（NF4）基座量化、双重量化与分页优化器技术，极大压缩微调显存门槛。
- **Adapter**：在 Transformer 模块层间插入瓶颈结构的轻量级 MLP 残差分支以注入下游任务知识。
- **IA³**：通过引入可学习的通道缩放向量对 Key、Value 激活值及 FFN 中间表征进行微调，参数量比 LoRA 更加轻量。
- **Prefix Tuning / P-Tuning**：在输入层或多层注意力层前拼接连续可学习虚拟 Token 表征，引导模型自适应不同下游任务。

## 训练基础设施 (Infra)

- **DP (Data Parallelism)**：各计算节点持有模型全量参数副本并独立分发数据微批次，通过梯度全约简（AllReduce）保持参数同步更新。
- **TP (Tensor Parallelism)**：将单层算子的权重矩阵按行或按列切分至多张 GPU 并行计算，依赖机内极高带宽互联。
- **PP (Pipeline Parallelism)**：将网络深度方向的若干层流水线切分至不同节点，借助微批次流水调度隐藏流水线空泡（Bubble）。
- **3D 并行**：融合数据并行、张量并行与流水线并行的多维混合策略，为超大规模万卡集群预训练的标准范式。
- **ZeRO (Zero Redundancy Optimizer)**：DeepSpeed 提出的显存冗余消除技术，通过在数据并行维度渐进式切分优化器状态、梯度与模型参数以释放显存。
- **FSDP (Fully Sharded Data Parallel)**：PyTorch 原生实现的完全分片数据并行机制，对应 ZeRO-3 级别的参数分片与通信编排。
- **FlashAttention**：利用 GPU 内存层次结构特性，通过分块重计算与在线 Softmax 避免将注意力矩阵物化写入显存，实现近线性显存占用与数倍吞吐加速。
- **Gradient Checkpointing**：在前向传播阶段丢弃大部分中间激活值，反向传播时按需局部重计算，以计算开销换取显存容量的大幅节省。
- **Mixed Precision**：采用 BF16 或 FP16 执行前向与反向张量乘法，并在优化器更新阶段保留 FP32 高精度主权重的混合训练机制。
- **FP8 训练**：利用前沿芯片张量核心原生支持的 8 位浮点格式（E4M3/E5M2）进行高吞吐量矩阵运算与显存通信。
- **NCCL**：NVIDIA 针对多 GPU 架构深度优化的集合通信库，构成分布式训练中 AllReduce、AllGather 等通信原语的底层标准。
- **NVLink / InfiniBand**：分别构筑节点内极高带宽跨卡互联与跨节点低延迟无阻塞网络互联的硬件底座。

## 推理 (Inference)

- **KV Cache**：自回归生成阶段缓存历史 Token 计算所得的 Key 与 Value 状态张量，避免重复计算并实现 $O(1)$ 单步增量复杂度。
- **PagedAttention**：借鉴操作系统虚拟内存分页思想，将 KV Cache 离散存储于非连续显存块中，从根本上解决显存碎片与预留浪费。
- **Continuous Batching**：在请求级别实施细粒度迭代调度，允许新请求在解码间隙即时插入并与完成请求解耦，极大提升服务并发吞吐。
- **Speculative Decoding (投机采样)**：利用小型草稿模型快速生成候选 Token 序列，主模型单次前向并行验证接受，显著降低单 Token 延迟。
- **Medusa**：在主模型顶层挂载多个并行轻量预测头生成未来 Token 候选，无需引入独立草稿模型即可实现投机加速。
- **Prefix Caching**：针对共有系统提示词或长文档前缀的 KV Cache 进行全局跨请求缓存与复用，显著削减预填充阶段的延迟与算力消耗。
- **量化 (Quantization)**：将连续浮点权重与激活值映射至低比特离散整数域（如 INT8/INT4/FP8），在极小精度退化下大幅降低显存开销与带宽负载。
- **GPTQ / AWQ**：主流离线后训练量化（PTQ）方法；AWQ 通过保护突出的显著权重通道，在低比特截断下表现出更稳健的表征保真度。

## 评测 (Evaluation)

- **Perplexity (PPL)**：测试语料在模型因果预测下的指数化交叉熵损失，直接反映模型对序列概率分布的建模与压缩能力。
- **MMLU**：跨越人文、社科、理工等 57 个学科维度的多选题基准，用于评估基座模型的通用世界知识与推理厚度。
- **GSM8K**：包含高质量小学至初中数学应用题的评测集，核心考察模型的多步逻辑推演与符号计算能力。
- **HumanEval / MBPP**：基于函数级单元测试通过率（Pass@k）的代码生成与算法实现能力评估基准。
- **MT-Bench**：以高阶大模型为裁判（LLM-as-a-Judge）对多轮开放式人机交互对话质量进行系统性评估的基准。
- **Chatbot Arena**：基于人类匿名双盲对战与 Elo 评分体系的大模型真实交互能力客观评估平台。
- **HELM**：斯坦福大学提出的多维度、全方位大模型综合评估框架。

## 数据 (Data)

- **CommonCrawl / C4 / The Pile / FineWeb**：支撑现代大模型预训练的典型大规模公开网络抓取与合成清洗语料库。
- **Deduplication (去重)**：基于 MinHash、LSH 或后缀数组等算法在文档级与句子级消除重复文本，提升训练泛化效率并防止记忆过拟合。
- **Quality Filter**：综合运用轻量级分类器、语言学规则与困惑度阈值滤除低质文本、机器乱码与垃圾信息。
- **Mixture / Curriculum**：预训练阶段多源语料的配比混合策略与随训练推进动态调整的渐进式数据课程设计。

## RAG / 检索 (Retrieval-Augmented Generation)

- **Embedding**：利用编码器将离散文本映射至高维连续语义几何流形空间的稠密向量表示。
- **Bi-encoder（双塔架构）**：查询项与文档项分别独立编码生成表征向量，推理期通过近似最近邻（ANN）点积快速计算相似度。
- **Cross-encoder（交叉编码器）**：将查询项与文档项拼接后统一输入模型进行全注意力交互，精度极高但计算复杂度不适于大规模候选召回。
- **Reranker (重排器)**：在两阶段检索架构中，利用交叉编码器对双塔召回的候选文档集进行高精度二次精细排序。
- **BM25**：基于词频、逆文档频率与文档长度归一化的经典词法稀疏检索算法，构筑混合检索的重要基线。
- **ColBERT / Late Interaction**：保留 Token 级别的细粒度多向量嵌入，在交互层计算最大相似度求和，兼具双塔的速度与交叉编码器的精度。
- **Hybrid Search**：融合 BM25 词法稀疏匹配与稠密向量语义检索的加权混合召回策略。

## 多模态 (Multimodality)

- **VLM (Vision-Language Model)**：具备跨视觉表征与自然语言联合感知、对齐与生成能力的统一多模态模型。
- **ViT (Vision Transformer)**：将图像网格解构为离散 Patch 序列并进行线性投影，直接复用 Transformer 骨干进行全局视觉表征建模。
- **CLIP**：基于大规模图文对比学习（Contrastive Learning）将视觉与文本模态映射至统一语义度量空间的基石模型。
- **VQA (Visual Question Answering)**：结合输入图像视觉信息与自然语言提问进行跨模态推理并生成精准回答的任务。
- **OCR-free**：模型凭借强大的视觉编码能力直接感知并解析图像中的排版与文本内容，摆脱独立的 OCR 预处理管线。

## 解码采样技巧 (Decoding Strategies)

- **Temperature**：控制 Softmax 概率分布平滑程度的温度超参数；数值越高生成越富有多样性，趋近于 0 则退化为贪心搜索（Greedy Search）。
- **Top-k / Top-p (Nucleus Sampling)**：截断低概率尾部候选词的采样策略；Top-k 保留最高概率的固定数量候选，Top-p 动态截取累积概率阈值内的候选子集。
- **Repetition Penalty**：对近期已生成的 Token 施加对数几率惩罚，有效抑制模型陷入局部自重复与死循环。
- **Beam Search**：在解码搜索树上动态维护固定宽度（Beam Width）最高累积概率路径的启发式搜索策略，常用于机器翻译等确定性生成场景。
- **CoT (Chain-of-Thought)**：通过显式生成中间思维推演链路，诱导模型逐步拆解复杂逻辑与推理步骤。
- **Few-shot**：在上下文提示词中提供少量示范样本进行情境自适应学习（In-Context Learning）。
- **System Prompt**：置于对话交互起始位置的全局指导性提示词，用于锚定模型的角色扮演、行为准则与输出约束。

---

[← 目录](../README.md) | [下一章 →](01-tokenizer.md)
