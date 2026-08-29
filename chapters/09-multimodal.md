[← 上一章](08-embedding-rag.md) | [目录](../README.md) | [下一章 →](10-safety-alignment.md)

---

# 第九章：多模态大模型 (Multimodal LLMs)

人类对世界的感知天然交织于视觉光影、声音韵律与离散符号之间。从纯文本大模型迈向多模态大模型（Multimodal Foundation Models），不仅是输入维度的拓宽，更是从符号逻辑闭环跃迁至物理世界具身感知的必由之路。本章将系统解析视觉语言模型（VLM）、扩散与自回归图像生成、时空视频建模以及端到端全模态（Omni）系统的底层架构演进与工程实践。

## 9.1 视觉语言模型 (Vision-Language Models, VLMs)

### 9.1.1 跨模态架构模式

如何将高维连续的像素张量与离散的词元空间有机缝合，是 VLM 架构设计的核心命题。业界演化出三种主流范式：

```
范式一: 交叉注意力融合 (Cross-Attention Fusion, 如 Flamingo / Claude 3)
  图像像素 → 视觉编码器 → 视觉 Token 序列
  文本 Token + 视觉 Token → 门控交叉注意力层 (Gated Cross-Attention) → 语言模型输出
  
  核心优势: 视觉特征与语言计算显式解耦，利于保留高分辨率细粒度空间拓扑。
  工程代价: 需侵入式修改预训练 LLM 内部网络层，推理引擎需定制优化。

范式二: 早期前置投影融合 (Early Fusion / Prefix Projection, 如 LLaVA / BLIP-2)
  图像像素 → 视觉编码器 (ViT) → 投影对齐层 (MLP / Resampler) → 视觉 Token 嵌入
  [视觉 Token 序列] 与 [文本 Token 序列] 在输入端无缝拼接 → 送入标准 Transformer 解码器
  
  核心优势: 完整复用现有成熟的开源 LLM 架构与生态基础设施。
  主要瓶颈: 高分辨率图像展开后占用大量上下文窗口，计算复杂度随图像分辨率呈二次方攀升。

范式三: 原生统一全模态 (Native Unified Multimodal, 如 Gemini 1.5 / GPT-4o)
  自预训练之初即在统一 Token 空间中对文本、像素、音频进行混合序列建模。
  
  核心优势: 跨模态表征对齐极其深刻，具备原生的多模态生成与长程上下文感知能力。
  工程代价: 算力开销与多模态数据均衡配比极其严苛，训练稳定性控制极难。
```

### 9.1.2 视觉编码器与特征压缩

**主流视觉编码器架构演进**:
- **CLIP ViT (Vision Transformer)**: 基于数亿图文对进行对比学习预训练，语义抽象度极高，但对局部细微空间坐标和密集文字辨识相对薄弱。
- **SigLIP (Sigmoid Loss for Language Image Pre-training)**: 采用逐对独立 Sigmoid 损失替代全局 Softmax，训练稳定性与下游零样本分类能力显著提升。
- **InternViT / DINOv2**: 具备数亿至数十亿参数的巨型视觉基座，利用自监督像素级掩码重建提取丰富的几何与纹理细节。

```
图像 (224×224 像素) → 划分为 14×14 图像块 (Patch) → 得到 256 个局部 Patch 嵌入
→ 穿过 12 至 24 层 ViT 编码器 → 输出 256 个 d 维连续视觉向量
```

**动态高分辨率切片 (Dynamic Resolution / AnyRes)**:
为避免高分辨率图像强行双线性插值导致的模糊，现代模型（如 LLaVA-NeXT、Qwen2-VL）采用动态网格切分机制：

```
输入任意长宽比高分辨率图像 (如 1024×1024):
1. 保持原图比例切分为 2×2 个标准局部切片 (Tiles) + 1 张全局下采样缩略图。
2. 每个 Tile 独立经过 ViT 提取特征，生成 4 × 256 = 1024 个局部细粒度 Token。
3. 结合全局概览 Token 进行拼接，保留微观文字与宏观构图信息。
```

**视觉 Token 压缩与降采样**:
- **Perceiver Resampler**: 设定固定数量（如 64 个）的可学习查询向量（Latent Queries），通过交叉注意力机制从数千个视觉 Token 中提炼关键特征。
- **2D 空间平均池化 (Spatial Pooling)**: 将相邻 $2 \times 2$ 的视觉 Token 拼接降采样为 1 个 Token，压缩率达 75%。
- **Token 动态合并 (Token Merging)**: 在自注意力层根据余弦相似度动态合并冗余的背景 Token。

### 9.1.3 三阶段多模态训练流水线

```
第一阶段: 跨模态特征对齐预训练 (Pre-alignment)
  - 参数策略: 严格冻结 Vision Encoder 与 LLM 基座，仅微调投影层 (Projector MLP)
  - 训练数据: 海量弱监督图文对 (如 LAION、CC3M、DataComp 等数千万样本)
  - 核心目标: 将视觉编码器的潜在空间对齐映射至 LLM 文本词嵌入空间

第二阶段: 视觉指令微调 (Visual Instruction Tuning)
  - 参数策略: 解冻 LLM 基座，联合微调 Projector 与 LLM (可选解冻 Vision Encoder)
  - 训练数据: 复杂视觉问答 (VQA)、文档 OCR 解析、科学图表理解、空间定位 (Grounding)
  - 核心目标: 激发模型理解图像细微细节并遵循复杂推理指令的能力

第三阶段: 多模态偏好对齐与防幻觉强化 (Multimodal RLHF / DPO)
  - 核心目标: 针对多模态特有的“对象幻觉”（Object Hallucination，凭空捏造图像中不存在的物体）进行惩罚对齐
```

### 9.1.4 旗舰视觉语言模型一览

| 模型 | 激活规模 | 视觉编码器基座 | 分辨率处理机制 | 核心技术亮点 |
|------|---------|--------------|--------------|-------------|
| **GPT-4o** | 未公开 | 原生多模态统一基座 | 动态切片投影 | 毫秒级流式响应，综合感知能力顶尖 |
| **Claude 3.5 Sonnet** | 未公开 | 混合注意力架构 | 细粒度网格采样 | 卓越的图表分析、代码界面还原与文档理解 |
| **Gemini 1.5 Pro** | MoE 架构 | 原生多模态统一基座 | 原生多模态序列 | 支持 1 小时以上长视频与百万级上下文 |
| [**LLaVA-OneVision**](https://arxiv.org/abs/2408.03326) | 7B / 72B | SigLIP-SO400M | AnyRes 动态切分 | 开源多模态基线，兼顾单图、多图与视频理解 |
| [**InternVL 2.5**](https://arxiv.org/abs/2412.05819) | 1B 至 78B | InternViT-6B | 动态高分辨率切片 | 工业级开源视觉 SOTA，中文 OCR 与图表极强 |
| [**Qwen2-VL**](https://arxiv.org/abs/2409.12191) | 2B / 7B / 72B | 动态分辨率 ViT | 原生动态长宽比 (Naive Dynamic) | 任意分辨率无损输入，支持秒级时空视频定位 |

## 9.2 图像生成与统一多模态生成

### 9.2.1 扩散模型与潜空间流形 (Diffusion & Flow Matching)

图像生成领域历经了从生成对抗网络（GAN）向扩散模型（Diffusion Models）与流匹配（Flow Matching）的深刻演进。

```
前向加噪过程 (物理扩散): 从真实数据分布 x_0 逐步注入高斯白噪声，经过 T 步退化为纯噪声 x_T
反向去噪过程 (逆向积分): 训练神经网络 ε_θ(x_t, t, c) 预测所注入的噪声向量，沿得分函数反向求解

优化目标 (Score Matching): 
  L_diffusion = E_{t, x_0, ε} [ || ε - ε_θ(x_t, t, c) ||² ]
  (其中 c 为文本提示词的 CLIP / T5 语义编码条件)
```

**潜在扩散模型 (Latent Diffusion Model, LDM)**:
在像素空间直接执行数十步迭代去噪的计算复杂度极高。LDM 先利用变分自编码器（VAE）将 $512 \times 512 \times 3$ 的高维像素无损压缩至 $64 \times 64 \times 4$ 的连续潜在流形空间，并在紧凑潜在空间上进行扩散运算，最终由 VAE 解码器重建高清图像。

```
文本 Prompt ──▶ Text Encoder (T5 / CLIP) ──────────────────────┐
                                                                ▼
高斯噪声 ──────▶ 潜在空间扩散去噪网络 (U-Net / DiT) ────▶ 潜在特征图 ────▶ VAE Decoder ──▶ 高清图像
```

### 9.2.2 扩散 Transformer (DiT) 与流匹配架构

传统扩散模型普遍基于卷积 U-Net 结构，参数扩增时容易遭遇容量瓶颈。**DiT (Diffusion Transformer)** ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) 将潜在空间特征切分为 Patch 序列，以标准 Transformer 作为去噪主体，使得图像生成领域完整继承了 Scaling Law 的算力红利。

| 模型 | 骨干架构 | 核心数学机制 | 典型生成特性 |
|------|---------|-------------|-------------|
| [**FLUX.1**](https://github.com/black-forest-labs/flux) | MMDiT (多模态 DiT) | Rectified Flow (直角流匹配) | 极致逼真的人体质感与复杂排版文字生成 |
| [**Stable Diffusion 3**](https://arxiv.org/abs/2403.03206) | 双流 MMDiT | Flow Matching ODE | 文本与视觉在独立权重流中交叉注意力交互 |
| **Midjourney v6** | 专有 DiT 体系 | 深度美学对齐 | 艺术表现力与商业构图质感顶尖 |
| **DALL-E 3** | U-Net 扩散模型 | 深度 Prompt 自动化重写 | 严谨遵从极其复杂的长文本提示词 |

### 9.2.3 自回归图像生成与统一表征范式

大模型领域的另一条探索路径是将图像彻底离散化，使用与语言模型完全一致的自回归（Autoregressive）方式逐词生成图像：
1. **离散分词化 (Discrete Tokenization)**: 利用 VQ-GAN 或 FSQ（有限标量量化）将图像切块离散映射为码本整数序列（如 $32 \times 32 = 1024$ 个离散 Token）。
2. **统一自回归预测**: 将文本与图像 Token 混编为长序列，用标准因果掩码交叉熵损失训练单个 Transformer。
3. **统一混合架构 (Transfusion)** ([Meta, 2024](https://arxiv.org/abs/2408.11039)): 在同一个 Transformer 内部，对文本 Token 采用自回归交叉熵损失，对连续图像 Patch 采用扩散流匹配损失，真正实现了离散与连续模态的端到端融合。

## 9.3 音频与语音建模 (Audio & Speech)

### 9.3.1 语音识别与声学表征 (ASR)

**[Whisper](https://arxiv.org/abs/2212.04356) 架构体系**:
将连续声学波形转换为对数梅尔频谱图（Log-Mel Spectrogram），通过一维卷积与 Transformer 编码器提取声学特征，再由自回归解码器在特殊控制 Token（如 `<|transcribe|>`, `<|en|>`）引导下输出多语言转录文本。

```
原始音频波形 ──▶ 80通道 Log-Mel 频谱 ──▶ Conv1D 降采样 ──▶ Transformer Encoder ──▶ 交叉注意力解码 ──▶ 文字序列
```

### 9.3.2 神经音频编解码与语音合成 (TTS)

现代高质量零样本语音克隆依赖于神经音频编解码器（Neural Audio Codec，如 EnCodec、SoundStream、DAC）：
- **残差向量量化 (RVQ)**: 将 24kHz 高保真音频波形压缩为每秒数百个多层离散 Codec Token。
- **自回归生成 (如 VALL-E)**: 将文本与目标说话人的 3 秒参考提示音频拼接，自回归预测第一层声学 Token，后续层并行生成。
- **流匹配生成 (如 F5-TTS)**: 基于非自回归 Flow Matching 进行音频梅尔谱去噪重建，生成速度极快且极富自然语调起伏。

## 9.4 时空视频建模 (Video Intelligence)

### 9.4.1 视频理解与长程时序建模

视频由多帧图像序列构成，兼具二维空间几何与一维时间运动维度。
- **时空解耦注意力 (Factorized Spatial-Temporal Attention)**: 在空间层内执行单帧内注意力，在时间层内沿同一空间位置的时序序列执行时间注意力，计算复杂度大幅低于全时空 3D 注意力。
- **动态抽帧与长上下文 (Gemini / Qwen2-VL)**: 针对长视频按 1-2 FPS 动态采样，结合 3D Patch 编码与超长上下文机制（1M+ tokens），实现小时级长视频的精确定位与情节理解。

### 9.4.2 视频生成前沿与物理世界模拟

视频生成是物理世界动力学模拟的初级形态。模型必须在保持单帧高保真图像质量的同时，严格保证帧间运动的一致性与因果连续性。

```
文本/参考图条件 ──▶ 3D 潜在空间时空 DiT (Spatial-Temporal DiT) ──▶ 3D VAE 解码器 ──▶ 连贯视频帧流
```

**代表性工业 SOTA 系统**:
- **Sora (OpenAI)**: 采用时空切片（Spacetime Patches）将任意分辨率与时长的视频统一表示为序列，以巨型 DiT 实现分钟级高连贯性生成。
- **HunyuanVideo / Wan (腾讯 / 阿里)**: 开源双流/单流混合 DiT 视频生成架构，具备极强的时间连贯性与精细物理规律还原度。
- **Kling (快手)**: 在复杂大动作生成与高分辨率长镜头运镜方面表现卓越。

## 9.5 全模态统一系统 (Omni Models)

大模型发展的终极形态是建立能够全双工、低延迟处理任意模态输入与输出的统一智能体（Omni Foundation Models）。

```
任意输入流 (文字 / 图像视频流 / 麦克风实时音频)
                    │
                    ▼
     统一离散/连续多模态 Token 抽象表示层
                    │
                    ▼
      全模态因果注意力基座 (Omni-Transformer)
                    │
                    ▼
 专用解码头 (文本 Logits / 神经音频解码流 / 视觉生成潜在向量)
                    │
                    ▼
任意输出流 (流式文字 / 实时自然对话语音 / 动态生成图像视频)
```

**核心工程挑战**:
1. **统一分词与表征鸿沟**: 如何在离散文本符号与高熵连续物理信号之间建立统一且高效的信息瓶颈。
2. **实时流式全双工交互**: 实现端到端毫秒级中断响应（Voice Interruption），摒弃传统的“ASR → LLM → TTS”级联架构。
3. **跨模态算力均衡**: 避免高开销视觉与音频数据冲垮轻量化文本推理的低延迟边界。

## 关键论文

- [Radford et al. (2021): Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)：对比学习开创图文多模态特征对齐新纪元。
- [Liu et al. (2023): Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)：奠定现代开源视觉指令微调的标准两阶段范式。
- [Peebles & Xie (2023): Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)：以 Transformer 替代 U-Net，引领现代文生图与文生视频架构变革。
- [Radford et al. (2022): Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356)：弱监督大规模语音表征与通用识别标杆。
- [Alayrac et al. (2022): Flamingo: A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)：基于门控交叉注意力的少样本视觉交互范式。

## 进阶参考

- [LLaVA 官方开源项目](https://github.com/haotian-liu/LLaVA)：包含完整的端到端视觉指令微调代码与数据处理管线。
- [Qwen2-VL 技术报告与源码](https://github.com/QwenLM/Qwen2-VL)：动态分辨率与时空长视频理解的工程参考实现。
- [Black Forest Labs FLUX.1](https://github.com/black-forest-labs/flux)：现代流匹配 DiT 图像生成的工业级实现。

## 练习题

1. **VLM 幻觉检测实战**: 使用 Qwen2.5-VL-7B 对一组包含细微文字、反常物理场景与遮挡物体的复杂图像进行描述，设计 Prompt 探测并统计模型的对象幻觉发生率。
2. **视觉编码器特征对比**: 在同一组图像分类与检索任务上，分别提取 CLIP-ViT-L/14 与 SigLIP-SO400M 的输出向量，计算不同类目在潜在空间中的类内聚类度与类间分离度。
3. **跨模态以文搜图**: 基于 sentence-transformers 与预训练 CLIP 模型，构建包含 10,000 张图片的本地图文向量索引，实现自然语言语义检索并对比单模态与多模态表征差异。

---

[← 上一章](08-embedding-rag.md) | [目录](../README.md) | [下一章 →](10-safety-alignment.md)
