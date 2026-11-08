[← 上一章](08-embedding-rag.md) | [目录](../README.md) | [下一章 →](10-safety-alignment.md)

# 第九章：多模态大模型（Multimodal LLMs）

纯文本大模型只能处理离散的文字 token，而图像、音频、视频都是连续的信号。要把这些信号接进大模型，核心问题只有两个：怎么把图像或声音编码成模型能读懂的向量序列，以及怎么用自回归或扩散过程把这些模态生成出来。本章依次介绍视觉语言模型（VLM）、图像生成、语音建模、视频理解与生成，以及端到端全模态（Omni）模型。

## 9.1 视觉语言模型（Vision-Language Models, VLMs）

### 9.1.1 三种架构模式

把视觉和语言拼在一起，主流有三种做法：

```
方式一: Cross-Attention 融合 (Flamingo / Llama 3.2 Vision 风格)
  Image → Vision Encoder → Visual Tokens
  Text Tokens + Visual Tokens → 插入门控交叉注意力层 (Gated Cross-Attention) → 文本输出

  优势: 视觉 token 不直接挤占文本的上下文长度，适合高分辨率多图
  劣势: 需要在 LLM 内部额外插层，改动了标准 Transformer 解码器结构

方式二: Early Fusion 前置拼接 (LLaVA / Qwen-VL / InternVL 风格) ← 开源界最主流
  Image → Vision Encoder (ViT) → Projector (MLP) → 视觉 Token 嵌入
  [视觉 Token] 与 [文本 Token] 直接在输入层拼接 → 送入标准 LLM → 文本输出

  优势: 不改 LLM 内部结构，直接复用现成的预训练底座与推理框架 (vLLM/SGLang)
  劣势: 高分辨率图像会产生几千个视觉 token，大量占用上下文窗口与 KV Cache

方式三: Native Multimodal 原生多模态 (Gemini / GPT-4o 风格)
  从预训练第一天起，就把文本、图像、音频转成统一的 token 序列混合训练

  优势: 跨模态对齐最深，不仅能看图说话，还能原生输出图像和语音
  劣势: 从头预训练成本极高，多模态数据配比与训练稳定性很难调
```

### 9.1.2 视觉编码器（Vision Encoder）与 Token 压缩

开源 VLM 通常拿现成的视觉编码器把图片切成小方格（Patch），编码成一串连续向量：

```
输入图像 (224×224) → 切成 14×14 的小块 (共 16×16 = 256 个 patches)
→ 过 12-24 层 ViT Encoder
→ 得到 256 个 d 维视觉向量 (Visual Tokens)
```

常用的视觉编码器基座：
- **[CLIP](https://arxiv.org/abs/2103.00020) ViT**：用数亿图文对通过对比学习训练出来，擅长抓全局语义，但对密集小字（OCR）和精确位置不太敏感。
- **SigLIP**：Google 对 CLIP 的改进版，把全局 Softmax 换成逐对独立的 Sigmoid 损失，在小 batch 下也训得稳，是目前开源 VLM（如 LLaVA-OneVision、PaliGemma）最常用的视觉底座。
- **InternViT / DINOv2**：前者是参数量高达 6B 的大尺寸视觉底座；后者靠自监督学习，空间几何与局部细节还原更好。

**高分辨率处理（Dynamic Resolution / AnyRes）**：
早期 CLIP 只吃固定的 $224 \times 224$ 或 $336 \times 336$ 方图，遇到长条形网页截图或密密麻麻的 PDF 文档，强行缩放就会模糊不清。现在主流模型采用两种办法：
1. **动态切块（Tile Splitting，如 LLaVA-NeXT、InternVL）**：按原图长宽比切成多个标准小块（比如 $2 \times 2$ 共 4 块）分别过 ViT，再加上一张缩略图提供全局视野。以 336 输入、patch 14 的 ViT 为例，每块产生 $24 \times 24 = 576$ 个 token，一张 $672 \times 672$ 的图切成 4 块再加缩略图，共 $(4+1) \times 576 = 2880$ 个视觉 token。
2. **原生动态分辨率（Naive Dynamic Resolution，如 Qwen2-VL）**：去掉绝对位置编码，改用 2D-RoPE，直接把任意长宽比的图片按原始分辨率切成 patch 送进 ViT。

**视觉 Token 压缩**：
分辨率一高，单张图就会产生两三千个 token，传十张图或者一段视频就会超出显存。常见压缩手段包括：
- **Pixel Shuffle / 空间池化**：把相邻 $2 \times 2$ 的 4 个视觉 token 在通道维拼起来再过一层线性投影压成 1 个，token 数直接减少 75%（InternVL、Qwen2-VL 均采用此法）。
- **Perceiver Resampler / Q-Former**：用固定数量（如 64 个）的可学习查询向量去和视觉特征做交叉注意力，不管原图多大都只抽出 64 个 token。缺点是容易丢掉细碎文字信息。

### 9.1.3 三阶段训练流程

把一个纯文本 LLM 改造成 VLM，通常分三步走：

```
Stage 1: 特征对齐预训练 (Feature Alignment)
  - 冻结 Vision Encoder 和 LLM，只训练中间的 Projector (通常是个两层 MLP)
  - 数据: 几百万图文对 (LAION, CC3M 等图像描述数据)
  - 目的: 让 MLP 学会把 ViT 吐出的视觉向量翻译到 LLM 认识的词向量空间里

Stage 2: 视觉指令微调 (Visual Instruction Tuning)
  - 解冻 LLM + Projector (高分辨率阶段往往连 Vision Encoder 一起解冻)
  - 数据: 几十万到几百万条高质量视觉问答 (VQA)、OCR 文档解析、图表阅读、多轮图文对话
  - 目的: 让模型学会看图回答复杂问题、提取表格与推理

Stage 3: 偏好对齐 (DPO / RLHF，可选)
  - 目的: 重点抑制多模态幻觉 (减少模型顺着文本惯性“脑补”图中根本没有的物体)
```

### 9.1.4 代表性视觉语言模型

| 模型 | 参数规模 | 视觉编码器 | 分辨率方案 | 特点 |
|------|---------|--------------|--------------|-------------|
| **GPT-4o** | 未公开 | 原生多模态 | 动态切块 | 响应快，综合图文与语音交互能力强 |
| **Claude 3.5 Sonnet** | 未公开 | 未公开 | 高分辨率采样 | 读图表、看长文档与前端 UI 代码还原极准 |
| **Gemini 1.5 Pro** | MoE | 原生多模态 | 原生多模态序列 | 1M-2M 上下文，可直接扔进 1 小时视频或整本书 |
| [**LLaVA-OneVision**](https://arxiv.org/abs/2408.03326) | 0.5B / 7B / 72B | SigLIP-400M | AnyRes 动态切块 | 训练配方与数据完全开源，支持单图、多图与视频 |
| [**InternVL 2.5**](https://arxiv.org/abs/2412.05819) | 1B - 78B | InternViT-6B | 动态切块 + Pixel Shuffle | 开源多模态综合榜单前列，中文 OCR 表现突出 |
| [**Qwen2-VL**](https://arxiv.org/abs/2409.12191) | 2B / 7B / 72B | ViT-600M (2D-RoPE) | 原生动态分辨率 | 支持任意长宽比图片输入与长视频理解 |

## 9.2 图像生成

### 9.2.1 扩散模型（Diffusion Models）

扩散模型的思路很直观：前向过程往真实图片里一步步加高斯噪声，直到变成纯雪花噪点；反向过程训练一个神经网络，学会从噪点里一步步把噪声减掉，还原出清晰图像。

```
前向过程 (加噪): x_0 → x_1 → ... → x_T (逐步混入高斯噪声 ε)
反向过程 (去噪): x_T → x_{T-1} → ... → x_0 (模型一步步预测并减去噪声)

训练目标:
  L = E_{t, x_0, ε} [ || ε - ε_θ(x_t, t, c) ||² ]
  # ε: 实际加进去的噪声
  # ε_θ: 模型预测出来的噪声
  # c: 文本提示词的 Embedding 条件 (来自 CLIP 或 T5)
```

**潜在扩散模型（Latent Diffusion，[Rombach et al., 2022](https://arxiv.org/abs/2112.10752)）**：
直接在 $1024 \times 1024$ 的像素图上跑几十步去噪太慢。Latent Diffusion 先用一个训练好的变分自编码器（VAE）把原图长宽各缩小 8 倍，压到紧凑的潜在空间（Latent Space，例如 $128 \times 128 \times 16$），在里面做完扩散去噪后，最后再用 VAE Decoder 还原回像素图，计算量骤降。

```
Text Prompt ──▶ Text Encoder (CLIP / T5) ─────────────┐
                                                      ▼
随机噪声 ──────▶ Latent Diffusion (U-Net / DiT) ──▶ 去噪后的 Latent ──▶ VAE Decoder ──▶ 生成图像
```

### 9.2.2 从 U-Net 到 DiT 与 Flow Matching

早期的 Stable Diffusion 1.5/XL 用卷积 U-Net 做去噪骨干网。**[DiT（Diffusion Transformer）](https://arxiv.org/abs/2212.09748)** 把 Latent 特征图也切成 Patch 序列，改用标准 Transformer 做去噪网络，发现参数量和计算量往上堆时，生成质量像 LLM 一样平滑提升。

新一代生图模型（如 SD3、FLUX）还把传统的扩散过程升级成了 **Rectified Flow（流匹配，Flow Matching）**：不再走弯弯曲曲的随机微分方程路径，而是在噪声分布和真实图像分布之间拉一条直线路径，训练模型预测直线方向的速度场，只需更少的采样步数就能生成高质量图像。

| 模型 | 架构 | 特点 |
|------|------|------|
| [**FLUX.1**](https://github.com/black-forest-labs/flux) | MMDiT + Rectified Flow | 目前开源生图主力，人手结构与图中英文排版文字准确 |
| [**Stable Diffusion 3**](https://arxiv.org/abs/2403.03206) | MMDiT（图文双流注意力） | 开源，文本分支与图像分支各自保留独立权重并做联合注意力 |
| **Midjourney v6** | 未公开 | 审美调教与构图光影质感突出 |
| **DALL-E 3** | Diffusion + LLM 改写 | 先用 GPT-4 把用户的简短提示词扩写成详细画面描述，生图听话度高 |
| **Imagen 3** | Diffusion | Google 出品，写实细节与文字渲染扎实 |

### 9.2.3 自回归图像生成与统一模型

除了扩散模型，另一条路线是照搬 LLM 的做法，用自回归（Next-Token Prediction）方式生成图像：

```
路线一: 离散视觉词表 (VQVAE / VQGAN, 如 Parti, Chameleon, Emu3)
  把图像通过离散编码器量化成一串整数编号 (如 1024 个码本 token)
  和文本 token 混在一起，直接用标准自回归 Transformer 逐个预测

路线二: 自回归 + 扩散混合 (Transfusion / MAR / Show-o)
  同一个 Transformer 里，遇到文本走自回归交叉熵损失，遇到图像走连续 Patch 扩散损失
  兼顾文本的逻辑推理与图像的连续细节质量
```

## 9.3 音频与语音（Audio & Speech）

### 9.3.1 语音识别（ASR）

OpenAI 的 **[Whisper](https://arxiv.org/abs/2212.04356)** 是目前最通用的语音转文字基座：

```
原始音频波形 → 转成 80 通道 Log-Mel 频谱图 → 2 层卷积 + Transformer Encoder
→ Transformer Decoder (通过 Cross-Attention 读取音频特征) → 输出文字与时间戳

训练数据: 68 万小时多语言弱监督音频
特点: 支持 98 种语言转录与翻译，抗背景噪音能力强
```

> 开源仓库：[openai/whisper](https://github.com/openai/whisper)（生产部署常用更快的重实现如 `faster-whisper` 或 `whisper.cpp`）。

### 9.3.2 语音合成（TTS）

现代 TTS 普遍引入了神经音频编解码器（Neural Audio Codec，如 EnCodec、SNAC），先把连续声波压成离散的声学 token，再用自回归或流匹配模型生成：

| 模型 | 方法 | 特点 |
|------|------|------|
| [**VALL-E**](https://arxiv.org/abs/2301.02111) | 自回归预测 Codec Tokens | 给 3 秒参考音频就能克隆说话人音色 |
| [**CosyVoice**](https://github.com/FunAudioLLM/CosyVoice) | LLM + Flow Matching | 阿里开源，中文自然度与音色克隆效果好 |
| [**F5-TTS**](https://arxiv.org/abs/2410.06885) | 非自回归 Flow Matching + DiT | 开源，无需音素对齐，合成速度快 |
| [**Bark**](https://github.com/suno-ai/bark) | 自回归生成语义与声学 token | 开源，支持生成笑声、叹气等副语言音效 |
| **GPT-4o Audio** | 端到端原生语音 token | 延迟极低，支持随时打断与情绪控制 |

### 9.3.3 语音理解大模型（Audio-Language Models）

```
Audio → Audio Encoder (如 Whisper Encoder) → Projector → LLM (与文本一起输入)

用途: 不只是转成文字，还能直接听出说话人的语气情绪、背景环境音和音乐风格
代表模型: Qwen2-Audio, SALMONN, Gemini 1.5
```

## 9.4 视频理解与生成

### 9.4.1 视频理解（Video Understanding）

视频可以看作一连串按时间排列的图像帧加音轨。最大的麻烦是 token 数量膨胀太快：每秒抽 2 帧，每帧 256 个 token，一段 10 分钟的视频就有超过 30 万个视觉 token。

```
做法一: 均匀抽帧 + 空间池化
  按固定帧率抽帧 → 每帧过 ViT → 用池化或双线性插值把每帧压到 64-144 个 token → 拼起来送入长上下文 LLM

做法二: 3D 时空 Patch 编码 (如 Qwen2-VL)
  把相邻两帧的同位置像素打包成一个 3D 立方块 (Tubelet)，在编码阶段直接融掉一半时间步

做法三: 原生百万级上下文硬吃 (如 Gemini 1.5 Pro)
  依靠 Ring Attention 等长序列架构，把整段 1 小时视频的所有帧和音频直接放进 2M 上下文窗口
```

### 9.4.2 视频生成（Video Generation）

主流视频生成模型都转向了 **3D 时空 DiT（Spatial-Temporal DiT）** 架构：

```
文本或首帧图片 → 3D 时空 DiT 在潜在空间去噪 → 3D VAE 解码还原成连续视频帧

核心难点:
  1. 时间一致性: 镜头移动或人物转身时，背景和长相不能闪烁变形
  2. 物理规律: 液体流动、碰撞、重力与遮挡关系要符合常识
  3. 算力消耗: 比单张图片多出一个时间维，3D 注意力的序列长度动辄数万
```

代表性视频生成模型：
- **Sora（OpenAI）**：提出 Spacetime Patches，用 3D VAE 把视频在时间和空间上同时压缩，再用大参数 DiT 训练，支持生成长达 1 分钟的多镜头视频。
- **Veo 2（Google）**：物理运动规律还原度与高分辨率画质表现突出。
- **HunyuanVideo（腾讯） / Wan 2.1（阿里）**：目前开源界效果最好的两个视频生成底座，支持文生视频与图生视频。
- **Kling（快手可灵） / Runway Gen-3**：商用视频生成代表，擅长大幅度动作与镜头控制。

## 9.5 全模态模型（Omni Models）

传统的语音对话助手是三个独立模型串起来的流水线：`语音转文字 (ASR) → 文本大模型 (LLM) → 文字转语音 (TTS)`。这种级联做法有两个硬伤：一是延迟叠加动辄两三秒，二是 ASR 转成纯文字时把用户的笑声、停顿、愤怒语气全丢光了。

**Omni（全模态）模型**（如 GPT-4o、Gemini 2.0、Qwen2.5-Omni）的目标是用同一个神经网络直接吃进文本、图像、视频和音频流，并直接吐出文本或语音波形：

```
任意模态输入 (文本 / 摄像头画面 / 麦克风实时语音)
                    ↓
      各模态 Tokenizer 转为统一序列表示
                    ↓
         统一自回归 Transformer 基座
                    ↓
      文本解码头 / 流式神经语音解码器
                    ↓
      实时返回文字 + 自然语音 (支持随时打断)
```

实现这类端到端模型需要解决三个工程问题：
1. **跨模态对齐而不损害语言能力**：早期把语音和图像直接混进 LLM 训练时，经常发现模型的纯文本逻辑推理能力明显退化，需要精细设计多阶段解耦训练或双流架构（如 Qwen2.5-Omni 的 Thinker-Talker 架构，一个大脑负责思考文本，另一个轻量头负责实时合成语音流）。
2. **流式全双工交互**：模型在说话的同时还要持续监听麦克风输入，一旦用户插话就立刻停下重听。
3. **帧率控制**：把每秒语音压到 12.5Hz 或 25Hz 的极低帧率离散 token，才能让自回归生成跑赢实时播放速度。

## 关键论文

- [Radford et al. (2021): CLIP](https://arxiv.org/abs/2103.00020)：用对比学习把图像和文本拉到同一个向量空间，是现代 VLM 和文生图模型的共同地基。
- [Rombach et al. (2022): Latent Diffusion Models (Stable Diffusion)](https://arxiv.org/abs/2112.10752)：把扩散过程搬到 VAE 潜在空间，大幅降低生图计算开销。
- [Alayrac et al. (2022): Flamingo](https://arxiv.org/abs/2204.14198)：用 Perceiver Resampler 和门控交叉注意力把视觉特征接进冻结的 LLM。
- [Radford et al. (2022): Whisper](https://arxiv.org/abs/2212.04356)：基于 68 万小时多语言弱监督数据训练的通用语音识别模型。
- [Peebles & Xie (2023): DiT (Scalable Diffusion Models with Transformers)](https://arxiv.org/abs/2212.09748)：用 Transformer 替换 U-Net 做扩散模型骨干网。
- [Liu et al. (2023): LLaVA](https://arxiv.org/abs/2304.08485)：确立“ViT + MLP + LLM”两阶段视觉指令微调的极简开源配方。

## 进一步阅读

- [LLaVA 开源仓库](https://github.com/haotian-liu/LLaVA)：代码简洁，适合入门多模态两阶段训练。
- [Cambrian-1](https://arxiv.org/abs/2406.16860)：对各种视觉编码器、连接器与训练配方的系统性消融实验报告。
- [Qwen2-VL 技术报告](https://arxiv.org/abs/2409.12191)：原生动态分辨率与 2D/3D-RoPE 的实现细节。
- [Transfusion](https://arxiv.org/abs/2408.11039)：在同一个 Transformer 里对文本做下一个 token 预测、对图像 patch 做扩散的统一模型。

## 练习题

1. **VLM 幻觉观察**：找一张包含密集小字或反常识细节的图片喂给开源 VLM（如 Qwen2-VL-7B），故意追问图中不存在的物体（例如“图里左下角那只猫是什么颜色的？”），观察模型是否会顺着提问编造答案。
2. **视觉编码器对比**：分别加载 `CLIP-ViT-L/14` 和 `SigLIP-SO400M`，抽取同一批含文字图片和自然风景图的特征，对比它们在图文检索上的表现差异。
3. **以文搜图小系统**：用 CLIP 把本地 1000 张图片编码成向量存入 ChromaDB 或 FAISS，写一个支持用自然语言搜图的命令行小工具。

---

[← 上一章](08-embedding-rag.md) | [目录](../README.md) | [下一章 →](10-safety-alignment.md)
