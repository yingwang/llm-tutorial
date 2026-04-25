[← 上一章](08-evaluation.md) | [目录](../README.md) | [下一章 →](10-sota-models.md)

---

# 第九章：多模态 (Multimodal)

## 9.1 Vision-Language Models (VLMs)

### 9.1.1 架构模式

```
方式一: Cross-Attention (Flamingo/Claude 3 风格)
  Image → Vision Encoder → Visual Tokens
  Text + Visual Tokens → Cross-Attention 层 → Text 输出
  
  优势: 视觉和文本交互更灵活
  劣势: 需要修改 LLM 架构

方式二: Early Fusion (LLaVA 风格) ← 当前主流
  Image → Vision Encoder → MLP Projector → 视觉 Token
  [视觉 Token] + [文本 Token] → 标准 LLM → 输出
  
  优势: 不改 LLM 架构，简单
  劣势: 视觉 token 占 context 窗口

方式三: Native Multimodal (Gemini, GPT-4o 风格)
  从预训练开始就是多模态的
  图像和文本在同一个 token 空间
  
  优势: 最强的多模态能力
  劣势: 训练成本极高
```

### 9.1.2 Vision Encoder

**主流选择**: [CLIP](https://arxiv.org/abs/2103.00020) ViT (Vision Transformer)

```
Image (224×224) → Patch Embedding (14×14 patches = 256 tokens)
→ ViT Encoder (12-24 layers)
→ 256 个 visual tokens (每个 d 维)

高分辨率处理:
  - 动态分辨率: 把图像分成 tiles (如 2×2)
  - 每个 tile 独立编码，合并
  - LLaVA-Next: 最大 672×672 → 4 tiles × 256 tokens = 1024 tokens

更好的 encoder:
  - SigLIP: Google 的改进 CLIP (sigmoid loss)
  - InternViT: 6B 参数的大 ViT
  - DINOv2: 自监督 ViT，更好的空间理解
```

**Visual Token 压缩**:
```
问题: 高分辨率图像 → 大量 visual token → 占 context
     1024×1024 图像 → 4096 tokens → 一张图占了一半 context

方案:
  - Perceiver Resampler: 用固定数量的 learnable queries 压缩 (如 256 → 64)
  - Average Pooling: 空间维度降采样
  - C-Abstractor: CNN 做空间降采样
  - Token Merging: 合并相似的 visual token
```

### 9.1.3 训练流程

```
Stage 1: Vision-Language Alignment (预训练)
  - 冻结 Vision Encoder + LLM
  - 只训练 Projector (MLP)
  - 数据: 图文对 (caption 数据, 如 LAION, CC3M)
  - 目标: 对齐视觉和语言空间
  - 量级: 几百万到几千万 pairs

Stage 2: Visual Instruction Tuning (微调)
  - 解冻 LLM (可选: 解冻 Vision Encoder)
  - 训练 Projector + LLM
  - 数据: 视觉指令数据 (VQA, OCR, chart understanding, etc.)
  - 量级: 几十万到几百万条

Stage 3: Preference Optimization (可选)
  - 对视觉问答做 DPO/RLHF
  - 改善幻觉 (减少模型"编造"图中不存在的内容)
```

### 9.1.4 SOTA VLMs

| 模型 | 参数量 | Vision Encoder | 特点 |
|------|--------|---------------|------|
| GPT-4o | ? | 原生多模态 | 最强综合性能 |
| Claude 3.5 Sonnet | ? | Cross-attention | 强文档理解 |
| Gemini 1.5 Pro | ? | 原生多模态 | 超长上下文 (1M+) |
| [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) | 7B/72B | SigLIP | 开源 SOTA |
| [InternVL 2.5](https://arxiv.org/abs/2412.05819) | 78B | InternViT-6B | 开源 SOTA |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | 72B | ViT-600M | 强 OCR/文档 |

## 9.2 Image Generation

### 9.2.1 Diffusion Models

```
前向过程: x_0 → x_1 → ... → x_T (逐步加噪声)
反向过程: x_T → x_{T-1} → ... → x_0 (逐步去噪声, 模型学这个)

训练目标: 
  L = E[||ε - ε_θ(x_t, t, c)||²]
  # ε = 加入的噪声
  # ε_θ = 模型预测的噪声
  # c = 条件 (文本 embedding)
```

**Latent Diffusion** ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752)):
```
Image → VAE Encoder → Latent (64×64) → Diffusion → VAE Decoder → Image
                                          ↑
                                    Text Encoder (CLIP)
```

在 latent space 做 diffusion，计算量小很多。

### 9.2.2 Text-to-Image

| 模型 | 架构 | 特点 |
|------|------|------|
| [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) | DiT (Diffusion Transformer) | 开源, MMDiT |
| DALL-E 3 | Diffusion + Caption rewriting | 强 prompt 遵循 |
| Midjourney v6 | ? | 最佳美学 |
| [FLUX](https://github.com/black-forest-labs/flux) | Rectified Flow Transformer | 新 SOTA 开源 |
| Imagen 3 | Cascaded Diffusion | Google |

**DiT** ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)): 用 Transformer 替代 U-Net 作为 diffusion 的 backbone。FLUX, SD3 都基于 DiT。

### 9.2.3 Autoregressive Image Generation

**新趋势**: 用和 LLM 一样的自回归方式生成图像。

```
方式一: Visual Tokenizer (VQVAE/VQGAN)
  Image → Discrete tokens → Autoregressive LLM 生成
  代表: DALL-E (original), Parti, Chameleon

方式二: Continuous Autoregressive
  Image → Continuous patches → AR with diffusion head
  代表: MAR, Transfusion
```

**[Transfusion](https://arxiv.org/abs/2408.11039) (Meta)**: 统一文本 (AR) 和图像 (diffusion) 在同一个模型中。

## 9.3 Audio & Speech

### 9.3.1 Speech-to-Text (ASR)

**[Whisper](https://arxiv.org/abs/2212.04356) (OpenAI)**:
```
Audio → Mel Spectrogram → Audio Encoder (Transformer)
→ Cross-Attention with Text Decoder → Transcription

训练: 680K 小时多语言标注音频
特点: 98种语言, 极强鲁棒性
```

> 代码: [openai/whisper](https://github.com/openai/whisper)

### 9.3.2 Text-to-Speech (TTS)

| 模型 | 方法 | 特点 |
|------|------|------|
| [VALL-E](https://arxiv.org/abs/2301.02111) | AR codec tokens | 3秒 voice clone |
| [Bark](https://github.com/suno-ai/bark) | AR + Diffusion | 多语言, 开源 |
| [F5-TTS](https://arxiv.org/abs/2410.06885) | Flow matching | 快速, 高质量 |
| GPT-4o | 端到端多模态 | 最自然 |

### 9.3.3 Audio Understanding

**Unified Audio-Language Models**:
```
Audio → Audio Encoder (Whisper encoder 或 HuBERT)
→ Projector → LLM (和文本一起处理)

能做: 语音理解、音乐分析、环境声音识别
代表: SALMONN, Qwen-Audio, Gemini
```

## 9.4 Video

### 9.4.1 Video Understanding

```
Video = 多帧图像 + 时序信息

方式一: 均匀采样帧 → 每帧独立编码 → 所有帧 token 送入 LLM
  简单但 token 数爆炸 (16帧 × 256 tokens = 4096)

方式二: 时空编码
  3D patch embedding (空间 + 时间)
  Video ViT 或 TimeSformer

方式三: 关键帧 + 运动信息
  选关键帧编码 + 光流/运动 token

代表模型: Gemini 1.5 (100万token, 1小时视频), GPT-4o, LLaVA-Video
```

### 9.4.2 Video Generation

```
Text/Image → Spatial-Temporal DiT → Video

关键挑战:
  - 时间一致性 (帧间连贯)
  - 运动合理性 (物理规律)
  - 计算量巨大 (比图像多一个维度)

SOTA:
  - Sora (OpenAI): 空间-时间 patch + DiT, 最长1分钟
  - Veo 2 (Google): 4K, 超过1分钟
  - Kling (快手): 中国 SOTA
  - HunyuanVideo (腾讯): 开源 SOTA
  - Wan (阿里): 开源 SOTA
```

## 9.5 Omni Models (全模态)

**终极目标**: 一个模型处理所有模态（文本、图像、音频、视频）的输入和输出。

```
GPT-4o: 原生多模态 — 文本/图像/音频 输入输出
Gemini 2: 多模态输入输出 + 工具使用
任意模态 → 统一 Token 空间 → Autoregressive Generation → 任意模态
```

**技术路线**:
1. **Tokenize everything**: 把所有模态转成 token (文本 BPE, 图像 VQVAE, 音频 codec)
2. **Mixed training**: 在统一 token 序列上做 next-token prediction
3. **Modality-specific heads**: 不同模态的解码头

---

[← 上一章](08-evaluation.md) | [目录](../README.md) | [下一章 →](10-sota-models.md)
