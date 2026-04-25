[← Previous Chapter](08-evaluation.md) | [Table of Contents](README.md) | [Next Chapter →](10-sota-models.md)

---

# Chapter 9: Multimodal

## 9.1 Vision-Language Models (VLMs)

### 9.1.1 Architecture Patterns

```
Approach 1: Cross-Attention (Flamingo/Claude 3 style)
  Image → Vision Encoder → Visual Tokens
  Text + Visual Tokens → Cross-Attention Layers → Text Output
  
  Pros: More flexible vision-text interaction
  Cons: Requires modifying the LLM architecture

Approach 2: Early Fusion (LLaVA style) ← Current mainstream
  Image → Vision Encoder → MLP Projector → Visual Tokens
  [Visual Tokens] + [Text Tokens] → Standard LLM → Output
  
  Pros: No LLM architecture changes, simple
  Cons: Visual tokens consume context window

Approach 3: Native Multimodal (Gemini, GPT-4o style)
  Multimodal from pretraining
  Images and text share the same token space
  
  Pros: Strongest multimodal capability
  Cons: Extremely high training cost
```

### 9.1.2 Vision Encoder

**Mainstream choice**: [CLIP](https://arxiv.org/abs/2103.00020) ViT (Vision Transformer)

```
Image (224×224) → Patch Embedding (14×14 patches = 256 tokens)
→ ViT Encoder (12-24 layers)
→ 256 visual tokens (each d-dimensional)

High-resolution handling:
  - Dynamic resolution: split image into tiles (e.g., 2×2)
  - Each tile encoded independently, then merged
  - LLaVA-Next: max 672×672 → 4 tiles × 256 tokens = 1024 tokens

Better encoders:
  - SigLIP: Google's improved CLIP (sigmoid loss)
  - InternViT: 6B-parameter large ViT
  - DINOv2: Self-supervised ViT, better spatial understanding
```

**Visual Token Compression**:
```
Problem: High-res image → many visual tokens → eats context
     1024×1024 image → 4096 tokens → one image takes half the context

Solutions:
  - Perceiver Resampler: Compress with fixed number of learnable queries (e.g., 256 → 64)
  - Average Pooling: Spatial downsampling
  - C-Abstractor: CNN-based spatial downsampling
  - Token Merging: Merge similar visual tokens
```

### 9.1.3 Training Pipeline

```
Stage 1: Vision-Language Alignment (Pretraining)
  - Freeze Vision Encoder + LLM
  - Train only the Projector (MLP)
  - Data: Image-text pairs (caption data, e.g., LAION, CC3M)
  - Goal: Align visual and language spaces
  - Scale: Millions to tens of millions of pairs

Stage 2: Visual Instruction Tuning (Fine-tuning)
  - Unfreeze LLM (optionally: unfreeze Vision Encoder)
  - Train Projector + LLM
  - Data: Visual instruction data (VQA, OCR, chart understanding, etc.)
  - Scale: Hundreds of thousands to millions of examples

Stage 3: Preference Optimization (Optional)
  - Apply DPO/RLHF to visual QA
  - Reduce hallucinations (prevent the model from "making up" content not in the image)
```

### 9.1.4 SOTA VLMs

| Model | Parameters | Vision Encoder | Highlights |
|-------|-----------|---------------|------------|
| GPT-4o | ? | Native multimodal | Strongest overall performance |
| Claude 3.5 Sonnet | ? | Cross-attention | Strong document understanding |
| Gemini 1.5 Pro | ? | Native multimodal | Ultra-long context (1M+) |
| [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) | 7B/72B | SigLIP | Open-source SOTA |
| [InternVL 2.5](https://arxiv.org/abs/2412.05819) | 78B | InternViT-6B | Open-source SOTA |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | 72B | ViT-600M | Strong OCR/document |

## 9.2 Image Generation

### 9.2.1 Diffusion Models

```
Forward process: x_0 → x_1 → ... → x_T (progressively add noise)
Reverse process: x_T → x_{T-1} → ... → x_0 (progressively denoise; model learns this)

Training objective: 
  L = E[||ε - ε_θ(x_t, t, c)||²]
  # ε = noise added
  # ε_θ = model-predicted noise
  # c = condition (text embedding)
```

**Latent Diffusion** ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752)):
```
Image → VAE Encoder → Latent (64×64) → Diffusion → VAE Decoder → Image
                                          ↑
                                    Text Encoder (CLIP)
```

Performing diffusion in latent space is far more computationally efficient.

### 9.2.2 Text-to-Image

| Model | Architecture | Highlights |
|-------|-------------|------------|
| [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) | DiT (Diffusion Transformer) | Open-source, MMDiT |
| DALL-E 3 | Diffusion + Caption rewriting | Strong prompt adherence |
| Midjourney v6 | ? | Best aesthetics |
| [FLUX](https://github.com/black-forest-labs/flux) | Rectified Flow Transformer | New open-source SOTA |
| Imagen 3 | Cascaded Diffusion | Google |

**DiT** ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)): Replaces U-Net with Transformer as the diffusion backbone. FLUX and SD3 are both built on DiT.

### 9.2.3 Autoregressive Image Generation

**Emerging trend**: Generating images autoregressively, the same way as LLMs.

```
Approach 1: Visual Tokenizer (VQVAE/VQGAN)
  Image → Discrete tokens → Autoregressive LLM generation
  Examples: DALL-E (original), Parti, Chameleon

Approach 2: Continuous Autoregressive
  Image → Continuous patches → AR with diffusion head
  Examples: MAR, Transfusion
```

**[Transfusion](https://arxiv.org/abs/2408.11039) (Meta)**: Unifies text (AR) and image (diffusion) within a single model.

## 9.3 Audio & Speech

### 9.3.1 Speech-to-Text (ASR)

**[Whisper](https://arxiv.org/abs/2212.04356) (OpenAI)**:
```
Audio → Mel Spectrogram → Audio Encoder (Transformer)
→ Cross-Attention with Text Decoder → Transcription

Training: 680K hours of multilingual labeled audio
Features: 98 languages, extremely robust
```

> Code: [openai/whisper](https://github.com/openai/whisper)

### 9.3.2 Text-to-Speech (TTS)

| Model | Method | Highlights |
|-------|--------|------------|
| [VALL-E](https://arxiv.org/abs/2301.02111) | AR codec tokens | 3-second voice cloning |
| [Bark](https://github.com/suno-ai/bark) | AR + Diffusion | Multilingual, open-source |
| [F5-TTS](https://arxiv.org/abs/2410.06885) | Flow matching | Fast, high-quality |
| GPT-4o | End-to-end multimodal | Most natural |

### 9.3.3 Audio Understanding

**Unified Audio-Language Models**:
```
Audio → Audio Encoder (Whisper encoder or HuBERT)
→ Projector → LLM (processed together with text)

Capabilities: Speech understanding, music analysis, environmental sound recognition
Examples: SALMONN, Qwen-Audio, Gemini
```

## 9.4 Video

### 9.4.1 Video Understanding

```
Video = Multiple frames + temporal information

Approach 1: Uniformly sample frames → Encode each frame independently → Feed all frame tokens into LLM
  Simple but token count explodes (16 frames × 256 tokens = 4096)

Approach 2: Spatiotemporal encoding
  3D patch embedding (spatial + temporal)
  Video ViT or TimeSformer

Approach 3: Keyframes + motion information
  Encode keyframes + optical flow/motion tokens

Notable models: Gemini 1.5 (1M tokens, 1-hour video), GPT-4o, LLaVA-Video
```

### 9.4.2 Video Generation

```
Text/Image → Spatial-Temporal DiT → Video

Key challenges:
  - Temporal consistency (cross-frame coherence)
  - Motion plausibility (physical realism)
  - Enormous compute (one extra dimension compared to images)

SOTA:
  - Sora (OpenAI): Spatial-temporal patches + DiT, up to 1 minute
  - Veo 2 (Google): 4K, over 1 minute
  - Kling (Kuaishou): Chinese SOTA
  - HunyuanVideo (Tencent): Open-source SOTA
  - Wan (Alibaba): Open-source SOTA
```

## 9.5 Omni Models

**Ultimate goal**: A single model that handles all modalities (text, image, audio, video) for both input and output.

```
GPT-4o: Native multimodal — text/image/audio input and output
Gemini 2: Multimodal input/output + tool use
Any modality → Unified Token Space → Autoregressive Generation → Any modality
```

**Technical approach**:
1. **Tokenize everything**: Convert all modalities to tokens (text BPE, image VQVAE, audio codec)
2. **Mixed training**: Next-token prediction on unified token sequences
3. **Modality-specific heads**: Different decoding heads for different modalities

---

[← Previous Chapter](08-evaluation.md) | [Table of Contents](README.md) | [Next Chapter →](10-sota-models.md)
