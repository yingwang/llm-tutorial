[← Previous Chapter](08-embedding-rag.md) | [Table of Contents](README.md) | [Next Chapter →](10-safety-alignment.md)

---

# Chapter 9: Multimodal Models and Omni Systems

Moving beyond text-only paradigms, modern AI systems unify visual perception, speech synthesis, video comprehension, and cross-modal reasoning into cohesive generative architectures.

## 9.1 Vision-Language Models (VLMs)

### 9.1.1 Architectural Integration Patterns

```
Pattern 1: Cross-Attention Injection (Flamingo / DeepMind)
  Image ──> Vision Encoder ──> Visual Feature Tokens
  Text Tokens + Visual Tokens ──> Interleaved Gated Cross-Attention Layers ──> Output
  Characteristics: Preserves frozen LLM weights; adds dedicated cross-attention parameters.

Pattern 2: Linear Projection & Early Fusion (LLaVA / Qwen-VL) [De Facto Standard]
  Image ──> Vision Encoder (e.g. SigLIP) ──> MLP Projector ──> Visual Token Sequence
  Concatenate: [Visual Tokens, Text Tokens] ──> Standard Autoregressive LLM ──> Text
  Characteristics: Simple architecture, treats visual tokens identically to text embeddings.

Pattern 3: Native Unified Pretraining (GPT-4o / Gemini 1.5)
  Joint multi-modal pretraining from initial step 0.
  Discrete audio, image patches, and text tokens share a homogeneous causal attention space.
  Characteristics: Maximal cross-modal synergy at higher pretraining compute cost.
```

### 9.1.2 Vision Encoders and Spatial Patching

**Vision Transformer (ViT)**: Treats non-overlapping 2D image patches (e.g., $14 \times 14$ pixels) as discrete input tokens passed through Transformer blocks ([Radford et al., 2021 (CLIP)](https://arxiv.org/abs/2103.00020)):

$$\text{Image } (H \times W \times C) \longrightarrow N_{\text{patches}} = \frac{H \cdot W}{P^2} \text{ tokens of dimension } D$$

```
High-Resolution Dynamic Tiling (AnyRes / LLaVA-NeXT):
  Standard ViTs downsample high-resolution images, destroying fine-grained text and layout.
  Dynamic Resolution Strategy:
  1. Partition a 1024x1024 image into a grid of 2x2 local sub-tiles (each 336x336).
  2. Append a downsampled global overview tile.
  3. Encode tiles independently through ViT and concatenate visual tokens with newline delimiters.
```

**Advanced Visual Feature Extractors**:
- **SigLIP** ([Zhai et al., 2023](https://arxiv.org/abs/2303.15343)): Replaces standard softmax contrastive loss with pairwise sigmoid loss, improving visual grounding.
- **InternViT-6B** ([Chen et al., 2024](https://arxiv.org/abs/2412.05819)): Scales vision encoder capacity to 6B parameters for fine-grained OCR and document parsing.
- **DINOv2** ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)): Self-supervised visual representation capturing fine geometric and depth cues.

### 9.1.3 Multi-Stage VLM Alignment Pipeline

```
Stage 1: Multi-Modal Feature Alignment (Pretraining)
  - Freeze Vision Backbone and LLM Transformer layers.
  - Train strictly the 2-layer MLP projection adapter.
  - Data: Filtered image-caption pairs (LAION-5B, CC12M).
  - Objective: Map visual embeddings into the language model's latent embedding manifold.

Stage 2: End-to-End Visual Instruction Tuning (SFT)
  - Unfreeze LLM weights (and optionally fine-tune vision encoder with lower learning rate).
  - Data: Multi-turn visual instruction conversations, document OCR, chart reasoning, spatial grounding.
  - Objective: Teach multi-step visual reasoning and formatting adherence.

Stage 3: Multi-Modal Preference Optimization (DPO / RLHF)
  - Align visual responses against human preference pairs.
  - Mitigate visual hallucination (penalizing descriptions of objects not present in the input image).
```

### 9.1.4 Frontier Vision-Language Models

| Model Family | Total Scale | Vision Backbone | Distinguishing Capability |
|--------------|-------------|-----------------|---------------------------|
| GPT-4o | Undisclosed | Native Multimodal | End-to-end multi-modal reasoning and low-latency interaction |
| Claude 3.5 Sonnet | Undisclosed | High-Res Cross-Attention | State-of-the-art document, diagram, and code reasoning |
| Gemini 1.5 Pro | Sparse MoE | Native Multi-Modal | 1M-2M token context for whole-video comprehension |
| [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) | 7B / 72B | SigLIP | Open-source multi-image and video reasoning |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | 7B / 72B | NaViT Dynamic ViT | Native dynamic resolution without fixed grid constraints |

## 9.2 Visual Synthesis and Generative Foundations

### 9.2.1 Diffusion Models & Rectified Flow

$$\text{Forward SDE (Noise Injection): } \mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\mathrm{d}t + g(t)\mathrm{d}\mathbf{w}_t$$
$$\text{Reverse Denoising Objective: } \mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})\|^2 \right]$$

**Latent Diffusion Architecture (LDM)** ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752)): Executes the forward and reverse diffusion processes within the low-dimensional latent space of a pretrained Variational Autoencoder (VAE), reducing spatial dimensions by 8x and compute requirements by over an order of magnitude.

**Diffusion Transformers (DiT)** ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)): Replaces traditional convolutional U-Net backbones with standard Transformer architectures operating on visual latent patches, inheriting the compute scaling laws of LLMs (powers FLUX and Stable Diffusion 3).

### 9.2.2 Autoregressive Visual Generation

- **Discrete Tokenization (VQ-VAE / VQ-GAN)**: Quantizes continuous pixel patches into discrete visual codebook indices ($z_q \in \{1, \dots, V\}$), enabling standard autoregressive cross-entropy next-token prediction over interleaved image-text streams ([Chameleon](https://arxiv.org/abs/2405.09818)).
- **Transfusion** ([Chung et al., 2024](https://arxiv.org/abs/2408.11039)): Unifies causal language modeling (cross-entropy on discrete text tokens) with continuous diffusion objectives (MSE on continuous visual latent patches) within a single Transformer backbone.

## 9.3 Audio, Speech, and Temporal Perception

### 9.3.1 Robust Speech Transcription (Whisper)

([Radford et al., 2022: Whisper](https://arxiv.org/abs/2212.04356)): Encoder-Decoder Transformer trained over 680,000 hours of weakly supervised multilingual audio. Converts raw log-Mel filterbank spectrograms directly into transcribed and translated text sequences without external acoustic models.

### 9.3.2 Neural Audio Codecs & Generative Speech

- **Neural Audio Tokenizers (EnCodec / SoundStream)**: Compresses continuous audio waveforms into multi-codebook discrete tokens via Residual Vector Quantization (RVQ).
- **Autoregressive Audio Generation (VALL-E / F5-TTS)**: Models acoustic tokens autoregressively or via continuous flow matching, enabling high-fidelity zero-shot voice cloning from 3-second reference prompts.

## 9.4 Video Understanding & Spatiotemporal Generation

### 9.4.1 Spatiotemporal Video Encoding

Video inputs combine spatial image coordinates with temporal dynamics ($T \times H \times W \times C$):
1. **Dynamic Frame Sampling**: Uniformly samples keyframes (e.g. 1 to 2 FPS), encoding each independently while prepending relative temporal positional embeddings.
2. **3D Spatiotemporal Patch Tiling**: Partitions video volumes into 3D voxel patches ($t_{\text{patch}} \times h_{\text{patch}} \times w_{\text{patch}}$), processing space and time jointly in self-attention layers.

### 9.4.2 Video Synthesis Engines

- **Spatial-Temporal DiT (Sora, HunyuanVideo, Wan)**: Flattens video latents into sequences of 3D spacetime tokens, optimizing rectified flow matching trajectories over high-capacity Transformer backbones.

## 9.5 Omni-Modal Unified Systems

**The Frontier Direction**: Single unified models accepting arbitrary interleaved modalities (text, audio, vision, video) and emitting real-time streaming multimodal responses.

```
Universal Representation Architecture:
Arbitrary Inputs ──> Modality-Specific Tokenizers ──> Unified Causal Backbone ──> Streaming Modality Heads
```

- **Speech-to-Speech Streaming**: Bypasses intermediate ASR and TTS text translation stages, preserving emotional prosody, conversational interruptions, and sub-300ms interaction latencies.

## Key Papers

- [Radford et al. (2021): Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020): Foundational multimodal contrastive representation paper.
- [Liu et al. (2023): Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485): Landmark visual instruction tuning and projector architecture.
- [Peebles & Xie (2023): Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748): Introduction of Transformer architectures to diffusion generative models.
- [Radford et al. (2022): Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356): Milestone speech transcription architecture.
- [Chung et al. (2024): Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039): Unified discrete text and continuous visual generation.

## Further Reading

- Haotian Liu: [LLaVA Project Repository](https://github.com/haotian-liu/LLaVA) (Open-source visual instruction tuning code and checkpoints).
- Qwen Team: [Qwen2-VL Architecture and Benchmark Suite](https://github.com/QwenLM/Qwen2-VL) (Dynamic resolution visual modeling).
- OpenAI: [Whisper Architecture and Codebase](https://github.com/openai/whisper) (Open-source robust speech transcription).

## Exercises

1. **VLM Perception Inspection**: Deploy an open-weight VLM (such as Qwen2-VL-7B); test complex document OCR, table structure recovery, and chart numerical reasoning, noting where spatial grounding fails.
2. **Vision Encoder Ablation**: Compare visual question answering accuracy on identical instruction data when pairing an LLM backbone with CLIP-ViT-L versus SigLIP.
3. **Multimodal RAG Implementation**: Index a collection of technical diagrams and text passages using CLIP/BGE-M3 embeddings; evaluate text-to-image retrieval precision and multimodal synthesis quality.

---

[← Previous Chapter](08-embedding-rag.md) | [Table of Contents](README.md) | [Next Chapter →](10-safety-alignment.md)
