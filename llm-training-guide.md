# LLM 训练工程师完全指南

> 从 Tokenizer 到 Post-Training，从单卡到万卡集群，从纯文本到多模态。
> 目标读者：想系统掌握 LLM 全栈训练能力的工程师。

---

## 目录

- [第一章：Tokenizer](#第一章tokenizer)
- [第二章：模型架构](#第二章模型架构)
- [第三章：预训练 (Pretraining)](#第三章预训练-pretraining)
- [第四章：Post-Training（后训练）](#第四章post-training后训练)
- [第五章：参数高效微调 (PEFT)](#第五章参数高效微调-peft)
- [第六章：训练基础设施 (Infra)](#第六章训练基础设施-infra)
- [第七章：多模态 (Multimodal)](#第七章多模态-multimodal)
- [第八章：评估与对齐 (Evaluation & Alignment)](#第八章评估与对齐-evaluation--alignment)
- [第九章：SOTA 模型深度解析](#第九章sota-模型深度解析)
- [第十章：知识蒸馏与模型合并](#第十章知识蒸馏与模型合并)
- [第十一章：实战路线图](#第十一章实战路线图)
- [附录：关键论文清单](#附录关键论文清单)

---

# 第一章：Tokenizer

Tokenizer 是 LLM 的入口——把原始文本变成模型能处理的 token ID 序列。Tokenizer 的设计直接影响模型的词汇覆盖率、序列长度、多语言能力和训练效率。

## 1.1 为什么需要 Tokenizer

神经网络处理的是数值张量，不是字符串。最简单的做法是 character-level（每个字符一个 token），但这样：
- 序列太长（一个句子几十上百个 token），attention 的 O(n²) 代价太高
- 没有语义粒度，模型要自己学单词边界

另一个极端是 word-level（每个单词一个 token），但：
- 词表爆炸（英语就有几十万词形）
- 无法处理 OOV (out-of-vocabulary) 词
- 对中日韩等语言不友好（没有空格分词）

Subword tokenization 是折中方案：高频词保留为整词，低频词拆成子词片段。

## 1.2 核心算法

### 1.2.1 BPE (Byte Pair Encoding)

**原理**: 从字符（或字节）级别开始，反复合并出现频率最高的相邻 pair，直到词表达到目标大小。

**训练过程**:
```
初始词表: 所有单字符（或字节 0-255）
循环:
    1. 统计所有相邻 token pair 的频率
    2. 合并频率最高的 pair → 新 token
    3. 更新语料中的 token 序列
    4. 词表大小达到目标 → 停止
```

**举例**:
```
语料: "low lower lowest"
初始: ['l','o','w',' ','l','o','w','e','r',' ','l','o','w','e','s','t']
第1轮: 'l'+'o' → 'lo' (出现3次, 最高频)
第2轮: 'lo'+'w' → 'low' (出现3次)
第3轮: 'low'+'e' → 'lowe' (出现2次)
...
```

**GPT 系列的 BPE**: OpenAI 的 [tiktoken](https://github.com/openai/tiktoken) 在字节级 BPE 上做了改进：
- 基于 UTF-8 字节而非 Unicode 字符，天然支持任何语言
- 用正则预分词（把文本先切成大块再跑 BPE），避免跨词合并
- GPT-4 用的 `cl100k_base` 词表有 ~100K tokens

```python
# tiktoken 使用
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
print(enc.decode(tokens))  # "Hello, world!"
```

### 1.2.2 WordPiece

**原理**: 和 BPE 类似，但合并标准不同。BPE 选频率最高的 pair，WordPiece 选使语言模型 likelihood 提升最大的 pair。

**公式**: 选择合并 (x, y) → xy 使得:
```
score(x, y) = freq(xy) / (freq(x) × freq(y))
```

这等价于 pointwise mutual information (PMI)。高 PMI 意味着 x 和 y 经常一起出现，合并后信息量大。

**使用者**: BERT、DistilBERT 等。词表标记用 `##` 前缀表示续接（如 `playing` → `play` + `##ing`）。

### 1.2.3 Unigram (SentencePiece)

**原理**: 反向操作——从一个大词表开始，逐步删除使语料 likelihood 下降最小的 token，直到词表缩到目标大小。

**训练过程**:
```
1. 初始化: 用所有子串 + 字符构建大词表 (如 100万)
2. 用 EM 算法估计每个 token 的概率 P(token)
3. 计算每个 token 的 loss 贡献: 如果去掉它，语料的 log-likelihood 下降多少
4. 删掉 loss 贡献最小的 10-20% token
5. 重复 2-4 直到词表大小达到目标
```

**优势**: Unigram 可以输出 N-best 分词结果（概率化），对正则化有好处。

**使用者**: T5、LLaMA、Gemma 等都用 [SentencePiece](https://github.com/google/sentencepiece) 的 Unigram 模型。

### 1.2.4 对比

| 特征 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| 方向 | 自底向上合并 | 自底向上合并 | 自顶向下删减 |
| 合并标准 | 频率 | Likelihood/PMI | Likelihood (EM) |
| 确定性 | 确定 | 确定 | 概率化 (可采样) |
| 代表模型 | GPT系列, LLaMA 2 | BERT | T5, LLaMA 3, Gemma |

## 1.3 字节级 vs 字符级

**Byte-level BPE** (GPT-2/3/4, LLaMA):
- 基础单元是 UTF-8 字节 (0-255)，而非 Unicode 字符
- 优势：永远不会遇到 OOV，任何字节序列都能编码
- 劣势：非 ASCII 字符（如中文）一个字可能要 3 个字节起步

**Byte-level Fallback** (SentencePiece):
- 正常走 Unigram/BPE，遇到 OOV 字符时回退到字节表示
- LLaMA 3 用这种方式

**纯字节模型** ([ByT5](https://arxiv.org/abs/2105.13626)):
- 完全不用 tokenizer，直接输入 UTF-8 字节
- 优势：零预处理，鲁棒性极强
- 劣势：序列长度暴增 3-4 倍，计算量大

## 1.4 实战：训练你自己的 Tokenizer

```python
# 用 HuggingFace tokenizers 库训练 BPE tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    show_progress=True,
)

# 从文件训练
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("my_tokenizer.json")

# 测试
output = tokenizer.encode("Hello, 世界!")
print(output.tokens)
print(output.ids)
```

> 库: [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | 最小化实现: [karpathy/minbpe](https://github.com/karpathy/minbpe)

**关键决策**:
- **词表大小**: 32K-256K。越大 → 序列越短但 embedding 层越大。LLaMA 2 用 32K，LLaMA 3 用 128K，GPT-4 用 100K
- **特殊 token**: `<bos>`, `<eos>`, `<pad>`, `<unk>`，chat 模型还需要 `<|im_start|>`, `<|im_end|>` 等
- **预分词**: 用正则把文本切块，防止跨"自然边界"合并（如数字和字母、标点和单词）
- **规范化**: 是否做 NFKC Unicode 归一化、大小写折叠等

## 1.5 多语言 Tokenizer 设计

多语言是 tokenizer 的大挑战：

**问题**: 如果训练数据以英语为主，BPE 合并偏向英文 — 中文每个字可能需要 3-4 个 token（因为 UTF-8 编码），而英文一个单词只需 1 个 token。这导致中文输入的有效上下文窗口只有英文的 1/3。

**解决方案**:
1. **平衡训练语料**: 按语言采样，确保每种语言有足够数据参与 merge
2. **扩大词表**: LLaMA 3 从 32K→128K，大幅改善中文效率
3. **语言特定预分词**: 中文用 jieba/sentencepiece 先分词再跑 BPE
4. **字符覆盖**: 确保高频汉字/日文假名等作为单独 token 存在

**Fertility 指标**: 衡量 tokenizer 对某语言的效率。`fertility = tokens / words`，越接近 1 越好。GPT-2 对中文 fertility ~3.5，LLaMA 3 ~1.5。

## 1.6 SOTA Tokenizer 技巧

- **Token Healing** ([guidance](https://github.com/guidance-ai/guidance)): 生成时修复 tokenizer 导致的 prompt 边界问题
- **Byte Fallback**: SentencePiece 的 `byte_fallback=True`，OOV 回退到字节
- **Split digits**: 把数字拆成单独的 digit token（`2024` → `2` `0` `2` `4`），提升数学能力
- **Whitespace handling**: 保留前导空格作为 token 一部分（GPT 风格）vs 单独 token
- **Code tokens**: 保留缩进（tab、多空格）作为特殊 token，提升代码生成

---

# 第二章：模型架构

## 2.1 Transformer 基础

Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) 是现代 LLM 的骨架。核心组件：

### 2.1.1 Self-Attention

**直觉**: 对于序列中的每个 token，计算它应该"关注"其他哪些 token。

**计算过程**:
```
输入: X ∈ ℝ^(n×d)  # n个token, d维

Q = X @ W_Q  # Query  ∈ ℝ^(n×d_k)
K = X @ W_K  # Key    ∈ ℝ^(n×d_k)
V = X @ W_V  # Value  ∈ ℝ^(n×d_v)

Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**为什么除以 √d_k**: 点积的方差随维度增长，不缩放的话 softmax 会饱和（梯度消失）。

**Multi-Head Attention (MHA)**:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

多个 head 让模型在不同子空间学习不同类型的注意力模式（如语法关系、语义关系、位置关系）。

### 2.1.2 FFN (Feed-Forward Network)

每个 Transformer 层里，attention 之后是一个 position-wise FFN：
```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

- 原始 Transformer 用 ReLU
- GPT-2 用 GELU
- LLaMA/现代模型用 SwiGLU ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)):
```
SwiGLU(x) = (x @ W_1 · σ(x @ W_gate)) @ W_2
```
SwiGLU 用了 gating mechanism，实际效果更好（但参数多 50%）。

### 2.1.3 Layer Normalization

**位置很重要**:
- **Post-Norm** (原始 Transformer): `x + LayerNorm(SubLayer(x))`
  - 训练不稳定，需要 warmup
- **Pre-Norm** (GPT-2+, 现代 LLM 标配): `x + SubLayer(LayerNorm(x))`
  - 训练更稳定，但最终性能可能略差
- **RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) (LLaMA, Gemma): Pre-Norm 用 RMSNorm 替代 LayerNorm
  ```
  RMSNorm(x) = x / RMS(x) * γ
  RMS(x) = √(mean(x²))
  ```
  去掉了均值中心化（减去 mean），计算更快，效果不输 LayerNorm

### 2.1.4 Decoder-Only vs Encoder-Decoder

| 架构 | 代表模型 | 训练目标 | 适用场景 |
|------|---------|---------|---------|
| Encoder-only | BERT, RoBERTa | Masked LM | 分类、NER、检索 |
| Encoder-Decoder | T5, BART | Seq2Seq | 翻译、摘要 |
| **Decoder-only** | **GPT, LLaMA, Claude** | **Causal LM** | **生成、对话、通用** |

**为什么 Decoder-only 赢了**:
- Scaling law 表现最好（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)）
- 训练简单：一个 loss (next token prediction)
- 天然支持 in-context learning
- 统一了理解和生成

## 2.2 位置编码 (Positional Encoding)

Attention 是置换不变的（permutation invariant），所以必须注入位置信息。

### 2.2.1 绝对位置编码

**正弦位置编码** (原始 Transformer):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- 优势：理论上可外推到任意长度
- 劣势：实践中外推效果差

**可学习位置编码** (GPT-1/2):
- 每个位置一个可学习向量 `P[i] ∈ ℝ^d`
- 直接加到 token embedding 上
- 劣势：固定最大长度，不能外推

### 2.2.2 RoPE (Rotary Position Embedding)

**现代 LLM 的标准选择** ([Su et al., 2021](https://arxiv.org/abs/2104.09864))。LLaMA, Qwen, Mistral, Gemma 等全部使用。

**核心思想**: 不加位置编码到 embedding 上，而是在 attention 计算时旋转 Q 和 K。

```
# 把 d 维向量看成 d/2 个二维平面
# 在每个平面上按位置旋转角度 θ

R(pos, i) = [[cos(pos·θ_i), -sin(pos·θ_i)],
              [sin(pos·θ_i),  cos(pos·θ_i)]]

# θ_i = 10000^(-2i/d)

q_rotated = R(m) @ q   # position m
k_rotated = R(n) @ k   # position n

# 关键性质: q_rotated^T @ k_rotated 只依赖于相对位置 (m-n)
```

**为什么好**:
- 天然编码相对位置
- 远距离 token 间的 attention 自然衰减
- 可以通过调整 θ 的 base 来扩展上下文窗口（见 2.2.3）

### 2.2.3 长上下文扩展

预训练通常只用 2K-8K context，但推理时想用 128K+。

**NTK-aware Interpolation** ([Reddit/bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), Code LLaMA 风格):
```python
# 不直接做位置插值（会模糊近距离信息），
# 而是修改 RoPE 的 base frequency
base_new = base_old * (scale_factor ** (d / (d - 2)))
# d = dimension, scale_factor = target_length / trained_length
```

**YaRN** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071)):
- 把 RoPE 的频率维度分为三组：低频（不改）、中频（插值）、高频（NTK 缩放）
- 加一个 temperature 修正 attention 分布
- LLaMA 3.1 的 128K context 就用了 YaRN 变体

**ALiBi** ([Press et al., 2022](https://arxiv.org/abs/2108.12409)):
- 不用位置编码，直接在 attention score 上加线性偏置：`score -= m * |i - j|`
- m 是 head-specific 的斜率
- 训练短、推理长，外推效果好
- 用于 BLOOM、MPT

## 2.3 现代架构改进

### 2.3.1 GQA (Grouped-Query Attention)

([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245))

**问题**: MHA 中每个 head 有独立的 K、V，KV cache 随 head 数线性增长，是长上下文推理的瓶颈。

**解决方案**: 多个 Q head 共享一组 K、V head。

```
MHA:  h个Q head, h个K head, h个V head  (如 h=32)
MQA:  h个Q head, 1个K head, 1个V head  (极端共享)
GQA:  h个Q head, g个K head, g个V head  (如 h=32, g=8)
```

- **GQA** 是 MHA 和 MQA 的折中
- KV cache 减少 h/g 倍（如 32/8 = 4倍）
- 质量损失很小
- **LLaMA 2 70B, LLaMA 3, Mistral, Gemma 2 全部使用 GQA**

```python
# GQA 伪代码
def grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads):
    # Q: [batch, seq, num_q_heads, head_dim]
    # K, V: [batch, seq, num_kv_heads, head_dim]
    
    group_size = num_q_heads // num_kv_heads  # e.g., 4
    
    # 扩展 K, V 到 Q 的 head 数（每组共享）
    K = K.repeat_interleave(group_size, dim=2)
    V = V.repeat_interleave(group_size, dim=2)
    
    return standard_attention(Q, K, V)
```

### 2.3.2 MoE (Mixture of Experts)

**核心思想**: FFN 层变成多个 "expert" FFN，每个 token 只激活其中 top-k 个。([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538))

```
# 标准 FFN:
output = FFN(x)

# MoE:
gate_scores = softmax(x @ W_gate)  # [n_experts]
top_k_experts = topk(gate_scores, k=2)

output = Σ(gate_scores[i] * Expert_i(x))  for i in top_k_experts
```

**优势**: 总参数量大但每个 token 的计算量小。[Mixtral 8x7B](https://arxiv.org/abs/2401.04088) 有 47B 参数但推理只激活 ~13B。

**挑战**:
- **Load balancing**: 需要 auxiliary loss 防止所有 token 涌入同一个 expert
  ```
  L_balance = α * n_experts * Σ(f_i * P_i)
  # f_i = fraction of tokens routed to expert i
  # P_i = average gate probability for expert i
  ```
- **通信开销**: expert 分布在不同 GPU 上，token 路由需要 all-to-all 通信
- **训练不稳定**: 容易出现 expert collapse（某些 expert 学废了，永远不被选）

**SOTA MoE 模型**: [Mixtral 8x7B/8x22B](https://arxiv.org/abs/2401.04088), [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), Grok-1, [DeepSeek-V2/V3](https://arxiv.org/abs/2412.19437), Qwen2-MoE

**DeepSeek-V2/V3 的 MoE 改进** ([DeepSeek-V2](https://arxiv.org/abs/2405.04434)):
- **DeepSeekMoE**: 更多更小的 expert (如 160 个 expert, top-6 激活)，替代传统的 8 expert top-2
- **Shared expert**: 保留 1-2 个 expert 始终激活（处理通用知识），其余 expert 竞争路由
- **Auxiliary-loss-free load balancing**: 用 bias term 而非 loss 来平衡负载

### 2.3.3 MLA (Multi-head Latent Attention)

[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 引入, DeepSeek-V3/R1 沿用。

**问题**: GQA 虽然减少了 KV cache，但仍有信息损失。

**方案**: 把 KV 压缩到一个低维 latent 向量，推理时从 latent 恢复 KV。

```
# 训练时:
c_kv = x @ W_DKV           # 压缩: d_model → d_latent (如 4096 → 512)
K = c_kv @ W_UK             # 解压 Key
V = c_kv @ W_UV             # 解压 Value

# 推理 KV cache 只存 c_kv (512维) 而非 K+V (4096*2维)
```

KV cache 压缩到原来的 ~1/16，同时质量不输 MHA。

### 2.3.4 Sliding Window Attention

[Mistral 7B](https://arxiv.org/abs/2310.06825) 引入: 不同层交替使用全局 attention 和滑动窗口 attention。

```
# 滑动窗口: 每个 token 只能看前后 W 个 token
# 但因为层叠效果，信息可以跨层传播到更远
# L 层滑动窗口的有效感受野 = L * W

# Mistral 7B: W=4096, 32层 → 理论感受野 131072
```

**优势**: 前面的层用局部 attention（捕捉局部模式），后面的层用全局 attention（整合全局信息）。减少计算量但保持性能。

### 2.3.5 其他现代技巧

**Parallel Attention + FFN** (GPT-J, [PaLM](https://arxiv.org/abs/2204.02311) 风格):
```python
# 标准: 串行
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# 并行: 计算更快 (attention 和 FFN 可以并行)
x = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))
```

**QK-Norm**: 对 Q 和 K 做 LayerNorm/RMSNorm，防止注意力 logits 爆炸。[Gemma 2](https://arxiv.org/abs/2408.00118), Cohere 等使用。

**Logit Soft-Capping** ([Gemma 2](https://arxiv.org/abs/2408.00118)): `logits = soft_cap * tanh(logits / soft_cap)`，限制 attention logits 和 final logits 的范围。

## 2.4 模型规模设计

### Scaling Laws (Chinchilla)

[Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) 发现：

```
给定计算预算 C (FLOPs):
最优模型大小 N ∝ C^0.5
最优数据量 D ∝ C^0.5
即: N 和 D 应该同步增长

经验公式:
最优 tokens 数 ≈ 20 × 参数量
```

| 参数量 | 最优 tokens | 代表模型 |
|--------|------------|---------|
| 1B | 20B | [TinyLlama](https://github.com/jzhang38/TinyLlama) |
| 7B | 140B | LLaMA 2 7B (2T 实际用了更多) |
| 13B | 260B | LLaMA 2 13B |
| 70B | 1.4T | LLaMA 2 70B |

**实际趋势**: 后来的模型普遍 over-train（用远超 Chinchilla 最优的数据量），因为推理成本 > 训练成本，smaller model + more data 在推理时更划算。LLaMA 3 8B 训练了 15T tokens。

### 常见架构参数

| 模型 | 参数量 | Layers | Hidden | Heads | KV Heads | Vocab | Context |
|------|--------|--------|--------|-------|----------|-------|---------|
| [LLaMA 2 7B](https://arxiv.org/abs/2307.09288) | 6.7B | 32 | 4096 | 32 | 32 (MHA) | 32K | 4K |
| [LLaMA 3 8B](https://arxiv.org/abs/2407.21783) | 8.0B | 32 | 4096 | 32 | 8 (GQA) | 128K | 128K |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 7.3B | 32 | 4096 | 32 | 8 (GQA) | 32K | 32K |
| [Qwen2 72B](https://arxiv.org/abs/2407.10671) | 72.7B | 80 | 8192 | 64 | 8 (GQA) | 152K | 128K |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 671B | 61 | 7168 | 128 | MLA | 128K | 128K |

---

# 第三章：预训练 (Pretraining)

## 3.1 训练目标

### 3.1.1 Causal Language Modeling (CLM)

**所有现代 LLM 的核心目标**: 预测下一个 token。

```
给定序列 x = (x_1, x_2, ..., x_n)
Loss = -Σ log P(x_t | x_1, ..., x_{t-1})
```

**为什么 next-token prediction 这么强大**:
- 压缩即理解：要准确预测下一个 token，模型必须理解语法、语义、世界知识、推理
- Self-supervised：不需要人工标注，互联网文本就是训练数据
- Scaling：简单的目标 + 更多数据 + 更大模型 = 持续的能力提升

### 3.1.2 Fill-in-the-Middle (FIM)

在 CLM 基础上加入中间填充能力（对代码补全尤其重要）([Bavarian et al., 2022](https://arxiv.org/abs/2207.14255)):

```
原始: A B C D E
变换: <fim_prefix> A B <fim_suffix> D E <fim_middle> C

# 模型学会了在知道前后文的情况下填充中间
# PSM (prefix-suffix-middle) 格式
```

- GPT-4, [StarCoder](https://arxiv.org/abs/2305.06161), [CodeLlama](https://arxiv.org/abs/2308.12950) 等代码模型都用 FIM
- 通常只对一部分数据做 FIM 变换（如 50%），其余保持 CLM

### 3.1.3 Masked Language Modeling (MLM)

[BERT](https://arxiv.org/abs/1810.04805) 风格，随机 mask 15% 的 token，预测被 mask 的 token：
```
输入: The [MASK] sat on the [MASK]
目标: cat, mat
```
- 双向注意力，适合理解任务
- 不适合生成（推理时没有 mask）
- 现在主要用于 encoder 模型（embedding、检索）

## 3.2 数据 (最关键的部分)

### 3.2.1 数据来源

| 来源 | 规模 | 质量 | 代表数据集 |
|------|------|------|-----------|
| Web crawl | PB级 | 低 | [Common Crawl](https://commoncrawl.org/), [C4](https://huggingface.co/datasets/allenai/c4), [OSCAR](https://oscar-project.github.io/documentation/) |
| 百科/知识 | TB级 | 高 | [Wikipedia](https://dumps.wikimedia.org/), Wikidata |
| 书籍 | TB级 | 高 | Books3, [Project Gutenberg](https://www.gutenberg.org/) |
| 代码 | TB级 | 中 | GitHub, [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| 科学论文 | TB级 | 高 | [arXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [S2ORC](https://github.com/allenai/s2orc) |
| 对话/论坛 | TB级 | 中 | Reddit, StackOverflow |
| 数学 | GB级 | 高 | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), [ProofPile](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |

### 3.2.2 数据处理 Pipeline

```
Raw Web Crawl
    │
    ▼
① URL 过滤 (去掉成人/垃圾/违规网站)
    │
    ▼
② 语言识别 (fastText lid.176.bin)
    │
    ▼
③ 文本提取 (trafilatura, resiliparse)
    │  - 去 HTML boilerplate
    │  - 提取正文
    │
    ▼
④ 质量过滤
    │  - 长度过滤 (去掉太短的)
    │  - 困惑度过滤 (KenLM, 去掉 perplexity 太高的)
    │  - 分类器过滤 (训练一个 fastText 分类器区分高质量/低质量)
    │  - 规则过滤 (去掉重复词比例高、特殊字符比例高的)
    │
    ▼
⑤ 去重 (Deduplication)
    │  - Exact dedup: hash-based (SHA-256)
    │  - Fuzzy dedup: MinHash + LSH (jaccard similarity > 0.8 → 去重)
    │  - 跨文档去重: 全局 MinHash
    │
    ▼
⑥ PII 去除
    │  - 正则匹配邮箱、电话、身份证号
    │  - NER 模型识别人名、地址
    │
    ▼
⑦ Tokenize → 二进制格式 (存为 memory-mapped 文件)
```

> 工具: [HuggingFace datatrove](https://github.com/huggingface/datatrove) — 完整的数据处理框架 | [dolma](https://github.com/allenai/dolma) — Allen AI 的数据 toolkit

### 3.2.3 数据配比 (Data Mix)

**关键决策**：不同来源的数据按什么比例混合。

LLaMA 3 的大致配比:
```
Web (英文): ~50%
Web (多语言): ~15%
代码: ~17%
数学/科学: ~5%
书籍: ~5%
百科/知识: ~5%
对话: ~3%
```

**配比调优方法**:
1. **小模型代理实验**: 用 1B 模型跑多组不同配比，选下游任务最好的
2. **DoReMi** ([Xie et al., 2023](https://arxiv.org/abs/2305.10429)): 用 DRO (Distributionally Robust Optimization) 自动学习最优配比
3. **经验法则**: 代码比例↑提升推理能力，数学比例↑提升数学能力，对话比例↑提升指令遵循

### 3.2.4 Data Curriculum

**不是一股脑把所有数据灌进去，而是按阶段调整数据**:

1. **Phase 1** (大部分训练): 通用 web 数据，大量、多样
2. **Phase 2** (后期): 增加高质量数据比例（代码、数学、知识）
3. **Annealing** (最后几%的训练): 
   - 大幅提升高质量数据比例
   - 降低学习率到接近 0
   - LLaMA 3 的 annealing 阶段把高质量数据比例提到 ~50%

### 3.2.5 Synthetic Data

**当前最大趋势**: 用强模型生成数据训练弱模型。

| 类型 | 方法 | 用途 |
|------|------|------|
| Instruction data | 强模型 (GPT-4) 生成 QA pairs | SFT |
| Code data | 模型生成代码 + 执行验证 | Code pretraining |
| Math data | 模型生成证明步骤 + 验证 | Math reasoning |
| Textbook-quality | ["Textbooks Are All You Need"](https://arxiv.org/abs/2306.11644) (Phi) | 预训练 |
| Rephrased data | 用强模型重写 web 文本为教科书风格 | 预训练质量提升 |

**[Phi 系列](https://arxiv.org/abs/2404.14219) (Microsoft)**: 证明高质量 synthetic data + 小模型可以达到大模型级别的性能。Phi-3 3.8B ≈ LLaMA 3 8B。

## 3.3 训练过程

### 3.3.1 优化器

**AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) (几乎所有 LLM 的标准选择):
```
m_t = β_1 * m_{t-1} + (1 - β_1) * g_t          # 一阶矩 (momentum)
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t²         # 二阶矩 (adaptive LR)
m̂_t = m_t / (1 - β_1^t)                         # bias correction
v̂_t = v_t / (1 - β_2^t)
θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + wd * θ_{t-1})  # weight decay
```

**典型超参**:
- β_1 = 0.9, β_2 = 0.95 (LLaMA), 有的用 0.999
- ε = 1e-8
- weight_decay = 0.1
- 峰值学习率: 3e-4 (小模型) ~ 1.5e-4 (大模型)

**Adam 的内存开销**: 每个参数需要存 m (fp32) + v (fp32) + params (fp16/bf16)。一个 70B 模型光优化器状态就要 ~560GB。

**替代优化器**:
- **[Adafactor](https://arxiv.org/abs/1804.04235)**: 用矩阵分解近似 v，内存减半。T5 用
- **[LION](https://arxiv.org/abs/2302.06675)**: Google 提出，只用 sign(momentum)，内存更小。但需要更仔细的调参
- **[Sophia](https://arxiv.org/abs/2305.14342)**: 用二阶信息（Hessian diagonal 估计）做 adaptive LR。收敛更快但计算更贵
- **[MUON](https://arxiv.org/abs/2502.16982)**: 最新的优化器，用 momentum 的 SVD 来做更新方向。某些设置下大幅加速收敛

### 3.3.2 学习率调度

**标准方案**: Linear Warmup + Cosine Decay

```
# Warmup: 前 2000 步线性增到峰值
if step < warmup_steps:
    lr = peak_lr * step / warmup_steps

# Cosine decay: 缓慢降到最小值 (通常是峰值的 10%)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
```

**WSD (Warmup-Stable-Decay)** ([MiniCPM](https://arxiv.org/abs/2404.06395)):
```
Warmup → 恒定学习率（大部分训练时间）→ 快速 decay
```
- 优势：可以随时在 stable 阶段"分支"出一个快速 decay 的 checkpoint
- MiniCPM, DeepSeek 等使用
- 方便做 continual pretraining

### 3.3.3 训练稳定性

**Loss spike**: 训练中 loss 突然飙升的现象，常见于大模型。

**原因和对策**:
| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 梯度爆炸 | loss 突然 NaN/Inf | gradient clipping (max_norm=1.0) |
| 数据问题 | loss spike 后可恢复 | 跳过坏 batch (loss > threshold 就跳过) |
| 学习率过大 | 反复 spike | 降低 peak LR |
| 注意力发散 | QK 点积爆炸 | QK-Norm, Logit soft-capping |
| BF16 溢出 | embedding 层 spike | embedding 层用 FP32 |
| z-loss | logits 绝对值过大 | 加 z-loss 正则化: α * log²(Σexp(logits)) |

**PaLM 的经验**: Google 训练 [PaLM 540B](https://arxiv.org/abs/2204.02311) 时遇到 ~20 次 loss spike，处理方式是回滚到 spike 前的 checkpoint，跳过导致 spike 的数据。

### 3.3.4 Batch Size 策略

```
有效 batch size = micro_batch_size × gradient_accumulation × dp_world_size

典型值:
- LLaMA 2: 4M tokens per batch
- GPT-4 (推测): 60M+ tokens per batch
```

**Batch size warmup**: 训练初期用小 batch（更好的泛化），后期增大（更稳定、更快）。

**Critical batch size** ([McCandlish et al., 2018](https://arxiv.org/abs/1812.06162)): 低于此值，增大 batch 几乎线性加速；高于此值，收益递减。经验公式:
```
B_crit ≈ B_noise / L   # B_noise 是梯度噪声规模，L 是当前 loss
```

### 3.3.5 数值精度

| 精度 | 位宽 | 范围 | 用途 |
|------|------|------|------|
| FP32 | 32位 | ±3.4e38 | 优化器状态、master weights |
| TF32 | 19位 | 同 FP32 | NVIDIA Ampere+ 默认 matmul |
| BF16 | 16位 | ±3.4e38 | **主流训练精度** (范围同 FP32) |
| FP16 | 16位 | ±65504 | 旧一代训练精度 (需要 loss scaling) |
| FP8 | 8位 | E4M3/E5M2 | H100+ 前沿训练 |

**Mixed Precision Training** ([Micikevicius et al., 2018](https://arxiv.org/abs/1710.03740)):
```python
# PyTorch AMP
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# 梯度计算用 BF16，优化器更新用 FP32
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**FP8 Training** (H100/B200):
- 前向用 E4M3 (更高精度)
- 反向用 E5M2 (更大范围)
- 需要 per-tensor scaling
- DeepSeek-V3 成功在 FP8 下训练了 671B MoE

## 3.4 上下文长度训练

### 3.4.1 短→长训练策略

1. **Phase 1**: 在 4K/8K context 上预训练大部分 token
2. **Phase 2**: 在长序列 (32K-128K) 上继续训练少量 token
   - 调整 RoPE base frequency
   - 只需要 ~1-5% 的总训练 token 数

**为什么不直接在长序列上训练**:
- Attention 的 O(n²) 计算量，128K 是 4K 的 1024 倍
- 长文档数据稀缺，短文档更多样
- 先学好语言基础，再学长距离依赖

### 3.4.2 Ring Attention / Context Parallelism

([Liu et al., 2023](https://arxiv.org/abs/2310.01889)) 超长序列不能放到一张卡上时，把序列切成 chunks 分布到多张卡，用 ring 通信传递 KV。

```
GPU 0: tokens 0-32K      → 计算 attention chunk
GPU 1: tokens 32K-64K    → 计算 attention chunk
GPU 2: tokens 64K-96K    → 计算 attention chunk
GPU 3: tokens 96K-128K   → 计算 attention chunk
           ↻ KV blocks 在 GPU 间 ring 传递
```

---

# 第四章：Post-Training（后训练）

预训练得到的是一个"文本补全器"，post-training 把它变成有用的"助手"。

## 4.1 概览

```
Base Model (预训练)
    │
    ▼
SFT (Supervised Fine-Tuning)
    │  - 学会按指令格式回答
    │
    ▼
Preference Optimization (RLHF / DPO / ...)
    │  - 学会什么是"好"回答
    │
    ▼
Safety Training
    │  - 拒绝有害请求
    │
    ▼
Deployed Model
```

## 4.2 SFT (Supervised Fine-Tuning)

### 4.2.1 数据格式

**Chat Format** (标准):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

**Loss masking**: 只对 assistant 的回复计算 loss，不对 system 和 user 消息计算 loss。

### 4.2.2 SFT 数据来源

| 来源 | 规模 | 质量 | 例子 |
|------|------|------|------|
| 人工标注 | 10K-100K | 极高 | OpenAI 的内部数据, Anthropic 的标注数据 |
| 开源数据集 | 100K-1M | 中-高 | [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1), [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| Synthetic | 1M+ | 中 | [Self-Instruct](https://arxiv.org/abs/2212.10560), [Evol-Instruct](https://arxiv.org/abs/2304.12244), [Magpie](https://arxiv.org/abs/2406.08464) |
| Distillation | 1M+ | 中-高 | 用 GPT-4/Claude 生成回答 |

**LIMA 发现** ([Zhou et al., 2023](https://arxiv.org/abs/2305.11206)): "Less Is More for Alignment" — 仅 1000 条高质量 SFT 数据就能让模型有不错的对话能力。质量 > 数量。

### 4.2.3 SFT 技巧

**Instruction Diversity**: 比回答质量更重要。覆盖：代码、数学、创意写作、摘要、翻译、角色扮演、工具使用...

**Rejection Sampling**: 对每个 prompt 生成 N 个回答，用 reward model 或规则选最好的。这是 Meta 训练 LLaMA 3 的关键技巧。

**超参**:
- 学习率: 1e-5 ~ 2e-5（比预训练低 10-100 倍）
- Epochs: 2-5（数据少则多 epoch，数据多则 1-2 epoch）
- Batch size: 128-512 samples

### 4.2.4 Long-Context SFT

针对长上下文能力做专门 SFT:
- 包含长文档 QA、多文档摘要、长代码理解等任务
- 训练数据长度分布应覆盖目标上下文窗口
- "Needle in a haystack" 测试验证长上下文能力

## 4.3 RLHF (Reinforcement Learning from Human Feedback)

### 4.3.1 完整 Pipeline

([Ouyang et al., 2022 — InstructGPT](https://arxiv.org/abs/2203.02155))

```
Step 1: Reward Model Training
    收集人类偏好数据: (prompt, chosen, rejected)
    训练 RM: RM(chosen) > RM(rejected)

Step 2: PPO Training
    对每个 prompt:
        1. 当前策略 π_θ 生成回答
        2. RM 给回答打分
        3. 用 PPO 优化策略，最大化 reward 同时不偏离 SFT 模型太远
```

### 4.3.2 Reward Model

```python
# RM 架构: 和 LLM 一样的 Transformer，最后一层换成一个 scalar head
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.backbone = base_model
        self.head = nn.Linear(hidden_size, 1)  # 输出标量 reward
    
    def forward(self, input_ids):
        hidden = self.backbone(input_ids).last_hidden_state[:, -1, :]
        return self.head(hidden)  # scalar reward

# Bradley-Terry Loss
def bt_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

**偏好数据收集**:
- 标注员看到同一个 prompt 的两个回答，选择更好的那个
- 可以用 Likert scale (1-7 分) 或 ranking (排序多个回答)
- 每个 prompt 通常需要 3-5 个标注员，取多数

**RM 的问题**:
- **Reward hacking**: 模型学会讨好 RM 而不是真正变好（如总是说长回答、用连接词）
- **Over-optimization**: 优化 RM score 过头后性能反而下降 ([Gao et al., 2023](https://arxiv.org/abs/2210.10760))
- **Distribution shift**: RM 在 SFT 模型的输出上训练，对 PPO 后的输出可能不准

### 4.3.3 PPO (Proximal Policy Optimization)

([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))

```python
# PPO objective:
L = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

# r_t = π_θ(a|s) / π_θ_old(a|s)  # importance sampling ratio
# A_t = advantage estimate (reward - baseline)
# ε = 0.2 typically (clipping range)

# KL penalty 防止模型偏离 SFT baseline 太远:
reward_total = reward_rm - β * KL(π_θ || π_ref)
# π_ref = SFT model (frozen)
# β = 0.01-0.1
```

**PPO 的工程挑战**:
- 需要同时跑 4 个模型: policy (训练中), reference policy (frozen), reward model, value model
- 显存需求极大（70B 模型需要 ~512 张 A100）
- 训练不稳定，超参敏感
- 大量工程优化: 异步生成、vLLM 加速推理、critic model 和 policy model 共享

### 4.3.4 实际优化

**REINFORCE Leave-One-Out (RLOO)** ([Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)):
- 替代 PPO 的更简单方法
- 对每个 prompt 生成 K 个回答，用 leave-one-out baseline 估计 advantage
- 不需要 value model，内存减半
- DeepSeek、LLaMA 3 实际用的是 RLOO 变体

**GRPO (Group Relative Policy Optimization)** ([Shao et al., 2024 — DeepSeekMath](https://arxiv.org/abs/2402.03300)):
- DeepSeek-R1 使用
- 对每个 prompt 采样一组回答
- 用组内相对排序作为 reward signal
- 不需要外部 reward model

## 4.4 DPO (Direct Preference Optimization)

### 4.4.1 核心思想

([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))

**问题**: RLHF 太复杂——需要训练 RM，需要 PPO，需要 4 个模型。

**DPO 的洞察**: 可以把 RM + PPO 合并成一个简单的 supervised loss。

```python
# DPO loss:
L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) 
                     - log π_θ(y_l|x)/π_ref(y_l|x)))]

# y_w = chosen (winning) response
# y_l = rejected (losing) response
# π_ref = reference policy (SFT model, frozen)
# β = temperature (通常 0.1-0.5)
```

**直觉**: 增大 chosen 的概率，降低 rejected 的概率，幅度由 β 控制。reference model 防止偏移太远。

### 4.4.2 DPO 变体

| 变体 | 改进 | 论文 |
|------|------|------|
| **IPO** | 防止 overfitting 到偏好数据 | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) |
| **KTO** | 只需要好/坏标签，不需要 pair | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) |
| **ORPO** | 不需要 reference model | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) |
| **SimPO** | 用序列平均 log-prob 作为 reward | [Meng et al., 2024](https://arxiv.org/abs/2405.14734) |

### 4.4.3 Online DPO vs Offline DPO

**Offline DPO**: 用 SFT 模型生成的偏好数据训练 → 简单但效果有上限

**Online/Iterative DPO** ([Xu et al., 2024](https://arxiv.org/abs/2404.07503)): 
```
循环:
    1. 用当前策略 π_θ 生成回答
    2. 用 RM (或人类/强模型) 标注偏好
    3. 用 DPO loss 更新 π_θ
```
- 效果接近 RLHF
- 解决了 offline DPO 的 distribution shift 问题

## 4.5 Constitutional AI (CAI)

([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) Anthropic 提出的方法，用一组"宪法规则"进行自我对齐:

```
Step 1: Red-teaming
    让模型生成有害回答

Step 2: Critique & Revision
    模型根据宪法规则自我批评并修改回答
    规则如: "Choose the response that is least harmful"

Step 3: RL from AI Feedback (RLAIF)
    用修改后的数据训练 RM
    用 RM + PPO 训练
```

## 4.6 RLVR (Reinforcement Learning with Verifiable Rewards)

**[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 的关键创新**: 用可验证的 reward (如数学答案正确性、代码执行结果) 做 RL，不需要人类偏好数据。

```
对于数学题:
    1. 模型生成 chain-of-thought + 最终答案
    2. 检查答案是否正确 → reward = 1 or 0
    3. 用 GRPO 优化

对于代码题:
    1. 模型生成代码
    2. 跑测试用例 → reward = pass rate
    3. 用 GRPO 优化
```

**惊人发现**: 纯 RL (不用 SFT) 就能让模型涌现 chain-of-thought 推理能力。DeepSeek-R1-Zero 在没有任何 SFT 的情况下，仅通过 RLVR 就学会了长链推理。

## 4.7 Reasoning Models

### 4.7.1 Chain-of-Thought Training

**OpenAI o1/o3** 和 **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)** 代表了一种新范式:

```
传统: prompt → answer
推理: prompt → <think>长链推理过程</think> → answer
```

训练方法:
1. **Process Reward Model (PRM)** ([Lightman et al., 2023](https://arxiv.org/abs/2305.20050)): 对每一步推理打分，而非只看最终答案 (Outcome Reward Model, ORM)
2. **Monte Carlo Tree Search (MCTS)**: 在推理树上搜索最优路径
3. **RLVR**: 用可验证 reward 训练长链推理

### 4.7.2 Test-Time Compute Scaling

核心洞察：与其用更大的模型，不如让模型在推理时"思考"更长时间。([Snell et al., 2024](https://arxiv.org/abs/2408.03314))

```
传统 scaling: 增大模型 → 更好的结果
新 scaling:   增大推理时的 compute → 更好的结果

具体方法:
- Best-of-N: 生成 N 个回答，选最好的
- Majority voting: 生成 N 个回答，投票
- Chain-of-thought: 让模型生成长推理链
- Tree search: 在推理空间中搜索
- Iterative refinement: 让模型反复改进回答
```

## 4.8 Tool Use & Agent Training

### 4.8.1 Function Calling

训练模型输出结构化的工具调用:

```json
{"name": "search", "arguments": {"query": "weather in Stockholm"}}
```

**训练数据**: 收集 (prompt, tool_call, tool_result, final_answer) 的 trajectory

### 4.8.2 Code Execution

```
User: What is 7^23?
Model: <code>print(7**23)</code>
System: [Execution result: 27368747340080916343]
Model: 7^23 = 27368747340080916343
```

### 4.8.3 Multi-step Agent

训练模型执行多步任务:

```
Observation → Thought → Action → Observation → Thought → Action → ... → Answer
```

**[SWE-Agent](https://github.com/princeton-nlp/SWE-agent)/[SWE-bench](https://www.swebench.com/)** 风格训练:
- 给模型一个 GitHub issue
- 模型读代码、编辑文件、跑测试
- reward = 测试通过

---

# 第五章：参数高效微调 (PEFT)

全量微调一个 70B 模型需要 ~1TB 显存（参数 + 优化器状态 + 梯度 + 激活）。PEFT 方法只训练一小部分参数，大幅降低资源需求。

## 5.1 LoRA (Low-Rank Adaptation)

([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) — **目前最主流的 PEFT 方法。**

### 5.1.1 核心思想

预训练权重冻结，旁路加入低秩分解的可训练矩阵：

```
# 原始: Y = X @ W       (W ∈ ℝ^{d×d})
# LoRA: Y = X @ W + X @ A @ B
#       A ∈ ℝ^{d×r}, B ∈ ℝ^{r×d}, r << d

# 参数量对比 (d=4096, r=16):
# 全量: 4096 × 4096 = 16.7M
# LoRA: 4096 × 16 + 16 × 4096 = 131K (减少 128 倍!)
```

### 5.1.2 关键超参

| 超参 | 常用值 | 说明 |
|------|--------|------|
| `r` (rank) | 8-64 | 越大越接近全量微调，但参数越多 |
| `alpha` | 16-64 | 缩放因子。实际 scaling = alpha/r |
| `target_modules` | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj | 对哪些层加 LoRA |
| `dropout` | 0.05-0.1 | LoRA 层的 dropout |

**经验法则**:
- `r=16, alpha=32` 是很好的起点
- 对 attention 层 (q, k, v, o) + FFN 层 (gate, up, down) 都加 LoRA 效果最好
- rank 太小会欠拟合，太大浪费资源且可能过拟合

### 5.1.3 代码实现

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.52%
```

> 库: [huggingface/peft](https://github.com/huggingface/peft)

### 5.1.4 LoRA 合并

训练完后可以把 LoRA 权重合并回原模型，推理时零额外开销：

```python
merged_model = model.merge_and_unload()
# W_new = W + A @ B
# 推理时和全量微调的模型完全相同
```

## 5.2 QLoRA

([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) — **让单卡 24GB GPU 也能微调 70B 模型。**

### 5.2.1 核心创新

在 LoRA 基础上加入三个内存优化：

1. **4-bit NormalFloat (NF4) 量化**: 把 frozen 参数量化到 4-bit
2. **Double Quantization**: 量化 scaling factor 本身（再省内存）
3. **Paged Optimizers**: 用 CPU 内存作为 GPU 内存的交换空间

```
内存对比 (LLaMA 65B):
全量微调: ~780GB (需要多节点)
LoRA FP16: ~130GB (2× A100 80GB)
QLoRA 4-bit: ~48GB (1× A100 80GB!)
```

### 5.2.2 使用

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
)

# 然后像普通 LoRA 一样加 adapter
model = get_peft_model(model, lora_config)
```

## 5.3 其他 PEFT 方法

| 方法 | 原理 | 参数量 | 论文 |
|------|------|--------|------|
| **Prefix Tuning** | 在每层 attention 前加可训练 prefix tokens | 0.1% | [Li & Liang, 2021](https://arxiv.org/abs/2101.00190) |
| **Prompt Tuning** | 在输入前加可训练 soft prompts | 0.01% | [Lester et al., 2021](https://arxiv.org/abs/2104.08691) |
| **Adapter** | 在每层 Transformer 后插入小型 MLP | 1-5% | [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751) |
| **IA3** | 学习对 K, V, FFN 的缩放向量 | 0.01% | [Liu et al., 2022](https://arxiv.org/abs/2205.05638) |
| **DoRA** | LoRA + 权重幅度分解 | ~LoRA | [Liu et al., 2024](https://arxiv.org/abs/2402.09353) |

**实际选择**: 绝大多数场景用 LoRA 或 QLoRA 即可。其他方法在特定场景有优势但生态支持不如 LoRA。

## 5.4 PEFT 实战建议

**什么时候用 PEFT vs 全量微调**:
- 数据量 < 100K：PEFT（全量微调容易过拟合）
- 数据量 > 1M + 大预算：全量微调
- 预算有限（单卡/几卡）：QLoRA
- 需要切换多个任务：LoRA（可以同时加载多个 adapter）

**常见工具**:
- [huggingface/trl](https://github.com/huggingface/trl) — SFT, DPO, PPO + LoRA
- [axolotl](https://github.com/axolotl-ai-cloud/axolotl) — 一站式微调（支持 LoRA/QLoRA + 各种格式）
- [unsloth](https://github.com/unslothai/unsloth) — 2x 加速 LoRA 微调（手写 kernel）
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 中文社区常用的微调框架

---

# 第六章：训练基础设施 (Infra)

## 6.1 硬件

### 6.1.1 GPU 选型

| GPU | 显存 | BF16 TFLOPS | FP8 TFLOPS | 互连 | 用途 |
|-----|------|-------------|------------|------|------|
| A100 80GB | 80GB HBM2e | 312 | - | NVLink 600GB/s | 主流训练 (2022-2024) |
| H100 SXM | 80GB HBM3 | 989 | 1979 | NVLink 900GB/s | 当前主力 |
| H200 | 141GB HBM3e | 989 | 1979 | NVLink 900GB/s | 大模型 (更大显存) |
| B200 | 192GB HBM3e | 2250 | 4500 | NVLink 1800GB/s | 下一代主力 |
| MI300X (AMD) | 192GB HBM3 | 1307 | 2614 | Infinity Fabric | AMD 替代方案 |

**关键指标**:
- **显存容量**: 决定能放多大的模型
- **显存带宽**: 决定推理速度（推理是 memory-bound）
- **算力 (TFLOPS)**: 决定训练速度（训练是 compute-bound）
- **互连带宽**: 决定多卡并行效率

### 6.1.2 集群网络

```
单机内:
  GPU ←NVLink 900GB/s→ GPU    (8卡全连接)

跨机:
  Node ←InfiniBand 400Gb/s→ Node
  
  400Gb IB ≈ 50GB/s (每个方向)
  NVLink ≈ 900GB/s
  
  所以跨机通信比机内慢 ~18倍！
  → 并行策略必须最小化跨机通信
```

**大集群拓扑** (如 16K GPU):
```
GPU (8) → Node → Leaf switch (32 nodes) → Spine switch → Fat-tree/Dragonfly
```

**网络问题**: 训练大模型时，一个慢节点或丢包就会拖慢整个训练。需要：
- [NCCL](https://github.com/NVIDIA/nccl) 调优
- 网络拓扑感知的进程放置
- 容错机制（自动检测并替换故障节点）

### 6.1.3 存储

```
训练数据读取:
  → 分布式文件系统 (Lustre, GPFS, WekaFS)
  → 或对象存储 (S3) + 本地 NVMe 缓存

Checkpoint 写入:
  → 70B 模型一个 checkpoint ~500GB
  → 每 1000 步存一次 → 每天几十TB
  → 异步 checkpoint (不阻塞训练)
  → Nebula/分布式 checkpoint 存储
```

## 6.2 分布式训练策略

### 6.2.1 数据并行 (Data Parallelism, DP)

**最简单的并行方式**: 每张卡存完整模型副本，不同卡处理不同数据。

```
GPU 0: Model copy + Data batch 0 → gradient_0
GPU 1: Model copy + Data batch 1 → gradient_1
GPU 2: Model copy + Data batch 2 → gradient_2
GPU 3: Model copy + Data batch 3 → gradient_3
          ↓ AllReduce: average gradients ↓
GPU 0-3: 用平均梯度更新模型
```

**限制**: 每张卡要放整个模型。7B 模型 (BF16) = 14GB 参数 + 优化器状态 ~56GB → 一张 80GB 卡勉强放下。70B 模型放不下。

### 6.2.2 ZeRO (Zero Redundancy Optimizer)

([Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)) **[DeepSpeed](https://github.com/microsoft/DeepSpeed) 的核心贡献**: 数据并行中的冗余太多——每张卡都存完整模型 + 优化器状态。ZeRO 把它们切分。

```
ZeRO Stage 1: 切分优化器状态
  → 内存减少 4倍

ZeRO Stage 2: 切分优化器状态 + 梯度
  → 内存减少 8倍

ZeRO Stage 3: 切分优化器状态 + 梯度 + 模型参数
  → 内存减少 N倍 (N = GPU数)
  → 相当于 FSDP
```

**PyTorch FSDP (Fully Sharded Data Parallel)**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    auto_wrap_policy=transformer_auto_wrap_policy,
)
```

**FSDP2** (PyTorch 最新):
- 更细粒度的 sharding (per-parameter)
- 更好地配合 tensor parallel
- 支持 FP8

### 6.2.3 张量并行 (Tensor Parallelism, TP)

([Shoeybi et al., 2019 — Megatron-LM](https://arxiv.org/abs/1909.08053))

**把单个层的矩阵乘法切分到多张卡上**:

```
# 线性层 Y = X @ W (W ∈ ℝ^{d×d})
# 把 W 按列切分到 2 张卡:

GPU 0: Y_0 = X @ W_0   (W_0 = W[:, :d/2])
GPU 1: Y_1 = X @ W_1   (W_1 = W[:, d/2:])

# AllGather 或 ReduceScatter 合并结果
```

**Megatron-LM 风格 TP**:
- Self-Attention: Q, K, V 按 head 切分
- FFN: 第一个线性层按列切，第二个按行切
- 每层需要 2 次 AllReduce

**适用场景**: 机内（NVLink 带宽足够），通常 TP=8（一台机器内）

### 6.2.4 流水线并行 (Pipeline Parallelism, PP)

([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473))

**把模型按层切分到不同机器**:

```
GPU 0: Layers 0-7    (Stage 0)
GPU 1: Layers 8-15   (Stage 1)
GPU 2: Layers 16-23  (Stage 2)
GPU 3: Layers 24-31  (Stage 3)

数据从 Stage 0 → 1 → 2 → 3 流水线式处理
```

**问题**: 朴素 PP 有大量"bubble"（GPU 空闲等待）。

**解决方案**:
- **GPipe** ([Huang et al., 2019](https://arxiv.org/abs/1811.06965)): 把 micro-batch 切成更小的 mini-batch，增加流水线并行度
- **1F1B** (one-forward-one-backward): 前向和反向交替调度，减少 bubble
  ```
  Stage 0: F0 F1 F2 F3 B0 B1 B2 B3   (朴素, 大bubble)
  Stage 0: F0 F1 F2 F3 B0 F4 B1 F5   (1F1B, 小bubble)
  ```
- **Interleaved PP**: 每张卡放非连续的层（如 GPU 0 放 layer 0,8,16,24），减少 bubble
- **Zero-bubble PP** ([Qi et al., 2024](https://arxiv.org/abs/2401.10241)): 通过重新调度把 bubble 降到接近 0

### 6.2.5 Context Parallelism (CP)

**长序列并行**: 把序列切分到多张卡，每张卡处理一部分序列。

```
序列长度 128K, 4张卡:
GPU 0: tokens 0-32K
GPU 1: tokens 32K-64K
GPU 2: tokens 64K-96K
GPU 3: tokens 96K-128K

Attention 通过 Ring Attention 计算:
- 每张卡本地计算 QK^T 的一部分
- KV 通过 ring 传递到下一张卡
- 重复直到所有 KV 都遍历过
```

### 6.2.6 Expert Parallelism (EP)

**MoE 模型专用**: 不同 expert 放在不同 GPU 上。

```
GPU 0: Expert 0, 1
GPU 1: Expert 2, 3
GPU 2: Expert 4, 5
GPU 3: Expert 6, 7

Token routing:
1. Gate 计算每个 token 要去哪个 expert → All-to-All
2. 每张卡计算自己的 expert → All-to-All
3. 结果送回原来的卡
```

**通信瓶颈**: All-to-All 通信量和 token 数 × hidden_size 成正比。

### 6.2.7 3D/4D/5D 并行组合

实际大模型训练组合多种并行:

```
DeepSeek-V3 (2048 H800 GPUs):
  TP=1 (不用TP，因为用了MLA和MoE)
  PP=16 (16个pipeline stage)
  DP=128 (128路数据并行)
  EP=64 (64路expert并行)
  
  2048 = 16 × 128 = 16 × 2 × 64

LLaMA 3 405B (16384 H100 GPUs):
  TP=8 (机内)
  PP=16 (跨机)
  DP=128 (数据并行)
  CP 用于长上下文训练
  
  16384 = 8 × 16 × 128
```

## 6.3 训练框架

### 6.3.1 框架对比

| 框架 | 公司 | 并行策略 | 适用规模 |
|------|------|---------|---------|
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA | TP+PP+DP, MoE | 百B-万B |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft | ZeRO, PP, MoE | 十B-千B |
| [FSDP](https://pytorch.org/docs/stable/fsdp.html) (PyTorch) | Meta | ZeRO-3 | 十B-百B |
| [ColossalAI](https://github.com/hpcaitech/ColossalAI) | HPC-AI Tech | 多种 | 十B-百B |
| [Levanter](https://github.com/stanford-crfm/levanter) | Stanford | JAX-based | 研究 |
| [NanoGPT](https://github.com/karpathy/nanoGPT) | Karpathy | DDP | 学习/小规模 |

### 6.3.2 Megatron-LM 核心

```python
# Megatron-LM 的 3D 并行配置
# launch: torchrun --nproc_per_node=8 --nnodes=64

args = {
    "tensor_model_parallel_size": 8,     # TP: 机内
    "pipeline_model_parallel_size": 16,   # PP: 跨机
    "data_parallel_size": 32,             # DP: auto = total / TP / PP
    "sequence_parallel": True,            # 序列并行 (和 TP 搭配)
    "use_flash_attn": True,
    "bf16": True,
    "micro_batch_size": 1,
    "global_batch_size": 1024,
}
```

### 6.3.3 高效训练技巧

**Gradient Checkpointing (Activation Recomputation)** ([Chen et al., 2016](https://arxiv.org/abs/1604.06174)):
```
正常: 前向保存所有中间激活 → 反向使用
问题: 激活占内存太多 (正比于 batch_size × seq_len × hidden × layers)

Gradient Checkpointing: 只保存部分层的激活，反向时重算
  → 内存减少 √L 倍 (L = 层数)
  → 计算增加 ~33%

选择性 checkpointing: 只 checkpoint 占内存大的操作 (attention)
```

**Flash Attention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)):
```
标准 Attention:
  S = Q @ K^T          → O(n²d) compute, O(n²) memory
  P = softmax(S)       → 存 n² 矩阵
  O = P @ V            

Flash Attention (Tri Dao):
  不 materialize n² attention matrix
  用 tiling + online softmax 在 SRAM 中分块计算
  → 内存 O(n) 而非 O(n²)
  → 速度快 2-4x (减少 HBM 读写)
  
  Flash Attention 2: 更好的并行化
  Flash Attention 3: H100 优化, FP8 支持
```

> 代码: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

**Compiled/Fused Kernels**:
```python
# torch.compile 自动 fuse 操作
model = torch.compile(model, mode="max-autotune")

# 手动 fuse 的关键 kernel:
# - Fused Attention (FlashAttention)
# - Fused LayerNorm + Dropout
# - Fused SwiGLU
# - Fused RoPE
# - Fused Cross-Entropy (lm_head + CE loss)
```

**Communication-Computation Overlap**:
```
不 overlap:
  Forward → AllReduce → Forward → AllReduce ...

Overlap:
  Forward_layer_N → [AllReduce_layer_{N-1} 同时进行] → ...
  
  异步启动通信，计算的同时传输梯度
```

## 6.4 容错与效率

### 6.4.1 Checkpoint 策略

```python
# 异步 checkpoint (不阻塞训练)
def async_save_checkpoint(model, optimizer, step):
    # 把 state_dict copy 到 CPU (非阻塞)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    # 在后台线程写入存储
    threading.Thread(target=torch.save, args=(state, f"ckpt_{step}.pt")).start()

# 分布式 checkpoint (每张卡只存自己的 shard)
from torch.distributed.checkpoint import save, load
save(model.state_dict(), storage_writer=FileSystemWriter(path))
```

### 6.4.2 故障恢复

大集群训练中故障是常态:
- 16K GPU 训练，平均每 2-3 小时一次硬件故障
- GPU 内存错误、网络抖动、节点宕机

**DeepSeek-V3 的解决方案**:
- 每 10 分钟快速 checkpoint 到本地 NVMe (ramdisk)
- 故障时从最近的快速 checkpoint 恢复
- 坏节点自动被替换节点顶替
- 训练仅丢失 10 分钟内的进度

### 6.4.3 MFU (Model FLOPS Utilization)

**衡量训练效率的核心指标**:
```
MFU = 实际计算量 / (理论峰值算力 × 训练时间)

好的 MFU:
  单机: 50-60%
  小集群 (64-256 GPU): 40-50%
  大集群 (1K+ GPU): 30-45%
  超大集群 (16K GPU): 35-40% (LLaMA 3 报告 38-43%)
```

**影响 MFU 的因素**:
- 通信开销（并行策略选择）
- Bubble 比例（PP 调度）
- 数据加载延迟
- Kernel 效率
- Gradient checkpointing 的重计算开销

## 6.5 推理优化 (Inference)

### 6.5.1 KV Cache

```
自回归生成时，每个新 token 需要和所有前文做 attention
朴素: 每次重算所有 K, V → O(n²) per token

KV Cache: 缓存历史 K, V，每次只计算新 token 的 K, V
→ O(n) per token
→ 但缓存占内存: 2 * layers * kv_heads * head_dim * seq_len * bytes

70B 模型, 128K context, BF16:
KV cache ≈ 2 * 80 * 8 * 128 * 128K * 2 ≈ 42GB per request!
```

**KV Cache 优化**:
- **GQA**: KV heads 减少 → cache 减少 (见 2.3.1)
- **MLA**: 压缩到 latent → cache 大幅减少 (见 2.3.3)
- **PagedAttention** ([vLLM](https://github.com/vllm-project/vllm)): 用虚拟内存分页管理 KV cache，消除碎片
- **KV Cache Quantization**: 把 KV cache 量化到 INT8/FP8

### 6.5.2 量化推理

| 方法 | 精度 | 速度提升 | 质量损失 |
|------|------|---------|---------|
| BF16 | 16-bit | 1x (baseline) | 无 |
| INT8 (W8A8) | 8-bit | ~2x | 极小 |
| FP8 | 8-bit | ~2x | 极小 |
| INT4 ([GPTQ](https://arxiv.org/abs/2210.17323)/[AWQ](https://arxiv.org/abs/2306.00978)) | 4-bit | ~3-4x | 小 |
| GGUF Q4_K_M | 4-bit mixed | ~3-4x | 小 |
| INT2-3 | 2-3 bit | ~5x+ | 明显 |

**GPTQ**: Post-training quantization，用校准数据最小化量化误差
**AWQ**: Activation-aware Weight Quantization，保护 salient weights
**GGUF**: [llama.cpp](https://github.com/ggerganov/llama.cpp) 格式，CPU/GPU 混合推理

### 6.5.3 推理框架

| 框架 | 特点 | 适用 |
|------|------|------|
| [**vLLM**](https://github.com/vllm-project/vllm) | PagedAttention, continuous batching | 生产部署首选 |
| [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 优化, FP8 | 最大吞吐 |
| [**SGLang**](https://github.com/sgl-project/sglang) | RadixAttention, prefix caching | 复杂 prompt |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp) | CPU/Apple Silicon, GGUF | 本地推理 |
| [**MLC-LLM**](https://github.com/mlc-ai/mlc-llm) | 编译优化, 多平台 | 端侧 |

### 6.5.4 Speculative Decoding

([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192))

```
问题: LLM 自回归生成是 memory-bound，GPU 算力利用率很低

方案: 用小模型 (draft) 快速生成 K 个 token，大模型 (target) 并行验证

Draft model: 快速生成 t1, t2, t3, t4, t5  (5 tokens)
Target model: 并行验证 → 接受 t1, t2, t3, 拒绝 t4 → 从 t3 后重新生成

效果: 大模型一次前向验证 K 个 token，实际输出 ~K/2 个 token
     速度提升 2-3x，输出质量和大模型完全相同
```

**[EAGLE](https://arxiv.org/abs/2401.15077)** (高级 speculative decoding):
- Draft model 用 target model 的中间层特征
- 比独立 draft model 更准确
- 速度提升 3x+

---

# 第七章：多模态 (Multimodal)

## 7.1 Vision-Language Models (VLMs)

### 7.1.1 架构模式

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

### 7.1.2 Vision Encoder

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

### 7.1.3 训练流程

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

### 7.1.4 SOTA VLMs

| 模型 | 参数量 | Vision Encoder | 特点 |
|------|--------|---------------|------|
| GPT-4o | ? | 原生多模态 | 最强综合性能 |
| Claude 3.5 Sonnet | ? | Cross-attention | 强文档理解 |
| Gemini 1.5 Pro | ? | 原生多模态 | 超长上下文 (1M+) |
| [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) | 7B/72B | SigLIP | 开源 SOTA |
| [InternVL 2.5](https://arxiv.org/abs/2412.05819) | 78B | InternViT-6B | 开源 SOTA |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | 72B | ViT-600M | 强 OCR/文档 |

## 7.2 Image Generation

### 7.2.1 Diffusion Models

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

### 7.2.2 Text-to-Image

| 模型 | 架构 | 特点 |
|------|------|------|
| [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) | DiT (Diffusion Transformer) | 开源, MMDiT |
| DALL-E 3 | Diffusion + Caption rewriting | 强 prompt 遵循 |
| Midjourney v6 | ? | 最佳美学 |
| [FLUX](https://github.com/black-forest-labs/flux) | Rectified Flow Transformer | 新 SOTA 开源 |
| Imagen 3 | Cascaded Diffusion | Google |

**DiT** ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)): 用 Transformer 替代 U-Net 作为 diffusion 的 backbone。FLUX, SD3 都基于 DiT。

### 7.2.3 Autoregressive Image Generation

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

## 7.3 Audio & Speech

### 7.3.1 Speech-to-Text (ASR)

**[Whisper](https://arxiv.org/abs/2212.04356) (OpenAI)**:
```
Audio → Mel Spectrogram → Audio Encoder (Transformer)
→ Cross-Attention with Text Decoder → Transcription

训练: 680K 小时多语言标注音频
特点: 98种语言, 极强鲁棒性
```

> 代码: [openai/whisper](https://github.com/openai/whisper)

### 7.3.2 Text-to-Speech (TTS)

| 模型 | 方法 | 特点 |
|------|------|------|
| [VALL-E](https://arxiv.org/abs/2301.02111) | AR codec tokens | 3秒 voice clone |
| [Bark](https://github.com/suno-ai/bark) | AR + Diffusion | 多语言, 开源 |
| [F5-TTS](https://arxiv.org/abs/2410.06885) | Flow matching | 快速, 高质量 |
| GPT-4o | 端到端多模态 | 最自然 |

### 7.3.3 Audio Understanding

**Unified Audio-Language Models**:
```
Audio → Audio Encoder (Whisper encoder 或 HuBERT)
→ Projector → LLM (和文本一起处理)

能做: 语音理解、音乐分析、环境声音识别
代表: SALMONN, Qwen-Audio, Gemini
```

## 7.4 Video

### 7.4.1 Video Understanding

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

### 7.4.2 Video Generation

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

## 7.5 Omni Models (全模态)

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

# 第八章：评估与对齐 (Evaluation & Alignment)

## 8.1 预训练评估

### 8.1.1 标准 Benchmark

| Benchmark | 评测能力 | 数据量 | 指标 |
|-----------|---------|--------|------|
| [MMLU](https://arxiv.org/abs/2009.03300) | 知识 (57科目) | 15K | Accuracy |
| [HellaSwag](https://arxiv.org/abs/1905.07830) | 常识推理 | 10K | Accuracy |
| [ARC-Challenge](https://arxiv.org/abs/1803.05457) | 科学问答 | 1.2K | Accuracy |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | 指代消解 | 1.7K | Accuracy |
| [GSM8K](https://arxiv.org/abs/2110.14168) | 小学数学 | 1.3K | Accuracy |
| [MATH](https://arxiv.org/abs/2103.03874) | 竞赛数学 | 5K | Accuracy |
| [HumanEval](https://arxiv.org/abs/2107.03374) | 代码 (Python) | 164 | pass@1 |
| [MBPP](https://arxiv.org/abs/2108.07732) | 代码 (Python) | 974 | pass@1 |
| TriviaQA | 事实问答 | 95K | F1/EM |

### 8.1.2 进阶 Benchmark

| Benchmark | 评测能力 | 特点 |
|-----------|---------|------|
| [**MMLU-Pro**](https://arxiv.org/abs/2406.01574) | 更难的知识测试 | 10选项 + 推理题 |
| [**GPQA**](https://arxiv.org/abs/2311.12022) | PhD级科学问答 | 领域专家出题 |
| **MATH-500** | 数学推理 | 更多样 |
| [**LiveCodeBench**](https://livecodebench.github.io/) | 代码 (持续更新) | 防数据泄露 |
| [**SWE-bench**](https://www.swebench.com/) | 软件工程 | 修真实 GitHub issue |
| **AIME 2024/2025** | 数学竞赛 | 最难的数学评测 |
| [**Codeforces**](https://codeforces.com/) | 竞赛编程 | ELO rating |
| [**IFEval**](https://arxiv.org/abs/2311.07911) | 指令遵循 | 格式、约束 |
| [**Arena-Hard**](https://github.com/lm-sys/arena-hard-auto) | 综合对话 | 模拟人类偏好 |

### 8.1.3 多模态评估

| Benchmark | 模态 | 能力 |
|-----------|------|------|
| [MMMU](https://arxiv.org/abs/2311.16502) | 图像+文本 | 多学科视觉问答 |
| [MathVista](https://arxiv.org/abs/2310.02255) | 图像+文本 | 数学视觉推理 |
| [DocVQA](https://arxiv.org/abs/2007.00398) | 文档图像 | 文档理解 |
| ChartQA | 图表 | 图表理解 |
| OCRBench | 图像 | OCR 能力 |
| [VideoMME](https://arxiv.org/abs/2405.21075) | 视频+文本 | 视频理解 |

> 评估框架: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 统一跑各种 benchmark

## 8.2 人类评估

### 8.2.1 Chatbot Arena (LMSYS)

**最权威的 LLM 排名**: 真实用户盲测两个模型，选更好的。

> [https://chat.lmsys.org/](https://chat.lmsys.org/) | [排行榜](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

```
用户提问 → 模型 A 和模型 B 分别回答（匿名）→ 用户选哪个更好
→ 用 Bradley-Terry 模型计算 ELO rating
```

### 8.2.2 Red Teaming

系统性地测试模型的安全边界:
- Jailbreak 测试
- 有害内容生成
- 隐私泄露
- 偏见检测

## 8.3 Alignment 技术总结

```
Level 0: 预训练数据过滤 (去有害内容)
    ↓
Level 1: SFT (学会有帮助的格式)
    ↓
Level 2: RLHF/DPO (学会人类偏好)
    ↓
Level 3: Safety training (学会拒绝)
    ↓
Level 4: Constitutional AI (原则性自我对齐)
    ↓
Level 5: Scalable oversight (可扩展的监督)
    - Debate: 两个模型辩论，人类裁判
    - Recursive reward modeling: 用 AI 辅助人类标注
    - Weak-to-strong generalization: 弱模型监督强模型
```

---

# 第九章：SOTA 模型深度解析

## 9.1 LLaMA 3 / 3.1 / 3.3 (Meta)

> 论文: [LLaMA 3](https://arxiv.org/abs/2407.21783) | 权重: [meta-llama](https://huggingface.co/meta-llama)

```
参数: 8B, 70B, 405B
架构: Dense Transformer, GQA, RoPE, SwiGLU, RMSNorm
词表: 128K (扩大后的 tiktoken BPE)
上下文: 128K (通过 YaRN 扩展)
训练数据: 15T tokens
训练硬件: 16384 H100 GPUs
训练时间: ~54天 (405B)

关键创新:
- 超大词表 (128K) 大幅改善多语言和代码
- Over-training: 8B 模型用 15T tokens (远超 Chinchilla 最优)
- 3阶段 Post-training: SFT → Rejection Sampling → DPO
- 代码执行反馈: 用代码执行结果作为 reward
```

## 9.2 DeepSeek-V3 / R1

> 论文: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 权重: [deepseek-ai](https://huggingface.co/deepseek-ai)

```
DeepSeek-V3:
  参数: 671B (MoE, 37B 激活)
  架构: MLA + DeepSeekMoE (256 experts, top-8 + 1 shared)
  训练: 14.8T tokens, 2048 H800 GPUs
  成本: ~$5.5M (极低!)
  
  关键创新:
  - MLA: KV cache 压缩 93%
  - FP8 混合精度训练 (首次大规模成功)
  - Auxiliary-loss-free load balancing
  - Multi-Token Prediction (预测未来多个 token)

DeepSeek-R1:
  基于 V3，加了推理能力
  
  关键创新:
  - R1-Zero: 纯 RL (GRPO + 可验证 reward) → 涌现 CoT
  - R1: SFT (cold start) → RL → Rejection Sampling → SFT
  - 在 AIME 2024 上接近 OpenAI o1 水平
  - 完全开源 (权重 + 论文)
```

## 9.3 Claude 3/4 (Anthropic)

> 文档: [anthropic.com/claude](https://www.anthropic.com/claude)

```
架构: 未公开 (推测 dense transformer, cross-attention multimodal)
系列: Haiku (小) → Sonnet (中) → Opus (大)

关键特点:
- Constitutional AI: 基于原则的自我对齐
- 超长上下文: 200K tokens
- 强文档/代码理解
- 安全性领先
- Claude 4: 超强 coding, agentic 能力
```

## 9.4 Gemini 2 (Google)

> 论文: [Gemini](https://arxiv.org/abs/2312.11805) | [Gemini 1.5](https://arxiv.org/abs/2403.05530)

```
架构: 原生多模态 Transformer (MoE)
训练: TPU v5p/v6e 集群

关键特点:
- 原生多模态 (文本, 图像, 音频, 视频 一起预训练)
- 超长上下文: Gemini 1.5 支持 2M tokens
- Project Astra: 实时视频 + 音频理解
- Gemini 2 Flash: 极快推理
```

## 9.5 GPT-4 / o1 / o3 (OpenAI)

> 文档: [platform.openai.com](https://platform.openai.com/docs)

```
GPT-4:
  参数: ~1.8T (MoE, 8 experts, 推测)
  训练: ~13T tokens on ~25000 A100s

o1/o3 (reasoning):
  关键创新:
  - 大规模 RL 训练 (推测用了 PRM + MCTS)
  - Test-time compute scaling
  - Hidden chain-of-thought (用户不可见)
  - o3 在 ARC-AGI 上达到 87.5%
```

## 9.6 Qwen 2.5 / QwQ (Alibaba)

> 论文: [Qwen2](https://arxiv.org/abs/2407.10671) | 权重: [Qwen](https://huggingface.co/Qwen)

```
参数: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B + MoE
架构: GQA, RoPE, SwiGLU
词表: 152K
训练: 18T tokens

关键特点:
- 最全的开源模型家族 (覆盖所有规模)
- 强多语言 (中英日韩等)
- Qwen2-VL: 强视觉理解
- QwQ: 推理模型 (类 o1)
- Qwen-Agent: 工具使用框架
```

## 9.7 Mistral / Mixtral

> 论文: [Mistral 7B](https://arxiv.org/abs/2310.06825) | [Mixtral](https://arxiv.org/abs/2401.04088) | 权重: [mistralai](https://huggingface.co/mistralai)

```
Mistral 7B:
  创新: Sliding Window Attention, GQA
  效果: 7B 超过 LLaMA 2 13B

Mixtral 8x7B:
  创新: 开源首个 MoE LLM
  47B 参数, 13B 激活
  效果: 接近 GPT-3.5

Mistral Large 2:
  123B dense model
  强代码和多语言
```

## 9.8 小模型 (Small Language Models)

| 模型 | 参数量 | 训练数据 | 特点 |
|------|--------|---------|------|
| [Phi-3/3.5](https://arxiv.org/abs/2404.14219) | 3.8B/14B | Synthetic重 | "教科书级"数据质量 |
| [Gemma 2](https://arxiv.org/abs/2408.00118) | 2B/9B/27B | 大规模过训练 | Google 开源 |
| [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 135M-1.7B | 高质量 web data | HuggingFace |
| [TinyLlama](https://github.com/jzhang38/TinyLlama) | 1.1B | 3T tokens | 极度过训练 |

**趋势**: 小模型 + 高质量数据 + 过训练 → 性价比极高。端侧部署 (手机、PC) 是重要方向。

---

# 第十章：知识蒸馏与模型合并

## 10.1 知识蒸馏 (Knowledge Distillation)

([Hinton et al., 2015](https://arxiv.org/abs/1503.02531))

**核心思想**: 用大模型 (teacher) 的 soft labels 训练小模型 (student)。

```python
# 标准 KD Loss
L = α * CE(student_logits, hard_labels) + (1-α) * KL(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T²

# T = temperature (通常 2-20)，让 soft labels 更"软"
# α = hard/soft label 的权重平衡
```

### LLM 蒸馏方法

| 方法 | 说明 | 例子 |
|------|------|------|
| **Logit distillation** | 学 teacher 的 output distribution | 经典 KD |
| **On-policy distillation** | student 生成 → teacher 打分 → 训练 student | GKD ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)) |
| **Synthetic data** | teacher 生成回答，student 当 SFT 数据 | Alpaca, Vicuna |
| **Reasoning distillation** | teacher 生成 CoT → student 学 CoT | DeepSeek-R1 蒸馏版 |

**DeepSeek-R1 蒸馏**: 用 R1 (671B) 的 reasoning traces 蒸馏出 1.5B-70B 的小模型，效果惊人地好。

## 10.2 模型合并 (Model Merging)

不额外训练，直接在权重空间合并多个模型。

### 10.2.1 方法

| 方法 | 原理 | 论文 |
|------|------|------|
| **Linear** | `W = α * W_A + (1-α) * W_B` | - |
| **SLERP** | 球面线性插值 | - |
| **TIES** | 修剪冲突参数后合并 | [Yadav et al., 2023](https://arxiv.org/abs/2306.01708) |
| **DARE** | 随机 drop 并 rescale delta weights | [Yu et al., 2024](https://arxiv.org/abs/2311.03099) |
| **Model Soups** | 平均多个 fine-tune checkpoint | [Wortsman et al., 2022](https://arxiv.org/abs/2203.05482) |

### 10.2.2 工具

> [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) — 最流行的模型合并工具

```yaml
# mergekit 配置示例 (YAML)
models:
  - model: base_model
    parameters:
      weight: 0.5
  - model: math_model
    parameters:
      weight: 0.3
  - model: code_model
    parameters:
      weight: 0.2
merge_method: linear
dtype: bfloat16
```

**实际用途**: 合并一个通用模型 + 一个数学模型 + 一个代码模型 → 得到一个三者兼备的模型，不需要额外训练。[Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) 上很多顶级模型是合并得到的。

---

# 第十一章：实战路线图

## 11.1 学习路径

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
  - 推荐: nanoGPT, litGPT, minGPT

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

□ 长上下文
  - 理解 RoPE 扩展
  - 实现 Ring Attention (概念)
  - 训练支持 32K+ context 的模型

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

## 11.2 推荐资源

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

**Infra**:
12. [Megatron-LM](https://arxiv.org/abs/1909.08053) (Shoeybi 2019)
13. [ZeRO: Memory Optimizations](https://arxiv.org/abs/1910.02054) (Rajbhandari 2020)
14. [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao 2022)
15. [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao 2023)

**多模态**:
16. [Learning Transferable Visual Models (CLIP)](https://arxiv.org/abs/2103.00020) (Radford 2021)
17. [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) (Liu 2023)
18. [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) (Google 2023)

**PEFT**:
19. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu 2021)
20. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers 2023)

### 代码项目

**入门**:
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — 从头训练 GPT-2
- [karpathy/llm.c](https://github.com/karpathy/llm.c) — C语言实现 GPT-2 训练
- [karpathy/minbpe](https://github.com/karpathy/minbpe) — 最小化 BPE 实现

**进阶**:
- [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) — 生产级 LLM 训练
- [pytorch/torchtune](https://github.com/pytorch/torchtune) — PyTorch 原生微调框架
- [huggingface/trl](https://github.com/huggingface/trl) — SFT, DPO, PPO, RLOO
- [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) — 一站式微调
- [unslothai/unsloth](https://github.com/unslothai/unsloth) — 2x 加速 LoRA 微调

**Infra**:
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — 大规模预训练
- [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) — 分布式训练
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — 高效推理
- [sgl-project/sglang](https://github.com/sgl-project/sglang) — 推理 + serving

**多模态**:
- [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) — 视觉语言模型
- [huggingface/transformers](https://github.com/huggingface/transformers) — 统一框架
- [openai/whisper](https://github.com/openai/whisper) — 语音识别

**数据处理**:
- [huggingface/datatrove](https://github.com/huggingface/datatrove) — 数据处理 pipeline
- [allenai/dolma](https://github.com/allenai/dolma) — 数据 toolkit
- [bigcode-project/bigcode-dataset](https://github.com/bigcode-project/bigcode-dataset) — 代码数据处理

**评估**:
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 统一评估框架
- [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) — 软件工程评估
- [openai/human-eval](https://github.com/openai/human-eval) — 代码评估

### 课程

- [Andrej Karpathy: Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (YouTube, 免费)
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) (课件/视频免费)
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/) (2024, 课件免费)
- [CMU 11-868: Large Language Models](https://llms-11-868.github.io/) (2024, 课件免费)
- [Maxime Labonne: LLM Course](https://github.com/mlabonne/llm-course) (GitHub, 免费)
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) (视频免费)

## 11.3 硬件建议

### 个人学习
```
入门: 1x RTX 4090 (24GB) — 够训练 1B 模型, LoRA 微调 7B
进阶: 2x RTX 4090 — 学习多卡训练
或: 云 GPU (Lambda, RunPod, vast.ai) 按需租用
```

### 小团队
```
4-8x A100/H100 — 可以预训练 7B, 全量微调 70B
云方案: AWS p5, GCP a3, Azure ND H100
```

### 大规模
```
256+ GPU — 预训练 70B+
自建或云上长期租约
需要专职 infra 工程师
```

---

# 附录：关键论文清单

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
- [Lepikhin et al. (2021) — GShard](https://arxiv.org/abs/2006.16668)
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
- [Lightman et al. (2023) — Process Reward Models](https://arxiv.org/abs/2305.20050)
- [Snell et al. (2024) — Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314)

## Infra
- [Shoeybi et al. (2019) — Megatron-LM](https://arxiv.org/abs/1909.08053)
- [Rajbhandari et al. (2020) — ZeRO](https://arxiv.org/abs/1910.02054)
- [Narayanan et al. (2021) — Pipeline Parallelism](https://arxiv.org/abs/2104.04473)
- [Kwon et al. (2023) — PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Leviathan et al. (2023) — Speculative Decoding](https://arxiv.org/abs/2211.17192)

---

> 最后更新: 2026-04-25
> 这份教程覆盖了 LLM 训练的核心知识。持续学习最好的方式：读论文 → 跑代码 → 读论文。
