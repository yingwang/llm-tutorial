[← Table of Contents](README.md) | [Next Chapter →](02-architecture.md)

# Chapter 1: Tokenizer

The tokenizer is the entry point of an LLM — it converts raw text into a sequence of token IDs that the model can process. Tokenizer design directly affects vocabulary coverage, sequence length, multilingual capability, and training efficiency.

## 1.1 Why We Need a Tokenizer

Neural networks operate on numerical tensors, not strings. The simplest approach is character-level (one token per character), but this has problems:
- Sequences are too long (a single sentence becomes dozens to hundreds of tokens), making attention's O(n²) cost prohibitive
- No semantic granularity — the model has to learn word boundaries on its own

The other extreme is word-level (one token per word), but:
- The vocabulary explodes (English alone has hundreds of thousands of word forms)
- Cannot handle OOV (out-of-vocabulary) words
- Unfriendly to CJK languages (no whitespace-based word segmentation)

Subword tokenization is the compromise: frequent words are kept as whole tokens, while rare words are split into subword fragments.

## 1.2 Core Algorithms

### 1.2.1 BPE (Byte Pair Encoding)

**Principle**: Start from character (or byte) level, repeatedly merge the most frequent adjacent pair until the vocabulary reaches the target size.

**Training process**:
```
Initial vocabulary: all single characters (or bytes 0-255)
Loop:
    1. Count frequency of all adjacent token pairs
    2. Merge the most frequent pair → new token
    3. Update the token sequences in the corpus
    4. Stop when vocabulary reaches target size
```

**Example**:
```
Corpus: "low lower lowest"
Initial: ['l','o','w',' ','l','o','w','e','r',' ','l','o','w','e','s','t']
Round 1: 'l'+'o' → 'lo' (appears 3 times, highest frequency)
Round 2: 'lo'+'w' → 'low' (appears 3 times)
Round 3: 'low'+'e' → 'lowe' (appears 2 times)
...
```

**GPT series BPE**: OpenAI's [tiktoken](https://github.com/openai/tiktoken) improves on byte-level BPE:
- Based on UTF-8 bytes rather than Unicode characters, natively supporting any language
- Uses regex pre-tokenization (splits text into large chunks before running BPE) to prevent cross-word merges
- GPT-4 uses the `cl100k_base` vocabulary with ~100K tokens

```python
# tiktoken usage
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
print(enc.decode(tokens))  # "Hello, world!"
```

### 1.2.2 WordPiece

**Principle**: Similar to BPE, but with a different merge criterion. BPE selects the most frequent pair; WordPiece selects the pair that maximizes the language model likelihood improvement.

**Formula**: Select merge (x, y) → xy such that:
```
score(x, y) = freq(xy) / (freq(x) × freq(y))
```

This is equivalent to pointwise mutual information (PMI). High PMI means x and y frequently co-occur, so merging them captures more information.

**Users**: BERT, DistilBERT, etc. Continuation subwords are marked with the `##` prefix (e.g., `playing` → `play` + `##ing`).

### 1.2.3 Unigram (SentencePiece)

**Principle**: Works in reverse — starts from a large vocabulary and iteratively removes tokens whose removal causes the smallest decrease in corpus likelihood, until the vocabulary shrinks to the target size.

**Training process**:
```
1. Initialize: build a large vocabulary from all substrings + characters (e.g., 1M)
2. Estimate each token's probability P(token) using EM
3. Compute each token's loss contribution: how much the corpus log-likelihood drops if removed
4. Remove the 10-20% of tokens with the smallest loss contribution
5. Repeat 2-4 until vocabulary reaches target size
```

**Advantage**: Unigram can output N-best segmentation results (probabilistic), which benefits regularization.

**Users**: T5, LLaMA, Gemma, etc. all use [SentencePiece](https://github.com/google/sentencepiece)'s Unigram model.

### 1.2.4 Comparison

| Feature | BPE | WordPiece | Unigram |
|---------|-----|-----------|---------|
| Direction | Bottom-up merging | Bottom-up merging | Top-down pruning |
| Merge criterion | Frequency | Likelihood/PMI | Likelihood (EM) |
| Determinism | Deterministic | Deterministic | Probabilistic (can sample) |
| Representative models | GPT series, LLaMA 2 | BERT | T5, LLaMA 3, Gemma |

## 1.3 Byte-Level vs Character-Level

**Byte-level BPE** (GPT-2/3/4, LLaMA):
- Base units are UTF-8 bytes (0-255), not Unicode characters
- Advantage: never encounters OOV — any byte sequence can be encoded
- Disadvantage: non-ASCII characters (e.g., Chinese) may require 3+ bytes per character

**Byte-level Fallback** (SentencePiece):
- Normally uses Unigram/BPE; falls back to byte representation for OOV characters
- LLaMA 3 uses this approach

**Pure byte models** ([ByT5](https://arxiv.org/abs/2105.13626)):
- No tokenizer at all — directly inputs UTF-8 bytes
- Advantage: zero preprocessing, extremely robust
- Disadvantage: sequence length increases 3-4x, high computational cost

## 1.4 Hands-On: Training Your Own Tokenizer

```python
# Train a BPE tokenizer with HuggingFace tokenizers library
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

# Train from files
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("my_tokenizer.json")

# Test
output = tokenizer.encode("Hello, 世界!")
print(output.tokens)
print(output.ids)
```

> Libraries: [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | Minimal implementation: [karpathy/minbpe](https://github.com/karpathy/minbpe)

**Key decisions**:
- **Vocabulary size**: 32K-256K. Larger → shorter sequences but larger embedding layer. LLaMA 2 uses 32K, LLaMA 3 uses 128K, GPT-4 uses 100K
- **Special tokens**: `<bos>`, `<eos>`, `<pad>`, `<unk>`; chat models also need `<|im_start|>`, `<|im_end|>`, etc.
- **Pre-tokenization**: Use regex to split text into chunks, preventing merges across "natural boundaries" (e.g., digits and letters, punctuation and words)
- **Normalization**: Whether to apply NFKC Unicode normalization, case folding, etc.

## 1.5 Multilingual Tokenizer Design

Multilingual support is a major challenge for tokenizers:

**Problem**: If training data is predominantly English, BPE merges are biased toward English — a Chinese character may require 3-4 tokens (due to UTF-8 encoding), while an English word needs only 1 token. This means the effective context window for Chinese is only 1/3 of English.

**Solutions**:
1. **Balanced training corpus**: Sample by language to ensure each language has sufficient data for merges
2. **Larger vocabulary**: LLaMA 3 went from 32K→128K, dramatically improving Chinese efficiency
3. **Language-specific pre-tokenization**: Use jieba/sentencepiece for Chinese word segmentation before running BPE
4. **Character coverage**: Ensure high-frequency Chinese characters / Japanese kana exist as individual tokens

**Fertility metric**: Measures a tokenizer's efficiency for a given language. `fertility = tokens / words` — closer to 1 is better. GPT-2's Chinese fertility is ~3.5, LLaMA 3's is ~1.5.

## 1.6 SOTA Tokenizer Techniques

- **Token Healing** ([guidance](https://github.com/guidance-ai/guidance)): Fixes prompt boundary issues caused by the tokenizer during generation
- **Byte Fallback**: SentencePiece's `byte_fallback=True` — OOV falls back to bytes
- **Split digits**: Split numbers into individual digit tokens (`2024` → `2` `0` `2` `4`) to improve math ability
- **Whitespace handling**: Preserve leading spaces as part of the token (GPT style) vs separate tokens
- **Code tokens**: Preserve indentation (tabs, multiple spaces) as special tokens to improve code generation

## Key Papers

- [Sennrich et al. (2016) — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — the BPE landmark for NMT
- [Kudo & Richardson (2018) — SentencePiece](https://arxiv.org/abs/1808.06226) — frees tokenization from whitespace assumptions
- [Kudo (2018) — Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959) — a probabilistic alternative to BPE

## Further Reading

- Karpathy — [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — 2-hour walkthrough writing BPE from scratch
- HuggingFace — [Tokenizers docs](https://huggingface.co/docs/tokenizers) — industrial-strength implementation
- OpenAI — [tiktoken](https://github.com/openai/tiktoken) — official GPT-family tokenizer

## Exercises

1. **Implement BPE from scratch**: train a 1K-token BPE vocab on a small English corpus (e.g. Shakespeare); compare against the `tokenizers` library.
2. **Multilingual comparison**: encode the same Chinese passage with the GPT-2 and Llama 3 tokenizers; compare token counts. Understand why Chinese-focused models need vocabulary extension.
3. **Number handling**: find text with digits (dates, prices, phone numbers); compare how GPT-2 vs Qwen tokenizers split them and which favors math.

---

[← Table of Contents](README.md) | [Next Chapter →](02-architecture.md)
