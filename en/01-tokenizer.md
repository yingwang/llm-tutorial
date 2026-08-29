[← Table of Contents](README.md) | [Next Chapter →](02-architecture.md)

# Chapter 1: Tokenizer

The tokenizer serves as the primary gateway to a Large Language Model, transforming continuous human language into discrete sequences of token IDs. The design of the tokenization vocabulary and algorithm directly dictates sequence compression ratios, vocabulary coverage, multilingual expressiveness, and overall pretraining throughput.

## 1.1 Why We Need a Tokenizer

Neural networks operate over numerical tensors rather than character strings. The simplest mapping is character-level tokenization (one token per character), but this strategy presents severe limitations:
- Excessive sequence lengths: A single sentence expands into hundreds of tokens, making the $O(N^2)$ computational complexity of standard self-attention prohibitively expensive.
- Absence of semantic granularity: The model must expend representational capacity rediscovering basic morphemic and word boundaries from scratch.

The opposite extreme is word-level tokenization (one token per word), which also breaks down in practice:
- Vocabulary explosion: Natural languages contain millions of distinct inflected forms, driving parameter count in embedding layers to unsustainable scales.
- Inability to handle Out-Of-Vocabulary (OOV) tokens: Unseen words or rare compound terms cannot be represented.
- Poor fit for non-segmented languages: Scripts such as Chinese and Japanese do not use whitespace to delimit word boundaries.

Subword tokenization resolves this dilemma: frequent lexical items remain intact as atomic tokens, while rare or morphologically complex terms decompose into reusable subword fragments.

## 1.2 Core Algorithms

### 1.2.1 BPE (Byte Pair Encoding)

**Principle**: A bottom-up data compression algorithm that starts from individual characters (or bytes) and iteratively merges the most frequently adjacent symbol pairs until the target vocabulary budget is reached.

**Training process**:
```
Initial vocabulary: all base characters (or bytes 0-255)
Loop:
    1. Count frequencies of all adjacent token pairs across the corpus
    2. Merge the most frequent pair into a single new token
    3. Update all occurrences in the corpus token sequences
    4. Terminate when vocabulary reaches the target size
```

**Example**:
```
Corpus: "low lower lowest"
Initial: ['l','o','w',' ','l','o','w','e','r',' ','l','o','w','e','s','t']
Round 1: 'l'+'o' → 'lo' (frequency = 3)
Round 2: 'lo'+'w' → 'low' (frequency = 3)
Round 3: 'low'+'e' → 'lowe' (frequency = 2)
...
```

**Byte-Level BPE in Modern LLMs**: OpenAI's [tiktoken](https://github.com/openai/tiktoken) and Hugging Face's tokenizers implement byte-level BPE with crucial engineering refinements:
- Operates directly over raw UTF-8 byte sequences rather than Unicode code points, eliminating OOV tokens across arbitrary languages.
- Applies regex-based pre-tokenization to partition text before merge counting, preventing merges across whitespace, punctuation, and numeric boundaries.
- GPT-4 and GPT-4o utilize `cl100k_base` and `o200k_base` vocabularies containing ~100K and ~200K tokens, respectively.

```python
# tiktoken usage
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
print(enc.decode(tokens))  # "Hello, world!"
```

### 1.2.2 WordPiece

**Principle**: Similar to BPE in its bottom-up construction, but selects merge candidates using a likelihood maximization objective rather than raw co-occurrence frequency.

**Scoring Criterion**: Selects the merge pair $(x, y) \to xy$ that maximizes:
```
score(x, y) = freq(xy) / (freq(x) * freq(y))
```

This formulation directly mirrors Pointwise Mutual Information (PMI). A high score signifies that $x$ and $y$ co-occur far more frequently than expected by chance, maximizing the predictive information gained by fusing them.

**Usage**: Used prominently in BERT and DistilBERT. Non-initial subwords are conventionally prefixed with `##` (e.g., `playing` decomposes into `play` and `##ing`).

### 1.2.3 Unigram (SentencePiece)

**Principle**: Operates top-down: begins with an over-complete candidate vocabulary (e.g., millions of substrings) and iteratively prunes items whose removal minimizes the penalty to the corpus log-likelihood under a unigram language model.

**Training process**:
```
1. Initialize an over-complete candidate vocabulary from corpus substrings
2. Estimate marginal token probabilities P(x) using the Expectation-Maximization (EM) algorithm
3. Compute the loss penalty for each token: the reduction in corpus log-likelihood if removed
4. Prune the bottom 10-20% of tokens with minimal loss impact
5. Repeat steps 2-4 until the vocabulary meets the target size
```

**Advantage**: Because Unigram defines a true generative probabilistic model over segmentations, it enables subword regularization (sampling alternative segmentations during pretraining to improve downstream robustness).

**Usage**: Adopted by T5, LLaMA, and Gemma via Google's [SentencePiece](https://github.com/google/sentencepiece) library.

### 1.2.4 Algorithmic Comparison

| Feature | BPE | WordPiece | Unigram |
|---------|-----|-----------|---------|
| Direction | Bottom-up merging | Bottom-up merging | Top-down pruning |
| Selection Metric | Frequency | Likelihood ratio / PMI | Marginal likelihood (EM) |
| Determinism | Deterministic | Deterministic | Probabilistic (supports sampling) |
| Representative Models | GPT series, LLaMA 2 | BERT | T5, LLaMA 3, Gemma |

## 1.3 Byte-Level vs Character-Level Representation

**Byte-Level BPE** (GPT-4, LLaMA):
- Base alphabet consists of the 256 individual UTF-8 bytes rather than Unicode characters.
- Advantage: Completely immune to out-of-vocabulary (OOV) tokens; any arbitrary byte sequence can be represented.
- Trade-off: Non-ASCII characters (such as Asian scripts) may decompose into multiple byte tokens unless adequately represented in the merge table.

**Byte-Level Fallback** (SentencePiece):
- Retains standard character-level or subword merges for common tokens, but routes rare and unseen characters to individual byte tokens rather than `<unk>`.
- Adopted by LLaMA 3 and Mistral.

**Pure Byte Models** ([ByT5](https://arxiv.org/abs/2105.13626)):
- Operates directly over raw UTF-8 byte streams without subword vocabulary extraction.
- Advantage: Zero preprocessing pipeline, robust against character noise and typographic perturbations.
- Disadvantage: Expands sequence length by 3x to 4x, imposing significant attention and memory overhead.

## 1.4 Hands-On: Training a Custom BPE Tokenizer

```python
# Train a BPE tokenizer using Hugging Face tokenizers
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

# Train from corpus files
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("my_tokenizer.json")

# Test encoding and decoding
output = tokenizer.encode("Hello, 世界!")
print(output.tokens)
print(output.ids)
```

> Production libraries: [huggingface/tokenizers](https://github.com/huggingface/tokenizers) (Rust-backed) | Pedagogical reference: [karpathy/minbpe](https://github.com/karpathy/minbpe)

**Key Architectural Decisions**:
- **Vocabulary Budget**: Typically 32K to 256K tokens. Larger vocabularies compress text into fewer tokens per sequence (saving attention compute), but enlarge the parameter and memory footprint of input embeddings and output projection heads. LLaMA 2 used 32K; LLaMA 3 scaled to 128K; GPT-4o expanded to 200K.
- **Special Tokens**: Crucial delimiter tokens including `<bos>`, `<eos>`, `<pad>`, `<unk>`, alongside chat markup tokens like `<|im_start|>` and `<|im_end|>`.
- **Pre-Tokenization Rules**: Regex rules preventing merges across heterogeneous lexical categories (e.g., isolating digits from letters, and punctuation from alphanumeric strings).
- **Text Normalization**: Deciding whether to apply NFKC Unicode normalization, lowercase folding, or whitespace collapsing.

## 1.5 Multilingual Tokenizer Optimization

Multilingual tokenization presents a distinct set of engineering challenges:

**The Compression Disparity Problem**: When pretraining datasets are dominated by English text, BPE merges naturally prioritize English subwords. Consequently, a single Chinese or Arabic character may decompose into 2 to 4 byte tokens, whereas a complete English word requires only a single token. This token inflation effectively degrades the usable context window and increases inference costs for non-Latin languages.

**Engineering Mitigations**:
1. **Stratified Corpus Sampling**: Ensure non-English corpora are sufficiently represented during tokenizer training to allocate adequate merge capacity.
2. **Vocabulary Expansion**: Scaling vocabulary from 32K (LLaMA 2) to 128K (LLaMA 3) drastically reduces token fertility for non-Latin scripts.
3. **Domain Pre-Tokenization**: Applying specialized segmentation (e.g., Jieba or SentencePiece) prior to merge extraction.
4. **Guaranteed Character Coverage**: Ensuring all frequently used CJK ideographs and Arabic/Cyrillic scripts exist as individual atomic tokens.

**Fertility**: The average number of tokens required to encode a single word or character in a given language. Ratios approaching 1.0 reflect optimal encoding efficiency. GPT-2 exhibits a fertility rate around 3.5 on Chinese text, whereas modern multilingual tokenizers reduce this figure below 1.5.

## 1.6 Modern Tokenization Techniques

- **Token Healing** ([guidance](https://github.com/guidance-ai/guidance)): Resolves prompt boundary bias caused by tokenizer greedy prefix matching during autoregressive continuation.
- **Byte Fallback**: SentencePiece `byte_fallback=True`, routing unmapped characters to raw byte tokens rather than generic `<unk>` placeholders.
- **Digit Splitting**: Splitting numerical values into individual isolated digit tokens (`2024` $\to$ `2` `0` `2` `4`), preventing arbitrary number chunking and improving arithmetic reasoning.
- **Whitespace Preservation**: Preserving leading and trailing whitespace within tokens to retain exact indentation for programming language understanding.
- **Code Indentation Tokens**: Dedicated multi-space and tab tokens to compress Python and YAML indentation levels efficiently.

## Key Papers

- [Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): Foundational paper introducing BPE to neural NLP.
- [Kudo & Richardson (2018): SentencePiece](https://arxiv.org/abs/1808.06226): Language-independent subword tokenizer eliminating language-specific pre-tokenization.
- [Kudo (2018): Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959): Probabilistic subword sampling for robust representation learning.

## Further Reading

- Karpathy: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (Complete first-principles BPE walkthrough).
- Hugging Face: [Tokenizers Documentation](https://huggingface.co/docs/tokenizers) (High-performance industrial implementation).
- OpenAI: [tiktoken Repository](https://github.com/openai/tiktoken) (Fast BPE tokenizer used in production GPT models).

## Exercises

1. **BPE from First Principles**: Implement a minimal BPE training loop in pure Python; train a 1,000-token vocabulary on a small corpus (e.g., Shakespeare) and verify outputs against the Hugging Face `tokenizers` library.
2. **Multilingual Fertility Analysis**: Encode identical parallel passages in English, Chinese, and Arabic using the GPT-2, LLaMA 2, and LLaMA 3 tokenizers. Quantify the compression ratio improvements gained by expanding vocabulary size.
3. **Numerical Tokenization Inspection**: Compare how different tokenizers (GPT-2 vs. Qwen vs. LLaMA 3) tokenize complex numeric expressions (e.g., dates, floating-point numbers, and mathematical equations), and evaluate how digit splitting affects arithmetic modeling.

---

[← Table of Contents](README.md) | [Next Chapter →](02-architecture.md)
