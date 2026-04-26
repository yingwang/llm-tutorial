[← 目录](../README.md) | [下一章 →](02-architecture.md)

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

## 关键论文

- [Sennrich et al. (2016) — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 在 NMT 中的开山之作
- [Kudo & Richardson (2018) — SentencePiece](https://arxiv.org/abs/1808.06226) — 把分词从空格依赖中解放出来
- [Kudo (2018) — Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959) — 与 BPE 不同的概率化路径

## 进一步阅读

- Karpathy — [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)：从头写 BPE 的 2 小时视频
- HuggingFace — [Tokenizers 文档](https://huggingface.co/docs/tokenizers)：工业级实现
- OpenAI — [tiktoken](https://github.com/openai/tiktoken)：GPT 系列官方 tokenizer

## 练习题

1. **从零实现 BPE**：在英文小语料（如莎士比亚）上手写训练 BPE 词表（目标 1K tokens），与 `tokenizers` 库结果对比。
2. **多语言对比**：用 GPT-2 和 Llama 3 的 tokenizer 分别 encode 同一段中文，比较 token 数量。理解为什么中文模型需要扩词表。
3. **数字处理**：找一段含数字（日期、价格、电话）的文本，看 GPT-2 vs Qwen tokenizer 怎么切，哪种更利于数学能力。

---

[← 目录](../README.md) | [下一章 →](02-architecture.md)
