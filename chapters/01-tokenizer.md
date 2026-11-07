[← 目录](../README.md) | [下一章 →](02-architecture.md)

# 第一章：Tokenizer

分词器（Tokenizer）是大语言模型处理文本的第一步，负责把原始字符串切分成子词片段，再映射为模型可以计算的 Token ID 序列。词表怎么建、文本怎么切，直接关系到同一段话会占多少个 Token，也会影响多语言压缩率、数学与代码的处理效果，以及训练和推理的计算开销。

## 1.1 为什么需要 Tokenizer

神经网络只能对数值张量做矩阵运算，不能直接接收字符串。要把文本变成数字序列，最简单的做法是按字符切分（Character-level），每个字符对应一个 Token，但这会带来两个明显的问题：
- 序列会被拉得非常长，一句话就要占几十上百个 Token，自注意力层 $\mathcal{O}(n^2)$ 的计算和显存开销会迅速变得难以承受；
- 单个字符本身几乎不带完整的语义信息，模型得在浅层额外花不少容量去拼凑词的边界。

另一个极端是按完整的词切分（Word-level），每个单词对应一个 Token，同样行不通：
- 词表会急剧膨胀，仅英语里的词形变化、派生词和专有名词就能把词表撑到几十万以上；
- 推理时只要遇到训练没见过的生词（Out-of-Vocabulary, OOV），模型就只能退化成统一的 `<unk>`，丢失具体信息；
- 中文、日文这类书写时没有空格分隔的语言，很难事先定出统一的词边界。

因此，现代大模型普遍采用子词分词（Subword Tokenization）作为折中方案：常见的高频词直接保留为完整的 Token，低频词或生词则拆成更小的词根、词缀甚至字节片段。这样既控制住了词表规模，又不会因为生词丢信息。

## 1.2 核心算法

### 1.2.1 BPE (Byte Pair Encoding)

**原理**：从单字符（或单字节）组成的基础词表出发，统计语料里所有相邻 Token 对的出现频次，每次把出现最多的一对合并成新 Token，如此反复，直到词表达到预设大小。

**训练过程**：
```
初始词表: 所有单字符（或字节 0-255）
循环:
    1. 统计所有相邻 token pair 的频率
    2. 合并频率最高的 pair → 新 token
    3. 更新语料中的 token 序列
    4. 词表大小达到目标 → 停止
```

**举例**：
```
语料: "low lower lowest"
初始: ['l','o','w',' ','l','o','w','e','r',' ','l','o','w','e','s','t']
第1轮: 'l'+'o' → 'lo' (出现3次, 最高频)
第2轮: 'lo'+'w' → 'low' (出现3次)
第3轮: 'low'+'e' → 'lowe' (出现2次)
...
```

**GPT 系列的 BPE**：OpenAI 的 [tiktoken](https://github.com/openai/tiktoken) 在字节级 BPE 上做了几项关键的工程改进：
- 直接以 UTF-8 字节（0–255）为基础单元，而不是 Unicode 字符，任何语言和特殊符号都能编码，不会出现 OOV；
- 先用正则表达式做预分词（Pre-tokenization），按空格、标点、数字和字母边界把文本切成小块，再在块内跑 BPE，防止把单词和后面的标点硬拼成一个 Token；
- GPT-4 使用的 `cl100k_base` 词表包含约 100K 个 Token。

```python
# tiktoken 使用
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
print(enc.decode(tokens))  # "Hello, world!"
```

### 1.2.2 WordPiece

**原理**：同样是自底向上合并相邻子词，但合并标准和 BPE 不同。BPE 只看哪个组合出现次数最多，而 WordPiece 选择合并后能让语料在语言模型下的似然概率提升最大的一对。

**公式**：每次选择合并 $(x, y) \to xy$，使下面的分数最高：
```
score(x, y) = freq(xy) / (freq(x) × freq(y))
```

这个比值对应的就是点互信息（Pointwise Mutual Information, PMI）。如果 $x$ 和 $y$ 各自到处出现、只是碰巧挨在一起，分母就会很大，得分并不高；只有当 $x$ 和 $y$ 总是结伴出现时，它们才会被优先合并。

**代表模型**：BERT、DistilBERT 等。词表里会用 `##` 前缀标记词中或词尾的子词片段（例如 `playing` 会切分成 `play` + `##ing`）。

### 1.2.3 Unigram (SentencePiece)

**原理**：走的是自顶向下的删减路线。先准备一个非常大的初始候选词表，然后每次评估删掉某个 Token 后语料整体似然概率会下降多少，把影响最小的一批 Token 删掉，逐步缩减到目标词表大小。

**训练过程**：
```
1. 初始化: 用所有高频子串 + 单字符构建大词表 (如 100万)
2. 用 EM 算法估计每个 token 的概率 P(token)
3. 计算每个 token 的 loss 贡献: 如果去掉它，语料的 log-likelihood 下降多少
4. 删掉 loss 贡献最小的 10-20% token
5. 重复 2-4 直到词表大小达到目标
```

**优势**：由于每个 Token 都有对应的概率，Unigram 不仅可以给出最优切分，还能按概率采样出多种切分结果（Subword Regularization），在训练时起到类似数据增强的作用。

**代表模型**：T5、LLaMA、Gemma 等都使用了 [SentencePiece](https://github.com/google/sentencepiece) 工具包中的 Unigram 或 BPE 实现。

### 1.2.4 对比

| 特征 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| 方向 | 自底向上合并 | 自底向上合并 | 自顶向下删减 |
| 合并/删减标准 | 共现频率 | Likelihood / PMI | Likelihood (EM) |
| 确定性 | 确定 | 确定 | 概率化 (可采样) |
| 代表模型 | GPT 系列, LLaMA 2 | BERT | T5, LLaMA 3, Gemma |

## 1.3 字节级 vs 字符级

**Byte-level BPE**（GPT-2/3/4, LLaMA）：
- 基础单元是 UTF-8 字节（0–255），而不是 Unicode 字符；
- 好处是基础词表固定只有 256 项，任何字节序列都能编码，永远不会遇到 OOV；
- 代价是对非 ASCII 字符不够紧凑，比如一个生僻汉字在 UTF-8 里占 3 个字节，如果没有被合并进词表，就会拆成 3 个独立的字节 Token。

**Byte-level Fallback**（SentencePiece）：
- 平时按字符级 Unigram 或 BPE 正常切分，只有遇到词表里没有的生僻字符时，才回退成 UTF-8 字节表示；
- LLaMA 3 等模型采用了这种机制，兼顾常见字符的切分效率与生僻字符的兜底能力。

**纯字节模型**（[ByT5](https://arxiv.org/abs/2105.13626)）：
- 完全去掉分词器，直接把原始 UTF-8 字节序列喂给模型；
- 好处是不需要任何文本预处理，对拼写错误和噪声非常稳健；
- 代价是同样的文本序列长度会拉长 3 到 4 倍，训练和推理的计算开销明显增加。

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

> 常用库：[huggingface/tokenizers](https://github.com/huggingface/tokenizers) | 极简教学实现：[karpathy/minbpe](https://github.com/karpathy/minbpe)

**关键决策**：
- **词表大小**：通常选在 32K 到 256K 之间。词表越大，同样的文本切出来的序列越短，但输入 Embedding 层和输出 LM Head 的参数量与显存占用也会跟着变大。LLaMA 2 用 32K，LLaMA 3 扩充到了 128K，GPT-4 约为 100K。
- **特殊 Token**：除了基础的 `<bos>`、`<eos>`、`<pad>`、`<unk>`，对话模型还需要预留 `<|im_start|>`、`<|im_end|>` 等控制标记。
- **预分词**：通过正则把文本先按自然边界切开，防止数字和字母、单词和标点符号跨边界合并不该合并的片段。
- **规范化**：决定是否在分词前做 NFKC 等 Unicode 归一化或大小写转换（现代大模型为了保留原始文本与代码格式，通常不做破坏性的归一化）。

## 1.5 多语言 Tokenizer 设计

做多语言模型时，分词器很容易出现语种之间的效率失衡：

**问题**：如果训练分词器的语料以英文为主，BPE 的合并名额就会大量分给英文词根和单词。一个英文单词通常只占 1 个 Token，而中文由于 UTF-8 编码一个字占 3 个字节，又没分到多少合并名额，一个汉字往往要被切成 2 到 3 个 Token。这样一来，同样长的上下文窗口，中文能装下的实际内容只有英文的三分之一左右，生成速度也变慢。

**解决方案**：
1. **平衡训练语料**：训练分词器时按语种调整采样比例，给低资源语言和非拉丁语系留出足够的合并机会；
2. **扩大词表**：LLaMA 3 把词表从 32K 扩大到 128K，多出来的名额容纳了大量多语言常见词与汉字，中文编码效率明显改善；
3. **语言特定预分词**：针对中文等无空格语言，先用分词工具或正则规则切分，避免跨词边界乱合；
4. **字符覆盖**：设置较高的 `character_coverage`（如 0.9995），保证常用汉字、日文假名等直接作为独立 Token 收入词表。

**Fertility 指标**：用来衡量分词器对某种语言的压缩效率，定义为 `fertility = tokens / words`（中文常按字或词统计），数值越接近 1 说明切分越紧凑。GPT-2 对中文的 Fertility 大约是 3.5，而 LLaMA 3 降到了 1.5 左右。

## 1.6 SOTA Tokenizer 技巧

- **Token Healing**（[guidance](https://github.com/guidance-ai/guidance)）：在生成阶段自动回退并重切 Prompt 末尾的边界 Token，修补因末尾空格或半个词引起的贪心分词偏差。
- **Byte Fallback**：开启 SentencePiece 的 `byte_fallback=True`，遇到未登录字符时自动拆成 UTF-8 字节，避免产生 `<unk>`。
- **Split digits**：把多位数字按位拆成独立的单数字 Token（如 `2024` 切成 `2` `0` `2` `4`），避免同一个数字因为切分方式不一致而干扰算术推理。
- **Whitespace handling**：把前导空格并入词首作为子词的一部分（GPT 风格，如 `Ġworld`），或者把空格当成独立 Token 处理。
- **Code tokens**：把连续多个空格、缩进和制表符（Tab）合并成专门的单个 Token，大幅缩短 Python 等代码的 Token 序列长度。

## 关键论文

- [Sennrich et al. (2016): Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：把 BPE 引入神经机器翻译的经典工作
- [Kudo & Richardson (2018): SentencePiece](https://arxiv.org/abs/1808.06226)：摆脱空格预分词依赖的通用分词工具
- [Kudo (2018): Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959)：基于 Unigram 概率模型的子词采样与正则化方法

## 进一步阅读

- Karpathy：[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)，从零手写 BPE 分词器的两小时实战视频
- HuggingFace：[Tokenizers 文档](https://huggingface.co/docs/tokenizers)，Rust 编写的高性能分词库说明
- OpenAI：[tiktoken](https://github.com/openai/tiktoken)，GPT 系列官方开源的字节级 BPE 实现

## 练习题

1. **从零实现 BPE**：在英文小语料（如莎士比亚全集）上手写训练一个 1K 词表的 BPE 分词器，并与 `tokenizers` 库的切分结果做对比。
2. **多语言对比**：分别用 GPT-2、LLaMA 2 和 LLaMA 3 的 Tokenizer 编码同一段中文技术文章，统计各自的 Token 数量，算一下词表扩充对上下文窗口利用率的实际影响。
3. **数字处理**：找一段包含年份、价格、电话号码和算式的文本，对比 GPT-2 与 Qwen/LLaMA 3 的 Tokenizer 分别怎么切分数字，看看按位切分对加减法对齐有什么帮助。

---

[← 目录](../README.md) | [下一章 →](02-architecture.md)
