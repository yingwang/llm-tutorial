[← 目录](../README.md) | [下一章 →](02-architecture.md)

# 第一章：分词器 (Tokenizer)

分词器（Tokenizer）构成了大语言模型与离散文本交互的第一道门户：它将连续的字符流切分并映射为模型可计算的 Token ID 序列。分词器的构造机制不仅决定了模型的词汇覆盖边界与有效序列长度，更在深层次上制约着多语言表征迁移、数学符号推理以及端到端训练与推理的计算效率。

## 1.1 为什么需要分词表征

深度神经网络的底层算子仅能运算连续数值张量，无法直接吞吐离散字符串。若直接采用字符级（Character-level）表征：
- 序列长度急剧膨胀，致使自注意力算子的 $\mathcal{O}(n^2)$ 计算与显存开销不堪重负；
- 单字符缺乏独立的语义信息密度，迫使模型在底层耗费大量表征容量自底向上重构词素边界。

反之，若走向词级（Word-level）表征的另一极端：
- 词表规模将因屈折变化、复合派生词与专业术语而产生维度灾难；
- 面对未登录词（Out-of-Vocabulary, OOV）时表征能力彻底失效；
- 对于汉语、日语等缺乏天然空格分界符的表意语言体系，难以构建统一的词法边界。

子词分词（Subword Tokenization）在二者之间取得了优美的平衡：它将高频词固化为独立语义单元，将低频词与生僻词解构为更基础的子词片段，兼顾了表征紧凑性与开集词汇的无限覆盖能力。

## 1.2 核心算法

### 1.2.1 BPE (Byte Pair Encoding)

**核心原理**：从基础字符或单字节集合起步，通过贪心统计与迭代合并语料中出现频率最高的相邻 Token 对，直至词表达到预设容量。

**算法流程**：
```
初始词表: 所有基础单字符集合（或 0-255 基础字节）
迭代循环:
    1. 统计当前语料库中所有相邻 Token 对的共现频次
    2. 选取共现频次最高的一对 (t_i, t_j) 融合成新 Token t_new
    3. 全局更新语料库中的 Token 序列
    4. 当词表规模达到目标阈值时终止迭代
```

**示例推演**：
```
语料: "low lower lowest"
初始: ['l','o','w',' ','l','o','w','e','r',' ','l','o','w','e','s','t']
第1轮: 'l'+'o' → 'lo' (出现3次, 最高频)
第2轮: 'lo'+'w' → 'low' (出现3次)
第3轮: 'low'+'e' → 'lowe' (出现2次)
...
```

**现代字节级 BPE**：OpenAI 的 [tiktoken](https://github.com/openai/tiktoken) 在经典 BPE 基础上进行了关键工业级革新：
- 直接以 UTF-8 原始字节为基底，天然具备对任意未知语言与特殊二进制字符的零 OOV 编码能力；
- 引入严密的正则表达式预分词（Pre-tokenization），在语法边界（如字母与标点、数字序列）预先切分，阻断跨语义块的错误合并；
- GPT-4 所采用的 `cl100k_base` 词表规模约达 100,000 个 Token。

```python
# tiktoken 使用
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
print(enc.decode(tokens))  # "Hello, world!"
```

### 1.2.2 WordPiece

**核心原理**：与 BPE 的自底向上贪心合并机制类似，但合并准则从单纯的频次统计演进为最大化训练语料在单变量语言模型下的似然增益。

**打分公式**：选择使得互信息（PMI）最大化的子词对 $(x, y) \to xy$ 进行合并：
```
score(x, y) = freq(xy) / (freq(x) × freq(y))
```

点互信息（Pointwise Mutual Information）从统计学角度度量了两个子词片段在语料中的非随机绑定强度：得分越高，说明二者的结合越具结构特异性而非偶然共现。

**工程代表**：BERT 与 DistilBERT。词表内部采用 `##` 前缀显式标识子词的续接状态（例如 `playing` 拆解为 `play` + `##ing`）。

### 1.2.3 Unigram (SentencePiece)

**核心原理**：与 BPE 的自底向上合并相反，Unigram 采用自顶向下的剪枝策略：预先构建一个过饱和的庞大候选子词库，随后通过期望最大化（EM）算法逐步裁剪掉对语料总似然贡献最小的 Token，直至收缩至预设词表规模。

**算法流程**：
```
1. 初始化: 提取语料中所有高频子串与基础字符构建超大初始词表（如 100 万词条）
2. 运行 EM 算法估计当前词表中每个子词的先验概率 P(token)
3. 评估损失增量: 计算剔除各 Token 后语料对数似然（Log-Likelihood）的下降幅度
4. 批量裁剪对总似然损失影响最小的 10%–20% 候选 Token
5. 重复步骤 2–4 直至词表收敛至目标尺寸
```

**概率化优势**：Unigram 分词天然基于概率图输出 $N$-best 分词切分或进行分词采样（Subword Regularization），能够在训练阶段为模型注入输入扰动，提升鲁棒性。

**工程代表**：T5、LLaMA、Gemma 等模型均深度依赖 [SentencePiece](https://github.com/google/sentencepiece) 的 Unigram 模式。

### 1.2.4 算法特性对比

| 特征 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| 构词方向 | 自底向上贪心合并 | 自底向上贪心合并 | 自顶向下概率剪枝 |
| 合并/剪枝准则 | 共现频次 | 似然增益 / PMI | 语料总似然贡献 (EM) |
| 确定性 | 确定性切分 | 确定性切分 | 概率化分布 (支持正则化采样) |
| 代表模型 | GPT 系列, LLaMA 2 | BERT | T5, LLaMA 3, Gemma |

## 1.3 字节级 vs 字符级

**Byte-level BPE** (GPT-2/3/4, LLaMA)：
- 以 UTF-8 基础字节（0–255）为原子表征起点，彻底消除 OOV 困境；
- 缺陷在于对非 ASCII 字符（如中文单个汉字通常占 3 个字节）产生更细碎的初始切分，拉长输入序列。

**Byte-level Fallback** (SentencePiece)：
- 常规分词走 Unigram 或 BPE 路径，仅在遭遇未登记字符时动态回退至 UTF-8 原始字节序列；
- LLaMA 3 融合了这一设计以确保对生僻符号的绝对兜底。

**纯字节模型** ([ByT5](https://arxiv.org/abs/2105.13626))：
- 完全摒弃分词器预处理，端到端直接吞吐原始 UTF-8 字节流；
- 优势在于具备极致的噪声鲁棒性与跨语言统一性；
- 劣势在于序列长度剧增 3 至 4 倍，大幅加重自注意力计算的二次方负担。

## 1.4 实战：训练专有分词器

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

> 工业级核心库：[huggingface/tokenizers](https://github.com/huggingface/tokenizers)；教学极简实现：[karpathy/minbpe](https://github.com/karpathy/minbpe)

**关键工程决策**：
- **词表容量权衡**：通常在 32K 至 256K 之间。词表越大，文本压缩率越高（序列越短），但输入 Embedding 与输出 LM Head 的参数量和显存开销随之线性剧增。LLaMA 2 设定为 32K，LLaMA 3 扩充至 128K，GPT-4 采用约 100K；
- **特殊 Token 拓扑**：精确规划 `<bos>`、`<eos>`、`<pad>`、`<unk>` 以及对话结构标识符（如 `<|im_start|>`、`<|im_end|>`）；
- **预分词规则（Regex Pre-tokenization）**：设计高效正则表达式隔离自然语义分界（如标点、连续数字、字母缩写），防止有害的跨类型子词合并；
- **文本正则化**：权衡是否施加 NFKC Unicode 归一化或大小写折叠。

## 1.5 多语言分词器设计

在跨语言场景下，分词器设计面临显著的表征失衡挑战：

**语言不均问题**：若训练数据中英语占据绝对主导，BPE 算法将优先将高频英文词组压缩为单一 Token；而中文等表意文字由于单字对应 3 字节 UTF-8 编码且缺乏足够的合并频次，往往退化为单字甚至按字节拆分（一个汉字消耗 2 至 3 个 Token）。这导致中文输入的有效上下文窗口被严重压缩至英文的数分之一。

**工业级优化策略**：
1. **平衡采样语料**：在分词器训练阶段对多语言语料实施温度加权采样，确保低资源语言的高频词素获得充分合并机会；
2. **扩充词表容量**：LLaMA 3 将词表从 32K 跨越式增至 128K，直接使中文与多语言的压缩率提升数倍；
3. **语言特异性预分词**：对中文等无空格语言引入结巴分词或字词切分前处理，阻断无意义的字边界交错；
4. **高频字符硬覆盖**：强制将常用汉字、日文假名及各语言基础文字作为不可分割的独立单元纳入基础词表。

**Fertility 指标**：用于量化分词器在特定语言上的压缩效率：
$$\text{Fertility} = \frac{\text{Tokens 数量}}{\text{Words/Characters 数量}}$$
Fertility 越趋近于 1，表明分词表征越紧凑。GPT-2 对中文的 Fertility 高达约 3.5，而经过充分多语言扩表优化的 LLaMA 3 已降至约 1.5。

## 1.6 前沿 SOTA 分词技巧

- **Token Healing** ([guidance](https://github.com/guidance-ai/guidance))：消除自回归解码时由于 Prompt 尾部空格与贪心分词边界不匹配造成的表征偏置；
- **Byte Fallback**：启用 SentencePiece 的 `byte_fallback=True`，在遭遇生僻 Unicode 字符时平滑降级至单字节序列；
- **数字按位拆分 (Split Digits)**：强制将多位数字解构为单数字 Token（例如 `2024` 切分为 `2` `0` `2` `4`），避免大数字组合爆炸并显著增强数学运算与数值推理的一致性；
- **前导空格归一 (Whitespace Handling)**：保留前导空格作为子词前缀（GPT 风格）还是拆分为独立空格 Token；
- **代码缩进与排版专用 Token**：为连续制表符（Tab）或特定数量空格预置专用 Token，有效压缩代码文件的有效上下文长度并提升代码生成质量。

## 经典文献

- [Sennrich et al. (2016) — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：子词 BPE 在神经机器翻译中的奠基之作
- [Kudo & Richardson (2018) — SentencePiece](https://arxiv.org/abs/1808.06226)：将分词表征从语言特定空格依赖中彻底解放的通用框架
- [Kudo (2018) — Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959)：提出基于概率图采样的子词正则化方法

## 进阶参考

- Andrej Karpathy：[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)（从底层逻辑推演并手写 BPE 的经典解析）
- HuggingFace：[Tokenizers 官方文档](https://huggingface.co/docs/tokenizers)（工业级 Rust 高性能分词流水线指南）
- OpenAI：[tiktoken 仓库](https://github.com/openai/tiktoken)（GPT 系列官方极速分词实现）

## 实践训练

1. **从零实现 BPE 算法**：在纯文本微型语料上手写 BPE 训练逻辑（构建 1K 容量词表），并将分词切分结果与 HuggingFace `tokenizers` 工业实现进行一致性对比。
2. **多语言表征压缩率分析**：使用 GPT-2、LLaMA 2 与 LLaMA 3 的分词器分别编码同一段技术性中文语料，统计 Token 生成总量与 Fertility 指标，深入理解词表扩充对上下文承载力的物理意义。
3. **数字分词机制与数学推理相关性探究**：选取包含长数字、时间戳与浮点算式的文本，对比分析 GPT-2 贪心合并策略与 Qwen 独立数字切分策略在词法结构上的异同。

---

[← 目录](../README.md) | [下一章 →](02-architecture.md)
