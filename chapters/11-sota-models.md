[← 上一章](10-safety-alignment.md) | [目录](../README.md) | [下一章 →](12-distillation-merging.md)

# 第十一章：主流模型技术解析

前面十章拆解了模型架构、训练工程与后训练方法，本章把这些零件装回真实模型里，看看当前几家主流开源与闭源模型（LLaMA 3、DeepSeek-V3/R1、Claude、Gemini、GPT-4/o1/o3、Qwen 2.5、Mistral 以及端侧小模型）各自是怎么做架构取舍和训练配方的。

## 11.1 LLaMA 3 / 3.1 / 3.3（Meta）

> 论文：[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | 权重：[meta-llama](https://huggingface.co/meta-llama)

LLaMA 3 选择了最稳妥的工程路线：不在模型结构上搞激进改动（连 MoE 都没上），而是靠扎实的稠密（Dense）Transformer 结构、15T 高质量 token 和 1.6 万张 H100 的工程稳定性把效果推到上限。

```
参数规模: 8B, 70B, 405B (后续推出蒸馏优化版 3.3-70B 与端侧版 3.2-1B/3B)
模型结构: 标准 Dense Transformer, GQA (8 个 KV 头), RoPE (基频调至 500,000), SwiGLU, RMSNorm
词表大小: 128K (基于 tiktoken 的 BPE，相比 LLaMA 2 的 32K 大幅提升了代码与多语言压缩率)
上下文长度: 128K (在预训练最后阶段分 6 步从 8K 逐步扩到 128K)
预训练数据: 15.6T tokens (清洗并调高了代码与数学比例)
训练硬件: 16,384 张 H100 80GB (405B 模型训练耗时约 54 天，MFU 达到 38%-43%)

核心做法:
- 远超 Chinchilla 比例的过训练 (Over-training): 8B 小模型也喂满 15T tokens，牺牲一点一次性训练算力，换取极低的线上推理成本
- 后训练不靠复杂强化学习: 采用“SFT → 拒绝采样 (Rejection Sampling) → DPO”多轮迭代，每轮用最新模型生成数据再筛优训练
- 代码执行反馈: 训代码能力时直接把生成的代码丢进沙盒运行，拿编译和单元测试结果做过滤信号
```

## 11.2 DeepSeek-V3 / R1（DeepSeek）

> 论文：[DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 权重：[deepseek-ai](https://huggingface.co/deepseek-ai)

DeepSeek-V3 和 R1 代表了另一种思路：在算力受限的条件下，靠架构创新（MLA + 细粒度 MoE）、底层系统通信优化（DualPipe + PTX 调优）与强化学习后训练把单卡效率榨干。

```
DeepSeek-V3:
  参数规模: 总参数 671B，单 token 激活 37B
  核心结构: MLA (多头潜在注意力) + DeepSeekMoE (每层 1 个共享专家 + 256 个路由专家，每次激活 Top-8)
  训练规模: 14.8T tokens，使用 2048 张 H800 GPU 训练
  算力账单: 约 278.8 万 H800 卡时 (折合租价约 550 万美元)

  关键技术:
  - MLA 压缩: 把 KV 压成低维潜向量缓存，推理时 KV Cache 占用减少约 93%
  - 原生 FP8 混合精度训练: 首次在超大规模 MoE 上跑通细粒度分块 (Tile/Block-wise) FP8 前反向矩阵乘
  - 无辅助损失负载均衡 (Auxiliary-Loss-Free): 用动态调整的路由偏置项替代传统辅助 Loss，既防止专家坍缩又不伤害主任务精度
  - 多词元预测 (Multi-Token Prediction, MTP): 每个位置额外顺带预测下一个词，既加厚训练信号，又能在推理时直接拿来做投机采样加速

DeepSeek-R1:
  基于 V3 基座训练的深度推理模型

  关键发现:
  - DeepSeek-R1-Zero: 完全跳过 SFT，直接拿 V3 Base 跑纯强化学习 (GRPO 算法 + 数学/代码规则判分奖励)，模型自发学会了拉长思维链、自我验算和回溯反思
  - DeepSeek-R1 完整四阶段流程:
    1. 几千条长思维链冷启动 SFT (解决纯 RL 输出可读性差、中英混杂的问题)
    2. 面向推理任务的二阶段大规模 GRPO 强化学习
    3. 拒绝采样收集推理与通用数据，混入 V3 通用数据重新做一轮全量 SFT
    4. 全场景强化学习对齐 (兼顾推理准确性与安全有用性)
  - 数学与代码推理表现对齐 OpenAI o1，且完整开放权重与技术报告
```

## 11.3 Claude 3 / 3.5 / 4（Anthropic）

> 文档：[anthropic.com/claude](https://www.anthropic.com/claude)

Anthropic 不公开具体参数量与底层架构，但 Claude 3.5 Sonnet 与 Claude 4 系列在日常写代码、读长文档和跑 Agent 工作流上积累了极高口碑。

```
型号档位: Haiku (轻量快速) → Sonnet (主力均衡，编码与 Agent 首选) → Opus (大尺寸深度推理)
上下文窗口: 200K tokens (大海捞针与长文档跨段落细节召回扎实)

主要特点:
- Constitutional AI (宪法 AI): 用一套成文准则让模型对自己的回答做自我批评和修正 (RLAIF)，减少人工标注依赖，同时避免生硬说教式的拒答
- 真实软件工程能力强: 在 SWE-bench、复杂代码重构、前端界面还原与 Computer Use (直接看屏幕操作鼠标键盘) 上表现突出
- 混合推理模式 (Extended Thinking): 支持在普通秒回模式与长思维链深度思考模式之间按需切换
```

## 11.4 Gemini 1.5 / 2（Google）

> 论文：[Gemini 1.0](https://arxiv.org/abs/2312.11805) | [Gemini 1.5](https://arxiv.org/abs/2403.05530)

Google 的优势在于自研 TPU 集群与全模态长上下文架构：

```
底层结构: 原生多模态稀疏 MoE Transformer
训练硬件: Google TPU v5p / v6e (Trillium) 集群

主要特点:
- 从头原生多模态: 从预训练起就把文本、图像、音频、视频帧混在同一个序列里训，而不是拿纯文本模型后期外挂视觉编码器
- 1M-2M 超长上下文: Gemini 1.5 Pro 支持直接塞进 200 万 tokens (相当于 2 小时视频、几十小时录音或整个大型代码仓库)，长上下文检索准确率极高
- Flash 系列: 针对高频低延迟场景深度优化的轻量 MoE 版本，性价比极高
```

## 11.5 GPT-4 / GPT-4o / o1 / o3（OpenAI）

> 文档：[platform.openai.com](https://platform.openai.com/docs)

OpenAI 先后探出了两条技术路线：GPT-4 跑通了超大规模预训练与 RLHF 对齐，而 o1/o3 开启了测试期算力扩展（Test-Time Compute Scaling）。

```
GPT-4 / GPT-4o:
  - GPT-4 业界推测为 ~1.8T 参数的 MoE 模型 (16 个专家，每次激活 2 个)
  - GPT-4o ("o" 代表 Omni) 打通了文本、视觉与实时语音的端到端统一表示，把语音对话延迟压到了两三百毫秒

o1 / o3 系列 (Reasoning Models):
  - 核心转向: 当预训练堆文本数据撞上墙后，把算力花在“强化学习后训练”和“推理阶段多想一会儿”上
  - 训练机制: 通过大规模强化学习训练模型在输出最终答案前先生成长串隐式思维链 (Hidden CoT)，模型在思维链里学会自己拆解问题、试错、发现死胡同并掉头重试
  - Test-Time Scaling: 给模型的思考时间 (Reasoning Tokens) 越多，在 AIME 数学竞赛、Codeforces 编程和 GPQA 科学难题上的准确率就越高
  - o3 进一步支持在思维链内部穿插调用工具 (边想边搜网页、写 Python 代码跑计算、裁剪放大看图)
```

## 11.6 Qwen 2.5 / QwQ（Alibaba）

> 论文：[Qwen2](https://arxiv.org/abs/2407.10671) | [Qwen2.5](https://arxiv.org/abs/2412.15115) | 权重：[Qwen](https://huggingface.co/Qwen)

阿里的通义千问（Qwen）是目前开源社区尺寸最全、衍生模型最丰富的模型家族之一。

```
参数尺寸: 稠密模型覆盖 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B，另有 MoE 版本 (如 Qwen2.5-Turbo/Max)
架构细节: GQA, RoPE, SwiGLU, 保留 Attention QKV Bias，采用 152K 大词表
训练数据: 18T tokens，中英文与多语言综合质量高

家族矩阵:
- Qwen2.5-Coder (1.5B - 32B): 专攻代码生成与修复的开源主力
- Qwen2.5-Math: 专攻数学推理与过程奖励模型 (PRM)
- Qwen2-VL / Qwen2.5-VL: 原生动态分辨率与多模态理解标杆
- QwQ-32B: 基于强化学习训练的开源推理模型，32B 体积即可在数学与代码推理上逼近数倍体积的大模型
```

## 11.7 Mistral / Mixtral（Mistral AI）

> 论文：[Mistral 7B](https://arxiv.org/abs/2310.06825) | [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | 权重：[mistralai](https://huggingface.co/mistralai)

欧洲开源代表 Mistral AI 早期推动了滑动窗口注意力与开源 MoE 的普及：

```
Mistral 7B:
  引入滑动窗口注意力 (Sliding Window Attention, SWA) 与 GQA，用 7B 参数打平甚至反超早期的 LLaMA-2-13B

Mixtral 8x7B / 8x22B:
  引爆开源 MoE 热潮的代表作。每层 8 个专家，每个 token 路由激活 2 个专家 (8x7B 总参数 47B，实际激活仅 13B)

Mistral Small 3 / Large 2:
  回归高密度稠密模型 (24B / 123B)，主打低延迟指令遵循、多语言与代码能力
```

## 11.8 端侧小模型（Small Language Models, SLMs）

随着数据清洗与合成数据技术成熟，1B 到 4B 级别的小模型已经能在手机、笔记本电脑甚至浏览器里跑出以前 13B 模型的水平：

| 模型 | 参数量 | 数据与训练特点 | 适用定位 |
|------|--------|---------|------|
| [**Phi-3 / Phi-4**](https://arxiv.org/abs/2404.14219) (Microsoft) | 3.8B / 14B | 重度依赖精筛网页与大模型生成的“教科书级”合成推理语料 | 逻辑、数学与代码得分远超同尺寸模型 |
| [**Gemma 2**](https://arxiv.org/abs/2408.00118) (Google) | 2B / 9B / 27B | 放弃死磕纯 Cross-Entropy 预训练，全程用大模型的输出概率分布做知识蒸馏 (Logit Distillation)，并加入局部滑动注意力与 Logit Soft-Capping | 同体积下对话质感与指令遵循极稳 |
| [**SmolLM2**](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) (HuggingFace) | 135M / 360M / 1.7B | 公开完整的预训练数据配方（FineWeb-Edu、DCLM、Stack-Edu），过训练 11T tokens | 极小体积端侧部署与科研复现参考 |
| [**TinyLlama**](https://github.com/jzhang38/TinyLlama) | 1.1B | 用 3T tokens 把 1.1B 小模型喂饱 | 投机采样草稿模型、教学实验 |

**核心规律**：小模型的容量有限，记不住海量冷门百科事实（World Knowledge），但如果用大模型帮它把脏数据洗干净、改写成推理严密的合成教材，或者直接用大模型做 Logit 蒸馏，小模型也能具备相当强的逻辑推理与读写能力。

## 关键论文

- [OpenAI (2023): GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)：展示预训练算力预测（Predictable Scaling）与多模态评测结果。
- [Mistral AI (2023): Mistral 7B](https://arxiv.org/abs/2310.06825) & [Mixtral of Experts (2024)](https://arxiv.org/abs/2401.04088)：小模型架构优化与开源稀疏 MoE 代表作。
- [Llama Team (2024): The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)：长达 90 多页的工程百科全书，详述了 15T 数据清洗、1.6 万卡集群故障容错与多轮 DPO 后训练。
- [DeepSeek-AI (2024): DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：详解 MLA、无辅助损失 MoE、FP8 训练与 DualPipe 通信重叠。
- [DeepSeek-AI (2025): DeepSeek-R1](https://arxiv.org/abs/2501.12948)：公开如何使用强化学习（GRPO）激发出长思维链推理能力与蒸馏小模型的方法。
- [Qwen Team (2024): Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)：全尺寸开源模型家族的数据配比与后训练细节。

## 进一步阅读

- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)：开源模型客观指标对比。
- [Chatbot Arena (LMArena)](https://lmarena.ai/)：闭源与开源模型同台盲测榜单。
- Sebastian Raschka: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/)：主流开源大模型架构细节图文对比。

## 练习题

1. **精读一篇技术报告**：从 `Llama 3`、`DeepSeek-V3`、`DeepSeek-R1` 或 `Qwen2.5` 的技术报告中任选一篇从头读到尾，整理一份包含数据清洗流水线、模型结构超参数、训练并行策略与后训练阶段划分的读书笔记。
2. **同尺寸模型盲测**：选 10 道你自己平时工作常遇到的真实问题（如写脚本、改 bug、中英互译、总结长文），隐去模型名字对比 `Llama-3.1-8B`、`Qwen2.5-7B` 和 `Gemma-2-9B` 的回答质量。
3. **动手复现一个模块**：挑选 DeepSeek-V3 里的一个具体设计（如无辅助损失的动态偏置路由，或 Multi-Token Prediction 头），用几十行 PyTorch 写一个最小可运行模块。

---

[← 上一章](10-safety-alignment.md) | [目录](../README.md) | [下一章 →](12-distillation-merging.md)
