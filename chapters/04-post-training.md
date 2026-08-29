[← 上一章](03-pretraining.md) | [目录](../README.md) | [下一章 →](05-peft.md)

# 第四章：后训练 (Post-Training)

预训练赋予基座模型以广博的世界知识与统计关联表征；而后训练（Post-Training）则通过指令微调、偏好对齐、可验证强化学习与安全约束，将无约束的自回归文本补全器淬炼为具备意图理解、复杂推演与安全协作能力的实用智能体。

## 4.1 全局流程架构

```
预训练基座模型 (Base Model)
    │
    ▼
有监督指令微调 (SFT, Supervised Fine-Tuning)
    │  - 掌握多轮对话协议、角色扮演与指令遵循规范
    │
    ▼
人类/AI 偏好对齐 (Preference Optimization: RLHF / DPO / GRPO)
    │  - 引导生成分布贴合人类审美、帮助性与逻辑严密性
    │
    ▼
安全与边界对齐 (Safety & Guardrails)
    │  - 拒绝越狱攻击、违法违规与有害输入
    │
    ▼
可部署线上对齐模型 (Aligned / Instruct Model)
```

## 4.2 有监督指令微调 (SFT)

### 4.2.1 数据格式与损失掩码

**标准对话格式 (Chat Template)**：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

**损失掩码机制 (Loss Masking)**：因果自回归模型在计算交叉熵损失时，通过将 Prompt 区域（System 提示词与 User 输入）的 Label 设为 `-100` 进行掩码，仅对 Assistant 生成的回复部分计算梯度反向传播，防止模型退化为 Prompt 记忆器。

### 4.2.2 SFT 语料拓扑

| 语料范畴 | 规模体量 | 质量密度 | 典型代表数据集 |
|---------|---------|---------|--------------|
| 专业人工精标 | 10K–100K | 极致严谨 | 商业对齐专有数据集、专家编写的高难指令 |
| 开源社区精选 | 100K–1M | 优质异构 | [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1), [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| 规则演化合成 | 1M+ | 结构多样 | [Self-Instruct](https://arxiv.org/abs/2212.10560), [Evol-Instruct](https://arxiv.org/abs/2304.12244), [Magpie](https://arxiv.org/abs/2406.08464) |
| 顶尖模型蒸馏 | 1M+ | 高度流利 | 采用 GPT-4/Claude 生成多轮深度推演与解答 |

**LIMA 核心启示** ([Zhou et al., 2023](https://arxiv.org/abs/2305.11206))：“对齐表面化假设”（Superficial Alignment Hypothesis）表明，模型的大部分知识与能力均在预训练阶段习得，SFT 核心在于激活表达范式。仅需约 1000 条高质量、多样化的人工精调指令，即可激发极强的通用指令遵循能力。

### 4.2.3 SFT 核心工程技巧

- **指令多样性（Instruction Diversity）**：覆盖代码推演、形式化数学、结构化信息抽取、角色扮演、创意写作与工具调用等多元分布；
- **拒绝采样微调（Rejection Sampling / Best-of-N Fine-Tuning）**：利用 SFT 中期模型针对每个 Prompt 采样 $N$ 个候选回答，借助奖励模型（Reward Model）或确定性规则打分过滤出最高质量样本重新加入微调集，为 LLaMA 3 后训练的核心迭代机制；
- **超参数设定**：
  - 学习率：$1\times 10^{-5}$ 至 $2\times 10^{-5}$（约为预训练峰值学习率的 1/10 至 1/20）；
  - 训练轮次：2–5 Epochs（小规模精标数据集可微调 3–4 轮，百万级数据集通常 1–2 轮即可收敛）；
  - 批次大小：128–512 样本序列。

### 4.2.4 长上下文针对性 SFT

在长上下文基座上实施微调时需构建特化的长程数据：
- 引入长篇学术文献问答、超长代码库协同理解、全书级多跳推理与长程会议纪要摘要；
- 在微调验证集中嵌入大海捞针（Needle In A Haystack, NIAH）基准，确保在跨越 128K 长度时信息检索召回率逼近 100%。

## 4.3 人类反馈强化学习 (RLHF)

### 4.3.1 经典三阶段流水线

([Ouyang et al., 2022 — InstructGPT](https://arxiv.org/abs/2203.02155))

```
第一阶段: SFT 模型冷启动训练
    构建基准对话模型 π_SFT

第二阶段: 奖励模型 (Reward Model, RM) 训练
    构建成对偏好数据集: (prompt x, 获胜回答 y_w, 失败回答 y_l)
    优化 Bradley-Terry 排序损失使 RM(x, y_w) > RM(x, y_l)

第三阶段: PPO 在线策略梯度优化
    策略模型 π_θ 依据 Prompt 生成回答
    RM 计算偏好得分，结合 KL 散度惩罚构成总奖励
    通过 PPO 算法更新策略权重
```

### 4.3.2 奖励模型机制与挑战

```python
# 奖励模型架构: 复用预训练语言模型骨干，将最终输出头替换为标量线性投影层
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.head = nn.Linear(base_model.config.hidden_size, 1, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        # 取最后一个有效 Token 的隐层表征计算标量奖励
        return self.head(hidden[:, -1, :])

# Bradley-Terry 对数似然损失函数
def bradley_terry_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

**奖励模型的本质痛点**：
- **奖励作弊（Reward Hacking）**：策略模型倾向于探索出奖励模型的局部漏洞（例如盲目堆砌修饰词、无意义拉长篇幅、机械使用连接词），使 RM 给出高分但实际可读性劣化；
- **过优化陷阱（Over-Optimization）**：过度最大化 RM 得分会导致策略模型在真实人类偏好基准上反而大幅回落 ([Gao et al., 2023](https://arxiv.org/abs/2210.10760))；
- **分布偏移（Distribution Shift）**：RM 在离线 SFT 输出上拟合，面对 PPO 探索产生的新分布缺乏可靠泛化。

### 4.3.3 近端策略优化 (PPO)

([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))

```python
# PPO 裁剪目标函数:
L_CLIP(θ) = E_t [ min( r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]

# 重要性采样比率: r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)
# 广义优势估计: A_t 由 Critic 价值网络估算
# 裁剪阈值: ε 通常设为 0.1 ~ 0.2

# 复合奖励函数 (注入 KL 散度惩罚约束):
Reward_total = Reward_RM(x, y) - β * D_KL(π_θ(y|x) || π_ref(y|x))
# π_ref: 冻结的 SFT 参考模型
# β: KL 正则系数，防止策略崩塌
```

**PPO 的工程复杂性**：
- **四模型并发开销**：单次训练需在集群中并发调度当前策略模型（Policy, 更新中）、参考模型（Reference, 冻结）、奖励模型（RM, 冻结）与价值模型（Critic, 更新中）；
- **通信与显存压力**：全量 70B 级别 PPO 训练往往需耗费数百张高端 GPU；
- **工程解法**：采用异步推演流水线（Colocated Rollout）、vLLM 高并发批处理集成与 Critic/Policy 共享主干架构。

### 4.3.4 轻量强化学习演进：RLOO 与 GRPO

**REINFORCE Leave-One-Out (RLOO)** ([Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740))：
- 针对每个 Prompt 采样 $K$ 个回答，以其余 $K-1$ 个回答的平均奖励作为基线，直接计算优势值；
- 彻底摒弃独立的 Critic 价值网络，显存与计算开销减少近半，为 LLaMA 3 等前沿项目所采纳。

**组相对策略优化 (Group Relative Policy Optimization, GRPO)** ([Shao et al., 2024](https://arxiv.org/abs/2402.03300))：
- DeepSeekMath 与 DeepSeek-R1 的核心优化算子；
- 对每个 Prompt 采样一组回答序列 $\{y_1, y_2, \dots, y_G\}$，计算组内奖励的均值与方差并完成标准化：
  $$A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$$
- 无需 Critic 模型，极大简化超大集群分布式强化学态。

## 4.4 直接偏好优化 (Direct Preference Optimization, DPO)

### 4.4.1 数学机理

([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))

经典 RLHF 流程较为繁复：需预先训练独立的奖励模型，并在 PPO 在线采样中协同调度策略模型、参考模型、奖励模型与价值模型共 4 组网络。

**DPO 的理论突破**：从带有 KL 正则项的强化学习最优解出发，反解出奖励函数与最优策略之间的封闭解析关系：
$$r^*(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

将此形式代入 Bradley-Terry 偏好似然中，常数配分函数 $Z(x)$ 自然抵消，直接推导出端到端的监督分类损失：

```python
# DPO 损失函数形式:
L_DPO = -E_{(x, y_w, y_l)} [ log σ( β * ( log(π_θ(y_w|x) / π_ref(y_w|x)) 
                                         - log(π_θ(y_l|x) / π_ref(y_l|x)) ) ) ]

# y_w: 人类或判定胜出的优选回答 (Chosen)
# y_l: 被淘汰的次优回答 (Rejected)
# π_ref: 冻结的基准 SFT 参考模型
# β: 偏好强度调节温度超参 (通常设为 0.1 ~ 0.5)
```

**直觉解释**：自适应增大胜出样本 $y_w$ 相对于参考模型的隐式奖励，压低淘汰样本 $y_l$ 的相对几率。

### 4.4.2 现代 DPO 衍生变体

| 变体名称 | 核心设计改进 | 适用场景与代表文献 |
|---------|------------|------------------|
| **IPO** | 引入均方误差替代 Sigmoid，避免对偏好标签产生极端过拟合 | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) |
| **KTO** | 仅依赖单样本的独立好/坏二元信号，无需严格配对数据 | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) |
| **ORPO** | 将单步因果似然与逐样本 Odds Ratio 损失直接融合，免除参考模型 | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) |
| **SimPO** | 采用序列归一化的对数概率差异作为隐式奖励，消除长度偏置 | [Meng et al., 2024](https://arxiv.org/abs/2405.14734) |

### 4.4.3 离线与迭代式在线 DPO

- **离线 DPO (Offline DPO)**：直接基于静态固定的偏好数据集训练，实现简单但容易受困于分布外泛化瓶颈；
- **迭代式在线 DPO (Online / Iterative DPO)** ([Xu et al., 2024](https://arxiv.org/abs/2404.07503))：
  ```
  迭代循环:
      1. 利用当前最新策略模型 π_θ 针对 Prompt 采样生成候选回答
      2. 利用高阶裁判模型或奖励模型对候选进行动态成对排序
      3. 基于最新偏好数据计算 DPO 损失并更新策略 π_θ
  ```
  该模式有效弥合了离线偏好与动态生成之间的分布鸿沟，效果逼近完整 PPO 流程。

## 4.5 宪法人工智能 (Constitutional AI, CAI)

([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) Anthropic 提出的原则驱动对齐框架，利用显式规则集实现自监督反思与自我对齐：

```
步骤一: 诱导对抗生成 (Red-Teaming)
    引导基座模型针对潜在敏感或恶意请求生成初步回答

步骤二: 规则批评与自我迭代修改 (Critique & Revision)
    模型依据宪法原则（如“请基于无害性与客观中立原则修改上述回答”）进行自我批判并改写为安全版本

步骤三: 机器反馈强化学习 (RLAIF)
    利用自我修正数据训练奖励模型，并依托 PPO/DPO 完成规模化自对齐
```

## 4.6 可验证奖励强化学习 (RLVR)

**[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 范式革新**：在形式化数学、逻辑推演与单元测试代码等领域，使用客观确定的真实正确性（Ground Truth）作为可验证奖励信号，彻底消除对主观人类偏好标注的依赖。

```
形式化数学推导场景:
    1. 模型自回归生成完整的长思维链 (Chain-of-Thought) 与最终结论
    2. 符号判题器或 Lean 证明器自动验证结论真实性 → 获得 1 或 0 离散奖励
    3. 结合格式奖励 (如严格包裹 <think></think> 标签) 进行 GRPO 优化

代码算法生成场景:
    1. 模型生成完整函数与算法实现
    2. 沙盒环境自动编译并执行单元测试用例 → 奖励 = 测试通过率 (Pass Rate)
    3. 通过 GRPO 策略梯度反向更新网络参数
```

**纯强化学习的长思维链涌现**：DeepSeek-R1-Zero 证实，在完全不依赖人工精标 SFT 数据的前提下，纯粹依托可验证奖励与大规模强化学习，模型能够自发涌现反思、回溯试错、多角度验证与超长链条推理能力。

## 4.7 深度推理模型与测试期算力扩展

### 4.7.1 显式思维链 (Long CoT) 与过程奖励

**OpenAI o1/o3** 与 **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)** 确立的推理架构：
```
传统生成: Prompt → 直接输出最终解答
推理模型: Prompt → <think>长程多步推导、回溯、反思、验算</think> → 精炼解答
```

**过程奖励模型 (Process Reward Model, PRM)** ([Lightman et al., 2023](https://arxiv.org/abs/2305.20050))：
- 相比于仅对最终答案打分的结果奖励模型（Outcome Reward Model, ORM），PRM 对思维链中的每一个推理步骤实施细粒度打分；
- 有效识别复杂推导中“步骤错误但巧合蒙对答案”的假阳性现象，为蒙特卡洛树搜索（MCTS）提供高精度的分支剪枝引导。

### 4.7.2 测试期算力标度律 (Test-Time Compute Scaling)

突破传统“参数量 + 训练数据”扩展的单一边界，在推理推演阶段动态扩展计算量以换取解题精度的跃升 ([Snell et al., 2024](https://arxiv.org/abs/2408.03314))：

```
计算扩展双轮驱动:
  训练期标度律 (Train-Time Scaling): 扩展参数量与预训练 Token 预算
  测试期标度律 (Test-Time Scaling):  在推理阶段动态扩展采样步数与搜索深度

典型推演扩展路径:
- 组采样多数表决 (Self-Consistency / Majority Voting): 采样多条独立路径并聚合众数
- 树搜索推理 (MCTS / Beam Search over PRM): 在高维推演空间进行剪枝与最优搜索
- 自适应反思修订 (Iterative Self-Correction): 依据验证器反馈在上下文内部循环纠错
```

## 4.8 工具调用与智能体强化 (Agent Training)

### 4.8.1 函数调用微调 (Function Calling)

训练模型将自然语言意图精准解析为结构化的 JSON/API 调用契约：

```json
{"name": "search", "arguments": {"query": "weather in Stockholm"}}
```

**轨迹训练机制**：构建涵盖 `(User Prompt, Model Tool Call, System Tool Response, Model Final Answer)` 的多轮状态转移轨迹（Trajectories），实施端到端指令微调。

### 4.8.2 外部代码执行环境协同 (Program-Aided LLMs)

```
User: 计算 7 的 23 次方。
Model: <code>print(7**23)</code>
Environment: [执行输出: 27368747340080916343]
Model: 7 的 23 次方等于 27368747340080916343。
```

### 4.8.3 复杂多步智能体 (Multi-step Agent)

依托 ReAct 范式构建感知-思考-行动闭环：
```
Observation (观察) → Thought (思考) → Action (行动) → Observation (新状态) → ... → Final Response
```

**软件工程基准（[SWE-bench](https://www.swebench.com/)）导向训练**：
- 给定真实的 GitHub Issue 描述与完整代码仓库上下文；
- 模型自主执行跨文件索引、补丁修改、执行本地测试用例；
- 以单元测试通过率作为确定性奖励闭环微调。

## 关键论文

- [Ouyang et al. (2022) — InstructGPT](https://arxiv.org/abs/2203.02155): 经典 RLHF 三阶段框架奠基
- [Rafailov et al. (2023) — DPO](https://arxiv.org/abs/2305.18290): 跳过显式奖励模型的直接偏好优化
- [Shao et al. (2024) — DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300): 免 Critic 组相对策略优化
- [Bai et al. (2022) — Constitutional AI](https://arxiv.org/abs/2212.08073): 原则驱动的自监督机器对齐
- [DeepSeek-AI (2025) — DeepSeek-R1](https://arxiv.org/abs/2501.12948): 纯可验证强化学习激发长思维链推理

## 进阶参考

- HuggingFace: [TRL 官方库](https://github.com/huggingface/trl)（工业级 SFT / DPO / GRPO / PPO 训练套件）
- Nathan Lambert: [Interconnects.ai](https://www.interconnects.ai/)（深入剖析后训练技术演进的专业专栏）
- Lightman et al.: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)（过程奖励模型 PRM 权威论著）

## 实践训练

1. **SFT 对齐实战**：使用 TRL 库在清洗后的指令数据集上微调 1.5B 参数开源基座，并在标准对话评测集上量化指令遵循收益。
2. **DPO 与 SFT 效果消融**：在统一基线模型上分别运行 SFT 与在线/离线 DPO，通过成对盲测（LLM-as-a-Judge）评估胜率（Win Rate）迁移。
3. **可验证奖励强化学习复现**：针对初等代数或逻辑谜题，基于 GRPO 算法实现免 Critic 的自适应强化学习训练，观察模型输出中自发反思与推理链长度的演进趋势。

---

[← 上一章](03-pretraining.md) | [目录](../README.md) | [下一章 →](05-peft.md)
