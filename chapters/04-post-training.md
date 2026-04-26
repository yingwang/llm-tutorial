[← 上一章](03-pretraining.md) | [目录](../README.md) | [下一章 →](05-peft.md)

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

## 关键论文

- [Ouyang et al. (2022) — InstructGPT](https://arxiv.org/abs/2203.02155) — 经典 RLHF 三阶段
- [Rafailov et al. (2023) — DPO](https://arxiv.org/abs/2305.18290) — 跳过 reward model 的偏好优化
- [Shao et al. (2024) — DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) — DeepSeek-R1 用的 RL 算法
- [Bai et al. (2022) — Constitutional AI](https://arxiv.org/abs/2212.08073) — RLAIF 与原则驱动对齐
- [Bai et al. (2022) — HH-RLHF](https://arxiv.org/abs/2204.05862) — Helpful & Harmless 数据集与方法

## 进一步阅读

- HuggingFace — [TRL 库](https://github.com/huggingface/trl)：SFT / DPO / GRPO / PPO 工业实现
- Nathan Lambert — [Interconnects.ai](https://www.interconnects.ai/)：post-training 持续追踪
- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)：纯 RL 涌现推理能力

## 练习题

1. **SFT 微调**：用 TRL 在 Alpaca-cleaned 上对一个 1.5B base 模型做 SFT，对比 base 与 SFT 模型在 MT-Bench 上的差异。
2. **DPO 实验**：用相同 SFT 检查点 + UltraFeedback 数据跑 DPO；观察 win rate 变化。
3. **奖励黑客观察**：跑 PPO 时故意把 reward model 训得不充分，看模型如何"骗"reward（重复、特殊字符、夸张回答）。

---

[← 上一章](03-pretraining.md) | [目录](../README.md) | [下一章 →](05-peft.md)
