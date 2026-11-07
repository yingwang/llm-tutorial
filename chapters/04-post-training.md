[← 上一章](03-pretraining.md) | [目录](../README.md) | [下一章 →](05-peft.md)

# 第四章：Post-Training（后训练）

预训练出来的基座模型本质上只是一个“文本续写器”：给它一段话，它只会顺着往下接，甚至可能拿另一个问题来接你的问题。后训练（Post-Training）要做的，就是通过指令微调、偏好对齐、强化学习与安全训练，把这个续写器变成听得懂指令、答得准问题、守得住边界的助手。

## 4.1 概览

```
Base Model (预训练)
    │
    ▼
SFT (Supervised Fine-Tuning)
    │  - 学会按指令格式回答
    │
    ▼
Preference Optimization (RLHF / DPO / GRPO)
    │  - 学会区分什么是"好"回答、强化复杂推理
    │
    ▼
Safety Training
    │  - 拒绝有害请求与越狱诱导
    │
    ▼
Deployed Model
```

## 4.2 SFT (Supervised Fine-Tuning)

### 4.2.1 数据格式

**Chat Format**（标准多轮对话模板）：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

**Loss Masking（损失掩码）**：计算交叉熵损失时，只对 `assistant` 回复部分的 Token 算 Loss（把 `system` 和 `user` 部分的标签设为 `-100` 忽略掉），让模型专心学怎么回答，而不是去背用户的提问。

### 4.2.2 SFT 数据来源

| 来源 | 规模 | 质量 | 例子 |
|------|------|------|------|
| 人工标注 | 10K-100K | 极高 | OpenAI / Anthropic 内部专家标注数据 |
| 开源数据集 | 100K-1M | 中-高 | [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1), [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| 规则演化合成 | 1M+ | 中-高 | [Self-Instruct](https://arxiv.org/abs/2212.10560), [Evol-Instruct](https://arxiv.org/abs/2304.12244), [Magpie](https://arxiv.org/abs/2406.08464) |
| 强模型蒸馏 | 1M+ | 高 | 用 GPT-4 / Claude / DeepSeek-R1 生成多轮回答与推导过程 |

**LIMA 的发现**（[Zhou et al., 2023](https://arxiv.org/abs/2305.11206)）：论文提出“对齐表面化假说”（Superficial Alignment Hypothesis），认为模型的知识和能力绝大部分在预训练阶段就已经学到了，SFT 只是教模型用什么样的格式和语气把知识调出来。实验表明，仅用 1000 条精挑细选的高质量样本做 SFT，就能训练出对答相当得体的对话模型。在 SFT 阶段，数据质量远比单纯堆数量重要。

### 4.2.3 SFT 技巧

- **指令多样性（Instruction Diversity）**：Prompt 覆盖的任务类型越丰富越好，应涵盖代码编写、数学解题、摘要提取、翻译、创意写作、角色扮演、结构化 JSON 输出与工具调用等不同场景；
- **拒绝采样（Rejection Sampling）**：对同一个 Prompt 让当前模型采样生成 $N$ 个候选回答，再用奖励模型（Reward Model）或校验规则挑出得分最高的一条放回 SFT 训练集。这是 LLaMA 2/3 和 DeepSeek 后训练阶段反复使用的核心提纯技巧；
- **常用超参**：
  - 学习率：`1e-5` ~ `2e-5`（比预训练峰值学习率低一个数量级，防止冲掉预训练知识）；
  - 训练轮数：2–5 个 Epoch（精选小数据集训 3–4 轮，百万级合成数据通常 1–2 轮即可，训多了容易过拟合）；
  - Batch Size：通常为 128–512 条样本（常将多条短对话拼接 Packing 成一条长序列以提高吞吐）。

### 4.2.4 Long-Context SFT

如果基座模型把上下文窗口扩到了 128K，SFT 阶段也必须跟上专门的长文本指令数据：
- 加入长文档问答、多文档对比摘要、跨文件代码库理解等真实长输入任务（一般混入 1%–5% 的长样本，其余仍保持短对话以防短任务退化）；
- 用“大海捞针”（Needle in a Haystack）测试检验模型在长窗口各个深度位置的信息提取准确率。

## 4.3 RLHF (Reinforcement Learning from Human Feedback)

### 4.3.1 完整 Pipeline

（[Ouyang et al., 2022: InstructGPT](https://arxiv.org/abs/2203.02155)）

```
Step 1: Reward Model Training
    收集人类偏好数据: (prompt, chosen, rejected)
    训练 RM: 让 RM(chosen) > RM(rejected)

Step 2: PPO Training
    对每个 prompt:
        1. 当前策略 π_θ 生成回答
        2. RM 给回答打分
        3. 用 PPO 优化策略，最大化 reward 同时用 KL 惩罚约束它不偏离 SFT 模型太远
```

### 4.3.2 Reward Model

奖励模型（RM）通常直接在 SFT 模型的基础上改造，把顶部的词表预测头（LM Head）换成一个输出标量分数的线性层：

```python
# RM 架构: 和 LLM 一样的 Transformer，最后一层换成一个 scalar head
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.head = nn.Linear(hidden_size, 1)  # 输出标量 reward
    
    def forward(self, input_ids):
        hidden = self.backbone(input_ids).last_hidden_state[:, -1, :]
        return self.head(hidden)  # 取最后一个 token 的隐状态算分

# Bradley-Terry 成对排序损失
def bt_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

**偏好数据怎么收集**：
- 给标注员看同一个 Prompt 下的两个（或多个）模型回答，让他们选出更好的一个；
- 每个样本通常会分给 3–5 名标注员独立判断，取多数一致的结果入库。

**Reward Model 常见的问题**：
- **钻奖励空子（Reward Hacking）**：策略模型在强化学习过程中很容易摸出 RM 的评分癖好，比如不管问什么都写得又臭又长、拼命列序号、堆砌漂亮的过渡词，把 RM 分数刷得很高，实际回答却变得啰嗦空洞；
- **过优化（Over-optimization）**：随着 PPO 把 RM 得分越推越高，真实人类评估效果会先升后降（[Gao et al., 2023](https://arxiv.org/abs/2210.10760)）；
- **分布偏移（Distribution Shift）**：RM 是在旧版 SFT 模型的输出分布上训练出来的，等策略模型更新几百步之后，面对新分布下的古怪输出，RM 往往会打错分。

### 4.3.3 PPO (Proximal Policy Optimization)

（[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)）

```python
# PPO objective:
L = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

# r_t = π_θ(a|s) / π_θ_old(a|s)  # 重要性采样比率
# A_t = advantage estimate (由价值网络 Critic 配合 GAE 估计)
# ε = 0.2 typically (裁剪幅度范围，防止单步更新过猛)

# KL 惩罚项防止策略模型偏离 SFT 基准太远:
reward_total = reward_rm - β * KL(π_θ || π_ref)
# π_ref = SFT model (冻结参数)
# β = 0.01-0.1
```

**PPO 的工程难点**：
- **同时维护 4 个模型**：显存里需要同时放策略模型 Policy（可训练）、参考模型 Reference（冻结）、奖励模型 RM（冻结）和价值模型 Critic（可训练）；
- **显存与通信开销巨大**：训练一个 70B 模型的 PPO 流水线往往要动用上百张到数百张 A100/H100；
- **超参敏感且难调**：Rollout 生成速度慢（受限于自回归解码），现代框架（如 OpenRLHF、veRL）通常把 vLLM 推理引擎和训练引擎混部在同一批卡上，异步完成采样与反向传播。

### 4.3.4 轻量化替代方案：RLOO 与 GRPO

既然 PPO 里的 Critic 价值模型既占显存又难训，能不能直接把 Critic 砍掉？

**REINFORCE Leave-One-Out (RLOO)**（[Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)）：
- 对每个 Prompt 采样生成 $K$ 个回答，算第 $i$ 个回答的优势（Advantage）时，直接拿其余 $K-1$ 个回答的平均得分当基线（Baseline）；
- 完全省去了 Critic 模型，显存省下一大截，训练也更稳。

**GRPO (Group Relative Policy Optimization)**（[Shao et al., 2024: DeepSeekMath](https://arxiv.org/abs/2402.03300)）：
- DeepSeekMath 与 DeepSeek-R1 采用的核心强化学习算法；
- 同样去掉了独立的 Critic 模型，对每个 Prompt 采样一组回答 $\{y_1, y_2, \dots, y_G\}$，拿到各自的奖励分数 $\{r_1, r_2, \dots, r_G\}$ 后，直接用组内的均值和标准差做归一化来充当优势函数：
  $$A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$$
- 特别适合配合规则判题器做大规模推理强化学习。

## 4.4 DPO (Direct Preference Optimization)

### 4.4.1 核心思想

（[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)）

**痛点**：标准 RLHF 流程太重，先得训一个 RM，再搭一套 4 模型的 PPO 在线强化学习系统。

**DPO 的推导思路**：带 KL 散度约束的强化学习目标函数其实有解析最优解，奖励函数 $r(x, y)$ 可以直接用最优策略 $\pi^*$ 和参考策略 $\pi_{\text{ref}}$ 的对数概率比表示出来：
$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

把这个式子代进 Bradley-Terry 成对偏好损失里，配分函数 $Z(x)$ 刚好一正一负抵消掉，整个偏好优化就直接变成了一个简单的有监督分类 Loss：

```python
# DPO loss:
L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) 
                     - log π_θ(y_l|x)/π_ref(y_l|x)))]

# y_w = chosen (更优回答)
# y_l = rejected (较差回答)
# π_ref = reference policy (冻结的 SFT 模型)
# β = 温度系数 (通常取 0.1-0.5)
```

**直觉**：训练时同时只驻留策略模型和参考模型两个网络，直接拉高好回答 $y_w$ 相对于参考模型的概率比、压低差回答 $y_l$ 的概率比，不需要单独训 RM，也不需要在训练循环里做慢吞吞的在线采样。

### 4.4.2 DPO 变体

| 变体 | 改进点 | 论文 |
|------|--------|------|
| **IPO** | 加入正则项防止对成对偏好标签过拟合 | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) |
| **KTO** | 只需要单条回答的“好/坏”标签，不要求成对比较数据 | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) |
| **ORPO** | 把 SFT 损失与优势比（Odds Ratio）合二为一，连参考模型也省掉 | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) |
| **SimPO** | 用序列长度平均的对数概率代替与参考模型的比值，消除长度偏置且更省显存 | [Meng et al., 2024](https://arxiv.org/abs/2405.14734) |

### 4.4.3 Online DPO vs Offline DPO

- **Offline DPO**：直接拿现成的固定偏好数据集跑完训练。实现最简单，但因为数据不是当前策略模型自己生成的，训到一定程度容易遇到分布偏移上限。
- **Online / Iterative DPO**（[Xu et al., 2024](https://arxiv.org/abs/2404.07503)）：
  ```
  循环多轮:
      1. 用当前策略模型 π_θ 对一批 Prompt 生成多个候选回答
      2. 用奖励模型（或强裁判模型）给候选回答打分，拼成新的 (chosen, rejected) 对
      3. 在这批新鲜的偏好对上用 DPO Loss 更新 π_θ
  ```
  通过分轮次在线采样刷新偏好数据，既保留了 DPO 训练稳定的优点，又把最终效果拉到了接近在线 RL 的水平（LLaMA 3 便采用了多轮迭代式的偏好优化）。

## 4.5 Constitutional AI (CAI)

（[Bai et al., 2022](https://arxiv.org/abs/2212.08073)）Anthropic 提出的对齐方法，核心是用一套事先写好的“章程原则”（Constitution）代替大量人工标注，让模型自己批评、自己改写：

```
Step 1: Red-teaming (诱导提问)
    用带攻击性或敏感的 Prompt 让初始模型生成可能有害的回答

Step 2: Critique & Revision (自我批评与改写)
    把章程原则（如“挑出更没有伤害、更客观中立的回答”）喂给模型，
    让模型指出刚才回答里的问题，并改写出一版安全合规的回答 -> 用于 SFT

Step 3: RL from AI Feedback (RLAIF)
    让模型对照章程原则对成对回答打分，训练出偏好奖励模型，再跑强化学习对齐
```

## 4.6 RLVR (Reinforcement Learning with Verifiable Rewards)

**[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 的关键突破**：在数学、代码、逻辑推理这类有客观标准答案的任务上，完全不用容易被钻空子的人类偏好神经网络 RM，而是直接拿**可验证的客观结果**（Verifiable Rewards）当奖励信号：

```
对于数学题:
    1. 模型生成思考过程 <think>...</think> 和最终答案
    2. 判题规则直接核对最终答案对不对 → 对得 1 分，错得 0 分（外加少量格式分）
    3. 用 GRPO 算法更新策略模型

对于代码题:
    1. 模型生成代码解答
    2. 丢进沙盒跑单元测试 → 按测试用例通过率给奖励分
    3. 用 GRPO 算法更新策略模型
```

**重要发现**：DeepSeek-R1-Zero 证明了哪怕跳过 SFT 阶段，直接从预训练 Base 模型出发做大规模可验证强化学习（RLVR），模型为了拿高分也会自发拉长思考过程，学会在 `<think>` 标签里反复验算、自我纠错和回溯排查。

## 4.7 Reasoning Models

### 4.7.1 Chain-of-Thought Training

以 **OpenAI o1/o3** 和 **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)** 为代表的推理模型，把过去靠 Prompt 诱导的思维链（CoT）内化成了模型的原生生成习惯：

```
传统对话模型: prompt → 直接输出答案
深度推理模型: prompt → <think>多步拆解、试错、反思、验算</think> → 最终答案
```

常用的训练与搜索手段包括：
1. **RLVR（可验证结果奖励）**：只管最终答案对不对和格式合不合规，靠 GRPO 等算法让模型自己在试错中摸索出有效的长思维链；
2. **Process Reward Model (PRM)**（[Lightman et al., 2023](https://arxiv.org/abs/2305.20050)）：给推导过程中的每一步单独打分，而不是只看最后结果（Outcome Reward Model, ORM），能及时揪出“中间过程算错、最后碰巧蒙对”的情况；
3. **搜索与拒绝采样**：结合蒙特卡洛树搜索（MCTS）或 Best-of-N 采样生成高质量长推理轨迹，再用来反哺模型或蒸馏给小模型。

### 4.7.2 Test-Time Compute Scaling

（[Snell et al., 2024](https://arxiv.org/abs/2408.03314)）过去的 Scaling Law 拼的是训练阶段投入多少算力；推理模型则开辟了第二条轴线：在推理（测试）阶段遇到难题时，让模型多花一点计算时间去“多想一会儿”，用推理算力换取更高的准确率。

```
传统 scaling: 增大模型参数与预训练数据 → 提升能力
测试期 scaling: 增大推理时的思考 token 数或采样数 → 提升难题准确率

常见做法:
- Long Chain-of-Thought: 让模型在回答前生成几千甚至上万 token 的自我推演与验算
- Majority Voting (Self-Consistency): 独立采样 N 条推理路径，对最终答案投票取众数
- Best-of-N: 生成 N 个候选解答，用奖励模型或验证器挑出最高分的一个
- Iterative Refinement: 根据编译器报错或测试反馈，在上下文中多轮修改答案
```

## 4.8 Tool Use & Agent Training

### 4.8.1 Function Calling

教模型在遇到需要查外部资料或调接口的请求时，不再硬编答案，而是按约定的 JSON Schema 输出结构化的函数调用：

```json
{"name": "search", "arguments": {"query": "weather in Stockholm"}}
```

**训练数据构造**：收集或合成包含 `(用户提问, 模型发起 tool_call, 环境返回 tool_result, 模型汇总给出最终回答)` 的完整多步交互轨迹做 SFT 和强化学习。

### 4.8.2 Code Execution

遇到复杂算术、数据分析或画图需求时，让模型直接写一段 Python 代码交给沙盒运行，拿到真实运行结果后再继续往下推理：

```
User: What is 7^23?
Model: <code>print(7**23)</code>
System: [Execution result: 27368747340080916343]
Model: 7^23 = 27368747340080916343
```

### 4.8.3 Multi-step Agent

训练模型在真实环境里循环执行“观察 → 思考 → 行动”（ReAct 模式），自主完成需要几十步操作的复杂任务：

```
Observation → Thought → Action → Observation → Thought → Action → ... → Answer
```

**[SWE-Agent](https://github.com/princeton-nlp/SWE-agent) / [SWE-bench](https://www.swebench.com/)** 是代码智能体训练的典型场景：
- 给模型一个真实的 GitHub Issue 描述和整个代码仓库；
- 模型自己调命令行搜索代码、定位 Bug、修改文件并运行测试；
- 最终以仓库单元测试是否通过作为 RL 训练的客观奖励信号。

## 关键论文

- [Ouyang et al. (2022): InstructGPT](https://arxiv.org/abs/2203.02155)：奠定 SFT + RM + PPO 经典 RLHF 三阶段流程
- [Rafailov et al. (2023): Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)：跳过独立奖励模型，直接用成对偏好数据优化策略
- [Shao et al. (2024): DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)：提出免 Critic 的组相对策略优化（GRPO）
- [Bai et al. (2022): Constitutional AI](https://arxiv.org/abs/2212.08073)：用原则驱动的自我批评与 RLAIF 替代纯人工标注
- [DeepSeek-AI (2025): DeepSeek-R1](https://arxiv.org/abs/2501.12948)：通过大规模可验证奖励强化学习（RLVR）激发长思维链推理能力

## 进一步阅读

- HuggingFace：[TRL 库](https://github.com/huggingface/trl)，集成了 SFT、DPO、GRPO、PPO、KTO 的主流开源后训练库
- Nathan Lambert：[Interconnects.ai](https://www.interconnects.ai/)，长期跟踪拆解 RLHF 与后训练最新进展的技术博客与 [RLHF Book](https://rlhfbook.com/)
- Lightman et al. (2023)：[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)，对比过程奖励模型（PRM）与结果奖励模型（ORM）的经典研究

## 练习题

1. **SFT 微调实战**：使用 HuggingFace TRL 在精选指令集（如 Alpaca-cleaned 或 UltraChat 子集）上对一个 1.5B Base 模型做 SFT（注意开启 Prompt Loss Masking），对比微调前后的多轮问答表现。
2. **DPO 对齐实验**：基于上一题得到的 SFT 检查点，在 UltraFeedback 偏好数据集上跑一轮 DPO，观察模型回答风格的变化以及不同 $\beta$ 值（如 0.05 vs 0.5）的影响。
3. **GRPO 算术推理实验**：用 TRL 的 `GRPOTrainer` 在 GSM8K 数学题上对一个 1.5B/3B 模型做规则奖励强化学习（答案算对得 1 分，包含 `<think>...</think>` 标签得 0.2 分），观察模型输出长度与准确率随训练步数的变化。

---

[← 上一章](03-pretraining.md) | [目录](../README.md) | [下一章 →](05-peft.md)
