[← Previous Chapter](03-pretraining.md) | [Table of Contents](README.md) | [Next Chapter →](05-peft.md)

# Chapter 4: Post-Training

Pretraining produces a "text completer" — post-training turns it into a useful "assistant."

## 4.1 Overview

```
Base Model (pretrained)
    │
    ▼
SFT (Supervised Fine-Tuning)
    │  - Learn to respond in instruction format
    │
    ▼
Preference Optimization (RLHF / DPO / ...)
    │  - Learn what makes a "good" response
    │
    ▼
Safety Training
    │  - Refuse harmful requests
    │
    ▼
Deployed Model
```

## 4.2 SFT (Supervised Fine-Tuning)

### 4.2.1 Data Format

**Chat Format** (standard):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

**Loss masking**: Only compute loss on the assistant's responses, not on the system or user messages.

### 4.2.2 SFT Data Sources

| Source | Scale | Quality | Examples |
|--------|-------|---------|----------|
| Human annotation | 10K-100K | Very high | OpenAI's internal data, Anthropic's annotated data |
| Open-source datasets | 100K-1M | Medium-High | [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1), [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| Synthetic | 1M+ | Medium | [Self-Instruct](https://arxiv.org/abs/2212.10560), [Evol-Instruct](https://arxiv.org/abs/2304.12244), [Magpie](https://arxiv.org/abs/2406.08464) |
| Distillation | 1M+ | Medium-High | Use GPT-4/Claude to generate responses |

**LIMA finding** ([Zhou et al., 2023](https://arxiv.org/abs/2305.11206)): "Less Is More for Alignment" — just 1000 high-quality SFT examples can give a model decent conversational ability. Quality > quantity.

### 4.2.3 SFT Tips

**Instruction Diversity**: More important than response quality. Cover: code, math, creative writing, summarization, translation, role-playing, tool use...

**Rejection Sampling**: Generate N responses per prompt, select the best one using a reward model or rules. This is a key technique Meta used to train LLaMA 3.

**Hyperparameters**:
- Learning rate: 1e-5 ~ 2e-5 (10-100x lower than pretraining)
- Epochs: 2-5 (more epochs for less data, 1-2 epochs for more data)
- Batch size: 128-512 samples

### 4.2.4 Long-Context SFT

Specialized SFT for long-context capability:
- Include long-document QA, multi-document summarization, long code comprehension tasks
- Training data length distribution should cover the target context window
- "Needle in a haystack" test to verify long-context capability

## 4.3 RLHF (Reinforcement Learning from Human Feedback)

### 4.3.1 Full Pipeline

([Ouyang et al., 2022 — InstructGPT](https://arxiv.org/abs/2203.02155))

```
Step 1: Reward Model Training
    Collect human preference data: (prompt, chosen, rejected)
    Train RM: RM(chosen) > RM(rejected)

Step 2: PPO Training
    For each prompt:
        1. Current policy π_θ generates a response
        2. RM scores the response
        3. Use PPO to optimize the policy, maximizing reward while staying close to the SFT model
```

### 4.3.2 Reward Model

```python
# RM architecture: same Transformer as the LLM, with final layer replaced by a scalar head
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.backbone = base_model
        self.head = nn.Linear(hidden_size, 1)  # output scalar reward
    
    def forward(self, input_ids):
        hidden = self.backbone(input_ids).last_hidden_state[:, -1, :]
        return self.head(hidden)  # scalar reward

# Bradley-Terry Loss
def bt_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

**Preference data collection**:
- Annotators see two responses to the same prompt and choose the better one
- Can use Likert scale (1-7) or ranking (rank multiple responses)
- Each prompt typically needs 3-5 annotators, with majority vote

**RM issues**:
- **Reward hacking**: The model learns to game the RM rather than genuinely improving (e.g., always giving long responses, using filler connectives)
- **Over-optimization**: Pushing the RM score too far causes performance to degrade ([Gao et al., 2023](https://arxiv.org/abs/2210.10760))
- **Distribution shift**: The RM is trained on SFT model outputs and may be inaccurate on post-PPO outputs

### 4.3.3 PPO (Proximal Policy Optimization)

([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))

```python
# PPO objective:
L = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

# r_t = π_θ(a|s) / π_θ_old(a|s)  # importance sampling ratio
# A_t = advantage estimate (reward - baseline)
# ε = 0.2 typically (clipping range)

# KL penalty to prevent the model from drifting too far from the SFT baseline:
reward_total = reward_rm - β * KL(π_θ || π_ref)
# π_ref = SFT model (frozen)
# β = 0.01-0.1
```

**PPO engineering challenges**:
- Requires running 4 models simultaneously: policy (being trained), reference policy (frozen), reward model, value model
- Extremely high memory demand (a 70B model needs ~512 A100s)
- Unstable training, sensitive hyperparameters
- Extensive engineering optimizations: async generation, vLLM for inference acceleration, sharing critic model and policy model

### 4.3.4 Practical Alternatives

**REINFORCE Leave-One-Out (RLOO)** ([Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)):
- A simpler alternative to PPO
- Generate K responses per prompt, use leave-one-out baseline to estimate advantage
- No value model needed, halving memory requirements
- DeepSeek and LLaMA 3 actually use RLOO variants

**GRPO (Group Relative Policy Optimization)** ([Shao et al., 2024 — DeepSeekMath](https://arxiv.org/abs/2402.03300)):
- Used by DeepSeek-R1
- Sample a group of responses per prompt
- Use within-group relative ranking as the reward signal
- No external reward model needed

## 4.4 DPO (Direct Preference Optimization)

### 4.4.1 Core Idea

([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))

**Problem**: RLHF is too complex — it requires training an RM, running PPO, and managing 4 models.

**DPO's insight**: The RM + PPO can be collapsed into a simple supervised loss.

```python
# DPO loss:
L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) 
                     - log π_θ(y_l|x)/π_ref(y_l|x)))]

# y_w = chosen (winning) response
# y_l = rejected (losing) response
# π_ref = reference policy (SFT model, frozen)
# β = temperature (typically 0.1-0.5)
```

**Intuition**: Increase the probability of the chosen response, decrease the probability of the rejected response, with magnitude controlled by β. The reference model prevents excessive drift.

### 4.4.2 DPO Variants

| Variant | Improvement | Paper |
|---------|-------------|-------|
| **IPO** | Prevents overfitting to preference data | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) |
| **KTO** | Only needs good/bad labels, no pairs required | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) |
| **ORPO** | No reference model needed | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) |
| **SimPO** | Uses sequence-average log-prob as reward | [Meng et al., 2024](https://arxiv.org/abs/2405.14734) |

### 4.4.3 Online DPO vs Offline DPO

**Offline DPO**: Train on preference data generated by the SFT model → simple but has a performance ceiling

**Online/Iterative DPO** ([Xu et al., 2024](https://arxiv.org/abs/2404.07503)):
```
Loop:
    1. Generate responses with current policy π_θ
    2. Label preferences with RM (or humans / stronger model)
    3. Update π_θ with DPO loss
```
- Performance approaches RLHF
- Solves the distribution shift problem of offline DPO

## 4.5 Constitutional AI (CAI)

([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) Proposed by Anthropic — uses a set of "constitutional rules" for self-alignment:

```
Step 1: Red-teaming
    Have the model generate harmful responses

Step 2: Critique & Revision
    Model self-critiques and revises based on constitutional rules
    Rules such as: "Choose the response that is least harmful"

Step 3: RL from AI Feedback (RLAIF)
    Train RM on the revised data
    Train with RM + PPO
```

## 4.6 RLVR (Reinforcement Learning with Verifiable Rewards)

**[DeepSeek-R1](https://arxiv.org/abs/2501.12948)'s key innovation**: Use verifiable rewards (such as math answer correctness, code execution results) for RL, without needing human preference data.

```
For math problems:
    1. Model generates chain-of-thought + final answer
    2. Check if answer is correct → reward = 1 or 0
    3. Optimize with GRPO

For code problems:
    1. Model generates code
    2. Run test cases → reward = pass rate
    3. Optimize with GRPO
```

**Surprising finding**: Pure RL (without SFT) can cause chain-of-thought reasoning to emerge. DeepSeek-R1-Zero learned long-chain reasoning through RLVR alone, without any SFT.

## 4.7 Reasoning Models

### 4.7.1 Chain-of-Thought Training

**OpenAI o1/o3** and **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)** represent a new paradigm:

```
Traditional: prompt → answer
Reasoning:   prompt → <think>long chain-of-thought</think> → answer
```

Training methods:
1. **Process Reward Model (PRM)** ([Lightman et al., 2023](https://arxiv.org/abs/2305.20050)): Score each reasoning step, not just the final answer (Outcome Reward Model, ORM)
2. **Monte Carlo Tree Search (MCTS)**: Search for the optimal path in the reasoning tree
3. **RLVR**: Train long-chain reasoning with verifiable rewards

### 4.7.2 Test-Time Compute Scaling

Core insight: instead of using a larger model, let the model "think" longer at inference time. ([Snell et al., 2024](https://arxiv.org/abs/2408.03314))

```
Traditional scaling: larger model → better results
New scaling:         more inference compute → better results

Specific methods:
- Best-of-N: generate N responses, pick the best
- Majority voting: generate N responses, take a vote
- Chain-of-thought: have the model produce a long reasoning chain
- Tree search: search through reasoning space
- Iterative refinement: have the model repeatedly improve its response
```

## 4.8 Tool Use & Agent Training

### 4.8.1 Function Calling

Train models to output structured tool calls:

```json
{"name": "search", "arguments": {"query": "weather in Stockholm"}}
```

**Training data**: Collect (prompt, tool_call, tool_result, final_answer) trajectories

### 4.8.2 Code Execution

```
User: What is 7^23?
Model: <code>print(7**23)</code>
System: [Execution result: 27368747340080916343]
Model: 7^23 = 27368747340080916343
```

### 4.8.3 Multi-step Agent

Train models to execute multi-step tasks:

```
Observation → Thought → Action → Observation → Thought → Action → ... → Answer
```

**[SWE-Agent](https://github.com/princeton-nlp/SWE-agent)/[SWE-bench](https://www.swebench.com/)** style training:
- Give the model a GitHub issue
- Model reads code, edits files, runs tests
- reward = tests pass

## Key Papers

- [Ouyang et al. (2022) — InstructGPT](https://arxiv.org/abs/2203.02155) — the classic three-stage RLHF
- [Rafailov et al. (2023) — DPO](https://arxiv.org/abs/2305.18290) — preference optimization without a reward model
- [Shao et al. (2024) — DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) — the RL algorithm behind DeepSeek-R1
- [Bai et al. (2022) — Constitutional AI](https://arxiv.org/abs/2212.08073) — RLAIF and principle-driven alignment
- [Bai et al. (2022) — HH-RLHF](https://arxiv.org/abs/2204.05862) — Helpful & Harmless dataset and methodology

## Further Reading

- HuggingFace — [TRL library](https://github.com/huggingface/trl) — production SFT / DPO / GRPO / PPO
- Nathan Lambert — [Interconnects.ai](https://www.interconnects.ai/) — ongoing post-training coverage
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) — pure RL eliciting reasoning

## Exercises

1. **SFT fine-tune**: use TRL to SFT a 1.5B base model on Alpaca-cleaned; compare base vs. SFT on MT-Bench.
2. **DPO experiment**: from the same SFT checkpoint, run DPO on UltraFeedback; observe win-rate changes.
3. **Reward hacking watch**: deliberately undertrain a reward model and run PPO; observe how the policy "games" the reward (repetition, special characters, exaggerated outputs).

---

[← Previous Chapter](03-pretraining.md) | [Table of Contents](README.md) | [Next Chapter →](05-peft.md)
