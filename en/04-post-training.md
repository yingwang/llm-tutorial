[← Previous Chapter](03-pretraining.md) | [Table of Contents](README.md) | [Next Chapter →](05-peft.md)

# Chapter 4: Post-Training

Pretraining constructs an unconstrained statistical next-token predictor; post-training aligns that raw representational capacity into a steerable, safe, and task-oriented assistant.

## 4.1 The Post-Training Alignment Hierarchy

```
Base Model (Pretrained Foundation)
    │
    ▼
Supervised Fine-Tuning (SFT)
    │  - Format adherence, dialogue role-play, structured response syntax
    │
    ▼
Preference Optimization & Policy Alignment (RLHF / DPO / GRPO)
    │  - Calibrating human values, stylistic conciseness, epistemic honesty
    │
    ▼
Safety & Constitutional Alignment
    │  - Adversarial robustness, jailbreak resistance, policy compliance
    │
    ▼
Production Deployment Model
```

## 4.2 Supervised Fine-Tuning (SFT)

### 4.2.1 Formatting Conventions and Loss Masking

**Chat Markup Standard** (ChatML / LLaMA-Instruct standard):
```
<|im_start|>system
You are an expert systems engineer.<|im_end|>
<|im_start|>user
Explain the memory layout of FlashAttention.<|im_end|>
<|im_start|>assistant
FlashAttention optimizes GPU SRAM utilization by...<|im_end|>
```

**Loss Masking (Selective Backpropagation)**: Cross-entropy loss is strictly calculated over assistant completion tokens. Loss computation over system prompts, user turns, and structural markup delimiters is masked to zero, ensuring gradients optimize generation behavior rather than memorizing prompt distributions.

### 4.2.2 SFT Data Taxonomy

| Data Provenance | Typical Scale | Quality Ceiling | Primary Applications |
|-----------------|---------------|-----------------|----------------------|
| Expert Human Demonstrations | 10K-100K | Very High | Complex creative synthesis, subtle safety boundaries |
| Open Academic Corpora | 100K-1M | Medium to High | [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1), [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) |
| Synthetic Expansion Pipelines | 1M+ | Scalable / Clean | [Self-Instruct](https://arxiv.org/abs/2212.10560), [Evol-Instruct](https://arxiv.org/abs/2304.12244), [Magpie](https://arxiv.org/abs/2406.08464) |
| Frontier Model Distillation | 1M-10M | High | Synthetic teacher completions (GPT-4o, Claude 3.5 Sonnet) |

**The LIMA Hypothesis** ([Zhou et al., 2023](https://arxiv.org/abs/2305.11206)): "Less Is More for Alignment." The authors demonstrated that a base model fine-tuned on merely 1,000 meticulously curated, high-diversity instruction pairs achieves conversational fluency comparable to models trained on hundreds of thousands of web-scraped examples. Foundational knowledge is acquired during pretraining; SFT primarily acts as a superficial formatting and tone-setting layer.

### 4.2.3 SFT Production Best Practices

- **Instruction Surface Diversity**: Orthogonal task coverage (system administration, algorithmic coding, creative translation, constrained reasoning, JSON serialization) yields far higher transfer capability than redundant variations of repetitive QA tasks.
- **Rejection Sampling (Best-of-N Distillation)**: Generate $K$ stochastic candidate completions per prompt using the current checkpoint, score candidates against an auxiliary reward model or execution verifier, and retain only top-ranked trajectories for the next SFT iteration (central to the LLaMA 3 alignment pipeline).
- **Optimization Hyperparameters**:
  - Learning Rate: $1.0 \times 10^{-5}$ to $2.0 \times 10^{-5}$ (one to two orders of magnitude lower than pretraining).
  - Training Epochs: 2 to 4 epochs for small curated datasets; 1 epoch when training over multi-million synthetic corpora to prevent overfitting and stylistic degeneration.

### 4.2.4 Long-Context Instruction Tuning

- Synthesize tasks requiring deep retrieval and reasoning across full context budgets (multi-document cross-referencing, codebase architecture reviews, multi-hour meeting transcripts).
- Ensure sequence length distributions span the entire context spectrum.
- Validate retrieval fidelity across all context depths using synthetic "Needle in a Haystack" (NIAH) benchmarks.

## 4.3 Reinforcement Learning from Human Feedback (RLHF)

### 4.3.1 The Canonical Three-Stage Pipeline

([Ouyang et al., 2022: InstructGPT](https://arxiv.org/abs/2203.02155))

```
Stage 1: Supervised Fine-Tuning (SFT)
    Train policy π_SFT on curated instruction demonstrations.

Stage 2: Reward Model (RM) Training
    Gather paired preference comparisons: D = {(x, y_w, y_l)}
    Train scalar discriminator RM: r_ψ(x, y_w) > r_ψ(x, y_l)

Stage 3: Policy Optimization via PPO
    Optimize active policy π_θ against scalar reward signals,
    penalizing KL divergence relative to frozen reference policy π_SFT.
```

### 4.3.2 Reward Model Architecture and Bradley-Terry Modeling

```python
# Reward Model: Transformer backbone with scalar projection head
class RewardModel(nn.Module):
    def __init__(self, base_backbone):
        super().__init__()
        self.backbone = base_backbone
        self.value_head = nn.Linear(base_backbone.config.hidden_size, 1, bias=False)
    
    def forward(self, input_ids, attention_mask):
        hidden_states = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        # Extract last token representation
        last_token_repr = hidden_states[:, -1, :]
        return self.value_head(last_token_repr)

# Bradley-Terry Preference Loss Formulation
def bradley_terry_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

**Pathologies in Reward Modeling**:
- **Reward Hacking**: The active policy exploits flaws in the reward model's proxy function (e.g., generating superficial structural formatting, excessive verbosity, or sycophantic phrasing) rather than producing genuinely superior reasoning.
- **Over-Optimization Plateau**: Aggressive optimization against a static reward model causes actual response quality to degrade after an initial peak ([Gao et al., 2023](https://arxiv.org/abs/2210.10760)).
- **Distributional Shift**: The reward model, trained exclusively on historical SFT candidate outputs, yields unreliable out-of-distribution scores when evaluating completions from a heavily modified RL policy.

### 4.3.3 Proximal Policy Optimization (PPO)

([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))

$$\mathcal{L}_{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1 - \varepsilon, 1 + \varepsilon) \hat{A}_t \right) \right]$$

where $\rho_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$, and $\hat{A}_t$ represents Generalized Advantage Estimation (GAE).

**Regularized Composite Reward**:
$$R(x, y) = r_\psi(x, y) - \beta D_{\text{KL}}\left( \pi_\theta(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x) \right)$$

**Systems Complexity in PPO**:
- Requires coordinating four distinct model instances concurrently in cluster memory: Policy Network (trainable), Value Critic Network (trainable), Reference Policy (frozen), and Reward Model (frozen).
- High hardware footprint requiring multi-node tensor-parallel orchestration.

### 4.3.4 Modern Efficient RL Formulations

**REINFORCE Leave-One-Out (RLOO)** ([Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)):
- Samples $K$ parallel completions per prompt, constructing baseline rewards directly from the empirical leave-one-out mean of the group:
  $$\hat{A}_i = r(x, y_i) - \frac{1}{K - 1} \sum_{j \ne i} r(x, y_j)$$
- Eliminates the auxiliary value critic network entirely, saving ~50% GPU memory during training.

**Group Relative Policy Optimization (GRPO)** ([Shao et al., 2024 (DeepSeekMath)](https://arxiv.org/abs/2402.03300)):
- Computes normalized advantage scores over a group of sampled completions:
  $$\hat{A}_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$$
- Powers the reinforcement learning foundation of DeepSeek-R1 without requiring an external critic model.

## 4.4 Direct Preference Optimization (DPO)

### 4.4.1 Mathematical Formulation

([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))

**Foundational Insight**: Under the Bradley-Terry preference model, the optimal policy $\pi^*$ can be expressed in closed form relative to the ground-truth reward function. Inverting this relationship allows the reward model to be analytically substituted out of the objective, formulating preference optimization directly over policy probabilities.

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

where $y_w$ and $y_l$ denote the winning and losing completions, $\pi_{\text{ref}}$ represents the frozen SFT baseline, and $\beta$ controls the strength of the implicit KL constraint.

### 4.4.2 Preference Optimization Taxonomy

| Paradigm | Distinguishing Mechanism | Reference Paper |
|----------|--------------------------|-----------------|
| **IPO** | Adds quadratic identity regularization to prevent over-fitting | [Azar et al., 2023](https://arxiv.org/abs/2310.12036) |
| **KTO** | Formulates optimization over unpaired binary feedback signals | [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306) |
| **ORPO** | Monolithic SFT + preference loss without reference model | [Hong et al., 2024](https://arxiv.org/abs/2403.07691) |
| **SimPO** | Directly uses length-normalized log-probabilities with target margin | [Meng et al., 2024](https://arxiv.org/abs/2405.14734) |

### 4.4.3 Iterative Online DPO

While standard offline DPO trains over a static dataset of historical completions, **Iterative Online DPO** ([Xu et al., 2024](https://arxiv.org/abs/2404.07503)) samples fresh rollouts from the active policy $\pi_\theta$ at each iteration, labels preferences dynamically using an automated judge, and updates the weights. This mitigates out-of-distribution drift and matches the empirical frontier of online PPO.

## 4.5 Constitutional AI (CAI) and RLAIF

([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) Developed by Anthropic to automate alignment using explicit behavioral constitutions:

```
Stage 1: Critique and Revision (Supervised Phase)
    Prompt model with red-teaming queries.
    Model critiques its initial draft against constitutional principles.
    Model iteratively rewrites response to satisfy safety and helpfulness guidelines.

Stage 2: Reinforcement Learning from AI Feedback (RLAIF)
    Train an AI-directed preference model on constitutionally scored pairwise completions.
    Align policy using RL against the resulting AI preference function.
```

## 4.6 Reinforcement Learning with Verifiable Rewards (RLVR)

**The DeepSeek-R1 Paradigm** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)): Replaces soft preference models with deterministic, verifiable outcome oracles (e.g., formal compiler execution, automated unit test suites, mathematical proof checkers):

```
For Symbolic and Algorithmic Domains:
    1. Sample completion rollouts with unstructured chain-of-thought blocks (<think>...</think>).
    2. Extract boxed answers or executable code blocks.
    3. Deterministic scoring: Reward = 1.0 (Correct / Pass), Reward = 0.0 (Incorrect / Fail).
    4. Policy optimization via GRPO.
```

**Emergence of Autonomous Reasoning Chains**: DeepSeek-R1-Zero proved that large-scale RLVR applied directly to a base foundation model without prior SFT triggers spontaneous emergence of self-reflection, backtracking, alternative path exploration, and elongated chain-of-thought rollouts.

## 4.7 Reasoning Models and Test-Time Compute

### 4.7.1 Process Reward Models (PRM)

([Lightman et al., 2023](https://arxiv.org/abs/2305.20050)) Rather than assigning a single scalar reward to the final answer (Outcome Reward Model, ORM), PRMs assign step-level reward scores to every discrete deductive transition in a solution trajectory, enabling fine-grained search and error localization.

### 4.7.2 Test-Time Compute Scaling Laws

Frontier systems (OpenAI o1/o3, DeepSeek-R1) scale performance along a complementary axis: expanding test-time compute ([Snell et al., 2024](https://arxiv.org/abs/2408.03314)):

$$\text{Downstream Accuracy} = f(\text{Pretraining Compute}, \text{Test-Time Inference Compute})$$

**Search and Sampling Strategies**:
- **Best-of-$N$ (Rejection Sampling)**: Sample $N$ independent trajectories; select candidate maximizing PRM score.
- **Self-Consistency (Majority Voting)**: Sample multiple paths and aggregate final symbolic conclusions via majority consensus.
- **Tree Search (MCTS)**: Perform lookahead expansion and value backpropagation over step-level reasoning graphs.

## 4.8 Tool Integration and Agentic Post-Training

### 4.8.1 Function Calling and Structured Interleaving

Condition models to emit deterministic JSON action payloads bounded by special token schemas:

```json
{"name": "execute_query", "parameters": {"sql": "SELECT count(*) FROM clusters WHERE status = 'active';"}}
```

### 4.8.2 Multi-Turn Environment Interaction (SWE-Bench)

Train models over extended agentic loops:
$$\text{Observation} \longrightarrow \text{Internal Thought} \longrightarrow \text{Action} \longrightarrow \text{Environment Feedback}$$
Supervised and RL training over repository-level issue resolution (e.g., [SWE-Agent](https://github.com/princeton-nlp/SWE-agent)) reinforces autonomous debugging, patch generation, and test validation.

## Key Papers

- [Ouyang et al. (2022): Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155): Landmark RLHF methodology.
- [Rafailov et al. (2023): Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290): Closed-form preference optimization framework.
- [Shao et al. (2024): DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300): Introduction of GRPO.
- [DeepSeek-AI (2025): DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948): Frontier RLVR and pure RL reasoning emergence.
- [Bai et al. (2022): Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073): Self-critique alignment and RLAIF principles.

## Further Reading

- Hugging Face: [TRL (Transformer Reinforcement Learning) Library](https://github.com/huggingface/trl) (Production-ready toolchain for SFT, DPO, and GRPO).
- Nathan Lambert: [Interconnects.ai](https://www.interconnects.ai/) (In-depth analysis of post-training architectures and alignment literature).
- OpenAI: [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) (Exploration of test-time compute and reasoning models).

## Exercises

1. **End-to-End SFT Pipeline**: Using Hugging Face TRL, fine-tune a 1.5B base model on a high-quality instruction dataset (such as OpenAssistant or UltraFeedback); evaluate format adherence and refusal rates.
2. **DPO vs. SFT Comparative Analysis**: Implement a DPO training run initialized from your SFT checkpoint using paired preference data; benchmark win-rate improvements on MT-Bench.
3. **Reward Hacking Exploration**: Train a deliberately small reward model on a biased preference set (e.g., preferring verbose answers); run policy optimization and observe how the policy exploits length bias at the expense of substantive accuracy.

---

[← Previous Chapter](03-pretraining.md) | [Table of Contents](README.md) | [Next Chapter →](05-peft.md)
