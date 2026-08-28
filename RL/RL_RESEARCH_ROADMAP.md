# Reinforcement Learning: Complete Ordered Checklist

This list assumes you already know machine learning, deep learning, and PyTorch. It contains only reinforcement learning. Complete the **required core** in order, then choose only the specialization that matches your research.

## How to study each item

1. Copy the prompt under one checkbox into an LLM.
2. Learn the concept, derive the equations, implement the exercise, and pass the exit test.
3. Skip an item only if you can already pass its exit test without help.
4. Verify important equations against a textbook or the original paper; an LLM can make mistakes.

Use this tutor instruction at the start of the chat:

> Be my rigorous RL tutor. Assume I know machine learning, deep learning, transformers, and PyTorch. Teach only the RL topic I give you. First test me with 3 diagnostic questions. Then: build intuition, define every symbol, derive the equations step by step, work through a tiny numerical example, show minimal PyTorch-style pseudocode with tensor shapes and detach points, explain common bugs, and give me one coding exercise plus an oral exam. Do not reveal solutions until I attempt them. Challenge vague or incorrect answers. Do not move to the next topic until I pass.

---

# Part I — Required RL core

Everyone should complete Topics 1–12 in this order.

## 1. The RL problem

- [ ] Agent, environment, observation/state, action, reward, transition, episode, horizon, trajectory.
- [ ] Markov property, MDP, and POMDP.
- [ ] Policy, environment dynamics, initial-state distribution, and trajectory probability.
- [ ] Finite-horizon versus continuing tasks; terminal versus truncated episodes.

**Ask the LLM:**

> Teach me how to formulate a problem as an MDP or POMDP. Make me define the state/observation space, action space, transition distribution, reward, initial-state distribution, horizon, terminal conditions, policy, and trajectory probability for three examples. Be strict about state versus observation and terminal versus time-limit truncation.

**Exit test:** Given any sequential decision problem, you can write its MDP/POMDP and factorize the probability of a trajectory.

## 2. Return and RL objectives

- [ ] Immediate reward, return, discounted return, and reward-to-go.
- [ ] Discount factor and finite-horizon objectives.
- [ ] Expected return under a policy.
- [ ] Reward design, sparse/dense rewards, credit assignment, and reward hacking.

**Ask the LLM:**

> Teach return and the RL objective from first principles. Explain when discounting is mathematical convenience versus part of the task, calculate returns for a numerical trajectory, and compare sparse, dense, terminal, and shaped rewards. Include reward misspecification and reward hacking.

**Exit test:** You can calculate every time-step return and precisely state what objective an agent is optimizing.

## 3. Value functions and Bellman equations

- [ ] State value `V^pi`, action value `Q^pi`, and advantage `A^pi`.
- [ ] Bellman expectation equations.
- [ ] Bellman optimality equations.
- [ ] Prediction versus control.
- [ ] Bootstrapping and contraction/fixed-point intuition.

**Ask the LLM:**

> Derive V^pi, Q^pi, A^pi, the Bellman expectation equations, and the Bellman optimality equations without skipping algebra. Use one tiny MDP that I solve by hand. Explain prediction versus control, bootstrapping, and fixed-point intuition.

**Exit test:** You can derive each Bellman equation and solve a small MDP manually.

## 4. Dynamic programming

- [ ] Iterative policy evaluation.
- [ ] Policy improvement theorem.
- [ ] Policy iteration.
- [ ] Value iteration.
- [ ] Generalized policy iteration.

**Ask the LLM:**

> Teach dynamic programming for a known finite MDP. Derive and compare policy evaluation, policy iteration, and value iteration. Make me implement them in NumPy and explain why full dynamic programming is usually impossible in large real environments.

**Exit test:** You can implement policy iteration and value iteration from scratch and explain their convergence conditions.

## 5. Monte Carlo learning

- [ ] Sampling episodes without a known transition model.
- [ ] First-visit and every-visit prediction.
- [ ] Monte Carlo control.
- [ ] Exploring starts and epsilon-soft policies.
- [ ] Bias, variance, and episodic limitations.

**Ask the LLM:**

> Derive Monte Carlo policy evaluation and control from sampled trajectories. Compare first-visit and every-visit methods, explain exploration requirements, and make me implement an epsilon-soft Monte Carlo agent. Emphasize estimator bias and variance.

**Exit test:** You can implement Monte Carlo prediction/control and explain what is estimated from each episode.

## 6. Temporal-difference learning

- [ ] TD target and TD error.
- [ ] TD(0) prediction.
- [ ] SARSA, Expected SARSA, and Q-learning.
- [ ] On-policy versus off-policy learning.
- [ ] Behavior policy versus target policy.
- [ ] Exploration: epsilon-greedy and optimistic initialization.

**Ask the LLM:**

> Derive TD(0), SARSA, Expected SARSA, and Q-learning from their update targets. Use identical notation for all four. Explain on-policy versus off-policy using behavior and target policies, then make me implement and compare SARSA and Q-learning on a small environment.

**Exit test:** Given one transition, you can write every algorithm's exact target and say whether it is on-policy or off-policy.

## 7. Multi-step learning

- [ ] n-step return.
- [ ] Eligibility traces.
- [ ] TD(lambda) and lambda-return.
- [ ] Bias–variance trade-off.
- [ ] Correct handling of terminal and truncated episodes.

**Ask the LLM:**

> Derive n-step returns, eligibility traces, TD(lambda), and the forward/backward views. Give me one numerical trajectory to calculate manually. Explain how gamma and lambda change bias and variance and how terminal versus truncated steps must be masked.

**Exit test:** You can compute n-step and lambda-returns by hand and implement them with correct masks.

## 8. Function approximation

- [ ] Value approximation and semi-gradient updates.
- [ ] State distribution and projection intuition.
- [ ] The deadly triad: function approximation, bootstrapping, and off-policy learning.
- [ ] Stability, target scale, normalization, gradient clipping, and representation issues.

**Ask the LLM:**

> Teach value-function approximation and semi-gradient TD. Explain why the update is called a semi-gradient, what distribution weights the error, and how the deadly triad can cause divergence. Give me diagnostic examples and a minimal neural value-prediction exercise.

**Exit test:** You can identify every component of the deadly triad in an algorithm and explain every stopped gradient in a TD loss.

## 9. Policy gradients

- [ ] Stochastic policies and expected-return objective.
- [ ] Log-derivative/likelihood-ratio trick.
- [ ] Policy-gradient theorem.
- [ ] REINFORCE.
- [ ] Reward-to-go and causality.
- [ ] Baselines and variance reduction.
- [ ] Entropy regularization.

**Ask the LLM:**

> Derive REINFORCE from the expected-return objective and the log-derivative trick. Derive reward-to-go and prove why an action-independent baseline does not bias the expected policy gradient. Cover entropy regularization, variance, and credit assignment. Make me implement REINFORCE from scratch.

**Exit test:** You can derive and implement REINFORCE and explain why the estimator is unbiased but high variance.

## 10. Actor–critic

- [ ] Actor and critic roles.
- [ ] Advantage actor–critic.
- [ ] Critic targets and value loss.
- [ ] TD error as an advantage estimator.
- [ ] Actor/critic gradient flow and detach points.
- [ ] Bias introduced by an approximate critic.

**Ask the LLM:**

> Derive a one-step and an n-step actor–critic algorithm. Trace a batch through actor loss, critic target, critic loss, and optimizer updates. Mark every detach point, explain how critic errors affect the actor, and make me implement a minimal advantage actor–critic agent.

**Exit test:** You can write both losses, identify every gradient path, and explain the actor–critic bias–variance trade-off.

## 11. Generalized Advantage Estimation

- [ ] TD residuals.
- [ ] GAE derivation.
- [ ] Effects of gamma and lambda.
- [ ] Advantage normalization.
- [ ] Terminal, truncation, padding, and variable-length masks.

**Ask the LLM:**

> Derive GAE as an exponentially weighted sum of TD residuals and connect it to lambda-returns. Give me a batched numerical example containing a terminal episode, a time-limit truncation, and padding. Make me implement GAE from blank code with correct masks.

**Exit test:** You can derive, calculate, and implement GAE without looking it up.

## 12. Proximal Policy Optimization

- [ ] Old and new policy probabilities.
- [ ] Importance probability ratio.
- [ ] Clipped surrogate objective.
- [ ] Value loss and optional value clipping.
- [ ] Entropy bonus.
- [ ] Rollout batches, minibatches, and multiple update epochs.
- [ ] Approximate KL, clip fraction, explained variance, and early stopping.
- [ ] What PPO clipping does **not** guarantee.

**Ask the LLM:**

> Derive PPO-Clip from the policy-gradient surrogate and importance ratio. Then trace the complete loop: collect on-policy rollouts, store old log-probabilities/values, compute GAE and returns, normalize advantages, train on shuffled minibatches for several epochs, and monitor KL, entropy, clip fraction, and value quality. Include tensor shapes, pseudocode, detach points, and silent implementation bugs.

**Exit test:** Starting from blank functions, you can implement the GAE calculation and PPO actor/value losses, and diagnose zero KL, exploding values, collapsing entropy, and saturated clipping.

---

# Part II — Required practical checkpoint

Do this before choosing advanced topics.

- [ ] Implement a multi-armed bandit.
- [ ] Implement tabular SARSA and Q-learning.
- [ ] Implement REINFORCE.
- [ ] Implement GAE and the PPO losses from blank functions.
- [ ] Train PPO on one small environment using a trusted implementation or your own minimal version.
- [ ] Run at least 5 seeds and plot individual runs plus an aggregate.
- [ ] Log reward, episode length, losses, entropy, KL, clip fraction, value predictions, explained variance, and gradient norms.
- [ ] Deliberately introduce one bug, predict its symptom, observe it, and fix it.

**Core readiness test:** You can read a basic deep-RL paper, derive its objective, trace one batch through its implementation, and determine what data distribution it trains on.

---

# Part III — Choose your specialization

Do **not** learn every branch now. Choose by problem structure:

| Your problem | Choose first |
|---|---|
| Discrete actions; value-based control | Branch A: DQN |
| Continuous actions; sample efficiency matters | Branch B: SAC |
| LLM, VLM, or speech sequence post-training | Branch C: Sequence RL and preference optimization |
| You only have logged/fixed data | Branch D: Offline RL |
| You can learn or use environment dynamics for planning | Branch E: Model-based RL |
| Demonstrations are the main supervision | Branch F: Imitation learning |
| Multiple learning agents interact | Branch G: Multi-agent RL |

If several rows apply, begin with the branch that matches your **data-collection constraint**. For example, fixed logged data means offline RL even if the action space is continuous.

## Branch A — Value-based deep RL

- [ ] DQN: replay buffer, target network, epsilon-greedy collection, Huber loss.
- [ ] Overestimation bias and Double DQN.
- [ ] Dueling networks and prioritized replay at a conceptual level.
- [ ] Distributional RL only if relevant to your paper.

> Derive DQN's target and loss, then explain replay buffers and target networks through the deadly triad. Derive Double DQN's target and compare it with vanilla DQN. Make me implement DQN and debug common failures.

## Branch B — Continuous-control RL

- [ ] Gaussian policies and action squashing.
- [ ] Deterministic policy gradient and TD3 overview.
- [ ] Maximum-entropy objective.
- [ ] Soft Actor-Critic: twin critics, target critics, reparameterized actor loss, temperature.

> Derive Soft Actor-Critic with identical notation for its actor, critics, targets, entropy temperature, and replay distribution. Explain tanh action correction, detach points, automatic temperature tuning, and why SAC is off-policy. Compare it with PPO and TD3.

## Branch C — Sequence RL and preference optimization

- [ ] Autoregressive generation as an MDP: prefix state, token action, EOS, variable horizon.
- [ ] Sequence log-probability and token/response-level rewards.
- [ ] KL-regularized RL with a reference policy.
- [ ] Sparse outcome rewards, process rewards, and credit assignment.
- [ ] Learned reward models and pairwise preference loss.
- [ ] PPO-based RLHF: policy, reference, reward/verifier, value head, rollout, advantage, updates.
- [ ] Verifiable rewards and reward hacking.
- [ ] Direct preference methods such as DPO; understand that they are not automatically online RL.
- [ ] Group-relative or online policy-optimization methods; derive the exact estimator from the chosen paper.
- [ ] On-policy data freshness and reusing generated trajectories.

> Formulate autoregressive generation as an MDP and derive trajectory probability, sequence log-probability, a KL-regularized objective, per-token reward shaping, advantages, and the PPO-based RLHF update. Trace tensors and masks through generation, reward, value prediction, GAE, and optimization. Then compare this data flow with reward modeling, DPO-style offline preference training, rejection sampling, and online group-relative methods. Clearly distinguish what is RL from what is supervised preference optimization.

## Branch D — Offline RL

- [ ] Dataset coverage and support mismatch.
- [ ] Extrapolation error and out-of-distribution actions.
- [ ] Off-policy evaluation and importance sampling.
- [ ] Conservative value learning.
- [ ] Behavior regularization.
- [ ] Decision-transformer-style sequence modeling as a contrasting approach.

> Teach offline RL around the central problem of distribution shift. Derive importance-sampling evaluation, explain why naive Q-learning exploits out-of-distribution errors, and compare conservative value learning, behavior regularization, and sequence-modeling approaches. Make me diagnose dataset coverage before choosing an algorithm.

## Branch E — Model-based RL

- [ ] Known versus learned dynamics.
- [ ] Planning, model-predictive control, and tree search.
- [ ] Dyna-style learning.
- [ ] Model error and compounding error.
- [ ] Uncertainty-aware planning.

> Teach model-based RL by separating model learning, planning, policy learning, and data collection. Compare Dyna, model-predictive control, and tree-search approaches. Explain compounding model error and uncertainty, then make me design a small model-based agent.

## Branch F — Imitation learning

- [ ] Behavioral cloning.
- [ ] Covariate shift and compounding errors.
- [ ] DAgger.
- [ ] Inverse RL and adversarial imitation conceptually.
- [ ] Relationship to supervised fine-tuning and preference learning.

> Derive behavioral cloning and explain sequential covariate shift. Teach DAgger and the intuition of inverse RL. Compare imitation learning, offline RL, supervised fine-tuning, and preference optimization by their data, assumptions, and objectives.

## Branch G — Multi-agent RL

- [ ] Cooperative, competitive, and mixed settings.
- [ ] Non-stationarity caused by learning agents.
- [ ] Centralized training with decentralized execution.
- [ ] Credit assignment and equilibrium concepts.
- [ ] Self-play and population-based evaluation.

> Teach the minimum multi-agent RL foundation: Markov games, cooperative versus competitive objectives, non-stationarity, centralized training with decentralized execution, multi-agent credit assignment, self-play, and evaluation. Make me formulate one two-agent problem mathematically.

---

# Topics to postpone unless your project needs them

- [ ] Contextual-bandit regret proofs.
- [ ] Distributional RL.
- [ ] Hierarchical RL and options.
- [ ] Meta-RL.
- [ ] Safe/constrained RL.
- [ ] Risk-sensitive RL.
- [ ] Advanced exploration theory.
- [ ] Multi-objective RL.
- [ ] Distributed RL infrastructure.
- [ ] Formal convergence proofs beyond the core derivations.

These are valuable, but none is a universal prerequisite for beginning RL research.

---

# Final RL readiness checklist

You are ready to begin a research implementation when you can:

- [ ] Formulate the task as an MDP/POMDP and write the exact objective.
- [ ] Explain Bellman learning, Monte Carlo learning, TD learning, policy gradients, actor–critic, GAE, and PPO without notes.
- [ ] Identify whether the method is on-policy, off-policy, or offline.
- [ ] Derive the method's targets and losses from the paper.
- [ ] Trace one batch through the implementation, including shapes, masks, and detach points.
- [ ] Explain its expected failure modes and the metrics that expose them.
- [ ] Implement or modify the relevant specialization branch.

Once these are true, stop expanding the checklist and start the research experiment. Learn additional RL topics only when the experiment demands them.
