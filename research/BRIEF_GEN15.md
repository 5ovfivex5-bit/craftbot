# Deep Research: Reverse-Engineering GEN-1.5 Robot Learning (user brief)

Mission: act as a senior AI researcher (robotics, multimodal foundation models, imitation learning,
transformers, RL, in-context learning). Investigate how Generalist AI developed and trained GEN-1.5
(announced August 19, 2026) and how its ability to learn new physical tasks from demonstrations emerged.
Focus on learning mechanisms, not demos.

Primary question: How can a neural network learn general physical behaviors from observation-action
sequences during pretraining, then adapt to a completely new task using a short demonstration without
any gradient updates? Understand it deeply enough to design an original smaller-scale experiment.
Do NOT write implementation code. Produce a technically grounded report + experimental plan.

## 1. Primary sources
1. GEN-1.5 — Embodied Foundation Models are One-Shot Learners https://generalistai.com/blog/gen-1.5
2. GEN-1 — Scaling Embodied Foundation Models to Mastery https://generalistai.com/blog/gen-1
3. GEN-0 — Embodied Foundation Models That Scale with Physical Interaction https://generalistai.com/blog/gen-0
4. Towards Machines with a Thousand Hands https://generalistai.com/blog/towards-machines-with-a-thousand-hands
5. ICRT — In-Context Imitation Learning via Next-Token Prediction https://arxiv.org/abs/2408.15980 , https://github.com/Max-Fu/icrt
6. Hugging Face LeRobot / SmolVLA https://github.com/huggingface/lerobot
Find additional papers, tech reports, datasets, open-source code. Prefer primary sources and official code.
CRITICAL: GEN-1.5's full architecture, weights, dataset and recipe are not public. Never invent details.
Label every technical claim: **Confirmed** (explicitly disclosed), **Inferred** (supported, not confirmed),
**Proposed** (alternative we could test), **Unknown**. Cite with direct links throughout.

## 2. Training data
Demonstration types; handheld grippers / wearables capture; what each trajectory records; alignment of RGB,
motion trajectories, body states, actions; multi-camera; action representation; actual pretraining data
sources; scaling collection across homes/factories/warehouses; diversity, repetition, quality; reward
labeling?; language descriptions required?; how continuous recordings become training examples.
Differences: teleop vs human manipulation vs simulation vs robot self-generated data. How human-collected
data transfers to robot hardware.
Required output: proposed training-data spec (observation repr, action repr, episode/sequence boundaries,
timestamps/sync, sensor metadata, normalization, demo quality checks, train/val/test splits, example
serialized records). Separate confirmed GEN design from proposed reproduction format.

## 3. Architecture
How images/sensors are encoded; continuous actions as inputs/outputs; obs-action sequence interaction;
temporal encoding; history/adaptation; multiple embodiments sharing a model; real-time inference.
Special attention: GEN-0 "Harmonic Reasoning" (asynchronous sensing/acting streams, continuous-time
processing) — what is known vs proprietary.
Compare: causal transformers, multimodal transformers, VLAs, autoregressive action prediction,
action chunking, diffusion policies, flow matching, state-space/recurrent — how each works, pros/cons,
compute, suitability for in-context physical learning. Do not claim GEN uses an undisclosed architecture
because another model does. Propose a small-scale architecture with justified choices + data-flow diagram.

## 4. Pretraining from first principles (most important)
Generalist reports >8 months training on randomly sampled continuous sequences of real-world physical
interaction, monitoring next-action prediction error on held-out data. Explain what this teaches.
Next-action prediction vs behavior cloning; can action prediction teach object interaction and dynamics;
generalized movement skills; obs-action-outcome relationships; reusable manipulation representations;
role of temporal context; why diversity matters; why bigger models benefit from heterogeneous data;
multimodal/equally valid actions; sequences spanning multiple activity phases. Separate evidence from
explanations. Compare losses: autoregressive token, continuous regression, probabilistic, diffusion/flow —
which are verified for Generalist vs alternatives. Full conceptual loop: collection → preprocessing →
sequence sampling → multimodal encoding → prediction → loss → backprop → validation → checkpoint eval.
How to measure progress beyond training loss.

## 5. Emergent in-context learning (central objective)
GEN-1.5 reportedly learns new short-horizon tasks from a 3–12 s sensorimotor demo inserted into context,
no weight changes. Answer: demo representation in context; what action info it has; combining past demos
with current obs; inferring intended task; transfer to different starting conditions; why no fine-tuning
needed; memorizing vs copying vs pattern matching vs genuine policy adaptation; ICL vs meta-learning;
emergence without explicit meta-learning objective; how to establish genuine adaptation experimentally.
Generalist hypotheses (treat as hypotheses): burstiness, Zipfian structure, repetitive physical behaviors,
sequence pattern recognition, learned task-intent representations. Compare with ICRT and other published
in-context imitation systems. Why using ordinary continuous training sequences is interesting.
Required experiments: single demo vs none; one-shot vs multiple; correct vs mismatched prompts; shuffled
temporal order; new objects/environments; different initial conditions; tasks excluded from pretraining;
interrupted/incomplete demos; longer vs shorter context; larger vs smaller models; more data diversity and
compute. State which results support/refute which hypotheses.

## 6. Few-step gradient adaptation
Reported useful adaptation with 1–10 gradient steps on ~1–5 min of demos. Explain how one step can adapt
a large policy; why pretraining enables it; what changes; ICL vs fine-tuning; why fewer updates preserve
generalization; new skills vs recombination; effect of LR, batch, model size, data quality. Investigate the
reported finding that ten steps changed weights very little. Propose a reproducible comparison: zero-gradient
prompting vs 1-step vs 10-step vs long fine-tuning, reporting success and compute cost.

## 7. Open reproduction starting points
Prioritize ICRT, LeRobot, SmolVLA, OpenVLA, Octo, Diffusion Policy, other in-context imitation systems.
Compare table: training objective, architecture, dataset, in-context learning (without fine-tuning?),
compute (GPU memory/time), reproducibility (code/weights/data), evaluation benchmarks. Pick the best
reference. Don't confuse fine-tuning-required models with true in-context adaptation. Smallest experiment
demonstrating a limited version of the principle.

## 8. Resource-efficient original experiments
Resources: CPU-only dev, maybe an NVIDIA T4, limited memory/budget, no robot or private data, Python+PyTorch.
Design: A — minimal proof of concept (lightweight sim, simple interactions, discrete/low-dim actions; does a
small sequence model use previous demos to change behavior without weight updates?). B — vision-based
(simulated visual env, demos, multiple related tasks, held-out task adaptation). C — scaling study (model
sizes / data variants: diversity, capacity, context length, pretraining duration vs in-context adaptation).
For each: environment, dataset generation, architecture, objective, model size, compute+VRAM estimate,
data volume, training stages, evaluation, baselines, limitations, success/failure criteria. Transparent
compute estimates, no unsupported promises; do not promise T4-scale reproduces GEN-1.5.

## 9. Optional: Minecraft
Could the same ideas apply to a Minecraft AI? Synchronized demos (visual obs, position/orientation, mouse,
keyboard, inventory changes, block/entity interactions, rewards/progress, events/state). Could pretraining
on diverse player demos teach general behaviors (navigate, collect, craft, interact)? Method for giving a new
demo at inference without weight changes. Separate low-level sensorimotor policy from high-level language
interpretation/planning. Which robotics insights transfer and which don't. Compare in-context imitation vs
behavior cloning vs RL vs hierarchical LLM-planner system.
(Project context: we already have a C++ Minecraft-like sim with batched envs, a PPO+GRU goal-conditioned
agent, and a Fabric mod running the policy in real Minecraft 26.2 — see /home/deck/claude/ai/test/craftbot
PROTOCOL.md for the obs/action spaces. Use this to make section 9 concrete.)

## 10. Deliverable
`GEN15_TRAINING_RESEARCH.md` with sections:
1 Executive summary · 2 Chronology of GEN-0, GEN-1, GEN-1.5 · 3 Confirmed technical findings ·
4 Data collection and preparation · 5 Model architecture and known unknowns · 6 Pretraining methodology ·
7 Learning-objective analysis · 8 Emergent in-context learning · 9 Few-step gradient adaptation ·
10 Comparison with related academic research · 11 Open-source reproduction opportunities ·
12 Proposed lightweight architecture · 13 Dataset specification · 14 Detailed experimental training plan ·
15 Hardware and compute estimates · 16 Evaluation and ablation experiments ·
17 Failure modes and scientific limitations · 18 Optional Minecraft adaptation ·
19 Remaining open questions · 20 Sources and bibliography
Standards: citations for facts/results; primary papers and repos; flag missing info instead of inventing;
distinguish author claims from independent replication; equations with variables explained; tensor
shapes, sequence formatting, causal conditioning, loss functions for any proposed reproduction; diagrams
and pseudocode where helpful; simplest meaningful experiment first; implementation roadmap but no build.
Be rigorous, skeptical, technical, reproducibility-focused.
