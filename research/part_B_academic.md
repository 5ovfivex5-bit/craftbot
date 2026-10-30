# Part B: Academic Comparison, Proposed Architecture, and Experiment Plan (Sections 10–19)

Scope: sections 10–19 of `BRIEF_GEN15.md`. Generalist AI primary sources are covered by Part A and are referenced here only as "Generalist claims (per brief)", not re-verified.

**Labels used throughout**

- **Confirmed (by authors)**: stated by the paper or repo authors. Not independently replicated unless noted.
- **Inferred**: derived by us from confirmed facts or from a stated config. Not stated by the authors.
- **Proposed**: a design or experiment we recommend. Not yet run.
- **Unknown**: not stated in any source checked here.

**Verification caveat.** Facts below come from `WebFetch` summaries (a small model reads the page and answers a prompt). Abstract-level claims were checked against the abstract text. Numbers taken from full text (ICRT config, prompt length, success table) should be spot-checked against the PDF before any citation in a publication.

---

## 10. Comparison with related academic research

### 10.1 Verification status of the requested list

| Work | Link | Status | Notes |
|---|---|---|---|
| ICRT (Fu et al., 2024) | arXiv 2408.15980; github.com/Max-Fu/icrt | Verified (abstract, HTML, README, TRAIN.md) | Core in-context imitation reference |
| One-Shot Imitation Learning (Duan et al., 2017) | arXiv 1703.07326 | Verified (abstract) | Meta-learned; test-time demo conditioning |
| Algorithm Distillation (Laskin et al., 2022) | arXiv 2210.14215 | Verified (abstract, via export.arxiv.org) | In-context RL from learning histories |
| Decision-Pretrained Transformer (Lee et al., 2023) | arXiv 2306.14892 | Verified (abstract) | **Correction:** the DPT name belongs to 2306.14892, not 2206.13499 |
| Prompt-DT (Xu et al., 2022) | arXiv 2206.13499 | Verified (abstract) | The brief's "Prompt-DT" reference is correct |
| Data distributional properties (Chan et al., 2022) | arXiv 2205.05055 | Verified (abstract) | Burstiness, latent classes, Zipf |
| Decision Transformer (Chen et al., 2021) | arXiv 2106.01345 | Verified (abstract) | Return-conditioned sequence model |
| RT-1 (Brohan et al., 2022) | arXiv 2212.06817 | Verified (abstract only) | No architecture or dataset numbers in fetched text |
| RT-2 (Brohan et al., 2023) | arXiv 2307.15818 | Verified (abstract) | No parameter counts in fetched text |
| Octo (Octo Team, 2024) | arXiv 2405.12213; github.com/octo-models/octo | Verified (abstract, README) | Diffusion-style action head per README |
| OpenVLA (Kim et al., 2024) | arXiv 2406.09246; github.com/openvla/openvla | Verified (abstract, README) | 7B parameters |
| π0 (Black et al., 2024) | arXiv 2410.24164 | Verified (abstract) | Flow matching on VLM backbone |
| Flow Matching (Lipman et al., 2022) | arXiv 2210.02747 | Verified (abstract) | Generative method behind π0-style action heads |
| Diffusion Policy (Chi et al., 2023) | arXiv 2303.04137 | Verified (abstract) | Conditional denoising over action sequences |
| ACT (Zhao et al., 2023) | arXiv 2304.13705 | Verified (abstract) | Action chunking. CVAE detail not in abstract; not verified here |
| SmolVLA (Shukor et al., 2025) | arXiv 2506.01844; github.com/huggingface/lerobot | Verified (abstract, LeRobot README) | Parameter count and compute not in fetched text |
| Vid2Robot (Jain et al., 2024) | arXiv 2403.12943 | Verified (abstract) | Video-prompted policy, cross-attention |
| VPT (Baker et al., 2022) | arXiv 2206.11795; github.com/openai/Video-Pre-Training | Verified (abstract, README) | Minecraft; released weights listed |
| STEVE-1 (Lifshitz et al., 2023) | arXiv 2306.00937 | Verified (abstract) | Minecraft instruction-following; author-stated $60 compute |
| GROOT (Cai et al., 2023) | arXiv 2310.08235 | Verified (abstract) | Minecraft video-instruction following; no test-time adaptation claim |

No independent replication of any of these works was checked in this session.

### 10.2 Relevance to the GEN-1.5 question

The central question is whether a sequence model can use a demonstration placed in its context to change behavior, with no weight updates. The closest published evidence, ordered by how directly it addresses that question:

1. **ICRT (Fu et al., 2024).** Confirmed (by authors): test-time prompting with teleop trajectories, with no parameter updates. This is the most direct analogue to the GEN-1.5 claim, at a much smaller scale than the Generalist claims.
2. **Algorithm Distillation (Laskin et al., 2022).** Confirmed (by authors): an in-context RL algorithm improves across episodes without weight updates. The mechanism is in-context improvement over learning histories, not demonstrations.
3. **Decision-Pretrained Transformer (Lee et al., 2023).** Confirmed (by authors): a transformer trained by supervised learning on offline interaction data acts in-context on bandit and MDP tasks, and generalizes beyond its pretraining distribution. The authors show a Bayesian posterior-sampling interpretation.
4. **Prompt-DT (Xu et al., 2022).** Confirmed (by authors): trajectory prompts of a few timesteps enable few-shot generalization on five MuJoCo tasks with no fine-tuning on the unseen target task.
5. **Vid2Robot (Jain et al., 2024).** Confirmed (by authors): a human video prompt conditions a robot policy via cross-attention. No test-time gradient step is claimed; the fetched text did not confirm whether any is used.
6. **One-Shot Imitation Learning (Duan et al., 2017).** Confirmed (by authors, abstract only): one demonstration conditions a meta-learned policy on a new block-stacking instance. This is meta-learning: the adaptation ability was trained explicitly across task pairs.

### 10.3 Pretraining objective families (where each fits)

| Family | Representative work | Objective | In-context use claimed? |
|---|---|---|---|
| Causal sequence over sensorimotor tokens | ICRT; Decision Transformer | Next-token prediction on (obs, action) | ICRT yes (prompt trajectories); DT is return-conditioned, not prompted |
| Vision-language-action with discretized tokens | RT-2 | Co-fine-tuned token prediction | No in-context claim in fetched text; claims emergent generalization |
| Transformer policy with diffusion head | Octo | Diffusion over action chunks | Finetuning and instruction/goal conditioning; no in-context claim |
| VLM + flow-matching action expert | π0 | Flow matching on action chunks | Zero-shot after pretraining; new skills via fine-tuning. No in-context claim |
| 7B VLA with tokenized actions | OpenVLA | Token prediction | Fine-tuning (LoRA or full). No in-context claim |
| Conditional diffusion, visuomotor | Diffusion Policy | Denoising score matching over action sequences | No; trained per task |
| CVAE-style action chunking | ACT | Chunk prediction | No; trained per task |
| Video-conditioned cross-attention | Vid2Robot | Behavior cloning plus contrastive alignment | Prompt-conditioned; unseen-task transfer claimed |
| Minecraft: inverse-dynamics labeling then BC | VPT | Behavioral cloning on pseudo-labels, then fine-tuning | No in-context claim |
| Minecraft: instruction via latent prior | STEVE-1 | Behavioral cloning and hindsight relabeling | No test-time adaptation claim in fetched text |
| Minecraft: video instruction encoder | GROOT | Encoder-decoder over reference videos | Prompt-by-video; no test-time adaptation claim |

### 10.4 Data-distribution theory for in-context emergence

- **Confirmed (by authors), Chan et al. (2022):** in-context learning emerges when training data is bursty and has many rare latent classes. Naturalistic distributions favored in-context learning in transformers over recurrent models. In-context and in-weights learning traded off, but a Zipfian mix allowed both.
- **Inferred (for our project):** a GRU policy such as our PPO+GRU agent is a recurrent model, so Chan et al. predicts it would learn less in-context behavior than a transformer on the same data. This is a prediction to test, not a finding.

### 10.5 Where the ICL claim is strongest and weakest

- **Strongest, confirmed by authors:** ICRT, AD, DPT, Prompt-DT, all with no weight updates at test time per their abstracts.
- **Weakest:** RT-2, OpenVLA, Octo, π0, SmolVLA, and Diffusion Policy do not claim in-context learning from demonstrations. Their claims concern fine-tuning, zero-shot language conditioning, or emergent generalization. Treating them as in-context evidence would be a category error (the brief warns about this).
- **Generalist claims (per brief, unverified here):** GEN-1.5 is described as learning from a 3–12 s demo in context. Part A is responsible for whether that claim is primary-sourced.

---

## 11. Open-source reproduction opportunities

### 11.1 Comparison table (brief section 7)

"ICL without fine-tuning" means the authors state that a new task is executed from demonstrations supplied at inference, without parameter updates on those demonstrations.

| System | Objective | Architecture | Dataset | ICL w/o fine-tuning? (evidence) | Compute (stated) | Reproducibility | Evaluation |
|---|---|---|---|---|---|---|---|
| **ICRT** | Next-token prediction over (state, action) tokens; predicts next h=16 actions | Default: randomly initialized Llama-2-style causal transformer, 12 layers, latent 768 (Confirmed per paper). Frozen ViT-Base vision encoder (CrossMAE pretraining). Variant: Llama-2-7B + LoRA rank 32 | DROID subset (~2k trajectories after filtering, stated); in-house ICRT-MT: 1,098 trajectories, 29 tasks | **Yes**, confirmed by authors: "without any additional training on the new demonstrations." Demonstrations are context with KV caching. No ablation on prompt length in time steps | DROID pretraining 56 min; ICRT-MT finetune 18 h (paper). TRAIN.md: 8-GPU DDP examples; joint vision finetune "more than 2 day to train on 8xA100" | Code and checkpoints on HF (`mlfu7/ICRT`) and dataset (`Ravenh97/ICRT-MT`). DROID subset still "to be released" per README to-do | Real Franka, 79.2% average (ICRT) vs 20.0% goal-conditioned, 9.2% Octo, 7.5% OpenVLA (paper table) |
| **Octo** | Diffusion-based action head over action chunks | "Transformer-based diffusion policies" (README). Octo-Small 27M, Octo-Base 93M params (README) | Open X-Embodiment, 800k trajectories (abstract) | **No** in-context claim. Instruction via language or goal image; finetuning to new sensors/actions | Pretraining on TPUv4-128: 8 h (Octo-S), 14 h (Octo-B) (README). Inference 17 it/s (S) on RTX 4090 | Open weights and code (README); license Unknown in fetched text | Robot benchmarks in paper |
| **OpenVLA** | Tokenized action prediction on VLM backbone | 7B params; Llama 2 + DINOv2 + SigLIP (abstract/README) | ~970k real-world demonstrations (abstract) | **No** in-context claim. LoRA fine-tuning on one A100 80GB; full fine-tune on 8×A100 (README) | Pretraining compute: Unknown in fetched text | Open weights and code (README) | Robot benchmarks in paper |
| **SmolVLA** | VLA; action-head design Unknown in fetched text (LeRobot lists it as a VLA policy) | Compact; parameter count Unknown in fetched text | Community datasets (Unknown count in fetched text) | Not stated | Designed to train on a single GPU (abstract); GPU and hours Unknown | LeRobot Apache-2.0 (README) | Simulation and real benchmarks in paper |
| **π0** | Flow matching over action chunks on VLM | Parameters Unknown in fetched text | Large multi-platform dataset (counts Unknown) | **No**: zero-shot plus fine-tuning, no in-context claim | Unknown in fetched text | Weights availability: Unknown (not checked against openpi repo) | Robot benchmarks in paper |
| **Diffusion Policy** | Conditional denoising of action sequences | Diffusion transformer or CNN backbone (Unknown exact in fetched text) | Per-task demonstrations | **No**: trained per task | Inference via iterative denoising; no numbers in abstract | Code available (repo not fetched in this session) | 12 tasks in 4 benchmarks; author-reported +46.9% average improvement |
| **ACT** | Action chunking, generative over action sequences | Transformer; CVAE per paper body (not verified here) | "10 minutes" of demonstrations | **No** in-context claim | Unknown | LeRobot includes ACT (README) | 6 real tasks at 80–90% (author-reported) |
| **VPT** | Behavioral cloning on inverse-dynamics pseudo-labels; then RL fine-tuning | Unknown in fetched text | Large unlabeled Minecraft videos plus small labeled set (hours Unknown) | **No** in-context claim. Released weights are BC and RL-fine-tuned variants (README) | Unknown in fetched text | Released `.model` and `.weights` files (README) | Minecraft diamond-tool claim (author-stated) |
| **Decision Transformer** | Return-conditioned sequence prediction | Causal transformer (Confirmed by authors) | Atari, Gym, Key-to-Door | **No**; return-conditioned, not prompted | Unknown | Code: Unknown (not checked) | Offline RL benchmarks |
| **Prompt-DT** | Trajectory-prompted sequence prediction | Transformer with prompt segment | Five MuJoCo tasks | **Yes** in author claim: few-shot "without any extra finetuning on unseen target tasks" | Unknown | Code: Unknown (not checked) | MuJoCo few-shot |

### 11.2 Best reference and smallest meaningful experiment

**Best reference: ICRT.** It is the only open system whose authors explicitly claim test-time prompting with no weight updates and that ships code, checkpoints and a dataset.

Caveats that limit how far ICRT is a template:

- Its default model is a randomly initialized 12-layer, 768-wide transformer (about 85M transformer parameters by the standard $12 d^2$ per layer approximation; Inferred, the paper gives no total). It also uses a frozen ViT-Base (about 86M parameters; Inferred from common ViT-B size). That is roughly 100× our proposed model, and it needs a real robot dataset.
- Its prompt is "the first k trajectories" of the same task, but the paper does not fix k or report a prompt-length ablation (Confirmed absence in the fetched text).
- It is not a demonstration that a *small* model can do this; the paper's claim is about its own scale.

**Second reference: Prompt-DT / Decision Transformer** for the small-model, no-robot version. Their claims rest on MuJoCo, not on a robot or game.

**Smallest meaningful experiment (Proposed):** Experiment A in section 14. It uses our existing craftsim, a causal transformer of about 0.8M parameters, discrete actions, and a held-out-goal test with a prompt-versus-no-prompt comparison. It is the minimum that can show the effect with a prompt, rather than only a goal one-hot.

---

## 12. Proposed lightweight architecture

### 12.1 Design choices and justification

| Choice | Reason | Source of evidence |
|---|---|---|
| Causal transformer over (state, action) tokens | Prompt retrieval by attention; matches ICRT and AD | ICRT, AD (Confirmed by authors) |
| Per-step token = fused observation embedding, then action embedding | Matches ICRT's (state, action) interleaving at small scale | ICRT (Confirmed by authors, per fetched HTML) |
| Discrete action head over 23 actions | Our sim's action space is discrete (`craftsim.py`) | PROTOCOL / `craftsim.py` |
| Vector obs (39-dim) plus optional 32×32×2 image via a small CNN | Experiment A uses vec only; Experiment B adds the image | `Obs.java`, PROTOCOL |
| Prompt is a prefix of same-goal demonstrations, separated by a segment token | Burstiness (Chan et al.) | Chan et al. (Confirmed by authors) |
| No recurrence across demos; full attention over prompt | Recurrent models showed weaker in-context behavior in Chan et al. | Chan et al. (Confirmed by authors) |
| Cross-entropy only (no value or auxiliary heads) | Keep the test about in-context use, not RL | Proposed |

### 12.2 Data-flow diagram

```
 PROMPT (K demos, same goal g)                    QUERY (current episode, goal g)
 ┌──────────────────────────────┐                 ┌────────────────────────────────┐
 │ demo_1: (v_t, img_t, a_t-1)  │                 │ query: (v_t, img_t, a_t-1)     │
 │ <SEP_1>                      │                 │ generated online               │
 │ demo_2 ...                   │                 │                                │
 │ <SEP_K>                      │                 │                                │
 └──────────────┬───────────────┘                 └───────────────┬────────────────┘
                │ concatenate along time (segment embedding: P=0, Q=1)
                ▼
   obs encoder:  v_t ─ Linear(39→d) ;  img_t ─ CNN(2×32×32 → d)     (Exp. B only)
   action enc:   a_{t-1} ─ Embed(23 → d)
   token per step:  x_t = obs_emb + act_emb + pos_emb(t) + seg_emb
                │
                ▼
   causal transformer  (L layers, d_model=d, H heads, FFN 4d, causal mask)
                │
                ▼
   policy head:  h_t (at obs token position) ─ Linear(d → 23) ─ softmax ─ π(a_t | ·)
```

### 12.3 Tensor shapes

Let $B$ = batch, $K$ = number of prompt demos, $H_k$ = length of prompt demo $k$ (decisions), $T$ = query length (decisions), $N = \sum_k H_k + T$ = total steps per sequence, $d$ = model width, $A = 23$ actions.

| Tensor | Shape | Notes |
|---|---|---|
| `vec` | $[B, N, 39]$ | 39-dim vector obs (`VEC_DIM`) |
| `img` (Exp. B) | $[B, N, 2, 32, 32]$ uint8 → float | block id and depth channels (`RES = 32`) |
| `act_prev` | $[B, N]$ int64 in $\{0..22\}$ | previous action; `act_prev[:,0]` = special start token |
| `seg` | $[B, N]$ in $\{0,1\}$ | 0 = prompt, 1 = query |
| `x` (tokens) | $[B, N, d]$ | one token per step (obs+act fused) |
| hidden | $[B, N, d]$ | after $L$ causal blocks |
| logits | $[B, N, 23]$ | next-action distribution at each step |

Sequence length in tokens is $N$ (one fused token per step). If we instead use two tokens per step (state token, action token, as in ICRT), $N_{\text{tok}} = 2N$. The analysis in section 15 uses $N_{\text{tok}}=2N$ as a conservative bound.

**Context budget (Proposed):** $K \le 3$ prompt demos of $\le 60$ decisions (12 s at an assumed 0.2 s per decision; see section 15), plus a query of $T \le 140$ decisions, giving $N \le 320$ steps, i.e. $\le 640$ tokens at two tokens per step.

### 12.4 Causal masking

Attention mask $M_{ij}$ for token positions $i, j \in \{1..n_{\text{tok}}\}$:

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

Consequences:

- Query step $t$ attends to all prompt tokens (earlier in the sequence) and to query steps $\le t$.
- Prompt tokens cannot attend to the query. This prevents the prompt from being conditioned on the query, which is the leak to avoid.
- Prompt-internal order is preserved, so the shuffled-order ablation (section 16) is meaningful.

### 12.5 Loss

Let $\pi_\theta$ be the policy. Let $\mathcal{P}$ = prompt demos, $o_{\le t}$ = query observations up to $t$, $a_{<t}$ = query actions before $t$.

Per-step action cross-entropy, computed only on query positions:

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_{i} |Q_i|} \sum_{i} \sum_{t \in Q_i} \log \pi_\theta\!\left(a_t^{(i)} \mid \mathcal{P}^{(i)},\, o_{\le t}^{(i)},\, a_{<t}^{(i)}\right)
$$

- $i$ indexes sequences in the batch.
- $Q_i$ = query positions of sequence $i$. Prompt positions contribute no loss.
- Optional label smoothing $\epsilon$ on the one-hot target (default 0, Proposed).

Pretraining distribution (Proposed): sample a goal $g$, sample $K \in \{0,1,2,3\}$ prompt demos of goal $g$ from other world seeds, then one query episode of goal $g$. With probability $p_{\text{mis}}$ (default 0.1, Proposed) replace the prompt with demos of a different goal, so the model learns to read the prompt rather than assume it always matches.

### 12.6 Pseudocode (short)

```
for batch of sequences:
    prompt, query = sample(goal g, K)           # prompt from other seeds
    x = embed(prompt + query)                   # [B, N, d]
    h = causal_transformer(x, mask=causal)      # [B, N, d]
    logits = head(h[query_positions])           # [B, T, 23]
    loss = cross_entropy(logits, actions[query_positions], ignore prompt)
    loss.backward(); opt.step()
```

### 12.7 Model sizes

| Name | $d$ | $L$ | Block params (formula $12 d^2 L$) | Use |
|---|---|---|---|---|
| S | 64 | 2 | 98,304 | Exp. C small |
| M | 128 | 4 | 786,432 | Exp. A and B default |
| L | 256 | 6 | 4,718,592 | Exp. C large |

Formula per layer: $4d^2$ (attention Q, K, V, out) + $8d^2$ (FFN with 4× expansion) $= 12 d^2$. Biases, layer norms, and embeddings are excluded. The measured existing `MineMindPolicy` has 933,522 parameters (measured in this session via `trainer/model.py`), so the M model is comparable in size to our current policy.

---

## 13. Dataset specification

This is a proposed reproduction format. Confirmed GEN-1.5 format details are not public (Part A).

### 13.1 Format

- **Container:** one `.npz` per shard of about 500 episodes, plus a JSON Lines index `index.jsonl` (one record per episode, metadata only). Images are stored as uint8 arrays so that the 32×32×2 observations stay compact.
- **Episode boundary:** each episode is one contiguous sequence of decisions. Episodes are never concatenated in storage. Concatenation into prompts happens at sample time.
- **Timestamps:** decision index $t$ (0-based), plus `ticks_per_decision = 4` (from `CraftSim(ticks=4)`). Physical time $= t \times$ tick duration $\times 4$. The tick rate of the sim is Unknown in this session; the Minecraft tick rate of 20 TPS is an assumption used only for demo-length arithmetic.

### 13.2 Fields per episode

| Field | Type | Shape | Source | Notes |
|---|---|---|---|---|
| `episode_id` | str | — | generator | unique |
| `world_seed` | uint64 | — | `reset_env(i, seed)` | split key |
| `goal` | int | — | `state(i)["goal"]` | 0..13 index into `ITEMS` |
| `img` | uint8 | $[H, 32, 32, 2]$ | `step()` output | channel 0 = block id, channel 1 = depth |
| `vec` | float32 | $[H, 39]$ | `step()` output | layout per sim spec (`VEC_DIM = 39`) |
| `action` | int8 | $[H]$ | expert | in $\{0..22\}$ (`NUM_ACTIONS = 23`) |
| `reward` | float32 | $[H]$ | `step()` | logged for analysis only; unused in loss |
| `ach` | uint32 | $[H]$ | `step()` | bit 31 = goal achieved |
| `success` | bool | — | derived | `ach` bit 31 at termination |
| `expert` | str | — | generator | identity of demonstrator |
| `kit` | bool | — | `state(i)["kit"]` | exclude from pure ICL tests |
| `split` | str | — | assigned | `train` / `val` / `test_goal` / `test_seed` |

### 13.3 Example serialized record (JSON Lines index entry)

```json
{"episode_id":"ep_000431","world_seed":912734501,"goal":12,"goal_name":"diamond","length":137,"success":true,"expert":"ppo_goal_v2","kit":false,"split":"train","shard":"shard_0007.npz","shard_key":"ep_000431","action_hist":{"noop":3,"forward":41,"attack":52}}
```

Corresponding arrays in the shard (names and shapes, not values):

```
ep_000431/img     uint8   (137, 32, 32, 2)
ep_000431/vec     float32 (137, 39)
ep_000431/action  int8    (137,)
ep_000431/reward  float32 (137,)
ep_000431/ach     uint32  (137,)
```

### 13.4 Normalization

- `img` channel 0: block ids are categorical (0..17, `NUM_BLOCKS = 18`), so embed them with a learned lookup (as `FastEmbedding` in `trainer/model.py` does), not as floats.
- `img` channel 1: depth scaled to $[0,1]$ by dividing by 255.
- `vec`: not normalized beyond what the sim already applies (height is pre-scaled to $(y - \text{SEA\_LEVEL})/32$ per PROTOCOL). Standardize per dimension with train-split mean and std (Proposed), stored in `stats.json`.

### 13.5 Demo quality checks (Proposed)

A demo is accepted only if all hold:

1. `length` within $[15, 320]$ decisions (3–64 s at 0.2 s per decision, assumption).
2. `success == true` (for prompt demos). Failed episodes are kept for the "interrupted or incomplete demo" ablation, tagged `success=false`.
3. All actions in $\{0..22\}$; no NaN or inf in `vec` or `img`.
4. `kit == false` for pure in-context tests.
5. Expert identity logged; no mixing of expert versions within a test split.

### 13.6 Splits (Proposed)

- Split by **world seed** and **goal**, not by episode. Otherwise a prompt demo and a test episode can share a world, which leaks layout information.
- `train`: goals $G_\text{train}$ (e.g. 10 of 14 items), seeds $S_\text{train}$.
- `test_seed`: goals $G_\text{train}$, unseen seeds $S_\text{test}$. Tests in-context use of demos on new layouts.
- `test_goal`: goals $G_\text{held}$ never seen in training, unseen seeds. Tests task generalization.
- Prompts for test episodes are drawn from the same split as their query, from different seeds.

---

## 14. Detailed experimental training plan

All three experiments use only Python, PyTorch, and the existing craftsim. No robot, no private data, no external art assets (consistent with the project's procedural-only convention).

### 14.1 Hardware assumptions (Proposed plan, stated)

- **CPU:** 8 cores (`nproc` = 8), 14 GB RAM, of which about 8 GB is already in use (measured this session). Budget about 5 GB for data and training.
- **GPU:** one NVIDIA T4 (16 GB GDDR6, 320+ GB/s, 65 FP16 TFLOPS mixed-precision tensor core, 8.1 FP32 TFLOPS, 70 W) per the official NVIDIA T4 page. The T4 is used only as an option; a CPU-only run is the baseline.

### 14.2 Experiment A: minimal proof of concept (Proposed)

| Field | Specification |
|---|---|
| **Question** | Does a small causal transformer, given same-goal demos in context, perform better on held-out goals than with no prompt, with no weight updates? |
| **Environment** | Existing `craftsim` (`CraftSim`), vec-only observation (39-dim), 23 discrete actions, `max_steps` capped at 200 decisions per episode for the experiment |
| **Dataset generation** | Expert = a trained goal-conditioned PPO+GRU policy from `runs/` if a suitable checkpoint exists (Unknown which is current); otherwise a scripted expert. Generate 2,000 successful episodes per training goal for 10 goals = 20,000 episodes. Held-out goals: 4 goals never used in training. Test: 50 world seeds per held-out goal = 200 test episodes. |
| **Architecture** | Causal transformer, $d=128$, $L=4$, 4 heads, FFN 512. One fused token per step (vec embed + previous-action embed + positional + segment). ≈ 0.79M block parameters (section 12.7). |
| **Objective** | Cross-entropy over 23 actions on query positions only (section 12.5). Prompt positions have no loss. |
| **Model size** | ≈ 0.8M parameters (about the same as our current 933,522-parameter PPO+GRU). |
| **Compute and VRAM** | CPU: about 4.8 h per 100M tokens at an assumed 50 GFLOPS sustained (section 15). T4: about 5 min per 100M tokens at assumed 3 TFLOPS sustained. VRAM < 1 GB (rough, unmeasured). Benchmark first. |
| **Data volume** | 20,000 episodes × ≤200 steps × 2 tokens = 8.0M tokens per epoch (section 15). Train ~12 epochs = ~100M tokens. |
| **Training stages** | Stage 1 (pretraining): sample sequences per section 12.5 with $K \in \{0..3\}$ and $p_\text{mis}=0.1$. Stage 2 (none): no fine-tuning; evaluation uses prompts only. |
| **Evaluation** | Held-out goal success rate (`ach` bit 31) for: (a) no prompt, (b) correct-goal prompt, (c) mismatched-goal prompt, (d) shuffled-order prompt. Report with 95% Wilson intervals. |
| **Baselines** | (i) Same transformer with $K=0$ only (goal one-hot, no prompt). (ii) Existing PPO+GRU goal-conditioned agent evaluated zero-shot on held-out goals. (iii) Same transformer with 1 and 10 gradient steps on the prompt demos (section 6 comparison). |
| **Limitations** | Vector obs already contains the goal one-hot, so the model could ignore the prompt. Vec-only obs may not contain layout information that a prompt can supply. Discrete actions. Simulator is not Minecraft. |
| **Success criterion (Proposed, pre-register)** | Correct-prompt success exceeds no-prompt success on held-out goals by ≥ 15 percentage points, with non-overlapping 95% intervals; mismatched prompt gives no gain over no prompt; shuffled prompt degrades relative to correct prompt. |
| **Failure criterion (Proposed)** | Correct-prompt gain < 5 points on held-out goals, or the gain is equally present with mismatched prompts (indicates the model learned from something other than the demo content). |

### 14.3 Experiment B: vision-based (Proposed)

| Field | Specification |
|---|---|
| **Question** | Does in-context use survive when the policy must use the 32×32×2 image rather than vec only, and when test worlds differ from prompt worlds? |
| **Environment** | Same `craftsim`, image plus vec observation. Use the same 10 train and 4 held-out goals as A, with new world seeds. |
| **Dataset generation** | As in A, but store `img`. Subsample to 8,000 episodes for memory (see below). |
| **Architecture** | As M in section 12.7, plus a CNN per step: conv 2→32 (3×3, stride 2), 32→64 (3×3, stride 2), 64→128 (3×3, stride 2), flatten 2048 → Linear 128. CNN ≈ 0.36M parameters (computed). Total ≈ 1.15M. |
| **Objective** | Same as A. |
| **Model size** | ≈ 1.15M parameters. |
| **Compute and VRAM** | CPU: ≈ 9.4 h per 100M tokens at assumed 50 GFLOPS sustained (section 15). T4: ≈ 10 min per 100M tokens at 3 TFLOPS sustained. VRAM < 1 GB (rough). |
| **Data volume** | Images: 20,000 episodes × 200 steps × 2 × 32 × 32 bytes = 8.2 GB raw. 8,000 episodes = 3.3 GB. Use `np.memmap` per shard. Still uses most of the free RAM, so stream from disk. |
| **Training stages** | Pretraining as A. No fine-tuning. |
| **Evaluation** | Same conditions as A, plus **new layouts**: prompt from world seeds $S_\text{test}$, query from seeds $S_\text{train}$, and the reverse. The intent is to show the prompt supplies layout information. |
| **Baselines** | Image-only without prompt; vec-only transformer (from A); existing PPO+GRU (image+vec). |
| **Limitations** | Image is 32×32 block ids and depth, not Minecraft pixels. Vision is simpler than real rendering. A CNN may learn the layout without the prompt. |
| **Success criterion (Proposed)** | Prompt gain on new-layout held-out goals ≥ 10 points, and the gain is absent when the prompt uses the wrong layout. |
| **Failure criterion (Proposed)** | No gain on new layouts; or the gain only appears on same-layout test (points to copying or memorizing layout, not adaptation). |

### 14.4 Experiment C: scaling study (Proposed)

| Field | Specification |
|---|---|
| **Question** | How do model size, data diversity, context length, and pretraining duration change in-context gain on held-out goals? |
| **Environment** | As B. |
| **Design** | Factorial grid (reduced): model size $\{S, M, L\}$ × train-goal count $\{3, 6, 10\}$ × prompt count $K \in \{0, 1, 3\}$. That is 27 runs. Pretraining budget per run: 10M tokens (reduced from 100M to fit the CPU budget). |
| **Dataset** | Subsets of the Experiment B data. |
| **Architecture** | S, M, L from section 12.7 (block params 98k, 786k, 4.7M), each with the CNN of B. |
| **Compute** | Per-run CPU at 10M tokens: S ≈ 1.57M FLOPs/token (block only) → fast; M ≈ 9.4 h × 0.1 = 0.94 h; L ≈ 40.1M FLOPs/token → about 2.2 h per 10M tokens on CPU at assumed 50 GFLOPS (50e9 / 40.1e6 ≈ 1,247 tokens/s; section 15). On a T4: L ≈ 0.04 h per 10M tokens at assumed 3 TFLOPS. The full grid is only practical on a T4 for L and on CPU for S and M. |
| **Evaluation** | Gain = success with $K$ prompts minus success with $K=0$, on held-out goals. Plot gain against each factor. |
| **Baselines** | $K=0$ at each size and data count. |
| **Limitations** | 10M-token budget is short; results at the low end may reflect undertraining, not architecture. Only one seed per cell unless budget allows three. |
| **Success criterion (Proposed)** | Gain increases with train-goal count (diversity) and with $K$ up to a plateau; gain increases with size at fixed data. |
| **Failure criterion (Proposed)** | Gain flat across all factors (no evidence that scale or diversity drives in-context use at this budget). |

### 14.5 Common protocol

- Single seed for A and B; three seeds for the final chosen configuration of each.
- Pre-register success criteria before running.
- Log: tokens seen, loss, held-out goal success, prompt gain, wall-clock, and peak memory.
- Compare with equal token budget and equal parameter count where possible.

---

## 15. Hardware and compute estimates

All throughput numbers are **assumptions**, not measurements. Benchmark before relying on them.

### 15.1 Parameters

| Model | Formula | Parameters |
|---|---|---|
| S block | $L \cdot 12 d^2$ with $d=64, L=2$ | 98,304 |
| M block | $4 \cdot 12 \cdot 128^2$ | 786,432 |
| L block | $6 \cdot 12 \cdot 256^2$ | 4,718,592 |
| CNN (Exp. B) | conv1 $2\cdot9\cdot32+32$; conv2 $32\cdot9\cdot64+64$; conv3 $64\cdot9\cdot128+128$; linear $2048\cdot128+128$ | 355,232 |
| Experiment B total (M + CNN) | 786,432 + 355,232 + embeddings | ≈ 1.15M (computed, includes embeddings) |
| Existing PPO+GRU policy | measured in this session | 933,522 |
| ICRT default (Inferred) | 12 layers × $12 \cdot 768^2$ | ≈ 85M transformer parameters (no total stated in the paper) |

### 15.2 FLOPs per token

Training FLOPs per token use the standard approximation $6N$ (forward plus backward) plus an attention term $12 L d \cdot \text{ctx}$.

For M with context 640 tokens:

$$
\text{FLOPs/token} \approx 6 \times 786{,}432 + 12 \times 4 \times 128 \times 640 = 4{,}718{,}592 + 3{,}932{,}160 = 8{,}650{,}752 \approx 8.65 \text{ MFLOP}
$$

For Experiment B, the CNN runs once per step (two tokens per step), so per token it adds half of its per-step cost. CNN MACs per image: conv1 $16 \cdot 16 \cdot 32 \cdot 18 = 147{,}456$; conv2 $8 \cdot 8 \cdot 64 \cdot 288 = 1{,}179{,}648$; conv3 $4 \cdot 4 \cdot 128 \cdot 576 = 1{,}179{,}648$; linear $2048 \cdot 128 = 262{,}144$. Total $= 2{,}768{,}896$ MACs (≈ 5.5 MFLOP forward). Training ≈ $3 \times 5.5 = 16.6$ MFLOP per step, or 8.3 MFLOP per token.

$$
\text{FLOPs/token (B)} \approx 8.65 + 8.31 = 16.96 \approx 17.0 \text{ MFLOP}
$$

Other sizes: S ≈ 1.57 MFLOP/token; L ≈ 40.1 MFLOP/token (block and attention terms only).

### 15.3 Tokens and wall-clock

Dataset tokens per epoch (Experiment A): $20{,}000 \text{ episodes} \times 200 \text{ steps} \times 2 \text{ tokens} = 8.0 \times 10^6$.

Assumed sustained throughput:

- CPU: 50 GFLOPS (assumption; 8 cores, fp32; to be measured).
- T4: 3 TFLOPS (assumption; the 65 TFLOPS mixed-precision peak is not reached by small models; a 3 TFLOPS sustained rate is about 5% of peak).

| Model | Platform | Tokens/s | 100M tokens | 1B tokens |
|---|---|---|---|---|
| M (Exp. A) | CPU @50 GFLOPS | 5,780 | 4.8 h | 48 h |
| M (Exp. A) | T4 @3 TFLOPS | 346,800 | 0.08 h (~5 min) | 0.8 h |
| B (M + CNN) | CPU @50 GFLOPS | 2,940 | 9.4 h | 94 h |
| B (M + CNN) | T4 @3 TFLOPS | 176,000 | 0.16 h (~10 min) | 1.6 h |
| L | T4 @3 TFLOPS | 74,800 | 0.37 h | 3.7 h |

Real throughput will be lower because of data loading and Python overhead. The T4 number above assumes the data pipeline keeps up, which is not guaranteed for the image experiments.

### 15.4 Memory

- M parameters in fp32: $786{,}432 \times 4$ bytes ≈ 3.1 MB. With gradients and two Adam moments, about 13 MB. The whole model fits trivially in VRAM.
- KV cache at inference (fp16): $2 \cdot d \cdot L \cdot 2$ bytes per token $= 2 \cdot 128 \cdot 4 \cdot 2 = 2{,}048$ bytes per token; for 640 tokens, about 1.3 MB per sequence.
- Activations for batch 32 × 640 tokens: rough upper estimate under 1 GB (unmeasured).
- **The binding constraint is data memory, not model memory.** Experiment B raw images for 20,000 episodes are 8.2 GB, more than the free RAM on this machine. Use 8,000 episodes (3.3 GB) or memmap.

### 15.5 Timing of demonstrations

- One decision = 4 physics ticks (`ticks=4`). At an assumed 20 TPS, one decision = 0.2 s (assumption).
- A 3 s demo = 15 decisions; a 12 s demo = 60 decisions (arithmetic).
- Context of 640 tokens at 2 tokens per step = 320 steps = 64 s of assumed play. This allows roughly three 60-decision prompt demos plus a 140-decision query.

---

## 16. Evaluation and ablation experiments

Mapping of the brief's section 5 experiment list to hypotheses. Each row: the experiment, the hypothesis it tests, the metric, and what supports or refutes it.

**Hypotheses (Proposed)**

- **H1 (existence):** a prompt of same-goal demos raises held-out-goal success above no-prompt, with no weight updates.
- **H2 (content-sensitivity):** H1's gain requires the prompt to be correct (same goal, temporal order intact).
- **H3 (information):** the gain comes from information the prompt carries beyond the current observation (layout), not from the goal one-hot alone.
- **H4 (number):** more prompt demos increase gain up to a plateau.
- **H5 (context):** longer context helps only up to the window used in training.
- **H6 (scale and diversity):** gain grows with model size and with number of training goals.
- **H7 (gradient-free vs gradient):** prompting with zero gradient steps matches or beats 1 gradient step on the same demos, at zero compute cost.

| # | Experiment (brief §5) | Hypothesis | Metric | Supports | Refutes |
|---|---|---|---|---|---|
| 1 | Single demo vs none | H1 | Held-out success, K=1 vs K=0 | K=1 > K=0 by ≥ 15 points | No difference |
| 2 | One-shot vs multiple demos | H4 | Success vs K = 0, 1, 2, 3 | Monotone increase, then plateau | Flat or decrease |
| 3 | Correct vs mismatched prompt | H2 | Success, correct vs other-goal prompt | Correct > mismatched | Equal (prompt not used) |
| 4 | Shuffled temporal order | H2 | Success, original vs shuffled timesteps | Original > shuffled | Equal (order not used) |
| 5 | New objects / environments | H3 | Success on new seeds with prompts from old seeds | Gain persists | Gain only on same seeds |
| 6 | Different initial conditions | H3 | Success with prompt from a different start position | Gain persists | Gain only on matched starts |
| 7 | Tasks excluded from pretraining | H1, H6 | Gain on held-out goals | Gain > 0 with CI excluding 0 | Gain = 0 |
| 8 | Interrupted or incomplete demos | H2 | Success with truncated prompt (first 50%) | Partial gain | Gain requires full demo, or none |
| 9 | Longer vs shorter context | H5 | Gain at K=3 with context 320 vs 160 steps | Longer ≥ shorter | Shorter ≥ longer |
| 10 | Larger vs smaller model | H6 | Gain for S, M, L | Gain grows with size | Flat |
| 11 | More data diversity (goal count) | H6 | Gain at 3, 6, 10 train goals | Gain grows | Flat |
| 12 | More compute (pretraining tokens) | H6 | Gain at 10M, 30M, 100M tokens | Gain grows | Flat |
| 13 | Zero-gradient vs 1-step vs 10-step adaptation | H7 | Success and compute for each | Prompt ≥ 1-step at zero compute | 1-step or 10-step clearly better |
| 14 | Memorization check | H3 | Prompt demos from training world seeds reused as test | Gain only on same seeds | — (diagnostic) |

**Control needed for every row:** a no-prompt baseline with the same model and equal training tokens. Without it, a gain may come from the model simply being trained longer.

### 16.1 Experiment 13 protocol (Section 6 comparison)

- Start from the no-prompt pretrained model.
- Arm 1: zero gradient steps, prompt in context.
- Arm 2: one gradient step on the prompt demos (a single minibatch of the prompt, cross-entropy, learning rate swept over $\{10^{-4}, 10^{-3}\}$).
- Arm 3: ten gradient steps, same learning rate.
- Arm 4: full fine-tuning on the demos (upper bound, long).
- Report success, wall-clock, and parameter change $\|\theta_\text{after} - \theta_\text{before}\|_2 / \|\theta_\text{before}\|_2$.

The brief's observation that "ten steps changed weights very little" is only checkable with this relative-change metric. We have not measured it.

---

## 17. Failure modes and scientific limitations

### 17.1 Failure modes of the proposed experiments

1. **Goal leakage.** The goal one-hot is in `vec` (indices 25–38). The model may solve held-out goals from the one-hot alone, with or without a prompt. Control: no-prompt baseline and mismatched-prompt test. Note: a held-out goal has no training examples, so pure one-hot generalization is unlikely. Verify.
2. **Copying rather than adaptation.** A model may replay the prompt action sequence. Control: shuffled-order test, and new world seeds (section 16, rows 4–5).
3. **Prompt ignored.** Attention may not learn to retrieve prompt content. Control: mismatched prompt and the probability-of-prompt sweep $p_\text{mis}$.
4. **Shortcut through layout memorization.** Training seeds may repeat, so the model learns the layout. Control: strict seed split (section 13.6).
5. **Recurrent baseline confound.** Our PPO+GRU is recurrent, so comparison with a transformer is a change of both architecture and objective. Control: report objective and architecture separately (an extra cross-entropy GRU run).
6. **Expert quality.** If the expert is weak, prompts teach weak behavior. Control: filter by success (section 13.5).
7. **Discretization mismatch.** Human-style continuous mouse input would be quantized into 4-tick decisions. This is an issue for Minecraft (section 18), not for craftsim.
8. **Under-training.** 10M-token scaling runs may fall in an undertrained regime, making scale look flat when it is not. Mitigation: the reduced grid is labeled as such.
9. **Evaluation noise.** 200 test episodes give wide intervals. A 15-point gap is needed for a reliable call. Use Wilson intervals and three seeds for the final configuration.
10. **Chan et al. trade-off.** In-context and in-weights learning trade off. A model with strong in-context gain may show weaker held-out performance on training goals. Track both.

### 17.2 Scientific limitations of the whole plan

- **Scale.** A ~1M-parameter model with 100M tokens tests the *principle*. It does not test the claim that a 3–12 s demo produces the GEN-1.5 effect. Do not state results as GEN-1.5 replication.
- **Simulation.** Craftsim is a Minecraft-like abstraction (14 items, 18 blocks). Results may not transfer to Minecraft's real physics or rendering.
- **Single domain.** One simulator, one action space. No claim of generality across embodiments.
- **Author claims.** The ICRT, AD, DPT, Prompt-DT, and Chan et al. results are author claims. This session did not check independent replications, so the plan treats them as hypotheses.
- **Unknown Generalist recipe.** Part A reports what is public; the recipe is not. Anything we attribute to GEN-1.5's mechanism is an inference.
- **Hardware figures are assumptions.** The CPU and T4 throughput rates in section 15 are not measured. The official T4 specs are confirmed; sustained throughput is not.

---

## 18. Optional Minecraft adaptation

### 18.1 What the project already has (from the repo)

- Observation: `Obs.java` builds the exact observation used in the C++ sim: 32×32 raycast of block id and depth, plus a 39-float vector. This is the key asset: demos recorded through the same builder are compatible with the trained policy.
- Action space: 23 discrete actions (`craftsim.py`, `Bot.java`), with 4-tick decisions and a hold-until-broken `attack`.
- Policy bridge: `PolicyClient.java` sends obs to `mcbridge/bridge.py` over `127.0.0.1:7788`, one JSON object per line. `Bot.java decide()` drives the per-tick decision.
- Policy: `MineMindPolicy`, 933,522 parameters, GRU hidden 256.

### 18.2 How to record demos from the Fabric mod (Proposed)

- Add a recorder in the server tick path, alongside `Bot.decide()`. On each decision, log `(obs_img, obs_vec, action, goal, tick, outcome)`, using the same `Obs` builder so the format matches training.
- Human demonstrator: map human input to the 23 discrete actions. Gap: human continuous mouse movement must be quantized to `turn_left`, `turn_right`, `look_up`, `look_down` over 4-tick decisions. Log the raw input too, for analysis.
- Store a single demo as one episode, with the goal recorded in `vec`. Apply the quality checks in section 13.5.
- Keep the recorder off by default and gate it behind a flag (Proposed).

### 18.3 How a demo would be used at inference with no weight updates (Proposed)

The sequence policy holds a context of prompt tokens and query tokens (section 12).

1. Load the demo episode (a list of `(img, vec, action)` decisions).
2. Build the prompt token sequence as in section 12.3, with segment tag 0.
3. For each live decision, append the live `(img, vec, previous action)` as a query token and run one forward pass. The model's weights are never updated.
4. Use the KV cache for the prompt, so only the query tokens are recomputed per step.

Contrast with the current GRU: the GRU's hidden state carries information forward but does not retrieve from a stored demo. A transformer over a demo prefix can do retrieval, which is the mechanism Chan et al. and ICRT point to. This is a hypothesis to test, not a conclusion.

### 18.4 Comparison with our current PPO+GRU RL agent

| Dimension | PPO+GRU (current) | In-context sequence policy (Proposed) |
|---|---|---|
| Learns from | Reward (`rew`), with curiosity, Go-Explore, and kits (PROTOCOL v2–v4) | Demonstrations only |
| New goal at test | Goal one-hot in `vec`; trained goals only | Demo of the new goal in context; held-out goals testable |
| Test-time weight change | None | None |
| Sample cost | High (PPO over many env steps) | Low for the demo; high for pretraining data |
| Failure mode | Reward hacking, local optima | Copying, demo quality limits ceiling |
| Compute | 933k params, PPO on batched sim | ~0.8M–1.15M params, supervised |

The two approaches are complementary. RL can generate the demonstrations (the expert), and the sequence policy could give test-time adaptation to goals RL never trained on.

### 18.5 Comparison with behavior cloning

- BC: supervised on demos, one policy per goal or a goal-conditioned policy. It cannot use a demo at test time without retraining.
- The in-context policy is BC trained on sequences that include prompts (Proposed). The difference is what happens at test time: BC changes weights for a new task, the in-context policy does not.

### 18.6 Comparison with a hierarchical LLM planner

- An LLM planner decomposes a goal into subgoals in text and calls low-level skills (for example a crafting primitive). It needs no demos but needs language grounding and a skill library.
- Our sim has no language input. A planner would need a text-to-goal mapping, which is trivial for 14 items but is a separate system.
- The in-context policy is a low-level skill learner. The two are not competing. An LLM could choose which demo to put in context, which is a plausible composition (Proposed).

### 18.7 Which robotics insights transfer and which do not

| Insight | Transfers? | Reason |
|---|---|---|
| Causal sequence over (obs, action) tokens | Yes | Architecture is observation-agnostic (ICRT) |
| Burstiness and rare classes drive in-context learning | Yes (hypothesis) | Chan et al. is domain-agnostic; testable here |
| Prompt must be from same task but different scene | Yes | This is the test design in section 13.6 |
| Action chunking | Partial | Minecraft decisions are already discrete 4-tick steps; chunking may help smooth human demos |
| Diffusion and flow-matching action heads | Not needed | Our action space is 23 discrete classes; cross-entropy is adequate |
| Continuous proprioception and multi-camera fusion | No | Our obs is a 32×32 raycast and a 39-float vector |
| Real-robot sensor noise and calibration | No | Not in the sim or the mod |
| Large VLA pretraining (7B) | No | Compute and data outside scope; we propose ~1M parameters |
| Language-conditioned VLA | No | No language in our obs; the planner section covers this separately |

---

## 19. Remaining open questions

1. **GEN-1.5 mechanism.** Does GEN-1.5's in-context effect come from the sequence model alone, or from a retrieval or conditioning component? Unknown (Part A).
2. **Prompt length.** ICRT does not ablate prompt length in time steps. What is the minimum demo length that supports adaptation? Unknown.
3. **Prompt information.** Does a prompt's gain come from layout information (H3), from action patterns (H2), or from both? Experiments 5–6 and 4 separate these, but we have no result.
4. **Recurrent vs transformer.** Is the in-context effect a property of the transformer, or of the data? A GRU baseline with the same data would separate the two. Unknown.
5. **Sustained throughput.** Real CPU and T4 throughput for our model is unmeasured. The 50 GFLOPS and 3 TFLOPS figures are assumptions.
6. **Expert identity.** Which PPO+GRU checkpoint in `runs/` is current and good enough to be an expert? Unknown in this session.
7. **Tick rate.** The Minecraft tick rate of 20 TPS is used for demo-length arithmetic. Confirm from the server config and craftsim's time model.
8. **Sim obs layout.** The full 39-dim vector spec (indices 0–24) is referenced but not reproduced here. Confirm against `src/sim.cpp` before writing the dataset writer.
9. **Continuous vs discrete demos.** How much does quantizing human mouse input into discrete turn actions lose? Unmeasured.
10. **Stochastic prompts.** Do prompts need to vary across episodes to avoid overfitting to one demo? Unknown.
11. **ICRT parameter total.** The paper does not state a total parameter count for its default model; our 85M figure is Inferred from the stated config.
12. **Independent replication.** No replication of ICRT, Prompt-DT, DPT, or AD was checked. Do any independent papers confirm the no-update effect at matched scale? Unknown.
13. **OpenVLA and π0 compute.** Pretraining compute for OpenVLA and π0 is not stated in the fetched text. Unknown.
14. **Weights for π0.** Whether open π0 weights are released was not verified against the openpi repository. Unknown.

---

## Sources used

**Papers (arXiv)**

1. Fu et al., "In-Context Imitation Learning via Next-Token Prediction," arXiv:2408.15980. https://arxiv.org/abs/2408.15980 and full text https://arxiv.org/html/2408.15980v2. Supported: ICRT architecture (12-layer, 768-dim, frozen ViT-Base), prompt construction, no weight update at test time, real-robot table, data sizes.
2. Fu et al., ICRT code. https://github.com/Max-Fu/icrt (README, TRAIN.md). Supported: checkpoints on HF (`mlfu7/ICRT`), dataset (`Ravenh97/ICRT-MT`), training commands, Llama-2-7B LoRA variant, 8×A100 note.
3. Duan et al., "One-Shot Imitation Learning," arXiv:1703.07326. https://arxiv.org/abs/1703.07326. Supported: meta-learned single-demo conditioning (abstract only).
4. Laskin et al., "In-context Reinforcement Learning with Algorithm Distillation," arXiv:2210.14215. https://export.arxiv.org/abs/2210.14215 (arxiv.org blocked by the fetch tool's domain check; export mirror used). Supported: in-context improvement without weight updates (author claim).
5. Lee et al., "Supervised Pretraining Can Learn In-Context Reinforcement Learning" (DPT), arXiv:2306.14892. https://arxiv.org/abs/2306.14892. Supported: DPT in-context generalization, posterior-sampling interpretation (author claim).
6. Xu et al., "Prompting Decision Transformer for Few-Shot Policy Generalization," arXiv:2206.13499. https://export.arxiv.org/abs/2206.13499. Supported: trajectory prompts, no finetuning on unseen target tasks (author claim).
7. Chan et al., "Data Distributional Properties Drive Emergent In-Context Learning in Transformers," arXiv:2205.05055. https://export.arxiv.org/abs/2205.05055. Supported: burstiness, latent classes, Zipf, transformers vs recurrent models (author claim).
8. Chen et al., "Decision Transformer," arXiv:2106.01345. https://arxiv.org/abs/2106.01345. Supported: return-conditioned causal transformer over trajectories.
9. Brohan et al., "RT-1," arXiv:2212.06817. https://arxiv.org/abs/2212.06817. Supported: existence and abstract only; no numbers retrieved.
10. Brohan et al., "RT-2," arXiv:2307.15818. https://arxiv.org/abs/2307.15818. Supported: actions as text tokens, co-fine-tuning, 6k evaluation trials (author claim).
11. Octo Team, "Octo: An Open-Source Generalist Robot Policy," arXiv:2405.12213. https://arxiv.org/abs/2405.12213. Supported: 800k trajectories, finetuning claim.
12. Octo repository. https://github.com/octo-models/octo. Supported: 27M and 93M model sizes, "transformer-based diffusion policies," TPUv4-128 pretraining times, inference speed on RTX 4090.
13. Kim et al., "OpenVLA," arXiv:2406.09246. https://arxiv.org/abs/2406.09246. Supported: 7B, Llama 2 with DINOv2 and SigLIP, ~970k demonstrations.
14. OpenVLA repository. https://github.com/openvla/openvla. Supported: fine-tuning hardware (A100 80GB LoRA; 8×A100 full), 7.5B parameters.
15. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164. https://arxiv.org/abs/2410.24164. Supported: flow matching on VLM, zero-shot plus fine-tuning claims.
16. Lipman et al., "Flow Matching for Generative Modeling," arXiv:2210.02747. https://arxiv.org/abs/2210.02747. Supported: flow matching definition.
17. Chi et al., "Diffusion Policy," arXiv:2303.04137. https://arxiv.org/abs/2303.04137. Supported: conditional denoising over actions, 12 tasks, +46.9% average (author claim).
18. Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), arXiv:2304.13705. https://arxiv.org/abs/2304.13705. Supported: action chunking, 80–90% on 6 tasks (author claim).
19. Shukor et al., "SmolVLA," arXiv:2506.01844. https://arxiv.org/abs/2506.01844. Supported: single-GPU training design goal, community data.
20. LeRobot repository. https://github.com/huggingface/lerobot. Supported: policy list (ACT, Diffusion, SmolVLA, Pi0), LeRobotDataset format, Apache-2.0.
21. Jain et al., "Vid2Robot," arXiv:2403.12943. https://arxiv.org/abs/2403.12943. Supported: video-prompted cross-attention policy, beats BC-Z by >20% with human-video prompts (author claim).
22. Baker et al., "Video PreTraining (VPT)," arXiv:2206.11795. https://arxiv.org/abs/2206.11795. Supported: inverse-dynamics labeling, diamond-tool claim (author claim).
23. VPT repository. https://github.com/openai/Video-Pre-Training. Supported: released foundation, BC, and RL weight files.
24. Lifshitz et al., "STEVE-1," arXiv:2306.00937. https://arxiv.org/abs/2306.00937. Supported: VPT with MineCLIP prior, $60 compute claim (author claim).
25. Cai et al., "GROOT: Learning to Follow Instructions by Watching Gameplay Videos," arXiv:2310.08235. https://arxiv.org/abs/2310.08235. Supported: video-instruction encoder, 70% win rate vs best generalist baseline (author claim).

**Project files (local)**

26. `/home/deck/claude/ai/test/craftbot/BRIEF_GEN15.md` (path: `/home/deck/claude/ai/test/craftbot/research/BRIEF_GEN15.md`). Scope and required sections.
27. `/home/deck/claude/ai/test/craftbot/PROTOCOL.md`. Obs and action definitions, goal one-hot, decision ticks, v2–v4 features.
28. `/home/deck/claude/ai/test/craftbot/craftsim.py`. Action list (23), item and block lists.
29. `/home/deck/claude/ai/test/craftbot/trainer/model.py`. `MineMindPolicy` structure, measured 933,522 parameters (this session).
30. `/home/deck/claude/ai/test/craftbot/mcmod/src/main/java/com/minemind/craftbot/Obs.java`. Observation builder, `RES = 32`, `VEC_DIM = 39`.
31. `/home/deck/claude/ai/test/craftbot/mcmod/src/main/java/com/minemind/craftbot/PolicyClient.java`. Bridge protocol, `127.0.0.1:7788`.

**Hardware reference**

32. NVIDIA Tesla T4 product page. https://www.nvidia.com/en-us/data-center/tesla-t4/. Supported: 16 GB GDDR6, 320+ GB/s, 65 FP16 TFLOPS mixed precision, 8.1 FP32 TFLOPS, 70 W, 130 INT8 TOPS.
