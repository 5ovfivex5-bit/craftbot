# GEN-1.5 Training Research: How Embodied Models Learn Physical Skills from Demonstrations

*MineMind project · research report · 2026-10-15*
*Compiled from two parallel research passes (Generalist primary sources; academic literature and experiment design), merged and spot-checked by the coordinator.*

## How to read this report

Every technical claim is labelled:

- **Confirmed**: explicitly stated by the cited source's authors. For Generalist, this means the company blog. For papers, it means that paper's authors. Not independently replicated unless stated.
- **Inferred**: supported by cited evidence, but not stated by the source.
- **Proposed**: a design or experiment we recommend. Not yet run.
- **Unknown**: not disclosed in any source checked.

**Evidence caveats.**
- All GEN-0, GEN-1 and GEN-1.5 numbers are company-reported blog claims. No paper, technical report, dataset, weights or code exists for any of the three models.
- Research passes read pages through a summarising fetch tool. The coordinator re-fetched the GEN-1.5 post and confirmed these figures verbatim against it:
  - 59% (±10% s.d.) one-shot in-context success over 10 tasks, from 3–12 s of a single demonstration.
  - 83% (±9% s.d.) with 10 gradient steps on 5 minutes of data per task.
  - 30 s video memory and 100 Hz action trajectories.
  - Pretraining "continuously for over eight months" on "randomly sampled continuous spans".
  - Ten steps change held-out weights by "less than 0.15%".
  - "no architectural changes to promote in-context learning", with no inner/outer (meta-learning) loop.

  Full-text numbers from academic papers (e.g. ICRT configurations) should still be checked against the PDFs before any publication use.

---

## 1. Executive summary

**What GEN-1.5 is (Confirmed, company-reported).** Generalist AI's GEN-1.5 is a large multimodal robot model. It takes 30 s of video memory plus other sensor, language and proprioceptive inputs, and produces 100 Hz action trajectories. It was pretrained for more than eight months on randomly sampled continuous spans of real-world physical interaction. GEN-1's base pretraining used human-worn wearable devices, without robot data. Progress was tracked by next-action prediction error on held-out data.

**The headline result (Confirmed, company-reported).** Inserting a single 3–12 s demonstration into the context window yields 59% average success across 10 tasks, with zero weight updates. Ten gradient steps on 5 minutes of data per task yields 83%, while changing held-out weights by less than 0.15%.

**What is not disclosed (Unknown).** Parameter count, layer structure, tokenisation, action representation, the exact loss, dataset composition, and the mechanism of "Harmonic Reasoning". GEN-0 describes Harmonic Reasoning only as asynchronous, continuous-time sensing/acting token streams. Nothing in this report fills those gaps.

**Why the result is scientifically interesting (Inferred).**
- The in-context ability was not trained for: there was no meta-learning objective and no architectural change. It emerged from ordinary next-action prediction on continuous data.
- This mirrors findings in language models. Chan et al. 2022 showed that in-context learning emerges when training data is "bursty" (items recur within a context) and Zipf-distributed. Generalist offers exactly these as untested hypotheses.
- Continuous real-world recordings naturally contain repeated behaviours within one context window, so the mechanism is plausible. It is not demonstrated.

**Closest open research (Confirmed by those authors, not replicated).** Only ICRT, Algorithm Distillation, the Decision-Pretrained Transformer and Prompt-DT claim in-context adaptation without fine-tuning (Section 10). Octo, OpenVLA, π0, SmolVLA and Diffusion Policy require fine-tuning for new tasks. ICRT is the most direct open analogue and is the recommended reference implementation (Section 11).

**Scale caveat for this project.**
- GEN-0 reports that models around 1B parameters "ossify" and that a phase transition appears near 7B.
- Our current MineMind policy has about 0.9M parameters (933,522 measured by the Part B pass on the current trainer).
- Any in-context effect we reproduce will be a limited, small-scale version of the principle, not GEN-1.5's capability.

**Recommended first experiment (Proposed, Section 14, Experiment A).**
1. Use the existing CraftSim with vector observations (39-dim) and 23 discrete actions.
2. Train a ~0.8M-parameter causal transformer on 10 goals, holding out 4. The prompt is 0–3 same-goal demonstrations from other world seeds; 10% of prompts are mismatched.
3. Loss: cross-entropy on query actions only.
4. Pre-registered success criterion:
   - With a correct prompt, success on held-out goals beats no prompt by at least 15 points, with non-overlapping 95% intervals.
   - Mismatched and shuffled prompts give no gain.
5. Estimated cost: about 5 CPU-hours or minutes on a T4 (arithmetic estimate, Section 15). Benchmark before committing.

**Minecraft relevance (Section 18).** Our project already has the matching data interface: the Fabric mod can record a player in the policy's exact observation and action format. Demonstrations recorded that way can serve both purposes:
- pretraining data for next-action prediction
- prompts for an in-context policy at inference, with no weight updates

Section 18 compares this against our PPO+GRU RL agent, behaviour cloning, and a hierarchical LLM planner.

---

## 2. Chronology of GEN-0, GEN-1, GEN-1.5

| Date | Item | Status | Source |
|---|---|---|---|
| Nov 2025 | GEN-0 blog post: "Embodied Foundation Models That Scale with Physical Interaction" | **Confirmed** (month as shown in the fetch summary of the page) | [gen-0](https://generalistai.com/blog/gen-0) |
| Undated in source | "Towards Machines with a Thousand Hands" (end-effector generalization, GEN-1 era) | **Unknown** date. The page did not return a date in the fetch | [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands) |
| 2 Apr 2026 | GEN-1 blog post: "Scaling Embodied Foundation Models to Mastery" | **Confirmed** (date as shown in the fetch summary) | [gen-1](https://generalistai.com/blog/gen-1) |
| 19 Aug 2026 | GEN-1.5 blog post: "Embodied Foundation Models are One-Shot Learners" | **Confirmed** (date given in the research brief and on the fetch summary) | [gen-1.5](https://generalistai.com/blog/gen-1.5) |

Model-lineage facts:

- **Confirmed.** GEN-0 reports model sizes 1B, 6B, 7B, and "scaled GEN-0 to 10B+ model sizes." [gen-0](https://generalistai.com/blog/gen-0)
- **Confirmed.** GEN-1 is "a large multimodal model that emits actions in real-time" (quoted by the fetch tool). [gen-1](https://generalistai.com/blog/gen-1)
- **Confirmed.** The GEN-1 post says "for GEN-0 and GEN-1 the base foundation model is trained without any robot data" (partial sentence as returned). [gen-1](https://generalistai.com/blog/gen-1)
- **Confirmed.** The GEN-1.5 page refers to a pretrained base that "already adapts in very few steps without any such machinery" (quoted by the fetch tool). [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Inferred.** That base is the GEN-1 family model. The page's wording is consistent with this, but it is not stated as a direct identity.
- **Inferred.** GEN-1.5 is a post-training and prompting result on the GEN-1 family rather than a new architecture. The page does not say this directly; it is the most consistent reading of the text.
- **Unknown.** Whether GEN-1 and GEN-1.5 share a single set of weights, and how many checkpoints exist.

---

## 3. Confirmed technical findings

Each row is sourced from the cited page. Quotes are the fetch tool's text, with partial sentences marked.

| # | Finding | Label | Source |
|---|---|---|---|
| F1 | GEN-1.5 learns new tasks from a 3 to 12 s single demonstration inserted in context. | **Confirmed** | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F2 | Zero-step in-context result: 59% (±10% std. dev.) average success across 10 tasks. | **Confirmed** | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F3 | Ten gradient steps on 5 min of data per task: 83% (±9%) average success. | **Confirmed** | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F4 | One gradient step on one minute of data: 66.5% success on a held-out task. | **Confirmed** (single task, as stated) | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F5 | Ten steps change held-out model weights by "less than 0.15%". The metric is not defined in the fetched text. | **Confirmed** (claim); metric **Unknown** | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F6 | GEN-1.5 takes video input with a **30-second** context window, plus other sensor, language and proprioceptive inputs, and "produces 100 Hz action trajectories." | **Confirmed** (inputs partly stated; action space not stated) | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F7 | Demonstrations are "sensorimotor examples (i.e. sensor data plus action trajectories)" inserted into the 30 s window; the rest holds rolling observations. | **Confirmed** | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F8 | Pretraining is on randomly sampled continuous spans of real-world interaction data from homes, warehouses and factories, and "contains no simulation data." Training ran "continuously for over eight months." | **Confirmed** | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F9 | The pretraining objective the authors monitor is "next action prediction error on a held-out validation set." The loss function is not named. | **Confirmed** (monitoring); loss form **Unknown** | [gen-1.5](https://generalistai.com/blog/gen-1.5), [gen-0](https://generalistai.com/blog/gen-0) |
| F10 | No explicit in-context, meta-learning, inner/outer loop, or auxiliary objectives were used; "no architectural changes to promote in-context learning." | **Confirmed** (as quoted by the fetch tool) | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F11 | Burstiness and Zipfian structure are offered as possible causes; the authors say "Why this capability emerges from pretraining is hard to say." | **Confirmed** (as hypotheses) | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F12 | Pretraining data: "over half a million hours of high-fidelity physical interaction data" (partial sentence). | **Confirmed** | [gen-1](https://generalistai.com/blog/gen-1) |
| F13 | Pretraining data grows by "10,000+ new hours per week" and totals 270,000 hours at the time of the GEN-0 post. | **Confirmed** (as dated in GEN-0 post) | [gen-0](https://generalistai.com/blog/gen-0) |
| F14 | GEN-0 reports a "phase transition at 7B where smaller models exhibit ossification while larger ones continue to improve." | **Confirmed** | [gen-0](https://generalistai.com/blog/gen-0) |
| F15 | GEN-0 reports "a strong power-law relationship" between pretraining data scale and downstream post-training error, in the form L(D) = (D_c / D)^α_D. | **Confirmed** (form as stated on the page; constants not reported in the fetch) | [gen-0](https://generalistai.com/blog/gen-0) |
| F16 | GEN-0 describes Harmonic Reasoning as "an interplay between asynchronous, continuous-time streams of sensing and acting tokens." | **Confirmed** (description only) | [gen-0](https://generalistai.com/blog/gen-0) |
| F17 | GEN-1 pretraining data: "data from low-cost wearable devices on humans doing millions of activities" (partial sentence). | **Confirmed** | [gen-1](https://generalistai.com/blog/gen-1) |
| F18 | GEN-1 claims "each of the results shown are built with only approximately one hour of robot data." | **Confirmed** (company claim) | [gen-1](https://generalistai.com/blog/gen-1) |
| F19 | GEN-1 without pretraining "exhibit[s] very poor performance (average 19%)." | **Confirmed** (company claim; no task list in the fetch) | [gen-1](https://generalistai.com/blog/gen-1) |
| F20 | Weight change is defined as ΔW = W_ft − W₀. Relative parameter-norm changes across end effectors range from 2.5% to 11.4%. | **Confirmed** | [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands) |
| F21 | GEN-1 adapted mid-rollout after its end effector was physically swapped; no success rates were given. | **Confirmed** (qualitative) | [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands) |
| F22 | Human-to-robot transfer uses "a pair of handheld grippers" or "a person can demonstrate a task with their own hands" (GEN-1.5 page). | **Confirmed** | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| F23 | No paper, technical report, dataset release, or code release is announced for GEN-0, GEN-1 or GEN-1.5. | **Confirmed** (absence on the posts) | all three posts |

---

## 4. Data collection and preparation

### 4.1 What is confirmed

- **Demonstration types.** Human data is recorded "either as human data (with a pair of handheld grippers) or as rollouts from the robot itself" (partial sentence as quoted). Human hands are also used for transfer. Simulation prompts "can come from a scripted policy, an RL agent, or a human teleoperating a simulated robot." [gen-1.5](https://generalistai.com/blog/gen-1.5) **Confirmed.**
- **Wearables.** GEN-1 says its pretraining "uses data from low-cost wearable devices on humans doing millions of activities." [gen-1](https://generalistai.com/blog/gen-1) **Confirmed.**
- **Robot data.** GEN-0 and GEN-1 base models are pretrained "without any robot data" (partial sentence). [gen-1](https://generalistai.com/blog/gen-1) **Confirmed.** This conflicts with the GEN-0 post's "real-world manipulation trajectories" framing, which does not specify a source. The conflict is recorded as **Unknown**: the posts do not say whether "trajectories" means wearable traces, robot traces, or both.
- **Scale and sources.** GEN-0: "270,000 hours of real-world manipulation trajectories" collected "across diverse activities in 1,000s of homes, warehouses, and workplaces worldwide," via "multiple data foundry partners." [gen-0](https://generalistai.com/blog/gen-0) **Confirmed.** GEN-1: "over half a million hours." [gen-1](https://generalistai.com/blog/gen-1) **Confirmed.**
- **Diversity of end effectors.** GEN-1 pretraining spans "many end effectors," with about 9,000 variations including off-the-shelf tools, printed parts and modified two-finger grippers. [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands) **Confirmed.**
- **Data engine.** The GEN-1 post says "this progression also validates the data engine behind these models." [gen-1](https://generalistai.com/blog/gen-1) **Confirmed.** The engine is not described.
- **Infrastructure.** GEN-0 cites "a global network of hardware and 1,000s of data collection devices and robots" and "6.85 years of real-world manipulation experience per day of training." [gen-0](https://generalistai.com/blog/gen-0) **Confirmed** (company statement).
- **Sequence sampling.** Pretraining samples are "randomly sampled continuous spans" of interaction. [gen-1.5](https://generalistai.com/blog/gen-1.5) **Confirmed.** The span length used in pretraining is not stated; 30 s is the inference-time context.
- **Post-training data.** Real-robot GEN-0 results used 5.6 hours of task data, with the best results using "550+ hours." [gen-0](https://generalistai.com/blog/gen-0) **Confirmed.**
- **Quality and diversity.** The GEN-0 post says data quality and diversity matter more than raw volume, and that "data mixtures change model behavior." It compares eight datasets from different data-collection partners using prediction error and reverse KL divergence (Table 1). [gen-0](https://generalistai.com/blog/gen-0) **Confirmed** (claim and metrics; Table values not in the fetch).
- **Reward labels.** No reward labels are mentioned on any of the three posts. **Unknown** whether any are used.
- **Language descriptions.** GEN-1.5 mentions a language input in its list of inputs, but the posts do not say whether task language is required for demonstrations. **Unknown.**
- **Simulation.** GEN-1.5 says the model "contains no simulation data, neither rendered video nor simulated dynamics" in pretraining, but uses sim-recorded prompts at inference. [gen-1.5](https://generalistai.com/blog/gen-1.5) **Confirmed.**

### 4.2 Teleoperation vs human manipulation vs simulation vs robot self-generated data

| Source | Confirmed use | Status |
|---|---|---|
| Wearable devices on humans | GEN-1 pretraining | Confirmed |
| Handheld grippers recording human demos | GEN-1.5 prompts | Confirmed |
| Bare human hands | Transfer demos | Confirmed (qualitative) |
| Robot self-generated rollouts | Named as a prompt source | Confirmed (qualitative) |
| Simulation (scripted, RL, teleoperated sim) | Prompts only | Confirmed |
| Real-robot teleoperation | Not described for GEN-1.5 | Unknown |

### 4.3 How human-collected data transfers to robot hardware

- **Confirmed.** Human demonstrations are "reproduced with robot hands" in GEN-1.5 (as the brief states and the fetch confirmed the topic). Specifics are not given.
- **Confirmed.** The GEN-1 end-effector work argues that each hand has distinct contact and force dynamics, and compares cross-embodiment training to multilingual language-model training. [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands)
- **Unknown.** How the human wearable signal is mapped into the robot's observation and action spaces. The GEN-1 end-effector post says the interface is "a different sensorimotor interface" but does not say how actions or poses are represented or mapped.

### 4.4 Proposed training-data specification (reproduction format)

**Proposed.** This is our reproduction format, not Generalist's. Generalist's format is **Unknown**.

Fields per timestep t in an episode (all sensor fields are synchronized to a shared clock):

- `episode_id`, `source_type` (sim / robot_teleop / robot_self / human_wearable / human_handheld), `task_id`, `embodiment_id`, `sensor_meta` (camera intrinsics, extrinsics, frame rate, latency measured).
- `t_ms` (integer milliseconds from episode start, monotonic, step 1000/hz).
- `rgb[t]`: uint8 image, shape (H, W, 3), per camera.
- `proprio[t]`: float32, shape (D_p,), joint or end-effector state.
- `action[t]`: float32, shape (D_a,), chosen action space (see below).
- `done[t]`: bool, episode boundary.
- `lang` (optional): string, task description, with a `null` allowed.

Action representation options (all **Proposed**):

- Absolute joint position, or end-effector pose plus gripper, with per-dim normalization computed on the training split only.
- Action chunks of length K with stride 1, so the model predicts a window of future actions at each step.

Episode boundaries and sequences:

- Episodes are cut into contiguous sequences of length L (for example 30 s at the chosen rate). Sequences may cross task boundaries if the source recording is continuous (this mirrors GEN-1.5's "continuous spans" description, but the boundary rule is **Unknown**).

Sync and quality checks (**Proposed**):

- Timestamp alignment checked by cross-correlating proprio velocity against image optical-flow magnitude; reject episodes above a fixed lag threshold.
- Drop frames where proprio is missing for more than a fixed run length.
- Demo quality score: fraction of successful task completions from a rule-based checker, plus a manual audit sample.

Splits (**Proposed**):

- Split by episode and by task, not by frame. Hold out whole tasks for the zero-shot generalization test (see section 8).
- Hold out whole scenes or objects for the new-environment test.

Serialized example (**Proposed**, JSONL-like; not code):

```
{"episode_id":"ep_000812","source_type":"human_handheld","task_id":"pour_cup_03",
 "embodiment_id":"gripper_v2","hz":50,"t_ms":120,
 "rgb_ref":"cam0/000812_000120.jpg","proprio":[...7 floats...],
 "action":[...7 floats...],"done":false,"lang":null,
 "sensor_meta":{"cam0":{"fx":...,"fy":...,"lat_ms":34}}}
```

Normalization (**Proposed**): z-score computed on the training split; stored with the checkpoint, not recomputed at test time.

---

## 5. Model architecture and known unknowns

### 5.1 Confirmed architectural facts

- **Confirmed.** GEN-1 is "a large multimodal model that emits actions in real-time." [gen-1](https://generalistai.com/blog/gen-1)
- **Confirmed.** GEN-1.5 processes video with a 30 s context window, plus other sensor, language and proprioceptive inputs, and outputs 100 Hz action trajectories. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Confirmed.** GEN-1 "introduces an evolution of the way we do inference with Harmonic Reasoning." [gen-1](https://generalistai.com/blog/gen-1)
- **Confirmed.** GEN-1 inference uses "new forms of paged attention to enable real-time inference" (partial sentence). [gen-1](https://generalistai.com/blog/gen-1)
- **Confirmed.** The architecture is said to decompose updates "across sensor processing, harmonic reasoning, and actuation" (GEN-1 end-effector post, as quoted). [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands)
- **Confirmed.** The architecture was tested on 6DoF, 7DoF and 16+DoF robots. [gen-0](https://generalistai.com/blog/gen-0)
- **Confirmed.** GEN-0 reports that "a new multimodal architecture" made GEN-0 possible. [gen-1](https://generalistai.com/blog/gen-1)

### 5.2 Harmonic Reasoning: what is known vs proprietary

| Element | Status | Source |
|---|---|---|
| Name and high-level idea: "models are trained to simultaneously think and act seamlessly" | **Confirmed** | [gen-0](https://generalistai.com/blog/gen-0) |
| Mechanism: "asynchronous, continuous-time streams of sensing and acting tokens" | **Confirmed** (phrase only) | [gen-0](https://generalistai.com/blog/gen-0) |
| Claim: it avoids System 1 / System 2 separation and inference-time guidance | **Inferred** from the author framing in the GEN-0 summary; not verified as a direct quote | [gen-0](https://generalistai.com/blog/gen-0) |
| What the streams predict (next token, next chunk, MSE on which quantity) | **Unknown** | none |
| How sensing and acting tokens are tokenized | **Unknown** | none |
| Whether the streams run at different rates | **Unknown** | none |
| Relation to "100 Hz action trajectories" | **Unknown** | none |
| Any formal definition or equation | **Unknown** (no published paper) | none |

Our reading of the public description is that "harmonic reasoning" names an interleaved, asynchronous sensing-and-acting token stream, but the public record does not say how it is computed. We do not assume it is causal attention, a recurrent state, or diffusion.

### 5.3 Known unknowns (not disclosed)

| Item | Status |
|---|---|
| Parameter count for GEN-1 and GEN-1.5 | **Unknown** (GEN-0 family reported at 1B, 6B, 7B, 10B+) |
| Number of layers, width, heads | **Unknown** |
| Camera count and resolution | **Unknown** |
| Visual encoder (ViT, CNN, tokenizer) | **Unknown** |
| Language encoder | **Unknown** (language input is said to exist) |
| Action space (joint vs end-effector) | **Unknown** |
| Action tokenization or continuous head | **Unknown** |
| Real-time inference hardware and latency | **Unknown** (compute normalized to GEN-0 7B = 1.0; no absolute FLOPs) |
| Multi-embodiment conditioning mechanism | **Unknown** (shared model across hands is said to work) |
| Whether history is handled by the 30 s window only | **Inferred** (the 30 s window is said to hold demo and rolling observations) |

### 5.4 Multiple embodiments sharing a model

- **Confirmed.** A single model handles many end effectors. The GEN-1 end-effector post says "some hands provide less learning signal than others." [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands)
- **Confirmed.** Adaptation changes weights more for some tools than others: "Controlling a power screwdriver... takes more re-education than controlling a pair of tongs." [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands)
- **Unknown.** How embodiment identity is supplied to the model.

### 5.5 Candidate-architecture comparison

Not Generalist disclosures. These describe families that a reproduction could choose from. We do not claim GEN uses any of them.

| Family | How it works | Pros | Cons | Suitability for in-context physical learning |
|---|---|---|---|---|
| Causal transformer over obs-action tokens (ICRT-style) | Next-token prediction over an interleaved sequence of demo and current observations and actions | Directly supports prompts in context; simple; reproducible | Long sequences are expensive; action tokenization choices matter | **High**, and the closest published precedent (ICRT, arXiv 2408.15980) |
| Multimodal VLM transformer with action tokens (RT-2 / OpenVLA style) | Pretrained vision-language backbone emits discretized action tokens | Language grounding; reuses web-scale pretraining | Large (OpenVLA 7B on 970k demos); discretization loses precision | Medium: prompts exist but are rarely used as in-context demos in published work |
| Action chunking transformer (ACT) | Predicts a chunk of future actions from current observation (conditional VAE in the original) | Efficient on small data; smooth actions | Not designed for demo-in-context; training is per task | Low to medium |
| Diffusion policy | Conditional denoising over action sequences | Multimodal action distributions; strong in-domain results | Inference needs several denoising steps; conditioning on demos is not native | Medium: would need demo conditioning added |
| Flow-matching VLA (pi0 style) | Regresses a velocity field over action chunks on a VLM backbone | Continuous actions; multimodal output | Large; in-context demo use not reported | Medium |
| State-space / recurrent (SSM, GRU) | Compresses history into a fixed state | Cheap long context | Harder to attend to a specific earlier demo; recurrent models did not show ICL in the Chan et al. setup | Low for demo lookup (see section 8) |

Sources for family descriptions: ICRT [arXiv 2408.15980](https://arxiv.org/abs/2408.15980); OpenVLA [arXiv 2406.09246](https://arxiv.org/abs/2406.09246); ACT [arXiv 2304.13705](https://arxiv.org/abs/2304.13705); Diffusion Policy [arXiv 2303.04137](https://arxiv.org/abs/2303.04137); pi0 [arXiv 2410.24164](https://arxiv.org/abs/2410.24164); Chan et al. on recurrent models [arXiv 2205.05055](https://arxiv.org/abs/2205.05055). The description of the families in the table is **Inferred** from the cited abstracts and our general knowledge of these methods, not from any Generalist source.

### 5.6 Real-time inference

- **Confirmed.** GEN-1 emits actions in real time and uses paged attention for it (partial sentence). [gen-1](https://generalistai.com/blog/gen-1)
- **Confirmed.** GEN-1.5 outputs 100 Hz action trajectories. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Unknown.** Inference latency, hardware, and any action-chunk length.

---

## 6. Pretraining methodology

### 6.1 Confirmed

- **Duration.** "It has now been training continuously for over eight months" (GEN-1.5 post). [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Sampling.** Randomly sampled continuous spans from the data engine. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Monitoring.** Next-action prediction error on a held-out validation set. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Metric.** GEN-0 post: "Performance is measured in terms of validation prediction M.S.E." It also reports reverse KL, estimated by a Monte-Carlo estimator. [gen-0](https://generalistai.com/blog/gen-0)
- **Post-training.** GEN-0 post describes supervised fine-tuning "simultaneously on 16 different task sets." [gen-0](https://generalistai.com/blog/gen-0)
- **Scaling.** Power-law relation between pretraining data and downstream post-training error, with the form L(D) = (D_c / D)^α_D. Model-size scaling laws exist per the post but their form is not given. [gen-0](https://generalistai.com/blog/gen-0)
- **Ossification.** 1B models "struggle to absorb complex and diverse sensorimotor data during pretraining"; the authors call this "ossification," and say it has been seen in language models at O(10M) rather than O(1B) parameters. [gen-0](https://generalistai.com/blog/gen-0)

### 6.2 Inferred

- **Inferred.** The pretraining loss is a next-action prediction objective on the sampled sequences, because the authors describe only next-action error. The loss form (MSE, cross-entropy on tokens, flow, or diffusion) is **Unknown**.
- **Inferred.** The GEN-0 "MSE" is measured on continuous actions. This follows from the MSE metric, but the post does not explicitly say which quantity is regressed.

### 6.3 Loss families compared (what Generalist has verified vs not)

| Loss | Verified for GEN? | Notes |
|---|---|---|
| Autoregressive token cross-entropy | **Unknown** | The "tokens" language in GEN-0 is consistent with it, but not confirmed |
| Continuous regression (MSE / L1) | **Partly confirmed**: validation MSE is reported | The regression target is not named |
| Probabilistic (Gaussian NLL, mixture) | **Unknown** | |
| Diffusion | **Unknown** | No mention |
| Flow matching | **Unknown** | No mention |

Our note: the GEN posts do not claim any of these by name except MSE. We do not assign a loss family to GEN.

### 6.4 Conceptual loop (Confirmed pieces; the rest Inferred or Unknown)

1. **Collection.** Wearables, handheld grippers, robot rollouts, simulation prompts. Confirmed (4.1).
2. **Preprocessing.** Unknown.
3. **Sequence sampling.** Random continuous spans. Confirmed.
4. **Multimodal encoding.** Video, sensors, language, proprioception. Confirmed as inputs; encoders Unknown.
5. **Prediction.** Next action, continuous, 100 Hz output. Confirmed as output rate.
6. **Loss.** Next-action error. Form Unknown.
7. **Backprop.** Standard, inferred.
8. **Validation.** Held-out next-action MSE. Confirmed.
9. **Checkpoint evaluation.** Downstream post-training success. Confirmed (GEN-0 and GEN-1.5).

### 6.5 Why random continuous sequences could teach behaviour (explanation, not evidence)

These are our explanations. None is confirmed by Generalist.

- **Inferred.** Predicting the next action from a long history forces the model to represent what the agent is doing and what will happen next, not only the current pose.
- **Inferred.** Diversity of tasks, homes and end effectors increases the number of distinct contact and dynamics regimes the model must fit. GEN-1's end-effector post says different hands give different learning signal (confirmed statement; the mechanism is our inference).
- **Inferred.** Bigger models absorbing heterogeneous data is consistent with the GEN-0 ossification observation (confirmed statement; the mechanism is our inference).

### 6.6 Measuring progress beyond training loss (Proposed)

- Held-out next-action error by source type and by task.
- Downstream success after a fixed post-training budget.
- In-context success with a fixed prompt set (section 8).
- Stability under perturbation: object pose, lighting, tool swap.

---

## 7. Learning-objective analysis

### 7.1 Confirmed objective-level facts

- Next-action prediction error on held-out data is the monitored quantity. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- Validation prediction MSE and reverse KL (Monte-Carlo) are reported. [gen-0](https://generalistai.com/blog/gen-0)
- No inner or outer meta-learning loop. [gen-1.5](https://generalistai.com/blog/gen-1.5)

### 7.2 Proposed equations (reproduction; variables explained)

**Proposed.** Next-action regression over a sequence (continuous actions):

L_NAP(θ) = E over sequences x_{1:T} ~ D [ (1/T) Σ_t || a_t − f_θ(x_{<t}) ||² ]

- θ: model parameters.
- D: distribution of training sequences (random continuous spans, section 4).
- x_{<t}: all observations and actions before time t in the sequence (causal conditioning).
- a_t: ground-truth action at step t.
- f_θ: the model's predicted action from the causal prefix.
- ||·||²: squared Euclidean norm over action dimensions.

Note: this is the reproduction target we would use. It is not claimed to be GEN's loss.

**Proposed.** Chunked version (predict K future actions at each step):

L_chunk(θ) = E [ (1/T) Σ_t Σ_{k=0}^{K−1} w_k || a_{t+k} − f_θ(x_{<t}, k) ||² ]

- K: chunk length.
- w_k: per-offset weight (uniform by default).

**Proposed.** Token cross-entropy, if actions are discretized into B bins per dimension:

L_CE(θ) = − E [ Σ_t log p_θ(z_t | z_{<t}) ]

- z_t: discrete action token at step t, one per dimension or a joint code.
- p_θ: model's softmax over tokens.

**Proposed.** Flow-matching action loss (for comparison only):

L_FM(θ) = E_{t, a_0, ε} [ || v_θ(a_τ, τ, x_{<t}) − (ε − a_0) ||² ], a_τ = (1 − τ) a_0 + τ ε

- τ: interpolation time in [0, 1].
- a_0: target action chunk; ε: Gaussian noise.
- v_θ: learned velocity field.

**Proposed.** Reverse-KL style evaluation, as used in GEN-0: KL(p_data || p_θ) estimated with Monte-Carlo samples from p_θ; the GEN-0 post reports the estimator and not the formula. Treat the formula as **Unknown** for GEN.

### 7.3 Power-law fit (Confirmed form, Unknown constants)

L(D) = (D_c / D)^{α_D}

- L: asymptotic next-action prediction error on a post-trained model.
- D: pretraining dataset size, in action trajectories (GEN-0 Fig. 4 caption).
- D_c: scale constant.
- α_D: exponent.

The constants D_c and α_D are **Unknown** from the fetched text.

### 7.4 Multimodal and equally valid actions

- **Unknown.** Whether GEN handles multimodal action distributions, and how. The posts do not discuss it.
- **Proposed.** If a reproduction sees multimodal demos, test a mixture-density head or a flow head against MSE.

---

## 8. Emergent in-context learning

### 8.1 Confirmed behaviour

- A 3 to 12 s single demonstration, as sensorimotor data, is inserted into the 30 s window and steers the policy on a new task. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- Zero-step one-shot average is 59% across 10 tasks; one-step result 66.5% on a held-out task. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- Chaining demos, sim-recorded prompts, human-hand demos and unseen tools are described qualitatively; no numbers given. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- The page says in-context learning sometimes beats "1–5 gradient steps on the same demonstration data." [gen-1.5](https://generalistai.com/blog/gen-1.5)
- The in-context skills are "more brittle than finetuned models." [gen-1.5](https://generalistai.com/blog/gen-1.5) **Confirmed** (author limitation).

### 8.2 Demo representation in context (what we can say)

- **Confirmed.** The demo is sensor data plus action trajectories. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Confirmed.** The demo sits in the same window as the rolling current observations. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Unknown.** The exact token layout, how the demo is separated from live context, and whether a prompt-type marker is used.
- **Unknown.** Whether action information in the demo is used directly or only through the observation stream.

### 8.3 Questions asked in the brief

| Question | Status and answer |
|---|---|
| Demo representation in context | Confirmed: sensorimotor sequence in the window (F7). Format Unknown. |
| What action info the demo carries | Confirmed: "action trajectories" are part of the demo. |
| Combining past demos with current observations | Confirmed: the window holds both. Mechanism Unknown. |
| Inferring the intended task | Unknown. Not described. |
| Transfer to different starting conditions | Not reported in the posts with numbers. |
| Why no fine-tuning is needed | Unknown. The authors say the capability "appears to emerge directly from pretraining." [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| Memorizing vs copying vs pattern matching vs policy adaptation | Unknown. No experiment in the posts separates these. |
| ICL vs meta-learning | Confirmed: no meta-learning objective. Classification of the mechanism is open. |
| Emergence without explicit meta-learning | Confirmed as the authors' claim: "To our surprise" the model does this out of the box. [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| Genuine adaptation test | Not performed by the authors (see experiment list below). |

### 8.4 Hypotheses from the authors (treated as hypotheses)

| Hypothesis | Source of claim | Status |
|---|---|---|
| Burstiness in the training data | Authors suggest it may be "linked to in-context learning in language models" | Hypothesis; untested by authors |
| Zipfian structure | Same | Hypothesis; untested |
| Repetitive physical behaviours | Authors say physical work may contain "naturally repetitive cycles" | Hypothesis; untested |
| Sequence pattern recognition | Our framing of the same mechanism | **Inferred** |
| Learned task-intent representation | Our framing | **Proposed** (no evidence in posts) |

### 8.5 Published literature on burstiness and Zipf (background)

- **Chan et al., 2022.** "Data Distributional Properties Drive Emergent In-Context Learning in Transformers." Authors: Stephanie C.Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H. Richemond, Jay McClelland, Felix Hill. NeurIPS 2022 (oral). [arXiv 2205.05055](https://arxiv.org/abs/2205.05055). **Confirmed** (abstract):
  - ICL emerges when the training data has properties "such as burstiness," and "having large numbers of rarely occurring classes."
  - ICL traded off against in-weights learning in early experiments.
  - Both could "co-exist in a single model when it was trained on data following a skewed Zipfian distribution."
  - Naturalistic distributions elicited ICL in transformers but not in recurrent models.
- **Chan, Chen, György, Schuurmans, 2024/2025.** "Toward Understanding In-context vs. In-weight Learning." ICLR 2025. [arXiv 2410.23042](https://arxiv.org/abs/2410.23042). **Confirmed** (abstract): ICL can emerge and then decline with further training; a stylized gated model of in-context vs in-weight predictors explains both; the authors extend it to LLM fine-tuning on prompt collections.
- **Brown et al., 2020.** "Language Models are Few-Shot Learners." [arXiv 2005.14165](https://arxiv.org/abs/2005.14165). **Confirmed:** demonstrations "specified purely via text interaction with the model" (the origin of the term in this line of work).
- **Akyürek et al., 2022/2023.** "What learning algorithm is in-context learning?" ICLR 2023. [arXiv 2211.15661](https://arxiv.org/abs/2211.15661). **Confirmed** (abstract): transformers can implement gradient descent and ridge regression for linear regression.
- **von Oswald et al., 2022/2023.** "Transformers learn in-context by gradient descent." [arXiv 2212.07677](https://arxiv.org/abs/2212.07677). **Confirmed** (abstract): trained Transformers can act as mesa-optimizers; a linear self-attention layer can reproduce gradient descent on a regression loss.

Relevance: Chan et al. is the strongest published reason to expect burstiness and Zipf to matter. It is a small-scale study (not robotics), so transfer to GEN-1.5 is **Inferred**, not confirmed.

### 8.6 Comparison with ICRT and other in-context imitation systems

- **ICRT.** Causal transformer; prompt of teleoperated trajectories; autoregressive prediction; no test-time gradients; experiments on a Franka robot. [arXiv 2408.15980](https://arxiv.org/abs/2408.15980) **Confirmed** (abstract). Code is Apache-2.0 and checkpoints are on Hugging Face, but the GitHub page states "Release DROID subset that is used for pre-training ICRT" as an open to-do, and gives no model sizes. [github.com/Max-Fu/icrt](https://github.com/Max-Fu/icrt) **Confirmed.**
- **One-shot imitation (Duan et al., 2017).** Meta-learning framework; one demo at test time. [arXiv 1703.07326](https://arxiv.org/abs/1703.07326) **Confirmed** (abstract). This is a meta-learning approach, unlike GEN's stated absence of meta-learning objectives.
- **GEN-1.5 vs ICRT.** Both use demos in context without weight changes. GEN-1.5 is a large pretrained model with a 30 s window. ICRT is smaller and openly described. Whether the two share a mechanism is **Unknown**.

### 8.7 Why ordinary continuous training sequences are interesting

- **Inferred.** If the same next-action loss on continuous spans produces in-context behaviour, no special prompt-training objective is needed. This is the interesting claim. It is supported only by the company's results, not by an ablation.

### 8.8 Required experiment list (what each outcome would show)

These are **Proposed** experiments for us to run or evaluate. Each row says what outcome would support or refute which hypothesis (H1 burstiness, H2 Zipf, H3 repetitive cycles, H4 pattern recognition, H5 task-intent representation).

| # | Experiment | Supports | Refutes / weakens |
|---|---|---|---|
| E1 | Single demo vs no demo | Demo-driven gain supports H4 and H5 | No gain: the effect is not from the demo |
| E2 | One-shot vs multiple demos | Monotone gain with more demos supports H5 (intent aggregation) | Flat gain: demo count does not matter, suggesting copying |
| E3 | Correct vs mismatched prompt (wrong task demo) | Large drop on mismatch supports H5 | No drop: the model ignores the prompt (no ICL) |
| E4 | Shuffled temporal order within demo | Drop supports H4 (sequence structure used) | No drop: the model uses only a bag of frames |
| E5 | New objects / environments | Success supports H5 (transfer, not copying) | Failure only in new scenes: memorisation of training scenes |
| E6 | Different initial conditions | Success supports H5 and genuine adaptation | Failure: copying of demo trajectory |
| E7 | Tasks excluded from pretraining | Success supports genuine adaptation | Failure: the gain is task memorisation |
| E8 | Interrupted or incomplete demo | Partial success supports H4 | Full dependence on the complete demo: copy-like |
| E9 | Longer vs shorter context | Gain with longer context supports H3 (repetitive cycles, longer evidence) | No gain: context length is not used |
| E10 | Larger vs smaller model | Larger better supports the data-diversity explanation (GEN-0 ossification) | No size effect: capacity is not the limit |
| E11 | More data diversity | Gain supports H1/H2 (distributional properties) | No gain: diversity is not the driver |
| E12 | More pretraining compute | Gain without data change supports the scaling claim | Plateau: ICL does not scale |
| E13 | Burstiness ablation: pretraining with i.i.d. vs bursty sampling | Gain only in bursty supports H1 | Equal gain: burstiness is not needed |
| E14 | Zipf ablation: uniform vs Zipf task frequencies | Gain only in Zipf supports H2 | Equal gain: Zipf is not needed |

A **genuine adaptation** claim requires E5, E6 and E7 to pass together, with E3 and E4 as controls. Without E3 (mismatched prompt) a gain can be explained by generic visual priors.

---

## 9. Few-step gradient adaptation

### 9.1 Confirmed numbers

| Setting | Data | Success | Source |
|---|---|---|---|
| 0 steps (in-context) | 3 to 12 s demo in context | 59% (±10%), 10 tasks | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| 1 step | 1 min data | 66.5% on a held-out task | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| 10 steps | 5 min data per task | 83% (±9%) average | [gen-1.5](https://generalistai.com/blog/gen-1.5) |
| Long fine-tune | Not reported | "tens of thousands of gradient steps" cited for prior robot models, no GEN run reported | [gen-1.5](https://generalistai.com/blog/gen-1.5) |

Other confirmed points:

- Hyperparameters "similar to pretraining"; the procedure was "not tuned," with no adaptation-specific sweep. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- Larger batches and higher learning rates improved one-step results. [gen-1.5](https://generalistai.com/blog/gen-1.5) Numeric values are not given.
- Ten steps change held-out weights by "less than 0.15%." [gen-1.5](https://generalistai.com/blog/gen-1.5)
- The brief describes "1–5 min" of data. The page's one-step result uses one minute, and the 10-step result uses five minutes per task. **Discrepancy noted:** the brief's range is a paraphrase; the page supports 1 min (one-step) and 5 min (ten-step).
- The fetch summary gives "about 50 demonstrations" for 5 minutes. This is **Inferred** from the data duration and is not stated as a count.

### 9.2 Investigating "ten steps changed weights very little"

- **Confirmed.** The authors report <0.15% change in held-out weights after ten steps, alongside an 83% result. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Unknown.** The norm used to measure "change." The GEN-1 end-effector post defines ΔW = W_ft − W₀ and reports relative parameter-norm changes of 2.5% to 11.4% across tools. [thousand-hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands) The GEN-1.5 metric is not defined in the fetched text and may differ.
- **Inferred.** A very small global relative norm change can still shift behaviour, if the change is concentrated in a few layers or directions, or if a global norm averages over many unchanged parameters. The posts do not test this.
- **Inferred.** The two results (<0.15% GEN-1.5 vs 2.5 to 11.4% GEN-1 across tools) are not directly comparable: different tasks, different metrics and not stated to share a norm definition.
- **Proposed.** Report per-layer and per-module ΔW norms, the fraction of parameters above a threshold, and the rank of ΔW, alongside success.

### 9.3 Why one step can adapt a large policy (explanation)

- **Inferred.** If the pretrained model is already near a region where a small step in the right direction selects the task, one step can shift behaviour. This is consistent with the "adapts in very few steps without any such machinery" statement. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Proposed.** Test by measuring the loss on the demo before and after 1 step, and the loss gradient norm. Large gradient norm with small weight change would support a sharp-direction explanation.

### 9.4 ICL vs fine-tuning

- **Confirmed.** In-context skills are "more brittle than finetuned models" (author statement). [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Confirmed.** In-context sometimes beats 1 to 5 gradient steps on the same data. [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Inferred.** The two routes may use one mechanism (demo-conditioned prediction) or two. The posts do not test this.

### 9.5 New skills vs recombination

- **Unknown.** The posts do not say whether the adapted tasks are new skills or recombinations of pretraining skills. The GEN-1.5 description says tasks are "simple and short-horizon." [gen-1.5](https://generalistai.com/blog/gen-1.5)

### 9.6 Effect of LR, batch, model size, data quality

- **Confirmed.** Larger batch and higher LR helped one-step results (no numbers). [gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Confirmed.** Model size effect: 1B models ossify in pretraining; 6B and 7B+ absorb data (GEN-0). [gen-0](https://generalistai.com/blog/gen-0) Adaptation numbers by size are not given.
- **Confirmed.** Data quality and mixture matter for downstream behaviour (GEN-0 claim). [gen-0](https://generalistai.com/blog/gen-0)

### 9.7 Proposed reproducible comparison (zero-gradient vs 1-step vs 10-step vs long fine-tune)

**Proposed.** Same held-out tasks, same demo budget, same base checkpoint.

| Arm | Gradient steps | Data for adaptation | What is reported |
|---|---|---|---|
| A. In-context prompt only | 0 | One demo in context | Success rate, trial count, 95% CI |
| B. One step | 1 | One demo (same as A) | Success, ΔW per module, compute |
| C. Ten steps | 10 | One demo, and separately 5 min of data | Success, ΔW per module, compute |
| D. Long fine-tune | e.g. 1000 or until validation plateau | Same data as C | Success, ΔW per module, compute |

Controls and reporting:

- Matched learning rate and batch across B, C, D; a small LR sweep reported separately.
- Compute reported as wall-clock GPU time and peak VRAM, and number of forward and backward passes.
- Success: fixed task checker, same trial count for all arms, with seeds.
- Reported as success vs compute (a Pareto plot), and as ΔW norm vs success.

Pseudocode is intentionally omitted.

---

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

---

## 20. Sources and bibliography

### 20.1 Generalist primary sources and background (Part A)

Successfully read (primary):

- GEN-1.5, "Embodied Foundation Models are One-Shot Learners," Generalist AI, 19 Aug 2026. https://generalistai.com/blog/gen-1.5 (read via WebFetch, three passes; summary tool, not raw HTML). Supported: F1-F11, section 4 demos and sim, section 6, section 8, section 9 numbers.
- GEN-1, "Scaling Embodied Foundation Models to Mastery," Generalist AI, 2 Apr 2026. https://generalistai.com/blog/gen-1 (read via WebFetch, three passes). Supported: wearable pretraining data, no robot data in base model, half a million hours, Harmonic Reasoning mention, paged attention, 19% without pretraining, 1 hr robot data claim.
- GEN-0, "Embodied Foundation Models That Scale with Physical Interaction," Generalist AI, Nov 2025. https://generalistai.com/blog/gen-0 (read via WebFetch, three passes). Supported: 7B phase transition, ossification, 270,000 hours, scaling law form, Harmonic Reasoning description, MSE and reverse-KL metrics, 16 task-set SFT.
- Towards Machines with a Thousand Hands, Generalist AI (undated in fetch). https://generalistai.com/blog/towards-machines-with-a-thousand-hands (read via WebFetch, two passes). Supported: ΔW definition, 2.5% to 11.4% relative norm change, end-effector diversity, on-the-fly swap.
- ICRT, "In-Context Imitation Learning via Next-Token Prediction," arXiv 2408.15980. https://arxiv.org/abs/2408.15980 (abstract). Supported: causal transformer, test-time prompting without weight updates.
- ICRT repository. https://github.com/Max-Fu/icrt (README). Supported: Apache-2.0, Hugging Face checkpoints, DROID pretraining subset not yet released.
- LeRobot repository. https://github.com/huggingface/lerobot (README). Supported: supported-policy list (ACT, Diffusion, pi0, SmolVLA, others), LeRobotDataset format. No in-context prompting mentioned.
- SmolVLA, arXiv 2506.01844 (search-result summary only; abstract page not fetched). https://arxiv.org/abs/2506.01844 . Supported: under 30k community episodes, single-GPU training, released code and data (as summarized).
- Chan et al., "Data Distributional Properties Drive Emergent In-Context Learning in Transformers," arXiv 2205.05055. https://arxiv.org/abs/2205.05055 (abstract). Supported: burstiness, rare classes, Zipf, ICL/IWL trade-off.
- Chan, Chen, György, Schuurmans, "Toward Understanding In-context vs. In-weight Learning," arXiv 2410.23042 (ICLR 2025). https://arxiv.org/abs/2410.23042 (abstract).
- Brown et al., "Language Models are Few-Shot Learners," arXiv 2005.14165. https://arxiv.org/abs/2005.14165 (abstract).
- Akyürek et al., "What learning algorithm is in-context learning?," arXiv 2211.15661. https://arxiv.org/abs/2211.15661 (abstract).
- von Oswald et al., "Transformers learn in-context by gradient descent," arXiv 2212.07677. https://arxiv.org/abs/2212.07677 (abstract).
- Kaplan et al., "Scaling Laws for Neural Language Models," arXiv 2001.08361. https://arxiv.org/abs/2001.08361 (abstract).
- Duan et al., "One-Shot Imitation Learning," arXiv 1703.07326. https://arxiv.org/abs/1703.07326 (abstract).
- Chi et al., "Diffusion Policy," arXiv 2303.04137. https://arxiv.org/abs/2303.04137 (abstract).
- Black et al., "pi0: A Vision-Language-Action Flow Model for General Robot Control," arXiv 2410.24164. https://arxiv.org/abs/2410.24164 (abstract).
- Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), arXiv 2304.13705. https://arxiv.org/abs/2304.13705 (abstract).
- Kim et al., "OpenVLA," arXiv 2406.09246. https://arxiv.org/abs/2406.09246 (abstract).
- Octo Model Team, "Octo: An Open-Source Generalist Robot Policy," arXiv 2405.12213. https://arxiv.org/abs/2405.12213 (abstract).

Secondary (used only to check for official material, not as evidence for technical claims):

- Humanoid Guide, GEN-1.5 coverage. https://humanoid.guide/generalist-unveils-gen-1-5-for-one-shot-physical-learning/ (confirms no independent validation at time of coverage).
- The Robot Report, GEN-1 coverage. https://www.therobotreport.com/generalist-introduces-gen-1-general-purpose-model-for-physical-ai/ (search snippet only; the "custom kernels" and "paged attention" wording comes from this search summary and is not verified against the primary text).
- KuCoin news flash on GEN-1.5 (search snippet only). https://www.kucoin.com/news/flash/generalist-ai-launches-gen-1-5-seen-as-gpt-3-moment-for-robotics

Failed to read: none of the primary URLs failed. Not checked: no arXiv paper, technical report, or dataset for any GEN model was found by search; a targeted search for a GEN-1 paper returned only the company blog and press.

Process note: one early search-guessed arXiv ID (2505.09012) returned an unrelated paper. It was not used anywhere in this report.

### 20.2 Academic papers, repositories and project files (Part B)

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
