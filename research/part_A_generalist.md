# Part A: Generalist AI Primary Sources (Sections 1-9)

Scope: sections 1-9 of the GEN15 research deliverable. Academic comparisons, open-source reproduction, experiments and Minecraft are covered by the other agent.

## Method and evidence caveats

- **Labels.** Every technical claim is tagged **Confirmed** (explicitly stated in a cited source), **Inferred** (supported by sources but not stated), **Proposed** (a test or design we would run), or **Unknown** (not disclosed).
- **Fetch limitation.** The WebFetch tool returned summaries and short partial quotes, not raw page text. Quotes below are reproduced as the tool returned them, with the partial-sentence cases marked. Where a claim matters for reproduction, the original page should be checked by hand before it is treated as final.
- **Failures.** No primary URL failed to load: all four generalistai.com posts, the arXiv abstracts, and both GitHub repos were read. Nothing was filled in from memory.
- **Company-reported numbers.** All GEN-0, GEN-1 and GEN-1.5 performance, scaling and data numbers are company claims. No paper, technical report, dataset or code release was found for any of the three models.
- **Secondary coverage.** Press and aggregator pages (Humanoid Guide, The Robot Report, KuCoin, a LinkedIn post) were used only to confirm that no independent validation exists. They are not used as evidence for any technical claim.

---

## 1. Executive summary (draft; will be revised)

- **Confirmed.** Generalist AI has published three blog posts, not papers: GEN-0 (Nov 2025), GEN-1 (2 Apr 2026) and GEN-1.5 (19 Aug 2026). Each is a "foundation model" for robots trained on large amounts of real-world physical interaction data. [generalistai.com/blog/gen-0](https://generalistai.com/blog/gen-0), [generalistai.com/blog/gen-1](https://generalistai.com/blog/gen-1), [generalistai.com/blog/gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Confirmed.** The headline GEN-1.5 result is in-context prompting: a 3 to 12 second demonstration inserted into the context window gives 59% (±10%) average success across 10 tasks with zero gradient updates. Ten gradient steps on 5 minutes of data per task give 83% (±9%). [generalistai.com/blog/gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Confirmed.** The pretraining objective the authors name is next-action prediction error on a held-out validation set. No explicit in-context, meta-learning, or auxiliary objective is described, and the authors say there were no architectural changes to promote in-context learning. [generalistai.com/blog/gen-1.5](https://generalistai.com/blog/gen-1.5), [generalistai.com/blog/gen-0](https://generalistai.com/blog/gen-0)
- **Confirmed.** The GEN-1.5 authors offer burstiness and Zipfian structure as hypotheses for the emergence. They say they did not test them. [generalistai.com/blog/gen-1.5](https://generalistai.com/blog/gen-1.5)
- **Unknown.** Parameter count, layer structure, action space, tokenization, loss function, dataset composition, and the mechanism of Harmonic Reasoning are not disclosed for GEN-1 or GEN-1.5. Nothing in this report fills those gaps.
- **Draft conclusion.** The public evidence supports "pretraining on long continuous sensorimotor sequences plus prompting with a demonstration in context" as a description of the GEN-1.5 behaviour. It does not establish the mechanism, and the performance figures are not independently replicated.

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

## Sources used

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
