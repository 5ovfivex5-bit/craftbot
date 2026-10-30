# MineMind Training Studio — requirements (from user, verbatim summary)
See the user's full spec; key points every part must honor:
- C++20 sim + Dear ImGui (docking) + SDL3 + OpenGL3 dashboard; PyTorch trainer; pybind11 sim bindings.
- Sim, model, GUI independent. Sim always runs headless. Closing dashboard never affects training.
- Trainer and dashboard are separate processes speaking PROTOCOL.md; dashboard can reconnect.
- Dashboard panels (dockable, dark theme, layouts saved/restored): Live 3D viewport (first-person,
  third-person follow, free cam; env selector; overlay: action, coords, velocity, health, task,
  episode, reward, env id; toggle to disable), Neural Network Inspector (real architecture from
  `model` msg: node/edge graph Encoder->Hidden->GRU->Heads, click layer for details; activation
  view: heatmaps, histograms, layer summaries with labeled color scale; action probs, entropy,
  grad norms, weight stats, GRU stats, NaN/Inf warning; clearly labeled activations vs probs vs
  weights vs grads; unsubscribe nn when panel hidden), Training stats + progress bar + ETA
  ("Calculating" when null), Controls (start/pause/resume/stop/reset/save/load, total steps,
  num envs w/ presets 1,4,8,16,32,64,128,256,512,1000 + custom, lr, minibatch, rollout len,
  ckpt freq, visualize toggle, mode fast/live/debug, workers), Agent inspector (id, seed, episode,
  state, last action/reward, ep reward, coords, target block, obs image, action history,
  debug-step buttons when paused), Graphs (reward, sps, policy loss, value loss, entropy,
  success, ep length, cpu/mem; selectable time range; bounded rolling history), Console/logs,
  Checkpoint browser (load / evaluate without overwriting run), Glossary tooltip explaining
  Environments vs Agents vs Policies vs Workers.
- ONE shared policy batched over all envs. Never 1000 networks.
- No fake data: every number/visual comes from real telemetry.
- Trainer: PPO with GRU policy; pause/resume/stop with final ckpt; config changes queued to
  rollout boundaries; atomic checkpoints (model, optimizer, config, counters, curriculum, RNG
  states, schema version, metadata); full history to JSONL log; rolling in-memory history
  bounded; EMA-based ETA; psutil CPU/RSS; memory warnings; env-scaling benchmark script;
  forward-hook activations only for selected env + selected layers at nn_hz, only when subscribed.
- Modes: fast (no viz/instrumentation), live (throttled viz + nn), debug (fixed seeds, step-by-step).
