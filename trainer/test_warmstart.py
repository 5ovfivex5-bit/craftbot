"""Warm-start tests (PROTOCOL 6.8 / 6.10): v5 checkpoint -> v6 model.

Run:  .venv/bin/python -m trainer.test_warmstart      (pytest is not installed)

Exact parts, asserted to 1e-4:
  - block_embed rows 0..17, policy_head rows 0..22 and value_head are copies;
  - the vec input layer: a v6 vector with teammate/chest/crosshair features zero gives the same
    vec_mlp output as the v5 vector it came from;
  - img_fc input expansion 4x4 -> 8x8: for a conv feature map that is constant on 2x2 blocks, the v6
    img_fc on the 8x8 map equals the v5 img_fc on the 4x4 map (channel-major flatten).

Not exact (asserted as closeness, with the numbers printed): the end-to-end step-0 policy. A 3x3
stride-2 conv at 64x64 does not see the same pixels as the 32x32 conv on the 2x upsampled image: its
three taps read X32[j-1], X32[j], X32[j] (the duplicated pixel), where the v5 conv reads X32[j-1],
X32[j], X32[j+1]. No weight assignment reproduces the lost X32[j+1] tap, and copying the conv weights
as-is already gives the best mapping of the taps (checked: remapping the 1st and 2nd tap rows gives
the same outputs). The error is therefore a property of the architecture change, not of the mapping,
and it is largest where neighbouring pixels differ (block edges, the depth channel). The observations
used are real sim observations (single-agent, res 32) upsampled 2x (nearest) to 64.
"""
import os
import sys

import numpy as np
import torch

from . import checkpoint as _ckpt  # noqa: F401  (keeps the package import path identical to the trainer)
from .model import MineMindPolicy
from .warmstart import load_warm_start

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5_CKPT = os.path.join(ROOT, "runs", "v5-off", "checkpoints", "ckpt_000013000704_u003174.pt")
EXACT_ATOL = 1e-4


def _load_v5_model(src):
    m5 = MineMindPolicy(res=32, vec_dim=39, num_actions=23, num_blocks=18, size="small",
                        resblock=False, num_msgs=8).eval()
    # aux_head / spr_trans have v5 shapes (15 classes, 151 inputs): unused by the forward pass
    sd = {k: v for k, v in src["model"].items()
          if k in m5.state_dict() and m5.state_dict()[k].shape == v.shape}
    m5.load_state_dict(sd, strict=False)
    return m5.eval()


def _load_v6(src_path):
    m6 = MineMindPolicy(res=64, vec_dim=102, num_actions=28, num_blocks=20, size="small",
                        resblock=False, num_msgs=8)
    src, report = load_warm_start(src_path, m6)
    return m6.eval(), src, report


def _real_observations(n=1024, seed=3):
    """Real v6-sim observations at res 32 (single-agent worlds), random actions."""
    import craftsim as cs
    sim = cs.CraftSim(num_envs=64, seed=seed, res=32, agents=1, action_mode="v6")
    sim.reset()
    rng = np.random.default_rng(seed)
    imgs, vecs = [], []
    while sum(len(x) for x in imgs) < n:
        img, vec, _, _, _ = sim.step(rng.integers(0, cs.NUM_ACTIONS, sim.n).astype(np.int32))
        imgs.append(img.copy())
        vecs.append(vec.copy())
    return np.concatenate(imgs)[:n], np.concatenate(vecs)[:n].astype(np.float32)


def test_exact_row_and_column_copies():
    src = torch.load(V5_CKPT, map_location="cpu", weights_only=False)
    m6, _, _ = _load_v6(V5_CKPT)
    sd6 = m6.state_dict()
    assert torch.equal(sd6["block_embed.weight"][:18], src["model"]["block_embed.weight"])
    assert torch.equal(sd6["policy_head.weight"][:23], src["model"]["policy_head.weight"])
    assert torch.equal(sd6["policy_head.bias"][:23], src["model"]["policy_head.bias"])
    expect_bias = src["model"]["policy_head.bias"].mean()
    assert torch.allclose(sd6["policy_head.bias"][23:], expect_bias.expand(5), atol=EXACT_ATOL)
    assert torch.equal(sd6["value_head.weight"], src["model"]["value_head.weight"])
    assert torch.equal(sd6["conv2.0.weight"], src["model"]["conv2.0.weight"])
    # teammate, chest and crosshair input columns are zero
    W = sd6["vec_mlp.0.weight"]
    assert float(W[:, 14].abs().max()) == 0.0 and float(W[:, 40:].abs().max()) == 0.0


def test_vec_input_mapping_exact():
    src = torch.load(V5_CKPT, map_location="cpu", weights_only=False)
    m5 = _load_v5_model(src)
    m6, _, _ = _load_v6(V5_CKPT)
    rng = np.random.default_rng(0)
    B = 32
    inv = rng.integers(0, 6, (B, 14)).astype(np.float32)
    scal = rng.uniform(-1, 1, (B, 11)).astype(np.float32)
    goal = np.zeros((B, 14), np.float32)
    goal[np.arange(B), rng.integers(0, 14, B)] = 1
    v5 = torch.from_numpy(np.concatenate([inv, scal, goal], 1))
    v6 = torch.zeros(B, 102)
    v6[:, 0:14] = torch.from_numpy(inv)
    v6[:, 15:26] = torch.from_numpy(scal)
    v6[:, 26:40] = torch.from_numpy(goal)
    with torch.no_grad():
        d = (m6.vec_mlp(v6) - m5.vec_mlp(v5)).abs().max()
    assert float(d) < EXACT_ATOL, f"vec_mlp mismatch {float(d):.2e}"


def test_img_fc_expansion_exact():
    src = torch.load(V5_CKPT, map_location="cpu", weights_only=False)
    m5 = _load_v5_model(src)
    m6, _, _ = _load_v6(V5_CKPT)
    torch.manual_seed(0)
    f4 = torch.randn(16, 32, 4, 4)
    f8 = f4.repeat_interleave(2, 2).repeat_interleave(2, 3)     # constant on 2x2 blocks
    with torch.no_grad():
        d = (m5.img_fc(f4.flatten(1)) - m6.img_fc(f8.flatten(1))).abs().max()
    assert float(d) < EXACT_ATOL, f"img_fc expansion mismatch {float(d):.2e}"


def test_step0_policy_closeness_real_observations():
    """End-to-end step-0 comparison on real observations (see module docstring for why not 1e-4)."""
    src = torch.load(V5_CKPT, map_location="cpu", weights_only=False)
    m5 = _load_v5_model(src)
    m6, _, _ = _load_v6(V5_CKPT)
    img32, vec6 = _real_observations()
    vec5 = np.concatenate([vec6[:, 0:14], vec6[:, 15:26], vec6[:, 26:40]], 1).astype(np.float32)
    img64 = np.repeat(np.repeat(img32, 2, 1), 2, 2)           # 2x2 nearest upsampling
    h = torch.zeros(len(img32), 256)
    with torch.no_grad():
        l5, _, v5, _ = m5(torch.from_numpy(img32), torch.from_numpy(vec5), h)
        l6, _, v6, _ = m6(torch.from_numpy(img64), torch.from_numpy(vec6), h)
    l6 = l6[:, :23]
    dl = (l6 - l5).abs()
    p5 = torch.softmax(l5, -1)
    kl = (p5 * (torch.log(p5) - torch.log_softmax(l6, -1))).sum(-1)
    agree = float((l6.argmax(-1) == l5.argmax(-1)).float().mean())
    dv = (v6 - v5).abs()
    print(f"step-0 v6 vs v5 on {len(img32)} real observations:")
    print(f"  logits |diff| max {float(dl.max()):.4f} mean {float(dl.mean()):.4f} (|logit| mean {float(l5.abs().mean()):.3f})")
    print(f"  value |diff| max {float(dv.max()):.4f} mean {float(dv.mean()):.4f} (|value| mean {float(v5.abs().mean()):.3f})")
    print(f"  policy KL(v5||v6) mean {float(kl.mean()):.4f} max {float(kl.max()):.4f}; argmax agreement {agree:.3f}")
    # measured in the run that wrote this test: logit mean 0.156 (max 3.6), value mean 0.107, KL mean 0.045,
    # argmax agreement 0.891 (value |diff| max 0.90)
    assert float(dl.mean()) < 0.25, "mean logit deviation too large"
    assert float(kl.mean()) < 0.1, "mean policy KL too large"
    assert agree > 0.8, "argmax agreement too low"
    assert float(dv.mean()) < 0.5, "mean value deviation too large"


def test_checkpoint_not_modified():
    before = os.path.getmtime(V5_CKPT), os.path.getsize(V5_CKPT)
    _load_v6(V5_CKPT)
    assert before == (os.path.getmtime(V5_CKPT), os.path.getsize(V5_CKPT))


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
