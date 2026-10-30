"""Recurrent PPO: rollout buffer, batched inference, GAE, truncated-BPTT updates.

v4 additions (all flags in RecurrentPPO):
- two value streams: extrinsic (episodic, gamma) and intrinsic (RND, non-episodic, gamma_int);
  advantages are added; each stream has a critic `value` = MSE scalar or HL-Gauss classifier;
- HL-Gauss (Farebrother et al. 2024): categorical value over fixed bins, target = Gaussian CDF mass
  per bin with sigma = 0.75 * bin width;
- RND predictor loss on the detached fused feature (Burda et al. 2018);
- next-achievement auxiliary cross-entropy (Moon et al. 2023, Achievement Distillation), 14 items + none.

GPU backend (gpusim): the same functions accept torch tensors on the CUDA device. Numpy inputs keep
the CPU behaviour; torch inputs stay on the device (no per-step host copies). `amp=True` runs the
network under fp16 autocast with a GradScaler; losses and heads are computed in fp32.
"""
import contextlib
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from . import heads
from .envs import NUM_ACTIONS, NUM_ITEMS, NUM_MSGS, V7
from .model import VALUE_BINS

VALUE_SUPPORT = 30.0      # HL-Gauss support [-V, V] in scaled-return units
HL_SIGMA_BINS = 0.75      # sigma = 0.75 * bin width
GAMMA_INT = 0.99          # intrinsic (RND) discount, non-episodic
ACH_BITS = NUM_ITEMS      # achievement bits 0..NUM_ITEMS-1 (ITEMS order; chest is the last one in v6)
NONE_CLASS = NUM_ITEMS    # aux class index for "no further achievement in this episode"
ACH_MASK = (1 << ACH_BITS) - 1


def _autocast(enabled):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class Rollout:
    """Time-major buffers [T, N, ...]. hid[t] is the GRU input hidden state at step t.
    prev[t]: achievement mask the step's events are measured against (episode start mask at a new
    episode); mask[t]: achievement mask after step t.
    With `device` set, every buffer is a torch tensor on that device (GPU backend)."""

    def __init__(self, T, N, res, vec_dim, gru_hidden, device=None, act_cols=0):
        self.T, self.N = T, N
        self.device = device
        # act_cols 0: act is [T, N] (v6). act_cols 3: [T, N, 3] = (primary, recipe, item) (v7).
        atail = (int(act_cols),) if act_cols else ()
        if device is None:
            z = np.zeros
            self.img = z((T, N, res, res, 2), np.uint8)
            self.vec = z((T, N, vec_dim), np.float32)
            self.hid = z((T, N, gru_hidden), np.float32)
            self.act = z((T, N) + atail, np.int64)
            self.msg = z((T, N), np.int64)        # v6 chat token per agent step
            self.logp = z((T, N), np.float32)
            self.val = z((T, N), np.float32)
            self.val_int = z((T, N), np.float32)
            self.rnd = z((T, N), np.float32)      # raw RND error (before normalization)
            self.rew = z((T, N), np.float32)
            self.done = z((T, N), np.float32)
            self.mask = z((T, N), np.int64)
            self.prev = z((T, N), np.int64)
        else:
            d = device
            self.img = torch.zeros((T, N, res, res, 2), dtype=torch.uint8, device=d)
            self.vec = torch.zeros((T, N, vec_dim), dtype=torch.float32, device=d)
            self.hid = torch.zeros((T, N, gru_hidden), dtype=torch.float32, device=d)
            self.act = torch.zeros((T, N) + atail, dtype=torch.int64, device=d)
            self.msg = torch.zeros((T, N), dtype=torch.int64, device=d)
            self.logp = torch.zeros((T, N), dtype=torch.float32, device=d)
            self.val = torch.zeros((T, N), dtype=torch.float32, device=d)
            self.val_int = torch.zeros((T, N), dtype=torch.float32, device=d)
            self.rnd = torch.zeros((T, N), dtype=torch.float32, device=d)
            self.rew = torch.zeros((T, N), dtype=torch.float32, device=d)
            self.done = torch.zeros((T, N), dtype=torch.float32, device=d)
            self.mask = torch.zeros((T, N), dtype=torch.int64, device=d)
            self.prev = torch.zeros((T, N), dtype=torch.int64, device=d)
        self.t = 0


def _pick(logits, sample):
    """Sampled (Gumbel-max, same distribution as Categorical.sample) or greedy index of logits [N, K]."""
    if sample:
        u = torch.rand_like(logits).clamp_(1e-10, 1.0)
        return (logits - torch.log(-torch.log(u))).argmax(-1)
    return logits.argmax(-1)


@torch.inference_mode()
def act(model, img, vec, h, sample=True, value_mode="hlgauss", rnd=False, amp=False):
    """Batched inference for N agent rows with one shared policy.

    numpy img/vec (CPU backend) -> numpy outputs; torch img/vec -> torch outputs. h is a torch tensor
    on the same device. Returns (action, msg, logp, value_ext, value_int, h_new, rnd_error); logp is
    the joint log-prob of (action, msg).

    Sampling is Gumbel-max over the logits without the Distribution object, whose validation and
    wrapping cost ~1 ms per call at 64 envs."""
    on_device = isinstance(img, torch.Tensor)
    dev = h.device
    if on_device:
        img_t, vec_t = img, vec
    else:
        img_t = torch.from_numpy(np.ascontiguousarray(img)).to(dev, non_blocking=True)
        vec_t = torch.from_numpy(np.ascontiguousarray(vec)).to(dev, non_blocking=True)
    v7 = bool(getattr(model, "v7", False))
    with _autocast(amp):
        feat = model.encode(img_t, vec_t)
        out, hn = model.recur(feat, h)
        logits = model.policy_head(out).float()
        mlogits = model.msg_head(out).float()
        if v7:
            rlogits = model.recipe_head(out).float()
            ilogits = model.item_head(out).float()
        v_ext = value_estimate(model, out, "ext", value_mode).float()
        v_int = value_estimate(model, out, "int", value_mode).float()
        err = model.rnd_error(feat).float() if rnd else torch.zeros(feat.shape[0], device=dev)
    logp_all = torch.log_softmax(logits, -1)
    mlogp_all = torch.log_softmax(mlogits, -1)
    a = _pick(logits, sample)
    m = _pick(mlogits, sample)
    lp = logp_all.gather(-1, a[:, None])[:, 0] + mlogp_all.gather(-1, m[:, None])[:, 0]
    if v7:
        # recipe / item heads: masked by the inventory and active only for their primaries (heads.py)
        rmask, imask, r_act, i_act = heads.masks(vec_t.float(), a)
        rl = heads.masked(rlogits, rmask)
        il = heads.masked(ilogits, imask)
        rv = _pick(rl, sample)
        iv = _pick(il, sample)
        rlp = torch.log_softmax(rl, -1).gather(-1, rv[:, None])[:, 0]
        ilp = torch.log_softmax(il, -1).gather(-1, iv[:, None])[:, 0]
        lp = lp + rlp * r_act.float() + ilp * i_act.float()
        a = torch.stack([a, rv, iv], -1)
    if on_device:
        return a, m, lp, v_ext, v_int, hn.float(), err
    return a.numpy(), m.numpy(), lp.numpy(), v_ext.numpy(), v_int.numpy(), hn, err.numpy()


def act_by_slot(models, img, vec, h, sample=True, value_mode="hlgauss", rnd=False):
    """Population inference (PROTOCOL 10.1): row i = w*K + k is driven by models[k] (team slot k).
    numpy img/vec/h rows are split by slot (rows p, p+K, ...), each slot runs its own policy, and the
    outputs are scattered back to global row order. Same return tuple as `act`."""
    K = len(models)
    N = img.shape[0]
    if N % K:
        raise ValueError(f"{N} rows do not split into {K} slots")
    tail = (3,) if getattr(models[0], "v7", False) else ()
    a = np.empty((N,) + tail, np.int64)
    m = np.empty(N, np.int64)
    lp = np.empty(N, np.float32)
    v = np.empty(N, np.float32)
    vi = np.empty(N, np.float32)
    err = np.empty(N, np.float32)
    hn = torch.empty((N, h.shape[1]), dtype=h.dtype, device=h.device)
    for p, model in enumerate(models):
        sl = slice(p, None, K)
        ap, mp, lpp, vp, vip, hnp, errp = act(model, img[sl], vec[sl], h[sl], sample=sample,
                                             value_mode=value_mode, rnd=rnd)
        a[sl], m[sl], lp[sl], v[sl], vi[sl], err[sl] = ap, mp, lpp, vp, vip, errp
        hn[sl] = hnp
    return a, m, lp, v, vi, hn, err


def policy_act(model, img, vec, h, sample=True):
    """Policy-only inference (eval, debug, dashboard). Returns (action, msg, logp, value_ext, h_new)."""
    a, m, lp, v, _vi, hn, _err = act(model, img, vec, h, sample=sample)
    return a, m, lp, v, hn


# ----- value heads -----
def value_estimate(model, o, which, mode):
    """Expected scalar value from the selected stream's head. o: [..., gru]."""
    v, _ = _value_pair(model, o, which, mode)
    return v


def _value_pair(model, o, which, mode):
    if mode == "hlgauss":
        lg = (model.value_cls if which == "ext" else model.int_value_cls)(o)
        return hl_expect(lg), lg
    head = model.value_head if which == "ext" else model.int_value_head
    return head(o).squeeze(-1), None


def hl_centers(device=None):
    return torch.linspace(-VALUE_SUPPORT, VALUE_SUPPORT, VALUE_BINS, device=device)


def hl_expect(logits):
    p = torch.softmax(logits.float(), -1)
    return (p * hl_centers(logits.device)).sum(-1)


def hl_gauss_loss(logits, target):
    """Cross-entropy to the HL-Gauss target: the Gaussian(target, sigma) mass in each bin."""
    logits = logits.float()
    centers = hl_centers(logits.device)
    w = float(centers[1] - centers[0])
    edges = torch.cat([centers - w / 2, centers[-1:] + w / 2])          # [BINS+1]
    tgt = target.float().clamp(-VALUE_SUPPORT, VALUE_SUPPORT)
    z = (edges - tgt.unsqueeze(-1)) / (HL_SIGMA_BINS * w)
    cdf = torch.special.ndtr(z)
    probs = cdf[..., 1:] - cdf[..., :-1]
    probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-8)
    return -(probs * torch.log_softmax(logits, -1)).sum(-1).mean()


def value_loss(v, lg, target, mode):
    if mode == "hlgauss":
        return hl_gauss_loss(lg, target)
    return 0.5 * (target - v.float()).pow(2).mean()


def next_ach_targets(mask, prev, done):
    """Aux targets [T,N] in 0..14: the next item (bit index) newly obtained later in the same
    episode within this rollout, else NONE_CLASS. Events: bits set in mask[t] but not prev[t].
    numpy in -> numpy out; torch in -> torch out (same values)."""
    if isinstance(mask, torch.Tensor):
        T = mask.shape[0]
        new = (mask & ACH_MASK) & ~(prev & ACH_MASK)
        ev = torch.full_like(mask, -1)
        for b in reversed(range(ACH_BITS)):
            ev = torch.where(((new >> b) & 1) == 1, torch.full_like(ev, b), ev)
        tgt = torch.full_like(mask, NONE_CLASS)
        for t in reversed(range(T - 1)):
            cont = done[t] == 0
            nxt = torch.where(ev[t + 1] >= 0, ev[t + 1], tgt[t + 1])
            tgt[t] = torch.where(cont, nxt, torch.full_like(nxt, NONE_CLASS))
        return tgt
    T = mask.shape[0]
    new = (mask & ACH_MASK) & ~(prev & ACH_MASK)
    ev = np.full(mask.shape, -1, np.int64)
    for b in reversed(range(ACH_BITS)):          # lowest newly set bit wins
        ev = np.where((new >> b) & 1 == 1, b, ev)
    tgt = np.full(mask.shape, NONE_CLASS, np.int64)
    for t in reversed(range(T - 1)):
        cont = done[t] == 0
        nxt = np.where(ev[t + 1] >= 0, ev[t + 1], tgt[t + 1])
        tgt[t] = np.where(cont, nxt, NONE_CLASS)
    return tgt


def gae(rew, val, done, last_val, gamma, lam):
    """rew/val/done are [T, N]; done[t] is 1 when the episode ended after step t.
    numpy in -> numpy out; torch in -> torch out."""
    if isinstance(rew, torch.Tensor):
        T = rew.shape[0]
        adv = torch.zeros_like(rew)
        last = torch.zeros_like(rew[0])
        for t in reversed(range(T)):
            nxt_v = last_val if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + gamma * nxt_v * nonterm - val[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        return adv, adv + val
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1], np.float32)
    for t in reversed(range(T)):
        nxt_v = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt_v * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def _var(x):
    if isinstance(x, torch.Tensor):
        return float(x.double().var(unbiased=False))
    return float(np.var(x))


def load_optimizer_state(opt, saved):
    """Load Adam state saved before new parameters were added: the old parameters keep their
    indices (new modules are registered last), the new ones start with fresh moments."""
    n_new = sum(len(g["params"]) for g in opt.param_groups)
    groups = copy.deepcopy(saved["param_groups"])
    n_saved = sum(len(g["params"]) for g in groups)
    if n_saved == n_new:
        opt.load_state_dict(saved)
        return 0
    if n_saved > n_new or len(groups) != 1:
        raise ValueError(f"optimizer state has {n_saved} params, model has {n_new}")
    groups[0]["params"] = list(range(n_new))
    opt.load_state_dict({"state": saved["state"], "param_groups": groups})
    return n_new - n_saved


def _to_dev(x, dev):
    return x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x)).to(dev)


class RecurrentPPO:
    """curiosity: 'rnd' (RND error, predictor trained), 'spr' (SPR latent error, computed per rollout)
    or 'none'. `rnd` below means "intrinsic stream on" (curiosity != 'none')."""

    def __init__(self, model, lr=3e-4, gamma=0.995, lam=0.95, clip=0.2, ent_coef=0.01,
                 vf_coef=0.5, max_grad_norm=0.5, epochs=2, bptt=16, minibatch_size=4096,
                 value_mode="hlgauss", rnd=False, aux=False, aux_coef=0.1, rnd_norm=None,
                 curiosity="rnd", spr=False, spr_coef=1.0, spr_k=5, spr_tau=0.99, weight_decay=0.0,
                 amp=False, msg_ent_coef=None):
        self.model = model
        self.v7 = bool(getattr(model, "v7", False))
        if self.v7 and (spr or aux):
            raise ValueError("v7 observation: --spr and --aux-ach are not supported (SPR's action input and the "
                             "achievement targets are v6 layouts); run them with --obs v6")
        self.amp = bool(amp)
        # v6: entropy bonus of the chat-token head (default = the action entropy coefficient)
        self.msg_ent_coef = float(ent_coef if msg_ent_coef is None else msg_ent_coef)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp) if self.amp else None
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.epochs, self.bptt, self.minibatch_size = epochs, bptt, minibatch_size
        self.value_mode, self.aux, self.aux_coef = value_mode, aux, aux_coef
        self.curiosity = curiosity if rnd else "none"
        self.rnd = self.curiosity != "none"          # intrinsic value stream on
        self.rnd_train = self.curiosity == "rnd"     # RND predictor loss on
        self.spr, self.spr_coef, self.spr_k, self.spr_tau = spr, spr_coef, spr_k, spr_tau
        self.wd = float(weight_decay)
        # AdamW-style decoupled decay on weight matrices/convs only (not biases, norms, embeddings)
        self.decay_params = [p for n, p in model.named_parameters()
                             if p.ndim >= 2 and p.requires_grad and "block_embed" not in n]
        self.target = None
        # DAPG-style demo loss (run.py --demos): a DemoSampler with a `coef` attribute, set by the trainer
        self.demo = None
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        if self.spr:
            self.reset_target()

    @property
    def dev(self):
        return next(self.model.parameters()).device

    def reset_target(self):
        """SPR target encoder: an EMA copy of the online model (rebuilt on load; not checkpointed)."""
        self.target = copy.deepcopy(self.model)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _ema_update(self):
        torch._foreach_mul_(list(self.target.parameters()), self.spr_tau)
        torch._foreach_add_(list(self.target.parameters()), list(self.model.parameters()),
                            alpha=1.0 - self.spr_tau)

    def _post_step(self):
        lr = self.opt.param_groups[0]["lr"]
        if self.wd > 0:
            with torch.no_grad():
                for p in self.decay_params:
                    p.mul_(1.0 - lr * self.wd)
        if self.spr:
            self._ema_update()

    @torch.no_grad()
    def spr_intrinsic(self, roll, last_img, last_vec):
        """Curiosity = one-step SPR latent prediction error per rollout step, [T, N] (0 at episode
        ends). Uses the online projector/transition and the EMA target encoder."""
        T, N = roll.T, roll.N
        dev = self.dev
        ri, rv = _to_dev(roll.img, dev), _to_dev(roll.vec, dev)
        li, lv = _to_dev(last_img, dev), _to_dev(last_vec, dev)
        nxt_img = torch.cat([ri[1:], li[None]], 0).reshape(T * N, *ri.shape[2:])
        nxt_vec = torch.cat([rv[1:], lv[None]], 0).reshape(T * N, -1)
        cur_img = ri.reshape(T * N, *ri.shape[2:])
        cur_vec = rv.reshape(T * N, -1)
        m, tg = self.model, self.target
        m.eval()
        with _autocast(self.amp):
            z = m.spr_proj(m.encode(cur_img, cur_vec))
            acts = F.one_hot(_to_dev(roll.act, dev).reshape(-1), NUM_ACTIONS).float()
            pred = m.spr_pred(m.spr_trans(torch.cat([z, acts], -1)))
            tgt = tg.spr_proj(tg.encode(nxt_img, nxt_vec))
        err = 1.0 - F.cosine_similarity(pred.float(), tgt.float(), dim=-1)
        err = err.reshape(T, N).float()
        done = _to_dev(roll.done, dev)
        return err * (1.0 - done)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def _spr_loss(self, m, z, tproj, acts, dones):
        """SPR: from z_t and actions a_t..a_{t+k-1} predict the target projection of z_{t+k}, for
        k = 1..K (BYOL-style cosine loss through the predictor). Pairs that cross an episode end are
        masked. z, tproj: [B, L, P]; acts, dones: [B, L]."""
        B, L = acts.shape
        K = min(self.spr_k, L - 1)
        Lv = L - K
        base = z[:, :Lv]
        valid = torch.ones(B, Lv, device=z.device)
        total = 0.0
        for k in range(1, K + 1):
            a = acts[:, k - 1:k - 1 + Lv]
            base = m.spr_trans(torch.cat([base, F.one_hot(a, NUM_ACTIONS).float()], -1))
            valid = valid * (1.0 - dones[:, k - 1:k - 1 + Lv])
            pred = m.spr_pred(base)
            cos = F.cosine_similarity(pred.float(), tproj[:, k:k + Lv].float(), dim=-1)
            total = total + ((1.0 - cos) * valid).sum() / valid.sum().clamp_min(1.0)
        return total / K

    def _chunk_len(self, T):
        for L in range(min(self.bptt, T), 0, -1):
            if T % L == 0:
                return L
        return 1

    def update(self, r, last_val, last_val_int=None, pump=None, grad_hook=None, rew=None,
               rew_int=None):
        """One PPO update over a full Rollout. `rew` / `rew_int` are the scaled extrinsic and
        intrinsic rewards (default: raw extrinsic, zero intrinsic). Returns averaged stats.
        Torch rollouts (GPU backend) stay on the device; numpy rollouts (CPU backend) are moved over."""
        T, N = r.T, r.N
        dev = self.dev
        torch_in = isinstance(r.val, torch.Tensor)
        last_val = _to_dev(last_val, dev) if torch_in else last_val
        rew_ = r.rew if rew is None else rew
        adv, ret = gae(rew_, r.val, r.done, last_val, self.gamma, self.lam)
        if self.rnd:
            ri = (torch.zeros_like(r.rew) if torch_in else np.zeros_like(r.rew)) if rew_int is None else rew_int
            zeros_done = torch.zeros_like(r.done) if torch_in else np.zeros_like(r.done)
            if last_val_int is None:
                lvi = torch.zeros(N, device=dev) if torch_in else np.zeros(N, np.float32)
            else:
                lvi = _to_dev(last_val_int, dev) if torch_in else last_val_int
            adv_i, ret_i = gae(ri, r.val_int, zeros_done, lvi, GAMMA_INT, self.lam)
            adv = adv + adv_i
        else:
            ret_i = torch.zeros_like(ret) if torch_in else np.zeros_like(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tgt_aux = next_ach_targets(r.mask, r.prev, r.done) if self.aux else None

        L = self._chunk_len(T)
        C = T // L
        m = self.model

        def chunked(a):  # [T, N, ...] -> [C, N, L, ...]
            a = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a))
            return a.reshape(C, L, N, *a.shape[2:]).transpose(1, 2)

        imgC = chunked(r.img)
        vecC = chunked(r.vec)
        actC = chunked(r.act)
        msgC = chunked(r.msg)
        logpC = chunked(r.logp)
        advC = chunked(adv)
        retC = chunked(ret)
        retIC = chunked(ret_i)
        doneC = chunked(r.done)
        hidC = chunked(r.hid)
        auxC = chunked(tgt_aux) if tgt_aux is not None else None
        if imgC.device != dev:
            imgC, vecC, actC, msgC, logpC, advC, retC, retIC, doneC, hidC = [
                x.to(dev) for x in (imgC, vecC, actC, msgC, logpC, advC, retC, retIC, doneC, hidC)]
            auxC = auxC.to(dev) if auxC is not None else None

        n_seq = C * N
        seq_per_mb = max(1, self.minibatch_size // L)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [],
                 "rnd_loss": [], "aux_loss": [], "aux_ach_acc": [], "spr_loss": [], "msg_entropy": [],
                 "recipe_entropy": [], "item_entropy": [], "demo_ce": [], "demo_acc_primary": [],
                 "demo_acc_recipe": [], "demo_acc_item": []}
        v7 = self.v7
        skipped = 0
        m.train()
        amp = self.amp
        for _ in range(self.epochs):
            perm = torch.randperm(n_seq, device=dev)
            for start in range(0, n_seq, seq_per_mb):
                idx = perm[start:start + seq_per_mb]
                ci = idx // N
                ni = idx % N
                B = idx.numel()
                img = imgC[ci, ni].reshape(B * L, *imgC.shape[3:])
                vec = vecC[ci, ni].reshape(B * L, -1)
                with _autocast(amp):
                    feat = m.encode(img, vec).view(B, L, -1)
                    h = hidC[ci, ni, 0]
                    outs = []
                    for t in range(L):
                        if t > 0:
                            h = h * (1.0 - doneC[ci, ni, t - 1]).unsqueeze(-1)
                        out, h = m.recur(feat[:, t], h)
                        outs.append(out)
                    o = torch.stack(outs, 1)
                    logits = m.policy_head(o).float()
                    mlogits = m.msg_head(o).float()
                    v, lg = _value_pair(m, o, "ext", self.value_mode)
                    v = v.float()
                    dist = Categorical(logits=logits, validate_args=False)
                    mdist = Categorical(logits=mlogits, validate_args=False)
                    act_all = actC[ci, ni]
                    if v7:
                        act_ = act_all[..., 0]
                        rec_ = act_all[..., 1]
                        itm_ = act_all[..., 2]
                    else:
                        act_ = act_all
                    logp = dist.log_prob(act_) + mdist.log_prob(msgC[ci, ni])
                    ent_extra = 0.0
                    if v7:
                        Bl = B * L
                        rmask, imask, r_act, i_act = heads.masks(vec.float(), act_.reshape(Bl))
                        rl = heads.masked(m.recipe_head(o).float(), rmask.view(B, L, -1))
                        il = heads.masked(m.item_head(o).float(), imask.view(B, L, -1))
                        dist_r = Categorical(logits=rl, validate_args=False)
                        dist_i = Categorical(logits=il, validate_args=False)
                        ra = r_act.view(B, L).float()
                        ia = i_act.view(B, L).float()
                        logp = logp + dist_r.log_prob(rec_) * ra + dist_i.log_prob(itm_) * ia
                        ent_r = (dist_r.entropy() * ra).mean()
                        ent_i = (dist_i.entropy() * ia).mean()
                        ent_extra = ent_r + ent_i
                        stats["recipe_entropy"].append(ent_r.detach())
                        stats["item_entropy"].append(ent_i.detach())
                    old_logp = logpC[ci, ni]
                    a = advC[ci, ni]
                    rr = retC[ci, ni]
                    ratio = torch.exp(logp - old_logp)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a).mean()
                    vl = value_loss(v, lg, rr, self.value_mode)
                    ent = dist.entropy().mean()
                    ent_m = mdist.entropy().mean()
                    loss = (pg + self.vf_coef * vl - self.ent_coef * (ent + ent_extra)
                            - self.msg_ent_coef * ent_m)
                    if self.rnd:
                        vi, lgi = _value_pair(m, o, "int", self.value_mode)
                        vli = value_loss(vi.float(), lgi, retIC[ci, ni], self.value_mode)
                        loss = loss + self.vf_coef * vli
                        if self.rnd_train:
                            fd = feat.detach().float()
                            rnd_l = (m.rnd_pred(fd).float() - m.rnd_target(fd).float()).pow(2).mean()
                            loss = loss + rnd_l
                            stats["rnd_loss"].append(rnd_l.detach())
                    if self.spr:
                        with torch.no_grad(), _autocast(amp):
                            tproj = self.target.spr_proj(self.target.encode(img, vec).view(B, L, -1))
                        z = m.spr_proj(feat)
                        spr_l = self._spr_loss(m, z, tproj, actC[ci, ni], doneC[ci, ni])
                        loss = loss + self.spr_coef * spr_l
                        stats["spr_loss"].append(spr_l.detach())
                    if self.aux:
                        alog = m.aux_head(o).float()
                        tg = auxC[ci, ni]
                        aux_l = F.cross_entropy(alog.reshape(-1, alog.shape[-1]), tg.reshape(-1))
                        loss = loss + self.aux_coef * aux_l
                        stats["aux_loss"].append(aux_l.detach())
                        stats["aux_ach_acc"].append((alog.argmax(-1) == tg).float().mean().detach())
                    dm = self.demo
                    if dm is not None and dm.coef > 0:
                        # demo batch (one per minibatch, own draw per population member): masked multi-head CE
                        # of the expert actions, no value term (see trainer/demos.py)
                        d_loss, d_st = dm.loss(m)
                        loss = loss + dm.coef * d_loss
                        stats["demo_ce"].append(d_st["loss"])
                        stats["demo_acc_primary"].append(d_st["ok_prim"] / max(1, d_st["n"]))
                        if d_st["n_rec"]:
                            stats["demo_acc_recipe"].append(d_st["ok_rec"] / d_st["n_rec"])
                        if d_st["n_item"]:
                            stats["demo_acc_item"].append(d_st["ok_item"] / d_st["n_item"])
                if not torch.isfinite(loss):
                    skipped += 1
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.opt.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                else:
                    loss.backward()
                if grad_hook is not None:
                    grad_hook(m)
                torch.nn.utils.clip_grad_norm_(m.parameters(), self.max_grad_norm)
                if self.scaler is not None:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                self._post_step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - old_logp)).mean()
                stats["policy_loss"].append(pg.detach())
                stats["value_loss"].append(vl.detach())
                stats["entropy"].append(ent.detach())
                stats["msg_entropy"].append(ent_m.detach())
                stats["approx_kl"].append(kl.detach())
                if pump is not None:
                    pump()
        m.eval()
        out = {}
        for k, v in stats.items():
            if v:
                out[k] = float(torch.stack([torch.as_tensor(x, dtype=torch.float32, device=dev) for x in v]).mean())
            else:
                out[k] = None
        out["skipped_nonfinite"] = skipped
        out["explained_var"] = float(1.0 - _var(ret - r.val) / (_var(ret) + 1e-8))
        return out


class ReturnNormalizer:
    """Reward scaling by the running std of the discounted return (VecNormalize-style).

    The +10 goal reward makes raw value targets huge and badly scaled; dividing rewards by the
    return std keeps the value loss and advantages in a stable range. Only the scale is learned
    (no mean subtraction), so reward signs and ordering are unchanged. Stats are checkpointed."""

    def __init__(self, gamma, clip=10.0):
        self.gamma, self.clip = gamma, clip
        self.ret = None
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def reset_envs(self, N):
        self.ret = np.zeros(N, np.float64)

    def update(self, rew, done):
        """rew, done: [T, N] numpy. Updates running return statistics."""
        if self.ret is None or self.ret.shape[0] != rew.shape[1]:
            self.reset_envs(rew.shape[1])
        T = rew.shape[0]
        rets = np.empty(rew.shape, np.float64)
        run = self.ret.copy()
        for t in range(T):
            run = run * self.gamma + rew[t]
            rets[t] = run
            run = run * (1.0 - done[t])
        self.ret = run
        flat = rets.reshape(-1)
        b_count = flat.size
        b_mean, b_var = float(flat.mean()), float(flat.var())
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta * delta * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot

    def scale(self, rew):
        s = math.sqrt(self.var + 1e-8)
        if isinstance(rew, torch.Tensor):
            return (rew / s).clamp(-self.clip, self.clip)
        return np.clip(rew / s, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd):
        self.mean, self.var, self.count = float(sd["mean"]), float(sd["var"]), float(sd["count"])


class RunningStd:
    """Running variance of raw RND errors (only the scale is used; no mean subtraction)."""

    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1)
        b_count = x.size
        b_mean, b_var = float(x.mean()), float(x.var())
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta * delta * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot

    def scale(self, x):
        return np.asarray(x, np.float32) / math.sqrt(self.var + 1e-8)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd):
        self.mean, self.var, self.count = float(sd["mean"]), float(sd["var"]), float(sd["count"])
