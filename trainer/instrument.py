"""Neural-network instrumentation for the dashboard.

Forward hooks are registered only while a client is subscribed to `nn` (and mode != fast) and
removed when nobody is. Even while registered they are disarmed except during the instrumented
forward pass for ONE selected env, so training forwards/backwards pay nothing.
"""
import math

import torch

from .model import MineMindPolicy

HIST_BINS = 16
SAMPLE_MAX = 256


def _tensor_stats(t):
    flat = t.detach().float().flatten()
    finite_mask = torch.isfinite(flat)
    nan_inf = not bool(finite_mask.all())
    vals = flat[finite_mask] if nan_inf else flat
    if vals.numel() == 0:
        return None, nan_inf
    mn, mx = float(vals.min()), float(vals.max())
    mean, std = float(vals.mean()), float(vals.std()) if vals.numel() > 1 else 0.0
    if mx > mn:
        counts = torch.histc(vals, bins=HIST_BINS, min=mn, max=mx).long().tolist()
    else:
        counts = [0] * HIST_BINS
        counts[HIST_BINS // 2] = int(vals.numel())
    edges = [mn + (mx - mn) * i / HIST_BINS for i in range(HIST_BINS + 1)]
    n = vals.numel()
    if n > SAMPLE_MAX:
        idx = torch.linspace(0, n - 1, SAMPLE_MAX, device=vals.device).long()
        sample = vals[idx]
    else:
        sample = vals
    return {
        "mean": mean, "std": std, "min": mn, "max": mx,
        "hist": {"edges": edges, "counts": counts},
        "sample": [round(float(x), 6) for x in sample.tolist()],
    }, nan_inf


class Instrument:
    def __init__(self, model: MineMindPolicy):
        self.model = model
        self.layer_names = [n for n, _ in model.named_children()]
        self.active = False
        self._armed = False
        self._handles = []
        self.captured = {}
        self.layers = set(self.layer_names)
        self.grad_norms = {}
        self.weight_stats = {}
        self._weights_fresh = False

    # ----- hook lifecycle -----
    def set_active(self, on):
        on = bool(on)
        if on == self.active:
            return
        self.active = on
        if on:
            for name in self.layer_names:
                mod = getattr(self.model, name)
                self._handles.append(mod.register_forward_hook(self._make_hook(name)))
            self.refresh_weights()
        else:
            for h in self._handles:
                h.remove()
            self._handles = []
            self.captured = {}

    def _make_hook(self, name):
        def hook(_mod, _inp, out):
            if not self._armed:
                return
            t = out[0] if isinstance(out, tuple) else out
            self.captured[name] = t.detach()
        return hook

    def set_layers(self, names):
        if names:
            self.layers = {n for n in names if n in self.layer_names}
        else:
            self.layers = set(self.layer_names)

    # ----- stats -----
    def record_grads(self, model):
        """Called after backward (inside PPO update) while active."""
        norms = {}
        for name in MineMindPolicy.TOP_LEVEL:
            mod = getattr(model, name, None)
            if mod is None:          # v7 heads on a v6 network
                continue
            sq = 0.0
            for p in mod.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().pow(2).sum())
            norms[name] = math.sqrt(sq)
        self.grad_norms = norms

    def refresh_weights(self):
        out = {}
        with torch.no_grad():
            for name in MineMindPolicy.TOP_LEVEL:
                mod = getattr(self.model, name, None)
                if mod is None:
                    continue
                ps = [p.detach().flatten().float() for p in mod.parameters()]
                if not ps:
                    continue
                flat = torch.cat(ps)
                out[name] = {"mean": float(flat.mean()), "std": float(flat.std()),
                             "min": float(flat.min()), "max": float(flat.max())}
        self.weight_stats = out
        self._weights_fresh = True

    def probe(self, img_np, vec_np, h, env_id, step):
        """Instrumented forward for ONE env (batch of 1).

        Returns (activations_payload, new_hidden [1, gru_hidden]) or (None, None) when inactive.
        h is a torch tensor [1, gru_hidden]; the caller's training state is not modified.
        """
        if not self.active:
            return None, None
        if not self._weights_fresh:
            self.refresh_weights()
        dev = next(self.model.parameters()).device
        if isinstance(img_np, torch.Tensor):      # gpusim backend: device tensors
            img_t = img_np[None].clone().to(dev)
            vec_t = vec_np[None].clone().to(dev)
            h = h.to(dev)
        else:
            img_t = torch.from_numpy(img_np[None].copy()).to(dev)
            vec_t = torch.from_numpy(vec_np[None].copy()).to(dev)
        self.captured = {}
        self._armed = True
        try:
            with torch.no_grad():
                logits, mlogits, value, hn = self.model(img_t, vec_t, h)
        finally:
            self._armed = False
        nan_inf = False
        layers = {}
        for name in self.layer_names:
            if name not in self.layers or name not in self.captured:
                continue
            t = self.captured[name][0]  # drop batch dim
            st, bad = _tensor_stats(t)
            nan_inf = nan_inf or bad
            if st is None:
                continue
            st["shape"] = list(t.shape)
            layers[name] = st
        probs = torch.softmax(logits[0], -1)
        ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
        mprobs = torch.softmax(mlogits[0], -1)
        gru_out = hn[0].detach()
        gst, bad = _tensor_stats(gru_out)
        nan_inf = nan_inf or bad
        gru = {"mean": gst["mean"], "std": gst["std"], "min": gst["min"], "max": gst["max"],
               "norm": float(gru_out.norm())} if gst else None
        for p in self.model.parameters():
            if not torch.isfinite(p).all():
                nan_inf = True
                break
        payload = {
            "env_id": int(env_id),
            "step": int(step),
            "action_probs": [round(float(x), 6) for x in probs.tolist()],
            "msg_probs": [round(float(x), 6) for x in mprobs.tolist()],
            "entropy": ent,
            "value": float(value[0]),
            "layers": layers,
            "gru": gru,
            "grad_norms": dict(self.grad_norms),
            "weight_stats": dict(self.weight_stats),
            "nan_inf": bool(nan_inf),
        }
        return payload, hn.detach()
