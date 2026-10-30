"""v7 action heads: primary / recipe / item_select / msg (PROTOCOL 12, src/README_v7.md).

The sim's action columns: [primary (21), recipe (0 none, r + 1), item_select (0 keep, i + 1), msg (0..7)].

Conditional heads (documented choice, see README):
- recipe is active only for primary in {craft, smelt}; its logits are masked to the recipes whose
  ingredients the agent holds (counts decoded from the inventory block of the vec obs); craft / smelt
  never choose "none". If no recipe is feasible the mask is lifted (the sim rejects the craft).
- item_select is active only for primary in {use, drop, give}; its logits are masked to items with
  count > 0 plus "keep".
- an inactive head is restricted to its neutral value (recipe 0, item 0 = keep), so it contributes exactly
  zero to log-prob and entropy and the sim receives the neutral value. Its logits are never sampled.
- msg is always active.
All masks are a pure function of (vec obs, primary action), so the PPO update recomputes the same masks.
"""
import numpy as np
import torch

from . import envs as E

MASK_FILL = -1e9   # finite (not -inf) so masked entropies stay 0 * finite


def _prim_index(name):
    return list(E.ACTIONS).index(name)


if E.V7:
    PRIM_CRAFT = (_prim_index("craft"), _prim_index("smelt"))
    PRIM_ITEM = (_prim_index("use"), _prim_index("drop"), _prim_index("give"))
    PRIM_USE = _prim_index("use")
else:
    PRIM_CRAFT = PRIM_ITEM = ()
    PRIM_USE = -1

INV_LEN = E.NUM_ITEMS if E.V7 else 0


def _recipe_need_matrix():
    """float32 [NUM_RECIPES, NUM_ITEMS]: items each recipe consumes (from craftsim.RECIPES)."""
    import craftsim
    need = np.zeros((E.NUM_RECIPES, E.NUM_ITEMS), np.float32)
    for r, rec in enumerate(craftsim.RECIPES):
        for k, n in rec["ingredients"].items():
            item = E.ITEMS.index(k) if isinstance(k, str) else int(k)
            need[r, item] += float(n)
    return need


if E.V7:
    NEED = torch.from_numpy(_recipe_need_matrix())
else:
    NEED = None


def decode_counts(vec):
    """Inventory counts from the vec obs: column i = log2(1 + n_i) / 7 (README_v7), so n = 2^(7 v) - 1.
    vec: float tensor [..., VEC_DIM] -> float tensor [..., NUM_ITEMS] of whole counts."""
    v = vec[..., :INV_LEN].double()
    return torch.round(torch.exp2(7.0 * v) - 1.0).float()


def masks(vec, primary):
    """vec: float tensor [B, VEC_DIM]; primary: long tensor [B] (sampled / stored primary action).
    Returns (recipe_mask bool [B, NUM_RECIPE_CLASSES], item_mask bool [B, NUM_ITEM_CLASSES],
    recipe_active bool [B], item_active bool [B])."""
    cnt = decode_counts(vec)
    is_craft = torch.zeros_like(primary, dtype=torch.bool)
    for p in PRIM_CRAFT:
        is_craft |= primary == p
    is_item = torch.zeros_like(primary, dtype=torch.bool)
    for p in PRIM_ITEM:
        is_item |= primary == p
    feas = (cnt[:, None, :] >= NEED.to(cnt.device)[None]).all(-1)          # [B, NUM_RECIPES]
    any_feas = feas.any(-1, keepdim=True)
    allowed = torch.where(any_feas, feas, torch.ones_like(feas))
    rmask = torch.zeros(primary.shape[0], E.NUM_RECIPE_CLASSES, dtype=torch.bool, device=vec.device)
    rmask[:, 1:] = allowed & is_craft[:, None]
    rmask[:, 0] = ~is_craft
    imask = torch.zeros(primary.shape[0], E.NUM_ITEM_CLASSES, dtype=torch.bool, device=vec.device)
    imask[:, 1:] = (cnt > 0) & is_item[:, None]
    imask[:, 0] = True
    return rmask, imask, is_craft, is_item


def masked(logits, mask):
    return torch.where(mask, logits, torch.full_like(logits, MASK_FILL))


def pack(prim, rec, item, msg):
    """numpy int arrays [N] -> the sim's [N, 4] action columns."""
    return np.column_stack([prim, rec, item, msg]).astype(np.int32)
