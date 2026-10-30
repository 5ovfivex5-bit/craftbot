"""Warm start a v6 policy from a v5 checkpoint (PROTOCOL 6.8 and 6.10).

v5 layout (res 32, VEC_DIM 39, NUM_BLOCKS 18, NUM_ACTIONS 23) -> v6 layout (res 64, VEC_DIM 102,
NUM_BLOCKS 20, NUM_ACTIONS 28):
  - tensors with equal shapes are copied (conv stack, vec_mlp.2, fuse, gru, value heads, ...);
  - vec_mlp.0 input columns are remapped (6.5): new[0..13] = old[0..13], new[15..25] = old[14..24],
    new[26..39] = old[25..38]; every other column (inventory chest count, teammates, chest, crosshair,
    goal "chest") is zero, so those inputs contribute nothing at step 0;
  - block_embed rows 0..17 copied (rows 18, 19 fresh: chest and the pseudo agent block);
  - policy_head rows 0..22 copied; rows 23..27 keep their fresh weights and get the mean of the old
    biases as their bias;
  - img_fc.0 input weights are expanded from 4x4 to 8x8 spatial cells (channel-major flatten, C,H,W):
    each old cell's weights go to its 2x2 children divided by 4;
  - msg_head is fresh; value heads and everything else follow the equal-shape rule.
The v5 observation normalizer does not exist (v5 has only the reward normalizer), so nothing else is
remapped. The reward normalizer is carried over by `load_warm_start` (see its docstring).
"""
import torch

V5_OBS = {"obs_version": 2, "res": 32, "vec_dim": 39, "num_blocks": 18, "num_actions": 23}
OLD_INV = (0, 14)        # old inventory columns (14 items)
OLD_SCALARS = (14, 25)   # old scalar columns (11)
OLD_GOAL = (25, 39)      # old goal one-hot (14 items)
NEW_INV = (0, 14)
NEW_SCALARS = (15, 26)
NEW_GOAL = (26, 41)      # new goal one-hot is 15 items (26..40); the last one (chest) stays zero


def check_v5_obs(obs):
    if not isinstance(obs, dict):
        raise ValueError("warm start source has no observation signature (not a v2+ checkpoint)")
    bad = [f"{k}: source={obs.get(k)!r} expected={v!r}" for k, v in V5_OBS.items() if obs.get(k) != v]
    if bad:
        raise ValueError("warm start supports only the v5 layout (res 32, 39-dim vec, 18 blocks, "
                         "23 actions); " + "; ".join(bad))


def map_v5_state(dst_sd, src_sd):
    """dst_sd: fresh v6 state dict. src_sd: v5 model state dict. Returns (new_sd, report)."""
    new = {k: v.detach().clone() for k, v in dst_sd.items()}
    remapped = {}

    # vec input layer: 6.5 column remap, every other column zero
    old = src_sd["vec_mlp.0.weight"]
    W = torch.zeros_like(new["vec_mlp.0.weight"])
    W[:, NEW_INV[0]:NEW_INV[1]] = old[:, OLD_INV[0]:OLD_INV[1]]
    W[:, NEW_SCALARS[0]:NEW_SCALARS[1]] = old[:, OLD_SCALARS[0]:OLD_SCALARS[1]]
    W[:, NEW_GOAL[0]:NEW_GOAL[0] + (OLD_GOAL[1] - OLD_GOAL[0])] = old[:, OLD_GOAL[0]:OLD_GOAL[1]]
    new["vec_mlp.0.weight"] = W
    new["vec_mlp.0.bias"] = src_sd["vec_mlp.0.bias"].detach().clone()
    remapped["vec_mlp.0"] = "columns remapped per 6.5, rest zero"

    # block embedding: rows 0..17 of 18 copied; rows 18, 19 stay fresh
    n_blk = src_sd["block_embed.weight"].shape[0]
    new["block_embed.weight"][:n_blk] = src_sd["block_embed.weight"]
    remapped["block_embed"] = f"rows 0..{n_blk - 1} copied, rest fresh"

    # policy head: rows 0..22 copied; rows 23..27 fresh weights, bias = mean of the old biases
    n_act = src_sd["policy_head.weight"].shape[0]
    new["policy_head.weight"][:n_act] = src_sd["policy_head.weight"]
    new["policy_head.bias"][:n_act] = src_sd["policy_head.bias"]
    new["policy_head.bias"][n_act:] = src_sd["policy_head.bias"].mean()
    remapped["policy_head"] = f"rows 0..{n_act - 1} copied, rest fresh with mean bias"

    # img_fc input: channel-major flatten of a [C, s, s] map; each old cell goes to its f x f children / f^2
    old_fc = src_sd["img_fc.0.weight"]
    out_dim = old_fc.shape[0]
    s_old = int(round((old_fc.shape[1] // 32) ** 0.5))   # C = 32 for the v5 conv3 output
    C = old_fc.shape[1] // (s_old * s_old)
    s_new = int(round((new["img_fc.0.weight"].shape[1] // C) ** 0.5))
    f = s_new // s_old
    cells = old_fc.reshape(out_dim, C, s_old, s_old)
    expanded = cells.repeat_interleave(f, dim=2).repeat_interleave(f, dim=3) / float(f * f)
    new["img_fc.0.weight"] = expanded.reshape(out_dim, C * s_new * s_new).clone()
    new["img_fc.0.bias"] = src_sd["img_fc.0.bias"].detach().clone()
    remapped["img_fc.0"] = f"input weights expanded {s_old}x{s_old} -> {s_new}x{s_new} (children / {f * f})"

    done = set(remapped_keys(remapped))
    copied, fresh = [], []
    for k, v in dst_sd.items():
        if k in done:
            continue
        if k in src_sd and tuple(src_sd[k].shape) == tuple(v.shape):
            new[k] = src_sd[k].detach().clone()
            copied.append(k)
        else:
            fresh.append(k)
    return new, {"copied": sorted(copied), "remapped": remapped, "fresh": sorted(fresh)}


def remapped_keys(remapped):
    keys = {"vec_mlp.0.weight", "vec_mlp.0.bias", "block_embed.weight", "policy_head.weight",
            "policy_head.bias", "img_fc.0.weight", "img_fc.0.bias"}
    return keys


def load_warm_start(path, model):
    """Load a v5 checkpoint into the fresh v6 `model`, or a v6 checkpoint into a fresh v7 model (in
    place; dispatched on the source observation version). Returns (src_payload, report).

    Carried over from the source checkpoint: model weights (mapped above) and the reward normalizer
    (its return scale matches the copied value heads). Not carried over: counters, optimizer, the
    curriculum, the RND normalizer and RNG state; the caller starts those fresh."""
    src = torch.load(path, map_location="cpu", weights_only=False)
    obs = src.get("obs")
    if isinstance(obs, dict) and obs.get("obs") == "v7":
        return load_warm_start_v7(src, model)
    if isinstance(obs, dict) and obs.get("obs_version") == 3:
        return load_warm_start_v6(path, model)
    check_v5_obs(obs)
    new_sd, report = map_v5_state(model.state_dict(), src["model"])
    model.load_state_dict(new_sd)
    return src, report


# ----- v6 -> v7 (src/README_v7.md, trainer README "Warm start from v6") -----
# Primary actions with an obvious v7 equivalent (same name in both lists, or the v6 turn / look pair);
# every other primary row (place_*, craft_*, smelt_iron, deposit, withdraw, give) stays fresh.
V6_PRIMARY_KEEP = ("noop", "forward", "back", "left", "right", "jump", "forward_jump",
                   "turn_left", "turn_right", "look_up", "look_down", "attack", "deposit", "withdraw", "give")
V6_OBS_KEYS = {"obs_version": 3, "vec_dim": 102, "num_blocks": 20, "num_actions": 28, "num_items": 15,
               "num_msgs": 8}


def check_v6_obs(obs):
    if not isinstance(obs, dict):
        raise ValueError("warm start source has no observation signature")
    bad = [f"{k}: source={obs.get(k)!r} expected={v!r}" for k, v in V6_OBS_KEYS.items() if obs.get(k) != v]
    if bad:
        raise ValueError("warm start supports only the v6 layout (102-dim vec, 20 blocks, 28 actions, 15 items); "
                         + "; ".join(bad))


def map_v6_state(dst_sd, src_sd, v6_names, v7_names, v6_actions, v7_actions, v6_blocks, v7_blocks):
    """dst_sd: fresh v7 state dict; src_sd: v6 model state dict. Returns (new_sd, report).

    - vec_mlp.0 input columns: by name (v6 VEC_NAMES -> v7 VEC_NAMES); v7 columns without a v6 name are zero.
    - block_embed rows: by block name (v6 ids 0..19 are the v7 ids 0..19).
    - policy_head rows: by primary-action name for V6_PRIMARY_KEEP; the other rows keep their init.
    - every other tensor with an equal shape is copied (CNN, img_fc, fuse, GRU, value heads, msg_head,
      res_block, ...); aux_head and spr_trans change shape and stay fresh; recipe_head / item_head are new.
    """
    new = {k: v.detach().clone() for k, v in dst_sd.items()}
    remapped = {}
    v6_index = {n: i for i, n in enumerate(v6_names)}
    cols_src, cols_dst = [], []
    for j, n in enumerate(v7_names):
        if n in v6_index:
            cols_dst.append(j)
            cols_src.append(v6_index[n])
    W = torch.zeros_like(new["vec_mlp.0.weight"])
    W[:, cols_dst] = src_sd["vec_mlp.0.weight"][:, cols_src]
    new["vec_mlp.0.weight"] = W
    new["vec_mlp.0.bias"] = src_sd["vec_mlp.0.bias"].detach().clone()
    remapped["vec_mlp.0"] = (f"{len(cols_dst)} of {len(v7_names)} v7 columns from v6 names "
                             f"({len(v6_names) - len(cols_src)} v6 columns dropped), rest zero")

    v6_bi = {n: i for i, n in enumerate(v6_blocks)}
    rows_dst, rows_src = [], []
    for j, n in enumerate(v7_blocks):
        if n in v6_bi and v6_bi[n] < src_sd["block_embed.weight"].shape[0]:
            rows_dst.append(j)
            rows_src.append(v6_bi[n])
    new["block_embed.weight"][rows_dst] = src_sd["block_embed.weight"][rows_src]
    remapped["block_embed"] = f"{len(rows_dst)} of {len(v7_blocks)} block rows copied by name"

    v7_ai = {n: i for i, n in enumerate(v7_actions)}
    v6_ai = {n: i for i, n in enumerate(v6_actions)}
    pr_dst, pr_src = [], []
    for n in V6_PRIMARY_KEEP:
        if n in v6_ai and n in v7_ai:
            pr_dst.append(v7_ai[n])
            pr_src.append(v6_ai[n])
    new["policy_head.weight"][pr_dst] = src_sd["policy_head.weight"][pr_src]
    new["policy_head.bias"][pr_dst] = src_sd["policy_head.bias"][pr_src]
    remapped["policy_head"] = f"{len(pr_dst)} of {len(v7_actions)} primary rows copied by name, rest fresh"

    done = set(remapped_keys(remapped))
    done |= {"vec_mlp.0.weight", "vec_mlp.0.bias", "block_embed.weight", "policy_head.weight",
             "policy_head.bias"}
    copied, fresh = [], []
    for k, v in dst_sd.items():
        if k in done:
            continue
        if k in src_sd and tuple(src_sd[k].shape) == tuple(v.shape):
            new[k] = src_sd[k].detach().clone()
            copied.append(k)
        else:
            fresh.append(k)
    report = {"copied": sorted(copied), "remapped": remapped, "fresh": sorted(fresh),
              "vec_columns": {"mapped": len(cols_dst), "v7": len(v7_names), "v6": len(v6_names)},
              "block_rows": len(rows_dst), "primary_rows": len(pr_dst), "primary_total": len(v7_actions)}
    return new, report


def load_warm_start_v6(path, model):
    """Load a v6 checkpoint into a fresh v7 model (in place). Returns (src_payload, report)."""
    import craftsim
    src = torch.load(path, map_location="cpu", weights_only=False)
    check_v6_obs(src.get("obs"))
    if not getattr(model, "v7", False):
        raise ValueError("a v6 checkpoint warm starts a v7 model: run with --obs v7 (the default)")
    new_sd, report = map_v6_state(
        model.state_dict(), src["model"],
        v6_names=craftsim.VEC_NAMES_V6, v7_names=craftsim.VEC_NAMES,
        v6_actions=list(craftsim.ACTIONS), v7_actions=list(craftsim.PRIMARY_NAMES),
        v6_blocks=list(craftsim.BLOCKS[:craftsim.NUM_BLOCKS_V6]), v7_blocks=list(craftsim.BLOCKS))
    model.load_state_dict(new_sd)
    return src, report


def load_warm_start_v7(src, model):
    """v7 -> v7 (e.g. a behavior-cloned checkpoint from trainer/bc_v7.py): copy every tensor whose name and
    shape match; anything else stays fresh. Returns (src_payload, report)."""
    if not getattr(model, "v7", False):
        raise ValueError("a v7 checkpoint warm starts a v7 model: run with --obs v7 (the default)")
    sd, ssd = model.state_dict(), src["model"]
    copied, fresh = [], []
    for k, v in sd.items():
        if k in ssd and tuple(ssd[k].shape) == tuple(v.shape):
            sd[k] = ssd[k].to(v.dtype).clone()
            copied.append(k)
        else:
            fresh.append(k)
    model.load_state_dict(sd)
    return src, {"copied": copied, "remapped": [], "fresh": fresh}
