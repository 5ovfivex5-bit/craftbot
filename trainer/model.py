"""MineMindPolicy: one shared recurrent actor-critic batched over all envs.

Graph (named submodules are the layers shown in the dashboard):
  img_in ─> block_embed ─> conv1 ─> conv2 ─> conv3 ─> img_fc ─┐
                                                             ├─> fuse (hidden) ─> gru ─┬─> policy_head (actions)
  vec_in ─> vec_mlp ─────────────────────────────────────────┘                        ├─> msg_head (chat token, v6)
                                                                                      └─> value_head

v6 (teams): every agent emits (action, msg). The action and msg heads are sampled independently; the
joint log-prob is the sum of both, and the policy loss / entropy bonus sum over both heads.

Sizes (`size=`):
  small  the v2-v5 network (about 0.9M parameters). Unchanged layout and parameter order.
  100m   IMPALA-style residual CNN encoder (conv + max-pool + 2 residual convs per stage, 128/256/512
         channels), wide img/vec/fused features (2560 / 768), SimBa residual block on the fused
         feature and a 2048-unit GRU. About 101M parameters including the auxiliary heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs import NUM_ACTIONS, NUM_BLOCKS, NUM_ITEMS, NUM_MSGS, RES, VEC_DIM

VALUE_BINS = 61        # HL-Gauss support bins (value space is the scaled return)
NUM_ACH_CLASSES = NUM_ITEMS + 1   # next-achievement prediction: 14 items + "none"
RND_DIM = 64           # RND target/predictor output size
SPR_DIM = 128          # SPR projection / latent size

SIZES = {
    "small": dict(embed_dim=8, feat=256, vec_hidden=128, gru_hidden=256, cnn=None, res_expand=2),
    "100m": dict(embed_dim=16, feat=2560, vec_hidden=768, gru_hidden=2048,
                 cnn=(128, 256, 512), res_expand=2),
}


class ResBlock(nn.Module):
    """SimBa-style residual MLP block (Lee et al. 2025): x + MLP(LayerNorm(x)). The last linear layer
    is zero-initialized, so the block is the identity at init and a v2/v3 policy keeps its function."""

    def __init__(self, dim, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, expand * dim), nn.ReLU(), nn.Linear(expand * dim, dim))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        return self.mlp(self.norm(x))


class ImpalaStage(nn.Module):
    """IMPALA residual stage: 3x3 conv, stride-2 max-pool, then `n_res` residual blocks (relu-conv-relu-conv)."""

    def __init__(self, cin, cout, n_res=2):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1),
                          nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1))
            for _ in range(n_res)
        ])

    def forward(self, x):
        x = self.pool(self.conv(x))
        for blk in self.blocks:
            x = x + blk(x)
        return x


class _EmbedGrad(torch.autograd.Function):
    """Embedding lookup whose weight gradient is an index_add_ (about 2.5x faster on CPU than
    aten::embedding_dense_backward, which dominated the PPO update for [B*H*W] lookups)."""

    @staticmethod
    def forward(ctx, ids, weight):
        ctx.save_for_backward(ids)
        ctx.num_rows = weight.shape[0]
        return F.embedding(ids, weight)

    @staticmethod
    def backward(ctx, grad):
        (ids,) = ctx.saved_tensors
        gw = grad.new_zeros(ctx.num_rows, grad.shape[-1])
        gw.index_add_(0, ids.reshape(-1), grad.reshape(-1, grad.shape[-1]))
        return None, gw


class FastEmbedding(nn.Embedding):
    """Drop-in nn.Embedding (same state_dict, module hooks still fire) with the faster backward."""

    def forward(self, ids):
        return _EmbedGrad.apply(ids, self.weight)


class MineMindPolicy(nn.Module):
    def __init__(self, res=RES, vec_dim=VEC_DIM, num_actions=NUM_ACTIONS, num_blocks=NUM_BLOCKS,
                 size="small", resblock=True, num_msgs=NUM_MSGS, num_recipe_classes=0,
                 num_item_classes=0):
        super().__init__()
        if size not in SIZES:
            raise ValueError(f"model size must be one of {tuple(SIZES)}")
        p = SIZES[size]
        self.size = size
        embed_dim, feat, vec_hidden, gru_hidden = p["embed_dim"], p["feat"], p["vec_hidden"], p["gru_hidden"]
        self.resblock = bool(resblock)
        self.res = res
        self.vec_dim = vec_dim
        self.num_actions = num_actions
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.feat = feat
        self.vec_hidden = vec_hidden
        self.gru_hidden = gru_hidden

        # Observation encoder: block-id embedding + depth channel -> CNN.
        self.block_embed = FastEmbedding(num_blocks, embed_dim)
        s = res
        for _ in range(3):
            s = (s + 1) // 2
        if p["cnn"] is None:
            self.conv1 = nn.Sequential(nn.Conv2d(embed_dim + 1, 16, 3, stride=2, padding=1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
            self.conv3 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU())
            cnn_out = 32
        else:
            c1, c2, c3 = p["cnn"]
            self.conv1 = ImpalaStage(embed_dim + 1, c1)
            self.conv2 = ImpalaStage(c1, c2)
            self.conv3 = ImpalaStage(c2, c3)
            cnn_out = c3
        self.conv_out_hw = s
        self.img_fc = nn.Sequential(nn.Linear(cnn_out * s * s, feat), nn.ReLU())

        # Vector observation encoder.
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_dim, vec_hidden), nn.ReLU(),
            nn.Linear(vec_hidden, vec_hidden), nn.ReLU(),
        )

        # Fused hidden layer, recurrent core, heads.
        self.fuse = nn.Sequential(nn.Linear(feat + vec_hidden, feat), nn.Tanh())
        self.gru = nn.GRU(feat, gru_hidden)
        self.policy_head = nn.Linear(gru_hidden, num_actions)
        self.value_head = nn.Linear(gru_hidden, 1)

        # v4 heads, registered last so the v2/v3 parameter order (and optimizer state) is unchanged.
        # value_head / int_value_head: scalar MSE critics; value_cls / int_value_cls: HL-Gauss
        # classifiers. Only the ones selected by `--value-loss` are trained; the others stay unused.
        self.value_cls = nn.Linear(gru_hidden, VALUE_BINS)
        self.int_value_head = nn.Linear(gru_hidden, 1)
        self.int_value_cls = nn.Linear(gru_hidden, VALUE_BINS)
        self.aux_head = nn.Linear(gru_hidden, NUM_ACH_CLASSES)
        # RND (Burda et al. 2018): fixed random target, trained predictor, on the fused feature.
        self.rnd_target = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, RND_DIM))
        self.rnd_pred = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, 256),
                                      nn.ReLU(), nn.Linear(256, RND_DIM))
        for p_ in self.rnd_target.parameters():
            p_.requires_grad_(False)
        # v5, registered last (keeps the old parameter prefix for optimizer state):
        # SimBa residual block on the fused hidden layer (identity at init, see ResBlock)
        self.res_block = ResBlock(feat, expand=p["res_expand"])
        # SPR (Schwarzer et al. 2021): projector, predictor and action-conditioned latent transition
        self.spr_proj = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Linear(256, SPR_DIM))
        self.spr_pred = nn.Sequential(nn.Linear(SPR_DIM, SPR_DIM), nn.ReLU(), nn.Linear(SPR_DIM, SPR_DIM))
        self.spr_trans = nn.Sequential(nn.Linear(SPR_DIM + NUM_ACTIONS, 256), nn.ReLU(),
                                       nn.Linear(256, SPR_DIM))
        # v6, registered last: the chat token head (one token per agent per step, 0 = silent)
        self.num_msgs = num_msgs
        self.msg_head = nn.Linear(gru_hidden, num_msgs)
        # v7, registered last: recipe logits (0 none, r + 1) and item_select logits (0 keep, i + 1).
        # num_recipe_classes == 0 is the v6 network (no such heads).
        self.v7 = int(num_recipe_classes) > 0
        if self.v7:
            self.recipe_head = nn.Linear(gru_hidden, int(num_recipe_classes))
            self.item_head = nn.Linear(gru_hidden, int(num_item_classes))

    # ----- forward pieces (used separately by PPO and instrumentation) -----
    def encode(self, img, vec):
        """img: uint8 [B,H,W,2] tensor, vec: float [B,VEC_DIM] -> fused feature [B,feat]."""
        ids = img[..., 0].long()
        depth = img[..., 1].float().div(255.0).unsqueeze(-1)
        emb = self.block_embed(ids)
        # channels_last is ~2x faster for these convs on CPU
        x = torch.cat([emb, depth], dim=-1).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        x = self.conv3(self.conv2(self.conv1(x)))
        if self.size != "small":
            x = F.relu(x)
        x = self.img_fc(x.flatten(1))
        v = self.vec_mlp(vec)
        h = self.fuse(torch.cat([x, v], dim=-1))
        if self.resblock:
            h = h + self.res_block(h)
        return h

    def recur(self, feat, h):
        """feat [B,feat], h [B,gru] -> (out [B,gru], h_new [B,gru]) for one time step."""
        out, hn = self.gru(feat.unsqueeze(0), h.unsqueeze(0))
        return out[0], hn[0]

    def heads(self, out):
        return self.policy_head(out), self.value_head(out).squeeze(-1)

    def forward(self, img, vec, h):
        """-> (action logits, msg logits, value, h_new)."""
        out, hn = self.recur(self.encode(img, vec), h)
        logits, value = self.heads(out)
        return logits, self.msg_head(out), value, hn

    def rnd_error(self, feat):
        """Per-sample RND novelty: squared error between predictor and target. feat [B,feat] -> [B]."""
        return (self.rnd_pred(feat) - self.rnd_target(feat)).pow(2).mean(-1)

    def zero_hidden(self, batch, device=None):
        return torch.zeros(batch, self.gru_hidden, device=device)

    # ----- architecture description for the dashboard `model` message -----
    def describe(self):
        s0 = self.res
        s1 = (s0 + 1) // 2
        s2 = (s1 + 1) // 2
        s3 = (s2 + 1) // 2
        E = self.embed_dim
        feat = self.feat
        vh = self.vec_hidden
        g = self.gru_hidden

        def nparams(m):
            return int(sum(p.numel() for p in m.parameters()))

        def ntrain(m):
            return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

        if self.size == "small":
            conv_spec = [
                ("conv1", "Conv2d+ReLU", self.conv1, [E + 1, s0, s0], [16, s1, s1], "encoder"),
                ("conv2", "Conv2d+ReLU", self.conv2, [16, s1, s1], [32, s2, s2], "encoder"),
                ("conv3", "Conv2d+ReLU", self.conv3, [32, s2, s2], [32, s3, s3], "encoder"),
            ]
        else:
            c1, c2, c3 = SIZES[self.size]["cnn"]
            conv_spec = [
                ("conv1", "IMPALA stage (conv+pool+2 res)", self.conv1, [E + 1, s0, s0], [c1, s1, s1], "encoder"),
                ("conv2", "IMPALA stage (conv+pool+2 res)", self.conv2, [c1, s1, s1], [c2, s2, s2], "encoder"),
                ("conv3", "IMPALA stage (conv+pool+2 res)", self.conv3, [c2, s2, s2], [c3, s3, s3], "encoder"),
            ]
        cnn_out = conv_spec[-1][4][0]
        spec = [
            ("img_in", "Input", None, [s0, s0, 2], [s0, s0, 2], "input"),
            ("vec_in", "Input", None, [self.vec_dim], [self.vec_dim], "input"),
            ("block_embed", "Embedding", self.block_embed, [s0, s0], [s0, s0, E], "encoder"),
        ] + conv_spec + [
            ("img_fc", "Linear+ReLU", self.img_fc, [cnn_out * s3 * s3], [feat], "encoder"),
            ("vec_mlp", "MLP(2x Linear+ReLU)", self.vec_mlp, [self.vec_dim], [vh], "encoder"),
            ("fuse", "Linear+Tanh", self.fuse, [feat + vh], [feat], "hidden"),
            ("res_block", "SimBa residual MLP", self.res_block, [feat], [feat], "hidden"),
            ("gru", "GRU", self.gru, [feat], [g], "recurrent"),
            ("policy_head", "Linear", self.policy_head, [g], [self.num_actions], "heads"),
            ("msg_head", "Linear (chat token)", self.msg_head, [g], [self.num_msgs], "heads"),
        ] + ([
            ("recipe_head", "Linear (recipe, masked)", self.recipe_head, [g], [self.recipe_head.out_features], "heads"),
            ("item_head", "Linear (item select, masked)", self.item_head, [g], [self.item_head.out_features], "heads"),
        ] if self.v7 else []) + [
            ("value_head", "Linear", self.value_head, [g], [1], "heads"),
            ("value_cls", "Linear (HL-Gauss)", self.value_cls, [g], [VALUE_BINS], "heads"),
            ("int_value_head", "Linear", self.int_value_head, [g], [1], "heads"),
            ("int_value_cls", "Linear (HL-Gauss)", self.int_value_cls, [g], [VALUE_BINS], "heads"),
            ("aux_head", "Linear", self.aux_head, [g], [NUM_ACH_CLASSES], "heads"),
            ("rnd_target", "MLP (frozen)", self.rnd_target, [feat], [RND_DIM], "rnd"),
            ("rnd_pred", "MLP", self.rnd_pred, [feat], [RND_DIM], "rnd"),
            ("spr_proj", "MLP", self.spr_proj, [feat], [SPR_DIM], "spr"),
            ("spr_pred", "MLP", self.spr_pred, [SPR_DIM], [SPR_DIM], "spr"),
            ("spr_trans", "MLP", self.spr_trans, [SPR_DIM + self.num_actions], [SPR_DIM], "spr"),
        ]
        layers = []
        for name, typ, mod, ins, outs, group in spec:
            layers.append({
                "name": name,
                "type": typ,
                "group": group,
                "params": nparams(mod) if mod is not None else 0,
                "in_shape": ins,
                "out_shape": outs,
                "trainable": ntrain(mod) > 0 if mod is not None else False,
            })
        edges = [
            ["img_in", "block_embed"], ["block_embed", "conv1"], ["conv1", "conv2"],
            ["conv2", "conv3"], ["conv3", "img_fc"], ["img_fc", "fuse"],
            ["vec_in", "vec_mlp"], ["vec_mlp", "fuse"], ["fuse", "res_block"], ["res_block", "gru"],
            ["gru", "policy_head"], ["gru", "msg_head"], ["gru", "value_head"],
        ] + ([["gru", "recipe_head"], ["gru", "item_head"]] if self.v7 else []) + [
            ["gru", "value_cls"], ["gru", "int_value_head"], ["gru", "int_value_cls"],
            ["gru", "aux_head"], ["res_block", "rnd_target"], ["res_block", "rnd_pred"],
            ["res_block", "spr_proj"],
        ]
        return {
            "name": ("MineMindPolicy-v7" if self.v7 else "MineMindPolicy-v6") if self.size == "small"
            else f"MineMindPolicy-{self.size}-" + ("v7" if self.v7 else "v6"),
            "size": self.size,
            "params_total": int(sum(p.numel() for p in self.parameters())),
            "input": {"img": [self.res, self.res, 2], "vec": [self.vec_dim]},
            "gru_hidden": self.gru_hidden,
            "layers": layers,
            "edges": edges,
        }

    # Top-level module names, used for grad / weight statistics.
    TOP_LEVEL = ["block_embed", "conv1", "conv2", "conv3", "img_fc", "vec_mlp", "fuse", "gru",
                 "policy_head", "msg_head", "value_head", "recipe_head", "item_head"]
