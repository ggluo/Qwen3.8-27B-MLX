"""
A small, faithful DeepSeek-V3 in MLX. Dependencies: mlx only (rest is stdlib).

Implements the two things that make V3 "V3":
  * MLA   -- Multi-head Latent Attention with a low-rank KV cache + decoupled RoPE
  * MoE   -- fine-grained experts, a shared expert, sigmoid gating with an
             aux-loss-free correction bias, and group-limited (node-limited) routing

Usage:
    python deepseek_v3.py shapes            # arch walkthrough, tensor shapes, param counts
    python deepseek_v3.py test              # correctness checks (cache vs. no-cache, etc.)
    python deepseek_v3.py train             # train on the embedded corpus, then sample
    python deepseek_v3.py train --data f.txt --steps 800
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    vocab_size: int = 256  # byte-level tokenizer
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8

    # --- MLA ---
    q_lora_rank: int = 192  # rank of the query down-projection (params only)
    kv_lora_rank: int = 96  # rank of the KV latent -- THIS is what gets cached
    qk_nope_head_dim: int = 32  # per-head content dims  (compressible)
    qk_rope_head_dim: int = 16  # per-head position dims (NOT compressible)
    v_head_dim: int = 32

    # --- MoE ---
    intermediate_size: int = 768  # hidden dim of the DENSE mlp in early layers
    moe_intermediate_size: int = 192  # hidden dim of each fine-grained expert
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4  # top-k
    n_group: int = 4  # experts are partitioned into this many groups
    topk_group: int = 2  # a token may only route within its best `topk_group`
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 1  # layers [0, k) use a dense MLP, not MoE
    bias_update_speed: float = 1e-3  # aux-loss-free balancing step size (gamma)

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False  # V3 does not tie; tiny models benefit

    def __post_init__(self):
        assert self.n_routed_experts % self.n_group == 0
        assert self.topk_group <= self.n_group
        # top-k must fit inside the experts reachable via the selected groups
        assert self.num_experts_per_tok <= self.topk_group * (
            self.n_routed_experts // self.n_group
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


# The real thing, for scale. Don't try to instantiate this.
V3_671B = dict(
    vocab_size=129280, hidden_size=7168, num_layers=61, num_heads=128,
    q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
    qk_rope_head_dim=64, v_head_dim=128, intermediate_size=18432,
    moe_intermediate_size=2048, n_routed_experts=256, n_shared_experts=1,
    num_experts_per_tok=8, n_group=8, topk_group=4, first_k_dense_replace=3,
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """x / rms(x) * w. No mean subtraction, no bias -- cheaper than LayerNorm."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # normalize in fp32 so bf16 training doesn't lose the variance
        out = mx.fast.rms_norm(x.astype(mx.float32), self.weight.astype(mx.float32), self.eps)
        return out.astype(x.dtype)


def apply_rope(x: mx.array, offset: int, theta: float) -> mx.array:
    """Rotary embedding on the last axis of (..., L, D). `offset` = tokens already cached.

    Uses the "rotate_half" layout (pair dim i with i+D/2), i.e. the HF convention.
    Real V3 checkpoints store q_pe/k_pe interleaved, so loading them requires a
    permutation of those slices -- irrelevant here since we train from scratch,
    as long as we are self-consistent.
    """
    D = x.shape[-1]
    half = D // 2
    pos = mx.arange(offset, offset + x.shape[-2], dtype=mx.float32)[:, None]
    inv_freq = theta ** (-mx.arange(0, half, dtype=mx.float32) / half)[None, :]
    ang = pos * inv_freq  # (L, D/2)
    cos, sin = mx.cos(ang), mx.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def causal_mask(L: int, offset: int, dtype) -> Optional[mx.array]:
    """(L, L+offset) additive mask. Query i sits at absolute position offset+i.

    Correctly handles offset > 0 (chunked prefill), which the naive (L, L) mask
    silently broadcasts wrong or crashes on.
    """
    if L == 1:
        return None  # single query attends to everything in the cache
    q = mx.arange(offset, offset + L)[:, None]
    k = mx.arange(offset + L)[None, :]
    # use dtype's most-negative rather than -1e9: -1e9 overflows to -inf in fp16
    return mx.where(k <= q, 0.0, mx.finfo(dtype).min).astype(dtype)


# ---------------------------------------------------------------------------
# Multi-head Latent Attention
# ---------------------------------------------------------------------------


class MLA(nn.Module):
    """
    Cache footprint per token per layer is (kv_lora_rank + qk_rope_head_dim),
    NOT (num_heads * (qk_head_dim + v_head_dim)). For V3-671B that is 576 vs
    40960 numbers -- a 71x reduction.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_heads
        self.scale = cfg.qk_head_dim ** -0.5  # V3 rescales this via YaRN mscale; we don't

        # ---- queries: hidden -> c_Q (rank r_q) -> per-head [nope | pe] ----
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.n_heads * cfg.qk_head_dim, bias=False)

        # ---- keys/values: hidden -> [c_KV (rank r_kv) | k_pe] in ONE matmul ----
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)  # note: c_KV only
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank,
            self.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.n_heads * cfg.v_head_dim, cfg.hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        cfg = self.cfg
        B, L, _ = x.shape
        offset = 0 if cache is None else cache[0].shape[1]

        # ---------------- queries ----------------
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.n_heads, cfg.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope = q[..., : cfg.qk_nope_head_dim]
        q_pe = apply_rope(q[..., cfg.qk_nope_head_dim :], offset, cfg.rope_theta)

        # ---------------- latent K/V ----------------
        kv = self.kv_a_proj_with_mqa(x)
        c_kv = self.kv_a_layernorm(kv[..., : cfg.kv_lora_rank])  # (B, L, r_kv)
        k_pe = kv[..., cfg.kv_lora_rank :][:, None]  # (B, 1, L, d_pe) -- MQA-style, 1 head
        k_pe = apply_rope(k_pe, offset, cfg.rope_theta)

        # The cache holds ONLY these two. Everything else is recomputed.
        if cache is not None:
            c_kv = mx.concatenate([cache[0], c_kv], axis=1)
            k_pe = mx.concatenate([cache[1], k_pe], axis=2)
        new_cache = (c_kv, k_pe)

        # ---------------- decompress ----------------
        # The memory-optimal path folds kv_b_proj into q_b_proj and attends against
        # c_kv directly ("absorption"), never materializing per-head K/V. We do the
        # explicit version: same numerics, clearer, fine at this scale.
        T = c_kv.shape[1]
        kv = self.kv_b_proj(c_kv).reshape(
            B, T, self.n_heads, cfg.qk_nope_head_dim + cfg.v_head_dim
        ).transpose(0, 2, 1, 3)
        k_nope, v = mx.split(kv, [cfg.qk_nope_head_dim], axis=-1)

        # the single positional key head is shared by all query heads (free broadcast)
        k_pe_b = mx.broadcast_to(k_pe, (B, self.n_heads, T, cfg.qk_rope_head_dim))
        k = mx.concatenate([k_nope, k_pe_b], axis=-1)
        q = mx.concatenate([q_nope, q_pe], axis=-1)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * cfg.v_head_dim)
        return self.o_proj(o), new_cache


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)). Used for dense layers + shared expert."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SwitchGLU(nn.Module):
    """All N experts as stacked weights; gather_mm touches only the selected k.

    This is what makes MoE actually cheap: FLOPs scale with k, not N. The
    "loop over every expert and mask" pattern is N/k times more expensive.
    """

    def __init__(self, dim: int, hidden: int, n_experts: int):
        super().__init__()
        s_in, s_h = dim ** -0.5, hidden ** -0.5
        shape_in, shape_out = (n_experts, hidden, dim), (n_experts, dim, hidden)
        self.gate_proj = mx.random.uniform(-s_in, s_in, shape_in)
        self.up_proj = mx.random.uniform(-s_in, s_in, shape_in)
        self.down_proj = mx.random.uniform(-s_h, s_h, shape_out)

    def __call__(self, x: mx.array, idx: mx.array) -> mx.array:
        """x: (T, D), idx: (T, k)  ->  (T, k, D)"""
        x = mx.expand_dims(x, (-2, -3))  # (T, 1, 1, D)
        g = mx.gather_mm(x, self.gate_proj.swapaxes(-1, -2), rhs_indices=idx)
        u = mx.gather_mm(x, self.up_proj.swapaxes(-1, -2), rhs_indices=idx)
        h = nn.silu(g) * u  # (T, k, 1, H)
        out = mx.gather_mm(h, self.down_proj.swapaxes(-1, -2), rhs_indices=idx)
        return out.squeeze(-2)  # (T, k, D)


class MoEGate(nn.Module):
    """V3 gating: sigmoid scores, group-limited top-k, aux-loss-free bias."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.weight = mx.random.uniform(
            -cfg.hidden_size ** -0.5, cfg.hidden_size ** -0.5,
            (cfg.n_routed_experts, cfg.hidden_size),
        )
        # Added to scores for SELECTION ONLY, never to the output weight -- so
        # load balancing produces zero interfering gradient. Nudged by +/-gamma
        # each step toward whichever experts are under/over-loaded.
        self.e_score_correction_bias = mx.zeros((cfg.n_routed_experts,))

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        cfg = self.cfg
        # gate always in fp32: routing decisions are discrete and bf16 ties are ugly
        scores = mx.sigmoid(x.astype(mx.float32) @ self.weight.astype(mx.float32).T)
        biased = scores + self.e_score_correction_bias.astype(mx.float32)

        # ---- group-limited routing: keep only the best `topk_group` groups ----
        T = scores.shape[0]
        per_group = cfg.n_routed_experts // cfg.n_group
        g = biased.reshape(T, cfg.n_group, per_group)
        # V3 scores a group by the sum of its top-2 experts
        g_score = mx.topk(g, min(2, per_group), axis=-1).sum(-1)  # (T, n_group)
        keep = mx.argsort(-g_score, axis=-1)[:, : cfg.topk_group]  # (T, topk_group)
        gmask = mx.zeros((T, cfg.n_group)).at[mx.arange(T)[:, None], keep].add(1.0)
        masked = mx.where(
            mx.repeat(gmask, per_group, axis=-1) > 0, biased, mx.finfo(mx.float32).min
        )

        # ---- top-k within the surviving groups ----
        # stop_gradient: expert choice is discrete, so no gradient flows through the
        # index path (gather_mm has no VJP w.r.t. indices anyway). The gate still
        # learns -- via `w` below, which multiplies the expert output.
        idx = mx.stop_gradient(mx.argsort(-masked, axis=-1)[:, : cfg.num_experts_per_tok])
        w = mx.take_along_axis(scores, idx, axis=-1)  # un-biased scores!
        if cfg.norm_topk_prob:
            w = w / (w.sum(-1, keepdims=True) + 1e-20)
        return idx, (w * cfg.routed_scaling_factor).astype(x.dtype)


class MoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.gate = MoEGate(cfg)
        self.experts = SwitchGLU(cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts)
        # always-on expert: absorbs patterns every token needs, so the routed
        # experts don't each burn capacity relearning them
        self.shared_experts = MLP(
            cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts
        )
        self.expert_load = mx.zeros((cfg.n_routed_experts,))  # diagnostics / bias update

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        xf = x.reshape(-1, D)
        idx, w = self.gate(xf)
        y = (self.experts(xf, idx) * w[..., None]).sum(axis=1)  # (T, D)
        return (y + self.shared_experts(xf)).reshape(B, L, D), idx


# ---------------------------------------------------------------------------
# Blocks & model
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = MLA(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.is_moe = layer_idx >= cfg.first_k_dense_replace
        self.mlp = MoE(cfg) if self.is_moe else MLP(cfg.hidden_size, cfg.intermediate_size)

    def __call__(self, x, mask=None, cache=None):
        h, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + h
        f = self.mlp(self.post_attention_layernorm(x))
        routing = None
        if self.is_moe:
            f, routing = f
        return x + f, cache, routing


class DeepSeekV3(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Block(cfg, i) for i in range(cfg.num_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, ids: mx.array, cache: Optional[List] = None):
        x = self.embed_tokens(ids)
        offset = 0 if cache is None else cache[0][0].shape[1]
        mask = causal_mask(ids.shape[1], offset, x.dtype)

        new_cache, routings = [], []
        for i, layer in enumerate(self.layers):
            x, c, r = layer(x, mask, None if cache is None else cache[i])
            new_cache.append(c)
            if r is not None:
                routings.append(r)

        x = self.norm(x)
        logits = self.embed_tokens.as_linear(x) if self.cfg.tie_word_embeddings else self.lm_head(x)
        return logits, new_cache, routings

    # -- aux-loss-free load balancing: shift the selection bias, no gradients --
    def update_expert_bias(self, routings: List[mx.array]):
        cfg = self.cfg
        for layer in self.layers:
            if not layer.is_moe:
                continue
            r = routings.pop(0).flatten()
            counts = mx.zeros((cfg.n_routed_experts,)).at[r].add(1.0)
            gate = layer.mlp.gate
            # overloaded experts get their bias lowered, underloaded raised
            err = counts.mean() - counts
            gate.e_score_correction_bias = (
                gate.e_score_correction_bias + cfg.bias_update_speed * mx.sign(err)
            )
            layer.mlp.expert_load = layer.mlp.expert_load + counts


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(model, prompt: mx.array, max_tokens=200, temp=0.8, top_k=40):
    logits, cache, _ = model(prompt)  # prefill
    out = []
    for _ in range(max_tokens):
        z = logits[:, -1, :].astype(mx.float32)
        if temp == 0:
            nxt = mx.argmax(z, axis=-1)
        else:
            if top_k:
                kth = mx.topk(z, top_k, axis=-1)[:, :1]  # smallest kept logit
                z = mx.where(z < kth, mx.finfo(mx.float32).min, z)
            nxt = mx.random.categorical(z / temp, axis=-1)
        out.append(nxt)
        logits, cache, _ = model(nxt[:, None], cache)
    return mx.stack(out, axis=1)


# ---------------------------------------------------------------------------
# Tiny byte-level training loop
# ---------------------------------------------------------------------------

CORPUS = """The transformer reads a sequence and predicts what comes next.
Attention lets every token look at every earlier token. That is the whole trick.
Multi-head latent attention compresses the keys and the values into one small
latent vector before it stores them, so the cache stays small while the model
stays wide. The position information is carved out into its own narrow slice,
because a rotation that depends on two positions cannot be folded into a
matrix that depends on none. Content is compressed. Position is not.
A mixture of experts replaces one wide feed forward network with many narrow
ones. A router looks at each token and picks a few experts to run. One expert
is shared and always runs, so the routed experts never waste their capacity
learning the things that every token needs. The router adds a small bias when
it chooses, and that bias is nudged up for the experts that are idle and down
for the experts that are busy, so the load spreads out without any extra loss
term fighting the gradient of the language model itself.
Compression saves memory. Sparsity saves compute. Neither one changes the
shape of the thing being learned, which is only ever the next token.
"""


def batches(data: mx.array, batch_size: int, seq_len: int):
    while True:
        i = mx.random.randint(0, len(data) - seq_len - 1, (batch_size,))
        w = i[:, None] + mx.arange(seq_len + 1)[None, :]
        chunk = data[w]
        yield chunk[:, :-1], chunk[:, 1:]


def train(cfg: Config, text: str, steps: int, lr: float, batch_size: int, seq_len: int):
    data = mx.array(list(text.encode("utf-8")))
    print(f"corpus: {len(data)} bytes | {len(set(text))} distinct chars")

    model = DeepSeekV3(cfg)
    mx.eval(model.parameters())
    n = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"params: {n / 1e6:.2f}M")

    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps), weight_decay=0.01)

    def loss_fn(model, x, y):
        logits, _, routings = model(x)
        loss = nn.losses.cross_entropy(logits.astype(mx.float32).reshape(-1, cfg.vocab_size),
                                       y.reshape(-1), reduction="mean")
        return loss, routings

    grad_fn = nn.value_and_grad(model, loss_fn)
    t0 = time.time()
    for step, (x, y) in enumerate(batches(data, batch_size, seq_len), 1):
        (loss, routings), grads = grad_fn(model, x, y)
        opt.update(model, grads)
        model.update_expert_bias(routings)  # gradient-free, so after the optimizer
        mx.eval(model.parameters(), opt.state, loss)
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  "
                  f"({(time.time() - t0) / step * 1000:.0f} ms/step)")
        if step >= steps:
            break

    # how evenly did the router spread work? 1.0 == perfectly uniform
    load = sum(l.mlp.expert_load for l in model.layers if l.is_moe)
    print(f"\nexpert load balance: min/mean = {(load.min() / load.mean()).item():.2f}, "
          f"max/mean = {(load.max() / load.mean()).item():.2f}")
    return model


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def show_shapes(cfg: Config):
    model = DeepSeekV3(cfg)
    mx.eval(model.parameters())
    ids = mx.array([list(b"attention is all")])
    logits, cache, routings = model(ids)
    mx.eval(logits)

    B, L = ids.shape
    print(f"input ids            {ids.shape}")
    print(f"logits               {logits.shape}")
    print("\n--- MLA cache, per layer ---")
    print(f"c_KV (latent)        {cache[0][0].shape}   <- rank {cfg.kv_lora_rank}")
    print(f"k_pe (positional)    {cache[0][1].shape}   <- 1 head, shared by all {cfg.num_heads}")
    naive = cfg.num_heads * (cfg.qk_head_dim + cfg.v_head_dim)
    mla = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    print(f"per token per layer: MLA {mla} vs MHA {naive} numbers -> {naive / mla:.1f}x smaller")
    v = V3_671B
    n_, m_ = v["num_heads"] * (v["qk_nope_head_dim"] + v["qk_rope_head_dim"] + v["v_head_dim"]), \
        v["kv_lora_rank"] + v["qk_rope_head_dim"]
    print(f"  (V3-671B: {m_} vs {n_} -> {n_ / m_:.0f}x)")

    print("\n--- MoE ---")
    print(f"dense layers         0..{cfg.first_k_dense_replace - 1}")
    print(f"MoE layers           {cfg.first_k_dense_replace}..{cfg.num_layers - 1}")
    print(f"routing idx          {routings[0].shape}  (tokens x top-{cfg.num_experts_per_tok})")
    print(f"experts chosen by token 0: {routings[0][0].tolist()} "
          f"of {cfg.n_routed_experts}, groups of {cfg.n_routed_experts // cfg.n_group}")

    print("\n--- params ---")
    tot = 0
    for name, p in tree_flatten(model.parameters()):
        tot += p.size
    print(f"total                {tot / 1e6:.2f}M")
    moe = sum(p.size for nm, p in tree_flatten(model.parameters()) if ".experts." in nm)
    per_tok = moe * cfg.num_experts_per_tok / cfg.n_routed_experts
    print(f"routed-expert params {moe / 1e6:.2f}M, of which active per token "
          f"{per_tok / 1e6:.2f}M ({cfg.num_experts_per_tok}/{cfg.n_routed_experts})")
    print(f"active per token     {(tot - moe + per_tok) / 1e6:.2f}M")


def run_tests(cfg: Config):
    mx.random.seed(0)
    model = DeepSeekV3(cfg)
    mx.eval(model.parameters())
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    # 1. incremental decoding must equal a single full forward pass
    ids = mx.array([list(b"the quick brown fox jumps")])
    full, _, _ = model(ids)
    step_logits, cache = [], None
    for t in range(ids.shape[1]):
        lg, cache, _ = model(ids[:, t : t + 1], cache)
        step_logits.append(lg)
    inc = mx.concatenate(step_logits, axis=1)
    check(f"cached decode == full forward (max diff {mx.abs(full - inc).max().item():.2e})",
          mx.allclose(full, inc, atol=2e-4))

    # 2. chunked prefill (offset > 0 with L > 1) -- the mask bug
    a, b = ids[:, :10], ids[:, 10:]
    lg1, c1, _ = model(a)
    lg2, _, _ = model(b, c1)
    chunked = mx.concatenate([lg1, lg2], axis=1)
    check(f"chunked prefill == full forward (max diff {mx.abs(full - chunked).max().item():.2e})",
          mx.allclose(full, chunked, atol=2e-4))

    # 3. causality: perturbing token t must not change logits before t
    ids2 = mx.array(ids.tolist())
    ids2[0, 15] = 65
    alt, _, _ = model(ids2)
    check("future tokens do not leak into the past",
          mx.allclose(full[:, :15], alt[:, :15], atol=1e-5)
          and not mx.allclose(full[:, 15:], alt[:, 15:], atol=1e-5))

    # 4. routing respects group-limited constraint
    _, _, routings = model(ids)
    per_group = cfg.n_routed_experts // cfg.n_group
    groups = routings[0] // per_group
    n_groups_used = mx.array([len(set(row)) for row in groups.tolist()]).max().item()
    check(f"tokens span <= topk_group={cfg.topk_group} groups (max used {n_groups_used})",
          n_groups_used <= cfg.topk_group)

    # 5. top-k indices are distinct per token
    distinct = all(len(set(r)) == cfg.num_experts_per_tok for r in routings[0].tolist())
    check("top-k experts distinct per token", distinct)

    # 6. gate weights sum to routed_scaling_factor when norm_topk_prob
    xf = mx.random.normal((7, cfg.hidden_size))
    _, w = model.layers[-1].mlp.gate(xf)
    check("gate weights sum to routed_scaling_factor",
          mx.allclose(w.sum(-1), mx.full((7,), cfg.routed_scaling_factor), atol=1e-4))

    # 7. SwitchGLU gather path == running the expert densely
    sg = SwitchGLU(8, 16, 6)
    mx.eval(sg.parameters())
    x, idx = mx.random.normal((3, 8)), mx.array([[0, 5], [2, 1], [4, 3]])
    got = sg(x, idx)
    ref = mx.stack([mx.stack([
        (nn.silu(x[t] @ sg.gate_proj[e].T) * (x[t] @ sg.up_proj[e].T)) @ sg.down_proj[e].T
        for e in idx[t].tolist()]) for t in range(3)])
    check(f"SwitchGLU == dense reference (max diff {mx.abs(got - ref).max().item():.2e})",
          mx.allclose(got, ref, atol=1e-5))

    # 8. bias update is gradient-free and moves the right way
    g = model.layers[-1].mlp.gate
    before = g.e_score_correction_bias
    hot = mx.zeros((4, cfg.num_experts_per_tok), dtype=mx.int32)  # everyone picks expert 0
    model.update_expert_bias([hot] * sum(l.is_moe for l in model.layers))
    after = model.layers[-1].mlp.gate.e_score_correction_bias
    check("overloaded expert's bias decreased", (after[0] < before[0]).item())

    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return ok


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["shapes", "test", "train"], nargs="?", default="shapes")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--prompt", type=str, default="The transformer")
    a = ap.parse_args()

    cfg = Config()
    if a.mode == "shapes":
        show_shapes(cfg)
    elif a.mode == "test":
        raise SystemExit(0 if run_tests(cfg) else 1)
    else:
        text = open(a.data).read() if a.data else CORPUS
        mx.random.seed(0)
        model = train(cfg, text, a.steps, a.lr, a.batch_size, a.seq_len)
        prompt = mx.array([list(a.prompt.encode("utf-8"))])
        for temp in (0.0, 0.7):
            out = generate(model, prompt, max_tokens=220, temp=temp)
            txt = bytes(out[0].tolist()).decode("utf-8", errors="replace")
            print(f"\n--- temp={temp} ---\n{a.prompt}{txt}")


if __name__ == "__main__":
    main()
