"""
Qwen3.5-27B (`model_type: qwen3_5`) text decoder, from scratch in MLX.

The architecture, straight from config.json:

    64 layers, hidden 5120, head_dim 256, vocab 248320
    layer_types = [linear, linear, linear, full] x 16

  * 16 FULL-ATTENTION layers -- GQA 24 Q heads / 4 KV heads, per-head QK-RMSNorm,
    a sigmoid OUTPUT GATE (q_proj is double width: [Q | gate] per head), and
    RoPE on only the first 64 of 256 head dims (partial_rotary_factor 0.25).

  * 48 GATED-DELTANET layers -- linear attention. No KV cache: a fixed
    128x128 state matrix per head, updated by the gated delta rule

        S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T
        o_t = S_t^T q_t

    with a_t = exp(-exp(A_log) * softplus(a + dt_bias)) in (0,1) as a forget
    gate and b_t = sigmoid(b) as the delta-rule write strength. Q/K/V first pass
    through a depthwise causal conv (kernel 4) + silu; the output is swish-gated
    and RMSNormed. 16 K heads serve 48 V heads (3 each).

  That 3:1 ratio of linear to full layers is why 256K context is affordable:
  three quarters of the layers cost O(1) memory per token instead of O(n).

VERIFIED: every tensor name and shape against the checkpoint; the architecture
against the model card ("16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated
Attention -> FFN))"); and every convention below against the mlx-vlm reference
implementation (mlx_vlm/models/qwen3_5/):
  * per-head [Q | gate] split inside q_proj  (reference splits after
    reshape(B, L, n_heads, -1), so Q and gate interleave per head)
  * rotate_half RoPE pairing. "mrope_interleaved" in the config refers to how
    M-RoPE assigns the t/h/w position axes across frequency bands, NOT to the
    rotary pairing -- the reference maps style "interleaved" to _HALF_SPLIT.
  * contiguous [q | k | v] inside in_proj_qkv, with repeat_interleave 16 -> 48
  * normalize-then-gate in the gated RMSNorm (see GatedRMSNorm -- getting this
    backwards costs 3.2 nats/token and is invisible except as bad output)
  * the gated delta rule, verified algebraically equivalent to the reference

Two things that make this model hard to reimplement blind, both found here the
hard way and both silent (fluent nonsense, never an error): the zero-centered
norm gammas (see RMSNorm) and the gated-norm order (see GatedRMSNorm).

Text-only: M-RoPE (mrope_section [11,11,10]) degenerates EXACTLY to standard
RoPE when the time/height/width position components are all equal, which is the
case for text tokens. Images need the real 3-way split; see note in `rope`.
The vision tower is not loaded. Multi-token prediction IS loaded (see
MTPDraft) and drives speculative decoding in speculative.py.
"""

import glob
import json
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

import delta_kernel


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class TextConfig:
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    intermediate_size: int = 17408
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    attn_output_gate: bool = True
    layer_types: Tuple[str, ...] = ()
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    mtp_num_hidden_layers: int = 0

    @classmethod
    def from_json(cls, path: str) -> "TextConfig":
        raw = json.load(open(path))
        t = raw["text_config"]
        rope = t.get("rope_parameters", {})
        keep = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in t.items() if k in keep}
        kw["layer_types"] = tuple(t["layer_types"])
        kw["rope_theta"] = float(rope.get("rope_theta", t.get("rope_theta", 1e7)))
        kw["partial_rotary_factor"] = rope.get(
            "partial_rotary_factor", t.get("partial_rotary_factor", 0.25)
        )
        return cls(**kw)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """x_hat * (1 + weight)  -- ZERO-CENTERED gamma.

    Determined from the checkpoint, not assumed: input_layernorm gammas have
    mean -0.033 and post_attention_layernorm gammas are entirely <= 0 (mean
    -0.217). As a plain multiplicative gain those would negate and annihilate
    the block input, so the stored value is an offset from 1. Confirmed by
    perplexity: plain w gives 13.6 nats/token, (1 + w) gives 5.1.

    (1 + w) is computed in fp32 -- bf16 resolution near 1.0 is only ~0.004,
    which would quantize a gamma of -0.033 into uselessness.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps
        self._gamma = None      # leading underscore => MLX does not treat it as a param

    def __call__(self, x):
        # Cache (1 + w) in fp32. Recomputing it per call costs two extra kernel
        # launches on a 5120-element array, and there are ~161 live norms per
        # forward pass -- ~322 launches per token for a value that never changes.
        if self._gamma is None or self._gid != id(self.weight):
            self._gamma = 1.0 + self.weight.astype(mx.float32)
            self._gid = id(self.weight)
            mx.eval(self._gamma)
        return mx.fast.rms_norm(x.astype(mx.float32), self._gamma, self.eps).astype(x.dtype)


class GatedRMSNorm(nn.Module):
    """out = silu(z) * rmsnorm(x, weight)  -- NORMALIZE FIRST, THEN GATE.

    The order matters and is easy to get backwards. Gating inside the norm --
    rmsnorm(x * silu(z)) -- divides the gate's own magnitude back out, so the
    gate can only rotate the direction of the vector, never scale it. Here the
    gate survives, which is the whole point of it.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        self._gamma = None

    def __call__(self, x, z):
        if self._gamma is None or self._gid != id(self.weight):
            self._gamma = self.weight.astype(mx.float32)
            self._gid = id(self.weight)
            mx.eval(self._gamma)
        h = mx.fast.rms_norm(x.astype(mx.float32), self._gamma, self.eps)
        return h * nn.silu(z.astype(mx.float32))


def rope(x: mx.array, offset: int, rotary_dim: int, theta: float) -> mx.array:
    """RoPE on the first `rotary_dim` dims of (B, H, L, head_dim); rest passes through.

    For text this is exactly M-RoPE: with mrope_section [11,11,10] the three
    sections index time/height/width positions, and for a text token all three
    are the token index, so every section sees the same value. Images break that
    equality and need the real 3-way position split.
    """
    B, H, L, D = x.shape
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    pos = mx.arange(offset, offset + L, dtype=mx.float32)[:, None]
    inv = theta ** (-mx.arange(0, half, dtype=mx.float32) / half)[None, :]
    ang = pos * inv
    cos, sin = mx.cos(ang), mx.sin(ang)
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return mx.concatenate([rotated.astype(x.dtype), x_pass], axis=-1)


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


# ---------------------------------------------------------------------------
# caches
# ---------------------------------------------------------------------------


class KVCache:
    """Pre-allocated, growable KV cache for the full-attention layers.

    Naive `mx.concatenate([cache, new], axis=2)` on every decode step
    reallocates and copies the entire cache each token: O(n) per step, O(n^2)
    over a generation. Here the buffer is over-allocated in blocks of `step`
    and new keys/values are written in place, so the amortized cost per token is
    O(1) and a reallocation happens only once every `step` tokens.

    `offset` is the true logical length; the buffer is usually longer, so always
    hand attention the `[..., :offset, :]` slice rather than the raw buffer.
    """

    step = 256

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        prev = self.offset
        L = keys.shape[2]

        if self.keys is None or prev + L > self.keys.shape[2]:
            B, H, _, dk = keys.shape
            dv = values.shape[3]
            grow = ((L + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((B, H, grow, dk), keys.dtype)
            new_v = mx.zeros((B, H, grow, dv), values.dtype)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                # drop any unused tail before extending so the buffer stays dense
                if prev < self.keys.shape[2]:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

        self.offset = prev + L
        self.keys[..., prev:self.offset, :] = keys
        self.values[..., prev:self.offset, :] = values
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

    def trim(self, length: int):
        """Roll back to `length` tokens. Rejected speculative keys/values are left
        in the buffer as garbage beyond `offset`; the next write overwrites them,
        and nothing ever reads past `offset`."""
        self.offset = length


class DeltaCache:
    """State for one Gated-DeltaNet layer, with rollback support.

    The conv ring holds kernel_size-1 timesteps and `state` is one (H, Dk, Dv)
    matrix per head -- both fixed size no matter how long the context gets.

    Rollback is the hard part of speculative decoding for a recurrent layer. A KV
    cache rolls back by truncation, but delta-rule updates cannot be un-applied:
    S_t = a(I - b k k^T)S_{t-1} + b k v^T destroys information about S_{t-1}.
    So in `record` mode we do NOT advance the state. Instead we keep every
    per-step state the forward pass already computed, and `commit(n)` just picks
    the n-th one. Costs ~3.1 MB per step per layer (~750 MB for a 5-token block
    across 48 layers) and zero extra compute.
    """

    def __init__(self):
        self.conv: Optional[mx.array] = None
        self.state: Optional[mx.array] = None
        self.offset = 0
        self.record = False
        self.pending = None
        self.base_offset = 0

    def commit(self, n: int):
        """Accept the first n tokens of the recorded block and drop the rest."""
        conv_input, S_prev, states = self.pending
        L = len(states)
        state = S_prev if n == 0 else states[n - 1]
        kw = conv_input.shape[1] - L
        conv = conv_input[:, n:n + kw]
        # mx.contiguous DETACHES these from the graph that produced them. Without
        # it, holding a slice keeps the whole block's computation alive -- all L
        # states and the conv input -- and it compounds every round.
        self.state = mx.contiguous(state)
        self.conv = mx.contiguous(conv)
        self.offset = self.base_offset + n
        self.pending = None
        self.record = False


# ---------------------------------------------------------------------------
# gated delta rule: chunked (prefill) and single-step (decode) forms
# ---------------------------------------------------------------------------


# Sequences at least this long use the chunked scan; shorter ones (i.e. decode)
# use the per-token recurrence. Set to a huge number to force sequential.
CHUNK_THRESHOLD = 16

# The fused Metal kernel (delta_kernel.py) beats both MLX paths up to ~L=384.
# Above that the chunked scan's time-parallelism wins: the kernel is serial in t
# (one thread per (b,h,dv)), which is memory-optimal but exposes no parallelism
# across time, while the chunked scan turns time into matmuls.
# Measured, 48 chained layers: L=1 2.4 vs 6.7ms | L=64 20 vs 133ms |
#                              L=256 88 vs 138ms | L=512 190 vs 164ms
METAL_MAX_L = 384
USE_METAL_DELTA = True


def _invert_unit_lower(Tm: mx.array, C: int) -> mx.array:
    """Inverse of a batched unit-lower-triangular matrix by forward substitution.

    Sequential in C, but C is a fixed 64 and every chunk/head/batch is solved
    simultaneously -- so this is 64 steps regardless of sequence length, which is
    the whole point. A log-depth Neumann/squaring inverse would be shallower but
    overflows here: correlated keys give KK off-diagonals of order 1.
    """
    X = mx.broadcast_to(mx.eye(C, dtype=mx.float32), Tm.shape) + mx.zeros_like(Tm)
    rows = [X[..., 0:1, :]]
    for i in range(1, C):
        rows.append(X[..., i:i + 1, :] - Tm[..., i:i + 1, :i] @ mx.concatenate(rows, -2))
    return mx.concatenate(rows, axis=-2)


def chunked_delta(q, k, v, alpha, beta, S, C: int = 64):
    """Chunked parallel form of the gated delta rule. Exactly equivalent to the
    per-token recurrence, but the sequential depth drops from L to L/C.

    The recurrence S_t = a_t(I - b_t k k^T)S_{t-1} + b_t k v^T is affine in
    S_{t-1}, so within a chunk of C tokens the whole thing can be unrolled into
    matmuls. The catch is the (I - b k k^T) factors compose into a unit lower
    triangular matrix that must be inverted -- one CxC triangular solve per
    chunk, batched. What remains sequential is only the chunk-to-chunk state
    hand-off: L/64 steps instead of L.

    q,k: (B,L,H,Dk)  v: (B,L,H,Dv)  alpha,beta: (B,L,H)  S: (B,H,Dk,Dv)
    """
    B, L, H, Dk = q.shape
    Dv = v.shape[-1]

    pad = (C - L % C) % C
    if pad:  # alpha=1, beta=0 makes padded steps exact no-ops
        q = mx.concatenate([q, mx.zeros((B, pad, H, Dk), q.dtype)], axis=1)
        k = mx.concatenate([k, mx.zeros((B, pad, H, Dk), k.dtype)], axis=1)
        v = mx.concatenate([v, mx.zeros((B, pad, H, Dv), v.dtype)], axis=1)
        alpha = mx.concatenate([alpha, mx.ones((B, pad, H), alpha.dtype)], axis=1)
        beta = mx.concatenate([beta, mx.zeros((B, pad, H), beta.dtype)], axis=1)
    Lp, nC = L + pad, (L + pad) // C

    def rc(x, D):  # (B,Lp,H,D) -> (B,H,nC,C,D)
        return x.reshape(B, nC, C, H, D).transpose(0, 3, 1, 2, 4).astype(mx.float32)

    q, k, v = rc(q, Dk), rc(k, Dk), rc(v, Dv)
    g = alpha.reshape(B, nC, C, H).transpose(0, 3, 1, 2).astype(mx.float32)
    bt = beta.reshape(B, nC, C, H).transpose(0, 3, 1, 2).astype(mx.float32)

    # cumulative decay within each chunk, in log space.
    # clip off 0: alpha can underflow to exactly 0 and log(0) would poison
    # every decay ratio with NaN.
    lcg = mx.cumsum(mx.log(mx.clip(g, 1e-6, 1.0)), axis=-1)
    cumg = mx.exp(lcg)
    lower = mx.tril(mx.ones((C, C), mx.float32), 0)
    slower = mx.tril(mx.ones((C, C), mx.float32), -1)
    # dr[i,j] = cumg_i/cumg_j. Mask the exponent to the lower triangle BEFORE
    # exp: the upper triangle would overflow (alpha<1) and inf*0 is NaN.
    dr = mx.exp(mx.where(lower > 0, lcg[..., :, None] - lcg[..., None, :], -1e30))

    A = bt[..., :, None] * dr * (k @ mx.swapaxes(k, -1, -2)) * slower
    Tinv = _invert_unit_lower(mx.eye(C, dtype=mx.float32) + A, C)

    U0 = Tinv @ (bt[..., :, None] * v)                       # (B,H,nC,C,Dv)
    Kt = Tinv @ ((bt * cumg)[..., :, None] * k)              # (B,H,nC,C,Dk)
    M = dr * (q @ mx.swapaxes(k, -1, -2)) * lower
    Qeff = cumg[..., :, None] * q - M @ Kt
    MU0 = M @ U0
    cumg_last = cumg[..., -1]
    ratio_last = mx.exp(lcg[..., -1, None] - lcg)            # cumg_last / cumg_i

    if S is None:
        S = mx.zeros((B, H, Dk, Dv), mx.float32)
    else:
        S = mx.swapaxes(S, -1, -2)      # (B,H,Dv,Dk) -> (B,H,Dk,Dv) internally
    ys = []
    for c in range(nC):                                       # only L/C steps
        ys.append(MU0[:, :, c] + Qeff[:, :, c] @ S)
        Uc = U0[:, :, c] - Kt[:, :, c] @ S
        Us = ratio_last[:, :, c][..., None] * Uc
        S = (cumg_last[:, :, c][..., None, None] * S
             + mx.swapaxes(k[:, :, c], -1, -2) @ Us)
    Y = mx.stack(ys, axis=2).transpose(0, 2, 3, 1, 4).reshape(B, Lp, H, Dv)[:, :L]
    return Y, mx.swapaxes(S, -1, -2)    # back to (B,H,Dv,Dk)


@partial(mx.compile, shapeless=True)
def _delta_step(q_t, k_t, v_t, a_t, b_t, S):
    """One gated-delta step. State is (B, H, Dv, Dk).

    The (Dv, Dk) layout is deliberate: it makes each Dv row contiguous over Dk,
    which is what the fused Metal kernel in delta_kernel.py needs so that both
    reductions stay inside one thread. mx.compile fuses this chain as far as MLX
    can; the Metal kernel goes further (state read once instead of ~7 times).

    q_t, k_t: (B,H,1,Dk)   v_t: (B,H,Dv,1)   a_t,b_t: (B,H,1,1)
    """
    kS = mx.sum(k_t * S, axis=-1, keepdims=True)         # k . S  -> (B,H,Dv,1)
    S = a_t * (S - b_t * (kS * k_t)) + b_t * (v_t * k_t)
    return mx.sum(q_t * S, axis=-1), S                   # q . S  -> (B,H,Dv)


def sequential_delta(q, k, v, alpha, beta, S, collect=False):
    """Per-token reference recurrence. Used for decode (L=1), for speculative
    verification, and to test the chunked form.

    With collect=True it also returns the per-step states [S_1..S_L]. These are
    materialized by the loop anyway, so keeping references costs no extra compute
    and no extra allocation -- it only keeps them alive (~3.1 MB per step per
    layer). That turns speculative rollback into an index lookup instead of a
    replay, which is worth it: replaying cost ~45 ms per round in kernel launches.
    """
    B, L, H, Dk = q.shape
    if S is None:
        S = mx.zeros((B, H, v.shape[-1], Dk), mx.float32)
    outs, states = [], []
    for t in range(L):
        o_t, S = _delta_step(q[:, t][:, :, None, :],
                             k[:, t][:, :, None, :],
                             v[:, t][..., None],
                             alpha[:, t][:, :, None, None],
                             beta[:, t][:, :, None, None],
                             S)
        outs.append(o_t)
        if collect:
            states.append(S)
    Y = mx.stack(outs, axis=1)
    return (Y, S, states) if collect else (Y, S)


# ---------------------------------------------------------------------------
# full attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.scale = self.hd ** -0.5

        # double width: per head this emits [query (hd) | gate (hd)]
        q_out = self.n_heads * self.hd * (2 if cfg.attn_output_gate else 1)
        self.q_proj = nn.Linear(cfg.hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)  # per-head, on head_dim
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None, rope_offset=None):
        cfg = self.cfg
        B, L, _ = x.shape
        # read the offset BEFORE update_and_fetch advances it
        offset = cache.offset if cache is not None else 0
        # The MTP drafter's KV cache starts mid-sequence, so its cache index and
        # its absolute position differ; let the caller supply the true position.
        pos = offset if rope_offset is None else rope_offset

        if cfg.attn_output_gate:
            # reshape to (B, L, n_heads, 2*hd) then split per head -- NOT a global
            # first-half/second-half split of the 12288 columns.
            qg = self.q_proj(x).reshape(B, L, self.n_heads, 2 * self.hd)
            q, gate = qg[..., : self.hd], qg[..., self.hd :]
            gate = gate.reshape(B, L, self.n_heads * self.hd)
        else:
            q = self.q_proj(x).reshape(B, L, self.n_heads, self.hd)
            gate = None

        k = self.k_proj(x).reshape(B, L, self.n_kv, self.hd)
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.hd)

        q = self.q_norm(q).transpose(0, 2, 1, 3)  # QK-Norm before RoPE
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = rope(q, pos, cfg.rotary_dim, cfg.rope_theta)
        k = rope(k, pos, cfg.rotary_dim, cfg.rope_theta)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.hd)
        if gate is not None:
            o = o * mx.sigmoid(gate)
        return self.o_proj(o)


# ---------------------------------------------------------------------------
# gated deltanet (linear attention)
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.nk = cfg.linear_num_key_heads      # 16
        self.nv = cfg.linear_num_value_heads    # 48
        self.dk = cfg.linear_key_head_dim       # 128
        self.dv = cfg.linear_value_head_dim     # 128
        self.rep = self.nv // self.nk           # 3 value heads per key head
        self.K = cfg.linear_conv_kernel_dim     # 4

        self.qkv_dim = self.nk * self.dk * 2 + self.nv * self.dv  # 10240
        self.z_dim = self.nv * self.dv                            # 6144

        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.qkv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, self.z_dim, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, self.nv, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, self.nv, bias=False)
        self.out_proj = nn.Linear(self.z_dim, cfg.hidden_size, bias=False)

        self.conv1d = _DepthwiseConv(self.qkv_dim, self.K)
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps)
        self.A_log = mx.zeros((self.nv,))
        self.dt_bias = mx.zeros((self.nv,))

    def __call__(self, x, cache=None):
        B, L, _ = x.shape
        conv_state = None if cache is None else cache.conv
        S = None if cache is None else cache.state

        qkv, conv_input = self.conv1d(self.in_proj_qkv(x), conv_state)
        qkv = nn.silu(qkv)

        # contiguous [q | k | v] layout
        nkd = self.nk * self.dk
        q = qkv[..., :nkd].reshape(B, L, self.nk, self.dk)
        k = qkv[..., nkd : 2 * nkd].reshape(B, L, self.nk, self.dk)
        v = qkv[..., 2 * nkd :].reshape(B, L, self.nv, self.dv)

        q, k = l2norm(q), l2norm(k)
        # Reference scaling: k gets plain l2norm; q additionally gets 1/sqrt(dk).
        # (rms_norm == sqrt(d)*l2norm, so their inv_scale factors reduce to this.)
        # The q factor cannot change the output -- the gated RMSNorm downstream is
        # scale-invariant -- but match it anyway so the intermediates line up.
        q = q * (self.dk ** -0.5)
        # repeat_interleave: key head i serves value heads 3i, 3i+1, 3i+2
        q = mx.repeat(q, self.rep, axis=2).astype(mx.float32)
        k = mx.repeat(k, self.rep, axis=2).astype(mx.float32)
        v = v.astype(mx.float32)

        beta = mx.sigmoid(self.in_proj_b(x).astype(mx.float32))            # (B,L,nv)
        a = self.in_proj_a(x).astype(mx.float32)
        A = mx.exp(self.A_log.astype(mx.float32))
        alpha = mx.exp(-A * nn.softplus(a + self.dt_bias.astype(mx.float32)))

        if S is None:
            S = mx.zeros((B, self.nv, self.dv, self.dk), mx.float32)
        S_prev = S

        # Prefill uses the chunked scan (sequential depth L/64 instead of L);
        # decode is a single step, where chunking would only pad 1 -> 64.
        # CHUNK_THRESHOLD is module-level so tests can force either path.
        recording = cache is not None and cache.record
        use_metal = (USE_METAL_DELTA and delta_kernel.available()
                     and L <= METAL_MAX_L)
        states = None
        if recording:
            # speculative blocks are short; collect per-step states for rollback
            if use_metal:
                o, S, states = delta_kernel.gated_delta_metal(
                    q, k, v, alpha, beta, S, collect=True)
            else:
                o, S, states = sequential_delta(q, k, v, alpha, beta, S,
                                                collect=True)
        elif use_metal:
            o, S = delta_kernel.gated_delta_metal(q, k, v, alpha, beta, S)
        elif L >= CHUNK_THRESHOLD:
            o, S = chunked_delta(q, k, v, alpha, beta, S)
        else:
            o, S = sequential_delta(q, k, v, alpha, beta, S)

        z = self.in_proj_z(x).reshape(B, L, self.nv, self.dv)
        o = self.norm(o, z).astype(x.dtype).reshape(B, L, self.z_dim)
        if cache is not None:
            if recording:
                # speculative: do not advance the state; keep the per-step states
                cache.base_offset = cache.offset
                cache.pending = (conv_input, S_prev, states)
            else:
                cache.conv = conv_input[:, -(self.K - 1):]
                cache.state = S
                cache.offset += L
        return self.out_proj(o)


class _DepthwiseConv(nn.Module):
    """Causal depthwise conv, kernel 4, no bias. Weight is (C, 1, K) as stored.

    Written as an explicit sum over the K taps rather than mx.conv1d(groups=C):
    it is unambiguous about layout and avoids a slow grouped-conv path.
    """

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.K = kernel
        self.weight = mx.zeros((channels, 1, kernel))

    def __call__(self, x, state=None):
        B, L, C = x.shape
        pad = mx.zeros((B, self.K - 1, C), x.dtype) if state is None else state
        xp = mx.concatenate([pad, x], axis=1)              # (B, L+K-1, C)
        w = self.weight[:, 0, :].T                          # (K, C)
        y = sum(xp[:, j : j + L, :] * w[j] for j in range(self.K))
        return y, xp   # xp is the full padded input; the last K-1 rows are the state


# ---------------------------------------------------------------------------
# blocks & model
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: TextConfig, i: int, force_full: bool = False):
        super().__init__()
        self.layer_type = "full_attention" if force_full else cfg.layer_types[i]
        self.is_linear = self.layer_type == "linear_attention"
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)

    def __call__(self, x, mask=None, cache=None, rope_offset=None):
        h = self.input_layernorm(x)
        # caches are mutated in place, so nothing is returned
        if self.is_linear:
            h = self.linear_attn(h, cache)
        else:
            h = self.self_attn(h, mask, cache, rope_offset)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class MTPDraft(nn.Module):
    """Multi-Token Prediction head, used here as a self-speculative drafter.

    Trained as an auxiliary objective (predict token t+2 as well as t+1), which
    densifies the training signal and forces h_t to encode more than the
    immediate next token. At inference it doubles as a draft model:

        h_t (target's final hidden, post-norm) ---> pre_fc_norm_hidden ---.
        emb(token t+1) (target's embedding table) -> pre_fc_norm_embedding -'
                            concat -> fc (10240 -> 5120)
                            -> 1 full-attention decoder layer
                            -> norm -> the target's OWN lm_head

    ~424M params, 1.6% of the model. Note the layer is full attention, not
    DeltaNet, and both the embedding table and the LM head are shared with the
    target (`mtp_use_dedicated_embeddings: false`, and no mtp lm_head is stored).
    """

    def __init__(self, cfg: TextConfig):
        super().__init__()
        h = cfg.hidden_size
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = RMSNorm(h, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(h, cfg.rms_norm_eps)
        self.layers = [Layer(cfg, 0, force_full=True)
                       for _ in range(max(1, cfg.mtp_num_hidden_layers))]
        self.norm = RMSNorm(h, cfg.rms_norm_eps)

    def make_cache(self) -> List:
        return [KVCache() for _ in self.layers]

    def __call__(self, token_embed, hidden, cache, rope_offset):
        h = mx.concatenate([self.pre_fc_norm_embedding(token_embed),
                            self.pre_fc_norm_hidden(hidden)], axis=-1)
        h = self.fc(h)
        L = h.shape[1]
        mask = None
        if L > 1:
            off = cache[0].offset
            qq = mx.arange(off, off + L)[:, None]
            kk = mx.arange(off + L)[None, :]
            mask = mx.where(kk <= qq, 0.0, mx.finfo(h.dtype).min).astype(h.dtype)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, rope_offset)
        return self.norm(h)


class Qwen35(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Layer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # named `mtp` so the checkpoint's mtp.* keys land here directly, which
        # means strict loading and the quantization predicate just work
        if cfg.mtp_num_hidden_layers:
            self.mtp = MTPDraft(cfg)

    def make_cache(self) -> List:
        """One cache object per layer: KVCache for full attention, DeltaCache otherwise."""
        return [DeltaCache() if t == "linear_attention" else KVCache()
                for t in self.cfg.layer_types]

    def hidden_states(self, ids: mx.array, cache: Optional[List] = None):
        x = self.embed_tokens(ids)
        L = ids.shape[1]

        if cache is None:
            cache = self.make_cache()
        # every layer's cache tracks the same logical length
        offset = cache[0].offset

        mask = None
        if L > 1:
            q = mx.arange(offset, offset + L)[:, None]
            kk = mx.arange(offset + L)[None, :]
            mask = mx.where(kk <= q, 0.0, mx.finfo(x.dtype).min).astype(x.dtype)

        for layer, c in zip(self.layers, cache):
            x = layer(x, mask, c)

        # post-norm hidden: this is exactly the tensor fed to lm_head, and it is
        # what the MTP drafter consumes (it applies its own pre_fc_norm_hidden)
        return self.norm(x), cache

    def __call__(self, ids: mx.array, cache: Optional[List] = None,
                 all_logits: bool = True):
        h, cache = self.hidden_states(ids, cache)
        # Generation only ever needs the last position. Projecting all L positions
        # through a 248320-wide lm_head costs 5120*248320 flops per token and
        # materializes an L x 248320 tensor (1 GB at L=2048), for nothing.
        if not all_logits:
            h = h[:, -1:]
        return self.lm_head(h), cache


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def find_model(path: str) -> str:
    """Resolve a model directory given as a bare name or a path.

    The sources live in python/ while the checkpoints sit at the repository root,
    so `--model qwen3.5-27b-4bit` has to work whether you run from the root, from
    python/, or from anywhere else via the `ai` launcher. Tries the path as given,
    then relative to this file, then relative to its parent.
    """
    if os.path.isdir(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here)):
        cand = os.path.join(base, path)
        if os.path.isdir(cand):
            return cand
    raise SystemExit(f"model directory not found: {path}")


def load(path: str, verbose: bool = True) -> Tuple[Qwen35, TextConfig]:
    path = find_model(path)
    cfg_path = os.path.join(path, "config.json")
    cfg = TextConfig.from_json(cfg_path)
    raw = json.load(open(cfg_path))
    quant = raw.get("quantization")

    model = Qwen35(cfg)

    if quant:
        qmods = set(quant["quantized_modules"])
        # our module path -> the HF name recorded in the file
        def pred(p, m):
            return p in qmods or ("model.language_model." + p) in qmods
        nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"],
                    class_predicate=pred)

    weights = {}
    skipped = 0
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        for k, v in mx.load(f).items():
            if k.startswith("model.visual."):
                skipped += 1
                continue
            weights[k.removeprefix("model.language_model.")] = v

    # strict=True is the real structural check: it fails loudly if any name or
    # shape in our module tree disagrees with the checkpoint.

    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    if verbose:
        nbytes = sum(v.nbytes for v in weights.values())
        print(f"loaded {len(weights)} tensors, {nbytes / 1e9:.2f} GB "
              f"({skipped} vision tensors skipped)")
    return model, cfg
