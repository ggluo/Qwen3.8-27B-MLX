# How a tensor gets through the model

A walk through one forward pass of Qwen3.8-27B (`model_type: qwen3_5`) as this
repo implements it: the shape of the tensor at every step, the module that
produces it, and what each module costs — parameters, bytes, FLOPs. Then what the
KV cache is, and why prefill and decode behave so differently.

Names follow the C++ port (`cpp/src/model.cpp`); the Python port
(`python/qwen35.py`) uses the same ones. Parameter and byte counts are summed
from the 4-bit checkpoint's own safetensors headers
(`qwen3.5-27b-4bit-uncensored`). Timings are for the M3 Max this was built on:
40-core GPU, 400 GB/s, ~14.3 TFLOP/s peak.

[`PERFORMANCE_MODEL.md`](PERFORMANCE_MODEL.md) is the companion: it takes these
numbers and builds the roofline from them.

---

## Notation

Every tensor is written as `(batch, tokens, ...)`. Batch is always 1 here.

| symbol | meaning |
|---|---|
| **L** | tokens in **this** forward pass: the prefill window (up to 1024), or 1 for decode, or k+1 for a speculative verify |
| **P** | tokens the caches **already** hold: the position of this pass's first token |
| **S** | `P + L`, the context the attention layers see |

`L` and `P` are independent. Keeping them apart is the point of most of the
shape bookkeeping below; see §1 of `PERFORMANCE_MODEL.md`.

From `config.json`:

| | |
|---|---|
| hidden size `H` | 5120 |
| layers | 64 = `[DeltaNet, DeltaNet, DeltaNet, Attention] × 16` |
| MLP intermediate `I` | 17408 |
| vocabulary `V` | 248320 |
| full attention | 24 query heads, 4 key/value heads, head dim 256, RoPE on 64 of 256 dims |
| Gated DeltaNet | 16 key heads, 48 value heads, head dim 128, conv width 4 |

---

## 1. The whole model

```
token ids                 (1, L)            int32
  │  embed_tokens         248320 × 5120 table, a lookup
  ▼
x                         (1, L, 5120)      bf16       one 5120-vector per token
  │
  │  64 layers. Each one:
  │      x = x + mixer(input_layernorm(x))           tokens exchange information here
  │      x = x + mlp(post_attention_layernorm(x))    each token on its own
  │  The mixer is a Gated DeltaNet in 48 layers and full attention in 16.
  │  The shape never changes: (1, L, 5120) all the way down.
  ▼
norm                      (1, L, 5120)
  │  keep the last position only
  ▼
                          (1, 1, 5120)
  │  lm_head              5120 → 248320
  ▼
logits                    (1, 1, 248320)    a score for every vocabulary entry; sample one
```

`x` is the **residual stream**. Every layer reads it, computes something, and
adds the result back, so information that a layer does not touch passes straight
through. The mixer is the only place where one token's vector is influenced by
another's. The norms, the MLP and every projection treat each of the `L` tokens
independently — which is what makes them batchable (§6).

Only the last position goes through `lm_head`, because only the last position is
sampled. Projecting all of them would cost 2.5 GFLOP per token and, at `L = 2048`,
a 1 GB logits tensor for nothing.

---

## 2. Inside a layer

### RMSNorm — `input_layernorm`, `post_attention_layernorm` (every layer)

```
x  (1, L, 5120)  →  x / rms(x) * (1 + γ)  →  (1, L, 5120)
```

Normalizes each token's vector to unit RMS, then scales each channel. The stored
γ is an **offset from 1** — a silent convention, see the root README. 5120
parameters, ~4 FLOPs per element: negligible in time, but it runs 128 times per
pass.

### MLP — SwiGLU (every layer)

```
x                (1, L, 5120)
├─ gate_proj  →  (1, L, 17408)
├─ up_proj    →  (1, L, 17408)
│     silu(gate) * up
└─ down_proj  →  (1, L, 5120)
```

Three 5120 × 17408 matrices. Per token it is three matrix–vector products,
`2 · 5120 · 17408` FLOPs each. **The MLP is two thirds of the model's
arithmetic and two thirds of its bytes**; attention gets the attention, the MLP
does the work.

### Full attention (16 layers: 3, 7, 11, …, 63)

```
x                         (1, L, 5120)

q_proj                 →  (1, L, 12288)   double width: per head [query | gate]
   reshape, split per head:
   q                      (1, L, 24, 256)
   gate                   (1, L, 6144)            kept for the output
k_proj                 →  (1, L, 1024)  → (1, L, 4, 256)
v_proj                 →  (1, L, 1024)  → (1, L, 4, 256)

q_norm, k_norm            RMSNorm over each head's 256 dims
transpose                 q (1, 24, L, 256)   k (1, 4, L, 256)   v (1, 4, L, 256)
RoPE                      rotate dims 0..63 by positions P .. P+L-1

KV cache                  write k, v into slots [P, P+L); read back slots [0, S):
                          K (1, 4, S, 256)    V (1, 4, S, 256)

attention                 each key/value head serves 6 query heads (GQA)
   scores = q·Kᵀ/16       (1, 24, L, S)       every new token against every cached one
   + causal mask          (L, S)              a token cannot see anything after it
   softmax · V         →  (1, 24, L, 256)

transpose, reshape     →  (1, L, 6144)
* sigmoid(gate)           the output gate
o_proj                 →  (1, L, 5120)
```

The `q·Kᵀ` and `·V` products use **no weights**. Their cost grows with the
context `S`, not the parameter count — negligible at 1K tokens, and more than
all of the layer's weight FLOPs past ~30K (§3).

Note the mask is `(L, S)`, not `(L, L)`: the `L` new queries sit at positions
`P .. P+L-1` and see every one of the `S` keys up to their own position. A square
mask happens to work for a first prefill (`P = 0`) and for decode (`L = 1`), and
silently misaligns everything else.

### Gated DeltaNet (48 layers: 0–2, 4–6, …, 60–62)

```
x                         (1, L, 5120)

in_proj_qkv            →  (1, L, 10240)          [q | k | v] = 2048 + 2048 + 6144
conv1d                    depthwise, width 4, causal. The conv cache's last 3 rows go in
                          front: (1, 3+L, 10240) → (1, L, 10240), then silu
split                     q (1, L, 16, 128)   k (1, L, 16, 128)   v (1, L, 48, 128)
l2norm q and k; q *= 1/√128
repeat q, k 16 → 48 heads, to fp32       (1, L, 48, 128)

in_proj_b → sigmoid    →  β  (1, L, 48)          write strength
in_proj_a → decay      →  α  (1, L, 48)          α = exp(−exp(A_log)·softplus(a + dt_bias))

state                     S  (1, 48, 128, 128)   fp32, FIXED SIZE whatever the context
recurrence, for t = 0 .. L-1     (the Metal kernel in cpp/src/delta.cpp)
   S ← α_t (S − β_t (S k_t) k_tᵀ) + β_t v_t k_tᵀ
   o_t = S q_t
                       →  o  (1, L, 48, 128)

in_proj_z              →  z  (1, L, 6144) → (1, L, 48, 128)
gated norm                rmsnorm(o) * silu(z)    normalize FIRST, then gate
reshape                →  (1, L, 6144)
out_proj               →  (1, L, 5120)
```

Instead of keeping every past token, the layer folds the whole history into one
128×128 matrix per head. Each token decays it (`α`), erases what the state
currently predicts for `k` (`− β (S k) kᵀ`), and writes `v` in its place
(`+ β v kᵀ`). Reading is one matrix–vector product, `S q`.

The recurrence is the only part of the model that is **serial in the tokens**:
step `t` needs the state left by step `t−1`. It is tiny in FLOPs, but it cannot
be turned into a matrix multiply over `L` the way everything else can. That is
why it has a hand-written kernel.

---

## 3. What each module costs

Per token, for one layer. Matmul FLOPs are `2 × parameters`: one multiply and one
add per weight. **Bytes** is what the 4-bit checkpoint stores: 4 bits per weight,
plus a bf16 scale and bias per 64 weights, which is 4.5 bits, or 0.5625
bytes per weight. A few small tensors are kept in bf16 (marked below).

### Full-attention layer

| module | shape | parameters | bytes | FLOPs / token |
|---|---|---:|---:|---:|
| `input_layernorm` | 5120 | 5,120 | 10 KB | ~20 K |
| `q_proj` | 5120 → 12288 | 62.9 M | 35.4 MB | 125.8 M |
| `k_proj` | 5120 → 1024 | 5.2 M | 2.9 MB | 10.5 M |
| `v_proj` | 5120 → 1024 | 5.2 M | 2.9 MB | 10.5 M |
| `q_norm`, `k_norm`, RoPE | per head | 512 | 1 KB | ~0.1 M |
| scores `q·Kᵀ` + `·V` | context `S` | — | reads the KV cache (§4) | **24,576 × S** |
| `o_proj` | 6144 → 5120 | 31.5 M | 17.7 MB | 62.9 M |
| `post_attention_layernorm` | 5120 | 5,120 | 10 KB | ~20 K |
| MLP: `gate`, `up`, `down` | 5120 ↔ 17408 | 267.4 M | 150.4 MB | 534.8 M |
| **layer total** | | **372.2 M** | **209.4 MB** | **744.5 M + 24,576 × S** |

The score term per layer: `2 · 24 heads · 256 · S` for `q·Kᵀ`, the same again
for `·V`. At 1K context that is 25 MFLOP (3% of the layer); at 30K it equals the
weight FLOPs; at 100K it is 2.5 GFLOP, more than three times them.

### Gated DeltaNet layer

| module | shape | parameters | bytes | FLOPs / token |
|---|---|---:|---:|---:|
| `input_layernorm` | 5120 | 5,120 | 10 KB | ~20 K |
| `in_proj_qkv` | 5120 → 10240 | 52.4 M | 29.5 MB | 104.9 M |
| `conv1d` | 4 taps × 10240 | 41 K | 82 KB (bf16) | 0.08 M |
| `in_proj_z` | 5120 → 6144 | 31.5 M | 17.7 MB | 62.9 M |
| `in_proj_a`, `in_proj_b` | 5120 → 48, twice | 0.5 M | 0.98 MB (bf16) | 1.0 M |
| recurrence | state 48 × 128 × 128 | — | 3.1 MB state, read + written | **7.1 M** |
| gated norm | 6144 | 128 | — | ~0.05 M |
| `out_proj` | 6144 → 5120 | 31.5 M | 17.7 MB | 62.9 M |
| `post_attention_layernorm` | 5120 | 5,120 | 10 KB | ~20 K |
| MLP: `gate`, `up`, `down` | 5120 ↔ 17408 | 267.4 M | 150.4 MB | 534.8 M |
| **layer total** | | **383.2 M** | **216.4 MB** | **773.6 M** (fixed) |

The recurrence is ~9 FLOPs per state element per token (`S·k`, the update,
`S·q`) × 48 × 128 × 128. It is under 1% of the layer's FLOPs. Its cost is
latency instead: step `t` cannot start until step `t−1` is done.

### The whole model

| | count | parameters | bytes | FLOPs / token |
|---|---:|---:|---:|---:|
| `embed_tokens` | 1 | 1.27 B | 715 MB | 0 (a lookup: only the `L` rows used are read) |
| full-attention layers | 16 | 5.96 B | 3.35 GB | 11.9 G + 0.39 M × S |
| Gated DeltaNet layers | 48 | 18.39 B | 10.39 GB | 37.1 G |
| final `norm` | 1 | 5,120 | 10 KB | ~20 K |
| `lm_head` | 1 | 1.27 B | 715 MB | 2.54 G (last position only) |
| **text model** | | **26.89 B** | **15.17 GB** | **51.6 G + 0.39 M × S** |
| MTP drafter (bf16, speculative decoding only) | 1 | 0.42 B | 0.85 GB | 0.85 G per drafted token, plus `lm_head` |
| vision tower (first picture only) | 1 | — | 0.46 GB | per image |

The loader's "16.02 GB" is the text model plus the MTP head. One forward pass
**reads 14.45 GB**: every layer and `lm_head`, but not the embedding table, of
which only the rows for this pass's tokens are gathered.

Grouped by what does the work:

| | FLOPs / token | share |
|---|---:|---:|
| MLP × 64 | 34.2 G | 66 % |
| DeltaNet projections × 48 | 11.1 G | 22 % |
| attention projections × 16 | 3.4 G | 7 % |
| `lm_head` (decode; prefill runs it once per window) | 2.5 G | 5 % |
| DeltaNet recurrence × 48 | 0.34 G | 0.7 % |
| attention scores × 16, at 1K context | 0.39 G | 0.8 % |

---

## 4. The KV cache

### What problem it solves

To produce token 1025, each attention layer compares it with **every earlier
token**: `scores = q · Kᵀ` over all 1025 positions. So the layer needs the key
and value vectors of tokens 0 … 1024.

Those vectors **never change**. Attention is causal, so token 37's key at layer 7
depends only on tokens 0 … 37, and nothing generated later can alter it. Without
a cache, every new token would push the whole prefix through all 64 layers again
just to rebuild `K` and `V`: a 1025-token pass to get one token, then a
1026-token pass for the next.

The KV cache computes each token's `k` and `v` once and keeps them:

```
prefill:    k, v for tokens 0..1023   ──write──▶  cache slots [0, 1024)
decode 1:   k, v for token 1024       ──write──▶  cache slot  1024
            attention reads           ◀──────────  cache slots [0, 1025)
decode 2:   k, v for token 1025       ──write──▶  cache slot  1025
            attention reads           ◀──────────  cache slots [0, 1026)
```

In the code that is `KVCache::update_and_fetch`: write the new `k, v` at offset
`P`, return the slice `[0, P + L)`.

### How big it is

```
one token, one full-attention layer:  2 (k and v) × 4 heads × 256 dims × 2 bytes = 4 KB
× 16 full-attention layers                                                       = 64 KB per token
```

| context | KV cache | vs the 14.45 GB read per pass |
|---:|---:|---:|
| 1 K | 67 MB | 0.5 % |
| 10 K | 655 MB | 4.5 % |
| 100 K | 6.6 GB | 45 % |
| 256 K | 17 GB | more than the weights |

Two architectural choices keep it this small:

- **Grouped-query attention.** 4 key/value heads serve 24 query heads, so the
  cache is 6× smaller than with one key/value head per query head.
- **Only 16 of 64 layers have one.** The 48 DeltaNet layers keep a fixed state
  instead — a 128×128 matrix per head (3.1 MB per layer) plus the conv cache's
  last 3 rows (60 KB) — about 151 MB in all, at 10 tokens or at 200K. If all 64
  layers were full attention, the cache would be 4× bigger, and 256K context
  would need ~69 GB for it alone.

### What it means for the code

- **It is pre-allocated** in blocks of 256 tokens. Concatenating on every decode
  step would copy the whole cache every token; instead, new entries are written
  in place.
- **It rolls back by truncation.** When speculative decoding rejects drafted
  tokens, `trim()` moves the offset back, and the rejected entries are
  overwritten later. The DeltaNet state cannot be rolled back like that —
  `S ← α(I − βkkᵀ)S + βkvᵀ` destroys information about the old `S` — so
  `DeltaCache` keeps the state after every drafted token and `commit(n)` picks
  the one to keep.
- **It is what makes multi-turn cheap.** A follow-up message prefills only its
  own tokens; the conversation before it is already cached. Serve mode keeps the
  cache across requests for the same reason.
- **Decode reads it on every token**, on top of the weights: 64 KB × context.

---

## 5. One pass, prefill against decode

Same model, two passes. Prefill: a 1024-token prompt in one window. Then the
first decode step.

| | prefill (`L = 1024`, `P = 0`) | decode (`L = 1`, `P = 1024`) |
|---|---|---|
| `x` | (1, 1024, 5120) | (1, 1, 5120) |
| attention `q` | (1, 24, 1024, 256) | (1, 24, 1, 256) |
| new `k, v` computed | for 1024 tokens | for **1** token |
| `K, V` used | (1, 4, 1024, 256) | (1, 4, **1025**, 256): 1024 cached + 1 new |
| scores | (1, 24, 1024, 1024) | (1, 24, 1, 1025) |
| DeltaNet state | stepped 1024 times | stepped once |
| every projection, every MLP | matrix × **matrix** (1024 columns) | matrix × **vector** |
| `lm_head` | last position only | the one position |

---

## 6. Why prefill reads the weights once, and decode once per token

**Decode generates one token at a time because it has to.** Token 1025 cannot be
computed until token 1024 is known, and token 1024 is the output of the previous
pass. So each pass has one vector to push through each weight matrix:

```
weight (17408 × 5120)  ×  one vector (5120)  =  (17408)
```

Each weight is fetched from memory, used for one multiply-add, and not used
again until the next pass fetches it again. Per token that is the full 14.45 GB:

```
14.45 GB ÷ 400 GB/s = 36 ms per token at the very least  →  ≤ 28 tok/s
measured: ~20 tok/s plain decode
```

The arithmetic units are mostly idle, waiting on memory. Decode is
**memory-bound**.

**Prefill already has every token of the prompt.** Nothing waits on the model's
output, so all `L` tokens go through together: every token through layer 0, then
every token through layer 1, and so on. Each projection becomes a matrix × matrix
product:

```
weight (17408 × 5120)  ×  1024 vectors (5120 × 1024)  =  (17408 × 1024)
```

The GPU brings a tile of weights on-chip and uses it against all 1024 tokens
before moving on. Each weight crosses the memory bus once per window and is used
1024 times. Per token:

| | weight bytes per token | limited by |
|---|---:|---|
| decode | 14.45 GB | memory bandwidth, 400 GB/s |
| prefill, 1024-token window | 14.45 GB / 1024 ≈ 14 MB | arithmetic, ~14.3 TFLOP/s |

The FLOPs per token are the same either way, ~49 G. What batching removes is the
repeated read. Once it is gone, arithmetic is the limit, and the prefill ceiling
comes from the FLOP count instead:

```
48.7 GFLOP/token (no lm_head) ÷ 14.3 TFLOP/s  =  3.4 ms per token  →  ~290 tok/s
48.7 GFLOP/token ÷ 12.9 TFLOP/s (MLX's dense bf16 GEMM)          →  ~265 tok/s
measured: ~228 tok/s at a 2048-token prompt
```

**What about the attention between prompt tokens?** Prompt tokens do depend on
each other — but only through attention, and attention for all `L` positions can
be computed at once, with the causal mask keeping each position from seeing later
ones. No token's *output* is needed as another token's *input*. The DeltaNet
recurrence is the one genuinely sequential step, and it runs on its small
fixed-size state, not on the weights.

**Speculative decoding is this same trick, inside decode.** The MTP head guesses
the next 3 tokens; the full model checks all 4 positions (the known token plus 3
guesses) in one `L = 4` pass. The weights are read once for up to 4 tokens
instead of 4 times, which is how plain decode's ~19 tok/s becomes ~33 tok/s when
most guesses are accepted.

---

## 7. Where the time goes, per token

Decode is priced in bytes, prefill in FLOPs. Floors at 400 GB/s and 14.3 TFLOP/s:

| | decode: bytes → ms | prefill: FLOPs → ms |
|---|---:|---:|
| MLP × 64 | 9.62 GB → 24.1 | 34.2 G → 2.39 |
| DeltaNet projections × 48 | 3.16 GB → 7.9 | 11.1 G → 0.78 |
| attention projections × 16 | 0.94 GB → 2.4 | 3.4 G → 0.23 |
| `lm_head` | 0.72 GB → 1.8 | once per window, ~0 |
| DeltaNet state × 48 | 0.30 GB → 0.75 | serial, ~0.1 measured |
| KV cache / attention scores, 1K context | 0.07 GB → 0.17 | 0.4 G → 0.03 |
| KV cache / attention scores, 100K context | 6.6 GB → 16.4 | 19.7 G → 1.4 |
| **floor at 1K context** | **37 ms → 27 tok/s** | **3.55 ms → 282 tok/s** |
| measured | ~50 ms → 20 tok/s | 4.4 ms → 228 tok/s |

For decode the context row is the cache read for one token at that context. For
prefill it is the per-token average over a prompt of that length: MLX computes
each window's full `(L, S)` rectangle of scores, masked half included, so a
1K-token prompt in one window costs 1K positions per token, and a 100K prompt
averages ~50K.

Prefill reaches ~80% of its floor; the rest is MLX's matmul efficiency (§6) plus
the norms, conv and elementwise ops. Decode reaches ~74%; the rest is mostly the
fixed cost of the many small kernels in a pass, which weighs heavily against a
37 ms budget.
