# How this code was built, step by step

A chronological record of building a from-scratch MLX implementation of
**Qwen3.8-27B** (`model_type: qwen3_5`) — a 27B hybrid linear-attention
vision-language model — on an M3 Max with 48 GB, with no network access.

Including the wrong turns, because those were where the learning was.

```
deepseek_v3.py    634 lines   earlier exercise: DeepSeek-V3 arch at toy scale
tokenizer.py      338 lines   byte-level BPE, stdlib only
quantize.py       126 lines   bf16 -> 4/8-bit, streamed shard by shard
qwen35.py         764 lines   the model
generate.py       121 lines   sampling + streaming
test_model.py     173 lines   16 correctness tests
speculative.py    369 lines   MTP self-speculative decoding
```

---

## Stage 0 — DeepSeek-V3 at toy scale (`deepseek_v3.py`)

The project started as "explain DeepSeek-V3 and implement it minimally". Another
model's answer was provided as a starting point; reviewing it produced the first
lessons.

**Bugs found in the provided answer:**

1. **Causal mask breaks on chunked prefill.** `(1,1,L,L)` mask added to
   `(B,H,L,L+offset)` scores. Works for `offset=0` prefill and `L=1` decode;
   raises on anything else. → the `L` vs context distinction.
2. **Softmax router gating.** V3 uses **sigmoid** + a per-expert correction bias
   + `routed_scaling_factor=2.5`, not softmax.
3. **No group-limited routing, no dense first layers.** V3 partitions 256 experts
   into 8 groups and restricts each token to its best 4; layers 0–2 are dense.
4. **The MoE wasn't sparse.** It ran `expert(x)` for *every* expert and then
   masked — dense compute at 8× the cost, i.e. the exact thing MoE exists to
   avoid. Also `if mx.any(...)` forced a GPU→CPU sync per expert per layer.
5. `-1e9` as a mask value overflows in fp16; `mx.repeat` materializes a broadcast
   that `mx.broadcast_to` gives for free.

**A bug of my own:** `mx.gather_mm` has no VJP with respect to its indices, so
the MoE wouldn't backprop at all until the routing indices were wrapped in
`mx.stop_gradient`. Expert selection is discrete — it *must* be detached. The
gate still learns, through the weights that multiply the expert outputs.

Result: 14.88 M params, 8/8 tests pass (cached decode == full forward, chunked
prefill, causality, group-routing constraint, SwitchGLU vs dense reference,
bias-update direction). Training was left **undertrained** — loss 5.66 → 2.09
over 400 steps, samples are English-shaped mush. Stated as such rather than
dressed up.

---

## Stage 1 — Picking a model that actually fits

Sequence of corrections, all driven by the user:

1. DeepSeek-V3 is 671B. At 1.58-bit it is ~131 GB — will not fit in 48 GB.
2. The smallest model with the *same* MLA + MoE architecture is
   **DeepSeek-V2-Lite** (16B / 2.4B active). Note the R1-Distill models are *not*
   this architecture — they are dense Qwen/Llama finetunes.
3. The user redirected to `qwen3.8:27b`.

**"qwen3.8" postdated my knowledge, and there was no network** — `WebFetch`
returned "Socket is closed", `curl` gave HTTP 000 even outside the sandbox, and
`pip3` hit a proxy 403. I declined to invent an architecture for it. The user
cloned the repo locally instead, which resolved it.

> **Lesson.** When you cannot verify, say so and ask, rather than producing
> plausible code for a model you are guessing at. A wrong architecture produces
> fluent nonsense, not an error.

---

## Stage 2 — Reading the architecture out of the checkpoint

Before writing anything, extract ground truth. Three independent sources:

**`config.json`:**

```
64 layers, hidden 5120, head_dim 256, vocab 248320, 256K context
layer_types = [linear, linear, linear, full] × 16   (full_attention_interval=4)
attn_output_gate: true          partial_rotary_factor: 0.25
mrope_section: [11,11,10]       rope_theta: 1e7
linear_num_key_heads: 16        linear_num_value_heads: 48
mtp_num_hidden_layers: 1        + a SigLIP-style ViT
```

**Tensor shapes**, read straight from the safetensors headers with `struct` +
`json` — no need to load 55 GB to see the layout:

```
layers.0  (linear): linear_attn.{in_proj_qkv[10240,5120], in_proj_z[6144,5120],
                    in_proj_a[48,5120], in_proj_b[48,5120], conv1d[10240,1,4],
                    A_log[48], dt_bias[48], norm[128], out_proj[5120,6144]}
layers.3  (full)  : self_attn.{q_proj[12288,5120], k_proj[1024,5120],
                    v_proj[1024,5120], o_proj[5120,6144], q_norm[256], k_norm[256]}
```

The shapes settle things the config leaves ambiguous:
- `q_proj` is `12288 = 2 × 24 × 256` → **double width**, confirming a per-head
  output gate.
- `10240 = 2048 + 2048 + 6144` → 16 Q heads, 16 K heads, **48** V heads.
- `q_norm/k_norm` are `[256] = head_dim` → per-head QK-RMSNorm.

**The model card** (65 KB `README.md` in the repo, easy to overlook) states the
layout in prose: *"16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention →
FFN))"*, plus "Rotary Position Embedding Dimension: 64" and "MTP: trained with
multiple steps". Every inference above confirmed.

> **Lesson.** Read the weights, not just the config. And read the README — it
> took one `grep` and confirmed the whole architecture.

---

## Stage 3 — `tokenizer.py`

`tokenizer.json` is a GPT-4-style byte-level BPE: NFC normalize → split on a
regex → byte-to-safe-unicode → greedy lowest-rank merges. 248 044 vocab +
33 special tokens.

The obstacle: the split pattern uses `\p{L}`, `\p{N}`, `\p{M}`, which Python's
stdlib `re` cannot do. Rather than add a dependency, `_pretokenize` is a
hand-written scanner over `unicodedata.category()`, with a self-test that diffs
against the `regex` module when it happens to be installed.

Verification: 12/12 round-trip cases (CJK, emoji, accents, CRLF, contractions,
double spaces), exact special-token IDs, and — importantly — **canonical
output**:

```
'The history of computing' -> ['The', 'Ġhistory', 'Ġof', 'Ġcomputing']
```

Round-tripping is necessary but **not sufficient**: a tokenizer can round-trip
perfectly and still produce a different segmentation than the reference, which
would silently inflate perplexity. Inspecting the actual tokens is the real check.

---

## Stage 4 — `quantize.py`

55.6 GB of bf16 will not fit in 48 GB. So: load one shard → quantize the big 2-D
weights → write → free → next. Never more than one shard resident.

Kept in bf16 deliberately: everything 1-D (norms, biases), the depthwise
`conv1d`, and the DeltaNet decay parameters `A_log`, `dt_bias`, `in_proj_a`,
`in_proj_b`. Those feed `exp()`/`softplus()`, where quantization error compounds
through a recurrence — and they total ~0.25 M params, so it costs nothing.

```
55.6 GB bf16 -> 15.9 GB (4-bit, group 64)  in 15 s, peak RAM 5.1 GB
55.6 GB bf16 -> 29.7 GB (8-bit, group 64)  in 18 s
```

The 8-bit build existed purely as a debugging control (see Stage 6).

---

## Stage 5 — `qwen35.py`, first attempt

Written from the extracted spec, with every uncertain convention marked
`[INFERRED]` in the docstring. It loaded with `strict=True` — 1655 tensors, every
name and shape matching — and produced **garbage**:

```
The capital of France is → ��-ﾞ...ﾞ-ﾞ-ﾞ-_dt-меть-
```

`strict=True` is a strong structural check and a useless semantic one.

---

## Stage 6 — Debugging: 14 experiments, one real finding, one dead end

This was the bulk of the work. In order:

| # | experiment | outcome |
|---|---|---|
| 1 | Layer-wise activation stats | Healthy (resid 1.1→5.1, no blowup) but logits *flat* (max 4.59, rms 0.99) → semantic bug, not numerical |
| 2 | Dequantization fidelity | 9.7 % weight error — **normal** for 4-bit affine (theory: $\approx 0.115\sigma / 0.8\sigma$). Not the cause |
| 3 | Ablate each attention type | **Flawed test.** "No attention → garbage" is the *expected* result of removing 64 layers. Proved nothing; I said so and moved on |
| 4 | **Parameter statistics** | `post_attention_layernorm` gammas are *all ≤ 0*, mean −0.217. As a multiplicative gain that would negate and annihilate the block input → these are **zero-centered gammas**, $\hat{x}\cdot(1+w)$ |
| 5 | Perplexity as the objective | Replaced eyeballing with NLL/token. Plain `w`: 13.565. `(1+w)`: **5.14**. First real progress |
| 6 | Sweep norm groups | `+1` on pre-norms, QK-norms, final norm — but **not** the DeltaNet gated norm (whose gammas already center at 0.87) |
| 7 | Grid DeltaNet layout | contiguous `[q|k|v]` vs grouped, repeat-interleave vs tile, gate-then-norm vs norm-then-gate → baseline best |
| 8 | Grid attention | per-head vs global split, rotate_half vs interleaved, sigmoid/silu/none → baseline best |
| 9 | Logit lens by depth | NLL flat at ~9 from layer 0 to 58, *worsening* mid-stack, collapsing to 5.1 only in the last 5 layers |
| 10 | Echo test | Predicting the *current* token scored 6.34 vs 5.14 for the next → **not** an off-by-one |
| 11 | 8-bit vs 4-bit | 6.213 vs 6.253 → quantization **definitively** ruled out |
| 12 | Induction (repeat a sentence) | 2nd copy 8.02 → 2.33 → retrieval partially works |
| 13 | Temperature fit on logits | Optimum scale = 1.0 → not a calibration/scale bug. Top-1 19.6 % (want 45–60 %) |
| 14 | **Joint random search**, 70 of 2048 configs | Still 5.112. Every convention I had chosen was confirmed locally optimal |

At this point I reported being blocked, listed what was ruled out, and asked for
a reference implementation — rather than continuing to guess.

> **Lesson (the big one).** Experiments 7, 8 and 14 all *confirmed* a wrong
> configuration. Coordinate descent gets stuck in local optima, and 70 samples of
> an 11-dimensional space does not reliably break a two-way interaction. Without
> a reference, **a self-consistent local optimum is indistinguishable from the
> truth.**

> **Lesson.** Experiment 3 was a test that could not have informed anything. Say
> so and discard it; don't retrofit a story onto it.

---

## Stage 7 — The reference, and the actual bug

The user cloned **mlx-vlm**, which has `mlx_vlm/models/qwen3_5/`. The bug took
about ten minutes to find:

```python
# mine  — the norm divides the gate's own magnitude back out, so the gate can
#         only rotate the vector, never scale it
rmsnorm(out * silu(z)) * w

# reference — normalize first, THEN gate, so the gate survives
silu(z) * rmsnorm(out, w)
```

Across 48 of 64 layers that was worth **3.2 nats/token**:

| metric | before | after |
|---|---:|---:|
| NLL/token (prose) | 5.112 | **1.872** |
| top-1 next-token | 19.6 % | **56.5 %** |
| "The capital of France is" | " the" | **" Paris"** (17.5, clear top-1) |

Why the searches missed it: norm-then-gate was tested twice and scored *worse*
both times, because RoPE pairing was wrong in the same runs. Entangled variables,
exactly the failure mode above.

The reference also **confirmed** the independently-derived zero-centered gamma
finding — its `NORM_WEIGHT_SUFFIXES` list is exactly the four groups the sweep
identified, and pointedly excludes `linear_attn.norm`.

And it corrected one of my own claims: I had read `style="interleaved"` as the
rotary pairing. It isn't — `_pairing_for_style` maps it to `_HALF_SPLIT`;
"interleaved" describes how M-RoPE assigns the t/h/w *position axes* across
frequency bands. The original rotate_half was right.

> **Lesson.** Two conventions in this model are **silent** — wrong gives fluent
> nonsense, never an error: the zero-centered gammas and the gated-norm order.
> Both are now documented in the code with the reasoning, not just the answer.

---

## Stage 8 — `test_model.py`

Writing tests exposed a measurement problem. The naive assertion
`max|a−b| < 0.05` **failed** on a correct implementation: a 64-layer bf16 stack
differs by ~0.19 in logits between `L=1` and `L=10` purely from matmul tiling.

The fix was to assert what actually matters:

```python
same  = argmax agreement          # must be exact
cos   = cosine similarity         # > 0.9999
drift = per_t[-1] / per_t[0]      # ~1.0 => error does NOT compound
```

`drift` is the discriminating one: rounding is flat across positions, a stale
cache compounds. The observed per-position error was **0.1875 even at t=0**,
where no cache exists — conclusive proof it was precision, not a cache bug.

A second self-inflicted bug: `same == 1.0` is exact, but the display
`f"{same*100:.0f}%"` rounds 99.86 % to "100%". The test reported "100 %" and
failed. Now it counts mismatches explicitly.

16 tests: cache equivalence (3 forms), causality, cache shapes, block-growth
amortization, perplexity, top-1 accuracy, chunked-scan equivalence, 4 factual
probes.

---

## Stage 9 — Pre-allocated KV cache

`mx.concatenate` per decode step reallocates the whole cache: O(n) per token,
O(n²) per generation. Replaced with a buffer over-allocated in blocks of 256,
written in place, plus `offset` as the logical length.

**Honest result: no measurable gain at realistic lengths** — 17.2 vs 17.3 tok/s
over 480 tokens. A 16 MB copy is nothing beside a 15 GB matmul. The win is
asymptotic:

| ctx | prealloc | concat | penalty |
|---|---:|---:|---:|
| 8 K | 0.94 ms | 9.59 ms | 8.7 ms |
| 32 K | 2.34 ms | 37.6 ms | 35 ms |
| 128 K | 8.79 ms | 177 ms | **168 ms** |

A decode step is ~58 ms, so concat would be a ~3× slowdown at 128 K — worth
fixing for a model advertising 256 K, even though current prompts won't notice.

---

## Stage 10 — Chunked prefill scan

The recurrence $S_t = \alpha_t(I - \beta_t kk^{\top})S_{t-1} + \beta_t kv^{\top}$
is **affine in $S_{t-1}$**, so within a chunk of C=64 it unrolls into matmuls.
The $(I - \beta kk^{\top})$ factors compose
into a unit lower-triangular matrix needing one batched C×C triangular solve per
chunk. Only the chunk-to-chunk handoff stays sequential: **L/64 steps instead of
L**.

Taken from the reference rather than invented: the triangular inverse uses
forward substitution, *not* a log-depth Neumann/squaring inverse, which overflows
because correlated keys give `KKᵀ` off-diagonals of order 1.

Verified against the per-token recurrence at L = 1, 5, 16, 63, 64, 65, 130, 200
(rel-err ≤ 1.8e-6 in fp32) — the boundaries 63/64/65 are where an off-by-one
would hide.

**Result: ~2× end-to-end prefill**, consistently across 256–4096 tokens. *Not*
the ~20× the reference cites for the scan in isolation — measured in isolation it
*is* 6–9× (0.284 s → 0.031 s per layer at L=2048), but the other ~10 s is MLP and
attention matmuls this doesn't touch. Amdahl, not a bad implementation.

Two further fixes found here:
- **`lm_head` on the last position only.** Projecting all L positions through a
  248320-wide head materialized a 1 GB tensor at L=2048 for nothing.
- **Windowed prefill.** Peak memory scales with the window, not the prompt.

| prefill_step | peak | prefill | decode |
|---|---:|---:|---:|
| 4096 (one shot) | 32.8 GB | 166 tok/s | **0.6 tok/s** |
| 1024 | 21.3 GB | 188 tok/s | 17.1 tok/s |

That 0.6 tok/s is real: 32.8 GB on a 48 GB machine pushed decode into swap.

> **Mistake worth recording.** I first reported a 0.8× *slowdown* at 2048 tokens.
> That was my benchmark's fault — running both variants in one process exceeded
> memory and caused swapping. Benchmark in **separate processes**. I also
> initially blamed the triangular inverse; measured, it was 0.8 s of 31 s.
> Profile before optimizing.

---

## Stage 11 — MTP speculative decoding

`config.json` has `mtp_num_hidden_layers: 1` and the checkpoint has 15 `mtp.*`
tensors: an MTP head (~424 M params, 1.6 % of the model) that takes the target's
final hidden state plus the embedding of token *t+1* and predicts *t+2*.

```
h_t --> pre_fc_norm_hidden ---.
emb(tok t+1) --> pre_fc_norm_embedding -'
        concat -> fc (10240->5120) -> 1 full-attention layer -> norm -> target's lm_head
```

Registering it as a submodule named `mtp` made the checkpoint's `mtp.*` keys land
directly, so strict loading and the quantization predicate worked unchanged.

**The hard part is rollback.** A KV cache rolls back by truncation, but
delta-rule updates *cannot be un-applied* —
$S_t = \alpha(I - \beta kk^{\top})S_{t-1} + \beta kv^{\top}$
destroys information about `S_{t−1}`. Three approaches, tried in order:

1. **Tape + replay** — store `(q,k,v,α,β,conv_input)` (~21 MB) and replay the
   accepted prefix. Correct, but cost **45 ms/round** in kernel launches.
2. **Keep the per-step states** the forward pass already computes. `commit(n)`
   becomes an index lookup. ~750 MB for a 5-token block, zero extra compute.
   Overhead dropped to **~0 ms**. (The reference supports both; it has a Metal
   kernel that makes replay cheap.)
3. **`mx.contiguous` to detach.** Learned from MTPLX — without it, holding
   `states[n-1]` keeps the whole block's graph alive, compounding every round.

Then the user cloned **MTPLX**, which changed the design substantially:

- **Exact rejection sampling** (Leviathan/Chen). Mine was greedy-only. Accept `d`
  with probability $\min\!\big(1,\ p(d)/q(d)\big)$; on rejection emit a draw from
  the normalized residual $\max(0,\ p-q)$. Exact at *any* temperature, and greedy falls
  out as the case where `to_probs` returns a one-hot — one code path.
- **The draft temperature is free.** Since `p` and `q` are derived
  independently, it cannot affect correctness — only acceptance rate.
- **`speculative_output_marginal` as a correctness oracle.** Summing over every
  possible draft token must recover `p` exactly. Now `--selftest`, 200 random
  distributions, exact to 1e-5. This catches the silent failure mode (a missing
  clamp, an unnormalized residual) that would otherwise appear only as a slow
  drift in output style.

**Results:** greedy k=3 → **26.2 vs 18.2 tok/s = 1.44×** (later 1.93× with the
Metal kernel of Stage 12).

> **Correction, found later.** I initially suggested `--draft-temp 0.3` on the
> strength of one noisy single-run sweep. Re-measured properly it is **worse**:
> at a target temp of 0.7, matching the draft temp to the target gives 78 %
> acceptance and 31.8 tok/s, while 0.3 gives 63 % and 28.1. The theory says so
> too — acceptance is $\sum_d \min\big(p(d), q(d)\big) = 1 - \operatorname{TV}(p,q)$,
> maximized when $q \approx p$, so
> sharpening the draft moves it away from the target. The default (draft temp =
> target temp) was always right; the advice was not.

Two findings:

- **Acceptance is prompt-dependent, and my first k-sweep was misleading.** Run on
  a prose prompt it showed k=2–3 optimal and k=8 at 0.63×. Re-run on code and
  structured output, acceptance jumps from 44–55 % to 74–88 % and k=3 gives
  **1.85–1.91×**, with k=4–6 still worth 1.67–1.72×. Each extra verified position
  costs ~18 ms of real matmul FLOPs (see `PERFORMANCE_MODEL.md`), so the useful
  depth is however far the drafter stays accurate — which depends on the text.
  Lesson: **benchmark on the workload you care about**; one prompt is not a sweep.
- **Greedy output is not bit-identical, and that is precision, not a bug.**
  Divergence at token 65 of 124: the target's top-2 were `' strongly'` and
  `' efficiently'`, **both at logit 21.2500 — an exact bf16 tie**. Tie-breaking
  depends on reduction order, which differs between an L=1 decode and an L=4
  verify. `--compare` now measures the gap at any divergence and reports which
  case it is, rather than failing.

---

## Stage 12 — A fused Metal kernel for the gated-delta step (`delta_kernel.py`)

### Picking the target by measurement, not intuition

The obvious target looked wrong at first. Attribution by **ablation inside the
real model** (replace one component with a no-op, time the difference) gave the
per-extra-token cost as:

| component | share of marginal cost |
|---|---:|
| MLP | 49 % |
| **DeltaNet recurrence** | **28 %** |
| attention | ~0 % |

And for the decode path specifically:

| | recurrence cost | share |
|---|---:|---:|
| L=1 (decode) | 10.4 ms of 53.6 ms | 19.5 % |
| L=8 (verify) | 43.6 ms of 168.8 ms | 25.8 % |
| **bandwidth floor** (state read once, written once) | **0.86 ms** | — |

12x off the floor, on an operation that is 0.6 % of the model's FLOPs. That is a
pure state-traffic problem: ~7 unfused elementwise/reduce ops per step, each
streaming the whole 3.1 MB per-layer state. The MLP's 49 % is already MLX's
quantized matmul and not worth attacking.

> **Two measurement bugs found on the way here.** First, my microbenchmark built
> N identical graphs and only `mx.eval`-ed the last one, so MLX never computed
> the other $N-1$ — it reported 1655 GB/s on a 400 GB/s machine. Second, once
> fixed, per-call `mx.eval` overhead (~0.4 ms) dominated everything at this
> scale. The fix was to stop microbenchmarking and attribute cost by ablation
> inside the real model, where 53 ms >> overhead. An earlier estimate of "the
> recurrence is only 6 % of the cost" came from the broken timer and was wrong.

### The kernel

One thread per `(batch, head, dv)`. Each thread owns state row `S[b,h,v,:]` —
128 contiguous floats — and computes both reductions over its **own** row:

$$
\begin{aligned}
kS[v] \;&=\; \textstyle\sum_{d_k} k[d_k]\, S[v, d_k] \\
S[v, d_k] \;&\leftarrow\; a\big(S[v,d_k] - b\, kS[v]\, k[d_k]\big) + b\, v[v]\, k[d_k] \\
o[v] \;&=\; \textstyle\sum_{d_k} q[d_k]\, S[v, d_k]
\end{aligned}
$$

No cross-thread communication, no threadgroup memory, every thread independent.

That only works with the state laid out **(Dv, Dk)** so the row is contiguous.
Choosing (Dk, Dv) instead would make `kS` a 128-way reduction *across* threads.
This is why both reference implementations store the state that way — a detail
that looked arbitrary until the kernel made the reason obvious. The repo's state
layout was switched to match, which also simplified the MLX path.

For `L > 1` the time loop lives inside the kernel, so the state is read and
written once per **block** rather than once per step. A `collect` variant also
dumps every per-step state for speculative rollback.

### Results

Verified to ~4e-7 relative error against the per-token recurrence across
B/L/H shapes, including the `collect` variant.

48 chained calls (one model's worth of layers):

| L | MLX seq | MLX chunked | **Metal** | speedup |
|---|---:|---:|---:|---:|
| 1 | 6.7 ms | — | **2.4 ms** | 2.75× |
| 4 | 20.3 ms | — | **2.4 ms** | 8.6× |
| 64 | 191 ms | 133 ms | **20.1 ms** | 6.6× vs chunked |
| 256 | — | 138 ms | **88 ms** | 1.6× |
| 512 | — | **164 ms** | 190 ms | chunked wins |

Crossover at $L \approx 384$: the kernel is serial in time (memory-optimal, no
time-parallelism), while the chunked scan turns time into matmuls. So the model
now picks the kernel below 384 and the chunked scan above it.

End-to-end, A/B on the same build (`USE_METAL_DELTA` toggles it):

| | MLX | Metal |
|---|---:|---:|
| decode | 18.8 tok/s | **19.7 tok/s** (+4.8 %) |
| speculative k=3 (code) | 33.4 tok/s | **36.7 tok/s** (+10 %) |
| speculative speedup | 1.85× | **1.93×** |
| prefill | 183 tok/s | 183 tok/s (unchanged — uses the chunked path) |

The gain is larger for speculation than for decode because the verify pass runs
at L=k+1, where the kernel is 8.6× rather than 2.75×.

### A free side-effect: the prefill window

Since the kernel wins below L=384, the prefill window was retuned to 384. Prefill
throughput is flat across windows (190–198 tok/s) because it is MLP-compute-bound
— but peak memory is not:

| window | peak | tok/s |
|---|---:|---:|
| 384 | **16.9 GB** | 198 |
| 1024 | 21.4 GB | 193 |
| 2048 | 26.3 GB | 174 |

So the smaller window is free in time and saves 4.5 GB.

---

## Cross-cutting lessons

1. **Get an objective metric early.** Fourteen experiments became tractable only
   once NLL/token replaced eyeballing samples. "Looks like English" is not a
   measurement.
2. **A self-consistent local optimum looks exactly like the truth.** Grid search
   confirmed a wrong config three separate times. Reference implementations are
   not optional for reverse-engineering.
3. **Read the weights.** The zero-centered gamma discovery came from
   `mean(post_attention_layernorm) = −0.217`, nothing else.
4. **Test what matters, not what's easy to write.** `max|a−b|` fails on correct
   bf16 code; argmax + cosine + drift does not.
5. **Profile before optimizing.** I blamed the triangular inverse (0.8 s of 31 s)
   and considered optimizing `to_probs` (0.25 ms). Both wrong; measured both.
6. **Benchmark in separate processes.** A contaminated benchmark reported a 2×
   slowdown that was actually swapping.
7. **Distinguish the three bottleneck regimes** — memory, compute, launch. Each
   has a different fix and a different diagnostic. See `PERFORMANCE_MODEL.md`.
8. **Silent bugs are the dangerous ones.** Every real bug here (mask shape, gamma
   convention, gated-norm order, missing `stop_gradient`) produced plausible
   output rather than an error.

---

## Not implemented

- **Vision tower** (499 tensors, loaded-but-skipped) — text-only despite being a
  VLM. Needs the ViT, patch embedding, and the merger.
- **True M-RoPE for images.** For text, all three position axes are equal so
  M-RoPE reduces *exactly* to standard RoPE. Images break that equality.
- **YaRN scaling** for context beyond 262 144.
- **More Metal kernels.** The gated-delta step now has one (Stage 12). The
  references have ~99, including fused SwiGLU MLP, 2-pass paged SDPA, and MoE
  paths. The MLP is 49 % of the marginal cost and still uses MLX's stock
  quantized matmul.
- **Reclaimable disk:** `Qwen3.8-27B/.git/lfs` holds a duplicate ~52 GB, and the
  28 GB 8-bit build was only a debugging control.
