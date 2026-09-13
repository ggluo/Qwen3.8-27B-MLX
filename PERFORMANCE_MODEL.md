# Performance model: `L`, FLOPs/token, and the memory wall

Notes for the Qwen3.8-27B (`model_type: qwen3_5`) implementation in this repo.
Every number here was **measured on this machine** (M3 Max, 48 GB) or computed
from the model's own `config.json` — none of it is quoted from a spec sheet.

---

## 1. What `L` is

`L` is the **sequence length of a single forward pass** — how many token
positions go through the model at once. In the code it is literally:

```python
B, L, _ = x.shape
```

The thing to keep straight: **`L` is not the context length.** They are
independent numbers, and conflating them is the single most common source of
bugs in this kind of code.

| situation | `L` (positions this pass) | context (`cache.offset`) |
|---|---|---|
| prefill a 1000-token prompt | 1000 (or the window, 1024) | grows 0 → 1000 |
| decode one token | **1** | 1000, 1001, 1002 … |
| verify a speculative block | **k+1** (4 when k=3) | 1000 → 1003 |

So in the experiment tables in this project:

- "prefill L=2048" → 2048 positions in one pass
- "verify pass L=5" → k=4 drafted tokens plus the one already-known token
- "cached decode" → L=1, repeatedly

### Why the distinction bites

Position information comes from the **offset**, while the tensor has **`L`**
rows. That is why the causal mask is `(L, L + offset)` and not `(L, L)`:

```python
def causal_mask(L, offset, dtype):
    q = mx.arange(offset, offset + L)[:, None]   # absolute query positions
    k = mx.arange(offset + L)[None, :]           # every key in the cache
    return mx.where(k <= q, 0.0, mx.finfo(dtype).min)
```

The very first bug found in this project (in another model's answer) was exactly
this: a `(L, L)` mask that happens to work for `offset=0` prefill and for `L=1`
decode, and crashes or silently misaligns for anything else. The same confusion
shows up in RoPE — `rope(x, offset, ...)` must be given the offset, not `0`.

---

## 2. FLOPs per token

### The one rule that gets you 95% of the way

A matmul against a weight matrix of shape `(out, in)` costs
$2 \cdot \mathrm{in} \cdot \mathrm{out}$ FLOPs per token — one multiply and one
add per weight. Therefore:

$$\text{FLOPs/token} \;\approx\; 2 \times (\text{number of weights used})$$

That's it. For a dense transformer, forward FLOPs per token is just twice the
parameter count (minus the embedding table, which is a lookup, not a matmul).

### Worked out for this model

From `config.json`: `hidden=5120`, `64` layers, `intermediate=17408`,
`head_dim=256`, `24` Q heads / `4` KV heads, `vocab=248320`, 16 full-attention
layers and 48 DeltaNet layers.

| component | weights | GFLOP/token |
|---|---:|---:|
| MLP × 64 | 17.11 B | **34.23** |
| Attention × 16 | 1.68 B | 3.36 |
| DeltaNet × 48 | 5.56 B | 11.12 |
| `lm_head` | 1.27 B | 2.54 |
| embedding | 1.27 B | 0 (lookup) |
| **total** | **26.89 B** | **51.24** |

Cross-check: `2 × (26.89B − 1.27B) = 51.2 GFLOP` ✓ — and 26.89 B is the "27B"
on the tin.

Per-component derivations:

$$
\begin{aligned}
\text{MLP} \;&=\; 3 H I &&=\; 3 \cdot 5120 \cdot 17408 &&=\; 267\,\mathrm{M} \ \text{per layer} \\
\text{Attention} \;&=\; \underbrace{2 n_q d_h H}_{\texttt{q\_proj}} + \underbrace{2 n_{kv} d_h H}_{\texttt{k,v}} + \underbrace{n_q d_h H}_{\texttt{o\_proj}} &&&&=\; 105\,\mathrm{M} \ \text{per layer} \\
\text{DeltaNet} \;&=\; \underbrace{(2 n_k d_k + n_v d_v)H}_{\texttt{in\_proj\_qkv}} + \underbrace{n_v d_v H}_{\texttt{in\_proj\_z}} + \underbrace{2 n_v H}_{a,\,b} + \underbrace{n_v d_v H}_{\texttt{out\_proj}} &&&&=\; 116\,\mathrm{M} \ \text{per layer} \\
\text{lm\_head} \;&=\; V H &&=\; 248320 \cdot 5120 &&=\; 1.27\,\mathrm{B}
\end{aligned}
$$

Note $\texttt{q\_proj}$ is **double width** — it emits $[\,Q \mid \text{gate}\,]$
per head, hence the factor $2 n_q d_h$.

**The MLP is two-thirds of all the arithmetic.** Attention gets all the
attention; the feed-forward network does the computing.

### Two corrections the rule misses

**(a) Context-dependent attention.** `QK^T` and `AV` use no weights at all, so
they scale with context length `S`, not with parameters:

$$2 \cdot 2 \cdot n_\text{heads} \cdot d_\text{head} \cdot S
\qquad\text{per full-attention layer, per token}$$

| context | GFLOP/token | as % of weight FLOPs |
|---|---:|---:|
| 1 K | 0.39 | 0.8 % |
| 10 K | 3.93 | 7.7 % |
| 100 K | 39.3 | **77 %** |

Negligible at short context, dominant at long. This is precisely why the
architecture has only **16** full-attention layers out of 64 — and why MLA (in
DeepSeek-V3) and Gated DeltaNet (here) exist at all.

**(b) The DeltaNet recurrence.** `0.302` GFLOP/token — **0.6 %** of the total.
Yet measured, it cost ~10 ms of a 200 ms pass. FLOPs are simply the wrong unit
for it. See the launch-bound regime in §3.

---

## 3. Memory-bound vs compute-bound

### The three numbers

1. **Bytes moved** per pass. At 4-bit that is ~**15.4 GB** of weights — read
   *once per pass, independent of `L`*. This is the key asymmetry.
2. **FLOPs** = `51.24 × L` GFLOP.
3. **Arithmetic intensity** $I = \dfrac{\text{FLOPs}}{\text{bytes}}$ (FLOP per byte).

Compare $I$ against the **machine balance**:

$$B = \frac{\text{peak FLOP/s}}{\text{peak bandwidth}}
\qquad\text{if } I < B \text{ you are memory-bound.}$$

### Measured on this machine

```
L = 1        : 15.4 GB / 53 ms  =  291 GB/s     and  0.97 TFLOP/s
each extra L : ~0 GB extra                      and  2.85 TFLOP/s
```

291 GB/s is most of an M3 Max's available bandwidth (300–400 GB/s depending on
variant), while 0.97 TFLOP/s is barely a third of the 2.85 TFLOP/s the *same
chip* reaches once compute is the limit. So:

$$
\begin{aligned}
I &= \frac{51.24\ \mathrm{GFLOP}}{15.4\ \mathrm{GB}} = 3.3\ \text{FLOP/byte} \\[4pt]
B &= \frac{2.85 \times 10^{12}}{291 \times 10^{9}} = 9.8\ \text{FLOP/byte} \\[4pt]
I &\ll B \;\Longrightarrow\; \textbf{memory bound at } L=1
\end{aligned}
$$

At `L=1` the GPU spends most of its time waiting for weights to arrive. The
arithmetic units are ~66 % idle.

### The crossover — and why it explains everything

Compute time catches up with memory time at:

$$L^{*} = \frac{53\ \mathrm{ms}\ \text{(read all the weights)}}
{18\ \mathrm{ms}\ \text{(compute one more position)}} \approx 2.9$$

**This number sets the ceiling for speculative decoding in this repo.** Each
round costs $53 + 18(L-1)$ ms and emits $\text{tok/pass}$ tokens, so the
break-even depth depends entirely on how many drafts get accepted:

$$\text{speedup} \;=\; \frac{\text{tok/pass} \times 53}{53 + 18k}$$

Speculative decoding works **only** in the memory-bound regime: it spends
otherwise-idle compute to avoid re-reading the weights. Once `k` pushes the
verify pass into the compute-bound regime, extra depth costs real time — so the
useful depth is exactly however far the drafter can stay accurate before that.

**Acceptance is dominated by the prompt, not by k.** Measured at k=3, same model,
same sampler:

| task | accept | tok/pass | speedup |
|---|---:|---:|---:|
| list the first 15 primes | 88 % | 3.64 | **1.91×** |
| code: merge two sorted lists | 86 % | 3.55 | 1.85× |
| code: fizzbuzz + tests | 74 % | 3.24 | 1.69× |
| prose: why the sky is blue | 55 % | 2.68 | 1.40× |
| prose: history of the printing press | 44 % | 2.34 | 1.22× |

Predictable, templated text (code, lists, boilerplate) is easy to draft;
open-ended prose is not. On code, depth 4–6 still pays (1.72×, 1.67×); on prose
it does not. This is why MTPLX ships a `tune` step that measures the machine and
why its published acceptance figures (0.961 / 0.879 / 0.816 by depth) are quoted
against a *coding* instrument.

### The practical test (use this, not the arithmetic)

**Double `L` and time it.**

- Time barely changes → **memory-bound**. (Measured here: L=1→2 went 53→59 ms,
  only +11 % for double the work.)
- Time roughly doubles → **compute-bound**.

### A third regime: launch-bound

Worth naming because this project hit it. The DeltaNet recurrence was 0.6 % of
FLOPs and a trivial fraction of bandwidth, yet slow — because it was ~340 tiny
kernel launches per token (7 elementwise/reduce ops × 48 layers), each on a
3.1 MB state.

**Diagnostic:** if *fusing* the ops speeds it up, you were launch-bound.
`mx.compile` gave 1.5× (15 ms → 10 ms at L=9), which neither a bandwidth nor a
FLOP analysis would have predicted.

So the full checklist for "why is this slow":

| symptom | regime | fix |
|---|---|---|
| effective GB/s near peak | memory | quantize, smaller cache, fewer passes |
| effective TFLOP/s near peak | compute | fewer FLOPs, better algorithm |
| neither near peak, many small ops | launch | fuse (`mx.compile`), custom kernel |

**Worked example.** The gated-delta recurrence sat in the third row: 0.6 % of
FLOPs, and its state traffic floor (read once + write once = 302 MB/token) is
0.86 ms, yet it measured **10.4 ms**. Cause: ~7 unfused ops each streaming the
whole state. `mx.compile` recovered 1.5×; a fused Metal kernel
(`delta_kernel.py`) recovered 2.75× at L=1 and 8.6× at L=4, worth +4.8 % on
decode and +10 % on speculative decoding. Neither a bandwidth nor a FLOP count
would have predicted the problem — only the gap between the two did.

---

## 4. Why this framing explains the rest of the project

- **4-bit quantization** gives ~3.5× decode speedup *not* by reducing FLOPs (it
  reduces none) but by cutting bytes 55.6 GB → 15.9 GB. A pure memory-bound win.
  Correspondingly, 8-bit vs 4-bit made **no measurable difference to quality**
  (NLL 6.21 vs 6.25) but a large one to speed.
- **Prefill (L=1024)** is compute-bound. That is why the chunked DeltaNet scan
  bought only **2×** end-to-end even though the scan itself got **6–9×** in
  isolation — Amdahl, with the MLP matmuls as the serial remainder.
- **MLA (DeepSeek-V3)** and **Gated DeltaNet (here)** both attack *bytes*, not
  FLOPs. MLA shrinks the KV cache 71×; DeltaNet replaces it with an O(1) state.
  At batch size 1, bytes are what you pay.
- **The pre-allocated KV cache** showed no gain at 480 tokens (17.2 vs
  17.3 tok/s) but a 168 ms/token penalty avoided at 128 K context — because the
  copy only becomes a meaningful share of the byte budget when the cache
  approaches the weight size.
- **`lm_head` on all positions** wasted a 1 GB tensor at L=2048 for nothing.
  2.54 GFLOP/token × 2048 positions when only the last one is ever sampled.

---

## 5. Quick reference

$$
\begin{aligned}
\text{FLOPs/token} &\approx 2 \times (\text{params} - \text{embedding}) \\
\text{+ attention} &\approx 2 \cdot 2 \cdot n_\text{heads} \cdot d_\text{head}
                       \cdot \text{context} \cdot n_\text{full layers} \\
\text{bytes/pass} &\approx \text{model size on disk (weights dominate at batch 1)} \\
I &= \text{FLOPs} \,/\, \text{bytes} \\
B &= \text{peak FLOP/s} \,/\, \text{peak GB/s} \\
L^{*} &= \frac{\text{time to read the weights}}{\text{time to compute one position}}
\end{aligned}
$$

$$I < B \;\Rightarrow\; \text{memory bound} \;\Rightarrow\;
\text{speculation and quantization help}$$

$$I > B \;\Rightarrow\; \text{compute bound} \;\Rightarrow\;
\text{only fewer FLOPs help}$$

For this model on this machine: **51.24 GFLOP/token, 15.4 GB/pass,
$I = 3.3$, $B = 9.8$, speculation ceiling $L^{*} \approx 2.9$.**
