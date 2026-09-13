# [Qwen3.8-27B in MLX](https://huggingface.co/Qwen/Qwen3.8-27B)

A from-scratch [MLX](https://github.com/ml-explore/mlx) implementation of
**Qwen3.8-27B** (`model_type: qwen3_5`) — a 27B hybrid linear-attention
vision-language model — plus a terminal assistant built on top of it.

No Transformers, no `tokenizers`, no network. The model, the byte-level BPE
tokenizer, the quantizer, the speculative decoder, the Metal kernel and the line
editor are all implemented here and verified against the checkpoint. Built and
measured on an M3 Max with 48 GB of unified memory.

---

## The assistant

```bash
python3 chat.py
```

Or install the launcher and just type `ai`:

```bash
ln -s "$PWD/ai" /usr/local/bin/ai
ai
```

```
> Write a short email to my landlord reporting a broken heater. Polite but firm.
Subject: Urgent Maintenance Request: Broken Heater
...
> Add that I'm available any weekday after 4pm.
```

Follow-ups revise the previous answer — the conversation is kept in context.

| command | |
|---|---|
| `/new` | fresh conversation, clears context |
| `/think [on\|off]` | show reasoning (dimmed); off by default for speed |
| `/temp <float>` | sampling temperature (`0` = deterministic; `0.7` default) |
| `/system <text>` | system prompt; applies on the next `/new` |
| `/paste` | type a multi-line message, end with a single `.` line |
| `/stats` | tokens, tok/s, model passes, draft acceptance, context size |
| `/help`, `/exit` | Ctrl-D also quits |

**Editing and paste.** Paste a multi-line paragraph and press Return — it arrives
as a single message. `Ctrl-J` inserts a newline without sending; `↑`/`↓` recall
history; `←`/`→`, `Ctrl-A`/`Ctrl-E`, `Backspace`, `Ctrl-U` edit. `Ctrl-C`
interrupts a reply without quitting.

**Nothing is written to disk.** The conversation, the KV cache and the input
history live in memory only; on exit it prints `conversation discarded`. Verified
by diffing the directory listings of the project and `$HOME` across a session —
no files created in either.

**Per-turn cost stays flat** because the KV cache is reused across turns; only
the new user turn is prefilled. Measured over a growing conversation:

| turn | context before | total time |
|---|---:|---:|
| 1 | 0 tok | 3.5 s |
| 3 | 169 tok | 2.6 s |
| 5 | 308 tok | 2.2 s |

Startup is ~1.5–3 s (mmap of the 4-bit weights), so keep the session open rather
than relaunching per question. Peak memory ~16.3 GB.

### Why paste needed its own module

`input()` cannot read a pasted paragraph. The tty is in canonical mode, so it
hands over a line the moment it sees a newline: pasting two lines auto-submits
the first *before you press Return* and turns the rest into separate turns.
Draining the leftover lines from stdin recovers most cases, but not a paste whose
selection has no trailing newline — the common case when copying a paragraph —
and the wedged final line is **not** retrievable by switching to cbreak (measured).
This Python is backed by libedit, not GNU readline, so `enable-bracketed-paste`
is silently ignored, and `prompt_toolkit` was not installable.

`lineread.py` takes the tty out of line-buffered mode and assembles the paste
itself, using two independent mechanisms: bracketed paste (`\e[200~ … \e[201~`)
where the terminal supports it, and otherwise the fact that Return arrives as CR
while a bare LF can only come from a paste or `Ctrl-J`. **Pasting never submits
on its own — you press Return**, the same semantic as a modern shell.

---

## Why this model is interesting

Qwen3.8-27B mixes two layer types in a repeating `3×linear + 1×full` pattern:

| | Full attention (16 layers) | Gated DeltaNet (48 layers) |
|---|---|---|
| State per token | KV cache, O(context) | fixed 128×128 matrix per head, O(1) |
| Heads | GQA 24 Q / 4 KV, head_dim 256 | 16 K heads → 48 V heads, dim 128 |
| Distinguishing detail | sigmoid output gate, per-head QK-RMSNorm, RoPE on 64/256 dims | gated delta rule + depthwise conv + swish gate |

Because three-quarters of the layers carry **constant** state per token instead
of a growing KV cache, the model advertises a 256K-token context affordably.

The gated delta rule updates the state with a forget gate and a write strength:

$$
\begin{aligned}
S_t &= a_t\,(I - b_t\,k_t k_t^{\top})\,S_{t-1} \;+\; b_t\,k_t v_t^{\top} \\
o_t &= S_t^{\top} q_t
\end{aligned}
$$

with the decay $a_t = \exp\!\big(\!-\!\exp(A_{\log})\cdot\operatorname{softplus}(a + \mathrm{dt\_bias})\big) \in (0,1)$
and the write strength $b_t = \sigma(b)$.

Two conventions in this model are **silent** — get them wrong and you get fluent
nonsense, never an error. Both are documented in `qwen35.py` with the reasoning:

- **Zero-centered RMSNorm gammas** ($\hat{x}\cdot(1+w)$, not $\hat{x}\cdot w$) for the pre-norms,
  QK-norms and final norm — but *not* the DeltaNet gated norm. Found from the
  weight statistics: `post_attention_layernorm` values are all ≤ 0.
- **Normalize-then-gate** in the gated RMSNorm. Gating inside the norm divides
  the gate's own magnitude back out. Getting this backwards costs 3.2 nats/token.

## Repository layout

| File | Purpose |
|---|---|
| `chat.py` | The assistant: interactive REPL, cache reuse across turns, ephemeral |
| `ai` | Launcher shim (edit the path inside if you move the project) |
| `lineread.py` | Raw-mode line reader with correct multi-line paste handling |
| `qwen35.py` | The model: RMSNorm/GatedRMSNorm, RoPE, gated attention, GatedDeltaNet, MTP head, KV/Delta caches, chunked + sequential delta rule |
| `tokenizer.py` | Byte-level BPE, stdlib only (hand-rolled GPT-4 regex split) |
| `quantize.py` | bf16 → 4/8-bit, streamed shard by shard |
| `generate.py` | One-shot generation with streaming |
| `speculative.py` | MTP self-speculative decoding with exact rejection sampling |
| `delta_kernel.py` | Fused Metal kernel for the gated-delta recurrence (**in use**) |
| `mlp_kernel.py` | Batched quantized mat-vec kernel — a documented **negative result**, not wired in |
| `test_model.py` | 16 model correctness tests |
| `BUILD_LOG.md` | How this was built, wrong turns included |
| `PERFORMANCE_MODEL.md` | Measured FLOPs/token, roofline, and the memory wall |

## Requirements

- Apple Silicon Mac. The Metal kernel needs a GPU; without it the model falls
  back to the pure-MLX path automatically, just slower.
- `mlx` (`pip install mlx`). Nothing else — the rest is stdlib.
- A quantized checkpoint directory (below). The bf16 source is 55.6 GB and will
  not fit in 48 GB.

## Quantizing a checkpoint

```bash
python3 quantize.py Qwen3.8-27B qwen3.5-27b-4bit                    # default
python3 quantize.py Qwen3.8-27B qwen3.5-27b-8bit --bits 8
python3 quantize.py Qwen3.8-27B qwen3.5-27b-4bit-mtpbf16 --keep-mtp-bf16
```

One safetensors shard is resident at a time (peak ~5 GB, ~15 s total). Only the
large 2-D matmul weights are quantized. Kept in bf16: all norms and biases, the
depthwise conv, and the Gated-DeltaNet decay parameters (`A_log`, `dt_bias`,
`in_proj_a`, `in_proj_b`) — those feed `exp()`/`softplus()`, where quantization
error would compound through the recurrence.

| build | on-disk | notes |
|---|---:|---|
| bf16 (source) | 55.6 GB | |
| **4-bit** | **15.86 GB** | the default; every measurement here uses it |
| 8-bit | 29.97 GB | quality control — scored NLL 6.21 vs 4-bit's 6.25 |
| 4-bit, bf16 MTP head | 16.47 GB | `--keep-mtp-bf16`; see below |

`--bits 2` is accepted but untested here — no size or quality number is claimed.

**On `--keep-mtp-bf16`:** the reference runtime ships a "bf16 MTP sidecar",
never quantizing the drafter, on the theory that a drafter's job is to guess well.
Measured on a coding prompt at k=3, it moved acceptance 87% → 90% and the speedup
1.86× → 1.87× for +0.61 GB — i.e. noise. Off by default. `config.json` records
`"mtp_kept_bf16"` so you can tell which recipe built a directory.

## One-shot generation

```bash
python3 generate.py "Explain what a KV cache is." -n 400
python3 generate.py --raw "The capital of France is" --temp 0 -n 20
python3 speculative.py "Write a Python function to merge two sorted lists." -k 3 -n 200
```

Common flags: `--model`, `--temp`, `--top-p`, `--top-k`, `-n/--max-tokens`,
`--seed`, `--no-think`. The default `--model` is `qwen3.5-27b-4bit`; any
directory produced by `quantize.py` works.

## Speculative decoding

`speculative.py` uses the model's own MTP head (~424M params, 1.6% of the model,
8 modules) as the drafter:

1. **Draft** — the 1-layer MTP head proposes `k` tokens from its distribution `q`.
2. **Verify** — the full model scores all `k+1` positions in one pass.
3. **Accept** — Leviathan/Chen rejection sampling: accept `d_i` with probability
   $\min\!\big(1,\ p_i(d_i)/q_i(d_i)\big)$; on the first rejection emit a draw
   from the normalized residual $\max(0,\ p - q)$.

The output distribution is **identical** to sampling the target directly at any
temperature/top-p/top-k — greedy is just the case where the distribution is a
one-hot. `--selftest` proves the marginal numerically (200 random distributions,
exact to 1e-5).

```bash
python3 speculative.py --selftest
python3 speculative.py --raw "The capital of France is" -n 40 --compare
```

**Acceptance is dominated by the prompt, not by `k`.** Measured at k=3:

| task | acceptance | speedup |
|---|---:|---:|
| list the first 15 primes | 88% | **1.91×** |
| code: merge two sorted lists | 86% | 1.85× |
| code: fizzbuzz + tests | 74% | 1.69× |
| prose: why the sky is blue | 55% | 1.40× |
| prose: history of the printing press | 44% | 1.22× |

So expect **~1.9× on code and structured output, ~1.2–1.5× on prose.** Use `k=3`
for code, `k=2` for prose; on prose `k=4` can drop *below* baseline.

**Leave `--draft-temp` alone.** Acceptance is

$$\sum_{d} \min\big(p(d),\ q(d)\big) \;=\; 1 - \operatorname{TV}(p, q)$$

the overlap of the two distributions, so it is maximised when the draft temperature
*matches* the target. Sharpening the draft measurably hurts: at target 0.7,
draft 0.3 gives 63% acceptance versus 78% for the matching default.

Two honest caveats:

- In bf16, greedy output is not bit-identical to plain decoding. The verify pass
  computes logits at `L=k+1` while decode uses `L=1`, and matmul tiling makes
  those differ by ~0.19; on an **exact** top-2 tie the argmax can differ.
  `--compare` measures the gap at any divergence and reports which case it was.
- The hard part is rolling back **delta-rule state** — a KV cache truncates, but
  $S_t = a\,(I - b\,kk^{\top})S_{t-1} + b\,kv^{\top}$ destroys $S_{t-1}$. `DeltaCache` records
  per-step states during verification and indexes the accepted one on commit.

## Performance

This is a memory-wall story, not a FLOP story — `PERFORMANCE_MODEL.md` has the
full measured model: $51.24$ GFLOP/token over $15.4$ GB per pass is an arithmetic
intensity of $3.3$ FLOP/byte, against a machine balance of $9.8$ — so
**memory-bound at $L=1$**.

Measured on an M3 Max, 4-bit:

| | |
|---|---:|
| decode | **19.7 tok/s** (18.8 without the Metal kernel) |
| prefill | ~190 tok/s |
| speculative, code, k=3 | **35–37 tok/s** |
| peak memory | 16.3 GB |
| language modelling | NLL 1.872/token, top-1 56.5% |

**`delta_kernel.py`** is the one custom kernel that pays. The gated-delta
recurrence is only 0.6% of the model's FLOPs but ~20% of a decode step, because
it streams 151 MB of state through ~7 unfused ops — 12× off its bandwidth floor.
One thread per `(batch, head, dv)` keeps both reductions inside a single thread,
which is why the state is laid out `(Dv, Dk)`: the row must be contiguous.
Isolated it is 2.75× at L=1 and 8.6× at L=4; end to end that is **+4.8% decode
and +10% speculative**. Above $L \approx 384$ the chunked scan wins and is used instead.

**`mlp_kernel.py` is a negative result, kept for the record.** The MLP is 63% of
the model's bytes and stock MLX `quantized_matmul` sits in a dead zone at
`L=4..8` (115 GB/s against 361 achievable) — exactly the speculative verify
range. Three iterations (naive → coalesced SIMD-group → N-tiled with the
$\sum_k x_k (q_k s + b) = s\sum_k x_k q_k + b\sum_k x_k$ decomposition and full
unrolling) all lost to MLX at
every `L`. It is numerically correct and self-tested, just slower. The reference
runtime ran the same experiment, specialised to M=3, and did not ship it either.

## Tests

```bash
python3 tokenizer.py        # 12 round-trip cases, exact special-token ids
python3 test_model.py       # 16 model correctness tests
python3 lineread.py         # 7 line-editor tests, driven through a real pty
python3 speculative.py --selftest   # rejection-sampling marginal == target
python3 delta_kernel.py     # Metal kernel vs. the reference recurrence
python3 mlp_kernel.py       # qmv kernel vs. QuantizedLinear (correctness only)
```

The model tests cover the parts that actually break: cached decode == full
forward, chunked prefill == full forward, prefill+incremental == full forward,
causality, O(1) DeltaNet state, block-wise KV growth, chunked scan == per-token
recurrence, perplexity and top-1 accuracy, and four factual probes.

Note the cache tests do **not** assert on `max|a−b|` — a 64-layer bf16 stack
differs by ~0.19 in logits between `L=1` and `L=10` from matmul tiling alone, on
correct code. They assert identical argmax, cosine > 0.9999, and that error does
**not grow with position** (rounding is flat; a stale cache compounds).

## Notes and caveats

- **Text-only.** The vision tower is not loaded (499 `model.visual.*` tensors are
  skipped) despite this being a VLM. For text, M-RoPE degenerates exactly to
  standard RoPE because the time/height/width position axes are all equal.
- No YaRN scaling, so context beyond the native 262 144 is unsupported.
- A git-LFS clone of the source keeps a second copy in `.git/lfs`; that is
  ~52 GB you can reclaim once you no longer need to re-quantize.
- Everything here was built with no network access. `BUILD_LOG.md` records the
  wrong turns, because those were where the learning was.
