# [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), from scratch, twice

A from-scratch implementation of **Qwen3.8-27B** (`model_type: qwen3_5`) — a 27B
hybrid linear-attention vision-language model — plus a terminal assistant built on
top of it. Written twice against [MLX](https://github.com/ml-explore/mlx): first
in Python, then ported to C++.

No Transformers, no `tokenizers`, no network. The model, the byte-level BPE
tokenizer, the quantizer, the speculative decoder, the fused Metal kernel and the
line editor are all implemented here and verified against the checkpoint. Built
and measured on an M3 Max with 48 GB of unified memory.

```
python/     the original implementation, and the quantizer      -> python/README.md
cpp/        the port, using MLX's C++ API; no Python at runtime  -> cpp/README.md
```

```bash
python3 python/chat.py          # the assistant, Python
cd cpp && make && ./build/ai    # the assistant, C++
```

Checkpoint directories live here at the root, so both implementations share them.

| Also here | |
|---|---|
| [`PERFORMANCE_MODEL.md`](PERFORMANCE_MODEL.md) | Measured FLOPs/token, the roofline, and the memory wall |
| [`BUILD_LOG.md`](BUILD_LOG.md) | How this was built, wrong turns included |

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
nonsense, never an error. Both are documented in the source of both
implementations, with the reasoning:

- **Zero-centered RMSNorm gammas** ($\hat{x}\cdot(1+w)$, not $\hat{x}\cdot w$) for the pre-norms,
  QK-norms and final norm — but *not* the DeltaNet gated norm. Found from the
  weight statistics: `post_attention_layernorm` values are all ≤ 0.
- **Normalize-then-gate** in the gated RMSNorm. Gating inside the norm divides
  the gate's own magnitude back out. Getting this backwards costs 3.2 nats/token.

## Two optimizations that pay

**A fused Metal kernel for the gated-delta recurrence.** It is only 0.6% of the
model's FLOPs but ~20% of a decode step, because it streams 151 MB of state
through ~7 unfused ops — 12× off its bandwidth floor. One thread per
`(batch, head, dv)` keeps both reductions inside a single thread, which is why the
state is laid out `(Dv, Dk)`: the row must be contiguous. Worth +4.8% decode and
+10% speculative, end to end.

**Self-speculative decoding through the model's own MTP head** (~424M params, 1.6%
of the model). It drafts `k` tokens, the full model verifies all `k+1` in one pass,
and Leviathan/Chen rejection sampling keeps the output distribution *identical* to
sampling the target directly. This works because decode is memory-bound: reading
15.9 GB of weights to score `k+1` positions costs barely more than scoring one.
~1.9× on code, ~1.2–1.5× on prose — acceptance is dominated by the prompt, not by
`k`.

The hard part is rollback. A KV cache truncates, but the delta rule destroys
$S_{t-1}$, so the recurrent state cannot be un-applied; the per-step states are
recorded during verification and the accepted one is indexed on commit.

There is also a documented **negative result** — a hand-written fused
quantized-SwiGLU kernel for the MLP, correct but slower than stock MLX at every
`L`, across three iterations. It is kept in `python/mlp_kernel.py` with the
measurements, and deliberately not ported.

## Performance

A memory-wall story, not a FLOP story: $51.24$ GFLOP/token over $15.4$ GB per pass
is an arithmetic intensity of $3.3$ FLOP/byte against a machine balance of $9.8$,
so decode is **memory-bound at $L=1$**. See
[`PERFORMANCE_MODEL.md`](PERFORMANCE_MODEL.md).

M3 Max, 4-bit, identical prompt and settings, separate processes:

| | Python | C++ |
|---|---|---|
| plain decode | 18.2–18.4 tok/s | 19.1–19.4 tok/s |
| speculative, k=3 | 31.1 tok/s | 33.3 tok/s |
| draft accept, tok/pass | 84%, 3.49 | 84%, 3.49 |
| peak memory | 16.3 / 16.9 GB | 16.3 / 16.9 GB |
| prefill | ~176–190 tok/s | ~176 tok/s |

**The two are within ~5%, and that is the expected result.** Identical acceptance,
tok/pass and pass count mean both do precisely the same arithmetic; all the real
work is inside MLX's Metal kernels, and both dispatch the same ones. The host
language only builds and submits the graph — a couple of milliseconds against a
~52 ms token. Rewriting the dispatcher cannot speed up a memory-bound kernel, so
the ~5% is just the Python interpreter's per-token overhead disappearing.

What the port buys is therefore not speed but deployment: a single binary, no
interpreter or MLX wheel at runtime, `make bundle` for a movable tree, and every
tensor's shape declared at load. Also 120 greedy tokens of **byte-identical**
output against the Python, which is the strongest correctness check either
implementation has.

## Requirements

Apple Silicon Mac; `mlx` for the Python side, its headers and `libmlx.dylib` for
the C++ side. A quantized checkpoint — the bf16 source is 55.6 GB and will not fit
in 48 GB, so start with `python/quantize.py` (see
[`python/README.md`](python/README.md)). Without a GPU both fall back to the
pure-MLX path automatically, just slower.

## Notes and caveats

- **Text-only.** The vision tower is not loaded (499 `model.visual.*` tensors are
  skipped) despite this being a VLM. For text, M-RoPE degenerates exactly to
  standard RoPE because the time/height/width position axes are all equal.
- No YaRN scaling, so context beyond the native 262 144 is unsupported.
- A git-LFS clone of the source keeps a second copy in `.git/lfs`; that is
  ~52 GB you can reclaim once you no longer need to re-quantize.
- Everything here was built with no network access. `BUILD_LOG.md` records the
  wrong turns, because those were where the learning was.
