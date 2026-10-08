# [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), from scratch

A from-scratch implementation of **Qwen3.8-27B** (`model_type: qwen3_5`) — a 27B
hybrid linear-attention vision-language model — plus a terminal assistant built on
top of it that reads pictures as well as text. Written twice against
[MLX](https://github.com/ml-explore/mlx): first in Python, then ported to C++,
whose serve mode puts it behind an OpenAI-compatible API.

No Transformers, no `tokenizers`, no PIL, no network. The model, its vision tower,
the byte-level BPE tokenizer, the quantizer, the speculative decoder, the fused
Metal kernel and the line editor are all implemented here and verified against the
checkpoint. Built and measured on an M3 Max with 48 GB of unified memory.

```
python/     the original implementation, and the quantizer      -> python/README.md
cpp/        the port, using MLX's C++ API; no Python at runtime  -> cpp/README.md
```

```bash
python3 python/chat.py          # the assistant, Python
cd cpp && make && ./build/ai    # the assistant, C++
./build/ai --serve              # ... as an OpenAI-compatible server
```

In either chat, `/image photo.jpg` attaches a picture to the next message.

Checkpoint directories live here at the root, so both implementations share them;
`testdata/` holds the small pictures both test suites use.

| Also here | |
|---|---|
| [`MODEL_WALKTHROUGH.md`](MODEL_WALKTHROUGH.md) | One forward pass, module by module: tensor shapes, the cost of each module, the KV cache, and prefill vs decode |
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

with the decay $a_t = \exp\!\big(\!-\!\exp(A_{\log})\cdot\mathrm{softplus}(a + \mathrm{dt\_bias})\big) \in (0,1)$
and the write strength $b_t = \sigma(b)$.

Two conventions in this model are **silent** — get them wrong and you get fluent
nonsense, never an error. Both are documented in the source of both
implementations, with the reasoning:

- **Zero-centered RMSNorm gammas** ($\hat{x}\cdot(1+w)$, not $\hat{x}\cdot w$) for the pre-norms,
  QK-norms and final norm — but *not* the DeltaNet gated norm. Found from the
  weight statistics: `post_attention_layernorm` values are all ≤ 0.
- **Normalize-then-gate** in the gated RMSNorm. Gating inside the norm divides
  the gate's own magnitude back out. Getting this backwards costs 3.2 nats/token.

## Pictures

The checkpoint's 333 `model.visual.*` tensors are Qwen3-VL's vision encoder,
name for name and shape for shape — 27 ViT blocks with a 2-D rotary and a learned
48×48 position table, and a merger that folds each 2×2 block of 16-pixel patches
into one token. Its output replaces the embeddings of `<|image_pad|>` tokens in
the prompt, one per token: a picture costs one token per 32×32 pixels, capped by
default at 1024.

Because the encoder is Qwen3-VL's, transformers is a trustworthy reference for it,
and the Python port is held against it stage by stage: preprocessing gives
identical pixels, the tower agrees to ~2×10⁻⁵ in fp32, and the positions are
identical to `get_rope_index`. The C++ port is then held against the Python —
bit-identical patches, vision features and post-prefill hidden states, and
byte-identical replies about the test pictures.

What changes in the text model is the positions. Image tokens take **M-RoPE**: a
(time, row, column) position each, with the rotary frequencies split between the
three axes, interleaved `THWTHW…`. A picture of $r \times c$ tokens occupies $rc$
slots of the context but only $\max(r, c)$ positions, so after the first picture
the rope position runs ahead of the KV-cache offset by

$$\Delta = \sum_{\text{pictures}} \big(\max(r, c) - rc\big)$$

and every later prefill, decode step, MTP draft and speculative verify has to add
it. For text, M-RoPE reduces exactly to ordinary RoPE (all three axes equal), so a
conversation with no pictures is unchanged.

Two conventions in the tower fail silently, like the text model's: the block MLPs
use the tanh GELU but the merger the exact erf one, and the patches are ordered
2×2-block-major rather than in raster order, which the position table and rotary
coordinates have to follow.

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

- **Pictures, not video.** The tower's temporal patching would take frames, but
  neither port feeds them, nor the timestamps Qwen3-VL puts between them.
- No YaRN scaling, so context beyond the native 262 144 is unsupported.
- A git-LFS clone of the source keeps a second copy in `.git/lfs`; that is
  ~52 GB you can reclaim once you no longer need to re-quantize.
- Everything here was built with no network access. `BUILD_LOG.md` records the
  wrong turns, because those were where the learning was.
