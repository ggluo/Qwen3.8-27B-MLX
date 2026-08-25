# [Qwen3.8-27B in MLX](https://huggingface.co/Qwen/Qwen3.8-27B)

A from-scratch [MLX](https://github.com/ml-explore/mlx) implementation of
**Qwen3.8-27B** (`model_type: qwen3_5`) — a 27B hybrid linear-attention
vision-language model — with text-only decoding on Apple Silicon.

No Transformers, no `tokenizers`, no network. The model, the byte-level BPE
tokenizer, the quantizer, and the speculative decoder are all implemented here,
verified tensor-by-tensor against the checkpoint. Built and measured on an M3
Max with 48 GB of unified memory.

## Why this model is interesting

Qwen3.8-27B mixes two layer types in a repeating `3×linear + 1×full` pattern:

| | Full attention (16 layers) | Gated DeltaNet (48 layers) |
|---|---|---|
| State per token | KV cache, O(context) | fixed 128×128 matrix per head, O(1) |
| Heads | GQA 24 Q / 4 KV, head_dim 256 | 16 K heads → 48 V heads, dim 128 |
| Distinguishing detail | sigmoid output gate, per-head QK-RMSNorm, RoPE on 64/256 dims | gated delta rule + depthwise conv + swish gate |

Because three-quarters of the layers are linear attention with **constant**
memory per token (a fixed `128×128` state matrix per head instead of a growing
KV cache), the model advertises a 256K-token context at an affordable cost.

The gated delta rule updates the state with a forget gate and a write strength:

```
S_t = a_t (I − b_t k_t k_tᵀ) S_{t−1} + b_t k_t v_tᵀ
o_t = S_tᵀ q_t
```

where `a_t ∈ (0,1)` is the decay and `b_t = sigmoid(b)` is the delta-rule write
strength.

## Highlights

- **Hand-rolled, verified architecture.** Every tensor name/shape checked against
  the checkpoint; every convention checked against the reference implementation.
  Two subtleties that are otherwise silent (fluent nonsense, never an error) are
  documented in `qwen35.py`: the zero-centered RMSNorm gammas and the
  normalize-then-gate order in the gated RMSNorm.
- **4-bit / 2-bit quantization** streamed shard-by-shard (55.6 GB of bf16 never
  fits in memory at once), with the precision-sensitive decay parameters kept in
  bf16.
- **Self-speculative decoding** driven by the model's own multi-token-prediction
  (MTP) head, with exact rejection sampling — lossless in exact arithmetic.
- **Fused Metal kernels** for the two hot spots that stock MLX leaves on the
  table: the gated-delta recurrence and small-`L` quantized mat-vecs.
- **Correctness harness** (`test_model.py`) covering the hard part — the two
  kinds of recurrent state in a hybrid model — plus a numerical proof of
  speculative-decoding soundness (`--selftest`).

## Repository layout

| File | Purpose |
|---|---|
| `qwen35.py` | The model: config, RMSNorm/GatedRMSNorm, RoPE, full attention, GatedDeltaNet, MTP draft head, KV/Delta caches, chunked + sequential delta rule |
| `tokenizer.py` | Byte-level BPE tokenizer, stdlib-only (hand-rolled GPT-4 regex split) |
| `generate.py` | Chat/raw text generation with streaming |
| `quantize.py` | bf16 → 4/8-bit, streamed shard by shard |
| `speculative.py` | MTP self-speculative decoding with exact rejection sampling |
| `delta_kernel.py` | Fused Metal kernel for the gated-delta recurrence |
| `mlp_kernel.py` | Batched quantized mat-vec Metal kernel for small `L` |
| `test_model.py` | 16 correctness tests (cache, causality, quality, facts) |
| `BUILD_LOG.md` | Chronological record of how this was built, wrong turns included |
| `PERFORMANCE_MODEL.md` | Measured FLOPs/token and memory-wall analysis |

## Requirements

- Apple Silicon Mac (Metal GPU required for the custom kernels; the model runs
  on CPU without them, just slower)
- `mlx` (`pip install mlx`)
- A quantized checkpoint directory (see below). The unquantized bf16 checkpoint
  is ~55.6 GB and will not fit in 48 GB.

## Quick start

```bash
# chat (with the reasoning template)
python generate.py "Explain what a KV cache is." --model qwen3.8-27b-4bit -n 400

# raw completion, no chat template
python generate.py --raw "The capital of France is" --model qwen3.8-27b-4bit --temp 0 -n 20

# speculative decoding (MTP head, exact rejection sampling)
python speculative.py "Explain what a KV cache is." --model qwen3.8-27b-4bit -k 3 -n 200
```

Common flags (both scripts): `--temp`, `--top-p`, `--top-k`, `-n/--max-tokens`,
`--seed`, `--model`.

## Quantizing a checkpoint

```bash
python quantize.py Qwen3.8-27B qwen3.8-27b-4bit            # 4-bit, group 64 (default)
python quantize.py Qwen3.8-27B qwen3.8-27b-2bit --bits 2   # 2-bit
python quantize.py Qwen3.8-27B qwen3.8-27b-8bit --bits 8   # 8-bit
```

`quantize.py` streams one safetensors shard at a time and quantizes only the
large 2-D matmul weights. Kept in bf16: all norms/biases, the depthwise conv,
and the Gated-DeltaNet decay parameters (`A_log`, `dt_bias`, `in_proj_a`,
`in_proj_b`) — those feed `exp()`/`softplus()`, where 4-bit error would compound
through the recurrence.

| Precision | On-disk size |
|---|---|
| bf16 (source) | ~55.6 GB |
| 4-bit | 15.9 GB |
| 2-bit | 12.4 GB |

## Speculative decoding

`speculative.py` uses the model's MTP head (~424M params, 1.6% of the model) as
a drafter:

1. **Draft** — the 1-layer MTP head proposes `k` tokens.
2. **Verify** — the full model scores all `k+1` positions in one pass.
3. **Accept** — Leviathan/Chen rejection sampling: accept `d_i` with probability
   `min(1, p_i(d_i)/q_i(d_i))`.

The output distribution is **identical** to sampling the target directly, at any
temperature / top-p / top-k. The one caveat: in bf16 the verify pass computes
logits at `L=k+1` while plain decode computes them at `L=1`, which differ by
~0.19 from matmul tiling — on an exact top-2 tie the argmax can differ. Use
`--compare` to distinguish that precision artifact from a real bug, and
`--selftest` to prove the marginal numerically:

```bash
python speculative.py --selftest
python speculative.py --raw "The capital of France is" -n 40 --compare
```

The hard part of speculation in a hybrid model is rolling back **delta-rule
state** (KV caches truncate; a linear-attention state cannot be un-applied).
`qwen35.py` handles this by recording per-step states during the verify pass and
picking the accepted one on commit — see `DeltaCache` and `DeltaCache.commit`.

## Tests

```bash
python tokenizer.py          # tokenizer self-test (round-trips, special tokens)
python test_model.py qwen3.8-27b-4bit   # 16 model correctness tests
python delta_kernel.py       # Metal delta kernel vs. the reference recurrence
python mlp_kernel.py         # Metal qmv kernel vs. QuantizedLinear
```

The model tests cover cached decode == full forward, chunked prefill == full
forward, causality (no future leakage), O(1) DeltaNet state, block-wise KV
growth, and a couple of factual probes.

## Performance

The performance story is a memory-wall story, not a FLOP story — see
`PERFORMANCE_MODEL.md` for the full measured model. Briefly:

- The gated-delta recurrence is ~0.6% of the model's FLOPs but ~20% of a decode
  step, because it is dominated by streaming the 151 MB of state. The fused
  Metal kernel (`delta_kernel.py`) reads/writes the state once instead of ~7
  times.
- Stock MLX `quantized_matmul` is at the bandwidth floor at `L=1` but ~3× off
  both roofs at `L=4..8` — exactly the speculative verify range. `mlp_kernel.py`
  fixes that band with an algebraic dequantize-and-accumulate trick.
- Speculative decoding turns a memory-bandwidth-bound decoder into one that
  scores `k+1` positions for barely more than the cost of 1.

## Notes and caveats

- **Text-only.** The vision tower is not loaded (`model.visual.*` tensors are
  skipped), and M-RoPE degenerates to standard RoPE for text tokens.
- The model directories are named `qwen3.8-27b-{4bit,2bit}`; a couple of scripts
  still default their `--model` argument to a legacy `qwen3.5-27b-*` name, so
  pass `--model qwen3.8-27b-4bit` explicitly.
- Everything here was built with no network access; `BUILD_LOG.md` records the
  wrong turns, because those were where the learning was.
