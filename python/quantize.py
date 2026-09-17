"""
Stream the bf16 Qwen3.5-27B checkpoint into a 4-bit MLX checkpoint.

55.6 GB of bf16 will not fit in 48 GB of unified memory, so we never hold more
than one shard at a time: load shard -> quantize the big 2-D weights -> write
-> free -> next shard.

What gets quantized: every large 2-D matmul weight (all the *_proj, embeddings,
lm_head, vision qkv/mlp). What stays bf16: everything 1-D (norms, biases), the
depthwise conv1d, and the Gated-DeltaNet decay parameters (A_log, dt_bias,
in_proj_a, in_proj_b). Those last ones feed exp()/softplus(), where 4-bit error
compounds through the recurrence -- they are only ~0.25M params, so keeping them
in bf16 costs nothing and removes a whole class of risk.

Checkpoints live at the repository root, one level above this file. Bare names are
resolved there, so these work from either directory:

    python3 quantize.py Qwen3.8-27B qwen3.5-27b-4bit
    python3 quantize.py Qwen3.8-27B qwen3.5-27b-8bit --bits 8
    python3 quantize.py Qwen3.8-27B qwen3.5-27b-4bit-mtpbf16 --keep-mtp-bf16
"""

import argparse
import json
import os
import shutil
import time

import mlx.core as mx

import qwen35

# The repository root, where the checkpoint directories live.
_CHECKPOINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# substrings that force a tensor to stay in bf16
KEEP_FP = (
    "norm",          # all RMSNorm / LayerNorm weights and biases
    "conv1d",        # depthwise causal conv, tiny and precision-sensitive
    "A_log",         # SSM decay: goes through -exp()
    "dt_bias",       # SSM timestep bias: goes through softplus()
    "in_proj_a",     # per-head decay projection
    "in_proj_b",     # per-head delta-rule beta projection
    "pos_embed",     # ViT position table
    "patch_embed",   # 5-D conv kernel
)

# The MTP draft head (8 modules: mtp.fc plus mtp.layers.0's 4 attention and 3 MLP
# projections, ~424M params). The reference ships a "bf16 MTP sidecar" -- it never
# quantizes the drafter, on the theory that its job is to GUESS well and
# quantization error there costs acceptance rate on every round.
#
# MEASURED on a coding prompt at k=3: keeping it bf16 moved acceptance 87% -> 90%
# and the speedup 1.86x -> 1.87x, i.e. noise, for +0.61 GB. So it is OFF by
# default here. Kept as a flag because the effect may differ on other prompts.
KEEP_FP_MTP = ("mtp.",)


def should_quantize(name: str, arr: mx.array, group_size: int,
                    keep_mtp: bool) -> bool:
    if arr.ndim != 2:
        return False
    skip = KEEP_FP + (KEEP_FP_MTP if keep_mtp else ())
    if any(s in name for s in skip):
        return False
    if arr.shape[-1] % group_size != 0:
        return False
    if arr.size < 1 << 20:  # leave anything under ~1M params alone
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="Qwen3.8-27B")
    ap.add_argument("dst", nargs="?", default="qwen3.5-27b-4bit")
    ap.add_argument("--bits", type=int, default=4, choices=(2, 3, 4, 6, 8))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--keep-mtp-bf16", action="store_true",
                    help="leave the 8 MTP draft-head modules in bf16 (+0.61 GB; "
                         "measured worth ~0 for acceptance -- see KEEP_FP_MTP)")
    args = ap.parse_args()          # not `a`: the shard loop below uses `a` for arrays
    # `src` must already exist, so resolve it; `dst` is being created, so place a
    # bare name next to the other checkpoints rather than inside python/.
    src = qwen35.find_model(args.src)
    dst = args.dst if os.path.isabs(args.dst) or os.sep in args.dst \
        else os.path.join(_CHECKPOINT_ROOT, args.dst)
    BITS, GROUP_SIZE, KEEP_MTP = args.bits, args.group_size, args.keep_mtp_bf16
    os.makedirs(dst, exist_ok=True)

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    print(f"{len(shards)} shards, {index['metadata']['total_size'] / 1e9:.1f} GB bf16")

    new_map, quantized_names = {}, []
    in_bytes = out_bytes = 0
    t0 = time.time()

    for si, shard in enumerate(shards, 1):
        w = mx.load(os.path.join(src, shard))
        out = {}
        for name in sorted(w):
            a = w[name]
            in_bytes += a.nbytes
            if should_quantize(name, a, GROUP_SIZE, KEEP_MTP):
                wq, scales, biases = mx.quantize(a, group_size=GROUP_SIZE, bits=BITS)
                out[name] = wq
                out[name.removesuffix(".weight") + ".scales"] = scales
                out[name.removesuffix(".weight") + ".biases"] = biases
                quantized_names.append(name.removesuffix(".weight"))
            else:
                out[name] = a.astype(mx.bfloat16)
        mx.eval(out)

        path = os.path.join(dst, shard)
        mx.save_safetensors(path, out)
        for k in out:
            new_map[k] = shard
            out_bytes += out[k].nbytes

        del w, out
        mx.clear_cache()
        print(f"  [{si:2d}/{len(shards)}] {shard}  "
              f"-> {os.path.getsize(path) / 1e9:5.2f} GB  "
              f"({time.time() - t0:5.1f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB)")

    json.dump(
        {"metadata": {"total_size": out_bytes}, "weight_map": new_map},
        open(os.path.join(dst, "model.safetensors.index.json"), "w"),
        indent=2,
    )

    # carry over everything the model/tokenizer needs, and record the quant recipe
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["quantization"] = {
        "group_size": GROUP_SIZE,
        "bits": BITS,
        # explicit list so the loader knows exactly which modules to swap
        "quantized_modules": sorted(set(quantized_names)),
        "mtp_kept_bf16": KEEP_MTP,
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "preprocessor_config.json", "chat_template.jinja", "vocab.json",
              "merges.txt"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, dst)

    print(f"\n{in_bytes / 1e9:.1f} GB bf16 -> {out_bytes / 1e9:.1f} GB "
          f"({BITS}-bit, group {GROUP_SIZE})  =  {in_bytes / out_bytes:.2f}x smaller")
    print(f"{len(set(quantized_names))} modules quantized, "
          f"{len(new_map) - 3 * len(set(quantized_names))} tensors left in bf16"
          f"{'  (MTP head kept bf16)' if KEEP_MTP else ''}")
    print(f"wrote {dst}/  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
