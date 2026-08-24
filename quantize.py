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

    python quantize.py Qwen3.8-27B qwen3.5-27b-4bit
"""

import json
import os
import shutil
import sys
import time

import mlx.core as mx

GROUP_SIZE = 64
BITS = int(os.environ.get("QBITS", 4))

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


def should_quantize(name: str, arr: mx.array) -> bool:
    if arr.ndim != 2:
        return False
    if any(s in name for s in KEEP_FP):
        return False
    if arr.shape[-1] % GROUP_SIZE != 0:
        return False
    if arr.size < 1 << 20:  # leave anything under ~1M params alone
        return False
    return True


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.8-27B"
    dst = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5-27b-4bit"
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
            if should_quantize(name, a):
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
          f"{len(new_map) - 3 * len(set(quantized_names))} tensors left in bf16")
    print(f"wrote {dst}/  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
