"""
Batched quantized mat-vec Metal kernel, for the small-L regime.

WHY THIS EXISTS
---------------
MLX's `quantized_matmul` is excellent at L=1 and inefficient at L=4..8. Measured
on 64 chained quantized SwiGLU MLPs (the model's real MLP workload, 10.7 GB of
4-bit weights):

    L    time     GB/s   TFLOP/s
    1    29.6ms    361     1.16     <- at the bandwidth floor (30.6ms). Optimal.
    2    29.8ms    359     2.30     <- still free
    4    47.4ms    226     2.89
    8    92.9ms    115     2.95     <- 3x off BOTH roofs
    32  121.3ms     88     9.03     <- compute bound, efficient again

At L=8 it hits neither roof: 115 GB/s against 361 achievable, and 2.95 TFLOP/s
against 9.03 achievable. It is re-reading the weights roughly 3x, i.e. running a
GEMV per row instead of reusing each weight tile across the L rows.

L=4..8 is exactly the speculative-decoding verify range (L = k+1), so this is
the band that matters for throughput.

THE MAPPING
-----------
Structure follows MTPLX's `verify_qmv.py`, which set out to answer the same
question ("can reusing dequantized weight loads across the verify rows beat
stock quantized_matmul"). Three things matter, and two are not obvious:

1. COALESCING. One SIMD group per output *tile*, lane `i` owning 16 consecutive
   values (2 uint32 packs). The naive mapping -- one thread per output column,
   striding down its own row -- measured 2-4x SLOWER than MLX, because adjacent
   threads then read 2560 bytes apart: every load pulls a 128-byte line and uses
   4 bytes of it.

2. N-TILING. Each SIMD group computes RESULTS_PER_SIMDGROUP=4 output columns, so
   the x values held in registers are reused across 4 weight rows. Without this
   the inner loop is 1 x-load per FMA; with it, 1 per 4.

3. THE ALGEBRAIC DECOMPOSITION, which is the real trick:

       sum_k x[k] * (q[k]*s + b)  =  s * sum_k x[k]*q[k]  +  b * sum_k x[k]

   So accumulate `sum x` alongside the dot product and apply scale and bias ONCE
   PER GROUP instead of once per value. That removes the dequantize entirely from
   the inner loop -- no per-value multiply-add by (s, b).

   Combined with pre-scaling x by 1, 1/16, 1/256, 1/4096, the inner loop can
   multiply by the *unshifted* nibble mask (`packed & 0x00f0`), saving the shift
   too. Inner loop becomes 4 raw masks and 4 FMAs per uint16.

Weight traffic is independent of L -- that is the win over MLX's per-row GEMV.

4-bit layout, verified against mx.dequantize: 8 values per uint32, value i in
nibble i (low nibble first), `value = nibble * scale[group] + bias[group]`,
with 8 words per 64-element group.
"""

from typing import Optional

import mlx.core as mx

_CACHE = {}

_HEADER = """
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;          // 4-bit values per uint32
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;   // 512
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;  // 8

    // Load 16 x values into registers, pre-scaled so the dot product can use
    // raw (unshifted) nibble masks. Also return sum(x) for the bias term.
    inline float load_x16(const device float* x, thread float* xt) {
      float sum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        xt[i]     = x[i];
        xt[i + 1] = x[i + 1] / 16.0f;
        xt[i + 2] = x[i + 2] / 256.0f;
        xt[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    // s * sum(x*q) + b * sum(x)   -- scale/bias applied once, not per value
    inline float qdot16(const device uint8_t* w, const thread float* xt,
                        float s, float b, float sum) {
      const device uint16_t* ws = (const device uint16_t*)w;
      float acc = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
        uint16_t p = ws[i];
        acc += xt[4*i]     * float(p & 0x000f)
             + xt[4*i + 1] * float(p & 0x00f0)
             + xt[4*i + 2] * float(p & 0x0f00)
             + xt[4*i + 3] * float(p & 0xf000);
      }
      return s * acc + b * sum;
    }
"""

def _gen_source(K: int, N: int, L: int) -> str:
    """Emit the kernel with L fully unrolled into NAMED variables.

    Arrays indexed by a loop variable (xt[l], result[l][r]) do not get promoted
    to registers -- they spill to device memory, and the cost explodes: measured
    4059 ms at L=16 versus 120 ms for MLX. MTPLX sidesteps this by hardcoding
    M=3 with x0_thread/x1_thread/x2_thread and result0/1/2. Here the same thing
    is generated for arbitrary L, which keeps everything in registers as long as
    L stays small (16*L + 4*L floats; the budget is ~128).
    """
    decl_x = "\n".join(
        f"    float xt{l}[VALUES_PER_THREAD]; float xsum{l};\n"
        f"    const device float* xp{l} = x + {l} * {K} + int(simd_lid) * VALUES_PER_THREAD;"
        for l in range(L))
    decl_r = "\n".join(
        "    " + " ".join(f"float r{l}_{r} = 0.0f;" for r in range(4))
        for l in range(L))
    load_x = "\n".join(f"      xsum{l} = load_x16(xp{l}, xt{l});" for l in range(L))
    body = []
    for r in range(4):
        body.append(f"      if (out_row + {r} < {N}) {{")
        body.append(f"        const device uint8_t* wl{r} = ws + {r} * row_bytes;")
        body.append(f"        float s{r} = sc[{r} * row_groups];")
        body.append(f"        float b{r} = bs[{r} * row_groups];")
        for l in range(L):
            body.append(f"        r{l}_{r} += qdot16(wl{r}, xt{l}, s{r}, b{r}, xsum{l});")
        body.append("      }")
    adv = "\n".join(f"      xp{l} += BLOCK_SIZE;" for l in range(L))
    store = []
    for r in range(4):
        store.append(f"    if (out_row + {r} < {N}) {{")
        for l in range(L):
            store.append(f"      {{ float t = simd_sum(r{l}_{r});"
                         f" if (simd_lid == 0u) out[{l} * {N} + out_row + {r}] = t; }}")
        store.append("    }")
    return f"""
    uint n_tile   = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int SCALE_STEP = 64 / VALUES_PER_THREAD;
    int out_row    = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int row_bytes  = {K} * BYTES_PER_PACK / PACK_FACTOR;
    int row_groups = {K} / 64;

    const device uint8_t* ws = (const device uint8_t*)wq + out_row * row_bytes
                             + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device float*   sc = scales + out_row * row_groups + int(simd_lid) / SCALE_STEP;
    const device float*   bs = biases + out_row * row_groups + int(simd_lid) / SCALE_STEP;

{decl_x}
{decl_r}

    for (int k = 0; k < {K}; k += BLOCK_SIZE) {{
{load_x}
{chr(10).join(body)}
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / 64;
      bs += BLOCK_SIZE / 64;
{adv}
    }}

{chr(10).join(store)}
"""


def _kernel(K: int, N: int, L: int):
    key = (K, N, L)
    if key in _CACHE:
        return _CACHE[key]
    k = mx.fast.metal_kernel(
        name=f"qmv_batched_{K}_{N}_{L}",
        input_names=["x", "wq", "scales", "biases"],
        output_names=["out"],
        source=_gen_source(K, N, L),
        header=_HEADER,
    )
    _CACHE[key] = k
    return k


def available() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# Only worth using in the band where MLX is off both roofs. Outside it, MLX wins:
# at L<=2 MLX is already at the bandwidth floor, and at large L its GEMM path is
# far better than one-thread-per-column.
L_MIN, L_MAX = 3, 16


def qmv(x: mx.array, ql) -> mx.array:
    """out = x @ dequantize(ql.weight).T  for a QuantizedLinear `ql`.

    x: (..., L, K) -> returns (..., L, N). Only supports group_size=64, bits=4.
    """
    *lead, L, K = x.shape
    N = ql.weight.shape[0]
    xr = mx.contiguous(x.reshape(-1, L, K)[0]).astype(mx.float32)   # (L, K)
    tiles = (N + 7) // 8
    out = _kernel(K, N, L)(
        inputs=[xr, ql.weight, ql.scales.astype(mx.float32),
                ql.biases.astype(mx.float32)],
        grid=(64, tiles, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(L, N)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(*lead, L, N)


def usable(ql, L: int) -> bool:
    return (available() and L_MIN <= L <= L_MAX
            and getattr(ql, "bits", None) == 4
            and getattr(ql, "group_size", None) == 64
            and getattr(ql, "mode", "affine") == "affine"
            and "bias" not in ql)


# ---------------------------------------------------------------------------


def _selftest():
    import mlx.nn as nn
    mx.random.seed(0)
    print(f"metal available: {available()}")
    ok = True
    for (K, N) in [(5120, 17408), (17408, 5120), (5120, 5120), (512, 256)]:
        lin = nn.Linear(K, N, bias=False)
        mx.eval(lin.parameters())
        ql = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
        mx.eval(ql.parameters())
        for L in (1, 3, 4, 8, 16):
            x = mx.random.normal((1, L, K)).astype(mx.bfloat16)
            mx.eval(x)
            ref = ql(x).astype(mx.float32)
            got = qmv(x, ql)
            rel = (mx.abs(got - ref).max() / mx.abs(ref).max()).item()
            good = rel < 2e-3        # bf16 x vs fp32 accumulation
            ok &= good
            print(f"  [{'ok' if good else 'FAIL'}] K={K:5d} N={N:5d} L={L:2d}  "
                  f"rel err {rel:.2e}")
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
