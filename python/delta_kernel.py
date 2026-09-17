"""
Fused Metal kernel for the Qwen3.5 gated-delta recurrence.

WHY THIS EXISTS
---------------
Measured on an M3 Max, the recurrence is 19.5 % of a decode step (10.4 ms of
53.6 ms) and 25.8 % of an L=8 verify pass (43.6 ms of 168.8 ms). It is only
0.302 GFLOP/token -- 0.6 % of the model's arithmetic -- so this is not a compute
problem. It is state traffic:

    state = 48 layers x 48 heads x 128 x 128 x 4 B = 151 MB
    read once + written once per token                = 302 MB  = 0.86 ms @ 350 GB/s
    measured                                          = 10.4 ms  -> 12x off the floor

The unfused version is ~7 elementwise/reduce ops per step, each streaming the
whole 3.1 MB per-layer state through memory. Fusing them into one kernel reads
the state once and writes it once.

THE MAPPING
-----------
One thread per (batch, head, dv). Each thread owns state row S[b,h,v,:] -- 128
contiguous floats -- and computes

    kS[v]     = sum_dk k[dk] * S[v,dk]          (reduction, in-thread)
    S[v,dk]  <- a*(S[v,dk] - b*kS[v]*k[dk]) + b*v[v]*k[dk]
    o[v]      = sum_dk q[dk] * S[v,dk]          (reduction, in-thread)

Both reductions run over the thread's own row, so there is **no cross-thread
communication and no threadgroup memory** -- every thread is independent.

That only works if the state is laid out (Dv, Dk) rather than (Dk, Dv): the
thread's row must be contiguous for coalesced loads. Choosing (Dk, Dv) instead
would make kS a 128-way reduction *across* threads. This is why both reference
implementations store the state as (Dv, Dk).

For L > 1 the loop over time lives inside the kernel, so the state is read and
written once for the whole block rather than once per step.
"""

from typing import List, Optional, Tuple

import mlx.core as mx

_CACHE = {}

_SRC = """
    uint gid = thread_position_in_grid.x;

    const uint DK = {DK};
    const uint DV = {DV};
    const uint H  = {H};
    const uint L  = {L};

    uint vv = gid % DV;
    uint h  = (gid / DV) % H;
    uint b  =  gid / (DV * H);

    // this thread's state row: (b, h, vv, :) over DK, contiguous
    uint sbase = ((b * H + h) * DV + vv) * DK;

    float s[{DK}];
    for (uint d = 0; d < DK; ++d) {{
        s[d] = S_in[sbase + d];
    }}

    for (uint t = 0; t < L; ++t) {{
        uint hb = (b * L + t) * H + h;      // index into (B,L,H)
        uint kb = hb * DK;                  // index into (B,L,H,DK)
        float a  = alpha[hb];
        float bb = beta[hb];
        float vt = v[hb * DV + vv];

        // kS = k . s   (over this thread's own row)
        float kS = 0.0f;
        for (uint d = 0; d < DK; ++d) {{
            kS += k[kb + d] * s[d];
        }}

        // fused state update + output reduction in one pass over the row
        float ov = 0.0f;
        for (uint d = 0; d < DK; ++d) {{
            float kd = k[kb + d];
            float sn = a * (s[d] - bb * kS * kd) + bb * vt * kd;
            s[d] = sn;
            ov += q[kb + d] * sn;
        }}
        o[hb * DV + vv] = ov;
{COLLECT}
    }}

    for (uint d = 0; d < DK; ++d) {{
        S_out[sbase + d] = s[d];
    }}
"""

# when collecting per-step states (speculative rollback), also dump the row
# after every step into (B, H, L, DV, DK)
_COLLECT_BODY = """
        uint cb = (((b * H + h) * L + t) * DV + vv) * DK;
        for (uint d = 0; d < DK; ++d) {{
            S_all[cb + d] = s[d];
        }}
"""


def _kernel(DK: int, DV: int, H: int, L: int, collect: bool):
    key = (DK, DV, H, L, collect)
    if key in _CACHE:
        return _CACHE[key]
    body = _SRC.replace("{COLLECT}", _COLLECT_BODY if collect else "")
    src = body.format(DK=DK, DV=DV, H=H, L=L)
    k = mx.fast.metal_kernel(
        name=f"gated_delta_{DK}_{DV}_{H}_{L}_{int(collect)}",
        input_names=["q", "k", "v", "alpha", "beta", "S_in"],
        output_names=["o", "S_out", "S_all"] if collect else ["o", "S_out"],
        source=src,
    )
    _CACHE[key] = k
    return k


def available() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def gated_delta_metal(q, k, v, alpha, beta, S, collect: bool = False):
    """Drop-in for sequential_delta with the state laid out (B, H, Dv, Dk).

    q, k    : (B, L, H, Dk)   already l2-normalized and head-expanded
    v       : (B, L, H, Dv)
    alpha   : (B, L, H)       forget gate in (0,1)
    beta    : (B, L, H)       delta-rule write strength
    S       : (B, H, Dv, Dk)  or None
    returns : o (B, L, H, Dv), S_out, [states] if collect
    """
    B, L, H, Dk = q.shape
    Dv = v.shape[-1]
    if S is None:
        S = mx.zeros((B, H, Dv, Dk), mx.float32)

    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    alpha = alpha.astype(mx.float32)
    beta = beta.astype(mx.float32)
    S = S.astype(mx.float32)

    nthreads = B * H * Dv
    tg = 128 if nthreads % 128 == 0 else 32
    shapes = [(B, L, H, Dv), (B, H, Dv, Dk)]
    if collect:
        shapes.append((B, H, L, Dv, Dk))

    outs = _kernel(Dk, Dv, H, L, collect)(
        inputs=[q, k, v, alpha, beta, S],
        grid=(nthreads, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=shapes,
        output_dtypes=[mx.float32] * len(shapes),
    )
    if collect:
        o, S_out, S_all = outs
        # match the python path's [S_1 .. S_L] list of (B,H,Dv,Dk)
        return o, S_out, [S_all[:, :, t] for t in range(L)]
    return outs[0], outs[1]


# ---------------------------------------------------------------------------


def _selftest():
    import qwen35
    mx.random.seed(0)
    print(f"metal available: {available()}")
    ok = True
    for (B, L, H, Dk, Dv) in [(1, 1, 48, 128, 128), (1, 4, 48, 128, 128),
                              (1, 9, 48, 128, 128), (2, 3, 16, 128, 128),
                              (1, 64, 48, 128, 128)]:
        q = qwen35.l2norm(mx.random.normal((B, L, H, Dk))) * (Dk ** -0.5)
        k = qwen35.l2norm(mx.random.normal((B, L, H, Dk)))
        v = mx.random.normal((B, L, H, Dv))
        al = mx.random.uniform(0.85, 1.0, (B, L, H))
        bt = mx.random.uniform(0.0, 1.0, (B, L, H))
        S0 = mx.random.normal((B, H, Dv, Dk)) * 0.1
        mx.eval(q, k, v, al, bt, S0)

        ro, rS = qwen35.sequential_delta(q, k, v, al, bt, S0)
        ko, kS = gated_delta_metal(q, k, v, al, bt, S0)
        eo = (mx.abs(ko - ro).max() / mx.abs(ro).max()).item()
        es = (mx.abs(kS - rS).max() / mx.abs(rS).max()).item()
        good = max(eo, es) < 2e-5
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] B={B} L={L:2d} H={H}  "
              f"out {eo:.2e}  state {es:.2e}")

        # collect variant must return the same per-step states
        _, _, rst = qwen35.sequential_delta(q, k, v, al, bt, S0, collect=True)
        _, _, kst = gated_delta_metal(q, k, v, al, bt, S0, collect=True)
        e = max((mx.abs(a - b).max() / mx.abs(b).max()).item()
                for a, b in zip(kst, rst))
        good = e < 2e-5 and len(kst) == len(rst)
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}]   collect: {len(kst)} states, "
              f"max rel err {e:.2e}")
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
