// Fused Metal kernel for the Qwen3.5 gated-delta recurrence.
//
// WHY THIS EXISTS
// ---------------
// Measured on an M3 Max, the recurrence is 19.5% of a decode step (10.4 ms of
// 53.6 ms) and 25.8% of an L=8 verify pass. It is only 0.302 GFLOP/token --
// 0.6% of the model's arithmetic -- so this is not a compute problem. It is
// state traffic:
//
//     state = 48 layers x 48 heads x 128 x 128 x 4 B = 151 MB
//     read once + written once per token             = 302 MB = 0.86 ms @ 350 GB/s
//     measured                                       = 10.4 ms -> 12x off the floor
//
// The unfused version is ~7 elementwise/reduce ops per step, each streaming the
// whole 3.1 MB per-layer state through memory. Fusing them reads the state once
// and writes it once.
//
// THE MAPPING
// -----------
// One thread per (batch, head, dv). Each thread owns state row S[b,h,v,:] -- 128
// contiguous floats -- and computes
//
//     kS[v]    = sum_dk k[dk] * S[v,dk]        (reduction, in-thread)
//     S[v,dk] <- a*(S[v,dk] - b*kS[v]*k[dk]) + b*v[v]*k[dk]
//     o[v]     = sum_dk q[dk] * S[v,dk]        (reduction, in-thread)
//
// Both reductions run over the thread's own row, so there is no cross-thread
// communication and no threadgroup memory. That only works because the state is
// laid out (Dv, Dk) rather than (Dk, Dv): the thread's row must be contiguous.
// Choosing (Dk, Dv) would make kS a 128-way reduction ACROSS threads. This is
// why both reference implementations store the state as (Dv, Dk).
//
// For L > 1 the loop over time lives inside the kernel, so the state is read and
// written once for the whole block rather than once per step.
#pragma once

#include <mlx/mlx.h>

#include <optional>
#include <vector>

namespace mx = mlx::core;

namespace delta {

bool available();

struct Result {
  mx::array o;                      // (B, L, H, Dv)
  mx::array s_out;                  // (B, H, Dv, Dk)
  std::vector<mx::array> states;    // per-step states, only when collect=true
};

// Drop-in for the per-token recurrence, with the state laid out (B,H,Dv,Dk).
//
//   q, k  : (B,L,H,Dk)  already l2-normalized and head-expanded
//   v     : (B,L,H,Dv)
//   alpha : (B,L,H)     forget gate in (0,1)
//   beta  : (B,L,H)     delta-rule write strength
//   S     : (B,H,Dv,Dk) or nullopt for a zero state
Result run(const mx::array& q, const mx::array& k, const mx::array& v,
           const mx::array& alpha, const mx::array& beta,
           const std::optional<mx::array>& S, bool collect = false);

// Reference per-token recurrence in plain MLX ops. Used when Metal is
// unavailable, and as the oracle the kernel is tested against.
Result run_reference(const mx::array& q, const mx::array& k, const mx::array& v,
                     const mx::array& alpha, const mx::array& beta,
                     const std::optional<mx::array>& S, bool collect = false);

}  // namespace delta
