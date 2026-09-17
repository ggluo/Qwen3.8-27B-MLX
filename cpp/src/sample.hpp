// Sampling, shared by plain and speculative decoding.
//
// The rejection test in speculative decoding compares p(token) against q(token)
// as ACTUAL sampling distributions, so whatever truncation the sampler applies
// has to be reflected in the probability vector itself -- hence `to_probs`
// returning the exact vector that gets drawn from, rather than a sample.
#pragma once

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace sample {

// Logits (V,) -> the exact probability vector the sampler draws from.
// temp == 0 yields a one-hot at the argmax, which makes greedy a special case of
// the same code path rather than a separate branch.
mx::array to_probs(const mx::array& logits, float temp, float top_p, int top_k);

// Draw one index from a 1-D probability vector.
int sample_probs(const mx::array& p);

// Normalized max(0, p - q); falls back to p if the residual is empty.
mx::array residual(const mx::array& p, const mx::array& q);

}  // namespace sample
