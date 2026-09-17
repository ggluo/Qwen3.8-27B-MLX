// Small array helpers shared by the model and the delta kernel.
#pragma once

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace ops {

// x[..., start:stop] along `axis`. Negative axes count from the end.
mx::array slice_axis(const mx::array& x, int axis, int start, int stop);

// x[..., i] along `axis`, with that axis dropped.
mx::array index_axis(const mx::array& x, int axis, int i);

mx::array silu(const mx::array& x);
mx::array softplus(const mx::array& x);

// x * rsqrt(sum(x^2, -1) + eps) -- note this is l2, not RMS: no 1/sqrt(d).
mx::array l2norm(const mx::array& x, float eps = 1e-6f);

}  // namespace ops
