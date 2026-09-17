#include "ops.hpp"

namespace ops {

mx::array slice_axis(const mx::array& x, int axis, int start, int stop) {
  const int nd = static_cast<int>(x.ndim());
  if (axis < 0) axis += nd;
  mx::Shape lo(nd, 0), hi = x.shape();
  lo[axis] = start;
  hi[axis] = stop;
  return mx::slice(x, lo, hi);
}

mx::array index_axis(const mx::array& x, int axis, int i) {
  const int nd = static_cast<int>(x.ndim());
  if (axis < 0) axis += nd;
  return mx::squeeze(slice_axis(x, axis, i, i + 1), axis);
}

mx::array silu(const mx::array& x) { return mx::multiply(x, mx::sigmoid(x)); }

mx::array softplus(const mx::array& x) {
  // log1p(exp(x)) overflows for large x; MLX's logaddexp(0, x) is the stable form.
  return mx::logaddexp(mx::zeros_like(x), x);
}

mx::array l2norm(const mx::array& x, float eps) {
  mx::array sq = mx::sum(mx::multiply(x, x), -1, true);
  return mx::multiply(x, mx::rsqrt(mx::add(sq, mx::array(eps, x.dtype()))));
}

}  // namespace ops
