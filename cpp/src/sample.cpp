#include "sample.hpp"

#include <limits>

#include "ops.hpp"

namespace sample {

mx::array to_probs(const mx::array& logits_in, float temp, float top_p, int top_k) {
  mx::array logits = mx::astype(logits_in, mx::float32);
  const int V = logits.shape(-1);

  if (temp == 0.0f) {
    mx::array am = mx::argmax(logits, -1, /*keepdims=*/true);
    return mx::astype(mx::equal(mx::arange(V, mx::int32), am), mx::float32);
  }

  const float kNegInf = std::numeric_limits<float>::lowest();
  if (top_k > 0 && top_k < V) {
    // topk returns the k largest in ascending order, so element 0 is the k-th
    // largest -- the cutoff.
    mx::array kth = ops::slice_axis(mx::topk(logits, top_k, -1), -1, 0, 1);
    logits = mx::where(mx::less(logits, kth), mx::array(kNegInf), logits);
  }

  mx::array p = mx::softmax(mx::divide(logits, mx::array(temp)), -1);

  if (top_p > 0.0f && top_p < 1.0f) {
    mx::array order = mx::argsort(mx::negative(p), -1);
    mx::array ps = mx::take_along_axis(p, order, -1);
    mx::array cum = mx::cumsum(ps, -1);
    // (cum - ps) < top_p always keeps at least the top token
    mx::array keep = mx::less(mx::subtract(cum, ps), mx::array(top_p));
    ps = mx::where(keep, ps, mx::array(0.0f));
    p = mx::put_along_axis(mx::zeros_like(p), order, ps, -1);
    p = mx::divide(p, mx::sum(p, -1, true));
  }
  return p;
}

int sample_probs(const mx::array& p) {
  mx::array idx = mx::random::categorical(mx::log(mx::add(p, mx::array(1e-30f))));
  mx::eval(idx);
  return static_cast<int>(idx.item<uint32_t>());
}

mx::array residual(const mx::array& p, const mx::array& q) {
  mx::array r = mx::maximum(mx::subtract(p, q), mx::array(0.0f));
  mx::array tot = mx::sum(r);
  mx::eval(tot);
  if (tot.item<float>() <= 1e-12f) return mx::divide(p, mx::sum(p));
  return mx::divide(r, tot);
}

}  // namespace sample
