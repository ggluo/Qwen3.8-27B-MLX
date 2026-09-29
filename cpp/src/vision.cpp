#include "vision.hpp"

#include <cmath>

#include "ops.hpp"

namespace {

// A Python scalar in an MLX expression takes the array's dtype ("weak" typing);
// a C++ mx::array(float) would be float32 and promote a bf16 operand instead.
// Every scalar below goes through this so the arithmetic matches the Python port.
mx::array sc(double v, const mx::array& like) {
  return mx::array(static_cast<float>(v), like.dtype());
}

// gelu_pytorch_tanh, written as MLX's nn.gelu_approx writes it:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
mx::array gelu_tanh(const mx::array& x) {
  mx::array cube = mx::power(x, sc(3.0, x));
  mx::array inner = mx::add(x, mx::multiply(sc(0.044715, x), cube));
  mx::array t = mx::tanh(mx::multiply(sc(std::sqrt(2.0 / M_PI), x), inner));
  return mx::multiply(mx::multiply(sc(0.5, x), x), mx::add(sc(1.0, x), t));
}

// The exact (erf) GELU, as nn.gelu:  x * (1 + erf(x / sqrt(2))) / 2
mx::array gelu_erf(const mx::array& x) {
  mx::array e = mx::erf(mx::divide(x, sc(std::sqrt(2.0), x)));
  return mx::divide(mx::multiply(x, mx::add(sc(1.0, x), e)), sc(2.0, x));
}

// rotate_half RoPE over the whole head, in fp32 as the reference does.
// x: (N, H, hd); cos/sin: (N, hd).
mx::array rotate(const mx::array& x_in, const mx::array& cos, const mx::array& sin) {
  const mx::Dtype dt = x_in.dtype();
  mx::array x = mx::astype(x_in, mx::float32);
  const int half = x.shape(-1) / 2;
  mx::array rot = mx::concatenate(
      {mx::negative(ops::slice_axis(x, -1, half, 2 * half)), ops::slice_axis(x, -1, 0, half)},
      -1);
  mx::array y = mx::add(mx::multiply(x, mx::expand_dims(cos, 1)),
                        mx::multiply(rot, mx::expand_dims(sin, 1)));
  return mx::astype(y, dt);
}

// (row, col) of every patch, in the order the patches are laid out: block row,
// block col, row within block, col within block.
void merge_order(int gh, int gw, int m, std::vector<int>& rows, std::vector<int>& cols) {
  for (int br = 0; br < gh / m; ++br) {
    for (int bc = 0; bc < gw / m; ++bc) {
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < m; ++c) {
          rows.push_back(br * m + r);
          cols.push_back(bc * m + c);
        }
      }
    }
  }
}

}  // namespace

mx::array VisionModel::position_embedding(int gh, int gw) const {
  // The learned side x side table, bilinearly resized to (gh, gw). Sample points
  // are linspace(0, side-1, n): corner-aligned, so the table's edge rows land
  // exactly on the image's edges.
  const int s = static_cast<int>(std::lround(std::sqrt(cfg.num_position_embeddings)));
  auto axis = [s](int n) {
    std::vector<double> v;
    if (n == 1) return std::vector<double>{0.0};
    for (int i = 0; i < n; ++i) v.push_back(static_cast<double>(i * (s - 1)) / (n - 1));
    return v;
  };
  const std::vector<double> hs = axis(gh), ws = axis(gw);
  std::vector<int> rows, cols;
  merge_order(gh, gw, cfg.spatial_merge_size, rows, cols);

  std::vector<int> idx[4];
  std::vector<float> wt[4];
  for (size_t p = 0; p < rows.size(); ++p) {
    const double y = hs[static_cast<size_t>(rows[p])], x = ws[static_cast<size_t>(cols[p])];
    const int y0 = static_cast<int>(y), x0 = static_cast<int>(x);
    const int y1 = std::min(y0 + 1, s - 1), x1 = std::min(x0 + 1, s - 1);
    const double dy = y - y0, dx = x - x0;
    const int yy[4] = {y0, y0, y1, y1}, xx[4] = {x0, x1, x0, x1};
    const double w4[4] = {(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx};
    for (int k = 0; k < 4; ++k) {
      idx[k].push_back(yy[k] * s + xx[k]);
      wt[k].push_back(static_cast<float>(w4[k]));
    }
  }
  const int n = static_cast<int>(rows.size());
  std::optional<mx::array> out;
  for (int k = 0; k < 4; ++k) {
    mx::array ids(idx[k].begin(), {n}, mx::int32);
    mx::array rows_k = mx::astype(pos_embed(ids), mx::float32);
    mx::array term = mx::multiply(
        rows_k, mx::expand_dims(mx::array(wt[k].begin(), {n}, mx::float32), 1));
    out = out ? mx::add(*out, term) : term;
  }
  return *out;
}

std::pair<mx::array, mx::array> VisionModel::rotary(int gh, int gw) const {
  // Half the frequencies index the patch's row, half its column.
  const int dim = cfg.head_dim() / 2;  // 36 -> 18 frequencies
  mx::array inv = mx::divide(
      mx::array(1.0f),
      mx::power(mx::array(10000.0f),
                mx::divide(mx::arange(0.0, static_cast<double>(dim), 2.0, mx::float32),
                           mx::array(static_cast<float>(dim)))));
  std::vector<int> rows, cols;
  merge_order(gh, gw, cfg.spatial_merge_size, rows, cols);
  std::vector<float> rf(rows.begin(), rows.end()), cf(cols.begin(), cols.end());
  const int n = static_cast<int>(rows.size());
  mx::array r = mx::multiply(mx::expand_dims(mx::array(rf.begin(), {n}, mx::float32), 1),
                             mx::expand_dims(inv, 0));
  mx::array c = mx::multiply(mx::expand_dims(mx::array(cf.begin(), {n}, mx::float32), 1),
                             mx::expand_dims(inv, 0));
  mx::array emb = mx::concatenate({r, c}, -1);  // (N, 36)
  emb = mx::concatenate({emb, emb}, -1);        // (N, 72)
  return {mx::cos(emb), mx::sin(emb)};
}

mx::array VisionModel::operator()(const image::Image& img) const {
  if (img.t != 1) throw std::runtime_error("video is not supported");
  const mx::Dtype dtype = merger_fc2.bias->dtype();
  const int heads = cfg.num_heads, hd = cfg.head_dim();

  mx::array x = mx::astype(img.pixels, dtype);
  x = mx::add(mx::matmul(mx::astype(x, patch_w.dtype()), mx::transpose(patch_w)), patch_b);
  x = mx::astype(mx::add(mx::astype(x, mx::float32), position_embedding(img.gh, img.gw)), dtype);
  const std::pair<mx::array, mx::array> cs = rotary(img.gh, img.gw);
  const int n = x.shape(0);
  const float scale = static_cast<float>(std::pow(static_cast<double>(hd), -0.5));

  for (const VisionBlock& b : blocks) {
    // attention: bidirectional, every patch of the image sees every other
    mx::array qkv = mx::reshape(b.qkv(b.norm1(x)), {n, 3, heads, hd});
    mx::array q = rotate(ops::index_axis(qkv, 1, 0), cs.first, cs.second);
    mx::array k = rotate(ops::index_axis(qkv, 1, 1), cs.first, cs.second);
    mx::array v = ops::index_axis(qkv, 1, 2);
    auto heads_first = [](const mx::array& t) {
      return mx::expand_dims(mx::transpose(t, {1, 0, 2}), 0);  // (1, H, N, hd)
    };
    mx::array o = mx::fast::scaled_dot_product_attention(heads_first(q), heads_first(k),
                                                         heads_first(v), scale);
    o = mx::reshape(mx::transpose(ops::index_axis(o, 0, 0), {1, 0, 2}), {n, -1});
    x = mx::add(x, b.proj(o));
    // MLP: the TANH GELU
    x = mx::add(x, b.fc2(gelu_tanh(b.fc1(b.norm2(x)))));
  }

  // merger: LayerNorm per patch BEFORE folding 2x2 patches into one token, then
  // the EXACT (erf) GELU -- not the one the blocks use
  const int unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
  x = mx::reshape(merger_norm(x), {-1, x.shape(-1) * unit});
  return merger_fc2(gelu_erf(merger_fc1(x)));
}
