#include "delta.hpp"

#include <map>
#include <sstream>
#include <string>
#include <tuple>

#include "ops.hpp"

namespace delta {
namespace {

// The kernel body. MLX exposes the declared inputs/outputs as device pointers
// with these names; DK/DV/H/L/R are baked in so `float s[R][NP]` is a
// fixed-size register array rather than a dynamic allocation.
const char* kBody = R"METAL(
    const uint DK = @DK@;
    const uint DV = @DV@;
    const uint H  = @H@;
    const uint L  = @L@;
    const uint R  = @R@;
    const uint NP = DK / 32;

    uint lane = thread_position_in_grid.x;
    uint v0   = thread_position_in_grid.y * R;
    uint bh   = thread_position_in_grid.z;
    uint h = bh % H;
    uint b = bh / H;

    // this simdgroup's R state rows (b, h, v0..v0+R, :), each contiguous over
    // DK; lane owns elements lane, lane+32, lane+64, ... of every row
    uint sbase = (bh * DV + v0) * DK + lane;

    float s[R][NP];
    for (uint r = 0; r < R; ++r) {
        for (uint i = 0; i < NP; ++i) {
            s[r][i] = S_in[sbase + r * DK + i * 32];
        }
    }

    for (uint t = 0; t < L; ++t) {
        uint hb = (b * L + t) * H + h;      // index into (B,L,H)
        uint kb = hb * DK + lane;           // this lane's first element of (B,L,H,DK)
        float a  = alpha[hb];
        float bb = beta[hb];

        float kk[NP], qq[NP];
        for (uint i = 0; i < NP; ++i) {
            kk[i] = k[kb + i * 32];
            qq[i] = q[kb + i * 32];
        }

        for (uint r = 0; r < R; ++r) {
            float vt = v[hb * DV + v0 + r];

            // kS = k . s   (over the row, reduced across the simdgroup)
            float kS = 0.0f;
            for (uint i = 0; i < NP; ++i) {
                kS += kk[i] * s[r][i];
            }
            kS = simd_sum(kS);

            // fused state update + output reduction in one pass over the row
            float ov = 0.0f;
            for (uint i = 0; i < NP; ++i) {
                float sn = a * (s[r][i] - bb * kS * kk[i]) + bb * vt * kk[i];
                s[r][i] = sn;
                ov += qq[i] * sn;
            }
            ov = simd_sum(ov);
            if (lane == 0) {
                o[hb * DV + v0 + r] = ov;
            }
        }
@COLLECT@
    }

    for (uint r = 0; r < R; ++r) {
        for (uint i = 0; i < NP; ++i) {
            S_out[sbase + r * DK + i * 32] = s[r][i];
        }
    }
)METAL";

// When collecting per-step states (for speculative rollback), also dump the rows
// after every step into (B, H, L, DV, DK).
const char* kCollectBody = R"METAL(
        uint cb = (((b * H + h) * L + t) * DV + v0) * DK + lane;
        for (uint r = 0; r < R; ++r) {
            for (uint i = 0; i < NP; ++i) {
                S_all[cb + r * DK + i * 32] = s[r][i];
            }
        }
)METAL";

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  for (size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + to.size())) {
    s.replace(p, from.size(), to);
  }
}

using Key = std::tuple<int, int, int, int, int, bool>;

mx::fast::CustomKernelFunction& kernel(int DK, int DV, int H, int L, int R, bool collect) {
  static std::map<Key, mx::fast::CustomKernelFunction> cache;
  Key key{DK, DV, H, L, R, collect};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  std::string src = kBody;
  replace_all(src, "@COLLECT@", collect ? kCollectBody : "");
  replace_all(src, "@DK@", std::to_string(DK));
  replace_all(src, "@DV@", std::to_string(DV));
  replace_all(src, "@H@", std::to_string(H));
  replace_all(src, "@L@", std::to_string(L));
  replace_all(src, "@R@", std::to_string(R));

  std::ostringstream name;
  name << "gated_delta_" << DK << "_" << DV << "_" << H << "_" << L << "_" << R << "_"
       << (collect ? 1 : 0);
  std::vector<std::string> outputs{"o", "S_out"};
  if (collect) outputs.push_back("S_all");

  auto fn = mx::fast::metal_kernel(name.str(), {"q", "k", "v", "alpha", "beta", "S_in"},
                                   outputs, src);
  return cache.emplace(key, std::move(fn)).first->second;
}

mx::array f32(const mx::array& a) { return mx::astype(a, mx::float32); }

}  // namespace

bool available() {
  return mx::default_device() == mx::Device::gpu && mx::metal::is_available();
}

Result run(const mx::array& q_in, const mx::array& k_in, const mx::array& v_in,
           const mx::array& alpha_in, const mx::array& beta_in,
           const std::optional<mx::array>& S_in, bool collect) {
  const int B = q_in.shape(0), L = q_in.shape(1), H = q_in.shape(2),
            Dk = q_in.shape(3);
  const int Dv = v_in.shape(3);

  mx::array q = f32(q_in), k = f32(k_in), v = f32(v_in);
  mx::array alpha = f32(alpha_in), beta = f32(beta_in);
  mx::array S = S_in ? f32(*S_in) : mx::zeros({B, H, Dv, Dk}, mx::float32);

  if (Dk % 32 != 0) return run_reference(q, k, v, alpha, beta, S, collect);
  // R rows per simdgroup share each step's k and q loads; four simdgroups per
  // threadgroup. Both measured: R=4 beat 1, 2 and 8 at L=384.
  const int R = (Dv % 4 == 0) ? 4 : 1;
  const int groups = Dv / R;
  const int tgy = (groups % 4 == 0) ? 4 : 1;

  std::vector<mx::Shape> shapes{{B, L, H, Dv}, {B, H, Dv, Dk}};
  if (collect) shapes.push_back({B, H, L, Dv, Dk});
  std::vector<mx::Dtype> dtypes(shapes.size(), mx::float32);

  std::vector<mx::array> outs = kernel(Dk, Dv, H, L, R, collect)(
      {q, k, v, alpha, beta, S}, shapes, dtypes, {32, groups, B * H}, {32, tgy, 1}, {},
      std::nullopt, false, {});

  Result r{outs[0], outs[1], {}};
  if (collect) {
    // Match the reference path's [S_1 .. S_L] list of (B,H,Dv,Dk): S_all is
    // (B,H,L,Dv,Dk), so drop the time axis at each t.
    r.states.reserve(static_cast<size_t>(L));
    for (int t = 0; t < L; ++t) {
      r.states.push_back(ops::index_axis(outs[2], 2, t));
    }
  }
  return r;
}

Result run_reference(const mx::array& q, const mx::array& k, const mx::array& v,
                     const mx::array& alpha, const mx::array& beta,
                     const std::optional<mx::array>& S_in, bool collect) {
  const int B = q.shape(0), L = q.shape(1), H = q.shape(2), Dk = q.shape(3);
  const int Dv = v.shape(3);

  mx::array S = S_in ? f32(*S_in) : mx::zeros({B, H, Dv, Dk}, mx::float32);
  std::vector<mx::array> outs, states;
  outs.reserve(static_cast<size_t>(L));

  for (int t = 0; t < L; ++t) {
    // (B,1,H,D) reshapes straight to the step layouts: element order is
    // (b, h, d) either way, so none of these are copies.
    mx::array q_t = mx::reshape(ops::slice_axis(q, 1, t, t + 1), {B, H, 1, Dk});
    mx::array k_t = mx::reshape(ops::slice_axis(k, 1, t, t + 1), {B, H, 1, Dk});
    mx::array v_t = mx::reshape(ops::slice_axis(v, 1, t, t + 1), {B, H, Dv, 1});
    mx::array a_t = mx::reshape(ops::slice_axis(alpha, 1, t, t + 1), {B, H, 1, 1});
    mx::array b_t = mx::reshape(ops::slice_axis(beta, 1, t, t + 1), {B, H, 1, 1});

    mx::array kS = mx::sum(mx::multiply(k_t, S), -1, true);          // (B,H,Dv,1)
    S = mx::add(mx::multiply(a_t, mx::subtract(S, mx::multiply(b_t,
            mx::multiply(kS, k_t)))),
        mx::multiply(b_t, mx::multiply(v_t, k_t)));
    outs.push_back(mx::sum(mx::multiply(q_t, S), -1));               // (B,H,Dv)
    if (collect) states.push_back(S);
  }

  mx::array Y = L > 0 ? mx::stack(outs, 1) : mx::zeros({B, 0, H, Dv}, mx::float32);
  return Result{Y, S, states};
}

}  // namespace delta
