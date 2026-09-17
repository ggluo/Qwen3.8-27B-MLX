#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <dirent.h>

#include <stdexcept>

#include "delta.hpp"
#include "json.hpp"

namespace {

using ops::index_axis;
using ops::slice_axis;

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

std::string shape_str(const mx::array& a) {
  std::string s = "(";
  for (size_t i = 0; i < a.ndim(); ++i) {
    if (i) s += ", ";
    s += std::to_string(a.shape(static_cast<int>(i)));
  }
  return s + ")";
}

// Additive causal mask for L queries at positions [offset, offset+L) against
// offset+L keys. Built explicitly rather than via mask_mode="causal" so the
// alignment is stated here rather than assumed.
mx::array causal_mask(int offset, int L, mx::Dtype dtype) {
  mx::array q = mx::reshape(mx::arange(offset, offset + L, mx::int32), {L, 1});
  mx::array k = mx::reshape(mx::arange(0, offset + L, mx::int32), {1, offset + L});
  float neg = static_cast<float>(mx::finfo(dtype).min);
  return mx::astype(
      mx::where(mx::less_equal(k, q), mx::array(0.0f), mx::array(neg)), dtype);
}

// ---------------------------------------------------------------------------
// weight loading
// ---------------------------------------------------------------------------

struct Loader {
  Weights& W;
  const Config& cfg;
  size_t consumed = 0;

  mx::array take(const std::string& name) {
    auto it = W.find(name);
    if (it == W.end()) fail("checkpoint is missing tensor '" + name + "'");
    ++consumed;
    return it->second;
  }

  void expect_shape(const mx::array& a, const mx::Shape& want,
                    const std::string& name) {
    if (a.shape() != want) {
      std::string w = "(";
      for (size_t i = 0; i < want.size(); ++i) {
        if (i) w += ", ";
        w += std::to_string(want[i]);
      }
      fail("tensor '" + name + "' has shape " + shape_str(a) + ", expected " + w + ")");
    }
  }

  // Zero-centered gamma: the stored value is an offset from 1. Folded in here,
  // once, in fp32 -- recomputing (1 + w) per call would cost two extra kernel
  // launches on every one of the ~161 live norms per forward pass.
  RMSNorm rmsnorm(const std::string& path, int dims) {
    mx::array w = take(path + ".weight");
    expect_shape(w, {dims}, path + ".weight");
    return RMSNorm{mx::add(mx::astype(w, mx::float32), mx::array(1.0f)),
                   cfg.rms_norm_eps};
  }

  // The DeltaNet gated norm is NOT zero-centered (its stored gammas are ~1).
  GatedRMSNorm gated_rmsnorm(const std::string& path, int dims) {
    mx::array w = take(path + ".weight");
    expect_shape(w, {dims}, path + ".weight");
    return GatedRMSNorm{mx::astype(w, mx::float32), cfg.rms_norm_eps};
  }

  Linear linear(const std::string& path, int out_f, int in_f) {
    const bool q = cfg.quantized && cfg.is_quantized(path);
    mx::array w = take(path + ".weight");
    std::optional<mx::array> sc, bi;
    if (q) {
      sc = take(path + ".scales");
      bi = take(path + ".biases");
      // 4-bit packs 8 values per uint32, so the stored width is in_f/8; check the
      // dimensions that actually pin the layout down.
      expect_shape(*sc, {out_f, in_f / cfg.group_size}, path + ".scales");
      expect_shape(*bi, {out_f, in_f / cfg.group_size}, path + ".biases");
      if (w.shape(0) != out_f) {
        fail("tensor '" + path + ".weight' has shape " + shape_str(w) +
             ", expected " + std::to_string(out_f) + " rows");
      }
    } else {
      expect_shape(w, {out_f, in_f}, path + ".weight");
    }
    return Linear{w, sc, bi, cfg.group_size, cfg.bits};
  }

  Embedding embedding(const std::string& path, int rows, int dim) {
    const bool q = cfg.quantized && cfg.is_quantized(path);
    mx::array w = take(path + ".weight");
    std::optional<mx::array> sc, bi;
    if (q) {
      sc = take(path + ".scales");
      bi = take(path + ".biases");
      expect_shape(*sc, {rows, dim / cfg.group_size}, path + ".scales");
    } else {
      expect_shape(w, {rows, dim}, path + ".weight");
    }
    return Embedding{w, sc, bi, cfg.group_size, cfg.bits};
  }

  MLP mlp(const std::string& path) {
    const int h = cfg.hidden_size, f = cfg.intermediate_size;
    return MLP{linear(path + ".gate_proj", f, h), linear(path + ".up_proj", f, h),
               linear(path + ".down_proj", h, f)};
  }

  Attention attention(const std::string& path) {
    const int h = cfg.hidden_size, hd = cfg.head_dim;
    const int nq = cfg.num_attention_heads, nkv = cfg.num_key_value_heads;
    // q_proj is double width when gated: per head it emits [query | gate].
    const int q_out = nq * hd * (cfg.attn_output_gate ? 2 : 1);
    return Attention{nq,
                     nkv,
                     hd,
                     cfg.rotary_dim(),
                     1.0f / std::sqrt(static_cast<float>(hd)),
                     cfg.rope_theta,
                     cfg.attn_output_gate,
                     linear(path + ".q_proj", q_out, h),
                     linear(path + ".k_proj", nkv * hd, h),
                     linear(path + ".v_proj", nkv * hd, h),
                     linear(path + ".o_proj", h, nq * hd),
                     rmsnorm(path + ".q_norm", hd),
                     rmsnorm(path + ".k_norm", hd)};
  }

  GatedDeltaNet deltanet(const std::string& path) {
    const int h = cfg.hidden_size;
    const int nk = cfg.linear_num_key_heads, nv = cfg.linear_num_value_heads;
    const int dk = cfg.linear_key_head_dim, dv = cfg.linear_value_head_dim;
    const int K = cfg.linear_conv_kernel_dim;
    const int qkv_dim = nk * dk * 2 + nv * dv;  // 10240
    const int z_dim = nv * dv;                  // 6144

    mx::array conv_w = take(path + ".conv1d.weight");
    expect_shape(conv_w, {qkv_dim, 1, K}, path + ".conv1d.weight");
    mx::array A_log = take(path + ".A_log");
    expect_shape(A_log, {nv}, path + ".A_log");
    mx::array dt_bias = take(path + ".dt_bias");
    expect_shape(dt_bias, {nv}, path + ".dt_bias");

    return GatedDeltaNet{
        nk,
        nv,
        dk,
        dv,
        nv / nk,
        K,
        qkv_dim,
        z_dim,
        linear(path + ".in_proj_qkv", qkv_dim, h),
        linear(path + ".in_proj_z", z_dim, h),
        linear(path + ".in_proj_a", nv, h),
        linear(path + ".in_proj_b", nv, h),
        linear(path + ".out_proj", h, z_dim),
        // (C,1,K) -> (K,C) once, so the per-tap broadcast below is a plain index
        DepthwiseConv{mx::transpose(mx::squeeze(conv_w, 1)), K},
        gated_rmsnorm(path + ".norm", dv),
        mx::astype(A_log, mx::float32),
        mx::astype(dt_bias, mx::float32)};
  }

  Layer layer(const std::string& path, bool is_linear) {
    RMSNorm in_ln = rmsnorm(path + ".input_layernorm", cfg.hidden_size);
    RMSNorm post_ln = rmsnorm(path + ".post_attention_layernorm", cfg.hidden_size);
    std::optional<Attention> attn;
    std::optional<GatedDeltaNet> lin;
    if (is_linear) {
      lin = deltanet(path + ".linear_attn");
    } else {
      attn = attention(path + ".self_attn");
    }
    return Layer{is_linear, in_ln, post_ln, attn, lin, mlp(path + ".mlp")};
  }
};

std::vector<std::string> shard_paths(const std::string& dir) {
  std::vector<std::string> out;
  DIR* d = opendir(dir.c_str());
  if (!d) fail("cannot open model directory: " + dir);
  while (struct dirent* e = readdir(d)) {
    std::string n = e->d_name;
    if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) {
      out.push_back(dir + "/" + n);
    }
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  if (out.empty()) fail("no .safetensors files in " + dir);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

bool Config::is_quantized(const std::string& path) const {
  return quantized_modules.count(path) > 0 ||
         quantized_modules.count("model.language_model." + path) > 0;
}

Config Config::from_json(const std::string& config_path) {
  json::Value raw = json::parse_file(config_path);
  const json::Value& t = raw["text_config"];
  if (!t.is_obj()) fail("config.json has no text_config");

  Config c;
  c.hidden_size = static_cast<int>(t["hidden_size"].as_int(c.hidden_size));
  c.num_hidden_layers =
      static_cast<int>(t["num_hidden_layers"].as_int(c.num_hidden_layers));
  c.num_attention_heads =
      static_cast<int>(t["num_attention_heads"].as_int(c.num_attention_heads));
  c.num_key_value_heads =
      static_cast<int>(t["num_key_value_heads"].as_int(c.num_key_value_heads));
  c.head_dim = static_cast<int>(t["head_dim"].as_int(c.head_dim));
  c.intermediate_size =
      static_cast<int>(t["intermediate_size"].as_int(c.intermediate_size));
  c.vocab_size = static_cast<int>(t["vocab_size"].as_int(c.vocab_size));
  c.rms_norm_eps = static_cast<float>(t["rms_norm_eps"].as_num(c.rms_norm_eps));
  c.attn_output_gate = t["attn_output_gate"].as_bool(c.attn_output_gate);
  c.linear_num_key_heads =
      static_cast<int>(t["linear_num_key_heads"].as_int(c.linear_num_key_heads));
  c.linear_num_value_heads =
      static_cast<int>(t["linear_num_value_heads"].as_int(c.linear_num_value_heads));
  c.linear_key_head_dim =
      static_cast<int>(t["linear_key_head_dim"].as_int(c.linear_key_head_dim));
  c.linear_value_head_dim =
      static_cast<int>(t["linear_value_head_dim"].as_int(c.linear_value_head_dim));
  c.linear_conv_kernel_dim =
      static_cast<int>(t["linear_conv_kernel_dim"].as_int(c.linear_conv_kernel_dim));
  c.mtp_num_hidden_layers =
      static_cast<int>(t["mtp_num_hidden_layers"].as_int(c.mtp_num_hidden_layers));

  // rope_parameters wins over the flat keys, matching the reference loader.
  const json::Value& rope = t["rope_parameters"];
  c.rope_theta = static_cast<float>(
      rope["rope_theta"].as_num(t["rope_theta"].as_num(c.rope_theta)));
  c.partial_rotary_factor = static_cast<float>(rope["partial_rotary_factor"].as_num(
      t["partial_rotary_factor"].as_num(c.partial_rotary_factor)));

  const json::Value& lt = t["layer_types"];
  if (!lt.is_arr() || lt.size() == 0) fail("config.json has no text_config.layer_types");
  for (size_t i = 0; i < lt.size(); ++i) c.layer_types.push_back(lt[i].as_str());
  if (static_cast<int>(c.layer_types.size()) != c.num_hidden_layers) {
    fail("layer_types has " + std::to_string(c.layer_types.size()) +
         " entries but num_hidden_layers is " + std::to_string(c.num_hidden_layers));
  }

  const json::Value& q = raw["quantization"];
  if (q.is_obj()) {
    c.quantized = true;
    c.group_size = static_cast<int>(q["group_size"].as_int(64));
    c.bits = static_cast<int>(q["bits"].as_int(4));
    const json::Value& mods = q["quantized_modules"];
    for (size_t i = 0; i < mods.size(); ++i) c.quantized_modules.insert(mods[i].as_str());
  }
  return c;
}

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

mx::array RMSNorm::operator()(const mx::array& x) const {
  return mx::astype(mx::fast::rms_norm(mx::astype(x, mx::float32), gamma, eps),
                    x.dtype());
}

mx::array GatedRMSNorm::operator()(const mx::array& x, const mx::array& z) const {
  // Normalize FIRST, then gate. See the header.
  mx::array h = mx::fast::rms_norm(mx::astype(x, mx::float32), gamma, eps);
  return mx::multiply(h, ops::silu(mx::astype(z, mx::float32)));
}

mx::array Linear::operator()(const mx::array& x) const {
  if (quantized()) {
    return mx::quantized_matmul(x, w, *scales, biases, /*transpose=*/true,
                                group_size, bits);
  }
  return mx::matmul(x, mx::transpose(mx::astype(w, x.dtype())));
}

mx::array Embedding::operator()(const mx::array& ids) const {
  if (!scales) return mx::take(w, ids, 0);
  // Gather the packed rows and dequantize just those, rather than materializing
  // a 248320 x 5120 table.
  mx::array rows = mx::take(w, ids, 0);
  mx::array sc = mx::take(*scales, ids, 0);
  std::optional<mx::array> bi;
  if (biases) bi = mx::take(*biases, ids, 0);
  return mx::dequantize(rows, sc, bi, group_size, bits);
}

std::pair<mx::array, mx::array> DepthwiseConv::operator()(
    const mx::array& x, const std::optional<mx::array>& state) const {
  const int B = x.shape(0), L = x.shape(1), C = x.shape(2);
  mx::array pad = state ? *state : mx::zeros({B, K - 1, C}, x.dtype());
  mx::array xp = mx::concatenate({pad, x}, 1);  // (B, L+K-1, C)
  mx::array y = mx::multiply(slice_axis(xp, 1, 0, L), index_axis(wt, 0, 0));
  for (int j = 1; j < K; ++j) {
    y = mx::add(y, mx::multiply(slice_axis(xp, 1, j, j + L), index_axis(wt, 0, j)));
  }
  return {y, xp};
}

// ---------------------------------------------------------------------------
// caches
// ---------------------------------------------------------------------------

std::pair<mx::array, mx::array> KVCache::update_and_fetch(const mx::array& k,
                                                          const mx::array& v) {
  const int prev = offset;
  const int L = k.shape(2);

  if (!keys || prev + L > keys->shape(2)) {
    const int B = k.shape(0), H = k.shape(1), dk = k.shape(3), dv = v.shape(3);
    const int grow = ((L + kStep - 1) / kStep) * kStep;
    mx::array nk = mx::zeros({B, H, grow, dk}, k.dtype());
    mx::array nv = mx::zeros({B, H, grow, dv}, v.dtype());
    if (!keys) {
      keys = nk;
      values = nv;
    } else {
      // drop any unused tail before extending so the buffer stays dense
      if (prev < keys->shape(2)) {
        keys = slice_axis(*keys, 2, 0, prev);
        values = slice_axis(*values, 2, 0, prev);
      }
      keys = mx::concatenate({*keys, nk}, 2);
      values = mx::concatenate({*values, nv}, 2);
    }
  }

  offset = prev + L;
  // slice_update returns a new array, but MLX donates the buffer when the input
  // has no other reference -- so this is the in-place write it looks like.
  mx::Shape start(4, 0), stop = keys->shape();
  start[2] = prev;
  stop[2] = offset;
  keys = mx::slice_update(*keys, k, start, stop);
  stop = values->shape();
  stop[2] = offset;
  values = mx::slice_update(*values, v, start, stop);

  return {slice_axis(*keys, 2, 0, offset), slice_axis(*values, 2, 0, offset)};
}

void DeltaCache::commit(int n) {
  if (!pending) return;
  Pending p = std::move(*pending);
  const int L = static_cast<int>(p.states.size());
  if (n < 0 || n > L) fail("DeltaCache::commit out of range");

  mx::array st = (n == 0) ? p.s_prev : p.states[static_cast<size_t>(n) - 1];
  const int kw = p.conv_input.shape(1) - L;
  mx::array cv = slice_axis(p.conv_input, 1, n, n + kw);

  // mx::contiguous DETACHES these from the graph that produced them. Without it,
  // holding a slice keeps the whole block's computation alive -- all L states and
  // the conv input -- and it compounds every round.
  state = mx::contiguous(st);
  conv = mx::contiguous(cv);
  offset = base_offset + n;
  pending.reset();
  record = false;
}

// ---------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------

mx::array Attention::operator()(const mx::array& x,
                                const std::optional<mx::array>& mask,
                                KVCache* cache,
                                std::optional<int> rope_offset) const {
  const int B = x.shape(0), L = x.shape(1);
  // read the offset BEFORE update_and_fetch advances it
  const int offset = cache ? cache->offset : 0;
  const int pos = rope_offset ? *rope_offset : offset;

  mx::array q = mx::zeros({1});  // replaced immediately
  std::optional<mx::array> gate;
  if (output_gate) {
    // reshape to (B, L, n_heads, 2*hd) then split PER HEAD -- not a global
    // first-half/second-half split of the 12288 columns.
    mx::array qg = mx::reshape(q_proj(x), {B, L, n_heads, 2 * hd});
    q = slice_axis(qg, 3, 0, hd);
    gate = mx::reshape(slice_axis(qg, 3, hd, 2 * hd), {B, L, n_heads * hd});
  } else {
    q = mx::reshape(q_proj(x), {B, L, n_heads, hd});
  }
  mx::array k = mx::reshape(k_proj(x), {B, L, n_kv, hd});
  mx::array v = mx::reshape(v_proj(x), {B, L, n_kv, hd});

  q = mx::transpose(q_norm(q), {0, 2, 1, 3});  // QK-Norm before RoPE
  k = mx::transpose(k_norm(k), {0, 2, 1, 3});
  v = mx::transpose(v, {0, 2, 1, 3});

  // Partial RoPE: the first `rotary_dim` of 256 head dims, rotate_half pairing.
  // For text this is exactly M-RoPE -- with mrope_section [11,11,10] the three
  // sections index time/height/width, and for a text token all three are the
  // token index, so every section sees the same value.
  q = mx::fast::rope(q, rotary_dim, /*traditional=*/false, rope_theta, 1.0f, pos);
  k = mx::fast::rope(k, rotary_dim, /*traditional=*/false, rope_theta, 1.0f, pos);

  if (cache) {
    std::pair<mx::array, mx::array> kv = cache->update_and_fetch(k, v);
    k = kv.first;
    v = kv.second;
  }

  mx::array o = mask ? mx::fast::scaled_dot_product_attention(q, k, v, scale,
                                                              "array", {*mask})
                     : mx::fast::scaled_dot_product_attention(q, k, v, scale);
  o = mx::reshape(mx::transpose(o, {0, 2, 1, 3}), {B, L, n_heads * hd});
  if (gate) o = mx::multiply(o, mx::sigmoid(*gate));
  return o_proj(o);
}

mx::array GatedDeltaNet::operator()(const mx::array& x, DeltaCache* cache) const {
  const int B = x.shape(0), L = x.shape(1);
  std::optional<mx::array> conv_state = cache ? cache->conv : std::nullopt;
  std::optional<mx::array> S_in = cache ? cache->state : std::nullopt;

  std::pair<mx::array, mx::array> cv = conv1d(in_proj_qkv(x), conv_state);
  mx::array qkv = ops::silu(cv.first);
  const mx::array& conv_input = cv.second;

  // contiguous [q | k | v] layout inside in_proj_qkv
  const int nkd = nk * dk;
  mx::array q = mx::reshape(slice_axis(qkv, 2, 0, nkd), {B, L, nk, dk});
  mx::array k = mx::reshape(slice_axis(qkv, 2, nkd, 2 * nkd), {B, L, nk, dk});
  mx::array v = mx::reshape(slice_axis(qkv, 2, 2 * nkd, qkv_dim), {B, L, nv, dv});

  q = ops::l2norm(q);
  k = ops::l2norm(k);
  // Reference scaling: k gets plain l2norm, q additionally gets 1/sqrt(dk). The
  // q factor cannot change the output -- the gated RMSNorm downstream is
  // scale-invariant -- but match it so the intermediates line up.
  q = mx::multiply(q, mx::array(1.0f / std::sqrt(static_cast<float>(dk)), q.dtype()));
  // repeat_interleave: key head i serves value heads 3i, 3i+1, 3i+2
  q = mx::astype(mx::repeat(q, rep, 2), mx::float32);
  k = mx::astype(mx::repeat(k, rep, 2), mx::float32);
  v = mx::astype(v, mx::float32);

  mx::array beta = mx::sigmoid(mx::astype(in_proj_b(x), mx::float32));   // (B,L,nv)
  mx::array a = mx::astype(in_proj_a(x), mx::float32);
  mx::array A = mx::exp(A_log);
  mx::array alpha =
      mx::exp(mx::negative(mx::multiply(A, ops::softplus(mx::add(a, dt_bias)))));

  mx::array S = S_in ? *S_in : mx::zeros({B, nv, dv, dk}, mx::float32);
  const mx::array S_prev = S;

  // Speculative blocks need every per-step state kept, for rollback.
  const bool recording = cache && cache->record;
  // The fused kernel is correct at any L. Above L ~= 384 a chunked parallel scan
  // would be faster (it turns time into matmuls, while this kernel is serial in
  // t); that path is not ported, and prefill is windowed to 384 so it never runs
  // past the crossover anyway.
  delta::Result r = delta::available()
                        ? delta::run(q, k, v, alpha, beta, S, recording)
                        : delta::run_reference(q, k, v, alpha, beta, S, recording);

  mx::array z = mx::reshape(in_proj_z(x), {B, L, nv, dv});
  mx::array o = mx::reshape(mx::astype(norm(r.o, z), x.dtype()), {B, L, z_dim});

  if (cache) {
    if (recording) {
      // do not advance the state; keep the per-step states for commit()
      cache->base_offset = cache->offset;
      cache->pending = DeltaCache::Pending{conv_input, S_prev, r.states};
    } else {
      const int T = conv_input.shape(1);
      cache->conv = slice_axis(conv_input, 1, T - (K - 1), T);
      cache->state = r.s_out;
      cache->offset += L;
    }
  }
  return out_proj(o);
}

mx::array MLP::operator()(const mx::array& x) const {
  return down_proj(mx::multiply(ops::silu(gate_proj(x)), up_proj(x)));
}

mx::array Layer::operator()(const mx::array& x, const std::optional<mx::array>& mask,
                            LayerCache* cache, std::optional<int> rope_offset) const {
  mx::array h = input_layernorm(x);
  // caches are mutated in place, so nothing is returned
  if (is_linear) {
    h = (*linear_attn)(h, cache ? &cache->delta : nullptr);
  } else {
    h = (*self_attn)(h, mask, cache ? &cache->kv : nullptr, rope_offset);
  }
  mx::array y = mx::add(x, h);
  return mx::add(y, mlp(post_attention_layernorm(y)));
}

// ---------------------------------------------------------------------------
// MTP drafter
// ---------------------------------------------------------------------------

std::vector<LayerCache> MTPDraft::make_cache() const {
  return std::vector<LayerCache>(layers.size());
}

mx::array MTPDraft::operator()(const mx::array& token_embed, const mx::array& hidden,
                               std::vector<LayerCache>& cache, int rope_offset) const {
  mx::array h = mx::concatenate(
      {pre_fc_norm_embedding(token_embed), pre_fc_norm_hidden(hidden)}, -1);
  h = fc(h);
  const int L = h.shape(1);
  std::optional<mx::array> mask;
  if (L > 1) mask = causal_mask(cache[0].offset(), L, h.dtype());
  for (size_t i = 0; i < layers.size(); ++i) {
    h = layers[i](h, mask, &cache[i], rope_offset);
  }
  return norm(h);
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

std::vector<LayerCache> Model::make_cache() const {
  std::vector<LayerCache> c(cfg.layer_types.size());
  for (size_t i = 0; i < c.size(); ++i) {
    c[i].linear = cfg.layer_types[i] == "linear_attention";
  }
  return c;
}

mx::array Model::hidden_states(const mx::array& ids,
                               std::vector<LayerCache>& cache) const {
  mx::array x = embed_tokens(ids);
  const int L = ids.shape(1);
  // every layer's cache tracks the same logical length
  const int offset = cache.empty() ? 0 : cache[0].offset();

  std::optional<mx::array> mask;
  if (L > 1) mask = causal_mask(offset, L, x.dtype());

  for (size_t i = 0; i < layers.size(); ++i) {
    x = layers[i](x, mask, &cache[i], std::nullopt);
  }
  return norm(x);
}

mx::array Model::forward_last(const mx::array& ids,
                              std::vector<LayerCache>& cache) const {
  mx::array h = hidden_states(ids, cache);
  return lm_head(slice_axis(h, 1, h.shape(1) - 1, h.shape(1)));
}

Model load_model(const std::string& path, bool verbose) {
  Config cfg = Config::from_json(path + "/config.json");

  Weights W;
  size_t skipped = 0, nbytes = 0;
  for (const std::string& f : shard_paths(path)) {
    mx::SafetensorsLoad shard = mx::load_safetensors(f);
    for (auto& kv : shard.first) {
      if (kv.first.rfind("model.visual.", 0) == 0) {
        ++skipped;
        continue;
      }
      // Strip the language-model prefix so mtp.* (stored unprefixed) and the
      // decoder tensors share one flat namespace.
      const std::string prefix = "model.language_model.";
      std::string name = kv.first.rfind(prefix, 0) == 0
                             ? kv.first.substr(prefix.size())
                             : kv.first;
      nbytes += kv.second.nbytes();
      W.emplace(std::move(name), kv.second);
    }
  }

  Loader ld{W, cfg};
  Embedding embed = ld.embedding("embed_tokens", cfg.vocab_size, cfg.hidden_size);
  std::vector<Layer> layers;
  layers.reserve(static_cast<size_t>(cfg.num_hidden_layers));
  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    layers.push_back(ld.layer("layers." + std::to_string(i),
                              cfg.layer_types[static_cast<size_t>(i)] ==
                                  "linear_attention"));
  }
  RMSNorm final_norm = ld.rmsnorm("norm", cfg.hidden_size);
  Linear lm_head = ld.linear("lm_head", cfg.vocab_size, cfg.hidden_size);

  std::optional<MTPDraft> mtp;
  if (cfg.mtp_num_hidden_layers > 0) {
    const int h = cfg.hidden_size;
    std::vector<Layer> mtp_layers;
    for (int i = 0; i < cfg.mtp_num_hidden_layers; ++i) {
      // The MTP layer is FULL attention, not DeltaNet, whatever layer_types[0] says.
      mtp_layers.push_back(ld.layer("mtp.layers." + std::to_string(i), false));
    }
    mtp = MTPDraft{ld.linear("mtp.fc", h, 2 * h),
                   ld.rmsnorm("mtp.pre_fc_norm_embedding", h),
                   ld.rmsnorm("mtp.pre_fc_norm_hidden", h),
                   ld.rmsnorm("mtp.norm", h), std::move(mtp_layers)};
  }

  Model m{cfg, embed, std::move(layers), final_norm, lm_head, std::move(mtp)};

  // Force the weights (and the folded fp32 gammas) resident before the first
  // token, so timing does not blame decode for the load.
  std::vector<mx::array> all;
  for (auto& kv : W) all.push_back(kv.second);
  mx::eval(all);

  if (verbose) {
    printf("loaded %zu tensors, %.2f GB (%zu vision tensors skipped)\n", W.size(),
           static_cast<double>(nbytes) / 1e9, skipped);
    if (ld.consumed != W.size()) {
      printf("  note: %zu tensors in the checkpoint were not used\n",
             W.size() - ld.consumed);
    }
  }
  return m;
}
