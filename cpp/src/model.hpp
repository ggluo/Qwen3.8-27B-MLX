// Qwen3.5-27B (`model_type: qwen3_5`) text decoder, on MLX's C++ API.
//
// The architecture, straight from config.json:
//
//     64 layers, hidden 5120, head_dim 256, vocab 248320
//     layer_types = [linear, linear, linear, full] x 16
//
//   * 16 FULL-ATTENTION layers -- GQA 24 Q heads / 4 KV heads, per-head
//     QK-RMSNorm, a sigmoid OUTPUT GATE (q_proj is double width: [Q | gate] per
//     head), and RoPE on only the first 64 of 256 head dims.
//
//   * 48 GATED-DELTANET layers -- linear attention. No KV cache: a fixed
//     128x128 state matrix per head, updated by the gated delta rule
//
//         S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T
//         o_t = S_t^T q_t
//
//     Q/K/V first pass through a depthwise causal conv (kernel 4) + silu; the
//     output is swish-gated and RMSNormed. 16 K heads serve 48 V heads.
//
//   That 3:1 ratio is why 256K context is affordable: three quarters of the
//   layers cost O(1) memory per token instead of O(n).
//
// Two conventions here are silent if you get them wrong -- fluent nonsense, never
// an error -- and both cost real debugging time in the Python original:
// zero-centered norm gammas (see RMSNorm) and normalize-then-gate in the gated
// norm (see GatedRMSNorm). Do not "simplify" either.
#pragma once

#include <mlx/mlx.h>

#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ops.hpp"

namespace mx = mlx::core;

using Weights = std::unordered_map<std::string, mx::array>;

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

struct Config {
  int hidden_size = 5120;
  int num_hidden_layers = 64;
  int num_attention_heads = 24;
  int num_key_value_heads = 4;
  int head_dim = 256;
  int intermediate_size = 17408;
  int vocab_size = 248320;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1e7f;
  float partial_rotary_factor = 0.25f;
  bool attn_output_gate = true;
  std::vector<std::string> layer_types;
  // gated deltanet
  int linear_num_key_heads = 16;
  int linear_num_value_heads = 48;
  int linear_key_head_dim = 128;
  int linear_value_head_dim = 128;
  int linear_conv_kernel_dim = 4;
  int mtp_num_hidden_layers = 0;
  // quantization, from the checkpoint's own record
  bool quantized = false;
  int group_size = 64;
  int bits = 4;
  std::set<std::string> quantized_modules;

  int rotary_dim() const {
    return static_cast<int>(static_cast<float>(head_dim) * partial_rotary_factor);
  }
  bool is_quantized(const std::string& path) const;

  static Config from_json(const std::string& config_path);
};

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

// x_hat * gamma, with gamma held in fp32.
//
// For the pre-norms, QK-norms and the final norm the stored gamma is an OFFSET
// FROM 1: the checkpoint's post_attention_layernorm weights are all <= 0 (mean
// -0.217), which as a plain multiplicative gain would annihilate the block
// input. Perplexity confirms it: plain w gives 13.6 nats/token, (1 + w) gives
// 5.1. The (1 + w) is folded in at load time, in fp32 -- bf16 resolution near
// 1.0 is only ~0.004, which would quantize a gamma of -0.033 into uselessness.
struct RMSNorm {
  mx::array gamma;
  float eps;
  mx::array operator()(const mx::array& x) const;
};

// out = silu(z) * rmsnorm(x, gamma) -- NORMALIZE FIRST, THEN GATE.
//
// The order matters and is easy to get backwards. Gating inside the norm --
// rmsnorm(x * silu(z)) -- divides the gate's own magnitude back out, so the gate
// could only rotate the vector, never scale it. Getting this wrong costs 3.2
// nats/token. Note this norm's gamma is NOT zero-centered.
struct GatedRMSNorm {
  mx::array gamma;
  float eps;
  mx::array operator()(const mx::array& x, const mx::array& z) const;
};

// A weight matrix, 4-bit quantized or plain bf16, stored (out_features, in).
struct Linear {
  mx::array w;
  std::optional<mx::array> scales;
  std::optional<mx::array> biases;
  int group_size = 64;
  int bits = 4;

  bool quantized() const { return scales.has_value(); }
  mx::array operator()(const mx::array& x) const;
};

// The embedding table, which is also quantized in this checkpoint.
struct Embedding {
  mx::array w;
  std::optional<mx::array> scales;
  std::optional<mx::array> biases;
  int group_size = 64;
  int bits = 4;

  mx::array operator()(const mx::array& ids) const;  // (B,L) -> (B,L,hidden)
};

// Causal depthwise conv, kernel 4, no bias. `wt` is the stored (C,1,K) weight
// pre-transposed to (K,C); written as an explicit sum over taps rather than a
// grouped conv1d, which is unambiguous about layout and avoids a slow path.
struct DepthwiseConv {
  mx::array wt;
  int K;
  // Returns {y, padded_input}. The last K-1 rows of the padded input are the
  // next call's conv state.
  std::pair<mx::array, mx::array> operator()(
      const mx::array& x, const std::optional<mx::array>& state) const;
};

// ---------------------------------------------------------------------------
// caches
// ---------------------------------------------------------------------------

// Pre-allocated, growable KV cache for the full-attention layers.
//
// Concatenating on every decode step reallocates and copies the whole cache each
// token: O(n) per step, O(n^2) over a generation. Here the buffer is
// over-allocated in blocks of kStep and new keys/values are written in place.
//
// `offset` is the true logical length; the buffer is usually longer, so always
// hand attention the [..., :offset, :] slice rather than the raw buffer.
struct KVCache {
  static constexpr int kStep = 256;

  std::optional<mx::array> keys;
  std::optional<mx::array> values;
  int offset = 0;

  std::pair<mx::array, mx::array> update_and_fetch(const mx::array& k,
                                                   const mx::array& v);
  // Roll back to `length` tokens. Rejected speculative entries are left in the
  // buffer as garbage past `offset`; the next write overwrites them and nothing
  // ever reads beyond `offset`.
  void trim(int length) { offset = length; }
};

// State for one Gated-DeltaNet layer, with rollback support.
//
// The conv ring holds K-1 timesteps and `state` is one (H, Dv, Dk) matrix per
// head -- both fixed size no matter how long the context gets.
//
// Rollback is the hard part of speculative decoding for a recurrent layer. A KV
// cache rolls back by truncation, but delta-rule updates cannot be un-applied:
// S_t = a(I - b k k^T)S_{t-1} + b k v^T destroys information about S_{t-1}. So
// in `record` mode the state is NOT advanced; instead every per-step state the
// forward pass already computed is kept, and commit(n) picks the n-th one. That
// costs ~3.1 MB per step per layer and zero extra compute.
struct DeltaCache {
  struct Pending {
    mx::array conv_input;
    mx::array s_prev;
    std::vector<mx::array> states;
  };

  std::optional<mx::array> conv;
  std::optional<mx::array> state;
  int offset = 0;
  bool record = false;
  int base_offset = 0;
  std::optional<Pending> pending;

  // Accept the first n tokens of the recorded block and drop the rest.
  void commit(int n);
};

// One cache slot per layer. Only the member matching the layer type is used;
// both are cheap when empty, which keeps the call sites free of variant noise.
struct LayerCache {
  bool linear = false;
  KVCache kv;
  DeltaCache delta;

  int offset() const { return linear ? delta.offset : kv.offset; }
};

// ---------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------

struct Attention {
  int n_heads, n_kv, hd, rotary_dim;
  float scale, rope_theta;
  bool output_gate;
  Linear q_proj, k_proj, v_proj, o_proj;
  RMSNorm q_norm, k_norm;

  // `rope_offset` overrides the position used for RoPE. The MTP drafter's KV
  // cache starts mid-sequence, so its cache index and absolute position differ.
  mx::array operator()(const mx::array& x, const std::optional<mx::array>& mask,
                       KVCache* cache, std::optional<int> rope_offset) const;
};

struct GatedDeltaNet {
  int nk, nv, dk, dv, rep, K, qkv_dim, z_dim;
  Linear in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj;
  DepthwiseConv conv1d;
  GatedRMSNorm norm;
  mx::array A_log, dt_bias;  // fp32

  mx::array operator()(const mx::array& x, DeltaCache* cache) const;
};

struct MLP {
  Linear gate_proj, up_proj, down_proj;
  mx::array operator()(const mx::array& x) const;
};

struct Layer {
  bool is_linear = false;
  RMSNorm input_layernorm, post_attention_layernorm;
  std::optional<Attention> self_attn;
  std::optional<GatedDeltaNet> linear_attn;
  MLP mlp;

  mx::array operator()(const mx::array& x, const std::optional<mx::array>& mask,
                       LayerCache* cache, std::optional<int> rope_offset) const;
};

// Multi-Token Prediction head, used here as a self-speculative drafter.
//
//     h_t (target's final hidden, post-norm) -> pre_fc_norm_hidden ---.
//     emb(token t+1) (target's embedding table) -> pre_fc_norm_embedding -'
//                         concat -> fc (10240 -> 5120)
//                         -> 1 FULL-attention decoder layer
//                         -> norm -> the target's OWN lm_head
//
// ~424M params, 1.6% of the model. Both the embedding table and the LM head are
// shared with the target (`mtp_use_dedicated_embeddings: false`).
struct MTPDraft {
  Linear fc;
  RMSNorm pre_fc_norm_embedding, pre_fc_norm_hidden, norm;
  std::vector<Layer> layers;

  std::vector<LayerCache> make_cache() const;
  mx::array operator()(const mx::array& token_embed, const mx::array& hidden,
                       std::vector<LayerCache>& cache, int rope_offset) const;
};

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

struct Model {
  Config cfg;
  Embedding embed_tokens;
  std::vector<Layer> layers;
  RMSNorm norm;
  Linear lm_head;
  std::optional<MTPDraft> mtp;

  std::vector<LayerCache> make_cache() const;

  // Post-norm hidden: exactly the tensor fed to lm_head, and what the MTP
  // drafter consumes (it applies its own pre_fc_norm_hidden).
  mx::array hidden_states(const mx::array& ids, std::vector<LayerCache>& cache) const;

  mx::array logits(const mx::array& h) const { return lm_head(h); }

  // Convenience: hidden_states + lm_head on the last position only. Projecting
  // all L positions through a 248320-wide head materializes an L x 248320
  // tensor (1 GB at L=2048) for nothing.
  mx::array forward_last(const mx::array& ids, std::vector<LayerCache>& cache) const;
};

// Loads config.json + every *.safetensors shard. Throws if any tensor the model
// needs is missing or misshaped -- the structural check that the Python port
// got from `strict=True`.
Model load_model(const std::string& path, bool verbose = true);
