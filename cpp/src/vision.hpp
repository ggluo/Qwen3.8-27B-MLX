// The Qwen3.5 vision tower -- Qwen3-VL's encoder, tensor for tensor.
//
//     27 ViT blocks, hidden 1152, 16 heads of 72, 2-D rotary inside attention,
//     a learned 48x48 position table bilinearly resized to the image's grid, and
//     a merger that folds each 2x2 block of patches into one LLM token.
//
// A mirror of ../python/vision.py, which is held against transformers' own
// implementation to ~2e-5 in fp32. Three things here are easy to get wrong
// without an error, and each is marked where it happens:
//
//   * patch ORDER: merge-block-major, so 4 consecutive patches are one 2x2 block
//   * two GELUs: tanh-approximate in the block MLPs, exact erf in the merger
//   * the merger's LayerNorm runs per patch, BEFORE the 2x2 fold
#pragma once

#include <mlx/mlx.h>

#include <utility>
#include <vector>

#include "image.hpp"
#include "model.hpp"

struct VisionConfig {
  int depth = 27;
  int hidden_size = 1152;
  int intermediate_size = 4304;
  int num_heads = 16;
  int out_hidden_size = 5120;
  int num_position_embeddings = 2304;
  int patch_size = 16;
  int temporal_patch_size = 2;
  int spatial_merge_size = 2;
  int in_channels = 3;

  int head_dim() const { return hidden_size / num_heads; }
};

struct LayerNorm {
  mx::array weight, bias;
  float eps;
  mx::array operator()(const mx::array& x) const {
    return mx::fast::layer_norm(x, weight, bias, eps);
  }
};

struct VisionBlock {
  LayerNorm norm1, norm2;
  Linear qkv, proj;          // attention
  Linear fc1, fc2;           // MLP
};

struct VisionModel {
  VisionConfig cfg;
  // A Conv3d whose stride equals its kernel is a matmul over flattened patches;
  // the stored (1152, 3, 2, 16, 16) weight, reshaped, in the patch vector's
  // (C, T, kh, kw) order.
  mx::array patch_w, patch_b;
  Embedding pos_embed;
  std::vector<VisionBlock> blocks;
  LayerNorm merger_norm;
  Linear merger_fc1, merger_fc2;

  // One image -> (n_tokens, out_hidden_size), in the text model's dtype.
  mx::array operator()(const image::Image& img) const;

  mx::array position_embedding(int gh, int gw) const;
  std::pair<mx::array, mx::array> rotary(int gh, int gw) const;  // fp32 cos, sin
};
