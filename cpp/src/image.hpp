// Everything between an image file and the vision tower: decode, orient,
// resize, normalise, patch.
//
//     file bytes --ImageIO--> RGB uint8, EXIF orientation applied
//       --smart_resize--> both sides a multiple of 32, pixel count clamped
//       --bicubic--> resized RGB uint8
//       --(x/255 - 0.5)/0.5, 2 identical frames--> patches (N, 3*2*16*16)
//
// A mirror of ../python/vision.py, op for op: the same ImageIO decoder, the same
// double-precision resize weights, the same MLX calls in the same order. That is
// what lets the two ports produce byte-identical replies about the same picture.
// The Python side is held against transformers' own image processor -- the
// resized pixels are identical, the normalised values within one fp32 ULP.
#pragma once

#include <mlx/mlx.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace image {

constexpr int kPatch = 16;
constexpr int kTemporal = 2;  // a still image is duplicated to fill the 2-frame patch
constexpr int kMerge = 2;     // 2x2 patches -> one LLM token
constexpr int kFactor = kPatch * kMerge;

// The checkpoint's shortest_edge: 64 image tokens.
constexpr long long kMinPixels = 65536;
// The checkpoint says 16777216, which is 16384 tokens for one picture -- minutes
// of prefill here. ~1M pixels is 1024 tokens and keeps a screenshot's small
// text legible. Override per call.
constexpr long long kMaxPixels = 1024LL * kFactor * kFactor;

// Decoded pixels, upright, 3 bytes a pixel.
struct RGB {
  std::vector<uint8_t> px;
  int w = 0, h = 0;
};

// Encoded bytes (JPEG, PNG, HEIC, WebP, TIFF, ...) -> RGB. Transparency is
// composited over WHITE -- the reference drops alpha and keeps whatever colour a
// transparent pixel stored, usually black, which hides black text on a
// transparent background. Throws std::runtime_error on anything undecodable.
RGB decode(const std::string& bytes);
RGB load(const std::string& path);

// Apply an EXIF orientation (1..8) in place, as PIL.ImageOps.exif_transpose does.
void orient(RGB& img, int orientation);

// Qwen2-VL's resize rule: both sides a multiple of kFactor, the pixel count
// within [min_pixels, max_pixels], aspect ratio kept as closely as that allows.
// Returns (height, width). Throws on an aspect ratio beyond 200:1.
std::pair<int, int> smart_resize(int h, int w, long long min_pixels = kMinPixels,
                                 long long max_pixels = kMaxPixels);

// Row-major (n_out, n_in) antialiased bicubic weights, built exactly as the
// Python port builds them.
std::vector<double> resample_matrix(int n_in, int n_out);

// One picture, ready for the vision tower.
struct Image {
  mx::array pixels = mx::zeros({1});  // (N, 1536) float32 patches, merge-block order
  int t = 1, gh = 0, gw = 0;          // grid in PATCHES
  int w = 0, h = 0;                   // as decoded, after orientation
  int rw = 0, rh = 0;                 // as fed to the model

  int rows() const { return gh / kMerge; }  // LLM token grid
  int cols() const { return gw / kMerge; }
  int n_tokens() const { return t * rows() * cols(); }
};

// Must run on the thread that will run the model: it builds MLX arrays.
Image preprocess(const RGB& rgb, long long min_pixels = kMinPixels,
                 long long max_pixels = kMaxPixels);

// The (t, h, w) position of each of a rows x cols image's tokens, given the
// position `start` of its first one. Every token shares t = start. The text
// after the image resumes at start + max(rows, cols) -- not start + rows*cols --
// which is why the rope position and the cache offset part ways after an image.
void positions(int start, int rows, int cols, std::vector<int>& t, std::vector<int>& h,
               std::vector<int>& w);

// Standard base64, as a data: URL carries an image. Whitespace is skipped;
// anything else outside the alphabet throws.
std::string base64_decode(const std::string& text);

}  // namespace image
