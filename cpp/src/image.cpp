#include "image.hpp"

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace image {
namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

// Owns one CoreFoundation reference.
template <typename T>
struct CF {
  T ref = nullptr;
  explicit CF(T r = nullptr) : ref(r) {}
  ~CF() {
    if (ref) CFRelease(ref);
  }
  CF(const CF&) = delete;
  CF& operator=(const CF&) = delete;
  operator T() const { return ref; }
};

double cubic(double x) {
  const double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

// PIL rounds half up and clips at every pass; floor(x + 0.5) is that.
mx::array to_u8(const mx::array& x) {
  return mx::clip(mx::floor(mx::add(x, mx::array(0.5f))), mx::array(0.0f),
                  mx::array(255.0f));
}

mx::array weights(int n_in, int n_out) {
  std::vector<double> m = resample_matrix(n_in, n_out);
  std::vector<float> f(m.begin(), m.end());
  return mx::array(f.begin(), {n_out, n_in}, mx::float32);
}

// (3, H, W) float32 in 0..255 -> (3, out_h, out_w). Horizontal first, rounded to
// 8 bits in between: PIL's order and precision.
mx::array resize(mx::array chw, int out_h, int out_w) {
  const int h = chw.shape(1), w = chw.shape(2);
  if (out_w != w) chw = to_u8(mx::matmul(chw, mx::transpose(weights(w, out_w))));
  if (out_h != h) chw = to_u8(mx::matmul(weights(h, out_h), chw));
  return chw;
}

}  // namespace

RGB decode(const std::string& bytes) {
  if (bytes.empty()) fail("empty image");
  CF<CFDataRef> data(CFDataCreate(kCFAllocatorDefault,
                                  reinterpret_cast<const UInt8*>(bytes.data()),
                                  static_cast<CFIndex>(bytes.size())));
  CF<CGImageSourceRef> src(CGImageSourceCreateWithData(data, nullptr));
  if (!src) fail("not an image ImageIO can read");
  if (CGImageSourceGetCount(src) < 1) fail("image file holds no images");
  CF<CGImageRef> img(CGImageSourceCreateImageAtIndex(src, 0, nullptr));
  if (!img) fail("cannot decode image");

  int orientation = 1;
  CF<CFDictionaryRef> props(CGImageSourceCopyPropertiesAtIndex(src, 0, nullptr));
  if (props) {
    auto num = static_cast<CFNumberRef>(
        CFDictionaryGetValue(props, kCGImagePropertyOrientation));
    if (num) CFNumberGetValue(num, kCFNumberIntType, &orientation);
  }

  const size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
  if (w == 0 || h == 0) fail("image has no pixels");

  // An RGB image is drawn in its own colour space, so its values come through as
  // stored rather than colour-managed; anything else goes to sRGB.
  CGColorSpaceRef own = CGImageGetColorSpace(img);  // not owned
  CF<CGColorSpaceRef> made;
  CGColorSpaceRef cs = own;
  if (!own || CGColorSpaceGetModel(own) != kCGColorSpaceModelRGB) {
    made.ref = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    cs = made;
  }
  std::vector<uint8_t> rgba(w * h * 4);
  CGContextRef ctx = CGBitmapContextCreate(rgba.data(), w, h, 8, w * 4, cs,
                                           kCGImageAlphaPremultipliedLast);
  if (!ctx && !made) {
    // an embedded profile bitmap contexts refuse: fall back to sRGB
    made.ref = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    ctx = CGBitmapContextCreate(rgba.data(), w, h, 8, w * 4, made, kCGImageAlphaPremultipliedLast);
  }
  if (!ctx) fail("cannot create a bitmap context for this image");
  const CGRect rect = CGRectMake(0, 0, static_cast<CGFloat>(w), static_cast<CGFloat>(h));
  CGContextSetInterpolationQuality(ctx, kCGInterpolationNone);
  CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0);  // transparency over white
  CGContextFillRect(ctx, rect);
  CGContextDrawImage(ctx, rect, img);
  CGContextRelease(ctx);

  RGB out;
  out.w = static_cast<int>(w);
  out.h = static_cast<int>(h);
  out.px.resize(w * h * 3);
  for (size_t i = 0; i < w * h; ++i) {
    out.px[3 * i] = rgba[4 * i];
    out.px[3 * i + 1] = rgba[4 * i + 1];
    out.px[3 * i + 2] = rgba[4 * i + 2];
  }
  orient(out, orientation);
  return out;
}

RGB load(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) fail("cannot open " + path + ": " + strerror(errno));
  std::string bytes;
  char buf[1 << 16];
  for (size_t n; (n = fread(buf, 1, sizeof(buf), f)) > 0;) bytes.append(buf, n);
  fclose(f);
  return decode(bytes);
}

void orient(RGB& img, int orientation) {
  if (orientation < 2 || orientation > 8) return;
  const int W = img.w, H = img.h;
  const bool swap = orientation >= 5;  // 5..8 transpose the axes
  const int ow = swap ? H : W, oh = swap ? W : H;
  std::vector<uint8_t> out(img.px.size());
  // out(y, x) = in(sy, sx); each case is the Python port's slicing, written out:
  //   2 a[:, ::-1]   3 a[::-1, ::-1]   4 a[::-1, :]   5 T(a)
  //   6 T(a)[:, ::-1]   7 T(a)[::-1, ::-1]   8 T(a)[::-1, :]
  for (int y = 0; y < oh; ++y) {
    for (int x = 0; x < ow; ++x) {
      int sy = 0, sx = 0;
      switch (orientation) {
        case 2: sy = y;         sx = W - 1 - x; break;
        case 3: sy = H - 1 - y; sx = W - 1 - x; break;
        case 4: sy = H - 1 - y; sx = x;         break;
        case 5: sy = x;         sx = y;         break;
        case 6: sy = H - 1 - x; sx = y;         break;
        case 7: sy = H - 1 - x; sx = W - 1 - y; break;
        case 8: sy = x;         sx = W - 1 - y; break;
      }
      std::memcpy(&out[(static_cast<size_t>(y) * ow + x) * 3],
                  &img.px[(static_cast<size_t>(sy) * W + sx) * 3], 3);
    }
  }
  img.px.swap(out);
  img.w = ow;
  img.h = oh;
}

std::pair<int, int> smart_resize(int h, int w, long long min_pixels, long long max_pixels) {
  const int f = kFactor;
  if (static_cast<double>(std::max(h, w)) / std::min(h, w) > 200) {
    fail("aspect ratio is more than 200:1");
  }
  // nearbyint under the default rounding mode is half-to-even, which is what
  // Python's round() -- and so the reference -- does.
  int hb = static_cast<int>(std::nearbyint(static_cast<double>(h) / f)) * f;
  int wb = static_cast<int>(std::nearbyint(static_cast<double>(w) / f)) * f;
  const double hw = static_cast<double>(static_cast<long long>(h) * w);
  if (static_cast<long long>(hb) * wb > max_pixels) {
    const double beta = std::sqrt(hw / static_cast<double>(max_pixels));
    hb = std::max(f, static_cast<int>(std::floor(h / beta / f)) * f);
    wb = std::max(f, static_cast<int>(std::floor(w / beta / f)) * f);
  } else if (static_cast<long long>(hb) * wb < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) / hw);
    hb = static_cast<int>(std::ceil(h * beta / f)) * f;
    wb = static_cast<int>(std::ceil(w * beta / f)) * f;
  }
  return {hb, wb};
}

std::vector<double> resample_matrix(int n_in, int n_out) {
  // Same expressions, same order as the Python port: -ffp-contract=off keeps
  // clang from fusing any of them into an FMA.
  const double scale = static_cast<double>(n_in) / n_out;
  const double fs = std::max(scale, 1.0);
  const double support = 2.0 * fs;
  std::vector<double> m(static_cast<size_t>(n_out) * n_in, 0.0);
  std::vector<double> ws;
  for (int i = 0; i < n_out; ++i) {
    const double center = (i + 0.5) * scale;
    const int lo = std::max(static_cast<int>(center - support + 0.5), 0);
    const int hi = std::min(static_cast<int>(center + support + 0.5), n_in);
    ws.clear();
    double total = 0.0;
    for (int j = lo; j < hi; ++j) ws.push_back(cubic((j - center + 0.5) / fs));
    for (double v : ws) total += v;
    for (int j = lo; j < hi; ++j) {
      m[static_cast<size_t>(i) * n_in + j] = ws[static_cast<size_t>(j - lo)] / total;
    }
  }
  return m;
}

Image preprocess(const RGB& rgb, long long min_pixels, long long max_pixels) {
  const std::pair<int, int> r = smart_resize(rgb.h, rgb.w, min_pixels, max_pixels);
  const int rh = r.first, rw = r.second;
  mx::array x(rgb.px.begin(), {rgb.h, rgb.w, 3}, mx::uint8);
  x = mx::astype(mx::transpose(x, {2, 0, 1}), mx::float32);  // (3, H, W)
  x = resize(x, rh, rw);
  // rescale then normalise, in that order and in float32, as the reference does
  x = mx::divide(mx::subtract(mx::multiply(x, mx::array(static_cast<float>(1.0 / 255.0))),
                              mx::array(0.5f)),
                 mx::array(0.5f));
  const int gh = rh / kPatch, gw = rw / kPatch;
  x = mx::stack(std::vector<mx::array>(kTemporal, x));      // (T, 3, H, W)
  x = mx::reshape(x, {1, kTemporal, 3, gh / kMerge, kMerge, kPatch, gw / kMerge, kMerge, kPatch});
  x = mx::transpose(x, {0, 3, 6, 4, 7, 2, 1, 5, 8});
  x = mx::reshape(x, {gh * gw, 3 * kTemporal * kPatch * kPatch});

  Image im;
  im.pixels = x;
  im.t = 1;
  im.gh = gh;
  im.gw = gw;
  im.w = rgb.w;
  im.h = rgb.h;
  im.rw = rw;
  im.rh = rh;
  return im;
}

void positions(int start, int rows, int cols, std::vector<int>& t, std::vector<int>& h,
               std::vector<int>& w) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      t.push_back(start);
      h.push_back(start + r);
      w.push_back(start + c);
    }
  }
}

std::string base64_decode(const std::string& text) {
  static int8_t lut[256];
  static bool built = false;
  if (!built) {
    std::memset(lut, -1, sizeof(lut));
    const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) lut[static_cast<unsigned char>(abc[i])] = static_cast<int8_t>(i);
    // the URL-safe alphabet too; some clients send it
    lut[static_cast<unsigned char>('-')] = 62;
    lut[static_cast<unsigned char>('_')] = 63;
    built = true;
  }
  std::string out;
  out.reserve(text.size() * 3 / 4);
  uint32_t acc = 0;
  int bits = 0;
  for (unsigned char c : text) {
    if (c == '=') break;
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    const int v = lut[c];
    if (v < 0) fail("invalid base64 in image data");
    acc = (acc << 6) | static_cast<uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((acc >> bits) & 0xFF));
    }
  }
  return out;
}

}  // namespace image
