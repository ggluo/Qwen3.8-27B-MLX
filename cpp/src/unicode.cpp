#include "unicode.hpp"

#include <CoreFoundation/CoreFoundation.h>

namespace uni {

bool in_ranges(uint32_t cp, const Range* r, int n) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if (cp < r[mid].lo) {
      hi = mid - 1;
    } else if (cp > r[mid].hi) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

int seq_len(unsigned char b) {
  if (b < 0x80) return 1;
  if ((b & 0xE0) == 0xC0) return 2;
  if ((b & 0xF0) == 0xE0) return 3;
  if ((b & 0xF8) == 0xF0) return 4;
  return 1;  // continuation or invalid lead: treat as one bad byte
}

uint32_t decode(const std::string& s, size_t& i) {
  unsigned char b = static_cast<unsigned char>(s[i]);
  int n = seq_len(b);
  if (n == 1) {
    ++i;
    return b < 0x80 ? b : 0xFFFD;
  }
  if (i + n > s.size()) {
    ++i;
    return 0xFFFD;
  }
  static const uint32_t lead_mask[5] = {0, 0x7F, 0x1F, 0x0F, 0x07};
  uint32_t cp = b & lead_mask[n];
  for (int k = 1; k < n; ++k) {
    unsigned char c = static_cast<unsigned char>(s[i + k]);
    if ((c & 0xC0) != 0x80) {  // truncated sequence
      ++i;
      return 0xFFFD;
    }
    cp = (cp << 6) | (c & 0x3F);
  }
  i += n;
  return cp;
}

void encode(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string nfc(const std::string& s) {
  if (s.empty()) return s;
  CFStringRef in = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8*>(s.data()),
      static_cast<CFIndex>(s.size()), kCFStringEncodingUTF8, false);
  if (!in) return s;  // not valid UTF-8; leave it alone
  CFMutableStringRef m = CFStringCreateMutableCopy(kCFAllocatorDefault, 0, in);
  CFRelease(in);
  if (!m) return s;
  CFStringNormalize(m, kCFStringNormalizationFormC);

  CFIndex len = CFStringGetLength(m);
  CFIndex max = CFStringGetMaximumSizeForEncoding(len, kCFStringEncodingUTF8) + 1;
  std::string out(static_cast<size_t>(max), '\0');
  CFIndex used = 0;
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false,
                   reinterpret_cast<UInt8*>(&out[0]), max, &used);
  CFRelease(m);
  out.resize(static_cast<size_t>(used));
  return out;
}

}  // namespace uni
