// Character classes for the pre-tokenizer, plus UTF-8 and NFC helpers.
//
// The class tables live in the generated unicode_tables.cpp (see
// tools/gen_unicode.py). NFC normalization is delegated to CoreFoundation
// rather than table-generated: it needs canonical decomposition, combining
// classes and composition exclusions, and the system already has all of it.
#pragma once

#include <cstdint>
#include <string>

namespace uni {

struct Range {
  uint32_t lo, hi;
};

extern const Range kL[];
extern const int kLCount;
extern const Range kN[];
extern const int kNCount;
extern const Range kM[];
extern const int kMCount;
extern const Range kSPACE[];
extern const int kSPACECount;

bool in_ranges(uint32_t cp, const Range* r, int n);

inline bool is_L(uint32_t cp) { return in_ranges(cp, kL, kLCount); }
inline bool is_N(uint32_t cp) { return in_ranges(cp, kN, kNCount); }
inline bool is_M(uint32_t cp) { return in_ranges(cp, kM, kMCount); }
inline bool is_LM(uint32_t cp) { return is_L(cp) || is_M(cp); }
inline bool is_space(uint32_t cp) { return in_ranges(cp, kSPACE, kSPACECount); }

// Decode the UTF-8 codepoint starting at s[i]; advances i past it. Invalid
// bytes decode to U+FFFD and consume one byte, so this never stalls.
uint32_t decode(const std::string& s, size_t& i);

// Number of bytes in the UTF-8 sequence starting with byte b (1 if invalid).
int seq_len(unsigned char b);

void encode(uint32_t cp, std::string& out);

// Unicode NFC, via CoreFoundation.
std::string nfc(const std::string& s);

}  // namespace uni
