// Byte-level BPE for Qwen3.5. Reads the model's own tokenizer.json.
//
//     NFC normalize -> split on the GPT-4 regex -> map bytes to safe unicode
//     -> greedy lowest-rank BPE merges -> ids
//
// A direct port of tokenizer.py, kept deliberately structure-for-structure with
// it: the split pattern uses \p{L} / \p{N} / \p{M}, so `pretokenize` is a
// hand-written scanner over the generated category tables (see unicode.hpp)
// rather than a regex. Any divergence here would silently shift token
// boundaries, so tests.cpp diffs the ids against the Python implementation.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// LITERAL REGIONS
// ---------------
// A chat prompt is two kinds of text: the template's own structure, whose
// <|im_start|> and <tool_call> must become control tokens, and what someone
// wrote -- a message, a tool's output, a file an agent read -- which must not.
// Without the distinction a README that merely mentions <|im_end|> ends the
// turn it is quoted in.
//
// So text between kLiteralOpen and kLiteralClose (two Unicode noncharacters,
// which never occur in interchanged text) is literal: a special token's text in
// there is encoded as the ordinary text it is. It is still cut out as its own
// piece, exactly where a special token would have been, and the markers
// themselves vanish without creating a boundary -- so wrapping text that holds
// no special tokens changes nothing about how it is tokenized.
constexpr const char* kLiteralOpen = "\xEF\xB7\x90";   // U+FDD0
constexpr const char* kLiteralClose = "\xEF\xB7\x91";  // U+FDD1

// Wraps `text` as a literal region, first removing any of the markers it may
// carry (U+FDD0..U+FDD2) so that text cannot close its own region early.
std::string literal(const std::string& text);

// `text` without any U+FDD0..U+FDD2 markers.
std::string strip_marks(const std::string& text);

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json);

  // `allow_special` = false treats the whole text as literal.
  std::vector<int> encode(const std::string& text, bool allow_special = true) const;
  std::string decode(const std::vector<int>& ids, bool skip_special = false) const;

  // Exposed for the self-tests.
  std::vector<std::string> pretokenize(const std::string& text) const;

  int eos_id = -1;
  int bos_id = -1;
  size_t vocab_size() const { return vocab_.size(); }
  size_t merge_count() const { return ranks_.size(); }
  size_t special_count() const { return added_.size(); }

 private:
  std::vector<int> bpe(const std::string& piece) const;

  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> added_;      // special tokens
  std::vector<std::string> added_sorted_;           // longest first
  std::unordered_map<int, std::string> id_to_token_;

  // Merge ranks, keyed by the concatenation "left\x00right". One hash lookup per
  // adjacent pair; the NUL keeps the key unambiguous.
  std::unordered_map<std::string, int> ranks_;

  std::string byte_to_uni_[256];                    // GPT-2 byte -> unicode char
  std::unordered_map<uint32_t, uint8_t> uni_to_byte_;
  bool normalize_ = false;

  // BPE is pure, so results are memoized. `encode` is const, hence mutable.
  mutable std::unordered_map<std::string, std::vector<int>> cache_;
};
