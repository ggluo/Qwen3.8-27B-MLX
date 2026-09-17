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

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json);

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
