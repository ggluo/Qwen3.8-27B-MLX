#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

#include "json.hpp"
#include "unicode.hpp"

namespace {

// The pattern being reimplemented, for reference:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   |\p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
const char* kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

// GPT-2's byte<->unicode map: every byte becomes a printable, non-space
// codepoint so the BPE never sees whitespace it would have to special-case.
void build_byte_maps(std::string (&to_uni)[256],
                     std::unordered_map<uint32_t, uint8_t>& to_byte) {
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    std::string s;
    uni::encode(static_cast<uint32_t>(cs[i]), s);
    to_uni[bs[i]] = s;
    to_byte[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
  }
}

inline bool is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

std::string merge_key(const std::string& a, const std::string& b) {
  std::string k;
  k.reserve(a.size() + b.size() + 1);
  k += a;
  k.push_back('\0');
  k += b;
  return k;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& tokenizer_json) {
  json::Value spec = json::parse_file(tokenizer_json);
  const json::Value& model = spec["model"];
  if (model["type"].as_str() != "BPE") {
    throw std::runtime_error("tokenizer: expected a BPE model, got '" +
                             model["type"].as_str() + "'");
  }

  const json::Value& vocab = model["vocab"];
  if (!vocab.is_obj()) throw std::runtime_error("tokenizer: model.vocab missing");
  vocab_.reserve(vocab.obj.size() * 2);
  id_to_token_.reserve(vocab.obj.size() * 2);
  for (const auto& kv : vocab.obj) {
    int id = static_cast<int>(kv.second.as_int(-1));
    vocab_[kv.first] = id;
    id_to_token_[id] = kv.first;
  }

  const json::Value& merges = model["merges"];
  ranks_.reserve(merges.size() * 2);
  for (size_t i = 0; i < merges.size(); ++i) {
    const json::Value& m = merges[i];
    std::string left, right;
    if (m.type == json::Type::Str) {
      // "Ġ Ġ" -- a single space separates the two halves
      const std::string& s = m.str;
      size_t sp = s.find(' ');
      if (sp == std::string::npos) throw std::runtime_error("tokenizer: bad merge");
      left = s.substr(0, sp);
      right = s.substr(sp + 1);
    } else if (m.is_arr() && m.size() == 2) {
      left = m[0].as_str();
      right = m[1].as_str();
    } else {
      throw std::runtime_error("tokenizer: bad merge entry");
    }
    ranks_[merge_key(left, right)] = static_cast<int>(i);
  }

  for (size_t i = 0; i < spec["added_tokens"].size(); ++i) {
    const json::Value& t = spec["added_tokens"][i];
    std::string content = t["content"].as_str();
    int id = static_cast<int>(t["id"].as_int(-1));
    added_[content] = id;
    if (!vocab_.count(content)) vocab_[content] = id;
    if (!id_to_token_.count(id)) id_to_token_[id] = content;
    added_sorted_.push_back(content);
  }
  // Longest first so "<|im_start|>" wins over any prefix of it.
  std::sort(added_sorted_.begin(), added_sorted_.end(),
            [](const std::string& a, const std::string& b) {
              return a.size() != b.size() ? a.size() > b.size() : a < b;
            });

  build_byte_maps(byte_to_uni_, uni_to_byte_);
  normalize_ = spec["normalizer"]["type"].as_str() == "NFC";

  auto lookup = [&](const char* name) {
    auto it = added_.find(name);
    return it == added_.end() ? -1 : it->second;
  };
  eos_id = lookup("<|im_end|>");
  if (eos_id < 0) eos_id = lookup("<|endoftext|>");
  bos_id = lookup("<|endoftext|>");
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
  // Decode to codepoints once, keeping byte offsets, so the scanner below can be
  // written against codepoint indices exactly like the Python original. Doing it
  // directly on bytes would mean re-deriving character boundaries at every
  // lookahead, which is where an off-by-one would hide.
  std::vector<uint32_t> cp;
  std::vector<size_t> off;  // byte offset of each codepoint, plus a final sentinel
  for (size_t i = 0; i < text.size();) {
    off.push_back(i);
    cp.push_back(uni::decode(text, i));
  }
  off.push_back(text.size());

  const size_t n = cp.size();
  std::vector<std::string> out;
  auto emit = [&](size_t a, size_t b) {
    if (b > a) out.push_back(text.substr(off[a], off[b] - off[a]));
  };

  size_t i = 0;
  while (i < n) {
    uint32_t ch = cp[i];

    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    // Only U+0027 qualifies (the Python guards on `ch == "'"`), and the
    // contractions are ASCII, so ASCII-only case folding is equivalent here.
    if (ch == '\'' && i + 1 < n) {
      char low[3] = {0, 0, 0};
      for (size_t k = 0; k < 2 && i + 1 + k < n; ++k) {
        uint32_t c = cp[i + 1 + k];
        low[k] = (c < 128) ? static_cast<char>(tolower(static_cast<int>(c))) : '\1';
      }
      size_t hit = 0;
      for (const char* c : kContractions) {
        size_t len = strlen(c) - 1;  // characters after the quote
        if (strncmp(low, c + 1, len) == 0) {
          hit = len + 1;
          break;
        }
      }
      if (hit) {
        emit(i, i + hit);
        i += hit;
        continue;
      }
    }

    // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    size_t j = i;
    if (!is_crlf(ch) && !uni::is_L(ch) && !uni::is_N(ch)) j = i + 1;
    if (j < n && uni::is_LM(cp[j])) {
      size_t k = j;
      while (k < n && uni::is_LM(cp[k])) ++k;
      emit(i, k);
      i = k;
      continue;
    }

    // 3. \p{N} -- one digit at a time
    if (uni::is_N(ch)) {
      emit(i, i + 1);
      ++i;
      continue;
    }

    // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    j = (ch == ' ' && i + 1 < n) ? i + 1 : i;
    auto symbolic = [](uint32_t c) {
      return !uni::is_space(c) && !uni::is_LM(c) && !uni::is_N(c);
    };
    if (j < n && symbolic(cp[j])) {
      size_t k = j;
      while (k < n && symbolic(cp[k])) ++k;
      while (k < n && is_crlf(cp[k])) ++k;
      emit(i, k);
      i = k;
      continue;
    }

    // 5. \s*[\r\n]+ -- a whitespace run that contains a newline
    if (uni::is_space(ch)) {
      size_t k = i;
      while (k < n && uni::is_space(cp[k])) ++k;
      size_t last_nl = k;  // k == "none found"
      for (size_t p = k; p-- > i;) {
        if (is_crlf(cp[p])) {
          last_nl = p;
          break;
        }
      }
      if (last_nl != k) {
        emit(i, last_nl + 1);
        i = last_nl + 1;
        continue;
      }
      // 6. \s+(?!\S) -- trailing whitespace, nothing after it
      if (k == n) {
        emit(i, k);
        i = k;
        continue;
      }
      // 7. \s+, but leave the last space for the next token to claim (the regex
      //    is greedy and `behavior: Isolated` then re-splits)
      size_t stop = (k - 1 > i) ? k - 1 : k;
      emit(i, stop);
      i = stop;
      continue;
    }

    emit(i, i + 1);  // unreachable in practice; never drop input
    ++i;
  }
  return out;
}

std::vector<int> Tokenizer::bpe(const std::string& piece) const {
  auto cached = cache_.find(piece);
  if (cached != cache_.end()) return cached->second;

  std::vector<std::string> symbols;
  symbols.reserve(piece.size());
  for (unsigned char b : piece) symbols.push_back(byte_to_uni_[b]);

  while (symbols.size() > 1) {
    size_t best = 0;
    int best_rank = 0;
    bool found = false;
    for (size_t idx = 0; idx + 1 < symbols.size(); ++idx) {
      auto it = ranks_.find(merge_key(symbols[idx], symbols[idx + 1]));
      if (it != ranks_.end() && (!found || it->second < best_rank)) {
        best = idx;
        best_rank = it->second;
        found = true;
      }
    }
    if (!found) break;
    symbols[best] += symbols[best + 1];
    symbols.erase(symbols.begin() + static_cast<long>(best) + 1);
  }

  std::vector<int> ids;
  ids.reserve(symbols.size());
  for (const std::string& s : symbols) {
    auto it = vocab_.find(s);
    if (it != vocab_.end()) {
      ids.push_back(it->second);
      continue;
    }
    // No byte_fallback in this tokenizer: split the symbol into single chars.
    for (size_t p = 0; p < s.size();) {
      size_t start = p;
      uni::decode(s, p);
      auto c = vocab_.find(s.substr(start, p - start));
      if (c != vocab_.end()) ids.push_back(c->second);
    }
  }
  cache_.emplace(piece, ids);
  return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool allow_special) const {
  std::string norm = normalize_ ? uni::nfc(text) : text;

  // Split out the special tokens first; they are matched literally and never
  // reach the BPE. `false` marks ordinary text.
  std::vector<std::pair<std::string, bool>> chunks{{norm, false}};
  if (allow_special) {
    for (const std::string& tok : added_sorted_) {
      std::vector<std::pair<std::string, bool>> next;
      for (auto& c : chunks) {
        if (c.second || c.first.find(tok) == std::string::npos) {
          next.push_back(std::move(c));
          continue;
        }
        const std::string& s = c.first;
        size_t pos = 0;
        while (true) {
          size_t hit = s.find(tok, pos);
          if (hit == std::string::npos) break;
          if (hit > pos) next.emplace_back(s.substr(pos, hit - pos), false);
          next.emplace_back(tok, true);
          pos = hit + tok.size();
        }
        if (pos < s.size()) next.emplace_back(s.substr(pos), false);
      }
      chunks = std::move(next);
    }
  }

  std::vector<int> ids;
  for (const auto& c : chunks) {
    if (c.second) {
      ids.push_back(added_.at(c.first));
      continue;
    }
    for (const std::string& piece : pretokenize(c.first)) {
      std::vector<int> part = bpe(piece);
      ids.insert(ids.end(), part.begin(), part.end());
    }
  }
  return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string out;
  std::string buf;  // accumulated raw bytes, flushed at special-token boundaries
  for (int id : ids) {
    auto it = id_to_token_.find(id);
    if (it == id_to_token_.end()) continue;
    const std::string& tok = it->second;
    if (added_.count(tok)) {
      out += buf;
      buf.clear();
      if (!skip_special) out += tok;
      continue;
    }
    for (size_t p = 0; p < tok.size();) {
      uint32_t c = uni::decode(tok, p);
      auto b = uni_to_byte_.find(c);
      if (b != uni_to_byte_.end()) buf.push_back(static_cast<char>(b->second));
    }
  }
  out += buf;
  return out;
}
