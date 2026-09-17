// Minimal JSON reader: enough for config.json and tokenizer.json, nothing more.
//
// Objects keep INSERTION ORDER in a vector rather than hashing. The only large
// object here is tokenizer.json's 248k-entry vocab, and it is walked exactly
// once to build the tokenizer's own maps -- hashing it here would be pure waste.
// Member lookup is therefore linear, which is fine for the small config objects
// it is actually used on.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace json {

enum class Type { Null, Bool, Num, Str, Arr, Obj };

struct Value {
  Type type = Type::Null;
  bool boolean = false;
  double num = 0;
  std::string str;                                  // Str (already unescaped)
  std::vector<Value> arr;                           // Arr
  std::vector<std::pair<std::string, Value>> obj;   // Obj

  bool is_null() const { return type == Type::Null; }
  bool is_obj() const { return type == Type::Obj; }
  bool is_arr() const { return type == Type::Arr; }

  // Missing keys yield a static null Value, so `v["a"]["b"].is_null()` is safe.
  const Value& operator[](const std::string& key) const;
  const Value& operator[](size_t i) const;
  size_t size() const;

  // Typed reads with a default for missing/mistyped nodes.
  std::string as_str(const std::string& def = "") const;
  double as_num(double def = 0) const;
  int64_t as_int(int64_t def = 0) const;
  bool as_bool(bool def = false) const;
};

// Both throw std::runtime_error on malformed input or an unreadable file.
Value parse(const std::string& text);
Value parse_file(const std::string& path);

}  // namespace json
