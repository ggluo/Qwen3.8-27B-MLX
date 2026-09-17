#include "json.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "unicode.hpp"

namespace json {
namespace {

const Value kNull;

struct Parser {
  const char* p;
  const char* end;
  const char* begin;

  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("json: ") + what + " at offset " +
                            std::to_string(p - begin));
  }

  void ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }

  void expect(char c) {
    if (p >= end || *p != c) fail("unexpected character");
    ++p;
  }

  Value value() {
    ws();
    if (p >= end) fail("unexpected end of input");
    switch (*p) {
      case '{': return object();
      case '[': return array();
      case '"': {
        Value v;
        v.type = Type::Str;
        v.str = string();
        return v;
      }
      case 't':
        lit("true");
        return boolean(true);
      case 'f':
        lit("false");
        return boolean(false);
      case 'n':
        lit("null");
        return Value{};
      default: return number();
    }
  }

  void lit(const char* s) {
    size_t n = strlen(s);
    if (static_cast<size_t>(end - p) < n || memcmp(p, s, n) != 0) fail("bad literal");
    p += n;
  }

  Value boolean(bool b) {
    Value v;
    v.type = Type::Bool;
    v.boolean = b;
    return v;
  }

  Value number() {
    char* stop = nullptr;
    double d = strtod(p, &stop);
    if (stop == p) fail("bad number");
    p = stop;
    Value v;
    v.type = Type::Num;
    v.num = d;
    return v;
  }

  Value object() {
    expect('{');
    Value v;
    v.type = Type::Obj;
    ws();
    if (p < end && *p == '}') {
      ++p;
      return v;
    }
    while (true) {
      ws();
      std::string key = string();
      ws();
      expect(':');
      v.obj.emplace_back(std::move(key), value());
      ws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      expect('}');
      return v;
    }
  }

  Value array() {
    expect('[');
    Value v;
    v.type = Type::Arr;
    ws();
    if (p < end && *p == ']') {
      ++p;
      return v;
    }
    while (true) {
      v.arr.push_back(value());
      ws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      expect(']');
      return v;
    }
  }

  uint32_t hex4() {
    if (end - p < 4) fail("truncated \\u escape");
    uint32_t x = 0;
    for (int i = 0; i < 4; ++i) {
      char c = *p++;
      x <<= 4;
      if (c >= '0' && c <= '9') x |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') x |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') x |= static_cast<uint32_t>(c - 'A' + 10);
      else fail("bad hex digit");
    }
    return x;
  }

  std::string string() {
    expect('"');
    std::string out;
    // Fast path: copy runs with no escapes in one go. The vocab is 248k short
    // strings, most of them escape-free.
    const char* run = p;
    while (true) {
      if (p >= end) fail("unterminated string");
      unsigned char c = static_cast<unsigned char>(*p);
      if (c == '"') {
        out.append(run, static_cast<size_t>(p - run));
        ++p;
        return out;
      }
      if (c != '\\') {
        ++p;
        continue;
      }
      out.append(run, static_cast<size_t>(p - run));
      ++p;
      if (p >= end) fail("unterminated escape");
      char e = *p++;
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t cp = hex4();
          // Surrogate pair: 😀 is one codepoint, not two.
          if (cp >= 0xD800 && cp <= 0xDBFF && end - p >= 6 && p[0] == '\\' &&
              p[1] == 'u') {
            const char* save = p;
            p += 2;
            uint32_t lo = hex4();
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else {
              p = save;  // lone high surrogate; emit replacement below
            }
          }
          if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;  // unpaired surrogate
          uni::encode(cp, out);
          break;
        }
        default: fail("unknown escape");
      }
      run = p;
    }
  }
};

}  // namespace

const Value& Value::operator[](const std::string& key) const {
  if (type == Type::Obj) {
    for (const auto& kv : obj) {
      if (kv.first == key) return kv.second;
    }
  }
  return kNull;
}

const Value& Value::operator[](size_t i) const {
  if (type == Type::Arr && i < arr.size()) return arr[i];
  return kNull;
}

size_t Value::size() const {
  if (type == Type::Arr) return arr.size();
  if (type == Type::Obj) return obj.size();
  return 0;
}

std::string Value::as_str(const std::string& def) const {
  return type == Type::Str ? str : def;
}

double Value::as_num(double def) const { return type == Type::Num ? num : def; }

int64_t Value::as_int(int64_t def) const {
  return type == Type::Num ? static_cast<int64_t>(num) : def;
}

bool Value::as_bool(bool def) const {
  if (type == Type::Bool) return boolean;
  if (type == Type::Num) return num != 0;
  return def;
}

Value parse(const std::string& text) {
  Parser ps{text.data(), text.data() + text.size(), text.data()};
  Value v = ps.value();
  ps.ws();
  if (ps.p != ps.end) ps.fail("trailing content");
  return v;
}

Value parse_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    throw std::runtime_error("cannot open " + path + ": " + strerror(errno));
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n < 0) {
    fclose(f);
    throw std::runtime_error("cannot size " + path);
  }
  std::string buf(static_cast<size_t>(n), '\0');
  size_t got = n > 0 ? fread(&buf[0], 1, static_cast<size_t>(n), f) : 0;
  fclose(f);
  if (got != static_cast<size_t>(n)) throw std::runtime_error("short read on " + path);
  return parse(buf);
}

}  // namespace json
