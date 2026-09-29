// Serve mode. HTTP/1.1 on the wire, the OpenAI chat-completions API in the
// payload, down to the field names and the SSE framing, so a client needs no
// adapter to talk to it.
//
// Two things are worth knowing before reading on:
//
//   * Keep-alive is on for everything, including streaming replies. A streaming
//     reply is framed as chunked transfer-encoding, which is what gives it an
//     end marker; without that, the only way to say "done" would be to close the
//     connection.
//   * The session's KV cache is reused across requests. A request whose messages
//     extend the conversation the session is already holding appends only the
//     new turns; anything else resets and re-prefills. That is the same bet the
//     REPL makes, and it is why a chat client that echoes its history back does
//     not pay for that history again on every turn.
//
// Threading is the other thing to know, and the reason for ModelThread below: a
// connection thread never touches the model, it hands the request over.
#include "serve.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "json.hpp"

namespace {

// A body this size is a client that has lost the plot; refuse it before it is
// resident.
constexpr size_t kMaxRequest = 32u << 20;

// How long a connection may sit idle between requests. Without it, a client that
// opens a socket and says nothing parks a thread for the life of the process.
constexpr int kIdleTimeoutS = 600;

volatile std::sig_atomic_t g_stop = 0;

void on_sigint(int) {
  g_stop = 1;
  // Also stop the reply in flight, at its next round boundary -- the same place
  // Ctrl-C stops one in the REPL.
  request_interrupt();
}

// ---------------------------------------------------------------------------
// odds and ends
// ---------------------------------------------------------------------------

std::string lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

std::string trim(const std::string& s) {
  const size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Appends `s` as a JSON string. UTF-8 goes through untouched, which is what a
// JSON string is allowed to hold; only the characters that would break the
// syntax are escaped.
void put_str(std::string& out, const std::string& s) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char esc[8];
          snprintf(esc, sizeof(esc), "\\u%04x", c);
          out += esc;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

std::string quoted(const std::string& s) {
  std::string out;
  put_str(out, s);
  return out;
}

// <prefix>-<hex time><hex counter>: unique within the process, and obviously
// ours in a client's log.
std::string new_id(const char* prefix) {
  static std::atomic<unsigned long long> counter{0};
  char buf[64];
  snprintf(buf, sizeof(buf), "%s-%llx%06llx", prefix,
           static_cast<unsigned long long>(time(nullptr)),
           static_cast<unsigned long long>(++counter));
  return buf;
}

std::string usage_json(int prompt_n, int completion_n) {
  return "{\"prompt_tokens\":" + std::to_string(prompt_n) +
         ",\"completion_tokens\":" + std::to_string(completion_n) +
         ",\"total_tokens\":" + std::to_string(prompt_n + completion_n) + "}";
}

// ---------------------------------------------------------------------------
// the socket
// ---------------------------------------------------------------------------

bool send_all(int fd, const char* data, size_t n) {
  while (n > 0) {
    const ssize_t w = ::send(fd, data, n, 0);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;  // the peer is gone, so there is no one left to tell
    }
    data += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

bool send_all(int fd, const std::string& s) { return send_all(fd, s.data(), s.size()); }

int listen_socket(const std::string& host, int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error(std::string("socket: ") + strerror(errno));

  const int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (host == "localhost") {
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  } else if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    close(fd);
    throw std::runtime_error("--host takes an IPv4 address, or localhost; got '" + host + "'");
  }

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    const int e = errno;
    close(fd);
    throw std::runtime_error("cannot bind " + host + ":" + std::to_string(port) + ": " +
                             strerror(e));
  }
  if (listen(fd, 16) < 0) {
    const int e = errno;
    close(fd);
    throw std::runtime_error(std::string("listen: ") + strerror(e));
  }
  return fd;
}

// The port actually bound, which is not the one asked for when that was 0.
int bound_port(int fd) {
  sockaddr_in addr{};
  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) return 0;
  return ntohs(addr.sin_port);
}

// Everything a server-side socket wants set and a listening socket cannot pass
// down. SO_NOSIGPIPE matters most: without it, a client that disconnects
// mid-reply kills the process with SIGPIPE rather than failing a write.
void tune_client_socket(int fd) {
  const int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  struct timeval tv {
    kIdleTimeoutS, 0
  };
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

// ---------------------------------------------------------------------------
// responses
// ---------------------------------------------------------------------------

const char* status_text(int code) {
  switch (code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    default: return "Internal Server Error";
  }
}

bool send_response(int fd, int code, const char* type, const std::string& body) {
  std::string head = "HTTP/1.1 " + std::to_string(code) + " " + status_text(code) +
                     "\r\nContent-Type: " + type +
                     "\r\nContent-Length: " + std::to_string(body.size()) +
                     "\r\nConnection: keep-alive\r\n\r\n";
  return send_all(fd, head) && send_all(fd, body);
}

bool send_error(int fd, int code, const std::string& message) {
  const char* type = code == 401 ? "authentication_error"
                     : code >= 500 ? "server_error"
                                   : "invalid_request_error";
  const std::string body =
      "{\"error\":{\"message\":" + quoted(message) + ",\"type\":\"" + type +
      "\",\"code\":null}}";
  return send_response(fd, code, "application/json", body);
}

// The head of a streaming reply. There is no Content-Length: the body is framed
// as chunks, and the terminating chunk is what ends it.
bool send_sse_head(int fd) {
  return send_all(fd,
                  "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                  "Cache-Control: no-cache\r\nConnection: keep-alive\r\n"
                  "Transfer-Encoding: chunked\r\n\r\n");
}

bool send_chunk(int fd, const std::string& s) {
  char head[32];
  snprintf(head, sizeof(head), "%zx\r\n", s.size());
  return send_all(fd, head) && send_all(fd, s) && send_all(fd, "\r\n");
}

bool send_event(int fd, const std::string& data) {
  return send_chunk(fd, "data: " + data + "\n\n");
}

bool send_sse_end(int fd) {
  return send_event(fd, "[DONE]") && send_all(fd, "0\r\n\r\n");
}

// `{"id":..,"object":"chat.completion.chunk","created":..,"model":..,"choices":[`
std::string chunk_open(const std::string& id, long long created, const std::string& model) {
  return "{\"id\":" + quoted(id) + ",\"object\":\"chat.completion.chunk\",\"created\":" +
         std::to_string(created) + ",\"model\":" + quoted(model) + ",\"choices\":[";
}

bool send_delta(int fd, const std::string& id, long long created, const std::string& model,
                bool reasoning, const std::string& text) {
  std::string j = chunk_open(id, created, model);
  j += "{\"index\":0,\"delta\":{";
  j += reasoning ? "\"reasoning_content\":" : "\"content\":";
  put_str(j, text);
  j += "},\"finish_reason\":null}]}";
  return send_event(fd, j);
}

// One whole call in one delta: the whole of it is known the moment the closing
// tag arrives, and clients take the arguments as a string either way.
bool send_tool_delta(int fd, const std::string& id, long long created,
                     const std::string& model, size_t index, const ToolCall& call) {
  std::string j = chunk_open(id, created, model) +
                  "{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":" +
                  std::to_string(index) + ",\"id\":" + quoted(call.id) +
                  ",\"type\":\"function\",\"function\":{\"name\":" + quoted(call.name) +
                  ",\"arguments\":";
  put_str(j, call.arguments);
  j += "}}]},\"finish_reason\":null}]}";
  return send_event(fd, j);
}

bool send_finish(int fd, const std::string& id, long long created,
                 const std::string& model, const char* reason) {
  return send_event(fd, chunk_open(id, created, model) + "{\"index\":0,\"delta\":{}," +
                               "\"finish_reason\":\"" + reason + "\"}]}");
}

bool send_usage_chunk(int fd, const std::string& id, long long created,
                      const std::string& model, int prompt_n, int completion_n) {
  return send_event(fd, chunk_open(id, created, model) +
                           "],\"usage\":" + usage_json(prompt_n, completion_n) + "}");
}

// ---------------------------------------------------------------------------
// splitting the reply
// ---------------------------------------------------------------------------

// Hands the reply to two sinks, split at </think>: the reasoning block, then the
// answer. main.cpp's ThinkStyler does the same split to dim one of the halves;
// the API reports them in different fields instead, which is what clients that
// understand reasoning expect.
//
// The marker can arrive split across pieces, so hold back len(marker)-1 bytes
// until it is either found or the reply ends.
class ThinkSplitter {
 public:
  using Sink = std::function<void(bool reasoning, const std::string&)>;

  ThinkSplitter(bool thinking, Sink sink) : active_(thinking), sink_(std::move(sink)) {}

  void feed(const std::string& piece) {
    if (!active_) {
      sink_(false, piece);
      return;
    }
    buf_ += piece;
    const size_t i = buf_.find(kMark);
    if (i != std::string::npos) {
      const std::string rest = buf_.substr(i + strlen(kMark));
      const size_t nb = rest.find_first_not_of('\n');
      if (i > 0) sink_(true, buf_.substr(0, i));
      if (nb != std::string::npos) sink_(false, rest.substr(nb));
      buf_.clear();
      active_ = false;  // the rest of the reply is the answer
      return;
    }
    const size_t hold = strlen(kMark) - 1;
    if (buf_.size() > hold) {
      sink_(true, buf_.substr(0, buf_.size() - hold));
      buf_.erase(0, buf_.size() - hold);
    }
  }

  void finish() {
    if (!buf_.empty()) sink_(active_, buf_);
    buf_.clear();
    active_ = false;
  }

 private:
  static constexpr const char* kMark = "</think>";
  bool active_;
  Sink sink_;
  std::string buf_;
};

}  // namespace

// ---------------------------------------------------------------------------
// StopFilter
// ---------------------------------------------------------------------------

size_t utf8_complete(const char* s, size_t n) {
  // Walk back over the continuation bytes to whatever leads them.
  size_t i = n;
  for (size_t back = 0;
       i > 0 && back < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80;
       ++back) {
    --i;
  }
  if (i == 0) return n;  // continuation bytes with nothing leading them: nothing to fix
  const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
  const size_t need = lead < 0x80              ? 1
                      : (lead & 0xE0) == 0xC0  ? 2
                      : (lead & 0xF0) == 0xE0  ? 3
                      : (lead & 0xF8) == 0xF0  ? 4
                                               : 1;  // not a lead byte at all
  return need == 1 || n - (i - 1) >= need ? n : i - 1;
}

StopFilter::StopFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {
  for (const std::string& s : stops_) max_hold_ = std::max(max_hold_, s.size());
  max_hold_ = max_hold_ == 0 ? 0 : max_hold_ - 1;
}

std::string StopFilter::feed(const std::string& piece) {
  if (hit_) return "";
  buf_ += piece;

  // A stop string that has arrived in full ends the reply here.
  for (const std::string& s : stops_) {
    const size_t i = buf_.find(s);
    if (i != std::string::npos) {
      std::string out = buf_.substr(0, i);
      buf_.clear();
      hit_ = true;
      return out;
    }
  }

  // Otherwise hold back the tail that could still become one: at most len-1
  // bytes of each stop, which is what `max_hold_` already bounds.
  size_t keep = 0;
  for (size_t n = std::min(max_hold_, buf_.size()); n > 0; --n) {
    bool prefix = false;
    for (const std::string& s : stops_) {
      if (s.compare(0, n, buf_, buf_.size() - n, n) == 0) {
        prefix = true;
        break;
      }
    }
    if (prefix) {
      keep = n;
      break;
    }
  }
  const size_t cut = buf_.size() - keep;
  std::string out = buf_.substr(0, cut);
  buf_.erase(0, cut);
  return out;
}

std::string StopFilter::finish() {
  if (hit_) return "";
  std::string out = std::move(buf_);
  buf_.clear();
  return out;
}

// ---------------------------------------------------------------------------
// function calls
// ---------------------------------------------------------------------------

json::Value ToolCallParser::value_of(const std::string& tool, const std::string& key,
                                     const std::string& text) const {
  json::Value v;
  const std::string t = trim(text);
  std::string type;
  auto tool_it = types_.find(tool);
  if (tool_it != types_.end()) {
    auto type_it = tool_it->second.find(key);
    if (type_it != tool_it->second.end()) type = type_it->second;
  }

  const bool numeric = type == "number" || type == "integer";
  const bool boolean = type == "boolean";
  const bool structured = type == "object" || type == "array" ||
                          (!t.empty() && (t[0] == '{' || t[0] == '['));
  if (structured) {
    try {
      return json::parse(t);
    } catch (const std::exception&) {
      // Not JSON after all: fall through and keep it as the text it is.
    }
  } else if (numeric || boolean) {
    // Only if it really is one: a "5 " that the model meant as a string stays one.
    try {
      const json::Value parsed = json::parse(t);
      if (parsed.type == json::Type::Num || parsed.type == json::Type::Bool) return parsed;
    } catch (const std::exception&) {
    }
  }
  v.type = json::Type::Str;
  v.str = text;
  return v;
}

ToolCall ToolCallParser::parse_block(const std::string& raw) {
  ToolCall call;
  const std::string body = trim(raw);
  if (body.empty()) return call;

  if (body[0] == '{') {
    // {"name": ..., "arguments": {...}}, with or without a "function" wrapper.
    try {
      const json::Value v = json::parse(body);
      const json::Value& fn = v["function"].is_obj() ? v["function"] : v;
      call.name = fn["name"].as_str();
      const json::Value& args = fn["arguments"];
      call.arguments = args.type == json::Type::Str ? args.str : json::dump(args);
    } catch (const std::exception&) {
      call.name.clear();
    }
  } else {
    const std::string fn_tag = "<function=";
    const size_t fn_at = body.find(fn_tag);
    if (fn_at == std::string::npos) return call;
    const size_t name_end = body.find('>', fn_at);
    if (name_end == std::string::npos) return call;
    call.name = trim(body.substr(fn_at + fn_tag.size(), name_end - fn_at - fn_tag.size()));

    json::Value args;
    args.type = json::Type::Obj;
    const std::string param_tag = "<parameter=";
    const std::string param_end = "</parameter>";
    for (size_t at = name_end + 1;;) {
      const size_t p_at = body.find(param_tag, at);
      if (p_at == std::string::npos) break;
      const size_t p_gt = body.find('>', p_at);
      if (p_gt == std::string::npos) break;
      const std::string key =
          trim(body.substr(p_at + param_tag.size(), p_gt - p_at - param_tag.size()));
      const size_t v_end = body.find(param_end, p_gt);
      if (v_end == std::string::npos) break;
      // The template puts the value on its own lines; the newlines are not part
      // of it.
      std::string value = body.substr(p_gt + 1, v_end - p_gt - 1);
      if (!value.empty() && value.front() == '\n') value.erase(0, 1);
      if (!value.empty() && value.back() == '\n') value.pop_back();
      args.obj.emplace_back(key, value_of(call.name, key, value));
      at = v_end + param_end.size();
    }
    call.arguments = json::dump(args);
  }

  if (!call.name.empty()) call.id = new_id("call");
  return call;
}

ToolCallParser::Piece ToolCallParser::feed(const std::string& piece) {
  Piece out;
  buf_ += piece;

  for (;;) {
    const size_t start = buf_.find(kOpen);
    if (start == std::string::npos) {
      // Nothing that looks like a call here. Hold back a tail that could still
      // turn into the opening tag, and pass the rest on.
      size_t keep = 0;
      for (size_t n = std::min(buf_.size(), strlen(kOpen) - 1); n > 0; --n) {
        if (std::string(kOpen).compare(0, n, buf_, buf_.size() - n, n) == 0) {
          keep = n;
          break;
        }
      }
      out.text += buf_.substr(0, buf_.size() - keep);
      buf_.erase(0, buf_.size() - keep);
      return out;
    }

    const size_t end = buf_.find(kClose, start);
    if (end == std::string::npos) {
      // The call is still arriving: everything before it is content.
      out.text += buf_.substr(0, start);
      buf_.erase(0, start);
      return out;
    }

    out.text += buf_.substr(0, start);
    const size_t body_at = start + strlen(kOpen);
    const ToolCall call = parse_block(buf_.substr(body_at, end - body_at));
    if (call.name.empty()) {
      // Not a call after all. Keep the text, the way it arrived.
      out.text += buf_.substr(start, end + strlen(kClose) - start);
    } else {
      out.calls.push_back(call);
    }
    buf_.erase(0, end + strlen(kClose));
  }
}

ToolCallParser::Piece ToolCallParser::finish() {
  Piece out;
  // A call that never completed is not a call: send it as the text it is, rather
  // than swallowing what the model wrote.
  out.text = std::move(buf_);
  buf_.clear();
  return out;
}


// ---------------------------------------------------------------------------
// the request
// ---------------------------------------------------------------------------

namespace {

// The bytes of one picture, from the URL a client gave for it.
std::string image_bytes(const std::string& url) {
  if (url.rfind("data:", 0) == 0) {
    const size_t comma = url.find(',');
    if (comma == std::string::npos) throw std::runtime_error("malformed data: URL for an image");
    const std::string meta = url.substr(5, comma - 5);  // e.g. "image/png;base64"
    if (meta.size() < 7 || meta.compare(meta.size() - 7, 7, ";base64") != 0) {
      throw std::runtime_error("image data: URLs must be base64 (data:<type>;base64,...)");
    }
    std::string bytes = image::base64_decode(url.substr(comma + 1));
    if (bytes.empty()) throw std::runtime_error("empty image in data: URL");
    return bytes;
  }
  if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
    throw std::runtime_error("image URLs are not fetched -- this server never touches the "
                             "network; send the picture inline as a data:<type>;base64,... URL");
  }
  if (url.rfind("file:", 0) == 0) {
    throw std::runtime_error("file: image URLs are refused: they would let any client read "
                             "pictures off the server's disk; send the bytes as a data: URL");
  }
  throw std::runtime_error("unsupported image URL; send the picture as a data:<type>;base64,... URL");
}

}  // namespace

std::string flatten_content(const json::Value& content, std::vector<std::string>& images) {
  if (content.type == json::Type::Str) return content.str;
  if (content.is_null()) return "";
  if (!content.is_arr()) {
    throw std::runtime_error("message content must be a string or an array of parts");
  }
  std::string text;
  for (const json::Value& part : content.arr) {
    const std::string type = part["type"].as_str();
    if (type == "text" || type == "input_text") {
      text += part["text"].as_str();
    } else if (type == "image_url" || type == "input_image" || type == "image") {
      // {"image_url": {"url": ...}} is the API's shape; a bare string there, and
      // {"type": "image", "image": ...}, are what other clients send.
      const json::Value& iu = part["image_url"];
      std::string url = iu.is_obj() ? iu["url"].as_str() : iu.as_str();
      if (url.empty()) url = part["image"].as_str();
      if (url.empty()) throw std::runtime_error("an image part has no url");
      images.push_back(image_bytes(url));
      text += kImagePlaceholder;
    }
  }
  return text;
}

std::string HttpRequest::header(const std::string& name) const {
  const std::string want = lower(name);
  for (const auto& h : headers) {
    if (h.first == want) return h.second;
  }
  return "";
}

bool parse_request(std::string& buf, HttpRequest& out) {
  // Headers end at the first blank line, CRLF or LF.
  const size_t crlf = buf.find("\r\n\r\n");
  const size_t lf = buf.find("\n\n");
  size_t sep = 0, sep_len = 0;
  if (crlf != std::string::npos && (lf == std::string::npos || crlf <= lf)) {
    sep = crlf;
    sep_len = 4;
  } else if (lf != std::string::npos) {
    sep = lf;
    sep_len = 2;
  } else {
    return false;  // no blank line yet
  }

  HttpRequest r;
  const size_t line_end = buf.find('\n');
  std::string line = buf.substr(0, line_end);
  if (!line.empty() && line.back() == '\r') line.pop_back();

  const size_t sp1 = line.find(' ');
  const size_t sp2 = line.rfind(' ');
  if (sp1 == std::string::npos || sp2 == sp1 || sp2 + 1 >= line.size()) {
    throw std::runtime_error("malformed request line: " + trim(line));
  }
  r.method = line.substr(0, sp1);
  r.version = line.substr(sp2 + 1);
  const std::string url = line.substr(sp1 + 1, sp2 - sp1 - 1);
  const size_t q = url.find('?');
  r.target = url.substr(0, q);
  r.query = q == std::string::npos ? "" : url.substr(q + 1);
  if (r.target.empty() || r.target[0] != '/') {
    throw std::runtime_error("malformed request target: " + url);
  }

  // The header block runs from the end of the request line up to the blank
  // line; on a CRLF request the last header's own newline is the first half of
  // the separator, so the block ends on its '\r'.
  const std::string head =
      buf.substr(line_end + 1, sep > line_end + 1 ? sep - line_end - 1 : 0);
  for (size_t pos = 0; pos < head.size();) {
    const size_t e = head.find('\n', pos);
    const size_t stop = e == std::string::npos ? head.size() : e;
    std::string h = head.substr(pos, stop - pos);
    pos = stop + 1;
    if (!h.empty() && h.back() == '\r') h.pop_back();
    if (h.empty()) continue;
    const size_t c = h.find(':');
    if (c == std::string::npos) throw std::runtime_error("malformed header: " + trim(h));
    r.headers.emplace_back(lower(trim(h.substr(0, c))), trim(h.substr(c + 1)));
  }

  const std::string te = r.header("transfer-encoding");
  if (!te.empty()) {
    throw std::runtime_error("chunked request bodies are not supported (" + te + ")");
  }

  size_t want = 0;
  const std::string len = r.header("content-length");
  if (!len.empty()) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long v = strtoull(len.c_str(), &end, 10);
    if (errno != 0 || end == len.c_str() || *end != '\0') {
      throw std::runtime_error("bad Content-Length: " + len);
    }
    if (v > kMaxRequest) throw std::runtime_error("request body too large");
    want = static_cast<size_t>(v);
  }

  const size_t body_at = sep + sep_len;
  if (buf.size() < body_at + want) return false;  // body still arriving

  r.body = buf.substr(body_at, want);
  buf.erase(0, body_at + want);
  out = std::move(r);
  return true;
}

// ---------------------------------------------------------------------------
// the server
// ---------------------------------------------------------------------------

namespace {

// What a /v1/chat/completions request asks for, pulled out of the JSON.
struct ChatRequest {
  std::vector<Message> msgs;        // every message except the system ones
  std::string system;               // the system turn, and the only one there is
  std::vector<ToolSpec> tools;      // as sent, in order
  ToolCallParser::Types tool_types;  // for reading the model's arguments back
  std::vector<std::string> stops;
  bool think = false;
  bool stream = false;
  bool include_usage = false;
  float temp = 0.7f;
  float top_p = 0.95f;
  int top_k = 20;
  int max_tokens = 1024;
  unsigned dump_no = 0;  // --dump's file number for this request, 0 for none
};

// The template knows four roles. `developer` is the API's newer name for the
// system prompt and lands in the same block.
std::string render_role(const std::string& role) {
  if (role == "system" || role == "developer") return "system";
  if (role == "user" || role == "assistant" || role == "tool") return role;
  throw std::runtime_error("unsupported message role: '" + role + "'");
}

// An assistant turn's tool_calls, as the client echoed them back.
std::vector<ToolCall> parse_tool_calls(const json::Value& calls) {
  std::vector<ToolCall> out;
  if (!calls.is_arr()) return out;
  for (const json::Value& c : calls.arr) {
    const json::Value& fn = c["function"].is_obj() ? c["function"] : c;
    ToolCall call;
    call.id = c["id"].as_str();
    call.name = fn["name"].as_str();
    const json::Value& args = fn["arguments"];
    // OpenAI sends the arguments as a JSON string; a client that sends them as
    // an object means the same thing.
    call.arguments = args.type == json::Type::Str ? args.str : json::dump(args);
    if (!call.name.empty()) out.push_back(std::move(call));
  }
  return out;
}

// ---------------------------------------------------------------------------
// dumping
// ---------------------------------------------------------------------------

// One request and its reply, on disk, numbered in the order they arrived --
// which is the order of the log lines too, since replies are one at a time.
//
// Off unless --dump names a directory. It is the one thing in this binary that
// writes a conversation to disk, and it exists because the alternative is
// guessing why a prompt is 17k tokens long.

bool write_file(const std::string& path, const std::string& text) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
  fclose(f);
  return ok;
}

std::string dump_path(const std::string& dir, unsigned n, const char* what) {
  char name[32];
  snprintf(name, sizeof(name), "/%04u-%s", n, what);
  return dir + name;
}

void dump_text(const std::string& dir, unsigned n, const char* what,
               const std::string& text) {
  if (dir.empty() || n == 0) return;
  if (!write_file(dump_path(dir, n, what), text)) {
    fprintf(stderr, "  cannot write %s under %s\n", what, dir.c_str());
  }
}

// What the prompt for this request was made of. Bytes, not characters: the
// system prompts these clients send are full of emoji and CJK, and the number
// that has to reconcile with prompt.txt is the byte count.
//
// `from` is the first message this request actually carried. Everything before
// it is already in the session's context and was not sent again, so listing it
// would describe a prompt that never existed -- which is exactly the sort of
// thing this dump is meant to settle. `reused` is how many tokens the context
// already held.
std::string dump_breakdown(const ChatRequest& cr, size_t from, size_t reused) {
  size_t tools_bytes = 0;
  for (const ToolSpec& t : cr.tools) tools_bytes += t.json.size();
  std::string out = "{\"head_in_context\":" +
                    std::string(from > 0 ? "true" : "false") +
                    ",\"system_bytes\":" + std::to_string(cr.system.size()) +
                    ",\"tools_bytes\":" + std::to_string(tools_bytes) +
                    ",\"reused_tokens\":" + std::to_string(reused) + ",\"messages\":[";
  for (size_t i = from; i < cr.msgs.size(); ++i) {
    if (i > from) out += ',';
    out += "{\"role\":" + quoted(cr.msgs[i].role) +
           ",\"content_bytes\":" + std::to_string(cr.msgs[i].content.size());
    if (!cr.msgs[i].images.empty()) {
      size_t img_bytes = 0;
      for (const std::string& b : cr.msgs[i].images) img_bytes += b.size();
      out += ",\"images\":" + std::to_string(cr.msgs[i].images.size()) +
             ",\"image_bytes\":" + std::to_string(img_bytes);
    }
    out += "}";
  }
  return out + "]}";
}

// The parameters a tool declares, so that a value the schema calls a number can
// come back as one. Only the top level is read: that is what a model fills in.
ToolCallParser::Types parse_tool_types(const std::vector<ToolSpec>& tools,
                                       const std::vector<json::Value>& schemas) {
  ToolCallParser::Types types;
  for (size_t i = 0; i < schemas.size() && i < tools.size(); ++i) {
    const json::Value& fn = schemas[i]["function"].is_obj() ? schemas[i]["function"]
                                                            : schemas[i];
    const std::string name = fn["name"].as_str();
    if (name.empty()) continue;
    const json::Value& props = fn["parameters"]["properties"];
    if (!props.is_obj()) continue;
    std::map<std::string, std::string>& params = types[name];
    for (const auto& kv : props.obj) params[kv.first] = kv.second["type"].as_str();
  }
  return types;
}

ChatRequest parse_chat(const json::Value& body, const ServeOptions& opt) {
  const json::Value& messages = body["messages"];
  if (!messages.is_arr() || messages.arr.empty()) {
    throw std::runtime_error("messages must be a non-empty array");
  }

  ChatRequest cr;
  cr.msgs.reserve(messages.arr.size());
  std::string system = trim(opt.system);
  for (const json::Value& m : messages.arr) {
    Message msg;
    msg.role = render_role(m["role"].as_str());
    msg.content = flatten_content(m["content"], msg.images);
    if (!msg.images.empty() && msg.role != "user") {
      // the template raises on a picture anywhere but a user turn
      throw std::runtime_error("only user messages can contain images, not " + msg.role +
                               " messages");
    }
    msg.reasoning = m["reasoning_content"].as_str();
    msg.tool_calls = parse_tool_calls(m["tool_calls"]);
    // The template has one system block, at the head, so the system messages are
    // collected into it rather than left in the message list.
    if (msg.role == "system") {
      if (msg.content.empty()) continue;
      if (!system.empty()) system += "\n\n";
      system += trim(msg.content);
      continue;
    }
    cr.msgs.push_back(std::move(msg));
  }
  cr.system = system;

  if (cr.msgs.empty()) throw std::runtime_error("messages must not be only a system prompt");
  // An agent loop hands back a tool result and waits for the next step, so the
  // last message is a user turn or a tool's answer to one.
  const std::string& last = cr.msgs.back().role;
  if (last != "user" && last != "tool") {
    throw std::runtime_error("the last message must be from the user or a tool: the "
                             "template leaves an assistant turn open to be generated");
  }

  const json::Value& tools = body["tools"];
  if (tools.is_arr() && !tools.arr.empty()) {
    std::vector<json::Value> schemas;
    for (const json::Value& t : tools.arr) {
      schemas.push_back(t);
      cr.tools.push_back(ToolSpec{json::dump(t)});
    }
    cr.tool_types = parse_tool_types(cr.tools, schemas);
  }

  cr.temp = static_cast<float>(body["temperature"].as_num(0.7));
  cr.top_p = static_cast<float>(body["top_p"].as_num(0.95));
  cr.top_k = static_cast<int>(body["top_k"].as_int(20));
  cr.stream = body["stream"].as_bool(false);
  cr.include_usage = body["stream_options"]["include_usage"].as_bool(false);
  // `max_tokens` is the original spelling, `max_completion_tokens` the current
  // one; accept either.
  cr.max_tokens = static_cast<int>(body["max_tokens"].as_int(
      body["max_completion_tokens"].as_int(1024)));
  if (cr.max_tokens < 1) cr.max_tokens = 1;
  // Thinking is off unless asked for: `think` is this binary's own spelling, and
  // the other is what the vLLM/transformers Qwen templates use.
  cr.think = body["think"].as_bool(
      body["chat_template_kwargs"]["enable_thinking"].as_bool(opt.think));

  const json::Value& stop = body["stop"];
  if (stop.type == json::Type::Str && !stop.str.empty()) {
    cr.stops.push_back(stop.str);
  } else if (stop.is_arr()) {
    for (const json::Value& s : stop.arr) {
      if (s.type == json::Type::Str && !s.str.empty()) cr.stops.push_back(s.str);
    }
  }
  return cr;
}

// The conversation. Everything here is touched only by the model thread.
struct Server {
  Server(const Model& m, const Tokenizer& tk, const ServeOptions& options)
      : opt(options),
        model(m),
        session(m, tk, /*system=*/std::string(), opt.spec, opt.draft) {}

  const ServeOptions& opt;
  const Model& model;

  Session session;
  std::vector<Message> held;      // the messages the session's context already covers
  std::string held_system;        // ... and the system prompt it covers them with
  std::vector<ToolSpec> held_tools;  // ... and the tools
  bool held_ok = false;           // ... and covers exactly, so a request can extend it

  bool handle(const ChatRequest& cr, int fd);
};

// MLX ties every array to the stream of the thread that built it, and a graph
// built on one thread cannot be evaluated on another -- it fails with "There is
// no Stream(gpu, N) in current thread". The weights are one big graph, so the
// model has to be loaded on the thread that runs it, and every request has to
// come to that thread. That is this class: one thread that loads the model and
// then generates, one reply at a time, while the connection threads do the HTTP
// around it.
//
// The thread is never joined: serve() ends with _exit, so it dies with the
// process, like everything else it holds.
class ModelThread {
 public:
  ModelThread(std::string dir, const ServeOptions& opt)
      : dir_(std::move(dir)), opt_(opt), started_(time(nullptr)), thread_([this] { run(); }) {
    ready_future_ = ready_.get_future();
  }

  // Blocks until the model is up, or exits the process if it cannot be loaded.
  void wait_ready() { ready_future_.wait(); }

  // Runs one request on the model thread and writes the whole reply, headers
  // and all. Returns false if the connection should be closed.
  bool submit(const ChatRequest& cr, int fd) {
    std::promise<bool> done;
    std::future<bool> result = done.get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      job_ = Job{&cr, fd, &done};
      has_job_ = true;
    }
    cv_.notify_one();
    return result.get();
  }

  const ServeOptions& opt() const { return opt_; }
  long long started() const { return started_; }

  // The next --dump file number. Handed out in the order requests arrive, which
  // is the order the log lines come in too.
  unsigned next_dump_no() { return ++dump_seq_; }

 private:
  struct Job {
    const ChatRequest* cr;
    int fd;
    std::promise<bool>* done;
  };

  void run();

  std::string dir_;
  ServeOptions opt_;
  const long long started_;

  std::atomic<unsigned> dump_seq_{0};

  std::mutex mu_;
  std::condition_variable cv_;
  Job job_{};
  bool has_job_ = false;

  std::promise<void> ready_;
  std::future<void> ready_future_;
  std::thread thread_;
};

// The checkpoint's chat template as text: the .jinja file if it has one, and
// otherwise the copy embeddings carry inside tokenizer_config.json.
std::string chat_template_of(const std::string& dir) {
  FILE* f = fopen((dir + "/chat_template.jinja").c_str(), "rb");
  if (f) {
    std::string text;
    char buf[8192];
    for (size_t n = fread(buf, 1, sizeof(buf), f); n > 0; n = fread(buf, 1, sizeof(buf), f)) {
      text.append(buf, n);
    }
    fclose(f);
    if (!text.empty()) return text;
  }
  try {
    return json::parse_file(dir + "/tokenizer_config.json")["chat_template"].as_str();
  } catch (const std::exception&) {
    return "";
  }
}

void ModelThread::run() {
  // Nothing here may touch another thread's arrays, so the tokenizer, the
  // weights and every forward pass after them are built and used right here.
  try {
    Tokenizer tk(dir_ + "/tokenizer.json");
    Model model = load_model(dir_, /*verbose=*/false);

    // The prompt is written in C++ (session.cpp) rather than interpreted from
    // the checkpoint's Jinja, so this is where the two are held against each
    // other: a checkpoint whose template says something else must not be served
    // as if it said this.
    const std::string mismatch = template_mismatch(chat_template_of(dir_));
    if (!mismatch.empty()) fprintf(stderr, "  warning: %s\n", mismatch.c_str());

    Server server(model, tk, opt_);
    ready_.set_value();

    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return has_job_; });
        job = job_;
        has_job_ = false;
      }
      try {
        job.done->set_value(server.handle(*job.cr, job.fd));
      } catch (const std::exception& e) {
        // Nothing sensible can be written now: the reply may be half sent.
        fprintf(stderr, "  generation failed: %s\n", e.what());
        job.done->set_value(false);
      }
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "cannot serve: %s\n", e.what());
    fflush(nullptr);
    _exit(1);
  }
}

bool Server::handle(const ChatRequest& cr, int fd) {
  const std::string id = new_id("chatcmpl");
  const long long created = time(nullptr);
  const auto t0 = std::chrono::steady_clock::now();

  // Clear anything a previous stop sequence, or a Ctrl-C, left set: the decode
  // loop would otherwise stop before its first token.
  take_interrupt();

  // A request that is the conversation the session holds plus more -- which is
  // what every turn of a chat looks like -- only appends. Anything else (a new
  // conversation, another client, an edited history) starts over. The system
  // prompt and the tools are part of the head the session already holds, so a
  // change to either has to start over too.
  const bool same_head = held_system == cr.system && held_tools == cr.tools;
  const bool append =
      held_ok && same_head && cr.msgs.size() > held.size() &&
      std::equal(held.begin(), held.end(), cr.msgs.begin(),
                 [](const Message& a, const Message& b) {
                   // the pictures too: the same words about a different picture
                   // are a different conversation
                   return a.role == b.role && a.content == b.content && a.images == b.images;
                 });
  const size_t from = append ? held.size() : 0;

  const std::string prompt = append ? render_tail(cr.msgs, held.size(), cr.think)
                                    : render_chat(cr.msgs, cr.system, cr.think, cr.tools);

  // What actually went to the model. A cold render is the whole conversation,
  // system prompt and tools included; a continued one is the tail, and it is
  // named as a tail so that nobody reads half a conversation as a whole prompt.
  dump_text(opt.dump_dir, cr.dump_no, from > 0 ? "prompt-tail.txt" : "prompt.txt", prompt);

  // The pictures this prompt carries: only those in the messages being fed now,
  // since the earlier ones are already in the context. Decoded here, before
  // anything is sent, so a bad picture is a plain 400 rather than a stream that
  // breaks off -- and here rather than on the connection thread, because the
  // preprocessing builds MLX arrays and must run on the model's thread.
  std::vector<image::Image> pictures;
  {
    size_t n = 0;
    for (size_t i = from; i < cr.msgs.size(); ++i) n += cr.msgs[i].images.size();
    size_t pads = 0;
    for (size_t at = prompt.find("<|image_pad|>"); at != std::string::npos;
         at = prompt.find("<|image_pad|>", at + 1)) {
      ++pads;
    }
    std::string bad;
    if (n > 0 && !model.has_vision()) {
      bad = "this model has no vision tower; it cannot take images";
    } else if (pads != n) {
      bad = "the messages contain a literal <|image_pad|> token";
    }
    for (size_t i = from; i < cr.msgs.size() && bad.empty(); ++i) {
      for (size_t k = 0; k < cr.msgs[i].images.size() && bad.empty(); ++k) {
        try {
          pictures.push_back(image::preprocess(image::decode(cr.msgs[i].images[k]),
                                               image::kMinPixels, opt.max_pixels));
        } catch (const std::exception& e) {
          bad = "image " + std::to_string(pictures.size() + 1) + ": " + e.what();
        }
      }
    }
    if (!bad.empty()) {
      // Refused before the session is touched -- including the reset below -- so
      // it still holds exactly `held`, and the next good request continues it.
      dump_text(opt.dump_dir, cr.dump_no, "response.json", "{\"error\":" + quoted(bad) + "}");
      fprintf(stderr, "  refused: %s\n", bad.c_str());
      return send_error(fd, 400, bad);
    }
  }

  if (!append) {
    session.reset();
    held.clear();
    held_ok = false;
  }
  // What the session held before this turn: the difference between this and the
  // prompt is what the KV cache reuse bought.
  const size_t reused = session.context_tokens();

  // A streaming reply opens with the assistant role, the way the API does; some
  // clients will not accept a first delta that is only content.
  if (cr.stream &&
      !(send_sse_head(fd) &&
        send_event(fd, chunk_open(id, created, opt.model_name) +
                           "{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
                           "\"finish_reason\":null}]}"))) {
    return false;
  }

  // Everything that reaches the client passes through here, so this is also
  // where a character split across two pieces is put back together: the tail
  // that is not yet a whole character waits in `carry` for the rest of itself.
  std::string content, reasoning;  // the non-streaming reply, accumulated
  std::string carry;
  bool carry_reasoning = false;
  bool ok = true;
  auto emit = [&](bool is_reasoning, const std::string& text) {
    if (!ok || text.empty()) return;
    if (carry.empty()) carry_reasoning = is_reasoning;
    const std::string whole = carry + text;
    const size_t n = utf8_complete(whole.data(), whole.size());
    carry.assign(whole, n, std::string::npos);
    if (n == 0) return;
    const std::string ready = whole.substr(0, n);
    // Kept whether or not it is streamed: the non-streaming reply needs it, and
    // so does the dump of a streamed one.
    if (is_reasoning) {
      reasoning += ready;
    } else {
      content += ready;
    }
    if (cr.stream) {
      ok = send_delta(fd, id, created, opt.model_name, is_reasoning, ready);
      // The client hung up: stop generating rather than fill a socket nobody is
      // reading, at the next round boundary.
      if (!ok) request_interrupt();
    }
  };

  StopFilter stops(cr.stops);
  ToolCallParser tools(cr.tool_types);
  std::vector<ToolCall> calls;
  auto take_calls = [&](ToolCallParser::Piece piece) {
    if (!piece.text.empty()) {
      const std::string safe = stops.feed(piece.text);
      if (!safe.empty()) emit(false, safe);
      // What follows a stop sequence is not worth generating: ask the decode loop
      // to stop at its next round boundary, where token list and caches agree.
      if (stops.hit()) request_interrupt();
    }
    for (const ToolCall& call : piece.calls) {
      const size_t index = calls.size();
      calls.push_back(call);
      if (cr.stream) ok = ok && send_tool_delta(fd, id, created, opt.model_name,
                                                index, call);
    }
  };

  ThinkSplitter splitter(cr.think, [&](bool is_reasoning, const std::string& text) {
    if (is_reasoning) {
      emit(true, text);  // a stop sequence in the reasoning block would be the
      return;            // model quoting one, not using it
    }
    take_calls(tools.feed(text));
  });

  session.turn_prompt(prompt, cr.temp, cr.top_p, cr.top_k, cr.max_tokens,
                      [&](const std::string& piece) { splitter.feed(piece); }, pictures);
  splitter.finish();
  emit(false, stops.finish());
  // A reply that stopped mid-character gets the replacement character for a
  // tail, rather than leaving a JSON string holding half of one.
  if (!carry.empty()) emit(carry_reasoning, "\xEF\xBF\xBD");
  take_interrupt();  // ours, from the stop sequence above

  if (g_stop || !ok) return false;  // shutting down, or the client is gone

  const Stats& st = session.last();
  const char* finish = !calls.empty()                  ? "tool_calls"
                       : st.n >= cr.max_tokens         ? "length"
                                                       : "stop";
  held = cr.msgs;
  held_system = cr.system;
  held_tools = cr.tools;
  // A reply cut at a stop sequence leaves the session holding tokens the client
  // never saw, so the next request cannot simply append to it.
  held_ok = !stops.hit();

  if (cr.stream) {
    ok = send_finish(fd, id, created, opt.model_name, finish);
    if (ok && cr.include_usage) {
      ok = send_usage_chunk(fd, id, created, opt.model_name, st.prompt_n, st.n);
    }
    if (ok) ok = send_sse_end(fd);
  } else {
    std::string body = "{\"id\":" + quoted(id) + ",\"object\":\"chat.completion\"" +
                       ",\"created\":" + std::to_string(created) +
                       ",\"model\":" + quoted(opt.model_name) + ",\"choices\":[{\"index\":0," +
                       "\"message\":{\"role\":\"assistant\",\"content\":";
    put_str(body, content);
    if (cr.think) {
      body += ",\"reasoning_content\":";
      put_str(body, reasoning);
    }
    if (!calls.empty()) {
      body += ",\"tool_calls\":[";
      for (size_t i = 0; i < calls.size(); ++i) {
        if (i) body += ',';
        body += "{\"index\":" + std::to_string(i) + ",\"id\":" + quoted(calls[i].id) +
                ",\"type\":\"function\",\"function\":{\"name\":" + quoted(calls[i].name) +
                ",\"arguments\":";
        put_str(body, calls[i].arguments);
        body += "}}";
      }
      body += "]";
    }
    body += "},\"finish_reason\":\"" + std::string(finish) + "\"}],\"usage\":" +
            usage_json(st.prompt_n, st.n) + "}";
    ok = send_response(fd, 200, "application/json", body);
  }

  // The split matters when a client feels slow: prefill is what the cache reuse
  // saves, decode is what the model costs, and they move independently.
  const double dt =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const double decode_t = st.dt > 0 ? st.dt : 1e-9;
  char pics[64] = "";
  if (st.images) {
    snprintf(pics, sizeof(pics), " [%d picture%s, %.1fs vision]", st.images,
             st.images == 1 ? "" : "s", st.vision_t);
  }
  fprintf(stderr, "  %s %d in + %d out in %.1fs (prefill %.1fs %.0f tok/s,"
                  " decode %.1fs %.1f tok/s)%s%s%s\n",
          append ? "continued" : "prefilled", st.prompt_n, st.n, dt, st.prefill_t,
          st.prompt_n / (st.prefill_t > 0 ? st.prefill_t : 1e-9), decode_t,
          st.n / decode_t, pics, cr.think ? " [think]" : "",
          stops.hit() ? " [stop sequence]"
                      : (calls.empty() ? "" : " [called a tool]"));

  if (cr.dump_no) {
    std::string out = "{\"continued\":" + std::string(append ? "true" : "false") +
                      ",\"prompt\":" + dump_breakdown(cr, from, reused) +
                      ",\"usage\":" + usage_json(st.prompt_n, st.n) +
                      ",\"finish_reason\":" + quoted(finish) + ",\"timings\":{\"prefill_s\":" +
                      std::to_string(st.prefill_t) + ",\"decode_s\":" + std::to_string(st.dt) +
                      ",\"total_s\":" + std::to_string(dt) + "},\"message\":{\"role\":\"assistant\"";
    if (cr.stream) {
      out += ",\"streamed\":true";
    }
    out += ",\"content\":";
    put_str(out, content);
    if (cr.think) {
      out += ",\"reasoning_content\":";
      put_str(out, reasoning);
    }
    if (!calls.empty()) {
      out += ",\"tool_calls\":[";
      for (size_t i = 0; i < calls.size(); ++i) {
        if (i) out += ',';
        out += "{\"id\":" + quoted(calls[i].id) + ",\"name\":" + quoted(calls[i].name) +
               ",\"arguments\":" + quoted(calls[i].arguments) + "}";
      }
      out += "]";
    }
    dump_text(opt.dump_dir, cr.dump_no, "response.json", out + "}}");
  }
  return ok;
}

bool dispatch(ModelThread& model, const HttpRequest& req, int fd) {
  const ServeOptions& opt = model.opt();

  if (req.method == "GET" && req.target == "/health") {
    return send_response(fd, 200, "application/json", "{\"status\":\"ok\"}");
  }
  if (!opt.api_key.empty() &&
      req.header("authorization") != "Bearer " + opt.api_key) {
    return send_error(fd, 401, "expected an `Authorization: Bearer <key>` header");
  }
  if (req.method == "GET" && req.target == "/v1/models") {
    const std::string body =
        "{\"object\":\"list\",\"data\":[{\"id\":" + quoted(opt.model_name) +
        ",\"object\":\"model\",\"created\":" + std::to_string(model.started()) +
        ",\"owned_by\":\"local\"}]}";
    return send_response(fd, 200, "application/json", body);
  }
  if (req.target == "/v1/chat/completions") {
    if (req.method != "POST") return send_error(fd, 405, "use POST here");
    // The request is read and checked here, so a client that sends nonsense gets
    // its answer without waking the model or queueing behind a reply in flight.
    ChatRequest cr;
    try {
      cr = parse_chat(json::parse(req.body), opt);
    } catch (const std::exception& e) {
      // A request that was refused is worth keeping too -- it is what the client
      // sent, which is usually the thing being asked about.
      if (!opt.dump_dir.empty()) {
        const unsigned n = model.next_dump_no();
        dump_text(opt.dump_dir, n, "request.json", req.body);
        dump_text(opt.dump_dir, n, "response.json",
                  "{\"error\":" + quoted(e.what()) + "}");
      }
      return send_error(fd, 400, e.what());
    }
    if (!opt.dump_dir.empty()) {
      cr.dump_no = model.next_dump_no();
      dump_text(opt.dump_dir, cr.dump_no, "request.json", req.body);
    }
    return model.submit(cr, fd);
  }
  return send_error(fd, 404, "no route for " + req.method + " " + req.target +
                                 "; try GET /health, GET /v1/models or "
                                 "POST /v1/chat/completions");
}

void handle_connection(ModelThread& model, int fd) {
  std::string buf;
  while (!g_stop) {
    HttpRequest req;
    bool have = false;
    while (!have) {
      try {
        have = parse_request(buf, req);
      } catch (const std::runtime_error& e) {
        send_error(fd, 400, e.what());
        close(fd);
        return;
      }
      if (have) break;

      char tmp[16 * 1024];
      const ssize_t n = ::read(fd, tmp, sizeof(tmp));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {  // EOF, a timeout, or a client that went away
        close(fd);
        return;
      }
      buf.append(tmp, static_cast<size_t>(n));
      if (buf.size() > kMaxRequest) {
        send_error(fd, 413, "request too large");
        close(fd);
        return;
      }
    }

    // Nothing here may take the process down with it: an exception that escapes
    // a thread's entry point calls std::terminate.
    bool keep = false;
    try {
      keep = dispatch(model, req, fd);
    } catch (const std::exception& e) {
      send_error(fd, 500, e.what());
      break;
    }
    // An HTTP/1.0 client that did not ask for keep-alive is waiting for the
    // close to know the reply ended.
    const bool close_asked = lower(req.header("connection")) == "close";
    const bool http10 = req.version == "HTTP/1.0" &&
                        lower(req.header("connection")) != "keep-alive";
    if (!keep || close_asked || http10) break;
  }
  close(fd);
}

}  // namespace

void serve_socket(const std::string& model_dir, const ServeOptions& opt, int fd) {
  // One model thread for the life of the process, as serve() has. It is never
  // freed, for the same reason serve()'s is never joined.
  static ModelThread* model = nullptr;
  static std::string loaded;
  if (!model) {
    model = new ModelThread(model_dir, opt);
    loaded = model_dir;
  } else if (loaded != model_dir) {
    throw std::runtime_error("serve_socket: already serving " + loaded);
  }
  model->wait_ready();
  handle_connection(*model, fd);
}

void serve(const std::string& model_dir, const ServeOptions& opt) {
  int lfd = -1;
  try {
    lfd = listen_socket(opt.host, opt.port);
  } catch (const std::exception& e) {
    // A port in use, or one we may not bind, is the user's to fix, not a crash.
    fprintf(stderr, "cannot serve: %s\n", e.what());
    fflush(nullptr);
    _exit(1);
  }
  const int port = bound_port(lfd);

  // Replace the REPL's handler: SIGINT here stops the server, not one reply.
  struct sigaction sa {};
  sa.sa_handler = on_sigint;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // no SA_RESTART: let the blocked accept() return EINTR
  sigaction(SIGINT, &sa, nullptr);

  // The dump directory is the one thing here that writes to disk, so it is
  // created only when it was asked for, and only its owner can read it.
  if (!opt.dump_dir.empty() && mkdir(opt.dump_dir.c_str(), 0700) != 0 && errno != EEXIST) {
    fprintf(stderr, "cannot make the dump directory %s: %s\n", opt.dump_dir.c_str(),
            strerror(errno));
    return;
  }

  fprintf(stderr, "loading %s ...\n", model_dir.c_str());
  ModelThread model(model_dir, opt);
  model.wait_ready();
  fprintf(stderr, "serving %s on http://%s:%d/v1   OpenAI chat completions; Ctrl-C to stop\n",
          opt.model_name.c_str(), opt.host.c_str(), port);
  if (!opt.dump_dir.empty()) {
    fprintf(stderr, "dumping every request and reply to %s, oldest first\n",
            opt.dump_dir.c_str());
  }

  while (!g_stop) {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    const int fd = accept(lfd, reinterpret_cast<sockaddr*>(&peer), &len);
    if (fd < 0) {
      if (errno == EINTR) continue;  // a SIGINT lands here, and g_stop ends the loop
      perror("accept");
      break;
    }
    tune_client_socket(fd);
    std::thread(handle_connection, std::ref(model), fd).detach();
  }

  fprintf(stderr, "stopped\n");
  // _exit, not return: connection threads are detached and may be mid-write, and
  // they must not race the C++ runtime tearing down its own state. There is
  // nothing of the user's to flush -- the conversation dies with the process.
  fflush(nullptr);
  _exit(0);
}
