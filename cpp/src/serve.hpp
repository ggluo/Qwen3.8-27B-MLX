// Serve mode: the same assistant behind an HTTP endpoint, speaking the OpenAI
// chat-completions protocol, so anything that already talks to vLLM, llama.cpp
// or Ollama can talk to this binary instead.
//
// Threads: one per connection, plus one that owns the model. See the note on
// ModelThread in serve.cpp for why the model cannot be shared between them.
// Requests are answered one at a time -- there is one conversation -- while
// /health and /v1/models are answered by the connection threads without waking
// the model at all, so a client can always tell that the process is alive.
#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"
#include "session.hpp"

struct ServeOptions {
  std::string host = "127.0.0.1";  // an IPv4 literal, or "localhost"
  int port = 8080;
  std::string api_key;     // empty => no Authorization check
  std::string model_name;  // what every reply and /v1/models calls this model
  std::string system;      // system prompt, ahead of whatever the request sends
  bool think = false;      // default, for requests that do not ask
  bool spec = true;
  int draft = 3;
  // Where to write every request and reply, verbatim, for when a conversation
  // is not doing what it looks like it should. Empty -- the default -- writes
  // nothing: this is the one thing here that touches the disk, and it only does
  // so when asked.
  std::string dump_dir;
};

// Binds, loads the model on the thread that will run it, then serves until
// SIGINT. Never returns.
void serve(const std::string& model_dir, const ServeOptions& opt);

// ---------------------------------------------------------------------------
// Pieces the self-tests drive on their own. None of them touch the model.
// ---------------------------------------------------------------------------

// Length of the longest prefix of s[0..n) that does not end in the middle of a
// UTF-8 sequence. A byte-level BPE token can be a single continuation byte, so
// a piece of the reply can end mid-character; the rest has to wait for the next
// piece, because half a character inside a JSON string is not a string.
size_t utf8_complete(const char* s, size_t n);

// Holds back anything that could still turn out to be the start of a stop
// sequence, so that a stop string never reaches the client. feed() returns the
// text that is safe to send now; finish() flushes what is left when the reply
// ends without one.
class StopFilter {
 public:
  explicit StopFilter(std::vector<std::string> stops);

  std::string feed(const std::string& piece);
  std::string finish();
  bool hit() const { return hit_; }

 private:
  std::vector<std::string> stops_;
  std::string buf_;
  bool hit_ = false;
  size_t max_hold_ = 0;  // longest stop, minus one: more than that and it would have matched
};

// The model writes a call the way the checkpoint's template taught it:
//
//     <tool_call>
//     <function=name>
//     <parameter=key>
//     value
//     </parameter>
//     </function>
//     </tool_call>
//
// Some checkpoints in the same family, and anything a client pasted into a
// history, use a JSON body in the same tags instead. Both are read.
//
// A block is only handed on once it is complete, so a call never reaches the
// client in pieces; a `<tool_call>` that turns out to be prose stays prose.
class ToolCallParser {
 public:
  // tool name -> parameter name -> declared type. The type is what decides
  // whether a parameter that looks like a number comes back as one.
  using Types = std::map<std::string, std::map<std::string, std::string>>;

  explicit ToolCallParser(Types types) : types_(std::move(types)) {}

  struct Piece {
    std::string text;             // what is left of the content
    std::vector<ToolCall> calls;  // calls that arrived in full
  };

  Piece feed(const std::string& piece);
  Piece finish();

 private:
  ToolCall parse_block(const std::string& body);
  json::Value value_of(const std::string& tool, const std::string& key,
                       const std::string& text) const;

  static constexpr const char* kOpen = "<tool_call>";
  static constexpr const char* kClose = "</tool_call>";
  Types types_;
  std::string buf_;
};

// One HTTP request, parsed off the wire.
struct HttpRequest {
  std::string method;
  std::string target;  // path, without the query string
  std::string query;
  std::string version;                                       // "HTTP/1.1"
  std::vector<std::pair<std::string, std::string>> headers;  // names lowercased
  std::string body;

  std::string header(const std::string& name) const;  // case-insensitive name
};

// Takes one request off the front of `buf`, erasing the bytes it consumed.
// Returns false -- touching nothing -- while the request is still incomplete, so
// the caller can read more. Throws std::runtime_error on a request that no
// amount of further reading would make servable.
bool parse_request(std::string& buf, HttpRequest& out);
