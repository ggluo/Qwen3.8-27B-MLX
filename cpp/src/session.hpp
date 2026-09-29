// One conversation: the tokens, the caches that mirror them, and the two decode
// loops. Nothing is written to disk; everything here dies with the process.
//
// Two things make it feel fast:
//
//   * The KV cache is REUSED across turns. Only the new user turn is prefilled,
//     not the whole conversation -- otherwise turn N costs O(total tokens) and a
//     long chat crawls.
//   * Speculative decoding via the MTP head, ~1.4x on prose and ~1.9x on code.
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "image.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

struct Stats {
  int n = 0;         // tokens emitted
  int prompt_n = 0;  // tokens consumed as the prompt
  int rounds = 0;    // target passes (speculative only)
  int drafted = 0;
  int accepted = 0;
  double prefill_t = 0;  // seconds spent consuming the prompt, pictures included
  double dt = 0;         // seconds spent generating, excluding prefill
  int images = 0;        // pictures in this turn's prompt
  double vision_t = 0;   // seconds of prefill_t spent in the vision tower
};

// Set from the SIGINT handler; the decode loops poll it at round boundaries.
// Returns the previous value and clears it.
bool take_interrupt();
void install_interrupt_handler();
bool interrupt_pending();
// Asks a running decode loop to stop at the next round boundary, the same way
// SIGINT does. Async-signal-safe, so a signal handler other than the one above
// may call it -- serve mode's does, so that Ctrl-C both stops the reply in
// flight and shuts the server down.
void request_interrupt();

// ---------------------------------------------------------------------------
// the chat template
// ---------------------------------------------------------------------------

// What the template writes for one picture in a message. The single
// <|image_pad|> is expanded to one per image token when the prompt is tokenized
// (expand_image_pads), so a rendered prompt holds exactly one of these per image.
constexpr const char* kImagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>";

// One function call, either one the model asked for or one a client echoed back.
struct ToolCall {
  std::string id;         // clients send one; we mint it when we emit one
  std::string name;
  std::string arguments;  // a JSON object, as text
};

// One message of a chat, content already flattened to text.
struct Message {
  Message() = default;
  Message(std::string r, std::string c) : role(std::move(r)), content(std::move(c)) {}

  std::string role;  // system | user | assistant | tool
  std::string content;
  std::string reasoning;             // assistant only, the part before </think>
  std::vector<ToolCall> tool_calls;  // assistant only
  // The pictures in `content`, as the encoded bytes the client sent, in the
  // order of their kImagePlaceholder in the text. Kept as bytes so two requests
  // can be compared exactly: the same words with a different picture are a
  // different message.
  std::vector<std::string> images;
};

// One entry of a request's `tools` array, already back out as JSON text. The
// template prints these verbatim into the prompt, so the text has to be stable
// from request to request or the KV cache cannot be reused.
struct ToolSpec {
  std::string json;

  bool operator==(const ToolSpec& other) const { return json == other.json; }
};

// The whole conversation as the model sees it: an optional system block, one
// block per message, then the assistant opener and the marker that closes the
// reasoning block.
//
// The last message is the one to be answered; the opener goes on the end
// unconditionally, so callers must end with a message that is not the
// assistant's.
//
// `tools` puts the checkpoint's own function-calling preamble, and the schemas,
// at the head of the system block -- the model was trained on exactly that text,
// so it is not ours to rephrase. `system` is the whole system turn; a request's
// system message belongs there rather than in `msgs`.
std::string render_chat(const std::vector<Message>& msgs, const std::string& system,
                        bool think, const std::vector<ToolSpec>& tools = {});

// The tail of that string, covering msgs[from..] and the opener. A session that
// closed its last turn with <|im_end|>\n already holds everything before `from`,
// so feeding it this one string continues the chat in the place -- and in the
// order -- render_chat() would have put it.
//
// The tools are not re-rendered: they live in the head, which is already there.
//
// `render_tail(msgs, 0, think)` is `render_chat(msgs, "", think)`.
std::string render_tail(const std::vector<Message>& msgs, size_t from, bool think);

// The renderers above are this model family's chat template, written out in C++
// rather than interpreted -- the real one is a Jinja file, and running it would
// mean carrying a Jinja engine. So a checkpoint that brought a different
// template would be prompted in a format it was not trained on, silently.
//
// This reads the checkpoint's template (its text, from chat_template.jinja or
// tokenizer_config.json) and returns what is wrong with it, or "" when it is the
// template these renderers implement.
std::string template_mismatch(const std::string& chat_template);

class Session {
 public:
  Session(const Model& model, const Tokenizer& tk, std::string system, bool spec,
          int draft_k, int prefill_step = 384);

  // Drops the conversation and starts over.
  void reset();

  // Runs one exchange, handing decoded text to `emit` as it is produced.
  //
  // An interrupt stops at a ROUND BOUNDARY, where the token list and every cache
  // already agree, so there is nothing to roll back -- unlike the Python original,
  // which had to unwind a half-finished speculative round after a KeyboardInterrupt.
  //
  // `images` are shown to the model ahead of the text, the way the model card
  // lays a picture out.
  void turn(const std::string& text, bool think, float temp, float top_p, int top_k,
            int max_tokens, const std::function<void(const std::string&)>& emit,
            const std::vector<image::Image>& images = {});

  // The same, with the caller supplying the prompt text. `turn()` is this plus
  // render_chat() on one user message; serve mode renders a whole chat from the
  // request's messages, so it comes through here instead. `system` is not
  // consulted -- whatever is in the prompt is what the model sees.
  //
  // The prompt carries one `<|vision_start|><|image_pad|><|vision_end|>` per
  // picture, and `images` are those pictures in the order they appear. Throws,
  // leaving the conversation untouched, if the two do not match up or the model
  // has no vision tower.
  void turn_prompt(const std::string& prompt, float temp, float top_p, int top_k,
                   int max_tokens, const std::function<void(const std::string&)>& emit,
                   const std::vector<image::Image>& images = {});

  const Stats& last() const { return last_; }
  size_t context_tokens() const { return tokens_.size(); }
  const std::vector<int>& tokens() const { return tokens_; }
  int rope_delta() const { return rope_delta_; }
  bool speculative() const { return spec_; }

  std::string system;

 private:
  // Feeds everything the model has not seen yet; returns the last hidden state.
  mx::array sync();
  bool is_eos(int t) const { return t == tk_.eos_id || t == tk_.bos_id; }
  void close_turn();

  void plain_loop(mx::array h, int first, std::vector<int>& emitted, float temp,
                  float top_p, int top_k, int max_tokens,
                  const std::function<void(const std::string&)>& emit);
  void spec_loop(mx::array h, int first, std::vector<int>& emitted, float temp,
                 float top_p, int top_k, int max_tokens,
                 const std::function<void(const std::string&)>& emit);

  const Model& model_;
  const Tokenizer& tk_;
  bool spec_;
  int k_;
  int prefill_step_;

  std::vector<int> tokens_;
  std::vector<LayerCache> cache_;
  int turns_ = 0;
  Stats last_;

  // After an image the rope position runs ahead of the token index -- an image
  // of r x c tokens advances it by max(r, c), not r*c -- and this is by how much.
  // Every decode step and every draft uses cache offset + rope_delta_. Decode
  // loops stop only at round boundaries, never inside a prefill, so an image is
  // always consumed whole and this never needs recomputing from a prefix.
  int rope_delta_ = 0;
  std::vector<ImageFeatures> pending_;  // pictures in tokens_ not yet fed to the model
};
