#include "session.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <stdexcept>

#include "json.hpp"
#include "ops.hpp"
#include "sample.hpp"

namespace {

volatile std::sig_atomic_t g_interrupt = 0;

void on_sigint(int) { g_interrupt = 1; }

double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// A (1, n) int32 array of token ids.
mx::array row(const std::vector<int>& ids) {
  return mx::array(ids.begin(), {1, static_cast<int>(ids.size())}, mx::int32);
}

mx::array one(int id) { return mx::array({id}, mx::Shape{1, 1}, mx::int32); }

// logits (B, L, V) -> the last position as a 1-D (V,) vector.
mx::array last_row(const mx::array& logits) {
  return ops::index_axis(ops::index_axis(logits, 1, logits.shape(1) - 1), 0, 0);
}

// x[:, -1:] -- keeps the length axis.
mx::array last_step(const mx::array& x) {
  return ops::slice_axis(x, 1, x.shape(1) - 1, x.shape(1));
}

// The replacement character, i.e. "this token sequence ends mid-codepoint".
const char* kReplacement = "\xEF\xBF\xBD";

// One `<|im_start|>role\ncontent<|im_end|>\n` block. The closing newline matters:
// every block is followed by the next one, and the template never puts anything
// else between them.
std::string block(const std::string& role, const std::string& content) {
  return "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
}

}  // namespace

namespace {

// The function-calling preamble, verbatim from the checkpoint's chat template.
// The model was trained on this exact wording, so it is not ours to rephrase.
// The two empty lines at the front are the blank line the template puts before
// "If you choose to call a function" and the one before that.
const char* const kToolFormatLines[] = {
    "",
    "",
    "If you choose to call a function ONLY reply in the following format with NO suffix:",
    "",
    "<tool_call>",
    "<function=example_function_name>",
    "<parameter=example_parameter_1>",
    "value_1",
    "</parameter>",
    "<parameter=example_parameter_2>",
    "This is the value for the second parameter",
    "that can span",
    "multiple lines",
    "</parameter>",
    "</function>",
    "</tool_call>",
    "",
    "<IMPORTANT>",
    "Reminder:",
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags",
    "- Required parameters MUST be specified",
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after",
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls",
    "</IMPORTANT>",
};

// A tool result, which the template renders inside a user block. Trailing
// whitespace is not trimmed: the template does not, and a client's history is
// compared against its own previous request to decide on cache reuse.
bool is_tool_response(const std::string& content) {
  const size_t a = content.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return false;
  const std::string s = content.substr(a);
  return s.rfind("<tool_response>", 0) == 0 &&
         s.size() >= strlen("</tool_response>") &&
         s.compare(s.size() - strlen("</tool_response>"), strlen("</tool_response>"),
                   "</tool_response>") == 0;
}

std::string trim(const std::string& s) {
  const size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  const size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// The arguments a client echoed, reparsed so that each parameter can go back
// into the template's <parameter=> lines. Anything that is not a JSON object
// leaves the call with no parameters, which is all a malformed echo deserves.
json::Value parse_arguments(const std::string& text) {
  json::Value v;
  try {
    v = json::parse(text);
  } catch (const std::exception&) {
    return json::Value{};
  }
  return v.is_obj() ? v : json::Value{};
}

// What the template writes for one assistant turn's tool calls, in the format it
// taught the model to use.
std::string render_tool_calls(const Message& m) {
  std::string out;
  for (size_t c = 0; c < m.tool_calls.size(); ++c) {
    const ToolCall& call = m.tool_calls[c];
    if (c == 0 && trim(m.content).empty()) {
      out += "<tool_call>\n";
    } else if (c == 0) {
      out += "\n\n<tool_call>\n";
    } else {
      out += "\n<tool_call>\n";
    }
    out += "<function=" + call.name + ">\n";
    // Values go back exactly as the template would have written them: a string
    // as itself, anything structured as JSON.
    const json::Value args = parse_arguments(call.arguments);
    for (const auto& kv : args.obj) {
      out += "<parameter=" + kv.first + ">\n";
      if (kv.second.type == json::Type::Str) {
        out += kv.second.str;
      } else {
        out += json::dump(kv.second);
      }
      out += "\n</parameter>\n";
    }
    out += "</function>\n</tool_call>";
  }
  return out;
}

}  // namespace

std::string template_mismatch(const std::string& chat_template) {
  if (chat_template.empty()) {
    return "the checkpoint carries no chat template, so the prompt cannot be checked "
           "against one";
  }
  // The template's own text has to carry these; each is what one part of the
  // renderer stands in for.
  struct Marker {
    const char* text;
    const char* part;
  };
  static const Marker kCore[] = {
      {"<|im_start|>", "<|im_start|> turn markers"},
      {"<|im_end|>", "<|im_end|> turn markers"},
  };
  static const Marker kTools[] = {
      {"# Tools", "a tools preamble"},
      {"<tool_call>", "tool_call tags"},
      {"<function=", "a <function=...> call body"},
      {"<parameter=", "<parameter=...> arguments"},
      {"<tool_response>", "a tool_response result block"},
  };
  for (const Marker& m : kCore) {
    if (chat_template.find(m.text) == std::string::npos) {
      return std::string("the checkpoint's chat template has no ") + m.part +
             ", which is the format these renderers write; prompts would not be what "
             "the model was trained on";
    }
  }
  std::string missing;
  for (const Marker& m : kTools) {
    if (chat_template.find(m.text) == std::string::npos) {
      if (!missing.empty()) missing += ", ";
      missing += m.part;
    }
  }
  if (!missing.empty()) {
    return "the checkpoint's chat template has no " + missing +
           "; requests that pass tools cannot be rendered the way the model expects";
  }
  return "";
}

std::string render_chat(const std::vector<Message>& msgs, const std::string& system,
                        bool think, const std::vector<ToolSpec>& tools) {
  const std::string sys = trim(system);
  std::string p;
  if (!tools.empty()) {
    p += "<|im_start|>system\n";
    p += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const ToolSpec& t : tools) {
      p += "\n";
      p += t.json;
    }
    p += "\n</tools>";
    for (const char* line : kToolFormatLines) {
      p += "\n";
      p += line;
    }
    if (!sys.empty()) p += "\n\n" + sys;
    p += "<|im_end|>\n";
  } else if (!sys.empty()) {
    p += block("system", sys);
  }
  return p + render_tail(msgs, 0, think);
}

std::string render_tail(const std::vector<Message>& msgs, size_t from, bool think) {
  // The template only wraps an assistant turn in a reasoning block when that
  // turn comes after the last real user query -- the turn being continued, not
  // one replayed from the history.
  size_t last_query = msgs.size();
  for (size_t i = 0; i < msgs.size(); ++i) {
    if (msgs[i].role == "user" && !is_tool_response(msgs[i].content)) last_query = i;
  }

  std::string p;
  for (size_t i = from; i < msgs.size(); ++i) {
    const Message& m = msgs[i];
    if (m.role == "tool") {
      // Consecutive tool results share one user block.
      if (i == 0 || msgs[i - 1].role != "tool") p += "<|im_start|>user";
      p += "\n<tool_response>\n" + trim(m.content) + "\n</tool_response>";
      if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") p += "<|im_end|>\n";
      continue;
    }
    // The template trims every message's content; matching it matters, because
    // the tokens are what the KV cache is keyed on.
    std::string body = trim(m.content);
    if (m.role == "assistant") {
      if (i > last_query) body = "<think>\n" + trim(m.reasoning) + "\n</think>\n\n" + body;
      if (!m.tool_calls.empty()) body += render_tool_calls(m);
    }
    p += block(m.role, body);
  }
  p += "<|im_start|>assistant\n";
  // The template always opens a reasoning block; closing it immediately is how
  // you disable thinking.
  p += think ? "<think>\n" : "<think>\n\n</think>\n\n";
  return p;
}

void install_interrupt_handler() {
  struct sigaction sa {};
  sa.sa_handler = on_sigint;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // no SA_RESTART: let a blocked read() return EINTR
  sigaction(SIGINT, &sa, nullptr);
}

bool interrupt_pending() { return g_interrupt != 0; }

bool take_interrupt() {
  bool was = g_interrupt != 0;
  g_interrupt = 0;
  return was;
}

void request_interrupt() { g_interrupt = 1; }

Session::Session(const Model& model, const Tokenizer& tk, std::string system,
                 bool spec, int draft_k, int prefill_step)
    : system(std::move(system)),
      model_(model),
      tk_(tk),
      spec_(spec && model.mtp.has_value()),
      k_(draft_k),
      prefill_step_(prefill_step) {
  reset();
}

void Session::reset() {
  tokens_.clear();
  cache_ = model_.make_cache();
  turns_ = 0;
  rope_delta_ = 0;
  pending_.clear();
  mx::clear_cache();
}

mx::array Session::sync() {
  const int lo = cache_.empty() ? 0 : cache_[0].offset();
  const std::vector<int> need(tokens_.begin() + lo, tokens_.end());
  std::pair<mx::array, int> r = prefill(model_, cache_, need, pending_, rope_delta_, prefill_step_);
  rope_delta_ = r.second;
  pending_.clear();
  return r.first;
}

void Session::close_turn() {
  // Close the assistant turn so the next one continues correctly. These tokens
  // are consumed by the next sync().
  std::vector<int> end = tk_.encode("<|im_end|>\n");
  tokens_.insert(tokens_.end(), end.begin(), end.end());
  ++turns_;
}

void Session::turn(const std::string& text, bool think, float temp, float top_p,
                   int top_k, int max_tokens,
                   const std::function<void(const std::string&)>& emit,
                   const std::vector<image::Image>& images) {
  Message user;
  user.role = "user";
  for (size_t i = 0; i < images.size(); ++i) user.content += kImagePlaceholder;
  user.content += text;
  turn_prompt(render_chat({user}, turns_ == 0 ? system : std::string(), think), temp,
              top_p, top_k, max_tokens, emit, images);
}

void Session::turn_prompt(const std::string& prompt, float temp, float top_p,
                          int top_k, int max_tokens,
                          const std::function<void(const std::string&)>& emit,
                          const std::vector<image::Image>& images) {
  const double t0 = now_s();
  // Everything that can fail happens before the token list is touched, so a bad
  // picture leaves the conversation exactly as it was.
  std::vector<int> ids = tk_.encode(prompt);
  const int pad = model_.cfg.image_token_id;
  std::vector<ImageFeatures> feats;
  if (images.empty()) {
    if (std::find(ids.begin(), ids.end(), pad) != ids.end()) {
      throw std::runtime_error("the message contains a literal <|image_pad|> token");
    }
  } else {
    std::vector<int> counts;
    for (const image::Image& im : images) counts.push_back(im.n_tokens());
    ids = expand_image_pads(ids, counts, pad);
    std::vector<mx::array> embeds;
    for (const image::Image& im : images) {
      feats.push_back(ImageFeatures{model_.embed_image(im), im.rows(), im.cols()});
      embeds.push_back(feats.back().embeds);
    }
    mx::eval(embeds);
  }
  const double vision_t = now_s() - t0;

  const size_t before = tokens_.size();
  tokens_.insert(tokens_.end(), ids.begin(), ids.end());
  pending_.insert(pending_.end(), feats.begin(), feats.end());

  last_ = Stats{};
  last_.prompt_n = static_cast<int>(tokens_.size() - before);
  last_.images = static_cast<int>(images.size());
  last_.vision_t = vision_t;

  mx::array h = sync();
  int first = sample::sample_probs(
      sample::to_probs(last_row(model_.logits(h)), temp, top_p, top_k));
  last_.prefill_t = now_s() - t0;
  const double t1 = now_s();

  if (is_eos(first)) {
    close_turn();
    return;
  }

  tokens_.push_back(first);
  last_.n = 1;
  std::vector<int> emitted{first};
  emit(tk_.decode(emitted));

  if (spec_) {
    spec_loop(h, first, emitted, temp, top_p, top_k, max_tokens, emit);
  } else {
    plain_loop(h, first, emitted, temp, top_p, top_k, max_tokens, emit);
  }

  last_.dt = now_s() - t1;
  close_turn();
}

void Session::plain_loop(mx::array h, int tok, std::vector<int>& emitted, float temp,
                         float top_p, int top_k, int max_tokens,
                         const std::function<void(const std::string&)>& emit) {
  std::string printed = tk_.decode(emitted);
  while (last_.n < max_tokens) {
    if (interrupt_pending()) return;
    // after an image the rope position is the cache offset plus rope_delta_
    h = model_.hidden_states(one(tok), cache_, Rope::at(cache_[0].offset() + rope_delta_));
    tok = sample::sample_probs(
        sample::to_probs(last_row(model_.logits(h)), temp, top_p, top_k));
    if (is_eos(tok)) return;
    tokens_.push_back(tok);
    emitted.push_back(tok);
    ++last_.n;

    std::string full = tk_.decode(emitted);
    if (full.size() > printed.size() &&
        full.find(kReplacement, printed.size()) == std::string::npos) {
      emit(full.substr(printed.size()));
      printed = full;
    }
  }
}

void Session::spec_loop(mx::array h, int first, std::vector<int>& emitted, float temp,
                        float top_p, int top_k, int max_tokens,
                        const std::function<void(const std::string&)>& emit) {
  const MTPDraft& mtp = *model_.mtp;
  const int k = k_;

  // Fresh MTP cache each turn: its entries are indexed by cache position, and
  // injecting a user turn would leave a positional gap. The drafter only needs
  // local context and a wrong draft costs nothing but a rejection, so this is
  // safe -- just slightly fewer accepts at the start of a turn.
  std::vector<LayerCache> mtp_cache = mtp.make_cache();
  std::string printed = tk_.decode(emitted);

  int P = cache_[0].offset();
  // Every position the drafter and the verify pass use is shifted by the rope
  // delta -- constant for the whole reply, which holds no images.
  const int d = rope_delta_;
  mx::array A_tok = one(first);
  mx::array A_hid = h;
  int A_pos = P + d;
  mx::array next_tok = one(first);

  while (last_.n < max_tokens) {
    // Poll only here, at a round boundary: the token list and every cache agree
    // at this point, so stopping needs no rollback.
    if (interrupt_pending()) return;

    const int P_old = P;

    // ---- 1. draft: the MTP head proposes k tokens autoregressively ----
    mx::array hd = mtp(model_.embed_tokens(A_tok), A_hid, mtp_cache, A_pos);
    const int grounded = mtp_cache[0].offset();
    mx::array h_prev = last_step(hd);
    std::vector<mx::array> qs;
    std::vector<int> dl;
    for (int i = 0; i < k; ++i) {
      if (i) {
        h_prev = mtp(model_.embed_tokens(one(dl.back())), h_prev, mtp_cache,
                     P_old + i + d);
      }
      mx::array q =
          sample::to_probs(last_row(model_.logits(h_prev)), temp, top_p, top_k);
      qs.push_back(q);
      dl.push_back(sample::sample_probs(q));
    }

    // ---- 2. verify: ONE target pass over [known, d1..dk] ----
    mx::array X = mx::concatenate({next_tok, row(dl)}, 1);
    for (LayerCache& c : cache_) {
      if (c.linear) c.delta.record = true;
    }
    mx::array h_v = model_.hidden_states(X, cache_, Rope::at(P_old + d));
    mx::array tl = model_.logits(h_v);
    std::vector<mx::array> ps;
    ps.reserve(static_cast<size_t>(k) + 1);
    for (int i = 0; i <= k; ++i) {
      ps.push_back(sample::to_probs(ops::index_axis(ops::index_axis(tl, 1, i), 0, 0),
                                    temp, top_p, top_k));
    }

    // ---- 3. rejection test, one sync for the whole block ----
    mx::array di = mx::reshape(row(dl), {k, 1});
    std::vector<mx::array> ps_k(ps.begin(), ps.begin() + k);
    mx::array pd =
        ops::index_axis(mx::take_along_axis(mx::stack(ps_k), di, -1), -1, 0);
    mx::array qd = ops::index_axis(mx::take_along_axis(mx::stack(qs), di, -1), -1, 0);
    mx::array u = mx::random::uniform(0.0f, 1.0f, {k});
    // accept iff u <= p/q, guarding q == 0 (then accept iff p > 0)
    mx::array acc = mx::where(mx::greater(qd, mx::array(0.0f)),
                              mx::less_equal(mx::multiply(u, qd), pd),
                              mx::greater(pd, mx::array(0.0f)));
    mx::eval({acc, h_v});

    const bool* accepted = acc.data<bool>();
    int j = 0;
    while (j < k && accepted[j]) ++j;
    int corrected = sample::sample_probs(j == k ? ps[static_cast<size_t>(k)]
                                                : sample::residual(ps[j], qs[j]));

    // ---- roll the caches back to the accepted prefix ----
    for (LayerCache& c : cache_) {
      if (c.linear) {
        c.delta.commit(j + 1);
      } else {
        c.kv.trim(P_old + j + 1);
      }
    }
    mtp_cache[0].kv.trim(grounded);

    last_.drafted += k;
    last_.accepted += j;
    ++last_.rounds;

    bool stop = false;
    std::vector<int> accepted_toks(dl.begin(), dl.begin() + j);
    accepted_toks.push_back(corrected);
    for (int t : accepted_toks) {
      if (is_eos(t)) {
        stop = true;
        break;
      }
      tokens_.push_back(t);
      emitted.push_back(t);
      ++last_.n;
      if (last_.n >= max_tokens) {
        stop = true;
        break;
      }
    }

    std::string full = tk_.decode(emitted);
    if (full.size() > printed.size() &&
        full.find(kReplacement, printed.size()) == std::string::npos) {
      emit(full.substr(printed.size()));
      printed = full;
    }
    if (stop) return;

    P = P_old + j + 1;
    next_tok = one(corrected);
    accepted_toks.back() = corrected;  // already true; kept explicit
    A_tok = row(accepted_toks);
    A_hid = ops::slice_axis(h_v, 1, 0, j + 1);
    A_pos = P_old + 1 + d;
  }
}
