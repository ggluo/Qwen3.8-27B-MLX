#include "session.hpp"

#include <chrono>
#include <csignal>

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

}  // namespace

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
  mx::clear_cache();
}

std::vector<int> Session::prompt_tokens(const std::string& text, bool think) const {
  std::string p;
  if (turns_ == 0 && !system.empty()) {
    p += "<|im_start|>system\n" + system + "<|im_end|>\n";
  }
  p += "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
  // The template always opens a reasoning block; closing it immediately is how
  // you disable thinking.
  p += think ? "<think>\n" : "<think>\n\n</think>\n\n";
  return tk_.encode(p);
}

mx::array Session::sync() {
  const int lo = cache_.empty() ? 0 : cache_[0].offset();
  const int total = static_cast<int>(tokens_.size());
  mx::array h = mx::zeros({1, 1, model_.cfg.hidden_size});
  for (int s = lo; s < total; s += prefill_step_) {
    const int end = std::min(s + prefill_step_, total);
    mx::array ids(tokens_.begin() + s, {1, end - s}, mx::int32);
    h = model_.hidden_states(ids, cache_);
    mx::eval(h);
  }
  return last_step(h);
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
                   const std::function<void(const std::string&)>& emit) {
  std::vector<int> prompt = prompt_tokens(text, think);
  tokens_.insert(tokens_.end(), prompt.begin(), prompt.end());

  const double t0 = now_s();
  last_ = Stats{};

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
    h = model_.hidden_states(one(tok), cache_);
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
  mx::array A_tok = one(first);
  mx::array A_hid = h;
  int A_pos = P;
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
                     P_old + i);
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
    mx::array h_v = model_.hidden_states(X, cache_);
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
    A_pos = P_old + 1;
  }
}
