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
#include <vector>

#include "model.hpp"
#include "tokenizer.hpp"

struct Stats {
  int n = 0;         // tokens emitted
  int rounds = 0;    // target passes (speculative only)
  int drafted = 0;
  int accepted = 0;
  double prefill_t = 0;  // seconds spent consuming the prompt
  double dt = 0;         // seconds spent generating, excluding prefill
};

// Set from the SIGINT handler; the decode loops poll it at round boundaries.
// Returns the previous value and clears it.
bool take_interrupt();
void install_interrupt_handler();
bool interrupt_pending();

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
  void turn(const std::string& text, bool think, float temp, float top_p, int top_k,
            int max_tokens, const std::function<void(const std::string&)>& emit);

  const Stats& last() const { return last_; }
  size_t context_tokens() const { return tokens_.size(); }
  bool speculative() const { return spec_; }

  std::string system;

 private:
  std::vector<int> prompt_tokens(const std::string& text, bool think) const;
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
};
