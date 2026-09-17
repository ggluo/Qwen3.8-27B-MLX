// Self-tests. Run `make test`, or `./build/tests --model <dir>`.
//
// The tokenizer tests are pinned against the Python implementation (see
// tools/gen_golden.py) and need only tokenizer.json. The model tests need the
// weights and are skipped if the directory is absent, so `make test` stays
// useful without a 16 GB download.
#include <sys/select.h>
#include <sys/wait.h>
#include <unistd.h>
#include <util.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "delta.hpp"
#include "lineread.hpp"
#include "model.hpp"
#include "session.hpp"
#include "ops.hpp"
#include "sample.hpp"
#include "tokenizer.hpp"
#include "unicode.hpp"

namespace {

int g_pass = 0, g_fail = 0;

// max|a-b| / max|b| -- the same relative measure the Python self-tests used.
float rel_err(const mx::array& a, const mx::array& b) {
  mx::array num = mx::max(mx::abs(mx::subtract(a, b)));
  mx::array den = mx::max(mx::abs(b));
  mx::eval({num, den});
  float d = den.item<float>();
  return num.item<float>() / (d > 0 ? d : 1.0f);
}

void check(bool ok, const std::string& what) {
  printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
  }
}

struct GoldenCase {
  const char* text;
  std::vector<int> ids;
};

#include "tokenizer_golden.inc"

std::string join(const std::vector<int>& v, size_t limit = 16) {
  std::string s;
  for (size_t i = 0; i < v.size() && i < limit; ++i) {
    if (i) s += " ";
    s += std::to_string(v[i]);
  }
  if (v.size() > limit) s += " ...";
  return s;
}

std::string brief(const std::string& s, size_t n = 40) {
  std::string out;
  for (char c : s) {
    if (out.size() >= n) {
      out += "...";
      break;
    }
    if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else if (c == '\r') out += "\\r";
    else out += c;
  }
  return out;
}

void test_unicode() {
  printf("\nunicode\n");
  check(uni::is_L('a') && uni::is_L(0x4E2D) && !uni::is_L('1'), "is_L");
  check(uni::is_N('7') && uni::is_N(0x2160) && !uni::is_N('a'), "is_N (Nd + Nl)");
  check(uni::is_M(0x0301) && !uni::is_M('a'), "is_M (combining marks)");
  check(uni::is_space(' ') && uni::is_space('\n') && uni::is_space(0x3000) &&
            !uni::is_space('a'),
        "is_space");
  // NFC must compose e + combining acute into a single precomposed codepoint.
  check(uni::nfc("cafe\xCC\x81") == "caf\xC3\xA9", "nfc composes e+U+0301 -> e-acute");
  check(uni::nfc("plain ascii") == "plain ascii", "nfc leaves ascii alone");

  std::string round;
  for (uint32_t cp : {0x41u, 0xE9u, 0x4E2Du, 0x1F600u}) {
    round.clear();
    uni::encode(cp, round);
    size_t i = 0;
    check(uni::decode(round, i) == cp && i == round.size(),
          "utf-8 round trip U+" + std::to_string(cp));
  }
}

void test_tokenizer(const std::string& path) {
  printf("\ntokenizer (%s)\n", path.c_str());
  Tokenizer tk(path);
  printf("  vocab %zu  merges %zu  special %zu  eos %d  bos %d\n", tk.vocab_size(),
         tk.merge_count(), tk.special_count(), tk.eos_id, tk.bos_id);
  check(tk.eos_id == 248046 && tk.bos_id == 248044, "eos/bos ids");

  // 1. Ids identical to the reference Python implementation.
  int mismatch = 0;
  for (const GoldenCase& g : kGolden) {
    std::vector<int> got = tk.encode(g.text);
    if (got != g.ids) {
      ++mismatch;
      printf("  [FAIL] ids differ for %s\n         got  %s\n         want %s\n",
             brief(g.text).c_str(), join(got).c_str(), join(g.ids).c_str());
    }
  }
  check(mismatch == 0,
        "ids match the Python reference on all " + std::to_string(sizeof(kGolden) / sizeof(kGolden[0])) +
            " cases");

  // 2. Round trip. Comparing against NFC(input), since encode normalizes.
  int bad = 0;
  for (const GoldenCase& g : kGolden) {
    std::string want = uni::nfc(g.text);
    std::string back = tk.decode(tk.encode(g.text));
    if (back != want) {
      ++bad;
      printf("  [FAIL] round trip %s\n         got  %s\n", brief(g.text).c_str(),
             brief(back).c_str());
    }
  }
  check(bad == 0, "decode(encode(x)) == NFC(x)");

  // 3. Special tokens must be single ids, matched literally.
  struct {
    const char* s;
    int id;
  } specials[] = {{"<|im_start|>", 248045}, {"<|im_end|>", 248046},
                  {"<|endoftext|>", 248044}};
  bool sp_ok = true;
  for (auto& s : specials) {
    std::vector<int> got = tk.encode(s.s);
    sp_ok &= (got.size() == 1 && got[0] == s.id);
  }
  check(sp_ok, "special tokens map to their exact single ids");

  // 4. With allow_special=false they must go through the BPE instead.
  check(tk.encode("<|im_end|>", false).size() > 1,
        "allow_special=false runs specials through the BPE");

  // 5. Every id in range.
  bool in_range = true;
  for (const GoldenCase& g : kGolden) {
    for (int id : tk.encode(g.text)) in_range &= (id >= 0 && id < 248320);
  }
  check(in_range, "all ids within vocab_size");
}

void test_delta() {
  printf("\ndelta kernel (metal available: %s)\n", delta::available() ? "yes" : "no");
  if (!delta::available()) {
    printf("  [skip] no Metal device\n");
    return;
  }
  mx::random::seed(0);
  struct Case {
    int B, L, H, Dk, Dv;
  };
  for (const Case& c : {Case{1, 1, 48, 128, 128}, Case{1, 4, 48, 128, 128},
                        Case{1, 9, 48, 128, 128}, Case{2, 3, 16, 128, 128},
                        Case{1, 64, 48, 128, 128}, Case{1, 384, 8, 128, 128}}) {
    mx::array q = mx::multiply(
        ops::l2norm(mx::random::normal({c.B, c.L, c.H, c.Dk})),
        mx::array(1.0f / std::sqrt(static_cast<float>(c.Dk))));
    mx::array k = ops::l2norm(mx::random::normal({c.B, c.L, c.H, c.Dk}));
    mx::array v = mx::random::normal({c.B, c.L, c.H, c.Dv});
    mx::array al = mx::random::uniform(0.85f, 1.0f, {c.B, c.L, c.H});
    mx::array bt = mx::random::uniform(0.0f, 1.0f, {c.B, c.L, c.H});
    mx::array S0 = mx::multiply(mx::random::normal({c.B, c.H, c.Dv, c.Dk}),
                                mx::array(0.1f));
    mx::eval({q, k, v, al, bt, S0});

    delta::Result ref = delta::run_reference(q, k, v, al, bt, S0);
    delta::Result got = delta::run(q, k, v, al, bt, S0);
    float eo = rel_err(got.o, ref.o);
    float es = rel_err(got.s_out, ref.s_out);
    char buf[160];
    snprintf(buf, sizeof(buf), "B=%d L=%3d H=%2d  out %.2e  state %.2e", c.B, c.L,
             c.H, eo, es);
    check(std::max(eo, es) < 2e-5f, buf);

    // The collect variant must return the identical per-step states -- this is
    // what speculative rollback indexes into.
    delta::Result rc = delta::run_reference(q, k, v, al, bt, S0, true);
    delta::Result kc = delta::run(q, k, v, al, bt, S0, true);
    float worst = 0;
    for (size_t t = 0; t < rc.states.size() && t < kc.states.size(); ++t) {
      worst = std::max(worst, rel_err(kc.states[t], rc.states[t]));
    }
    snprintf(buf, sizeof(buf), "  collect: %zu states, max rel err %.2e",
             kc.states.size(), worst);
    check(kc.states.size() == rc.states.size() &&
              kc.states.size() == static_cast<size_t>(c.L) && worst < 2e-5f,
          buf);
  }
}

// ---------------------------------------------------------------------------
// model tests -- need the weights, so they are skipped if the directory is absent
// ---------------------------------------------------------------------------

mx::array row(const std::vector<int>& ids) {
  return mx::array(ids.begin(), {1, static_cast<int>(ids.size())}, mx::int32);
}

// Mean next-token NLL (nats) and top-1 accuracy over `ids`.
std::pair<float, float> score(const Model& m, const std::vector<int>& ids) {
  std::vector<LayerCache> cache = m.make_cache();
  mx::array h = m.hidden_states(row(ids), cache);
  mx::array logits = mx::astype(m.logits(h), mx::float32);       // (1, L, V)
  const int L = static_cast<int>(ids.size());

  // log softmax, then gather the log-prob of each realized next token
  mx::array lp = mx::subtract(logits, mx::logsumexp(logits, -1, true));
  mx::array pred = ops::slice_axis(lp, 1, 0, L - 1);             // predicts ids[1:]
  std::vector<int> targets(ids.begin() + 1, ids.end());
  mx::array tgt = mx::reshape(row(targets), {1, L - 1, 1});
  mx::array chosen = mx::take_along_axis(pred, tgt, -1);
  mx::array nll = mx::negative(mx::mean(chosen));

  mx::array am = mx::argmax(pred, -1);
  mx::array hit = mx::mean(mx::astype(mx::equal(am, mx::reshape(row(targets),
                                                               {1, L - 1})),
                                      mx::float32));
  mx::eval({nll, hit});
  return {nll.item<float>(), hit.item<float>()};
}

void test_model(const std::string& path) {
  printf("\nmodel (%s)\n", path.c_str());
  Model m = load_model(path, /*verbose=*/true);
  Tokenizer tk(path + "/tokenizer.json");

  const std::string text =
      "The KV cache stores the key and value projections for every token the "
      "model has already processed, so that generating the next token does not "
      "require recomputing attention over the entire prefix. Without it, decoding "
      "would cost O(n^2) work in the sequence length rather than O(n).";
  std::vector<int> ids = tk.encode(text);
  printf("  scoring %zu tokens\n", ids.size());

  // 1. The model must actually model the text. A silently wrong convention (the
  //    zero-centered gammas, or gate-then-normalize) shows up here as a jump of
  //    several nats and nowhere else.
  std::pair<float, float> s = score(m, ids);
  char buf[200];
  snprintf(buf, sizeof(buf), "NLL %.3f nats/token, top-1 %.1f%% (expect < 2.5 nats)",
           s.first, 100.0 * s.second);
  check(s.first < 2.5f && s.second > 0.35f, buf);

  // 2. Segmented prefill must equal a single forward pass. Everything about
  //    cache reuse across turns rests on this.
  std::vector<LayerCache> c1 = m.make_cache();
  mx::array full = m.logits(m.hidden_states(row(ids), c1));
  std::vector<LayerCache> c2 = m.make_cache();
  mx::array part = mx::zeros({1});
  const int step = 7;  // deliberately not a divisor of the length
  for (size_t i = 0; i < ids.size(); i += step) {
    std::vector<int> chunk(ids.begin() + static_cast<long>(i),
                          ids.begin() + static_cast<long>(std::min(i + step, ids.size())));
    part = m.logits(m.hidden_states(row(chunk), c2));
  }
  mx::array full_last = ops::index_axis(full, 1, full.shape(1) - 1);
  mx::array part_last = ops::index_axis(part, 1, part.shape(1) - 1);
  mx::array a1 = mx::argmax(full_last, -1), a2 = mx::argmax(part_last, -1);
  mx::eval({a1, a2});
  const float e = rel_err(mx::astype(part_last, mx::float32),
                          mx::astype(full_last, mx::float32));
  snprintf(buf, sizeof(buf),
           "chunked prefill (step %d) == full forward: same argmax %s, rel err %.2e",
           step, a1.item<uint32_t>() == a2.item<uint32_t>() ? "yes" : "NO", e);
  check(a1.item<uint32_t>() == a2.item<uint32_t>() && e < 5e-2f, buf);
  check(c1[0].offset() == c2[0].offset() &&
            c1[0].offset() == static_cast<int>(ids.size()),
        "both caches ended at the same offset");

  // 3. Every layer's cache must track the same logical length -- the invariant
  //    hidden_states() relies on when it reads cache[0].offset().
  bool same = true;
  for (const LayerCache& c : c1) same &= (c.offset() == c1[0].offset());
  check(same, "all 64 layer caches agree on the offset");

  // 4. Speculative decoding must be LOSSLESS: at temp 0 it has to reproduce
  //    plain decoding token for token.
  if (m.mtp) {
    const char* prompt = "List the first five prime numbers, one per line.";
    std::string plain, spec;
    Session s1(m, tk, "", /*spec=*/false, 3);
    s1.turn(prompt, false, 0.0f, 0.95f, 20, 48,
            [&](const std::string& p) { plain += p; });
    Session s2(m, tk, "", /*spec=*/true, 3);
    s2.turn(prompt, false, 0.0f, 0.95f, 20, 48,
            [&](const std::string& p) { spec += p; });
    snprintf(buf, sizeof(buf),
             "speculative == plain at temp 0 (%d vs %d tokens, %.0f%% accept)",
             s1.last().n, s2.last().n,
             100.0 * s2.last().accepted / std::max(s2.last().drafted, 1));
    check(plain == spec && s1.last().n > 0, buf);
    if (plain != spec) {
      printf("        plain: %s\n         spec: %s\n", brief(plain, 80).c_str(),
             brief(spec, 80).c_str());
    }
  } else {
    printf("  [skip] no MTP head in this checkpoint\n");
  }
}

// ---------------------------------------------------------------------------
// sampler
// ---------------------------------------------------------------------------

void test_sampler() {
  printf("\nsampler\n");
  mx::random::seed(0);

  // The property that makes speculative decoding sound: summed over every
  // possible draft token, the marginal distribution of what gets emitted must
  // equal the TARGET distribution exactly. It is easy to break subtly (a missing
  // clamp, an unnormalized residual) in a way that only shows up as a slow drift
  // in output style, so check it exactly on small distributions.
  bool ok = true;
  float worst = 0;
  const int V = 6;
  for (int trial = 0; trial < 200; ++trial) {
    mx::array p = mx::softmax(mx::multiply(mx::random::normal({V}), mx::array(2.0f)), -1);
    mx::array q = mx::softmax(mx::multiply(mx::random::normal({V}), mx::array(2.0f)), -1);
    mx::eval({p, q});
    mx::array marg = mx::zeros({V});
    for (int d = 0; d < V; ++d) {
      mx::array pd = ops::index_axis(p, 0, d), qd = ops::index_axis(q, 0, d);
      mx::eval({pd, qd});
      const float pv = pd.item<float>(), qv = qd.item<float>();
      const float a = qv > 0 ? std::min(1.0f, pv / qv) : 1.0f;
      mx::array onehot =
          mx::astype(mx::equal(mx::arange(V, mx::int32), mx::array(d)), mx::float32);
      marg = mx::add(marg, mx::multiply(mx::array(qv * a), onehot));
      if (a < 1.0f) {
        marg = mx::add(marg, mx::multiply(mx::array(qv * (1.0f - a)),
                                          sample::residual(p, q)));
      }
    }
    mx::array err = mx::max(mx::abs(mx::subtract(marg, p)));
    mx::eval(err);
    worst = std::max(worst, err.item<float>());
  }
  ok = worst < 1e-5f;
  char buf[160];
  snprintf(buf, sizeof(buf),
           "rejection-sampling marginal == target (200 random 6-way dists, "
           "max err %.2e)", worst);
  check(ok, buf);

  // Whatever truncation the sampler applies, the result must still be a
  // normalized distribution -- the rejection test divides by these.
  mx::array lg = mx::random::normal({50});
  struct TP { float top_p; int top_k; };
  for (const TP& t : {TP{1.0f, 0}, TP{0.9f, 0}, TP{1.0f, 5}, TP{0.5f, 10}}) {
    mx::array pr = sample::to_probs(lg, 0.7f, t.top_p, t.top_k);
    mx::array s = mx::sum(pr);
    mx::array lo = mx::min(pr);
    mx::eval({s, lo});
    const bool good = std::abs(s.item<float>() - 1.0f) < 1e-5f && lo.item<float>() >= 0;
    snprintf(buf, sizeof(buf), "to_probs normalized (top_p=%.2f, top_k=%d) sum=%.6f",
             t.top_p, t.top_k, s.item<float>());
    check(good, buf);
  }

  // temp == 0 has to be a one-hot, so greedy is the same code path.
  mx::array g = sample::to_probs(lg, 0.0f, 1.0f, 0);
  mx::array gs = mx::sum(g), gm = mx::max(g);
  mx::eval({gs, gm});
  check(std::abs(gs.item<float>() - 1.0f) < 1e-6f && gm.item<float>() == 1.0f,
        "temp=0 gives a one-hot (greedy is a special case)");
}

// ---------------------------------------------------------------------------
// line reader -- driven through a real pty, because the behaviour under test IS
// the tty interaction. Each case sends its bytes as separate chunks with a gap
// between them, which is what really happens: the paste arrives as one burst,
// then the human presses Return a moment later. Pasting must never submit.
// ---------------------------------------------------------------------------

std::string escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else out += c;
  }
  return out;
}

// The child half: read messages from the pty and report what came through.
int lineread_child() {
  lineread::LineReader r;
  for (int i = 0; i < 8; ++i) {
    lineread::Line l = r.read("> ");
    if (l.status == lineread::Status::Eof) {
      printf("EOF\n");
      break;
    }
    printf("MSG=%s\n", escape(l.text).c_str());
    fflush(stdout);
  }
  return 0;
}

double mono() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Read from `fd` for `seconds`, returning everything that arrived. The pty must
// be drained continuously: if its buffer fills, the child blocks inside write()
// and the test deadlocks.
std::string drain(int fd, double seconds) {
  std::string out;
  const double end = mono() + seconds;
  while (mono() < end) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv {0, 20000};
    if (select(fd + 1, &fds, nullptr, nullptr, &tv) > 0) {
      char buf[4096];
      ssize_t n = ::read(fd, buf, sizeof(buf));
      if (n <= 0) break;
      out.append(buf, static_cast<size_t>(n));
    }
  }
  return out;
}

void test_lineread(const char* self) {
  printf("\nline reader (through a pty)\n");
  struct Case {
    const char* label;
    std::vector<std::string> chunks;
    const char* want;
  };
  const std::vector<Case> cases = {
      {"typed + backspace", {"helloX\x7f", "\r"}, "hello"},
      {"paste, trailing NL", {"aaa\nbbb\nccc\n", "\r"}, "aaa\\nbbb\\nccc\\n"},
      {"paste, NO trailing NL", {"aaa\nbbb\nccc", "\r"}, "aaa\\nbbb\\nccc"},
      {"bracketed paste", {"\x1b[200~one\ntwo\x1b[201~", "\r"}, "one\\ntwo"},
      {"CRLF paste", {"win\r\ndows\r\n", "\r"}, "win\\ndows\\n"},
      {"Ctrl-J newline", {"one", "\n", "two", "\r"}, "one\\ntwo"},
      {"history recall (up)", {"\x1b[A", "\r"}, "one\\ntwo"},
  };

  int master = -1;
  pid_t pid = forkpty(&master, nullptr, nullptr, nullptr);
  if (pid < 0) {
    printf("  [skip] forkpty failed\n");
    return;
  }
  if (pid == 0) {
    execl(self, self, "--lineread-child", nullptr);
    _exit(127);
  }

  drain(master, 0.4);  // let the child reach its first read
  for (const Case& c : cases) {
    // Accumulate everything for this case: the child prints its MSG= line as
    // soon as it sees the Return, i.e. during the gap after that write, not in
    // some final settling window.
    std::string seen;
    for (const std::string& chunk : c.chunks) {
      ssize_t ignored = write(master, chunk.data(), chunk.size());
      (void)ignored;
      seen += drain(master, 0.30);  // gap: a burst, then a keystroke
    }
    seen += drain(master, 0.35);

    // Take the last MSG= line the child reported.
    std::string got;
    size_t p = seen.rfind("MSG=");
    if (p != std::string::npos) {
      size_t e = seen.find('\n', p);
      got = seen.substr(p + 4, (e == std::string::npos ? seen.size() : e) - p - 4);
      while (!got.empty() && (got.back() == '\r' || got.back() == '\n')) got.pop_back();
    }
    const bool ok = got == c.want;
    check(ok, std::string(c.label) + " -> \"" + got + "\"");
    if (!ok) printf("        wanted \"%s\"\n", c.want);
  }

  close(master);
  int status = 0;
  waitpid(pid, &status, 0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model = "../qwen3.5-27b-4bit-uncensored";
  std::string tok_json;
  bool want_model = true;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--lineread-child")) return lineread_child();
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tok_json = argv[++i];
    else if (!strcmp(argv[i], "--no-model")) want_model = false;
    else {
      printf("usage: tests [--model DIR] [--tokenizer FILE] [--no-model]\n");
      return 1;
    }
  }
  if (tok_json.empty()) tok_json = model + "/tokenizer.json";

  test_unicode();
  try {
    test_tokenizer(tok_json);
  } catch (const std::exception& e) {
    printf("  [skip] tokenizer tests: %s\n", e.what());
  }
  test_sampler();
  test_delta();
  test_lineread(argv[0]);
  if (want_model) {
    try {
      test_model(model);
    } catch (const std::exception& e) {
      printf("  [skip] model tests: %s\n", e.what());
    }
  }

  printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
