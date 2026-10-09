// Self-tests. Run `make test`, or `./build/tests --model <dir>`.
//
// The tokenizer tests are pinned against the Python implementation (see
// tools/gen_golden.py) and need only tokenizer.json. The model tests need the
// weights and are skipped if the directory is absent, so `make test` stays
// useful without a 16 GB download.
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <util.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "delta.hpp"
#include "image.hpp"
#include "lineread.hpp"
#include "model.hpp"
#include "serve.hpp"
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

// True if `f` throws -- for the paths that have to refuse input rather than
// answer it.
bool throws(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
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

  // 2. Round trip. Comparing against NFC(input), since encode normalizes, less
  //    the literal-region markers, which encode drops by design.
  int bad = 0;
  for (const GoldenCase& g : kGolden) {
    std::string want = uni::nfc(strip_marks(g.text));
    std::string back = tk.decode(tk.encode(g.text));
    if (back != want) {
      ++bad;
      printf("  [FAIL] round trip %s\n         got  %s\n", brief(g.text).c_str(),
             brief(back).c_str());
    }
  }
  check(bad == 0, "decode(encode(x)) == NFC(x), markers aside");

  // 3a. Literal regions: special-token text inside one is text, and wrapping text
  //     that holds none changes nothing.
  {
    const std::string mention = "The README says: tokens end with <|im_end|> here.";
    const std::vector<int> ids =
        tk.encode("<|im_start|>tool\n" + literal(mention) + "<|im_end|>\n");
    const long ends = std::count(ids.begin(), ids.end(), tk.eos_id);
    check(ends == 1, "a quoted <|im_end|> in a literal region is text: " + std::to_string(ends) +
                         " end-of-turn token, the template's own");
    const std::string body = "Hello there, world.\n\nx  ";
    check(tk.encode("<|im_start|>user\n" + body + "<|im_end|>") ==
              tk.encode("<|im_start|>user\n" + literal(body) + "<|im_end|>"),
          "wrapping ordinary text as literal leaves its tokens unchanged");
  }

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

// ---------------------------------------------------------------------------
// the chat template and the HTTP layer behind serve mode
// ---------------------------------------------------------------------------

const char* kOpener = "<|im_start|>assistant\n";
const char* kNoThink = "<think>\n\n</think>\n\n";
const char* kThink = "<think>\n";

// What the model reads: a render without its literal-region markers. The
// template tests compare that against the template's own text.
std::string rc(const std::vector<Message>& msgs, const std::string& system, bool think,
               const std::vector<ToolSpec>& tools = {}) {
  return strip_marks(render_chat(msgs, system, think, tools));
}
std::string rt(const std::vector<Message>& msgs, size_t from, bool think) {
  return strip_marks(render_tail(msgs, from, think));
}

std::string blk(const char* role, const std::string& text) {
  return std::string("<|im_start|>") + role + "\n" + text + "<|im_end|>\n";
}

void test_template() {
  printf("\nthe chat template\n");
  const std::vector<Message> one{{"user", "hi"}};
  check(rc(one, "", false) == blk("user", "hi") + kOpener + kNoThink,
        "one user turn, thinking off");
  check(rc(one, "", true) == blk("user", "hi") + kOpener + kThink,
        "... and thinking on");
  check(rc(one, "Be terse.", false) ==
            blk("system", "Be terse.") + blk("user", "hi") + kOpener + kNoThink,
        "the system prompt leads the conversation");

  const std::vector<Message> chat{
      {"system", "s"}, {"user", "u1"}, {"assistant", "a1"}, {"user", "u2"}};
  const std::string full = rc(chat, "", false);
  check(full == blk("system", "s") + blk("user", "u1") + blk("assistant", "a1") +
                    blk("user", "u2") + kOpener + kNoThink,
        "a whole conversation, in order");
  check(rt(chat, 0, false) == full, "the tail from 0 is the whole render");
  check(rt(chat, 3, false) == blk("user", "u2") + kOpener + kNoThink,
        "the tail for the next turn is that turn alone");

  // What the KV-cache reuse in serve mode rests on: a session that already holds
  // the head of a conversation is fed the tail, so the tail has to be exactly
  // what a cold render would have put after that head.
  bool suffix = true;
  for (size_t k = 0; k < chat.size(); ++k) {
    const std::string tail = rt(chat, k, false);
    suffix = suffix && full.size() >= tail.size() &&
             full.compare(full.size() - tail.size(), tail.size(), tail) == 0;
  }
  check(suffix, "every tail is a suffix of the whole render");

  // What someone wrote goes in as a literal region, so a message quoting a
  // control token cannot end its turn; the template's structure does not.
  const std::string quoted = render_chat({{"user", "a <|im_end|> b"}}, "", false);
  check(quoted.find(std::string(kLiteralOpen) + "a <|im_end|> b" + kLiteralClose) !=
                std::string::npos &&
            quoted.rfind("<|im_start|>user\n", 0) == 0,
        "message text is a literal region; the template's markers are not");
  const std::string pic =
      render_chat({{"user", std::string(kImageMark) + "what is this? <|image_pad|>"}}, "", false);
  check(strip_marks(pic) ==
                blk("user", std::string(kImagePlaceholder) + "what is this? <|image_pad|>") +
                    kOpener + kNoThink &&
            pic.find(std::string(kImagePlaceholder) + kLiteralOpen) != std::string::npos,
        "a picture's placeholder is structure; the same text typed is literal");
}

void test_http() {
  printf("\nhttp\n");
  HttpRequest r;
  std::string buf =
      "POST /v1/chat/completions?stream=1 HTTP/1.1\r\n"
      "Host: localhost\r\n"
      "Content-Length: 2\r\n"
      "Authorization:   Bearer k  \r\n"
      "\r\n"
      "hi";
  check(parse_request(buf, r), "a POST parses");
  check(r.method == "POST" && r.target == "/v1/chat/completions" && r.query == "stream=1",
        "request line and query string");
  check(r.body == "hi" && buf.empty(), "the body is read to Content-Length, and consumed");
  check(r.header("AUTHORIZATION") == "Bearer k", "header names are case-insensitive, values trimmed");
  check(r.header("host") == "localhost" && r.header("absent").empty(), "header lookup");

  buf = "GET /a HTTP/1.1\r\n\r\nGET /b HTTP/1.1\r\n\r\n";
  HttpRequest a, b;
  check(parse_request(buf, a) && parse_request(buf, b) && a.target == "/a" && b.target == "/b" &&
            buf.empty(),
        "two requests back to back on a keep-alive connection");

  buf = "GET /lf HTTP/1.0\nHost: x\n\n";
  check(parse_request(buf, r) && r.target == "/lf" && r.version == "HTTP/1.0",
        "LF-only line endings");

  buf = "GET /half HTTP/1.1\r\nHost: x\r\n";
  const std::string before = buf;
  check(!parse_request(buf, r) && buf == before, "a half-arrived request is left alone");

  buf = "POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nshort";
  check(!parse_request(buf, r), "so is a half-arrived body");

  buf = "POST /x HTTP/1.1\r\nContent-Length: nosense\r\n\r\n";
  check(throws([&] { parse_request(buf, r); }), "a bad Content-Length throws");
  buf = "GARBAGE\r\n\r\n";
  check(throws([&] { parse_request(buf, r); }), "a malformed request line throws");
  buf = "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
  check(throws([&] { parse_request(buf, r); }), "chunked bodies are refused by name");
}

// Runs pieces through a filter the way a reply arrives, one token at a time.
std::string through_stops(const std::vector<std::string>& stops,
                          const std::vector<std::string>& pieces) {
  StopFilter f(stops);
  std::string out;
  for (const std::string& p : pieces) out += f.feed(p);
  out += f.finish();
  return out;
}

void test_stop_filter() {
  printf("\nstop sequences\n");
  check(through_stops({"STOP"}, {"hello ", "world"}) == "hello world",
        "with no stop in sight, everything comes out");
  check(through_stops({"STOP"}, {"hello ST", "OP and more"}) == "hello ",
        "a stop split across two pieces still stops");
  check(through_stops({"STOP", "###"}, {"a", "###", "b"}) == "a", "the earliest stop wins");
  check(through_stops({"STOP"}, {"S", "x", "y"}) == "Sxy", "a near miss is let through");
  check(through_stops({"ab"}, {"a"}) == "a", "a stop that never arrives comes out at the end");
  check(through_stops({"STOP"}, {"STOP"}) == "", "a stop at the very start leaves nothing");

  // The reply is bytes, and a piece can end in the middle of a character.
  check(utf8_complete("abc", 3) == 3, "ascii is complete");
  check(utf8_complete("caf\xC3\xA9", 5) == 5, "a whole 2-byte character is complete");
  check(utf8_complete("caf\xC3", 4) == 3, "a lead byte with no continuation is not");
  check(utf8_complete("\xE4\xB8", 2) == 0, "nor a 3-byte character cut short twice over");
  check(utf8_complete("\xF0\x9F\x98\x80", 4) == 4, "a whole 4-byte character is complete");
  check(utf8_complete("\xF0\x9F\x98", 3) == 0, "nor a 4-byte one cut short");
}

void test_tools() {
  printf("\ntool calls\n");

  // ---- the template ----
  const std::vector<ToolSpec> tools{
      ToolSpec{R"({"type": "function", "function": {"name": "bash"}})"}};
  const std::vector<Message> one{{"user", "list the files"}};
  const std::string p = rc(one, "", false, tools);
  const std::string head =
      "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n"
      "<tools>\n" + tools[0].json + "\n</tools>";
  check(p.rfind(head, 0) == 0, "the schemas open the prompt, in the template's own words");
  check(p.find("<function=example_function_name>") != std::string::npos,
        "...followed by the format the model was taught");
  check(p.find("\n\n<IMPORTANT>\nReminder:\n") != std::string::npos,
        "...and the rules that go with it");
  check(p.find("<|im_end|>\n<|im_start|>user\nlist the files<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n") != std::string::npos,
        "...then the messages, then the opener");
  check(rc(one, "Be terse.", false, tools)
            .find("</IMPORTANT>\n\nBe terse.<|im_end|>\n") != std::string::npos,
        "the system prompt shares that block, after the tools");
  check(rc(one, "", false).find("# Tools") == std::string::npos,
        "and with no tools there is no preamble at all");

  // ---- a conversation with a call in it ----
  std::vector<Message> chat{{"user", "list the files"}, {"assistant", ""},
                            {"tool", "a.txt\nb.txt"}, {"user", "thanks"}};
  chat[1].tool_calls.push_back(ToolCall{"", "bash", R"({"command": "ls"})"});
  const std::string r = rc(chat, "", false);
  check(r.find("<|im_start|>assistant\n<tool_call>\n<function=bash>\n<parameter=command>\n"
               "ls\n</parameter>\n</function>\n</tool_call><|im_end|>\n") !=
            std::string::npos,
        "an echoed call goes back in the model's own format");
  check(r.find("<|im_start|>user\n<tool_response>\na.txt\nb.txt\n</tool_response>"
               "<|im_end|>\n") != std::string::npos,
        "a tool result is a user block");
  check(rc({{"assistant", "done"}, {"tool", "1"}, {"tool", "2"}}, "", false)
            .find("<|im_start|>user\n<tool_response>\n1\n</tool_response>\n"
                  "<tool_response>\n2\n</tool_response><|im_end|>\n") != std::string::npos,
        "consecutive results share one user block");

  // In an agent loop the call is the last assistant turn, so it carries the
  // reasoning block the template opens for the turn being continued.
  std::vector<Message> loop{{"user", "list the files"}, {"assistant", ""}, {"tool", "a.txt"}};
  loop[1].tool_calls.push_back(ToolCall{"", "bash", R"({"command": "ls"})"});
  check(rc(loop, "", false)
            .find("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n") !=
            std::string::npos,
        "the turn after the last question keeps its reasoning block");

  // ---- reading the model's calls back ----
  ToolCallParser::Types types{{"bash", {{"command", "string"}, {"timeout", "number"}}}};
  ToolCallParser parse(types);
  ToolCallParser::Piece piece = parse.feed(
      "Sure.\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n"
      "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>");
  check(piece.text == "Sure.\n" && piece.calls.size() == 1, "a call comes out of the content");
  check(piece.calls.size() == 1 && piece.calls[0].name == "bash", "...with its name");
  check(piece.calls.size() == 1 &&
            piece.calls[0].arguments == R"({"command":"ls -la","timeout":30})",
        "...and arguments typed as the schema declares them");
  check(piece.calls.size() == 1 && piece.calls[0].id.rfind("call-", 0) == 0,
        "...and an id for the client to answer with");

  ToolCallParser split({});
  std::string text;
  size_t calls = 0;
  for (const char* part : {"before <tool", "_call>\n<function=f>\n<parameter=a>\n1\n</pa",
                           "rameter>\n</function>\n</tool_call> after"}) {
    ToolCallParser::Piece out = split.feed(part);
    text += out.text;
    calls += out.calls.size();
  }
  check(text == "before  after" && calls == 1,
        "a call split across pieces is still one call, and the prose survives");

  ToolCallParser::Piece json_form = ToolCallParser({}).feed(
      R"(<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>)");
  check(json_form.calls.size() == 1 &&
            json_form.calls[0].arguments == R"({"command":"ls"})",
        "the JSON spelling of a call works too");

  ToolCallParser open_tag({});
  ToolCallParser::Piece held = open_tag.feed("use <tool_call> to call a function");
  check(held.calls.empty() && held.text == "use ", "an unclosed tag is held back");
  check(open_tag.finish().text == "<tool_call> to call a function",
        "...and comes back out as text at the end");

  ToolCallParser::Piece not_call =
      ToolCallParser({}).feed("<tool_call>just words</tool_call>");
  check(not_call.calls.empty() && not_call.text == "<tool_call>just words</tool_call>",
        "a block that is not a call stays text");

  check(json::dump(json::parse(R"({"a": 1, "b": [true, null, "x"], "c": 1.5, "d": "q\"t"})")) ==
            R"({"a":1,"b":[true,null,"x"],"c":1.5,"d":"q\"t"})",
        "json::dump writes back what it read");

  // ---- the checkpoint's own template, held against the renderer ----
  const std::string qwen =
      "<|im_start|>system\n# Tools\n\n<tools>{}\n</tools>\n<tool_call><function=<parameter="
      "<tool_response><|im_end|>";
  check(template_mismatch(qwen).empty(), "this family's template passes");
  check(template_mismatch("<|start_header_id|>user<|end_header_id|>").find("no <|im_start|>") !=
            std::string::npos,
        "another family's template is refused");
  const std::string no_tools = template_mismatch("<|im_start|>user\nhi<|im_end|>\n");
  check(no_tools.find("tool_call tags") != std::string::npos &&
            no_tools.find("tool_response") != std::string::npos,
        "a template with no tool format says what is missing");
  check(!template_mismatch("").empty(), "so does a checkpoint with no template at all");
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


// ---------------------------------------------------------------------------
// images -- pinned to the Python port, which is pinned to transformers
// ---------------------------------------------------------------------------

struct ImageGolden {
  const char* file;
  long long max_pixels;
  int gh, gw;
  uint64_t hash;
};

#include "image_golden.inc"

uint64_t fnv1a64(const void* data, size_t n) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  uint64_t h = 0xCBF29CE484222325ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 0x100000001B3ULL;
  }
  return h;
}

std::string read_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::string s;
  char buf[1 << 16];
  for (size_t n; (n = fread(buf, 1, sizeof(buf), f)) > 0;) s.append(buf, n);
  fclose(f);
  return s;
}

void test_image(const std::string& data) {
  printf("\nimages (%s)\n", data.c_str());
  using image::smart_resize;
  check(smart_resize(480, 640) == std::make_pair(480, 640),
        "smart_resize keeps an in-range multiple of 32");
  {
    const std::pair<int, int> r = smart_resize(3024, 4032);
    check(r.first % 32 == 0 && r.second % 32 == 0 &&
              static_cast<long long>(r.first) * r.second <= image::kMaxPixels,
          "smart_resize shrinks a 12 MP photo to " + std::to_string(r.second) + "x" +
              std::to_string(r.first) + ", <= kMaxPixels");
    const std::pair<int, int> t = smart_resize(120, 200);
    check(static_cast<long long>(t.first) * t.second >= image::kMinPixels,
          "smart_resize grows a tiny image to >= kMinPixels");
  }
  // 80/32 = 2.5 and 112/32 = 3.5: half-to-even gives (64, 128), half-up (96, 128)
  check(smart_resize(80, 112, 1, 1000000000LL) == std::make_pair(64, 128),
        "rounding is half-to-even, as Python's round() in the reference is");
  bool threw = false;
  try {
    smart_resize(10, 3000);
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "an aspect ratio beyond 200:1 is refused");

  check(image::base64_decode("aGVsbG8gd29ybGQ=") == "hello world" &&
            image::base64_decode("aGVs\nbG8=") == "hello",
        "base64 decodes, skipping whitespace");
  threw = false;
  try {
    image::base64_decode("aGV$bG8=");
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "base64 outside the alphabet is refused");

  // Pixels identical to the Python port's, which match transformers' processor:
  // decode, EXIF orientation, alpha, resize, normalisation and patch order at once.
  for (const ImageGolden& g : kImageGolden) {
    image::Image im = image::preprocess(image::load(data + "/" + g.file), image::kMinPixels,
                                        g.max_pixels);
    mx::array px = mx::astype(im.pixels, mx::float32);
    mx::eval(px);
    const uint64_t h = fnv1a64(px.data<float>(), px.nbytes());
    char buf[200];
    snprintf(buf, sizeof(buf), "%-16s max %8lld: grid %dx%d, patches bit-identical to Python",
             g.file, g.max_pixels, im.gh, im.gw);
    check(im.gh == g.gh && im.gw == g.gw && h == g.hash, buf);
  }

  // The EXIF-6 JPEG is stored sideways; once righted it must look like the
  // upright PNG it was made from (JPEG noise aside), not like a rotation of it.
  {
    image::RGB a = image::load(data + "/arrow_exif6.jpg");
    image::RGB b = image::load(data + "/arrow_upright.png");
    double diff = 0;
    if (a.w == b.w && a.h == b.h) {
      for (size_t i = 0; i < a.px.size(); ++i) diff += std::abs(a.px[i] - b.px[i]);
      diff /= static_cast<double>(a.px.size());
    }
    char buf[160];
    snprintf(buf, sizeof(buf), "EXIF orientation 6 righted: %dx%d, mean |diff| %.2f vs upright",
             a.w, a.h, diff);
    check(a.w == b.w && a.h == b.h && diff < 2.0, buf);
  }

  // positions and placeholders
  std::vector<int> t, hh, ww;
  image::positions(10, 2, 3, t, hh, ww);
  check(t == std::vector<int>(6, 10) && hh == std::vector<int>{10, 10, 10, 11, 11, 11} &&
            ww == std::vector<int>{10, 11, 12, 10, 11, 12},
        "image positions: shared t, rows on h, cols on w");
  check(delta_contribution(ImageSpan{0, 3, 5}) == 5 - 15,
        "a 3x5 image advances positions by 5, not 15");
  const int pad = 248056;
  check(expand_image_pads({1, pad, 2, pad, 3}, {2, 3}, pad) ==
            std::vector<int>{1, pad, pad, 2, pad, pad, pad, 3},
        "one <|image_pad|> expands into one per token");
  const std::vector<ImageSpan> sp = find_image_spans({pad, pad, pad, pad, pad}, {{1, 2}, {1, 3}}, pad, 100);
  check(sp.size() == 2 && sp[0].start == 100 && sp[1].start == 102 && sp[1].n() == 3,
        "adjacent images in one pad run are split by their grids");
  std::string axes;
  for (int a : mrope_axes(64, {11, 11, 10})) axes += "THW"[a];
  check(axes == "THWTHWTHWTHWTHWTHWTHWTHWTHWTHWTH", "M-RoPE frequencies interleave THW...TH");
}

// ---------------------------------------------------------------------------
// the model looking at pictures
// ---------------------------------------------------------------------------

std::string lower_copy(std::string s) {
  for (char& c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
  return s;
}

void test_vision_model(const std::string& path, const std::string& data) {
  printf("\nvision (%s)\n", path.c_str());
  Model m = load_model(path, /*verbose=*/false);
  if (!m.has_vision()) {
    printf("  [skip] checkpoint has no vision tower\n");
    return;
  }
  Tokenizer tk(path + "/tokenizer.json");
  auto img = [&](const char* f) { return image::preprocess(image::load(data + "/" + f)); };
  auto ask = [&](Session& s, const std::vector<image::Image>& ims, const std::string& q, int n) {
    std::string out;
    s.turn(q, false, 0.0f, 0.95f, 20, n, [&](const std::string& p) { out += p; }, ims);
    return out;
  };
  {
    Session s(m, tk, "", true, 3);
    const std::string a =
        ask(s, {img("shapes.png")}, "What number is written in the image? The number only.", 16);
    check(a.find("42") != std::string::npos, "reads the number: \"" + brief(a) + "\"");
  }
  {
    Session s(m, tk, "", true, 3);
    const std::string a = ask(s, {img("arrow_exif6.jpg")},
                              "Which way does the arrow point: up, down, left or right? One word.", 8);
    check(lower_copy(a).rfind("up", 0) == 0,
          "EXIF orientation applied: the arrow points \"" + brief(a) + "\"");
  }
  {
    Session s(m, tk, "", true, 3);
    const std::string a = ask(s, {img("alpha_text.png")}, "What word is written? The word only.", 8);
    check(lower_copy(a).find("cat") != std::string::npos,
          "black text on a transparent background is legible: \"" + brief(a) + "\"");
  }
  {
    // A follow-up with no picture, then a second picture: the rope delta must
    // carry across turns and accumulate across images.
    Session s(m, tk, "", true, 3);
    ask(s, {img("shapes.png")}, "What colour is the square? One word.", 8);
    const std::string a2 = ask(s, {}, "And the circle? One word.", 8);
    ask(s, {img("arrow_exif6.jpg")}, "Which way does this arrow point? One word.", 8);
    const std::string a4 = ask(s, {}, "In the FIRST picture, what number was written? Number only.", 8);
    check(lower_copy(a2).find("blue") != std::string::npos && a4.find("42") != std::string::npos,
          "multi-turn: follow-up \"" + brief(a2) + "\", recall across images \"" + brief(a4) +
              "\" (rope delta " + std::to_string(s.rope_delta()) + ")");
  }
  {
    // Speculative decoding must not drift from plain with a picture in context:
    // the draft and the verify pass both take the shifted positions. A wrong
    // shift diverges at once, on a clearly-decided token. What is allowed is the
    // bf16 caveat speculative decoding always has -- the verify pass scores at
    // L=k+1 and plain decode at L=1, and on an EXACT tie the argmax can differ --
    // so a divergence passes only if the plain model's top two there are within
    // bf16 resolution of each other.
    const image::Image im = img("shapes.png");
    const std::string q = "List the shapes and their colours.";
    Session p(m, tk, "", false, 3), sp(m, tk, "", true, 3);
    const std::string a = ask(p, {im}, q, 40);
    const std::string b = ask(sp, {im}, q, 40);
    const std::vector<int>& ta = p.tokens();
    const std::vector<int>& tb = sp.tokens();
    size_t i = 0;
    while (i < ta.size() && i < tb.size() && ta[i] == tb[i]) ++i;
    if (a == b) {
      check(!a.empty(), "speculative == plain at temp 0, image in context");
    } else {
      // replay the plain path to the divergence and read its logits there
      const size_t n0 = static_cast<size_t>(p.last().prompt_n);
      std::vector<LayerCache> cache = m.make_cache();
      std::pair<mx::array, int> r =
          prefill(m, cache, std::vector<int>(ta.begin(), ta.begin() + static_cast<long>(n0)),
                  {ImageFeatures{m.embed_image(im), im.rows(), im.cols()}}, 0, 384);
      mx::array h = r.first;
      for (size_t k = n0; k < i; ++k) {
        mx::array tok({ta[k]}, mx::Shape{1, 1}, mx::int32);
        h = m.hidden_states(tok, cache, Rope::at(cache[0].offset() + r.second));
      }
      mx::array lg = mx::astype(ops::index_axis(ops::index_axis(m.logits(h), 1, 0), 0, 0), mx::float32);
      mx::array top = mx::topk(lg, 2);  // ascending: [second, first]
      mx::eval(top);
      const float gap = top.data<float>()[1] - top.data<float>()[0];
      char buf[200];
      snprintf(buf, sizeof(buf),
               "speculative vs plain, image in context: diverge at reply token %zu on a bf16 "
               "tie (top-2 gap %.4f)",
               i - n0, gap);
      check(i > n0 && gap <= 0.25f, buf);
    }
  }
  {
    // A message that merely quotes the placeholder -- a README, say -- is text:
    // it gets an answer, not a refusal, and no image tokens.
    Session s(m, tk, "", true, 3);
    const std::string a =
        ask(s, {}, "Repeat this exactly: <|image_pad|>", 12);
    long pads = std::count(s.tokens().begin(), s.tokens().end(), m.cfg.image_token_id);
    check(!a.empty() && pads == 0,
          "a quoted <|image_pad|> is text: answered (\"" + brief(a) + "\"), no image tokens");
  }
}

// ---------------------------------------------------------------------------
// serve mode with pictures -- over a socketpair, since a test may not bind a port
// ---------------------------------------------------------------------------

std::string base64_encode(const std::string& in) {
  static const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  size_t i = 0;
  for (; i + 2 < in.size(); i += 3) {
    const uint32_t v = (static_cast<unsigned char>(in[i]) << 16) |
                       (static_cast<unsigned char>(in[i + 1]) << 8) |
                       static_cast<unsigned char>(in[i + 2]);
    out += abc[(v >> 18) & 63];
    out += abc[(v >> 12) & 63];
    out += abc[(v >> 6) & 63];
    out += abc[v & 63];
  }
  if (i < in.size()) {
    uint32_t v = static_cast<unsigned char>(in[i]) << 16;
    if (i + 1 < in.size()) v |= static_cast<unsigned char>(in[i + 1]) << 8;
    out += abc[(v >> 18) & 63];
    out += abc[(v >> 12) & 63];
    out += i + 1 < in.size() ? abc[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

struct Reply {
  int status = 0;
  std::string body;
};

// One request/response on a keep-alive socket.
Reply http_post(int fd, const std::string& body) {
  const std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n"
                          "Content-Type: application/json\r\nContent-Length: " +
                          std::to_string(body.size()) + "\r\n\r\n" + body;
  for (size_t off = 0; off < req.size();) {
    const ssize_t w = ::write(fd, req.data() + off, req.size() - off);
    if (w <= 0) return Reply{};
    off += static_cast<size_t>(w);
  }
  std::string buf;
  char tmp[16384];
  auto more = [&]() {
    const ssize_t n = ::read(fd, tmp, sizeof(tmp));
    if (n <= 0) return false;
    buf.append(tmp, static_cast<size_t>(n));
    return true;
  };
  size_t head_end;
  while ((head_end = buf.find("\r\n\r\n")) == std::string::npos) {
    if (!more()) return Reply{};
  }
  Reply r;
  r.status = atoi(buf.c_str() + 9);  // "HTTP/1.1 200"
  const std::string head = lower_copy(buf.substr(0, head_end));
  const size_t cl = head.find("content-length:");
  if (cl != std::string::npos) {
    const size_t n = static_cast<size_t>(atol(head.c_str() + cl + 15));
    while (buf.size() < head_end + 4 + n) {
      if (!more()) break;
    }
    r.body = buf.substr(head_end + 4, n);
  } else {  // chunked: read to the terminating chunk
    while (buf.find("\r\n0\r\n\r\n", head_end) == std::string::npos) {
      if (!more()) break;
    }
    r.body = buf.substr(head_end + 4);
  }
  return r;
}

std::string image_part(const std::string& bytes, const char* mime = "image/png") {
  return "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:" + std::string(mime) +
         ";base64," + base64_encode(bytes) + "\"}}";
}

std::string text_part(const std::string& t) {
  json::Value v;
  v.type = json::Type::Str;
  v.str = t;
  return "{\"type\":\"text\",\"text\":" + json::dump(v) + "}";
}

std::string msg(const std::string& role, const std::string& content_json) {
  return "{\"role\":\"" + role + "\",\"content\":" + content_json + "}";
}

std::string jtext(const std::string& t) {
  json::Value v;
  v.type = json::Type::Str;
  v.str = t;
  return json::dump(v);
}

std::string chat_body(const std::vector<std::string>& msgs, int max_tokens, bool stream = false) {
  std::string m;
  for (size_t i = 0; i < msgs.size(); ++i) m += (i ? "," : "") + msgs[i];
  return "{\"model\":\"test\",\"temperature\":0,\"max_tokens\":" + std::to_string(max_tokens) +
         ",\"stream\":" + (stream ? "true" : "false") + ",\"messages\":[" + m + "]}";
}

void test_serve_parse() {
  printf("\nserve: image content parts\n");
  auto parse = [](const std::string& content, std::vector<std::string>& imgs) {
    return flatten_content(json::parse(content), imgs);
  };
  std::vector<std::string> imgs;
  const std::string t = parse("[{\"type\":\"text\",\"text\":\"a\"},"
                              "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,aGk=\"}},"
                              "{\"type\":\"text\",\"text\":\"b\"},"
                              "{\"type\":\"image_url\",\"image_url\":\"data:image/jpeg;base64,eW8=\"},"
                              "{\"type\":\"image\",\"image\":\"data:image/png;base64,Pz8=\"}]",
                              imgs);
  check(t == std::string("a") + kImageMark + "b" + kImageMark + kImageMark &&
            imgs == std::vector<std::string>{"hi", "yo", "??"},
        "parts in order: text and placeholders interleave, bytes decoded (object, string and "
        "`image` forms)");
  auto refused = [&](const std::string& url, const char* want) {
    std::vector<std::string> im;
    try {
      parse("[{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + url + "\"}}]", im);
    } catch (const std::exception& e) {
      return std::string(e.what()).find(want) != std::string::npos;
    }
    return false;
  };
  check(refused("https://example.com/cat.png", "never touches the network"),
        "an https image URL is refused: no network");
  check(refused("file:///etc/passwd", "disk"), "a file: image URL is refused: would read the disk");
  check(refused("data:image/png,rawbytes", "base64"), "a non-base64 data: URL is refused");
}

void test_serve_vision(const std::string& path, const std::string& data) {
  printf("\nserve: pictures over the real HTTP path (%s)\n", path.c_str());
  int sv[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
    printf("  [skip] socketpair failed\n");
    return;
  }
  const int one = 1;
  setsockopt(sv[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
  struct timeval tv {600, 0};
  setsockopt(sv[0], SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ServeOptions opt;
  opt.model_name = "test";
  std::thread server([&] { serve_socket(path, opt, sv[1]); });

  auto content_of = [](const Reply& r) {
    try {
      return json::parse(r.body)["choices"][0]["message"]["content"].as_str();
    } catch (const std::exception&) {
      return std::string();
    }
  };
  // usage.prompt_tokens is the whole prompt; what this request actually fed is
  // that minus the part the context already held.
  auto usage_of = [](const Reply& r, const char* field) {
    try {
      const json::Value u = json::parse(r.body)["usage"];
      if (std::string(field) == "cached_tokens") {
        return static_cast<int>(u["prompt_tokens_details"]["cached_tokens"].as_int(-1));
      }
      return static_cast<int>(u[field].as_int(-1));
    } catch (const std::exception&) {
      return -1;
    }
  };
  auto prompt_tokens = [&](const Reply& r) {
    return usage_of(r, "prompt_tokens") - usage_of(r, "cached_tokens");
  };
  auto error_of = [](const Reply& r) {
    try {
      return json::parse(r.body)["error"]["message"].as_str();
    } catch (const std::exception&) {
      return std::string();
    }
  };

  // What a continuing request should feed: its new user turn and the assistant
  // opener, and nothing else -- in particular not the previous reply again,
  // which the context already holds as the model wrote it.
  Tokenizer tk(path + "/tokenizer.json");
  auto tail_tokens = [&](const std::string& user_text) {
    return static_cast<int>(tk.encode(render_tail({Message("user", user_text)}, 0, false)).size());
  };

  const std::string shapes = read_file(data + "/shapes.png");
  const std::string alpha = read_file(data + "/alpha_text.png");
  const std::string q = "What is written in the image? Answer with exactly what is written.";
  const std::string u1 = msg("user", "[" + image_part(shapes) + "," + text_part(q) + "]");

  // 1. a picture, cold
  const Reply r1 = http_post(sv[0], chat_body({u1}, 12));
  const std::string c1 = content_of(r1);
  const int p1 = prompt_tokens(r1);
  check(r1.status == 200 && c1.find("42") != std::string::npos && p1 > 300,
        "picture in, cold: \"" + brief(c1) + "\", " + std::to_string(p1) + " prompt tokens");

  // 2. the client echoes the history and asks more: the picture is already in the
  //    context, so only the new turn is fed
  const std::string a1 = msg("assistant", jtext(c1));
  const std::string t2 = "What colour is the circle? One word.";
  const std::string u2 = msg("user", jtext(t2));
  const Reply r2 = http_post(sv[0], chat_body({u1, a1, u2}, 8));
  const std::string c2 = content_of(r2);
  const int p2 = prompt_tokens(r2);
  check(r2.status == 200 && lower_copy(c2).find("blue") != std::string::npos &&
            p2 == tail_tokens(t2),
        "echoed history continues: \"" + brief(c2) + "\", " + std::to_string(p2) +
            " new prompt tokens -- exactly the new turn (" + std::to_string(tail_tokens(t2)) +
            "), the echoed reply not fed again");

  // ... while usage reports the whole context, so a client can size its window:
  //    everything the first turn left behind is counted as cached
  const int whole1 = usage_of(r1, "prompt_tokens") + usage_of(r1, "completion_tokens");
  check(usage_of(r1, "cached_tokens") == 0 && usage_of(r2, "cached_tokens") > whole1 &&
            usage_of(r2, "prompt_tokens") == usage_of(r2, "cached_tokens") + p2,
        "usage counts the whole context: " + std::to_string(usage_of(r2, "prompt_tokens")) +
            " prompt tokens, " + std::to_string(usage_of(r2, "cached_tokens")) + " of them cached");

  // ... and again: a third turn in a row feeds only its own turn too
  const std::string a2 = msg("assistant", jtext(c2));
  const std::string t3 = "And the square? One word.";
  const std::string u3a = msg("user", jtext(t3));
  const Reply r2b = http_post(sv[0], chat_body({u1, a1, u2, a2, u3a}, 8));
  const std::string c2b = content_of(r2b);
  check(r2b.status == 200 && lower_copy(c2b).find("red") != std::string::npos &&
            prompt_tokens(r2b) == tail_tokens(t3),
        "a third turn continues the same way: \"" + brief(c2b) + "\", " +
            std::to_string(prompt_tokens(r2b)) + " new prompt tokens");

  // An agent's side request between two turns -- a title, a recap, each with a
  // system prompt of its own -- is a conversation of its own. It takes another
  // slot, and the next turn of this one still feeds only its own tokens.
  const Reply rs = http_post(sv[0], chat_body({msg("system", jtext("You are a title generator.")),
                                               msg("user", jtext("hi"))},
                                              4));
  const std::string a3 = msg("assistant", jtext(c2b));
  const std::string t4 = "And the background? One word.";
  const Reply r4 = http_post(sv[0], chat_body({u1, a1, u2, a2, u3a, a3, msg("user", jtext(t4))}, 8));
  check(rs.status == 200 && usage_of(rs, "cached_tokens") == 0 && r4.status == 200 &&
            prompt_tokens(r4) == tail_tokens(t4),
        "a side request between turns leaves the conversation cached: the next turn feeds " +
            std::to_string(prompt_tokens(r4)) + " new prompt tokens, exactly its own");

  // A history that leaves out the reply is not the conversation the context
  // holds, so it has to start over rather than append after the reply.
  const Reply r2c = http_post(sv[0], chat_body({u1, u2}, 8));
  check(r2c.status == 200 && prompt_tokens(r2c) > 300,
        "a history without the reply starts over: " + std::to_string(prompt_tokens(r2c)) +
            " prompt tokens");

  // 3. the same words about a DIFFERENT picture is a different conversation: it
  //    must not be appended to the cached one
  const std::string u1b = msg("user", "[" + image_part(alpha) + "," + text_part(q) + "]");
  const std::string u3 = msg("user", jtext("Say OK."));
  const Reply r3 = http_post(sv[0], chat_body({u1b, a1, u2, a2, u3}, 8));
  const int p3 = prompt_tokens(r3);
  check(r3.status == 200 && p3 > 72,
        "a swapped picture starts over: " + std::to_string(p3) + " prompt tokens, not a tail");

  // 4. refusals, each a 400 that leaves the cached conversation alone
  struct Bad {
    const char* what;
    std::string body;
    const char* want;
  };
  const std::vector<Bad> bads = {
      {"https URL",
       chat_body({msg("user", "[{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://x/y.png\"}}]")}, 4),
       "network"},
      {"file: URL",
       chat_body({msg("user", "[{\"type\":\"image_url\",\"image_url\":{\"url\":\"file:///etc/hosts\"}}]")}, 4),
       "disk"},
      {"bytes that are not a picture",
       chat_body({msg("user", "[" + image_part("hello, not an image") + "," + text_part("?") + "]")}, 4),
       "image 1"},
      {"a picture in a system message",
       chat_body({msg("system", "[" + image_part(shapes) + "]"), msg("user", jtext("hi"))}, 4),
       "only user messages"},
  };
  for (const Bad& b : bads) {
    const Reply r = http_post(sv[0], b.body);
    const std::string e = error_of(r);
    check(r.status == 400 && e.find(b.want) != std::string::npos,
          std::string("refused with 400: ") + b.what + " (\"" + brief(e, 60) + "\")");
  }

  // 5. ... and after them the conversation still continues from where it was
  const std::string c3 = content_of(r3);
  const std::string t5 = "Say OK again.";
  const Reply r5 = http_post(sv[0], chat_body({u1b, a1, u2, a2, u3, msg("assistant", jtext(c3)),
                                               msg("user", jtext(t5))},
                                              8));
  const int p5 = prompt_tokens(r5);
  check(r5.status == 200 && p5 == tail_tokens(t5),
        "after the refusals the cached conversation continues: " + std::to_string(p5) +
            " new prompt tokens, exactly the new turn");

  // 6. What an agent does to this repository: read a file that quotes the
  //    template's control tokens, and hand it back as a tool result. That is
  //    text -- no refusal, and no turn boundary injected into the prompt.
  {
    const std::string readme =
        "Pictures replace <|image_pad|> tokens; every turn ends with <|im_end|>.";
    const std::string call =
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\",\"type\":"
        "\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}]}";
    const Reply r = http_post(
        sv[0],
        "{\"model\":\"test\",\"temperature\":0,\"max_tokens\":16,\"messages\":[" +
            msg("user", jtext("Read README.md and quote its first line.")) + "," + call + "," +
            "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":" + jtext(readme) + "}]," +
            "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":"
            "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}}}]}");
    // the rendered prompt, tokenized as the server does, holds exactly the
    // template's own end-of-turn tokens -- the quoted one is text
    std::vector<Message> msgs{Message("user", "Read README.md and quote its first line."),
                              Message("assistant", ""), Message("tool", readme)};
    msgs[1].tool_calls.push_back(ToolCall{"c1", "read", "{\"path\":\"README.md\"}"});
    const std::vector<int> ids = tk.encode(render_chat(msgs, "", false));
    const long ends = std::count(ids.begin(), ids.end(), tk.eos_id);
    const long pads = std::count(ids.begin(), ids.end(), 248056);
    check(r.status == 200 && ends == 3 && pads == 0,
          "a tool result quoting <|image_pad|> and <|im_end|> is text: " +
              std::to_string(r.status) + ", " + std::to_string(ends) +
              " end-of-turn tokens (one per block), no image tokens");
  }

  // 7. streaming, with a picture
  const Reply r6 = http_post(
      sv[0], chat_body({msg("user", "[" + image_part(shapes) + "," +
                                        text_part("What number is written? Number only.") + "]")},
                       8, /*stream=*/true));
  // Digits are tokenized one at a time, so "42" arrives as two deltas: put the
  // stream back together before looking for it.
  std::string streamed;
  for (size_t at = r6.body.find("data: {"); at != std::string::npos;
       at = r6.body.find("data: {", at + 1)) {
    const size_t end = r6.body.find("\n\n", at);
    try {
      streamed += json::parse(r6.body.substr(at + 6, end - at - 6))["choices"][0]["delta"]["content"]
                      .as_str();
    } catch (const std::exception&) {
    }
  }
  check(r6.status == 200 && streamed.find("42") != std::string::npos &&
            r6.body.find("[DONE]") != std::string::npos,
        "a streamed reply about a picture: \"" + brief(streamed) + "\"");

  // 8. The slots are bounded: a conversation survives slots-1 others started
  //    after it, and is displaced by the slots-th.
  auto side_requests = [&](int n, const std::string& tag) {
    for (int i = 0; i < n; ++i) {
      http_post(sv[0], chat_body({msg("system", jtext("Side request " + tag + std::to_string(i))),
                                  msg("user", jtext("Say OK."))},
                                 2));
    }
  };
  const std::string y1 = msg("user", jtext("Say yes."));
  const Reply ry1 = http_post(sv[0], chat_body({y1}, 4));
  side_requests(opt.slots - 1, "a");
  const std::string y2 = msg("user", jtext("Say no."));
  const std::vector<std::string> y_hist{y1, msg("assistant", jtext(content_of(ry1))), y2};
  const Reply ry2 = http_post(sv[0], chat_body(y_hist, 4));
  check(ry2.status == 200 && prompt_tokens(ry2) == tail_tokens("Say no."),
        "a conversation outlives " + std::to_string(opt.slots - 1) +
            " others started after it: " + std::to_string(prompt_tokens(ry2)) +
            " new prompt tokens");
  side_requests(opt.slots, "b");
  std::vector<std::string> y_more = y_hist;
  y_more.push_back(msg("assistant", jtext(content_of(ry2))));
  y_more.push_back(msg("user", jtext("Say maybe.")));
  const Reply ry3 = http_post(sv[0], chat_body(y_more, 4));
  check(ry3.status == 200 && usage_of(ry3, "cached_tokens") == 0,
        "... and the " + std::to_string(opt.slots) + "th displaces it: " +
            std::to_string(prompt_tokens(ry3)) + " prompt tokens from scratch");

  close(sv[0]);  // the server sees EOF, closes its end and returns
  server.join();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model = "../qwen3.5-27b-4bit-uncensored";
  std::string tok_json;
  bool want_model = true;
  // the shared test pictures live at the repository root, next to cpp/
  std::string data = std::string(argv[0]).rfind('/') == std::string::npos
                         ? "../testdata"
                         : std::string(argv[0]).substr(0, std::string(argv[0]).rfind('/')) +
                               "/../../testdata";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--lineread-child")) return lineread_child();
    if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tok_json = argv[++i];
    else if (!strcmp(argv[i], "--no-model")) want_model = false;
    else if (!strcmp(argv[i], "--data") && i + 1 < argc) data = argv[++i];
    else {
      printf("usage: tests [--model DIR] [--tokenizer FILE] [--data DIR] [--no-model]\n");
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
  test_template();
  test_tools();
  test_http();
  test_stop_filter();
  test_lineread(argv[0]);
  test_serve_parse();
  try {
    test_image(data);
  } catch (const std::exception& e) {
    printf("  [skip] image tests: %s\n", e.what());
  }
  if (want_model) {
    try {
      test_model(model);
    } catch (const std::exception& e) {
      printf("  [skip] model tests: %s\n", e.what());
    }
    mx::clear_cache();
    try {
      test_vision_model(model, data);
    } catch (const std::exception& e) {
      printf("  [skip] vision tests: %s\n", e.what());
    }
    mx::clear_cache();
    // last: its model thread holds the model for the rest of the process
    test_serve_vision(model, data);
  }

  printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
