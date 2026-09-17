// ai -- a local terminal assistant. Interactive chat, nothing written to disk.
//
//     ./ai                        # start chatting
//     ./ai --think                # show the model's reasoning
//     ./ai --system "Be terse."
//     ./ai --prompt "..." -n 200  # one-shot, non-interactive
//
// Everything lives in memory: the conversation, the KV cache, the editing
// history. Quit and it is gone. No files are created, and none are read except
// the model weights.
//
// In-chat commands: /help /new /think /temp /system /paste /stats /exit
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

#include "lineread.hpp"
#include "model.hpp"
#include "session.hpp"
#include "tokenizer.hpp"

namespace {

const char* kDim = "\033[2m";
const char* kReset = "\033[0m";
const char* kBold = "\033[1m";

const char* kHelp = R"(
  /new              start a fresh conversation (clears context)
  /think [on|off]   show the model's reasoning (slower)
  /temp <float>     sampling temperature (0 = deterministic)
  /system <text>    set a system prompt (applies to the next /new)
  /paste            type a multi-line message, end with a single "." line
  /stats            timing for the last reply
  /help             this
  /exit             quit (Ctrl-D also works)

  Paste a multi-line paragraph and press Return -- it arrives as one message.
  Ctrl-J inserts a newline without sending. Ctrl-U clears the line.
  Ctrl-C interrupts a reply without quitting.
)";

bool is_dir(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string dirname_of(const std::string& p) {
  size_t s = p.rfind('/');
  return s == std::string::npos ? "." : p.substr(0, s);
}

// Resolve the model relative to the binary, not the cwd, so a launcher on PATH
// works from any directory.
std::string resolve_model(const std::string& want, const char* argv0) {
  if (is_dir(want)) return want;
  const std::string here = dirname_of(argv0);
  for (const std::string& base : {here, here + "/..", here + "/../.."}) {
    std::string p = base + "/" + want;
    if (is_dir(p)) return p;
  }
  fprintf(stderr, "model directory not found: %s\n", want.c_str());
  exit(1);
}

// Dims the reasoning block and swallows the `</think>` marker.
//
// The marker can arrive split across pieces, so hold back len(marker)-1 bytes
// until it is either found or the reply ends.
class ThinkStyler {
 public:
  explicit ThinkStyler(bool thinking) : active_(thinking) {
    if (active_) fputs(kDim, stdout);
  }

  void feed(const std::string& piece) {
    if (!active_) {
      write(piece);
      return;
    }
    buf_ += piece;
    size_t i = buf_.find(kMark);
    if (i != std::string::npos) {
      std::string rest = buf_.substr(i + strlen(kMark));
      size_t nb = rest.find_first_not_of('\n');
      write(buf_.substr(0, i) + kReset + (nb == std::string::npos ? "" : rest.substr(nb)));
      buf_.clear();
      active_ = false;  // rest of the reply goes out undimmed
      return;
    }
    const size_t hold = strlen(kMark) - 1;
    if (buf_.size() > hold) {
      write(buf_.substr(0, buf_.size() - hold));
      buf_.erase(0, buf_.size() - hold);
    }
  }

  void finish() {
    if (!buf_.empty()) write(buf_);
    if (active_) fputs(kReset, stdout);
    buf_.clear();
    active_ = false;
  }

 private:
  static void write(const std::string& s) {
    fwrite(s.data(), 1, s.size(), stdout);
    fflush(stdout);
  }
  static constexpr const char* kMark = "</think>";
  bool active_;
  std::string buf_;
};

struct Args {
  std::string model = "qwen3.5-27b-4bit-uncensored";
  std::string system;
  std::string prompt;  // non-empty => one-shot mode
  bool think = false;
  bool no_spec = false;
  float temp = 0.7f;
  float top_p = 0.95f;
  int top_k = 20;
  int draft = 3;
  int max_tokens = 1024;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", f.c_str());
        exit(1);
      }
      return argv[++i];
    };
    if (f == "--model") a.model = next();
    else if (f == "--system") a.system = next();
    else if (f == "--prompt") a.prompt = next();
    else if (f == "--think") a.think = true;
    else if (f == "--no-spec") a.no_spec = true;
    else if (f == "--temp") a.temp = strtof(next().c_str(), nullptr);
    else if (f == "--top-p") a.top_p = strtof(next().c_str(), nullptr);
    else if (f == "--top-k") a.top_k = atoi(next().c_str());
    else if (f == "-k" || f == "--draft") a.draft = atoi(next().c_str());
    else if (f == "-n" || f == "--max-tokens") a.max_tokens = atoi(next().c_str());
    else if (f == "-h" || f == "--help") {
      printf("usage: ai [--model DIR] [--system TEXT] [--think] [--temp F]\n"
             "          [--top-p F] [--top-k N] [-k N] [--no-spec] [-n N]\n"
             "          [--prompt TEXT]\n%s",
             kHelp);
      exit(0);
    } else {
      fprintf(stderr, "unknown option: %s (try --help)\n", f.c_str());
      exit(1);
    }
  }
  return a;
}

void print_stats(const Session& s) {
  const Stats& st = s.last();
  if (st.n == 0 && st.rounds == 0) {
    printf("%sno reply yet%s\n", kDim, kReset);
    return;
  }
  char buf[320];
  int len = snprintf(buf, sizeof(buf),
                     "prefill %.2fs | %d tokens in %.2fs (%.1f tok/s)", st.prefill_t,
                     st.n, st.dt, st.n / (st.dt > 0 ? st.dt : 1e-9));
  if (st.rounds) {
    len += snprintf(buf + len, sizeof(buf) - static_cast<size_t>(len),
                    " | %d passes, %.2f tok/pass, accept %.0f%%", st.rounds,
                    static_cast<double>(st.n) / st.rounds,
                    100.0 * st.accepted / std::max(st.drafted, 1));
  }
  snprintf(buf + len, sizeof(buf) - static_cast<size_t>(len),
           " | context %zu tokens | peak %.1f GB", s.context_tokens(),
           static_cast<double>(mx::get_peak_memory()) / 1e9);
  printf("%s%s%s\n", kDim, buf, kReset);
}

// Reads a multi-line message terminated by a lone "." line (the /paste command).
std::string read_dot_terminated() {
  printf("%smulti-line input; end with a single '.' line%s\n", kDim, kReset);
  std::string text, line;
  while (std::getline(std::cin, line)) {
    if (line == ".") break;
    text += line;
    text += '\n';
  }
  while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.pop_back();
  return text;
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

void generate(Session& s, const std::string& text, bool think, const Args& a,
              float temp) {
  ThinkStyler styler(think);
  s.turn(text, think, temp, a.top_p, a.top_k, a.max_tokens,
         [&](const std::string& piece) { styler.feed(piece); });
  styler.finish();
  printf("\n");
  if (take_interrupt()) printf("%s^C interrupted%s\n", kDim, kReset);
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);
  const std::string path = resolve_model(a.model, argv[0]);

  const auto t0 = std::chrono::steady_clock::now();
  fprintf(stderr, "loading %s ...\n", a.model.c_str());
  Tokenizer tk(path + "/tokenizer.json");
  Model model = load_model(path, /*verbose=*/false);
  Session s(model, tk, a.system, !a.no_spec, a.draft);
  bool think = a.think;
  float temp = a.temp;
  fprintf(stderr, "ready in %.1fs%s   /help for commands, /exit to quit\n\n",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
          s.speculative() ? "" : "  (no MTP head: plain decoding)");

  install_interrupt_handler();

  // One-shot mode: generate once and exit. Useful for scripting and for diffing
  // against the Python implementation.
  if (!a.prompt.empty()) {
    generate(s, a.prompt, think, a, temp);
    print_stats(s);
    return 0;
  }

  lineread::LineReader reader;
  while (true) {
    lineread::Line line = reader.read(std::string(kBold) + ">" + kReset + " ");
    if (line.status == lineread::Status::Eof) {
      printf("\n");
      break;
    }
    if (line.status == lineread::Status::Interrupted) {
      take_interrupt();
      continue;
    }
    // A trailing backslash continues onto another line.
    while (!line.text.empty() && line.text.back() == '\\') {
      line.text.pop_back();
      line.text += '\n';
      lineread::Line more = reader.read(std::string(kDim) + "…" + kReset + " ");
      if (more.status != lineread::Status::Ok) break;
      line.text += more.text;
    }

    std::string text = trim(line.text);
    if (text.empty()) continue;

    // Only a single line can be a command, so a pasted paragraph that happens to
    // start with "/" is still treated as text.
    bool send = false;
    if (text[0] == '/' && text.find('\n') == std::string::npos) {
      const size_t sp = text.find(' ');
      const std::string cmd = text.substr(1, sp == std::string::npos ? std::string::npos
                                                                     : sp - 1);
      const std::string arg =
          sp == std::string::npos ? "" : trim(text.substr(sp + 1));

      if (cmd == "exit" || cmd == "quit" || cmd == "q") break;
      if (cmd == "help") {
        printf("%s", kHelp);
      } else if (cmd == "new") {
        s.reset();
        printf("%snew conversation%s\n", kDim, kReset);
      } else if (cmd == "think") {
        think = arg.empty() ? !think : (arg != "off");
        printf("%sthinking %s%s\n", kDim, think ? "on" : "off", kReset);
      } else if (cmd == "temp") {
        if (!arg.empty()) {
          temp = strtof(arg.c_str(), nullptr);
          printf("%stemperature %.2f%s\n", kDim, temp, kReset);
        } else {
          printf("%stemperature is %.2f%s\n", kDim, temp, kReset);
        }
      } else if (cmd == "system") {
        s.system = arg;
        printf("%ssystem prompt %s; takes effect on /new%s\n", kDim,
               arg.empty() ? "cleared" : "set", kReset);
      } else if (cmd == "stats") {
        print_stats(s);
      } else if (cmd == "paste") {
        text = read_dot_terminated();
        if (text.empty()) continue;
        send = true;  // fall through to generation
      } else {
        printf("%sunknown command /%s -- try /help%s\n", kDim, cmd.c_str(), kReset);
      }
      if (!send) continue;
    }

    generate(s, text, think, a, temp);
  }

  fprintf(stderr, "%sconversation discarded%s\n", kDim, kReset);
  return 0;
}
