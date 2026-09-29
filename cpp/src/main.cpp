// ai -- a local terminal assistant. Interactive chat, nothing written to disk.
//
//     ./ai                        # start chatting
//     ./ai --think                # show the model's reasoning
//     ./ai --system "Be terse."
//     ./ai --prompt "..." -n 200  # one-shot, non-interactive
//     ./ai --image photo.jpg --prompt "What is this?"
//     ./ai --serve                # OpenAI-compatible API on 127.0.0.1:8080
//
// Everything lives in memory: the conversation, the KV cache, the editing
// history. Quit and it is gone. No files are created, and none are read except
// the model weights.
//
// In-chat commands: /help /new /think /temp /system /image /paste /stats /exit
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "image.hpp"
#include "lineread.hpp"
#include "model.hpp"
#include "serve.hpp"
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
  /image <path>...  attach pictures to your next message (drag a file here);
                    /image alone lists them, /image clear drops them
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

std::string basename_of(const std::string& p) {
  size_t s = p.rfind('/');
  return s == std::string::npos ? p : p.substr(s + 1);
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
  std::string model = "/Users/gluo/Documents/Qwen3.8-27B-MLX/qwen3.5-9b-4bit-uncensored";
  std::string system;
  std::string prompt;  // non-empty => one-shot mode
  bool think = false;
  bool no_spec = false;
  float temp = 0.7f;
  float top_p = 0.95f;
  int top_k = 20;
  int draft = 3;
  int max_tokens = 1024;
  std::vector<std::string> images;  // attached to the first message
  long long max_pixels = image::kMaxPixels;

  // serve mode
  bool serve = false;
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string api_key;
  std::string model_name;  // defaults to the model directory's own name
  std::string dump_dir;    // non-empty => write every request and reply there
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
    else if (f == "--image") a.images.push_back(next());
    else if (f == "--max-pixels") a.max_pixels = atoll(next().c_str());
    else if (f == "--serve") a.serve = true;
    else if (f == "--host") a.host = next();
    else if (f == "--port") a.port = atoi(next().c_str());
    else if (f == "--api-key") a.api_key = next();
    else if (f == "--model-name") a.model_name = next();
    else if (f == "--dump") a.dump_dir = next();
    else if (f == "-h" || f == "--help") {
      printf("usage: ai [--model DIR] [--system TEXT] [--think] [--temp F]\n"
             "          [--top-p F] [--top-k N] [-k N] [--no-spec] [-n N]\n"
             "          [--prompt TEXT] [--image PATH]... [--max-pixels N]\n"
             "       ai --serve [--host IP] [--port N] [--api-key KEY]\n"
             "          [--model-name NAME] [--dump DIR] [--max-pixels N]\n\n"
             "  --image attaches a picture to the first message (repeatable).\n"
             "  --max-pixels downscales pictures above N pixels; every 32x32 is one\n"
             "  token, and the default %lld caps a picture at %lld tokens.\n%s",
             image::kMaxPixels, image::kMaxPixels / (image::kFactor * image::kFactor), kHelp);
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
  char buf[400];
  int len = snprintf(buf, sizeof(buf), "prefill %.2fs", st.prefill_t);
  if (st.images) {
    len += snprintf(buf + len, sizeof(buf) - static_cast<size_t>(len),
                    " (%d picture%s, %.2fs in the vision tower)", st.images,
                    st.images == 1 ? "" : "s", st.vision_t);
  }
  len += snprintf(buf + len, sizeof(buf) - static_cast<size_t>(len),
                  " | %d tokens in %.2fs (%.1f tok/s)", st.n, st.dt,
                  st.n / (st.dt > 0 ? st.dt : 1e-9));
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

// Splits a command argument into paths the way a shell would: whitespace
// separates, quotes group, and a backslash escapes the next character -- which
// is what Terminal writes when a file is dragged in (My\ Photo.png). Throws on
// an unterminated quote.
std::vector<std::string> split_paths(const std::string& arg) {
  std::vector<std::string> out;
  std::string cur;
  bool have = false;
  char quote = 0;
  for (size_t i = 0; i < arg.size(); ++i) {
    const char c = arg[i];
    if (quote) {
      if (c == quote) {
        quote = 0;
      } else if (c == '\\' && quote == '"' && i + 1 < arg.size()) {
        cur += arg[++i];
      } else {
        cur += c;
      }
    } else if (c == '\'' || c == '"') {
      quote = c;
      have = true;
    } else if (c == '\\' && i + 1 < arg.size()) {
      cur += arg[++i];
      have = true;
    } else if (c == ' ' || c == '\t') {
      if (have) out.push_back(cur);
      cur.clear();
      have = false;
    } else {
      cur += c;
      have = true;
    }
  }
  if (quote) throw std::runtime_error("unterminated quote");
  if (have) out.push_back(cur);
  return out;
}

std::string expand_home(const std::string& p) {
  if (p.size() >= 1 && p[0] == '~' && (p.size() == 1 || p[1] == '/')) {
    const char* home = getenv("HOME");
    if (home) return std::string(home) + p.substr(1);
  }
  return p;
}

struct Attachment {
  std::string name;
  image::Image img;
};

// Decodes and preprocesses the pictures now, so a bad path is reported at once;
// the vision tower runs when the message is sent.
void attach(const Model& model, const std::vector<std::string>& paths, long long max_pixels,
            std::vector<Attachment>& attached) {
  if (!model.has_vision()) {
    printf("%sthis checkpoint has no vision tower; it cannot take pictures%s\n", kDim, kReset);
    return;
  }
  for (const std::string& raw : paths) {
    const std::string p = expand_home(raw);
    try {
      image::Image im = image::preprocess(image::load(p), image::kMinPixels, max_pixels);
      const std::string name = basename_of(p);
      printf("%sattached %s: %dx%d", kDim, name.c_str(), im.w, im.h);
      if (im.w != im.rw || im.h != im.rh) printf(" -> %dx%d", im.rw, im.rh);
      printf(", %d tokens%s\n", im.n_tokens(), kReset);
      attached.push_back(Attachment{name, im});
    } catch (const std::exception& e) {
      printf("%scannot attach %s: %s%s\n", kDim, p.c_str(), e.what(), kReset);
    }
  }
}

void generate(Session& s, const std::string& text, bool think, const Args& a,
              float temp, const std::vector<image::Image>& images = {}) {
  ThinkStyler styler(think);
  try {
    s.turn(text, think, temp, a.top_p, a.top_k, a.max_tokens,
           [&](const std::string& piece) { styler.feed(piece); }, images);
  } catch (const std::exception& e) {
    styler.finish();
    printf("%s%s%s\n", kDim, e.what(), kReset);
    return;
  }
  styler.finish();
  printf("\n");
  if (take_interrupt()) printf("%s^C interrupted%s\n", kDim, kReset);
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);
  const std::string path = resolve_model(a.model, argv[0]);

  // Serve mode takes over the process before any of the terminal machinery is
  // set up; it never reads the keyboard and never writes a reply to stdout. It
  // loads the model itself, on the thread that will run it -- see serve.cpp.
  if (a.serve) {
    ServeOptions opt;
    opt.host = a.host;
    opt.port = a.port;
    opt.api_key = a.api_key;
    opt.model_name = a.model_name.empty() ? basename_of(path) : a.model_name;
    opt.system = a.system;
    opt.think = a.think;
    opt.spec = !a.no_spec;
    opt.draft = a.draft;
    opt.dump_dir = a.dump_dir;
    opt.max_pixels = a.max_pixels;
    serve(path, opt);
    return 0;  // unreachable: serve() exits
  }

  const auto t0 = std::chrono::steady_clock::now();
  fprintf(stderr, "loading %s ...\n", a.model.c_str());
  Tokenizer tk(path + "/tokenizer.json");
  Model model = load_model(path, /*verbose=*/false);
  Session s(model, tk, a.system, !a.no_spec, a.draft);
  bool think = a.think;
  float temp = a.temp;
  fprintf(stderr, "ready in %.1fs%s%s   /help for commands, /exit to quit\n\n",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
          s.speculative() ? "" : "  (no MTP head: plain decoding)",
          model.has_vision() ? "" : "  (no vision tower: text only)");

  std::vector<Attachment> attached;  // for the next message
  attach(model, a.images, a.max_pixels, attached);
  auto take_attached = [&attached]() {
    std::vector<image::Image> out;
    for (Attachment& at : attached) out.push_back(at.img);
    attached.clear();  // whatever happens, they went with this message
    return out;
  };

  install_interrupt_handler();

  // One-shot mode: generate once and exit. Useful for scripting and for diffing
  // against the Python implementation.
  if (!a.prompt.empty()) {
    generate(s, a.prompt, think, a, temp, take_attached());
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
      } else if (cmd == "image") {
        if (arg == "clear") {
          attached.clear();
          printf("%sattachments dropped%s\n", kDim, kReset);
        } else if (!arg.empty()) {
          try {
            attach(model, split_paths(arg), a.max_pixels, attached);
          } catch (const std::exception& e) {
            printf("%scannot read the path: %s%s\n", kDim, e.what(), kReset);
          }
        } else if (!attached.empty()) {
          std::string names;
          for (const Attachment& at : attached) names += (names.empty() ? "" : ", ") + at.name;
          printf("%sattached: %s%s\n", kDim, names.c_str(), kReset);
        } else {
          printf("%snothing attached; /image <path> to attach%s\n", kDim, kReset);
        }
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

    generate(s, text, think, a, temp, take_attached());
  }

  fprintf(stderr, "%sconversation discarded%s\n", kDim, kReset);
  return 0;
}
