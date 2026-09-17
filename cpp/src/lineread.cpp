#include "lineread.hpp"

#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <iostream>

#include "unicode.hpp"

namespace lineread {
namespace {

const char* kPasteStart = "\x1b[200~";
const char* kPasteEnd = "\x1b[201~";
const char* kPasteOn = "\x1b[?2004h";
const char* kPasteOff = "\x1b[?2004l";

// A keystroke never follows this fast, so anything already waiting is a paste.
constexpr double kBurstGap = 0.01;

void out(const std::string& s) {
  fwrite(s.data(), 1, s.size(), stdout);
  fflush(stdout);
}

bool more_pending(double gap) {
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(STDIN_FILENO, &fds);
  struct timeval tv;
  tv.tv_sec = static_cast<time_t>(gap);
  tv.tv_usec = static_cast<suseconds_t>((gap - tv.tv_sec) * 1e6);
  return select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) > 0;
}

// One raw byte. Returns false at EOF.
bool getbyte(unsigned char& b) {
  ssize_t n;
  do {
    n = ::read(STDIN_FILENO, &b, 1);
  } while (n < 0 && errno == EINTR);
  return n == 1;
}

size_t count_cp(const std::string& s, size_t from, size_t to) {
  size_t n = 0;
  for (size_t i = from; i < to && i < s.size();) {
    i += static_cast<size_t>(uni::seq_len(static_cast<unsigned char>(s[i])));
    ++n;
  }
  return n;
}

size_t prev_cp(const std::string& s, size_t cur) {
  if (cur == 0) return 0;
  size_t i = cur - 1;
  // step back over continuation bytes (10xxxxxx)
  while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;
  return i;
}

size_t next_cp(const std::string& s, size_t cur) {
  if (cur >= s.size()) return s.size();
  size_t n = static_cast<size_t>(uni::seq_len(static_cast<unsigned char>(s[cur])));
  return std::min(cur + n, s.size());
}

std::string crlf(const std::string& s) {
  std::string r;
  for (char c : s) {
    if (c == '\n') r += "\r\n";
    else r += c;
  }
  return r;
}

}  // namespace

bool available() { return isatty(STDIN_FILENO) && isatty(STDOUT_FILENO); }

namespace {

// Repaint the current visual line -- the text after the last newline.
void redraw(const std::string& prompt, const std::string& buf, size_t cur) {
  size_t nl = buf.rfind('\n');
  size_t line_start = (nl == std::string::npos) ? 0 : nl + 1;
  std::string tail = buf.substr(line_start);
  // The prompt only belongs on the first visual line.
  const std::string prefix = (nl == std::string::npos) ? prompt : "";

  std::string s = "\r" + prefix + tail + "\x1b[K";
  size_t back = count_cp(buf, cur, buf.size());
  if (back > 0) s += "\x1b[" + std::to_string(back) + "D";
  out(s);
}

std::string read_paste() {
  // Read up to the bracketed-paste end marker.
  std::string acc;
  const std::string end = kPasteEnd;
  unsigned char b;
  while (acc.size() < end.size() ||
         acc.compare(acc.size() - end.size(), end.size(), end) != 0) {
    if (!getbyte(b)) break;
    acc.push_back(static_cast<char>(b));
  }
  if (acc.size() >= end.size() &&
      acc.compare(acc.size() - end.size(), end.size(), end) == 0) {
    acc.resize(acc.size() - end.size());
  }
  // Normalize line endings the way the terminal may have mangled them.
  std::string r;
  for (size_t i = 0; i < acc.size(); ++i) {
    if (acc[i] == '\r') {
      if (i + 1 < acc.size() && acc[i + 1] == '\n') ++i;
      r += '\n';
    } else {
      r += acc[i];
    }
  }
  return r;
}

Line loop(const std::string& prompt, std::vector<std::string>& history) {
  std::string buf;
  size_t cur = 0;
  size_t hidx = history.size();
  std::string pending;   // partial escape sequence
  bool skip_lf = false;  // just consumed the CR of a CRLF pair

  const std::string paste_start = kPasteStart;

  while (true) {
    unsigned char b;
    if (!getbyte(b)) return {Status::Eof, ""};
    char ch = static_cast<char>(b);

    // ---- escape sequences: paste markers, arrows ----
    if (ch == '\x1b' || !pending.empty()) {
      pending += ch;
      if (pending == "\x1b") continue;
      if (paste_start.compare(0, pending.size(), pending) == 0) {
        if (pending == paste_start) {
          std::string text = read_paste();
          buf.insert(cur, text);
          cur += text.size();
          // Echo only up to the final newline; redraw() owns the current visual
          // line. Echoing the whole paste AND then redrawing prints the last
          // line twice.
          size_t nl = text.rfind('\n');
          if (nl != std::string::npos) out(crlf(text.substr(0, nl + 1)));
          redraw(prompt, buf, cur);
          pending.clear();
        }
        continue;
      }
      if (pending == "\x1b[" || pending == "\x1bO") continue;

      const char key = pending.back();
      pending.clear();
      if (key == 'D' && cur > 0) {              // left
        cur = prev_cp(buf, cur);
      } else if (key == 'C' && cur < buf.size()) {  // right
        cur = next_cp(buf, cur);
      } else if (key == 'A' && buf.find('\n') == std::string::npos) {  // up
        if (hidx > 0) {
          --hidx;
          buf = history[hidx];
          cur = buf.size();
        }
      } else if (key == 'B' && buf.find('\n') == std::string::npos) {  // down
        if (hidx + 1 < history.size()) {
          ++hidx;
          buf = history[hidx];
          cur = buf.size();
        } else {
          hidx = history.size();
          buf.clear();
          cur = 0;
        }
      } else if (key == 'H') {
        cur = 0;
      } else if (key == 'F') {
        cur = buf.size();
      }
      redraw(prompt, buf, cur);
      continue;
    }

    // ---- newline handling ----
    // Return arrives as CR (raw mode disables ICRNL). A bare LF therefore only
    // comes from a paste or from Ctrl-J, so LF always inserts and only CR can
    // submit. CRLF pastes are collapsed via skip_lf.
    if (ch == '\n') {
      if (skip_lf) {
        skip_lf = false;
        continue;
      }
      buf.insert(cur, 1, '\n');
      ++cur;
      out("\r\n");
      continue;
    }
    if (ch == '\r') {
      if (more_pending(kBurstGap)) {  // more bytes behind it => a paste
        buf.insert(cur, 1, '\n');
        ++cur;
        skip_lf = true;  // swallow a following LF
        out("\r\n");
        continue;
      }
      out("\r\n");
      // trailing-whitespace check mirrors Python's `buf.strip()`
      if (buf.find_first_not_of(" \t\r\n") != std::string::npos) {
        history.push_back(buf);
      }
      return {Status::Ok, buf};
    }
    if (ch == '\x03') {  // Ctrl-C
      out("\r\n");
      return {Status::Interrupted, ""};
    }
    if (ch == '\x04') {  // Ctrl-D
      if (buf.empty()) {
        out("\r\n");
        return {Status::Eof, ""};
      }
      continue;
    }
    if (ch == '\x7f' || ch == '\b') {  // backspace
      if (cur > 0) {
        size_t p = prev_cp(buf, cur);
        const bool crossed = buf[p] == '\n';
        buf.erase(p, cur - p);
        cur = p;
        // cannot repaint upward; move up and start the line clean
        if (crossed) out("\x1b[A\r\x1b[K");
      }
      redraw(prompt, buf, cur);
      continue;
    }
    if (ch == '\x15') {  // Ctrl-U: clear the current line
      size_t nl = buf.rfind('\n');
      buf.resize(nl == std::string::npos ? 0 : nl + 1);
      cur = buf.size();
      redraw(prompt, buf, cur);
      continue;
    }
    if (ch == '\x01') {  // Ctrl-A
      cur = 0;
      redraw(prompt, buf, cur);
      continue;
    }
    if (ch == '\x05') {  // Ctrl-E
      cur = buf.size();
      redraw(prompt, buf, cur);
      continue;
    }
    if (b < 0x20 && ch != '\t') continue;  // ignore other controls

    skip_lf = false;
    buf.insert(cur, 1, ch);
    ++cur;
    if (cur == buf.size() && more_pending(0)) {
      // Mid-burst (a paste on a terminal without bracketed paste): append-only
      // echo. A full redraw per character is O(n^2) output and visibly flickers.
      // The last byte of the burst falls through to the real redraw below.
      out(std::string(1, ch));
    } else {
      redraw(prompt, buf, cur);
    }
  }
}

}  // namespace

Line LineReader::read(const std::string& prompt) {
  if (!available()) {  // piped stdin: plain line read
    std::string line;
    if (!std::getline(std::cin, line)) return {Status::Eof, ""};
    return {Status::Ok, line};
  }

  struct termios old {};
  if (tcgetattr(STDIN_FILENO, &old) != 0) return {Status::Eof, ""};
  out(prompt + kPasteOn);

  struct termios raw = old;
  // cfmakeraw, then TCSANOW rather than TCSAFLUSH: TCSAFLUSH DISCARDS pending
  // input, which would silently eat a paste that arrives while the prompt is
  // being drawn, and it blocks until output drains.
  cfmakeraw(&raw);
  raw.c_cc[VMIN] = 1;
  raw.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSANOW, &raw);

  Line r = loop(prompt, history_);

  tcsetattr(STDIN_FILENO, TCSADRAIN, &old);
  out(kPasteOff);
  return r;
}

}  // namespace lineread
