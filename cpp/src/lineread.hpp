// Raw-mode line reader with correct multi-line paste handling.
//
// WHY THIS EXISTS
// ---------------
// A line-buffered read cannot take a pasted paragraph. The tty in canonical mode
// hands over a line as soon as it sees a newline, so pasting
//
//     Reply to this note:
//     The meeting moved to 3pm.
//
// auto-submits "Reply to this note:" before you touch Enter, and the rest becomes
// separate turns. Draining the leftover lines recovers most of it, but not a
// paste whose selection has no trailing newline -- and that is the common case
// when copying a paragraph. The final partial line stays wedged in the tty's line
// buffer and is not retrievable.
//
// The only real fix is to stop the tty from buffering lines. In raw mode every
// byte arrives immediately, so a paste lands as one burst that we assemble
// ourselves. Two independent mechanisms decide what is a paste:
//
//   * BRACKETED PASTE. We ask the terminal to wrap pastes in \e[200~ ... \e[201~
//     (Terminal.app, iTerm2, VS Code, tmux all support it). Everything between
//     the markers is literal text, newlines included.
//   * BURST HEURISTIC, for terminals that lack it. On a newline, if more input is
//     already waiting, it came from a paste rather than a keystroke -- no human
//     produces the next byte within 10 ms of pressing Return.
//
// Editing is deliberately modest: characters, backspace, left/right, home/end,
// and in-memory history, plus Ctrl-J to insert a newline. Once the buffer
// contains a newline (i.e. a paste happened) history navigation is disabled,
// because redrawing a wrapped multi-line buffer correctly is a much bigger job
// than this tool needs -- after pasting a paragraph you press Return.
//
// The buffer is UTF-8 and the cursor is a byte offset, but it only ever moves by
// whole codepoints, and cursor-motion escapes are counted in codepoints -- which
// is what the reference implementation did. Double-width glyphs (CJK) therefore
// draw one cell narrower than they measure; harmless for editing a prompt.
//
// Nothing is persisted. History dies with the process.
#pragma once

#include <string>
#include <vector>

namespace lineread {

enum class Status { Ok, Eof, Interrupted };

struct Line {
  Status status = Status::Ok;
  std::string text;
};

// True when both stdin and stdout are a tty; otherwise `read` falls back to a
// plain getline so the binary still works under a pipe.
bool available();

class LineReader {
 public:
  Line read(const std::string& prompt);

 private:
  std::vector<std::string> history_;  // in memory only, never written
};

}  // namespace lineread
