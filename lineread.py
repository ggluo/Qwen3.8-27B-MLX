"""
Raw-mode line reader with correct multi-line paste handling.

WHY THIS EXISTS
---------------
`input()` plus libedit cannot read a pasted paragraph. The tty is in canonical
mode, so it hands over a line as soon as it sees a newline: pasting

    Reply to this note:
    The meeting moved to 3pm.

auto-submits "Reply to this note:" before you touch Enter, and the rest becomes
separate turns. Draining the leftover lines from stdin recovers most of it, but
not a paste whose selection has no trailing newline -- and that is the common
case when copying a paragraph. The final partial line stays wedged in the tty's
line buffer; measured, it is NOT retrievable by switching to cbreak and reading.

The only real fix is to stop the tty from buffering lines. In raw mode every
byte arrives immediately, so a paste lands as one burst that we can assemble
ourselves. Two independent mechanisms decide what is a paste:

  * BRACKETED PASTE. We ask the terminal to wrap pastes in \\e[200~ ... \\e[201~
    (supported by Terminal.app, iTerm2, VS Code, tmux). Everything between the
    markers is literal text, newlines included.
  * BURST HEURISTIC, for terminals that lack it. On a newline, if more input is
    already waiting, it came from a paste rather than a keystroke -- no human
    produces the next byte within 10 ms of pressing Return.

Editing is deliberately modest: characters, backspace, left/right, home/end, and
in-memory history, plus Ctrl-J to insert a newline. Once the buffer contains a newline (i.e. a paste happened),
history navigation is disabled, because redrawing a wrapped multi-line buffer
correctly is a much bigger job than this tool needs -- after pasting a paragraph
you press Return.

Nothing is persisted. History dies with the process.
"""

import codecs
import select
import sys

try:
    import termios
    import tty
    _HAVE_TTY = True
except ImportError:                      # non-POSIX
    _HAVE_TTY = False

PASTE_START, PASTE_END = "\x1b[200~", "\x1b[201~"
BURST_GAP = 0.01                         # a keystroke never follows this fast


def available() -> bool:
    return _HAVE_TTY and sys.stdin.isatty() and sys.stdout.isatty()


class LineReader:
    def __init__(self):
        self.history = []                # in memory only, never written
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")

    # ------------------------------------------------------------------
    def read(self, prompt: str) -> str:
        """Read one message. Raises EOFError on Ctrl-D, KeyboardInterrupt on Ctrl-C."""
        if not available():              # piped stdin, or no tty: plain readline
            line = sys.stdin.readline()
            if not line:
                raise EOFError
            return line.rstrip("\n")

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        sys.stdout.write(prompt + PASTE_ON)
        sys.stdout.flush()
        try:
            # TCSANOW, not tty.setraw's default TCSAFLUSH: TCSAFLUSH DISCARDS
            # pending input, which would silently eat a paste that arrives while
            # the prompt is being drawn, and it blocks until output drains.
            tty.setraw(fd, termios.TCSANOW)
            return self._loop(prompt)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write(PASTE_OFF)
            sys.stdout.flush()

    # ------------------------------------------------------------------
    def _more_pending(self, gap=BURST_GAP) -> bool:
        return bool(select.select([sys.stdin], [], [], gap)[0])

    def _getch(self) -> str:
        """One decoded character (blocking), handling split UTF-8 sequences."""
        while True:
            b = sys.stdin.buffer.raw.read(1)
            if not b:
                raise EOFError
            s = self._dec.decode(b)
            if s:
                return s

    def _redraw(self, prompt, buf, cur):
        """Repaint the current visual line -- the text after the last newline."""
        head, _, tail = buf.rpartition("\n")
        prefix = "" if head or buf.endswith("\n") else prompt
        line_start = len(buf) - len(tail)
        sys.stdout.write("\r" + prefix + tail + "\x1b[K")
        back = len(tail) - (cur - line_start)
        if back > 0:
            sys.stdout.write(f"\x1b[{back}D")
        sys.stdout.flush()

    def _loop(self, prompt):
        buf, cur, hidx = "", 0, len(self.history)
        pending = ""                      # partial escape sequence
        skip_lf = False                   # just consumed the CR of a CRLF pair

        while True:
            ch = self._getch()

            # ---- escape sequences: paste markers, arrows ----
            if ch == "\x1b" or pending:
                pending += ch if not pending else ch
                if pending == "\x1b":
                    continue
                if PASTE_START.startswith(pending):
                    if pending == PASTE_START:
                        text = self._read_paste()
                        buf = buf[:cur] + text + buf[cur:]
                        cur += len(text)
                        # Echo only up to the final newline; _redraw owns the
                        # current visual line. Echoing the whole paste AND then
                        # redrawing prints the last line twice.
                        if "\n" in text:
                            head = text[: text.rindex("\n") + 1]
                            sys.stdout.write(head.replace("\n", "\r\n"))
                        self._redraw(prompt, buf, cur)
                        pending = ""
                    continue
                if pending in ("\x1b[", "\x1bO"):
                    continue
                key, pending = pending, ""
                if key.endswith("D") and cur > 0:            # left
                    cur -= 1
                elif key.endswith("C") and cur < len(buf):   # right
                    cur += 1
                elif key.endswith("A") and "\n" not in buf:  # up: history
                    if hidx > 0:
                        hidx -= 1
                        buf = self.history[hidx]
                        cur = len(buf)
                elif key.endswith("B") and "\n" not in buf:  # down
                    if hidx < len(self.history) - 1:
                        hidx += 1
                        buf = self.history[hidx]
                        cur = len(buf)
                    else:
                        hidx, buf, cur = len(self.history), "", 0
                elif key.endswith("H"):
                    cur = 0
                elif key.endswith("F"):
                    cur = len(buf)
                self._redraw(prompt, buf, cur)
                continue

            # ---- newline handling ----
            # Return arrives as CR (raw mode disables ICRNL). A bare LF therefore
            # only comes from a paste or from Ctrl-J, so LF always inserts and
            # only CR can submit. CRLF pastes are collapsed via skip_lf.
            if ch == "\n":
                if skip_lf:
                    skip_lf = False
                    continue
                buf = buf[:cur] + "\n" + buf[cur:]
                cur += 1
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                continue
            if ch == "\r":
                if self._more_pending():      # more bytes behind it => a paste
                    buf = buf[:cur] + "\n" + buf[cur:]
                    cur += 1
                    skip_lf = True            # swallow a following LF
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    continue
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                if buf.strip():
                    self.history.append(buf)
                return buf
            if ch == "\x03":                                  # Ctrl-C
                sys.stdout.write("\r\n")
                raise KeyboardInterrupt
            if ch == "\x04":                                  # Ctrl-D
                if not buf:
                    sys.stdout.write("\r\n")
                    raise EOFError
                continue
            if ch in ("\x7f", "\b"):                          # backspace
                if cur > 0:
                    crossed = buf[cur - 1] == "\n"
                    buf = buf[:cur - 1] + buf[cur:]
                    cur -= 1
                    if crossed:            # cannot repaint upward; start clean
                        sys.stdout.write("\x1b[A\r\x1b[K")
                self._redraw(prompt, buf, cur)
                continue
            if ch == "\x15":                                  # Ctrl-U: clear line
                head, sep, _ = buf.rpartition("\n")
                buf = head + sep
                cur = len(buf)
                self._redraw(prompt, buf, cur)
                continue
            if ch == "\x01":                                  # Ctrl-A
                cur = 0
                self._redraw(prompt, buf, cur)
                continue
            if ch == "\x05":                                  # Ctrl-E
                cur = len(buf)
                self._redraw(prompt, buf, cur)
                continue
            if ch < " " and ch not in ("\t",):
                continue                                      # ignore other controls

            skip_lf = False
            buf = buf[:cur] + ch + buf[cur:]
            cur += 1
            if cur == len(buf) and self._more_pending(0):
                # Mid-burst (a paste on a terminal without bracketed paste):
                # append-only echo. A full redraw per character is O(n^2) output
                # and visibly flickers. The last character of the burst falls
                # through to the real redraw below.
                sys.stdout.write(ch)
            else:
                self._redraw(prompt, buf, cur)

    def _read_paste(self) -> str:
        """Read up to the bracketed-paste end marker."""
        out = ""
        while not out.endswith(PASTE_END):
            out += self._getch()
        return out[: -len(PASTE_END)].replace("\r\n", "\n").replace("\r", "\n")


PASTE_ON, PASTE_OFF = "\x1b[?2004h", "\x1b[?2004l"


# ----------------------------------------------------------------------


def _selftest():
    """Drive a real pty: typing, editing, history, and both paste paths."""
    import os
    import pty
    import time

    script = (
        "import sys;sys.path.insert(0,%r)\n"
        "from lineread import LineReader\n"
        "r = LineReader()\n"
        "for _ in range(7):\n"
        "    try: m = r.read('> ')\n"
        "    except EOFError: print('EOF');break\n"
        "    print('MSG=' + repr(m))\n"
        "    sys.stdout.flush()\n"
    ) % os.path.dirname(os.path.abspath(__file__))
    open("/tmp/_lr_probe.py", "w").write(script)

    # Each case is a list of byte chunks sent with a gap between them, which is
    # what really happens: the paste arrives as one burst, then the human presses
    # Return a moment later. Pasting never submits on its own.
    cases = [
        ("typed + backspace",      [b"helloX\x7f", b"\r"],            "'hello'"),
        ("paste, trailing NL",     [b"aaa\nbbb\nccc\n", b"\r"],       "'aaa\\nbbb\\nccc\\n'"),
        ("paste, NO trailing NL",  [b"aaa\nbbb\nccc", b"\r"],         "'aaa\\nbbb\\nccc'"),
        ("bracketed paste",        [b"\x1b[200~one\ntwo\x1b[201~", b"\r"], "'one\\ntwo'"),
        ("CRLF paste",             [b"win\r\ndows\r\n", b"\r"],       "'win\\ndows\\n'"),
        ("Ctrl-J newline",         [b"one", b"\n", b"two", b"\r"],    "'one\\ntwo'"),
        ("history recall (up)",    [b"\x1b[A", b"\r"],                "'one\\ntwo'"),
    ]
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("python3", ["python3", "/tmp/_lr_probe.py"])

    # A pty master must be drained continuously: if its output buffer fills, the
    # child blocks inside tcsetattr/write and the test deadlocks.
    import threading
    seen, lock, stop = [], threading.Lock(), []

    def drain():
        while not stop:
            try:
                if select.select([fd], [], [], 0.05)[0]:
                    c = os.read(fd, 4096)
                    if not c:
                        break
                    with lock:
                        seen.append(c.decode(errors="replace"))
            except OSError:
                break

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    time.sleep(1.0)
    ok = True
    for label, chunks, want in cases:
        with lock:
            seen.clear()
        for c in chunks:
            os.write(fd, c)
            time.sleep(0.35)          # gap: a burst, then a keystroke
        time.sleep(0.6)
        with lock:
            chunk = "".join(seen)
        import re as _re
        got = _re.findall(r"MSG=(.*)", chunk)
        good = bool(got) and got[-1].strip() == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {label:28s} -> "
              f"{got[-1].strip() if got else '(nothing)'}")
        if not good:
            print(f"        wanted {want}\n        raw: {chunk[:300]!r}")
    stop.append(1)
    try:
        os.close(fd)
    except OSError:
        pass
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
