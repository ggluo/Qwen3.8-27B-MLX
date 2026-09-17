"""
ai -- a local terminal assistant. Interactive chat, nothing written to disk.

    python3 chat.py                       # start chatting
    python3 chat.py --think               # show the model's reasoning
    python3 chat.py --system "Be terse."

Everything lives in memory: the conversation, the KV cache, the readline
editing history. Quit and it is gone. No files are created or read except the
model weights.

Two things make it feel fast:

  * The KV cache is REUSED across turns. Only the new user turn is prefilled,
    not the whole conversation -- otherwise turn N costs O(total tokens) and a
    long chat crawls. Correctness of this relies on segmented prefill being
    exact (test_model.py: "chunked prefill == full forward").
  * Speculative decoding via the MTP head, ~1.4x on prose and ~1.9x on code.

In-chat commands: /help /new /think /temp /system /exit
"""

import argparse
import os
import sys
import time

import mlx.core as mx

import lineread
import qwen35
from speculative import residual, sample_probs, to_probs
from tokenizer import Tokenizer

EOS = (248046, 248044)          # <|im_end|>, <|endoftext|>


class Session:
    """Owns the conversation tokens and the caches that mirror them.

    Invariant: `self.tokens` is the full context, and `cache[0].offset` is how
    much of it the model has actually consumed. Everything past that offset is
    fed on the next `_sync()`.
    """

    def __init__(self, model, tk, system=None, spec=True, k=3, prefill_step=384):
        self.model, self.tk = model, tk
        self.system = system
        self.spec = spec and hasattr(model, "mtp")
        self.k = k
        self.prefill_step = prefill_step
        self.reset()

    def reset(self):
        self.tokens = []
        self.cache = self.model.make_cache()
        self.turns = 0
        mx.clear_cache()

    # -- feed everything the model has not seen yet; return hidden at the end --
    def _sync(self):
        lo = self.cache[0].offset
        need = self.tokens[lo:]
        h = None
        for s in range(0, len(need), self.prefill_step):
            h, self.cache = self.model.hidden_states(
                mx.array([need[s:s + self.prefill_step]]), self.cache)
            mx.eval(h)
        return h[:, -1:]

    def _prompt_tokens(self, text, think):
        p = ""
        if self.turns == 0 and self.system:
            p += f"<|im_start|>system\n{self.system}<|im_end|>\n"
        p += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        # the template always opens a reasoning block; closing it immediately
        # is how you disable thinking
        p += "<think>\n" if think else "<think>\n\n</think>\n\n"
        return self.tk.encode(p)

    def turn(self, text, think=False, temp=0.7, top_p=0.95, top_k=20,
             max_tokens=1024):
        """Generator yielding decoded text as it is produced."""
        self.tokens += self._prompt_tokens(text, think)
        start_len = len(self.tokens)
        h = self._sync()

        p0 = to_probs(self.model.lm_head(h)[0, -1], temp, top_p, top_k)
        first = sample_probs(p0)
        stats = {"n": 0, "rounds": 0, "drafted": 0, "accepted": 0,
                 "t0": time.time()}

        if first in EOS:
            self._close_turn(start_len, stats)
            return
        self.tokens.append(first)
        stats["n"] = 1
        emitted = [first]
        yield self.tk.decode(emitted)

        gen = (self._spec_loop if self.spec else self._plain_loop)
        for piece in gen(h, first, emitted, stats, temp, top_p, top_k, max_tokens):
            yield piece
        self._close_turn(start_len, stats)

    def interrupt(self):
        """Recover from a Ctrl-C mid-generation.

        The danger is a half-finished speculative round: hidden_states() has
        already advanced the KV caches and stashed `pending` on every DeltaCache,
        but commit()/trim() never ran. The two cache families then disagree on
        the offset and the next turn feeds the wrong tokens.

        Roll everything back to the last committed position -- commit(0) restores
        a DeltaCache's pre-block state exactly -- then truncate the token list to
        match and close the turn so the transcript stays well-formed.
        """
        base = None
        for c in self.cache:
            if isinstance(c, qwen35.DeltaCache):
                if c.pending is not None:
                    base = c.base_offset
                    c.commit(0)
                c.record = False
                if base is None:
                    base = c.offset
        if base is None:
            base = self.cache[0].offset
        for c in self.cache:
            if not isinstance(c, qwen35.DeltaCache):
                c.trim(base)
        del self.tokens[base:]          # drop tokens the model never consumed
        self._close_turn(base, {"n": 0, "rounds": 0, "drafted": 0,
                                "accepted": 0, "t0": time.time()})

    def _close_turn(self, start_len, stats):
        # close the assistant turn so the next one continues correctly
        self.tokens += self.tk.encode("<|im_end|>\n")
        self.turns += 1
        stats["dt"] = time.time() - stats["t0"]
        self.last = stats

    # ---------------- plain decoding ----------------

    def _plain_loop(self, h, tok, emitted, stats, temp, top_p, top_k, max_tokens):
        printed = self.tk.decode(emitted)
        while stats["n"] < max_tokens:
            h, self.cache = self.model.hidden_states(mx.array([[tok]]), self.cache)
            p = to_probs(self.model.lm_head(h)[0, -1], temp, top_p, top_k)
            tok = sample_probs(p)
            if tok in EOS:
                return
            self.tokens.append(tok)
            emitted.append(tok)
            stats["n"] += 1
            full = self.tk.decode(emitted)
            if len(full) > len(printed) and "�" not in full[len(printed):]:
                yield full[len(printed):]
                printed = full

    # ---------------- speculative decoding ----------------

    def _spec_loop(self, h, first, emitted, stats, temp, top_p, top_k, max_tokens):
        model, k = self.model, self.k
        # Fresh MTP cache each turn: its entries are indexed by cache position,
        # and injecting a user turn would leave a positional gap. The drafter
        # only needs local context, and a wrong draft costs nothing but a
        # rejection, so this is safe -- just slightly fewer accepts at turn start.
        mtp_cache = model.mtp.make_cache()
        printed = self.tk.decode(emitted)
        P = self.cache[0].offset
        A_tok, A_hid, A_pos = mx.array([[first]]), h, P
        next_tok = mx.array([[first]])

        while stats["n"] < max_tokens:
            P_old = P
            hd = model.mtp(model.embed_tokens(A_tok), A_hid, mtp_cache, A_pos)
            grounded = mtp_cache[0].offset
            h_prev = hd[:, -1:]
            qs, dl = [], []
            for i in range(k):
                if i:
                    h_prev = model.mtp(model.embed_tokens(mx.array([[dl[-1]]])),
                                       h_prev, mtp_cache, P_old + i)
                q = to_probs(model.lm_head(h_prev)[0, -1], temp, top_p, top_k)
                qs.append(q)
                dl.append(sample_probs(q))

            X = mx.concatenate([next_tok, mx.array([dl])], axis=1)
            for c in self.cache:
                if isinstance(c, qwen35.DeltaCache):
                    c.record = True
            h_v, self.cache = model.hidden_states(X, self.cache)
            tl = model.lm_head(h_v)[0]
            ps = [to_probs(tl[i], temp, top_p, top_k) for i in range(k + 1)]

            di = mx.array(dl)[:, None]
            pd = mx.take_along_axis(mx.stack(ps[:k]), di, axis=-1)[:, 0]
            qd = mx.take_along_axis(mx.stack(qs), di, axis=-1)[:, 0]
            u = mx.random.uniform(shape=(k,))
            acc = mx.where(qd > 0, u * qd <= pd, pd > 0)
            mx.eval(acc, h_v)
            accl = acc.tolist()
            j = 0
            while j < k and accl[j]:
                j += 1
            corrected = sample_probs(ps[k] if j == k else residual(ps[j], qs[j]))

            for c in self.cache:
                if isinstance(c, qwen35.DeltaCache):
                    c.commit(j + 1)
                else:
                    c.trim(P_old + j + 1)
            mtp_cache[0].trim(grounded)

            stats["drafted"] += k
            stats["accepted"] += j
            stats["rounds"] += 1

            stop = False
            for t in dl[:j] + [corrected]:
                if t in EOS:
                    stop = True
                    break
                self.tokens.append(t)
                emitted.append(t)
                stats["n"] += 1
                if stats["n"] >= max_tokens:
                    stop = True
                    break
            full = self.tk.decode(emitted)
            if len(full) > len(printed) and "�" not in full[len(printed):]:
                yield full[len(printed):]
                printed = full
            if stop:
                return

            P = P_old + j + 1
            next_tok = mx.array([[corrected]])
            A_tok = mx.array([dl[:j] + [corrected]])
            A_hid = h_v[:, :j + 1]
            A_pos = P_old + 1


DIM, RESET = "\033[2m", "\033[0m"

_MARK = "</think>"


def style_stream(pieces, thinking):
    """Dim the reasoning block and swallow the `</think>` marker.

    The marker can arrive split across pieces, so hold back len(_MARK)-1
    characters until it is either found or the reasoning ends.
    """
    if not thinking:
        yield from pieces
        return
    yield DIM
    buf = ""
    for p in pieces:
        buf += p
        i = buf.find(_MARK)
        if i >= 0:
            yield buf[:i] + RESET + buf[i + len(_MARK):].lstrip("\n")
            yield from pieces          # rest of the reply, undimmed
            return
        hold = len(_MARK) - 1
        if len(buf) > hold:
            yield buf[:-hold]
            buf = buf[-hold:]
    if buf:
        yield buf
    yield RESET


_READER = lineread.LineReader()


def read_message(prompt):
    """One message from the user, with multi-line paste handled correctly.

    See lineread.py: `input()` plus libedit cannot do this -- the tty submits at
    the first newline, so pasting a paragraph fires off its first line before you
    press Return and orphans the rest. The raw-mode reader takes the tty out of
    line-buffered mode so a paste arrives whole.
    """
    return _READER.read(prompt)


HELP = """
  /new              start a fresh conversation (clears context)
  /think [on|off]   show the model's reasoning (slower)
  /temp <float>     sampling temperature (0 = deterministic)
  /system <text>    set a system prompt (applies to the next /new)
  /stats            timing for the last reply
  /help             this
  /exit             quit (Ctrl-D also works)

  Paste a multi-line paragraph and press Return -- it arrives as one message.
  Ctrl-J inserts a newline without sending. Ctrl-U clears the line.
  /paste             type a multi-line message, end with a single "." line
  Ctrl-C interrupts a reply without quitting.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-27b-4bit-uncensored")
    ap.add_argument("--system", default=None)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("-k", "--draft", type=int, default=3)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("-n", "--max-tokens", type=int, default=1024)
    a = ap.parse_args()

    # Resolve the model relative to this script, not the cwd, so a launcher on
    # PATH works from any directory. Checkpoints live at the repository root,
    # one level above this file.
    path = qwen35.find_model(a.model)

    t0 = time.time()
    print(f"loading {os.path.basename(path)} ...", file=sys.stderr, flush=True)
    tk = Tokenizer(f"{path}/tokenizer.json")
    model, _ = qwen35.load(path, verbose=False)
    s = Session(model, tk, a.system, spec=not a.no_spec, k=a.draft)
    think, temp = a.think, a.temp
    print(f"ready in {time.time()-t0:.1f}s"
          f"{'' if s.spec else '  (no MTP head: plain decoding)'}"
          f"   /help for commands, /exit to quit\n", file=sys.stderr)

    while True:
        try:
            line = read_message("\033[1m>\033[0m ")
            while line.endswith("\\"):
                line = line[:-1] + "\n" + read_message("\033[2m…\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = line.strip()
        if not text:
            continue

        # only a single line can be a command, so a pasted paragraph that happens
        # to start with "/" is still treated as text
        send = False
        if text.startswith("/") and "\n" not in text:
            cmd, _, arg = text[1:].partition(" ")
            arg = arg.strip()
            if cmd in ("exit", "quit", "q"):
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "new":
                s.reset()
                print("\033[2mnew conversation\033[0m")
            elif cmd == "think":
                think = arg != "off" if arg else not think
                print(f"\033[2mthinking {'on' if think else 'off'}\033[0m")
            elif cmd == "temp":
                try:
                    temp = float(arg)
                    print(f"\033[2mtemperature {temp}\033[0m")
                except ValueError:
                    print(f"\033[2mtemperature is {temp}\033[0m")
            elif cmd == "system":
                s.system = arg or None
                print(f"\033[2msystem prompt {'set' if arg else 'cleared'};"
                      f" takes effect on /new\033[0m")
            elif cmd == "paste":
                print(f"{DIM}multi-line input; end with a single '.' line{RESET}")
                lines = []
                while True:
                    try:
                        l = input()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if l.strip() == ".":
                        break
                    lines.append(l)
                text = "\n".join(lines).strip()
                if not text:
                    continue
                send = True             # fall through to generation
            elif cmd == "stats":
                st = getattr(s, "last", None)
                if st:
                    msg = (f"{st['n']} tokens in {st['dt']:.1f}s "
                           f"({st['n']/max(st['dt'],1e-9):.1f} tok/s)")
                    if st["rounds"]:
                        msg += (f", {st['rounds']} model passes, "
                                f"{st['accepted']/max(st['drafted'],1)*100:.0f}% "
                                f"draft accept")
                    msg += f", context {len(s.tokens)} tokens"
                    print(f"\033[2m{msg}\033[0m")
                else:
                    print("\033[2mno reply yet\033[0m")
            else:
                print(f"\033[2munknown command /{cmd} -- try /help\033[0m")
            if not send:
                continue

        try:
            stream = s.turn(text, think, temp, a.top_p, a.top_k, a.max_tokens)
            for piece in style_stream(stream, think):
                sys.stdout.write(piece)
                sys.stdout.flush()
            print()
        except KeyboardInterrupt:
            s.interrupt()          # resync caches, keep the context usable
            print("\n\033[2m^C interrupted\033[0m")

    print("\033[2mconversation discarded\033[0m", file=sys.stderr)


if __name__ == "__main__":
    main()
