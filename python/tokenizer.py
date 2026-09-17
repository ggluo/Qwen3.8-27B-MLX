"""
Byte-level BPE tokenizer for Qwen3.5, stdlib only (json + unicodedata).

Reads the model's own tokenizer.json. Reimplements what HF `tokenizers` does:

    NFC normalize -> split on the GPT-4 regex -> map bytes to safe unicode
    -> greedy lowest-rank BPE merges -> ids

The split pattern uses \\p{L} / \\p{N} / \\p{M}, which Python's `re` does not
support, so `_pretokenize` is a hand-written scanner over unicodedata
categories rather than a regex. `python tokenizer.py` self-tests it, and will
diff against the `regex` module if that happens to be installed.
"""

import json
import unicodedata
from functools import lru_cache
from typing import Dict, List, Tuple

# The pattern we are reimplementing, kept here for reference:
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


# --------------------------------------------------------------------------
# unicode character classes
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _cat(ch: str) -> str:
    return unicodedata.category(ch)


def _is_L(ch): return _cat(ch)[0] == "L"
def _is_N(ch): return _cat(ch)[0] == "N"
def _is_M(ch): return _cat(ch)[0] == "M"
def _is_LM(ch): return _cat(ch)[0] in ("L", "M")


def _is_space(ch: str) -> bool:
    """Match Rust regex \\s: ASCII whitespace + unicode Z* + a few controls."""
    return ch.isspace() or _cat(ch)[0] == "Z"


# --------------------------------------------------------------------------
# byte <-> unicode (the GPT-2 trick: keep every byte printable and non-space)
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _byte_encoder() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# --------------------------------------------------------------------------


class Tokenizer:
    def __init__(self, tokenizer_json: str):
        with open(tokenizer_json, "r", encoding="utf-8") as f:
            spec = json.load(f)

        model = spec["model"]
        assert model["type"] == "BPE", model["type"]
        self.vocab: Dict[str, int] = model["vocab"]

        merges = model["merges"]
        self.ranks: Dict[Tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ") if isinstance(m, str) else list(m)
            assert len(parts) == 2, m
            self.ranks[(parts[0], parts[1])] = i

        # special tokens are matched literally, before any splitting
        self.added: Dict[str, int] = {}
        for t in spec.get("added_tokens", []):
            self.added[t["content"]] = t["id"]
            self.vocab.setdefault(t["content"], t["id"])
        # longest-first so <|im_start|> wins over any prefix of it
        self._added_sorted = sorted(self.added, key=len, reverse=True)

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.byte_encoder = _byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._normalize = spec.get("normalizer", {}).get("type") == "NFC"
        self._cache: Dict[str, List[int]] = {}

        self.eos_id = self.added.get("<|im_end|>", self.added.get("<|endoftext|>"))
        self.bos_id = self.added.get("<|endoftext|>")

    # ---------------- pre-tokenization ----------------

    def _pretokenize(self, text: str) -> List[str]:
        """Hand-rolled equivalent of SPLIT_PATTERN with behavior=Isolated."""
        out: List[str] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]

            # 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if ch in ("'", "’") and i + 1 < n:
                low = text[i:i + 3].lower()
                hit = next((c for c in CONTRACTIONS if low.startswith(c) and ch == "'"), None)
                if hit:
                    out.append(text[i:i + len(hit)])
                    i += len(hit)
                    continue

            # 2. [^\r\n\p{L}\p{N}]? [\p{L}\p{M}]+
            j = i
            if ch not in "\r\n" and not _is_L(ch) and not _is_N(ch):
                j = i + 1
            if j < n and _is_LM(text[j]):
                k = j
                while k < n and _is_LM(text[k]):
                    k += 1
                out.append(text[i:k])
                i = k
                continue

            # 3. \p{N}   (single digit at a time)
            if _is_N(ch):
                out.append(ch)
                i += 1
                continue

            # 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            j = i + 1 if (ch == " " and i + 1 < n) else i
            if j < n and not _is_space(text[j]) and not _is_LM(text[j]) and not _is_N(text[j]):
                k = j
                while k < n and not _is_space(text[k]) and not _is_LM(text[k]) and not _is_N(text[k]):
                    k += 1
                while k < n and text[k] in "\r\n":
                    k += 1
                out.append(text[i:k])
                i = k
                continue

            # 5. \s*[\r\n]+   -- run of whitespace that contains a newline
            if _is_space(ch):
                k = i
                while k < n and _is_space(text[k]):
                    k += 1
                # longest prefix of the whitespace run ending in newlines
                last_nl = -1
                for p in range(k - 1, i - 1, -1):
                    if text[p] in "\r\n":
                        last_nl = p
                        break
                if last_nl >= 0:
                    out.append(text[i:last_nl + 1])
                    i = last_nl + 1
                    continue
                # 6. \s+(?!\S)  -- whitespace not followed by a non-space
                if k == n:
                    out.append(text[i:k])
                    i = k
                    continue
                # 7. \s+ but leave the final space for the next token to claim
                #    (regex \s+ is greedy; Isolated behavior then re-splits)
                end = k - 1 if k - 1 > i else k
                out.append(text[i:end])
                i = end
                continue

            out.append(ch)  # unreachable in practice; never drop input
            i += 1
        return [t for t in out if t]

    # ---------------- BPE ----------------

    def _bpe(self, piece: str) -> List[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        symbols = [self.byte_encoder[b] for b in piece.encode("utf-8")]
        while len(symbols) > 1:
            best, best_rank = None, None
            for idx in range(len(symbols) - 1):
                r = self.ranks.get((symbols[idx], symbols[idx + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = idx, r
            if best is None:
                break
            symbols[best:best + 2] = [symbols[best] + symbols[best + 1]]

        ids = []
        for s in symbols:
            if s in self.vocab:
                ids.append(self.vocab[s])
            else:  # no byte_fallback in this tokenizer; split to single bytes
                ids.extend(self.vocab[c] for c in s)
        self._cache[piece] = ids
        return ids

    # ---------------- public API ----------------

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        if self._normalize:
            text = unicodedata.normalize("NFC", text)

        chunks: List[Tuple[str, bool]] = [(text, False)]  # (text, is_special)
        if allow_special:
            for tok in self._added_sorted:
                nxt: List[Tuple[str, bool]] = []
                for s, is_sp in chunks:
                    if is_sp or tok not in s:
                        nxt.append((s, is_sp))
                        continue
                    parts = s.split(tok)
                    for p_i, p in enumerate(parts):
                        if p_i:
                            nxt.append((tok, True))
                        if p:
                            nxt.append((p, False))
                chunks = nxt

        ids: List[int] = []
        for s, is_sp in chunks:
            if is_sp:
                ids.append(self.added[s])
            else:
                for piece in self._pretokenize(s):
                    ids.extend(self._bpe(piece))
        return ids

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        buf = bytearray()
        out = []
        for i in ids:
            tok = self.id_to_token.get(int(i))
            if tok is None:
                continue
            if tok in self.added:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not skip_special:
                    out.append(tok)
                continue
            buf.extend(self.byte_decoder[c] for c in tok)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


# --------------------------------------------------------------------------


def _default_tokenizer() -> str:
    """First checkpoint directory that has a tokenizer.json.

    The sources live in python/ and the checkpoints at the repository root, so a
    fixed relative path would only work from one of the two.
    """
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("Qwen3.8-27B", "qwen3.5-27b-4bit-uncensored", "qwen3.5-27b-4bit"):
        cand = os.path.join(root, name, "tokenizer.json")
        if os.path.exists(cand):
            return cand
    raise SystemExit("no checkpoint with a tokenizer.json found; pass one as argv[1]")


def _self_test(path=None):
    if path is None:
        path = _default_tokenizer()
    tk = Tokenizer(path)
    print(f"vocab {len(tk.vocab)}  merges {len(tk.ranks)}  special {len(tk.added)}")
    print(f"eos_id {tk.eos_id}  bos_id {tk.bos_id}")

    cases = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "  leading and trailing   ",
        "line one\nline two\n\n\nline three",
        "don't can't I'LL we've",
        "numbers 1234567890 and 3.14159",
        "def f(x): return x**2  # comment",
        "你好，世界！これはテストです。",
        "emoji \U0001f600\U0001f680 and accents café naïve",
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        "tabs\tand\r\ncrlf",
        "",
    ]
    ok = True
    for s in cases:
        ids = tk.encode(s)
        back = tk.decode(ids)
        want = unicodedata.normalize("NFC", s)
        good = back == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {len(ids):3d} tok  {s[:42]!r}")
        if not good:
            print(f"        got {back!r}\n       want {want!r}")

    # every id must be in range
    big = tk.encode("".join(cases))
    assert all(0 <= i < 248320 for i in big), "id out of range"

    # specials must map to their exact ids
    for name, want in [("<|im_start|>", 248045), ("<|im_end|>", 248046),
                       ("<|endoftext|>", 248044)]:
        got = tk.encode(name)
        good = got == [want]
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] special {name} -> {got} (want [{want}])")

    # cross-check the pretokenizer against the `regex` module if available
    try:
        import regex
        pat = regex.compile(SPLIT_PATTERN)
        diffs = 0
        for s in cases:
            if not s:
                continue
            mine = tk._pretokenize(unicodedata.normalize("NFC", s))
            theirs = pat.findall(unicodedata.normalize("NFC", s))
            if mine != theirs:
                diffs += 1
                print(f"  [DIFF] {s[:40]!r}\n     mine   {mine}\n     regex  {theirs}")
        print(f"  [{'ok' if not diffs else 'WARN'}] pretokenizer vs regex module: "
              f"{len(cases)-1-diffs}/{len(cases)-1} identical")
    except ImportError:
        print("  [skip] `regex` not installed -- cannot diff pretokenizer against it")

    print("\nROUND-TRIP OK" if ok else "\nFAILURES ABOVE")
    return ok


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(0 if _self_test(p) else 1)
