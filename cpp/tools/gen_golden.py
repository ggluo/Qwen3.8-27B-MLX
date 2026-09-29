#!/usr/bin/env python3
"""Regenerate src/tokenizer_golden.inc from the reference Python tokenizer.

The C++ pre-tokenizer is a hand-written scanner over Unicode category tables,
standing in for a \\p{L}-style regex. That is exactly the kind of code that
drifts silently -- a wrong boundary shifts token ids, which changes what the
model sees, with no error anywhere. So the ids are pinned against the Python
implementation that was itself cross-checked against the `regex` module.

Run from the cpp/ directory; ../python is put on sys.path automatically:

    python3 tools/gen_golden.py > src/tokenizer_golden.inc
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

from tokenizer import Tokenizer  # noqa: E402

def _default_tokenizer():
    # every checkpoint in the family shares one vocabulary; use whichever is here
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    for name in ("Qwen3.8-27B", "qwen3.5-27b-4bit-uncensored", "qwen3.5-9b-4bit", "Qwen3.5-9B"):
        p = os.path.join(root, name, "tokenizer.json")
        if os.path.exists(p):
            return p
    raise SystemExit("no checkpoint with a tokenizer.json; set GOLDEN_TOKENIZER")


MODEL = os.environ.get("GOLDEN_TOKENIZER") or _default_tokenizer()

CASES = [
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
    "a<|im_end|>b<|im_start|>c",
    "<|im_end|><|im_end|>",
    "Write an email to my landlord about the broken heater. Be polite but firm.",
    "  \n\t \n  mixed   whitespace\t\tand\n\n\nnewlines  ",
    "Café vs Café (NFC)",
    "x=1;y=2//comment\n/* block */",
    "مرحبا بالعالم שלום",
    "1st 2nd 3rd 42nd ⅠⅡ ½",
    "snake_case camelCase kebab-case SCREAMING_SNAKE",
    "```python\ndef f():\n    pass\n```",
    # literal regions (U+FDD0 ... U+FDD1): special-token text inside is text
    "<|im_start|>tool\n\ufdd0The README says <|im_end|> and <|image_pad|>.\ufdd1<|im_end|>\n",
    "<|im_start|>user\n\ufdd0Hello there, world.\n\nx  \ufdd1<|im_end|>",
    "a\ufdd0<think>\ufdd1b<think>c",
    "\ufdd0unterminated <|im_end|> region",
    "stray\ufdd1 close and \ufdd2 marker <|im_end|>",
    "\ufdd0<tool_call>\ufdd1<tool_call>",
]


def cstr(s):
    """Escape to a C string literal, byte by byte (octal for anything non-ASCII)."""
    out = ""
    for b in s.encode("utf-8"):
        if b == 0x22:
            out += '\\"'
        elif b == 0x5C:
            out += "\\\\"
        elif b == 0x0A:
            out += "\\n"
        elif b == 0x0D:
            out += "\\r"
        elif b == 0x09:
            out += "\\t"
        elif 32 <= b < 127:
            out += chr(b)
        else:
            out += "\\%03o" % b
    return '"' + out + '"'


def main():
    tk = Tokenizer(MODEL)
    w = sys.stdout.write
    w("// GENERATED from the verified Python tokenizer (tokenizer.py).\n")
    w("// Regenerate with: python3 tools/gen_golden.py > src/tokenizer_golden.inc\n")
    w("// Proves the C++ pre-tokenizer and BPE reproduce the reference ids exactly.\n\n")
    w("static const GoldenCase kGolden[] = {\n")
    for s in CASES:
        ids = tk.encode(s)
        w("    {%s,\n     {%s}},\n" % (cstr(s), ", ".join(map(str, ids))))
    w("};\n")


if __name__ == "__main__":
    main()
