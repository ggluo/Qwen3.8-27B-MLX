# Qwen3.5-27B in C++

A C++ port of the from-scratch Qwen3.5-27B assistant in
[`../python`](../python). Same model, same architecture work, same terminal
assistant — MLX is used through its **C++ API** directly, so there is no Python at
runtime.

```
make            # builds build/ai and build/tests
make test       # runs the self-tests
make config     # show which MLX the build resolved to
make bundle     # detach build/ from the MLX install
./build/ai      # start chatting
```

## Why C++ and not C

The original plan was C against [`mlx-c`](https://github.com/ml-explore/mlx-c).
MLX's own library is C++, and `mlx-c` is a generated thin wrapper over it, so
going through C would have meant either vendoring that wrapper or hand-rolling an
equivalent — and then reintroducing, in C, the reference counting that
`mlx::core::array` already does for free. Using the C++ API directly removes a
whole layer: `array` is a refcounted handle, so RAII handles every temporary in a
64-layer forward pass with no arena, no scope stack, and no manual frees.

Nothing else here needs C++ features to speak of. It is mostly C with
`std::string`, `std::vector` and destructors.

## Building

MLX is used as an ordinary dynamic library — its headers plus `libmlx.dylib`.
Nothing else is needed at build or run time.

```
make            # autodetects MLX, builds build/ai and build/tests
make config     # print which MLX was resolved, and its version
make test
```

Requirements: macOS on Apple silicon, Xcode command line tools, MLX ≥ 0.29
(checked against 0.29.3 and 0.32.3). CoreFoundation supplies Unicode NFC; Metal
runs the fused delta kernel.

### Pointing the build at MLX

Autodetection tries the `mlx` Python wheel first, then `/opt/mlx`, `/usr/local`
and `/opt/homebrew`. Three ways to override, in order of precedence:

```
make MLX_DIR=/opt/mlx
        # a prefix laid out as <prefix>/include/mlx/mlx.h and
        # <prefix>/lib/libmlx.dylib -- what `cmake --install` and the wheel
        # both produce, so this is the usual answer

make MLX_INCLUDE=~/src/mlx MLX_LIB=~/src/mlx-build/mlx
        # when headers and library are not under a common prefix, e.g. an
        # UNINSTALLED cmake build tree, where the headers are the source
        # checkout itself. Either may be given alone; the other falls back
        # to MLX_DIR.

make PYTHON=python3.12
        # several interpreters, and only one has mlx
```

The binary records `MLX_LIB` as its runtime search path, so it works from any
directory. Every failure names the knob that fixes it, and lists where it looked:

```
$ make config
MLX version   0.29.3
MLX_INCLUDE   /Users/…/site-packages/mlx/include
MLX_LIB       /Users/…/site-packages/mlx/lib
RPATH         /Users/…/site-packages/mlx/lib
mlx.metallib  /Users/…/site-packages/mlx/lib/mlx.metallib
CXX           c++ (Apple clang version 21.0.0)
```

`make config` also flags a missing `mlx.metallib`, which is easy to overlook: MLX
loads it at runtime from *next to* `libmlx.dylib`, and without it every GPU kernel
fails even though the build succeeded.

### Detaching from the MLX install

A default build hardcodes an rpath into whichever MLX it found, so upgrading or
removing that install breaks the binary. `make bundle` copies `libmlx.dylib` and
`mlx.metallib` into `build/lib` and links against `@loader_path/lib`, making
`build/` movable and shippable as a unit:

```
make bundle
otool -L build/ai | grep mlx      # -> @rpath/libmlx.dylib
```

Verified by copying `build/ai` and `build/lib` elsewhere and running it there;
removing `build/lib` then fails with the bundled path as the *only* one attempted,
which is what proves the original install is no longer in the picture. Note
`mlx.metallib` is 100 MB and has to travel with the dylib.

Static linking is not offered. macOS has no static libSystem, so `Metal`,
`Foundation`, `Accelerate` and `libc++` stay dynamic regardless; `mlx.metallib`
remains a separate runtime file either way; and a static MLX has to be compiled
from source, which needs the `metal` compiler from full Xcode. `make bundle`
delivers what static linking would actually have bought here.

## Usage

```
./build/ai                                   # interactive
./build/ai --think                           # show the model's reasoning
./build/ai --system "Be terse."
./build/ai --temp 0 -k 4                     # greedy, draft 4 tokens per round
./build/ai --no-spec                         # plain decoding, no MTP drafting
./build/ai --prompt "..." -n 200             # one-shot, non-interactive
```

In-chat commands: `/help` `/new` `/think` `/temp` `/system` `/paste` `/stats`
`/exit`.

Paste a multi-line paragraph and press Return — it arrives as one message. `Ctrl-J`
inserts a newline without sending, `Ctrl-U` clears the line, `Ctrl-C` interrupts a
reply without quitting, `Ctrl-D` exits.

Nothing is written to disk. The conversation, the KV cache and the editing history
all live in memory and die with the process.

`./ai` is a launcher you can symlink onto `PATH`; the binary resolves the model
directory relative to itself, so it works from any working directory.

## Layout

| File | What it is |
| --- | --- |
| `src/model.{hpp,cpp}` | The decoder: config, norms, attention, Gated DeltaNet, MLP, MTP head, caches, weight loading |
| `src/delta.{hpp,cpp}` | The fused Metal kernel for the gated-delta recurrence, plus its plain-MLX reference |
| `src/session.{hpp,cpp}` | One conversation: token bookkeeping, cache reuse, the plain and speculative decode loops |
| `src/sample.{hpp,cpp}` | `to_probs` / sampling / the rejection-sampling residual |
| `src/tokenizer.{hpp,cpp}` | Byte-level BPE over the model's own `tokenizer.json` |
| `src/unicode.{hpp,cpp}` | Character classes for the pre-tokenizer; NFC via CoreFoundation |
| `src/unicode_tables.cpp` | Generated `\p{L}` / `\p{N}` / `\p{M}` / `\s` ranges (`make unicode`) |
| `src/json.{hpp,cpp}` | Minimal JSON reader for `config.json` and the 12.8 MB `tokenizer.json` |
| `src/lineread.{hpp,cpp}` | Raw-mode line reader with correct multi-line paste handling |
| `src/ops.{hpp,cpp}` | Small array helpers (`slice_axis`, `silu`, `softplus`, `l2norm`) |
| `src/main.cpp` | Argument parsing, the REPL, reasoning-block dimming |
| `src/tests.cpp` | Self-tests |

## What is verified

`make test` — 46 checks, all passing:

* **Tokenizer ids are identical to the Python implementation** on 22 cases
  covering contractions, CJK, emoji, combining accents, tabs/CRLF, code fences,
  Arabic/Hebrew, Roman numerals and adjacent special tokens. The pre-tokenizer is
  a hand-written scanner standing in for a `\p{L}`-style regex, which is exactly
  the kind of code that drifts silently, so the ids are pinned
  (`tools/gen_golden.py` regenerates them).
* **The delta kernel matches its reference** to ≤ 6e-7 relative error across
  `L = 1 … 384`, including the per-step states that speculative rollback indexes.
* **Rejection sampling reproduces the target distribution** exactly (max error
  2.4e-7 over 200 random distributions) — the property that makes speculative
  decoding lossless.
* **The line reader handles both paste paths**, driven through a real pty: a paste
  with and without a trailing newline, bracketed paste, CRLF, `Ctrl-J`, history.
* **NLL 1.485 nats/token, top-1 62.5%** on held-out prose — the check that catches
  a silently wrong convention, which costs several nats and shows up nowhere else.
* **Segmented prefill equals a single forward pass** (same argmax; every layer's
  cache lands on the same offset). All cache reuse across turns rests on this.
* **Speculative decoding equals plain decoding** at temp 0, token for token.

Beyond the suite, the port was checked against the Python end to end: **120 greedy
tokens on a code prompt are byte-identical** between the two implementations.

## Measured on an M3 Max (48 GB), 4-bit, `qwen3.5-27b-4bit-uncensored`

Head to head against the Python implementation: identical prompt, identical
settings, separate processes, repeated.

| | Python | C++ | |
| --- | --- | --- | --- |
| plain decode | 18.2 / 18.4 / 18.4 tok/s | 19.1 / 19.1 / 19.4 tok/s | +5% |
| speculative, k=3 | 31.1 tok/s | 33.3 tok/s | +7% |
| draft accept | 84% | 84% | identical |
| tok/pass, passes | 3.49, 43 | 3.49, 43 | identical |
| peak memory | 16.3 / 16.9 GB | 16.3 / 16.9 GB | identical |
| model load | 0.7 s | 0.5 s | |
| wall clock, trivial prompt | 1.37 s | 0.98 s | −0.4 s |

Prefill ~176 tok/s (1634 tokens in 9.3 s); peak 17.5 GB on a 1.6K-token prompt.

**The throughput is a wash, and that is the expected result.** Identical accept
rate, tok/pass and pass count mean both implementations do precisely the same
arithmetic, so this measures host overhead and nothing else. Decode reads 15.4 GB
of weights per forward pass at an arithmetic intensity of ≈3.3 against a machine
balance of ≈9.8 — it is memory-bandwidth-bound, all the real work happens inside
MLX's Metal kernels, and both versions dispatch the *same* kernels. The host
language only builds and submits the graph: a couple of milliseconds against a
~52 ms token. The ~5% is the Python interpreter's per-token overhead
disappearing; there was never room for more than that.

What the port buys is therefore not speed: no interpreter or wheel at runtime, a
single binary, `make bundle` for a movable tree, and every tensor's shape declared
at load.

Acceptance is strongly prompt-dependent in both — 84% on this code prompt, 55% on
open-ended prose (24.7 tok/s, 1.30×) — and the speedup follows it.

## Deliberate differences from the Python

**Interrupts are cooperative, and simpler for it.** Python got interruption free
via `KeyboardInterrupt`, but an exception could land mid-speculative-round, with
the KV caches already advanced and `pending` stashed on every `DeltaCache` while
`commit()`/`trim()` never ran — the two cache families then disagreed on the
offset, and `Session.interrupt()` existed to unwind that. Here `SIGINT` sets a
flag that the decode loops poll **only at a round boundary**, where the token list
and every cache already agree. There is nothing to roll back, so there is no
rollback code.

**Gammas are folded at load time.** `(1 + w)` in fp32 is computed once during
loading rather than lazily memoized per norm object.

**`--prompt` runs one shot.** Added for scripting, and for diffing against the
Python implementation.

## Not ported

* **The chunked parallel delta scan.** The fused Metal kernel is correct at any
  `L`; above `L ≈ 384` a chunked scan is *faster*, because it turns time into
  matmuls while the kernel is serial in `t`. Prefill is windowed to 384, so that
  crossover is never reached, and the ~200 lines of triangular-inverse code would
  be dead. `delta::run_reference` is kept as the oracle and the no-Metal fallback.
* **`../python/mlp_kernel.py`.** A documented negative result — three
  attempts at a fused quantized-SwiGLU kernel, all slower than stock MLX. It was
  never wired into the model, so there is nothing to port.
* **`../python/quantize.py`.** Quantization is a one-off offline step; keep using
  the Python tool to produce a checkpoint, which this binary then loads.
* **The vision tower.** Text only, as in the Python. The 499 `model.visual.*`
  tensors are skipped at load.
