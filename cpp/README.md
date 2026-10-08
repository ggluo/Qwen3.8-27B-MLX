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
(checked against 0.29.3 and 0.32.3). CoreFoundation supplies Unicode NFC, ImageIO
and CoreGraphics decode pictures, and Metal runs the fused delta kernel.

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
./build/ai --image photo.jpg --prompt "What is in this picture?"
./build/ai --serve                           # OpenAI-compatible API on 127.0.0.1:8080
```

In-chat commands: `/help` `/new` `/think` `/temp` `/system` `/image` `/paste`
`/stats` `/exit`.

Paste a multi-line paragraph and press Return — it arrives as one message. `Ctrl-J`
inserts a newline without sending, `Ctrl-U` clears the line, `Ctrl-C` interrupts a
reply without quitting, `Ctrl-D` exits.

Nothing is written to disk. The conversation, the KV cache and the editing history
all live in memory and die with the process.

`./ai` is a launcher you can symlink onto `PATH`; the binary resolves the model
directory relative to itself, so it works from any working directory.

## Pictures

Every Qwen3.5 checkpoint carries a vision tower, and the assistant uses it:

```
> /image ~/Desktop/receipt.jpg
attached receipt.jpg: 4032x3024 -> 1152x864, 972 tokens
> What's the total, and which shop is it from?
```

`/image` takes one or more paths — drag a file into the terminal to type its path;
the backslash-escaped form Terminal writes is understood — and attaches them to
your next message. `/image` alone lists what is attached, `/image clear` drops it.
`--image` does the same for the first message, and works with `--prompt`.
Follow-up questions about a picture need no re-attaching: it stays in the context,
and later pictures accumulate alongside it.

JPEG, PNG, HEIC, WebP, TIFF and anything else ImageIO reads. Phone photos are
turned upright from their EXIF orientation, as the reference loader does.

**Each 32×32 pixels is one token**, so size is what costs. A picture is scaled so
both sides are multiples of 32 and its area is at least 64 tokens and at most
`--max-pixels` — by default 1,048,576, i.e. 1024 tokens. The checkpoint's own
limit is 16.7 M pixels, 16,384 tokens for a single photo, which is minutes of
prefill on this hardware; 1024 keeps a screenshot's small text legible.

What happens to a picture, in `image.cpp` then `vision.cpp`:

    bytes --ImageIO--> RGB, upright --smart_resize, bicubic--> multiples of 32
      --(x/255 - 0.5)/0.5, 2 identical frames--> 16x16 patches, 2x2-block order
      --27 ViT blocks, 2-D rotary, learned 48x48 position table-->
      --merger: 2x2 patches -> 1 token--> embeddings for the <|image_pad|> tokens

The text model then reads the image tokens with real **M-RoPE**: each carries a
(time, row, column) position instead of one index, and the rotary frequencies are
split between the three axes, interleaved `THWTHW…`. An image of `r×c` tokens
takes `r·c` slots in the context but only `max(r, c)` positions, so after the
first picture the rope position runs ahead of the KV-cache offset. `Session`
carries that difference — the *rope delta* — into every later prefill, every
decode step, the MTP drafter and the speculative verify pass.

Two deliberate differences from the reference loader: **transparency is composited
over white** (the reference drops alpha and keeps whatever colour a transparent
pixel stored — usually black, which makes black text on a transparent background
disappear), and the pixel cap above. Video is not supported.

## Serve mode

`--serve` puts the same assistant behind an HTTP endpoint speaking the OpenAI
chat-completions API — the field names, the SSE framing, the error shape — so
anything that already talks to vLLM, llama.cpp or Ollama can talk to this binary
without an adapter.

```
./build/ai --serve --port 8080
curl -s http://127.0.0.1:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"stream":true}'
```

| Route | |
| --- | --- |
| `POST /v1/chat/completions` | The API. `stream`, `temperature`, `top_p`, `top_k`, `max_tokens`/`max_completion_tokens`, `stop` and `stream_options.include_usage` are honoured. `think: true` (or the Qwen templates' `chat_template_kwargs.enable_thinking`) reports the reasoning block as `reasoning_content`, the way DeepSeek-style clients expect |
| `GET /v1/models` | The one model, named after its directory; `--model-name` overrides |
| `GET /health` | `{"status":"ok"}`, answered without waking the model, so it stays fast while a reply is being written |

**Tool calls work**, which is what an agentic client needs. The schemas go into
the prompt exactly as the checkpoint's template writes them, `<tool_call>`
blocks come back out as OpenAI `tool_calls` — streamed and whole, with arguments
that follow the declared types — a call's `finish_reason` is `tool_calls`, and
`role: "tool"` results are rendered back as `<tool_response>` blocks so the loop
can continue. So an agent like opencode can drive it:

```
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "List the files in /tmp with a tool."}],
  "tools": [{"type": "function", "function": {"name": "bash", "description": "Run a command",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                            "required": ["command"]}}}]}'
```

**Pictures work too**, as the API's `image_url` content parts:

```
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -i photo.jpg)"'"}},
    {"type": "text", "text": "What is in this picture?"}]}]}'
```

Parts are rendered in order, so text and pictures can interleave. The bare-string
form `"image_url": "data:..."` and `{"type": "image", "image": "data:..."}` are
accepted too, since clients send both. Pictures must arrive **inline, as base64
`data:` URLs**: an `http(s)` URL is refused (this server never touches the
network), and so is `file://`, which would let anyone who can reach the port read
pictures off this disk. A picture outside a user message, bytes that are not a
picture, or a model with no vision tower are all a `400`, answered before
anything is streamed — and before the session is touched, so the cached
conversation survives a bad request. Cache reuse compares the pictures' bytes as
well as the text: the same words about a different picture are a different
conversation. `--max-pixels` sets the size cap, as in the REPL.

Also `--host` (default `127.0.0.1`: this is a local model, and the default keeps
it local), `--api-key` (requires `Authorization: Bearer <key>`; `/health` stays
open), plus the same `--system`, `--think`, `-k`, `--no-spec` and `-n` knobs the
REPL takes, as defaults for requests that do not ask.

### Seeing what a client actually sends

`--dump DIR` writes every request and reply there as they happen, numbered in
arrival order — the same order as the log lines:

| File | |
| --- | --- |
| `NNNN-request.json` | the request body, verbatim |
| `NNNN-prompt.txt` | the exact text handed to the model, when the request was rendered from scratch: the whole conversation, system prompt and tool schemas included |
| `NNNN-prompt-tail.txt` | the same, when the request only extended a conversation the session was already holding. That prompt really is only the new turns — everything before it is already in the KV cache — and the file is named so that half a conversation cannot be mistaken for a whole prompt |
| `NNNN-response.json` | the reply — content, reasoning, tool calls — with its timings, and `prompt`: what this request's prompt was made of |

`prompt` in the response is what the breakdown has to be read as:

```json
"prompt": {
  "head_in_context": true,        // system prompt + tools already in the cache
  "system_bytes": 56048,          // bytes, not characters: these prompts are
  "tools_bytes": 13150,           //   full of emoji and CJK
  "reused_tokens": 17070,         // tokens the session already held
  "messages": [                   // only the messages this prompt carried
    {"role": "assistant", "content_bytes": 2938},
    {"role": "user", "content_bytes": 29}
  ]
}
```

`usage` itself follows the API: `prompt_tokens` is the whole prompt the request
stands for, and `prompt_tokens_details.cached_tokens` is the part the session
already held (`reused_tokens` here). So `prompt_tokens + completion_tokens` is the
context the next turn starts from, which is what an agent client measures its
context window by, and `prompt_tokens - cached_tokens` is what this request
actually prefilled.

That breakdown is how you answer "why is this prompt 17k tokens long": a prompt
that size is almost always the client's own doing — an agent's instructions, the
environment block, its tool schemas, whatever it pasted in from the repository.
Requests that were refused are dumped too, which is usually the quickest way to
see why a client is unhappy.

This is the one thing in this binary that writes a conversation to disk, and it
only happens when asked. The directory is created mode 0700.

Four things worth knowing:

* **The KV cache is reused across requests.** A request whose `messages` extend
  the conversation the server is already holding appends only the new turns;
  anything else resets and re-prefills. A client that echoes its history back
  pays for that history once, not on every turn. The log line says which
  happened: `continued` or `prefilled`.

  The held conversation includes the **last reply**, recorded as the assistant
  message the client gets back — content and tool calls, as sent. The context
  already holds that reply as the model wrote it, so a request that echoes it
  unchanged feeds only what follows; one that echoes it altered (or leaves it
  out) is a different conversation and starts over. A reply that ended early —
  the client hung up, Ctrl-C, a stop string — leaves the context holding tokens
  the next request cannot account for, so that one starts over too.

  It is all or nothing because of the 48 linear-attention layers: their state
  is overwritten by every token and cannot be rewound, so there is no reusing
  the shared part of an edited history the way a pure-attention server can by
  truncating its KV cache.
* **One generation at a time.** There is one conversation and one set of caches,
  so a second request queues behind the first — while `/health` and `/v1/models`
  are still answered immediately.
* **One thread owns the model.** MLX ties every array to the stream of the
  thread that created it, so the weights are loaded by, and only ever touched
  by, a single thread; connection threads hand it one request at a time. Loading
  on one thread and generating on another fails with `There is no Stream(gpu, N)
  in current thread` — which is exactly what a thread-per-request server does by
  accident.
* **What a client sends is text, never control tokens.** Message content, tool
  results, reasoning, arguments, the system prompt and the tool schemas all go
  into the prompt as literal regions (see `tokenizer.hpp`), so only the
  template's own structure becomes `<|im_start|>`, `<|im_end|>`, `<tool_call>`
  and the rest. Without that, an agent that reads a file quoting `<|im_end|>` —
  this repository's READMEs do — injects a turn boundary into its own
  conversation, and one quoting `<|image_pad|>` claimed a picture that was never
  sent. Only a real `image_url` part places image tokens.
* **The prompt is the checkpoint's chat template, written out in C++.** The real
  one is a Jinja file, and interpreting it would mean shipping a Jinja engine, so
  serve mode reads `<model>/chat_template.jinja` at load and checks it against
  the format these renderers write — the tools preamble, the `<function=...>` /
  `<parameter=...>` call body, the `<tool_response>` block. A checkpoint whose
  template says something else is reported at startup rather than prompted in a
  format it was never trained on. That file is the specification for
  `session.cpp`, and for checkpoints in this family it is byte-identical to the
  copy in `tokenizer_config.json`.
* **No CORS and no TLS.** This is the same in-memory assistant behind a socket,
  not a deployment: it writes nothing to disk, and nothing is exposed beyond
  loopback unless `--host` says so.

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
| `src/serve.{hpp,cpp}` | Serve mode: the HTTP/1.1 server, the OpenAI routes, and the one thread that owns the model |
| `src/image.{hpp,cpp}` | Pictures to patches: ImageIO decoding, EXIF orientation, the resize rule, bicubic, normalisation, patching; base64 |
| `src/vision.{hpp,cpp}` | The vision tower: patch embedding, position table, 27 ViT blocks, the 2x2 merger |
| `tools/gen_image_golden.py` | Regenerates `src/image_golden.inc`, which pins the preprocessing to the Python port's |
| `src/main.cpp` | Argument parsing, the REPL, reasoning-block dimming |
| `src/tests.cpp` | Self-tests |

## What is verified

`make test` — 146 checks, all passing (123 of them need no weights at all):

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
* **The chat template and the HTTP layer are pinned too.** The renderer is
  checked against literal template strings, including the property serve mode's
  cache reuse rests on — that a continuation is the *suffix* of a full render,
  so appending it lands where a cold render would have put it. The request
  parser is checked on keep-alive, LF-only line endings and half-arrived input;
  the stop filter on stop strings split across pieces, and on characters split
  across pieces, which is what a byte-level BPE token can do to a JSON string.
* **Tool calls are pinned at every step**: the preamble and the format the model
  was taught, a call echoed back into the history, a tool result as a user block,
  consecutive results sharing one, and — on the way out — a call read back out of
  the model's text, including when the tags themselves are split across pieces,
  when the arguments have to be typed as the schema declares, and when a
  `<tool_call>` turns out to be prose after all.
* **Picture preprocessing is bit-identical to the Python port's** on every test
  picture — a PNG, an EXIF-rotated JPEG, a transparent PNG, a greyscale one, at
  sizes that are kept, shrunk and grown — which is itself held against
  transformers' Qwen2-VL image processor (identical pixels, values within one
  fp32 ULP). `tools/gen_image_golden.py` records the Python's patch tensors.
* **The model reads pictures**: the number in one, the direction of an arrow
  stored sideways with an EXIF tag, black text on a transparent background; a
  follow-up with no picture, a second picture, then a question about the first —
  which only works if the rope delta carries across turns and adds up across
  pictures; and speculative decoding against plain decoding with a picture in
  context.
* **Serve mode takes pictures over its real HTTP path**, driven through a
  `socketpair()` — a test cannot bind a port, but a socket pair needs no network:
  a tool result that quotes `<|image_pad|>` and `<|im_end|>`, read as text
  with exactly one end-of-turn token per block; a picture cold; an echoed
  history that continues from the cache, feeding
  *exactly* the new turn's tokens and not the echoed reply again, two turns in
  a row; a history that leaves out the reply, and a swapped picture, both of
  which must start over; every refusal as a `400` with the cached conversation
  still intact after it; and a streamed reply.

Beyond the suite, the port was checked against the Python end to end: **greedy
output is byte-identical** between the two, for text and for pictures alike —
plain and speculative decoding, 150 tokens each, on four test pictures and a code
prompt, and on both checkpoints the suite runs. For pictures the vision features,
and the hidden state after the prefill, are bit-identical too.

Where plain and speculative decoding differ, it is the one caveat speculative
decoding always has in bf16: the verify pass scores `k+1` positions at once and
plain decode one, so on an **exact** tie of the top two logits the argmax can go
either way. The tests allow that and nothing else — a divergence passes only if
the plain model's top two are within bf16 resolution there. On the 27B with a
picture in context it happens 15 tokens into a list: `1.` or `-`, 25.500 against
25.375, one bf16 step apart.

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

Prefill ~228 tok/s (2048 tokens in 9.0 s; it was 176 tok/s); peak 18.7 GB on a
2K-token prompt. Prefill runs in 1024-token windows, and three things set its pace:

| | 2048-token prefill |
| --- | --- |
| before | 11.7 s, 175 tok/s |
| delta kernel: one simdgroup per four state rows, not one thread per row | 10.1 s, 203 tok/s |
| windows of 1024, and quantized weights dequantized for dense matmul at >= 512 rows | 9.0 s, 228 tok/s |

At that point the matmuls are ~93% of the pass, so what is left is MLX's GEMM.
On a 16K-token prompt the window matters more: 78 s against 96 s at the old
384, because each window re-reads every weight. Attention past ~10K context feeds
a window's queries in slices, so the larger window does not raise the peak there.

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

### Pictures

A 12 MP photo (4032×3024, downscaled to 1152×864, 972 tokens), asked to read the
text in it:

| | 9B, 4-bit | 27B, 4-bit |
| --- | --- | --- |
| prefill, picture included | 2.4 s | 6.9 s |
| … of which the vision tower | 0.7 s | 0.8 s |
| peak memory | 7.8 GB | 18.6 GB |

The tower's share is small; what a picture costs is its tokens going through the
text model. The first picture of a session also reads the tower's weights from
disk — they are left out of the load so a text-only session never pays for them —
and a 640×480 picture with warm weights takes 0.15 s in the tower. Peak memory is
~1.7–2.5 GB above text-only.

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

**Rope for text positions is `mx::fast::rope` in both ports.** It used to be a
hand-written rotation in the Python, mathematically the same as the fused kernel
but not bit for bit. Text-only replies never landed on a close enough margin to
show it; describing a picture did, a handful of words in. Now both ports call the
same kernel for text positions, and write out the same explicit rotation for the
(t, h, w) positions of a chunk holding image tokens, which has no fused kernel.

## Not ported

* **The chunked parallel delta scan.** The fused Metal kernel is correct at any
  `L`, and serial in `t`; a chunked scan turns time into matmuls instead. With
  the kernel spread across simdgroups it costs ~2% of a 1024-token prefill
  window, which leaves the ~200 lines of triangular-inverse code nothing to win.
  `delta::run_reference` is kept as the oracle and the no-Metal fallback.
* **`../python/mlp_kernel.py`.** A documented negative result — three
  attempts at a fused quantized-SwiGLU kernel, all slower than stock MLX. It was
  never wired into the model, so there is nothing to port.
* **`../python/quantize.py`.** Quantization is a one-off offline step; keep using
  the Python tool to produce a checkpoint, which this binary then loads.
* **Video.** The tower's temporal patching would take it, but neither port feeds
  frames or the timestamps Qwen3-VL puts between them.
