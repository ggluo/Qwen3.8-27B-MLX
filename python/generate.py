"""
Text generation for the from-scratch Qwen3.5-27B.

    python generate.py --raw "The capital of France is" --temp 0 -n 20
    python generate.py "Explain what a KV cache is." -n 400
    python generate.py "Hi" --no-think -n 200
"""

import argparse
import time

import mlx.core as mx

import qwen35
from tokenizer import Tokenizer

EOS = (248046, 248044)  # <|im_end|>, <|endoftext|>


def chat_prompt(user: str, system: str = None, think: bool = True) -> str:
    p = ""
    if system:
        p += f"<|im_start|>system\n{system}<|im_end|>\n"
    p += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    # the template opens a reasoning block; closing it immediately disables thinking
    p += "<think>\n" if think else "<think>\n\n</think>\n\n"
    return p


def sample(logits: mx.array, temp: float, top_p: float, top_k: int) -> mx.array:
    if temp == 0:
        return mx.argmax(logits, axis=-1)
    logits = logits.astype(mx.float32)
    if top_k and top_k < logits.shape[-1]:
        kth = mx.topk(logits, top_k, axis=-1)[..., :1]
        logits = mx.where(logits < kth, mx.finfo(mx.float32).min, logits)
    logits = logits / temp
    if top_p and top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        order = mx.argsort(-probs, axis=-1)
        sorted_p = mx.take_along_axis(probs, order, axis=-1)
        cum = mx.cumsum(sorted_p, axis=-1)
        # keep the smallest prefix whose mass exceeds top_p (always >= 1 token)
        keep = (cum - sorted_p) < top_p
        masked = mx.where(keep, mx.log(sorted_p + 1e-20), mx.finfo(mx.float32).min)
        pick = mx.random.categorical(masked, axis=-1)
        return mx.take_along_axis(order, pick[..., None], axis=-1).squeeze(-1)
    return mx.random.categorical(logits, axis=-1)


def generate(model, tk, prompt: str, max_tokens=200, temp=1.0, top_p=0.95,
             top_k=20, stream=True, prefill_step=384):
    ids = tk.encode(prompt)
    x = mx.array([ids])

    t0 = time.time()
    # Prefill in windows. Peak memory scales with the window, not the prompt:
    # measured on a 2048-token prefill, peak was 16.9 GB at window 384 vs 21.4 GB
    # at 1024 and 26.3 GB at 2048 (which pushed decode into swap). Throughput is
    # flat (190-198 tok/s) because prefill is MLP-compute-bound, so the small
    # window is free. 384 also matches METAL_MAX_L, so the fused delta kernel --
    # not the chunked scan -- handles prefill. Segmented prefill is exact (see
    # test_model.py "chunked prefill == full forward").
    logits, cache = None, model.make_cache()
    for s in range(0, x.shape[1], prefill_step):
        logits, cache = model(x[:, s:s + prefill_step], cache, all_logits=False)
        mx.eval(logits)
    prefill_t = time.time() - t0

    out, buf = [], ""
    t1 = time.time()
    for _ in range(max_tokens):
        nxt = sample(logits[:, -1, :], temp, top_p, top_k)
        mx.eval(nxt)
        tid = int(nxt.item())
        if tid in EOS:
            break
        out.append(tid)
        if stream:
            # decode incrementally, holding back incomplete utf-8 / partial tokens
            piece = tk.decode(out)
            if len(piece) > len(buf) and "�" not in piece[len(buf):]:
                print(piece[len(buf):], end="", flush=True)
                buf = piece
        logits, cache = model(nxt[:, None], cache, all_logits=False)
    gen_t = time.time() - t1

    if stream:
        rest = tk.decode(out)[len(buf):]
        if rest:
            print(rest, end="", flush=True)
        print()
    print(f"\n[prefill {len(ids)} tok in {prefill_t:.2f}s ({len(ids)/prefill_t:.1f} tok/s) | "
          f"decode {len(out)} tok in {gen_t:.2f}s ({len(out)/max(gen_t,1e-9):.1f} tok/s) | "
          f"peak {mx.get_peak_memory()/1e9:.1f} GB]")
    return tk.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="Hello")
    ap.add_argument("--model", default="qwen3.5-27b-4bit")
    ap.add_argument("--raw", default=None, help="skip the chat template entirely")
    ap.add_argument("--system", default=None)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("-n", "--max-tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mx.random.seed(a.seed)
    path = qwen35.find_model(a.model)
    tk = Tokenizer(f"{path}/tokenizer.json")
    model, _ = qwen35.load(path)

    p = a.raw if a.raw is not None else chat_prompt(a.prompt, a.system, not a.no_think)
    print(f"--- prompt ---\n{p}\n--- output ---")
    generate(model, tk, p, a.max_tokens, a.temp, a.top_p, a.top_k)


if __name__ == "__main__":
    main()
