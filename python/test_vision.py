"""
Vision tests. Each stage is held against transformers' Qwen3-VL implementation
where one exists -- the checkpoint's vision tower IS Qwen3-VL's encoder, tensor
for tensor -- and against the model's own answers where it does not.

    python3 test_vision.py                        # everything
    python3 test_vision.py --model qwen3.5-9b-4bit

The reference comparisons need torch, transformers and PIL. They are only test
dependencies, and are skipped (not failed) when missing; the implementation
itself needs none of them.
"""

import argparse
import os
import sys
import types

import mlx.core as mx

import qwen35
import vision

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "testdata")

ok_all = True


def check(ok, what):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")


def skip(what):
    print(f"  [skip] {what}")


def have(*mods):
    try:
        for m in mods:
            __import__(m)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# pure functions
# ---------------------------------------------------------------------------


def test_pure():
    print("\nresize rule, positions, placeholders")
    check(vision.smart_resize(480, 640) == (480, 640), "smart_resize keeps an in-range multiple of 32")
    h, w = vision.smart_resize(3024, 4032)
    check(h % 32 == 0 and w % 32 == 0 and h * w <= vision.MAX_PIXELS and abs(w / h - 4 / 3) < 0.05,
          f"smart_resize shrinks a 12 MP photo to {w}x{h}, <= MAX_PIXELS, aspect kept")
    h, w = vision.smart_resize(120, 200)
    check(h * w >= vision.MIN_PIXELS, f"smart_resize grows a tiny image to {w}x{h}, >= MIN_PIXELS")
    # 80/32 = 2.5 and 112/32 = 3.5: half-to-even takes them to 2 and 4 (64, 128),
    # where half-up would give 3 and 4 (96, 128)
    check(vision.smart_resize(80, 112, min_pixels=1, max_pixels=10 ** 9) == (64, 128),
          "rounding is half-to-even, as Python's round() in the reference is")

    t, hh, ww = vision.image_positions(10, 2, 3)
    check(t == [10] * 6 and hh == [10, 10, 10, 11, 11, 11] and ww == [10, 11, 12, 10, 11, 12],
          "image_positions: shared t, rows on h, cols on w")
    sp = qwen35.ImageSpan(0, 3, 5)
    check(qwen35.delta_contribution(sp) == 5 - 15, "a 3x5 image advances positions by 5, not 15")
    # 7 tokens of a 3x5 image: rows 0..1, cols 0..4, so the largest is 4 and the
    # text resumes at +5 -- a delta of 5 - 7
    check(qwen35.delta_contribution(sp, 7) == 5 - 7,
          "a partly-fed image resumes one past its largest position")

    pad = 248056
    ids = qwen35.expand_image_pads([1, pad, 2, pad, 3], [2, 3], pad)
    check(ids == [1, pad, pad, 2, pad, pad, pad, 3], "one <|image_pad|> expands into one per token")
    spans = qwen35.find_image_spans([pad] * 5, [(1, 2), (1, 3)], pad, base=100)
    check([(s.start, s.n) for s in spans] == [(100, 2), (102, 3)],
          "adjacent images in one pad run are split by their grids")
    try:
        qwen35.expand_image_pads([1, pad, pad], [4], pad)
        check(False, "a placeholder/image count mismatch is refused")
    except ValueError:
        check(True, "a placeholder/image count mismatch is refused")

    axes = qwen35.mrope_axes(64, (11, 11, 10))
    check(''.join('THW'[a] for a in axes) == "THW" * 10 + "TH", "M-RoPE frequencies interleave THWTHW...TH")
    # t = h = w must reduce to plain rope: that is what keeps text unchanged. The
    # int path is the fused kernel and the array path is written out, so they
    # agree to rounding rather than to the bit.
    x = mx.random.normal((1, 2, 5, 256))
    a = qwen35.rope(x, 7, 64, 1e7)
    b = qwen35.rope(x, mx.array([[7, 8, 9, 10, 11]] * 3), 64, 1e7)
    err = mx.abs(a - b).max().item()
    check(err < 1e-5, f"M-RoPE with t = h = w is plain RoPE (max |diff| {err:.1e})")


# ---------------------------------------------------------------------------
# decoding and preprocessing, against PIL and transformers
# ---------------------------------------------------------------------------

IMAGES = ["shapes.png", "arrow_exif6.jpg", "alpha_text.png", "gray.png"]


def test_decode():
    print("\ndecoding (ImageIO), against PIL")
    if not have("PIL", "numpy"):
        skip("PIL/numpy not installed")
        return
    import numpy as np
    from PIL import Image, ImageOps
    for f in IMAGES:
        p = os.path.join(DATA, f)
        rgb, w, h = vision.load(p)
        mine = np.frombuffer(rgb, np.uint8).reshape(h, w, 3).astype(int)
        im = ImageOps.exif_transpose(Image.open(p))
        if im.mode == "RGBA":   # the documented deviation: composite over white
            im = Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im)
        ref = np.asarray(im.convert("RGB")).astype(int)
        good = mine.shape == ref.shape and np.abs(mine - ref).max() <= (1 if f.endswith(".jpg") else 0)
        check(good, f"{f:16s} {w}x{h}, max |diff| "
                    f"{np.abs(mine - ref).max() if mine.shape == ref.shape else 'shape!'}"
                    f"{'  (JPEG IDCTs differ by 1)' if f.endswith('.jpg') else ''}")


def test_preprocess():
    print("\npreprocessing, against transformers' Qwen2VLImageProcessor")
    if not have("PIL", "numpy", "transformers"):
        skip("PIL/numpy/transformers not installed")
        return
    import numpy as np
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    for f, maxp, what in [("shapes.png", vision.MAX_PIXELS, "no resize"),
                          ("shapes.png", 100_000, "downscale"),
                          ("gray.png", vision.MAX_PIXELS, "upscale"),
                          ("arrow_exif6.jpg", 70_000, "jpeg, downscale")]:
        rgb, w, h = vision.load(os.path.join(DATA, f))
        mine = vision.preprocess(rgb, w, h, vision.MIN_PIXELS, maxp)
        proc = Qwen2VLImageProcessor(min_pixels=vision.MIN_PIXELS, max_pixels=maxp, patch_size=16,
                                     temporal_patch_size=2, merge_size=2,
                                     image_mean=[.5] * 3, image_std=[.5] * 3)
        out = proc(images=[Image.frombytes("RGB", (w, h), rgb)], return_tensors="np")
        grid = tuple(int(g) for g in out["image_grid_thw"][0])
        a, b = np.array(mine.pixels), out["pixel_values"]
        # Back to 0..255: the resized pixels themselves must be identical. What is
        # left after that is the normalisation, which the reference does at a
        # different precision -- one fp32 ULP, against 7.8e-3 for one pixel level.
        level = lambda x: np.round((x * 0.5 + 0.5) * 255.0).astype(int)
        same_px = a.shape == b.shape and (level(a) == level(b)).all()
        d = np.abs(a - b).max() if a.shape == b.shape else float("inf")
        check(grid == mine.grid and same_px and d < 1e-6,
              f"{f:16s} {what:16s} grid {mine.grid}, pixels identical, values within {d:.1e}")


# ---------------------------------------------------------------------------
# the tower and the positions, against transformers
# ---------------------------------------------------------------------------


def test_tower(bf16_dir):
    print("\nvision tower, fp32, against transformers' Qwen3VLVisionModel")
    if not have("torch", "numpy", "transformers"):
        skip("torch/numpy/transformers not installed")
        return
    if not bf16_dir:
        skip("needs an unquantized checkpoint with a vision tower (--bf16)")
        return
    import json
    import numpy as np
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    vc = json.load(open(os.path.join(bf16_dir, "config.json")))["vision_config"]
    index = json.load(open(os.path.join(bf16_dir, "model.safetensors.index.json")))["weight_map"]
    W = {}
    for shard in sorted({s for k, s in index.items() if k.startswith("model.visual.")}):
        for k, v in mx.load(os.path.join(bf16_dir, shard)).items():
            if k.startswith("model.visual."):
                W[k[len("model.visual."):]] = v.astype(mx.float32)
    mine = vision.VisionModel(vision.VisionConfig.from_dict(vc))
    mine.load_weights(list(W.items()), strict=True)
    cfg = Qwen3VLVisionConfig(**{k: v for k, v in vc.items() if k != "model_type"})
    cfg._attn_implementation = "eager"
    ref = Qwen3VLVisionModel(cfg).float().eval()
    ref.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in W.items()}, strict=True)
    for f, maxp in [("shapes.png", 100_000), ("arrow_exif6.jpg", vision.MAX_PIXELS)]:
        img = vision.load_image(os.path.join(DATA, f), max_pixels=maxp)
        a = np.array(mine(img))
        with torch.no_grad():
            b = ref(torch.from_numpy(np.array(img.pixels)), torch.tensor([img.grid]))[0].numpy()
        rel = np.abs(a - b).max() / np.abs(b).max()
        check(a.shape == b.shape and rel < 2e-4, f"{f:16s} grid {img.grid} -> {a.shape}, rel err {rel:.1e}")


def test_rope_index():
    print("\nM-RoPE positions, against transformers' get_rope_index")
    if not have("torch", "transformers"):
        skip("torch/transformers not installed")
        return
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    IMG, VS, VE = 248056, 248053, 248054
    grids = [(3, 5), (4, 2)]
    ids = [1, 2, 3, VS] + [IMG] * 15 + [VE, 7, 8, VS] + [IMG] * 8 + [VE, 9, 10, 11]
    fake = types.SimpleNamespace(config=types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2), image_token_id=IMG,
        video_token_id=248057, vision_start_token_id=VS))
    ref, ref_delta = Qwen3VLModel.get_rope_index(
        fake, torch.tensor([ids]), image_grid_thw=torch.tensor([[1, 2 * r, 2 * c] for r, c in grids]))
    ref = ref[:, 0].tolist()

    seen = []

    class Stub:     # captures the positions prefill hands each window
        cfg = types.SimpleNamespace(image_token_id=IMG)

        def hidden_states(self, chunk, cache, rope=None, image_embeds=None):
            n = chunk.shape[1]
            seen.append(rope.tolist() if isinstance(rope, mx.array) else [[rope + i for i in range(n)]] * 3)
            cache[0].offset += n
            return mx.zeros((1, n, 4)), cache

    for step in (384, 5):
        seen.clear()
        cache = [types.SimpleNamespace(offset=0)]
        _, delta = qwen35.prefill(Stub(), cache, ids, [(mx.zeros((r * c, 4)), r, c) for r, c in grids], 0, step)
        mine = [sum((w[a] for w in seen), []) for a in range(3)]
        check(mine == ref and delta == ref_delta.item(),
              f"two images, windows of {step}: positions identical, delta {delta}")


# ---------------------------------------------------------------------------
# the model's answers
# ---------------------------------------------------------------------------


def test_answers(model_dir):
    print(f"\nanswers ({os.path.basename(model_dir)})")
    import chat
    from tokenizer import Tokenizer
    tk = Tokenizer(os.path.join(model_dir, "tokenizer.json"))
    m, _ = qwen35.load(model_dir, verbose=False)
    if not m.has_vision:
        skip("checkpoint has no vision tower")
        return

    def ask(images, q, spec=False, n=24):
        s = chat.Session(m, tk, spec=spec)
        return "".join(s.turn(q, temp=0.0, max_tokens=n,
                              images=[vision.load_image(os.path.join(DATA, f)) for f in images]))

    a = ask(["shapes.png"], "What number is written in the image? Answer with the number only.")
    check("42" in a, f"reads the number: {a.strip()!r}")
    a = ask(["arrow_exif6.jpg"], "Which way does the arrow point: up, down, left or right? One word.")
    check(a.strip().lower().startswith("up"), f"EXIF orientation applied, the arrow points {a.strip()!r}")
    a = ask(["alpha_text.png"], "What word is written? The word only.")
    check("cat" in a.lower(), f"black text on a transparent background is legible: {a.strip()!r}")

    # A follow-up turn with no image, then a second image: the rope delta has to
    # carry across turns and accumulate across images.
    s = chat.Session(m, tk, spec=True)
    img = lambda f: vision.load_image(os.path.join(DATA, f))
    "".join(s.turn("What colour is the square? One word.", temp=0.0, max_tokens=8, images=[img("shapes.png")]))
    a2 = "".join(s.turn("And the circle? One word.", temp=0.0, max_tokens=8))
    "".join(s.turn("Which way does this arrow point? One word.", temp=0.0, max_tokens=8,
                   images=[img("arrow_exif6.jpg")]))
    a4 = "".join(s.turn("In the FIRST picture, what number was written? Number only.", temp=0.0, max_tokens=8))
    check("blue" in a2.lower() and "42" in a4,
          f"multi-turn: follow-up {a2.strip()!r}, recall across images {a4.strip()!r} "
          f"(rope delta {s.rope_delta})")

    # Speculative decoding must not drift from plain decoding with an image in
    # context: the draft and verify passes both take the shifted positions. A
    # wrong shift diverges at once, on a clearly-decided token; what is allowed
    # is the bf16 caveat speculative decoding always has -- verify scores at
    # L=k+1, plain decode at L=1, and on an EXACT tie the argmax can differ -- so
    # a divergence passes only if the plain model's top two there are within bf16
    # resolution of each other.
    q = "List the shapes and their colours."
    im = vision.load_image(os.path.join(DATA, "shapes.png"))
    runs = {}
    for spec in (False, True):
        s = chat.Session(m, tk, spec=spec)
        "".join(s.turn(q, temp=0.0, max_tokens=40, images=[im]))
        runs[spec] = s.tokens
    a, b = runs[False], runs[True]
    if a == b:
        check(True, "speculative == plain at temp 0, image in context")
    else:
        i = next(n for n in range(min(len(a), len(b))) if a[n] != b[n])
        s = chat.Session(m, tk, spec=False)
        n0 = len(s._prompt_tokens(q, False, [im]))
        cache = m.make_cache()
        h, delta = qwen35.prefill(m, cache, a[:n0], [(m.embed_image(im),) + im.llm_grid])
        for t in a[n0:i]:
            h, cache = m.hidden_states(mx.array([[t]]), cache, rope=cache[0].offset + delta)
        lg = m.lm_head(h)[0, -1].astype(mx.float32)
        top = mx.sort(lg)[-2:].tolist()
        gap = top[1] - top[0]
        check(i > n0 and gap <= 0.25,
              f"speculative vs plain, image in context: diverge at reply token {i - n0} "
              f"on a bf16 tie (top-2 gap {gap:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b-4bit")
    ap.add_argument("--bf16", default="Qwen3.5-9B",
                    help="an unquantized checkpoint, for the fp32 tower comparison")
    a = ap.parse_args()

    test_pure()
    test_decode()
    test_preprocess()
    try:
        bf16 = qwen35.find_model(a.bf16)
    except SystemExit:
        bf16 = None
    test_tower(bf16)
    test_rope_index()
    try:
        model_dir = qwen35.find_model(a.model)
    except SystemExit:
        model_dir = None
    if model_dir:
        test_answers(model_dir)
    else:
        skip(f"model tests: {a.model} not found")
    print("\nALL PASS" if ok_all else "\nFAILURES ABOVE")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
