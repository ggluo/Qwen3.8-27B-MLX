"""
The Qwen3.5 vision tower, and everything between an image file and it.

The checkpoint's `model.visual.*` tensors are, name for name and shape for
shape, Qwen3-VL's vision encoder (verified against transformers'
Qwen3VLVisionModel: all 333 tensors, no extras, no shape differences). So
transformers is a trustworthy numerical reference here, and test_vision.py
holds every stage below against it.

    file bytes --ImageIO--> RGB uint8 (EXIF orientation applied)
      --smart_resize--> both sides a multiple of 32, pixel count clamped
      --bicubic--> resized RGB uint8
      --(x/255 - 0.5)/0.5, 2 identical frames--> patches (N, 3*2*16*16)
      --VisionModel--> (N/4, text hidden size): one embedding per LLM token

    27 ViT blocks, hidden 1152, 16 heads of 72, 2-D rotary inside attention,
    a learned 48x48 position table bilinearly resized to the image's grid, and
    a merger that folds each 2x2 block of patches into one token.

Three things here are easy to get wrong without an error:

  * The patch ORDER. Patches are laid out merge-block-major -- (block row,
    block col, row in block, col in block) -- not raster order, so that each
    run of 4 consecutive patches is one 2x2 block the merger can fold with a
    reshape. The position table and the rotary coordinates follow that order.
  * Two different GELUs. The block MLPs use the tanh approximation
    (`gelu_pytorch_tanh`); the merger uses the exact erf form (`nn.GELU()`).
  * The merger's LayerNorm runs per patch (1152 wide) BEFORE the 2x2 fold, not
    on the folded 4608-wide vector.

Decoding goes through macOS ImageIO rather than PIL. That keeps this side
stdlib-only, handles JPEG/PNG/HEIC/WebP/TIFF, and -- the real reason -- is the
same decoder the C++ port calls, so the two implementations see identical
pixels and their outputs can be compared byte for byte.
"""

import ctypes
import math
from dataclasses import dataclass
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# preprocessing constants, from preprocessor_config.json
# ---------------------------------------------------------------------------

PATCH = 16
TEMPORAL = 2          # a still image is duplicated to fill the 2-frame patch
MERGE = 2             # 2x2 patches -> one LLM token
FACTOR = PATCH * MERGE

MIN_PIXELS = 65536    # the checkpoint's shortest_edge: 64 image tokens

# The checkpoint says 16777216 (longest_edge), which is 16384 tokens for one
# image -- minutes of prefill on this hardware. ~1M pixels is 1024 tokens, which
# keeps small text in a screenshot legible. Override with max_pixels=.
MAX_PIXELS = 1024 * FACTOR * FACTOR


# ---------------------------------------------------------------------------
# decoding: macOS ImageIO through ctypes
# ---------------------------------------------------------------------------


class _CGRect(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double),
                ("w", ctypes.c_double), ("h", ctypes.c_double)]


_IIO = None


def _frameworks():
    """Load and type the handful of CoreFoundation/CoreGraphics/ImageIO calls."""
    global _IIO
    if _IIO is not None:
        return _IIO
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    io = ctypes.CDLL("/System/Library/Frameworks/ImageIO.framework/ImageIO")
    p = ctypes.c_void_p

    def sig(lib, name, res, *args):
        f = getattr(lib, name)
        f.restype, f.argtypes = res, list(args)
        return f

    ns = type("IIO", (), {})()
    ns.CFDataCreate = sig(cf, "CFDataCreate", p, p, ctypes.c_char_p, ctypes.c_long)
    ns.CFRelease = sig(cf, "CFRelease", None, p)
    ns.CFDictionaryGetValue = sig(cf, "CFDictionaryGetValue", p, p, p)
    ns.CFNumberGetValue = sig(cf, "CFNumberGetValue", ctypes.c_bool, p, ctypes.c_long, p)
    ns.CGImageSourceCreateWithData = sig(io, "CGImageSourceCreateWithData", p, p, p)
    ns.CGImageSourceGetCount = sig(io, "CGImageSourceGetCount", ctypes.c_size_t, p)
    ns.CGImageSourceCreateImageAtIndex = sig(io, "CGImageSourceCreateImageAtIndex", p, p,
                                             ctypes.c_size_t, p)
    ns.CGImageSourceCopyPropertiesAtIndex = sig(io, "CGImageSourceCopyPropertiesAtIndex",
                                                p, p, ctypes.c_size_t, p)
    ns.kOrientation = p.in_dll(io, "kCGImagePropertyOrientation")
    ns.CGImageGetWidth = sig(cg, "CGImageGetWidth", ctypes.c_size_t, p)
    ns.CGImageGetHeight = sig(cg, "CGImageGetHeight", ctypes.c_size_t, p)
    ns.CGImageRelease = sig(cg, "CGImageRelease", None, p)
    ns.CGImageGetColorSpace = sig(cg, "CGImageGetColorSpace", p, p)
    ns.CGColorSpaceGetModel = sig(cg, "CGColorSpaceGetModel", ctypes.c_int, p)
    ns.CGColorSpaceCreateWithName = sig(cg, "CGColorSpaceCreateWithName", p, p)
    ns.CGColorSpaceRelease = sig(cg, "CGColorSpaceRelease", None, p)
    ns.kSRGB = p.in_dll(cg, "kCGColorSpaceSRGB")
    ns.CGBitmapContextCreate = sig(cg, "CGBitmapContextCreate", p, p, ctypes.c_size_t,
                                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, p,
                                   ctypes.c_uint32)
    ns.CGContextRelease = sig(cg, "CGContextRelease", None, p)
    ns.CGContextSetInterpolationQuality = sig(cg, "CGContextSetInterpolationQuality",
                                              None, p, ctypes.c_int)
    ns.CGContextSetRGBFillColor = sig(cg, "CGContextSetRGBFillColor", None, p,
                                      ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double)
    ns.CGContextFillRect = sig(cg, "CGContextFillRect", None, p, _CGRect)
    ns.CGContextDrawImage = sig(cg, "CGContextDrawImage", None, p, _CGRect, p)
    _IIO = ns
    return ns


_RGB_MODEL = 1                      # kCGColorSpaceModelRGB
_PREMULTIPLIED_LAST = 1             # kCGImageAlphaPremultipliedLast: RGBA bytes
_CF_INT = 9                         # kCFNumberIntType
_INTERPOLATION_NONE = 1


def decode(data: bytes) -> Tuple[bytes, int, int]:
    """Encoded image bytes -> (RGB bytes, width, height), upright.

    The image is drawn into an 8-bit RGBA bitmap that was first filled white, so
    any transparency is composited over WHITE. That departs from the reference
    loader, which drops alpha and keeps whatever colour a transparent pixel
    stored -- usually black, which makes black text on a transparent background
    vanish. Opaque images are unaffected.

    An RGB image is drawn in its own colour space, so its values come through
    as stored rather than colour-managed; anything else (grey, CMYK, indexed) is
    converted to sRGB.
    """
    f = _frameworks()
    if not data:
        raise ValueError("empty image")
    cfdata = f.CFDataCreate(None, data, len(data))
    src = f.CGImageSourceCreateWithData(cfdata, None)
    f.CFRelease(cfdata)
    if not src:
        raise ValueError("not an image ImageIO can read")
    try:
        if f.CGImageSourceGetCount(src) < 1:
            raise ValueError("image file holds no images")
        img = f.CGImageSourceCreateImageAtIndex(src, 0, None)
        if not img:
            raise ValueError("cannot decode image")
        orientation = 1
        props = f.CGImageSourceCopyPropertiesAtIndex(src, 0, None)
        if props:
            num = f.CFDictionaryGetValue(props, f.kOrientation)
            if num:
                v = ctypes.c_int(1)
                if f.CFNumberGetValue(num, _CF_INT, ctypes.byref(v)):
                    orientation = v.value
            f.CFRelease(props)
    finally:
        f.CFRelease(src)

    try:
        w, h = f.CGImageGetWidth(img), f.CGImageGetHeight(img)
        own = f.CGImageGetColorSpace(img)
        made = None
        cs = own
        if not own or f.CGColorSpaceGetModel(own) != _RGB_MODEL:
            made = cs = f.CGColorSpaceCreateWithName(f.kSRGB)
        buf = ctypes.create_string_buffer(w * h * 4)
        ctx = f.CGBitmapContextCreate(buf, w, h, 8, w * 4, cs, _PREMULTIPLIED_LAST)
        if not ctx and made is None:
            # an exotic embedded profile that bitmap contexts refuse: fall back
            made = cs = f.CGColorSpaceCreateWithName(f.kSRGB)
            ctx = f.CGBitmapContextCreate(buf, w, h, 8, w * 4, cs, _PREMULTIPLIED_LAST)
        if made is not None:
            f.CGColorSpaceRelease(made)
        if not ctx:
            raise ValueError("cannot create a bitmap context for this image")
        rect = _CGRect(0, 0, w, h)
        f.CGContextSetInterpolationQuality(ctx, _INTERPOLATION_NONE)
        f.CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0)
        f.CGContextFillRect(ctx, rect)
        f.CGContextDrawImage(ctx, rect, img)
        f.CGContextRelease(ctx)
    finally:
        f.CGImageRelease(img)

    rgba = buf.raw
    rgb = bytearray(w * h * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = rgba[0::4], rgba[1::4], rgba[2::4]
    return orient(bytes(rgb), w, h, orientation)


def orient(rgb: bytes, w: int, h: int, orientation: int) -> Tuple[bytes, int, int]:
    """Apply an EXIF orientation (1..8), as PIL.ImageOps.exif_transpose does.

    Phone photos are stored sideways with a tag saying so; the reference loader
    rights them before anything else, and so must we or the model sees the
    photo on its side.
    """
    if orientation in (0, 1):
        return rgb, w, h
    a = mx.array(memoryview(rgb), dtype=mx.uint8).reshape(h, w, 3)
    t = lambda x: mx.transpose(x, (1, 0, 2))
    ops = {
        2: lambda x: x[:, ::-1],          # mirror
        3: lambda x: x[::-1, ::-1],       # rotate 180
        4: lambda x: x[::-1, :],          # flip vertical
        5: lambda x: t(x),                # transpose
        6: lambda x: t(x)[:, ::-1],       # rotate 90 clockwise
        7: lambda x: t(x)[::-1, ::-1],    # transverse
        8: lambda x: t(x)[::-1, :],       # rotate 90 counter-clockwise
    }
    if orientation not in ops:
        return rgb, w, h
    a = mx.contiguous(ops[orientation](a))
    return bytes(memoryview(a)), a.shape[1], a.shape[0]


def load(path: str) -> Tuple[bytes, int, int]:
    with open(path, "rb") as fh:
        return decode(fh.read())


# ---------------------------------------------------------------------------
# resizing and patching
# ---------------------------------------------------------------------------


def _round_half_even(x: float) -> int:
    # Python's round() is banker's rounding, and smart_resize relies on it; the
    # C++ port uses nearbyint() under the default rounding mode, which agrees.
    return round(x)


def smart_resize(h: int, w: int, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS, factor: int = FACTOR) -> Tuple[int, int]:
    """Qwen2-VL's resize rule, verbatim: both sides a multiple of `factor`, the
    pixel count clamped to [min_pixels, max_pixels], aspect ratio kept as
    closely as that allows."""
    if max(h, w) / min(h, w) > 200:
        raise ValueError(f"aspect ratio {max(h, w) / min(h, w):.0f}:1 is more than 200:1")
    hb = _round_half_even(h / factor) * factor
    wb = _round_half_even(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def _cubic(x: float, a: float = -0.5) -> float:
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def resample_matrix(n_in: int, n_out: int) -> List[float]:
    """Row-major (n_out, n_in) bicubic weights, antialiased the way PIL is.

    When shrinking, the kernel is stretched by the scale factor so every input
    pixel contributes -- plain bicubic would alias badly at 4x. Built in double
    precision with a fixed operation order; the C++ port builds the same matrix
    the same way, so both feed identical weights to the same MLX matmul.
    """
    scale = n_in / n_out
    fs = max(scale, 1.0)
    support = 2.0 * fs
    m = [0.0] * (n_out * n_in)
    for i in range(n_out):
        center = (i + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        ws = [_cubic((j - center + 0.5) / fs) for j in range(lo, hi)]
        total = sum(ws)
        for j, wj in zip(range(lo, hi), ws):
            m[i * n_in + j] = wj / total
    return m


def _to_u8(x: mx.array) -> mx.array:
    # PIL rounds half up and clips at every pass; floor(x + 0.5) is that.
    return mx.clip(mx.floor(x + 0.5), 0, 255)


def resize(chw: mx.array, out_h: int, out_w: int) -> mx.array:
    """(3, H, W) float32 holding 0..255 -> (3, out_h, out_w), same range.

    Two separable passes as matmuls, horizontal first and rounded to 8 bits in
    between, which is the order and precision PIL uses.
    """
    _, h, w = chw.shape
    if out_w != w:
        wh = mx.array(resample_matrix(w, out_w), dtype=mx.float32).reshape(out_w, w)
        chw = _to_u8(chw @ wh.T)
    if out_h != h:
        wv = mx.array(resample_matrix(h, out_h), dtype=mx.float32).reshape(out_h, h)
        chw = _to_u8(wv @ chw)
    return chw


@dataclass
class Image:
    """One image, ready for the vision tower."""
    pixels: mx.array      # (N, 1536) float32 patches, merge-block order
    grid: Tuple[int, int, int]   # (t, h, w) in PATCHES; tokens = t*h*w // 4
    size: Tuple[int, int]        # original (width, height), after EXIF orientation
    resized: Tuple[int, int]     # (width, height) fed to the model

    @property
    def n_tokens(self) -> int:
        t, h, w = self.grid
        return t * (h // MERGE) * (w // MERGE)

    @property
    def llm_grid(self) -> Tuple[int, int]:
        """(rows, cols) of LLM tokens -- what M-RoPE indexes."""
        return self.grid[1] // MERGE, self.grid[2] // MERGE


def preprocess(rgb: bytes, w: int, h: int, min_pixels: int = MIN_PIXELS,
               max_pixels: int = MAX_PIXELS) -> Image:
    rh, rw = smart_resize(h, w, min_pixels, max_pixels)
    x = mx.array(memoryview(rgb), dtype=mx.uint8).reshape(h, w, 3)
    x = mx.transpose(x, (2, 0, 1)).astype(mx.float32)          # (3, H, W)
    x = resize(x, rh, rw)
    # rescale then normalise, in that order and in float32, as the reference does
    x = (x * (1.0 / 255.0) - 0.5) / 0.5
    gh, gw = rh // PATCH, rw // PATCH
    x = mx.stack([x] * TEMPORAL)                                 # (T, 3, H, W)
    x = x.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    x = x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)
    return Image(pixels=x, grid=(1, gh, gw), size=(w, h), resized=(rw, rh))


def load_image(path: str, **kw) -> Image:
    rgb, w, h = load(path)
    return preprocess(rgb, w, h, **kw)


def image_from_bytes(data: bytes, **kw) -> Image:
    rgb, w, h = decode(data)
    return preprocess(rgb, w, h, **kw)


# ---------------------------------------------------------------------------
# the vision tower
# ---------------------------------------------------------------------------


@dataclass
class VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    out_hidden_size: int = 5120
    num_position_embeddings: int = 2304
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    deepstack_visual_indexes: Tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, d: dict) -> "VisionConfig":
        keep = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in keep}
        kw["deepstack_visual_indexes"] = tuple(d.get("deepstack_visual_indexes") or ())
        return cls(**kw)


class PatchEmbed(nn.Module):
    """A Conv3d whose stride equals its kernel is a matmul over flattened
    patches. The weight stays in its stored (1152, 3, 2, 16, 16) shape so the
    checkpoint loads as is; its (C, T, kh, kw) order is the patch vector's."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        k = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        self.proj = _Conv3dAsLinear(cfg.hidden_size, k, cfg)

    def __call__(self, x):
        return self.proj(x)


class _Conv3dAsLinear(nn.Module):
    def __init__(self, out_dim, in_dim, cfg: VisionConfig):
        super().__init__()
        self.weight = mx.zeros((out_dim, cfg.in_channels, cfg.temporal_patch_size,
                                cfg.patch_size, cfg.patch_size))
        self.bias = mx.zeros((out_dim,))
        self.in_dim = in_dim

    def __call__(self, x):
        w = self.weight.reshape(self.weight.shape[0], self.in_dim)
        return x.astype(w.dtype) @ w.T + self.bias


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.heads = cfg.num_heads
        self.hd = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=True)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def __call__(self, x, cos, sin):
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.heads, self.hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]              # (N, H, hd)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))  # (1, H, N, hd)
        # Bidirectional: every patch of an image sees every other. One image per
        # call, so there are no cross-image blocks to mask out.
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5)
        return self.proj(o[0].transpose(1, 0, 2).reshape(n, -1))


def _rotate(x, cos, sin):
    """rotate_half RoPE over the whole head, computed in fp32 as the reference
    does. cos/sin are (N, hd)."""
    dt = x.dtype
    x = x.astype(mx.float32)
    half = x.shape[-1] // 2
    rot = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return (x * cos[:, None, :] + rot * sin[:, None, :]).astype(dt)


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))   # tanh GELU


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def __call__(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.unit = cfg.spatial_merge_size ** 2
        wide = cfg.hidden_size * self.unit
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)   # per patch, BEFORE folding
        self.linear_fc1 = nn.Linear(wide, wide, bias=True)
        self.linear_fc2 = nn.Linear(wide, cfg.out_hidden_size, bias=True)

    def __call__(self, x):
        x = self.norm(x).reshape(-1, x.shape[-1] * self.unit)  # 4 patches -> 1 token
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))     # exact erf GELU


class VisionModel(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        if cfg.deepstack_visual_indexes:
            # Qwen3-VL proper feeds intermediate layers into the LLM ("deepstack");
            # this checkpoint lists none, and the text model has nowhere to put them.
            raise NotImplementedError("deepstack vision features are not supported")
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)
        self.blocks = [VisionBlock(cfg) for _ in range(cfg.depth)]
        self.merger = PatchMerger(cfg)
        self.side = int(round(math.sqrt(cfg.num_position_embeddings)))   # 48

    # -- positions, both in merge-block order --

    def _merge_order(self, gh: int, gw: int):
        """(row, col) of every patch, in the order the patches are laid out."""
        m = self.cfg.spatial_merge_size
        rows, cols = [], []
        for br in range(gh // m):
            for bc in range(gw // m):
                for r in range(m):
                    for c in range(m):
                        rows.append(br * m + r)
                        cols.append(bc * m + c)
        return rows, cols

    def position_embedding(self, gh: int, gw: int) -> mx.array:
        """The learned 48x48 table, bilinearly resized to (gh, gw).

        Sample points are linspace(0, 47, n) on each axis -- corner-aligned, so
        the first and last rows of the table land exactly on the image's edges.
        """
        s = self.side

        def axis(n):
            if n == 1:
                return [0.0]
            return [i * (s - 1) / (n - 1) for i in range(n)]

        hs, ws = axis(gh), axis(gw)
        rows, cols = self._merge_order(gh, gw)
        idx = [[], [], [], []]
        wt = [[], [], [], []]
        for r, c in zip(rows, cols):
            y, x = hs[r], ws[c]
            y0, x0 = int(y), int(x)
            y1, x1 = min(y0 + 1, s - 1), min(x0 + 1, s - 1)
            dy, dx = y - y0, x - x0
            for k, (yy, xx, ww) in enumerate(((y0, x0, (1 - dy) * (1 - dx)),
                                              (y0, x1, (1 - dy) * dx),
                                              (y1, x0, dy * (1 - dx)),
                                              (y1, x1, dy * dx))):
                idx[k].append(yy * s + xx)
                wt[k].append(ww)
        out = None
        for k in range(4):
            # through the module, not .weight, so a quantized table works too
            rows_k = self.pos_embed(mx.array(idx[k])).astype(mx.float32)
            term = rows_k * mx.array(wt[k], dtype=mx.float32)[:, None]
            out = term if out is None else out + term
        return out

    def rotary(self, gh: int, gw: int):
        """2-D rotary for the attention: half the frequencies index the patch's
        row, half its column. Returns fp32 (cos, sin), each (N, head_dim)."""
        hd = self.cfg.hidden_size // self.cfg.num_heads     # 72
        dim = hd // 2                                        # 36 -> 18 frequencies
        inv = 1.0 / (10000.0 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        rows, cols = self._merge_order(gh, gw)
        r = mx.array(rows, dtype=mx.float32)[:, None] * inv[None]
        c = mx.array(cols, dtype=mx.float32)[:, None] * inv[None]
        emb = mx.concatenate([r, c], axis=-1)                # (N, 36)
        emb = mx.concatenate([emb, emb], axis=-1)            # (N, 72)
        return mx.cos(emb), mx.sin(emb)

    def __call__(self, image: Image) -> mx.array:
        """One image -> (n_tokens, out_hidden_size), in the LLM's dtype."""
        t, gh, gw = image.grid
        if t != 1:
            raise NotImplementedError("video is not supported")
        dtype = self.merger.linear_fc2.bias.dtype
        x = self.patch_embed(image.pixels.astype(dtype))
        x = (x.astype(mx.float32) + self.position_embedding(gh, gw)).astype(dtype)
        cos, sin = self.rotary(gh, gw)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.merger(x)


# ---------------------------------------------------------------------------
# M-RoPE positions for the text model
# ---------------------------------------------------------------------------


def image_positions(start: int, rows: int, cols: int) -> Tuple[List[int], List[int], List[int]]:
    """The (t, h, w) position of each of an image's rows*cols tokens, given the
    position `start` of its first token.

    Every token shares t = start; h and w count rows and columns from there. The
    text after the image resumes at start + max(rows, cols) -- NOT at
    start + rows*cols, which is why the rope position and the cache offset part
    ways after the first image (see Session.rope_delta).
    """
    t = [start] * (rows * cols)
    h = [start + r for r in range(rows) for _ in range(cols)]
    w = [start + c for _ in range(rows) for c in range(cols)]
    return t, h, w
