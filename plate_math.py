"""The numpy maths behind Fit to Plate and Relight Transfer.

Frames are (H, W, C) float arrays and masks (H, W); stacks add a first axis. Nothing here imports
torch or ComfyUI, so every function is tested in plain Python (tests/test_plate_math.py).

Modified from mathops.py of AInVFX Paint a plate (https://github.com/AInVFX/ainvfx-paintaplate,
commit 63854ef), Copyright AInVFX, licensed under the Apache License 2.0 (see LICENSE and NOTICE).
Changes: kept only what fit_to_plate and relight_frame use (level_fill, follow_matte, hdr_match,
keyframe_video and acescct_preview removed), moved GRAIN_UNIT next to _grain_unit, and made
noise_sigma ignore empty (0) pixels so a CG plate on black no longer measures NaN grain.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage

MASK_SIDE = 480  # fit_to_plate works on masks and colour fields of about this long side, then enlarges them
ALIGN_SIDE = 512  # and measures alignment at about this one


def luma(img: np.ndarray) -> np.ndarray:
    """Rec. 709 luma. A weighted sum, not a matrix product: no BLAS threads inside worker threads."""
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def structure(img: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Log luma minus its blur: edges and texture, with the lighting removed.

    Two frames that differ only by their light give similar structure maps, so
    alignment still works after a relight.
    """
    log = np.log(np.maximum(luma(img), 0.0) + 1e-3)
    return log - ndimage.gaussian_filter(log, sigma)


def phase_shift(ref: np.ndarray, img: np.ndarray, max_shift: float) -> tuple[float, float]:
    """Sub-pixel (dy, dx) that moves `img` onto `ref` (2D float arrays)."""
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fa, fb = np.fft.rfft2(ref * win), np.fft.rfft2(img * win)
    cross = fa * np.conj(fb)
    corr = np.fft.irfft2(cross / (np.abs(cross) + 1e-9), s=(h, w))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    def refine(c_m, c_0, c_p):  # parabola through three samples around the peak
        d = c_m - 2 * c_0 + c_p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (c_m - c_p) / d

    dy = py + refine(corr[py - 1, px], corr[py, px], corr[(py + 1) % h, px])
    dx = px + refine(corr[py, px - 1], corr[py, px], corr[py, (px + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return 0.0, 0.0  # not a small misalignment: leave the frame alone
    return float(dy), float(dx)


def noise_sigma(img: np.ndarray) -> np.ndarray:
    """Per-channel grain level: robust spread of the fine detail in flat areas."""
    sig = np.zeros(3, np.float32)
    y = luma(img)
    flat = ndimage.gaussian_filter(np.abs(ndimage.sobel(y)), 2.0)
    # Only pixels with picture in them: on a CG plate with an empty (0) background the 30th percentile
    # is 0, `flat < 0` kept nothing and the median of nothing was NaN, which switched grain off.
    valid = np.abs(y) > 1e-6  # not `y > 0`: the zero-mean noise field of _grain_unit must keep all its pixels
    if valid.sum() < 64:
        return sig
    keep = valid & (flat <= np.percentile(flat[valid], 30))
    for c in range(3):
        detail = img[..., c] - ndimage.gaussian_filter(img[..., c], 1.0)
        v = detail[keep]
        sig[c] = 1.4826 * np.median(np.abs(v - np.median(v)))
    return sig


def add_grain_level(img: np.ndarray, level: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add soft grain of a fixed per-channel `level`, as measured by noise_sigma.

    The level is decided once per shot, so the amount of grain does not change from frame to frame.
    """
    if not np.any(level > 0):
        return img
    noise = ndimage.gaussian_filter(rng.standard_normal(img.shape[:2] + (3,)).astype(np.float32), (0.6, 0.6, 0))
    out = img.copy()
    out[..., :3] += noise * (level / GRAIN_UNIT)
    return out


def _grain_unit() -> float:
    """How noise_sigma sees unit grain from add_grain_level: measured once on a fixed noise field."""
    noise = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal((256, 256, 3)).astype(np.float32), (0.6, 0.6, 0))
    return float(np.median(noise_sigma(noise)))


GRAIN_UNIT = _grain_unit()


def shrink(img: np.ndarray, f: int) -> np.ndarray:
    """Box average by an integer factor `f` over the first two axes; the edges are cropped to a multiple of f."""
    if f <= 1:
        return img
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    x = img[:h, :w]
    return x.reshape(h // f, f, w // f, f, *x.shape[2:]).mean((1, 3))


def _axis_taps(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each output index along one axis: the two input indices to blend and the weight of the second."""
    src = np.clip((np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
    i0 = np.floor(src).astype(np.intp)
    return i0, np.minimum(i0 + 1, n_in - 1), (src - i0).astype(np.float32)


def resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resample the first two axes to exactly (h, w): separable bilinear, pixel centres aligned, edges clamped.

    The same result as scipy's zoom (order 1, grid_mode) and torch's bilinear interpolate without align_corners,
    in a quarter of the time: two gathers and two blends in numpy, no spline machinery. Enlarging the masks and
    colour fields to full size was the composite's largest cost.
    """
    if img.shape[:2] == (h, w):
        return img
    x = img.astype(np.float32, copy=False)
    if x.shape[0] != h:
        i0, i1, wy = _axis_taps(x.shape[0], h)
        wy = wy.reshape((-1,) + (1,) * (x.ndim - 1))
        x = x[i0] * (1.0 - wy) + x[i1] * wy
    if x.shape[1] != w:
        j0, j1, wx = _axis_taps(x.shape[1], w)
        wx = wx.reshape((1, -1) + (1,) * (x.ndim - 2))
        x = x[:, j0] * (1.0 - wx) + x[:, j1] * wx
    return x


def push_pull(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fill a (H, W, C) field where `weight` (H, W, 0 to 1) is zero, from the values around it.

    Weighted averages at coarser and coarser scales are blended back where the weight
    is low, so a hole takes a smooth interpolation of its border.
    """
    levels = []
    v, w = values * weight[..., None], weight.astype(np.float32)
    while min(v.shape[:2]) > 1:
        levels.append((v, w))
        ph, pw = v.shape[0] % 2, v.shape[1] % 2
        if ph or pw:
            v = np.pad(v, ((0, ph), (0, pw), (0, 0)), mode="edge")
            w = np.pad(w, ((0, ph), (0, pw)), mode="edge")
        v = v.reshape(v.shape[0] // 2, 2, v.shape[1] // 2, 2, -1).sum((1, 3))
        w = w.reshape(w.shape[0] // 2, 2, w.shape[1] // 2, 2).sum((1, 3))
    fill = v / np.maximum(w, 1e-6)[..., None]
    for lv, lw in reversed(levels):
        up = resize_to(fill, lv.shape[0], lv.shape[1])
        a = np.clip(lw, 0.0, 1.0)[..., None]
        fill = lv / np.maximum(lw, 1e-6)[..., None] * a + up * (1.0 - a)
    return fill



def steady_masks(masks: np.ndarray, radius: int, grow: float, feather: float) -> np.ndarray:
    """Masks (N, H, W) whose edge moves smoothly from frame to frame.

    Each frame takes the widest coverage of its neighbours within `radius` frames, grows by
    `grow` pixels, is averaged over the same neighbours, then softened by `feather` pixels.
    A tracker's edge that changes every frame becomes one steady, soft edge.
    """
    m = np.clip(masks.astype(np.float32), 0.0, 1.0)
    span = 2 * radius + 1
    if radius > 0 and len(m) > 1:
        m = ndimage.maximum_filter1d(m, span, axis=0, mode="nearest")
    if grow > 0:
        m = dilate(m, grow)
    if radius > 0 and len(m) > 1:
        m = ndimage.uniform_filter1d(m, span, axis=0, mode="nearest")
    if feather > 0:
        m = ndimage.gaussian_filter(m, (0, feather / 2.0, feather / 2.0))
    return np.clip(m, 0.0, 1.0)


def dilate(m: np.ndarray, grow: float) -> np.ndarray:
    """Grow a mask (or a stack of masks) by `grow` pixels: two running maxima, along rows then columns.

    As fast at 60 px as at 6; the square corners it leaves are rounded by the feather that follows.
    """
    size = 2 * int(np.ceil(grow)) + 1
    return ndimage.maximum_filter1d(ndimage.maximum_filter1d(m, size, axis=-1, mode="nearest"), size, axis=-2, mode="nearest")


def smooth_track(track: np.ndarray, radius: int) -> np.ndarray:
    """A per-frame measurement (N, K) made steady: the median over 2r+1 frames, then their mean."""
    track = np.asarray(track, np.float64)
    if radius <= 0 or len(track) < 3:
        return track
    t = ndimage.median_filter(track, size=(2 * radius + 1, 1), mode="nearest")
    return ndimage.uniform_filter1d(t, 2 * radius + 1, axis=0, mode="nearest")


def seam_field(plate: np.ndarray, gen: np.ndarray, mask: np.ndarray, band: float) -> np.ndarray:
    """Log ratio (H, W, 3) that brings `gen` to the colour and exposure of `plate` at the edge of `mask`.

    It is measured on a ring just outside the mask, where the composite shows the plate, and
    spread smoothly inside it, as a Poisson blend would: colour and exposure drift are fixed,
    texture is left alone. Work at low resolution: the field is smooth by design.
    """
    outside = mask < 0.02
    if outside.all() or not outside.any():
        return np.zeros(plate.shape[:2] + (3,), np.float32)
    ring = outside & (ndimage.distance_transform_edt(outside) <= band)
    if ring.sum() < 16:
        return np.zeros(plate.shape[:2] + (3,), np.float32)
    d = np.log(np.maximum(plate[..., :3], 1e-4)) - np.log(np.maximum(gen[..., :3], 1e-4))
    lo, hi = np.percentile(d[ring], [10, 90], axis=0)  # another object crossing the ring must not pull the field
    field = push_pull(np.clip(d, lo, hi).astype(np.float32), ring.astype(np.float32))
    field = ndimage.gaussian_filter(field, (band / 2.0, band / 2.0, 0))
    return np.clip(field, -0.4, 0.4).astype(np.float32)


def fit_to_plate(n, h, w, plate_at, gen_at, masks, *, align=True, max_shift=8.0, match_color=True, grain=1.0,
                 grow=8, feather=12, steady=2, workers=8, progress=None):
    """Fit `n` generated frames to the plate they are composited onto. Returns (images, masks), full size.

    `plate_at(i)` and `gen_at(i)` give frame i as (h, w, C) float arrays of the same size; `masks` is
    (N or 1, any height, any width). The mask edge, the shift and the colour match are smoothed over
    `steady` frames on each side, so nothing in the composite changes from one frame to the next unless
    the shot does. Masks and colour fields are computed at about 480 px, alignment at about 512 px.
    """
    tick = progress or (lambda: None)
    lo = max(1, round(max(h, w) / MASK_SIDE))
    al = max(1, round(max(h, w) / ALIGN_SIDE))
    hl, wl = max(1, h // lo), max(1, w // lo)
    masks = np.asarray(masks, np.float32)[:n]
    if len(masks) < n:  # a shorter matte holds its last frame
        masks = np.concatenate([masks, np.repeat(masks[-1:], n - len(masks), 0)])
    small = np.stack([resize_to(m, hl, wl) for m in masks])
    small = steady_masks(small, steady, grow / lo, feather / lo)
    band = max(4.0, (grow + feather) / lo)
    ha, wa = max(1, h // al), max(1, w // al)

    def measure(i):  # pass 1: the shift and the colour field of one frame, at low resolution
        p, g = plate_at(i), gen_at(i)
        dy = dx = 0.0
        if align:
            keep = 1.0 - resize_to(small[i], ha, wa)
            dy, dx = phase_shift(structure(shrink(p, al)) * keep, structure(shrink(g, al)) * keep, max_shift / al)
            dy, dx = dy * al, dx * al
        field = None
        if match_color and small[i].max() > 0.02:
            gl = shrink(g, lo)
            if dy or dx:
                gl = ndimage.shift(gl, (dy / lo, dx / lo, 0), order=1, mode="nearest")
            field = seam_field(shrink(p, lo), gl, small[i], band)
        tick()
        return (dy, dx), field

    with ThreadPoolExecutor(workers) as pool:  # numpy and scipy release the GIL
        measured = list(pool.map(measure, range(n)))
    shifts = smooth_track(np.array([s for s, _ in measured]), steady + 1)
    fields = None
    if any(f is not None for _, f in measured):
        stack = np.stack([np.zeros((hl, wl, 3), np.float32) if f is None else f for _, f in measured])
        fields = ndimage.uniform_filter1d(stack, 2 * steady + 1, axis=0, mode="nearest") if steady else stack
    level = np.zeros(3, np.float32)
    if grain > 0:  # one grain level for the whole shot: the plate's grain minus what the generation already has
        picks = list(range(0, n, max(1, n // 3)))[:3]
        target = np.median([noise_sigma(plate_at(i)) for i in picks], 0)
        own = np.median([noise_sigma(gen_at(i)) for i in picks], 0)
        level = np.sqrt(np.maximum(target ** 2 - own ** 2, 0.0)) * grain

    def apply(i):  # pass 2: full size, only around the edit
        g = gen_at(i)
        dy, dx = shifts[i]
        if abs(dy) > 0.05 or abs(dx) > 0.05:
            g = ndimage.shift(g, (dy, dx, 0), order=1, mode="nearest")
        m = resize_to(small[i], h, w)
        ys, xs = np.nonzero(m > 1.0 / 1024)
        g = np.array(g, np.float32, copy=True)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            if fields is not None:  # the colour field, enlarged like the mask
                g[y0:y1, x0:x1, :3] *= np.exp(resize_to(fields[i], h, w)[y0:y1, x0:x1])
            if np.any(level > 0):
                g[y0:y1, x0:x1] = add_grain_level(g[y0:y1, x0:x1], level, np.random.default_rng(i))
        tick()
        return g, m.astype(np.float32)

    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(apply, range(n)))
    return [d[0] for d in done], [d[1] for d in done]




def box(x: np.ndarray, r: int) -> np.ndarray:
    return ndimage.uniform_filter(x, size=(2 * r + 1, 2 * r + 1) + (1,) * (x.ndim - 2), mode="reflect")


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-aware smoothing of `src` (H, W, C) following a gray `guide` (H, W)."""
    g = guide[..., None]
    mean_g, mean_s = box(g, r), box(src, r)
    cov = box(g * src, r) - mean_g * mean_s
    var = box(g * g, r) - mean_g**2
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return box(a, r) * g + box(b, r)


# Light may change by 4 stops down and 3 up: night needs about that. At 0.25 (2 stops) a night relight stayed in dusk.
RATIO_RANGE = (1.0 / 16.0, 8.0)
# A channel holding a small share of the luma (the blue of grass, of a red roof) is too dark to divide by on its own: below
# this share of the luma, its ratio leans on the luma's. Divided alone, the blue of the mossy cliff in the lighthouse plate
# swung between the limits of RATIO_RANGE from frame to frame, and drew saturated, flickering specks (28 Oct 2026).
CHANNEL_FLOOR = 0.1
# The colour of the new light changes slowly across the frame; its brightness may have sharp edges (new shadows). The colour
# is smoothed over this many radii, so the regenerated shot's colour noise never reaches single pixels of the plate.
TINT_SPREAD = 4.0


def light_ratio(source: np.ndarray, relit: np.ndarray, radius: int) -> np.ndarray:
    """The change of light from `source` to `relit` (H, W, 3), both linear light at the same size.

    Brightness: the ratio of the two lumas, smoothed by an edge-aware filter that follows `source`, so a new shadow keeps
    its edge. Colour: each channel's ratio over that brightness, with every channel floored at CHANNEL_FLOOR of the luma
    and smoothed over TINT_SPREAD radii, so a dark channel cannot swing the colour of a pixel.
    """
    guide = np.log(luma(source).clip(0) + 1e-3)
    eps = 1e-2 * max(guide.var(), 1e-6)  # smooth enough that fine detail never enters the ratio
    lit = guided_filter(guide, relit[..., :3].clip(0), radius, eps).clip(0)
    was = guided_filter(guide, source[..., :3].clip(0), radius, eps).clip(0)
    y_lit, y_was = luma(lit)[..., None], luma(was)[..., None]
    bright = (y_lit + 1e-3) / (y_was + 1e-3)
    tint = np.log((lit + CHANNEL_FLOOR * y_lit + 1e-3) / (was + CHANNEL_FLOOR * y_was + 1e-3) / bright)
    f = max(1, round(TINT_SPREAD * radius / 4))  # the colour is smooth: blur it at a fraction of the size
    sigma = TINT_SPREAD * radius / f
    tint = resize_to(ndimage.gaussian_filter(shrink(tint, f), (sigma, sigma, 0), mode="nearest"), *tint.shape[:2])
    ratio = ndimage.median_filter(bright * np.exp(tint), size=(3, 3, 1))  # isolated specks at sharp edges go
    return ndimage.gaussian_filter(np.clip(ratio, *RATIO_RANGE), (1.0, 1.0, 0)).astype(np.float32)


def relight(plate: np.ndarray, source: np.ndarray, relit: np.ndarray,
            radius: int, strength: float = 1.0) -> np.ndarray:
    """Move the light of `relit` onto `plate`, keeping the plate's own detail.

    All three images are linear light. `source` and `relit` share the working size; `plate` can be larger. The change
    of light (`light_ratio`) multiplies the plate, so its texture, grain and values above 1.0 stay its own.
    """
    ratio = light_ratio(source, relit, radius) ** strength
    if ratio.shape[:2] != plate.shape[:2]:
        ratio = resize_to(ratio, plate.shape[0], plate.shape[1])
    out = plate.copy()
    out[..., :3] = plate[..., :3] * ratio
    return out


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """The sRGB transfer function, decoded (IEC 61966-2-1): display values to linear light."""
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.04045) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """The sRGB transfer function, encoded: linear light to display values. Values above 1.0 stay above 1.0."""
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.maximum(x, 0.0031308) ** (1 / 2.4) - 0.055).astype(np.float32)


def relight_frame(plate: np.ndarray, source: np.ndarray, relit: np.ndarray, radius: int, strength: float = 1.0,
                  align: bool = True, plate_space: str = "linear", shot_space: str = "linear") -> np.ndarray:
    """One frame of Relight Transfer: `relit` aligned to `source`, then its light moved onto `plate`.

    The ratio of light is computed at the size of `source` and `relit`, then enlarged to the plate. Light
    is smooth, so the workflows give them at half the plate's size: converting them to linear light then
    costs a quarter. `radius` is in pixels of that size; a shift counts as misalignment up to 8 plate pixels.
    """
    if shot_space == "sRGB":  # source and relit as the model saw and made them
        source, relit = srgb_to_linear(source[..., :3]), srgb_to_linear(relit[..., :3])
    if plate_space == "sRGB":  # decoded here, frame by frame, then encoded back: far cheaper than core at full size
        plate = np.concatenate([srgb_to_linear(plate[..., :3]), plate[..., 3:]], -1)
    if align:
        dy, dx = phase_shift(structure(source), structure(relit), 8.0 * source.shape[1] / plate.shape[1])
        if dy or dx:
            relit = ndimage.shift(relit, (dy, dx, 0), order=1, mode="nearest")
    out = relight(plate, source, relit, max(1, radius), strength).astype(np.float32)
    if plate_space == "sRGB":
        out[..., :3] = linear_to_srgb(out[..., :3])
    return out
