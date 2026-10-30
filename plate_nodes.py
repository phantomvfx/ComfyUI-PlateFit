# Modified from nodes.py of AInVFX Paint a plate (https://github.com/AInVFX/ainvfx-paintaplate, commit 63854ef),
# Copyright AInVFX, Apache License 2.0 (see LICENSE and NOTICE). Changes: FitToPlate and RelightTransfer only,
# rewritten for the classic node API; batches shorter than the plate hold their last frame; relit is resized to source.
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from . import plate_math

try:
    from comfy.utils import ProgressBar
except ImportError:  # outside ComfyUI (tests)
    ProgressBar = None

CATEGORY = "image/PlateFit"
WORKERS = min(16, os.cpu_count() or 4)  # frames in parallel: numpy and scipy release the GIL


def progress(total):
    bar = ProgressBar(total) if ProgressBar is not None else None
    return (lambda: bar.update(1)) if bar is not None else (lambda: None)


def frame_np(images, i):
    # One [H, W, 3] float32 frame; batches shorter than the plate hold their last frame
    return images[min(i, images.shape[0] - 1), ..., :3].float().cpu().numpy()


def to_image(frames):
    return torch.from_numpy(np.ascontiguousarray(np.stack(frames), dtype=np.float32))


def resize(t, h, w):
    # [N, H, W(, C)] to [N, h, w(, C)], bicubic, antialiased when shrinking
    if tuple(t.shape[1:3]) == (h, w):
        return t
    x = t.unsqueeze(-1) if t.ndim == 3 else t
    x = F.interpolate(x.movedim(-1, 1).float(), size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    x = x.movedim(1, -1)
    return x[..., 0] if t.ndim == 3 else x


class FitToPlateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("IMAGE", {"tooltip": "The plate, in the same colour space and primaries as 'image' (e.g. both linear sRGB)."}),
                "image": ("IMAGE", {"tooltip": "Generated frames, any size; resized to the plate."}),
                "align": ("BOOLEAN", {"default": True, "tooltip": "Measure a small shift per frame, smooth it over time, remove it."}),
                "max_shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0, "step": 0.5,
                                        "tooltip": "Largest shift in plate pixels treated as misalignment."}),
                "match_color": ("BOOLEAN", {"default": True,
                                            "tooltip": "Match colour and exposure to the plate at the edge of the edit, spread smoothly inside it."}),
                "grain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                    "tooltip": "Add grain up to the plate's level. 0 adds none."}),
                "grow": ("INT", {"default": 8, "min": 0, "max": 256, "tooltip": "Grow the mask, in plate pixels."}),
                "feather": ("INT", {"default": 12, "min": 0, "max": 256, "tooltip": "Soften the mask edge, in plate pixels."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "The edited area, any size. Empty means no edit."}),
                "steady": ("INT", {"default": 2, "min": 0, "max": 12,
                                   "tooltip": "Frames on each side over which the mask edge, shift and colour match are smoothed."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "fit"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Prepares generated frames for compositing over the plate: resizes them to the plate, aligns them, "
                   "matches colour and grain at the edge of the edit, and returns the edit mask at plate size, grown, "
                   "softened and steady over time. Feed both outputs to ImageCompositeMasked (destination = plate).")

    def fit(self, plate, image, align, max_shift, match_color, grain, grow, feather, mask=None, steady=2):
        n, h, w = plate.shape[0], plate.shape[1], plate.shape[2]
        image = resize(image[:n], h, w)
        masks = np.zeros((1, 8, 8), np.float32) if mask is None else mask[:n].float().cpu().numpy()
        if masks.ndim == 2:
            masks = masks[None]
        tick = progress(2 * n)
        images, out = plate_math.fit_to_plate(
            n, h, w, lambda i: frame_np(plate, i), lambda i: frame_np(image, i), masks,
            align=align, max_shift=max_shift, match_color=match_color, grain=grain, grow=grow, feather=feather,
            steady=steady, workers=WORKERS, progress=tick)
        return (to_image(images), to_image(out))


class RelightTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("IMAGE", {"tooltip": "The plate, any size, linear light (or sRGB, see plate_space). Values above 1 are kept."}),
                "source": ("IMAGE", {"tooltip": "Exactly the image the model was given (e.g. the scaled sRGB frame). "
                                                "The light is measured at this size."}),
                "relit": ("IMAGE", {"tooltip": "The model's relit output; resized to 'source' if its size differs."}),
                "radius": ("INT", {"default": 8, "min": 1, "max": 256,
                                   "tooltip": "Detail size kept from the plate, in pixels of 'source'. Lower lets sharper new shadows through."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "align": ("BOOLEAN", {"default": True, "tooltip": "Remove a small shift of 'relit' first."}),
            },
            "optional": {
                "plate_space": (["linear", "sRGB"], {"default": "linear",
                                                     "tooltip": "How 'plate' and the result are encoded."}),
                "shot_space": (["sRGB", "linear"], {"default": "sRGB",
                                                    "tooltip": "How 'source' and 'relit' are encoded: sRGB as the model saw and made them."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "transfer"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Moves the new light of a model's relit image onto the full-resolution plate. Light, shadows and "
                   "colour of light come from the model; texture, grain, sharpness and values above 1 stay the plate's. "
                   "Plate, source and relit must share primaries (e.g. load ACEScg EXRs converted to linear sRGB).")

    def transfer(self, plate, source, relit, radius, strength, align, plate_space="linear", shot_space="sRGB"):
        n = plate.shape[0]
        relit = resize(relit, source.shape[1], source.shape[2])
        tick = progress(n)

        def one(i):
            out = plate_math.relight_frame(frame_np(plate, i), frame_np(source, i), frame_np(relit, i),
                                           radius, strength, align, plate_space, shot_space)
            tick()
            return out

        with ThreadPoolExecutor(WORKERS) as pool:
            out = list(pool.map(one, range(n)))
        return (to_image(out),)
