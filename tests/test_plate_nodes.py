"""Tests for the PlateFit nodes on ComfyUI-shaped tensors ([B, H, W, C] images, [B, H, W] masks).

Run with the ComfyUI venv (stdlib unittest, no pytest needed):
    <ComfyUI>/venv/Scripts/python.exe -m unittest discover -s tests -v

COMFYUI_ROOT points at the ComfyUI checkout (default L:/ComfyUI), so comfy.utils.ProgressBar is the real one.
"""
import importlib.util
import os
import sys
import unittest

COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", "L:/ComfyUI")
PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFYUI_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

spec = importlib.util.spec_from_file_location("platefit", os.path.join(PACK_DIR, "__init__.py"),
                                              submodule_search_locations=[PACK_DIR])
pack = importlib.util.module_from_spec(spec)
sys.modules["platefit"] = pack
spec.loader.exec_module(pack)
nodes = sys.modules["platefit.plate_nodes"]

rng = np.random.default_rng(5)


def textured(h, w, scale=1.0):
    return (ndimage.gaussian_filter(rng.random((h, w, 3)), (2, 2, 0)) * scale + 0.1).astype(np.float32)


class PlateFitNodeTests(unittest.TestCase):
    def test_pack_registers_both_nodes(self):
        self.assertEqual(set(pack.NODE_CLASS_MAPPINGS), {"PlateFit_FitToPlate", "PlateFit_RelightTransfer"})
        for cls in pack.NODE_CLASS_MAPPINGS.values():
            types = cls.INPUT_TYPES()
            self.assertTrue(hasattr(cls, cls.FUNCTION))
            self.assertIn("required", types)

    def fit(self, plate, image, mask=None, **kw):
        args = dict(align=True, max_shift=8.0, match_color=True, grain=1.0, grow=4, feather=6)
        args.update(kw)
        return nodes.FitToPlateNode().fit(plate, image, mask=mask, **args)

    def test_fit_resizes_to_an_odd_plate_and_leaves_outside_untouched(self):
        h, w = 101, 157  # not a multiple of 16, like a cropped plate
        plate = torch.from_numpy(textured(h, w, 3.0)[None])  # HDR values above 1
        gen = torch.from_numpy(textured(64, 96)[None] * 1.2)  # model output at its own size
        mask = torch.zeros(1, 64, 96)
        mask[:, 20:40, 30:60] = 1
        image, out_mask = self.fit(plate, gen, mask)
        self.assertEqual(tuple(image.shape), (1, h, w, 3))
        self.assertEqual(tuple(out_mask.shape), (1, h, w))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(out_mask[0, :, :20].abs().max().item(), 0.0)  # far from the edit: mask is exactly 0
        comp = out_mask[..., None] * image + (1 - out_mask[..., None]) * plate
        self.assertTrue(torch.equal(comp[0, :, :20], plate[0, :, :20]))
        self.assertGreater(plate.max().item(), 1.0)

    def test_fit_without_mask_returns_an_empty_mask(self):
        plate = torch.from_numpy(textured(48, 64)[None])
        image, mask = self.fit(plate, plate.clone())
        self.assertEqual(mask.abs().max().item(), 0.0)
        self.assertEqual(tuple(image.shape), (1, 48, 64, 3))

    def test_fit_holds_a_single_edit_over_a_plate_sequence(self):
        plate = torch.from_numpy(np.stack([textured(48, 64) for _ in range(4)]))
        gen = plate[:1] * 1.1
        mask = torch.zeros(1, 48, 64)
        mask[:, 10:30, 10:40] = 1
        image, out_mask = self.fit(plate, gen, mask, steady=1)
        self.assertEqual(image.shape[0], 4)
        self.assertEqual(out_mask.shape[0], 4)

    def test_relight_keeps_plate_size_and_hdr(self):
        h, w = 120, 200
        plate = textured(h, w, 4.0)
        plate[10:20, 10:20] = 9.0  # a highlight well above 1
        source = ndimage.zoom(plate, (60 / h, 100 / w, 1), order=1)
        gain = np.linspace(0.5, 2.0, 100, dtype=np.float32)[None, :, None]
        relit = source * gain
        relit = ndimage.zoom(relit, (64 / 60, 104 / 100, 1), order=1)  # the model rounded to a multiple of 8
        out, = nodes.RelightTransferNode().transfer(
            torch.from_numpy(plate[None]), torch.from_numpy(source[None]), torch.from_numpy(relit[None]),
            radius=4, strength=1.0, align=False, plate_space="linear", shot_space="linear")
        self.assertEqual(tuple(out.shape), (1, h, w, 3))
        ratio = (out[0] / torch.from_numpy(plate)).numpy()
        self.assertLess(ratio[60, 20:40].mean(), 1.0)  # dimmed on the left
        self.assertGreater(ratio[60, 160:180].mean(), 1.4)  # brightened on the right
        self.assertGreater(out[0, 15, 15].min().item(), 4.0)  # highlight scaled, not clipped

    def test_relight_srgb_shot_matches_manual_decode(self):
        plate = textured(64, 96, 2.0)
        src_srgb = np.clip(nodes.plate_math.linear_to_srgb(plate[::2, ::2] / 2.0), 0, 1)
        relit_srgb = np.clip(src_srgb * np.array([1.1, 1.0, 0.9], np.float32), 0, 1)
        out, = nodes.RelightTransferNode().transfer(
            torch.from_numpy(plate[None]), torch.from_numpy(src_srgb[None]), torch.from_numpy(relit_srgb[None]),
            radius=4, strength=1.0, align=False)  # defaults: plate linear, shot sRGB
        ref = nodes.plate_math.relight_frame(plate, src_srgb, relit_srgb, 4, 1.0, False, "linear", "sRGB")
        np.testing.assert_allclose(out[0].numpy(), ref, atol=1e-6)

    def test_relight_batch_shorter_than_plate_holds_last_frame(self):
        plate = torch.from_numpy(np.stack([textured(32, 48) for _ in range(3)]))
        source = plate[:1, ::2, ::2].contiguous()
        out, = nodes.RelightTransferNode().transfer(plate, source, source * 1.5, radius=2, strength=1.0,
                                                    align=False, plate_space="linear", shot_space="linear")
        self.assertEqual(tuple(out.shape), (3, 32, 48, 3))


if __name__ == "__main__":
    unittest.main()
