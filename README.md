# ComfyUI-PlateFit

Two ComfyUI nodes that put image-model results back onto a full-resolution VFX plate. Category `image/PlateFit`.

The maths is a trimmed copy of [AInVFX Paint a plate](https://github.com/AInVFX/ainvfx-paintaplate) (Apache-2.0, commit `63854ef`); see `NOTICE`. The nodes are rewritten in the classic node API, so they run on ComfyUI 0.42 and do not need that pack or its OpenEXR/OCIO dependencies. Dependencies: numpy, scipy and torch (all present in a ComfyUI venv).

## Fit to Plate (`PlateFit_FitToPlate`)
Prepares a model's edit for compositing over the plate. Feed `image` and `mask` to core **ImageCompositeMasked** (`destination` = plate, `source` = image, `mask` = mask, x = y = 0).

- Resizes `image` to the plate (bicubic). If the model changed the aspect ratio, the edit is stretched back.
- `align`: measures a sub-pixel shift between edit and plate outside the mask (phase correlation, robust to relighting), smooths it over time, removes it.
- `match_color`: measures the colour/exposure difference on a ring just outside the mask and spreads it smoothly inside, so there is no seam.
- `grain`: adds grain up to the plate's own level (measured once per shot).
- `grow`, `feather`, `steady`: the mask is grown, softened and steadied over `steady` frames on each side.
- `mask` empty: no edit; the mask output is all zeros.
- Batches: the plate decides the frame count. A shorter `image`/`mask` batch holds its last frame, so one edited still can be fitted over a sequence.

Where the output mask is 0 (outside grow + feather), the composite is the plate bit for bit.

## Relight Transfer (`PlateFit_RelightTransfer`)
Moves the new light of a relit model output onto the plate at full resolution. The node computes the *ratio* of light between `source` and `relit` at the model's size, smoothed edge-aware (brightness) and broadly (colour of light), and multiplies the plate by it. Texture, grain, sharpness and values above 1 stay the plate's.

- `source`: exactly the image the model was given. `relit`: the model's output; it is resized to `source` if the model rounded the size.
- `shot_space` (default `sRGB`): how `source`/`relit` are encoded. `plate_space` (default `linear`): how the plate and the result are encoded.
- `radius`: detail size kept from the plate, in pixels of `source`. Lower lets sharper new shadows through.
- `strength`: 0 = plate unchanged, 1 = full relight, up to 2.

## Colour
Both nodes work per channel, so plate and model images must share **primaries**. For ACEScg plates, load with `Load EXR (ACEScg)` set to *Convert ACEScg to Linear sRGB*, run the nodes in linear sRGB, and save with `Save EXR (ACEScg)` converting back. Feed the model an sRGB-encoded copy (core `ImageColorSpace` linear→sRGB) and use that same copy as Relight Transfer's `source`.

## Tests
```
<ComfyUI>/venv/Scripts/python.exe -m unittest discover -s tests -v
```
`tests/test_plate_math.py` is the adapted upstream maths suite; `tests/test_plate_nodes.py` checks the nodes on ComfyUI-shaped tensors (odd plate sizes, empty mask, short batches, HDR values, resized model output).

## Licence
Apache License 2.0 (`LICENSE`), with the attribution in `NOTICE`.
