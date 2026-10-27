"""
Prepare the profile picture for a clean ASCII conversion.

make_ascii_svg.py maps DARK pixels to DENSE characters, and dense characters are
what look bright on the dark terminal. So the image handed to it must be the
subject in "ink" on a white background:

  1. cut the subject out with OpenCV GrabCut (no ML model to download)
  2. smooth texture, boost local contrast (CLAHE) inside the subject
  3. optionally invert (INVERT=1) so the lit parts become dense characters
  4. background -> white (= blank), centred square

    python scripts/prep_photo.py [input.png] [output.png]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

# ---- load, flatten transparency onto white, square-crop ------------------
im = Image.open(INP).convert("RGBA")
flat = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
rgb = np.array(flat)
side = min(rgb.shape[:2])
y0, x0 = (rgb.shape[0] - side) // 2, (rgb.shape[1] - side) // 2
rgb = rgb[y0:y0 + side, x0:x0 + side]
rgb = cv2.resize(rgb, (600, 600), interpolation=cv2.INTER_AREA if side > 600 else cv2.INTER_CUBIC)
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
h, w = bgr.shape[:2]

# ---- 1. subject mask -----------------------------------------------------
mask = np.zeros((h, w), np.uint8)
rect = (int(w * 0.08), int(h * 0.06), int(w * 0.84), int(h * 0.94))
bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(bgr, mask, rect, bgm, fgm, 8, cv2.GC_INIT_WITH_RECT)
subj = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
subj = cv2.morphologyEx(subj, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(subj)
if n > 1:  # keep the largest blob only
    subj = np.where(lab == 1 + np.argmax(st[1:, 4]), 255, 0).astype(np.uint8)
if subj.mean() < 255 * 0.05:  # segmentation failed -> keep everything
    subj[:] = 255
alpha = cv2.GaussianBlur(subj.astype(np.float32) / 255.0, (0, 0), 1.2)

# ---- 2. tone ------------------------------------------------------------
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
gray = cv2.bilateralFilter(gray, 7, 30, 7)
gray = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(6, 6)).apply(gray)
# unsharp mask: eyes, glasses and beard are thin features that vanish at ascii scale
gray = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 3), -0.6, 0)

inside = subj > 128
# INVERT=1 renders the lit parts as dense characters (photographic look);
# the default keeps the classic "negative" ascii look, which reads faces better
dark_photo = os.environ.get("INVERT") == "1"

# ---- 3. orient so the subject's lit parts end up dark (= dense chars) ----
ink = (255 - gray) if dark_photo else gray
lo, hi = np.percentile(ink[inside], [3, 97])
ink = np.clip((ink.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
ink = np.power(ink, 0.8) * 0.82   # keep the subject below make_ascii's blank threshold

# ---- 4. background -> white ---------------------------------------------
out = (ink * alpha + 1.0 * (1 - alpha)) * 255

# ---- 5. zoom on the subject (square box, small margin) --------------------
ys, xs = np.where(subj > 128)
if len(xs):
    side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.08)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    canvas = np.full((side, side), 255.0)
    x0, y0 = cx - side // 2, cy - side // 2
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = out[sy0:sy1, sx0:sx1]
    out = canvas
Image.fromarray(out.astype(np.uint8), mode="L").save(OUT)
print("wrote", OUT, out.shape, "inverted" if dark_photo else "negative")
