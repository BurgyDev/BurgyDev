"""
Prepare the profile picture for a clean ASCII conversion.

Lightweight on purpose (no background-removal model, so the GitHub Action stays
fast): grayscale -> smooth texture -> local contrast (CLAHE) -> tone stretch ->
darken line work -> centred square crop.

If the result looks muddy, replace source-photo.png with a portrait on a plain
light background and re-run the workflow with "regenerate portrait" ticked.

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

LINE_WEIGHT = 0.5

im = Image.open(INP).convert("RGBA")
# transparent pixels -> white, so logos / cut-outs keep a blank background
bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
gray = np.array(Image.alpha_composite(bg, im).convert("L"))

side = min(gray.shape)
y0, x0 = (gray.shape[0] - side) // 2, (gray.shape[1] - side) // 2
gray = gray[y0:y0 + side, x0:x0 + side]
if side < 600:
    gray = cv2.resize(gray, (600, 600), interpolation=cv2.INTER_CUBIC)

smooth = gray
for _ in range(2):
    smooth = cv2.bilateralFilter(smooth, 9, 40, 9)

smooth = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(smooth)

lo, hi = np.percentile(smooth, [2, 95])
tone = np.clip((smooth.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)

fine = cv2.GaussianBlur(smooth, (0, 0), 1.5).astype(np.float32)
coarse = cv2.GaussianBlur(smooth, (0, 0), 6).astype(np.float32)
lines = np.clip((coarse - fine) / 40.0, 0, 1)
out = np.clip(tone - LINE_WEIGHT * lines, 0, 1) * 255

Image.fromarray(out.astype(np.uint8), mode="L").save(OUT)
print("wrote", OUT, out.shape)
