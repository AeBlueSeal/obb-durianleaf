# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from ultralytics import YOLO
from pathlib import Path
import cv2, os

WEIGHTS   = '../../models/yolo11m-obb.pt'
SOURCE    = '../../test_images/'
THRESHOLD = 0.5

model   = YOLO(WEIGHTS)
results = model.predict(source=SOURCE, conf=THRESHOLD, save=False)

false_positives = []
false_negatives = []

for r in results:
    boxes = r.boxes
    if len(boxes) == 0:
        false_negatives.append(r.path)

# %%
import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, path in zip(axes.flat, false_negatives[:8]):
    img = Image.open(path)
    ax.imshow(img)
    ax.set_title(Path(path).name, fontsize=8)
    ax.axis('off')
plt.suptitle('False Negatives — model missed objects', fontsize=12)
plt.tight_layout()
plt.show()
