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
import matplotlib.pyplot as plt
from PIL import Image

WEIGHTS   = '../../models/yolo11m-obb.pt'
SOURCE    = '../../test_images/'
THRESHOLD = 0.25

model   = YOLO(WEIGHTS)
results = model.predict(source=SOURCE, conf=THRESHOLD, save=False)

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, r in zip(axes.flat, results[:8]):
    img = Image.fromarray(r.plot()[:, :, ::-1])
    ax.imshow(img)
    ax.set_title(Path(r.path).name, fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()
