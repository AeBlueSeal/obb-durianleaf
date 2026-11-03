# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SAHI tile dataset -- slice datasetv3_2 (train/val/test) for tile-native training
#
# This notebook does **no training or inference**. It slices `datasetv3_2`'s
# train/val/test images into overlapping tiles using the exact same tile
# geometry `notebooks/SAHI_E-IRFS_optuna_datasetv3.py` already uses for
# *inference* (`sahi.slicing.get_slice_bboxes`, 640x640, 20% overlap), so a
# model trained directly on these tiles sees the same crop geometry SAHI will
# hand it at inference time -- rather than training on full images and only
# meeting tile-sized crops for the first time during sliced inference.
#
# Output is a new, self-contained YOLO-OBB dataset
# (`train/{images,labels}`, `val/{images,labels}`, `test/{images,labels}`,
# `classes.txt`, `data.yaml`) that any of the OBB training notebooks in this
# repo can point `DATA_ROOT`/`DATA_YAML`/`IMAGES_DIR` at directly, unchanged.
#
# ## Splits are sliced independently -- no leakage
#
# Each source image's tiles all land in the SAME split its source image came
# from. An image is never sliced once and split across train/val/test -- that
# would leak near-identical crops (adjacent, overlapping tiles of the same
# leaf) across the split boundary, which is exactly the kind of duplicate-like
# leakage `results/source_confound_pipeline.md`'s Phase D fix already had to
# clean up once for whole images. Slicing happens per-split, after the
# existing train/val/test assignment, never before it.
#
# ## What "keeping" a truncated instance means here
#
# An instance is kept in a tile if the fraction of its ORIGINAL polygon area
# still visible inside that tile (`shapely` intersection) is >=
# `VISIBILITY_THRESHOLD`. If kept, its label is **re-fit**: the visible
# intersection region is bounded by a new minimum-area rotated rectangle
# (`cv2.minAreaRect`) in the tile's local frame -- not the original box
# clamped/shifted as-is. This keeps every written label a valid 4-point OBB
# that never extends outside its tile (so normalized coordinates always land
# in [0, 1]), at the cost of the box being tighter around the *visible*
# portion of a truncated object rather than the object's true full extent.
# That is the standard, expected trade-off for tile-native training data --
# state it if reporting results trained on this dataset vs. one trained on
# full images.
#
# ## Background tiles
#
# Tiles with no surviving instance are kept as background negatives (empty
# label file) by default, same neutral treatment E-IRFS already gives
# background images elsewhere in this project (weight 1.0, never boosted --
# see `src/obb_common.py`). `EMPTY_TILE_KEEP_RATIO` below lets you subsample
# them if the tile grid produces far more background than object tiles.

# %% [markdown]
# ## file setup

# %%
# !pip3 install -q sahi opencv-python-headless shapely pyyaml numpy pillow

# %%
from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# %%
# !unzip -q /content/drive/MyDrive/Krong\ Si\ Su\ Pap/datasetv3_2.zip -d /content

# %%
# !mv /content/datasetv3_2 /content/data

# %%
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from sahi.slicing import get_slice_bboxes
from shapely.geometry import Polygon, box

sys.path.insert(0, PROJECT_ROOT)
from src.obb_common import (  # noqa: E402
    collect_items, load_class_names, read_yolo_obb_label,
)

# --- Source / destination ---
SRC_ROOT = "/content/data"                    # datasetv3_2, unzipped+renamed above
OUT_ROOT = "/content/data_sahi_tiles"          # sliced dataset written here
CLASSES_TXT = f"{SRC_ROOT}/classes.txt"

# --- Tile geometry -- MUST match SAHI_E-IRFS_optuna_datasetv3.py's inference
# config exactly, or a model trained here sees a different crop scale than
# SAHI will hand it at inference time. ---
SLICE_HEIGHT         = 640
SLICE_WIDTH           = 640
OVERLAP_HEIGHT_RATIO  = 0.2
OVERLAP_WIDTH_RATIO   = 0.2

# --- Instance-retention rules ---
VISIBILITY_THRESHOLD = 0.3     # min fraction of an instance's ORIGINAL area
                                # that must remain visible in a tile to keep it
MIN_KEPT_AREA_PX     = 16.0    # drop degenerate slivers even if the ratio above passes

# --- Background tiles ---
# This dataset's source images are large (measured: avg ~4800x3700px, up to
# 8064x6048) relative to a 640x640 tile, so slicing produces roughly two
# orders of magnitude more tiles than source images, and the overwhelming
# majority are background -- a 25-image sample measured 98/2289 (~4.3%)
# object tiles. Keeping every background tile at full scale would produce
# roughly 130k tiles for the train split alone (extrapolated from that
# sample), almost all background, which is a different and much more severe
# imbalance than E-IRFS elsewhere in this project is designed to counteract
# (E-IRFS assumes background is a minority, not >90% of the data). Default
# keeps 1 in 10 background tiles; watch the per-split summary printed below
# and adjust before a real training run.
KEEP_EMPTY_TILES       = True
EMPTY_TILE_KEEP_RATIO  = 0.10

SEED = 42
IMG_EXTS = [".jpg", ".jpeg", ".png"]

random.seed(SEED)
rng = random.Random(SEED)

CLASS_NAMES = load_class_names(CLASSES_TXT)
NUM_CLASSES = len(CLASS_NAMES)
print(f"{NUM_CLASSES} classes: {CLASS_NAMES}")

# %% [markdown]
# ### Slicing core
#
# `refit_visible_polygon` is the one function that decides what survives a
# cut. It is deliberately tested against synthetic cases (fully inside, fully
# outside, half-cut, mostly-cut-below-threshold, rotated-fully-inside) before
# being trusted against real data -- see the test cell right after it.

# %%
def slice_tile_boxes(img_w: int, img_h: int) -> list:
    """Tile boxes covering (img_w, img_h) -- same call SAHI's own inference
    path uses, so tile geometry here matches tile geometry at eval time.
    `auto_slice_resolution=False` pins the tile size to the constants above
    instead of SAHI silently recomputing a different size per image.
    """
    return get_slice_bboxes(
        image_height=img_h,
        image_width=img_w,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        auto_slice_resolution=False,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
    )


def refit_visible_polygon(poly_px, tile_box, visibility_threshold=VISIBILITY_THRESHOLD,
                          min_kept_area_px=MIN_KEPT_AREA_PX):
    """Decide whether an instance survives a tile cut, and if so, re-fit it.

    Returns (normalized_8_values, visible_ratio) if kept, else (None, visible_ratio).

    ``poly_px`` is the instance's 8-value polygon in ORIGINAL-image absolute
    pixels. The visible portion (shapely intersection with the tile rectangle)
    is re-fit with `cv2.minAreaRect` rather than kept as the raw clipped
    polygon, because a clipped rotated rectangle is in general a >4-vertex
    polygon and Ultralytics' OBB label format is strictly 4 points.
    """
    x0, y0, x1, y1 = tile_box
    obj = Polygon(np.asarray(poly_px, dtype=np.float64).reshape(4, 2))
    if not obj.is_valid:
        obj = obj.buffer(0)   # resolve self-intersecting quads to their valid outline
    if obj.is_empty or obj.area <= 0:
        return None, 0.0

    inter = obj.intersection(box(x0, y0, x1, y1))
    if inter.is_empty:
        return None, 0.0

    visible_ratio = inter.area / obj.area
    if visible_ratio < visibility_threshold:
        return None, visible_ratio

    if inter.geom_type == "Polygon":
        coords = np.array(inter.exterior.coords[:-1], dtype=np.float32)
    else:
        # Rare: a degenerate/self-intersecting source polygon can produce a
        # MultiPolygon intersection. Take the largest piece rather than
        # dropping the instance outright.
        parts = [g for g in getattr(inter, "geoms", []) if g.geom_type == "Polygon"]
        if not parts:
            return None, visible_ratio
        coords = np.array(max(parts, key=lambda g: g.area).exterior.coords[:-1], dtype=np.float32)
    if len(coords) < 3:
        return None, visible_ratio

    (cx, cy), (w, h), angle = cv2.minAreaRect(coords)
    if w * h < min_kept_area_px:
        return None, visible_ratio

    pts = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float64)
    pts[:, 0] = (pts[:, 0] - x0) / (x1 - x0)
    pts[:, 1] = (pts[:, 1] - y0) / (y1 - y0)
    # minAreaRect is fit on the CLIPPED points, so it cannot extend past the
    # tile -- this clip only guards float rounding at the exact tile edge.
    pts = np.clip(pts, 0.0, 1.0)
    return pts.reshape(-1), visible_ratio

# %%
# Sanity-check refit_visible_polygon against synthetic cases before trusting
# it on real data. A silently-wrong label here would poison every downstream
# training run using this dataset.
_tile = (0, 0, 640, 640)

# Fully inside -> unchanged, ratio 1.0
_, r = refit_visible_polygon([100, 100, 200, 100, 200, 200, 100, 200], _tile)
assert abs(r - 1.0) < 1e-6, r

# Fully outside -> dropped
refit, r = refit_visible_polygon([1000, 1000, 1100, 1000, 1100, 1100, 1000, 1100], _tile)
assert refit is None and r == 0.0

# 40% visible (above 0.3 threshold) -> kept, coords land in [0, 1]
refit, r = refit_visible_polygon([600, 100, 700, 100, 700, 200, 600, 200], _tile)
assert refit is not None and 0.35 < r < 0.45
assert refit.min() >= 0.0 and refit.max() <= 1.0

# 20% visible (below 0.3 threshold) -> dropped
refit, r = refit_visible_polygon([620, 100, 720, 100, 720, 200, 620, 200], _tile)
assert refit is None and 0.15 < r < 0.25

# Rotated box fully inside -> ratio 1.0, angle preserved by construction
_rect = cv2.boxPoints(((300, 300), (80, 40), 30))
_, r = refit_visible_polygon(_rect.reshape(-1), _tile)
assert abs(r - 1.0) < 1e-6

print("refit_visible_polygon: all synthetic checks passed.")

# %% [markdown]
# ### Slice one image, one split

# %%
def slice_image(img_path: Path, label_path: Path, out_images_dir: Path, out_labels_dir: Path):
    """Slice one (image, label) pair into tiles under out_images_dir/out_labels_dir.

    Returns a dict of per-image stats (tiles written, tiles with >=1 kept
    instance, instances kept/dropped) for the summary printed after each split.
    """
    img_path = Path(img_path)
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        cls, polys = read_yolo_obb_label(label_path)
        polys = polys.copy()
        if len(polys):
            polys[:, 0::2] *= w
            polys[:, 1::2] *= h

        tiles = slice_tile_boxes(w, h)
        stats = {"tiles": 0, "object_tiles": 0, "instances_kept": 0, "instances_dropped": 0}

        for ti, (x0, y0, x1, y1) in enumerate(tiles):
            lines = []
            for c, poly in zip(cls, polys):
                refit, ratio = refit_visible_polygon(poly, (x0, y0, x1, y1))
                if refit is None:
                    if ratio > 0:   # only count instances that touched this tile at all
                        stats["instances_dropped"] += 1
                    continue
                lines.append(f"{int(c)} " + " ".join(f"{v:.6f}" for v in refit))
                stats["instances_kept"] += 1

            if not lines:
                if not KEEP_EMPTY_TILES:
                    continue
                if rng.random() > EMPTY_TILE_KEEP_RATIO:
                    continue
            else:
                stats["object_tiles"] += 1

            tile_stem = f"{img_path.stem}_tile{ti:03d}"
            im.crop((x0, y0, x1, y1)).save(out_images_dir / f"{tile_stem}{img_path.suffix.lower()}")
            (out_labels_dir / f"{tile_stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            stats["tiles"] += 1

    return stats


def slice_split(split: str):
    """Slice one split (train/val/test) end-to-end and print a summary.

    Reads from SRC_ROOT/<split>/{images,labels}, writes to
    OUT_ROOT/<split>/{images,labels}. Uses collect_items (src/obb_common.py)
    for the same image/label pairing used across every other notebook in this
    comparison, so a label with no matching image is skipped consistently
    rather than crashing this script differently from the others.
    """
    src_images = f"{SRC_ROOT}/{split}/images"
    src_labels = f"{SRC_ROOT}/{split}/labels"
    out_images = Path(OUT_ROOT) / split / "images"
    out_labels = Path(OUT_ROOT) / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    items = collect_items(src_images, src_labels)
    totals = {"images": 0, "tiles": 0, "object_tiles": 0, "instances_kept": 0, "instances_dropped": 0}
    for img_path, label_path in items:
        stats = slice_image(img_path, label_path, out_images, out_labels)
        totals["images"] += 1
        for k in ("tiles", "object_tiles", "instances_kept", "instances_dropped"):
            totals[k] += stats[k]

    print(f"[{split}] {totals['images']} source images -> {totals['tiles']} tiles "
          f"({totals['object_tiles']} with >=1 instance, "
          f"{totals['tiles'] - totals['object_tiles']} background-only)")
    print(f"[{split}] instances kept={totals['instances_kept']} "
          f"dropped-below-threshold={totals['instances_dropped']}")
    return totals

# %% [markdown]
# ## Slice all three splits
#
# Each split is sliced independently from its own source directory -- an
# image's tiles all stay in the split its source image was already assigned
# to (see the leakage note in the intro markdown).

# %%
split_totals = {split: slice_split(split) for split in ("train", "val", "test")}

# %% [markdown]
# ### Write classes.txt + data.yaml for the sliced dataset
#
# Same shape as `datasetv3_2/data.yaml` -- any training notebook can point
# `DATA_ROOT`/`DATA_YAML`/`IMAGES_DIR`/`LABELS_DIR` at `OUT_ROOT` unchanged.

# %%
shutil.copy(CLASSES_TXT, Path(OUT_ROOT) / "classes.txt")

data_yaml = {
    "path": OUT_ROOT,
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": NUM_CLASSES,
    "names": CLASS_NAMES,
}
with open(Path(OUT_ROOT) / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"Wrote {OUT_ROOT}/data.yaml and {OUT_ROOT}/classes.txt")
print(yaml.dump(data_yaml, sort_keys=False))

# %% [markdown]
# ### Visual sanity check
#
# Draws the re-fit boxes on a handful of train tiles before trusting this
# dataset for a real training run -- catches an off-by-one or axis-swap bug
# by eye faster than a full training run would.

# %%
import matplotlib.pyplot as plt

sample_labels = sorted((Path(OUT_ROOT) / "train" / "labels").glob("*.txt"))
sample_labels = [p for p in sample_labels if p.stat().st_size > 0][:6]  # prefer non-empty tiles

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, label_path in zip(axes.flat, sample_labels):
    stem = label_path.stem
    img_path = next(Path(OUT_ROOT, "train", "images").glob(f"{stem}.*"))
    im = np.array(Image.open(img_path).convert("RGB"))
    h, w = im.shape[:2]
    cls, polys = read_yolo_obb_label(label_path)
    polys = polys.copy()
    polys[:, 0::2] *= w
    polys[:, 1::2] *= h
    for c, poly in zip(cls, polys):
        pts = poly.reshape(-1, 1, 2).astype(int)
        cv2.polylines(im, [pts], True, (255, 0, 0), 2)
    ax.imshow(im)
    ax.set_title(stem, fontsize=8)
    ax.axis("off")
for ax in axes.flat[len(sample_labels):]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Archive to Drive
#
# Zips the sliced dataset and drops it directly into the same `Krong Si Su
# Pap` Drive folder the source `datasetv3_2.zip`/`classes.txt` already live
# in (see the bootstrap cell at the top) -- not `ProjZ/data/`, and not the
# unzipped directory tree. A zip is one file transfer over Drive's slow FUSE
# mount instead of ~tens of thousands of individual tile files, and it drops
# in as a peer of `datasetv3_2.zip` ready for the exact same
# `!unzip .../<name>.zip -d /content` bootstrap pattern used everywhere else
# in this repo.

# %%
def _unique_drive_path(base_path: Path) -> Path:
    """Avoid overwriting a stale previous archive: if base_path already
    exists on Drive, return the first base_path stem "2", "3", ... (before
    the extension) that doesn't."""
    if not base_path.exists():
        return base_path
    n = 2
    while True:
        candidate = base_path.with_name(f"{base_path.stem}{n}{base_path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


RUN_LABEL = "datasetv3_2_sahi_tiles"
DRIVE_FOLDER = Path("/content/drive/MyDrive/Krong Si Su Pap")

zip_path = shutil.make_archive(str(Path(f"/content/{RUN_LABEL}")), "zip", OUT_ROOT)

zip_drive_dest = _unique_drive_path(DRIVE_FOLDER / f"{RUN_LABEL}.zip")
if zip_drive_dest.name != f"{RUN_LABEL}.zip":
    print(f"NOTE: '{RUN_LABEL}.zip' already exists on Drive -- archiving to '{zip_drive_dest.name}' instead.")
shutil.copy(zip_path, zip_drive_dest)

print("Local  :", OUT_ROOT, "->", split_totals)
print("Drive  :", zip_drive_dest)
