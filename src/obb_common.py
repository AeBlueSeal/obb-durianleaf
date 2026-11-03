"""Framework-agnostic pieces shared by every OBB comparison notebook.

The four models under comparison (YOLO11m-OBB, Strip R-CNN, Oriented-DETR,
RF-DETR-OBB) live in four different training frameworks. Anything that is
*not* the model itself must therefore be identical across them, or the
comparison measures framework plumbing rather than architecture. This module
is that shared layer:

  * ``make_folds`` .......... identical K-fold assignment for all four models
  * ``compute_eirfs`` ....... the E-IRFS class-imbalance weights, one
                              implementation, no per-framework re-derivation
  * ``yolo_obb_to_dota`` .... label conversion for the MMRotate / RF-DETR side
  * ``yolo_obb_to_coco`` .... label conversion for the Oriented-DETR side

The E-IRFS math here is a straight port of ``src/data_loader.py``'s
``YOLOEIRFSCappedDataset`` with the Ultralytics coupling removed -- same
alpha, same threshold, same cap rule, same "background negatives get a
neutral 1.0" behaviour. See results/source_confound_pipeline.md Phase 5.5 and
results/source_confound_scoping.md for why the cap exists at all.

Deliberately depends on nothing heavier than numpy + PIL so it can be
imported inside all four framework environments, several of which pin
mutually incompatible torch/mmcv versions.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm ships in every env this runs
    # in (Colab, all four training envs), but this module promises to stay
    # importable on numpy + PIL alone -- degrade to a no-op iterator rather
    # than add a hard dependency.
    def tqdm(iterable, **kwargs):
        return iterable

# Kept in sync with src/data_loader.py -- if these drift, the comparison is
# no longer holding the sampler constant across models.
EIRFS_ALPHA = 2.0
EIRFS_THRESHOLD = 0.0001
CAP_MULTIPLIER = 1.5
DEFAULT_CAPPED_CLASS_NAMES = {"Rhizoctonia"}

IMG_EXTS = [".jpg", ".jpeg", ".png"]


# --------------------------------------------------------------------------
# Dataset discovery + folds
# --------------------------------------------------------------------------
def find_image_for_label(label_path, images_dir):
    """Locate the image matching a label file, trying each known extension.

    Checks both the extension's usual case and its all-caps variant (e.g.
    ".jpg" and ".JPG") -- on a case-INsensitive filesystem (default macOS)
    the exact-case check alone already matches either, which is why this bug
    went unnoticed locally; on a case-sensitive one (Colab/Linux, where
    training actually runs) a lowercase-only check silently drops every
    upper-case-extension image, confirmed on this dataset to be ~27% of
    every split (datasetv3_2 mixes ".jpg" and ".JPG" filenames).
    """
    stem = Path(label_path).stem
    for ext in IMG_EXTS:
        for candidate_ext in (ext, ext.upper()):
            p = Path(images_dir) / f"{stem}{candidate_ext}"
            if p.exists():
                return p.resolve()
    return None


def collect_items(images_dir, labels_dir):
    """Collect (image, label) pairs. Labels without a matching image are dropped.

    Sorted by label filename so the ordering -- and therefore the fold
    assignment below -- is reproducible across machines and frameworks.
    """
    items = []
    for lf in sorted(Path(labels_dir).glob("*.txt")):
        img = find_image_for_label(lf, images_dir)
        if img is not None:
            items.append((str(img), str(lf.resolve())))
    return items


def kfold_assignments(n_items, k, seed, shuffle=True):
    """Round-robin fold assignment over a seeded shuffle.

    Reseeds ``random`` immediately before shuffling rather than relying on a
    module-level ``random.seed(SEED)`` set far earlier (which is what
    Template_Clean_E-IRFS_optuna_datasetv3.py does). Same result for that
    notebook, but it makes the fold assignment independent of how much other
    ``random`` consumption happened first -- which is the only way the four
    notebooks can be guaranteed to train and validate on byte-identical folds.
    """
    idxs = list(range(n_items))
    if shuffle:
        random.seed(seed)
        random.shuffle(idxs)
    fold_of = [None] * n_items
    for r, idx in enumerate(idxs):
        fold_of[idx] = r % k
    return fold_of


def make_folds(items, k, seed, shuffle=True):
    """Return a list of K dicts, each with the train/val item lists for that fold.

    Framework-neutral on purpose: each notebook materialises these into
    whatever on-disk form its trainer wants (YOLO txt lists, DOTA split dirs,
    COCO jsons) from the *same* index assignment.
    """
    fold_of = kfold_assignments(len(items), k, seed, shuffle)
    folds = []
    for f in range(k):
        train = [items[i] for i in range(len(items)) if fold_of[i] != f]
        val = [items[i] for i in range(len(items)) if fold_of[i] == f]
        if not val and train:
            val.append(train.pop())
        folds.append({"fold": f, "train": train, "val": val})
    return folds


# --------------------------------------------------------------------------
# Label IO
# --------------------------------------------------------------------------
def read_yolo_obb_label(label_path):
    """Read a YOLO-OBB label file -> (cls int array (n,), poly float array (n, 8)).

    Coordinates stay normalised to [0, 1] here; callers scale by image size.
    Malformed lines (fewer than 9 fields) are skipped rather than raising, so
    one bad row cannot take down a 3-fold x N-trial search hours in.
    """
    cls, polys = [], []
    p = Path(label_path)
    if not p.exists():
        return np.zeros(0, dtype=int), np.zeros((0, 8), dtype=np.float32)
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        cls.append(int(float(parts[0])))
        polys.append([float(v) for v in parts[1:9]])
    if not cls:
        return np.zeros(0, dtype=int), np.zeros((0, 8), dtype=np.float32)
    return np.asarray(cls, dtype=int), np.asarray(polys, dtype=np.float32)


def load_class_names(classes_txt):
    return [ln.strip() for ln in Path(classes_txt).read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------
# E-IRFS
# --------------------------------------------------------------------------
class EIRFSWeights:
    """E-IRFS per-image sampling weights, computed once, framework-free.

    Mirrors src/data_loader.py exactly:

        f_ic = image_counts   / sum(image_counts)
        f_bc = instance_counts / sum(instance_counts)
        w_c  = exp(alpha * sqrt(t / sqrt(f_ic * f_bc)))

    then capped classes get ``w_c`` clipped to ``cap_multiplier x median(w
    over uncapped classes)``, and each image takes the max weight over the
    classes it contains (background-only images: a neutral 1.0, never capped
    or boosted).

    ``instance_counts`` starts at ones rather than zeros -- same as the
    original -- which acts as a Laplace-ish guard against a divide-by-zero for
    a class absent from a given fold. ``image_counts`` is floored at 1 for the
    same reason.
    """

    def __init__(self, label_paths, num_classes, class_names,
                 capped_class_names=DEFAULT_CAPPED_CLASS_NAMES,
                 alpha=EIRFS_ALPHA, t=EIRFS_THRESHOLD,
                 cap_multiplier=CAP_MULTIPLIER, verbose=True):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.capped_class_names = set(capped_class_names or ())
        self.alpha, self.t, self.cap_multiplier = alpha, t, cap_multiplier
        # Kept so reorder_for() can rebuild the stem->weight mapping below --
        # self.weights is positional in THIS order, which is only valid for a
        # dataset that enumerates images in this same order (see reorder_for).
        self.label_paths = list(label_paths)

        per_image_cls = [read_yolo_obb_label(p)[0] for p in label_paths]

        self.instance_counts = np.ones(num_classes, dtype=np.float32)
        seen = [set() for _ in range(num_classes)]
        for i, cls_ids in enumerate(per_image_cls):
            for c in np.unique(cls_ids):
                seen[int(c)].add(i)
            for c in cls_ids:
                self.instance_counts[int(c)] += 1
        self.image_counts = np.array(
            [len(s) if len(s) > 0 else 1 for s in seen], dtype=np.float32
        )

        f_ic = self.image_counts / np.sum(self.image_counts)
        f_bc = self.instance_counts / np.sum(self.instance_counts)
        self.class_weights = np.exp(self.alpha * np.sqrt(self.t / np.sqrt(f_ic * f_bc)))

        self._cap()

        self.weights = np.array([
            np.max(self.class_weights[c.astype(int)]) if c.size else 1.0
            for c in per_image_cls
        ], dtype=np.float64)
        total = self.weights.sum()
        self.probabilities = (
            self.weights / total if total > 0
            else np.ones_like(self.weights) / max(len(self.weights), 1)
        )
        self.cum_probabilities = np.cumsum(self.probabilities)
        self.n_background = int(sum(1 for c in per_image_cls if not c.size))

        if verbose:
            self.print_stats()

    def _cap(self):
        capped_ids = [i for i, n in enumerate(self.class_names) if n in self.capped_class_names]
        uncapped_ids = [i for i in range(self.num_classes) if i not in capped_ids]
        if self.capped_class_names and not capped_ids:
            print(f"WARNING: E-IRFS cap requested for {sorted(self.capped_class_names)} but none "
                  f"matched this dataset's classes ({self.class_names}) -- cap NOT applied.")
        if not capped_ids or not uncapped_ids:
            return
        cap_value = self.cap_multiplier * float(np.median(self.class_weights[uncapped_ids]))
        for cid in capped_ids:
            if self.class_weights[cid] > cap_value:
                print(f"E-IRFS cap: {self.class_names[cid]} weight "
                      f"{self.class_weights[cid]:.3f} -> {cap_value:.3f}")
                self.class_weights[cid] = cap_value

    def print_stats(self):
        print("\n=== E-IRFS Class Statistics (capped) ===")
        for cid, name in enumerate(self.class_names):
            tag = " [CAPPED]" if name in self.capped_class_names else ""
            print(f"Class {cid:2d} ({name:15}): {int(self.image_counts[cid]):5d} images, "
                  f"{int(self.instance_counts[cid]):6d} instances, "
                  f"weight={self.class_weights[cid]:.3f}{tag}")
        print(f"Background-only negatives: {self.n_background} images, "
              f"sampling weight fixed at 1.0 (neutral)")
        print("=========================================\n")

    def sample_index(self):
        """Draw one index under the E-IRFS distribution.

        Uses the precomputed CDF + searchsorted rather than
        ``np.random.choice(p=...)``, which rebuilds that same cumsum on every
        single call. Identical semantics to the sampling in
        src/data_loader.py's ``__getitem__``, including the tail guard against
        float rounding pushing the index past the end.
        """
        if not len(self.cum_probabilities):
            return 0
        i = int(np.searchsorted(self.cum_probabilities, np.random.random(), side="right"))
        return min(i, len(self.cum_probabilities) - 1)

    def reorder_for(self, dataset_stems):
        """Return a copy with the per-image sampling distribution reordered
        to match ``dataset_stems`` -- the ACTUAL per-index image identity of
        the dataset ``attach_eirfs`` will be applied to.

        Needed because ``self.weights[i]`` is positional in the order
        ``label_paths`` was given in (``collect_items``'s alphabetically
        sorted glob), but not every framework's dataset class enumerates
        images in that same order. MMRotate's ``DOTADataset.load_annotations``
        is a confirmed case: it builds its image list via an unsorted
        ``glob.glob()``, so attaching weights without reordering silently
        applies each image's E-IRFS weight to a *different* image than
        intended. Matching is by filename stem, which survives even where an
        extension gets rewritten (e.g. MMRotate's own hardcoded ``.png``).
        """
        stem_to_weight = {Path(p).stem: w for p, w in zip(self.label_paths, self.weights)}
        missing = [s for s in dataset_stems if s not in stem_to_weight]
        if missing:
            raise KeyError(
                f"{len(missing)} dataset image(s) have no matching EIRFSWeights "
                f"entry (e.g. {missing[:3]}) -- label_paths and dataset_stems "
                f"must cover the same image set."
            )
        new = copy.copy(self)
        new.weights = np.array([stem_to_weight[s] for s in dataset_stems], dtype=np.float64)
        total = new.weights.sum()
        new.probabilities = (
            new.weights / total if total > 0
            else np.ones_like(new.weights) / max(len(new.weights), 1)
        )
        new.cum_probabilities = np.cumsum(new.probabilities)
        return new


def attach_eirfs(dataset, weights, train_mode=True):
    """Monkeypatch ``dataset.__getitem__`` to draw indices from ``weights``.

    Every framework in this comparison gets E-IRFS through this exact
    mechanism -- index substitution inside ``__getitem__``, sampling with
    replacement, epoch length unchanged -- which is what
    src/data_loader.py already does for Ultralytics. The alternative
    (a framework-native weighted Sampler in each of MMRotate / DETR /
    RF-DETR) would introduce three different interactions with distributed
    sampling, drop_last, and epoch boundaries, and the comparison would no
    longer be holding the sampler constant.

    Only this *instance* is affected, so a val/test dataset built from the
    same class is untouched -- but that is done by re-pointing
    ``dataset.__class__`` at a throwaway subclass, NOT by assigning
    ``dataset.__getitem__``. Assigning the instance attribute looks like it
    works (``dataset.__getitem__(i)`` does go through it) but is silently
    INERT under any DataLoader: Python resolves the ``dataset[i]`` subscript
    operator on ``type(dataset)`` and never consults the instance ``__dict__``
    (verified directly), and PyTorch's ``_MapDatasetFetcher.fetch`` batches
    with ``self.dataset[idx]``. That made E-IRFS a no-op for every arm routed
    through this function while the YOLO arms -- which get E-IRFS from a real
    ``YOLODataset`` *subclass* (see the notebooks' ``YOLOEIRFSCappedDataset``)
    -- had it genuinely active, i.e. the sampler was not actually being held
    constant across the comparison. A per-instance subclass keeps the
    original narrow blast radius and makes the override reachable from
    subscript.
    """
    if not train_mode:
        return dataset

    base = type(dataset)
    original_getitem = base.__getitem__

    def eirfs_getitem(self, index, _orig=original_getitem, _w=weights):
        return _orig(self, _w.sample_index())

    dataset.__class__ = type(f"EIRFS{base.__name__}", (base,),
                             {"__getitem__": eirfs_getitem})
    dataset._eirfs_weights = weights
    return dataset


# --------------------------------------------------------------------------
# Format conversion
# --------------------------------------------------------------------------
def image_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


def build_png_cache(items, cache_dir, size_cache=None):
    """Pre-populate a PNG re-encode cache for every unique source image.

    Pure CPU/PIL work with no framework dependency -- factored out of
    ``yolo_obb_to_dota`` so it can run standalone in a CPU-only Colab
    runtime (see ``notebooks/StripRCNN_png_cache_prep_cpu.py``) instead of
    burning GPU-runtime compute units on work that never touches the GPU.
    ``yolo_obb_to_dota(..., png_cache_dir=...)`` checks this same cache
    (keyed by filename stem) and skips re-encoding whatever is already
    present, so running this first turns its per-fold calls into cheap
    symlink + label-write operations.

    Also populates ``size_cache`` (if given) with each image's ``(w, h)``,
    the other repeated cost ``yolo_obb_to_dota`` would otherwise pay once
    per fold per item even on a PNG-cache hit.

    Returns the number of images newly encoded (0 on a fully-warm cache).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen, n_encoded = set(), 0
    for img_path, _ in tqdm(items, desc="Building PNG cache", unit="img"):
        img_path = Path(img_path)
        if img_path.stem in seen:
            continue
        seen.add(img_path.stem)
        cached_img = cache_dir / f"{img_path.stem}.png"
        if not cached_img.exists():
            with Image.open(img_path) as im:
                im.convert("RGB").save(cached_img, "PNG")
            n_encoded += 1
        if size_cache is not None and img_path.stem not in size_cache:
            size_cache[img_path.stem] = image_size(img_path)
    return n_encoded


def resized_copy(img_path, imgsz, cache_root):
    """Return ``(path, w, h)`` for a copy of ``img_path`` pre-scaled to the
    exact size S2A-Net's own pipeline would have resized it to, cached under
    ``cache_root/<imgsz>/``.

    **Why this is a cache and not a resampling change.** ``RotatedResize(
    img_scale=(imgsz, imgsz), keep_ratio=True)`` resizes through
    ``mmcv.imrescale``, which (mmcv 0.2.14, confirmed against its own source)
    computes::

        scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
        new_size     = (int(w * scale_factor + 0.5), int(h * scale_factor + 0.5))

    For a *square* ``scale`` that reduces to ``imgsz / max(h, w)``, so the
    result's long side is exactly ``imgsz``. Reproducing that arithmetic here
    and caching at ``new_size`` means the pipeline re-derives
    ``scale_factor == 1.0`` on the cached image and its own resize becomes a
    no-op -- the network sees the same pixels it would have seen from the
    original file. ``cv2.resize(..., INTER_LINEAR)`` is the same call
    ``mmcv.imresize`` makes for its default ``interpolation='bilinear'``, and
    the cache is PNG so re-decoding returns those pixels exactly, with no JPEG
    generation loss to account for.

    Worth doing because this dataset's images are ~8064x6048 (48.8 MP): one
    ``cv2.imread`` is ~0.5 s and ~146 MB, paid per image per epoch, to keep
    0.1 MP of it at ``imgsz=320``. Keyed by filename stem, so the K
    overlapping folds that each contain a given image share one copy -- the
    same reasoning as ``build_png_cache``'s cache and ``size_cache``.

    ``cv2`` is imported lazily on purpose: this module promises to stay
    importable on numpy + PIL alone (see the module docstring), and only the
    S2A-Net notebooks ever call this. PIL is not a substitute here -- its
    ``BILINEAR`` applies a scaled support filter when downscaling, so it does
    NOT produce the same pixels as OpenCV's ``INTER_LINEAR``, and the whole
    point of this function is to produce exactly what the pipeline would.
    """
    import cv2

    img_path = Path(img_path)
    cache_dir = Path(cache_root) / str(imgsz)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{img_path.stem}.png"
    if dst.exists():
        w, h = image_size(dst)
        return dst, w, h

    w0, h0 = image_size(img_path)
    scale_factor = min(imgsz / max(h0, w0), imgsz / min(h0, w0))
    if scale_factor >= 1.0:
        # Already no larger than the target, so the pipeline's own resize
        # would UPscale it. Caching an upscaled copy would only move that work
        # around (and cost disk), so hand back the original and let the
        # pipeline do exactly what it does today.
        return img_path, w0, h0

    new_w = int(w0 * scale_factor + 0.5)
    new_h = int(h0 * scale_factor + 0.5)
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(
            f"cv2 could not decode {img_path} -- a truncated or non-image file "
            f"would otherwise surface much later as a training-time crash."
        )
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Write-then-rename: a run killed mid-write (Colab disconnect, OOM) would
    # otherwise leave a truncated PNG that every later call trusts on sight,
    # since the only freshness check is `dst.exists()`.
    tmp = dst.with_name(dst.name + ".tmp.png")
    if not cv2.imwrite(str(tmp), out):
        raise RuntimeError(f"cv2 could not write {tmp}")
    tmp.replace(dst)
    return dst, new_w, new_h


def yolo_obb_to_dota(items, out_dir, class_names, link_images=True, convert_to_png=False,
                      png_cache_dir=None, size_cache=None):
    """Materialise a DOTA-style split: ``out_dir/{images,labelTxt}``.

    DOTA lines are ``x1 y1 x2 y2 x3 y3 x4 y4 <category> <difficulty>`` in
    ABSOLUTE pixels, so every polygon is denormalised by its own image size.
    Difficulty is written as 0 throughout -- this dataset has no difficulty
    annotation, and writing 1 anywhere would silently drop those instances
    from frameworks that filter on it by default.

    Images are symlinked rather than copied (a Colab disk holds one dataset,
    not K+1 copies of it); pass ``link_images=False`` to copy instead.

    ``convert_to_png``: re-encode each image as an actual PNG at
    ``<stem>.png`` regardless of its source format/extension, instead of
    symlinking/copying it under its original name. Needed for Strip R-CNN:
    MMRotate's ``DOTADataset.load_annotations`` hardcodes
    ``img_name = img_id + '.png'`` regardless of the real extension, so a
    symlinked ``.jpg``/``.JPG`` (this dataset's actual format) would never be
    found -- every image load would fail. A renamed symlink/copy wouldn't fix
    this: the bytes would still be JPEG-encoded under a ``.png`` name, which
    would fail to decode. This is real re-encoding work (slower, and PNG is
    typically larger on disk than JPEG), so it's opt-in -- RF-DETR-OBB, which
    shares this function and does not have this bug, leaves it off.

    ``png_cache_dir``: when ``convert_to_png=True``, an image's PNG bytes are
    identical no matter which split/fold uses it -- a source image that
    appears in K different folds (e.g. Strip R-CNN's K-fold notebook calling
    this function once per fold) would otherwise be re-decoded and
    re-encoded from scratch K times. Passing a directory here re-encodes each
    *source* image exactly once into that shared cache (keyed by stem) and
    symlinks it into every split's ``images/`` dir instead -- turning an
    O(K * N) PNG-encode cost into O(N). Leave unset (the default) to encode
    directly into ``out_dir`` as before, e.g. for a one-off single-split call.
    Pre-populate this cache with ``build_png_cache`` (locally, or restored
    from Drive) to skip encoding here entirely.

    ``size_cache``: an optional dict this function reads/writes, keyed by
    filename stem, to avoid re-opening a source image with ``image_size()``
    on every call across K folds for the same image. Share one dict across
    every ``yolo_obb_to_dota`` call in a notebook (same lifetime as
    ``png_cache_dir``) to benefit; leave unset for a one-off call.
    """
    import os
    import shutil

    out_dir = Path(out_dir)
    img_out, lbl_out = out_dir / "images", out_dir / "labelTxt"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    if convert_to_png and png_cache_dir is not None:
        png_cache_dir = Path(png_cache_dir)
        png_cache_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in tqdm(items, desc=f"Materialising {out_dir.name}", unit="img"):
        img_path, lbl_path = Path(img_path), Path(lbl_path)
        if convert_to_png:
            dst_img = img_out / f"{img_path.stem}.png"
            if png_cache_dir is not None:
                cached_img = png_cache_dir / f"{img_path.stem}.png"
                if not cached_img.exists():
                    with Image.open(img_path) as im:
                        im.convert("RGB").save(cached_img, "PNG")
                if not dst_img.exists():
                    os.symlink(cached_img, dst_img)
            elif not dst_img.exists():
                with Image.open(img_path) as im:
                    im.convert("RGB").save(dst_img, "PNG")
        else:
            dst_img = img_out / img_path.name
            if not dst_img.exists():
                if link_images:
                    os.symlink(img_path, dst_img)
                else:
                    shutil.copy2(img_path, dst_img)

        if size_cache is not None:
            stem = img_path.stem
            if stem not in size_cache:
                size_cache[stem] = image_size(img_path)
            w, h = size_cache[stem]
        else:
            w, h = image_size(img_path)
        cls, polys = read_yolo_obb_label(lbl_path)
        lines = []
        for c, poly in zip(cls, polys):
            xy = poly.copy()
            xy[0::2] *= w
            xy[1::2] *= h
            coords = " ".join(f"{v:.2f}" for v in xy)
            # Class names contain spaces ("Algal spot"), which would break
            # DOTA's whitespace-delimited parse -- underscore them here and
            # undo it when reading predictions back.
            lines.append(f"{coords} {class_names[int(c)].replace(' ', '_')} 0")
        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    return out_dir


def yolo_obb_to_coco(items, out_json, class_names, images_root=None):
    """Write a COCO JSON whose ``bbox`` field is an 8-value polygon.

    That is the convention Oriented-DETR's datasets/dota.py expects
    (``[s_tgt["category_id"]] + s_tgt["bbox"]`` is built into a 9-wide row,
    and poly2csl asserts the polygon part is 8 wide) -- it is NOT standard
    COCO ``[x, y, w, h]``. ``area`` and an axis-aligned ``bbox_aabb`` are
    included for any tooling that expects them, but the trainer reads
    ``bbox``.

    ``category_id`` is 0-based to match the class indices used everywhere
    else in this project; Oriented-DETR indexes its label vector directly
    rather than assuming COCO's 1-based ids.
    """
    images, annotations = [], []
    ann_id = 1
    for img_id, (img_path, lbl_path) in enumerate(items, start=1):
        img_path = Path(img_path)
        w, h = image_size(img_path)
        file_name = img_path.name if images_root is None else str(
            img_path.relative_to(images_root)
        )
        images.append({"id": img_id, "file_name": file_name, "width": w, "height": h})

        cls, polys = read_yolo_obb_label(lbl_path)
        for c, poly in zip(cls, polys):
            xy = poly.copy()
            xy[0::2] *= w
            xy[1::2] *= h
            xs, ys = xy[0::2], xy[1::2]
            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(c),
                "bbox": [float(v) for v in xy],          # 8-value polygon
                "bbox_aabb": [x0, y0, x1 - x0, y1 - y0],  # convenience only
                "area": float(_poly_area(xy)),
                "iscrowd": 0,
                "segmentation": [[float(v) for v in xy]],
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)],
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(coco))
    return out_json


def _poly_area(xy):
    """Shoelace area of a flat 8-value polygon."""
    x, y = xy[0::2], xy[1::2]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def poly_to_rotated_box(xy):
    """8-value ABSOLUTE-pixel polygon -> 5-value ``[cx, cy, w, h, angle_rad]``.

    Framework-free port of s2anet's own
    ``mmdet.core.bbox.transforms_rotated.poly_to_rotated_box_np`` (same repo
    S2A-Net's worker trains under) -- same angle convention: the angle is
    measured along whichever of the polygon's first two edges is LONGER
    (that edge becomes ``width``), then normalised into
    ``[-pi/4, 3*pi/4)`` via s2anet's own ``norm_angle``. Kept byte-identical
    to that function on purpose so a prediction decoded by
    ``src/s2anet_worker.py`` (which imports s2anet's own inverse function,
    ``rotated_box_to_poly_single``, directly, since mmdet is already
    importable there) round-trips through the SAME convention this side
    used to build the training annotations -- a mismatched angle convention
    would silently train and evaluate under two different geometries.
    """
    poly = np.asarray(xy, dtype=np.float64).reshape(4, 2)
    pt1, pt2, pt3, pt4 = poly
    edge1 = float(np.linalg.norm(pt1 - pt2))
    edge2 = float(np.linalg.norm(pt2 - pt3))
    width, height = max(edge1, edge2), min(edge1, edge2)
    if edge1 > edge2:
        angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    else:
        angle = np.arctan2(pt4[1] - pt1[1], pt4[0] - pt1[0])
    angle = (angle - (-np.pi / 4)) % np.pi + (-np.pi / 4)  # s2anet's norm_angle
    cx, cy = (pt1 + pt3) / 2
    return np.array([cx, cy, width, height, angle], dtype=np.float32)


def yolo_obb_to_s2anet_ann(items, out_dir, link_images=True, size_cache=None,
                           resize_to=None, resize_root=None):
    """Materialise an mmdet-1.x/"s2anet" split: ``out_dir/images/`` + an
    ``out_dir/ann.json`` annotation list, which ``src/s2anet_worker.py``
    converts into the pickle mmdet itself loads.

    S2A-Net's ``DotaDataset`` (a ``CustomDataset`` subclass, per
    ``mmdet/datasets/custom.py``) loads annotations via
    ``mmcv.load(ann_file)``, expecting a plain pickled list of dicts:
    ``{'filename', 'width', 'height', 'ann': {'bboxes' (n,5) float32
    [cx,cy,w,h,angle_rad], 'labels' (n,) int64, 'bboxes_ignore' (0,5),
    'labels_ignore' (0,)}}`` -- confirmed by reading that file's own
    docstring, and confirmed the ``bboxes`` field must be 5-value ROTATED
    boxes rather than 8-value polygons (unlike every other arm's on-disk
    format) by reading ``RotatedResize._resize_bboxes``
    (``mmdet/datasets/pipelines/transforms_rotated.py``), which calls
    ``rotated_box_to_poly_np`` on this exact field.

    Why this writes JSON when mmdet wants a pickle: this function runs in the
    notebook's own kernel (Colab Python 3.13, numpy 2.x) while mmdet loads the
    file over in the `s2anet` conda env (Python 3.7, numpy 1.x), and a pickle
    cannot cross that gap in this direction. numpy 2 pickles every array
    through ``numpy._core.multiarray._reconstruct``; numpy 1.x has no
    ``numpy._core`` at all, so unpickling dies with ``ModuleNotFoundError: No
    module named 'numpy._core'`` -- numpy 2 can read numpy 1's pickles, but not
    the reverse. Python's pickle protocol is a second, independent trap on the
    same line: Python 3.13 defaults to protocol 4, which 3.7 can read, but
    3.14 defaults to protocol 5, which it cannot -- so a Colab runtime bump
    would have broken this again for an unrelated reason. JSON has neither
    problem, and it stays readable when a fold's annotations need inspecting
    from either side. ``s2anet_worker.py``'s ``ann_json_to_pkl()`` rebuilds the
    arrays with the numpy that mmdet will actually use.

    No PNG re-encoding for its own sake (unlike ``yolo_obb_to_dota``'s Strip
    R-CNN-specific ``convert_to_png``): s2anet's ``LoadImageFromFile`` reads
    through ``mmcv.imread`` (OpenCV), which has no Strip-R-CNN-style hardcoded
    ``.png`` extension bug, so images are symlinked under their original
    name/extension.

    ``resize_to``/``resize_root``: when set, each image is symlinked from a
    shared ``resize_root/<resize_to>/`` cache of copies pre-scaled to exactly
    what the training pipeline's own ``RotatedResize`` would produce at
    ``img_scale=(resize_to, resize_to)``, and each record's ``width``/``height``
    are the CACHED dimensions. See ``resized_copy`` for why that leaves the
    pixels the network sees unchanged while removing a ~0.5 s 48.8 MP JPEG
    decode from every image of every epoch. The boxes need no separate
    rescaling: YOLO labels are normalised, so denormalising against the cached
    size is exact by construction. Leave both unset to symlink the originals
    and denormalise against their real size, as before.

    One sub-pixel honesty note. ``new_size`` rounds each axis to an integer, so
    the cached image's true per-axis ratio (``new_w / w``) can differ slightly
    from the float ``scale_factor`` that the UNCACHED pipeline multiplies its
    boxes by -- the two paths can place a box up to half a pixel apart, and the
    cached one is the more faithful (its boxes match its own pixels exactly).
    For this dataset it is moot: 8064x6048 is exactly 4:3, so both axes round
    exactly and the deviation is zero at every ``imgsz`` in the search space.

    Because the cached size depends on ``resize_to``, a split materialised at
    one ``resize_to`` is NOT reusable at another -- give each its own
    ``out_dir`` (the S2A-Net notebooks use ``fold<k>/train@<imgsz>``).
    """
    import os
    import shutil

    if resize_to is not None and resize_root is None:
        raise ValueError("resize_to requires resize_root (the shared cache directory)")

    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    records = []
    for img_path, lbl_path in tqdm(items, desc=f"Materialising {out_dir.name}", unit="img"):
        img_path, lbl_path = Path(img_path), Path(lbl_path)

        if resize_to is not None:
            # resized_copy's own on-disk cache makes a repeat call two syscalls
            # plus a PNG header read, so size_cache buys nothing here and would
            # have to be keyed by (stem, resize_to) to stay correct.
            src_img, w, h = resized_copy(img_path, resize_to, resize_root)
        else:
            src_img = img_path
            if size_cache is not None:
                stem = img_path.stem
                if stem not in size_cache:
                    size_cache[stem] = image_size(img_path)
                w, h = size_cache[stem]
            else:
                w, h = image_size(img_path)

        dst_img = img_out / src_img.name
        # `is_symlink() or exists()`, and an absolute target: exists() FOLLOWS
        # the link, so a symlink created from a relative source path reads as
        # "missing" (it cannot resolve from inside img_out) and this would both
        # re-raise FileExistsError and leave a dangling link behind. Every
        # notebook happens to pass absolute paths, so this never fired in
        # anger -- caught by a test that passed relative ones.
        if not (dst_img.is_symlink() or dst_img.exists()):
            if link_images:
                os.symlink(src_img.resolve(), dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        cls, polys = read_yolo_obb_label(lbl_path)
        boxes = np.zeros((len(cls), 5), dtype=np.float32)
        for i, poly in enumerate(polys):
            xy = poly.copy()
            xy[0::2] *= w
            xy[1::2] *= h
            boxes[i] = poly_to_rotated_box(xy)

        records.append({
            "filename": dst_img.name,
            "width": w,
            "height": h,
            "ann": {
                # Plain nested lists, not arrays -- see the note above on why
                # this file is JSON. s2anet_worker.py's ann_json_to_pkl()
                # restores the exact dtypes/shapes documented above.
                "bboxes": boxes.tolist(),
                # +1: mmdet-1.x labels are ONE-based, 0 being background.
                # `anchor_target` starts every anchor at label 0 and writes
                # `labels[pos_inds] = gt_labels[...]`, and the focal-loss
                # kernel scores channel d as positive only where
                # `target == d + 1` -- so a 0-based class 0 would be trained
                # as pure background (never detectable) and every other class
                # would land one output channel low, while
                # `multiclass_nms_rotated`/`bbox2result` hand predictions back
                # 0-based. This repo's own converter does the same thing:
                # `label_ids = {name: i + 1 ...}` in
                # DOTA_devkit/convert_dota_to_mmdet.py.
                "labels": (cls.astype(np.int64) + 1).tolist(),
                "bboxes_ignore": [],
                "labels_ignore": [],
            },
        })

    # Background-only images ARE trained on, but only because two things were
    # changed to make that possible; both are easy to lose in a refactor, so
    # they are named here where the empty records are produced.
    #
    #   1. The notebooks' `build_config` passes `LoadAnnotations(...,
    #      skip_img_without_anno=False)`. At mmdet-1.x's default of True,
    #      `_load_bboxes` returns None for a zero-instance image and
    #      `CustomDataset.__getitem__` quietly resamples a DIFFERENT image via
    #      `_rand_another(idx)` -- so an empty label file trained nothing, did
    #      not error, and merely duplicated some other image that epoch. Worse
    #      for the E-IRFS twin, where that resample is uniform and silently
    #      overrides E-IRFS's own choice for those draws.
    #   2. `src/s2anet_worker.py`'s `patch_empty_gt_support()`, because merely
    #      passing the image through then hits `MaxIoUAssigner.assign`'s
    #      `raise ValueError('No gt or bboxes')`.
    #
    # `ann_json_to_pkl`'s `reshape(-1, 5)` on these empty arrays is therefore
    # load-bearing now, not defensive.
    #
    # This dataset has label-free images on purpose -- see
    # analysis/dataset/phase4_background_negatives.py, which adds ~8% of train
    # as background-only field crops precisely so the model learns to predict
    # nothing on them. Still counted and reported, since the count silently
    # going to zero would mean the negatives stopped reaching this arm.
    n_empty = sum(1 for r in records if not r["ann"]["labels"])
    if n_empty:
        print(f"  NOTE: {n_empty}/{len(records)} images in {out_dir.name} have no "
              f"instances; these train as background (see yolo_obb_to_s2anet_ann).")

    ann_path = out_dir / "ann.json"
    ann_path.write_text(json.dumps(records))
    return img_out, ann_path
