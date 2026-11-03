"""Shared E-IRFS class-imbalance sampler for durian OBB training.

Single source of truth for the sampler previously copy-pasted into
notebooks/train.ipynb, notebooks/v4_confound_fix_train.ipynb,
notebooks/Template_Clean_E-IRFS_optuna_datasetv3.ipynb, and
notebooks/SAHI_E-IRFS_optuna_datasetv3.ipynb -- see
results/source_confound_pipeline.md (Phase 5.5) and
results/source_confound_scoping.md.

Usage in a notebook:

    import sys
    sys.path.insert(0, PROJECT_ROOT)   # wherever this repo's root lives
    from src.data_loader import YOLOEIRFSCappedDataset
    import ultralytics.data.build as build
    build.YOLODataset = YOLOEIRFSCappedDataset
"""
import csv
import os

import numpy as np
from ultralytics.data.dataset import YOLODataset

EIRFS_ALPHA = 2.0
EIRFS_THRESHOLD = 0.0001
CAP_MULTIPLIER = 1.5  # capped classes' weight is clipped to this x the median weight of uncapped classes

# Only Rhizoctonia is curated-only (source_confound_scoping.md: 8 field vs. 325
# curated instances) -- used if phase3_field_coverage_report.py hasn't been run
# yet / its output isn't reachable from wherever this module is imported.
# Healthy is deliberately excluded: it has plenty of field instances (179) and
# its AP gap was a background/preprocessing shortcut already addressed
# elsewhere, not a data-starvation problem -- see results/source_confound_pipeline.md
# Phase 5.5.
DEFAULT_CAPPED_CLASS_NAMES = {"Rhizoctonia"}

PHASE3_SCOPING_CSV = "analysis/dataset/outputs/resplit/phase3_scoping.csv"


def load_capped_class_names(scoping_csv=PHASE3_SCOPING_CSV, default=DEFAULT_CAPPED_CLASS_NAMES):
    """Derive which classes to cap from phase3_field_coverage_report.py's output
    instead of a hand-maintained allowlist, so a re-split that changes which
    classes are field-data-starved is picked up automatically the next time
    this is imported. Falls back to `default` if that report hasn't been
    (re-)run yet, or its CSV isn't reachable from the current working directory.
    """
    if not os.path.exists(scoping_csv):
        return set(default)
    capped = set()
    with open(scoping_csv) as f:
        for row in csv.DictReader(f):
            verdict = row.get("verdict", "")
            if verdict.startswith("CURATED-VALIDATED") or verdict.startswith("CURATED-ONLY"):
                capped.add(row["class"])
    return capped or set(default)


# Per-fold cache: avoids recomputing identical class-frequency/weight/cap
# statistics on every Optuna trial when the same fold's data yaml is reused
# across trials (previously recomputed from scratch on every single
# model.train() call, i.e. up to n_trials x K_FOLDS times for the same
# result). Keyed on the dataset's image file list, which is stable for a
# given split/fold and different across splits/folds.
_stats_cache = {}


def _cache_key(dataset):
    im_files = getattr(dataset, "im_files", None)
    if not im_files:
        return None
    return (len(dataset.data["names"]), tuple(im_files))


class YOLOEIRFSCappedDataset(YOLODataset):
    def __init__(self, *args, alpha=EIRFS_ALPHA, t=EIRFS_THRESHOLD,
                 capped_class_names=None, cap_multiplier=CAP_MULTIPLIER, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix
        self.alpha = alpha
        self.t = t
        self.capped_class_names = (
            load_capped_class_names() if capped_class_names is None else set(capped_class_names)
        )
        self.cap_multiplier = cap_multiplier
        self.num_classes = len(self.data["names"])

        cache_key = _cache_key(self)
        cached = _stats_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            self.image_counts, self.instance_counts, self.class_weights = (
                cached[0].copy(), cached[1].copy(), cached[2].copy()
            )
        else:
            # instance_counts is the real accumulator, incremented in-place
            # by _count_class_frequencies below; image_counts is fully
            # (re)built by that same call, so it needs no seed value here.
            self.instance_counts = np.ones(self.num_classes, dtype=np.float32)
            self._count_class_frequencies()
            self._compute_class_weights()
            self._cap_confounded_classes()
            if cache_key is not None:
                _stats_cache[cache_key] = (
                    self.image_counts.copy(), self.instance_counts.copy(), self.class_weights.copy()
                )

        self._compute_image_weights()
        self.print_class_stats()

    def print_class_stats(self):
        print("\n=== E-IRFS Class Statistics (capped) ===")
        for class_id, name in self.data["names"].items():
            img_count = int(self.image_counts[class_id])
            inst_count = int(self.instance_counts[class_id])
            tag = " [CAPPED]" if name in self.capped_class_names else ""
            print(f"Class {class_id:2d} ({name:15}): {img_count:5d} images, "
                  f"{inst_count:6d} instances, weight={self.class_weights[class_id]:.3f}{tag}")
        n_bg = sum(1 for label in self.labels if not label['cls'].size)
        print(f"Background-only negatives: {n_bg} images, sampling weight fixed at 1.0 (neutral)")
        print("=========================================\n")

    def _count_class_frequencies(self):
        """Recomputed fresh from THIS dataset's labels -- i.e. the new resplit
        + Phase 4.1 background negatives, not the pre-fix split (Phase 5.5)."""
        seen_in_image = [set() for _ in range(self.num_classes)]
        for label in self.labels:
            cls_ids = label['cls'].astype(int).flatten()
            unique = np.unique(cls_ids)
            for c in unique:
                seen_in_image[c].add(id(label))
            for c in cls_ids:
                self.instance_counts[c] += 1

        self.image_counts = np.array([len(s) if len(s) > 0 else 1 for s in seen_in_image], dtype=np.float32)

    def _compute_class_weights(self):
        f_ic = self.image_counts / np.sum(self.image_counts)
        f_bc = self.instance_counts / np.sum(self.instance_counts)
        self.class_weights = np.exp(self.alpha * np.sqrt(self.t / np.sqrt(f_ic * f_bc)))

    def _cap_confounded_classes(self):
        """Clip capped classes' weight so E-IRFS can't amplify the source
        confound for classes still data-starved on one source (scoping.md).

        Warns (rather than silently no-op'ing) if none of the requested
        capped class names match this dataset's classes -- a name mismatch
        (different dataset yaml, renamed class, typo) previously disabled the
        cap with zero observable signal.
        """
        names = self.data["names"]
        capped_ids = [cid for cid, name in names.items() if name in self.capped_class_names]
        uncapped_ids = [cid for cid in names if cid not in capped_ids]

        if self.capped_class_names and not capped_ids:
            print(f"WARNING: E-IRFS cap requested for {sorted(self.capped_class_names)} "
                  f"but none of these class names matched this dataset's classes "
                  f"({list(names.values())}) -- cap NOT applied.")

        if not capped_ids or not uncapped_ids:
            return

        cap_value = self.cap_multiplier * float(np.median(self.class_weights[uncapped_ids]))
        for cid in capped_ids:
            if self.class_weights[cid] > cap_value:
                print(f"E-IRFS cap: {names[cid]} weight {self.class_weights[cid]:.3f} -> {cap_value:.3f}")
                self.class_weights[cid] = cap_value

    def _compute_image_weights(self):
        self.weights = np.array([
            np.max(self.class_weights[label['cls'].astype(int).flatten()])
            if label['cls'].size else 1.0   # background-only negatives: neutral, never capped or boosted
            for label in self.labels
        ])
        total = np.sum(self.weights)
        self.probabilities = self.weights / total if total > 0 else np.ones_like(self.weights) / max(len(self.weights), 1)
        # Precomputed once here rather than rebuilt by np.random.choice(p=...)
        # on every single __getitem__ call (which internally recomputes this
        # cumsum from scratch every time it's given a `p=` array).
        self._cum_probabilities = np.cumsum(self.probabilities)

    def __getitem__(self, index):
        """Sample images based on precomputed probabilities (E-IRFS) during training."""
        if self.train_mode and len(self._cum_probabilities):
            index = int(np.searchsorted(self._cum_probabilities, np.random.random(), side="right"))
            index = min(index, len(self.labels) - 1)  # guard float-rounding at the tail
        return self.transforms(self.get_image_and_label(index))
