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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # RF-DETR-OBB (Roboflow, ICLR 2026) -- E-IRFS + Optuna, datasetv3
#
# Architecture arm 4 of 4. Structurally parallel to
# `notebooks/Template_Clean_E-IRFS_optuna_datasetv3.py` (YOLO11m-OBB): same
# folds, same E-IRFS sampler, same Optuna budget, same final-train protocol,
# same held-out metric. Framework-forced differences are enumerated in
# `results/obb_model_comparison.md`.
#
# ## Maturity warning -- this arm is not on equal footing with the other three
#
# RF-DETR's OBB support is **an open pull request, not a release**. As of
# 2026-08-18, [PR #921](https://github.com/roboflow/rf-detr/pull/921) is open
# against `develop` (opened 2026-04-03, last updated 2026-08-11) and the
# latest published release (1.9.3) has no OBB. This notebook installs from the
# contributor's branch. Three consequences you must state in any write-up:
#
# 1. **There are no OBB-pretrained weights.** The other three arms start from
#    DOTA- or COCO-pretrained detectors; this one starts from RF-DETR's
#    COCO-pretrained *axis-aligned* weights with a freshly-initialised angle
#    output. That is a real disadvantage and it is not an architectural one.
# 2. **The upstream API may change before merge.** The three monkeypatches
#    below are written against commit `6fb5051c`. Each asserts on what it
#    finds rather than assuming, so an upstream change fails loudly.
# 3. **RF-DETR's own OBB metric is not a rotated-IoU metric.** Its eval
#    callback scores the *axis-aligned envelope* of the predicted corners
#    (`coco_eval.py`: "the axis-aligned envelope of the 4 OBB corners is used").
#    That number is systematically optimistic and is not comparable to
#    anything else here. This notebook ignores it and scores through
#    `src/obb_eval.py` like every other arm.
#
# If the comparison needs all four arms on equal footing, this one is the
# candidate to drop or to report separately as a preview result.

# %% [markdown]
# ## file setup

# %%
# Installed from the OBB pull-request branch, pinned to the exact commit this
# notebook's monkeypatches were written against. Do NOT relax this to a branch
# name: the branch moves, and a silently-changed dataset loader would alter the
# training data without altering any number in this notebook.
# !pip3 install -q "rfdetr[train,loggers] @ git+https://github.com/farukalamai/rf-detr@6fb5051c9c1c"
# !pip3 install -q optuna pandas numpy pyyaml shapely

# %%
from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# %%
# !unzip -q /content/drive/MyDrive/Krong\ Si\ Su\ Pap/datasetv3_2.zip -d /content
# !cp /content/drive/MyDrive/Krong\ Si\ Su\ Pap/classes.txt /content/classes.txt

# %%
# !mv /content/datasetv3_2 /content/data

# %%
import json
import random
import shutil
import sys
import traceback
from pathlib import Path
from statistics import mean

import numpy as np
import optuna
import pandas as pd
import torch

sys.path.insert(0, PROJECT_ROOT)
from src.obb_common import (  # noqa: E402
    EIRFSWeights, attach_eirfs, collect_items, load_class_names, make_folds,
    yolo_obb_to_dota,
)
from src.obb_eval import (  # noqa: E402
    dump_predictions, evaluate_obb, load_ground_truth,
)

EIRFS_THRESHOLD = 0.0001
TRAIN_EPOCHS = 250

# --- Edit these paths ---
# NOTE: matches the YOLO template's current default -- this still points at a
# flat pre-fix dataset dump, same as Template_Clean_E-IRFS_optuna_datasetv3.py's
# own DATA_YAML. If reusing this notebook for the Phase 6.1 targeted HPO re-run,
# repoint DATA_ROOT at the grouped/dedup'd/stratified resplit dataset
# (data/dataset_resplit.yaml + Phase 4.1 background negatives) IN LOCKSTEP with
# the template -- otherwise this arm's E-IRFS frequencies and folds would be
# computed on a different split than the other three, breaking the "same data"
# guarantee in results/obb_model_comparison.md.
DATA_ROOT   = "/content/data"
IMAGES_DIR  = f"{DATA_ROOT}/train/images"
LABELS_DIR  = f"{DATA_ROOT}/train/labels"
TEST_IMAGES = f"{DATA_ROOT}/test/images"
TEST_LABELS = f"{DATA_ROOT}/test/labels"
CLASSES_TXT = f"{DATA_ROOT}/classes.txt"

# RFDETRMedium is the closest parameter-count match to YOLO11m and
# Strip R-CNN-S. Nano/Small would confound architecture with capacity.
MODEL_VARIANT = "RFDETRMedium"

# --- Training/Optuna knobs: held IDENTICAL to the YOLO template ---
DEVICE      = 0
PROJECT     = "/content/runs/optuna_rfdetr_obb"
EPOCHS_PER_TRIAL = 50
# Two different patience regimes, same intent as the YOLO template: short
# patience for cheap HPO trials, much longer patience once we commit to the
# real final-training run so it isn't cut off before it's had a chance to
# converge over its much larger epoch budget.
PATIENCE         = 7    # HPO trials only
FINAL_PATIENCE   = 50   # final training on the full split only
SEED             = 42
K_FOLDS          = 3
SHUFFLE_BEFORE_SPLIT = True

assert Path(PROJECT).is_absolute(), f"PROJECT must be absolute, got {PROJECT!r}"
Path(PROJECT).mkdir(parents=True, exist_ok=True)

CLASS_NAMES = load_class_names(CLASSES_TXT)
NUM_CLASSES = len(CLASS_NAMES)
DOTA_CLASSES = tuple(n.replace(" ", "_") for n in CLASS_NAMES)
print(f"{NUM_CLASSES} classes: {CLASS_NAMES}")

# %% [markdown]
# ### Dataset materialisation
#
# Same folds as every other arm. RF-DETR's OBB loader reads DOTA `labelTxt`
# directories, the same on-disk encoding the Strip R-CNN notebook produces, so
# `yolo_obb_to_dota` is reused verbatim. The split subdirectories must be named
# `train` and `val` (or `valid`) -- `build_dota` hardcodes those.

# %%
items = collect_items(IMAGES_DIR, LABELS_DIR)
print(f"Found {len(items)} labeled images.")

folds = make_folds(items, K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)
print("Fold sizes (train/val):", [(len(f["train"]), len(f["val"])) for f in folds])

FOLD_ROOT = Path(PROJECT) / "folds"
fold_specs = []
for f in folds:
    k = f["fold"]
    root = FOLD_ROOT / f"fold{k}"
    yolo_obb_to_dota(f["train"], root / "train", CLASS_NAMES)
    yolo_obb_to_dota(f["val"], root / "val", CLASS_NAMES)
    fold_specs.append({
        "fold": k,
        "dataset_dir": str(root),
        "val_items": f["val"],
        "train_labels": [lbl for _, lbl in f["train"]],
    })
print("Materialised DOTA-format folds under", FOLD_ROOT)

# %% [markdown]
# ### Upstream patches
#
# Two are required for correctness, one attaches E-IRFS.
#
# **Patch 1 -- class names.** `build_dota` constructs `DotaDetection(...)`
# without passing `class_names`, so it always falls back to the 15 hardcoded
# DOTA v1.0 categories. With this dataset's 7 classes, every annotation's
# category would land outside `class_to_idx`, be logged once as "unknown", and
# be **silently dropped** -- training would run to completion on entirely empty
# targets and report a plausible-looking loss curve. Rebinding
# `DOTA_V1_CLASSES` is not sufficient: it is bound to the function's default
# argument at definition time, so the module-global no longer affects it.
# Rebinding the *class* is, because `build_dota` looks that name up in its own
# module globals at call time.
#
# **Patch 2 -- E-IRFS.** Same mechanism as every other arm.
#
# **Patch 3 -- fail loudly.** The "unknown category" path is downgraded from a
# warning to an exception. A silent 100% annotation-drop is the single most
# expensive failure mode available here.

# %%
import rfdetr.datasets.dota_detection as rf_dota  # noqa: E402

_ORIGINAL_DOTA_DETECTION = rf_dota.DotaDetection
_ACTIVE_EIRFS = {"weights": None}


class DurianDotaDetection(_ORIGINAL_DOTA_DETECTION):
    """DotaDetection pinned to this project's classes, with a loud unknown-category path."""

    def __init__(self, root, transforms=None, class_names=None, include_difficult=False):
        super().__init__(root, transforms=transforms,
                         class_names=DOTA_CLASSES, include_difficult=include_difficult)
        # Verify the fix actually took, rather than trusting it.
        assert tuple(self.class_names) == DOTA_CLASSES, (
            f"class_names patch did not take: got {self.class_names}"
        )
        self._assert_annotations_survive()

    def _assert_annotations_survive(self):
        """Confirm at least one annotation parses into a known category.

        Cheap up-front check that turns the silent all-empty-targets failure
        into an immediate, readable error.
        """
        from rfdetr.datasets.dota_detection import parse_dota_annotation

        seen, kept = 0, 0
        for img_path in self.image_files[:50]:
            ann = self.labels_dir / f"{img_path.stem}.txt"
            if not ann.exists():
                continue
            for a in parse_dota_annotation(ann):
                seen += 1
                if a["category"] in self.class_to_idx:
                    kept += 1
        if seen and not kept:
            raise RuntimeError(
                f"All {seen} annotations sampled from {self.labels_dir} have categories "
                f"outside {self.class_names}. Training would silently run on empty "
                f"targets. Check that yolo_obb_to_dota wrote underscored class names."
            )


rf_dota.DotaDetection = DurianDotaDetection

# Guard against re-running this cell in the same kernel (common after fixing
# an earlier cell without restarting the runtime): a second run would
# capture the ALREADY-WRAPPED `eirfs_build_dota` as `_ORIGINAL_BUILD_DOTA`,
# and since `eirfs_build_dota` looks up the `_ORIGINAL_BUILD_DOTA` global by
# name at call time (not by closure value), the new wrapper would end up
# calling the old wrapper, which calls itself -- infinite recursion that
# only surfaces once training reaches dataset construction, not at
# cell-run time.
if not getattr(rf_dota.build_dota, "_is_eirfs_wrapper", False):
    _ORIGINAL_BUILD_DOTA = rf_dota.build_dota


def eirfs_build_dota(image_set, args, resolution):
    dataset = _ORIGINAL_BUILD_DOTA(image_set, args, resolution)
    if image_set == "train" and _ACTIVE_EIRFS["weights"] is not None:
        attach_eirfs(dataset, _ACTIVE_EIRFS["weights"], train_mode=True)
        print(f"E-IRFS attached to '{image_set}' dataset ({len(dataset)} images)")
    return dataset


eirfs_build_dota._is_eirfs_wrapper = True
rf_dota.build_dota = eirfs_build_dota
# rfdetr/datasets/__init__.py did `from ...dota_detection import build_dota` at
# import time, binding the original function object; rebinding the attribute on
# the source module alone would leave the dispatcher calling the old one.
import rfdetr.datasets as rf_datasets  # noqa: E402
rf_datasets.build_dota = eirfs_build_dota

print("Patched: DotaDetection class names, build_dota E-IRFS hook.")

# %% [markdown]
# ### Train / predict one fold

# %%
def build_model(params):
    """Instantiate the RF-DETR variant with the oriented head enabled."""
    import rfdetr

    cls = getattr(rfdetr, MODEL_VARIANT)
    model = cls(
        oriented=True,             # ModelConfig field added by PR #921
        num_classes=NUM_CLASSES,
        resolution=params["imgsz"],
        device=f"cuda:{DEVICE}",
    )
    assert getattr(model.model_config, "oriented", False), (
        "oriented=True did not reach ModelConfig -- the model would train an "
        "axis-aligned head and every rotated metric below would be meaningless."
    )
    return model


def train_one_fold(dataset_dir, output_dir, params, epochs, eirfs_weights, patience):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _ACTIVE_EIRFS["weights"] = eirfs_weights
    # model.train() below is never given a seed kwarg -- RF-DETR's own train()
    # API (an installed PR branch, not vendored here) may or may not accept
    # one, so explicitly seed the main process instead of guessing at its
    # signature. Covers weight init (build_model, right below) and any other
    # torch/numpy/random draws the framework makes internally.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    try:
        model = build_model(params)
        model.train(
            dataset_dir=str(dataset_dir),
            dataset_file="dota",
            output_dir=str(output_dir),
            epochs=epochs,
            batch_size=params["batch"],
            grad_accum_steps=params["grad_accum_steps"],
            lr=params["lr0"],
            lr_encoder=params["lr0"] * params["lr_encoder_mult"],
            weight_decay=params["weight_decay"],
            warmup_epochs=params["warmup_epochs"],
            lr_drop=max(1, int(epochs * params["lr_drop_frac"])),
            cls_loss_coef=params["cls_loss_coef"],
            ema_decay=params["ema_decay"],
            early_stopping=True,
            early_stopping_patience=patience,
            # build_dota warns that the OBB pipeline resizes to a fixed square
            # and ignores these; setting them False silences the warning and
            # makes the fixed-resolution behaviour explicit rather than implied.
            multi_scale=False,
            expanded_scales=False,
            tensorboard=False,
            num_workers=2,
            class_names=list(DOTA_CLASSES),
        )
    finally:
        # Clear unconditionally: a crashed fold must not leave fold k's class
        # frequencies attached to fold k+1's dataset.
        _ACTIVE_EIRFS["weights"] = None

    ckpt = Path(output_dir) / "checkpoint_best_total.pth"
    if not ckpt.exists():
        candidates = list(Path(output_dir).glob("checkpoint*.pth"))
        if not candidates:
            raise RuntimeError(f"no checkpoint written under {output_dir}")
        # mtime, not a string sort -- "checkpoint10.pth" < "checkpoint9.pth"
        # lexicographically, so sorted(...)[-1] could silently pick a stale
        # checkpoint if the upstream branch ever writes unpadded numbers.
        ckpt = max(candidates, key=lambda p: p.stat().st_mtime)
    return ckpt


@torch.no_grad()
def predict_split(model, items, params, conf=0.001, num_select=300):
    """Run inference and return records in the shared prediction format.

    Goes through `PostProcess(oriented=True)` directly rather than
    `model.predict()`. `predict()` returns a supervision `Detections`, which
    has no rotated-box field -- the `corners` are dropped and only the
    axis-aligned envelope survives, which would silently score this arm on a
    different geometry to the other three.

    `target_sizes` is each image's ORIGINAL (h, w), which is what makes the
    returned corners original-image pixels -- the frame `src/obb_eval.py`
    scores in.
    """
    from PIL import Image
    from rfdetr.models.postprocess import PostProcess
    import torchvision.transforms.functional as TF

    postprocess = PostProcess(num_select=num_select, oriented=True)
    net = model.model.model if hasattr(model.model, "model") else model.model
    net.eval()
    device = f"cuda:{DEVICE}"
    res = params["imgsz"]

    records = []
    for img_path, _ in items:
        img_path = Path(img_path)
        pil = Image.open(img_path).convert("RGB")
        orig_h, orig_w = pil.height, pil.width

        tensor = TF.to_tensor(pil.resize((res, res)))
        tensor = TF.normalize(tensor, model.means, model.stds).unsqueeze(0).to(device)

        outputs = net(tensor)
        target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
        results = postprocess(outputs, target_sizes)

        result = results[0]
        assert "corners" in result, (
            "PostProcess did not return 'corners' -- oriented=True did not take "
            "effect and these predictions are axis-aligned boxes."
        )
        corners = result["corners"].reshape(-1, 8).cpu().numpy()
        scores = result["scores"].reshape(-1).cpu().numpy()
        labels = result["labels"].reshape(-1).cpu().numpy().astype(int)
        for poly, score, label in zip(corners, scores, labels):
            if score < conf:
                continue
            records.append({"image": img_path.stem, "cls": int(label),
                            "score": float(score),
                            "poly": [float(v) for v in poly]})
    return records


def load_for_inference(ckpt, params):
    """Rebuild the model and load a trained checkpoint for inference."""
    model = build_model(params)
    state = torch.load(ckpt, map_location="cpu")
    net = model.model.model if hasattr(model.model, "model") else model.model
    net.load_state_dict(state["model"] if "model" in state else state)
    net.to(f"cuda:{DEVICE}").eval()
    return model

# %% [markdown]
# ## Optuna
#
# Same budget and same guards as the YOLO template. The search space is
# DETR-shaped for the same reasons as the Oriented-DETR arm (AdamW-only, split
# backbone/encoder learning rate, step LR drop). Augmentation is *not*
# searched here: RF-DETR's OBB pipeline resizes to a fixed square and exposes
# no per-augmentation knobs equivalent to YOLO's `hsv_*`/`mosaic`/`mixup`, so
# there is nothing to tune. That is the largest search-space asymmetry in the
# comparison and it is stated explicitly in
# `results/obb_model_comparison.md` -- do not present this arm's result as if
# it received the same tuning budget per axis.

# %%
def suggest_params(trial):
    return {
        "lr0": trial.suggest_float("lr0", 1e-5, 1e-3, log=True),
        "lr_encoder_mult": trial.suggest_float("lr_encoder_mult", 0.1, 3.0, log=True),
        "lr_drop_frac": trial.suggest_float("lr_drop_frac", 0.5, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
        "cls_loss_coef": trial.suggest_float("cls_loss_coef", 0.5, 4.0),
        "ema_decay": trial.suggest_float("ema_decay", 0.9, 0.999),
        # RF-DETR's resolution must be divisible by patch_size * num_windows --
        # 16 * 2 = 32 for RFDETRMediumConfig (confirmed in rfdetr/config.py),
        # NOT 56 as a previous version of this comment assumed. 320/640/832/1024
        # from the YOLO template are already multiples of 32, so they are valid
        # here unchanged (no need for a separate RF-DETR-specific set).
        "imgsz": trial.suggest_categorical("imgsz", [320, 640, 832, 1024]),
        "batch": trial.suggest_categorical("batch", [2, 4, 8]),
        "grad_accum_steps": trial.suggest_categorical("grad_accum_steps", [1, 2, 4]),
    }


def objective(trial):
    p = suggest_params(trial)
    fold_scores, rows, fold_errors = [], [], []

    for spec in fold_specs:
        k = spec["fold"]
        out_dir = Path(PROJECT) / f"trial_{trial.number:03d}_fold{k}"
        try:
            weights = EIRFSWeights(spec["train_labels"], NUM_CLASSES, CLASS_NAMES,
                                   t=EIRFS_THRESHOLD, verbose=(trial.number == 0))
            ckpt = train_one_fold(spec["dataset_dir"], out_dir, p, EPOCHS_PER_TRIAL, weights, PATIENCE)
            model = load_for_inference(ckpt, p)
            records = predict_split(model, spec["val_items"], p)
            gt = load_ground_truth(spec["val_items"])
            result = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
            score = float(result["mAP50-95"])
            if not np.isfinite(score):
                raise RuntimeError(f"shared evaluator returned {score} for fold {k}")
        except Exception as e:
            print(f"\n{'!'*70}\n[Trial {trial.number}] Fold {k} FAILED: {e}\n{'!'*70}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
            rows.append({"trial": trial.number, "fold": k, "mAP50-95": None, "error": str(e)})
            continue
        finally:
            # DETR-scale models plus an EMA copy will OOM the next fold if the
            # previous one's graph is still resident.
            torch.cuda.empty_cache()

        fold_scores.append(score)
        rows.append({"trial": trial.number, "fold": k, "mAP50-95": score, "error": None})
        trial.report(mean(fold_scores), step=k + 1)

    pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)

    if not fold_scores:
        detail = "; ".join(fold_errors)
        if trial.number == 0:
            raise RuntimeError(
                f"Trial 0 failed on all {len(fold_specs)} folds -- aborting the study. "
                f"Causes: {detail}"
            )
        raise optuna.TrialPruned(f"all folds failed: {detail}")

    if fold_errors:
        print(f"[Trial {trial.number}] WARNING: {len(fold_errors)}/{len(fold_specs)} folds failed.")
    return float(mean(fold_scores))

# %%
N_TRIALS = 10
N_STARTUP_TRIALS = 3

sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
pruner = optuna.pruners.PercentilePruner(
    percentile=50, n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2
)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS)

completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
print(f"Completed trials: {len(completed)} / {N_TRIALS}")
if not completed:
    raise RuntimeError("No trial completed successfully; best_params.json NOT written.")
distinct_values = {round(t.value, 10) for t in completed if t.value is not None}
if len(distinct_values) == 1 and len(completed) > 1:
    raise RuntimeError(
        f"All {len(completed)} completed trials returned the identical value "
        f"{distinct_values.pop()} -- trial 0 would win by tie-break regardless of the "
        f"search. best_params.json NOT written."
    )

print("Best CV mAP50-95:", study.best_value)
print("Best params:", study.best_params)

FINAL_DIR = Path(PROJECT) / "best_final_rfdetr_obb"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
(FINAL_DIR / "best_params.json").write_text(json.dumps(study.best_params, indent=2))

# %% [markdown]
# ### Final training on the full train split

# %%
best_params = json.loads((FINAL_DIR / "best_params.json").read_text())

FULL_ROOT = FOLD_ROOT / "full"
yolo_obb_to_dota(items, FULL_ROOT / "train", CLASS_NAMES)
val_items = collect_items(f"{DATA_ROOT}/val/images", f"{DATA_ROOT}/val/labels")
yolo_obb_to_dota(val_items, FULL_ROOT / "val", CLASS_NAMES)

final_weights = EIRFSWeights([lbl for _, lbl in items], NUM_CLASSES, CLASS_NAMES,
                             t=EIRFS_THRESHOLD)
FINAL_CKPT = train_one_fold(FULL_ROOT, FINAL_DIR, best_params, TRAIN_EPOCHS, final_weights, FINAL_PATIENCE)
print(f"Final run saved at: {FINAL_DIR} (checkpoint {FINAL_CKPT})")

# %% [markdown]
# ### test

# %%
test_items = collect_items(TEST_IMAGES, TEST_LABELS)
print(f"Test images: {len(test_items)}")

model = load_for_inference(FINAL_CKPT, best_params)
records = predict_split(model, test_items, best_params)
dump_predictions(FINAL_DIR / "test_predictions.json", records)

gt = load_ground_truth(test_items)
metrics = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
(FINAL_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

print(f"\n=== RF-DETR-{MODEL_VARIANT} OBB | held-out test (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

# %% [markdown]
# ### Archive to Drive

# %%
def _unique_drive_dir(base_dir: Path) -> Path:
    """Avoid merging into a stale previous archive: if base_dir already
    exists on Drive, return the first base_dir2, base_dir3, ... that doesn't."""
    if not base_dir.exists():
        return base_dir
    n = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}{n}")
        if not candidate.exists():
            return candidate
        n += 1


# Archives the ENTIRE local /content/runs tree, not just PROJECT -- each
# notebook gets its own Colab runtime, so its /content/runs only ever holds
# this notebook's own output -- into a notebook-labeled subfolder on Drive.
# Naming the Drive destination after RUN_LABEL rather than PROJECT's basename
# keeps runs/ readable across the whole comparison (results/obb_model_comparison.md)
# instead of five siblings named after internal PROJECT strings.
RUN_LABEL = "RFDETR_OBB_E-IRFS_Optuna"

LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = _unique_drive_dir(Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}"))
DRIVE_DEST.mkdir(parents=True, exist_ok=True)
if DRIVE_DEST.name != RUN_LABEL:
    print(f"NOTE: '{RUN_LABEL}' already exists on Drive -- archiving to '{DRIVE_DEST.name}' instead.")
shutil.copytree(LOCAL_RUNS, DRIVE_DEST, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("folds"))

print("Local  :", LOCAL_RUNS, "->", sorted(p.name for p in LOCAL_RUNS.iterdir()))
print("Drive  :", DRIVE_DEST, "->", sorted(p.name for p in DRIVE_DEST.iterdir()))

# %% [markdown]
# ## Re-test from Drive (resume without retraining)
#
# If you're in a fresh runtime and just want to re-verify a previously
# archived run without retraining, run this section after the mount/constants
# cells above and skip Optuna/training entirely.

# %%
# Restores the whole /content/runs tree from its RUN_LABEL folder on Drive --
# PROJECT is a subfolder of /content/runs, so this repopulates it along with
# everything else that was archived alongside it.
import shutil
from pathlib import Path

RUN_LABEL = "RFDETR_OBB_E-IRFS_Optuna"
LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}")

LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_DEST, LOCAL_RUNS, dirs_exist_ok=True)
print(f"Restored {LOCAL_RUNS} <- {DRIVE_DEST}")

FINAL_DIR = Path(PROJECT) / "best_final_rfdetr_obb"

# %%
# Re-prints the held-out test metrics already computed -- no re-inference
# needed since predict_split()/evaluate_obb() already wrote test_metrics.json
# during the original run.
metrics = json.loads((FINAL_DIR / "test_metrics.json").read_text())

print(f"\n=== RF-DETR-{MODEL_VARIANT} OBB | held-out test (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")
