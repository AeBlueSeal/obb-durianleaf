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
# # RF-DETR-OBB (Roboflow, ICLR 2026) -- E-IRFS + K-fold final training, datasetv3
#
# K-fold variant of `notebooks/RFDETR_OBB_E-IRFS_optuna_datasetv3.py`. The
# Optuna HPO search itself is **not** re-run here -- this notebook loads the
# `best_params.json` produced by that notebook's search (from this session's
# local run, or from its Drive archive under `RFDETR_OBB_E-IRFS_Optuna`) and
# replaces the single final full-dataset training run with K-fold final
# training: one model is trained per fold with the same best hyperparameters,
# and every fold's model is evaluated on the same held-out test set through
# the shared `src/obb_eval.py` evaluator, giving a mean +/- std test score
# instead of a single-split point estimate.
#
# The fold-construction *mechanics*, the upstream patches, and the
# train/predict helpers are unchanged from the Optuna notebook -- kept
# byte-identical on purpose. If the template's fold-construction or patch
# logic changes, re-sync this file the same way. One deliberate content
# difference: the *item pool* folded over is `train/` + `val/` pooled
# together, not `train/` alone -- a K-fold split carves its own held-out
# validation portion out of every fold, so the dataset's fixed `val/` split
# (which the Optuna notebook needed as final training's own validation)
# serves no purpose here. Fold membership is therefore *not* identical to
# the Optuna notebook's CV folds -- that's fine since only
# `best_params.json` (hyperparameters, not fold-conditioned results) is
# carried over from that search. `test/` is untouched throughout, as always.
#
# ## Maturity warning -- this arm is not on equal footing with the other three
#
# Same caveats as the source Optuna notebook: RF-DETR's OBB support is an
# open pull request (roboflow/rf-detr#921), not a release; there are no
# OBB-pretrained weights; the upstream API may change before merge; and
# RF-DETR's own OBB metric (axis-aligned envelope, not rotated IoU) is not
# used here -- everything is scored through `src/obb_eval.py`. State these
# explicitly in any write-up, same as for the Optuna notebook's numbers.

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
import time
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
    dump_predictions, evaluate_obb, load_ground_truth, poly_iou_matrix,
    precision_recall_f1,
)
from src.obb_logging import (  # noqa: E402
    Timer, collect_epoch_log, dump_study, fmt_dt, open_study,
    write_fold_composition, write_per_class_tables, write_run_env,
)

EIRFS_THRESHOLD = 0.0001
TRAIN_EPOCHS = 250

# --- Edit these paths ---
# NOTE: these still point at a flat pre-fix dataset dump, matching the Optuna
# notebook's own DATA_ROOT. Must match whatever split that notebook used to
# produce best_params.json -- see the note there before repointing.
DATA_ROOT   = "/content/data"
IMAGES_DIR  = f"{DATA_ROOT}/train/images"
LABELS_DIR  = f"{DATA_ROOT}/train/labels"
# K-fold carves its own held-out validation portion out of every fold, so a
# separate fixed val split serves no purpose here (unlike the single-split
# Optuna notebook, which used it as final training's own validation) --
# pooled into `items` below instead of left untouched throughout every fold.
VAL_IMAGES  = f"{DATA_ROOT}/val/images"
VAL_LABELS  = f"{DATA_ROOT}/val/labels"
TEST_IMAGES = f"{DATA_ROOT}/test/images"
TEST_LABELS = f"{DATA_ROOT}/test/labels"
CLASSES_TXT = f"{DATA_ROOT}/classes.txt"

# RFDETRMedium is the closest parameter-count match to YOLO11m and
# Strip R-CNN-S. Nano/Small would confound architecture with capacity.
MODEL_VARIANT = "RFDETRMedium"

# --- Training knobs: held IDENTICAL to the Optuna notebook ---
DEVICE      = 0
# Two different patience regimes, same intent as the YOLO template: short
# patience for cheap HPO trials, much longer patience once we commit to a
# real final-training run so it isn't cut off before it's had a chance to
# converge over its much larger epoch budget.
PATIENCE         = 7    # HPO trials only
FINAL_PATIENCE   = 50   # K-fold final training only
SEED             = 42
K_FOLDS          = 9
SHUFFLE_BEFORE_SPLIT = True
# Kept in a separate directory from the Optuna notebook's PROJECT so K-fold
# final training never collides with that search's trial runs.
PROJECT     = "/content/runs/kfold_final_rfdetr_obb"

# Where to find the archived Optuna run's best_params.json if this session
# doesn't already have one locally (see load_best_params() below).
OPTUNA_PROJECT_LOCAL = "/content/runs/optuna_rfdetr_obb"
OPTUNA_RUN_LABEL = "RFDETR_OBB_E-IRFS_Optuna"
OPTUNA_FINAL_SUBDIR = "best_final_rfdetr_obb"

# If True, this notebook runs its own Optuna search (identical search space
# and objective to notebooks/RFDETR_OBB_E-IRFS_optuna_datasetv3.py) against
# its own fold_specs below instead of loading a previously-archived
# best_params.json. Off by default -- the documented/comparable path is to
# reuse the sibling Optuna notebook's winning params (see the header
# markdown), which is cheaper and keeps HPO decoupled from K-fold reporting.
RUN_HPO = True
# HPO_K_FOLDS is deliberately separate from K_FOLDS above: the embedded
# search below (only used when RUN_HPO=True) mirrors the sibling Optuna
# notebook's own fold count, not this notebook's final-training fold count --
# they are different stages with different budgets.
HPO_K_FOLDS      = 3    # only used when RUN_HPO=True
EPOCHS_PER_TRIAL = 50   # only used when RUN_HPO=True
N_TRIALS         = 10   # only used when RUN_HPO=True
N_STARTUP_TRIALS = 3    # only used when RUN_HPO=True

assert Path(PROJECT).is_absolute(), f"PROJECT must be absolute, got {PROJECT!r}"
Path(PROJECT).mkdir(parents=True, exist_ok=True)

CLASS_NAMES = load_class_names(CLASSES_TXT)
NUM_CLASSES = len(CLASS_NAMES)
DOTA_CLASSES = tuple(n.replace(" ", "_") for n in CLASS_NAMES)
print(f"{NUM_CLASSES} classes: {CLASS_NAMES}")

# Best-effort run metadata -- platform, GPU, framework versions, git commit,
# and this arm's own control-variable constants. Never blocks training on
# failure; see src/obb_logging.py's module docstring.
write_run_env(Path(PROJECT) / "run_env.json", repo_root=PROJECT_ROOT, extra={
    "arm": "RFDETR_OBB_E-IRFS_KFold_Final", "SEED": SEED, "K_FOLDS": K_FOLDS,
    "TRAIN_EPOCHS": TRAIN_EPOCHS, "EIRFS_THRESHOLD": EIRFS_THRESHOLD,
    "PROJECT": PROJECT, "MODEL_VARIANT": MODEL_VARIANT, "RUN_HPO": RUN_HPO,
})

# %% [markdown]
# ### Dataset materialisation
#
# Same folds as every other arm. RF-DETR's OBB loader reads DOTA `labelTxt`
# directories, the same on-disk encoding the Strip R-CNN notebook produces, so
# `yolo_obb_to_dota` is reused verbatim. The split subdirectories must be named
# `train` and `val` (or `valid`) -- `build_dota` hardcodes those.

# %%
# Pool train + val into one fold-able set -- see the note by VAL_IMAGES
# above for why the fixed val split doesn't need to stay separate here.
items = collect_items(IMAGES_DIR, LABELS_DIR) + collect_items(VAL_IMAGES, VAL_LABELS)
print(f"Found {len(items)} labeled images (train + val pooled).")

folds = make_folds(items, K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)
print("Fold sizes (train/val):", [(len(f["train"]), len(f["val"])) for f in folds])
write_fold_composition(Path(PROJECT) / "fold_composition.csv", folds, CLASS_NAMES, NUM_CLASSES)

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
# ## Hyperparameters: run Optuna here, or load a previous search
#
# `RUN_HPO` (set above) picks one of two paths:
# - `False` (default): load the `best_params.json` produced by
#   `notebooks/RFDETR_OBB_E-IRFS_optuna_datasetv3.py`'s own search (from this
#   session's local run, or its Drive archive under `RFDETR_OBB_E-IRFS_Optuna`)
#   -- the documented, cheaper path used by the header markdown.
# - `True`: run that same search space and objective directly against this
#   notebook's own `fold_specs` (already built above) and skip loading
#   entirely. This costs `N_TRIALS * HPO_K_FOLDS` extra fold-trainings at
#   `EPOCHS_PER_TRIAL` epochs before K-fold final training even starts.

# %%
def load_best_params() -> dict:
    local_path = Path(OPTUNA_PROJECT_LOCAL) / OPTUNA_FINAL_SUBDIR / "best_params.json"
    if local_path.exists():
        print(f"Loading best_params from local Optuna run: {local_path}")
        return json.loads(local_path.read_text())

    drive_path = Path(f"{PROJECT_ROOT}/runs/{OPTUNA_RUN_LABEL}/{OPTUNA_FINAL_SUBDIR}/best_params.json")
    if drive_path.exists():
        print(f"Loading best_params from Drive archive: {drive_path}")
        return json.loads(drive_path.read_text())

    raise FileNotFoundError(
        f"No best_params.json found locally ({local_path}) or on Drive "
        f"({drive_path}). Run notebooks/RFDETR_OBB_E-IRFS_optuna_datasetv3.py's "
        f"Optuna search first and archive it to Drive (its 'Archive to Drive' "
        f"cell, RUN_LABEL={OPTUNA_RUN_LABEL!r}) before running this K-fold "
        f"final-training notebook, or set RUN_HPO=True above to search here "
        f"instead. If you archived more than once, _unique_drive_dir() may "
        f"have named it '{OPTUNA_RUN_LABEL}2', '{OPTUNA_RUN_LABEL}3', etc. -- "
        f"point OPTUNA_RUN_LABEL at the archive you want."
    )


# Same search space and objective as notebooks/RFDETR_OBB_E-IRFS_optuna_datasetv3.py
# -- kept byte-identical on purpose, just pointed at this notebook's own
# fold_specs instead of the sibling notebook's. Only defined/run when
# RUN_HPO=True.
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


# When RUN_HPO=True, build a SEPARATE train-only fold split for the embedded
# search -- mirrors notebooks/RFDETR_OBB_E-IRFS_optuna_datasetv3.py's own
# fold construction (train/ only, not train/+val/ pooled) so HPO searches
# over the same kind of data the sibling Optuna notebook would, instead of
# silently sharing the final-training folds above (which pool in val/, so
# HPO and final training would otherwise see the same held-out val images).
if RUN_HPO:
    hpo_items = collect_items(IMAGES_DIR, LABELS_DIR)
    print(f"[RUN_HPO] Found {len(hpo_items)} labeled images (train only).")
    hpo_folds = make_folds(hpo_items, HPO_K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)

    HPO_FOLD_ROOT = Path(PROJECT) / "hpo_folds"
    hpo_fold_specs = []
    for f in hpo_folds:
        k = f["fold"]
        root = HPO_FOLD_ROOT / f"fold{k}"
        yolo_obb_to_dota(f["train"], root / "train", CLASS_NAMES)
        yolo_obb_to_dota(f["val"], root / "val", CLASS_NAMES)
        hpo_fold_specs.append({
            "fold": k,
            "dataset_dir": str(root),
            "val_items": f["val"],
            "train_labels": [lbl for _, lbl in f["train"]],
        })
    print("Materialised train-only HPO folds under", HPO_FOLD_ROOT)


def objective(trial):
    p = suggest_params(trial)
    fold_scores, rows, fold_errors = [], [], []

    for spec in hpo_fold_specs:
        k = spec["fold"]
        out_dir = Path(PROJECT) / f"trial_{trial.number:03d}_fold{k}"
        fold_t0 = time.perf_counter()
        try:
            weights = EIRFSWeights(spec["train_labels"], NUM_CLASSES, CLASS_NAMES,
                                   t=EIRFS_THRESHOLD, verbose=(trial.number == 0))
            ckpt = train_one_fold(spec["dataset_dir"], out_dir, p, EPOCHS_PER_TRIAL, weights, PATIENCE)
            collect_epoch_log(out_dir, "rfdetr")
            model = load_for_inference(ckpt, p)
            records = predict_split(model, spec["val_items"], p)
            gt = load_ground_truth(spec["val_items"])
            result = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
            prf = precision_recall_f1(gt, records, NUM_CLASSES)
            score = float(result["mAP50-95"])
            if not np.isfinite(score):
                raise RuntimeError(f"shared evaluator returned {score} for fold {k}")
        except Exception as e:
            print(f"\n{'!'*70}\n[Trial {trial.number}] Fold {k} FAILED: {e}\n{'!'*70}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
            rows.append({"trial": trial.number, "fold": k, "mAP50-95": None, "mAP50": None,
                        "mAP75": None, "precision": None, "recall": None, "F1": None,
                        "seconds": time.perf_counter() - fold_t0, "error": str(e)})
            continue
        finally:
            # DETR-scale models plus an EMA copy will OOM the next fold if the
            # previous one's graph is still resident.
            torch.cuda.empty_cache()

        fold_scores.append(score)
        rows.append({"trial": trial.number, "fold": k, "mAP50-95": score,
                    "mAP50": result["mAP50"], "mAP75": result["mAP75"],
                    "precision": prf["precision"], "recall": prf["recall"], "F1": prf["F1"],
                    "seconds": time.perf_counter() - fold_t0, "error": None})
        trial.report(mean(fold_scores), step=k + 1)
        # Pruning is deliberately NOT enabled. `trial.report()` above feeds the
        # PercentilePruner configured below, but the objective never calls
        # `should_prune()`, so no trial is ever cut short: every trial runs
        # every fold to completion. This is a standing project decision (the
        # pruning path was judged not worth the implementation cost), not an
        # oversight -- do not "fix" it. Note the consequence for the
        # cross-arm comparison: Strip R-CNN and S2A-Net DO prune, so those two
        # arms search under a different HPO budget than this one. See
        # results/kfold_comparability_audit.md, Finding 2.

    pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)

    if not fold_scores:
        detail = "; ".join(fold_errors)
        if trial.number == 0:
            raise RuntimeError(
                f"Trial 0 failed on all {len(hpo_fold_specs)} folds -- aborting the study. "
                f"Causes: {detail}"
            )
        raise optuna.TrialPruned(f"all folds failed: {detail}")

    if fold_errors:
        print(f"[Trial {trial.number}] WARNING: {len(fold_errors)}/{len(hpo_fold_specs)} folds failed.")
    return float(mean(fold_scores))


def run_hpo() -> dict:
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
    pruner = optuna.pruners.PercentilePruner(
        percentile=50, n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2
    )
    # sqlite-backed (not in-memory) so a Colab disconnect mid-search does not
    # lose every trial's params/values -- same pattern S2A-Net already uses.
    study = open_study(
        "rfdetr_obb_kfold", Path(PROJECT) / "hpo.db",
        Path(PROJECT_ROOT) / "runs" / "RFDETR_OBB_E-IRFS_KFold_Final" / "hpo.db",
        sampler, pruner,
    )
    n_remaining = max(0, N_TRIALS - len(study.trials))
    if n_remaining < N_TRIALS:
        print(f"Resuming study: {len(study.trials)} trial(s) already recorded, "
              f"{n_remaining} remaining.")
    if n_remaining:
        study.optimize(objective, n_trials=n_remaining)

    completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    print(f"Completed trials: {len(completed)} / {N_TRIALS}")
    if not completed:
        raise RuntimeError("No trial completed successfully; best_params.json NOT written.")
    distinct_values = {round(t.value, 10) for t in completed if t.value is not None}
    if len(distinct_values) == 1 and len(completed) > 1:
        raise RuntimeError(
            f"All {len(completed)} completed trials returned the identical value "
            f"{distinct_values.pop()} -- trial 0 would win by tie-break regardless of "
            f"the search. best_params.json NOT written."
        )

    print("Best CV mAP50-95:", study.best_value)
    print("Best params:", study.best_params)

    final_dir = Path(PROJECT) / "best_final_rfdetr_obb"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    dump_study(study, final_dir)
    return study.best_params


best_params = run_hpo() if RUN_HPO else load_best_params()
print("Using best_params:", json.dumps(best_params, indent=2))

# %% [markdown]
# ## Precision / recall / F1 at the best-F1 confidence threshold
#
# `evaluate_obb()` (the shared AP metric) deliberately has no single
# confidence operating point -- AP integrates over the whole precision/recall
# curve. Ultralytics' own `box.mp`/`box.mr`/`box.f1` (used by the YOLO K-fold
# notebooks) DO report at one point: the confidence threshold that maximises
# mean F1 across classes. `src/obb_eval.precision_recall_f1` reproduces that
# same operating point so this arm's P/R/F1 are comparable to the YOLO arms'.
# This notebook used to carry its own inline copy of that function (verified
# byte-identical to the module version before deleting it) -- now imported
# above instead, so per-class P/R/F1 additions land in one place for every arm.

# %% [markdown]
# ## K-fold final training
#
# Trains K models -- one per fold above -- each for the full `TRAIN_EPOCHS`
# using `best_params`, then evaluates every fold's model on the same
# held-out test set through the shared `src/obb_eval.py` evaluator. This
# replaces the Optuna notebook's single full-dataset final training run with
# a mean +/- std test score across folds, which is more robust to reviewer
# questions about a single lucky/unlucky split. The fold with the best test
# mAP50-95 is kept as the representative "best" model for downstream use.
#
# `torch.cuda.empty_cache()` after each fold matters here for the same reason
# it did during the Optuna trials: a DETR-scale model plus its EMA copy will
# OOM the next fold if the previous one's graph is still resident.

# %%
KFOLD_FINAL_DIR = Path(PROJECT) / "kfold_final_rfdetr_obb"
KFOLD_FINAL_DIR.mkdir(parents=True, exist_ok=True)

test_items = collect_items(TEST_IMAGES, TEST_LABELS)
print(f"Test images: {len(test_items)}")
gt = load_ground_truth(test_items)

fold_rows = []
fold_errors = []
per_fold_per_class = []
for spec in fold_specs:
    k = spec["fold"]
    out_dir = KFOLD_FINAL_DIR / f"fold{k}"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        weights = EIRFSWeights(spec["train_labels"], NUM_CLASSES, CLASS_NAMES,
                               t=EIRFS_THRESHOLD, verbose=(k == 0))
        with Timer() as train_timer:
            ckpt = train_one_fold(spec["dataset_dir"], out_dir, best_params, TRAIN_EPOCHS, weights, FINAL_PATIENCE)
        collect_epoch_log(out_dir, "rfdetr")
        (out_dir / "resolved_params.json").write_text(json.dumps(best_params, indent=2))

        with Timer() as eval_timer:
            model = load_for_inference(ckpt, best_params)
            records = predict_split(model, test_items, best_params)
            dump_predictions(out_dir / "test_predictions.json", records)
            metrics = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
            prf = precision_recall_f1(gt, records, NUM_CLASSES, class_names=CLASS_NAMES)
        metrics["precision"] = prf["precision"]
        metrics["recall"] = prf["recall"]
        metrics["F1"] = prf["F1"]
        for _name, _pc in prf.get("per_class", {}).items():
            metrics["per_class"].setdefault(_name, {}).update(_pc)
        for _name, _c in prf.get("counts", {}).items():
            metrics["per_class"].setdefault(_name, {}).update(_c)
        per_fold_per_class.append((k, metrics["per_class"]))
        (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    except Exception as e:
        # One fold's failure (e.g. a transient CUDA OOM) must not take down
        # the whole run -- earlier folds' checkpoints/test_metrics.json are
        # already on disk, and kfold_final_results.csv/summary.json should
        # still get written from whatever folds did succeed. Mirrors
        # objective()'s own per-fold error handling above.
        print(f"\n{'!'*70}\n[Fold {k}] FAILED: {e}\n{'!'*70}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
        fold_rows.append({
            "fold": k, "test_mAP50-95": None, "test_mAP50": None, "test_mAP75": None,
            "test_precision": None, "test_recall": None, "test_F1": None,
            "ckpt": None, "run_dir": str(out_dir), "error": str(e),
            "started_at": started_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_seconds": None, "eval_seconds": None,
            "epochs_completed": None, "final_train_loss": None,
        })
        continue
    finally:
        torch.cuda.empty_cache()

    epoch_summary_path = out_dir / "epoch_log_summary.json"
    epoch_summary = (json.loads(epoch_summary_path.read_text())
                     if epoch_summary_path.exists() else {})

    fold_rows.append({
        "fold": k,
        "test_mAP50-95": metrics["mAP50-95"],
        "test_mAP50": metrics["mAP50"],
        "test_mAP75": metrics["mAP75"],
        "test_precision": prf["precision"],
        "test_recall": prf["recall"],
        "test_F1": prf["F1"],
        "ckpt": str(ckpt),
        "run_dir": str(out_dir),
        "error": None,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_seconds": train_timer.seconds,
        "eval_seconds": eval_timer.seconds,
        "epochs_completed": epoch_summary.get("epochs_completed"),
        "final_train_loss": epoch_summary.get("final_loss"),
    })
    print(f"[Fold {k}] test mAP50-95={metrics['mAP50-95']:.4f}  mAP50={metrics['mAP50']:.4f}  "
          f"P={prf['precision']:.4f}  R={prf['recall']:.4f}  F1={prf['F1']:.4f}"
          f"  | train {fmt_dt(train_timer.seconds or 0)}")

if fold_errors:
    print(f"WARNING: {len(fold_errors)}/{len(fold_specs)} folds failed: {'; '.join(fold_errors)}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(KFOLD_FINAL_DIR / "kfold_final_results.csv", index=False)
print(fold_df)
write_per_class_tables(KFOLD_FINAL_DIR, per_fold_per_class, CLASS_NAMES)

# %%
if fold_df["test_mAP50-95"].isna().all():
    raise RuntimeError(
        "Every fold's test mAP50-95 came back NaN -- scroll up for per-fold "
        "training/eval errors. kfold_final_summary.json NOT written."
    )

summary = {
    "n_folds": int(len(fold_df)),
    # Folds that raised and were skipped. The mean/std below are over the
    # SURVIVING folds only, so n_folds alone overstates the sample.
    "n_folds_failed": int(len(fold_errors)),
    "test_mAP50-95_mean": float(fold_df["test_mAP50-95"].mean()),
    "test_mAP50-95_std": float(fold_df["test_mAP50-95"].std(ddof=0)),
    "test_mAP50_mean": float(fold_df["test_mAP50"].mean()),
    "test_mAP50_std": float(fold_df["test_mAP50"].std(ddof=0)),
    "test_mAP75_mean": float(fold_df["test_mAP75"].mean()),
    "test_mAP75_std": float(fold_df["test_mAP75"].std(ddof=0)),
    "test_precision_mean": float(fold_df["test_precision"].mean()),
    "test_precision_std": float(fold_df["test_precision"].std(ddof=0)),
    "test_recall_mean": float(fold_df["test_recall"].mean()),
    "test_recall_std": float(fold_df["test_recall"].std(ddof=0)),
    "test_F1_mean": float(fold_df["test_F1"].mean()),
    "test_F1_std": float(fold_df["test_F1"].std(ddof=0)),
    "best_fold": int(fold_df.loc[fold_df["test_mAP50-95"].idxmax(), "fold"]),
    "best_params": best_params,
}
print("K-fold final test summary:", json.dumps(summary, indent=2))
(KFOLD_FINAL_DIR / "kfold_final_summary.json").write_text(json.dumps(summary, indent=2))

best_row = fold_df.loc[fold_df["test_mAP50-95"].idxmax()]
FINAL_CKPT = Path(best_row["ckpt"])
FINAL_DIR = Path(best_row["run_dir"])
print(f"Best fold: {summary['best_fold']} -> {FINAL_CKPT}")

# %% [markdown]
# ### test
#
# Re-prints the held-out test metrics for the best fold (already computed
# during the K-fold loop above) plus the full K-fold summary.

# %%
metrics = json.loads((FINAL_DIR / "test_metrics.json").read_text())

print(f"\n=== RF-DETR-{MODEL_VARIANT} OBB | held-out test, best fold {summary['best_fold']} (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
print(f"precision={metrics.get('precision', float('nan')):.4f}  "
      f"recall={metrics.get('recall', float('nan')):.4f}  "
      f"F1={metrics.get('F1', float('nan')):.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))

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
RUN_LABEL = "RFDETR_OBB_E-IRFS_KFold_Final"

LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = _unique_drive_dir(Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}"))
DRIVE_DEST.mkdir(parents=True, exist_ok=True)
if DRIVE_DEST.name != RUN_LABEL:
    print(f"NOTE: '{RUN_LABEL}' already exists on Drive -- archiving to '{DRIVE_DEST.name}' instead.")
shutil.copytree(LOCAL_RUNS, DRIVE_DEST, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("folds", "hpo_folds"))

print("Local  :", LOCAL_RUNS, "->", sorted(p.name for p in LOCAL_RUNS.iterdir()))
print("Drive  :", DRIVE_DEST, "->", sorted(p.name for p in DRIVE_DEST.iterdir()))

# %% [markdown]
# ## Re-test from Drive (resume without retraining)
#
# If you're in a fresh runtime and just want to re-verify a previously
# archived run without retraining, run this section after the mount/constants
# cells above and skip the K-fold training loop entirely.

# %%
# Restores the whole /content/runs tree from its RUN_LABEL folder on Drive,
# then reads kfold_final_summary.json to find which fold was best.
import shutil
from pathlib import Path

RUN_LABEL = "RFDETR_OBB_E-IRFS_KFold_Final"
LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}")

LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_DEST, LOCAL_RUNS, dirs_exist_ok=True)
print(f"Restored {LOCAL_RUNS} <- {DRIVE_DEST}")

summary = json.loads((KFOLD_FINAL_DIR / "kfold_final_summary.json").read_text())
fold_df = pd.read_csv(KFOLD_FINAL_DIR / "kfold_final_results.csv")
best_row = fold_df.loc[fold_df["test_mAP50-95"].idxmax()]
FINAL_CKPT = Path(best_row["ckpt"])
FINAL_DIR = Path(best_row["run_dir"])
print(f"Best fold: {summary['best_fold']} -> {FINAL_CKPT}")

# %%
# Re-prints the held-out test metrics already computed for the best fold --
# no re-inference needed since predict_split()/evaluate_obb() already wrote
# test_metrics.json during the original K-fold loop.
metrics = json.loads((FINAL_DIR / "test_metrics.json").read_text())

print(f"\n=== RF-DETR-{MODEL_VARIANT} OBB | held-out test, best fold {summary['best_fold']} (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
print(f"precision={metrics.get('precision', float('nan')):.4f}  "
      f"recall={metrics.get('recall', float('nan')):.4f}  "
      f"F1={metrics.get('F1', float('nan')):.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))
