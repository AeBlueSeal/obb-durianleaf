# OBB architecture comparison — what is held constant, and what cannot be

Five oriented-bounding-box detectors on datasetv3_2. This document is
the declaration of differences: read it before reporting any number from the
comparison notebooks.

> **Scope note (2026-09-12).** This document describes the earlier **Optuna
> generation**: five arms, 25 epochs/trial, a single train split, and YOLO
> metrics needing `rescore_baseline_yolo.py` to be comparable. The current
> **K-fold generation** has six arms (S2A-Net was added 2026-09-11), 50
> epochs/trial, `train`+`val` pooled into 9 folds, and every arm calling
> `src/obb_eval.evaluate_obb` directly, so the rescoring caveat in §1 no longer
> applies there. For the current list of what is and is not held constant, see
> [comparison_control_variables.md](comparison_control_variables.md). §2 of this
> document (search spaces, pretraining, angle representation, environments)
> remains the best reference for *why* certain things cannot be equalised.

| Arm | Notebook | Family | Framework |
|---|---|---|---|
| YOLO11m-OBB (baseline) | `notebooks/YOLO11mOBB_E-IRFS_optuna_datasetv3.py` (== `Template_Clean_E-IRFS_optuna_datasetv3.py`) | one-stage anchor-free | Ultralytics |
| YOLO26m-OBB | `notebooks/YOLO26mOBB_E-IRFS_optuna_datasetv3.py` | one-stage, NMS-free, DFL removed | Ultralytics |
| Strip R-CNN-S | `notebooks/StripRCNN_E-IRFS_optuna_datasetv3.py` | two-stage R-CNN | MMRotate 0.x |
| Oriented-DETR R50 | `notebooks/OrientedDETR_E-IRFS_optuna_datasetv3.py` | DETR set-prediction | standalone (DINO-derived) |
| RF-DETR-Medium OBB | `notebooks/RFDETR_OBB_E-IRFS_optuna_datasetv3.py` | DETR set-prediction | rfdetr (**open PR**) |

`YOLO11mOBB_E-IRFS_optuna_datasetv3.py` is a byte-identical copy of the
template except for its run/output paths, kept so both YOLO arms can be
named consistently with the other three and so the template's own file can
keep serving as the single source of truth to propagate fixes from (see
§6 below).

### YOLO26-specific caveats

YOLO26 is not a same-architecture weight swap for YOLO11 -- it removes DFL
(direct box regression instead), ships an NMS-free end-to-end inference
path, and its recommended recipe uses a new MuSGD optimizer, Progressive
Loss, and STAL label assignment that this notebook does **not** use (its
`optimizer` search space stays `["SGD", "Adam", "AdamW"]`, matching YOLO11's,
so the comparison stays apples-to-apples on search space at the cost of not
using YOLO26's own best-known recipe). The `dfl` hyperparameter is kept in
the search space for parity but is very likely inert under YOLO26's loss —
unverified until a real run. There is also an open, unresolved upstream
report (Ultralytics issue #23548) of `yolo26n-obb.pt` OBB fine-tuning
producing decreasing mAP over epochs on a small dataset — watch for the same
symptom before treating a low YOLO26 score here as a genuine result rather
than a training-instability artifact. Full detail in each notebook's header.

## 1. What is held constant

Enforced by shared code in `src/` for the three non-YOLO arms. The two YOLO
arms (11m, 26m) instead reuse the template's own already-fixed, already-tested
`YOLOEIRFSCappedDataset` / `make_folds_files` / `kfold_assignments` — kept as
their own copy rather than migrated to `src/obb_common`, so as not to re-touch
Ultralytics-specific code that was already debugged (see the "0.0 fold scores"
bug history in the template itself). Both implementations compute the
identical formula from the identical seeded shuffle of the identical sorted
item list, but they are two separate code paths, not one literally-shared
function — call this "held constant by parallel, not shared, implementation"
for those two rows.

| Held constant | Mechanism |
|---|---|
| Train/val/test images | `datasetv3_2.zip` unzipped to `/content/data` in all five (matches the YOLO template's current default — the flat pre-fix dump, **not** the resplit dataset; see the Phase 6.1 note in each notebook) |
| K-fold assignment | Non-YOLO arms: `src/obb_common.make_folds(items, 3, seed=42)`. YOLO arms: the template's own `make_folds_files`/`kfold_assignments`, same seed and sort order. Same round-robin-over-a-seeded-shuffle logic in both; not the same function. |
| Class-imbalance sampling | Non-YOLO arms: `src/obb_common.EIRFSWeights`. YOLO arms: the template's own `YOLOEIRFSCappedDataset`. Same E-IRFS formula (α=2.0, t=1e-4, Rhizoctonia capped at 1.5× the uncapped median) in both; not the same class. |
| Sampler *mechanism* | Index substitution inside `__getitem__`, with replacement, epoch length unchanged, in all five — `src/obb_common.attach_eirfs` for the non-YOLO arms, `YOLOEIRFSCappedDataset.__getitem__` for the YOLO arms. |
| Optuna budget | TPE, seed 42, `n_trials=10`, `n_startup_trials=3`, PercentilePruner(50, warmup 2) |
| Trial schedule | 25 epochs/trial, patience 5, 3-fold CV, mean fold mAP50-95 as objective |
| Final training | 250 epochs on the full train split with the winning params |
| **Reported metric** | Non-YOLO arms score through `src/obb_eval.evaluate_obb` directly. YOLO arms report Ultralytics' own `model.val()` mAP50-95 in-notebook — run `analysis/comparison/rescore_baseline_yolo.py` against each YOLO arm's `best.pt` (it accepts any Ultralytics OBB weights via `--weights`) to get the same COCO-style AP@0.50:0.05:0.95 over exact polygon IoU (shapely) as the other three. **Without doing this for both YOLO arms there is no valid five-way comparison.** |

### Why the metric had to be replaced

Each framework ships a different metric, and they do not agree:

- **Ultralytics** — its own OBB mAP50-95 with ProbIoU-based matching.
- **MMRotate 0.x** — DOTA eval is VOC-style AP at **IoU 0.5 only**. No 50:95 number exists.
- **Oriented-DETR** — DOTA-devkit VOC07 mAP50.
- **RF-DETR OBB** — its eval callback scores the **axis-aligned envelope** of the predicted corners (`coco_eval.py`: *"the axis-aligned envelope of the 4 OBB corners is used"*). It is not a rotated-IoU metric at all and is systematically optimistic.

Reporting those four side by side would report four metrics, not four models —
and the gap between metric implementations (1–3 mAP) is the same order as the
architectural effect being measured. Every arm therefore dumps predictions in
one common JSON format (absolute pixel polygons in the **original** image
frame) and is scored by the same function.

`analysis/comparison/rescore_baseline_yolo.py` produces the YOLO baseline's
number under that shared metric. **Without running it there is no valid
comparison** — the existing `model.val()` numbers are not on the same scale.

## 2. What cannot be held constant

### 2.1 Search space

The YOLO template searches 17 hyperparameters. They do not all exist elsewhere.

| YOLO template param | Strip R-CNN | Oriented-DETR | RF-DETR OBB |
|---|---|---|---|
| `optimizer` (SGD/Adam/AdamW) | ✅ searched | ❌ AdamW-only | ❌ AdamW-only |
| `lr0` | ✅ | ✅ | ✅ |
| `lrf` (final-LR fraction) | ✅ → `min_lr_ratio` (cosine) | ⚠️ → `lr_drop_frac` (**step**, not cosine) | ⚠️ → `lr_drop_frac` (**step**) |
| `momentum` | ⚠️ → `momentum` (SGD) or `betas[0]` (Adam) | ❌ n/a | ❌ n/a |
| `weight_decay` | ✅ | ✅ | ✅ |
| `warmup_epochs` | ⚠️ → `warmup_iters` | ❌ | ✅ |
| `warmup_momentum`, `warmup_bias_lr` | ⚠️ → `warmup_ratio` (one param, not two) | ❌ | ❌ |
| `box`, `cls`, `dfl` (loss weights) | ❌ **no two-stage counterpart** | ⚠️ → `set_cost_class`, `cls_loss_coef` | ⚠️ → `cls_loss_coef` |
| `imgsz` | ✅ (pipeline `Resize` scale) | ✅ | ⚠️ must be divisible by 56 → **336/616/840/1008**, not 320/640/832/1024 |
| `batch` | ✅ (smaller: two-stage memory) | ✅ (smaller: DETR memory) | ✅ + `grad_accum_steps` |
| `hsv_h/s/v` | ❌ not in MMRotate 0.x rotated pipeline | ✅ **same meaning** (YOLOv5 port) | ❌ not exposed |
| `translate`, `scale`, `shear`, `perspective` | ❌ | ✅ **same meaning** | ❌ |
| `mosaic`, `mixup` | ❌ | ✅ `mosaic` (no `mixup`) | ❌ |
| `flipud`, `fliplr` | ✅ (`RRandomFlip`) | ✅ | ❌ |
| — | + `rotate_ratio` (`PolyRandomRotate`) | + `degrees`, `lr_backbone_mult`, `clip_max_norm` | + `lr_encoder_mult`, `ema_decay` |
| **Total searched** (YOLO baseline: **17**) | **12** | **20** | **10** |

Consequences to state in any write-up:

- **Oriented-DETR is the closest match to the YOLO baseline's search space.** Its augmentation is a direct YOLOv5 port, so `hsv_*`, `translate`, `scale`, `shear`, `perspective`, `mosaic`, `flipud`, `fliplr`, `degrees` carry the same meaning.
- **RF-DETR-OBB receives the least tuning (10 axes, no augmentation search at all).** Its OBB pipeline resizes to a fixed square and exposes no per-augmentation knobs. Do not present it as having received equal tuning.
- 10 trials over 10 axes is not the same search density as 10 trials over 20. If per-arm tuning parity matters to the claim, scale `N_TRIALS` per arm rather than holding it fixed — the notebooks hold it fixed by default so the *compute* budget is comparable instead.

### 2.2 Pretraining — the largest confound

| Arm | Initialisation |
|---|---|
| YOLO11m-OBB | DOTAv1 OBB-pretrained (`yolo11m-obb.pt`) |
| YOLO26m-OBB | DOTAv1 OBB-pretrained (`yolo26m-obb.pt`) |
| Strip R-CNN-S | ImageNet-300ep backbone, detector head from scratch |
| Oriented-DETR R50 | COCO-pretrained (detection, axis-aligned) |
| RF-DETR-Medium OBB | COCO-pretrained axis-aligned + **freshly-initialised angle output** |

These are not equivalent starting points, and on a ~2.6k-image dataset the
initialisation gap plausibly exceeds the architecture gap. Only the two YOLO
arms start from *oriented* detection pretraining. **This is the single most
important caveat in the comparison** and cannot be removed without
pretraining all five identically, which is out of scope for this project's
compute budget.

### 2.3 Angle representation

Genuinely architectural — this is a difference you *want* to measure, not
control for:

| Arm | Representation | Boundary-discontinuity handling |
|---|---|---|
| YOLO11m-OBB | `(cx, cy, w, h, θ)` + ProbIoU loss | ProbIoU (Gaussian) |
| YOLO26m-OBB | `(cx, cy, w, h, θ)`, direct regression (DFL removed) | **Unverified** — whether OBB's ProbIoU rotation loss survives the DFL removal isn't confirmed from public Ultralytics docs as of 2026-08-20. Check the trainer's actual loss composition on first run rather than assuming parity with YOLO11m here. |
| Strip R-CNN-S | le90 `(cx, cy, w, h, θ)`, two-stage RoI | standard rotated smooth-L1 |
| Oriented-DETR | **point-axis** — no box regressed; the quad is derived at postprocess (`engine.poly2bbox`) | representation sidesteps angle regression entirely |
| RF-DETR OBB | `(cx, cy, w, h, θ)` | KLD regression loss, GWD matching cost |

### 2.4 Environments are mutually exclusive

Strip R-CNN needs MMRotate 0.x: `mmcv-full`, `mmdet` 2.x, torch ≤ 1.13,
Python ≤ 3.10. RF-DETR needs a modern torch. **They cannot share a runtime.**
Recent Colab runtimes ship Python 3.11+/torch 2.x, where `mmcv-full` has no
wheel and its source build fails — expect to need an older-Python runtime for
the Strip R-CNN arm specifically.

### 2.5 RF-DETR OBB is preview code

`roboflow/rf-detr` PR #921 is **open, not merged** (opened 2026-04-03, updated
2026-08-11; latest release 1.9.3 has no OBB). The notebook pins commit
`6fb5051c9c1c`. Two upstream defects are patched in-notebook:

1. **`build_dota` never passes `class_names`**, so `DotaDetection` falls back to the 15 hardcoded DOTA v1.0 categories. With 7 durian classes, *every* annotation lands outside `class_to_idx` and is silently dropped — training completes on empty targets with a plausible loss curve. Patched, and asserted at dataset construction.
2. **`predict()` returns a supervision `Detections`**, which has no rotated-box field, so `corners` are discarded. The notebook calls `PostProcess(oriented=True)` directly instead.

Report this arm as a preview result, or drop it. It is the natural candidate
to cut if reviewers press on equal footing.

## 3. Data encoding per arm

Same folds, different on-disk transcription (coordinate denormalisation and
file naming only — no filtering, clipping, or reordering):

| Arm | Format | Writer |
|---|---|---|
| YOLO | txt path lists, normalised polygons | template's `make_folds_files` |
| Strip R-CNN | DOTA `labelTxt/` + `images/`, absolute px | `yolo_obb_to_dota` |
| Oriented-DETR | COCO JSON with **8-value polygon in `bbox`** (not `[x,y,w,h]`) | `yolo_obb_to_coco` |
| RF-DETR OBB | DOTA `labelTxt/` + `images/`, absolute px | `yolo_obb_to_dota` |

Two encoding gotchas, both handled:

- Class names contain spaces (`"Algal spot"`); DOTA's format is whitespace-delimited. Names are underscored on disk and mapped back when reading predictions.
- Class indices: this project is 0-based; Oriented-DETR's `detect()` uses 1-based labels (`class_id = i + 1`). The prediction dump subtracts 1. Getting this wrong shifts every class by one and yields a plausible-looking but entirely wrong per-class table.

## 4. Cells not yet executed end-to-end

Written against the upstream sources cited above but **not run** — no GPU in
the authoring environment. Verified locally: `src/obb_common.py`,
`src/obb_eval.py` (exact polygon IoU against analytic values; perfect
predictions → 1.0000; empty → 0.0000; fold partition), and Python syntax of
all five notebooks. `YOLO11mOBB_...py` and `YOLO26mOBB_...py` carry over the
already-fixed, already-debugged template logic verbatim (absolute-`PROJECT`
assert, fail-loud fold errors, degenerate-study guard), so they don't carry
the same "never run against the real upstream API" risk as the other three —
their open questions are narrower and YOLO26-specific (below).

Verify on first GPU run, in this order:

1. **Strip R-CNN** — the config path `configs/strip_rcnn/strip_rcnn_s_fpn_1x_dota_le90.py` and the `roi_head.bbox_head.num_classes` override key. Confirm `predict_fold`'s `(n, 6)` assertion passes.
2. **Oriented-DETR** — that `--options` accepts every key passed in `make_args` (DINO-style configs reject unknown keys), and that `poly2bbox` returns 8 values.
3. **RF-DETR** — that `RFDETRMedium(oriented=True, ...)` is accepted, and that `model.model.model` is the right handle for the inner `nn.Module`.
4. **YOLO26m-OBB** — whether `dfl` measurably affects the loss/mAP curve across trials (§"YOLO26-specific caveats" above), and that `model.val()`'s `results.csv` still exposes `metrics/mAP50-95(B)` (or one of `read_final_map_from_results`'s other candidate column names) unchanged from YOLO11. Watch the first few epochs for the decreasing-mAP symptom reported in Ultralytics issue #23548.

Each of these is guarded by an assertion that fails loudly rather than
producing a wrong number quietly.

## 5. Relationship to Phase 7.5

**This comparison does not address the open Phase 7.5 problem.** The
background shortcut (Test C2: masked-image hit rate 61.7% → 83.7%) is a
data/augmentation artifact — the leading suspect is
`analysis/dataset/phase4_background_swap_augment.py`'s unfeathered binary-mask
paste. A different architecture will learn that shortcut too, and a stronger
backbone will likely learn it *more* efficiently.

Run the C1/C2 confound checks against any winner here before treating it as an
improvement, and do not let a better mAP50-95 in this table substitute for
that. Per CLAUDE.md, Phase 8 stays blocked until Phase 7.5 resolves.
