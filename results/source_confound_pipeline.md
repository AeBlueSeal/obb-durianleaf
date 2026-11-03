# Source-confound — pipeline file guide

What each script does, in run order. See `implementation_plan.md` for the
original 8-phase plan and `source_confound_scoping.md` for the Phase 3
Option B decision (Rhizoctonia field-data gap scoped, not filled).

**Current status:** Phases 1–7 complete (`v4_confound_fix` run, weights at
`runs/v4_confound_fix/train/weights/best.pt`). Phase 7 re-verification found
a **regression, not a pass** — the background shortcut (Test C2) got worse
and spread to 3 previously-clean classes. Now at **Phase 7.5** (diagnosis,
below) — don't treat Phase 4's output as "fixed" until that resolves.

## Phase 1 — Dedup / grouping

**`analysis/dataset/phase1_build_groups.py`** — local, CPU
Finds images that must never be split across train/val/test: near-duplicates
(perceptual hash, Hamming ≤ 5), field-photo burst sessions (same filename
prefix within a 10s window), and curated sequence runs. Merges all of these
into split-atomic groups via union-find.
Input: `data/datasetv3_2/**`. Output: `analysis/dataset/outputs/resplit/phase1_groups.{csv,json}`, `phase1_report.txt`.

## Phase 2 — Grouped stratified re-split

**`analysis/dataset/phase2_resplit.py`** — local, CPU
Reads Phase 1's groups, computes each group's (class, source) signature from
its OBB labels, and greedily assigns whole groups to train/val/test (80/10/10)
so no group — and therefore no duplicate/session/sequence — crosses a split
boundary. Materializes the new dataset.
Input: `phase1_groups.json` + `data/datasetv3_2/**`. Output: `data/datasetv3_2_resplit/{train,val,test}/{images,labels}`, `data/dataset_resplit.yaml`, `analysis/dataset/outputs/resplit/phase2_report.txt`.
Env: `PHASE2_SKIP_COPY=1` for a dry run (just prints the split plan, doesn't copy files).

## Phase 3 — Field-data gap + break the resolution fingerprint

**`analysis/dataset/phase3_field_coverage_report.py`** — local, CPU
Reports per-class curated vs. field instance/image counts on the new split;
flags any class with fewer than 50 field instances. Only Rhizoctonia (n=8)
is flagged — this is what justified the Option B scoping decision.

**`analysis/dataset/phase3_augment_curated.py`** — local, CPU
Curated images were all a fixed 400×400 — a zero-effort "this is curated"
fingerprint a model can learn instead of the actual lesion. This resizes/crops
each curated image to a random size from a pool (320–768px), removing the
fixed signature. Mutates `data/datasetv3_2_resplit` in place.

## Phase 4 — Break the background shortcut

**`analysis/dataset/phase4_background_negatives.py`** — local, CPU
Crops background-only patches (no leaf) out of existing field training images
using the OBB polygons as an exclusion mask, saves them with empty label
files (`bgneg_*`). Gives the model real "no lesion here" examples instead of
never seeing bare background.

**`analysis/dataset/phase4_background_swap_augment.py`** — local, CPU
Takes Algal-spot/Other leaf crops (the two classes Test C2 showed were
learnable from background alone) and composites them onto the `bgneg_*`
background pool, so the same lesion appears against varied backgrounds
(`bgswap_*`).
⚠️ **Prime suspect for the Phase 7.5 regression.** Current implementation is
a hard binary-mask paste — no feathering, no color/noise matching at the
leaf boundary. The visible seam this leaves is suspected to be a stronger,
more learnable shortcut than the natural background correlation it was
meant to remove (post-retrain C2 hit rates hit 5.5–5.6× chance, and the
regression spread to Healthy/Phomopsis/Rhizoctonia too). Don't reuse this
output as-is until Phase 7.5 Step 3 (feathered/Poisson blending + matched
noise/compression stats + wider pool + stochastic online swap) lands and
the Step 2 ablation confirms it's actually the cause.

**`analysis/dataset/phase4_hard_negative_mining.py`** — **Colab/GPU**
Runs the current `models/yolo11m-obb.pt` on the `bgneg_*` background-only
crops. Any Algal-spot/Other detection on an image with no leaf is a confirmed
false positive — proof the shortcut is live in the current model. Saves those
crops as extra hard negatives (`hardneg_*`). Needs the model loaded for
inference, so run wherever you have a GPU.

## Phase 5 — Standardize preprocessing

**`analysis/dataset/phase5_strip_exif.py`** — local, CPU
Re-saves every image in `data/datasetv3_2_resplit` via PIL, dropping EXIF
camera-model metadata (field photos carried an iPhone/DNY-NX9 tag, curated
images didn't — another zero-effort source fingerprint).

## Phase 5.5 — Re-sequence E-IRFS (class-imbalance sampling)

**`src/data_loader.py`** — canonical implementation, shared by the
Drive-mounted retrain notebooks (`notebooks/train.ipynb`,
`notebooks/v4_confound_fix_train.ipynb`) via
`from src.data_loader import YOLOEIRFSCappedDataset`. It overrides
`ultralytics.data.build.YOLODataset` with a repeat-factor sampler
(`YOLOEIRFSCappedDataset`) that caps specific classes' sampling weight.
`notebooks/Template_Clean_E-IRFS_optuna_datasetv3.ipynb` and
`notebooks/SAHI_E-IRFS_optuna_datasetv3.ipynb` don't mount Drive and can't
rely on that import path, so they keep their own copy of the same
implementation — kept in sync by hand, not a standalone script to run.

E-IRFS oversamples *images*, not *content*. Rhizoctonia is 97.6% curated from
a near-uniform cluster (fixed 400×400 pre-Phase-3), so re-running E-IRFS
unchanged would repeat that narrow source cluster and reinforce the confound
instead of fixing genuine imbalance.

- **Do not reapply E-IRFS on Rhizoctonia until Phase 3 is resolved.**
  Since `source_confound_scoping.md` recorded Option B (scoped reporting, no
  new field data collected), cap E-IRFS's repeat factor for Rhizoctonia
  specifically rather than letting it fully oversample it.
  `src/data_loader.load_capped_class_names()` derives the capped-class list
  from `analysis/dataset/outputs/resplit/phase3_scoping.csv` (Phase 3's own
  field-coverage verdict) when that file is present, falling back to
  `{"Rhizoctonia"}` otherwise — so a re-split that changes which classes are
  field-data-starved is picked up automatically instead of requiring a
  hand-edit. **Healthy is deliberately not capped**: it has plenty of field
  instances (179) and its AP gap was a background/preprocessing shortcut
  already addressed by Phases 3-5, not a data-starvation problem.
  ⚠️ **Open discrepancy — revisit during Phase 7.5.** `implementation_plan.md`'s
  original Phase 5.5 gated *both* Rhizoctonia and Healthy behind the Phase 3
  decision, and its Phase 7 target table tracks a joint "Rhizoctonia/Healthy
  AP gap after resampling" row. Healthy's exemption here assumed Phases 3-5
  were sufficient — but post-retrain, Healthy's curated−field AP gap
  *widened* (0.4015 → 0.5625) and it's now also flagged in C2. Phase 4.2
  doesn't touch Healthy directly, so this is likely a separate issue from
  the background-swap seam — but confirm via the Phase 7.5 ablation whether
  it's an E-IRFS interaction before treating "not capped" as final.
- **Recompute E-IRFS's frequencies on the new split** (`data/dataset_resplit.yaml`),
  not the original — Phase 1/2 regrouping and Phase 4.1's background negatives
  changed the image/instance counts the repeat-factor formula depends on.
- **Verify label-free background negatives (`bgneg_*`) get a neutral repeat
  factor (≈1)** — repeat-factor formulas are keyed on category frequency, and
  negatives have no category, so confirm Phase 4.1's negative ratio doesn't
  silently drift once sampling is applied.
- **Leave it as-is for Phomopsis, Anthracnose, and Phytophtora** (~99% field-sourced)
  — their imbalance is real and not source-tainted, no reason to cap them.
- **Watch the console for the cap warning.** If none of the requested capped
  class names match this dataset's classes (renamed class, wrong yaml,
  typo), the cap silently used to no-op with zero signal; it now prints a
  `WARNING: E-IRFS cap requested for ... but none of these class names
  matched ...` line instead. Absence of both this warning and a `[CAPPED]`
  tag in the printed class stats means the cap isn't engaging — don't trust
  the run until it is.
- **Re-verify after retraining** (Phase 7) — re-run Tests C1/C2 with specific
  attention to whether Rhizoctonia's numbers move again once E-IRFS is back
  in the loop; a re-widening gap means E-IRFS is re-amplifying the source
  shortcut.

## Phase 6 — Retrain

**Before running the scripts below**, see `implementation_plan.md` §6.1 for
what to re-search vs. reuse from the original HPO: augmentation params
(mosaic probability, HSV jitter, the new background-swap ratio) and the
`cls` loss weight need a fresh 3-fold CV pass on the new split; optimizer,
base LR, and momentum don't. This isn't wired up as a standalone script yet,
so it's easy to skip if you're only following this file.

**`scripts/base/train.py`** — **Colab/GPU**
Ultralytics YOLO OBB training entry point. Defaults now point at
`data/dataset_resplit.yaml` and add `--patience 50` (early stop, since epoch
count was never tuned). Deploy `runs/<name>/weights/best.pt`, not `last.pt`.

**`scripts/experiments/v4_confound_fix/config.yaml`** — **Colab/GPU**
Experiment config for `scripts/run.py --experiment v4_confound_fix`: same
retrain, wired through this repo's experiment-runner instead of calling
`train.py` directly.

## Phase 7 — Re-verify

**`analysis/dataset/source_confound_A.py`** — local, CPU
Cramér's V between class and image-source (curated/field). Dataset-property
check, doesn't need a trained model.

**`analysis/dataset/source_confound_B1.py`** — local, CPU
Checks curated-image resolution uniqueness and EXIF camera tags — confirms
Phases 3 and 5 actually removed the two fingerprints.

**`analysis/dataset/source_confound_D.py`** — local, CPU
Perceptual-hash sweep across the new train/val/test images — confirms Phase 1/2
grouping eliminated cross-split near-duplicate leakage.

All three take `CONFOUND_DATA_YAML` / `CONFOUND_OUTPUT_DIR` env vars so they
can be re-run against the new split without overwriting the original
baseline outputs, e.g.:
```
CONFOUND_DATA_YAML=data/dataset_resplit.yaml \
CONFOUND_OUTPUT_DIR=analysis/notebooks/outputs/source_confound_after_fix \
python analysis/dataset/source_confound_A.py
```

**`analysis/evaluation/source_confound_C1.py`** — **Colab/GPU**
Baseline per-source (curated vs. field) AP gap for each class, using the
*old* pre-fix model (`models/yolo11m-obb.pt`).  Run this once before retraining
to get the "before" numbers.

**`analysis/evaluation/source_confound_C2.py`** — **Colab/GPU**
Baseline background-only hit-rate for Algal spot/Other, same old model.

**`notebooks/v4_confound_fix_C1.py`** — **Colab/GPU**  *(new — after-fix)*
Same per-source AP breakdown as `source_confound_C1.py`, but pointed at the
retrained weights (`runs/v4_confound_fix/train/weights/best.pt`).  Drop-in
replacement — run this instead of the baseline version after Phase 6 completes.
Saves outputs to `runs/v4_confound_fix/test/C1_per_source_ap/`.

**`notebooks/v4_confound_fix_C2.py`** — **Colab/GPU**  *(new — after-fix)*
Same background-only probe as `source_confound_C2.py`, but pointed at the
retrained weights.  Saves outputs to
`runs/v4_confound_fix/test/C2_background_probe/`.

**Status:** A/B1/D and the after-fix C1/C2 have already been run once
(results in `results/source_confound_fix_phase7.md`). A/B1/D passed; C1/C2
got worse, not better — see Phase 7.5. Re-running `v4_confound_fix_C1.py` /
`_C2.py` again during the Phase 7.5 ablation will overwrite
`runs/v4_confound_fix/test/C1_per_source_ap/` and `.../C2_background_probe/`
— copy those outputs elsewhere first if you want to keep the regression
numbers for comparison.

---

## Phase 7.5 — Diagnose the `v4_confound_fix` regression (current status: ACTIVE)

Phase 7 confirmed B1, D, val-loss logging, and checkpoint selection are
fixed, but the confound itself got worse: C1's curated−field gap widened
(0.067 → 0.092; Healthy 0.4015 → 0.5625) and C2's masked-image hit rate rose
from 61.7% to 83.7%, now flagging 5/7 classes instead of 2/7. Leading
hypothesis: the hard binary-mask paste in `phase4_background_swap_augment.py`
(⚠️ flagged under Phase 4 above) left a visible seam that's a stronger, more
learnable shortcut than the natural background correlation it replaced.

**Step 1 — Visual audit** *(ad hoc, no script yet, ~15 min, no GPU)*
Pull 15–20 `*swap*.jpg` images from `phase4_background_swap_augment.py`'s
output directory and inspect the leaf boundary for a visible edge/halo,
lighting mismatch, or compression discontinuity. See `implementation_plan.md`
Phase 7.5 Step 1 for the plotting snippet. *To do: promote to
`analysis/dataset/phase7_5_visual_audit.py` if this gets re-run more than once.*

**Step 2 — Ablation retrain** *(config not yet created)* — **Colab/GPU**
Re-run Phase 6 with everything from `v4_confound_fix` unchanged (Phase 1/2
split, Phase 4.1 negatives, Phase 5.5 E-IRFS as currently configured) except
`phase4_background_swap_augment.py` disabled. Suggested: duplicate
`scripts/experiments/v4_confound_fix/config.yaml` as
`scripts/experiments/v4_confound_fix_ablation_no_bgswap/config.yaml` with the
swap-augmentation flag off, then re-run `notebooks/v4_confound_fix_C1.py` /
`_C2.py` against `runs/v4_confound_fix_ablation_no_bgswap/train/weights/best.pt`,
saving to sibling `.../test/C1_per_source_ap/` / `.../C2_background_probe/`
dirs so the Phase 7 regression numbers aren't overwritten.
```text
C2 hit rate drops back toward ~61.7% or lower
   → confirms background-swap augmentation is the cause. Proceed to Step 3.
C2 hit rate stays elevated even with the augmentation off
   → cause is elsewhere (E-IRFS repeat interaction, Phase 4.1 negative-pool
     composition, or Phase 5 preprocessing) — re-open Phase 5.5 and Phase 4.1
     before re-attempting 4.2.
```

**Step 3 — Fix the augmentation** *(once Step 2 confirms the cause)*
Edit `phase4_background_swap_augment.py`'s `swap_background()` to use
feathered/Poisson blending instead of a hard mask, match noise/JPEG-compression
statistics between leaf and background, widen the background pool, and
switch from a pre-generated fixed `bgswap_*` set to a stochastic online swap
(fresh background per epoch) so E-IRFS can't repeat the same artifact
verbatim. Re-run the ablation ladder: augmentation off → confirm baseline-ish
C2 → fixed augmentation on → confirm C2 improves *beyond* the off-state.

**Step 4 — Full re-verify** — re-run all of Phase 7's scripts again. Only
proceed to Phase 8 once C2 is below the *original* 61.7%/3.1×/2.9× baseline,
not just below the `v4_confound_fix` regression numbers.

---

## Phase 8 — Final reporting (blocked on Phase 7.5)

**`results/source_confound_fix_phase7.md`** — before/after table for Tests
A/B1/D (done, fixed) and C1/C2 (done, but the after-fix numbers are a
regression — see Phase 7.5).

**`results/source_confound_fix_phase8_template.md`** — fill in per-class AP
+ the mandatory confound-disclosure paragraph once Phase 7.5 lands on a
version where C2 is meaningfully below the original 61.7% baseline. Do not
publish numbers from this template using the current `v4_confound_fix` run
— those numbers show the shortcut got worse, not better.

---

## Quick reference: where to run what

| Local (CPU) | Colab (GPU) |
|---|---|
| phase1_build_groups.py | phase4_hard_negative_mining.py |
| phase2_resplit.py | notebooks/train.py |
| phase3_field_coverage_report.py | notebooks/v4_confound_fix_train.py |
| phase3_augment_curated.py | notebooks/v4_confound_fix_C1.py (after-fix) |
| phase4_background_negatives.py | notebooks/v4_confound_fix_C2.py (after-fix) |
| phase4_background_swap_augment.py ⚠️ regression suspect | analysis/evaluation/source_confound_C1.py (baseline) |
| src/data_loader.py (imported, not run standalone) | analysis/evaluation/source_confound_C2.py (baseline) |
| phase5_strip_exif.py | Template_Clean_E-IRFS_optuna_datasetv3.ipynb (Phase 5.5) |
| source_confound_A.py | SAHI_E-IRFS_optuna_datasetv3.ipynb (Phase 5.5) |
| source_confound_B1.py | v4_confound_fix_ablation_no_bgswap/config.yaml (Phase 7.5 Step 2, to create) |
| source_confound_D.py | v4_confound_fix_C1.py / _C2.py re-run vs. ablation weights (Phase 7.5 Step 4) |

**Phase 7.5 visual audit** (Step 1) is currently ad hoc — no tracked script
yet, run it inline per the section above.
