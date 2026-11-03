# Source-confound fix — Phase 7 verification

Re-running the confound-check suite (Tests A, B1, C1, C2, D) against the
fixed dataset (`data/dataset_resplit.yaml`, produced by
`analysis/dataset/phase1_build_groups.py` → `phase2_resplit.py` →
`phase3_*` → `phase4_*` → `phase5_strip_exif.py`). Baseline numbers are
from `analysis/notebooks/outputs/source_confound/`; new numbers are from
`analysis/notebooks/outputs/source_confound_after_fix/` (re-run with
`CONFOUND_DATA_YAML=data/dataset_resplit.yaml`).

| Test | Before | After | Status |
|---|---|---|---|
| A: Cramér's V (class↔source) | 0.855 | 0.842 | ⚠️ Barely moved — expected, not a bug. Rhizoctonia is still 97.6% curated because there are only 8 field instances in the whole dataset (a real data shortage, not a splitting artifact — see `results/source_confound_scoping.md`). Every *other* class already had good source mixing and stayed that way. |
| B1: curated resolution uniqueness | 1 / 307 | **63 / 307** | ✅ Fixed. Resolution jitter + crop-then-resize (`phase3_augment_curated.py`) removed the fixed-400×400 fingerprint. |
| B1: camera EXIF fingerprint | iPhone 15 Pro / DNY-NX9 / iPhone 16 Pro Max, field-only | **`unknown` for both sources** | ✅ Fixed. `phase5_strip_exif.py` removed camera-model tags dataset-wide. |
| D: cross-split near-duplicate pairs | 17 | **0** | ✅ Fixed. Every phash near-duplicate, field session-burst, and curated sequence-run is now a union-find group locked to a single split (`phase1_build_groups.py`); verified structurally (no group spans >1 split) and empirically (Test D re-run finds zero). |
| C1: Healthy AP gap (curated − field) | 0.4015 | *pending retrain* | Requires a model trained on `data/dataset_resplit.yaml` — the current `models/yolo11m-obb.pt` was trained on the old split. Re-run `analysis/evaluation/source_confound_C1.py` with `CONFOUND_DATA_YAML=data/dataset_resplit.yaml` once retrained. |
| C1: Rhizoctonia field AP | n=8 images | *pending retrain* | Still small-N by construction (Option B, not Option A) — report with the sample size stated, per `results/source_confound_scoping.md`. |
| C2: Algal spot / Other hit-rate ratio | 3.1× / 2.9× chance | *pending retrain* | `phase4_hard_negative_mining.py` already found live evidence of this exact shortcut on the *current* model (3 Algal spot + 2 Other false positives on pure background crops) and folded those crops back into train as hard negatives, on top of the 115 background negatives + 358 background-swapped Algal spot/Other images from `phase4_background_negatives.py` / `phase4_background_swap_augment.py`. Re-run C2 after retraining to confirm the ratio drops toward ~1.0×. |

## What's verified vs. what's pending

**Verified now, without needing a retrain** (these are dataset/pipeline properties, not model behavior):
- Split-atomic grouping eliminates all cross-split leakage (Test D: 17 → 0).
- Curated images no longer have a fixed resolution/aspect-ratio signature (Test B1).
- EXIF camera fingerprint removed dataset-wide (Test B1).
- New split holds 80/10/10 with per-class share within ~1.2pp of the original
  (`analysis/dataset/outputs/resplit/phase2_report.txt`).
- Rhizoctonia's class↔source imbalance is a genuine data-volume problem, not
  a split artifact — confirmed and explicitly scoped rather than masked.

**Requires retraining `models/yolo11m-obb.pt` (or an experiment run via
`scripts/experiments/v4_confound_fix/config.yaml`) on
`data/dataset_resplit.yaml` before these can be confirmed:**
- C1 per-source AP gap shrinking (Healthy, Rhizoctonia).
- C2 background-shortcut hit-rate ratio dropping toward chance (Algal spot, Other).

Do not report the retrained model's performance as reliable until both C1
and C2 are re-run and reviewed — that's what Phase 7 is actually for.
