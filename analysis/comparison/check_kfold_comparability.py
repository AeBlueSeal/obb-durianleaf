"""Guard the invariants that make the six K-fold arms' numbers comparable.

This project keeps re-discovering the same bug class: one arm quietly drifts
away from the shared protocol (a different metric, a different sampler, a
different evaluation operating point, a different fold split) and the
resulting table compares plumbing rather than architectures. Each time it has
been found by hand, after the runs were already archived.

This script checks the invariants statically, in seconds, so the next drift
is caught before a 9-fold x 250-epoch run is spent on it.

    python analysis/comparison/check_kfold_comparability.py

Exits non-zero on any violation. It is deliberately a source-text audit, not
an import: four of the six arms pin mutually incompatible torch/mmcv/Python
versions and cannot be imported in one interpreter -- which is exactly why
they drift in the first place.

What it does NOT check: anything that only shows up at runtime (a framework
silently ignoring a config field, a sampler that attaches but never fires).
Those need a live run; see results/eirfs_noop_nonyolo_arms.md for the
canonical example of one that a static check would have missed.
"""
from __future__ import annotations

import random
import re
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "notebooks"

# fold-training notebooks whose test numbers land in the comparison table
ARMS = {
    "YOLO11m":      "YOLO11mOBB_E-IRFS_kfold_datasetv3.py",
    "YOLO26m":      "YOLO26mOBB_E-IRFS_kfold_datasetv3.py",
    "OrientedDETR": "OrientedDETR_E-IRFS_kfold_datasetv3.py",
    "RFDETR_OBB":   "RFDETR_OBB_E-IRFS_kfold_datasetv3.py",
    "StripRCNN":    "StripRCNN_E-IRFS_kfold_datasetv3.py",
    "S2ANet":       "S2ANet_E-IRFS_kfold_datasetv3.py",
}
# E-IRFS-ablation twins: same protocol, so the ablation is an ablation
ABLATION = {
    "YOLO11m_NoEIRFS": "YOLO11mOBB_NoEIRFS_kfold_datasetv3.py",
    "YOLO26m_NoEIRFS": "YOLO26mOBB_NoEIRFS_kfold_datasetv3.py",
    "S2ANet_NoEIRFS":  "S2ANet_NoEIRFS_kfold_datasetv3.py",
}
# notebooks that run the hyperparameter search feeding a K-fold arm
HPO_SIBLINGS = {
    "YOLO11m_optuna": "YOLO11mOBB_E-IRFS_optuna_datasetv3.py",
    "YOLO26m_optuna": "YOLO26mOBB_E-IRFS_optuna_datasetv3.py",
}

failures: list[str] = []


def fail(arm: str, msg: str) -> None:
    failures.append(f"[{arm}] {msg}")


def src(name: str) -> str:
    return (NB / name).read_text()


def code(name: str) -> str:
    """``src`` with whole-line comments removed.

    Needed wherever a check greps for a BAD value. The arms that fixed such a
    bug typically carry a comment explaining the old value it replaced (e.g.
    "the base config sets test_cfg.rcnn.score_thr=0.05, which is applied
    inside the rotated NMS ..."), and a naive grep over the raw text then
    flags that prose as if it were live config -- reporting the bug loudest
    precisely where it was fixed and the reasoning was written down. Only
    full-line comments are stripped; a trailing comment on a code line keeps
    its code, which is all these checks look at.
    """
    return "\n".join(ln for ln in src(name).splitlines()
                     if not ln.lstrip().startswith("#"))


# --------------------------------------------------------------------------
# 1. Shared constants -- the sampler and the training budget must match
# --------------------------------------------------------------------------
def const(text: str, name: str) -> str | None:
    m = re.search(rf"^{re.escape(name)}\s*=\s*(.+?)\s*(?:#.*)?$", text, re.M)
    return m.group(1).strip() if m else None


SHARED_CONSTANTS = {
    "SEED": "42",
    "K_FOLDS": "9",
    "TRAIN_EPOCHS": "250",
}
# E-IRFS knobs only apply where E-IRFS is in play
EIRFS_CONSTANTS = {"EIRFS_THRESHOLD": "0.0001"}

for arm, f in {**ARMS, **ABLATION}.items():
    text = src(f)
    for k, want in SHARED_CONSTANTS.items():
        got = const(text, k)
        if got != want:
            fail(arm, f"{k} is {got!r}, every arm must use {want!r}")
    if "NoEIRFS" not in arm:
        for k, want in EIRFS_CONSTANTS.items():
            got = const(text, k)
            if got != want:
                fail(arm, f"{k} is {got!r}, expected {want!r}")

# --------------------------------------------------------------------------
# 2. Every arm reports the SHARED metric, not its framework's own
# --------------------------------------------------------------------------
for arm, f in {**ARMS, **ABLATION}.items():
    text = src(f)
    if "evaluate_obb(" not in text:
        fail(arm, "never calls evaluate_obb() -- its test score is on its "
                  "framework's own metric, not the shared one")
    if "precision_recall_f1(" not in text:
        fail(arm, "never calls precision_recall_f1() -- its P/R/F1 are at a "
                  "different operating point from the other arms'")
    for key in ('"test_mAP50-95"', '"test_mAP50"', '"test_mAP75"'):
        if key not in text:
            fail(arm, f"does not record {key} in its per-fold rows")
    # the YOLO arms may ALSO keep Ultralytics' own metric, but only under a
    # name that cannot be mistaken for the comparable one
    if "box.map" in text and "ultra_" not in text:
        fail(arm, "reads Ultralytics' box.map without an `ultra_*` label -- "
                  "that number is not cross-arm comparable")

# --------------------------------------------------------------------------
# 3. One evaluation operating point (conf/score_thr = 0.001) everywhere
# --------------------------------------------------------------------------
for arm, f in {**ARMS, **ABLATION}.items():
    if re.search(r"score_thr\s*=\s*0\.05", code(f)):
        fail(arm, "score_thr=0.05 truncates the P/R curve evaluate_obb() "
                  "integrates -- every arm evaluates at 0.001")

# The mmdet/MMRotate-family arms build their config on top of an upstream
# DOTA config that sets score_thr=0.05, applied inside nms *before* the
# worker's own predict_fold(score_thr=0.001) ever sees a detection. That
# value is INHERITED, so it is invisible to the grep above -- absence of
# "0.05" in the notebook proves nothing. Each must override it explicitly.
for arm in ("StripRCNN", "S2ANet"):
    if not re.search(r"score_thr\s*=\s*0\.001", code(ARMS[arm])):
        fail(arm, "does not explicitly set score_thr=0.001; its base config "
                  "inherits the stock 0.05, which caps recall and understates "
                  "this arm's AP against the arms that evaluate at 0.001")

# --------------------------------------------------------------------------
# 3b. Every arm trains on the dataset's background-only images
# --------------------------------------------------------------------------
# analysis/dataset/phase4_background_negatives.py adds label-free images to
# train/ deliberately, and Ultralytics treats an empty label file as background,
# so the YOLO arms train on them. Both mmdet-family arms discarded them by
# default -- in different ways, and both silently. As with the inherited
# score_thr above, the bad value is a DEFAULT, so absence of the fix string
# from the notebook is itself the violation; there is nothing to grep for.
BACKGROUND_FLAGS = {
    # mmdet 1.x: LoadAnnotations defaults to skip_img_without_anno=True, whose
    # None return makes CustomDataset.__getitem__ resample a DIFFERENT image
    "S2ANet": "skip_img_without_anno=False",
    "S2ANet_NoEIRFS": "skip_img_without_anno=False",
    # mmdet 2.x / MMRotate: CustomDataset defaults to filter_empty_gt=True,
    # which drops the image from the dataset outright
    "StripRCNN": "filter_empty_gt=False",
}
for arm, flag in BACKGROUND_FLAGS.items():
    notebook = {**ARMS, **ABLATION}[arm]
    if flag not in code(notebook):
        fail(arm, f"does not set {flag}; its framework then discards the "
                  f"background-only images that phase4_background_negatives.py "
                  f"adds to train/ on purpose, which the YOLO arms DO train on")

# --------------------------------------------------------------------------
# 4. Optuna pruning -- REPORTED, not enforced
# --------------------------------------------------------------------------
# Leaving pruning off is a standing project decision, so "no should_prune()"
# is not a violation here. But it is not uniform: Strip R-CNN and S2A-Net do
# prune, so they search under a different HPO budget from the arms that run
# every trial to completion. That is a real, accepted difference, and this
# block exists to keep it visible rather than to fail on it. If it ever
# becomes uniform in either direction, this can turn back into an assertion.
pruning = {}
for arm, f in {**ARMS, **HPO_SIBLINGS}.items():
    text = src(f)
    if "PercentilePruner" not in text:
        continue
    pruning[arm] = bool(re.search(r"^\s*if trial\.should_prune\(\):", text, re.M))

# --------------------------------------------------------------------------
# 5. One fold's failure must not destroy the whole run's results
# --------------------------------------------------------------------------
for arm, f in {**ARMS, **ABLATION}.items():
    text = src(f)
    # Scope the search to the FINAL-training loop. Looking for "fold_errors"
    # anywhere in the file gives a false negative: every arm's Optuna
    # objective() already has its own error handling far above, which is a
    # different loop and does not protect the 9-fold run.
    i = text.find("fold_rows = []")
    if i < 0:
        fail(arm, "no `fold_rows = []` -- cannot locate the final-training loop")
        continue
    final_loop = text[i:]
    if "fold_errors" not in final_loop or 'str(e)' not in final_loop:
        fail(arm, "the final K-fold training loop has no per-fold error "
                  "handling -- one fold's exception kills the whole run and "
                  "no CSV/summary is written")

# --------------------------------------------------------------------------
# 6b. Every arm logs the numbers src/obb_logging.py exists to persist
# --------------------------------------------------------------------------
# Added alongside the 2026-09 logging pass -- so the next arm added to this
# comparison does not quietly log less than the other six. A violation here
# means run_env.json / fold_composition.csv / per-epoch curves / the
# per-class table are missing for that arm, not that the reported mAP is
# wrong -- but those are exactly the numbers this project keeps having to
# reconstruct after the fact (see results/kfold_comparability_audit.md).
for arm, f in {**ARMS, **ABLATION}.items():
    text = src(f)
    if "write_run_env(" not in text:
        fail(arm, "never calls write_run_env() -- no run_env.json (platform/"
                  "GPU/framework-version/config record) for this run")
    if "write_fold_composition(" not in text:
        fail(arm, "never calls write_fold_composition() -- no per-fold "
                  "class-balance audit trail")
    if "collect_epoch_log(" not in text:
        fail(arm, "never calls collect_epoch_log() -- this arm's own "
                  "per-epoch loss/lr curve is never persisted")
    if "write_per_class_tables(" not in text:
        fail(arm, "never calls write_per_class_tables() -- no per-class "
                  "AP/P/R/F1 table across folds, the single most useful "
                  "missing number for this comparison")

# --------------------------------------------------------------------------
# 6. The YOLO arms' local fold split must equal src/obb_common's
# --------------------------------------------------------------------------
sys.path.insert(0, str(REPO))
try:
    import src.obb_common as shared
except Exception as e:  # numpy/PIL missing locally
    print(f"NOTE: skipping the fold-equivalence check ({e})")
else:
    for arm, f in {**ARMS, **ABLATION}.items():
        text = src(f)
        m = re.search(r"^def kfold_assignments.*?(?=\n(?:#|def |\n))",
                      text, re.S | re.M)
        if not m:
            continue  # this arm imports the shared one; nothing to compare
        mod = types.ModuleType(f"nb_{arm}")
        mod.random = random
        mod.SEED = 42
        exec(compile(m.group(0), f"<{f}>", "exec"), mod.__dict__)
        for n_items in (97, 400, 1234):
            for k in (3, 9):
                want = shared.kfold_assignments(n_items, k, 42, True)
                # clean run
                random.seed(42)
                if mod.kfold_assignments(n_items, k, True) != want:
                    fail(arm, f"local kfold_assignments disagrees with "
                              f"src/obb_common at n={n_items} k={k}")
                # after unrelated `random` consumption -- e.g. re-running the
                # fold cell alone in Colab without the config cell
                random.seed(42)
                for _ in range(17):
                    random.random()
                if mod.kfold_assignments(n_items, k, True) != want:
                    fail(arm, "local kfold_assignments does not reseed, so "
                              "its folds depend on prior `random` use and can "
                              "silently differ from the other arms'")
                    break
            else:
                continue
            break

# --------------------------------------------------------------------------
if pruning:
    on = sorted(a for a, v in pruning.items() if v)
    off = sorted(a for a, v in pruning.items() if not v)
    print("INFO  Optuna pruning (accepted difference, not enforced):")
    print(f"        prunes:        {', '.join(on) or '(none)'}")
    print(f"        runs in full:  {', '.join(off) or '(none)'}")
    if on and off:
        print("        -> these two groups search under different HPO budgets; "
              "see Finding 2 in\n           results/kfold_comparability_audit.md.")

if failures:
    print(f"\n{len(failures)} comparability violation(s):\n")
    for v in failures:
        print(f"  - {v}")
    print("\nSee results/kfold_comparability_audit.md for what each check means.")
    sys.exit(1)

print(f"OK -- {len(ARMS)} arms + {len(ABLATION)} ablation twins + "
      f"{len(HPO_SIBLINGS)} HPO siblings share one protocol.")
