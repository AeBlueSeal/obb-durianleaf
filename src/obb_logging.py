"""Shared logging/durability helpers for the OBB K-fold comparison notebooks.

Notebook-kernel-side only -- unlike ``src/obb_common.py`` (numpy/PIL-only, so
every conda-env worker can import it), this module assumes pandas and a
regular Python environment, and is imported by the notebooks themselves, not
by ``src/strip_rcnn_worker.py`` / ``src/s2anet_worker.py``.

Every function here is *fail-soft*: a logging bug must never cost a fold that
just spent real GPU time training. Each public function catches its own
exceptions, prints a warning, and returns ``None`` rather than raising --
callers should treat a ``None`` return as "nothing to log this time", not as
an error to propagate. This mirrors the standing project convention of
assertions that fail loudly for *correctness* bugs (see
``analysis/comparison/check_kfold_comparability.py``) -- logging is
deliberately the opposite: best-effort, because losing a number is much
cheaper than losing a fold.

Nothing in this file changes any training config, sampler, evaluation
operating point, or metric math -- it only writes additional files beside
the ones each arm already produces.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.obb_common import read_yolo_obb_label


def _warn_soft(fn_name, exc):
    warnings.warn(f"[obb_logging.{fn_name}] logging failed, continuing "
                  f"without it: {exc!r}", stacklevel=2)


def _fail_soft(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- logging must never kill a fold
            _warn_soft(fn.__name__, e)
            return None
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


# --------------------------------------------------------------------------
# Durability primitives -- promoted verbatim from
# notebooks/S2ANet_E-IRFS_kfold_datasetv3.py, the one arm that already had to
# solve "training spans several Colab sessions and /content does not survive
# one".
# --------------------------------------------------------------------------
def durable_dir(run_root, name):
    """A directory under ``run_root`` (Drive) for one fold's artifacts."""
    d = Path(run_root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def mirror_file(src, dst):
    """Copy write-then-rename, so an interrupted copy cannot leave a
    truncated file that a later session mistakes for a finished one."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(str(src), str(tmp))
    tmp.replace(dst)
    return dst


def fmt_dt(seconds):
    """Compact h/m/s, for progress lines only."""
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


# --------------------------------------------------------------------------
# Per-epoch curve collection
# --------------------------------------------------------------------------
def _read_ultralytics_log(run_dir):
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    loss_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("loss")]
    df["loss"] = df[loss_cols].sum(axis=1) if loss_cols else np.nan
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    df["lr"] = df[lr_cols[0]] if lr_cols else np.nan
    df["seconds"] = df["time"] if "time" in df.columns else np.nan
    map_cols = [c for c in df.columns if "mAP50-95" in c]
    best_epoch = best_metric = None
    if map_cols:
        best_idx = int(df[map_cols[0]].idxmax())
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_metric = float(df.loc[best_idx, map_cols[0]])
    return df, {
        "epochs_completed": int(df["epoch"].max()),
        "first_loss": float(df["loss"].iloc[0]),
        "final_loss": float(df["loss"].iloc[-1]),
        "min_loss": float(df["loss"].min()),
        "min_loss_epoch": int(df.loc[df["loss"].idxmin(), "epoch"]),
        "total_seconds": float(df["seconds"].iloc[-1]) if "time" in df.columns else None,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
    }


def _read_mmdet_log(work_dir):
    candidates = sorted(Path(work_dir).glob("*.log.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    path = candidates[-1]
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("mode") != "train":
            continue
        rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "epoch" not in df.columns:
        return None
    loss_cols = [c for c in df.columns if c == "loss" or c.startswith("loss_")]
    df["loss"] = df["loss"] if "loss" in df.columns else df[loss_cols].sum(axis=1)
    df["lr"] = df["lr"] if "lr" in df.columns else np.nan
    df["seconds"] = df["time"].cumsum() if "time" in df.columns else np.nan
    per_epoch = df.groupby("epoch", as_index=False).last()
    return per_epoch, {
        "epochs_completed": int(per_epoch["epoch"].max()),
        "first_loss": float(per_epoch["loss"].iloc[0]),
        "final_loss": float(per_epoch["loss"].iloc[-1]),
        "min_loss": float(per_epoch["loss"].min()),
        "min_loss_epoch": int(per_epoch.loc[per_epoch["loss"].idxmin(), "epoch"]),
        "total_seconds": float(per_epoch["seconds"].iloc[-1]) if "time" in per_epoch.columns else None,
        "best_epoch": None,
        "best_metric": None,
    }


def _read_detr_log(output_dir):
    """DINO/Oriented-DETR-style ``log.txt`` -- one JSON object per line."""
    path = Path(output_dir) / "log.txt"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    loss_cols = [c for c in df.columns if c == "train_loss"]
    df["loss"] = df[loss_cols[0]] if loss_cols else np.nan
    lr_cols = [c for c in df.columns if "lr" in c.lower()]
    df["lr"] = df[lr_cols[0]] if lr_cols else np.nan
    df["seconds"] = np.nan
    return df, {
        "epochs_completed": int(df["epoch"].max()),
        "first_loss": float(df["loss"].iloc[0]) if loss_cols else None,
        "final_loss": float(df["loss"].iloc[-1]) if loss_cols else None,
        "min_loss": float(df["loss"].min()) if loss_cols else None,
        "min_loss_epoch": int(df.loc[df["loss"].idxmin(), "epoch"]) if loss_cols else None,
        "total_seconds": None,
        "best_epoch": None,
        "best_metric": None,
    }


def _read_rfdetr_log(output_dir):
    """RF-DETR's own per-epoch log. Filename is unverified against a real
    run (no GPU here) -- glob for the DETR-family ``log.txt`` convention
    first, then fall back to any ``*.json`` in the output dir, and return
    ``None`` quietly if neither is present rather than guessing further."""
    result = _read_detr_log(output_dir)
    if result is not None:
        return result
    candidates = sorted(Path(output_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    for path in candidates:
        try:
            rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        except (json.JSONDecodeError, OSError):
            continue
        if not rows or "epoch" not in rows[0]:
            continue
        df = pd.DataFrame(rows)
        loss_cols = [c for c in df.columns if "loss" in c.lower()]
        df["loss"] = df[loss_cols[0]] if loss_cols else np.nan
        df["lr"] = np.nan
        df["seconds"] = np.nan
        return df, {
            "epochs_completed": int(df["epoch"].max()),
            "first_loss": float(df["loss"].iloc[0]) if loss_cols else None,
            "final_loss": float(df["loss"].iloc[-1]) if loss_cols else None,
            "min_loss": float(df["loss"].min()) if loss_cols else None,
            "min_loss_epoch": int(df.loc[df["loss"].idxmin(), "epoch"]) if loss_cols else None,
            "total_seconds": None, "best_epoch": None, "best_metric": None,
        }
    return None


_READERS = {
    "ultralytics": _read_ultralytics_log,
    "mmdet": _read_mmdet_log,
    "detr": _read_detr_log,
    "rfdetr": _read_rfdetr_log,
}


@_fail_soft
def collect_epoch_log(out_dir, framework):
    """Find, normalise and persist a framework's own per-epoch log.

    Writes ``epoch_log.csv`` (the framework's per-epoch columns, passed
    through unrenamed, plus a small common core: ``epoch``, ``loss``, ``lr``,
    ``seconds``) and ``epoch_log_summary.json`` next to the raw log, which is
    left in place. Returns the path to ``epoch_log.csv``, or ``None`` if no
    recognisable log was found -- silent by design, since not every framework
    reliably emits one under a known name (see ``_read_rfdetr_log``).

    ``framework`` is one of ``"ultralytics"``, ``"mmdet"``, ``"detr"``,
    ``"rfdetr"``.
    """
    reader = _READERS.get(framework)
    if reader is None:
        raise ValueError(f"unknown framework {framework!r}, expected one of {list(_READERS)}")
    result = reader(out_dir)
    if result is None:
        return None
    df, summary = result
    out_dir = Path(out_dir)
    csv_path = out_dir / "epoch_log.csv"
    df.to_csv(csv_path, index=False)
    (out_dir / "epoch_log_summary.json").write_text(json.dumps(summary, indent=2))
    return csv_path


# --------------------------------------------------------------------------
# Run environment
# --------------------------------------------------------------------------
def _probe_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None


def _probe_versions():
    versions = {}
    for mod in ("torch", "torchvision", "ultralytics", "mmcv", "mmdet",
                "numpy", "shapely", "optuna"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- version probing, not correctness
            versions[mod] = None
    return versions


def _probe_git_commit(repo_root):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


@_fail_soft
def write_run_env(path, extra=None, repo_root=None):
    """Write ``run_env.json``: timestamp, platform, GPU, framework versions,
    git commit, and whatever per-arm config the caller passes as ``extra``
    (SEED, K_FOLDS, TRAIN_EPOCHS, EIRFS_THRESHOLD, PROJECT, RUN_LABEL, ...).

    ``extra`` is also where an arm whose kernel has no torch at all (S2A-Net,
    Strip R-CNN -- both run their real training inside a conda subprocess)
    should pass in conda-probed versions, since ``_probe_versions()`` only
    sees this kernel's own imports.
    """
    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python_version": sys.version,
        "gpu": _probe_gpu(),
        "versions": _probe_versions(),
        "git_commit": _probe_git_commit(repo_root) if repo_root else None,
    }
    if extra:
        env["config"] = dict(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2, default=str))
    return path


# --------------------------------------------------------------------------
# Fold composition
# --------------------------------------------------------------------------
@_fail_soft
def write_fold_composition(path, folds, class_names, num_classes):
    """Per-(fold, split, class) image/instance counts.

    Deliberately a direct count, not a reuse of ``EIRFSWeights`` -- that
    class floors ``image_counts`` at 1 and starts ``instance_counts`` at 1
    per class (a Laplace-ish guard so a class absent from a fold doesn't
    divide by zero when computing a *sampling weight*), which would make a
    genuinely-absent class in this fold's table read as "1 image" instead of
    0. This function counts raw occurrences instead, so a fold with zero
    Rhizoctonia instances (relevant to
    ``results/source_confound_scoping.md``) shows zero, not one.

    ``folds`` is the list of fold dicts produced by ``make_folds``/
    ``kfold_assignments`` (each with a ``"fold"`` key and ``"train"``/``"val"``
    lists of ``(image_path, label_path)`` pairs).
    """
    rows = []
    for f in folds:
        for split in ("train", "val"):
            items = f.get(split, [])
            if not items:
                continue
            n_images = np.zeros(num_classes, dtype=int)
            n_instances = np.zeros(num_classes, dtype=int)
            n_background = 0
            for _, lbl_path in items:
                cls_ids, _ = read_yolo_obb_label(lbl_path)
                if not len(cls_ids):
                    n_background += 1
                    continue
                for cid in np.unique(cls_ids):
                    n_images[int(cid)] += 1
                for cid in cls_ids:
                    n_instances[int(cid)] += 1
            for cid, name in enumerate(class_names):
                rows.append({
                    "fold": f["fold"], "split": split, "class": name,
                    "n_images": int(n_images[cid]),
                    "n_instances": int(n_instances[cid]),
                })
            rows.append({
                "fold": f["fold"], "split": split, "class": "__background__",
                "n_images": n_background, "n_instances": 0,
            })
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Per-class aggregation across folds
# --------------------------------------------------------------------------
@_fail_soft
def write_per_class_tables(kfold_dir, per_fold_per_class, class_names):
    """Build the per-class table this comparison is missing.

    ``per_fold_per_class`` is a list of ``(fold, per_class_dict)`` pairs,
    where each ``per_class_dict`` is ``evaluate_obb()["per_class"]`` with
    ``precision_recall_f1()["per_class"]``/``["counts"]`` merged in by the
    caller -- see each arm's per-fold evaluation cell for the merge.

    Writes ``kfold_per_class_results.csv`` (one row per fold x class) and
    ``kfold_per_class_summary.csv`` (one row per class, mean +/- std across
    the folds that had GT for it).
    """
    rows = []
    for fold, per_class in per_fold_per_class:
        for name in class_names:
            m = per_class.get(name)
            if m is None:
                continue
            rows.append({
                "fold": fold, "class": name,
                "AP50-95": m.get("AP50-95"), "AP50": m.get("AP50"),
                "AP75": m.get("AP75"), "n_gt": m.get("n_gt"),
                "precision": m.get("precision"), "recall": m.get("recall"),
                "F1": m.get("F1"), "tp": m.get("tp"), "fp": m.get("fp"),
                "fn": m.get("fn"),
            })
    results_df = pd.DataFrame(rows)
    kfold_dir = Path(kfold_dir)
    kfold_dir.mkdir(parents=True, exist_ok=True)
    results_path = kfold_dir / "kfold_per_class_results.csv"
    results_df.to_csv(results_path, index=False)

    metric_cols = ["AP50-95", "AP50", "AP75", "precision", "recall", "F1"]
    summary_rows = []
    if not results_df.empty:
        for name, g in results_df.groupby("class"):
            row = {"class": name, "n_folds_with_gt": int(len(g))}
            for col in metric_cols:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=0))
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = kfold_dir / "kfold_per_class_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return results_path, summary_path


# --------------------------------------------------------------------------
# Optuna persistence
# --------------------------------------------------------------------------
def open_study(name, db_path, mirror_path, sampler, pruner):
    """S2A-Net's proven pattern: sqlite storage, restored from the Drive
    mirror if the local db is gone, so a disconnect does not lose the study.

    Resuming subtracts already-finished trials via Optuna's own
    ``load_if_exists=True`` + trial count, so ``N_TRIALS`` keeps meaning
    "this many total trials" rather than "this many more" -- callers must
    still cap their own HPO loop at ``N_TRIALS - len(study.trials)``, this
    function only opens the study.

    Not wrapped in ``@_fail_soft``: an Optuna study is load-bearing for the
    HPO loop that follows it (unlike everything else in this module, which
    is pure side-output), so a failure here should surface immediately
    rather than silently falling back to a fresh in-memory study that would
    then look like a successful resume.
    """
    import optuna

    db_path, mirror_path = Path(db_path), Path(mirror_path)
    if not db_path.exists() and mirror_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(mirror_path), str(db_path))
    return optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        storage=f"sqlite:///{db_path}", study_name=name, load_if_exists=True,
    )


@_fail_soft
def dump_study(study, out_dir):
    """Write ``hpo_trials.csv`` (``study.trials_dataframe()``) and
    ``hpo_best.json``, so the study is readable without an Optuna install or
    the sqlite file itself."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(out_dir / "hpo_trials.csv", index=False)
    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "n_trials": len(study.trials),
    }
    (out_dir / "hpo_best.json").write_text(json.dumps(best, indent=2))
    return out_dir / "hpo_trials.csv", out_dir / "hpo_best.json"


# --------------------------------------------------------------------------
# Per-fold / per-trial timing
# --------------------------------------------------------------------------
class Timer:
    """A tiny fail-soft stopwatch for one fold or trial.

    Usage::

        with Timer() as t:
            train_one_fold(...)
        row["train_seconds"] = t.seconds

    Never raises from ``__exit__`` -- a timer must not be the reason a fold's
    result row is lost.
    """

    def __init__(self):
        self.seconds = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.seconds = time.perf_counter() - self._t0
        except Exception:  # noqa: BLE001
            self.seconds = None
        return False  # never swallow the caller's own exception
