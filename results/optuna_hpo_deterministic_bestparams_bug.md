# Optuna HPO bug — `best_params.json` always identical regardless of model/epochs

**Status: diagnosed, not yet fixed.** Reported symptom: running the Optuna
search in `notebooks/Template_Clean_E-IRFS_optuna_datasetv3.py` produces the
exact same `best_params.json` every time, no matter what `PRETRAINED` model
or `EPOCHS_PER_TRIAL` is used.

The E-IRFS sampler code in this notebook (`YOLOEIRFSCappedDataset`) matches
the canonical `src/data_loader.py` implementation exactly — that is **not**
the source of the bug. The bug is in the Optuna wiring around it.

## Root cause

Two things compound to make the search fully deterministic and unrelated to
anything you configure:

**1. Every trial is silently tying at the same score.**
In `objective()`, each fold's training is wrapped in a broad
`try/except Exception` (lines 452–468):

```python
except Exception as e:
    print(f"[Trial {trial.number}] Fold {k} failed: {e}", file=sys.stderr)
    score = 0.0
```

If training fails on every fold (for whatever underlying reason — the
exception is currently only printed to `stderr` and never surfaced), every
trial's `cv_mean` ends up `0.0`. That's a tie across the entire study, not
just occasional pruning.

**2. Optuna's tie-break always keeps trial 0, and trial 0 is seed-fixed.**
`study.best_trial` / `study.best_params` only update on a *strict*
improvement — when every trial ties, the incumbent never changes, so the
"winner" is always trial 0. Confirmed directly:

```
best params {'x': 3.745401188473625, 'y': 'a'}        # trial 0, seed=42
run2 best params {'x': 3.745401188473625, 'y': 'a'}    # identical, second run, all trials tied
```

Trial 0's hyperparameters come from `sampler = optuna.samplers.TPESampler(seed=SEED)`
(line 522, `SEED=42`). `TPESampler`'s own `n_startup_trials` **defaults to
10**, which exactly equals `study.optimize(objective, n_trials=10)` (line
527) — so the sampler never leaves its seeded-random startup phase; real
TPE/Bayesian search never activates. Trial 0's draw is computed purely from
the fixed seed, *before any training happens*, so it cannot depend on
`PRETRAINED`, `EPOCHS_PER_TRIAL`, or any other run-to-run change.

Net effect: `best_params.json` (written at line 535) is always trial 0's
seed-determined params — identical across runs, models, and epoch counts,
exactly as reported.

## How to confirm

- Check the per-trial CSVs already being written each run:
  `runs/optuna_obb/cv_trial_XXX.csv` (line 480). If every row shows
  `mAP50-95 == 0.0`, that confirms every fold is failing/erroring.
- Scroll the Colab log for `[Trial N] Fold K failed: ...` — this line is
  currently easy to miss, buried under Ultralytics' own per-epoch training
  output.

## Recommended fixes

1. **Stop swallowing the real error.** Surface/re-raise it (or at least fail
   loudly after N consecutive fold failures) instead of silently scoring
   `0.0` — right now the actual exception (`e`) is discarded to stderr and
   never inspected, so the true underlying training failure is unknown.
2. **Fix `n_startup_trials` vs. `n_trials`.** Set `TPESampler(seed=SEED,
   n_startup_trials=3)` (matching the pruner's `n_startup_trials=3`) so the
   search does real Bayesian-guided sampling instead of pure seeded-random
   draws for its entire duration.
3. Once (1) is fixed, re-run and re-check the `cv_trial_XXX.csv` files to
   confirm scores actually vary across trials/folds before trusting
   `best_params.json` again.

## Files involved

- `notebooks/Template_Clean_E-IRFS_optuna_datasetv3.py` — lines 452–468
  (silent failure → tied 0.0 scores), lines 521–527 (seeded sampler /
  `n_startup_trials` collision), line 535 (`best_params.json` write).
- `src/data_loader.py` — canonical E-IRFS sampler; compared line-for-line
  against the notebook's local copy, no divergence found. Not implicated.
