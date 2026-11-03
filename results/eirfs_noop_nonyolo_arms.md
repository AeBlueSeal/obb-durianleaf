# E-IRFS was a silent no-op in the four non-YOLO arms

**Found:** 2026-09-11, during the S2A-Net comparability audit.
**Status:** fixed in `src/obb_common.py:302`. Archived non-YOLO results
predating the fix are affected — see [Implications](#implications).

Citations below are `file:line`. Project files are cited at commit `2845446`
where the pre-fix state matters (marked *pre-fix*) and at the current working
tree otherwise. Upstream files are cited at the exact tag each arm pins.

---

## Summary

`attach_eirfs()` installed the E-IRFS sampler by **assigning an instance
attribute**, `dataset.__getitem__ = eirfs_getitem`
(`src/obb_common.py:324` *pre-fix*). Python resolves the `dataset[i]`
subscript operator on `type(dataset)` and never consults the instance
`__dict__`, so that assignment had no effect on any index a DataLoader ever
requested. Every one of Strip R-CNN, Oriented-DETR, RF-DETR-OBB and S2A-Net
therefore trained with **stock uniform sampling** while reporting itself as an
E-IRFS run.

The two YOLO arms are unaffected: they install E-IRFS as a real **subclass**
(`notebooks/YOLO11mOBB_E-IRFS_kfold_datasetv3.py:348`), i.e. a class-level
override, which subscript does find.

The consequence is not "E-IRFS was slightly weaker in some arms." It is that
the sampler — the one variable this comparison exists to hold constant across
architectures (`src/obb_common.py:305-312` *pre-fix*, the function's own
docstring) — was **active in 2 of 6 arms and inert in the other 4**.

---

## The mechanism

### Why the assignment looks correct

`dataset.__getitem__ = fn` is a perfectly ordinary attribute assignment, and
it does work if you then call `dataset.__getitem__(i)` by hand. That is what
makes this fail silently rather than loudly: there is no error, no warning,
and the patched function is genuinely reachable — just never by the route the
training loop takes.

### Why it does not work

For special (dunder) methods invoked through their operator syntax, CPython
performs *implicit special method lookup*: it looks the method up on the
**type**, bypassing the instance `__dict__` entirely. `dataset[i]` is
therefore `type(dataset).__getitem__(dataset, i)`, which resolves to the
original class method regardless of what is sitting in the instance
dictionary.

This is documented language behaviour, not a CPython quirk:
<https://docs.python.org/3/reference/datamodel.html#special-lookup>

### Why the DataLoader always takes the failing route

PyTorch's map-style fetcher uses subscript syntax, in every version any arm
pins:

| Arm | torch | Fetcher line |
|---|---|---|
| S2A-Net | 1.3 | `torch/utils/data/_utils/fetch.py:44` (v1.3.0) |
| Strip R-CNN | 1.13.1 | `torch/utils/data/_utils/fetch.py:58` (v1.13.1) |
| Oriented-DETR, RF-DETR-OBB | 2.x | `torch/utils/data/_utils/fetch.py:52` (v2.5.0) |

All three are the same statement:

```python
data = [self.dataset[idx] for idx in possibly_batched_index]
```

There is no route around this. The single-sample path on the next line
(`fetch.py:46` / `:60` / `:54`) is `self.dataset[possibly_batched_index]` —
also subscript. The `__getitems__` fast path that torch ≥1.13 checks first
(`fetch.py:55` v1.13.1, `:49` v2.5.0) is a **different, plural-named** method
that `attach_eirfs` never touched, so it changes nothing here either.

And the DataLoader really is stock torch — e.g. mmdet builds it directly:
`mmdet/datasets/builder.py:195` (v2.28.2).

### Reproduction

`scratch/probe_eirfs_getitem_patch.py` reproduces both the failure and the
fix with plain classes and no framework stack installed (the behaviour under
test is a language rule, so it needs none):

```
  ds.__getitem__(5)            -> 999   (goes through the patch)
  ds[5]                        -> 5     (IGNORES the patch)
  torch-style fetch([1,2,3])   -> [1, 2, 3]  (IGNORES the patch)
```

---

## Which arms were affected

Every arm that routed E-IRFS through `attach_eirfs()` was affected. Call
sites:

| Arm | Call site |
|---|---|
| S2A-Net | `src/s2anet_worker.py:78` |
| Strip R-CNN | `src/strip_rcnn_worker.py:140` |
| Oriented-DETR (K-fold) | `notebooks/OrientedDETR_E-IRFS_kfold_datasetv3.py:376` |
| Oriented-DETR (Optuna) | `notebooks/OrientedDETR_E-IRFS_optuna_datasetv3.py:349` |
| RF-DETR-OBB (K-fold) | `notebooks/RFDETR_OBB_E-IRFS_kfold_datasetv3.py:273` |
| RF-DETR-OBB (Optuna) | `notebooks/RFDETR_OBB_E-IRFS_optuna_datasetv3.py:246` |

Note that S2A-Net never ran live, so it has no invalid archived results — it
only inherited the bug from shared code.

### Why the YOLO arms were not affected

The YOLO arms never call `attach_eirfs`. They define a genuine subclass and
swap it into Ultralytics' dataset factory, so the override lives on a
**class**:

- `notebooks/YOLO11mOBB_E-IRFS_kfold_datasetv3.py:348` — `class YOLOEIRFSCappedDataset(YOLODataset)`
- `notebooks/YOLO11mOBB_E-IRFS_kfold_datasetv3.py:439` — its `def __getitem__`, doing the index substitution
- `notebooks/YOLO11mOBB_E-IRFS_kfold_datasetv3.py:508` — `build.YOLODataset = YOLOEIRFSCappedDataset`

Same structure in the sibling YOLO notebooks:
`notebooks/YOLO26mOBB_E-IRFS_kfold_datasetv3.py:371`/`:462`/`:531` and
`notebooks/YOLO11mOBB_E-IRFS_optuna_datasetv3.py:337`/`:428`/`:596`.

A class-level `__getitem__` is exactly what implicit special method lookup
*does* find, so YOLO's E-IRFS has been live the whole time.

### A note on how long this hid

Nothing about the failure is observable from training logs — sampling still
produces valid batches of the right length, loss still decreases, and the
per-class weight computation that `EIRFSWeights` does
(`src/obb_common.py:161`, sampling at `src/obb_common.py:254`) still runs and
still prints its statistics. Only the *use* of those weights was dropped.

The belief that the patch was in effect had also been reasoned about
downstream and written into the code as fact — e.g.
`notebooks/StripRCNN_E-IRFS_kfold_datasetv3.py:538` argues that forked
DataLoader workers "inherit the patched method identically." That reasoning
is correct about forking, and irrelevant: there was no effective patch to
inherit.

---

## The fix

`src/obb_common.py:333-340`. Instead of assigning the instance attribute,
re-point `dataset.__class__` at a throwaway subclass carrying the override:

```python
base = type(dataset)
original_getitem = base.__getitem__

def eirfs_getitem(self, index, _orig=original_getitem, _w=weights):
    return _orig(self, _w.sample_index())

dataset.__class__ = type(f"EIRFS{base.__name__}", (base,),
                         {"__getitem__": eirfs_getitem})
```

This preserves the original design intent stated in the docstring — only this
*instance* changes, so a val/test dataset built from the same class is
untouched — while putting the override somewhere subscript will actually find
it. `isinstance(dataset, base)` still holds, so any framework code that
type-checks the dataset is unaffected. All three properties are asserted in
`scratch/probe_eirfs_getitem_patch.py`.

The fix is in shared code, so it reaches all four arms with no per-notebook
change.

### Why not patch the class directly

Patching `base.__getitem__` would work for subscript but would also hit every
other instance of that class in the process — including the val/test dataset
built from the same class in the same fold, which must **not** be resampled.
The per-instance subclass gets the reachability without the blast radius.

---

## Implications

**Any archived Strip R-CNN, Oriented-DETR or RF-DETR-OBB run labelled
"E-IRFS" predates this fix and did not actually use E-IRFS.** Those runs are
valid *stock-sampling* runs; they are not valid E-IRFS runs.

Concretely:

1. **E-IRFS-vs-NoEIRFS claims for the non-YOLO arms are unsupported.** The
   NoEIRFS ablation twins exist only for YOLO
   (`YOLO11mOBB_NoEIRFS_kfold_datasetv3.py` /
   `YOLO26mOBB_NoEIRFS_kfold_datasetv3.py`, see CLAUDE.md). For the other
   arms, the "E-IRFS" run *was* the ablation and nobody knew.
2. **Cross-architecture comparisons that predate this fix confound
   architecture with sampler.** YOLO arms had E-IRFS; the rest did not. Any
   YOLO-vs-other gap is partly a sampler gap.
3. **Archived `best_params.json` from non-YOLO arms was searched under stock
   sampling.** The search is not invalidated the way the truncated-dataset bug
   invalidated things (see CLAUDE.md's K-fold/Optuna entry) — it is a real
   search over real training runs — but it was tuned for a different sampler
   than the final runs will now use.
4. Re-running is the only way to recover the intended comparison. This is
   expensive: the non-YOLO arms have no early stopping and always train their
   full epoch budget (CLAUDE.md, "Patience/early-stopping across
   architectures").

---

## How to re-verify after a change

`scratch/probe_eirfs_getitem_patch.py` is the cheap regression check and
exits non-zero on failure:

```
python3 scratch/probe_eirfs_getitem_patch.py
```

To confirm on real Colab hardware that sampling is genuinely being
substituted, assert against the counter rather than trusting the attach call:

```python
seen = []
orig = type(datasets[0]).__getitem__
attach_eirfs(datasets[0], weights, train_mode=True)
_ = [datasets[0][i] for i in range(50)]   # subscript, like the DataLoader
# with E-IRFS live, the indices actually served will not be 0..49
```

The load-bearing detail is that the check must use **subscript**
(`dataset[i]`), not `dataset.__getitem__(i)` — a probe written the second way
passes against the broken code and proves nothing. That is precisely how this
bug survived as long as it did.

---

## See also

- `src/obb_common.py:302` — `attach_eirfs`, fixed, with the failure mode in its docstring
- `scratch/probe_eirfs_getitem_patch.py` — reproduction and regression check
- `results/obb_model_comparison.md` — what this comparison holds constant and what it cannot
- CLAUDE.md, "S2A-Net comparability audit" — the audit that surfaced this, and the four other bugs found alongside it
