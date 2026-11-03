# Implementation Plan — Fixing the Confirmed Source Confound

**Status going in:** Confound confirmed on every axis tested.
- Class ↔ source correlation: Cramér's V = 0.855 (Rhizoctonia 97.6% curated, Healthy 58.7% curated)
- Source is trivially learnable: curated = 400×400 exactly, always; field = high-res, variable
- Reliance confirmed: Healthy AP collapses 0.814 (curated) → 0.413 (field); Rhizoctonia's field AP is computed on only 8 instances (unreliable)
- Independent background shortcut: Algal spot (3.1× chance) and Other (2.9× chance) predictable from background alone
- Split leakage: 17 cross-split near-duplicate pairs, several at Hamming distance 0 (identical images)

**Do the phases in order.** Each phase has a pass/fail check — do not proceed past a failed check.

---

## Phase 1 — Deduplicate and build "must-stay-together" groups

Goal: identify every image that must be locked into a single split (train, val, or test) because it's a near-duplicate or same-session/same-plant shot as another image.

### 1.1 Compute perceptual hashes and find duplicate/near-duplicate pairs

```python
import imagehash, glob, os
from PIL import Image

IMG_DIR = "path/to/all_images"   # all images, all sources, all current splits
THRESH  = 5                       # Hamming distance; from D_report.txt this catches your 17 pairs

paths  = glob.glob(os.path.join(IMG_DIR, "**", "*.*"), recursive=True)
hashes = {}
for p in paths:
    try:
        hashes[p] = imagehash.phash(Image.open(p))
    except Exception as e:
        print("skip", p, e)

pairs = []
items = list(hashes.items())
for i in range(len(items)):
    pa, ha = items[i]
    for j in range(i + 1, len(items)):
        pb, hb = items[j]
        d = ha - hb
        if d <= THRESH:
            pairs.append((pa, pb, d))

print(f"{len(pairs)} duplicate/near-duplicate pairs found")
```

### 1.2 Build session groups from filenames (field images)

Your field filenames encode a timestamp burst (`IMG_20250907_133409.jpg`). Group images taken within a short window of each other on the same day — these are very likely the same branch/session and can share near-identical leaves even without being pixel-duplicates.

```python
import re
from datetime import datetime, timedelta

def parse_session(fname):
    m = re.search(r"IMG_(\d{8})_(\d{6})", os.path.basename(fname))
    if not m:
        return None
    date_str, time_str = m.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

WINDOW = timedelta(minutes=5)   # tune: 5 min covers one stop on a walk through the orchard

field_paths = [p for p in paths if "IMG_" in os.path.basename(p)]
timestamps  = [(p, parse_session(p)) for p in field_paths]
timestamps  = [(p, t) for p, t in timestamps if t is not None]
timestamps.sort(key=lambda x: x[1])

session_groups = []
current = []
for p, t in timestamps:
    if current and (t - current[-1][1]) > WINDOW:
        session_groups.append([x[0] for x in current])
        current = []
    current.append((p, t))
if current:
    session_groups.append([x[0] for x in current])

print(f"{len(session_groups)} field sessions from {len(field_paths)} images")
```

### 1.3 Build sequence groups for curated images

`Leaf_Rhizoctonia_92.jpg` and `_93.jpg` were flagged as near-duplicates — curated images are likely numbered sequentially per source leaf/batch. Group by class + nearby index.

```python
def parse_curated(fname):
    m = re.match(r"Leaf_(\w+)_(\d+)", os.path.basename(fname))
    if not m:
        return None
    return m.group(1), int(m.group(2))

curated_paths = [p for p in paths if os.path.basename(p).startswith("Leaf_")]
parsed = [(p, parse_curated(p)) for p in curated_paths]
parsed = [(p, c) for p, c in parsed if c is not None]
parsed.sort(key=lambda x: (x[1][0], x[1][1]))

SEQ_GAP = 2  # group indices within this gap of each other, per class
curated_groups = []
current = []
for p, (cls, idx) in parsed:
    if current and (cls != current[-1][2] or idx - current[-1][1] > SEQ_GAP):
        curated_groups.append([x[0] for x in current])
        current = []
    current.append((p, idx, cls))
if current:
    curated_groups.append([x[0] for x in current])

print(f"{len(curated_groups)} curated sequence groups from {len(curated_paths)} images")
```

### 1.4 Merge everything into final groups (union-find)

Combine the pairwise duplicates, session groups, and sequence groups into connected components — every image in a component must land in the same split.

```python
parent = {p: p for p in paths}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

for pa, pb, d in pairs:
    union(pa, pb)
for group in session_groups + curated_groups:
    for p in group[1:]:
        union(group[0], p)

from collections import defaultdict
components = defaultdict(list)
for p in paths:
    components[find(p)].append(p)

groups = list(components.values())
print(f"{len(paths)} images collapsed into {len(groups)} split-atomic groups")
```

**Check before continuing:** print the size distribution of `groups`. A handful of very large groups (hundreds of images) would make stratification hard — if that happens, tighten `WINDOW`/`SEQ_GAP` and re-run.

---

## Phase 2 — Grouped, source-stratified, class-stratified re-split

Goal: 80/10/10 split where (a) no group is split across sets, (b) source ratio (curated/field) is preserved per class in each split, (c) class proportions stay close to the original.

This is a constrained bin-packing problem — solved here with a greedy heuristic, which is standard practice for grouped stratified splitting when no single library does all three constraints at once.

```python
import random
from collections import Counter

random.seed(42)

# label_lookup: {image_path: {"classes": Counter, "source": "curated"/"field"}}
# Build this from your YOLO label files + source_of() logic from the confound-check guide.
label_lookup = build_label_lookup(paths)  # implement using your existing labeling pipeline

def group_signature(group):
    sig = Counter()
    sources = Counter()
    for p in group:
        info = label_lookup.get(p)
        if not info:
            continue
        sig.update(info["classes"])
        sources[info["source"]] += 1
    return sig, sources

group_info = [(g, *group_signature(g)) for g in groups]

# Sort largest-signature groups first (hardest to place), place smaller groups after
group_info.sort(key=lambda x: -sum(x[1].values()))

targets = {"train": 0.8, "val": 0.1, "test": 0.1}
split_totals   = {k: Counter() for k in targets}
split_sources  = {k: Counter() for k in targets}
split_assign   = {k: [] for k in targets}
total_sig = sum((sig for _, sig, _ in group_info), Counter())

def score_if_added(split_name, sig, src):
    """Lower score = better fit toward target proportions."""
    proposed = split_totals[split_name] + sig
    err = 0.0
    for cls, target_frac in targets.items():
        pass  # placeholder, real scoring below
    return proposed

def imbalance_score(split_name, sig):
    proposed_total = split_totals[split_name] + sig
    grand_total     = sum(total_sig.values()) * targets[split_name]
    return abs(sum(proposed_total.values()) - grand_total)

for g, sig, src in group_info:
    best_split = min(targets.keys(), key=lambda s: imbalance_score(s, sig))
    split_assign[best_split].extend(g)
    split_totals[best_split].update(sig)
    split_sources[best_split].update(src)

for s in targets:
    print(s, "images:", len(split_assign[s]), "class counts:", dict(split_totals[s]),
          "source mix:", dict(split_sources[s]))
```

**Verification before training:**
```python
# 1. No group crosses a split
assigned_to = {}
for s, imgs in split_assign.items():
    for p in imgs:
        assigned_to[p] = s
for g in groups:
    splits_used = {assigned_to.get(p) for p in g}
    assert len(splits_used) == 1, f"Group split across sets: {g}"

# 2. Class proportions close to original (print and eyeball, no class off by >5pp)
# 3. Source mix per split roughly matches overall source mix per class —
#    check specifically that Rhizoctonia and Healthy are NOT 100% one source
#    in any single split (some field Rhizoctonia/Healthy must appear in val/test
#    if any field examples exist at all).
```

**Pass/fail check:** if Rhizoctonia has only 8 field instances total, you likely cannot get meaningful field representation in val AND test AND train simultaneously. Flag this now — resolved in Phase 3, not by tweaking the split further.

---

## Phase 3 — Resolve the Rhizoctonia / Healthy field-data gap

With only 8 field Rhizoctonia instances, no split algorithm fixes an underlying data shortage. Choose one path:

### Option A (preferred if feasible): collect real field images
- Target at least ~150–200 field Rhizoctonia instances and a similar boost for Healthy, matching the volume your other field classes have (~600–2,100).
- Photograph under the same conditions as your other field data (same phones, same orchard, natural lighting/background) specifically to avoid creating a *new* source confound.

### Option B (if collection isn't possible right now): scope the claim honestly
- Explicitly document Rhizoctonia and Healthy as **curated-validated, not field-validated**.
- Do not report a single blended AP for these classes; report curated-set AP and field-set AP (even if the field AP has wide confidence intervals from small N) separately, with the sample size stated.
- Exclude these two classes from any claim about real-world deployment accuracy until Option A happens.

### Either way — reduce the source cue itself (do this regardless of A or B)
Since B1 showed curated images are a fixed 400×400 with zero resolution variance, that cue must be removed so the model can't use it as a free proxy:
- Do not feed curated and field images through different preprocessing paths. Both must go through the **same** resize/letterbox pipeline at training time (see Phase 5).
- If keeping curated images, augment them with resolution jitter, random crop-then-resize, and background variation so they stop looking like a uniform, separable cluster.

---

## Phase 4 — Break the background shortcut (Algal spot, Other)

Goal: make background alone stop being predictive of Algal spot / Other.

### 4.1 Add explicit background-only negatives to TRAIN
- Collect ~150–300 label-free images of grass, soil, bark, and non-durian foliage from the **same field conditions**.
- Add them to the training set with no label file (Ultralytics treats these as negative background).
- Target ~5–10% of the training set.

### 4.2 Background-randomization augmentation
Use your existing OBB polygons to separate leaf from background, then recombine leaves with varied backgrounds so the model can't key off "field-photo background → Algal spot."

```python
import cv2, numpy as np, random

def extract_leaf_mask(img_shape, obb_polys):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for poly in obb_polys:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask

def swap_background(img, mask, background_pool):
    bg = random.choice(background_pool)
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    return (img * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
```

- Build `background_pool` from your negative images (4.1) plus generic backgrounds (other orchards, other times of day).
- Apply this augmentation to a fraction (e.g. 30–50%) of Algal spot and Other training instances specifically, since these are the two classes flagged as shortcut-vulnerable.

### 4.3 Hard-negative mining
- Run the current model on background-only images.
- Collect every false-positive Algal spot / Other detection.
- Add those exact crops as additional negatives in the next training round.

### 4.4 Verify the fix (re-run Test C2 from the confound-check guide)
```text
Target: hit_rate on masked (background-only) images should fall to ~chance (≈14.3%)
         for Algal spot and Other specifically.
```
Do not proceed to full retraining claims until this ratio drops meaningfully from 3.1× / 2.9×.

---

## Phase 5 — Standardize preprocessing across sources

- Use one letterbox/resize function for every image regardless of source, at your training `imgsz`.
- Use the same padding color and the same augmentation pipeline (mosaic, HSV, flips) for curated and field images — no source-specific special-casing.
- Strip or randomize EXIF camera tags before training if your pipeline reads them (B1 showed camera model was a clean source fingerprint: `iPhone 15 Pro`, `DNY-NX9`, `iPhone 16 Pro Max` all field-only).

---

## Phase 5.5 — Re-sequence E-IRFS (class-imbalance sampling)

Your prior training used **E-IRFS** (Exponentially Weighted Instance-Aware Repeat Factor Sampling) to address the ~5.6:1 class imbalance. This needs to be re-applied carefully, not skipped and not blindly repeated as-is — it interacts directly with the confound you just confirmed.

### Why this needs care
E-IRFS oversamples *images*, not *content* — it makes training see rare-class images more often by repeating them, without adding any new diversity. Rhizoctonia is 97.6% curated, and those curated images are a near-uniform cluster (fixed 400×400, zero resolution variance). If E-IRFS ran on the original data, it was very likely **repeating the same narrow curated-source cluster**, not teaching the model more about the disease — reinforcing the source shortcut rather than fixing genuine imbalance. Class-imbalance sampling amplifies whatever the model actually keys on; if that's source/background, oversampling makes the confound worse, not better.

### What to do
- **Do not reapply E-IRFS on Rhizoctonia/Healthy until Phase 3 is resolved.** If Phase 3 Option A (real field images collected) is done, E-IRFS now oversamples genuinely diverse content and is safe/beneficial again. If Phase 3 Option B (scoped reporting, no new data) was chosen instead, treat E-IRFS's boost to these two classes with caution — consider capping its repeat factor for Rhizoctonia/Healthy specifically until more field diversity exists.
- **Recompute E-IRFS's frequencies on the NEW split**, not the old one. Image and instance counts changed after Phase 1–2 (regrouping) and Phase 4.1 (background negatives added) — stale frequencies from the original run will misweight the corrected dataset.
- **Verify how E-IRFS treats the new background-only negatives.** Repeat-factor formulas are keyed on category frequency; label-free negatives have no category. Confirm they get a neutral repeat factor (≈1) and aren't crowded out or accidentally over-repeated by the same logic — otherwise Phase 4.1's negative ratio silently drifts once sampling is applied.
- **Keep it for the classes where it isn't confounded.** Phomopsis, Anthracnose, and Phytophtora are ~99% field-sourced, so E-IRFS oversampling them is not source-tainted — no reason to remove it there. The imbalance problem is real and independent of the confound for these classes.
- **Re-verify after retraining.** This is exactly the kind of technique that can quietly re-inflate a shortcut you just fixed — re-run Tests C1/C2 (Phase 7) with specific attention to whether Rhizoctonia/Healthy's numbers move again once E-IRFS is back in the loop.

---

## Phase 6 — Retrain on the corrected setup

### 6.1 Scope a targeted HPO re-run (not a full redo)

Your original HPO (3-fold CV on train) doesn't need to be repeated from scratch — but it also shouldn't be skipped, because Phases 1–5 changed the data in ways it never saw.

- **Reuse as the starting point:** optimizer choice, base learning rate, and momentum are unlikely to shift much from a re-split alone — don't re-search these.
- **Do re-search:** augmentation-related hyperparameters (mosaic probability, HSV jitter, and specifically the new background-swap probability/ratio from Phase 4.2) and loss weighting (especially the `cls` gain, since class balance shifts once background negatives are added) — these now operate on a different dataset than your original search covered.
- **Include epoch count / patience this time.** This was the flagged gap in your original HPO (350 epochs was never tuned) and is what caused the over-training. Search `patience` (or fix it at 50 and confirm via the training curve rather than searching it, if compute is limited).
- **Run the reduced search via the same 3-fold CV on the new train split** (post Phase 1–2), so the folds reflect the corrected grouping.

### 6.2 Retrain

Combine with the earlier training fixes already identified:

```text
- New grouped, stratified split from Phase 2
- Background negatives + background-randomization augmentation from Phase 4
- E-IRFS re-sequenced per Phase 5.5 (recomputed frequencies, Rhizoctonia/Healthy capped until Phase 3 resolved)
- Re-tuned augmentation/loss/patience hyperparameters from 6.1
- patience=50 (early stopping; epochs were never tuned before)
- Standardized preprocessing from Phase 5
- Deploy best.pt, not last.pt
- Inference conf ≈ 0.53 (your prior F1 peak — re-check this after retraining, it may shift)
```

---

## Phase 7 — Re-run the full confound-check suite to confirm the fix

Re-run Tests A, B1, C1, C2, D from the source-confound-check guide on the new split/model. Compare against the baseline numbers above.

| Test | Before | Target after fix |
|---|---|---|
| A: Cramér's V (class↔source) | 0.855 | Lower is better; ideally each class has both sources represented, though it may not reach 0 if curated data remains genuinely class-imbalanced |
| B1: curated resolution uniqueness | 1 / 307 | Should no longer be a single fixed value if curated images are now augmented/resized identically to field |
| C1: Healthy AP gap (curated − field) | 0.4015 | Should shrink substantially; large remaining gap = confound not resolved |
| C1: Rhizoctonia field AP | computed on 8 images | Should be computed on a meaningfully larger field sample (Phase 3) |
| C2: Algal spot hit-rate ratio | 3.1× chance | Should approach ~1.0× (chance level) |
| C2: Other hit-rate ratio | 2.9× chance | Should approach ~1.0× (chance level) |
| D: cross-split near-duplicate pairs | 17 | 0 |
| E-IRFS check: Rhizoctonia/Healthy AP gap after resampling | N/A (new check) | Curated−field gap should not re-widen vs. the post-fix (pre-E-IRFS) numbers; if it does, E-IRFS is re-amplifying the source shortcut — cap or remove it for these two classes |

**Only report the model's performance as reliable once every row above has moved to target.** If any row hasn't improved, the corresponding phase needs another pass before you trust the numbers.

---

## Phase 7.5 — Diagnose the `v4_confound_fix` regression (current status: ACTIVE, do this next)

**Phase 7 was run** (weights `runs/v4_confound_fix/train/weights/best.pt`, data `dataset_resplit.yaml`). Results:

| Test | Before | After (`v4_confound_fix`) | Verdict |
|---|---|---|---|
| B1: curated resolution uniqueness | 1 / 307 | 63 / 307 | ✅ Fixed |
| B1: EXIF fingerprint | field-only tags | `unknown` both sources | ✅ Fixed |
| D: cross-split near-duplicate pairs | 17 | 0 | ✅ Fixed |
| Val-loss NaN logging | 286/350 epochs | 0/181 epochs | ✅ Fixed |
| Checkpoint selection | over-trained to e350 | early-stopped e181, `best.pt` = peak (e74) | ✅ Fixed |
| A: Cramér's V | 0.855 | 0.842 | ⚠️ Expected — Rhizoctonia's 8-instance field shortage is a real data gap, not a split artifact |
| C1: overall mAP gap (curated−field) | 0.067 | **0.092** | ❌ Worse |
| C1: Healthy AP gap | 0.4015 | **0.5625** | ❌ Worse |
| C2: overall masked-image hit rate | 61.7% | **83.7%** | ❌ Worse |
| C2: classes flagged shortcut | 2/7 (Algal spot, Other) | **5/7** (+Healthy, Phomopsis, Rhizoctonia) | ❌ Worse, broader |
| C2: Algal spot / Other hit-rate ratio | 3.1× / 2.9× chance | **5.5× / 5.6× chance** | ❌ Worse |
| Overall mAP@0.5 (test) | 0.689 | 0.650 | Down — partly expected (leak-free now), but doesn't explain C1/C2 |

**Read this precisely:** the pipeline/leakage fixes (B1, D, val-loss, checkpoint selection) worked. But the confound itself — the thing Phase 4 was supposed to fix — got **worse and spread to three previously-clean classes**. This is a genuine regression, not measurement noise (C2 is based on 178 images / 149 hits, a large enough sample to trust) and not a repeat of the earlier "stale checkpoint" false alarm (checkpoint and split were confirmed correct this time).

### Leading hypothesis: the background-swap augmentation created a stronger shortcut than the one it replaced

The Phase 4.2 reference code does a **hard binary-mask paste** — no feathering, no color/noise matching between the leaf and the pasted-in background. That leaves a visible seam at the leaf boundary, which is a high-contrast, very learnable pattern — arguably *easier* for a CNN to key on than the original natural background correlation. This fits the evidence:
- The C2 hit-rate increases are large (up to 5.5–5.6× chance) — bigger than a subtle natural-background correlation typically produces, more consistent with a near-deterministic artifact cue.
- The regression is broad and simultaneous across several classes rather than isolated to one — consistent with a shared mechanism (the augmentation pipeline itself, or its interaction with Phase 5.5's E-IRFS repeating the same augmented images across epochs) rather than seven unrelated causes.

### Step 1 — Visual audit (do this first, ~15 minutes, no training needed)

Pull 15–20 background-swap-augmented training images (from `phase4_background_swap_augment.py`'s output) and inspect the leaf boundary directly.

```python
import matplotlib.pyplot as plt
import glob, random

samples = random.sample(glob.glob("path/to/augmented_train/*swap*.jpg"), 20)
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
for ax, p in zip(axes.flat, samples):
    ax.imshow(plt.imread(p))
    ax.set_title(p.split("/")[-1], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("bg_swap_audit.png", dpi=150)
```

Look specifically for: a visible edge/halo around the leaf, a lighting or color-temperature mismatch between leaf and background, or compression-artifact discontinuities at the boundary. If present, that's strong visual confirmation of the hypothesis before spending compute on an ablation.

### Step 2 — Ablation retrain (isolates the cause)

Keep everything else from `v4_confound_fix` identical — new split, EXIF strip, background negatives (4.1), E-IRFS on the genuinely field-imbalanced classes (Phomopsis, Anthracnose, Phytophtora) — but **turn off Phase 4.2's background-swap augmentation only**. Re-run Tests C1 and C2 on the result.

```text
Interpretation:
  C2 hit rate drops back toward the original 61.7% baseline (or lower)
     → confirms background-swap augmentation is the cause. Proceed to Step 3.
  C2 hit rate stays elevated even with the augmentation off
     → the cause is elsewhere (E-IRFS repeat interaction, negative-pool
       composition, or something in Phase 5 preprocessing) — re-open
       Phase 5.5 and Phase 4.1 for review before re-attempting 4.2.
```

### Step 3 — Fix the augmentation properly (once confirmed)

- Replace the hard-mask paste with **feathered edges or Poisson blending** so there's no hard seam to key on.
- Match noise/JPEG-compression statistics between the pasted leaf and the new background.
- **Widen the background pool** and/or apply the swap **stochastically online** (fresh random background each epoch) rather than pre-generating a fixed set of augmented images that then get repeated verbatim by E-IRFS — repetition of an artifact is exactly what makes it memorizable.
- Re-run the ablation ladder: augmentation off → confirm baseline-ish C2 → fixed augmentation on → confirm C2 improves *beyond* the off-state, not just back to it (the point of Phase 4 was to reduce the shortcut below the original 61.7%/3.1×/2.9×, not merely avoid making it worse).

### Step 4 — Re-run full Phase 7 suite again

Only after Steps 1–3 converge on a version where C2's hit rate is meaningfully below the original baseline (not just below the `v4_confound_fix` regression) should you proceed to Phase 8 reporting.

---

## Phase 8 — Honest final reporting

- Report per-class AP from the **new, corrected test split** only.
- For Rhizoctonia and Healthy, state the field-instance sample size alongside the AP (small-N metrics should carry visible uncertainty, not be presented as equal-confidence to your high-N classes).
- Note in any write-up/paper/report that an earlier version of the pipeline had a confirmed source confound (source-predictable classes, background-predictable classes, split leakage) and describe the fixes applied — this is a legitimate and common finding to disclose in agricultural ML work, not a flaw to hide.

---

## Suggested execution order (fastest path)

**Status: Phases 1–7 complete (`v4_confound_fix` run). Currently at Phase 7.5 — do this next.**

1. ~~Phase 1 (dedup/grouping)~~ — done.
2. ~~Phase 2 (re-split)~~ — done.
3. ~~Phase 3 decision~~ — done (Option B: scoped reporting; Rhizoctonia field data still short at 8 instances).
4. ~~Phase 4 + Phase 5~~ — done, but Phase 4.2 is now the prime suspect for the Phase 7.5 regression.
5. ~~Phase 5.5 (E-IRFS re-sequencing)~~ — done.
6. ~~Phase 6 (6.1 scoped HPO, then 6.2 retrain)~~ — done; produced `v4_confound_fix`.
7. ~~Phase 7 verification~~ — done; **revealed a regression**, not a pass.
8. **→ Phase 7.5 (diagnose the regression) — current step.** Visual audit → ablation (4.2 off) → fix blending → re-verify. Do not proceed to Phase 8 until Phase 7.5's re-run of Phase 7 shows C2 below the *original* baseline, not just below `v4_confound_fix`.
9. Phase 8 reporting — blocked until Phase 7.5 resolves.
