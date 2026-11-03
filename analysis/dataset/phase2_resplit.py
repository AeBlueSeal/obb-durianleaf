"""
Phase 2 — Grouped, source-stratified, class-stratified re-split.

Consumes the split-atomic groups from Phase 1 and produces an 80/10/10
train/val/test split where:
  (a) no group is split across sets (enforced structurally — we assign
      whole groups, never individual images)
  (b) per-class source mix (curated/field) is preserved as closely as the
      group structure allows, with a rescue pass so any class that has
      *any* field examples has at least one in val and one in test
  (c) overall class proportions stay close to the original dataset

Materializes the result as a new dataset folder (default:
data/datasetv3_2_resplit) plus data/dataset_resplit.yaml, leaving the
original data/datasetv3_2 untouched.

Run from project root (after phase1_build_groups.py):
    python analysis/dataset/phase2_resplit.py
"""

import os, glob, json, shutil
from collections import Counter

import pandas as pd
import yaml

DATA_YAML   = "data/dataset.yaml"
OUTPUT_DIR  = "analysis/dataset/outputs/resplit"
GROUPS_JSON = os.path.join(OUTPUT_DIR, "phase1_groups.json")
NEW_DATASET_DIR = "data/datasetv3_2_resplit"
NEW_DATA_YAML   = "data/dataset_resplit.yaml"

TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}
RARE_THRESH = 20      # (class, source) combos with fewer total instances than
                       # this get a rescue pass to guarantee val/test presence
SIZE_WEIGHT = 1.0      # weight of overall image-count balance vs class/source balance

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

CLASSES   = cfg["names"]
DATA_ROOT = cfg["path"]
SPLITS    = ["train", "val", "test"]


def strip_uuid(fname):
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 and len(parts[0]) == 8 else stem


def source_of(filepath):
    name = strip_uuid(os.path.basename(filepath))
    return "curated" if name.startswith("Leaf_") else "field"


def label_path_for(img_path):
    split = next(s for s in SPLITS if f"/{s}/" in img_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(DATA_ROOT, split, "labels", stem + ".txt")


# ── Load groups from Phase 1 ────────────────────────────────────────────
with open(GROUPS_JSON) as f:
    groups_raw = json.load(f)
groups = list(groups_raw.values())
print(f"Loaded {len(groups)} groups covering "
      f"{sum(len(g) for g in groups)} images from Phase 1")


# ── Build label_lookup: {image_path: {"classes": Counter, "source": str}} ─
def build_label_lookup(groups):
    lookup = {}
    for g in groups:
        for p in g:
            lbl = label_path_for(p)
            classes = Counter()
            if os.path.exists(lbl):
                with open(lbl) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        cls_id = int(line.split()[0])
                        classes[CLASSES[cls_id]] += 1
            lookup[p] = {"classes": classes, "source": source_of(p)}
    return lookup


label_lookup = build_label_lookup(groups)


def group_signature(group):
    """Return (Counter[(class, source)] over annotations, image count)."""
    sig = Counter()
    for p in group:
        info = label_lookup[p]
        src = info["source"]
        for cls, cnt in info["classes"].items():
            sig[(cls, src)] += cnt
    return sig, len(group)


group_info = [(g, *group_signature(g)) for g in groups]

total_sig = Counter()
total_images = 0
for _, sig, n in group_info:
    total_sig.update(sig)
    total_images += n

print(f"\nTotal (class, source) annotation counts:")
for k, v in sorted(total_sig.items()):
    print(f"  {k}: {v}")

# Sort hardest (largest-signature) groups first
group_info.sort(key=lambda x: -sum(x[1].values()))

split_key_totals = {s: Counter() for s in SPLITS}
split_img_counts  = {s: 0 for s in SPLITS}
split_assign      = {s: [] for s in SPLITS}


def split_need(split, sig, n_imgs):
    # "Need" = how large a share of each split's still-unfilled quota this
    # group would satisfy, measured *before* the group is added. A squared
    # post-assignment error (tried first) has a cold-start trap: once a
    # split gets a head start on a class, small later groups can't move its
    # error enough to look better than continuing to (slightly) overshoot
    # the split that's already converging, so greedy piles everything into
    # one split. Ranking by pre-assignment remaining deficit avoids that —
    # deficit_frac starts at 1.0 identically for every split (0% filled) and
    # drops toward 0 (then negative, once overshot) independently per split
    # as it fills its own quota, so whichever split is furthest from its own
    # target always looks neediest, regardless of how big the group is.
    need = 0.0
    for k, cnt in sig.items():
        total_k = total_sig[k]
        target = TARGETS[split] * total_k
        current = split_key_totals[split].get(k, 0)
        deficit_frac = (target - current) / target
        need += (cnt / total_k) * deficit_frac

    target_imgs = TARGETS[split] * total_images
    size_deficit_frac = (target_imgs - split_img_counts[split]) / target_imgs
    need += SIZE_WEIGHT * size_deficit_frac

    return need


for g, sig, n_imgs in group_info:
    best_split = max(SPLITS, key=lambda s: split_need(s, sig, n_imgs))
    split_assign[best_split].extend(g)
    split_key_totals[best_split].update(sig)
    split_img_counts[best_split] += n_imgs

print("\nAfter greedy assignment:")
for s in SPLITS:
    print(f"  {s}: {split_img_counts[s]} images "
          f"({100 * split_img_counts[s] / total_images:.1f}%)")

# ── Rescue pass: guarantee rare (class, source) combos aren't 100% one split ──
# Only meaningful for the "field" source (curated is present in bulk almost
# everywhere); walk small/pure groups and move one to val, one to test if a
# rare (class, field) combo is currently absent there.
group_by_id = {id(g): g for g, _, _ in group_info}
assigned_to = {}
for s, imgs in split_assign.items():
    for p in imgs:
        assigned_to[p] = s


def move_group(g, from_split, to_split, sig, n_imgs):
    split_assign[from_split] = [p for p in split_assign[from_split] if p not in g]
    split_assign[to_split].extend(g)
    split_key_totals[from_split].subtract(sig)
    split_key_totals[to_split].update(sig)
    split_img_counts[from_split] -= n_imgs
    split_img_counts[to_split] += n_imgs
    for p in g:
        assigned_to[p] = to_split


rare_keys = [k for k, v in total_sig.items() if 0 < v < RARE_THRESH and k[1] == "field"]
for cls, src in rare_keys:
    key = (cls, src)
    missing_splits = [s for s in ["val", "test"] if split_key_totals[s].get(key, 0) == 0]
    if not missing_splits:
        continue
    # candidate donor groups: currently in a split that has surplus of this key,
    # prefer small/pure groups (fewest images, fewest distinct keys) as donors
    candidates = [
        (g, sig, n_imgs) for g, sig, n_imgs in group_info
        if sig.get(key, 0) > 0
    ]
    candidates.sort(key=lambda x: (len(x[0]), sum(x[1].values())))
    for target_split in missing_splits:
        donor = None
        for g, sig, n_imgs in candidates:
            cur_split = assigned_to[g[0]]
            if cur_split == target_split:
                continue
            # don't steal from a split that would then go empty on this key
            if split_key_totals[cur_split].get(key, 0) - sig.get(key, 0) < 0:
                continue
            donor = (g, sig, n_imgs, cur_split)
            break
        if donor:
            g, sig, n_imgs, cur_split = donor
            move_group(g, cur_split, target_split, sig, n_imgs)
            print(f"Rescue: moved group of {n_imgs} image(s) containing "
                  f"({cls}, {src}) from {cur_split} -> {target_split}")
        else:
            print(f"⚠️  No donor group available to rescue ({cls}, {src}) into {target_split}")

print("\nAfter rescue pass:")
for s in SPLITS:
    print(f"  {s}: {split_img_counts[s]} images "
          f"({100 * split_img_counts[s] / total_images:.1f}%)")

# ── Verification ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Verification")
print("=" * 70)

# 1. No group crosses a split (structural — but verify anyway)
bad = 0
for g in groups:
    splits_used = {assigned_to.get(p) for p in g}
    if len(splits_used) != 1:
        bad += 1
        print(f"❌ Group split across sets: {g}")
assert bad == 0, f"{bad} groups crossed a split boundary"
print("✅ No group crosses a split boundary.")

# 2. Class proportions vs original
orig_class_totals = Counter()
for (cls, src), v in total_sig.items():
    orig_class_totals[cls] += v
new_class_by_split = {s: Counter() for s in SPLITS}
for s in SPLITS:
    for k, v in split_key_totals[s].items():
        new_class_by_split[s][k[0]] += v

print("\nClass share of TRAIN split — original overall share vs new train share:")
rows = []
for cls in CLASSES:
    orig_share = orig_class_totals[cls] / sum(orig_class_totals.values()) * 100
    new_share = (new_class_by_split["train"][cls]
                 / max(sum(new_class_by_split["train"].values()), 1) * 100)
    diff = new_share - orig_share
    flag = "⚠️" if abs(diff) > 5 else "✅"
    rows.append({"class": cls, "orig_share_%": round(orig_share, 1),
                 "new_train_share_%": round(new_share, 1),
                 "diff_pp": round(diff, 1), "flag": flag})
class_share_df = pd.DataFrame(rows).set_index("class")
print(class_share_df.to_string())

# 3. Source mix per split per class — flag classes that are 100% one source
#    in any split despite having >0 of the other source overall
print("\nSource mix per split per class (flagging any split at 100% one source "
      "for a class that has both sources overall):")
flags = []
for cls in CLASSES:
    total_curated = total_sig.get((cls, "curated"), 0)
    total_field = total_sig.get((cls, "field"), 0)
    if total_curated == 0 or total_field == 0:
        continue  # genuinely single-source class, nothing to check
    for s in SPLITS:
        c = split_key_totals[s].get((cls, "curated"), 0)
        fld = split_key_totals[s].get((cls, "field"), 0)
        if c + fld == 0:
            continue
        if fld == 0 or c == 0:
            flags.append((s, cls, c, fld))

if flags:
    for s, cls, c, fld in flags:
        print(f"  ⚠️  {s}/{cls}: curated={c}, field={fld} — single-source in this split")
else:
    print("  ✅ Every class with both sources has both sources represented in every "
          "split that contains it.")

# ── Materialize new dataset folder ──────────────────────────────────────
if os.environ.get("PHASE2_SKIP_COPY"):
    print("\nPHASE2_SKIP_COPY set — skipping file materialization (assignment-only run).")
    raise SystemExit(0)

print(f"\nMaterializing new split -> {NEW_DATASET_DIR}")
for s in SPLITS:
    os.makedirs(os.path.join(NEW_DATASET_DIR, s, "images"), exist_ok=True)
    os.makedirs(os.path.join(NEW_DATASET_DIR, s, "labels"), exist_ok=True)

for s in SPLITS:
    for img_path in split_assign[s]:
        shutil.copy2(img_path, os.path.join(NEW_DATASET_DIR, s, "images"))
        lbl_path = label_path_for(img_path)
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(NEW_DATASET_DIR, s, "labels"))

new_cfg = {
    "path": os.path.abspath(NEW_DATASET_DIR),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": cfg["nc"],
    "names": CLASSES,
}
with open(NEW_DATA_YAML, "w") as f:
    yaml.dump(new_cfg, f, sort_keys=False)

print(f"Wrote {NEW_DATA_YAML} -> {NEW_DATASET_DIR}")

# ── Save assignment + reports ───────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
assign_rows = [{"path": p, "new_split": s} for s in SPLITS for p in split_assign[s]]
pd.DataFrame(assign_rows).to_csv(
    os.path.join(OUTPUT_DIR, "phase2_split_assignment.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "phase2_report.txt"), "w") as f:
    f.write("Split sizes:\n")
    for s in SPLITS:
        f.write(f"  {s}: {split_img_counts[s]} images "
                f"({100 * split_img_counts[s] / total_images:.1f}%)\n")
    f.write("\nClass share (train) — original vs new:\n")
    f.write(class_share_df.to_string() + "\n")
    f.write("\nSingle-source-in-split flags:\n")
    if flags:
        for s, cls, c, fld in flags:
            f.write(f"  {s}/{cls}: curated={c}, field={fld}\n")
    else:
        f.write("  none\n")

print(f"\nSaved -> {OUTPUT_DIR}/phase2_split_assignment.csv, phase2_report.txt")
print(f"New dataset config -> {NEW_DATA_YAML}")
