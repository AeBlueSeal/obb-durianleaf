"""
Phase 1 — Deduplicate and build "must-stay-together" groups.

Finds every image that must be locked into a single split (train/val/test)
because it's a near-duplicate, same-field-session, or same-curated-sequence
shot as another image. Combines:
  1. Perceptual-hash near-duplicate pairs (any source, any current split)
  2. Field session bursts (same day, within WINDOW of each other)
  3. Curated sequence runs (same class, nearby index)
into connected components via union-find.

Run from project root:
    python analysis/dataset/phase1_build_groups.py
"""

import os, re, glob, json
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import yaml
from PIL import Image
import imagehash

DATA_YAML  = "data/dataset.yaml"
SPLITS     = ["train", "val", "test"]
HASH_THRESH = 5                     # Hamming distance; matches Test D
# NOTE: field shooting cadence is near-continuous (median gap 4s, 95th pct 37s
# across the whole day) — a 5min window (as in the original plan) transitively
# chains an entire day's walk into one ~280-image group. 10s keeps sessions to
# genuine back-to-back bursts of the same leaf (max observed group size 16)
# while perceptual-hash pairs (1.1) still catch true near-duplicates regardless
# of time gap.
SESSION_WINDOW = timedelta(seconds=10)
SEQ_GAP    = 2                      # curated index gap to still be "same sequence"
OUTPUT_DIR = "analysis/dataset/outputs/resplit"

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

DATA_ROOT = cfg["path"]


def strip_uuid(fname):
    """Label-Studio prefixes files with an 8-char UUID + '-': strip it."""
    stem = os.path.splitext(fname)[0]
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 and len(parts[0]) == 8 else stem


def source_of(filepath):
    name = strip_uuid(os.path.basename(filepath))
    return "curated" if name.startswith("Leaf_") else "field"


# ── Gather every image, every split ────────────────────────────────────
paths = []
for split in SPLITS:
    paths += glob.glob(os.path.join(DATA_ROOT, split, "images", "*"))
paths = sorted(paths)
print(f"Found {len(paths)} images across {SPLITS}")

# ── 1.1 Perceptual hashes and near-duplicate pairs ─────────────────────
print("\nHashing images …")
hashes = {}
for p in paths:
    try:
        hashes[p] = imagehash.phash(Image.open(p))
    except Exception as e:
        print("skip", p, e)

items = list(hashes.items())
pairs = []
for i in range(len(items)):
    pa, ha = items[i]
    for j in range(i + 1, len(items)):
        pb, hb = items[j]
        d = ha - hb
        if d <= HASH_THRESH:
            pairs.append((pa, pb, int(d)))

print(f"{len(pairs)} duplicate/near-duplicate pairs found (Hamming <= {HASH_THRESH})")

# ── 1.2 Field session groups from IMG_YYYYMMDD_HHMMSS bursts ──────────
def parse_session(path):
    name = strip_uuid(os.path.basename(path))
    m = re.search(r"IMG_(\d{8})_(\d{6})", name)
    if not m:
        return None
    date_str, time_str = m.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")


field_paths = [p for p in paths if source_of(p) == "field"]
timestamps = [(p, parse_session(p)) for p in field_paths]
timestamps = [(p, t) for p, t in timestamps if t is not None]
timestamps.sort(key=lambda x: x[1])

session_groups = []
current = []
for p, t in timestamps:
    if current and (t - current[-1][1]) > SESSION_WINDOW:
        session_groups.append([x[0] for x in current])
        current = []
    current.append((p, t))
if current:
    session_groups.append([x[0] for x in current])
session_groups = [g for g in session_groups if len(g) > 1]

print(f"{len(session_groups)} multi-image field sessions "
      f"from {len(timestamps)}/{len(field_paths)} timestamped field images")

# ── 1.3 Curated sequence groups (same class, nearby index) ────────────
def parse_curated(path):
    name = strip_uuid(os.path.basename(path))
    m = re.match(r"Leaf_(\w+)_(\d+)", name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


curated_paths = [p for p in paths if source_of(p) == "curated"]
parsed = [(p, parse_curated(p)) for p in curated_paths]
parsed = [(p, c) for p, c in parsed if c is not None]
parsed.sort(key=lambda x: (x[1][0], x[1][1]))

curated_groups = []
current = []
for p, (cls, idx) in parsed:
    if current and (cls != current[-1][2] or idx - current[-1][1] > SEQ_GAP):
        curated_groups.append([x[0] for x in current])
        current = []
    current.append((p, idx, cls))
if current:
    curated_groups.append([x[0] for x in current])
curated_groups = [[x[0] if isinstance(x, tuple) else x for x in g] for g in curated_groups]
curated_groups = [g for g in curated_groups if len(g) > 1]

print(f"{len(curated_groups)} multi-image curated sequence groups "
      f"from {len(parsed)}/{len(curated_paths)} parsed curated images")

# ── 1.4 Union-find merge ────────────────────────────────────────────────
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

components = defaultdict(list)
for p in paths:
    components[find(p)].append(p)

groups = list(components.values())
print(f"\n{len(paths)} images collapsed into {len(groups)} split-atomic groups")

# ── Check: size distribution ───────────────────────────────────────────
sizes = sorted((len(g) for g in groups), reverse=True)
size_counter = defaultdict(int)
for s in sizes:
    size_counter[s] += 1

print("\nGroup size distribution (size: count of groups):")
for s in sorted(size_counter, reverse=True):
    print(f"  {s:>4}: {size_counter[s]}")

largest = sizes[:10]
print(f"\nLargest groups: {largest}")
if largest and largest[0] > 100:
    print("⚠️  Very large group(s) found (>100 images, i.e. 'hundreds') — "
          "consider tightening SESSION_WINDOW / SEQ_GAP and re-running before Phase 2.")
else:
    print("✅ No oversized groups (largest well under 100) — safe to proceed to Phase 2.")

# ── Save outputs ────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

group_id_of = {}
for gid, g in enumerate(groups):
    for p in g:
        group_id_of[p] = gid

rows = []
for p in paths:
    rows.append({
        "path": p,
        "split_current": next(s for s in SPLITS if f"/{s}/" in p),
        "source": source_of(p),
        "group_id": group_id_of[p],
        "group_size": len(components[find(p)]),
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTPUT_DIR, "phase1_groups.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "phase1_groups.json"), "w") as f:
    json.dump({str(gid): g for gid, g in enumerate(groups)}, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "phase1_report.txt"), "w") as f:
    f.write(f"Images: {len(paths)}\n")
    f.write(f"Near-duplicate pairs (Hamming <= {HASH_THRESH}): {len(pairs)}\n")
    f.write(f"Field sessions (>1 image): {len(session_groups)}\n")
    f.write(f"Curated sequences (>1 image): {len(curated_groups)}\n")
    f.write(f"Split-atomic groups: {len(groups)}\n\n")
    f.write("Group size distribution (size: count):\n")
    for s in sorted(size_counter, reverse=True):
        f.write(f"  {s:>4}: {size_counter[s]}\n")
    f.write(f"\nLargest groups: {largest}\n")

print(f"\nSaved -> {OUTPUT_DIR}/phase1_groups.csv, phase1_groups.json, phase1_report.txt")
