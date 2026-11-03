# Source-confound fix — scoping decision (Phase 3)

**Decision: Option B — scope honestly, do not block on new field photography.**

## Why

Phase 1/2 re-split (grouped, source+class stratified, dedup'd) gives every
class a full 80/10/10 split with per-class share within ~1pp of the original
dataset (`analysis/dataset/outputs/resplit/phase2_report.txt`). Running the
field-coverage report on that new split
(`analysis/dataset/outputs/resplit/phase3_report.txt`) shows only one class
fails to clear a 50-field-instance reliability bar:

| Class | Field instances | Curated instances | Verdict |
|---|---|---|---|
| Rhizoctonia | 8 | 325 | curated-validated only |
| Healthy | 179 | 254 | field-validated (plenty of both — the earlier AP gap was a shortcut-learning problem, not a sample-size problem; addressed by Phases 4/5) |
| Algal spot / Anthracnose / Other / Phomopsis / Phytophtora | 358–2101 | 0–14 | field-validated |

Only **Rhizoctonia** is data-starved on the field side. Collecting ~150-200
new field Rhizoctonia photos (Option A) is out of scope right now — it
requires physical orchard access this session can't provide.

## What this means for reporting (binds Phase 8)

- Rhizoctonia must be reported as **curated-validated, not field-validated**.
  Do not blend curated+field AP into one number for this class.
- Report Rhizoctonia's curated-set AP and field-set AP separately, and state
  the field sample size (n=8 instances / 5 images) directly next to the
  field AP so it isn't presented with equal confidence to high-N classes.
- Exclude Rhizoctonia from any claim about real-world (field) deployment
  accuracy until field data is actually collected.
- Anthracnose (1 curated instance) and Phytophtora (0 curated instances) are
  effectively field-only classes in this dataset — not a confound to fix,
  just worth stating plainly: there is no meaningful curated-side validation
  for them either.

## What's still done regardless of A vs B

Per the implementation plan, the *source cue itself* (fixed 400×400
resolution, uniform aspect ratio) gets removed independent of this decision
— see `analysis/dataset/phase3_augment_curated.py` and Phase 5's
preprocessing standardization. That work does not wait on a Rhizoctonia
data-collection decision.

## If Option A happens later

Target: ~150-200 new field Rhizoctonia instances, photographed with the same
phones/orchard/lighting as the existing field set (to avoid introducing a
*new* source confound). Re-run `phase1_build_groups.py` →
`phase2_resplit.py` → `phase3_field_coverage_report.py` once added; the
pipeline is idempotent and will just re-stratify with the new data.
