# Phase 8 — Honest final reporting (fill in after retraining + Phase 7 re-verification)

Do not publish/present numbers from this template until every row in
`results/source_confound_fix_phase7.md`'s "pending retrain" section has
been re-run and reviewed.

## Per-class AP — new test split only (`data/dataset_resplit.yaml`)

| Class | mAP@50 | mAP@50-95 | Field instances (n) | Curated instances (n) | Field-validated? |
|---|---|---|---|---|---|
| Algal spot | | | | | |
| Anthracnose | | | | | |
| Healthy | | | | | |
| Other | | | | | |
| Phomopsis | | | | | |
| Phytophtora | | | | | |
| Rhizoctonia | | | | | **No — curated-validated only, n=8 field instances. Report curated-set and field-set AP separately, do not blend. See `results/source_confound_scoping.md`.** |

## Required disclosure (do not omit)

State plainly in any write-up/paper/report using this model:

> An earlier version of this pipeline had a confirmed source confound:
> class was predictable from image source (curated vs. field) at
> Cramér's V = 0.855, curated images had a fixed 400×400 resolution acting
> as a zero-effort source fingerprint, two classes (Algal spot, Other) were
> predictable from background alone at 2.9-3.1× chance, and 17 near-duplicate
> image pairs leaked across train/val/test. These were fixed via grouped
> dedup+re-splitting, source-cue removal (resolution jitter, EXIF strip),
> background-negative/background-swap augmentation, and hard-negative
> mining (see `analysis/dataset/phase1-5_*.py`,
> `results/source_confound_fix_phase7.md`). Rhizoctonia remains
> curated-validated only due to a genuine field-data shortage (n=8), not
> a splitting artifact — treat its field-deployment accuracy as unknown
> until more field data is collected.

## Deployment-claim boundaries

- [ ] Rhizoctonia excluded from any "real-world / field deployment accuracy" claim
- [ ] Blended curated+field AP not reported for Rhizoctonia
- [ ] Field sample size (n) stated next to every field AP number, not just the headline metric
- [ ] Confound disclosure paragraph (above) included in any external write-up
