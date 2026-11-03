"""Rescore the existing YOLO11m-OBB baseline with the shared OBB evaluator.

The three new architecture notebooks all report `src/obb_eval.py`'s
rotated-IoU mAP50-95. The YOLO arm's existing numbers come from Ultralytics'
`model.val()`, which uses its own matching and its own AP integration. Putting
the two side by side in a table would be comparing metrics, not models.

This script produces the missing cell: the YOLO baseline's held-out test score
under exactly the code that scores the other three arms.

Run locally (inference only -- no training):

    python analysis/comparison/rescore_baseline_yolo.py \
        --weights runs/v4_confound_fix/train/weights/best.pt \
        --data-root data/datasetv3_2_resplit \
        --imgsz 1024 \
        --out results/comparison/yolo_baseline_shared_metric.json

NOTE ON WHICH WEIGHTS TO USE
----------------------------
`runs/v4_confound_fix` is NOT the fixed model -- see CLAUDE.md and
results/implementation_plan.md Phase 7.5: its background shortcut got worse
(masked-image hit rate 61.7% -> 83.7%). Rescoring it here gives a metric-
comparable baseline number, not a validated one. Whichever checkpoint Phase
7.5 settles on is the one that belongs in the final comparison table.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.obb_common import collect_items, load_class_names  # noqa: E402
from src.obb_eval import (  # noqa: E402
    dump_predictions, evaluate_obb, load_ground_truth, predict_ultralytics,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="Deliberately near-zero: AP integrates the whole PR curve, "
                         "so a higher threshold truncates it and understates AP.")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU during prediction.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="results/comparison/yolo_baseline_shared_metric.json")
    args = ap.parse_args()

    root = Path(args.data_root)
    class_names = load_class_names(root / "classes.txt")
    items = collect_items(root / args.split / "images", root / args.split / "labels")
    if not items:
        raise SystemExit(f"no labelled images under {root / args.split}")
    print(f"{len(items)} images in split '{args.split}', {len(class_names)} classes")

    device = int(args.device) if args.device.isdigit() else args.device
    records = predict_ultralytics(args.weights, items, imgsz=args.imgsz,
                                  conf=args.conf, iou=args.iou, device=device)
    print(f"{len(records)} raw predictions")

    gt = load_ground_truth(items)
    metrics = evaluate_obb(gt, records, len(class_names), class_names)
    metrics["_meta"] = {
        "weights": str(args.weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.iou,
        "n_images": len(items),
        "n_predictions": len(records),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    dump_predictions(out.with_name(out.stem + "_predictions.json"), records)

    print(f"\n=== YOLO11m-OBB baseline | {args.split} (shared evaluator) ===")
    print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"mAP50   : {metrics['mAP50']:.4f}")
    print(f"mAP75   : {metrics['mAP75']:.4f}")
    for name, m in metrics["per_class"].items():
        print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
