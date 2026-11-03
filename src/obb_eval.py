"""One rotated-box mAP implementation, used to score all four OBB models.

Why this file exists
--------------------
Each framework ships its own metric, and they do not agree:

  * Ultralytics computes its own OBB mAP50-95 with ProbIoU-based matching.
  * MMRotate 0.x's DOTA eval reports VOC-style mAP at IoU 0.5 ONLY -- there
    is no 50:95 number to compare against.
  * RF-DETR's OBB eval callback (PR #921) scores the *axis-aligned envelope*
    of the predicted corners, i.e. it is not a rotated-IoU metric at all.
  * Oriented-DETR reports DOTA-devkit VOC07 mAP50.

Reporting those four numbers side by side in a paper would be reporting four
different metrics. Differences of 1-3 mAP arise from the metric code alone,
which is the same order as the architectural differences being measured. So
every model in this comparison dumps predictions in one common format and is
scored here, by this code, with COCO-style AP at IoU 0.50:0.05:0.95 over
exact polygon IoU.

Common prediction format (JSON)::

    {"predictions": [{"image": "<stem>", "cls": 3, "score": 0.91,
                      "poly": [x1, y1, x2, y2, x3, y3, x4, y4]}, ...]}

Polygons are absolute pixel coordinates in the ORIGINAL image's frame -- each
framework must undo its own letterbox/resize before dumping, otherwise it is
being scored in a different coordinate space to the ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.obb_common import read_yolo_obb_label, image_size

DEFAULT_IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
# COCO's 101-point interpolated recall grid.
RECALL_THRS = np.linspace(0.0, 1.0, 101)


# --------------------------------------------------------------------------
# Polygon IoU
# --------------------------------------------------------------------------
def _shapely_polys(polys):
    from shapely.geometry import Polygon
    out = []
    for p in polys:
        poly = Polygon(np.asarray(p, dtype=np.float64).reshape(4, 2))
        if not poly.is_valid:
            # Self-intersecting quads (a bowtie from a mis-ordered corner
            # list) have undefined area; buffer(0) resolves them to the valid
            # polygon with the same outline instead of raising mid-eval.
            poly = poly.buffer(0)
        out.append(poly)
    return out


def poly_iou_matrix(polys_a, polys_b):
    """Pairwise IoU between two sets of 8-value polygons -> (len(a), len(b))."""
    if len(polys_a) == 0 or len(polys_b) == 0:
        return np.zeros((len(polys_a), len(polys_b)), dtype=np.float64)
    A, B = _shapely_polys(polys_a), _shapely_polys(polys_b)
    areas_a = np.array([p.area for p in A])
    areas_b = np.array([p.area for p in B])
    iou = np.zeros((len(A), len(B)), dtype=np.float64)
    for i, pa in enumerate(A):
        if areas_a[i] <= 0:
            continue
        # Cheap AABB reject before the exact intersection: on this dataset most
        # (pred, gt) pairs in an image do not overlap at all, and shapely's
        # exact intersection is ~50x the cost of a bbox comparison.
        ax0, ay0, ax1, ay1 = pa.bounds
        for j, pb in enumerate(B):
            if areas_b[j] <= 0:
                continue
            bx0, by0, bx1, by1 = pb.bounds
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            inter = pa.intersection(pb).area
            if inter <= 0:
                continue
            iou[i, j] = inter / (areas_a[i] + areas_b[j] - inter)
    return iou


# --------------------------------------------------------------------------
# Ground truth / prediction loading
# --------------------------------------------------------------------------
def load_ground_truth(items):
    """Build the GT structure from (image_path, label_path) pairs.

    Denormalises YOLO-OBB polygons into absolute pixels using each image's own
    size, so GT lands in the same coordinate frame the predictions must be
    dumped in.
    """
    gt = {}
    for img_path, lbl_path in items:
        img_path = Path(img_path)
        w, h = image_size(img_path)
        cls, polys = read_yolo_obb_label(lbl_path)
        polys = polys.copy()
        if len(polys):
            polys[:, 0::2] *= w
            polys[:, 1::2] *= h
        gt[img_path.stem] = {"cls": cls, "polys": polys, "width": w, "height": h}
    return gt


def dump_predictions(path, records):
    """Write the common prediction format. ``records`` is a list of dicts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"predictions": records}))
    return path


def load_predictions(path):
    return json.loads(Path(path).read_text())["predictions"]


# --------------------------------------------------------------------------
# COCO-style AP over polygon IoU
# --------------------------------------------------------------------------
def evaluate_obb(gt, predictions, num_classes, class_names=None,
                 iou_thrs=DEFAULT_IOU_THRS, max_dets=300):
    """COCO-style AP at each IoU threshold, over exact polygon IoU.

    Matching follows COCO: within an image and class, detections are taken in
    descending score order and each greedily claims the highest-IoU unclaimed
    GT above the threshold. Unmatched detections are false positives;
    unmatched GT are false negatives.

    AP is the 101-point interpolated precision, averaged over IoU thresholds
    and then over classes that actually have GT instances. Classes with zero
    GT are excluded rather than scored 0 -- scoring them 0 would let a class
    absent from a fold drag the mean down by 1/num_classes for reasons that
    have nothing to do with the model.

    Returns a dict with ``mAP50-95``, ``mAP50``, ``mAP75`` and per-class
    breakdowns.
    """
    iou_thrs = np.asarray(iou_thrs, dtype=np.float64)

    by_image = {}
    for p in predictions:
        by_image.setdefault(p["image"], []).append(p)

    # Cap detections per image, highest score first (COCO's maxDets rule).
    for stem, preds in by_image.items():
        preds.sort(key=lambda d: -d["score"])
        by_image[stem] = preds[:max_dets]

    ap_table = np.full((num_classes, len(iou_thrs)), np.nan)
    gt_counts = np.zeros(num_classes, dtype=int)

    for cid in range(num_classes):
        # Gather this class's GT and detections across all images.
        gt_by_image, n_gt = {}, 0
        for stem, g in gt.items():
            mask = g["cls"] == cid
            polys = g["polys"][mask]
            gt_by_image[stem] = polys
            n_gt += len(polys)
        gt_counts[cid] = n_gt
        if n_gt == 0:
            continue

        dets = []
        for stem, preds in by_image.items():
            if stem not in gt:
                # A prediction on an image not in this split would otherwise
                # count as a free false positive; surfacing it is more useful
                # than silently absorbing it.
                continue
            for p in preds:
                if p["cls"] == cid:
                    dets.append((p["score"], stem, p["poly"]))
        if not dets:
            ap_table[cid, :] = 0.0
            continue
        dets.sort(key=lambda d: -d[0])

        # Precompute IoU per image once, reused across all IoU thresholds.
        det_idx_by_image = {}
        for di, (_, stem, _) in enumerate(dets):
            det_idx_by_image.setdefault(stem, []).append(di)
        iou_cache = {}
        for stem, dis in det_idx_by_image.items():
            iou_cache[stem] = poly_iou_matrix(
                [dets[di][2] for di in dis], gt_by_image.get(stem, np.zeros((0, 8)))
            )
        # Position of each detection within its image's IoU matrix row set.
        row_of = {}
        for stem, dis in det_idx_by_image.items():
            for row, di in enumerate(dis):
                row_of[di] = row

        for ti, thr in enumerate(iou_thrs):
            claimed = {stem: np.zeros(len(g), dtype=bool) for stem, g in gt_by_image.items()}
            tp = np.zeros(len(dets), dtype=np.float64)
            fp = np.zeros(len(dets), dtype=np.float64)
            for di, (_, stem, _) in enumerate(dets):
                ious = iou_cache[stem][row_of[di]]
                best_iou, best_j = thr, -1
                for j, v in enumerate(ious):
                    if claimed[stem][j] or v < best_iou:
                        continue
                    best_iou, best_j = v, j
                if best_j >= 0:
                    claimed[stem][best_j] = True
                    tp[di] = 1
                else:
                    fp[di] = 1

            tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
            recall = tp_cum / n_gt
            precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
            ap_table[cid, ti] = _interpolated_ap(recall, precision)

    valid = gt_counts > 0
    per_class_5095 = np.nanmean(ap_table, axis=1)
    i50 = int(np.argmin(np.abs(iou_thrs - 0.5)))
    i75 = int(np.argmin(np.abs(iou_thrs - 0.75)))

    names = class_names or [str(i) for i in range(num_classes)]
    return {
        "mAP50-95": float(np.mean(per_class_5095[valid])) if valid.any() else float("nan"),
        "mAP50": float(np.mean(ap_table[valid, i50])) if valid.any() else float("nan"),
        "mAP75": float(np.mean(ap_table[valid, i75])) if valid.any() else float("nan"),
        "per_class": {
            names[c]: {
                "AP50-95": float(per_class_5095[c]),
                "AP50": float(ap_table[c, i50]),
                "AP75": float(ap_table[c, i75]),
                "n_gt": int(gt_counts[c]),
                "AP_per_iou": [float(v) for v in ap_table[c, :]],
            }
            for c in range(num_classes) if valid[c]
        },
        "classes_without_gt": [names[c] for c in range(num_classes) if not valid[c]],
        "iou_thrs": iou_thrs.tolist(),
    }


def _interpolated_ap(recall, precision):
    """COCO 101-point interpolated AP from a precision/recall curve."""
    # Make precision monotonically non-increasing from the right, then sample
    # it at the 101 fixed recall points.
    mpre = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_THRS, side="left")
    q = np.zeros(len(RECALL_THRS))
    valid = idx < len(mpre)
    q[valid] = mpre[idx[valid]]
    return float(q.mean())


# --------------------------------------------------------------------------
# Single-operating-point precision / recall / F1
# --------------------------------------------------------------------------
def precision_recall_f1(gt, predictions, num_classes, iou_thr=0.5, n_points=1000,
                        class_names=None):
    """Overall precision/recall/F1 at the confidence threshold that maximises
    mean F1 across classes with GT -- the same operating point Ultralytics'
    ap_per_class() uses for box.mp/box.mr/box.f1. Matching is greedy
    per-image IoU>=iou_thr (COCO-style, one GT claimed once), independent of
    evaluate_obb()'s multi-IoU-threshold AP computation.

    ``evaluate_obb()`` deliberately has no single confidence operating point
    -- AP integrates the whole precision/recall curve -- so P/R/F1 cannot
    come out of it. This lives here, beside the shared AP metric, because
    every arm's P/R/F1 column has to be produced at the SAME operating point
    or those columns are no more comparable than four different mAPs would
    be. The four non-YOLO K-fold notebooks each carry an inline copy of this
    function that predates this one (verified byte-identical apart from
    comments); they are unchanged and still correct -- this module-level copy
    is what the YOLO arms import, and is the version to consolidate the
    others onto.
    """
    by_image = {}
    for p in predictions:
        by_image.setdefault(p["image"], []).append(p)

    px = np.linspace(0.0, 1.0, n_points)
    p_curve = np.zeros((num_classes, n_points))
    r_curve = np.zeros((num_classes, n_points))
    tp_curve = np.zeros((num_classes, n_points))
    fp_curve = np.zeros((num_classes, n_points))
    valid = np.zeros(num_classes, dtype=bool)
    gt_counts = np.zeros(num_classes, dtype=int)

    names = class_names or [str(i) for i in range(num_classes)]

    for cid in range(num_classes):
        gt_by_image, n_gt = {}, 0
        for stem, g in gt.items():
            mask = g["cls"] == cid
            polys = g["polys"][mask]
            gt_by_image[stem] = polys
            n_gt += len(polys)
        gt_counts[cid] = n_gt
        if n_gt == 0:
            continue
        valid[cid] = True

        dets = []
        for stem, preds in by_image.items():
            if stem not in gt:
                continue
            for p in preds:
                if p["cls"] == cid:
                    dets.append((p["score"], stem, p["poly"]))
        if not dets:
            r_curve[cid, :] = 0.0
            p_curve[cid, :] = 1.0
            continue
        dets.sort(key=lambda d: -d[0])

        claimed = {stem: np.zeros(len(g), dtype=bool) for stem, g in gt_by_image.items()}
        tp = np.zeros(len(dets), dtype=np.float64)
        fp = np.zeros(len(dets), dtype=np.float64)
        for di, (_, stem, poly) in enumerate(dets):
            stem_gt = gt_by_image.get(stem, np.zeros((0, 8)))
            ious = poly_iou_matrix([poly], stem_gt)[0] if len(stem_gt) else np.zeros(0)
            best_iou, best_j = iou_thr, -1
            for j, v in enumerate(ious):
                if claimed[stem][j] or v < best_iou:
                    continue
                best_iou, best_j = v, j
            if best_j >= 0:
                claimed[stem][best_j] = True
                tp[di] = 1
            else:
                fp[di] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
        scores = np.array([d[0] for d in dets])

        # scores is sorted descending; np.interp needs ascending x, so both
        # the confidence grid and scores are negated. This reproduces the
        # step function "recall/precision using every detection with
        # score >= threshold" at each grid point, same trick ap_per_class()
        # uses to build its P/R-vs-confidence curves.
        r_curve[cid] = np.interp(-px, -scores, recall, left=0.0)
        p_curve[cid] = np.interp(-px, -scores, precision, left=1.0)
        # Raw TP/FP counts at the same operating points, purely for the
        # per-class confusion breakdown below -- not used by AP/P/R/F1 math.
        tp_curve[cid] = np.interp(-px, -scores, cum_tp, left=0.0)
        fp_curve[cid] = np.interp(-px, -scores, cum_fp, left=0.0)

    if not valid.any():
        return {"precision": 0.0, "recall": 0.0, "F1": 0.0, "conf": 0.0,
                "per_class": {}, "counts": {}}

    p_valid, r_valid = p_curve[valid], r_curve[valid]
    f1_curve = 2 * p_valid * r_valid / np.maximum(p_valid + r_valid, np.finfo(np.float64).eps)
    best_i = int(f1_curve.mean(axis=0).argmax())

    precision = float(p_valid[:, best_i].mean())
    recall = float(r_valid[:, best_i].mean())
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class = {}
    counts = {}
    for c in range(num_classes):
        if not valid[c]:
            continue
        pc, rc = float(p_curve[c, best_i]), float(r_curve[c, best_i])
        fc = 2 * pc * rc / (pc + rc) if (pc + rc) > 0 else 0.0
        per_class[names[c]] = {"precision": pc, "recall": rc, "F1": fc}
        tp_c = int(round(float(tp_curve[c, best_i])))
        fp_c = int(round(float(fp_curve[c, best_i])))
        counts[names[c]] = {
            "n_gt": int(gt_counts[c]), "tp": tp_c, "fp": fp_c,
            "fn": max(int(gt_counts[c]) - tp_c, 0),
        }

    return {"precision": precision, "recall": recall, "F1": f1,
            "conf": float(px[best_i]), "per_class": per_class, "counts": counts}


# --------------------------------------------------------------------------
# Ultralytics adapter -- lets the YOLO baseline be rescored by THIS metric
# --------------------------------------------------------------------------
def predict_ultralytics(weights, items, imgsz=1024, conf=0.001, iou=0.7,
                        device=0, batch=8):
    """Run a trained Ultralytics OBB model and return common-format records.

    Needed so the YOLO11m-OBB baseline is scored by the same code as the other
    three, rather than by ``model.val()``'s own mAP. Uses a low conf (0.001)
    on purpose: AP integrates over the whole precision/recall curve, so
    thresholding predictions at a "sensible" confidence truncates the curve
    and understates AP.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    records = []
    paths = [str(p) for p, _ in items]
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        results = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou,
                                device=device, task="obb", verbose=False)
        for path, res in zip(chunk, results):
            stem = Path(path).stem
            obb = getattr(res, "obb", None)
            if obb is None or obb.xyxyxyxy is None:
                continue
            polys = obb.xyxyxyxy.reshape(-1, 8).cpu().numpy()
            clss = obb.cls.cpu().numpy().astype(int)
            confs = obb.conf.cpu().numpy()
            for poly, c, s in zip(polys, clss, confs):
                records.append({"image": stem, "cls": int(c), "score": float(s),
                                "poly": [float(v) for v in poly]})
    return records
