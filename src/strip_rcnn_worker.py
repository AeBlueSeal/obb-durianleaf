"""Runs inside the `strip_rcnn` conda env (Python 3.10, torch 1.13.1,
mmcv-full 1.7.2, mmdet 2.28.2) that the Strip R-CNN notebooks bootstrap in
their "file setup" section -- this project's own Colab kernel can no longer
import that stack directly, since Colab's default runtime moved to Python
3.11+/3.13, for which no torch==1.13.1 wheel exists at all.

Invoked as a subprocess (not imported) from
notebooks/StripRCNN_E-IRFS_{optuna,kfold}_datasetv3.py's own Python-3.13
kernel, via the conda env's own Python binary:

    <conda_env>/bin/python src/strip_rcnn_worker.py <args_json_path>

`args_json_path` is a JSON file (see `main()` below for the exact keys) the
calling notebook writes before invoking this script, since a subprocess
boundary can't pass live Python objects. This script trains one fold and
predicts on the given test items, writing the shared prediction format
(same as every other arm's `src/obb_eval.py`-scored output) to
`predictions_out`, so the calling notebook's own Python-3.13 kernel can
score them with the framework-free `src/obb_eval.py` evaluator without ever
needing mmcv/mmdet/mmrotate itself.

`project_root` in the args JSON is added to `sys.path` so this script can
import `src.obb_common`'s `EIRFSWeights`/`attach_eirfs` directly -- both are
numpy-only (no mmcv dependency), and the conda env already needs numpy for
mmdet/mmrotate anyway, so no vendoring/duplication is needed for those.
`rboxes_to_polys` below is NOT importable that way (it's currently defined
directly inside the Strip R-CNN notebooks, not in `src/obb_common.py`), so
it's duplicated here -- kept in sync with the notebooks' own copy by hand,
same convention this project already uses for other small per-arm pieces.
"""
from __future__ import annotations

import os

# Must be set before mmdet is ever imported (mmdet.core.visualization pulls
# in matplotlib.pyplot at import time, which reads this to set
# rcParams['backend']). Belt-and-suspenders alongside the calling notebook's
# own subprocess.run(..., env=...) override: this script is invoked as a
# subprocess, but if MPLBACKEND is left inherited from a caller that runs
# under Colab's IPython kernel (which sets it to
# "module://matplotlib_inline.backend_inline"), matplotlib in THIS (separate,
# headless, non-notebook) conda env doesn't recognise that backend string and
# raises ValueError before training ever starts. setdefault, not overwrite --
# an explicit MPLBACKEND the caller actually wants is left alone.
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
from pathlib import Path

import numpy as np


def rboxes_to_polys(rboxes):
    """(n, 5) le90 [cx, cy, w, h, theta_rad] -> (n, 8) corner polygons.

    Kept in sync with the Strip R-CNN notebooks' own copy of this function --
    see their docstring for why it's written out rather than imported from
    `mmrotate.core.obb2poly_np`.
    """
    rboxes = np.asarray(rboxes, dtype=np.float64).reshape(-1, 5)
    cx, cy, w, h, a = rboxes.T
    cos, sin = np.cos(a), np.sin(a)
    dx = np.stack([-w / 2, w / 2, w / 2, -w / 2], axis=1)
    dy = np.stack([-h / 2, -h / 2, h / 2, h / 2], axis=1)
    xs = cx[:, None] + dx * cos[:, None] - dy * sin[:, None]
    ys = cy[:, None] + dx * sin[:, None] + dy * cos[:, None]
    return np.stack([xs, ys], axis=2).reshape(-1, 8)


def predict_fold(model, items, score_thr=0.001):
    """Run inference over `items` and return records in the shared format.

    `score_thr` is deliberately near-zero: AP integrates the full
    precision/recall curve, so thresholding at a "sensible" confidence
    truncates the curve and understates AP -- same reasoning as every other
    arm's own copy of this function.
    """
    from mmdet.apis import inference_detector

    records = []
    for img_path, _ in items:
        stem = Path(img_path).stem
        result = inference_detector(model, str(img_path))
        for cid, dets in enumerate(result):
            dets = np.asarray(dets)
            if dets.size == 0:
                continue
            assert dets.shape[1] == 6, (
                f"expected (n, 6) rotated dets [cx,cy,w,h,angle,score], got "
                f"{dets.shape} -- MMRotate's result format has changed and the "
                f"polygon conversion below would be silently wrong."
            )
            keep = dets[:, 5] >= score_thr
            dets = dets[keep]
            if not len(dets):
                continue
            polys = rboxes_to_polys(dets[:, :5])
            for poly, score in zip(polys, dets[:, 5]):
                records.append({"image": stem, "cls": int(cid),
                                "score": float(score),
                                "poly": [float(v) for v in poly]})
    return records


def main():
    args = json.loads(Path(sys.argv[1]).read_text())

    sys.path.insert(0, args["project_root"])
    from src.obb_common import EIRFSWeights, attach_eirfs  # noqa: E402

    from mmcv import Config  # noqa: E402
    from mmdet.apis import init_detector, set_random_seed, train_detector  # noqa: E402
    from mmrotate.datasets import build_dataset  # noqa: E402
    from mmrotate.models import build_detector  # noqa: E402

    cfg = Config.fromfile(args["cfg_path"])

    # train_detector() only threads cfg.seed into the DATALOADER's worker
    # seeding -- it does NOT seed the main process, so without this call,
    # model weight init (build_detector/init_weights, below) and any other
    # torch/numpy/random draws before the dataloader spins up are NOT
    # reproducible across runs despite cfg.seed being set (confirmed by
    # reading mmdet.apis.train.train_detector's actual source -- same fact
    # this project's notebooks already rely on elsewhere).
    set_random_seed(cfg.seed)

    datasets = [build_dataset(cfg.data.train)]

    # use_eirfs defaults True so any already-written args JSON (predating
    # this flag) keeps training with E-IRFS active, matching every call site
    # that exists before the NoEIRFS ablation twin -- see
    # src/s2anet_worker.py's identical convention.
    if args.get("use_eirfs", True):
        weights = EIRFSWeights(args["train_labels"], args["num_classes"], args["class_names"],
                               t=args["eirfs_threshold"], verbose=args["verbose"])

        # MMRotate's DOTADataset.load_annotations() enumerates images via an
        # unsorted glob.glob(), which does NOT match the alphabetically-sorted
        # order EIRFSWeights was built from -- reorder to the dataset's own
        # actual per-index order before attaching. See EIRFSWeights.reorder_for's
        # docstring in src/obb_common.py for the full explanation.
        dataset_stems = [Path(info["filename"]).stem for info in datasets[0].data_infos]
        weights = weights.reorder_for(dataset_stems)
        attach_eirfs(datasets[0], weights, train_mode=True)

    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))

    # StripNet's backbone config hardcodes norm_cfg=dict(type='SyncBN', ...)
    # (upstream's own multi-GPU DDP default), which builds real
    # torch.nn.SyncBatchNorm layers. In TRAINING mode, SyncBatchNorm.forward()
    # unconditionally calls torch.distributed.get_world_size() (confirmed
    # against torch 1.13.1's own source: `need_sync = bn_training and
    # self.training`, true whenever the module is in .train() mode) --
    # training here is single-process (distributed=False, no
    # init_process_group() call anywhere), so this raises
    # "Default process group has not been initialized" on the very first
    # forward pass. mmcv ships exactly this fix for exactly this scenario --
    # revert_sync_batchnorm recursively swaps SyncBatchNorm for a plain,
    # dimension-agnostic BatchNorm subclass while preserving weights/running
    # stats, the same utility mmdetection's own single-GPU test/demo scripts
    # use. Not needed for the later eval model (init_detector below runs in
    # .eval() mode, where SyncBatchNorm's forward never reaches that call).
    from mmcv.cnn.utils import revert_sync_batchnorm
    model = revert_sync_batchnorm(model)

    model.init_weights()
    model.CLASSES = datasets[0].CLASSES

    train_detector(model, datasets, cfg, distributed=False, validate=False,
                   timestamp=None, meta=dict(seed=cfg.seed))

    ckpt = Path(args["work_dir"]) / f"epoch_{args['epochs']}.pth"
    if not ckpt.exists():
        raise RuntimeError(
            f"no checkpoint at {ckpt} -- training produced no final weights. "
            f"Check the MMRotate log above."
        )

    eval_model = init_detector(args["cfg_path"], str(ckpt), device=f"cuda:{args['device']}")
    records = predict_fold(eval_model, args["test_items"])

    predictions_out = Path(args["predictions_out"])
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.write_text(json.dumps({"predictions": records}))

    print(f"Wrote {len(records)} predictions to {predictions_out}")
    print(f"CHECKPOINT_PATH={ckpt}")


if __name__ == "__main__":
    main()
