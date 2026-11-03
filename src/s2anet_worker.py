"""Runs inside the `s2anet` conda env (Python 3.7, torch 1.3, mmcv==0.2.14,
CUDA 10.0) that notebooks/S2ANet_E-IRFS_kfold_datasetv3.py's own "file setup"
section bootstraps -- this project's own Colab kernel can no longer import
that stack directly (torch 1.3 has no wheel for Python 3.11+, and S2A-Net's
own custom CUDA ops -- box_iou_rotated, nms_rotated, the alignment
convolution -- are compiled against that exact old torch/CUDA pair and
cannot be imported from a different Python/torch build). Same architecture
as src/strip_rcnn_worker.py, for the same reason.

Shared, unchanged, by both `notebooks/S2ANet_E-IRFS_kfold_datasetv3.py` and
its ablation twin `notebooks/S2ANet_NoEIRFS_kfold_datasetv3.py` -- the args
JSON's `use_eirfs` key (default `True` if absent) is the only thing that
differs between the two notebooks' worker invocations.

Invoked as a subprocess (not imported) via the conda env's own Python:

    <conda_env>/bin/python src/s2anet_worker.py <args_json_path>

`args_json_path` is a JSON file the calling notebook writes before invoking
this script (see `main()` below for the exact keys), since a subprocess
boundary can't pass live Python objects. This script trains one fold and
predicts on the given test items, writing the shared prediction format
(same as every other arm's own inference code) to `predictions_out`, so the
calling notebook's own kernel can score them with the framework-free
`src/obb_eval.py` evaluator without ever needing mmcv/mmdet itself.

Two behaviours here exist only in this file and are easy to lose sight of from
the notebook side. `patch_empty_gt_support()` makes mmdet train on images with
no instances rather than silently resampling past them (this dataset ships
background-only negatives on purpose). And when the args JSON carries an
optional `ckpt_mirror` path, checkpoints are mirrored there as they are written
and an interrupted fold resumes from the newest one -- without which a
multi-hour fold cannot survive Colab reclaiming the VM and wiping /content.

`project_root` in the args JSON is added to `sys.path` so this script can
import `src.obb_common`'s `EIRFSWeights`/`attach_eirfs` directly -- both are
numpy-only, no mmcv dependency, and the conda env already needs numpy for
mmdet anyway. The annotation files and image directories themselves are built
by the notebook's own kernel (via `src.obb_common`'s
`yolo_obb_to_s2anet_ann`, which is also numpy/PIL-only) BEFORE this worker
ever runs -- as JSON, which `ann_json_to_pkl()` below turns into the pickle
mmdet loads, because a numpy-2 pickle from that kernel is unreadable here
(see either function's docstring). They are built in the exact same item
order this worker was told to train on --
so unlike Strip R-CNN's MMRotate `DOTADataset` (which enumerates images via
an unsorted `glob.glob()` and needs `EIRFSWeights.reorder_for`), this
dataset's `img_infos[i]` already matches `train_labels[i]` 1:1 and no
reordering is needed before attaching E-IRFS.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    args = json.loads(Path(sys.argv[1]).read_text())

    sys.path.insert(0, args["project_root"])
    from src.obb_common import EIRFSWeights, attach_eirfs  # noqa: E402

    from mmcv import Config  # noqa: E402
    from mmdet.apis import init_detector, set_random_seed, train_detector  # noqa: E402
    from mmdet.datasets import DotaDataset, build_dataset  # noqa: E402
    from mmdet.models import build_detector  # noqa: E402

    # DotaDataset.CLASSES is a hardcoded 15-class DOTA v1.0 tuple with no
    # constructor override (unlike MMRotate's DOTADataset, which at least
    # takes a `classes=` kwarg) -- rebind the class attribute itself before
    # building any dataset from it. Cosmetic for training/inference (the
    # annotations carry integer labels, independent of this tuple), but keeps
    # `model.CLASSES` (set from the dataset at build time) correct for
    # anything downstream that prints/logs class names. Note CLASSES holds
    # the NUM_CLASSES real classes while the config's bbox_head gets
    # `num_classes=NUM_CLASSES + 1` and the annotations' labels are 1-based --
    # mmdet-1.x's background-is-0 convention, see yolo_obb_to_s2anet_ann.
    DotaDataset.CLASSES = tuple(args["class_names"])

    cfg = Config.fromfile(args["cfg_path"])

    # Each split's annotations arrive as JSON and are turned into the pickle
    # mmdet's CustomDataset actually loads, right here, with THIS env's numpy.
    # cfg.data.val and cfg.data.test deliberately share one ann_file, so the
    # cache keeps that from being converted twice.
    ann_pkls = {}
    for split in ("train", "val", "test"):
        sub = cfg.data.get(split)
        if sub is None or not str(sub.get("ann_file", "")).endswith(".json"):
            continue
        ann_json = sub["ann_file"]
        if ann_json not in ann_pkls:
            ann_pkls[ann_json] = ann_json_to_pkl(ann_json)
        sub["ann_file"] = ann_pkls[ann_json]

    patch_empty_gt_support()

    # Resume/mirror wiring. A fold here runs for hours and Colab reclaims the
    # VM (wiping /content) between sittings, so the only checkpoint that
    # survives is the one mirrored to Drive -- see patch_checkpoint_mirror.
    resume_from, already_trained = resolve_resume(
        args.get("ckpt_mirror"), args["work_dir"], args["epochs"])
    if args.get("ckpt_mirror"):
        patch_checkpoint_mirror(args["ckpt_mirror"])
    if resume_from:
        cfg.resume_from = resume_from
        print(f"resuming from {resume_from}", flush=True)

    # train_detector() here takes no seed/meta argument at all (confirmed by
    # reading this repo's own mmdet/apis/train.py) -- set_random_seed() must
    # be called explicitly, and BEFORE model construction, or weight init is
    # not reproducible despite everything else being seeded.
    set_random_seed(args["seed"])

    if already_trained:
        # resolve_resume() found the FINAL epoch in the mirror and restored it,
        # so this fold's training is done -- a previous sitting was interrupted
        # somewhere between the last epoch and writing its predictions. Skip
        # straight to inference rather than repeating hours of training.
        print(f"training already complete (epoch_{args['epochs']}.pth restored "
              f"from the mirror) -- re-predicting only", flush=True)
        assert_supported_gpu(args["device"])
    else:
        datasets = [build_dataset(cfg.data.train)]

        # use_eirfs defaults True so any already-written args JSON (predating
        # this flag) keeps its original behavior; the NoEIRFS notebook is the
        # only caller that sets it False.
        if args.get("use_eirfs", True):
            weights = EIRFSWeights(args["train_labels"], args["num_classes"],
                                   args["class_names"], t=args["eirfs_threshold"],
                                   verbose=args["verbose"])
            attach_eirfs(datasets[0], weights, train_mode=True)

        # Placed *after* build_dataset and E-IRFS, deliberately: those are
        # CPU-only and the first CUDA kernel does not launch until
        # train_detector below, so on an unusable GPU everything cheap still
        # gets exercised -- the JSON->pkl conversion, annotation loading, the
        # sampler -- before the run refuses. That makes an unsupported card
        # useful for testing the plumbing, which is how the L4 run that
        # motivated this check was actually being used.
        assert_supported_gpu(args["device"])

        model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        # NOT calling model.init_weights() separately: this era's BaseDetector
        # subclasses call it themselves at the end of their own __init__ using
        # model.cfg's `pretrained=` key (confirmed by the DOTA config's
        # top-level `pretrained='torchvision://resnet50'` -- a model-level key,
        # the mmdet-1.x convention, not the newer per-backbone `init_cfg` style
        # Strip R-CNN's mmdet 2.x uses, which DOES need an explicit call).
        model.CLASSES = datasets[0].CLASSES
        # Matches tools/train.py's own reference main(): without this, the saved
        # checkpoint has no CLASSES in its meta, and init_detector() below would
        # print a spurious "Class names are not saved in the checkpoint's meta
        # data" warning at eval time -- harmless either way (predict_fold below
        # never reads model.CLASSES), but free to avoid.
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(CLASSES=datasets[0].CLASSES)

        train_detector(model, datasets, cfg, distributed=False, validate=False)

    ckpt = Path(args["work_dir"]) / f"epoch_{args['epochs']}.pth"
    if not ckpt.exists():
        raise RuntimeError(
            f"no checkpoint at {ckpt} -- training produced no final weights. "
            f"Check the mmdet log above."
        )

    eval_model = init_detector(args["cfg_path"], str(ckpt), device=f"cuda:{args['device']}")
    records = predict_fold(eval_model, args["test_items"])

    predictions_out = Path(args["predictions_out"])
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.write_text(json.dumps({"predictions": records}))

    print(f"Wrote {len(records)} predictions to {predictions_out}")
    print(f"CHECKPOINT_PATH={ckpt}")


def assert_supported_gpu(device):
    """Refuse anything newer than Turing before a single CUDA kernel launches.

    Re-checked here and not only in the notebook's env-check cell, because that
    cell can be skipped or the runtime's GPU changed after it ran, and the cost
    of finding out late is extreme: on an L4 this arm spent ~38 minutes per fold
    JIT-ing torch's PTX before dying in the first conv2d with
    CUDNN_STATUS_EXECUTION_FAILED. Reading device properties launches no kernel,
    so this costs milliseconds.
    """
    import torch  # noqa: E402

    cap = torch.cuda.get_device_capability(device)
    if cap > (7, 5):
        raise RuntimeError(
            "GPU is sm_{}{} ({}), but this arm needs Turing or older (capability "
            "<= 7.5, e.g. a T4). Its ops are compiled against CUDA 10.0 / cuDNN "
            "7.6, which ship no kernels past Turing, and cuDNN's are closed "
            "binaries it cannot JIT from PTX -- so training here would spend tens "
            "of minutes JIT-ing and then fail in the first convolution. Everything "
            "before this point (annotations, dataset, sampler) did run.".format(
                cap[0], cap[1], torch.cuda.get_device_name(device)))


def patch_empty_gt_support():
    """Let this arm TRAIN on images that have no instances, instead of silently
    resampling a different image in their place.

    This dataset has label-free images on purpose -- see
    analysis/dataset/phase4_background_negatives.py, which adds background-only
    field crops so the model learns to predict nothing on them -- and the YOLO
    arms train on them (Ultralytics treats an empty label file as background).
    mmdet 1.x does not, and the notebooks' `LoadAnnotations(...,
    skip_img_without_anno=False)` alone is not enough to change that: two places
    upstream assume at least one box exists. Both are patched here rather than
    in the cloned repo so the change is version-controlled, survives the clone
    being rebuilt in a fresh runtime, and needs no byte-exact match against
    upstream source.

    1. `MaxIoUAssigner.assign` opens with
       `if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0: raise ValueError(
       'No gt or bboxes')`. With no gt, every anchor is simply a negative, which
       is exactly the training signal a background image should give -- so
       return that instead of raising. The background LABEL must be 0, not -1:
       mmdet 1.x is background-is-0 (its own `assign_wrt_overlaps` builds
       `assigned_labels` with `new_zeros`), whereas -1 is mmdet 2.x's
       "ignore" and would tell the loss to skip every anchor on these images,
       i.e. silently train on nothing at all. An empty `bboxes` still raises:
       that would be a real bug, not a background image.

    2. `rotated_box_to_poly_np` / `poly_to_rotated_box_np` both end in
       `np.array(<empty list>)` for empty input, returning shape `(0,)` instead
       of `(0, 8)` / `(0, 5)`, which breaks `RotatedResize._resize_bboxes`'
       column indexing. They are patched on the PIPELINE module rather than on
       the module that defines them, because
       `mmdet/datasets/pipelines/transforms_rotated.py` does a `from ... import`
       and therefore holds its own binding, which rebinding the original would
       not touch.

    Verified as needing no change: `anchor_target` computes `num_total_pos =
    sum([max(inds.numel(), 1) ...])`, so zero positives cannot divide by zero;
    `SamplingResult.__init__` indexes `gt_bboxes[<empty>, :]`, fine on `(0, 5)`;
    `RotatedRandomFlip.bbox_flip` is pure slicing; and `EIRFSWeights` already
    gives a background image the neutral weight 1.0.
    """
    import torch  # noqa: E402

    import mmdet.datasets.pipelines.transforms_rotated as rotated_tf  # noqa: E402
    from mmdet.core.bbox.assigners import MaxIoUAssigner  # noqa: E402
    from mmdet.core.bbox.assigners.assign_result import AssignResult  # noqa: E402

    if not getattr(MaxIoUAssigner.assign, "_allows_empty_gt", False):
        original_assign = MaxIoUAssigner.assign

        def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None,
                   _orig=original_assign):
            if gt_bboxes.shape[0] == 0 and bboxes.shape[0] > 0:
                n = bboxes.shape[0]
                gt_inds = bboxes.new_zeros((n,), dtype=torch.long)
                max_overlaps = bboxes.new_zeros((n,))
                labels = None if gt_labels is None else gt_inds.new_zeros((n,))
                # AssignResult(num_gts, gt_inds, max_overlaps, labels=...) --
                # positional order taken from assign_wrt_overlaps' own return.
                return AssignResult(0, gt_inds, max_overlaps, labels=labels)
            return _orig(self, bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels)

        assign._allows_empty_gt = True
        MaxIoUAssigner.assign = assign

    def empty_safe(fn, cols):
        if getattr(fn, "_empty_safe", False):
            return fn

        def wrapper(arr, _fn=fn, _cols=cols):
            if len(arr) == 0:
                return np.zeros((0, _cols), dtype=np.float32)
            return _fn(arr)

        wrapper._empty_safe = True
        return wrapper

    rotated_tf.rotated_box_to_poly_np = empty_safe(rotated_tf.rotated_box_to_poly_np, 8)
    rotated_tf.poly_to_rotated_box_np = empty_safe(rotated_tf.poly_to_rotated_box_np, 5)


def resolve_resume(mirror_dir, work_dir, total_epochs):
    """Return ``(resume_from, already_trained)`` from the Drive-side checkpoint
    mirror, so an interrupted fold continues instead of restarting.

    A fold here runs for hours, and between sittings Colab reclaims the VM and
    wipes ``/content`` -- including ``work_dir``. So the mirror (see
    ``patch_checkpoint_mirror``) is the only place a partial fold survives, and
    it is what gets consulted here.

    ``already_trained`` means the mirror holds the FINAL epoch: training
    finished and the previous run died somewhere between that and writing its
    predictions. The checkpoint is copied back into ``work_dir`` and the caller
    skips training entirely. Otherwise the newest epoch strictly below
    ``total_epochs`` is returned for ``cfg.resume_from``, which
    ``_non_dist_train`` hands to ``runner.resume()`` (restoring epoch, iteration
    and optimizer state, not just weights).
    """
    if not mirror_dir or not Path(mirror_dir).is_dir():
        return None, False

    found = {}
    for path in Path(mirror_dir).glob("epoch_*.pth"):
        m = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if m:
            found[int(m.group(1))] = path

    if total_epochs in found:
        dst = Path(work_dir) / f"epoch_{total_epochs}.pth"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(found[total_epochs]), str(dst))
        return None, True

    usable = [n for n in found if n < total_epochs]
    if not usable:
        return None, False
    return str(found[max(usable)]), False


def patch_checkpoint_mirror(mirror_dir):
    """Copy each new checkpoint to ``mirror_dir`` (on Drive) as it is written,
    keeping only the newest on either side.

    ``train_detector`` builds its runner internally and exposes no hook
    registration point, so this wraps ``CheckpointHook.after_train_epoch``
    (mmcv 0.2.14, ``mmcv/runner/hooks/checkpoint.py``) -- which is the method
    that calls ``runner.save_checkpoint`` on its interval. The file it has just
    written is ``epoch_{runner.epoch + 1}.pth``, because ``save_checkpoint``
    formats its name from ``self.epoch + 1`` and the runner only increments
    ``_epoch`` *after* the ``after_train_epoch`` hooks have run.

    Older checkpoints are deleted on both sides. Resume only ever wants the
    newest, and at 250 epochs with a 25-epoch interval keeping them all would be
    ~3.3 GB locally and the same again on Drive. The final epoch is the newest
    when training ends, so the file the caller and the archive step look for is
    always the one kept.
    """
    from mmcv.runner.hooks.checkpoint import CheckpointHook  # noqa: E402

    if getattr(CheckpointHook.after_train_epoch, "_mirrors_checkpoints", False):
        return

    mirror = Path(mirror_dir)
    mirror.mkdir(parents=True, exist_ok=True)
    original = CheckpointHook.after_train_epoch

    def after_train_epoch(self, runner, _orig=original, _mirror=mirror):
        _orig(self, runner)
        src = Path(self.out_dir or runner.work_dir) / f"epoch_{runner.epoch + 1}.pth"
        if not src.exists():
            return                      # not one of this hook's interval epochs
        # Write-then-rename, or a disconnect mid-copy leaves a truncated
        # checkpoint that resolve_resume() would hand straight to runner.resume.
        tmp = _mirror / (src.name + ".tmp")
        shutil.copyfile(str(src), str(tmp))
        tmp.replace(_mirror / src.name)
        for old in list(_mirror.glob("epoch_*.pth")) + list(src.parent.glob("epoch_*.pth")):
            if old.name != src.name:
                old.unlink()
        print(f"mirrored {src.name} -> {_mirror}", flush=True)

    after_train_epoch._mirrors_checkpoints = True
    CheckpointHook.after_train_epoch = after_train_epoch


def ann_json_to_pkl(ann_json):
    """Rebuild `ann_json` as the pickled annotation list mmdet loads, and
    return that path.

    `src.obb_common.yolo_obb_to_s2anet_ann` writes each split's annotations as
    JSON rather than as a pickle because it runs in the notebook's Python
    3.13 / numpy 2.x kernel, and a numpy-2 pickle cannot be read by the numpy
    1.x in this env at all -- numpy 2 reduces every array through
    `numpy._core.multiarray._reconstruct`, a module numpy 1 does not have, so
    `mmcv.load` on such a file dies with `ModuleNotFoundError: No module named
    'numpy._core'`. mmdet's `CustomDataset.load_annotations` hands its
    `ann_file` straight to `mmcv.load`, which picks its handler by suffix and
    would parse a `.json` into plain lists that the rotated pipeline then
    can't do array math on -- hence a real pickle, written here where the
    numpy version is the one mmdet will use.

    Skipped when the `.pkl` is already at least as new as its `.json`: the
    folds are materialised once and every HPO trial and final fold re-runs
    this worker against them.
    """
    import pickle

    ann_json = Path(ann_json)
    pkl_path = ann_json.with_name(ann_json.stem + ".pkl")
    if pkl_path.exists() and pkl_path.stat().st_mtime >= ann_json.stat().st_mtime:
        return str(pkl_path)

    records = json.loads(ann_json.read_text())
    for rec in records:
        ann = rec["ann"]
        # reshape(-1, 5) rather than asarray alone: an image with no instances
        # is `[]` in the JSON, which would come back as shape (0,) and break
        # `RotatedResize._resize_bboxes`' column indexing instead of being the
        # empty (0, 5) that mmdet expects.
        ann["bboxes"] = np.asarray(ann["bboxes"], dtype=np.float32).reshape(-1, 5)
        ann["labels"] = np.asarray(ann["labels"], dtype=np.int64)
        ann["bboxes_ignore"] = np.asarray(ann["bboxes_ignore"], dtype=np.float32).reshape(-1, 5)
        ann["labels_ignore"] = np.asarray(ann["labels_ignore"], dtype=np.int64)

    # Write-then-rename: a run killed mid-write (Colab disconnect, OOM) would
    # otherwise leave a truncated .pkl that is newer than its .json and so
    # passes the freshness check above forever.
    tmp_path = pkl_path.with_name(pkl_path.name + ".tmp")
    with open(str(tmp_path), "wb") as f:
        pickle.dump(records, f)
    tmp_path.replace(pkl_path)
    return str(pkl_path)


def predict_fold(model, items, score_thr=0.001):
    """Run inference over `items` and return records in the shared format.

    `score_thr` is deliberately near-zero: AP integrates the full
    precision/recall curve, so thresholding at a "sensible" confidence
    truncates the curve and understates AP -- same reasoning as every other
    arm's own copy of this function.
    """
    from mmdet.apis.inference import inference_detector
    from mmdet.core.bbox.transforms_rotated import rotated_box_to_poly_single

    records = []
    for img_path, _ in items:
        stem = Path(img_path).stem
        result = inference_detector(model, str(img_path))
        # bbox2result_rotated() (mmdet/core/bbox/transforms_rotated.py)
        # returns one (n, 6) array per real class -- [cx, cy, w, h, angle,
        # score] -- 0-based, no background-class offset (confirmed by
        # reading that function's own source).
        for cid, dets in enumerate(result):
            dets = np.asarray(dets)
            if dets.size == 0:
                continue
            assert dets.shape[1] == 6, (
                f"expected (n, 6) rotated dets [cx,cy,w,h,angle,score], got "
                f"{dets.shape} -- s2anet's result format has changed and the "
                f"polygon conversion below would be silently wrong."
            )
            keep = dets[:, 5] >= score_thr
            dets = dets[keep]
            for det in dets:
                poly = rotated_box_to_poly_single(det[:5])
                records.append({"image": stem, "cls": int(cid),
                                "score": float(det[5]),
                                "poly": [float(v) for v in poly]})
    return records


if __name__ == "__main__":
    main()
