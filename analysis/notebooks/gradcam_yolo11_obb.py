# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: .venv (3.14.6.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Grad-CAM for YOLOv11-OBB — Durian Leaf Disease Detection
#
# Visualises which image regions drive detections via Gradient-weighted Class Activation Maps.
#
# > **Kernel**: select the Python 3.9 interpreter that has `torch` installed  
# > (`/Library/Developer/CommandLineTools/usr/bin/python3.9` on this Mac)

# %%
# ── 0. Install / verify dependencies ──────────────────────────────────────
import importlib, subprocess, sys

def ensure(pkg, import_name=None):
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print(f'Installing {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

ensure('ultralytics')
ensure('opencv-python', 'cv2')
ensure('matplotlib')

import torch
from ultralytics import __version__ as ult_ver
print(f'ultralytics {ult_ver}  |  torch {torch.__version__}')
print(f'MPS available : {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')

# %%
# ── 1. Configuration ───────────────────────────────────────────────────────
import os
import torch

# ── paths (edit if needed) ─────────────────────────────────────────────────
MODEL_PATH = '../result/run-E-IRFS-optuna14_1_2026_new_data/obb/runs/optuna_obb/best_final_all_m/weights/best.pt'
OUTPUT_DIR = '../result/gradcam_outputs'

# ── model settings ─────────────────────────────────────────────────────────
IMG_SIZE    = 1024
CONF_THRESH = 0.25
IOU_THRESH  = 0.45

# Inference device: MPS on Apple Silicon, CUDA if available, else CPU
if torch.backends.mps.is_available():
    INFER_DEVICE = 'mps'
elif torch.cuda.is_available():
    INFER_DEVICE = 'cuda:0'
else:
    INFER_DEVICE = 'cpu'

# Grad-CAM backward pass: auto-falls back to CPU if MPS raises (see YOLOGradCAM.compute)
CAM_DEVICE = INFER_DEVICE

# ── Grad-CAM layer ─────────────────────────────────────────────────────────
# YOLOv11m-OBB layer map (24 total, index 0-23):
#   9  / -15  SPPF    – end of backbone,  32×32,  global context
#  13  / -11  C3k2    – neck P4,           64×64
#  16  /  -8  C3k2    – neck P3,          128×128  ← default (best localisation)
#  19  /  -5  C3k2    – neck P4 (merged),  64×64
#  22  /  -2  C3k2    – neck P5 (merged),  32×32
#  23  /  -1  OBB     – detection head
TARGET_LAYER_IDX = -8     # C3k2 at P3, 128×128 – tightest heatmaps on disease spots

# None → highest-confidence detection across all classes
# int  → specific class index (0=Algal spot … 6=Rhizoctonia)
TARGET_CLASS = None

# ── class metadata ─────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Algal spot', 'Anthracnose', 'Healthy', 'Other',
    'Phomopsis', 'Phytophtora', 'Rhizoctonia',
]
# BGR colours for OpenCV draw functions
COLORS = [
    (0,   165, 255),   # Algal spot  – orange
    (0,   0,   255),   # Anthracnose – red
    (0,   200, 0  ),   # Healthy     – green
    (200, 200, 200),   # Other       – grey
    (255, 0,   0  ),   # Phomopsis   – blue
    (255, 0,   255),   # Phytophtora – magenta
    (0,   255, 255),   # Rhizoctonia – yellow
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Inference device : {INFER_DEVICE}')
print(f'Grad-CAM device  : {CAM_DEVICE}')

# %%
# ── 2. Extract test images from zip ───────────────────────────────────────
import zipfile, pathlib

ZIP_PATH   = '../data/datasetv3_2.zip'
EXTRACT_TO = '../data/sample_images_for_gradcam'

if not os.path.isdir(EXTRACT_TO):
    os.makedirs(EXTRACT_TO, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        imgs = [
            f for f in z.namelist()
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and '__MACOSX' not in f
            and '/test/images/' in f
        ][:8]
        for f in imgs:
            dest = os.path.join(EXTRACT_TO, os.path.basename(f))
            with open(dest, 'wb') as out:
                out.write(z.read(f))
    print(f'Extracted {len(imgs)} test images → {EXTRACT_TO}/')
else:
    print(f'Using existing images in {EXTRACT_TO}/')

sample_images = sorted(
    list(pathlib.Path(EXTRACT_TO).glob('*.jpg')) +
    list(pathlib.Path(EXTRACT_TO).glob('*.png'))
)
print('Available:', [p.name for p in sample_images])

# %%
# ── 3. Load model & inspect layer map ─────────────────────────────────────
from ultralytics import YOLO

model     = YOLO(MODEL_PATH)
nn_model  = model.model          # underlying nn.Module
nn_model.eval()

print('── Layer index map ──')
for i, m in enumerate(nn_model.model):
    print(f'  [{i:>3}]  {type(m).__name__}')

target_layer = nn_model.model[TARGET_LAYER_IDX]
print(f'\n✓ Grad-CAM hook → [{TARGET_LAYER_IDX}] {type(target_layer).__name__}')

# %%
# ── 4. Grad-CAM implementation ─────────────────────────────────────────────
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np


class YOLOGradCAM:
    """
    Grad-CAM for ultralytics YOLOv11-OBB.

    Two root-cause fixes required for Apple Silicon / torch ≥ 2.7
    ---------------------------------------------------------------
    1. model.predict() calls fuse(), setting all param requires_grad=False.
       Fix → x.requires_grad_(True) anchors the computation graph via input.

    2. model.predict() runs under torch.no_grad(); on MPS this marks
       self.anchors / self.strides in OBBDetect as 'inference tensors'
       which cannot be saved for backward.
       Fix → reset head.shape = None before each Grad-CAM forward so the
       detection head recomputes anchors inside torch.enable_grad().

    Score function
    --------------
    Using cls_scores.sum() instead of .max() distributes gradient across
    ALL anchors at ALL FPN scales, giving meaningful signal at every layer.
    """

    def __init__(
        self,
        nn_model: torch.nn.Module,
        target_layer: torch.nn.Module,
        device: str = 'cpu',
    ):
        self._device     = device
        self._nn_model   = nn_model
        self._layer_idx  = list(nn_model.model).index(target_layer)
        self._target_lyr = nn_model.model[self._layer_idx]

        self._acts  = None
        self._grads = None
        self._fh    = self._target_lyr.register_forward_hook(self._fwd_hook)
        self._bh    = self._target_lyr.register_full_backward_hook(self._bwd_hook)

    # ── hooks ──────────────────────────────────────────────────────────────
    def _fwd_hook(self, module, inp, out):
        self._acts = (out[0] if isinstance(out, (list, tuple)) else out).detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        g = grad_out[0] if isinstance(grad_out, (list, tuple)) else grad_out
        self._grads = g.detach()

    # ── public API ─────────────────────────────────────────────────────────
    def compute(
        self,
        img_tensor: torch.Tensor,          # [1, 3, H, W]  float32  0–1
        target_class: Optional[int] = None,
        nc: int = len(CLASS_NAMES),
    ) -> np.ndarray:                       # [H, W]  float32  0–1

        self._acts  = None
        self._grads = None

        for dev in [self._device, 'cpu']:
            try:
                cam = self._compute_on_device(img_tensor, dev, target_class, nc)
                if dev != self._device:
                    print('[GradCAM] fell back to CPU.')
                return cam
            except Exception as e:
                if dev == 'cpu':
                    raise RuntimeError(f'GradCAM failed on CPU too: {e}') from e
                self._acts  = None
                self._grads = None

    def _compute_on_device(
        self,
        img_tensor: torch.Tensor,
        dev: str,
        target_class: Optional[int],
        nc: int,
    ) -> np.ndarray:

        self._nn_model.to(dev).eval()

        # FIX 2: model.predict() caches anchors as MPS inference tensors.
        # Reset head.shape so they are recomputed fresh inside enable_grad().
        head = self._nn_model.model[-1]
        if hasattr(head, 'shape'):
            head.shape = None

        x = img_tensor.to(dev)
        # FIX 1: fuse() sets all param requires_grad=False; anchor the
        # computation graph through the input tensor instead.
        x.requires_grad_(True)

        self._nn_model.zero_grad()

        with torch.enable_grad():
            raw_out = self._nn_model(x)

        # ultralytics eval → (decoded_preds, raw_features)
        preds = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
        # preds: [1, 5+nc, n_anchors] → normalise to [1, n_anchors, D]
        if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
            preds = preds.permute(0, 2, 1)

        D = preds.shape[-1]
        if D >= nc:
            # OBBDetect already sigmoid()s class scores; don't apply again.
            cls_scores = preds[..., -nc:]          # [1, n_anchors, nc]

            # Use SUM (not max) so gradient flows to every FPN scale/anchor.
            # max() routes all gradient through one anchor, starving other
            # layers; sum() gives meaningful signal at every network depth.
            score = (
                cls_scores[..., target_class].sum()
                if target_class is not None
                else cls_scores.sum()
            )
        else:
            score = preds.sum()

        self._nn_model.zero_grad()
        score.backward()

        acts  = self._acts
        grads = self._grads
        if acts is None or grads is None:
            raise RuntimeError('Hooks did not capture activations/gradients.')

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * acts).sum(dim=1)
        cam     = F.relu(cam).squeeze(0)

        cam_np = cam.cpu().float().numpy()
        lo, hi = cam_np.min(), cam_np.max()
        if hi - lo > 1e-8:
            cam_np = (cam_np - lo) / (hi - lo)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()


print('YOLOGradCAM class defined.')

# %%
# ── 5. Image helpers ───────────────────────────────────────────────────────
import cv2
import numpy as np
import torch


def load_and_preprocess(image_path: str, img_size: int = IMG_SIZE):
    """Read BGR image → letterboxed [1, 3, H, W] float tensor in [0, 1]."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    orig_h, orig_w = img_bgr.shape[:2]

    scale = img_size / max(orig_h, orig_w)
    nh, nw = int(round(orig_h * scale)), int(round(orig_w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    pad_y, pad_x = (img_size - nh) // 2, (img_size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    rgb     = canvas[:, :, ::-1].copy()          # BGR → RGB
    tensor  = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor  = tensor.unsqueeze(0)                 # [1, 3, H, W]

    meta = {
        'orig_bgr': img_bgr, 'canvas_bgr': canvas,
        'scale': scale, 'pad_x': pad_x, 'pad_y': pad_y,
    }
    return tensor, meta


def cam_to_heatmap(cam: np.ndarray, size: 'tuple') -> np.ndarray:
    """Resize [h,w] CAM to (W, H) and apply JET colourmap → BGR uint8."""
    resized  = cv2.resize(cam, size, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)


def overlay_cam(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap, alpha, 0)


def _xywhr_to_corners(xywhr: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h, angle_rad] → 4 corner points [4, 2]."""
    out = []
    for x, y, w, h, a in xywhr:
        cos_a, sin_a = np.cos(a), np.sin(a)
        half_w, half_h = w / 2, h / 2
        # corners relative to centre
        rel = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h],
        ])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = (rot @ rel.T).T + np.array([x, y])
        out.append(corners)
    return np.array(out)   # [N, 4, 2]


def draw_obb_on_image(
    img_bgr: np.ndarray,
    result,
    conf_thresh: float = CONF_THRESH,
) -> np.ndarray:
    """Draw rotated bounding boxes from a YOLO OBB result object."""
    out = img_bgr.copy()
    if result.obb is None or len(result.obb.conf) == 0:
        return out

    obb   = result.obb
    confs = obb.conf.cpu().numpy()
    clss  = obb.cls.cpu().numpy().astype(int)

    # Prefer xyxyxyxy (added in ultralytics 8.1); fall back to xywhr.
    if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None:
        corners_all = obb.xyxyxyxy.cpu().numpy()   # [N, 4, 2]
    else:
        xywhr = obb.xywhr.cpu().numpy()            # [N, 5]
        corners_all = _xywhr_to_corners(xywhr)

    for i in range(len(confs)):
        if confs[i] < conf_thresh:
            continue
        pts   = corners_all[i].astype(np.int32)    # [4, 2]
        color = COLORS[clss[i] % len(COLORS)]
        label = f'{CLASS_NAMES[clss[i]]} {confs[i]:.2f}'
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, color, 2)
        tx, ty = int(pts[0, 0]), int(pts[0, 1])
        cv2.putText(out, label, (tx, max(ty - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


print('Image helpers defined.')

# %%
# ── 6. Instantiate Grad-CAM ────────────────────────────────────────────────
grad_cam = YOLOGradCAM(
    nn_model     = nn_model,
    target_layer = target_layer,
    device       = CAM_DEVICE,
)
print(f'GradCAM hooked on layer [{TARGET_LAYER_IDX}] {type(target_layer).__name__}')

# %%
# ── 7. Run Grad-CAM on a single image ─────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def run_gradcam_single(
    image_path: str,
    target_class: Optional[int] = None,
    alpha: float = 0.45,
    save: bool = True,
):
    img_path = str(image_path)
    print(f'Processing: {img_path}')

    # 1. Preprocess → letterboxed canvas + input tensor
    tensor, meta = load_and_preprocess(img_path)
    canvas_bgr   = meta['canvas_bgr']

    # 2. YOLO inference — pass canvas_bgr (not img_path) so that the
    #    returned OBB coordinates are in canvas/model space (1024×1024),
    #    not in the original image space.
    nn_model.to(INFER_DEVICE).eval()
    with torch.no_grad():
        results = model.predict(
            source  = canvas_bgr,
            imgsz   = IMG_SIZE,
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            device  = INFER_DEVICE,
            verbose = False,
        )
    result = results[0]
    n_det  = len(result.obb.conf) if result.obb is not None else 0
    print(f'  Detections: {n_det}')

    # 3. Grad-CAM (handles device placement and MPS fallback internally)
    cam = grad_cam.compute(tensor, target_class=target_class)
    nn_model.to(INFER_DEVICE).eval()   # restore if Grad-CAM moved model to CPU

    # 4. Build panels
    sz       = (IMG_SIZE, IMG_SIZE)
    heatmap  = cam_to_heatmap(cam, sz)
    overlay  = overlay_cam(canvas_bgr, heatmap, alpha=alpha)
    with_obb = draw_obb_on_image(overlay.copy(), result)
    orig_obb = draw_obb_on_image(canvas_bgr.copy(), result)

    # 5. Plot  (all panels are BGR → flip last axis for matplotlib RGB)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    panels = [
        ('Original + OBB',      orig_obb[:, :, ::-1]),
        ('Grad-CAM Heatmap',    heatmap[:, :, ::-1]),
        ('Overlay',             overlay[:, :, ::-1]),
        ('Overlay + OBB Boxes', with_obb[:, :, ::-1]),
    ]
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel); ax.set_title(title, fontsize=11); ax.axis('off')

    cls_label = CLASS_NAMES[target_class] if target_class is not None else 'all classes'
    fig.suptitle(
        f'Grad-CAM  |  {os.path.basename(img_path)}'
        f'  |  target: {cls_label}'
        f'  |  layer [{TARGET_LAYER_IDX}] {type(target_layer).__name__}',
        fontsize=11,
    )
    patches = [
        mpatches.Patch(color=tuple(c / 255 for c in color[::-1]), label=name)
        for color, name in zip(COLORS, CLASS_NAMES)
    ]
    fig.legend(handles=patches, loc='lower center', ncol=len(CLASS_NAMES),
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()

    if save:
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f'{stem}_gradcam.jpg')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'  Saved → {out_path}')

    plt.show()
    return cam, result


# ── Run on first sample image ──────────────────────────────────────────────
if sample_images:
    cam, result = run_gradcam_single(sample_images[0], target_class=TARGET_CLASS)
else:
    print('No sample images found – add images to sample_images/ or set image_path manually.')

# %%
# ── 8. Batch: run on all sample images ────────────────────────────────────
for img_path in sample_images:
    run_gradcam_single(img_path, save=True)
    plt.close('all')


# %%
# ── 9. Class-specific Grad-CAM ─────────────────────────────────────────────
# Shows one heatmap per disease class detected in the image.

def run_gradcam_per_class(image_path: str, alpha: float = 0.40):
    img_path     = str(image_path)
    tensor, meta = load_and_preprocess(img_path)
    canvas_bgr   = meta['canvas_bgr']

    nn_model.to(INFER_DEVICE).eval()
    with torch.no_grad():
        results = model.predict(
            source=canvas_bgr, imgsz=IMG_SIZE,
            conf=CONF_THRESH, iou=IOU_THRESH,
            device=INFER_DEVICE, verbose=False,
        )
    result = results[0]

    if result.obb is not None and len(result.obb.cls):
        present = sorted(set(result.obb.cls.cpu().numpy().astype(int).tolist()))
    else:
        present = list(range(len(CLASS_NAMES)))

    n = len(present)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    if n == 0:
        return

    axes[0].imshow(draw_obb_on_image(canvas_bgr.copy(), result)[:, :, ::-1])
    axes[0].set_title('Original + OBB', fontsize=10)
    axes[0].axis('off')

    for ax, cls_idx in zip(axes[1:], present):
        cam     = grad_cam.compute(tensor, target_class=cls_idx)
        nn_model.to(INFER_DEVICE).eval()
        heatmap = cam_to_heatmap(cam, (IMG_SIZE, IMG_SIZE))
        overlay = draw_obb_on_image(overlay_cam(canvas_bgr, heatmap, alpha), result)
        ax.imshow(overlay[:, :, ::-1])
        ax.set_title(f'[{cls_idx}] {CLASS_NAMES[cls_idx]}', fontsize=10)
        ax.axis('off')

    fig.suptitle(f'Class-specific Grad-CAM  |  {os.path.basename(img_path)}', fontsize=11)
    plt.tight_layout()

    stem     = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f'{stem}_gradcam_per_class.jpg')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.show()


if sample_images:
    run_gradcam_per_class(sample_images[0])

# %%
# ── 10. Compare backbone/neck layers ───────────────────────────────────────
# Shows Grad-CAM at different network depths for the same image.
# Neck C3k2 layers give tighter disease-spot heatmaps than backbone SPPF.
COMPARE_LAYERS = [9, 13, 16, 19, 22]   # SPPF → neck P4 → P3 → P4' → P5'


def run_layer_comparison(image_path: str, alpha: float = 0.45):
    img_path     = str(image_path)
    tensor, meta = load_and_preprocess(img_path)
    canvas_bgr   = meta['canvas_bgr']

    nn_model.to(INFER_DEVICE).eval()
    with torch.no_grad():
        results = model.predict(
            source=canvas_bgr, imgsz=IMG_SIZE,
            conf=CONF_THRESH, iou=IOU_THRESH,
            device=INFER_DEVICE, verbose=False,
        )
    result = results[0]

    n = len(COMPARE_LAYERS)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    axes[0].imshow(draw_obb_on_image(canvas_bgr.copy(), result)[:, :, ::-1])
    axes[0].set_title('Original + OBB', fontsize=10)
    axes[0].axis('off')

    layer_res = {9: '32²', 13: '64²', 16: '128²', 19: '64²', 22: '32²'}

    for ax, idx in zip(axes[1:], COMPARE_LAYERS):
        layer = nn_model.model[idx]
        gcam  = YOLOGradCAM(nn_model, layer, device=CAM_DEVICE)
        cam   = gcam.compute(tensor, target_class=TARGET_CLASS)
        gcam.remove_hooks()
        nn_model.to(INFER_DEVICE).eval()

        hmap    = cam_to_heatmap(cam, (IMG_SIZE, IMG_SIZE))
        overlay = draw_obb_on_image(overlay_cam(canvas_bgr, hmap, alpha), result)
        ax.imshow(overlay[:, :, ::-1])
        res_label = layer_res.get(idx, '?')
        ax.set_title(f'Layer {idx}  {type(layer).__name__}\n{res_label}', fontsize=9)
        ax.axis('off')

    fig.suptitle(f'Layer comparison  |  {os.path.basename(img_path)}', fontsize=11)
    plt.tight_layout()

    stem     = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f'{stem}_gradcam_layers.jpg')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.show()


if sample_images:
    run_layer_comparison(sample_images[0])

# %%
# ── 11. Cleanup ────────────────────────────────────────────────────────────
grad_cam.remove_hooks()
# Restore model to inference device
nn_model.to(INFER_DEVICE).eval()
print('Done. Hooks removed.')
