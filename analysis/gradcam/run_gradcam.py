import cv2
import numpy as np
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

WEIGHTS            = 'models/yolo11m-obb.pt'
SOURCE_DIR         = 'test_images/'
OUTPUT_DIR         = 'analysis/gradcam/outputs/'
TARGET_LAYER_INDEX = -2

os.makedirs(OUTPUT_DIR, exist_ok=True)

model       = YOLO(WEIGHTS)
torch_model = model.model
target_layer = [torch_model.model[TARGET_LAYER_INDEX]]

cam = GradCAM(model=torch_model, target_layers=target_layer)

for img_path in Path(SOURCE_DIR).glob('*.[jJpP][pPnNgG]*'):
    img_bgr  = cv2.imread(str(img_path))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = np.float32(img_rgb) / 255.0

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

    grayscale_cam = cam(input_tensor=tensor)[0]
    visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

    out_path = os.path.join(OUTPUT_DIR, f'gradcam_{img_path.name}')
    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f'Saved → {out_path}')

print('Done')
