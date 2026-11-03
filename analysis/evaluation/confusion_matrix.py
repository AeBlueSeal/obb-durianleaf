from ultralytics import YOLO
import os

WEIGHTS    = 'models/yolo11m-obb.pt'
DATA_YAML  = 'data/dataset.yaml'
OUTPUT_DIR = 'analysis/evaluation/outputs/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(WEIGHTS)
model.val(data=DATA_YAML, plots=True, save_dir=OUTPUT_DIR)

print(f'Confusion matrix saved to {OUTPUT_DIR}')
