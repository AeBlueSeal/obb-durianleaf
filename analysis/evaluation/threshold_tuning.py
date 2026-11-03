import numpy as np
from ultralytics import YOLO

WEIGHTS   = 'models/yolo11m-obb.pt'
DATA_YAML = 'data/dataset.yaml'

model   = YOLO(WEIGHTS)
metrics = model.val(data=DATA_YAML)

p, r, f1, thresholds = (
    metrics.curves[0],
    metrics.curves[1],
    metrics.curves[2],
    metrics.curves[3],
)

best_idx    = np.argmax(f1)
best_thresh = thresholds[best_idx]
best_f1     = f1[best_idx]

print(f'Best threshold : {best_thresh:.3f}')
print(f'Best F1        : {best_f1:.4f}')
print(f'Precision      : {p[best_idx]:.4f}')
print(f'Recall         : {r[best_idx]:.4f}')
