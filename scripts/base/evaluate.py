from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True)
parser.add_argument('--data',    default='data/dataset.yaml')
args = parser.parse_args()

model   = YOLO(args.weights)
metrics = model.val(data=args.data)

print(f"mAP@50    : {metrics.box.map50:.4f}")
print(f"mAP@50-95 : {metrics.box.map:.4f}")
