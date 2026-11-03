from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True)
parser.add_argument('--source',  default='test_images/')
parser.add_argument('--conf',    type=float, default=0.25)
parser.add_argument('--save',    action='store_true')
args = parser.parse_args()

model = YOLO(args.weights)
model.predict(source=args.source, conf=args.conf, save=args.save)
