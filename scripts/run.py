import argparse, yaml, os
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', required=True)
args = parser.parse_args()

CONFIG_PATH = f'scripts/experiments/{args.experiment}/config.yaml'

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

cfg.pop('status', None)
cfg.pop('notes', None)

model = YOLO(cfg.pop('model'))
model.train(
    **cfg,
    project  = f'runs/{args.experiment}',
    name     = 'train',
    exist_ok = True,
)
