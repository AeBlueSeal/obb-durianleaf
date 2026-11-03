import os
from pathlib import Path
from collections import Counter
import yaml

DATA_YAML  = 'data/dataset.yaml'
OUTPUT_DIR = 'analysis/dataset/outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

class_names = cfg['names']
labels_dir  = Path(cfg['path']) / 'labels/train'
counter     = Counter()

for label_file in labels_dir.glob('*.txt'):
    for line in label_file.read_text().strip().splitlines():
        class_id = int(line.split()[0])
        counter[class_id] += 1

print(f'\n{"Class":<20} {"Count":>8} {"Share":>8}')
print('-' * 40)
total = sum(counter.values())
for class_id, count in sorted(counter.items()):
    name  = class_names[class_id]
    share = count / total * 100
    bar   = '█' * int(share / 2)
    print(f'{name:<20} {count:>8} {share:>7.1f}%  {bar}')

print(f'\nTotal annotations: {total}')
