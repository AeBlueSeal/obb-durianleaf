from pathlib import Path
import yaml

DATA_YAML = 'data/dataset.yaml'

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

base = Path(cfg['path'])

for split in ['train', 'val', 'test']:
    images_dir = base / 'images' / split
    labels_dir = base / 'labels' / split
    if not images_dir.exists():
        continue

    missing, empty = [], []
    for img in images_dir.glob('*.[jJpP][pPnNgG]*'):
        label = labels_dir / (img.stem + '.txt')
        if not label.exists():
            missing.append(img.name)
        elif label.stat().st_size == 0:
            empty.append(img.name)

    print(f'\n[{split}] Missing labels : {len(missing)}')
    print(f'[{split}] Empty labels   : {len(empty)}')
    for f in missing[:5]:
        print(f'  ✗ {f}')
