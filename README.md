# Rotation-Aware Detection of Durian Leaf Diseases Using YOLO-OBB Under Natural Orchard Conditions

[![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.17605%2FOSF.IO%2F2578K-darkgreen.svg)](https://doi.org/10.17605/OSF.IO/2578K)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11%20%2F%20YOLO26-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Google Play](https://img.shields.io/badge/Google%20Play-YOEDO%20App-34A853?logo=google-play&logoColor=white)](https://play.google.com/store/apps/details?id=com.cateatmars_dev.yoedodurian)

Code accompanying the manuscript *Rotation-Aware Detection of Durian Leaf Diseases Using YOLO-OBB Under Natural Orchard Conditions*. The repository contains the training, cross-validation and evaluation pipelines used to produce the results reported in the paper.

---

## Overview

Durian is Thailand's most valuable fruit export, and leaf disease is currently assessed by walk-through visual inspection of individual trees. Automatic detection of diseased leaves in the field is difficult. Durian leaves are long and narrow, lie at every angle and overlap against cluttered backgrounds, and axis-aligned boxes around such leaves enclose large areas of background and neighbouring leaves.

This study evaluates rotation-aware detection with oriented bounding boxes (OBB) for this task:

- A dataset of **1,811 orchard images** with **6,903 leaf instances** across **seven classes** was annotated with oriented bounding boxes and released publicly on OSF.
- Two oriented detectors, **YOLO11m-OBB** and **YOLO26m-OBB**, were trained under **nine-fold cross-validation** and evaluated on a **held-out test set of 188 images**.
- **RF-DETR-M OBB**, an oriented detection transformer, serves as a baseline outside the YOLO family.
- An ablation of **Exponentially Weighted Instance-Aware Repeat Factor Sampling (E-IRFS)** measures the effect of class-imbalance mitigation.

---

## Main Results

### Detection performance (held-out test set, mean ± SD over nine folds)

Test set: 188 images, 654 instances.

| Model | E-IRFS | mAP@0.5:0.95 | mAP@0.5 | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RF-DETR-M OBB | — | 0.329 ± 0.009 | 0.581 ± 0.011 | 0.640 ± 0.026 | 0.568 ± 0.017 | 0.602 ± 0.011 |
| YOLO11m-OBB | No | 0.505 ± 0.013 | 0.671 ± 0.018 | 0.678 ± 0.028 | 0.651 ± 0.025 | 0.664 ± 0.016 |
| YOLO11m-OBB | Yes | 0.502 ± 0.014 | 0.668 ± 0.014 | 0.675 ± 0.030 | 0.648 ± 0.029 | 0.661 ± 0.016 |
| YOLO26m-OBB | No | **0.525 ± 0.010** | **0.677 ± 0.020** | 0.693 ± 0.026 | **0.647 ± 0.026** | **0.668 ± 0.013** |
| YOLO26m-OBB | Yes | 0.522 ± 0.015 | 0.675 ± 0.016 | **0.698 ± 0.032** | 0.640 ± 0.017 | 0.667 ± 0.010 |

- YOLO26m-OBB gives the highest mean mAP@0.5:0.95 (0.525 against 0.505 for YOLO11m-OBB), and the margin is wider than the spread across folds.
- At the threshold of 0.5 the two YOLO models finish within 0.006 of each other. The gain of the newer architecture appears in localisation quality rather than in finding more diseased leaves.
- RF-DETR-M OBB was trained on the same folds and evaluated on the same test set at an input size of 320 × 320, the setting selected by its own hyperparameter search.

### Effect of E-IRFS

E-IRFS leaves overall accuracy unchanged within the fold-to-fold spread. Averaged over the nine folds, the rarest class benefits. **Rhizoctonia** gains **0.031** AP@0.5:0.95 with YOLO26m-OBB and **0.014** with YOLO11m-OBB, while Phytophthora and Algal spot lose ground. E-IRFS moves attention towards the rarest class at a cost to the more common classes and does not raise overall accuracy on this dataset.

### Per-class performance of the best fold

**YOLO26m-OBB, fold 3** (highest test mAP@0.5:0.95)

| Class | n | AP@0.5 (no E-IRFS) | AP@0.5:0.95 (no E-IRFS) | AP@0.5 (E-IRFS) | AP@0.5:0.95 (E-IRFS) | Δ AP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Algal spot | 212 | 0.691 | 0.529 | 0.707 | 0.550 | +0.021 |
| Other | 180 | 0.710 | 0.558 | 0.727 | 0.564 | +0.006 |
| Phomopsis | 70 | 0.510 | 0.431 | 0.483 | 0.389 | −0.042 |
| Anthracnose | 60 | 0.735 | 0.573 | 0.760 | 0.583 | +0.010 |
| Phytophthora | 53 | 0.672 | 0.557 | 0.705 | 0.591 | +0.034 |
| Healthy | 47 | 0.665 | 0.589 | 0.733 | 0.640 | +0.051 |
| Rhizoctonia | 32 | 0.657 | 0.484 | 0.758 | 0.577 | +0.093 |
| **All classes** | **654** | **0.663** | **0.532** | **0.696** | **0.556** | **+0.024** |

**YOLO11m-OBB, fold 2** (highest test mAP@0.5:0.95)

| Class | n | AP@0.5 (no E-IRFS) | AP@0.5:0.95 (no E-IRFS) | AP@0.5 (E-IRFS) | AP@0.5:0.95 (E-IRFS) | Δ AP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Algal spot | 212 | 0.649 | 0.492 | 0.692 | 0.507 | +0.015 |
| Other | 180 | 0.693 | 0.525 | 0.756 | 0.572 | +0.047 |
| Phomopsis | 70 | 0.498 | 0.406 | 0.568 | 0.444 | +0.038 |
| Anthracnose | 60 | 0.711 | 0.551 | 0.753 | 0.574 | +0.023 |
| Phytophthora | 53 | 0.639 | 0.521 | 0.564 | 0.467 | −0.054 |
| Healthy | 47 | 0.625 | 0.527 | 0.665 | 0.550 | +0.023 |
| Rhizoctonia | 32 | 0.751 | 0.485 | 0.821 | 0.529 | +0.044 |
| **All classes** | **654** | **0.652** | **0.501** | **0.688** | **0.521** | **+0.020** |

The best-fold tables are single-fold results. The nine-fold means above are the primary results.

### Computational efficiency

Measured on an NVIDIA RTX PRO 6000 GPU (96 GB VRAM) and averaged over the nine test evaluations.

| Model | Parameters (fused) | GFLOPs | Preprocess | Inference | Postprocess |
| :--- | :---: | :---: | :---: | :---: | :---: |
| YOLO11m-OBB | 20,883,880 | 71.5 | 1.4 ms | 3.8 ms | 1.8 ms |
| YOLO26m-OBB | 21,203,300 | 71.8 | 1.3 ms | 4.0 ms | 0.2 ms |

The training-time parameter counts are 20,907,240 and 23,481,352, because YOLO26 adds a one-to-many head that is dropped at inference. The short postprocessing time of YOLO26m-OBB follows from its NMS-free head and makes its total time per image shorter than that of YOLO11m-OBB.

---

## Dataset

The dataset contains **1,811 orchard images** with **6,903 oriented leaf instances**, annotated in Label Studio v1.21.0.

| Class | Train | Val | Test | Total | Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Algal spot | 1,700 | 203 | 212 | 2,115 | 30.6 |
| Other | 1,225 | 145 | 180 | 1,550 | 22.5 |
| Phomopsis | 899 | 109 | 70 | 1,078 | 15.6 |
| Anthracnose | 627 | 95 | 60 | 782 | 11.3 |
| Phytophthora | 496 | 63 | 53 | 612 | 8.9 |
| Healthy | 348 | 38 | 47 | 433 | 6.3 |
| Rhizoctonia | 274 | 27 | 32 | 333 | 4.8 |
| **Total** | **5,569** | **680** | **654** | **6,903** | **100.0** |

The **Other** class covers leaf damage that is not one of the six named classes, including damage from *Allocaridara* attacks and mechanically incomplete or torn leaves.

For cross-validation, the training and validation partitions are pooled and split into nine folds. The 188-image test set is held out and is never used for hyperparameter search or fold training.

**Download:** the dataset and the trained models are available on OSF at [https://doi.org/10.17605/OSF.IO/2578K](https://doi.org/10.17605/OSF.IO/2578K). Large data files and model weights are not stored in this repository.

---

## Training Configuration

Hyperparameters were selected by an Optuna search and used unchanged in every cross-validation fold. The search was repeated whenever the architecture changed. RF-DETR, which has a different set of hyperparameters, received a separate search.

Values selected for YOLO11m-OBB:

| Component | Selected value |
| :--- | :--- |
| Optimizer | SGD |
| `lr0` | 0.007688 |
| `lrf` (final LR factor) | 0.32269 |
| `momentum` | 0.90041 |
| `weight_decay` | 1.936 × 10⁻⁵ |
| `warmup_epochs` | 1.6018 |
| `warmup_momentum` | 0.75448 |
| `warmup_bias_lr` | 0.09973 |
| λ_box | 6.4127 |
| λ_cls | 1.4604 |
| λ_dfl | 0.96436 |
| `imgsz` | 1024 × 1024 |
| `batch` | 16 |

### Environment

| Component | Specification |
| :--- | :--- |
| Programming language | Python 3.10 |
| Deep learning framework | PyTorch 2.x |
| Annotation tool | Label Studio v1.21.0 |
| Detection models | YOLO11m-OBB and YOLO26m-OBB (Ultralytics) |
| Hardware | NVIDIA RTX PRO 6000 GPU (96 GB VRAM) |

---

## Repository Structure

```text
obb-durianleaf/
├── notebooks/                    # HPO and K-fold training pipelines (jupytext .py)
│   ├── YOLO11mOBB_E-IRFS_optuna_datasetv3.py   # Optuna search, YOLO11m-OBB
│   ├── YOLO26mOBB_E-IRFS_optuna_datasetv3.py   # Optuna search, YOLO26m-OBB
│   ├── YOLO11mOBB_E-IRFS_kfold_datasetv3.py    # 9-fold CV, with E-IRFS
│   ├── YOLO11mOBB_NoEIRFS_kfold_datasetv3.py   # 9-fold CV, without E-IRFS
│   ├── YOLO26mOBB_E-IRFS_kfold_datasetv3.py
│   ├── YOLO26mOBB_NoEIRFS_kfold_datasetv3.py
│   ├── RFDETR_OBB_E-IRFS_optuna_datasetv3.py   # RF-DETR baseline search
│   └── RFDETR_OBB_E-IRFS_kfold_datasetv3.py    # RF-DETR baseline 9-fold CV
├── src/
│   ├── obb_common.py             # K-fold partitioner, E-IRFS sampler, format converters
│   ├── obb_eval.py               # Framework-agnostic polygon-IoU OBB evaluator
│   └── obb_logging.py
├── analysis/
│   ├── comparison/               # Cross-arm comparability checks
│   ├── dataset/                  # Dataset grouping, re-splitting and label checks
│   ├── evaluation/               # Confusion matrices and threshold tuning
│   └── gradcam/                  # Grad-CAM visualisation for YOLO11-OBB
├── scripts/
│   ├── base/                     # train.py, evaluate.py, detect.py
│   └── run.py                    # Experiment launcher from a YAML config
├── results/                      # Experiment notes and comparison declarations
└── requirements.txt
```

The notebooks are stored as jupytext `.py` files. Regenerate a notebook with `jupytext --sync notebooks/<name>.py`.

The `notebooks/` folder also contains exploratory pipelines for other oriented detectors (Oriented-DETR, Strip R-CNN, S2A-Net) and SAHI tiling. These experiments are not reported in the paper.

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/AeBlueSeal/obb-durianleaf.git
cd obb-durianleaf

python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Data

Download the dataset from [OSF](https://doi.org/10.17605/OSF.IO/2578K) and extract it into `data/`. The dataset YAML files expected by the scripts are placed in the same folder.

### 3. Training and evaluation

Cross-validation runs (paper results) are produced by the notebooks in `notebooks/`. Run the Optuna notebook of a model first, then the matching K-fold notebook, which loads the selected hyperparameters.

Single runs and inference with the utility scripts:

```bash
# Train a single model
python scripts/base/train.py --model yolo26m-obb.pt --data data/dataset.yaml --imgsz 1024 --batch 16

# Evaluate a checkpoint
python scripts/base/evaluate.py --weights path/to/best.pt --data data/dataset.yaml

# Run inference on a folder of images
python scripts/base/detect.py --weights path/to/best.pt --source test_images/ --save
```

---

## Mobile Application: YOEDO (Durian Leaf Scanner)

The trained models have been ported to an Android application, **YOEDO: Durian Leaf Scanner**, for use by growers in the orchard. The app runs detection on the device and shows oriented boxes with confidence scores for the seven classes.

<p align="center">
  <a href="https://play.google.com/store/apps/details?id=com.cateatmars_dev.yoedodurian">
    <img src="https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png" alt="Get it on Google Play" height="80">
  </a>
</p>

The inference times reported above were measured on a server GPU and do not describe performance on mobile devices.

---

## Citation

The manuscript is under review. Citation details will be added upon publication. Until then, please cite the dataset:

```text
Dataset: Durian leaf disease oriented bounding box dataset.
Open Science Framework. https://doi.org/10.17605/OSF.IO/2578K
```

---

## Funding

This work was supported by Burapha University under grant number YR2568/13.

## License

The dataset is released under CC BY 4.0 on OSF.
