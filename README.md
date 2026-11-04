# Rotation-Aware Detection of Durian Leaf Diseases Using YOLO-OBB Under Natural Orchard Conditions

[![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.17605%2FOSF.IO%2F2578K-darkgreen.svg)](https://doi.org/10.17605/OSF.IO/2578K)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11%20%2F%20YOLO26-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)


---

## Overview

Durian (*Durio zibethinus*) is Thailand's premier commercial export fruit, yet orchard-level disease monitoring remains hindered by manual walk-through scouting. Automated detection in natural orchard environments presents severe computer vision challenges:
- **Leaf Geometry:** Durian leaves are elongated with extreme aspect ratios ($3:1$ to $6:1$) and lie at arbitrary orientations along branches. Standard horizontal/axis-aligned bounding boxes (HBB) enclose substantial background noise and trigger false suppression during NMS.
- **Complex In-Field Clutter:** Overlapping foliage, variable natural sunlight/shadows, and diffuse lesion boundaries make single-leaf classification models fail when deployed in field conditions.
- **Pathogen Coverage:** Crucial economic pathogens such as *Phytophthora palmivora* have been omitted from prior recognition benchmarks.

This repository provides an **Oriented Bounding Box (OBB)** framework, featuring a newly annotated multi-class durian leaf orchard dataset, an automated Optuna hyperparameter optimization suite, rigorous **9-fold cross-validation**, and standardized cross-framework polygon IoU evaluation.

---

## Key Highlights & Contributions

1. **First OBB Durian Leaf Disease Dataset:**
   - **1,811** field-collected orchard images with **6,903** oriented bounding box leaf instances across **7 categories** (including *Phytophthora*, *Rhizoctonia*, *Phomopsis*, *Anthracnose*, and *Algal spot*).
   - Released publicly under CC-BY 4.0 via **[OSF (DOI: 10.17605/OSF.IO/2578K)](https://doi.org/10.17605/OSF.IO/2578K)**.
2. **Comprehensive Architecture Benchmarking:**
   - Benchmarks next-generation anchor-free single-stage detectors (**YOLO11m-OBB** vs. **YOLO26m-OBB** with NMS-free dual head).
   - Cross-compared against oriented detection transformers (**RF-DETR Medium OBB**, **Oriented-DETR R50**) and two-stage detectors (**Strip R-CNN**, **S2A-Net**).
3. **Rigorous Experimental Integrity:**
   - Strict separation of a **188-image held-out test set** (never touched during HPO or fold training).
   - Cross-arm metric comparability enforced via a dedicated polygon IoU evaluator (`src/obb_eval.py`) and static invariant guard (`analysis/comparison/check_kfold_comparability.py`).
4. **Long-Tail Imbalance Analysis:**
   - Full ablation study of **Exponentially Weighted Instance-Aware Repeat Factor Sampling (E-IRFS)** across all folds.
5. **Real-Time Deployment Profiling:**
   - Sub-5ms inference latency per image on server GPU hardware, demonstrating edge readiness.

---

## Benchmark Results

### 1. Overall Performance (9-Fold Cross-Validation on Held-Out Test Set)

Evaluated on the independent held-out test set ($N = 188$ images, $654$ instances). Results report **Mean $\pm$ Standard Deviation across 9 folds**:

| Model Architecture | Framework / Type | Input Size | Fused Params | Latency (Inference) | mAP@0.5:0.95 | mAP@0.5 | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26m-OBB** (No E-IRFS) | YOLO-OBB (NMS-free dual head) | $1024 \times 1024$ | 21.20M | 4.0 ms | **0.525 ± 0.010** | **0.677 ± 0.020** | **0.668 ± 0.013** |
| **YOLO26m-OBB** (With E-IRFS) | YOLO-OBB (NMS-free dual head) | $1024 \times 1024$ | 21.20M | 4.0 ms | 0.522 ± 0.015 | 0.675 ± 0.016 | 0.667 ± 0.010 |
| **YOLO11m-OBB** (No E-IRFS) | YOLO-OBB (Anchor-free) | $1024 \times 1024$ | 20.88M | 3.8 ms | 0.505 ± 0.013 | 0.671 ± 0.018 | 0.664 ± 0.016 |
| **YOLO11m-OBB** (With E-IRFS) | YOLO-OBB (Anchor-free) | $1024 \times 1024$ | 20.88M | 3.8 ms | 0.502 ± 0.014 | 0.668 ± 0.013 | 0.661 ± 0.016 |
| **RF-DETR-M OBB** | Detection Transformer | $320 \times 320$ | — | — | 0.329 ± 0.009 | 0.581 ± 0.011 | 0.602 ± 0.011 |

> **Key Finding:** YOLO26m-OBB's NMS-free direct regression head delivers superior bounding box tightness (+0.020 mAP@50:95) over YOLO11m-OBB, while postprocessing overhead is eliminated ($0.2\text{ ms}$ vs $1.8\text{ ms}$).

### 2. Isolated Effect of E-IRFS Sampling

Averaged across 9 folds, E-IRFS selectively channels representation capacity to under-represented tail classes without hurting overall stability:
* **Rhizoctonia** (rarest class, $n=32$ test instances): **+0.031 AP@50:95** gain with YOLO26m-OBB.
* **Anthracnose** ($n=60$): **+0.015 AP@50:95** gain.

---

## Dataset & Annotation Protocol

The dataset consists of **1,811 orchard images** containing **6,903 oriented leaf instances**, annotated using Label Studio with a multi-stage consensus review.

| Class Name | Train Instances | Val Instances | Test Instances | Total Instances | Class Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Algal spot** | 1,700 | 203 | 212 | 2,115 | 30.6% |
| **Other** (Torn/Allocaridara) | 1,225 | 145 | 180 | 1,550 | 22.5% |
| **Phomopsis** | 899 | 109 | 70 | 1,078 | 15.6% |
| **Anthracnose** | 627 | 95 | 60 | 782 | 11.3% |
| **Phytophthora** | 496 | 63 | 53 | 612 | 8.9% |
| **Healthy** | 348 | 38 | 47 | 433 | 6.3% |
| **Rhizoctonia** | 274 | 27 | 32 | 333 | 4.8% |
| **Total** | **5,569** | **680** | **654** | **6,903** | **100.0%** |

* Download the dataset archive from OSF: [https://doi.org/10.17605/OSF.IO/2578K](https://doi.org/10.17605/OSF.IO/2578K)

---

## Repository Structure

```text
obb-durianleaf/
├── analysis/                     # Rigorous diagnostic and validation tools
│   ├── comparison/               # Invariant guard & baseline rescoring
│   │   ├── check_kfold_comparability.py  # Static audit ensuring apple-to-apple comparison
│   │   └── rescore_baseline_yolo.py
│   ├── dataset/                  # De-biasing, deduplication, and confound mitigation
│   │   ├── phase1_build_groups.py
│   │   ├── phase2_resplit.py
│   │   └── ...
│   ├── evaluation/               # Threshold tuning & confusion matrix generators
│   └── gradcam/                  # OBB visual explainability scripts
├── data/                         # Dataset configurations (.yaml) and splits
├── notebooks/                    # Self-contained HPO & K-fold training pipelines
│   ├── YOLO11mOBB_E-IRFS_kfold_datasetv3.py
│   ├── YOLO26mOBB_E-IRFS_kfold_datasetv3.py
│   ├── RFDETR_OBB_E-IRFS_kfold_datasetv3.py
│   ├── StripRCNN_E-IRFS_kfold_datasetv3.py
│   └── S2ANet_E-IRFS_kfold_datasetv3.py
├── results/                      # Audit trails, scoping notes, and control variable declarations
│   ├── comparison_control_variables.md
│   └── obb_model_comparison.md
├── scripts/                      # Modular execution scripts
│   ├── base/
│   │   ├── train.py              # Single-run training utility
│   │   ├── evaluate.py           # Checkpoint evaluation utility
│   │   └── detect.py             # Inference on test imagery
│   └── run.py                    # Experiment launcher via YAML config
├── src/                          # Core shared modules
│   ├── obb_common.py             # K-fold partitioner, E-IRFS sampler, format converters
│   ├── obb_eval.py               # Framework-agnostic Shapely polygon IoU evaluator
│   └── obb_logging.py
└── requirements.txt              # Environment dependencies
```

---

## Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/AeBlueSeal/obb-durianleaf.git
cd obb-durianleaf

# Create and activate a clean Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
