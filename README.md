# DurianTemp

DurianTemp is a computer-vision research project for developing and evaluating
object-detection models. The repository brings together dataset utilities,
training and inference scripts, experiment configurations, analysis tools, and
notebooks used throughout the machine-learning workflow.

The project is organized to support repeatable experiments: shared code lives
in reusable modules, experiment-specific settings are kept separate, and
results are collected for later comparison and analysis.

## Repository structure

```text
├── data/                 # Dataset configurations and metadata
├── scripts/              # Command-line training and inference utilities
│   ├── base/             # Shared scripts
│   ├── experiments/      # Experiment configurations and notes
│   └── run.py            # Experiment entry point
├── analysis/             # Dataset checks, evaluation, and visual analysis
├── notebooks/            # Interactive training and research notebooks
├── models/               # Local model weights (not tracked by Git)
├── test_images/          # Sample images for local testing
└── results/              # Experiment summaries and reports
```

