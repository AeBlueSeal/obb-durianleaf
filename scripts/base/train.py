from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',   default='yolo11m-obb.pt')
# Confound-fix default: data/dataset_resplit.yaml (grouped/dedup'd/stratified
# split from analysis/dataset/phase1_build_groups.py + phase2_resplit.py),
# not the original data/dataset.yaml — see results/source_confound_scoping.md.
parser.add_argument('--data',    default='data/dataset_resplit.yaml')
parser.add_argument('--epochs',  type=int, default=100)
parser.add_argument('--imgsz',   type=int, default=640)
parser.add_argument('--batch',   type=int, default=16)
# Epochs were never tuned before; early-stop instead of guessing a fixed count.
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--project', default='runs')
parser.add_argument('--name',    default='train')
args = parser.parse_args()

model = YOLO(args.model)
model.train(
    data     = args.data,
    epochs   = args.epochs,
    imgsz    = args.imgsz,
    batch    = args.batch,
    patience = args.patience,
    project  = args.project,
    name     = args.name,
)

# Confound-fix checklist (see implementation_plan.md):
#   - deploy runs/<name>/weights/best.pt, not last.pt
#   - re-check inference conf around ~0.53 after this run — the prior F1
#     peak was measured on the old split/model and may shift
#   - re-run analysis/dataset/source_confound_{A,B1,D}.py and
#     analysis/evaluation/source_confound_{C1,C2}.py against this weights
#     file + data/dataset_resplit.yaml before trusting the new numbers
