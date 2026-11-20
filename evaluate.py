import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader

from args import get_args
from dataset import knee_xray_dataset
from model import MyModel
from utils import (
    CLASS_NAMES,
    ensure_dir,
    save_confusion_matrix,
    save_fold_summary,
    save_predictions,
)


def evaluate_fold(fold, args, device):
    """Load one fold's checkpoint and evaluate its validation images."""
    csv_file = os.path.join(args.csv_dir, f"fold_{fold}_val.csv")
    model_file = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")

    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Validation CSV not found: {csv_file}")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_file}. Run main.py first."
        )

    dataframe = pd.read_csv(csv_file)
    if dataframe.empty:
        raise ValueError(f"Validation CSV is empty: {csv_file}")
    loader = DataLoader(
        knee_xray_dataset(dataframe),
        batch_size=args.bs,
        shuffle=False,
    )

    model = MyModel(backbone=args.backbone).to(device)
    # Only load checkpoints you trust, such as ones you trained yourself.
    state = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    names, actual, predicted, probabilities = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["xray"].to(device)
            scores = model(inputs)
            probs = torch.softmax(scores, dim=1).cpu().numpy()

            names.extend(batch["name"])
            actual.extend(batch["target"].cpu().numpy().tolist())
            predicted.extend(np.argmax(probs, axis=1).tolist())
            probabilities.extend(probs.tolist())

    results_dir = ensure_dir(os.path.join(args.out_dir, "evaluation"))
    save_predictions(
        names, actual, predicted, probabilities,
        os.path.join(results_dir, f"predictions_fold_{fold}.csv"),
    )
    save_confusion_matrix(
        actual, predicted,
        os.path.join(results_dir, f"confusion_matrix_fold_{fold}.png"),
        title=f"Knee OA - Fold {fold}",
    )

    report = classification_report(
        actual, predicted, labels=list(range(5)),
        target_names=CLASS_NAMES, zero_division=0,
    )
    with open(
        os.path.join(results_dir, f"classification_report_fold_{fold}.txt"),
        "w", encoding="utf-8",
    ) as file:
        file.write(report)

    metrics = {
        "Fold": fold,
        "Images": len(actual),
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced_Accuracy": balanced_accuracy_score(actual, predicted),
        "Macro_F1": f1_score(actual, predicted, labels=list(range(5)),
                             average="macro", zero_division=0),
    }
    print(f"\nFold {fold} ({len(actual)} validation images)")
    print(report)
    print("Balanced accuracy:", f"{metrics['Balanced_Accuracy']:.4f}")
    return metrics


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on:", device)

    results = [evaluate_fold(fold, args, device) for fold in range(5)]
    results_dir = ensure_dir(os.path.join(args.out_dir, "evaluation"))
    save_fold_summary(results, os.path.join(results_dir, "fold_summary.csv"))

    print("\nFive-fold validation summary")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    for metric in ["Accuracy", "Balanced_Accuracy", "Macro_F1"]:
        print(f"{metric}: {summary[metric].mean():.4f} "
              f"+/- {summary[metric].std(ddof=1):.4f}")
    print(f"\nReports saved in: {results_dir}")


if __name__ == "__main__":
    main()
