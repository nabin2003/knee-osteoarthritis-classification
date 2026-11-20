import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


CLASS_NAMES = ["KL 0", "KL 1", "KL 2", "KL 3", "KL 4"]


def ensure_dir(folder):
    """Create an output folder if it does not already exist."""
    os.makedirs(folder, exist_ok=True)
    return folder


def save_confusion_matrix(y_true, y_pred, filename, title="Confusion matrix"):
    """Save a five-class confusion matrix as a PNG image."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def save_predictions(names, y_true, y_pred, probabilities, filename):
    """Save one row per image, including predicted class probabilities."""
    data = {
        "Name": list(names),
        "Actual_KL": list(y_true),
        "Predicted_KL": list(y_pred),
    }
    probabilities = np.asarray(probabilities)
    for index in range(5):
        data[f"Probability_KL_{index}"] = probabilities[:, index]
    pd.DataFrame(data).to_csv(filename, index=False)


def save_fold_summary(rows, filename):
    """Save evaluation results for all folds in a CSV file."""
    pd.DataFrame(rows).to_csv(filename, index=False)
