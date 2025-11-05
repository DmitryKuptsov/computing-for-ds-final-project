import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_curve,
    confusion_matrix,
)


class Evaluator:
    """Evaluates binary classifier probabilities with optimized threshold and confusion matrix."""

    def __init__(self, metrics_path: str, figures_dir: str = "artifacts"):
        self.metrics_path = metrics_path
        self.figures_dir = figures_dir

    def _find_best_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find threshold that maximizes F1."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.nanargmax(f1_scores)
        return float(thresholds[best_idx])

    def _plot_confusion_matrix(self, y_true, y_pred, normalize: bool = True) -> None:
        """Plot and save confusion matrix (overwrites or timestamps file)."""
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

        os.makedirs(self.figures_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.figures_dir, f"confusion_matrix_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

        print(f"✅ Confusion matrix saved: {path}")

    def evaluate(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Compute key metrics using optimal threshold, save metrics and confusion matrix."""
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        best_threshold = self._find_best_threshold(y_true, y_proba)
        y_pred = (y_proba >= best_threshold).astype(int)

        metrics = {
            "ROC_AUC": float(roc_auc_score(y_true, y_proba)),
            "PR_AUC": float(average_precision_score(y_true, y_proba)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "F1": float(f1_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred)),
            "Recall": float(recall_score(y_true, y_pred)),
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Best_Threshold": best_threshold,
        }

        # Save metrics JSON
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # ✅ Plot confusion matrix
        self._plot_confusion_matrix(y_true, y_pred, normalize=True)

        return metrics
