import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score


class ThresholdAnalyzer:
    """
    Sweeps multiple probability thresholds to visualize precision/recall/F1 trade-offs
    and optionally saves the resulting plot.
    """

    def __init__(self, output_dir: str = "artifacts", step: float = 0.05):
        self.output_dir = output_dir
        self.step = step
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, y_true, y_proba, plot: bool = True) -> float:
        thresholds = np.arange(0.05, 0.95, self.step)
        precisions, recalls, f1s = [], [], []

        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred))

        # Choose threshold with best F1
        best_idx = int(np.argmax(f1s))
        best_thr = thresholds[best_idx]

        if plot:
            plt.figure(figsize=(7, 5))
            sns.lineplot(x=thresholds, y=precisions, label="Precision")
            sns.lineplot(x=thresholds, y=recalls, label="Recall")
            sns.lineplot(x=thresholds, y=f1s, label="F1-score")
            plt.axvline(best_thr, color="red", linestyle="--", label=f"Best F1 thr={best_thr:.2f}")
            plt.xlabel("Threshold"); plt.ylabel("Score")
            plt.title("Threshold Sweep — Precision/Recall/F1")
            plt.legend()
            path = os.path.join(self.output_dir, "threshold_sweep.png")
            plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

        print(f"📊 Best F1 threshold = {best_thr:.3f} (F1={f1s[best_idx]:.3f})")
        return float(best_thr)
