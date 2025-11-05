import os
import joblib
import warnings
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class ModelTrainer:
    """Trains a LightGBMClassifier with imbalance handling and cross-validation."""

    def __init__(self, params: dict, model_path: str, n_splits: int = 5):
        self.params = params
        self.model_path = model_path
        self.n_splits = n_splits
        self.models: list[LGBMClassifier] = []

    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """Compute class imbalance weight."""
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        return max(1.0, n_neg / max(n_pos, 1))

    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform stratified k-fold cross-validation, return mean ROC-AUC."""
        warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        aucs = []

        scale_weight = self._compute_scale_pos_weight(y)
        params = dict(self.params)
        params["scale_pos_weight"] = scale_weight
        params["verbosity"] = -1

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            aucs.append(auc)
            self.models.append(model)

        mean_auc = float(np.mean(aucs))
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.models[0], self.model_path)  # save first model for inference

        return mean_auc
