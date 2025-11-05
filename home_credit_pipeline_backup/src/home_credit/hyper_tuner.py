import optuna
import numpy as np
from typing import Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from .utils.logger import get_logger


class HyperTuner:
    """Tunes LightGBM hyperparameters using Optuna for ROC-AUC optimization."""

    def __init__(self, n_trials: int = 30, n_splits: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = get_logger(self.__class__.__name__)
        self.best_params_: dict[str, Any] | None = None

    def _objective(self, trial: optuna.Trial, X, y):
        """Objective function for Optuna."""
        # Define the search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": self.random_state,
            "verbosity": -1,
        }

        # Handle class imbalance automatically
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        # Stratified k-fold CV
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        aucs = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))

        mean_auc = float(np.mean(aucs))
        return mean_auc

    def tune(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Run Optuna optimization and return best parameters."""
        self.logger.info(f"🎯 Starting Optuna tuning ({self.n_trials} trials, {self.n_splits}-fold CV)...")

        study = optuna.create_study(direction="maximize", study_name="lgbm_tuning")
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)

        self.best_params_ = study.best_params
        best_auc = study.best_value
        self.logger.info(f"🏁 Best ROC-AUC: {best_auc:.4f}")
        self.logger.info(f"🧩 Best parameters: {self.best_params_}")

        return self.best_params_
