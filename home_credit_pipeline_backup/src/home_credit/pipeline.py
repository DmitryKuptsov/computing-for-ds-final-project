import os
import warnings
import numpy as np
import pandas as pd
from textwrap import indent
from scipy.sparse import issparse
from .config import Config
from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .preprocessor import Preprocessor
from .model_trainer import ModelTrainer
from .evaluator import Evaluator
from .hyper_tuner import HyperTuner
from .threshold_analyzer import ThresholdAnalyzer
from .balancer import Balancer
from .utils.logger import get_logger


class PipelineRunner:
    """
    End-to-end Home Credit Default Risk pipeline.

    Steps:
      1. Load and optionally sample training data
      2. Perform feature engineering
      3. Preprocess data (imputation, encoding, scaling)
      4. Optionally tune hyperparameters with Optuna
      5. Optionally balance classes (SMOTE, oversampling, undersampling)
      6. Train LightGBM model with stratified cross-validation
      7. Evaluate model (metrics, confusion matrix)
      8. Optionally analyze precision/recall trade-offs via threshold sweep
      9. Predict on test data and save Kaggle submission
    """

    def __init__(self, config_path: str):
        """Initialize pipeline with YAML configuration."""
        self.config = Config.from_yaml(config_path)
        self.logger = get_logger(self.__class__.__name__)
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn",
        )

    def run(self) -> None:
        """Run the full pipeline as defined in configuration."""
        cfg = self.config
        self.logger.info("🚀 Starting Home Credit FINAL TRAINING (full data)")

        # 1️⃣ Load training data
        df = DataLoader(cfg.data["path_train"], cfg.data.get("sample_size")).load()
        self.logger.info(f"📦 Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # 2️⃣ Feature engineering
        df = FeatureEngineer().transform(df)
        self.logger.info("🧩 Feature engineering completed successfully.")

        # 3️⃣ Preprocessing — imputation, encoding, scaling
        prep = Preprocessor(cfg.preprocessing.get("impute_strategy", "median"))
        transformer = prep.build(df.drop(columns=[cfg.data["target_col"]]))
        X_full = transformer.fit_transform(df.drop(columns=[cfg.data["target_col"]]))
        y_full = df[cfg.data["target_col"]].astype(int)

        def ensure_dense(X):
            """Convert sparse matrices to dense NumPy arrays."""
            return X.toarray() if issparse(X) else X

        X_full = ensure_dense(X_full)
        self.logger.info(f"🧹 Preprocessing complete. Final feature matrix shape: {X_full.shape}")

        # 4️⃣ Optional hyperparameter tuning (Optuna)
        if cfg.model.get("tune", False):
            self.logger.info("🔍 Hyperparameter tuning enabled.")
            tuner = HyperTuner(n_trials=cfg.model.get("n_trials", 30))
            best_params = tuner.tune(X_full, y_full)
            cfg.model["params"].update(best_params)
            self.logger.info("✅ Updated model parameters with tuned values.")
        else:
            self.logger.info("ℹ️ Skipping hyperparameter tuning (disabled in config).")

        # 5️⃣ Optional class balancing (SMOTE / oversample / undersample)
        balance_cfg = cfg.validation.get("balance_strategy", "none")
        if balance_cfg != "none":
            self.logger.info(f"⚖️ Applying class balancing strategy: {balance_cfg}")
            balancer = Balancer(strategy=balance_cfg)
            X_full, y_full = balancer.balance(X_full, y_full)
            self.logger.info(f"✅ Balancing complete. New shape: {X_full.shape}, class ratio = {y_full.mean():.3f}")
        else:
            self.logger.info("ℹ️ Skipping class balancing (strategy = 'none').")

        # 6️⃣ Train LightGBM model with stratified K-fold cross-validation
        trainer = ModelTrainer(cfg.model["params"], cfg.output["model_path"], n_splits=5)
        mean_auc = trainer.cross_validate(X_full, y_full)
        self.logger.info(f"🌲 Cross-validated ROC-AUC (5 folds, full data): {mean_auc:.4f}")

        # 7️⃣ Evaluate trained model on full data (Confusion Matrix + Metrics)
        model = trainer.models[-1]
        self.logger.info("📊 Evaluating model performance and generating confusion matrix...")
        y_pred_proba = model.predict_proba(X_full)[:, 1]

        evaluator = Evaluator(cfg.output["metrics_path"], "artifacts")
        metrics = evaluator.evaluate(y_full, y_pred_proba)

        metrics_str = indent(
            "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float)]),
            " " * 4,
        )
        self.logger.info(f"📈 Evaluation metrics:\n{metrics_str}")

        # 8️⃣ Optional threshold analysis (Precision/Recall/F1 sweep)
        if cfg.validation.get("analyze_thresholds", False):
            self.logger.info("🔎 Running threshold analysis (optional)...")
            analyzer = ThresholdAnalyzer("artifacts")
            best_thr = analyzer.run(y_full, y_pred_proba)
            self.logger.info(f"✅ Threshold analysis complete. Suggested best F1 threshold = {best_thr:.3f}")
        else:
            self.logger.info("ℹ️ Skipping threshold analysis (disabled in config).")

        # 9️⃣ Predict on test set and generate submission
        test_path = "data/application_test.csv"
        submission_path = "artifacts/submission.csv"

        if not os.path.exists(test_path):
            self.logger.error("❌ application_test.csv not found. Cannot create submission file.")
            return

        self.logger.info("📤 Generating final Kaggle submission...")
        test_df = pd.read_csv(test_path)
        test_features = FeatureEngineer().transform(test_df)
        X_test = transformer.transform(test_features)
        X_test = ensure_dense(X_test)

        y_test_pred = model.predict_proba(X_test)[:, 1]

        submission = pd.DataFrame({
            "SK_ID_CURR": test_df["SK_ID_CURR"],
            "TARGET": y_test_pred,
        })
        os.makedirs("artifacts", exist_ok=True)
        submission.to_csv(submission_path, index=False)

        # ✅ Sanity check & summary
        self.logger.info(f"✅ Submission saved: {submission_path} ({len(submission):,} rows)")
        self.logger.info(
            f"🔎 Test prediction stats — mean: {y_test_pred.mean():.4f}, "
            f"min: {y_test_pred.min():.4f}, max: {y_test_pred.max():.4f}"
        )

        self.logger.info("🏁 Final training pipeline completed successfully.")
