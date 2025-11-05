from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class Preprocessor:
    """Builds the sklearn ColumnTransformer for numeric + categorical data."""

    def __init__(self, impute_strategy: str = "median"):
        self.impute_strategy = impute_strategy
        self.transformer: ColumnTransformer | None = None

    def split(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float,
        random_state: int,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)
        if test_size == 0:
            return X, None, y, None
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )

    def build(self, X: pd.DataFrame) -> ColumnTransformer:
        # Identify column types
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Separate binary vs continuous numeric columns
        binary_cols = [col for col in numeric_cols if set(X[col].dropna().unique()).issubset({0, 1})]
        continuous_cols = [col for col in numeric_cols if col not in binary_cols]

        # Pipelines
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.impute_strategy)),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )

        bin_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combined transformer
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipe, continuous_cols),
                ("bin", bin_pipe, binary_cols),
                ("cat", cat_pipe, categorical_cols),
            ]
        )

        print(f"📊 Continuous: {len(continuous_cols)}, Binary: {len(binary_cols)}, Categorical: {len(categorical_cols)}")
        return self.transformer
