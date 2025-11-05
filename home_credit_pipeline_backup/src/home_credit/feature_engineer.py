import pandas as pd


class FeatureEngineer:
    """Domain-specific feature engineering for credit risk."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Robust denominators (+1) to avoid division by zero
        income = out["AMT_INCOME_TOTAL"].fillna(0) + 1
        credit = out["AMT_CREDIT"].fillna(0)
        annuity = out["AMT_ANNUITY"].fillna(0)
        days_emp = out["DAYS_EMPLOYED"].fillna(0)
        days_birth = out["DAYS_BIRTH"].fillna(-1)  # negative by convention

        out["CREDIT_INCOME_RATIO"] = credit / income
        out["ANNUITY_INCOME_RATIO"] = annuity / income
        # Convert negative day counters to positive magnitudes for interpretability
        out["AGE_YEARS"] = (-days_birth) / 365.25
        out["EMPLOYMENT_YEARS"] = (-days_emp.clip(upper=0)) / 365.25
        out["EMPLOYED_TO_AGE"] = out["EMPLOYMENT_YEARS"] / (out["AGE_YEARS"].replace(0, 1))

        return out
