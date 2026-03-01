import pandas as pd
import numpy as np

def get_valid_targets(df: pd.DataFrame) -> list:
    """
    Filter columns to only show valid target variable candidates.
    """
    valid = []
    total_rows = len(df)

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Exclude columns with >50% missing
        missing_pct = df[col].isna().sum() / total_rows
        if missing_pct > 0.5:
            continue

        # Exclude constant columns
        if series.nunique() <= 1:
            continue

        # Exclude ID-like columns (unique ratio > 95%)
        unique_ratio = series.nunique() / total_rows
        if unique_ratio > 0.95:
            continue

        # Exclude free-text (object columns with very high unique ratio > 50%)
        if df[col].dtype == 'object' and unique_ratio > 0.5:
            continue

        valid.append(col)

    return valid


def get_valid_predictors(df: pd.DataFrame, target_col: str) -> dict:
    """
    Get valid predictor columns after selecting a target.
    Returns dict with included columns and reasons for excluded ones.
    """
    total_rows = len(df)
    included = []
    excluded = []

    for col in df.columns:
        if col == target_col:
            continue

        series = df[col].dropna()

        # Exclude ID-like columns
        unique_ratio = series.nunique() / total_rows if total_rows > 0 else 0
        if unique_ratio > 0.95:
            excluded.append({"column": col, "reason": "ID-like column"})
            continue

        # Exclude constant columns
        if series.nunique() <= 1:
            excluded.append({"column": col, "reason": "Constant column"})
            continue

        # Exclude columns with >50% missing
        missing_pct = df[col].isna().sum() / total_rows
        if missing_pct > 0.5:
            excluded.append({"column": col, "reason": f"{round(missing_pct*100)}% missing values"})
            continue

        # Exclude free-text
        if df[col].dtype == 'object' and unique_ratio > 0.5:
            excluded.append({"column": col, "reason": "Free-text column"})
            continue

        # Check correlation leakage for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
            try:
                corr = abs(df[col].corr(df[target_col]))
                if corr > 0.9:
                    excluded.append({"column": col, "reason": f"High correlation with target ({round(corr, 2)})"})
                    continue
            except Exception:
                pass

        included.append(col)

    return {
        "included": included,
        "excluded": excluded
    }


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Detect if the target is classification or regression.
    """
    series = df[target_col].dropna()
    unique_count = series.nunique()

    if df[target_col].dtype == 'object' or unique_count <= 10:
        return 'classification'
    return 'regression'