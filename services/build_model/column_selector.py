import pandas as pd

def get_valid_targets(df: pd.DataFrame) -> list:
    return df.columns.tolist()

def get_valid_predictors(df: pd.DataFrame, target_col: str) -> dict:
    included = [col for col in df.columns if col != target_col]
    return {"included": included}

def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    series = df[target_col].dropna()
    if df[target_col].dtype == 'object' or series.nunique() <= 10:
        return 'classification'
    return 'regression'