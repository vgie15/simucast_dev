import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder


# Match the same thresholds used in model1.py / model2.py
UNIQUE_RATIO_ID_THRESHOLD = 0.98
MAX_CATEGORIES            = 100
MAX_CATEGORY_RATIO        = 0.20


def get_valid_targets(df: pd.DataFrame) -> list:
    return [col for col in df.columns if not str(col).startswith('Unnamed')]


def get_valid_predictors(df: pd.DataFrame, target_col: str) -> dict:
    included = [
        col for col in df.columns
        if col != target_col and not str(col).startswith('Unnamed')
    ]
    return {"included": included}


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    series = df[target_col].dropna()
    if len(series) == 0:
        return "unknown"
    if df[target_col].dtype != 'object':
        return 'regression'
    return 'classification'


def _is_datetime_like(series: pd.Series) -> bool:
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if series.dtype == 'object':
            sample = series.dropna().astype(str).head(50)
            if len(sample) == 0:
                return False
            parsed = pd.to_datetime(sample, errors='coerce')
            return float(parsed.notna().mean()) >= 0.80
        return False
    except Exception:
        return False


def compute_feature_importance(df: pd.DataFrame, target_col: str) -> dict:
    """
    Compute feature importance using Spearman rank correlation.

    Why Spearman:
      - Single unified method for both numeric and categorical columns
        (categoricals are ordinal-encoded before ranking, so everything
        is on the same scale)
      - Non-parametric — catches non-linear monotonic relationships that
        Pearson misses
      - Naturally insensitive to ID-like columns: random unique values
        produce near-zero rank correlation with the target
      - No model training required — runs in milliseconds on small datasets
      - All scores are in [-1, 1], so normalisation is honest and consistent

    Returns {column_name: normalised_score_0_to_1}.
    Columns that are filtered out (ID-like, high-cardinality, datetime,
    constant, mostly-missing) are returned with score 0.0 so the UI can
    still display them as Weak.
    """
    try:
        candidates = [
            c for c in df.columns
            if c != target_col and not str(c).startswith('Unnamed')
        ]

        df = df.dropna(subset=[target_col]).copy()
        if len(df) < 5:
            return {}

        n = len(df)

        # Encode target to numeric ranks
        y = df[target_col].copy()
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str)).astype(float)
        else:
            y = y.values.astype(float)

        raw_scores = {}

        for col in candidates:
            series = df[col].copy()

            # ── Pre-filter: mirror model pipeline exclusion rules ─────────────
            if _is_datetime_like(series):
                raw_scores[col] = 0.0
                continue

            nunique = int(series.nunique(dropna=True))
            unique_ratio = nunique / n if n else 0.0

            if nunique <= 1:
                raw_scores[col] = 0.0
                continue

            if unique_ratio > UNIQUE_RATIO_ID_THRESHOLD:
                raw_scores[col] = 0.0
                continue

            if not pd.api.types.is_numeric_dtype(series):
                if nunique > MAX_CATEGORIES or unique_ratio > MAX_CATEGORY_RATIO:
                    raw_scores[col] = 0.0
                    continue

            if series.isna().mean() > 0.50:
                raw_scores[col] = 0.0
                continue

            # ── Encode → Spearman ─────────────────────────────────────────────
            try:
                if pd.api.types.is_numeric_dtype(series):
                    x = series.fillna(series.median()).values.astype(float)
                else:
                    # Ordinal-encode categorical, then rank
                    filled = series.fillna('__missing__').astype(str)
                    x = LabelEncoder().fit_transform(filled).astype(float)

                # Pair-wise drop any remaining NaNs/infs
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 5:
                    raw_scores[col] = 0.0
                    continue

                corr, _ = spearmanr(x[mask], y[mask])
                raw_scores[col] = abs(float(corr)) if not np.isnan(corr) else 0.0

            except Exception:
                raw_scores[col] = 0.0

        # ── Normalise relative to the strongest real predictor ────────────────
        valid_scores = [v for v in raw_scores.values() if v > 0]
        max_score = max(valid_scores) if valid_scores else 1.0

        return {
            col: round(raw_scores.get(col, 0.0) / max_score, 4)
            if raw_scores.get(col, 0.0) > 0 else 0.0
            for col in candidates
        }

    except Exception:
        return {}