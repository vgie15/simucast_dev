"""
services/build_model/model_list/model1.py
SimuCast — Ridge Regression (Balanced)

Includes:
- Shared predictor filtering for BOTH auto + user-selected predictors
- Exclude datetime-like columns (prevents massive one-hot blowups)
- Exclude high-cardinality categorical columns
- OneHotEncoder uses sparse output (memory-safe)
- 80/20 holdout evaluation (NO Cross-Validation)
- Basic overfitting warning (train vs test performance gap)
- UI warnings are capped
- JSON dump safe with numpy types (default=str)
- Saves artifacts (pickle bundle + JSON summary) for Scenario / What-if
"""

import os
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Optional, Tuple, List

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# ── Thresholds (tune here, no touching logic below) ──────────────────────────
MIN_ROWS                  = 30
TEST_SIZE                 = 0.20
RANDOM_STATE              = 42

MAX_MISSING_RATIO         = 0.50
UNIQUE_RATIO_ID_THRESHOLD = 0.98
LEAKAGE_CORR_THRESHOLD    = 0.90

# High-cardinality categorical protection
MAX_CATEGORIES            = 100      # hard cap
MAX_CATEGORY_RATIO        = 0.20     # OR cap by ratio of unique categories to rows

# Overfitting warning threshold
OVERFIT_R2_GAP_THRESHOLD  = 0.15     # warn if train R² - test R² > this

# Ridge regularization strength
RIDGE_ALPHA               = 1.0

# UI safety
MAX_WARNINGS              = 50

MODEL_NAME = "Ridge Regression"
MODEL_KEY  = "ridge_regression"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — called by the build_model blueprint
# ─────────────────────────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    target_column: str,
    predictor_columns: Optional[list] = None,
    artifacts_dir: str = "artifacts",
    dataset_id: str = "dataset"
) -> dict:
    """
    Balanced Ridge Regression pipeline (holdout eval only).

    Returns:
    {
        "success":   bool,
        "error":     str | None,
        "warnings":  list[str],
        "model_id":  str,
        "model_name": str,
        "model_key": str,
        "metrics":   dict,
        "predictors": list,
        "feature_names_out": list,
        "target":    str,
        "artifact_path": str,
    }
    """
    ui_warnings: List[str] = []

    # ── 1) VALIDATE ───────────────────────────────────────────────────────────
    err = _validate(df, target_column)
    if err:
        return _fail(err)

    # Drop rows where target is missing (y imputation = drop)
    before = len(df)
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        ui_warnings.append(
            f"{dropped} row(s) dropped — target '{target_column}' was missing."
        )

    if len(df) < MIN_ROWS:
        return _fail(
            f"Only {len(df)} usable rows after dropping missing targets. "
            f"Need at least {MIN_ROWS}."
        )

    # ── 2) BUILD PREDICTOR LIST ───────────────────────────────────────────────
    if predictor_columns:
        x_cols, log = _select_user_predictors(df, predictor_columns, target_column)
    else:
        x_cols, log = _auto_select_predictors(df, target_column)

    ui_warnings.extend(log)

    if not x_cols:
        return _fail(
            "No valid predictors after filtering. "
            "Try selecting columns manually or cleaning the dataset further."
        )

    # ── 3) SPLIT X and y ──────────────────────────────────────────────────────
    X = df[x_cols].copy()
    y = df[target_column].copy()

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # ── 4) BUILD PIPELINE ─────────────────────────────────────────────────────
    pipeline = _build_pipeline(num_cols, cat_cols)

    # ── 5) HOLDOUT SPLIT + FIT ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)

    # Evaluate on train + test (for overfitting warning)
    train_scores = _evaluate(pipeline, X_train, y_train)
    test_scores  = _evaluate(pipeline, X_test, y_test)

    # Overfitting heuristic warning
    r2_gap = train_scores["r2"] - test_scores["r2"]
    if r2_gap > OVERFIT_R2_GAP_THRESHOLD:
        ui_warnings.append(
            f"Possible overfitting detected: Train R² ({train_scores['r2']:.3f}) "
            f"is much higher than Test R² ({test_scores['r2']:.3f})."
        )

    metrics = {
        # Holdout split (main metrics)
        "holdout_r2":   round(test_scores["r2"],   4),
        "holdout_rmse": round(test_scores["rmse"], 4),
        "holdout_mae":  round(test_scores["mae"],  4),

        # Train metrics (for diagnostics)
        "train_r2":     round(train_scores["r2"],   4),
        "train_rmse":   round(train_scores["rmse"], 4),
        "train_mae":    round(train_scores["mae"],  4),

        # Sample info
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "n_predictors": int(len(x_cols)),

        # Model params
        "ridge_alpha":  float(RIDGE_ALPHA),
    }

    # Cap warnings to avoid UI spam
    ui_warnings = _cap_warnings(ui_warnings)

    # ── 6) FEATURE NAMES (post-encoding) ─────────────────────────────────────
    feature_names_out = _get_feature_names(pipeline, num_cols, cat_cols)

    # ── 7) SAVE ARTIFACTS ─────────────────────────────────────────────────────
    model_id = f"ridge_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifact_path = _save_artifacts(
        artifacts_dir=artifacts_dir,
        model_id=model_id,
        pipeline=pipeline,
        predictors=x_cols,
        feature_names_out=feature_names_out,
        target=target_column,
        metrics=metrics
    )

    logger.info(
        "[%s] done. Test R²=%.4f  RMSE=%.4f  predictors=%d",
        MODEL_KEY, metrics["holdout_r2"], metrics["holdout_rmse"], len(x_cols)
    )

    return {
        "success":           True,
        "error":             None,
        "warnings":          ui_warnings,
        "model_id":          model_id,
        "model_name":        MODEL_NAME,
        "model_key":         MODEL_KEY,
        "metrics":           metrics,
        "predictors":        x_cols,
        "feature_names_out": feature_names_out,
        "target":            target_column,
        "artifact_path":     artifact_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame, target: str) -> Optional[str]:
    if target not in df.columns:
        return f"Target column '{target}' not found in the dataset."

    if len(df) < MIN_ROWS:
        return (
            f"Dataset has only {len(df)} rows. "
            f"{MODEL_NAME} needs at least {MIN_ROWS} rows."
        )

    if not pd.api.types.is_numeric_dtype(df[target]):
        return (
            f"Selected target '{target}' is not numeric. "
            f"{MODEL_NAME} is a regression model — "
            "please choose a numeric target column."
        )

    return None


def _auto_select_predictors(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    candidates = [c for c in df.columns if c != target]
    selected, log = _filter_predictors(df, candidates, target)
    return selected, log


def _select_user_predictors(df: pd.DataFrame, predictors: list, target: str) -> Tuple[List[str], List[str]]:
    valid: List[str] = []
    log: List[str] = []

    for col in predictors:
        if col == target:
            log.append(f"'{col}' skipped — cannot use target as a predictor.")
            continue
        if col not in df.columns:
            log.append(f"'{col}' skipped — column not found in dataset.")
            continue
        valid.append(col)

    if not valid:
        return [], log

    filtered, filter_log = _filter_predictors(df, valid, target)
    return filtered, log + filter_log


def _filter_predictors(df: pd.DataFrame, cols: list, target: str) -> Tuple[List[str], List[str]]:
    selected: List[str] = []
    log: List[str] = []
    n = len(df)

    for col in cols:
        # datetime-like exclusion (prevents huge one-hot)
        if _is_datetime_like(df[col]):
            log.append(f"'{col}' excluded — datetime-like column (not used for this model).")
            continue

        missing_ratio = float(df[col].isna().mean())
        nunique = int(df[col].nunique(dropna=True))
        unique_ratio = (nunique / n) if n else 0.0

        if unique_ratio > UNIQUE_RATIO_ID_THRESHOLD:
            log.append(f"'{col}' excluded — looks like an ID column (unique ratio {unique_ratio:.0%}).")
            continue

        if nunique <= 1:
            log.append(f"'{col}' excluded — constant column (only one unique value).")
            continue

        if missing_ratio > MAX_MISSING_RATIO:
            log.append(f"'{col}' excluded — {missing_ratio:.0%} values are missing.")
            continue

        # high-cardinality categorical exclusion
        if not pd.api.types.is_numeric_dtype(df[col]):
            ratio = (nunique / n) if n else 0.0
            if nunique > MAX_CATEGORIES or ratio > MAX_CATEGORY_RATIO:
                log.append(f"'{col}' excluded — too many categories ({nunique}).")
                continue

        # leakage check (numeric only)
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                corr = df[[col, target]].dropna().corr().loc[col, target]
                if pd.notna(corr) and abs(float(corr)) >= LEAKAGE_CORR_THRESHOLD:
                    log.append(
                        f"'{col}' excluded — correlation with target is {float(corr):.2f} "
                        "(possible data leakage)."
                    )
                    continue
            except Exception:
                pass

        selected.append(col)

    return selected, log


def _is_datetime_like(series: pd.Series) -> bool:
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        if series.dtype == "object":
            sample = series.dropna().astype(str).head(50)
            if len(sample) == 0:
                return False
            parsed = pd.to_datetime(sample, errors="coerce")
            return float(parsed.notna().mean()) >= 0.80

        return False
    except Exception:
        return False


def _cap_warnings(warnings_list: List[str]) -> List[str]:
    if len(warnings_list) <= MAX_WARNINGS:
        return warnings_list
    return warnings_list[:MAX_WARNINGS] + ["More columns were excluded (truncated)."]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    transformers = []

    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",        Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
    ])


def _evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "r2":   float(r2_score(y_test, y_pred)),
    }


def _get_feature_names(pipeline: Pipeline, num_cols: list, cat_cols: list) -> list:
    try:
        pre = pipeline.named_steps["preprocessor"]
        names = list(num_cols)

        if cat_cols:
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
            names += ohe.get_feature_names_out(cat_cols).tolist()

        return names
    except Exception:
        return num_cols + cat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Artifact save / load
# ─────────────────────────────────────────────────────────────────────────────

def _save_artifacts(
    artifacts_dir: str,
    model_id: str,
    pipeline: Pipeline,
    predictors: list,
    feature_names_out: list,
    target: str,
    metrics: dict
) -> str:
    os.makedirs(artifacts_dir, exist_ok=True)
    path = os.path.join(artifacts_dir, f"{model_id}.pkl")

    bundle = {
        "model_id":          model_id,
        "model_name":        MODEL_NAME,
        "model_key":         MODEL_KEY,
        "pipeline":          pipeline,
        "predictors":        predictors,
        "feature_names_out": feature_names_out,
        "target":            target,
        "metrics":           metrics,
        "trained_at":        datetime.now().isoformat(),
    }

    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    summary = {k: v for k, v in bundle.items() if k != "pipeline"}
    with open(path.replace(".pkl", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return path


def load_bundle(artifact_path: str) -> dict:
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Utility — quick model card for UI
# ─────────────────────────────────────────────────────────────────────────────

def model_card(metrics: dict) -> dict:
    """Display-ready summary dict for model card / comparison table."""
    r2 = float(metrics.get("holdout_r2", 0))
    return {
        "name": MODEL_NAME,
        "key":  MODEL_KEY,
        "primary_metric_label": "R² (Holdout)",
        "primary_metric_value": f"{r2:.4f}",
        "badge_color": _r2_badge(r2),
        "rows": [
            {"label": "Holdout R²",   "value": f"{metrics.get('holdout_r2', 0):.4f}"},
            {"label": "Holdout RMSE", "value": f"{metrics.get('holdout_rmse', 0):.4f}"},
            {"label": "Holdout MAE",  "value": f"{metrics.get('holdout_mae', 0):.4f}"},
            {"label": "Train R²",     "value": f"{metrics.get('train_r2', 0):.4f}"},
            {"label": "Train RMSE",   "value": f"{metrics.get('train_rmse', 0):.4f}"},
            {"label": "Train rows",   "value": str(metrics.get("n_train", "—"))},
            {"label": "Test rows",    "value": str(metrics.get("n_test", "—"))},
            {"label": "Predictors",   "value": str(metrics.get("n_predictors", "—"))},
        ],
    }


def _r2_badge(r2: float) -> str:
    if r2 >= 0.80:
        return "green"
    if r2 >= 0.60:
        return "yellow"
    return "red"


def _fail(msg: str) -> dict:
    return {
        "success": False,
        "error": msg,
        "warnings": [],
        "model_id": "",
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "metrics": {},
        "predictors": [],
        "feature_names_out": [],
        "target": "",
        "artifact_path": "",
    }