"""
services/build_model/model_list/model2.py
SimuCast — Random Forest Regressor (Balanced)
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
MIN_ROWS                  = 30
TEST_SIZE                 = 0.20
RANDOM_STATE              = 42

MAX_MISSING_RATIO         = 0.50
UNIQUE_RATIO_ID_THRESHOLD = 0.98
LEAKAGE_CORR_THRESHOLD    = 0.90

MAX_CATEGORIES            = 100
MAX_CATEGORY_RATIO        = 0.20

OVERFIT_R2_GAP_THRESHOLD  = 0.15

RF_N_ESTIMATORS           = 300
RF_MAX_DEPTH              = None
RF_MIN_SAMPLES_SPLIT      = 2
RF_MIN_SAMPLES_LEAF       = 1
RF_MAX_FEATURES           = "sqrt"
RF_N_JOBS                 = -1

MAX_WARNINGS              = 50
MAX_FEATURES_WARNING      = 5000

MODEL_NAME = "Random Forest Regressor"
MODEL_KEY  = "random_forest_regressor"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    target_column: str,
    predictor_columns: Optional[list] = None,
    artifacts_dir: str = "artifacts",
    dataset_id: str = "dataset",
    test_size: float = TEST_SIZE,
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth: Optional[int] = RF_MAX_DEPTH,
    min_samples_leaf: int = RF_MIN_SAMPLES_LEAF,
    ridge_alpha: float = 1.0   # accepted but ignored — keeps confirm_model loop simple
) -> dict:
    ui_warnings: List[str] = []

    # ── 1) VALIDATE ───────────────────────────────────────────────────────────
    err = _validate(df, target_column)
    if err:
        return _fail(err)

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

    # ── 2) PREDICTOR LIST ─────────────────────────────────────────────────────
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
    pipeline = _build_pipeline(
        num_cols, cat_cols,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )

    # ── 5) HOLDOUT SPLIT + FIT ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)

    train_scores = _evaluate(pipeline, X_train, y_train)
    test_scores  = _evaluate(pipeline, X_test,  y_test)

    r2_gap = train_scores["r2"] - test_scores["r2"]
    if r2_gap > OVERFIT_R2_GAP_THRESHOLD:
        ui_warnings.append(
            f"Possible overfitting detected: Train R² ({train_scores['r2']:.3f}) "
            f"is much higher than Test R² ({test_scores['r2']:.3f}). "
            "Consider reducing max_depth or increasing min_samples_leaf."
        )

    # ── 6) FEATURE NAMES ──────────────────────────────────────────────────────
    feature_names_out = _get_feature_names(pipeline, num_cols, cat_cols)

    if len(feature_names_out) > MAX_FEATURES_WARNING:
        ui_warnings.append(
            f"Encoded feature count is high ({len(feature_names_out)}). "
            "This may slow training. Consider removing high-cardinality columns."
        )

    metrics = {
        "holdout_r2":        round(test_scores["r2"],   4),
        "holdout_rmse":      round(test_scores["rmse"], 4),
        "holdout_mae":       round(test_scores["mae"],  4),
        "train_r2":          round(train_scores["r2"],   4),
        "train_rmse":        round(train_scores["rmse"], 4),
        "train_mae":         round(train_scores["mae"],  4),
        "n_train":           int(len(X_train)),
        "n_test":            int(len(X_test)),
        "n_predictors":      int(len(x_cols)),
        "n_features_out":    int(len(feature_names_out)),
        "rf_n_estimators":   int(n_estimators),
        "rf_max_depth":      max_depth,
        "rf_min_samples_leaf": int(min_samples_leaf),
        "test_size":         float(test_size),
    }

    ui_warnings = _cap_warnings(ui_warnings)

    # ── 7) SAVE ARTIFACTS ─────────────────────────────────────────────────────
    model_id = f"rf_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
# Validation
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


# ─────────────────────────────────────────────────────────────────────────────
# Predictor selection
# ─────────────────────────────────────────────────────────────────────────────

def _auto_select_predictors(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    candidates = [c for c in df.columns if c != target]
    return _filter_predictors(df, candidates, target)


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
        if _is_datetime_like(df[col]):
            log.append(f"'{col}' excluded — datetime-like column.")
            continue

        missing_ratio = float(df[col].isna().mean())
        nunique = int(df[col].nunique(dropna=True))
        unique_ratio = (nunique / n) if n else 0.0

        if unique_ratio > UNIQUE_RATIO_ID_THRESHOLD:
            log.append(f"'{col}' excluded — looks like an ID column (unique ratio {unique_ratio:.0%}).")
            continue

        if nunique <= 1:
            log.append(f"'{col}' excluded — constant column.")
            continue

        if missing_ratio > MAX_MISSING_RATIO:
            log.append(f"'{col}' excluded — {missing_ratio:.0%} values are missing.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            ratio = (nunique / n) if n else 0.0
            if nunique > MAX_CATEGORIES or ratio > MAX_CATEGORY_RATIO:
                log.append(f"'{col}' excluded — too many categories ({nunique}).")
                continue

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

def _build_pipeline(
    num_cols: list,
    cat_cols: list,
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth: Optional[int] = RF_MAX_DEPTH,
    min_samples_leaf: int = RF_MIN_SAMPLES_LEAF
) -> Pipeline:
    transformers = []

    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=min_samples_leaf,
        max_features=RF_MAX_FEATURES,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",        model),
    ])


def _evaluate(pipeline: Pipeline, X, y) -> dict:
    y_pred = pipeline.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae":  float(mean_absolute_error(y, y_pred)),
        "r2":   float(r2_score(y, y_pred)),
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
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def model_card(metrics: dict) -> dict:
    r2 = float(metrics.get("holdout_r2", 0))
    return {
        "name": MODEL_NAME,
        "key":  MODEL_KEY,
        "primary_metric_label": "R² (Holdout)",
        "primary_metric_value": f"{r2:.4f}",
        "badge_color": _r2_badge(r2),
        "rows": [
            {"label": "Holdout R²",       "value": f"{metrics.get('holdout_r2', 0):.4f}"},
            {"label": "Holdout RMSE",     "value": f"{metrics.get('holdout_rmse', 0):.4f}"},
            {"label": "Holdout MAE",      "value": f"{metrics.get('holdout_mae', 0):.4f}"},
            {"label": "Train R²",         "value": f"{metrics.get('train_r2', 0):.4f}"},
            {"label": "Train rows",       "value": str(metrics.get("n_train", "—"))},
            {"label": "Test rows",        "value": str(metrics.get("n_test", "—"))},
            {"label": "Predictors",       "value": str(metrics.get("n_predictors", "—"))},
            {"label": "Trees",            "value": str(metrics.get("rf_n_estimators", "—"))},
            {"label": "Max Depth",        "value": str(metrics.get("rf_max_depth") or "Auto")},
            {"label": "Min Samples Leaf", "value": str(metrics.get("rf_min_samples_leaf", "—"))},
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
        "success":           False,
        "error":             msg,
        "warnings":          [],
        "model_id":          "",
        "model_name":        MODEL_NAME,
        "model_key":         MODEL_KEY,
        "metrics":           {},
        "predictors":        [],
        "feature_names_out": [],
        "target":            "",
        "artifact_path":     "",
    }