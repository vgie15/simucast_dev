import pandas as pd
import numpy as np

def apply_outlier_cleaning(filepath: str, options: dict, output_path: str = None) -> dict:
    """
    Apply outlier cleaning to a dataset using IQR method.

    Args:
        filepath:    Path to the CSV or Excel file.
        options:     Dict of selected cleaning actions.
                     Keys: 'cap' (cap extreme values), 'remove' (remove outlier rows)
        output_path: Where to save the cleaned file (defaults to overwrite).

    Returns:
        Dict with before/after stats.
    """
    ext = '.csv' if filepath.endswith('.csv') else '.xlsx'
    df = pd.read_csv(filepath) if ext == '.csv' else pd.read_excel(filepath)

    original_rows = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    total_outliers_before = 0

    # Calculate IQR bounds per column
    bounds = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[col] = (lower, upper)
        total_outliers_before += int(((df[col] < lower) | (df[col] > upper)).sum())

    df_clean = df.copy()

    if options.get('cap'):
        # Cap values to IQR bounds (Winsorization)
        for col in numeric_cols:
            lower, upper = bounds[col]
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

    elif options.get('remove'):
        # Remove rows that have any outlier in any numeric column
        mask = pd.Series([True] * len(df_clean), index=df_clean.index)
        for col in numeric_cols:
            lower, upper = bounds[col]
            mask &= (df_clean[col] >= lower) & (df_clean[col] <= upper)
        df_clean = df_clean[mask]

    # Count remaining outliers
    total_outliers_after = 0
    for col in numeric_cols:
        lower, upper = bounds[col]
        total_outliers_after += int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())

    removed_rows = original_rows - len(df_clean)

    # Save
    save_path = output_path or filepath
    if ext == '.csv':
        df_clean.to_csv(save_path, index=False)
    else:
        df_clean.to_excel(save_path, index=False)

    return {
        "original_rows": original_rows,
        "remaining_rows": len(df_clean),
        "removed_rows": removed_rows,
        "outliers_before": total_outliers_before,
        "outliers_after": total_outliers_after,
        "fixed": total_outliers_before - total_outliers_after,
    }