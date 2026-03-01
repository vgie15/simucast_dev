import pandas as pd

def apply_missing_cleaning(filepath: str, options: dict, output_path: str = None) -> dict:
    """
    Apply missing value cleaning to a dataset.

    Args:
        filepath:    Path to the CSV or Excel file.
        options:     Dict of selected cleaning actions.
                     Keys: 'remove_rows', 'mean', 'median', 'mode'
        output_path: Where to save the cleaned file (defaults to overwrite).

    Returns:
        Dict with before/after stats.
    """
    ext = '.csv' if filepath.endswith('.csv') else '.xlsx'
    df = pd.read_csv(filepath) if ext == '.csv' else pd.read_excel(filepath)

    original_missing = int(df.isna().sum().sum())
    original_rows = len(df)

    df_clean = df.copy()

    # Remove rows first — if selected, skip imputation
    if options.get('remove_rows'):
        df_clean = df_clean.dropna()
    else:
        # Mean imputation — numeric columns
        if options.get('mean'):
            for col in df_clean.select_dtypes(include=['int64', 'float64']).columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

        # Median imputation — numeric columns
        if options.get('median'):
            for col in df_clean.select_dtypes(include=['int64', 'float64']).columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # Mode imputation — categorical/object columns
        if options.get('mode'):
            for col in df_clean.select_dtypes(include=['object', 'category']).columns:
                if df_clean[col].isna().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    remaining_missing = int(df_clean.isna().sum().sum())
    removed_rows = original_rows - len(df_clean)

    # Save
    save_path = output_path or filepath
    if ext == '.csv':
        df_clean.to_csv(save_path, index=False)
    else:
        df_clean.to_excel(save_path, index=False)

    return {
        "original_missing": original_missing,
        "remaining_missing": remaining_missing,
        "fixed": original_missing - remaining_missing,
        "removed_rows": removed_rows,
        "remaining_rows": len(df_clean)
    }