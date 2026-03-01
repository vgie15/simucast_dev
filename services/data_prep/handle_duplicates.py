import pandas as pd

def get_duplicate_info(filepath: str) -> dict:
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    duplicate_count = int(df.duplicated().sum())
    total_rows = len(df)
    pct = round(duplicate_count / total_rows * 100, 1) if total_rows > 0 else 0

    return {
        "count": duplicate_count,
        "pct": pct,
        "total_rows": total_rows
    }

def remove_duplicates(filepath: str, output_path: str = None) -> dict:
    ext = '.csv' if filepath.endswith('.csv') else '.xlsx'
    df = pd.read_csv(filepath) if ext == '.csv' else pd.read_excel(filepath)

    original_count = len(df)
    df_cleaned = df.drop_duplicates()
    removed_count = original_count - len(df_cleaned)

    if output_path is None:
        output_path = filepath  # overwrite in place

    if ext == '.csv':
        df_cleaned.to_csv(output_path, index=False)
    else:
        df_cleaned.to_excel(output_path, index=False)

    return {
        "original_count": original_count,
        "removed_count": removed_count,
        "remaining_count": len(df_cleaned)
    }