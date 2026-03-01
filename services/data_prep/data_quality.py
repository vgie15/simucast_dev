import pandas as pd

def analyze_data_quality(filepath):
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    total_rows = len(df)

    # Missing Values
    missing_per_col = df.isna().sum()
    missing_cols = [
        {"column": col, "count": int(count), "pct": round(count / total_rows * 100, 1)}
        for col, count in missing_per_col.items() if count > 0
    ]
    total_missing_pct = round(df.isna().sum().sum() / df.size * 100, 1)

    # Outliers (IQR method — numeric columns only)
    outlier_cols = []
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if outlier_count > 0:
            outlier_cols.append({"column": col, "count": outlier_count})
    total_outlier_pct = round(sum(o["count"] for o in outlier_cols) / total_rows * 100, 1)

    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_count / total_rows * 100, 1)
    
    return {
        "missing": {"pct": total_missing_pct, "columns": missing_cols},
        "outliers": {"pct": total_outlier_pct, "columns": outlier_cols},
        "duplicates": {"pct": duplicate_pct, "count": duplicate_count},
    }