import re
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


# -----------------------------
# Helpers: header heuristics
# -----------------------------

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _is_blank(x) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower().startswith("unnamed")


def _clean_cell(x) -> str:
    """Clean a header cell value (remove Unnamed, trim, return '')."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower().startswith("unnamed"):
        return ""
    return s


def looks_like_header_row(values: List[object]) -> bool:
    """
    Heuristic: header rows are mostly short-ish strings, not too numeric.
    Works reasonably for mixed datasets.
    """
    vals = [v for v in values if not _is_blank(v)]
    if not vals:
        return False

    s = [str(v).strip() for v in vals]
    numeric = sum(bool(_NUMERIC_RE.fullmatch(x)) for x in s)
    avg_len = sum(len(x) for x in s) / max(len(s), 1)

    # Header-ish if <40% numeric and average label length not too long
    return (numeric / len(s) < 0.4) and (avg_len < 35)


def _ffill_group_headers(top: List[str]) -> List[str]:
    """Forward fill group headers across blanks (for merged header look)."""
    out = top[:]
    for i in range(1, len(out)):
        if out[i] == "":
            out[i] = out[i - 1]
    return out


def _make_unique(cols: List[str]) -> List[str]:
    """Ensure column names are unique by appending __2, __3, ..."""
    seen = {}
    unique = []
    for c in cols:
        base = c if c else "Unnamed"
        if base in seen:
            seen[base] += 1
            unique.append(f"{base}__{seen[base]}")
        else:
            seen[base] = 1
            unique.append(base)
    return unique


def generate_safe_columns(n: int) -> List[str]:
    return [f"col_{i+1}" for i in range(n)]


# -----------------------------
# Public API
# -----------------------------

def read_dataset_with_header_preview(filepath: str) -> Tuple[pd.DataFrame, Optional[List[List[str]]]]:
    """
    Reads CSV/Excel and returns:
      df_clean: DataFrame with safe, unique columns
      preview_header_rows:
         - None for simple single header
         - [top_group_row, second_header_row] for displaying two-level headers
         - [top_group_row, generated_col_row] if headers are missing and we generated names

    This is designed for robust preview + downstream processing.
    """
    lower = filepath.lower()

    # ---- CSV: no merged headers; just read normally
    if lower.endswith(".csv"):
        df = pd.read_csv(filepath)
        df.columns = _make_unique([str(c).strip() for c in df.columns])
        return df, None

    # ---- Excel: probe first two rows raw (no header)
    probe = pd.read_excel(filepath, header=None, nrows=2)

    row0 = probe.iloc[0].tolist() if len(probe) > 0 else []
    row1 = probe.iloc[1].tolist() if len(probe) > 1 else []

    row0_is_header = looks_like_header_row(row0)
    row1_is_header = looks_like_header_row(row1)

    # Case A: Row0 is group header (merged) AND Row1 is the real column names -> header=1
    if row0_is_header and row1_is_header:
        df = pd.read_excel(filepath, header=1)
        # preview: show row0 as group headers (filled), row1 as actual columns
        top = _ffill_group_headers([_clean_cell(x) for x in row0])
        second = [str(c).strip() for c in df.columns]
        # clean + unique actual columns
        df.columns = _make_unique([_clean_cell(c) or f"col_{i+1}" for i, c in enumerate(df.columns)])
        # align second row to cleaned df.columns for UI (so clicking columns matches)
        second = df.columns.tolist()
        return df, [top, second]

    # Case B: Normal Excel with only 1 header row -> header=0
    if row0_is_header and not row1_is_header:
        df = pd.read_excel(filepath, header=0)
        df.columns = _make_unique([_clean_cell(c) or f"col_{i+1}" for i, c in enumerate(df.columns)])
        return df, None

    # Case C: Headers are missing or not reliable -> header=None, generate safe names
    df = pd.read_excel(filepath, header=None)
    generated = generate_safe_columns(df.shape[1])
    df.columns = _make_unique(generated)

    # Preview row: if row0 had group labels, show them; else show blanks
    top = _ffill_group_headers([_clean_cell(x) for x in row0]) if row0 else [""] * df.shape[1]
    return df, [top, df.columns.tolist()]


def format_preview_records(df: pd.DataFrame, n: int = 5) -> List[dict]:
    """
    Format preview so UI looks nicer:
      - NaN -> ""
      - 19.0 -> 19 (whole floats)
      - keep true decimals (3.16)
    """
    temp = df.head(n).copy()
    temp = temp.replace({np.nan: ""})

    def fix(x):
        if isinstance(x, float) and x != "" and float(x).is_integer():
            return int(x)
        return x

    # apply elementwise
    return temp.applymap(fix).to_dict(orient="records")