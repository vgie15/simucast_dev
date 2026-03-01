import os
import re
import json
import random
import pandas as pd
from pathlib import Path
from typing import Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / "key.env")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _complete(prompt: str, api_key: str, model: str) -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Prompt + parsing
# ---------------------------------------------------------------------------

def _build_prompt(columns_with_examples: dict[str, list[Any]], num_rows: int) -> str:
    column_block = "\n".join(
        f'  "{col}": {json.dumps([str(v) for v in examples])}'
        for col, examples in columns_with_examples.items()
    )
    column_names = list(columns_with_examples.keys())

    return f"""You are a synthetic data generator. Generate {num_rows} new rows that match the patterns in the examples below.

Column examples (up to 10 real values per column):
{{
{column_block}
}}

Rules:
- Infer the FORMAT and PATTERN of each column (e.g. ID structure, name style, allowed categories).
- Generate NEW, VARIED values — even if a column only has 2–3 unique examples, extrapolate more that fit the same pattern. Do not just repeat what you see.
- Keep values consistent across columns (e.g. coherent name/email, matching city/zip).
- Return ONLY a JSON array of {num_rows} objects with keys: {json.dumps(column_names)}.
- No explanation, no markdown, just the raw JSON array."""


def _parse_response(response: str) -> Optional[list[dict]]:
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
    json_str = match.group(1) if match else response

    if not json_str.strip().startswith("["):
        match = re.search(r"\[.*\]", json_str, re.DOTALL)
        if match:
            json_str = match.group(0)

    try:
        data = json.loads(json_str)
        if isinstance(data, list) and data:
            return data
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error: {e}")
        print(f"  Response preview: {response[:300]}")

    return None


# ---------------------------------------------------------------------------
# Fallback (no LLM)
# ---------------------------------------------------------------------------

def _fallback_column(series: pd.Series, num_rows: int) -> list[Any]:
    valid = series.dropna()
    if pd.api.types.is_integer_dtype(series):
        return [random.randint(int(valid.min()), int(valid.max())) for _ in range(num_rows)]
    if pd.api.types.is_float_dtype(series):
        return [random.uniform(valid.min(), valid.max()) for _ in range(num_rows)]
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.date_range(valid.min(), valid.max(), periods=num_rows).tolist()
    return random.choices(valid.tolist(), k=num_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    input_path: str,
    num_rows: int = 20,
    output_path: str = None,
    model: str = "gpt-4o-mini",
    api_key: str = None,
    debug: bool = False,
) -> None:
    """
    Append synthetic rows to a CSV or Excel file using OpenAI.

    Args:
        input_path:  Path to input CSV or Excel file.
        num_rows:    Number of synthetic rows to generate.
        output_path: Output path (defaults to synthetic_<input_name>.<ext>).
        model:       OpenAI model to use (default: gpt-4o-mini).
        api_key:     API key (falls back to OPENAI_API_KEY env variable).
        debug:       Print the raw LLM response.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = input_file.suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        raise ValueError(f"Unsupported format '{ext}'. Use .csv, .xlsx, or .xls.")

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("No API key found. Set the OPENAI_API_KEY environment variable.")

    if output_path is None:
        output_path = f"synthetic_{input_file.stem}{ext}"

    df = pd.read_csv(input_path) if ext == ".csv" else pd.read_excel(input_path)
    if df.empty:
        raise ValueError("Input file is empty.")

    print(f"📂 Loaded {len(df)} rows × {len(df.columns)} columns from '{input_path}'")
    print(f"🤖 Model: {model}  |  Rows to generate: {num_rows}\n")

    # Collect examples
    columns_with_examples: dict[str, list[Any]] = {}
    for col in df.columns:
        valid = df[col].dropna()
        if len(valid) > 0:
            columns_with_examples[col] = valid.sample(min(10, len(valid)), random_state=42).tolist()

    # Generate
    ai_rows: list[dict] = []
    try:
        prompt = _build_prompt(columns_with_examples, num_rows)
        print("⚙️  Sending all columns to OpenAI...")
        response = _complete(prompt, api_key=key, model=model)

        if debug:
            print(f"\n{'─'*60}\nRAW RESPONSE:\n{response}\n{'─'*60}\n")

        parsed = _parse_response(response)
        if parsed:
            while len(parsed) < num_rows:
                parsed.extend(parsed[:num_rows - len(parsed)])
            ai_rows = parsed[:num_rows]
            print(f"✅ Generated {len(ai_rows)} rows\n")
        else:
            print("⚠️  Could not parse response. Using fallback.\n")

    except Exception as e:
        print(f"⚠️  OpenAI error: {e}. Using fallback.\n")

    # Build synthetic DataFrame
    synthetic: dict[str, list[Any]] = {}
    for col in df.columns:
        if ai_rows and col in ai_rows[0]:
            synthetic[col] = [row.get(col) for row in ai_rows]
        else:
            print(f"  🔧 Fallback for column '{col}'")
            synthetic[col] = _fallback_column(df[col], num_rows)

    combined = pd.concat([df, pd.DataFrame(synthetic)], ignore_index=True)
    print(f"📊 Original: {len(df)}  |  Synthetic: {num_rows}  |  Total: {len(combined)}\n")

    try:
        if ext == ".csv":
            combined.to_csv(output_path, index=False)
        else:
            combined.to_excel(output_path, index=False)
        print(f"✅ Saved to '{output_path}'")
    except PermissionError:
        print(f"❌ Permission denied. Close '{output_path}' and retry.")
