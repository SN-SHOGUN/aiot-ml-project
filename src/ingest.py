"""Load a user table from an upload or pasted text."""
from __future__ import annotations

import csv
import io
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "ai4i2020.csv"
MAX_ROWS = 100_000


def limit_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Keep at most 100,000 rows."""
    if len(df) <= MAX_ROWS:
        return df.reset_index(drop=True), ""
    sampled = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
    return sampled, f"The file has {len(df):,} rows. A random sample of {MAX_ROWS:,} is used."


def read_uploaded(file) -> pd.DataFrame:
    """Read csv, tsv, txt, xlsx, or json from a file upload."""
    name = str(getattr(file, "name", "data.csv")).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".json"):
        return pd.read_json(file)
    if name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    return pd.read_csv(file)


def parse_text(text: str) -> pd.DataFrame:
    """Sniff CSV, TSV, or JSON from pasted text."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("The paste box is empty.")
    if raw[0] in "[{":
        try:
            frame = pd.read_json(io.StringIO(raw))
            if isinstance(frame, pd.DataFrame) and frame.shape[1] >= 1:
                return frame
        except ValueError:
            pass
    sep = ","
    try:
        sep = csv.Sniffer().sniff(raw[:4000], delimiters=",\t;|").delimiter
    except csv.Error:
        first = raw.splitlines()[0] if raw.splitlines() else ""
        sep = "\t" if "\t" in first else ","
    frame = pd.read_csv(io.StringIO(raw), sep=sep)
    if frame.shape[1] == 1 and raw.splitlines() and "," in raw.splitlines()[0]:
        frame = pd.read_csv(io.StringIO(raw))
    if frame.shape[1] < 1:
        raise ValueError("Could not parse that text. Use CSV, TSV, or JSON.")
    return frame


def load_example() -> pd.DataFrame:
    """Bundled AI4I table."""
    if not EXAMPLE.exists():
        raise FileNotFoundError("data/ai4i2020.csv is missing. Run python scripts/gen_data.py.")
    return pd.read_csv(EXAMPLE)
