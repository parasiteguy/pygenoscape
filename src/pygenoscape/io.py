from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_aligned_fasta(path: Path):
    headers: list[str] = []
    seqs: list[str] = []
    seq_parts: list[str] = []

    current_header = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    seq = "".join(seq_parts).upper()
                    seqs.append(seq)
                    seq_parts = []
                current_header = line[1:].strip()
                headers.append(current_header)
            else:
                seq_parts.append(line)

    if current_header is not None:
        seq = "".join(seq_parts).upper()
        seqs.append(seq)

    if not seqs:
        raise ValueError("FASTA appears empty or invalid.")

    L = len(seqs[0])
    bad = [i for i, s in enumerate(seqs) if len(s) != L]
    if bad:
        raise ValueError(f"FASTA is not aligned: sequences at indices {bad[:10]} differ in length from {L}.")

    return headers, seqs


def read_coords_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "lon", "lat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"coords.csv missing required columns: {sorted(missing)}")

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df = df.set_index("id", drop=True)

    df["lon"] = pd.to_numeric(df["lon"], errors="raise")
    df["lat"] = pd.to_numeric(df["lat"], errors="raise")
    return df


def read_dist_matrix_csv(path: Path) -> pd.DataFrame:
    """
    Read a square distance matrix CSV with row/col labels.

    Expected:
      - header row contains column ids
      - first column contains row ids
      - numeric entries

    We allow row/col sets to match even if order differs.
    """
    df = pd.read_csv(path, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Distance matrix must be square. Got {df.shape}.")

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if set(df.index) != set(df.columns):
        raise ValueError("Distance matrix row/col labels do not match as sets.")

    # Coerce to numeric (raise if bad)
    df = df.apply(pd.to_numeric, errors="raise")

    return df