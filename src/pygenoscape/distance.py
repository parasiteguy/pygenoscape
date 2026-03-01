from __future__ import annotations

import numpy as np


def pdistance_matrix(seqs: list[str]) -> np.ndarray:
    """
    p-distance: fraction of mismatches over comparable sites.

    v0.1 rules:
      - ignore sites where either sequence has gap '-' or ambiguous 'N'
      - if no comparable sites, distance is NaN then symmetrically imputed
    """
    n = len(seqs)
    L = len(seqs[0])

    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1").reshape(n, L)
    valid = (arr != b"-") & (arr != b"N")

    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            comp = valid[i] & valid[j]
            denom = int(comp.sum())
            if denom == 0:
                dij = np.nan
            else:
                mism = (arr[i] != arr[j]) & comp
                dij = float(mism.sum()) / denom
            D[i, j] = dij
            D[j, i] = dij

    if np.isnan(D).any():
        D = _nan_impute_symmetric(D)

    return D


def _nan_impute_symmetric(D: np.ndarray) -> np.ndarray:
    D2 = D.copy()
    n = D2.shape[0]

    for i in range(n):
        for j in range(n):
            if i == j:
                D2[i, j] = 0.0
            elif np.isnan(D2[i, j]):
                row = D2[i, :]
                mask = (~np.isnan(row)) & (np.arange(n) != i)
                D2[i, j] = float(row[mask].mean()) if mask.any() else 0.0

    D2 = 0.5 * (D2 + D2.T)
    np.fill_diagonal(D2, 0.0)
    return D2