from __future__ import annotations

import numpy as np


def compute_height(D: np.ndarray, *, ids: list[str], method: str = "pcoa1"):
    """
    Convert an NxN distance matrix into a per-sample height vector z (length N).

    Methods:
      - pcoa1 (default): first PCoA axis
      - mean: mean distance per sample
      - focal:<id>: distance to focal sample
    """
    method = method.strip()

    if method == "mean":
        z = D.mean(axis=1)
        info = {"method": "mean_distance"}
        return z, info

    if method.startswith("focal:"):
        focal_id = method.split(":", 1)[1]
        if focal_id not in ids:
            raise ValueError(f"focal id '{focal_id}' not found in ids.")
        k = ids.index(focal_id)
        z = D[:, k].copy()
        info = {"method": "focal_distance", "focal_id": focal_id}
        return z, info

    if method == "pcoa1":
        coords = pcoa(D, k=2)
        z = coords[:, 0]
        info = {"method": "pcoa", "axis": 1}
        return z, info

    raise ValueError("Unknown height method. Use: pcoa1 | mean | focal:<id>")


def pcoa(D: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Classical MDS / PCoA from a distance matrix.
    Returns an (N x k) coordinate matrix.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]

    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos = eigvals > 1e-12
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    if eigvals.size == 0:
        return np.zeros((n, k), dtype=float)

    m = min(k, eigvals.size)
    L = np.diag(np.sqrt(eigvals[:m]))
    V = eigvecs[:, :m]
    X = V @ L

    if m < k:
        X = np.hstack([X, np.zeros((n, k - m))])

    return X