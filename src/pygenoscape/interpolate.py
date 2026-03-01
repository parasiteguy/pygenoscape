from __future__ import annotations

import numpy as np
from scipy.interpolate import Rbf


def interpolate_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    method: str = "rbf",
    grid_size: int = 200,
    rbf_smooth: float = 0.1,
    idw_power: float = 2.0,
):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    gx = np.linspace(x.min(), x.max(), grid_size)
    gy = np.linspace(y.min(), y.max(), grid_size)
    GX, GY = np.meshgrid(gx, gy)

    method = method.lower().strip()
    if method == "rbf":
        rbf = Rbf(x, y, z, function="thin_plate", smooth=rbf_smooth)
        GZ = rbf(GX, GY)
        info = {"method": "rbf", "function": "thin_plate", "smooth": rbf_smooth, "grid_size": grid_size}
        return GX, GY, GZ, info

    if method == "idw":
        GZ = _idw(x, y, z, GX, GY, power=idw_power)
        info = {"method": "idw", "power": idw_power, "grid_size": grid_size}
        return GX, GY, GZ, info

    raise ValueError("Unknown interpolation method. Use: rbf | idw")


def _idw(x, y, z, GX, GY, power: float = 2.0):
    # Fully vectorized IDW
    xi = x.reshape(1, 1, -1)
    yi = y.reshape(1, 1, -1)
    zi = z.reshape(1, 1, -1)

    dx = GX[..., None] - xi
    dy = GY[..., None] - yi
    d2 = dx * dx + dy * dy

    # handle exact hits
    hit = d2 == 0.0
    if hit.any():
        out = np.empty(GX.shape, dtype=float)
        out[:] = np.nan

        hit_any = hit.any(axis=2)
        idx = hit.argmax(axis=2)
        out[hit_any] = z[idx[hit_any]]

        mask = ~hit_any
        d2m = d2[mask]
        wm = 1.0 / np.power(d2m, power / 2.0)
        out[mask] = (wm * zi.reshape(1, -1)).sum(axis=1) / wm.sum(axis=1)
        return out

    w = 1.0 / np.power(d2, power / 2.0)
    GZ = (w * zi).sum(axis=2) / w.sum(axis=2)
    return GZ