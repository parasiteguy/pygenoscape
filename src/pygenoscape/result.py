from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from .plotting import plot_surface_plotly


@dataclass
class LandscapeResult:
    ids: list[str]
    pops: list[str]
    lon: np.ndarray
    lat: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    D: np.ndarray
    GX: np.ndarray
    GY: np.ndarray
    GZ: np.ndarray
    metadata: dict

    def figure(self, title: str = "pygenoscape"):
        return plot_surface_plotly(
            self.GX, self.GY, self.GZ,
            self.x, self.y, self.z,
            self.ids, self.pops,
            title=title,
        )

    def to_html(self, path: str | Path, title: str | None = None):
        path = Path(path)
        fig = self.figure(title=title or path.stem)
        fig.write_html(str(path), include_plotlyjs="cdn")

    def save_grid(self, path: str | Path):
        path = Path(path)
        np.savez_compressed(path, GX=self.GX, GY=self.GY, GZ=self.GZ)

    def save_metadata(self, path: str | Path):
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)