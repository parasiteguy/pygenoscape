from __future__ import annotations

from pathlib import Path

from .io import read_aligned_fasta, read_coords_csv, read_dist_matrix_csv
from .parse import parse_headers
from .distance import pdistance_matrix
from .height import compute_height
from .projection import project_lonlat_auto_utm
from .interpolate import interpolate_grid
from .result import LandscapeResult


def landscape(
    *,
    coords: str | Path,
    fasta: str | Path | None = None,
    dist: str | Path | None = None,
    header_delim: str = "_",
    header_split: str = "last",   # "last" or "first"
    distance: str = "p",          # used only for fasta mode
    height: str = "pcoa1",        # "pcoa1" | "mean" | "focal:<id>"
    interp: str = "rbf",          # "rbf" | "idw"
    grid_size: int = 200,
    rbf_smooth: float = 0.1,
    idw_power: float = 2.0,
) -> LandscapeResult:
    """
    Build a genetic landscape from either:
      - aligned FASTA (compute distances), or
      - precomputed distance matrix CSV.

    Exactly one of `fasta` or `dist` must be provided.
    """
    coords = Path(coords)

    if (fasta is None) == (dist is None):
        raise ValueError("Provide exactly one of `fasta` or `dist`.")

    # ---- Determine ids/pops and distance matrix D ----
    fasta_path = None
    dist_path = None

    if dist is not None:
        dist_path = Path(dist)
        Ddf = read_dist_matrix_csv(dist_path)
        ids = list(Ddf.index.astype(str))
        pops = ["NA"] * len(ids)
        Ddf = Ddf.loc[ids, ids]  # enforce same ordering
        D = Ddf.to_numpy(dtype=float)
        distance_info = {"method": "precomputed", "path": str(dist_path)}

    else:
        fasta_path = Path(fasta)
        headers, seqs = read_aligned_fasta(fasta_path)
        ids, pops = parse_headers(headers, delim=header_delim, split=header_split)

        if distance != "p":
            raise ValueError("v0.1 supports only distance='p' (p-distance) in FASTA mode.")

        D = pdistance_matrix(seqs)
        distance_info = {"method": "p-distance", "gap_ambig_policy": "ignore '-' and 'N'"}

    # ---- Read and align coordinates ----
    coord_df = read_coords_csv(coords)

    missing = sorted(set(ids) - set(coord_df.index))
    if missing:
        raise ValueError(
            f"Coords file missing {len(missing)} ids required by input "
            f"(showing up to 15): {missing[:15]}"
        )

    coord_df = coord_df.loc[ids]
    lon = coord_df["lon"].to_numpy()
    lat = coord_df["lat"].to_numpy()

    # ---- Project lon/lat to meters ----
    x, y, proj_info = project_lonlat_auto_utm(lon, lat)

    # ---- Compute per-sample height ----
    z, height_info = compute_height(D, ids=ids, method=height)

    # ---- Interpolate surface ----
    GX, GY, GZ, interp_info = interpolate_grid(
        x, y, z,
        method=interp,
        grid_size=grid_size,
        rbf_smooth=rbf_smooth,
        idw_power=idw_power,
    )

    return LandscapeResult(
        ids=ids,
        pops=pops,
        lon=lon,
        lat=lat,
        x=x,
        y=y,
        z=z,
        D=D,
        GX=GX,
        GY=GY,
        GZ=GZ,
        metadata={
            "input": {
                "fasta": str(fasta_path) if fasta_path else None,
                "dist": str(dist_path) if dist_path else None,
                "coords": str(coords),
                "header_delim": header_delim if fasta_path else None,
                "header_split": header_split if fasta_path else None,
            },
            "distance": distance_info,
            "height": height_info,
            "projection": proj_info,
            "interpolation": interp_info,
        },
    )