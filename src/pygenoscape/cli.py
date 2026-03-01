from __future__ import annotations

import argparse
from pathlib import Path

from .api import landscape


def main() -> None:
    p = argparse.ArgumentParser(
        prog="pygenoscape",
        description="Genetic landscape surfaces from FASTA+coords or distance-matrix+coords.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Generate a genetic landscape surface.")

    mx = run.add_mutually_exclusive_group(required=True)
    mx.add_argument("--fasta", help="Aligned FASTA (headers like Ind1_Pop1).")
    mx.add_argument("--dist", help="Precomputed square distance matrix CSV (row/col labels are ids).")

    run.add_argument("--coords", required=True, help="CSV with columns id,lon,lat.")
    run.add_argument("--out", required=True, help="Output HTML filename (Plotly).")

    run.add_argument("--height", default="pcoa1", help="pcoa1 | mean | focal:<id>")
    run.add_argument("--interp", default="rbf", help="rbf | idw")
    run.add_argument("--grid", type=int, default=200, help="Grid size (e.g., 200 => 200x200).")

    run.add_argument("--header-delim", default="_", help="Delimiter between id and pop in FASTA header.")
    run.add_argument("--header-split", default="last", choices=["last", "first"], help="Split on first or last delim.")

    run.add_argument("--rbf-smooth", type=float, default=0.1, help="RBF smoothing parameter.")
    run.add_argument("--idw-power", type=float, default=2.0, help="IDW power parameter.")

    args = p.parse_args()
    if args.cmd != "run":
        p.error("Unknown command")

    res = landscape(
        coords=args.coords,
        fasta=args.fasta,
        dist=args.dist,
        header_delim=args.header_delim,
        header_split=args.header_split,
        height=args.height,
        interp=args.interp,
        grid_size=args.grid,
        rbf_smooth=args.rbf_smooth,
        idw_power=args.idw_power,
    )

    out_html = Path(args.out)
    res.to_html(out_html)

    stem = out_html.with_suffix("")
    res.save_grid(stem.with_name(stem.name + "_grid.npz"))
    res.save_metadata(stem.with_name(stem.name + "_meta.json"))

    print(f"Wrote: {out_html}")
    print(f"Wrote: {stem.with_name(stem.name + '_grid.npz')}")
    print(f"Wrote: {stem.with_name(stem.name + '_meta.json')}")