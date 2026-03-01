from __future__ import annotations

from typing import List, Tuple


def parse_headers(headers: List[str], *, delim: str = "_", split: str = "last") -> Tuple[list[str], list[str]]:
    """
    Parse FASTA headers into (id, pop) pairs.

    Default behavior:
      - If delim present, split on last delim:  Ind1_Pop1 -> id=Ind1, pop=Pop1
      - If delim absent, pop='NA' and id=full header
    """
    ids: list[str] = []
    pops: list[str] = []

    for h in headers:
        h = h.strip()
        if delim not in h:
            ids.append(h)
            pops.append("NA")
            continue

        if split == "last":
            left, right = h.rsplit(delim, 1)
        else:
            left, right = h.split(delim, 1)

        ids.append(left)
        pops.append(right)

    return ids, pops