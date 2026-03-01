from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer


def project_lonlat_auto_utm(lon: np.ndarray, lat: np.ndarray):
    """
    Project lon/lat (EPSG:4326) into an automatically selected local UTM zone (meters).
    Returns (x, y, info_dict).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    lon0 = float(np.nanmean(lon))
    lat0 = float(np.nanmean(lat))

    zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
    is_north = lat0 >= 0

    epsg = 32600 + zone if is_north else 32700 + zone
    crs_src = CRS.from_epsg(4326)
    crs_dst = CRS.from_epsg(epsg)

    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    x, y = transformer.transform(lon, lat)

    info = {
        "method": "auto_utm",
        "utm_zone": zone,
        "hemisphere": "north" if is_north else "south",
        "epsg": epsg,
    }
    return np.asarray(x, float), np.asarray(y, float), info