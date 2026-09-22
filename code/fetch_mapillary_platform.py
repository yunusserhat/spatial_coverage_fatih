#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yunus Serhat Bıçakçı
"""Snapshot all public Mapillary image locations inside the Fatih boundary.

Reads zoom 14 vector tiles of the public mly1_public tileset, which carry every
image point. Only the image identifier, location, capture time, compass angle,
panorama flag, and sequence identifier are kept. Contributor identifiers are
discarded on read. A Mapillary client token is read from MAPILLARY_ACCESS_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import mapbox_vector_tile
import pandas as pd
import requests

import analysis as A

TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
ZOOM = 14
KEEP = ["id", "captured_at", "compass_angle", "is_pano", "sequence_id"]


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def tile_pixel_to_lonlat(x: int, y: int, zoom: int, px: float, py: float, extent: int) -> tuple[float, float]:
    n = 2 ** zoom
    lon = (x + px / extent) / n * 360.0 - 180.0
    merc_y = math.pi * (1.0 - 2.0 * (y + py / extent) / n)
    lat = math.degrees(math.atan(math.sinh(merc_y)))
    return lon, lat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    package = args.package.resolve()
    token = os.environ["MAPILLARY_ACCESS_TOKEN"]

    boundary_geographic = A.read_boundary(package / "data" / "source" / "fatih_boundary_nominatim.json")[1]
    min_lon, min_lat, max_lon, max_lat = boundary_geographic.bounds
    x0, y1 = lonlat_to_tile(min_lon, min_lat, ZOOM)
    x1, y0 = lonlat_to_tile(max_lon, max_lat, ZOOM)

    rows = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            response = requests.get(TILE_URL.format(z=ZOOM, x=x, y=y), params={"access_token": token}, timeout=120)
            response.raise_for_status()
            decoded = mapbox_vector_tile.decode(response.content, default_options={"y_coord_down": True})
            layer = decoded.get("image")
            if not layer:
                continue
            extent = layer.get("extent", 4096)
            for feature in layer["features"]:
                px, py = feature["geometry"]["coordinates"]
                lon, lat = tile_pixel_to_lonlat(x, y, ZOOM, px, py, extent)
                properties = feature["properties"]
                rows.append({"lon": lon, "lat": lat, **{key: properties.get(key) for key in KEEP}})

    frame = pd.DataFrame(rows).drop_duplicates("id").rename(columns={"id": "image_id"})
    frame["image_id"] = frame["image_id"].astype(str)
    points = gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame["lon"], frame["lat"]), crs=A.WGS84)
    points = points.loc[points.within(boundary_geographic)].drop(columns="geometry")
    out = package / "data" / "source" / "mapillary_platform_images.parquet"
    pd.DataFrame(points).to_parquet(out, index=False)
    meta = {
        "tileset": "mly1_public",
        "zoom": ZOOM,
        "tiles": [x0, x1, y0, y1],
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records_in_boundary": int(len(points)),
        "fields": ["image_id", "lon", "lat"] + KEEP[1:],
        "note": "Contributor identifiers were not stored.",
    }
    A.json_dump(out.with_suffix(".provenance.json"), meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
