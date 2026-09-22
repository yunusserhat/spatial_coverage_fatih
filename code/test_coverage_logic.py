#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yunus Serhat Bıçakçı
"""Small runnable checks for the interval-union coverage rule."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import LineString
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
from analysis import (
    PROJECTED,
    build_network_context,
    canonical_line_key,
    coverage_from_matches,
    merge_intervals,
    unit_metrics,
    unit_metrics_from_atom_intervals,
)


def main() -> None:
    assert merge_intervals([(0, 10), (8, 15), (30, 35)]) == [(0.0, 15.0), (30.0, 35.0)]

    forward = LineString([(0, 0), (100, 0)])
    reverse = LineString([(100, 0), (0, 0)])
    assert canonical_line_key(forward, ("", "", "")) == canonical_line_key(reverse, ("", "", ""))

    streets = gpd.GeoDataFrame(
        {"edge_id": ["E00001"], "osm_id": ["1"]}, geometry=[forward], crs=PROJECTED
    )
    matches = pd.DataFrame(
        {"edge_index": [0, 0], "position_m": [10.0, 30.0], "edge_length_m": [100.0, 100.0]}
    )
    network_context = build_network_context(unary_union([forward]))
    coverage = coverage_from_matches(streets, network_context, matches, 15.0)
    assert abs(coverage["covered_length_m"] - 45.0) < 1e-9
    assert abs(coverage["uncovered_length_m"] - 55.0) < 1e-9
    atom_metrics = unit_metrics_from_atom_intervals(
        network_context, coverage["covered_atom_intervals"], 10.0
    )
    assert atom_metrics["covered_units"] == 5
    assert atom_metrics["total_units"] == 10

    horizontal = LineString([(0, 0), (100, 0)])
    vertical = LineString([(50, -50), (50, 50)])
    crossing_streets = gpd.GeoDataFrame(
        {"edge_id": ["E00001", "E00002"], "osm_id": ["1", "2"]},
        geometry=[horizontal, vertical],
        crs=PROJECTED,
    )
    crossing_context = build_network_context(unary_union([horizontal, vertical]))
    crossing_match = pd.DataFrame(
        {"edge_index": [0], "position_m": [50.0], "edge_length_m": [100.0]}
    )
    crossing_coverage = coverage_from_matches(
        crossing_streets, crossing_context, crossing_match, 15.0
    )
    assert abs(crossing_coverage["covered_length_m"] - 30.0) < 1e-9
    assert abs(crossing_coverage["uncovered_length_m"] - 170.0) < 1e-9

    partial = LineString([(0, 0), (5, 0)])
    metrics = unit_metrics(forward, partial, 10.0)
    assert metrics["total_units"] == 10
    assert metrics["covered_units"] == 1
    print("coverage logic checks passed")


if __name__ == "__main__":
    main()
