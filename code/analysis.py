#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yunus Serhat Bıçakçı
"""Reproducible spatial coverage analysis for the Fatih image manifest.

This script reads only the Hugging Face manifest plus cached boundary and OSM
network metadata. It does not read, download, or redistribute street images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import LineString, MultiLineString, distance, line_locate_point
from shapely.geometry import GeometryCollection, Point, box, shape
from shapely.ops import substring, unary_union
from shapely.strtree import STRtree


DATASET_ID = "yunusserhat/fatih"
DATASET_REVISION = "250dbb1efb580653b98d153dbc77341555d5a258"
MANIFEST_SHA256 = "75d8f360637e4f8a0411a162ec385d86569f772f95c7ada64d173c5597f75927"
MANIFEST_URL = (
    "https://huggingface.co/datasets/yunusserhat/fatih/resolve/"
    f"{DATASET_REVISION}/data/raw/manifest/train.parquet"
)
BOUNDARY_RELATION_ID = 1766104
WGS84 = "EPSG:4326"
PROJECTED = "EPSG:32635"
GRID_SIZE_M = 500.0
PRIMARY_MATCH_THRESHOLD_M = 20.0
PRIMARY_COVERAGE_DISTANCE_M = 25.0
UNIT_LENGTH_M = 10.0
NEAR_TIE_MARGIN_M = 2.0

ELIGIBLE_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "pedestrian",
    "road",
}
EXCLUDED_SERVICE_VALUES = {
    "driveway",
    "parking_aisle",
    "emergency_access",
    "drive-through",
    "parking",
}
EXCLUDED_ACCESS_VALUES = {"private", "no"}

MANIFEST_COLUMNS = [
    "image_id",
    "sequence_id",
    "lon",
    "lat",
    "computed_lon",
    "computed_lat",
    "captured_at",
    "captured_at_iso",
    "camera_type",
    "quality_score",
    "source_exists",
    "download_status",
    "error",
]


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_source(source: Path, destination: Path) -> None:
    """Copy a source cache into the package unless it is already that file."""
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def flatten_lines(geometry):
    """Yield all non-empty LineStrings in a possibly nested geometry."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from flatten_lines(part)


def canonical_line_key(line, vertical_key: tuple[str, str, str]) -> tuple:
    """Normalize orientation and millimetre-scale noise for exact duplicate removal."""
    coordinates = tuple((round(x, 2), round(y, 2)) for x, y in line.coords)
    reverse = tuple(reversed(coordinates))
    return vertical_key + (min(coordinates, reverse),)


def value_or_blank(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def safe_percent(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else 100.0 * numerator / denominator


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def fmt_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:,.{digits}f}"


def read_boundary(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature = payload[0] if isinstance(payload, list) else payload
    geometry = shape(feature["geojson"])
    if not geometry.is_valid:
        raise ValueError("The supplied Fatih boundary geometry is invalid.")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Expected a polygonal boundary, got {geometry.geom_type}.")
    projected = gpd.GeoSeries([geometry], crs=WGS84).to_crs(PROJECTED).iloc[0]
    return feature, geometry, projected


def read_manifest(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the necessary manifest fields and create explicit quality flags."""
    data = pd.read_parquet(path, columns=MANIFEST_COLUMNS)
    data.insert(0, "record_order", np.arange(len(data), dtype=np.int64))
    data["image_id"] = data["image_id"].astype("string")
    data["missing_image_id"] = data["image_id"].isna() | (data["image_id"].str.len() == 0)
    data["duplicate_image_id"] = data.duplicated("image_id", keep=False) & ~data["missing_image_id"]

    primary_valid = (
        data["lon"].notna()
        & data["lat"].notna()
        & data["lon"].between(-180, 180)
        & data["lat"].between(-90, 90)
    )
    computed_valid = (
        data["computed_lon"].notna()
        & data["computed_lat"].notna()
        & data["computed_lon"].between(-180, 180)
        & data["computed_lat"].between(-90, 90)
    )
    data["primary_coordinate_valid"] = primary_valid
    data["computed_coordinate_valid"] = computed_valid
    data["coordinate_source"] = np.select(
        [primary_valid, computed_valid], ["lon_lat", "computed_lon_lat"], default="none"
    )
    data["analysis_lon"] = np.where(primary_valid, data["lon"], data["computed_lon"])
    data["analysis_lat"] = np.where(primary_valid, data["lat"], data["computed_lat"])
    data["analysis_coordinate_valid"] = primary_valid | computed_valid
    data["capture_datetime"] = pd.to_datetime(data["captured_at_iso"], errors="coerce", utc=True)

    shared_coordinates = primary_valid & computed_valid
    displacement = np.full(len(data), np.nan)
    transformer = Transformer.from_crs(WGS84, PROJECTED, always_xy=True)
    raw_x, raw_y = transformer.transform(
        data.loc[shared_coordinates, "lon"].to_numpy(),
        data.loc[shared_coordinates, "lat"].to_numpy(),
    )
    computed_x, computed_y = transformer.transform(
        data.loc[shared_coordinates, "computed_lon"].to_numpy(),
        data.loc[shared_coordinates, "computed_lat"].to_numpy(),
    )
    displacement[shared_coordinates.to_numpy()] = np.hypot(
        np.asarray(raw_x) - np.asarray(computed_x),
        np.asarray(raw_y) - np.asarray(computed_y),
    )
    data["coordinate_disagreement_m"] = displacement

    # Exact repeated primary-coordinate locations are diagnostic only. They are
    # not treated as duplicate records because image identifiers remain distinct.
    coordinate_groups = data.loc[primary_valid].groupby(["lon", "lat"], dropna=False).size()
    data["repeat_primary_coordinate"] = False
    repeated_pairs = coordinate_groups[coordinate_groups > 1].index
    if len(repeated_pairs):
        data.loc[primary_valid, "repeat_primary_coordinate"] = pd.MultiIndex.from_frame(
            data.loc[primary_valid, ["lon", "lat"]]
        ).isin(repeated_pairs)

    # A deterministic one-row-per-identifier dataset for downstream geography.
    deduplicated = (
        data.loc[~data["missing_image_id"]]
        .sort_values("record_order")
        .drop_duplicates("image_id", keep="first")
        .copy()
    )
    report = {
        "manifest_records": int(len(data)),
        "manifest_unique_image_ids": int(data.loc[~data["missing_image_id"], "image_id"].nunique()),
        "manifest_missing_image_ids": int(data["missing_image_id"].sum()),
        "manifest_duplicate_identifier_records": int(data.duplicated("image_id", keep="first").sum()),
        "primary_coordinate_missing_or_invalid": int((~primary_valid).sum()),
        "computed_coordinate_missing_or_invalid": int((~computed_valid).sum()),
        "coordinate_fallback_records": int((~primary_valid & computed_valid).sum()),
        "records_without_usable_coordinates": int((~(primary_valid | computed_valid)).sum()),
        "unique_primary_coordinate_pairs": int(len(coordinate_groups)),
        "repeated_primary_coordinate_groups": int((coordinate_groups > 1).sum()),
        "records_at_repeated_primary_coordinates": int(coordinate_groups[coordinate_groups > 1].sum()),
        "maximum_records_at_one_primary_coordinate": int(coordinate_groups.max()),
        "coordinate_disagreement_median_m": float(np.nanmedian(displacement)),
        "coordinate_disagreement_p95_m": float(np.nanpercentile(displacement, 95)),
        "coordinate_disagreement_max_m": float(np.nanmax(displacement)),
        "coordinate_disagreement_gt_20m": int((displacement > 20).sum()),
        "capture_time_missing": int(data["capture_datetime"].isna().sum()),
        "capture_time_min_utc": data["capture_datetime"].min().isoformat(),
        "capture_time_max_utc": data["capture_datetime"].max().isoformat(),
        "capture_date_count": int(data["capture_datetime"].dt.date.nunique()),
        "sequence_count": int(data["sequence_id"].nunique(dropna=True)),
        "source_exists_true": int(data["source_exists"].eq(True).sum()),
        "downloaded_status_records": int(data["download_status"].eq("downloaded").sum()),
    }
    return deduplicated, report


def attach_existing_quality_labels(data: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Join the existing GeoAI v1 usability field without creating new labels."""
    columns = [
        "image_id",
        "usable",
        "parsed_quality_usable",
        "quality_issues_json",
        "quality_parse_status",
        "source_part",
    ]
    labels = pd.read_parquet(path, columns=columns)
    labels["image_id"] = labels["image_id"].astype("string")
    if labels["image_id"].isna().any() or labels["image_id"].duplicated().any():
        raise AssertionError("Existing quality-label file must have one non-missing row per image_id.")
    merged = data.merge(labels, on="image_id", how="left", validate="one_to_one")
    merged["quality_status"] = np.select(
        [merged["usable"].eq(True), merged["usable"].eq(False)],
        ["confirmed_usable", "explicitly_excluded"],
        default="unknown",
    )
    merged["analysis_quality_included"] = merged["quality_status"].eq("confirmed_usable")
    merged["quality_issues_json"] = merged["quality_issues_json"].fillna("")
    merged["quality_parse_status"] = merged["quality_parse_status"].fillna("no_quality_label")
    merged["quality_label_source"] = np.where(
        merged["quality_status"].eq("unknown"), "", "descriptions/geoai/v1"
    )
    exclusion_sets = merged.loc[merged["quality_status"].eq("explicitly_excluded"), "quality_issues_json"]
    exclusion_combinations = (
        exclusion_sets.value_counts(dropna=False)
        .rename_axis("documented_issue_set")
        .reset_index(name="excluded_images")
        .sort_values(["excluded_images", "documented_issue_set"], ascending=[False, True])
        .reset_index(drop=True)
    )
    report = {
        "quality_label_layer": "descriptions/geoai/v1",
        "quality_label_records": int(len(labels)),
        "quality_label_unique_image_ids": int(labels["image_id"].nunique()),
        "confirmed_usable_records": int(merged["quality_status"].eq("confirmed_usable").sum()),
        "explicitly_excluded_records": int(merged["quality_status"].eq("explicitly_excluded").sum()),
        "quality_unknown_records": int(merged["quality_status"].eq("unknown").sum()),
        "direct_and_nested_usable_disagreement": int(
            (
                merged["usable"].notna()
                & merged["parsed_quality_usable"].notna()
                & merged["usable"].ne(merged["parsed_quality_usable"])
            ).sum()
        ),
        "quality_parse_status_counts": {
            key: int(value)
            for key, value in merged["quality_parse_status"].value_counts(dropna=False).items()
        },
        "explicit_exclusion_reason_set_count": int(len(exclusion_combinations)),
    }
    if (
        report["confirmed_usable_records"]
        + report["explicitly_excluded_records"]
        + report["quality_unknown_records"]
        != len(merged)
    ):
        raise AssertionError("Quality status categories do not reconcile with the manifest.")
    if int(exclusion_combinations["excluded_images"].sum()) != report["explicitly_excluded_records"]:
        raise AssertionError("Exclusive documented quality issue sets do not reconcile with exclusions.")
    return merged, report, exclusion_combinations


def make_image_points(data: pd.DataFrame, boundary_projected):
    usable = data.loc[data["analysis_coordinate_valid"]].copy().reset_index(drop=True)
    geographic = gpd.GeoDataFrame(
        usable,
        geometry=gpd.points_from_xy(usable["analysis_lon"], usable["analysis_lat"]),
        crs=WGS84,
    )
    points = geographic.to_crs(PROJECTED)
    points["inside_boundary"] = [boundary_projected.covers(point) for point in points.geometry]
    return points


def read_eligible_streets(path: Path, boundary_projected):
    """Parse, filter, clip, and deduplicate raw Overpass highway ways."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    exclusions = Counter()
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        highway = value_or_blank(tags.get("highway")).lower()
        access = value_or_blank(tags.get("access")).lower()
        service = value_or_blank(tags.get("service")).lower()
        if highway not in ELIGIBLE_HIGHWAYS:
            exclusions["not_in_highway_filter"] += 1
            continue
        if access in EXCLUDED_ACCESS_VALUES:
            exclusions[f"access_{access}"] += 1
            continue
        if highway == "service" and service in EXCLUDED_SERVICE_VALUES:
            exclusions[f"service_{service}"] += 1
            continue
        if value_or_blank(tags.get("area")).lower() == "yes":
            exclusions["area_way"] += 1
            continue
        coordinates = [(node["lon"], node["lat"]) for node in element.get("geometry", [])]
        if len(coordinates) < 2:
            exclusions["fewer_than_two_nodes"] += 1
            continue
        rows.append(
            {
                "osm_id": str(element["id"]),
                "highway": highway,
                "name": value_or_blank(tags.get("name")),
                "access": access,
                "service": service,
                "bridge": value_or_blank(tags.get("bridge")),
                "tunnel": value_or_blank(tags.get("tunnel")),
                "layer": value_or_blank(tags.get("layer")),
                "oneway": value_or_blank(tags.get("oneway")),
                "geometry": LineString(coordinates),
            }
        )
    raw = gpd.GeoDataFrame(rows, crs=WGS84).to_crs(PROJECTED)
    clipped_rows = []
    short_parts = 0
    for row in raw.itertuples(index=False):
        clipped = row.geometry.intersection(boundary_projected)
        for part_number, line in enumerate(flatten_lines(clipped), start=1):
            if line.length < 0.1:
                short_parts += 1
                continue
            vertical_key = (row.layer, row.bridge, row.tunnel)
            clipped_rows.append(
                {
                    "osm_id": row.osm_id,
                    "part_number": part_number,
                    "highway": row.highway,
                    "name": row.name,
                    "access": row.access,
                    "service": row.service,
                    "bridge": row.bridge,
                    "tunnel": row.tunnel,
                    "layer": row.layer,
                    "oneway": row.oneway,
                    "geometry_key": canonical_line_key(line, vertical_key),
                    "geometry": line,
                }
            )
    clipped = gpd.GeoDataFrame(clipped_rows, crs=PROJECTED)
    clipped = clipped.sort_values(["osm_id", "part_number"]).reset_index(drop=True)
    before_duplicate_removal = len(clipped)
    streets = clipped.drop_duplicates("geometry_key", keep="first").copy().reset_index(drop=True)
    streets["edge_id"] = [f"E{i:05d}" for i in range(1, len(streets) + 1)]
    streets["length_m"] = streets.geometry.length
    report = {
        "overpass_way_records": int(len(payload.get("elements", []))),
        "eligible_way_records_before_clip": int(len(raw)),
        "excluded_way_records": dict(sorted(exclusions.items())),
        "short_clipped_parts_excluded": int(short_parts),
        "clipped_line_parts_before_duplicate_removal": int(before_duplicate_removal),
        "exact_duplicate_line_parts_removed": int(before_duplicate_removal - len(streets)),
        "eligible_line_parts": int(len(streets)),
        "eligible_highway_counts": {
            key: int(value) for key, value in streets["highway"].value_counts().sort_index().items()
        },
        "sum_of_canonical_line_lengths_m": float(streets["length_m"].sum()),
    }
    return streets, report


def nearest_matches(points: gpd.GeoDataFrame, streets: gpd.GeoDataFrame, max_distance_m: float):
    """Assign each point to a nearest eligible line with a stable tie break."""
    point_geometries = points.geometry.to_numpy()
    street_geometries = streets.geometry.to_numpy()
    tree = STRtree(street_geometries)
    indices, distances = tree.query_nearest(
        point_geometries,
        max_distance=max_distance_m,
        return_distance=True,
        all_matches=True,
    )
    selected: dict[int, tuple[int, float]] = {}
    edge_ids = streets["edge_id"].to_numpy()
    for point_index, street_index, match_distance in zip(indices[0], indices[1], distances):
        candidate = (int(street_index), float(match_distance))
        prior = selected.get(int(point_index))
        if prior is None or candidate[1] < prior[1] - 1e-9 or (
            abs(candidate[1] - prior[1]) <= 1e-9 and edge_ids[candidate[0]] < edge_ids[prior[0]]
        ):
            selected[int(point_index)] = candidate
    rows = []
    for point_index in sorted(selected):
        street_index, match_distance = selected[point_index]
        line = street_geometries[street_index]
        point = point_geometries[point_index]
        position = float(line_locate_point(line, point))
        rows.append(
            {
                "point_index": point_index,
                "image_id": str(points.iloc[point_index]["image_id"]),
                "edge_index": street_index,
                "edge_id": streets.iloc[street_index]["edge_id"],
                "osm_id": streets.iloc[street_index]["osm_id"],
                "highway": streets.iloc[street_index]["highway"],
                "street_name": streets.iloc[street_index]["name"],
                "nearest_distance_m": match_distance,
                "position_m": position,
                "edge_length_m": float(line.length),
                "distance_to_way_endpoint_m": float(min(position, line.length - position)),
            }
        )
    matches = pd.DataFrame(rows)
    return matches, tree


def ambiguity_diagnostics(
    points: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    tree: STRtree,
    threshold_m: float,
    margin_m: float,
):
    """Record near alternatives so that intersection and parallel-road cases are visible."""
    point_geometries = points.geometry.to_numpy()
    street_geometries = streets.geometry.to_numpy()
    pairs = tree.query(point_geometries, predicate="dwithin", distance=threshold_m)
    if pairs.shape[1] == 0:
        matches["candidate_count"] = 0
        matches["second_distance_m"] = np.nan
        matches["distance_margin_m"] = np.nan
        matches["near_tie"] = False
        matches["endpoint_proximate"] = False
        return matches, pd.DataFrame(), {"ambiguous_match_count": 0}

    candidate_distances = distance(point_geometries[pairs[0]], street_geometries[pairs[1]])
    candidates = pd.DataFrame(
        {
            "point_index": pairs[0].astype(int),
            "edge_index": pairs[1].astype(int),
            "candidate_distance_m": candidate_distances.astype(float),
        }
    )
    candidates["edge_id"] = streets.iloc[candidates["edge_index"].to_numpy()]["edge_id"].to_numpy()
    candidates = candidates.sort_values(["point_index", "candidate_distance_m", "edge_id"])
    match_lookup = matches.set_index("point_index")["nearest_distance_m"].to_dict()
    diagnostic_rows = []
    candidate_counts = {}
    second_distances = {}
    for point_index, group in candidates.groupby("point_index", sort=False):
        group = group.drop_duplicates("edge_index", keep="first")
        candidate_counts[int(point_index)] = int(len(group))
        nearest = match_lookup.get(int(point_index))
        eligible = group.loc[group["candidate_distance_m"] <= nearest + margin_m + 1e-9]
        if len(eligible) < 2:
            second_distances[int(point_index)] = np.nan
            continue
        first, second = eligible.iloc[0], eligible.iloc[1]
        second_distances[int(point_index)] = float(second["candidate_distance_m"])
        first_line = street_geometries[int(first["edge_index"])]
        second_line = street_geometries[int(second["edge_index"])]
        first_street = streets.iloc[int(first["edge_index"])]
        second_street = streets.iloc[int(second["edge_index"])]
        if bool(first_line.intersects(second_line)):
            relationship = "intersecting candidate streets"
        elif first_line.distance(second_line) <= margin_m:
            relationship = "closely spaced non-intersecting streets"
        else:
            relationship = "separate candidate streets"
        bridge_or_layer = any(
            value_or_blank(first_street[field]) or value_or_blank(second_street[field])
            for field in ["bridge", "layer", "tunnel"]
        )
        diagnostic_rows.append(
            {
                "point_index": int(point_index),
                "image_id": str(points.iloc[int(point_index)]["image_id"]),
                "nearest_edge_id": first_street["edge_id"],
                "nearest_osm_id": first_street["osm_id"],
                "nearest_highway": first_street["highway"],
                "nearest_name": first_street["name"],
                "nearest_distance_m": float(first["candidate_distance_m"]),
                "second_edge_id": second_street["edge_id"],
                "second_osm_id": second_street["osm_id"],
                "second_highway": second_street["highway"],
                "second_name": second_street["name"],
                "second_distance_m": float(second["candidate_distance_m"]),
                "distance_margin_m": float(second["candidate_distance_m"] - first["candidate_distance_m"]),
                "candidate_count_within_threshold": int(len(group)),
                "candidate_relationship": relationship,
                "bridge_layer_or_tunnel_tag_present": bool(bridge_or_layer),
                "lon": float(points.iloc[int(point_index)]["analysis_lon"]),
                "lat": float(points.iloc[int(point_index)]["analysis_lat"]),
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    matches = matches.copy()
    matches["candidate_count"] = matches["point_index"].map(candidate_counts).fillna(0).astype(int)
    matches["second_distance_m"] = matches["point_index"].map(second_distances)
    matches["distance_margin_m"] = matches["second_distance_m"] - matches["nearest_distance_m"]
    matches["near_tie"] = matches["distance_margin_m"].le(margin_m).fillna(False)
    matches["endpoint_proximate"] = matches["distance_to_way_endpoint_m"].le(5.0)
    report = {
        "ambiguity_threshold_m": threshold_m,
        "ambiguity_margin_m": margin_m,
        "ambiguous_match_count": int(matches["near_tie"].sum()),
        "endpoint_proximate_match_count": int(matches["endpoint_proximate"].sum()),
        "ambiguous_relationship_counts": {
            key: int(value)
            for key, value in diagnostics["candidate_relationship"].value_counts().sort_index().items()
        }
        if len(diagnostics)
        else {},
        "ambiguous_with_bridge_layer_or_tunnel_tag": int(
            diagnostics["bridge_layer_or_tunnel_tag_present"].sum()
        )
        if len(diagnostics)
        else 0,
    }
    return matches, diagnostics, report


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    result = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1] + 1e-8:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return [(float(start), float(end)) for start, end in result]


def build_network_context(network_geometry):
    """Create a non-overlapping, noded set of network atoms for exact accounting."""
    atoms = list(flatten_lines(network_geometry))
    if not atoms:
        raise AssertionError("Eligible network has no line atoms.")
    atom_array = np.asarray(atoms, dtype=object)
    lengths = np.asarray([line.length for line in atoms], dtype=float)
    return {
        "atoms": atom_array,
        "lengths_m": lengths,
        "tree": STRtree(atom_array),
        "total_length_m": float(lengths.sum()),
    }


def coverage_from_matches(
    streets: gpd.GeoDataFrame,
    network_context: dict,
    matches: pd.DataFrame,
    local_distance_m: float,
):
    """Map local edge intervals to disjoint network atoms before measuring length.

    The noded atoms have no overlapping interiors. Coverage and gaps are built
    as complementary intervals on each atom, which avoids unstable global
    line-difference operations and guarantees exact length reconciliation.
    """
    source_intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in matches[["edge_index", "position_m", "edge_length_m"]].itertuples(index=False):
        start = max(0.0, float(row.position_m) - local_distance_m)
        end = min(float(row.edge_length_m), float(row.position_m) + local_distance_m)
        if end > start:
            source_intervals[int(row.edge_index)].append((start, end))

    source_rows = []
    for edge_index, edge_intervals in source_intervals.items():
        line = streets.iloc[edge_index].geometry
        for start, end in merge_intervals(edge_intervals):
            geometry = substring(line, start, end)
            if geometry.length > 0:
                source_rows.append(
                    {
                        "edge_id": streets.iloc[edge_index]["edge_id"],
                        "osm_id": streets.iloc[edge_index]["osm_id"],
                        "start_m": start,
                        "end_m": end,
                        "length_m": float(geometry.length),
                        "geometry": geometry,
                    }
                )
    source_sections = gpd.GeoDataFrame(source_rows, geometry="geometry", crs=PROJECTED)

    atoms = network_context["atoms"]
    atom_intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    if len(source_sections):
        source_geometries = source_sections.geometry.to_numpy()
        pairs = network_context["tree"].query(source_geometries, predicate="intersects")
        for source_index, atom_index in zip(pairs[0], pairs[1]):
            atom = atoms[int(atom_index)]
            intersection_geometry = source_geometries[int(source_index)].intersection(atom)
            for part in flatten_lines(intersection_geometry):
                coordinates = list(part.coords)
                start = float(atom.project(Point(coordinates[0])))
                end = float(atom.project(Point(coordinates[-1])))
                if abs(end - start) > 1e-8:
                    atom_intervals[int(atom_index)].append((min(start, end), max(start, end)))

    covered_rows = []
    uncovered_parts = []
    covered_atom_intervals: dict[int, list[tuple[float, float]]] = {}
    for atom_index, atom in enumerate(atoms):
        merged = merge_intervals(atom_intervals.get(atom_index, []))
        covered_atom_intervals[atom_index] = merged
        prior_end = 0.0
        for start, end in merged:
            covered_part = substring(atom, start, end)
            if covered_part.length > 0:
                covered_rows.append(
                    {
                        "atom_id": f"A{atom_index + 1:05d}",
                        "start_m": start,
                        "end_m": end,
                        "length_m": float(covered_part.length),
                        "geometry": covered_part,
                    }
                )
            if start > prior_end + 1e-8:
                gap = substring(atom, prior_end, start)
                if gap.length > 0:
                    uncovered_parts.append(gap)
            prior_end = max(prior_end, end)
        if prior_end < atom.length - 1e-8:
            gap = substring(atom, prior_end, atom.length)
            if gap.length > 0:
                uncovered_parts.append(gap)

    sections = gpd.GeoDataFrame(covered_rows, geometry="geometry", crs=PROJECTED)
    covered_parts = list(sections.geometry) if len(sections) else []
    covered_geometry = GeometryCollection(covered_parts)
    uncovered_geometry = GeometryCollection(uncovered_parts)
    total_length = float(network_context["total_length_m"])
    covered_length = float(sum(part.length for part in covered_parts))
    uncovered_length = float(sum(part.length for part in uncovered_parts))
    residual = total_length - covered_length - uncovered_length
    if abs(residual) > 0.05:
        raise AssertionError(
            "Atom-based covered and uncovered lengths do not reconcile. "
            f"total={total_length:.6f}, covered={covered_length:.6f}, "
            f"uncovered={uncovered_length:.6f}, residual={residual:.6f}"
        )
    return {
        "sections": sections,
        "covered_geometry": covered_geometry,
        "uncovered_geometry": uncovered_geometry,
        "covered_length_m": covered_length,
        "uncovered_length_m": uncovered_length,
        "coverage_pct": safe_percent(covered_length, total_length),
        "interval_edge_count": int(len(source_intervals)),
        "interval_count": int(len(source_sections)),
        "covered_atom_intervals": covered_atom_intervals,
    }


def unit_metrics(network_geometry, covered_geometry, unit_length_m: float):
    """Split the geometric network into fixed units and classify units >=50% covered."""
    total_units = 0
    covered_units = 0
    total_unit_length = 0.0
    for line in flatten_lines(network_geometry):
        for start in np.arange(0.0, line.length, unit_length_m):
            end = min(float(start + unit_length_m), float(line.length))
            unit = substring(line, float(start), end)
            if unit.length <= 0:
                continue
            fraction = unit.intersection(covered_geometry).length / unit.length
            total_units += 1
            total_unit_length += unit.length
            if fraction >= 0.5:
                covered_units += 1
    return {
        "unit_length_m": unit_length_m,
        "total_units": total_units,
        "covered_units": covered_units,
        "covered_unit_pct": safe_percent(covered_units, total_units),
        "unit_length_sum_m": total_unit_length,
    }


def unit_metrics_from_atom_intervals(network_context: dict, atom_intervals: dict, unit_length_m: float):
    """Classify fixed units from the same exact atom intervals used for coverage."""
    total_units = 0
    covered_units = 0
    total_unit_length = 0.0
    for atom_index, atom in enumerate(network_context["atoms"]):
        intervals = atom_intervals.get(atom_index, [])
        active_interval = 0
        for start in np.arange(0.0, atom.length, unit_length_m):
            end = min(float(start + unit_length_m), float(atom.length))
            while (
                active_interval < len(intervals)
                and intervals[active_interval][1] <= start + 1e-9
            ):
                active_interval += 1
            covered_in_unit = 0.0
            interval_index = active_interval
            while interval_index < len(intervals) and intervals[interval_index][0] < end - 1e-9:
                interval_start, interval_end = intervals[interval_index]
                covered_in_unit += max(0.0, min(end, interval_end) - max(float(start), interval_start))
                interval_index += 1
            unit_length = end - float(start)
            total_units += 1
            total_unit_length += unit_length
            if covered_in_unit / unit_length >= 0.5 - 1e-9:
                covered_units += 1
    return {
        "unit_length_m": unit_length_m,
        "total_units": total_units,
        "covered_units": covered_units,
        "covered_unit_pct": safe_percent(covered_units, total_units),
        "unit_length_sum_m": total_unit_length,
    }


def build_grid(boundary_projected, size_m: float):
    min_x, min_y, max_x, max_y = boundary_projected.bounds
    x_start = math.floor(min_x / size_m) * size_m
    y_start = math.floor(min_y / size_m) * size_m
    rows = []
    row_number = 0
    y = y_start
    while y < max_y:
        column_number = 0
        x = x_start
        while x < max_x:
            clipped = box(x, y, x + size_m, y + size_m).intersection(boundary_projected)
            if not clipped.is_empty and clipped.area > 0:
                rows.append(
                    {
                        "grid_id": f"G{row_number + 1:02d}{column_number + 1:02d}",
                        "grid_row": row_number + 1,
                        "grid_column": column_number + 1,
                        "cell_area_m2": float(clipped.area),
                        "geometry": clipped,
                    }
                )
            column_number += 1
            x += size_m
        row_number += 1
        y += size_m
    grid = gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJECTED)
    grid["cell_area_km2"] = grid["cell_area_m2"] / 1_000_000.0
    return grid


def line_lengths_by_grid(lines, grid: gpd.GeoDataFrame) -> np.ndarray:
    """Allocate clipped line length by geometry, rather than by line centroids."""
    values = np.zeros(len(grid), dtype=float)
    if not lines:
        return values
    tree = STRtree(grid.geometry.to_numpy())
    pairs = tree.query(np.asarray(lines, dtype=object), predicate="intersects")
    for line_index, grid_index in zip(pairs[0], pairs[1]):
        piece = lines[int(line_index)].intersection(grid.geometry.iloc[int(grid_index)])
        values[int(grid_index)] += float(piece.length)
    return values


def make_grid_results(points, network_geometry, covered_geometry, grid_size_m: float, boundary_projected):
    grid = build_grid(boundary_projected, grid_size_m)
    joined = gpd.sjoin(
        points[["image_id", "geometry"]], grid[["grid_id", "geometry"]], how="left", predicate="within"
    )
    # GPS points exactly on a grid boundary are rare. Keep one deterministic cell if one occurs.
    joined = joined.sort_values(["image_id", "grid_id"]).drop_duplicates("image_id", keep="first")
    image_counts = joined["grid_id"].value_counts()
    grid["image_count"] = grid["grid_id"].map(image_counts).fillna(0).astype(int)

    network_lines = list(flatten_lines(network_geometry))
    covered_lines = list(flatten_lines(covered_geometry))
    grid["eligible_street_length_m"] = line_lengths_by_grid(network_lines, grid)
    grid["covered_street_length_m"] = line_lengths_by_grid(covered_lines, grid)
    grid["uncovered_street_length_m"] = (
        grid["eligible_street_length_m"] - grid["covered_street_length_m"]
    ).clip(lower=0.0)
    grid["street_coverage_pct"] = np.where(
        grid["eligible_street_length_m"] > 0,
        100.0 * grid["covered_street_length_m"] / grid["eligible_street_length_m"],
        np.nan,
    )
    grid["image_density_per_km2"] = grid["image_count"] / grid["cell_area_km2"]
    grid["images_per_km_street"] = np.where(
        grid["eligible_street_length_m"] > 0,
        grid["image_count"] / (grid["eligible_street_length_m"] / 1000.0),
        np.nan,
    )
    grid["uncovered_street_km_per_km2"] = (
        grid["uncovered_street_length_m"] / 1000.0 / grid["cell_area_km2"]
    )
    return grid


def map_extent(boundary_projected, margin_m: float = 280.0):
    min_x, min_y, max_x, max_y = boundary_projected.bounds
    return (min_x - margin_m, max_x + margin_m, min_y - margin_m, max_y + margin_m)


def base_map(ax, boundary_projected, extent):
    ax.set_facecolor("#FAFAF8")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    gpd.GeoSeries([boundary_projected], crs=PROJECTED).boundary.plot(
        ax=ax, color="#303030", linewidth=0.85, zorder=8
    )


def add_north_arrow(ax):
    ax.annotate(
        "",
        xy=(0.075, 0.93),
        xytext=(0.075, 0.855),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-|>", "color": "#303030", "lw": 1.0},
        zorder=20,
    )
    ax.text(
        0.075, 0.95, "N", transform=ax.transAxes, ha="center", va="center",
        fontsize=8, fontweight="bold", color="#303030", zorder=20,
    )


def add_scale_bar(ax, extent, length_m: float = 1000.0):
    x0 = (extent[0] + extent[1] - length_m) / 2.0
    y0 = extent[2] + 0.075 * (extent[3] - extent[2])
    half = length_m / 2.0
    height = 45.0
    for index, color in enumerate(["#303030", "#FFFFFF"]):
        rectangle = plt.Rectangle(
            (x0 + index * half, y0), half, height, facecolor=color, edgecolor="#303030", linewidth=0.55, zorder=20
        )
        ax.add_patch(rectangle)
    ax.text(x0, y0 - 65, "0", ha="center", va="top", fontsize=7, color="#303030")
    ax.text(x0 + half, y0 - 65, "0.5", ha="center", va="top", fontsize=7, color="#303030")
    ax.text(x0 + length_m, y0 - 65, "1 km", ha="center", va="top", fontsize=7, color="#303030")


def save_figure(figure, directory: Path, stem: str):
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor=figure.get_facecolor())
    figure.savefig(directory / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def generate_figures(
    figures_dir: Path,
    boundary_projected,
    network_geometry,
    covered_geometry,
    uncovered_geometry,
    grid: gpd.GeoDataFrame,
):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "figure.facecolor": "#FAFAF8",
        }
    )
    extent = map_extent(boundary_projected)
    network_lines = gpd.GeoSeries(list(flatten_lines(network_geometry)), crs=PROJECTED)
    covered_lines = gpd.GeoSeries(list(flatten_lines(covered_geometry)), crs=PROJECTED)
    uncovered_lines = gpd.GeoSeries(list(flatten_lines(uncovered_geometry)), crs=PROJECTED)

    # Figure 1
    figure, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    grid.plot(
        ax=ax,
        column="image_density_per_km2",
        cmap="cividis",
        linewidth=0.15,
        edgecolor="#F4F4F1",
        legend=True,
        legend_kwds={
            "label": "Image locations per km²", "orientation": "horizontal",
            "shrink": 0.68, "pad": 0.08,
        },
        zorder=3,
    )
    network_lines.plot(ax=ax, color="#D7D7D4", linewidth=0.24, zorder=4)
    base_map(ax, boundary_projected, extent)
    add_north_arrow(ax)
    add_scale_bar(ax, extent)
    save_figure(figure, figures_dir, "fig1_image_distribution")

    # Figure 2
    figure, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    network_lines.plot(ax=ax, color="#B8B8B8", linewidth=0.62, zorder=2)
    covered_lines.plot(ax=ax, color="#007C83", linewidth=0.85, zorder=4)
    base_map(ax, boundary_projected, extent)
    add_north_arrow(ax)
    add_scale_bar(ax, extent)
    legend_handles = [
        mpl.lines.Line2D([0], [0], color="#007C83", lw=2.2, label="Covered, local 25 m rule"),
        mpl.lines.Line2D([0], [0], color="#B8B8B8", lw=2.2, label="Uncovered"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=7,
    )
    save_figure(figure, figures_dir, "fig2_network_coverage")

    # Figure 3
    figure, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    has_network = grid.loc[grid["eligible_street_length_m"] > 0]
    no_network = grid.loc[grid["eligible_street_length_m"] <= 0]
    if len(no_network):
        no_network.plot(ax=ax, facecolor="#F6F6F4", edgecolor="#D7D7D4", linewidth=0.2, zorder=2)
    has_network.plot(
        ax=ax,
        column="street_coverage_pct",
        cmap="cividis",
        vmin=0,
        vmax=100,
        linewidth=0.2,
        edgecolor="#F4F4F1",
        legend=True,
        legend_kwds={
            "label": "Eligible street length covered (%)", "orientation": "horizontal",
            "shrink": 0.68, "pad": 0.08,
        },
        zorder=3,
    )
    network_lines.plot(ax=ax, color="#E2E2DF", linewidth=0.18, zorder=4)
    base_map(ax, boundary_projected, extent)
    add_north_arrow(ax)
    add_scale_bar(ax, extent)
    save_figure(figure, figures_dir, "fig3_grid_coverage")

    # Figure 4
    figure, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    has_network = grid.loc[grid["eligible_street_length_m"] > 0]
    no_network = grid.loc[grid["eligible_street_length_m"] <= 0]
    if len(no_network):
        no_network.plot(
            ax=ax, facecolor="#F6F6F4", edgecolor="#D7D7D4", linewidth=0.18, zorder=2
        )
    has_network.plot(
        ax=ax,
        column="uncovered_street_km_per_km2",
        cmap="magma",
        linewidth=0.18,
        edgecolor="#F4F4F1",
        legend=True,
        legend_kwds={
            "label": "Uncovered eligible street km per km²", "orientation": "horizontal",
            "shrink": 0.68, "pad": 0.08,
        },
        zorder=3,
    )
    uncovered_lines.plot(ax=ax, color="#3B1A20", linewidth=0.34, alpha=0.42, zorder=4)
    base_map(ax, boundary_projected, extent)
    add_north_arrow(ax)
    add_scale_bar(ax, extent)
    save_figure(figure, figures_dir, "fig4_gap_density")


def write_latex_table(path: Path, caption: str, columns: list[str], rows: list[list[str]], widths: str):
    header = " & ".join(tex_escape(column) for column in columns) + " \\\\" + "\n\\hline"
    body = "\n".join(" & ".join(cell for cell in row) + r" \\" for row in rows)
    content = textwrap.dedent(
        f"""\
        \\begin{{table}}[htbp]
        \\centering
        \\caption{{{tex_escape(caption)}}}
        \\small
        \\renewcommand{{\\arraystretch}}{{1.13}}
        \\begin{{tabular}}{{{widths}}}
        \\hline
        {header}
        {body}
        \\hline
        \\end{{tabular}}
        \\end{{table}}
        """
    )
    path.write_text(content, encoding="utf-8")


def csv_rows(path: Path, rows: list[dict]):
    pd.DataFrame(rows).to_csv(path, index=False)


def grid_summary(grid: gpd.GeoDataFrame):
    eligible = grid.loc[grid["eligible_street_length_m"] > 0].copy()
    statistics = []
    for label, quantile in [
        ("Minimum", 0.0),
        ("25th percentile", 0.25),
        ("Median", 0.5),
        ("75th percentile", 0.75),
        ("Maximum", 1.0),
    ]:
        statistics.append(
            {
                "statistic": label,
                "image_count": float(eligible["image_count"].quantile(quantile)),
                "image_density_per_km2": float(eligible["image_density_per_km2"].quantile(quantile)),
                "eligible_street_length_km": float(
                    (eligible["eligible_street_length_m"] / 1000.0).quantile(quantile)
                ),
                "street_coverage_pct": float(eligible["street_coverage_pct"].quantile(quantile)),
                "uncovered_street_km_per_km2": float(
                    eligible["uncovered_street_km_per_km2"].quantile(quantile)
                ),
            }
        )
    return statistics


def write_package_documents(out: Path, results: dict, provenance: dict, write_docs: bool = False):
    tables = out / "tables"
    figures = out / "figures"
    quality = results["quality"]
    network = results["network"]
    primary = results["primary"]
    sensitivity = results["sensitivity"]
    grid_stats = results["grid_summary"]
    comparison = results["original_filtered_comparison"]

    quality_rows = [
        {"metric": "Manifest records", "value": quality["manifest_records"], "unit": "records", "definition": "Rows in the pinned source manifest."},
        {"metric": "Existing GeoAI quality labels", "value": quality["quality_label_records"], "unit": "records", "definition": "Rows in descriptions/geoai/v1 with an existing usable field."},
        {"metric": "Confirmed usable", "value": quality["confirmed_usable_records"], "unit": "records", "definition": "Existing usable=true. No additional quality thresholds applied."},
        {"metric": "Explicitly excluded", "value": quality["explicitly_excluded_records"], "unit": "records", "definition": "Existing usable=false. Exact documented issue sets are in data/quality_exclusion_reason_sets.csv."},
        {"metric": "Quality status unknown", "value": quality["quality_unknown_records"], "unit": "records", "definition": "No existing label. Unknown is not treated as usable or unusable."},
        {"metric": "Confirmed usable in Fatih used in analysis", "value": quality["confirmed_usable_in_boundary_images"], "unit": "records", "definition": "Confirmed usable image locations inside the current boundary."},
        {"metric": "Explicitly excluded in Fatih", "value": quality["explicitly_excluded_in_boundary_images"], "unit": "records", "definition": "Existing usable=false locations inside the current boundary."},
        {"metric": "Unknown quality in Fatih", "value": quality["quality_unknown_in_boundary_images"], "unit": "records", "definition": "Not included because suitability is not confirmed."},
        {"metric": "Duplicate identifier records", "value": quality["manifest_duplicate_identifier_records"], "unit": "records", "definition": "Records beyond the first occurrence of an image_id."},
        {"metric": "Missing or invalid primary coordinates", "value": quality["primary_coordinate_missing_or_invalid"], "unit": "records", "definition": "lon or lat missing, non-finite, or outside geographic bounds."},
        {"metric": "Records at repeated primary coordinates", "value": quality["records_at_repeated_primary_coordinates"], "unit": "records", "definition": "Repeated locations are not duplicate records."},
        {"metric": "Capture period", "value": f"{quality['capture_time_min_utc'][:10]} to {quality['capture_time_max_utc'][:10]}", "unit": "UTC", "definition": "Range parsed from captured_at_iso."},
    ]
    csv_rows(tables / "data_quality.csv", quality_rows)
    write_latex_table(
        tables / "data_quality.tex",
        "Data quality control and confirmed-usable analytical cohort.",
        ["Metric", "Result"],
        [[tex_escape(row["metric"]), tex_escape(fmt_int(row["value"]) if isinstance(row["value"], (int, float)) else row["value"])] for row in quality_rows],
        "p{0.70\\linewidth}r",
    )

    network_rows = [
        {"analysis": "Original set", "setting": "All valid in-boundary locations, no quality filter", "matched_images": comparison["matched_images"], "covered_length_km": comparison["covered_length_m"] / 1000.0, "coverage_pct": comparison["coverage_pct"], "covered_unit_pct": comparison["covered_unit_pct"]},
        {"analysis": "Confirmed usable", "setting": "Existing usable=true, match ≤20 m, local coverage 25 m", "matched_images": primary["matched_images"], "covered_length_km": primary["covered_length_m"] / 1000.0, "coverage_pct": primary["coverage_pct"], "covered_unit_pct": primary["covered_unit_pct"]},
    ] + sensitivity
    csv_rows(tables / "network_coverage.csv", network_rows)
    latex_network_rows = []
    for row in network_rows:
        latex_network_rows.append(
            [
                tex_escape(row["analysis"]),
                tex_escape(row["setting"]),
                tex_escape(fmt_int(row["matched_images"])),
                tex_escape(fmt_num(row["covered_length_km"], 2)),
                tex_escape(fmt_num(row["coverage_pct"], 1) + "%"),
            ]
        )
    write_latex_table(
        tables / "network_coverage.tex",
        "Location-based street-network coverage and sensitivity results. All rows use the same eligible network.",
        ["Analysis", "Setting", "Matched images", "Covered km", "Coverage"],
        latex_network_rows,
        ">{\\raggedright\\arraybackslash}p{0.20\\linewidth}>{\\raggedright\\arraybackslash}p{0.31\\linewidth}r r r",
    )

    csv_rows(tables / "grid_coverage_summary.csv", grid_stats)
    latex_grid_rows = [
        [
            tex_escape(row["statistic"]),
            tex_escape(fmt_num(row["image_count"], 0)),
            tex_escape(fmt_num(row["eligible_street_length_km"], 2)),
            tex_escape(fmt_num(row["street_coverage_pct"], 1) + "%"),
            tex_escape(fmt_num(row["uncovered_street_km_per_km2"], 2)),
        ]
        for row in grid_stats
    ]
    write_latex_table(
        tables / "grid_coverage_summary.tex",
        "Distribution across 500 m regular grid cells with eligible street length.",
        ["Statistic", "Images", "Eligible km", "Coverage", "Uncovered km per km²"],
        latex_grid_rows,
        "p{0.25\\linewidth}r r r r",
    )

    # The briefing documents overwrite README.md and results.md in --out, so they are opt-in.
    if not write_docs:
        return

    brief = f"""\
\\section*{{Research briefing}}

\\subsection*{{Objective}}

This package supports a focused exploratory spatial analysis for the proposed Urban 21 Journal article, \\textit{{How Well Do Open Street Images Cover the City? Spatial Coverage and Data Gaps in Fatih, Istanbul}}. The study asks to what extent the analysed open street-level image dataset represents the eligible Fatih street network and how location-based gaps are distributed. The outcome is a measure of image-location support for street-network sections. It is not a measure of camera direction, visual visibility, image quality, or all imagery available on Mapillary.

\\subsection*{{Data and study area}}

The image source is the public \\texttt{{{DATASET_ID}}} repository at the pinned revision recorded in the package provenance \\cite{{fatih_dataset}}. The 123,907-row manifest supplies coordinates and timestamps. To reuse existing quality control, only \\texttt{{image\\_id}}, \\texttt{{usable}}, and the nested quality fields in the GeoAI description metadata were additionally read. No image bytes, segmentation masks, raw VLM responses, scene descriptions, or perception scores were used. The manifest preserves Mapillary image identifiers. The repository reports CC BY-SA 4.0 for the dataset and identifies Mapillary as the imagery source \\cite{{fatih_dataset,mapillary,creativecommons}}.

Fatih was represented by OpenStreetMap administrative relation {BOUNDARY_RELATION_ID}, tagged as an administrative level 6 district. The eligible road network was extracted from the same OpenStreetMap snapshot through a recorded Overpass query \\cite{{openstreetmap,odbl}}. All distance and area calculations used WGS 84 / UTM zone 35N, EPSG:32635. Fatih lies inside its stated 24°E to 30°E area of use. Geographic WGS 84 coordinates were retained only for exchange and data outputs.

\\subsection*{{Data quality and preparation}}

The source manifest contains {fmt_int(quality['manifest_records'])} records and {fmt_int(quality['manifest_unique_image_ids'])} unique image identifiers. Its existing GeoAI layer supplies {fmt_int(quality['quality_label_records'])} labels. The direct \\texttt{{usable}} field and nested \\texttt{{image\\_quality.usable\\_for\\_analysis}} field agree for all labelled images. It explicitly marks {fmt_int(quality['confirmed_usable_records'])} records usable and {fmt_int(quality['explicitly_excluded_records'])} records unsuitable. The remaining {fmt_int(quality['quality_unknown_records'])} records have no quality label and remain unknown. They are neither treated as usable nor as explicitly excluded.

The main analysis retains only existing \\texttt{{usable=true}} records. It does not create a new classifier, apply an undocumented \\texttt{{quality\\_score}} threshold, or exclude images based on individual issue words, vehicles, people, vegetation, or other ordinary street elements. The issue list is retained verbatim for explicitly excluded images. Exact issue sets form mutually exclusive exclusion categories in the supplied quality-control data, so their counts reconcile with the exclusion total. The dataset card says the GeoAI annotations are model-generated rather than human-verified ground truth. The status is therefore an existing documented filter, not an independent validation of each image.

The primary \\texttt{{lon}} and \\texttt{{lat}} fields are complete and valid for every record. There are {fmt_int(quality['repeated_primary_coordinate_groups'])} exact repeated-coordinate groups containing {fmt_int(quality['records_at_repeated_primary_coordinates'])} images. These are not duplicate records because identifiers remain distinct. The primary and computed coordinate pairs differ by a median of {fmt_num(quality['coordinate_disagreement_median_m'], 1)} m and exceed 20 m for {fmt_int(quality['coordinate_disagreement_gt_20m'])} records. Because the dataset card explicitly recommends \\texttt{{lon}} and \\texttt{{lat}} for quality checking while not defining the provenance of the computed pair, the former are the primary analytical coordinates.

After the documented quality checks and boundary filter, {fmt_int(quality['clean_in_boundary_images'])} confirmed-usable, valid, unique, in-boundary image locations enter the coverage analysis. {fmt_int(quality['explicitly_excluded_in_boundary_images'])} explicitly excluded and {fmt_int(quality['quality_unknown_in_boundary_images'])} unknown-quality locations within Fatih do not enter the analytical cohort. {fmt_int(quality['outside_boundary_records'])} valid records fall outside the current retrieved boundary. Capture timestamps span {quality['capture_time_min_utc'][:10]} to {quality['capture_time_max_utc'][:10]} UTC. The analysis is therefore about a temporally heterogeneous image collection, not a single-date survey.

\\subsection*{{Method}}

The eligible network includes public or unspecified-access OSM roads with highway values for major roads, local roads, service roads, pedestrian streets, and unclassified roads. Private and access=no ways are excluded. Service driveways, parking aisles, emergency accesses, drive-throughs, and area ways are also excluded. Footways, paths, cycleways, steps, tracks, construction, and proposed ways do not enter the street definition. Pedestrian streets are included because street-level imagery can reasonably be collected on them. The full filter is stored in the code and methods file.

Only the confirmed-usable image locations were assigned to their nearest eligible street geometry within 20 m using projected coordinates. The code saves distance, matched edge, projected position along the edge, near-tie diagnostics, and examples of alternative candidates. A matched image contributes only a 25 m local interval along its matched street. Source-edge intervals are merged, transferred to a noded non-overlapping network representation, and combined as exact along-atom intervals before length is measured. Thus, repeated images do not double count the same location and one image cannot cover a long way by itself. The main sensitivity analysis tests match thresholds of 10, 20, and 30 m. It separately tests local coverage distances of 15, 25, and 50 m. Network units are 10 m long and are classified as covered only when at least half of their length is covered.

\\subsection*{{Principal findings}}

The current boundary has an area of {fmt_num(network['study_area_km2'], 2)} km². The geometric eligible network totals {fmt_num(network['total_network_length_m'] / 1000.0, 2)} km. The primary 20 m rule matched {fmt_int(primary['matched_images'])} of {fmt_int(quality['clean_in_boundary_images'])} confirmed-usable image locations, or {fmt_num(primary['match_rate_pct'], 1)}\\%. Under the 25 m local rule, {fmt_num(primary['covered_length_m'] / 1000.0, 2)} km of network is covered and {fmt_num(primary['uncovered_length_m'] / 1000.0, 2)} km remains uncovered. The location-based coverage estimate is {fmt_num(primary['coverage_pct'], 1)}\\%. The 10 m unit result is {fmt_int(primary['covered_units'])} covered units out of {fmt_int(primary['total_units'])}, or {fmt_num(primary['covered_unit_pct'], 1)}\\%.

For context only, the original valid in-boundary location set yields {fmt_num(comparison['coverage_pct'], 1)}\\% coverage under the same 20 m and 25 m rules, compared with {fmt_num(primary['coverage_pct'], 1)}\\% for the confirmed-usable cohort. This comparison quantifies the spatial effect of the existing filter. It does not change the primary cohort or infer that excluded or unknown images lack any value for other purposes.

The maps distinguish image concentration from street coverage. Figure 1 maps image density by 500 m grid cell. Figure 2 is the principal evidence and maps covered and uncovered eligible street sections. Figure 3 shows coverage percentages by regular grid cell because a reliable official neighbourhood polygon layer was not obtained for this reproducible run. Figure 4 maps uncovered eligible street length per km². Local Moran analysis was deliberately omitted. A descriptive grid is sufficient for the stated research question, while inference from arbitrary grid units would add interpretation risk without changing the core result.

\\subsection*{{Contribution and interpretation}}

The package provides a transparent distinction between image density and location-based network coverage. A dense cluster of images can yield little additional street length after the local intervals overlap. Conversely, an image-sparse area can contain a substantial uncovered network. The results therefore help urban researchers and GeoAI users judge where this particular source can support street-level inference and where spatial sampling is weak. They do not establish why gaps occur, do not measure social or population exposure, and do not imply that a street absent from this dataset lacks imagery elsewhere on Mapillary.
"""
    (out / "analysis_brief.tex").write_text(textwrap.dedent(brief), encoding="utf-8")

    results_markdown = f"""# Verified results

All figures and values were generated by `code/analysis.py` from the pinned manifest and cached OSM source files. Values below are exact analysis outputs except where rounded for display. The machine-readable source is `data/results_summary.json`.

## Dataset and quality

- Source manifest records: {quality['manifest_records']:,} records. Output: `tables/data_quality.csv`.
- Unique image identifiers: {quality['manifest_unique_image_ids']:,} identifiers. Output: `tables/data_quality.csv`.
- Duplicate identifier records: {quality['manifest_duplicate_identifier_records']:,}. Output: `tables/data_quality.csv`.
- Existing GeoAI quality labels: {quality['quality_label_records']:,}. Output: `data/source/geoai_quality_labels.parquet`.
- Explicitly confirmed usable: {quality['confirmed_usable_records']:,}. Definition: existing `usable=true`; no new classifier or threshold. Output: `data/quality_filter_manifest.parquet`.
- Explicitly excluded: {quality['explicitly_excluded_records']:,}. Definition: existing `usable=false`. Exact, mutually exclusive documented issue sets are in `data/quality_exclusion_reason_sets.csv`.
- Quality status unknown: {quality['quality_unknown_records']:,}. Definition: no existing GeoAI quality label. Unknown is not counted as usable or unusable. Output: `data/quality_filter_manifest.parquet`.
- Valid primary `lon` and `lat` pairs: {quality['manifest_records'] - quality['primary_coordinate_missing_or_invalid']:,} records. Output: `data/quality_report.json`.
- Exact repeated primary-coordinate groups: {quality['repeated_primary_coordinate_groups']:,} groups containing {quality['records_at_repeated_primary_coordinates']:,} image records. Output: `data/quality_report.json`.
- Confirmed-usable in-boundary locations used in the main analysis: {quality['clean_in_boundary_images']:,}. Output: `data/clean_image_metadata.parquet`.
- Explicitly excluded and unknown in-boundary locations not used in the main analysis: {quality['explicitly_excluded_in_boundary_images']:,} and {quality['quality_unknown_in_boundary_images']:,}. Output: `data/quality_filter_manifest.parquet`.
- Capture period: {quality['capture_time_min_utc']} to {quality['capture_time_max_utc']}. Output: `data/quality_report.json`.

## Overall location-based network coverage

- Study area: {network['study_area_km2']:.6f} km². Output: `data/results_summary.json`.
- Eligible geometric network length: {network['total_network_length_m']:.3f} m. Output: `data/results_summary.json`.
- Clean image density: {network['image_density_per_km2']:.6f} images per km². Output: `data/results_summary.json`.
- Images per eligible network kilometre: {network['images_per_km_street']:.6f}. Output: `data/results_summary.json`.
- Primary nearest-street match rule: distance less than or equal to 20 m. Matched images: {primary['matched_images']:,} of {quality['clean_in_boundary_images']:,}, or {primary['match_rate_pct']:.6f}%. Output: `tables/network_coverage.csv`.
- Primary local coverage rule: 25 m along the matched street geometry. Covered length: {primary['covered_length_m']:.3f} m. Uncovered length: {primary['uncovered_length_m']:.3f} m. Coverage: {primary['coverage_pct']:.6f}%. Output: `tables/network_coverage.csv`.
- Street units: {primary['covered_units']:,} of {primary['total_units']:,} 10 m units have at least 50% covered length, or {primary['covered_unit_pct']:.6f}%. Output: `data/results_summary.json`.
- Near-tie match diagnostic: {primary['near_tie_matches']:,} matches have a second candidate within {NEAR_TIE_MARGIN_M:.0f} m of the nearest-candidate distance. Output: `data/ambiguous_match_examples.csv` and `data/image_match_summary.parquet`.
- Original-location comparison: {comparison['image_count']:,} valid in-boundary locations produce {comparison['coverage_pct']:.6f}% coverage under the same primary matching and local-distance rules. This is a context comparison only. Output: `tables/network_coverage.csv`.

## Sensitivity analysis

The following rows use the same eligible network and exact interval-union method. See `tables/network_coverage.csv` for unrounded values.

""" + "\n".join(
        f"- {row['analysis']}, {row['setting']}: {row['matched_images']:,} matched images, {row['covered_length_km']:.6f} km covered, {row['coverage_pct']:.6f}% coverage."
        for row in sensitivity
    ) + "\n\n## Grid results\n\n" + "\n".join(
        f"- {row['statistic']}: coverage {row['street_coverage_pct']:.6f}%, image count {row['image_count']:.6f}, eligible street length {row['eligible_street_length_km']:.6f} km. Output: `tables/grid_coverage_summary.csv`."
        for row in grid_stats
    ) + "\n"
    (out / "results.md").write_text(results_markdown, encoding="utf-8")

    methods = f"""# Methods

## Scope and source version

The analysis used the Hugging Face dataset `{DATASET_ID}` at commit `{DATASET_REVISION}`. The required source manifest was `data/raw/manifest/train.parquet`, whose SHA-256 is `{MANIFEST_SHA256}`. The existing GeoAI description layer supplied its image identifier, usability, and nested image-quality fields through byte-range retrieval of only needed metadata columns. The manifest has one row per stated image identifier. Raw WebDataset image shards, segmentation masks, raw VLM responses, scene descriptions, and perception scores were excluded from the workflow.

The boundary is OpenStreetMap administrative relation `{BOUNDARY_RELATION_ID}` for Fatih. The cached Nominatim response identifies it as an administrative boundary with `admin_level=6`. The OSM network source is the cached Overpass response listed in `data/source/eligible_streets_overpass.json`. Its query, response SHA-256, snapshot timestamp, and retrieval date are in `data/provenance.json`.

## Coordinate handling and data quality

The manifest `lon` and `lat` fields were selected as the primary coordinate pair because the dataset card explicitly identifies them for spatial filtering and quality checks. `computed_lon` and `computed_lat` were retained for comparison. A primary pair is valid only when both values are present and lie within geographic longitude and latitude bounds. A computed pair is used only as a fallback when the primary pair is invalid. The original `image_id` is retained throughout.

Duplicate identifiers are assessed independently of repeated locations. For a duplicate identifier, the first source row in original manifest order is retained for a deterministic one-row-per-image analysis, while the exclusion count remains in the quality report. Repeated coordinates are not removed. They can represent distinct valid images at one location.

## Existing quality filter

The repository contains no separate quality-control folder, exclusion-list file, or processed image subset. Its existing quality result is the GeoAI description-layer usable field, joined by image identifier. The nested parsed quality usability and issue values are retained as an audit check. The direct and nested usability fields agree for all labelled records in this analysis.

The analytical cohort uses the documented usable=true rule. Records with usable=false are explicitly excluded. Records without a GeoAI description row have quality status unknown, are not classified as usable or unusable, and are not included in a confirmed-usable analysis. No new image-quality model, score threshold, or word-based issue threshold is introduced. In particular, an image is not excluded merely for vehicles, pedestrians, vegetation, or ordinary urban objects.

The package retains a one-row-per-image quality-filter manifest with the original identifier, existing quality status, documented issue set, in-boundary flag, and analytical inclusion reason. The quality-exclusion reason-set CSV reports exact issue-set combinations. These are mutually exclusive by construction and reconcile to the explicit-exclusion count. Atomic issue labels can overlap, so they are not summed as exclusion categories.

Valid image points are transformed from EPSG:4326 to EPSG:32635. EPSG:32635 is WGS 84 / UTM zone 35N. Its area of use spans 24°E to 30°E in the northern hemisphere, which contains Fatih. Distances, line lengths, grid areas, and buffers are therefore calculated in metres. Geographic coordinates are used only for source data and exported exchange files.

## Eligible network

The Overpass query requests highway values `motorway`, `motorway_link`, `trunk`, `trunk_link`, `primary`, `primary_link`, `secondary`, `secondary_link`, `tertiary`, `tertiary_link`, `unclassified`, `residential`, `living_street`, `service`, `pedestrian`, and `road`. Ways with `access=private` or `access=no` are excluded. Service ways tagged `driveway`, `parking_aisle`, `emergency_access`, `drive-through`, or `parking` are excluded. Ways tagged as area features are excluded.

Pedestrian streets are included because street-level images can reasonably be collected on them. Footways, paths, cycleways, steps, tracks, construction, proposed ways, and non-street facilities are outside this street definition. The resulting linework is clipped to the boundary, exploded into LineStrings, and lines shorter than 0.1 m are removed. Reversed duplicate geometries with the same bridge, tunnel, and layer tags are removed. A final two-dimensional geometric union nodes the network and removes coincident interiors before length calculations.

## Location-based coverage

Each confirmed-usable in-boundary image point is assigned to its nearest eligible street geometry using a Shapely spatial index. The primary match threshold is 20 m. Ties are resolved by distance and then stable edge identifier. The process records nearest distance, projected position along the matched line, distance to line endpoints, all candidate count within threshold, and a near-tie flag. A near tie has a second candidate within {NEAR_TIE_MARGIN_M:.0f} m of the nearest-candidate distance. These cases are retained under the deterministic rule and are reported rather than silently discarded.

For an image matched at along-line position \\(s\\) on an edge of length \\(L\\), a local interval is formed from `max(0, s-r)` to `min(L, s+r)`, where \\(r\\) is the local coverage distance. The primary value is \\(r={PRIMARY_COVERAGE_DISTANCE_M:.0f}\\) m. Intervals are merged on each source edge and then intersected with the noded, non-overlapping network atoms. Along-atom intervals are merged. Covered and uncovered portions are formed as complementary intervals on each atom, and their lengths are summed across atoms. The code asserts that these two values reconcile with the total eligible network length. This method prevents overlapping images from double counting and does not assign a whole long OSM way to one image.

The primary result is supplemented by 10 m units. A unit is covered only if its covered fraction is at least 0.5. Exact covered length remains the primary metric.

## Grid and sensitivity analysis

Reliable official neighbourhood polygon data was not available in a documented, reproducible form for this run. The area is therefore partitioned into 500 m square cells, clipped to Fatih. This resolution produces a readable city-scale map while retaining local variation. For each cell, the analysis calculates image count, clipped cell area, image density per km², eligible street length, covered street length, coverage percentage, images per street kilometre, and uncovered street km per km². Lines are intersected with cells, not allocated by centroids.

Sensitivity tests vary the match threshold with a fixed 25 m local coverage distance at 10, 20, and 30 m. They also vary the local coverage distance with a fixed 20 m match threshold at 15, 25, and 50 m. A conservative diagnostic excludes near-tie matches under the primary thresholds. A comparison reruns the primary rule with all valid in-boundary locations before the existing usability filter. Assertions check quality-status reconciliation, covered and uncovered length reconciliation, percentage bounds, grid allocation, and expected monotonicity of coverage across local-distance sensitivity values.

## Spatial statistics

Moran's I and Local Moran statistics were not run. The evidence needed for the research question is descriptive and geographic. Grid-based inferential clustering would depend on an arbitrary cell partition and would invite causal interpretation of source-data gaps that this dataset cannot support.
"""
    (out / "methods.md").write_text(textwrap.dedent(methods), encoding="utf-8")

    limitations = f"""# Limitations

- This study measures location-based coverage of the analysed `{DATASET_ID}` manifest. It does not measure all imagery on Mapillary or any other street-imagery platform.
- Image coordinates alone do not establish camera direction, field of view, visibility, visual quality, temporal representativeness, or whether a street scene was actually visible from a matched location.
- The confirmed-usable cohort follows the repository's existing GeoAI `usable` field. The README says these annotations are model-generated and not human-verified ground truth. The 314 records without that label remain quality-unknown and are excluded from the confirmed-usable cohort without being classed as poor quality. This conservative rule can reduce estimated spatial support.
- Existing issue labels can be multi-valued and do not behave as a new deterministic threshold. Some usable images also retain issue words. The analysis preserves the repository's explicit usability decision and does not reinterpret individual words such as blur or partial view.
- The collection spans {quality['capture_time_min_utc'][:10]} to {quality['capture_time_max_utc'][:10]}. It is temporally heterogeneous.
- The current OSM Fatih boundary and street network were retrieved after the dataset build. The dataset card says it used an OSM boundary but does not record the original relation version or retrieval date. Boundary-based exclusions are therefore reported against the cached current source, not assumed to recreate the original filter exactly.
- The computed coordinate fields differ from primary `lon` and `lat` for some records, and their provenance is undocumented in the dataset card. The analysis uses the documented primary pair and retains the alternative fields for audit.
- Nearest-line matching is ambiguous around intersections, parallel roads, bridges, and closely spaced ways. The package records near-tie diagnostics, applies a deterministic tie rule, and reports a conservative near-tie exclusion result. Coordinates do not resolve vertical separation or road-level assignment.
- The eligible network is an operational OSM definition of public streets. OSM tagging completeness and the inclusion or exclusion rules for pedestrian and service streets affect denominators.
- The regular grid is an analytical reporting unit, not an administrative neighbourhood. No inference about neighbourhood characteristics, contributors, social inequality, or the causes of missing observations is made.
- OSM is licensed under ODbL. The source dataset card declares CC BY-SA 4.0 and attributes the images to Mapillary. This package redistributes no image bytes.
"""
    (out / "limitations.md").write_text(textwrap.dedent(limitations), encoding="utf-8")

    references = f"""@misc{{fatih_dataset,
  author = {{yunusserhat}},
  title = {{Fatih Mapillary Street-Level Images}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/datasets/yunusserhat/fatih}},
  note = {{Dataset revision {DATASET_REVISION}, accessed 2026-09-17. The dataset card declares CC BY-SA 4.0.}}
}}

@misc{{openstreetmap,
  author = {{OpenStreetMap contributors}},
  title = {{OpenStreetMap}},
  year = {{2026}},
  url = {{https://www.openstreetmap.org}},
  note = {{Fatih administrative relation {BOUNDARY_RELATION_ID} and eligible highway ways retrieved 2026-09-17.}}
}}

@misc{{mapillary,
  author = {{Mapillary}},
  title = {{Mapillary}},
  year = {{2026}},
  url = {{https://www.mapillary.com/}},
  note = {{Imagery platform identified by the source dataset card, accessed 2026-09-17.}}
}}

@misc{{odbl,
  author = {{Open Data Commons}},
  title = {{Open Database License (ODbL) v1.0}},
  year = {{2026}},
  url = {{https://opendatacommons.org/licenses/odbl/1-0/}},
  note = {{License stated by the Overpass API response for OpenStreetMap data.}}
}}

@misc{{creativecommons,
  author = {{Creative Commons}},
  title = {{Attribution-ShareAlike 4.0 International}},
  year = {{2026}},
  url = {{https://creativecommons.org/licenses/by-sa/4.0/}},
  note = {{License linked by the source dataset card.}}
}}
"""
    (out / "references.bib").write_text(references, encoding="utf-8")

    main_tex = """\\documentclass[10pt]{article}
\\usepackage[a4paper,margin=20mm]{geometry}
\\usepackage{graphicx}
\\usepackage{xcolor}
\\usepackage{hyperref}
\\usepackage{array}
\\usepackage{url}
\\usepackage[utf8]{inputenc}
\\DeclareUnicodeCharacter{2264}{\\ensuremath{\\leq}}
\\DeclareUnicodeCharacter{00B2}{\\textsuperscript{2}}
\\DeclareUnicodeCharacter{00B0}{\\ensuremath{^\\circ}}
\\hypersetup{colorlinks=true,linkcolor=black,citecolor=blue,urlcolor=blue}
\\graphicspath{{figures/}}
\\setlength{\\parskip}{0.45em}
\\setlength{\\parindent}{0pt}
\\begin{document}

{\\Large\\bfseries How Well Do Open Street Images Cover the City?\\par}
{\\large Spatial Coverage and Data Gaps in Fatih, Istanbul\\par}
\\vspace{0.7em}
{\\small Verified research briefing for OpenAI Prism\\par}

\\input{analysis_brief.tex}
\\clearpage
\\section*{Final tables}
\\input{tables/data_quality.tex}
\\input{tables/network_coverage.tex}
\\input{tables/grid_coverage_summary.tex}
\\clearpage
\\section*{Figures}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.94\\linewidth]{fig1_image_distribution.pdf}
\\caption{Spatial distribution of confirmed-usable image locations in Fatih. Grid shading represents image density in images per km². Density describes the distribution of observations and does not indicate street-network coverage. Source: pinned manifest, existing GeoAI quality filter, and OSM boundary.}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.94\\linewidth]{fig2_network_coverage.pdf}
\\caption{Location-based coverage of the eligible Fatih street network. Covered sections have at least one confirmed-usable image matched within 20 m and a local 25 m interval along the matched street geometry. This does not establish continuous visual or directional coverage. Source: pinned manifest, existing GeoAI quality filter, and OSM network.}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.94\\linewidth]{fig3_grid_coverage.pdf}
\\caption{Street-network coverage percentage by 500 m regular grid cell. The grid replaces neighbourhood polygons because a reliable official neighbourhood boundary source was not available for this reproducible analysis. Source: pinned manifest, existing GeoAI quality filter, and OSM network.}
\\end{figure}

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.94\\linewidth]{fig4_gap_density.pdf}
\\caption{Spatial concentration of uncovered eligible street length. Grid values show kilometres of uncovered eligible network per km². Cells without eligible street geometry are shown in light grey. Higher values identify concentrations of gaps in this dataset, not their causes or the absence of imagery elsewhere on Mapillary. Source: pinned manifest, existing GeoAI quality filter, and OSM network.}
\\end{figure}

\\clearpage
\\section*{Supporting notes}
The exact numerical record is in \\texttt{results.md}. Methods and assumptions are in \\texttt{methods.md}. Interpretation limits are in \\texttt{limitations.md}.
\\bibliographystyle{plain}
\\bibliography{references}
\\end{document}
"""
    (out / "main.tex").write_text(main_tex, encoding="utf-8")

    readme = f"""# Fatih spatial coverage research package

This compact package is directly importable into OpenAI Prism. Upload `fatih_spatial_coverage_prism.zip` or this folder. `main.tex` is the document entry point and uses only relative paths.

## Contents

- `main.tex` compiles the verified briefing, three final tables, and four figures.
- `analysis_brief.tex` is a concise research briefing, not a full article.
- `results.md`, `methods.md`, and `limitations.md` record results, procedure, and interpretation limits.
- `figures/` contains vector PDF and 400 dpi PNG exports.
- `tables/` contains CSV and LaTeX table versions.
- `data/` contains compact derived data, the quality-control summary and manifest, OSM source caches, diagnostics, and a machine-readable summary.
- `code/` contains the scripts and pinned environment requirements.

## Provenance

The image manifest comes from `{DATASET_ID}` at revision `{DATASET_REVISION}`. Its source checksum is `{MANIFEST_SHA256}`. The existing GeoAI quality labels were extracted from the same revision using only `image_id`, `usable`, and nested quality metadata. The raw 95 MB manifest is intentionally not included because it is publicly retrievable and the package retains only compact derived metadata. No street-image bytes, WebDataset shards, raw VLM text, segmentation masks, or perception outputs are included.

The boundary is OSM administrative relation {BOUNDARY_RELATION_ID}. The eligible network comes from the cached Overpass response in `data/source/`. OSM attribution and ODbL requirements apply. The source dataset card identifies Mapillary imagery and declares CC BY-SA 4.0. See `references.bib` and `data/provenance.json`.

## Reproduction

Use Python 3.12 or later in a clean environment.

```bash
python -m venv .venv
.venv/bin/pip install -r code/requirements.txt
python code/fetch_sources.py --output /tmp/fatih_inputs --manifest-only
.venv/bin/python code/extract_geoai_quality.py --output /tmp/fatih_inputs/geoai_quality_labels.parquet
.venv/bin/python code/analysis.py \\
  --manifest /tmp/fatih_inputs/manifest_train.parquet \\
  --quality /tmp/fatih_inputs/geoai_quality_labels.parquet \\
  --boundary data/source/fatih_boundary_nominatim.json \\
  --streets data/source/eligible_streets_overpass.json \\
  --out .
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/fatih_latex main.tex
```

The analysis script checks the pinned manifest checksum, quality-status reconciliation, covered plus uncovered network length against total network length, percentage bounds, grid allocation, and sensitivity monotonicity. The cached boundary and street source preserve the exact source snapshot used for the reported results. `fetch_sources.py` can also retrieve a fresh OSM source cache if needed, but that will reflect current OSM rather than the cached snapshot.

## Completed analysis

The package completes manifest inspection, existing-quality-filter reuse, a reproducible quality manifest, data quality assessment, boundary filtering, OSM eligible-network extraction, image density mapping, location-based coverage analysis, original-versus-filtered comparison, grid analysis, sensitivity analysis, ambiguity diagnostics, four figures, three tables, and local LaTeX validation. It does not perform Moran or Local Moran analysis because a descriptive gap map better fits the question and available evidence.

## Important interpretation boundary

The reported primary coverage is limited to confirmed-usable records in the analysed dataset, its timestamp range, current cached OSM geography, and the stated matching rules. It must not be interpreted as all Mapillary coverage, camera visibility, visual coverage, neighbourhood condition, population exposure, or a cause of missing data.
"""
    (out / "README.md").write_text(textwrap.dedent(readme), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--streets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--write-package-docs",
        action="store_true",
        help="Also write the briefing documents (README.md, results.md, main.tex and others) into --out.",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    figures_dir = out / "figures"
    tables_dir = out / "tables"
    data_dir = out / "data"
    source_dir = data_dir / "source"
    for directory in [figures_dir, tables_dir, data_dir, source_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    boundary_feature, boundary_geographic, boundary_projected = read_boundary(args.boundary)
    metadata, quality = read_manifest(args.manifest)
    metadata, quality_labels_report, exclusion_combinations = attach_existing_quality_labels(metadata, args.quality)
    quality.update(quality_labels_report)
    all_points = make_image_points(metadata, boundary_projected)
    in_boundary_points = all_points.loc[all_points["inside_boundary"]].copy().reset_index(drop=True)
    clean_points = in_boundary_points.loc[
        in_boundary_points["quality_status"].eq("confirmed_usable")
    ].copy().reset_index(drop=True)
    quality.update(
        {
            "valid_deduplicated_coordinate_records": int(len(all_points)),
            "outside_boundary_records": int((~all_points["inside_boundary"]).sum()),
            "original_in_boundary_images": int(len(in_boundary_points)),
            "confirmed_usable_in_boundary_images": int(len(clean_points)),
            "explicitly_excluded_in_boundary_images": int(
                in_boundary_points["quality_status"].eq("explicitly_excluded").sum()
            ),
            "quality_unknown_in_boundary_images": int(
                in_boundary_points["quality_status"].eq("unknown").sum()
            ),
            "clean_in_boundary_images": int(len(clean_points)),
            "primary_lon_min": float(metadata.loc[metadata["primary_coordinate_valid"], "lon"].min()),
            "primary_lon_max": float(metadata.loc[metadata["primary_coordinate_valid"], "lon"].max()),
            "primary_lat_min": float(metadata.loc[metadata["primary_coordinate_valid"], "lat"].min()),
            "primary_lat_max": float(metadata.loc[metadata["primary_coordinate_valid"], "lat"].max()),
        }
    )

    streets, street_report = read_eligible_streets(args.streets, boundary_projected)
    network_geometry = unary_union(list(streets.geometry))
    network_context = build_network_context(network_geometry)
    total_network_length_m = network_context["total_length_m"]
    street_report["geometric_network_length_m"] = total_network_length_m
    street_report["overlap_removed_by_2d_union_m"] = float(
        streets["length_m"].sum() - total_network_length_m
    )
    if total_network_length_m <= 0:
        raise AssertionError("Eligible network has zero length.")

    # Query all valid locations once. The filtered analysis retains only image IDs
    # explicitly confirmed usable by the existing repository quality label.
    all_matches, tree = nearest_matches(in_boundary_points, streets, max_distance_m=30.0)
    all_matches, ambiguity_examples, ambiguity_report = ambiguity_diagnostics(
        in_boundary_points, streets, all_matches, tree, PRIMARY_MATCH_THRESHOLD_M, NEAR_TIE_MARGIN_M
    )
    included_ids = set(clean_points["image_id"].astype(str))
    matches = all_matches.loc[all_matches["image_id"].isin(included_ids)].copy()
    original_primary_matches = all_matches.loc[
        all_matches["nearest_distance_m"] <= PRIMARY_MATCH_THRESHOLD_M
    ].copy()
    original_coverage = coverage_from_matches(
        streets, network_context, original_primary_matches, PRIMARY_COVERAGE_DISTANCE_M
    )
    original_units = unit_metrics_from_atom_intervals(
        network_context, original_coverage["covered_atom_intervals"], UNIT_LENGTH_M
    )
    original_comparison = {
        "dataset": "Original valid in-boundary locations",
        "image_count": int(len(in_boundary_points)),
        "matched_images": int(len(original_primary_matches)),
        "match_rate_pct": safe_percent(len(original_primary_matches), len(in_boundary_points)),
        "covered_length_m": original_coverage["covered_length_m"],
        "coverage_pct": original_coverage["coverage_pct"],
        "covered_unit_pct": original_units["covered_unit_pct"],
    }
    primary_matches = matches.loc[matches["nearest_distance_m"] <= PRIMARY_MATCH_THRESHOLD_M].copy()
    primary_coverage = coverage_from_matches(
        streets, network_context, primary_matches, PRIMARY_COVERAGE_DISTANCE_M
    )
    primary_units = unit_metrics_from_atom_intervals(
        network_context, primary_coverage["covered_atom_intervals"], UNIT_LENGTH_M
    )
    primary = {
        "match_threshold_m": PRIMARY_MATCH_THRESHOLD_M,
        "local_coverage_distance_m": PRIMARY_COVERAGE_DISTANCE_M,
        "matched_images": int(len(primary_matches)),
        "unmatched_images": int(len(clean_points) - len(primary_matches)),
        "match_rate_pct": safe_percent(len(primary_matches), len(clean_points)),
        "covered_length_m": primary_coverage["covered_length_m"],
        "uncovered_length_m": primary_coverage["uncovered_length_m"],
        "coverage_pct": primary_coverage["coverage_pct"],
        "covered_units": primary_units["covered_units"],
        "total_units": primary_units["total_units"],
        "covered_unit_pct": primary_units["covered_unit_pct"],
        "near_tie_matches": int(primary_matches["near_tie"].sum()),
        "endpoint_proximate_matches": int(primary_matches["endpoint_proximate"].sum()),
        "median_match_distance_m": float(primary_matches["nearest_distance_m"].median()),
        "p95_match_distance_m": float(primary_matches["nearest_distance_m"].quantile(0.95)),
        "max_match_distance_m": float(primary_matches["nearest_distance_m"].max()),
        "coverage_interval_count": primary_coverage["interval_count"],
        "coverage_interval_edge_count": primary_coverage["interval_edge_count"],
    }

    sensitivity = []
    threshold_results = {}
    for threshold in [10.0, 20.0, 30.0]:
        subset = matches.loc[matches["nearest_distance_m"] <= threshold]
        coverage = coverage_from_matches(streets, network_context, subset, PRIMARY_COVERAGE_DISTANCE_M)
        units = unit_metrics_from_atom_intervals(
            network_context, coverage["covered_atom_intervals"], UNIT_LENGTH_M
        )
        threshold_results[threshold] = coverage
        sensitivity.append(
            {
                "analysis": "Match threshold",
                "setting": f"Match ≤{threshold:.0f} m, local coverage 25 m",
                "matched_images": int(len(subset)),
                "covered_length_km": coverage["covered_length_m"] / 1000.0,
                "coverage_pct": coverage["coverage_pct"],
                "covered_unit_pct": units["covered_unit_pct"],
            }
        )
    local_results = {}
    for local_distance in [15.0, 25.0, 50.0]:
        coverage = coverage_from_matches(streets, network_context, primary_matches, local_distance)
        units = unit_metrics_from_atom_intervals(
            network_context, coverage["covered_atom_intervals"], UNIT_LENGTH_M
        )
        local_results[local_distance] = coverage
        sensitivity.append(
            {
                "analysis": "Local distance",
                "setting": f"Match ≤20 m, local coverage {local_distance:.0f} m",
                "matched_images": int(len(primary_matches)),
                "covered_length_km": coverage["covered_length_m"] / 1000.0,
                "coverage_pct": coverage["coverage_pct"],
                "covered_unit_pct": units["covered_unit_pct"],
            }
        )
    conservative_matches = primary_matches.loc[~primary_matches["near_tie"]]
    conservative_coverage = coverage_from_matches(
        streets, network_context, conservative_matches, PRIMARY_COVERAGE_DISTANCE_M
    )
    conservative_units = unit_metrics_from_atom_intervals(
        network_context, conservative_coverage["covered_atom_intervals"], UNIT_LENGTH_M
    )
    sensitivity.append(
        {
            "analysis": "Ambiguity check",
            "setting": "Match ≤20 m, local 25 m, exclude near ties",
            "matched_images": int(len(conservative_matches)),
            "covered_length_km": conservative_coverage["covered_length_m"] / 1000.0,
            "coverage_pct": conservative_coverage["coverage_pct"],
            "covered_unit_pct": conservative_units["covered_unit_pct"],
        }
    )

    if not (
        len(matches.loc[matches["nearest_distance_m"] <= 10.0])
        <= len(primary_matches)
        <= len(matches.loc[matches["nearest_distance_m"] <= 30.0])
    ):
        raise AssertionError("Match-threshold sensitivity is not monotonic.")
    if not (
        local_results[15.0]["covered_length_m"]
        <= local_results[25.0]["covered_length_m"] + 0.05
        <= local_results[50.0]["covered_length_m"] + 0.05
    ):
        raise AssertionError("Local-distance sensitivity is not monotonic.")

    grid = make_grid_results(
        clean_points,
        network_geometry,
        primary_coverage["covered_geometry"],
        GRID_SIZE_M,
        boundary_projected,
    )
    if abs(grid["eligible_street_length_m"].sum() - total_network_length_m) > 0.5:
        raise AssertionError("Grid network-length allocation does not reconcile with the network total.")
    if abs(grid["covered_street_length_m"].sum() - primary["covered_length_m"]) > 0.5:
        raise AssertionError("Grid covered-length allocation does not reconcile with covered total.")

    study_area_km2 = float(boundary_projected.area / 1_000_000.0)
    network = {
        "study_area_km2": study_area_km2,
        "total_network_length_m": total_network_length_m,
        "clean_image_count": int(len(clean_points)),
        "image_density_per_km2": len(clean_points) / study_area_km2,
        "images_per_km_street": len(clean_points) / (total_network_length_m / 1000.0),
        "grid_cell_size_m": GRID_SIZE_M,
        "grid_cell_count": int(len(grid)),
        "grid_cells_with_eligible_street_length": int((grid["eligible_street_length_m"] > 0).sum()),
    }
    results = {
        "quality": quality,
        "street_network": street_report,
        "matching": ambiguity_report,
        "network": network,
        "original_filtered_comparison": original_comparison,
        "primary": primary,
        "sensitivity": sensitivity,
        "grid_summary": grid_summary(grid),
    }

    # Compact derived data. No raw imagery, vision outputs, or raw manifest are copied.
    output_columns = [
        "image_id", "sequence_id", "lon", "lat", "computed_lon", "computed_lat", "analysis_lon", "analysis_lat",
        "coordinate_source", "captured_at", "captured_at_iso", "camera_type", "quality_score", "source_exists",
        "download_status", "repeat_primary_coordinate", "coordinate_disagreement_m", "quality_status",
        "quality_issues_json", "quality_parse_status", "quality_label_source", "analysis_quality_included",
        "inside_boundary", "geometry",
    ]
    clean_points[output_columns].to_crs(WGS84).to_parquet(data_dir / "clean_image_metadata.parquet", index=False)
    matches_export = matches.merge(
        clean_points[["image_id", "analysis_lon", "analysis_lat"]], on="image_id", how="left"
    )
    matches_export.to_parquet(data_dir / "image_match_summary.parquet", index=False)
    streets.drop(columns=["geometry_key"]).to_crs(WGS84).to_file(
        data_dir / "eligible_streets.geojson", driver="GeoJSON"
    )
    gpd.GeoDataFrame(
        {"network": ["eligible_geometric_union"]}, geometry=[network_geometry], crs=PROJECTED
    ).to_crs(WGS84).to_file(data_dir / "eligible_network_union.geojson", driver="GeoJSON")
    primary_coverage["sections"].to_crs(WGS84).to_file(data_dir / "covered_sections.geojson", driver="GeoJSON")
    gpd.GeoDataFrame(
        {"network": ["uncovered_geometric_union"]}, geometry=[primary_coverage["uncovered_geometry"]], crs=PROJECTED
    ).to_crs(WGS84).to_file(data_dir / "uncovered_sections.geojson", driver="GeoJSON")
    grid_output = grid.copy()
    centroids = grid_output.geometry.centroid
    centroid_wgs84 = gpd.GeoSeries(centroids, crs=PROJECTED).to_crs(WGS84)
    grid_output["centroid_lon"] = centroid_wgs84.x
    grid_output["centroid_lat"] = centroid_wgs84.y
    grid_output.drop(columns="geometry").to_csv(data_dir / "grid_500m.csv", index=False)
    grid_output.to_crs(WGS84).to_file(data_dir / "grid_500m.geojson", driver="GeoJSON")
    included_ambiguity_examples = ambiguity_examples.loc[
        ambiguity_examples["image_id"].isin(included_ids)
    ].copy()
    if len(included_ambiguity_examples):
        included_ambiguity_examples.sort_values(["candidate_relationship", "image_id"]).groupby(
            "candidate_relationship", group_keys=False
        ).head(12).to_csv(data_dir / "ambiguous_match_examples.csv", index=False)
    else:
        pd.DataFrame(columns=ambiguity_examples.columns).to_csv(
            data_dir / "ambiguous_match_examples.csv", index=False
        )

    inclusion_lookup = all_points[["image_id", "inside_boundary"]].copy()
    quality_manifest = metadata.merge(inclusion_lookup, on="image_id", how="left", validate="one_to_one")
    quality_manifest["inside_boundary"] = quality_manifest["inside_boundary"].fillna(False)
    quality_manifest["analysis_included"] = (
        quality_manifest["inside_boundary"] & quality_manifest["quality_status"].eq("confirmed_usable")
    )
    quality_manifest["analysis_inclusion_reason"] = np.select(
        [
            quality_manifest["analysis_included"],
            quality_manifest["quality_status"].eq("explicitly_excluded"),
            quality_manifest["quality_status"].eq("unknown"),
        ],
        [
            "confirmed_usable_and_in_boundary",
            "explicitly_excluded_by_existing_geoai_label",
            "quality_status_unknown_not_confirmed_usable",
        ],
        default="outside_study_boundary_or_invalid_coordinates",
    )
    quality_manifest_columns = [
        "image_id", "sequence_id", "lon", "lat", "computed_lon", "computed_lat", "captured_at_iso",
        "quality_status", "usable", "parsed_quality_usable", "quality_issues_json", "quality_parse_status",
        "quality_label_source", "inside_boundary", "analysis_included", "analysis_inclusion_reason",
    ]
    quality_manifest[quality_manifest_columns].to_parquet(
        data_dir / "quality_filter_manifest.parquet", index=False
    )
    quality_control_summary = [
        {
            "category": "Original manifest image records",
            "count": quality["manifest_records"],
            "definition": "All rows in the pinned raw manifest.",
        },
        {
            "category": "Existing GeoAI quality labels available",
            "count": quality["quality_label_records"],
            "definition": "Rows with the repository's descriptions/geoai/v1 usable field.",
        },
        {
            "category": "Confirmed usable",
            "count": quality["confirmed_usable_records"],
            "definition": "Existing usable=true label. No additional threshold applied.",
        },
        {
            "category": "Explicitly excluded",
            "count": quality["explicitly_excluded_records"],
            "definition": "Existing usable=false label.",
        },
        {
            "category": "Quality status unknown",
            "count": quality["quality_unknown_records"],
            "definition": "No existing GeoAI usability label. Not assumed usable or unusable.",
        },
        {
            "category": "Confirmed usable in Fatih used in main analysis",
            "count": quality["confirmed_usable_in_boundary_images"],
            "definition": "Confirmed usable records within the current analysis boundary.",
        },
        {
            "category": "Explicitly excluded in Fatih",
            "count": quality["explicitly_excluded_in_boundary_images"],
            "definition": "Existing usable=false records within the current analysis boundary.",
        },
        {
            "category": "Unknown quality in Fatih",
            "count": quality["quality_unknown_in_boundary_images"],
            "definition": "Not included because suitability is not confirmed.",
        },
    ]
    csv_rows(data_dir / "quality_control_summary.csv", quality_control_summary)
    exclusion_combinations["share_of_explicit_exclusions_pct"] = (
        100.0 * exclusion_combinations["excluded_images"] / quality["explicitly_excluded_records"]
    )
    exclusion_combinations.to_csv(data_dir / "quality_exclusion_reason_sets.csv", index=False)
    results["quality_control_summary"] = quality_control_summary
    results["quality_exclusion_reason_sets"] = exclusion_combinations.to_dict(orient="records")
    json_dump(data_dir / "quality_report.json", quality)
    json_dump(data_dir / "results_summary.json", results)

    boundary_gdf = gpd.GeoDataFrame(
        {"osm_relation_id": [BOUNDARY_RELATION_ID], "name": [boundary_feature.get("display_name", "Fatih")]},
        geometry=[boundary_geographic],
        crs=WGS84,
    )
    boundary_gdf.to_file(data_dir / "fatih_boundary.geojson", driver="GeoJSON")
    copy_source(args.boundary, source_dir / "fatih_boundary_nominatim.json")
    copy_source(args.streets, source_dir / "eligible_streets_overpass.json")
    copy_source(args.quality, source_dir / "geoai_quality_labels.parquet")
    quality_provenance = args.quality.with_suffix(".provenance.json")
    if quality_provenance.exists():
        copy_source(quality_provenance, source_dir / "geoai_quality_labels.provenance.json")
    provenance = {
        "analysis_run_utc": datetime.now(timezone.utc).isoformat(),
        "crs": {
            "geographic": WGS84,
            "projected": PROJECTED,
            "projected_name": CRS(PROJECTED).name,
            "projected_area_of_use": str(CRS(PROJECTED).area_of_use),
        },
        "hugging_face_dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "manifest_url": MANIFEST_URL,
            "manifest_sha256_expected": MANIFEST_SHA256,
            "manifest_sha256_observed": sha256(args.manifest),
            "manifest_bytes": args.manifest.stat().st_size,
            "license_declared_by_dataset": "CC BY-SA 4.0",
        },
        "existing_quality_filter": {
            "source_layer": "descriptions/geoai/v1",
            "source_file_sha256": sha256(args.quality),
            "source_file": args.quality.name,
            "main_analysis_rule": "Include only records with existing usable=true labels.",
            "unknown_quality_rule": "Do not classify missing labels. Exclude them from the confirmed-usable analytical cohort.",
            "summary": quality_labels_report,
        },
        "fatih_boundary": {
            "source": "Nominatim lookup for OpenStreetMap relation",
            "relation_id": BOUNDARY_RELATION_ID,
            "display_name": boundary_feature.get("display_name"),
            "source_sha256": sha256(args.boundary),
        },
        "openstreetmap_network": {
            "source": "Overpass API",
            "osm_snapshot_utc": json.loads(args.streets.read_text(encoding="utf-8")).get("osm3s", {}).get("timestamp_osm_base"),
            "source_sha256": sha256(args.streets),
            "network_filter": sorted(ELIGIBLE_HIGHWAYS),
            "excluded_access_values": sorted(EXCLUDED_ACCESS_VALUES),
            "excluded_service_values": sorted(EXCLUDED_SERVICE_VALUES),
            "license": "Open Database License 1.0 as stated by the Overpass response",
        },
        "parameters": {
            "grid_size_m": GRID_SIZE_M,
            "primary_match_threshold_m": PRIMARY_MATCH_THRESHOLD_M,
            "primary_local_coverage_distance_m": PRIMARY_COVERAGE_DISTANCE_M,
            "street_unit_length_m": UNIT_LENGTH_M,
            "covered_unit_minimum_fraction": 0.5,
            "near_tie_margin_m": NEAR_TIE_MARGIN_M,
        },
    }
    json_dump(data_dir / "provenance.json", provenance)

    generate_figures(
        figures_dir,
        boundary_projected,
        network_geometry,
        primary_coverage["covered_geometry"],
        primary_coverage["uncovered_geometry"],
        grid,
    )
    expected_figures = [
        "fig1_image_distribution",
        "fig2_network_coverage",
        "fig3_grid_coverage",
        "fig4_gap_density",
    ]
    for stem in expected_figures:
        for suffix in [".pdf", ".png"]:
            if not (figures_dir / f"{stem}{suffix}").exists():
                raise AssertionError(f"Missing expected figure: {stem}{suffix}")
    write_package_documents(out, results, provenance, write_docs=args.write_package_docs)
    print(
        "Analysis complete: "
        f"{quality['clean_in_boundary_images']:,} confirmed-usable in-boundary images, "
        f"{primary['coverage_pct']:.6f}% location-based network coverage."
    )


if __name__ == "__main__":
    main()
