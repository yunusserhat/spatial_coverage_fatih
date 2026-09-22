#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yunus Serhat Bıçakçı
"""Robustness and context analyses for the Fatih coverage audit.

Reuses the primary network construction, nearest-street matching, and exact
interval-union coverage from analysis.py. Outputs:

* coverage from a complete snapshot of public Mapillary images in Fatih;
* direction-aware coverage using the camera heading relative to the street;
* coverage and image density by OpenStreetMap neighbourhood (mahalle);
* coverage by length-weighted street betweenness class;
* sequence-level saturation curve;
* grid size sensitivity (250 m, 500 m, 1000 m);
* optional coverage against an end-of-2019 OpenStreetMap street network;
* revised figures for image density, grid results, neighbourhoods, and saturation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr
from shapely import LineString, Point
from shapely.ops import linemerge, polygonize, unary_union

import analysis as A
import extended_analysis as E

RANDOM_SEED = 20260922
DATASET_END_MS = pd.Timestamp("2025-08-17T10:22:43Z").value // 10**6
ALONG_STREET_MAX_DEG = 45.0
BETWEENNESS_SAMPLE = 1500
SEQUENCE_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REPEATS = 10


def coverage(streets, network_context, matches):
    return A.coverage_from_matches(streets, network_context, matches, A.PRIMARY_COVERAGE_DISTANCE_M)


def project_points(frame: pd.DataFrame, lon: str, lat: str) -> gpd.GeoDataFrame:
    points = gpd.GeoDataFrame(frame.copy(), geometry=gpd.points_from_xy(frame[lon], frame[lat]), crs=A.WGS84)
    return points.to_crs(A.PROJECTED).reset_index(drop=True)


# ---------- platform comparison ----------
def platform_rows(streets, network_context, platform: pd.DataFrame, manifest_ids: set[str]) -> list[dict]:
    rows = []
    variants = [
        ("Mapillary platform, all public images", platform),
        ("Mapillary platform, captured by 17 August 2025", platform.loc[platform["captured_at"] <= DATASET_END_MS]),
    ]
    for label, subset in variants:
        points = project_points(subset, "lon", "lat")
        matches, _ = A.nearest_matches(points, streets, A.PRIMARY_MATCH_THRESHOLD_M)
        result = coverage(streets, network_context, matches)
        rows.append(
            {
                "analysis": label,
                "images": int(len(subset)),
                "images_not_in_manifest": int((~subset["image_id"].isin(manifest_ids)).sum()),
                "matched_images": int(len(matches)),
                "covered_km": result["covered_length_m"] / 1000.0,
                "coverage_pct": result["coverage_pct"],
            }
        )
    return rows


# ---------- direction-aware coverage ----------
def street_bearing(line: LineString, position: float) -> float:
    start = line.interpolate(max(position - 2.0, 0.0))
    end = line.interpolate(min(position + 2.0, line.length))
    return math.degrees(math.atan2(end.x - start.x, end.y - start.y)) % 360.0


def direction_rows(streets, network_context, matches: pd.DataFrame, manifest: pd.DataFrame) -> tuple[list[dict], dict]:
    frame = matches.merge(manifest, on="image_id", how="left", validate="one_to_one")
    geometries = streets.geometry.to_numpy()
    bearings = np.array([street_bearing(geometries[e], p) for e, p in zip(frame["edge_index"], frame["position_m"])])
    panoramic = frame["camera_type"].isin(["spherical", "equirectangular"]).to_numpy()
    rows, summary = [], {}
    for heading_field in ["computed_compass_angle", "compass_angle"]:
        heading = frame[heading_field].to_numpy(dtype=float)
        valid = (heading >= 0) & (heading <= 360)
        difference = np.abs((heading - bearings + 180.0) % 360.0 - 180.0)
        folded = np.minimum(difference, 180.0 - difference)
        along = valid & (folded <= ALONG_STREET_MAX_DEG)
        keep = along | panoramic
        result = coverage(streets, network_context, frame.loc[keep])
        summary[heading_field] = {
            "valid_heading_share_pct": A.safe_percent(valid.sum(), len(frame)),
            "along_street_share_pct": A.safe_percent(along.sum(), len(frame)),
            "panoramic_images": int(panoramic.sum()),
            "retained_images": int(keep.sum()),
            "covered_km": result["covered_length_m"] / 1000.0,
            "coverage_pct": result["coverage_pct"],
        }
        if heading_field == "computed_compass_angle":
            rows.append(
                {
                    "analysis": "Along-street or panoramic views only",
                    "images": int(len(frame)),
                    "matched_images": int(keep.sum()),
                    "covered_km": result["covered_length_m"] / 1000.0,
                    "coverage_pct": result["coverage_pct"],
                }
            )
    return rows, summary


# ---------- zones ----------
def neighbourhood_polygons(path: Path, boundary_projected) -> gpd.GeoDataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for element in payload["elements"]:
        lines = [
            LineString([(node["lon"], node["lat"]) for node in member["geometry"]])
            for member in element.get("members", [])
            if member.get("type") == "way" and member.get("role") in ("outer", "") and len(member.get("geometry", [])) > 1
        ]
        polygons = list(polygonize(linemerge(unary_union(lines))))
        if not polygons:
            continue
        rows.append({"name": element["tags"].get("name", ""), "osm_relation": element["id"], "geometry": unary_union(polygons)})
    zones = gpd.GeoDataFrame(rows, crs=A.WGS84).to_crs(A.PROJECTED)
    zones["geometry"] = zones.geometry.intersection(boundary_projected)
    zones = zones.loc[~zones.geometry.is_empty].reset_index(drop=True)
    zones["name"] = zones["name"].str.replace(" Mahallesi", "", regex=False)
    return zones


def summarise_zones(zones: gpd.GeoDataFrame, points, network_geometry, covered_geometry) -> gpd.GeoDataFrame:
    zones = zones.copy()
    zones["zone_id"] = [f"Z{i:03d}" for i in range(1, len(zones) + 1)]
    joined = gpd.sjoin(points[["image_id", "geometry"]], zones[["zone_id", "geometry"]], how="left", predicate="within")
    joined = joined.sort_values(["image_id", "zone_id"]).drop_duplicates("image_id", keep="first")
    zones["image_count"] = zones["zone_id"].map(joined["zone_id"].value_counts()).fillna(0).astype(int)
    zones["area_km2"] = zones.geometry.area / 1e6
    zones["eligible_km"] = A.line_lengths_by_grid(list(A.flatten_lines(network_geometry)), zones) / 1000.0
    zones["covered_km"] = A.line_lengths_by_grid(list(A.flatten_lines(covered_geometry)), zones) / 1000.0
    zones["coverage_pct"] = np.where(zones["eligible_km"] > 0, 100.0 * zones["covered_km"] / zones["eligible_km"], np.nan)
    zones["image_density_per_km2"] = zones["image_count"] / zones["area_km2"]
    zones["images_per_eligible_km"] = zones["image_count"] / zones["eligible_km"]
    return zones


def grid_size_rows(points, network_geometry, covered_geometry, boundary_projected) -> list[dict]:
    rows = []
    for size in [250.0, 500.0, 1000.0]:
        grid = A.make_grid_results(points, network_geometry, covered_geometry, size, boundary_projected)
        cells = grid.loc[grid["eligible_street_length_m"] > 0, "street_coverage_pct"]
        weighted = grid.loc[grid["eligible_street_length_m"] >= 1000.0, "street_coverage_pct"]
        rows.append(
            {
                "grid_size_m": size,
                "cells_with_streets": int(len(cells)),
                "median_coverage_pct": float(cells.median()),
                "p25_coverage_pct": float(cells.quantile(0.25)),
                "p75_coverage_pct": float(cells.quantile(0.75)),
                "cells_with_at_least_1km": int(len(weighted)),
                "median_coverage_pct_cells_1km": float(weighted.median()),
            }
        )
    return rows


# ---------- betweenness ----------
def betweenness_rows(network_context, covered_atom_intervals) -> tuple[list[dict], float]:
    atoms = network_context["atoms"]
    graph = nx.Graph()
    edge_atoms: dict[tuple, list[int]] = {}
    for index, atom in enumerate(atoms):
        start = tuple(round(v, 2) for v in atom.coords[0])
        end = tuple(round(v, 2) for v in atom.coords[-1])
        if start == end:
            continue
        key = (start, end) if start <= end else (end, start)
        edge_atoms.setdefault(key, []).append(index)
        if not graph.has_edge(*key) or graph.edges[key]["length"] > atom.length:
            graph.add_edge(*key, length=atom.length)
    centrality = nx.edge_betweenness_centrality(
        graph, k=min(BETWEENNESS_SAMPLE, graph.number_of_nodes()), weight="length", normalized=True, seed=RANDOM_SEED
    )
    values = np.full(len(atoms), np.nan)
    for (u, v), value in centrality.items():
        key = (u, v) if u <= v else (v, u)
        for index in edge_atoms.get(key, []):
            values[index] = value
    lengths = network_context["lengths_m"]
    covered = E.covered_length_by_atom(network_context, covered_atom_intervals)
    frame = pd.DataFrame({"betweenness": values, "length": lengths, "covered": covered}).dropna()
    frame = frame.sort_values("betweenness").reset_index(drop=True)
    cumulative = frame["length"].cumsum() / frame["length"].sum()
    frame["class"] = np.minimum((cumulative * 5).apply(math.ceil), 5).clip(lower=1)
    rows = []
    for klass, group in frame.groupby("class"):
        rows.append(
            {
                "betweenness_class": int(klass),
                "eligible_km": group["length"].sum() / 1000.0,
                "covered_km": group["covered"].sum() / 1000.0,
                "coverage_pct": A.safe_percent(group["covered"].sum(), group["length"].sum()),
            }
        )
    share_classified = float(frame["length"].sum() / lengths.sum() * 100.0)
    return rows, share_classified


# ---------- sequences ----------
def sequence_curve(streets, network_context, matches) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    sequences = matches["sequence_id"].dropna().unique()
    rows = []
    for repeat in range(REPEATS):
        order = rng.permutation(sequences)
        for fraction in SEQUENCE_FRACTIONS:
            chosen = set(order[: int(round(fraction * len(order)))])
            subset = matches.loc[matches["sequence_id"].isin(chosen)]
            result = coverage(streets, network_context, subset)
            rows.append(
                {
                    "sequence_fraction": fraction,
                    "repeat": repeat,
                    "images": int(len(subset)),
                    "image_share_pct": A.safe_percent(len(subset), len(matches)),
                    "coverage_pct": result["coverage_pct"],
                }
            )
            if fraction == 1.0:
                break
    return pd.DataFrame(rows)


# ---------- historical network ----------
def historical_row(path: Path, boundary_projected, points) -> dict | None:
    if not path.exists():
        return None
    streets_2019, report = A.read_eligible_streets(path, boundary_projected)
    context_2019 = A.build_network_context(unary_union(list(streets_2019.geometry)))
    matches, _ = A.nearest_matches(points, streets_2019, A.PRIMARY_MATCH_THRESHOLD_M)
    result = A.coverage_from_matches(streets_2019, context_2019, matches, A.PRIMARY_COVERAGE_DISTANCE_M)
    return {
        "analysis": "End-of-2019 OpenStreetMap network",
        "network_km": context_2019["total_length_m"] / 1000.0,
        "matched_images": int(len(matches)),
        "covered_km": result["covered_length_m"] / 1000.0,
        "coverage_pct": result["coverage_pct"],
    }


LOG_TICKS = [1000, 2000, 5000, 10000, 20000, 50000]


def tidy_scale_bar(ax):
    """Drop the middle scale label, which collides with the end label in narrow panels."""
    for text in list(ax.texts):
        if text.get_text() == "0.5":
            text.remove()


def plain_log_axis(axis):
    axis.set_major_locator(mpl.ticker.FixedLocator(LOG_TICKS))
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    axis.set_minor_formatter(mpl.ticker.NullFormatter())


# ---------- figures ----------
def style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "figure.facecolor": "#FFFFFF"})


def density_figure(figures_dir, grid, boundary_projected, network_geometry):
    style()
    extent = A.map_extent(boundary_projected)
    figure, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    positive = grid.loc[grid["image_density_per_km2"] > 0]
    empty = grid.loc[grid["image_density_per_km2"] <= 0]
    if len(empty):
        empty.plot(ax=ax, facecolor="#F0F0EE", edgecolor="#D7D7D4", linewidth=0.2, zorder=2)
    positive.plot(
        ax=ax, column="image_density_per_km2", cmap="cividis",
        norm=LogNorm(vmin=max(positive["image_density_per_km2"].min(), 10), vmax=positive["image_density_per_km2"].max()),
        linewidth=0.15, edgecolor="#F4F4F1", legend=True,
        legend_kwds={"label": "Image locations per km² (log scale)", "orientation": "horizontal", "shrink": 0.68, "pad": 0.08},
        zorder=3,
    )
    gpd.GeoSeries(list(A.flatten_lines(network_geometry)), crs=A.PROJECTED).plot(ax=ax, color="#D7D7D4", linewidth=0.24, zorder=4)
    plain_log_axis(figure.axes[-1].xaxis)
    A.base_map(ax, boundary_projected, extent)
    ax.set_facecolor("#FFFFFF")
    A.add_north_arrow(ax)
    A.add_scale_bar(ax, extent)
    A.save_figure(figure, figures_dir, "fig1_image_distribution_log")


def grid_pair_figure(figures_dir, grid, boundary_projected, uncovered_geometry):
    style()
    extent = A.map_extent(boundary_projected)
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), constrained_layout=True)
    has_network = grid.loc[grid["eligible_street_length_m"] > 0]
    no_network = grid.loc[grid["eligible_street_length_m"] <= 0]
    panels = [
        ("street_coverage_pct", "cividis", "Eligible street length covered (%)", {"vmin": 0, "vmax": 100}),
        ("uncovered_street_km_per_km2", "magma", "Uncovered eligible street km per km²", {}),
    ]
    for ax, (column, cmap, label, limits), tag in zip(axes, panels, ["(a)", "(b)"]):
        if len(no_network):
            no_network.plot(ax=ax, facecolor="#F0F0EE", edgecolor="#D7D7D4", linewidth=0.2, zorder=2)
        has_network.plot(
            ax=ax, column=column, cmap=cmap, linewidth=0.2, edgecolor="#F4F4F1", legend=True,
            legend_kwds={"label": label, "orientation": "horizontal", "shrink": 0.8, "pad": 0.04}, zorder=3, **limits,
        )
        if column == "uncovered_street_km_per_km2":
            gpd.GeoSeries(list(A.flatten_lines(uncovered_geometry)), crs=A.PROJECTED).plot(
                ax=ax, color="#3B1A20", linewidth=0.25, alpha=0.4, zorder=4
            )
        A.base_map(ax, boundary_projected, extent)
        ax.set_facecolor("#FFFFFF")
        A.add_north_arrow(ax)
        A.add_scale_bar(ax, extent)
        tidy_scale_bar(ax)
        ax.set_title(tag, loc="left", fontsize=9, fontweight="bold")
    A.save_figure(figure, figures_dir, "fig4_grid_pair")


def neighbourhood_figure(figures_dir, zones, boundary_projected, rho):
    style()
    extent = A.map_extent(boundary_projected, margin_m=150.0)
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.25, 1.0]}, constrained_layout=True)
    ax = axes[0]
    zones.plot(
        ax=ax, column="coverage_pct", cmap="cividis", vmin=0, vmax=100, edgecolor="#FFFFFF", linewidth=0.4, legend=True,
        legend_kwds={"label": "Eligible street length covered (%)", "orientation": "horizontal", "shrink": 0.8, "pad": 0.04},
    )
    A.base_map(ax, boundary_projected, extent)
    ax.set_facecolor("#FFFFFF")
    A.add_north_arrow(ax)
    A.add_scale_bar(ax, extent)
    tidy_scale_bar(ax)
    ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold")
    ax = axes[1]
    ax.scatter(
        zones["image_density_per_km2"], zones["coverage_pct"], s=8 + 3 * zones["eligible_km"],
        color="#08519C", alpha=0.65, edgecolor="#FFFFFF", linewidth=0.4,
    )
    ax.set_xscale("log")
    plain_log_axis(ax.xaxis)
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Image locations per km² (log scale)")
    ax.set_ylabel("Eligible street length covered (%)")
    ax.grid(color="#E5E5E5", lw=0.5)
    ax.text(0.97, 0.04, f"Spearman ρ = {rho:.2f}", ha="right", transform=ax.transAxes, fontsize=7.5)
    ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold")
    A.save_figure(figure, figures_dir, "fig5_neighbourhoods")


def saturation_figure(figures_dir, image_curve, seq_curve, cumulative, n_matches):
    style()
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    ax = axes[0]
    image_summary = image_curve.groupby("images")["coverage_pct"].agg(["mean", "min", "max"]).reset_index()
    x = 100.0 * image_summary["images"] / n_matches
    ax.fill_between(x, image_summary["min"], image_summary["max"], color="#9ECAE1", alpha=0.6, lw=0)
    ax.plot(x, image_summary["mean"], color="#08519C", lw=1.4, marker="o", ms=2.5, label="Random images")
    seq = seq_curve.groupby("sequence_fraction")[["image_share_pct", "coverage_pct"]].mean().reset_index()
    ax.plot(seq["image_share_pct"], seq["coverage_pct"], color="#B2182B", lw=1.4, marker="s", ms=2.5, label="Random capture sequences")
    ax.set_xlabel("Matched image locations retained (%)")
    ax.set_ylabel("Eligible network covered (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.grid(color="#E5E5E5", lw=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold")
    ax = axes[1]
    ax.bar(cumulative["capture_year"], cumulative["images_in_year"] / 1000.0, color="#BDBDBD", width=0.75)
    ax.set_xlabel("Capture year")
    ax.set_ylabel("Matched images captured in year (thousands)")
    ax.grid(axis="y", color="#E5E5E5", lw=0.5)
    twin = ax.twinx()
    twin.plot(cumulative["capture_year"], cumulative["cumulative_coverage_pct"], color="#B2182B", lw=1.4, marker="o", ms=2.5)
    twin.set_ylabel("Cumulative network coverage (%)", color="#B2182B")
    twin.set_ylim(0, 50)
    twin.tick_params(axis="y", colors="#B2182B")
    ax.set_xticks(cumulative["capture_year"][::2])
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold")
    A.save_figure(figure, figures_dir, "fig6_saturation_temporal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--streets-2019", type=Path, default=None)
    args = parser.parse_args()
    package = args.package.resolve()
    data_dir, tables_dir, figures_dir = package / "data", package / "tables", package / "figures"
    source = data_dir / "source"

    _, _, boundary_projected = A.read_boundary(source / "fatih_boundary_nominatim.json")
    streets, _ = A.read_eligible_streets(source / "eligible_streets_overpass.json", boundary_projected)
    network_geometry = unary_union(list(streets.geometry))
    network_context = A.build_network_context(network_geometry)

    matches = pd.read_parquet(data_dir / "image_match_summary.parquet")
    metadata = pd.read_parquet(
        data_dir / "clean_image_metadata.parquet",
        columns=["image_id", "sequence_id", "captured_at", "analysis_lon", "analysis_lat"],
    )
    metadata["image_id"] = metadata["image_id"].astype(str)
    matches = matches.merge(metadata[["image_id", "sequence_id", "captured_at"]], on="image_id", how="left", validate="one_to_one")
    primary = matches.loc[matches["nearest_distance_m"] <= A.PRIMARY_MATCH_THRESHOLD_M].reset_index(drop=True)
    primary_result = coverage(streets, network_context, primary)
    if abs(primary_result["covered_length_m"] - 185610.191) > 0.01:
        raise AssertionError("Primary coverage does not reproduce the archived result.")
    clean_points = project_points(metadata, "analysis_lon", "analysis_lat")

    manifest = pd.read_parquet(args.manifest, columns=["image_id", "compass_angle", "computed_compass_angle", "camera_type"])
    manifest["image_id"] = manifest["image_id"].astype(str)
    platform = pd.read_parquet(source / "mapillary_platform_images.parquet")
    platform["image_id"] = platform["image_id"].astype(str)

    results: dict = {}
    robustness = platform_rows(streets, network_context, platform, set(manifest["image_id"]))
    direction, direction_summary = direction_rows(streets, network_context, primary, manifest)
    robustness += direction
    results["direction"] = direction_summary
    historical = historical_row(args.streets_2019, boundary_projected, clean_points) if args.streets_2019 else None
    if historical:
        robustness.append(historical)
    results["robustness"] = robustness
    pd.DataFrame(robustness).to_csv(tables_dir / "robustness_coverage.csv", index=False)

    zones = neighbourhood_polygons(source / "fatih_neighbourhoods_overpass.json", boundary_projected)
    zones = summarise_zones(zones, clean_points, network_geometry, primary_result["covered_geometry"])
    rho, p_value = spearmanr(zones["image_density_per_km2"], zones["coverage_pct"])
    rho_street, p_street = spearmanr(zones["images_per_eligible_km"], zones["coverage_pct"])
    zones.drop(columns="geometry").to_csv(tables_dir / "neighbourhood_coverage.csv", index=False)
    zones.to_crs(A.WGS84).to_file(data_dir / "neighbourhood_coverage.geojson", driver="GeoJSON")
    ordered = zones.sort_values("coverage_pct")
    results["neighbourhoods"] = {
        "count": int(len(zones)),
        "area_km2_total": float(zones["area_km2"].sum()),
        "eligible_km_total": float(zones["eligible_km"].sum()),
        "median_coverage_pct": float(zones["coverage_pct"].median()),
        "p25_coverage_pct": float(zones["coverage_pct"].quantile(0.25)),
        "p75_coverage_pct": float(zones["coverage_pct"].quantile(0.75)),
        "min": [ordered.iloc[0]["name"], float(ordered.iloc[0]["coverage_pct"])],
        "max": [ordered.iloc[-1]["name"], float(ordered.iloc[-1]["coverage_pct"])],
        "below_30_pct": int((zones["coverage_pct"] < 30).sum()),
        "above_60_pct": int((zones["coverage_pct"] > 60).sum()),
        "spearman_density_coverage": [float(rho), float(p_value)],
        "spearman_images_per_km_coverage": [float(rho_street), float(p_street)],
        "lowest_five": ordered[["name", "coverage_pct", "image_density_per_km2"]].head(5).values.tolist(),
        "highest_five": ordered[["name", "coverage_pct", "image_density_per_km2"]].tail(5).values.tolist(),
        "top_density_five": zones.sort_values("image_density_per_km2", ascending=False)[["name", "coverage_pct", "image_density_per_km2"]].head(5).values.tolist(),
    }

    grid_sizes = grid_size_rows(clean_points, network_geometry, primary_result["covered_geometry"], boundary_projected)
    results["grid_sizes"] = grid_sizes
    pd.DataFrame(grid_sizes).to_csv(tables_dir / "grid_size_sensitivity.csv", index=False)

    betweenness, share = betweenness_rows(network_context, primary_result["covered_atom_intervals"])
    results["betweenness"] = {"classes": betweenness, "length_share_classified_pct": share, "sample_nodes": BETWEENNESS_SAMPLE}
    pd.DataFrame(betweenness).to_csv(tables_dir / "betweenness_coverage.csv", index=False)

    seq_curve = sequence_curve(streets, network_context, primary)
    seq_curve.to_csv(tables_dir / "sequence_saturation_curve.csv", index=False)
    seq_summary = seq_curve.groupby("sequence_fraction")[["image_share_pct", "coverage_pct"]].agg(["mean", "min", "max"]).round(3)
    seq_summary.columns = ["_".join(column) for column in seq_summary.columns]
    results["sequence_curve"] = seq_summary.reset_index().to_dict(orient="records")

    grid = A.make_grid_results(clean_points, network_geometry, primary_result["covered_geometry"], A.GRID_SIZE_M, boundary_projected)
    density_figure(figures_dir, grid, boundary_projected, network_geometry)
    grid_pair_figure(figures_dir, grid, boundary_projected, primary_result["uncovered_geometry"])
    neighbourhood_figure(figures_dir, zones, boundary_projected, rho)
    image_curve = pd.read_csv(tables_dir / "saturation_curve.csv")
    cumulative = pd.read_csv(tables_dir / "temporal_cumulative_coverage.csv")
    saturation_figure(figures_dir, image_curve, seq_curve, cumulative, len(primary))

    A.json_dump(data_dir / "robustness_results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
