#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yunus Serhat Bıçakçı
"""Supplementary analyses that extend the primary Fatih coverage audit.

The script reuses the primary network construction and exact interval-union
coverage from analysis.py. It reads the derived match table written by the
primary run, so it adds no new matching rule. Outputs:

* coverage by OpenStreetMap road class, computed on the noded network atoms;
* a saturation curve from random subsamples of matched image locations;
* cumulative and recent-period coverage by capture year;
* concentration of matched images across 10 m street bins and sequences;
* a two-panel study area figure.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.ops import unary_union

import analysis as A


ROAD_CLASS_GROUPS = {
    "Major roads": {
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
    },
    "Tertiary roads": {"tertiary", "tertiary_link"},
    "Residential and local streets": {"residential", "unclassified", "living_street", "road"},
    "Service roads": {"service"},
    "Pedestrian streets": {"pedestrian"},
}
ROAD_CLASS_ORDER = list(ROAD_CLASS_GROUPS)
ROAD_CLASS_COLOURS = {
    "Major roads": "#B2182B",
    "Tertiary roads": "#EF8A62",
    "Residential and local streets": "#4D4D4D",
    "Service roads": "#67A9CF",
    "Pedestrian streets": "#1B7837",
}
SUBSAMPLE_FRACTIONS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SUBSAMPLE_REPEATS = 10
RANDOM_SEED = 20260922
BIN_LENGTH_M = 10.0


def road_class(highway: str) -> str:
    for group, members in ROAD_CLASS_GROUPS.items():
        if highway in members:
            return group
    raise ValueError(f"Unexpected highway value: {highway}")


def atom_road_classes(streets: gpd.GeoDataFrame, network_context: dict) -> np.ndarray:
    """Assign each noded atom the class of the eligible street nearest its midpoint."""
    tree = network_context_street_tree(streets)
    midpoints = [atom.interpolate(0.5, normalized=True) for atom in network_context["atoms"]]
    nearest = tree.query_nearest(midpoints, return_distance=False, all_matches=False)
    classes = streets["highway"].map(road_class).to_numpy()
    result = np.empty(len(midpoints), dtype=object)
    result[nearest[0]] = classes[nearest[1]]
    return result


def network_context_street_tree(streets: gpd.GeoDataFrame):
    from shapely.strtree import STRtree

    return STRtree(streets.geometry.to_numpy())


def covered_length_by_atom(network_context: dict, covered_atom_intervals: dict) -> np.ndarray:
    lengths = np.zeros(len(network_context["atoms"]), dtype=float)
    for atom_index, intervals in covered_atom_intervals.items():
        lengths[atom_index] = sum(end - start for start, end in intervals)
    return lengths


def road_class_table(streets, network_context, coverage, matches) -> pd.DataFrame:
    atom_classes = atom_road_classes(streets, network_context)
    covered = covered_length_by_atom(network_context, coverage["covered_atom_intervals"])
    frame = pd.DataFrame(
        {"road_class": atom_classes, "length_m": network_context["lengths_m"], "covered_m": covered}
    )
    grouped = frame.groupby("road_class")[["length_m", "covered_m"]].sum()
    image_counts = matches["highway"].map(road_class).value_counts()
    rows = []
    for group in ROAD_CLASS_ORDER:
        length = float(grouped.loc[group, "length_m"])
        covered_length = float(grouped.loc[group, "covered_m"])
        images = int(image_counts.get(group, 0))
        rows.append(
            {
                "road_class": group,
                "eligible_km": length / 1000.0,
                "network_share_pct": A.safe_percent(length, frame["length_m"].sum()),
                "matched_images": images,
                "image_share_pct": A.safe_percent(images, len(matches)),
                "covered_km": covered_length / 1000.0,
                "coverage_pct": A.safe_percent(covered_length, length),
                "images_per_covered_km": images / (covered_length / 1000.0) if covered_length else float("nan"),
            }
        )
    total_length = float(frame["length_m"].sum())
    total_covered = float(frame["covered_m"].sum())
    rows.append(
        {
            "road_class": "All eligible streets",
            "eligible_km": total_length / 1000.0,
            "network_share_pct": 100.0,
            "matched_images": int(len(matches)),
            "image_share_pct": 100.0,
            "covered_km": total_covered / 1000.0,
            "coverage_pct": A.safe_percent(total_covered, total_length),
            "images_per_covered_km": len(matches) / (total_covered / 1000.0),
        }
    )
    table = pd.DataFrame(rows)
    if abs(total_covered - coverage["covered_length_m"]) > 0.05:
        raise AssertionError("Road-class covered lengths do not reconcile with the primary result.")
    return table


def saturation_curve(streets, network_context, matches) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for fraction in SUBSAMPLE_FRACTIONS:
        repeats = 1 if fraction == 1.0 else SUBSAMPLE_REPEATS
        for repeat in range(repeats):
            size = int(round(fraction * len(matches)))
            chosen = rng.choice(len(matches), size=size, replace=False)
            subset = matches.iloc[np.sort(chosen)]
            coverage = A.coverage_from_matches(streets, network_context, subset, A.PRIMARY_COVERAGE_DISTANCE_M)
            rows.append(
                {
                    "fraction": fraction,
                    "repeat": repeat,
                    "images": size,
                    "covered_km": coverage["covered_length_m"] / 1000.0,
                    "coverage_pct": coverage["coverage_pct"],
                }
            )
    return pd.DataFrame(rows)


def temporal_coverage(streets, network_context, matches) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = matches["capture_year"]
    cumulative = []
    for year in range(int(years.min()), int(years.max()) + 1):
        subset = matches.loc[years <= year]
        coverage = A.coverage_from_matches(streets, network_context, subset, A.PRIMARY_COVERAGE_DISTANCE_M)
        cumulative.append(
            {
                "capture_year": year,
                "images_in_year": int((years == year).sum()),
                "cumulative_images": int(len(subset)),
                "cumulative_covered_km": coverage["covered_length_m"] / 1000.0,
                "cumulative_coverage_pct": coverage["coverage_pct"],
            }
        )
    windows = [
        ("2009 to 2019", 2009, 2019),
        ("2020 to 2025", 2020, 2025),
        ("2023 to 2025", 2023, 2025),
        ("2025 only", 2025, 2025),
    ]
    period_rows = []
    for label, start, end in windows:
        subset = matches.loc[years.between(start, end)]
        coverage = A.coverage_from_matches(streets, network_context, subset, A.PRIMARY_COVERAGE_DISTANCE_M)
        period_rows.append(
            {
                "period": label,
                "matched_images": int(len(subset)),
                "covered_km": coverage["covered_length_m"] / 1000.0,
                "coverage_pct": coverage["coverage_pct"],
            }
        )
    return pd.DataFrame(cumulative), pd.DataFrame(period_rows)


def concentration(matches) -> dict:
    bins = matches["edge_index"].astype(str) + ":" + np.floor(matches["position_m"] / BIN_LENGTH_M).astype(int).astype(str)
    counts = bins.value_counts().to_numpy()
    counts_sorted = np.sort(counts)[::-1]
    cumulative = np.cumsum(counts_sorted) / counts_sorted.sum()
    half_index = int(np.searchsorted(cumulative, 0.5)) + 1
    sequence_counts = matches["sequence_id"].value_counts().to_numpy()
    sequence_cumulative = np.cumsum(sequence_counts) / sequence_counts.sum()
    top_decile_sequences = max(1, int(np.ceil(0.1 * len(sequence_counts))))

    def gini(values: np.ndarray) -> float:
        ordered = np.sort(values.astype(float))
        n = len(ordered)
        return float((2 * np.arange(1, n + 1) - n - 1).dot(ordered) / (n * ordered.sum()))

    return {
        "occupied_10m_bins": int(len(counts)),
        "bins_holding_half_of_images": half_index,
        "bins_holding_half_of_images_pct": A.safe_percent(half_index, len(counts)),
        "max_images_in_one_bin": int(counts_sorted[0]),
        "median_images_per_occupied_bin": float(np.median(counts)),
        "gini_images_per_occupied_bin": gini(counts),
        "matched_sequences": int(len(sequence_counts)),
        "top_decile_sequences": top_decile_sequences,
        "top_decile_sequence_image_share_pct": float(100.0 * sequence_cumulative[top_decile_sequences - 1]),
        "largest_sequence_images": int(sequence_counts[0]),
    }


def study_area_figure(figures_dir: Path, admin_zip: Path, boundary_projected, streets) -> None:
    with zipfile.ZipFile(admin_zip) as archive:
        with archive.open("tur_admin2.geojson") as handle:
            districts = gpd.read_file(handle)
    istanbul = districts.loc[districts["adm1_pcode"].eq("TUR034")].to_crs(A.PROJECTED)
    fatih = istanbul.loc[istanbul["adm2_pcode"].eq("TUR034020")]

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "figure.facecolor": "#FFFFFF"})
    figure, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [1.0, 1.25]}, constrained_layout=True
    )
    ax = axes[0]
    istanbul.plot(ax=ax, facecolor="#EFEFEC", edgecolor="#9A9A96", linewidth=0.3)
    fatih.plot(ax=ax, facecolor="#B2182B", edgecolor="#6B0F19", linewidth=0.5)
    min_x, min_y, max_x, max_y = istanbul.total_bounds
    ax.set_xlim(min_x - 2000, max_x + 2000)
    # Extra space above the northernmost point keeps the north arrow off the coastline.
    ax.set_ylim(min_y - 2000, max_y + 19000)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("(a)", loc="left", fontsize=9, fontweight="bold")
    centre = fatih.geometry.iloc[0].centroid
    ax.annotate(
        "Fatih", xy=(centre.x, centre.y), xytext=(centre.x - 16000, centre.y - 22000),
        fontsize=8, arrowprops={"arrowstyle": "-", "color": "#303030", "lw": 0.6},
    )
    arrow_x = min_x + 6000
    ax.annotate(
        "", xy=(arrow_x, max_y + 13000), xytext=(arrow_x, max_y + 3000),
        arrowprops={"arrowstyle": "-|>", "color": "#303030", "lw": 1.0},
    )
    ax.text(arrow_x, max_y + 14000, "N", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#303030")
    x0, y0 = min_x + 4000, min_y + 3000
    for index, colour in enumerate(["#303030", "#FFFFFF"]):
        ax.add_patch(plt.Rectangle(
            (x0 + index * 10000, y0), 10000, 900, facecolor=colour, edgecolor="#303030", linewidth=0.55
        ))
    for offset, label in [(0, "0"), (20000, "20 km")]:
        ax.text(x0 + offset, y0 - 1200, label, ha="center", va="top", fontsize=7, color="#303030")

    ax = axes[1]
    extent = A.map_extent(boundary_projected, margin_m=150.0)
    plot_streets = streets.assign(road_class=streets["highway"].map(road_class))
    widths = {
        "Major roads": 1.1, "Tertiary roads": 0.8, "Residential and local streets": 0.35,
        "Service roads": 0.35, "Pedestrian streets": 0.6,
    }
    for group in ["Residential and local streets", "Service roads", "Pedestrian streets", "Tertiary roads", "Major roads"]:
        subset = plot_streets.loc[plot_streets["road_class"].eq(group)]
        subset.plot(ax=ax, color=ROAD_CLASS_COLOURS[group], linewidth=widths[group], zorder=3)
    A.base_map(ax, boundary_projected, extent)
    ax.set_facecolor("#FFFFFF")
    A.add_north_arrow(ax)
    A.add_scale_bar(ax, extent)
    ax.set_title("(b)", loc="left", fontsize=9, fontweight="bold")
    handles = [
        mpl.lines.Line2D([0], [0], color=ROAD_CLASS_COLOURS[g], lw=2.0, label=g) for g in ROAD_CLASS_ORDER
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
        frameon=False, fontsize=7,
    )
    A.save_figure(figure, figures_dir, "fig0_study_area")


def saturation_figure(figures_dir: Path, curve: pd.DataFrame, cumulative: pd.DataFrame) -> None:
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "figure.facecolor": "#FFFFFF"})
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)

    ax = axes[0]
    summary = curve.groupby("images")["coverage_pct"].agg(["mean", "min", "max"]).reset_index()
    ax.fill_between(summary["images"] / 1000.0, summary["min"], summary["max"], color="#9ECAE1", alpha=0.6, lw=0)
    ax.plot(summary["images"] / 1000.0, summary["mean"], color="#08519C", lw=1.4, marker="o", ms=2.5)
    ax.set_xlabel("Matched image locations retained (thousands)")
    ax.set_ylabel("Eligible network covered (%)")
    ax.set_ylim(0, 50)
    ax.grid(color="#E5E5E5", lw=0.5)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    bars = ax.bar(cumulative["capture_year"], cumulative["images_in_year"] / 1000.0, color="#BDBDBD", width=0.75)
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
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    A.save_figure(figure, figures_dir, "fig5_saturation_temporal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--admin-zip", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    data_dir, tables_dir, figures_dir = package / "data", package / "tables", package / "figures"

    _, _, boundary_projected = A.read_boundary(data_dir / "source" / "fatih_boundary_nominatim.json")
    streets, _ = A.read_eligible_streets(data_dir / "source" / "eligible_streets_overpass.json", boundary_projected)
    network_context = A.build_network_context(unary_union(list(streets.geometry)))

    matches = pd.read_parquet(data_dir / "image_match_summary.parquet")
    if not (streets["edge_id"].to_numpy()[matches["edge_index"].to_numpy()] == matches["edge_id"].to_numpy()).all():
        raise AssertionError("Edge indices in the match table do not align with the rebuilt network.")
    metadata = pd.read_parquet(
        data_dir / "clean_image_metadata.parquet", columns=["image_id", "sequence_id", "captured_at"]
    )
    metadata["image_id"] = metadata["image_id"].astype(str)
    matches = matches.merge(metadata, on="image_id", how="left", validate="one_to_one")
    matches["capture_year"] = pd.to_datetime(matches["captured_at"], unit="ms", utc=True).dt.year
    primary = matches.loc[matches["nearest_distance_m"] <= A.PRIMARY_MATCH_THRESHOLD_M].reset_index(drop=True)

    coverage = A.coverage_from_matches(streets, network_context, primary, A.PRIMARY_COVERAGE_DISTANCE_M)
    if abs(coverage["covered_length_m"] - 185610.191) > 0.01:
        raise AssertionError("Primary coverage does not reproduce the archived result.")

    classes = road_class_table(streets, network_context, coverage, primary)
    classes.to_csv(tables_dir / "road_class_coverage.csv", index=False)
    curve = saturation_curve(streets, network_context, primary)
    curve.to_csv(tables_dir / "saturation_curve.csv", index=False)
    cumulative, periods = temporal_coverage(streets, network_context, primary)
    cumulative.to_csv(tables_dir / "temporal_cumulative_coverage.csv", index=False)
    periods.to_csv(tables_dir / "temporal_period_coverage.csv", index=False)
    concentration_summary = concentration(primary)

    summary = curve.groupby("fraction")["coverage_pct"].agg(["mean", "min", "max"])
    results = {
        "primary_covered_km": coverage["covered_length_m"] / 1000.0,
        "road_classes": classes.to_dict(orient="records"),
        "saturation_mean_coverage_pct": {str(k): float(v) for k, v in summary["mean"].items()},
        "saturation_range_pct": {str(k): [float(a), float(b)] for k, a, b in zip(summary.index, summary["min"], summary["max"])},
        "temporal_cumulative": cumulative.to_dict(orient="records"),
        "temporal_periods": periods.to_dict(orient="records"),
        "concentration": concentration_summary,
        "random_seed": RANDOM_SEED,
        "subsample_repeats": SUBSAMPLE_REPEATS,
    }
    A.json_dump(data_dir / "extended_results.json", results)

    study_area_figure(figures_dir, args.admin_zip, boundary_projected, streets)
    saturation_figure(figures_dir, curve, cumulative)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
