#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yunus Serhat Bıçakçı
"""Extract existing GeoAI quality labels with HTTP byte ranges.

Only the image_id, usable, and parsed_json Parquet columns are requested.
No image bytes, raw responses, segmentation outputs, or vision models are used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests


REVISION = "250dbb1efb580653b98d153dbc77341555d5a258"
BASE_URL = (
    "https://huggingface.co/datasets/yunusserhat/fatih/resolve/"
    f"{REVISION}/data/derived/descriptions/geoai/v1"
)
PARTS = [
    {
        "name": "part-00000.parquet",
        "bytes": 106_978_242,
        "lfs_sha256": "295af3d12420d2dc4887c5bd299c95f11b90edb05f046f2176aa192a8ab954db",
    },
    {
        "name": "part-00001.parquet",
        "bytes": 25_321_098,
        "lfs_sha256": "8f7ed31bd190eece87032e25a73f5061e6bed65c118455d60ceea8f7e726b1ca",
    },
]
COLUMNS = ["image_id", "usable", "parsed_json"]
USER_AGENT = "fatih-spatial-coverage-research/1.0 (academic analysis)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranged_get(url: str, range_header: str) -> tuple[bytes, requests.Response]:
    response = requests.get(
        url, headers={"Range": range_header, "User-Agent": USER_AGENT}, timeout=300
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Expected partial response for {range_header}, got {response.status_code}.")
    return response.content, response


def footer_metadata(url: str) -> tuple[pq.ParquetFile, bytes, int]:
    last, response = ranged_get(url, "bytes=-8")
    if len(last) != 8 or last[-4:] != b"PAR1":
        raise RuntimeError("The remote object is not a valid Parquet file.")
    footer_length = int.from_bytes(last[:4], "little")
    tail, response = ranged_get(url, f"bytes=-{footer_length + 8}")
    content_range = response.headers.get("Content-Range", "")
    try:
        total_bytes = int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"Missing total size in Content-Range: {content_range}") from error
    return pq.ParquetFile(io.BytesIO(tail)), tail, total_bytes


def selected_chunk_ranges(metadata: pq.ParquetFile, columns: list[str]) -> list[tuple[int, int]]:
    names = metadata.schema_arrow.names
    ranges = []
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.metadata.row_group(row_group_index)
        for name in columns:
            column = row_group.column(names.index(name))
            start = column.dictionary_page_offset or column.data_page_offset
            end = start + column.total_compressed_size
            ranges.append((int(start), int(end)))
    merged = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def read_selected_columns(url: str) -> tuple[pd.DataFrame, dict]:
    parquet, footer_tail, total_bytes = footer_metadata(url)
    ranges = selected_chunk_ranges(parquet, COLUMNS)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as stream:
        sparse_path = Path(stream.name)
        stream.write(b"PAR1")
        for start, end in ranges:
            chunk, _ = ranged_get(url, f"bytes={start}-{end - 1}")
            if len(chunk) != end - start:
                raise RuntimeError("Returned chunk length does not match requested Parquet range.")
            stream.seek(start)
            stream.write(chunk)
        stream.seek(total_bytes - len(footer_tail))
        stream.write(footer_tail)
    try:
        table = pq.read_table(sparse_path, columns=COLUMNS)
    finally:
        sparse_path.unlink(missing_ok=True)
    info = {
        "rows": int(table.num_rows),
        "remote_bytes": total_bytes,
        "selected_range_count": len(ranges),
        "selected_bytes": int(sum(end - start for start, end in ranges)),
        "schema": str(table.schema),
    }
    return table.to_pandas(), info


def parse_quality(value) -> tuple[object, str, str]:
    """Return nested usability, a canonical issue-set string, and parse status."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, "", "missing_parsed_json"
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return None, "", "invalid_parsed_json"
    quality = parsed.get("image_quality") if isinstance(parsed, dict) else None
    if not isinstance(quality, dict):
        return None, "", "missing_image_quality"
    nested_usable = quality.get("usable_for_analysis")
    issues = quality.get("issues")
    if issues is None:
        issue_values = []
    elif isinstance(issues, list):
        issue_values = [str(issue).strip() for issue in issues if str(issue).strip()]
    else:
        issue_values = [str(issues).strip()] if str(issues).strip() else []
    return nested_usable, json.dumps(sorted(set(issue_values)), ensure_ascii=False), "parsed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tables = []
    part_info = []
    for part in PARTS:
        url = f"{BASE_URL}/{part['name']}"
        table, info = read_selected_columns(url)
        table["source_part"] = part["name"]
        tables.append(table)
        part_info.append({**part, "url": url, **info})
    labels = pd.concat(tables, ignore_index=True)
    if labels["image_id"].duplicated().any():
        raise RuntimeError("GeoAI quality labels contain duplicate image_id values.")

    parsed = labels["parsed_json"].map(parse_quality)
    labels["parsed_quality_usable"] = [row[0] for row in parsed]
    labels["quality_issues_json"] = [row[1] for row in parsed]
    labels["quality_parse_status"] = [row[2] for row in parsed]
    labels = labels.drop(columns="parsed_json")
    labels["image_id"] = labels["image_id"].astype("string")
    labels.to_parquet(args.output, index=False)

    direct = labels["usable"]
    nested = labels["parsed_quality_usable"]
    provenance = {
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": REVISION,
        "analysis_layer": "descriptions/geoai/v1",
        "selected_columns": COLUMNS,
        "parts": part_info,
        "output_rows": int(len(labels)),
        "output_unique_image_ids": int(labels["image_id"].nunique()),
        "usable_true": int(direct.eq(True).sum()),
        "usable_false": int(direct.eq(False).sum()),
        "usable_missing": int(direct.isna().sum()),
        "parsed_quality_usable_true": int(nested.eq(True).sum()),
        "parsed_quality_usable_false": int(nested.eq(False).sum()),
        "direct_nested_usable_disagreement": int(
            (direct.notna() & nested.notna() & direct.ne(nested)).sum()
        ),
        "quality_parse_status_counts": {
            key: int(value)
            for key, value in labels["quality_parse_status"].value_counts(dropna=False).items()
        },
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
