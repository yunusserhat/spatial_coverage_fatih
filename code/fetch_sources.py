#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yunus Serhat Bıçakçı
"""Fetch the small source cache and pinned Hugging Face manifest for this study.

The script never requests Mapillary image shards or derived vision layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REVISION = "250dbb1efb580653b98d153dbc77341555d5a258"
MANIFEST_SHA256 = "75d8f360637e4f8a0411a162ec385d86569f772f95c7ada64d173c5597f75927"
MANIFEST_URL = (
    "https://huggingface.co/datasets/yunusserhat/fatih/resolve/"
    f"{REVISION}/data/raw/manifest/train.parquet?download=true"
)
BOUNDARY_URL = "https://nominatim.openstreetmap.org/lookup?" + urlencode(
    {"format": "jsonv2", "polygon_geojson": "1", "osm_ids": "R1766104"}
)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """[out:json][timeout:180];
area(3601766104)->.a;
way(area.a)[\"highway\"~\"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service|pedestrian|road)$\"];
out body geom;"""
USER_AGENT = "fatih-spatial-coverage-research/1.0 (academic analysis)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, data: bytes | None = None) -> None:
    headers = {"User-Agent": USER_AGENT}
    request = Request(url, data=data, headers=headers)
    with urlopen(request, timeout=300) as response, destination.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = args.output / "manifest_train.parquet"
    download(MANIFEST_URL, manifest)
    observed_hash = sha256(manifest)
    if observed_hash != MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest SHA-256 mismatch. Expected {MANIFEST_SHA256}, got {observed_hash}."
        )

    provenance = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": REVISION,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": observed_hash,
    }
    if not args.manifest_only:
        boundary = args.output / "fatih_boundary_nominatim.json"
        streets = args.output / "eligible_streets_overpass.json"
        download(BOUNDARY_URL, boundary)
        download(OVERPASS_URL, streets, OVERPASS_QUERY.encode("utf-8"))
        provenance.update(
            {
                "boundary_url": BOUNDARY_URL,
                "boundary_relation_id": 1766104,
                "overpass_url": OVERPASS_URL,
                "overpass_query": OVERPASS_QUERY,
                "boundary_sha256": sha256(boundary),
                "streets_sha256": sha256(streets),
            }
        )
    (args.output / "fetch_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
