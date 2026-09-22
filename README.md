# Spatial coverage of open street-level imagery in Fatih, Istanbul

Code and compact data for a reproducible audit of how well open street-level image locations cover the street network of Fatih, Istanbul. The study separates **image abundance** from **street-network coverage**. Image points are matched to eligible OpenStreetMap streets, each match supports a short interval along its street, and overlapping intervals are merged before covered length is measured.

The accompanying article is currently under review. Until it is published, please cite this repository and the image dataset (see [Citation](#citation)).

## Main results

| Measure | Value |
|---|---|
| Confirmed-usable image locations in the district | 116,464 (7,167.7 per km²) |
| Eligible street network | 443.40 km |
| Covered length (match ≤ 20 m, local interval 25 m) | 185.61 km, **41.9%** |
| Coverage on major roads / residential and local streets | 85.5% / 24.0% |
| Coverage range across the 57 neighbourhoods | 3.8% to 84.1% |
| All public Mapillary images in the district (snapshot, 22 September 2026) | 124,033, of which 99.8% are in the analysed dataset, 42.4% coverage |
| Range under alternative distance rules, headings, and a 2019 network | 38.1% to 49.0% |

Every number is traced to its output file in [docs/results.md](docs/results.md). Methods are in [docs/methods.md](docs/methods.md) and interpretation limits in [docs/limitations.md](docs/limitations.md).

## Repository layout

| Path | Contents |
|---|---|
| `code/analysis.py` | Primary analysis: quality filter, boundary filter, eligible network, nearest-street matching, exact interval-union coverage, 500 m grid, sensitivity tests |
| `code/extended_analysis.py` | Road class coverage, image thinning, temporal accumulation, 10 m bin concentration, study area map |
| `code/robustness_analysis.py` | Platform comparison, heading-based coverage, neighbourhood summaries, betweenness classes, sequence thinning, grid size sensitivity, end-of-2019 network check |
| `code/fetch_sources.py` | Downloads the pinned manifest (SHA-256 checked) and, optionally, fresh OpenStreetMap sources |
| `code/extract_geoai_quality.py` | Extracts the existing usability labels from the dataset repository |
| `code/fetch_mapillary_platform.py` | Snapshots all public Mapillary image locations in the district |
| `code/test_coverage_logic.py` | Small checks of the interval-union rule |
| `data/source/` | Cached inputs: boundary, street networks (2026 and end of 2019), neighbourhood relations, quality labels, Mapillary platform snapshot |
| `data/` | Derived results: GeoJSON layers, grid and neighbourhood results, JSON summaries |
| `tables/` | Result tables in CSV form |
| `figures/` | Figures used in the article, as PDF and 400 dpi PNG |

Figure files and article figure numbers:

| Article | File |
|---|---|
| Figure 1 | `figures/fig0_study_area` |
| Figure 2 | `figures/fig1_image_distribution_log` |
| Figure 3 | `figures/fig2_network_coverage` |
| Figure 4 | `figures/fig4_grid_pair` |
| Figure 5 | `figures/fig5_neighbourhoods` |
| Figure 6 | `figures/fig6_saturation_temporal` |

## Reproduction

Python 3.12 or later is required.

```bash
python -m venv .venv
.venv/bin/pip install -r code/requirements.txt

# 1. Pinned image manifest (about 95 MB, checksum verified). Not stored in this repository.
.venv/bin/python code/fetch_sources.py --output inputs --manifest-only

# 2. Primary analysis from the cached sources. Regenerates the large parquet outputs
#    that are not stored in this repository.
.venv/bin/python code/analysis.py \
  --manifest inputs/manifest_train.parquet \
  --quality data/source/geoai_quality_labels.parquet \
  --boundary data/source/fatih_boundary_nominatim.json \
  --streets data/source/eligible_streets_overpass.json \
  --out .
.venv/bin/python code/test_coverage_logic.py

# 3. Extended analyses. District boundaries come from the OCHA COD-AB dataset on HDX.
curl -L -o inputs/tur_admin_boundaries.geojson.zip \
  https://data.humdata.org/dataset/d74086a0-f398-4474-9e12-1b9a70907bd0/resource/470bd810-2240-4ce0-b5c4-17434112ce41/download/tur_admin_boundaries.geojson.zip
.venv/bin/python code/extended_analysis.py --admin-zip inputs/tur_admin_boundaries.geojson.zip

# 4. Robustness analyses (about 20 minutes).
.venv/bin/python code/robustness_analysis.py \
  --manifest inputs/manifest_train.parquet \
  --streets-2019 data/source/eligible_streets_overpass_2019-12-31.json
```

Notes:

- Both supplementary scripts stop if they cannot reproduce the primary covered length of 185,610.191 m.
- The cached sources in `data/source/` preserve the exact snapshots used in the article. `fetch_sources.py` without `--manifest-only`, and `fetch_mapillary_platform.py`, retrieve the **current** state of OpenStreetMap and Mapillary and will overwrite those snapshots. `fetch_mapillary_platform.py` needs a Mapillary client token in `MAPILLARY_ACCESS_TOKEN` and does not store contributor identifiers.
- `analysis.py --write-package-docs` also writes the original briefing documents (including a README.md) into `--out`. Use a separate output directory if you need them.
- `extract_geoai_quality.py` is only needed to rebuild `data/source/geoai_quality_labels.parquet` from the dataset repository.

## Interpretation boundary

Coverage here means location-based support of the eligible street network under explicit distance rules. It is not visual coverage, it does not establish camera field of view or visibility, and it does not explain why gaps occur. The grid and the OpenStreetMap neighbourhood relations are reporting units only.

## Data sources and licences

The GNU GPL v3 licence of this repository applies to the source code and the documentation written for it. It does not relicense third-party data, which keep their own terms:

| Data | Source | Licence |
|---|---|---|
| Boundary, street networks, neighbourhood relations, and all derived GeoJSON layers | © OpenStreetMap contributors | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/) |
| Image manifest, quality labels, and Mapillary platform snapshot (image identifiers, locations, capture times, headings, sequence identifiers) | [yunusserhat/fatih](https://doi.org/10.57967/hf/10144) dataset and [Mapillary](https://www.mapillary.com) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| District boundaries used in Figure 1 (not stored here) | OCHA COD-AB Türkiye via the [Humanitarian Data Exchange](https://data.humdata.org/dataset/cod-ab-tur) | CC BY-IGO |

Figures and tables combine these sources. When reusing them, attribute OpenStreetMap contributors and Mapillary contributors as above.

No street-level image content, segmentation masks, model outputs, or contributor identifiers are included.

## Citation

Image dataset:

> Bıçakçı, Y. S. (2026). *fatih* (Revision 250dbb1) [Data set]. Hugging Face. https://doi.org/10.57967/hf/10144

This repository: see [CITATION.cff](CITATION.cff) or use the "Cite this repository" button on GitHub.

## Licence

Copyright (C) 2026 Yunus Serhat Bıçakçı.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. See [LICENSE](LICENSE) for details.
