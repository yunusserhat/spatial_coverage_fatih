# Methods

## Scope and source version

The analysis used the Hugging Face dataset `yunusserhat/fatih` at commit `250dbb1efb580653b98d153dbc77341555d5a258`. The required source manifest was `data/raw/manifest/train.parquet`, whose SHA-256 is `75d8f360637e4f8a0411a162ec385d86569f772f95c7ada64d173c5597f75927`. The existing GeoAI description layer supplied its image identifier, usability, and nested image-quality fields through byte-range retrieval of only needed metadata columns. The manifest has one row per stated image identifier. Raw WebDataset image shards, segmentation masks, raw VLM responses, scene descriptions, and perception scores were excluded from the workflow.

The boundary is OpenStreetMap administrative relation `1766104` for Fatih. The cached Nominatim response identifies it as an administrative boundary with `admin_level=6`. The OSM network source is the cached Overpass response listed in `data/source/eligible_streets_overpass.json`. Its query, response SHA-256, snapshot timestamp, and retrieval date are in `data/provenance.json`.

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

Each confirmed-usable in-boundary image point is assigned to its nearest eligible street geometry using a Shapely spatial index. The primary match threshold is 20 m. Ties are resolved by distance and then stable edge identifier. The process records nearest distance, projected position along the matched line, distance to line endpoints, all candidate count within threshold, and a near-tie flag. A near tie has a second candidate within 2 m of the nearest-candidate distance. These cases are retained under the deterministic rule and are reported rather than silently discarded.

For an image matched at along-line position \(s\) on an edge of length \(L\), a local interval is formed from `max(0, s-r)` to `min(L, s+r)`, where \(r\) is the local coverage distance. The primary value is \(r=25\) m. Intervals are merged on each source edge and then intersected with the noded, non-overlapping network atoms. Along-atom intervals are merged. Covered and uncovered portions are formed as complementary intervals on each atom, and their lengths are summed across atoms. The code asserts that these two values reconcile with the total eligible network length. This method prevents overlapping images from double counting and does not assign a whole long OSM way to one image.

The primary result is supplemented by 10 m units. A unit is covered only if its covered fraction is at least 0.5. Exact covered length remains the primary metric.

## Grid and sensitivity analysis

Reliable official neighbourhood polygon data was not available in a documented, reproducible form for this run. The area is therefore partitioned into 500 m square cells, clipped to Fatih. This resolution produces a readable city-scale map while retaining local variation. For each cell, the analysis calculates image count, clipped cell area, image density per km², eligible street length, covered street length, coverage percentage, images per street kilometre, and uncovered street km per km². Lines are intersected with cells, not allocated by centroids.

Sensitivity tests vary the match threshold with a fixed 25 m local coverage distance at 10, 20, and 30 m. They also vary the local coverage distance with a fixed 20 m match threshold at 15, 25, and 50 m. A conservative diagnostic excludes near-tie matches under the primary thresholds. A comparison reruns the primary rule with all valid in-boundary locations before the existing usability filter. Assertions check quality-status reconciliation, covered and uncovered length reconciliation, percentage bounds, grid allocation, and expected monotonicity of coverage across local-distance sensitivity values.

## Spatial statistics

Moran's I and Local Moran statistics were not run. The evidence needed for the research question is descriptive and geographic. Grid-based inferential clustering would depend on an arbitrary cell partition and would invite causal interpretation of source-data gaps that this dataset cannot support.
