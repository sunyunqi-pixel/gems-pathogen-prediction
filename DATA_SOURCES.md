# Data sources and why raw data are not in this repository

The GEMS and GEMS1A datasets are not redistributed here. Doing so would conflict with
the terms under which they are made available, and would risk re-identification in a
paediatric cohort. This repository therefore contains code plus **derived, aggregate or
model-output** files only.

## How to obtain the inputs

| Input | Source | Notes |
|---|---|---|
| GEMS / GEMS1A | ClinEpiDB <https://clinepidb.org> | open-access exploration and download |
| GEMS repository | University of Maryland <https://www.medschool.umaryland.edu/gems/> | data, stool and bacterial samples |
| ERA5-Land | Copernicus CDS <https://cds.climate.copernicus.eu> | 2-m temperature, 2-m dewpoint, total precipitation |
| CHIRPS v2.0 | Climate Hazards Center <https://www.chc.ucsb.edu/data/chirps> | primary daily precipitation series |

## What is derived rather than raw

Everything under `derived/` is either a model output (out-of-fold predictions, weights,
hyperparameters, SHAP values) or an aggregate summary. These contain no participant
identifiers beyond the facility and country labels that are already public in the GEMS
study design.

## Rebuilding the derived inputs

1. Obtain the GEMS/GEMS1A release from ClinEpiDB and place it under `data/raw/`
   (excluded by `.gitignore`).
2. Download ERA5-Land and CHIRPS per the bounding boxes in the manuscript's
   Additional file 6, and place the archives under `data/raw/`.
3. Run the pipeline in the order given in `README.md`. The first four scripts rebuild
   the episode master and the site-day climate series; the remaining scripts reproduce
   every number in the manuscript.
