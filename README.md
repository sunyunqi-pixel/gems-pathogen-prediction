# Pathogen-specific prediction of childhood diarrhoeal aetiology from climate and contextual exposures

Analysis code and derived results for a pathogen-specific prediction study using the
Global Enteric Multicenter Study (GEMS) and its GEMS1A follow-on phase, linked to
ERA5-Land and CHIRPS climate exposures.

Repository: <https://github.com/sunyunqi-pixel/gems-pathogen-prediction>

## What this repository contains

| Folder | Contents |
|---|---|
| `code/` | Full analysis pipeline, from record assembly to figures |
| `derived/oof/` | Out-of-fold predictions used for every reported metric |
| `derived/folds/` | Optimised ensemble weights per cross-validation fold |
| `derived/params/` | Tuned hyperparameters per pathogen and validation regime |
| `derived/summary/` | Aggregated result tables (AUC, AP, calibration, SHAP, sensitivity) |
| `derived/subgroup/` | Case-only / control-only subgroup outputs and the case_control label file |
| `figures/` | Scripts and outputs for the main-text figures |

## Data sources (not redistributed here)

**We do not redistribute the GEMS or GEMS1A data.** They are available from the
official sources below, and this repository contains the code needed to rebuild
every derived dataset from them.

- **GEMS / GEMS1A** - ClinEpiDB: <https://clinepidb.org>
  and the GEMS Repository, University of Maryland School of Medicine:
  <https://www.medschool.umaryland.edu/gems/>
- **ERA5-Land** reanalysis - Copernicus Climate Change Service Climate Data Store:
  <https://cds.climate.copernicus.eu>
- **CHIRPS v2.0** precipitation - Climate Hazards Center:
  <https://www.chc.ucsb.edu/data/chirps>

Users are responsible for complying with the terms of access of each provider.

## Environment

Python 3.14.3. Install the pinned environment with:

```bash
pip install -r requirements.txt
```

Key package versions are listed in `requirements.txt`. The analysis was run with
fixed random seeds (42) for cross-validation splitting, the Optuna sampler, and all
learners.

## Pipeline order

Run in this order from the repository root:

```bash
python code/03_build_episode_master.py            # episode-level analysis master
python code/03c_qc_site_country.py                # site and country reconciliation
python code/05_build_site_day_climate_base.py     # site-day climate base series
python code/05a_extract_and_inventory_era5_archives.py
python code/07b_build_pathogen_analytic_sets.py   # pathogen-specific analytic sets
python code/ensemble_cv_compare.py --all-pathogens --cv-mode random  --trials 200 --drop-behavior --tag-suffix _clean
python code/ensemble_cv_compare.py --all-pathogens --cv-mode site    --trials 200 --drop-behavior --tag-suffix _clean
python code/ensemble_cv_compare.py --all-pathogens --cv-mode country --trials 200 --drop-behavior --tag-suffix _clean
python code/run_final_shap_calibration.py         # calibration, CI, SHAP summaries
python code/final_CI.py                           # subgroup (case-only / control-only) AUCs and CIs
python code/make_figs.py                          # main-text figures
```

## Reproducing the reported results

| Manuscript result | Source in this repository |
|---|---|
| Table 1 (analytic samples) | `derived/summary/`, analytic-set QC files |
| Table 2 (AUC / AP / delta AUC) | `derived/oof/` |
| Table 3 (model comparison) | `derived/summary/`, per-regime result JSON |
| Table 4 (calibration) | `derived/summary/calibration_summary_site_clean.csv` |
| Table 5 (subgroups) | `derived/subgroup/` (incl. `case_control.csv`) |
| Figure 1 (record flow) | `derived/summary/` missing-reason tables |
| Figures 2-5 | `derived/oof/`, `derived/summary/` |
| Sensitivity analyses | `derived/summary/` ensemble and delta tables |

## Licence

Code is released under the MIT Licence (see `LICENSE`). Derived data files are released under CC BY 4.0.
See `DATA_SOURCES.md` for how to obtain the underlying study and climate data.

## Citation

If you use this code, please cite the archived release (see `CITATION.cff`) and the
associated manuscript.
