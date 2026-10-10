import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import xarray as xr
except ImportError:
    xr = None


# =========================================================
# basic helpers
# =========================================================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_colname(col: str) -> str:
    if col is None:
        return ""
    c = str(col).strip().lower()
    replacements = [
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        ("\\", "_"),
        (".", "_"),
        ("(", ""),
        (")", ""),
        ("[", ""),
        ("]", ""),
        (":", ""),
        (";", ""),
        (",", ""),
        ("__", "_"),
    ]
    for old, new in replacements:
        c = c.replace(old, new)
    while "__" in c:
        c = c.replace("__", "_")
    return c.strip("_")


def make_unique_columns(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def normalize_string_series(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    x = x.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "none": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "null": pd.NA,
        "NULL": pd.NA
    })
    return x


def canonical_country(x: Optional[str]) -> Optional[str]:
    if x is None or pd.isna(x):
        return pd.NA
    s = str(x).strip()
    key = s.lower()

    mapping = {
        "bangladesh": "Bangladesh",
        "india": "India",
        "pakistan": "Pakistan",
        "kenya": "Kenya",
        "mali": "Mali",
        "mozambique": "Mozambique",
        "gambia": "The Gambia",
        "the gambia": "The Gambia",
        "the_gambia": "The Gambia",
    }
    return mapping.get(key, s)


def detect_text_delimiter(file_path: Path, n_lines: int = 10) -> str:
    candidates = ["\t", "|", ",", ";"]
    scores = {c: 0 for c in candidates}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [next(f, "") for _ in range(n_lines)]

    lines = [x for x in lines if x.strip()]
    if not lines:
        return "\t"

    for cand in candidates:
        splits = [len(line.split(cand)) for line in lines]
        if len(splits) == 0:
            continue
        if max(splits) > 1:
            scores[cand] = sum(splits) / len(splits)

    best = max(scores, key=scores.get)
    return best if scores[best] > 1 else "\t"


def read_table_flexible(file_path: Path) -> pd.DataFrame:
    ext = file_path.suffix.lower()

    if ext == ".parquet":
        return pd.read_parquet(file_path)

    if ext == ".feather":
        return pd.read_feather(file_path)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    delim = detect_text_delimiter(file_path)
    return pd.read_csv(
        file_path,
        sep=delim,
        engine="python",
        dtype="string",
        encoding="utf-8",
        encoding_errors="ignore"
    )


def parse_datetime_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def first_non_null(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return pd.NA
    return s.iloc[0]


# =========================================================
# paths / config
# =========================================================
def resolve_paths(cfg: dict) -> dict:
    project_root = Path(cfg.get("project_root", ".")).resolve()
    raw = cfg.get("raw", {})
    derived = cfg.get("derived", {})

    extracted_root = project_root / derived.get("extracted_dir", "01_extracted")

    return {
        "project_root": project_root,
        "chirps_dir": project_root / raw.get("chirps_dir", "00_raw/CHIRPS"),
        "era5_raw_dir": project_root / raw.get("era5_dir", "00_raw/ERA5/era5_downloads"),
        "era5_extracted_dir": extracted_root / "ERA5",
        "cleaned_dir": project_root / derived.get("cleaned_dir", "02_intermediate/cleaned"),
        "qc_dir": project_root / derived.get("qc_dir", "02_intermediate/qc"),
        "climate_feature_dir": project_root / derived.get("climate_feature_dir", "03_features/climate_features"),
    }


def get_default_gems_site_boxes() -> Dict[str, dict]:
    return {
        "Bangladesh": {
            "site_7": "Bangladesh",
            "lat_min": 23.4, "lat_max": 24.2,
            "lon_min": 90.0, "lon_max": 90.8,
            "study_region_label": "Mirzapur"
        },
        "India": {
            "site_7": "India",
            "lat_min": 22.3, "lat_max": 22.8,
            "lon_min": 88.1, "lon_max": 88.7,
            "study_region_label": "Kolkata"
        },
        "Pakistan": {
            "site_7": "Pakistan",
            "lat_min": 24.75, "lat_max": 25.15,
            "lon_min": 66.85, "lon_max": 67.45,
            "study_region_label": "Karachi"
        },
        "Kenya": {
            "site_7": "Kenya",
            "lat_min": -0.1, "lat_max": 0.4,
            "lon_min": 34.1, "lon_max": 34.7,
            "study_region_label": "Siaya"
        },
        "Mali": {
            "site_7": "Mali",
            "lat_min": 12.45, "lat_max": 12.85,
            "lon_min": -8.2, "lon_max": -7.7,
            "study_region_label": "Bamako"
        },
        "Mozambique": {
            "site_7": "Mozambique",
            "lat_min": -25.7, "lat_max": -25.2,
            "lon_min": 32.5, "lon_max": 33.0,
            "study_region_label": "Manhica"
        },
        "The Gambia": {
            "site_7": "The Gambia",
            "lat_min": 13.15, "lat_max": 13.6,
            "lon_min": -15.0, "lon_max": -14.1,
            "study_region_label": "Basse"
        },
    }


def load_site_boxes_from_config(cfg: dict) -> Dict[str, dict]:
    raw_boxes = cfg.get("gems_site_boxes", None)
    default_boxes = get_default_gems_site_boxes()

    if not isinstance(raw_boxes, dict):
        return default_boxes

    out = {}
    for country, default_spec in default_boxes.items():
        spec = raw_boxes.get(country, {})
        merged = default_spec.copy()
        if isinstance(spec, dict):
            merged.update(spec)
        out[country] = merged

    for country, spec in raw_boxes.items():
        if country not in out and isinstance(spec, dict):
            out[country] = spec

    return out


# =========================================================
# episode master
# =========================================================
def load_episode_master_path(cleaned_dir: Path) -> Path:
    candidates = [
        cleaned_dir / "episode_master_modeling_ready.parquet",
        cleaned_dir / "episode_master.parquet",
        cleaned_dir / "episode_master_enriched.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No episode master parquet found in cleaned_dir.")


def harmonize_episode_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_columns([normalize_colname(c) for c in df.columns])

    if "site_7" not in df.columns:
        if "country" in df.columns:
            df["site_7"] = df["country"]
        elif "site_7_suggested" in df.columns:
            df["site_7"] = df["site_7_suggested"]

    if "country" not in df.columns and "site_7" in df.columns:
        df["country"] = df["site_7"]

    if "event_date" in df.columns:
        df["event_date"] = parse_datetime_series(df["event_date"])

    if "site_7" in df.columns:
        df["site_7"] = normalize_string_series(df["site_7"]).map(canonical_country)

    if "country" in df.columns:
        df["country"] = normalize_string_series(df["country"]).map(canonical_country)

    return df


def derive_target_sites_and_dates(episode_master: pd.DataFrame) -> Tuple[List[str], pd.Timestamp, pd.Timestamp]:
    sites = sorted([x for x in episode_master["site_7"].dropna().unique().tolist()])
    date_min = pd.to_datetime(episode_master["event_date"]).min()
    date_max = pd.to_datetime(episode_master["event_date"]).max()
    return sites, date_min, date_max


# =========================================================
# file scan
# =========================================================
SUPPORTED_NETCDF_EXT = {".nc", ".nc4", ".cdf"}


def scan_chirps_files(chirps_dir: Path) -> pd.DataFrame:
    rows = []
    if not chirps_dir.exists():
        return pd.DataFrame()

    for p in chirps_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_NETCDF_EXT:
            continue
        rows.append({
            "file_path": str(p),
            "file_name": p.name,
            "file_ext": p.suffix.lower(),
            "source_area": "CHIRPS",
            "source_region": "ALL",
            "source_mode": "raw_chirps",
        })
    return pd.DataFrame(rows)


def infer_region_from_zip_name(zip_name: str) -> str:
    z = str(zip_name).lower()
    if "south_asia" in z:
        return "South_Asia"
    if "east_south_africa" in z:
        return "East_South_Africa"
    if "west_africa" in z:
        return "West_Africa"
    return "UNKNOWN"


def scan_era5_extracted_files_from_registry(qc_dir: Path) -> pd.DataFrame:
    reg_path = qc_dir / "qc_era5_extracted_registry.csv"
    if not reg_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(reg_path, dtype="string")
    df.columns = [normalize_colname(c) for c in df.columns]

    if "extracted_path" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["extracted_file_ext"] = work["extracted_file_ext"].astype("string").str.lower()
    work["status"] = work["status"].astype("string").str.lower()

    mask = work["status"].isin(["extracted", "existing_skipped"]) & work["extracted_file_ext"].isin(list(SUPPORTED_NETCDF_EXT))
    work = work.loc[mask].copy()

    rows = []
    for _, row in work.iterrows():
        fp = Path(str(row["extracted_path"]))
        if not fp.exists():
            continue
        zip_name = str(row["source_zip_name"]) if "source_zip_name" in row.index else ""
        rows.append({
            "file_path": str(fp),
            "file_name": fp.name,
            "file_ext": fp.suffix.lower(),
            "source_area": "ERA5",
            "source_region": infer_region_from_zip_name(zip_name),
            "source_mode": "extracted_era5_registry",
        })
    return pd.DataFrame(rows)


# =========================================================
# netcdf helpers
# =========================================================
def detect_coord_name(ds, candidates: List[str]) -> Optional[str]:
    all_names = list(ds.coords) + list(ds.variables)
    norm_map = {normalize_colname(x): x for x in all_names}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    return None


def guess_lat_lon_time_names(ds) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    lat_name = detect_coord_name(ds, ["lat", "latitude", "y"])
    lon_name = detect_coord_name(ds, ["lon", "longitude", "x"])
    time_name = detect_coord_name(ds, ["time", "date", "day", "valid_time"])
    return lat_name, lon_name, time_name


def subset_coord(da, coord_name: str, vmin: float, vmax: float):
    coord_vals = da[coord_name].values
    if len(coord_vals) == 0:
        return da
    if coord_vals[0] <= coord_vals[-1]:
        return da.sel({coord_name: slice(vmin, vmax)})
    return da.sel({coord_name: slice(vmax, vmin)})


def classify_netcdf_var(var_name: str, da) -> Optional[str]:
    nc = normalize_colname(var_name)
    attrs_join = " ".join([str(v) for v in da.attrs.values()]).lower()

    if nc == "tp" or "total precipitation" in attrs_join:
        return "precip_mm"
    if nc == "t2m" or "2 metre temperature" in attrs_join or "2 meter temperature" in attrs_join:
        return "tmean_c"
    if nc == "d2m" or "dew point" in attrs_join or "dewpoint" in attrs_join:
        return "dewpoint_c"
    return None


def convert_netcdf_series_units(series: pd.Series, standard_var: str, var_name: str, source_area: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    vn = normalize_colname(var_name)

    if standard_var == "precip_mm":
        # CHIRPS already daily mm
        # ERA5 tp usually meters per time step
        if source_area == "ERA5":
            if x.notna().any() and float(x.quantile(0.99)) < 2:
                x = x * 1000.0
        return x

    if standard_var in ["tmean_c", "dewpoint_c"]:
        if x.notna().any() and float(x.quantile(0.5)) > 150:
            x = x - 273.15
        return x

    return x


def region_allows_site(region_tag: str, site_7: str) -> bool:
    mapping = {
        "South_Asia": {"Bangladesh", "India", "Pakistan"},
        "West_Africa": {"Mali", "The Gambia"},
        "East_South_Africa": {"Kenya", "Mozambique"},
        "ALL": {"Bangladesh", "India", "Pakistan", "Kenya", "Mali", "Mozambique", "The Gambia"},
        "UNKNOWN": {"Bangladesh", "India", "Pakistan", "Kenya", "Mali", "Mozambique", "The Gambia"},
    }
    allowed = mapping.get(region_tag, mapping["UNKNOWN"])
    return site_7 in allowed


def parse_netcdf_climate_file(
    file_path: Path,
    source_area: str,
    source_region: str,
    site_boxes: Dict[str, dict],
    target_sites: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if xr is None:
        raise ImportError("xarray is required to read NetCDF climate files. Please install xarray.")

    ds = xr.open_dataset(file_path)

    lat_name, lon_name, time_name = guess_lat_lon_time_names(ds)
    if lat_name is None or lon_name is None or time_name is None:
        summary = pd.DataFrame([{
            "file_path": str(file_path),
            "source_area": source_area,
            "source_region": source_region,
            "read_mode": "netcdf",
            "status": "skipped_missing_lat_lon_time",
            "date_column": time_name if time_name is not None else "",
            "n_rows_output": 0
        }])
        return pd.DataFrame(), summary

    candidate_rows = []
    all_frames = []

    for vn in ds.data_vars:
        da = ds[vn]
        std_var = classify_netcdf_var(vn, da)
        if std_var is None:
            continue

        candidate_rows.append({
            "file_path": str(file_path),
            "source_area": source_area,
            "source_region": source_region,
            "read_mode": "netcdf",
            "source_column": vn,
            "standard_variable": std_var,
            "selection_method": "netcdf_var_heuristic"
        })

        for site in target_sites:
            if site not in site_boxes:
                continue
            if not region_allows_site(source_region, site):
                continue

            box = site_boxes[site]
            lat_min = float(box["lat_min"])
            lat_max = float(box["lat_max"])
            lon_min = float(box["lon_min"])
            lon_max = float(box["lon_max"])

            try:
                da_sub = subset_coord(da, lat_name, lat_min, lat_max)
                da_sub = subset_coord(da_sub, lon_name, lon_min, lon_max)

                reduce_dims = [d for d in da_sub.dims if d != time_name]
                if len(reduce_dims) > 0:
                    da_mean = da_sub.mean(dim=reduce_dims, skipna=True)
                else:
                    da_mean = da_sub

                sdf = da_mean.to_dataframe(name=std_var).reset_index()
                if time_name not in sdf.columns:
                    continue

                sdf["datetime"] = pd.to_datetime(sdf[time_name], errors="coerce")
                sdf["date"] = sdf["datetime"].dt.floor("D")
                sdf[std_var] = convert_netcdf_series_units(sdf[std_var], std_var, vn, source_area)

                # ERA5 often 4 times/day -> aggregate to daily
                if source_area == "ERA5":
                    if std_var == "precip_mm":
                        sdf = sdf.groupby("date", dropna=False)[std_var].sum().reset_index()
                    else:
                        sdf = sdf.groupby("date", dropna=False)[std_var].mean().reset_index()
                else:
                    sdf = sdf.groupby("date", dropna=False)[std_var].mean().reset_index()

                sdf["site_7"] = site
                sdf["country"] = site
                sdf["source_area"] = source_area
                sdf["source_region"] = source_region
                sdf["source_file"] = file_path.name
                all_frames.append(sdf)

            except Exception as e:
                logging.warning(f"Failed bbox aggregation for {file_path.name} | {site} | {vn}: {e}")

    if len(all_frames) == 0:
        summary = pd.DataFrame([{
            "file_path": str(file_path),
            "source_area": source_area,
            "source_region": source_region,
            "read_mode": "netcdf",
            "status": "skipped_no_supported_variables",
            "date_column": time_name,
            "n_rows_output": 0
        }])
        return pd.DataFrame(), summary

    out = pd.concat(all_frames, axis=0, ignore_index=True, sort=False)
    out = out.loc[out["date"].notna() & out["site_7"].notna()].copy()

    summary = pd.DataFrame([{
        "file_path": str(file_path),
        "source_area": source_area,
        "source_region": source_region,
        "read_mode": "netcdf",
        "status": "parsed",
        "date_column": time_name,
        "n_rows_output": int(len(out))
    }])

    candidate_df = pd.DataFrame(candidate_rows)
    return out, pd.concat([summary, candidate_df], axis=0, ignore_index=True, sort=False)


# =========================================================
# combine climate
# =========================================================
STANDARD_CLIMATE_VARS = [
    "precip_mm",
    "tmean_c",
    "dewpoint_c",
    "relative_humidity_pct",
]


def derive_relative_humidity_pct(tmean_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    t = pd.to_numeric(tmean_c, errors="coerce")
    td = pd.to_numeric(dewpoint_c, errors="coerce")

    es = np.exp((17.625 * t) / (243.04 + t))
    ed = np.exp((17.625 * td) / (243.04 + td))
    rh = 100.0 * (ed / es)
    rh = rh.clip(lower=0, upper=100)
    return rh


def aggregate_climate_tables(all_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if len(all_frames) == 0:
        raise RuntimeError("No climate frames were parsed successfully.")

    big = pd.concat(all_frames, axis=0, ignore_index=True, sort=False)
    big["date"] = pd.to_datetime(big["date"], errors="coerce").dt.floor("D")
    big["site_7"] = normalize_string_series(big["site_7"]).map(canonical_country)
    big["country"] = normalize_string_series(big["country"]).map(canonical_country)

    big = big.loc[big["date"].notna() & big["site_7"].notna()].copy()

    # Source priority:
    # precip -> CHIRPS first, then ERA5
    # tmean/dewpoint -> ERA5 first
    priority_rows = []
    for _, row in big.iterrows():
        pr = 99
        if row["source_area"] == "CHIRPS":
            pr = 1
        elif row["source_area"] == "ERA5":
            pr = 2
        priority_rows.append(pr)
    big["source_priority"] = priority_rows

    agg_rows = []
    key_cols = ["site_7", "date"]

    for (site, date), sub in big.groupby(key_cols, dropna=False):
        row = {
            "site_7": site,
            "country": first_non_null(sub["country"]),
            "date": date,
        }

        # precip: prefer CHIRPS, fallback ERA5
        precip_chirps = sub.loc[(sub["source_area"] == "CHIRPS") & sub["precip_mm"].notna(), "precip_mm"] if "precip_mm" in sub.columns else pd.Series(dtype=float)
        precip_era5 = sub.loc[(sub["source_area"] == "ERA5") & sub["precip_mm"].notna(), "precip_mm"] if "precip_mm" in sub.columns else pd.Series(dtype=float)
        if len(precip_chirps) > 0:
            row["precip_mm"] = precip_chirps.iloc[0]
            row["precip_source_used"] = "CHIRPS"
        elif len(precip_era5) > 0:
            row["precip_mm"] = precip_era5.iloc[0]
            row["precip_source_used"] = "ERA5"
        else:
            row["precip_mm"] = pd.NA
            row["precip_source_used"] = pd.NA

        # tmean / dewpoint from ERA5
        for v in ["tmean_c", "dewpoint_c"]:
            if v in sub.columns:
                s = sub.loc[(sub["source_area"] == "ERA5") & sub[v].notna(), v]
                row[v] = s.iloc[0] if len(s) > 0 else pd.NA
            else:
                row[v] = pd.NA

        agg_rows.append(row)

    out = pd.DataFrame(agg_rows)

    out["relative_humidity_pct"] = pd.NA
    mask = out["tmean_c"].notna() & out["dewpoint_c"].notna()
    if mask.any():
        out.loc[mask, "relative_humidity_pct"] = derive_relative_humidity_pct(
            out.loc[mask, "tmean_c"],
            out.loc[mask, "dewpoint_c"]
        )

    # placeholders for variables not available in current downloads
    out["tmin_c"] = pd.NA
    out["tmax_c"] = pd.NA
    out["solar_radiation_j_m2"] = pd.NA
    out["solar_radiation_mj_m2"] = pd.NA

    out["year"] = out["date"].dt.year.astype("Int64")
    out["month"] = out["date"].dt.month.astype("Int64")
    out["day"] = out["date"].dt.day.astype("Int64")

    out = out.sort_values(["site_7", "date"]).reset_index(drop=True)
    return out


# =========================================================
# QC
# =========================================================
def build_coverage_qc(climate_base: pd.DataFrame, target_sites: List[str], date_min: pd.Timestamp, date_max: pd.Timestamp) -> pd.DataFrame:
    all_dates = pd.date_range(date_min.floor("D"), date_max.floor("D"), freq="D")
    expected_n_days = len(all_dates)

    rows = []
    for site in target_sites:
        sub = climate_base.loc[climate_base["site_7"] == site].copy()

        row = {
            "site_7": site,
            "expected_n_days_between_event_min_max": int(expected_n_days),
            "n_days_present": int(sub["date"].nunique()) if len(sub) > 0 else 0,
            "min_date_present": sub["date"].min() if len(sub) > 0 else pd.NaT,
            "max_date_present": sub["date"].max() if len(sub) > 0 else pd.NaT,
        }

        for v in [
            "precip_mm", "tmean_c", "tmin_c", "tmax_c",
            "dewpoint_c", "relative_humidity_pct",
            "solar_radiation_j_m2", "solar_radiation_mj_m2"
        ]:
            row[f"n_days_non_missing_{v}"] = int(sub[v].notna().sum()) if v in sub.columns else 0

        rows.append(row)

    return pd.DataFrame(rows)


def build_event_join_check(episode_master: pd.DataFrame, climate_base: pd.DataFrame) -> pd.DataFrame:
    ep = episode_master.copy()
    ep["event_date"] = pd.to_datetime(ep["event_date"], errors="coerce").dt.floor("D")
    ep["site_7"] = normalize_string_series(ep["site_7"]).map(canonical_country)

    cb = climate_base[["site_7", "date"]].drop_duplicates().copy()
    cb["climate_match"] = 1

    chk = ep.merge(cb, left_on=["site_7", "event_date"], right_on=["site_7", "date"], how="left")
    chk["climate_match"] = chk["climate_match"].fillna(0).astype(int)

    rows = []
    rows.append({
        "scope": "overall",
        "site_7": "__ALL__",
        "n_episode_rows": int(len(chk)),
        "n_episode_rows_with_climate_match": int(chk["climate_match"].sum()),
        "pct_episode_rows_with_climate_match": round(100.0 * chk["climate_match"].sum() / len(chk), 3)
    })

    for site, sub in chk.groupby("site_7", dropna=False):
        rows.append({
            "scope": "by_site_7",
            "site_7": site if pd.notna(site) else "__MISSING__",
            "n_episode_rows": int(len(sub)),
            "n_episode_rows_with_climate_match": int(sub["climate_match"].sum()),
            "pct_episode_rows_with_climate_match": round(100.0 * sub["climate_match"].sum() / len(sub), 3) if len(sub) > 0 else pd.NA
        })

    return pd.DataFrame(rows)


# =========================================================
# main
# =========================================================
def main():
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to project_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    site_boxes = load_site_boxes_from_config(cfg)

    ensure_dir(paths["qc_dir"])
    ensure_dir(paths["climate_feature_dir"])

    episode_master_path = load_episode_master_path(paths["cleaned_dir"])
    episode_master = pd.read_parquet(episode_master_path)
    episode_master = harmonize_episode_master(episode_master)

    target_sites, date_min, date_max = derive_target_sites_and_dates(episode_master)
    logging.info(f"Target broad sites: {target_sites}")
    logging.info(f"Event date range: {date_min.date()} to {date_max.date()}")

    chirps_files = scan_chirps_files(paths["chirps_dir"])
    era5_files = scan_era5_extracted_files_from_registry(paths["qc_dir"])

    if len(chirps_files) == 0 and len(era5_files) == 0:
        raise RuntimeError("No CHIRPS or extracted ERA5 climate files found.")

    file_inventory = pd.concat([chirps_files, era5_files], axis=0, ignore_index=True, sort=False)
    logging.info(f"Climate files queued: {len(file_inventory)}")

    all_frames = []
    source_rows = []
    variable_hit_rows = []

    for _, row in file_inventory.iterrows():
        fp = Path(row["file_path"])

        try:
            frame, summary = parse_netcdf_climate_file(
                file_path=fp,
                source_area=row["source_area"],
                source_region=row["source_region"],
                site_boxes=site_boxes,
                target_sites=target_sites
            )

            if len(frame) > 0:
                all_frames.append(frame)

            if len(summary) > 0:
                if "standard_variable" in summary.columns:
                    hit_mask = summary["standard_variable"].notna()
                    if hit_mask.any():
                        variable_hit_rows.append(summary.loc[hit_mask].copy())
                    non_hit = summary.loc[~hit_mask].copy()
                    if len(non_hit) > 0:
                        source_rows.append(non_hit)
                else:
                    source_rows.append(summary)

        except Exception as e:
            logging.warning(f"Failed climate file: {fp.name} | {e}")
            source_rows.append(pd.DataFrame([{
                "file_path": str(fp),
                "source_area": row["source_area"],
                "source_region": row["source_region"],
                "read_mode": "netcdf",
                "status": f"failed_{type(e).__name__}",
                "date_column": "",
                "n_rows_output": 0
            }]))

    climate_base = aggregate_climate_tables(all_frames)

    climate_base = climate_base.loc[
        climate_base["site_7"].isin(target_sites) &
        (climate_base["date"] >= date_min.floor("D")) &
        (climate_base["date"] <= date_max.floor("D"))
    ].copy()

    coverage_qc = build_coverage_qc(climate_base, target_sites, date_min, date_max)
    join_qc = build_event_join_check(episode_master, climate_base)

    source_df = pd.concat(source_rows, axis=0, ignore_index=True, sort=False) if len(source_rows) > 0 else pd.DataFrame()
    variable_hits_df = pd.concat(variable_hit_rows, axis=0, ignore_index=True, sort=False) if len(variable_hit_rows) > 0 else pd.DataFrame()

    climate_base_path = paths["climate_feature_dir"] / "site_day_climate_base.parquet"
    source_qc_path = paths["qc_dir"] / "qc_site_day_climate_source_files.csv"
    var_hits_path = paths["qc_dir"] / "qc_site_day_climate_variable_hits.csv"
    coverage_qc_path = paths["qc_dir"] / "qc_site_day_climate_coverage.csv"
    join_qc_path = paths["qc_dir"] / "qc_site_day_climate_event_join_check.csv"
    manifest_path = paths["qc_dir"] / "qc_site_day_climate_manifest.json"

    climate_base.to_parquet(climate_base_path, index=False)
    source_df.to_csv(source_qc_path, index=False, encoding="utf-8-sig")
    variable_hits_df.to_csv(var_hits_path, index=False, encoding="utf-8-sig")
    coverage_qc.to_csv(coverage_qc_path, index=False, encoding="utf-8-sig")
    join_qc.to_csv(join_qc_path, index=False, encoding="utf-8-sig")

    manifest = {
        "input_episode_master": str(episode_master_path),
        "target_sites": target_sites,
        "event_date_min": str(date_min),
        "event_date_max": str(date_max),
        "site_box_mode": "study_region_bbox",
        "era5_time_coord_supported": "valid_time",
        "era5_variables_expected": ["t2m", "d2m", "tp"],
        "derived_variables": ["relative_humidity_pct"],
        "n_files_scanned": int(len(file_inventory)),
        "n_rows_climate_base": int(len(climate_base)),
        "output_files": {
            "site_day_climate_base_parquet": str(climate_base_path),
            "qc_site_day_climate_source_files_csv": str(source_qc_path),
            "qc_site_day_climate_variable_hits_csv": str(var_hits_path),
            "qc_site_day_climate_coverage_csv": str(coverage_qc_path),
            "qc_site_day_climate_event_join_check_csv": str(join_qc_path),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logging.info(f"Wrote: {climate_base_path}")
    logging.info(f"Wrote: {source_qc_path}")
    logging.info(f"Wrote: {var_hits_path}")
    logging.info(f"Wrote: {coverage_qc_path}")
    logging.info(f"Wrote: {join_qc_path}")
    logging.info(f"Wrote: {manifest_path}")
    logging.info("05_build_site_day_climate_base.py completed successfully.")


if __name__ == "__main__":
    main()
