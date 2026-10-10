import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml


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
    for old, new in [
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
    ]:
        c = c.replace(old, new)
    while "__" in c:
        c = c.replace("__", "_")
    return c.strip("_")


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


def standardize_case_control(series: pd.Series) -> pd.Series:
    """
    Convert common case/control representations to:
    1 = case
    0 = control
    NaN = unknown
    """
    s = normalize_string_series(series).str.lower()

    case_values = {
        "1", "case", "cases", "msd", "yes", "true",
        "c", "patient", "participant_case"
    }
    control_values = {
        "0", "control", "controls", "no", "false",
        "ctrl", "comparison", "participant_control"
    }

    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out[s.isin(case_values)] = 1
    out[s.isin(control_values)] = 0

    try:
        numeric = pd.to_numeric(series, errors="coerce")
        out[numeric == 1] = 1
        out[numeric == 0] = 0
    except Exception:
        pass

    return out


def pick_existing_input(cleaned_dir: Path) -> Path:
    candidates = [
        cleaned_dir / "episode_master_modeling_ready.parquet",
        cleaned_dir / "episode_master.parquet",
        cleaned_dir / "episode_master_enriched.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No episode master parquet found. Expected one of:\n"
        + "\n".join(str(x) for x in candidates)
    )


# =========================================================
# harmonization
# =========================================================
def harmonize_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we can find site/country/case_control even if names vary slightly.
    """
    original_cols = list(df.columns)
    rename_map = {c: normalize_colname(c) for c in original_cols}
    df = df.rename(columns=rename_map)

    candidate_map = {
        "site": [
            "site",
            "study_center",
            "study_centre",
            "study_center_eupath_0010047",
            "study_site",
            "site_name",
        ],
        "country": [
            "country",
            "country_envo_00000009",
            "country_name",
            "country_of_residence",
        ],
        "case_control": [
            "case_control",
            "case_or_control_participant",
            "case_or_control_participant_eupath_0010375",
            "case_status",
            "status_case_control",
        ],
        "child_id": [
            "child_id",
            "participant_id",
            "subject_id",
        ],
        "episode_id": [
            "episode_id",
            "case_id",
            "case_id_eupath_0010049",
            "visit_id",
        ],
    }

    for target, candidates in candidate_map.items():
        if target not in df.columns:
            for c in candidates:
                if c in df.columns:
                    df[target] = df[c]
                    break

    if "site" in df.columns:
        df["site"] = normalize_string_series(df["site"])

    if "country" in df.columns:
        df["country"] = normalize_string_series(df["country"])

    if "case_control" in df.columns:
        df["case_control_std"] = standardize_case_control(df["case_control"])
    else:
        df["case_control_std"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if "child_id" in df.columns:
        df["child_id"] = normalize_string_series(df["child_id"])

    if "episode_id" in df.columns:
        df["episode_id"] = normalize_string_series(df["episode_id"])

    return df


# =========================================================
# country normalization / broad grouping
# =========================================================
def normalize_country_value(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()

    mapping = {
        "gambia": "The Gambia",
        "the gambia": "The Gambia",
        "gambie": "The Gambia",
        "bangladesh": "Bangladesh",
        "india": "India",
        "pakistan": "Pakistan",
        "kenya": "Kenya",
        "mali": "Mali",
        "mozambique": "Mozambique",
    }

    key = s.lower()
    return mapping.get(key, s)


def add_broad_grouping_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "country" in work.columns:
        work["country_std"] = work["country"].apply(normalize_country_value).astype("string")
    else:
        work["country_std"] = pd.Series(pd.NA, index=work.index, dtype="string")

    # current raw site is facility-level
    if "site" in work.columns:
        work["site_facility"] = work["site"].astype("string")
    else:
        work["site_facility"] = pd.Series(pd.NA, index=work.index, dtype="string")

    # recommended broad grouping for now: country as site_7
    work["site_7"] = work["country_std"]

    return work


# =========================================================
# outputs
# =========================================================
def build_site_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Facility-level site distribution.
    """
    work = df.copy()

    if "site_facility" not in work.columns:
        raise ValueError("Column 'site_facility' not found.")

    total_n = len(work)

    grp = work.groupby("site_facility", dropna=False)

    out = grp.size().rename("n_records").reset_index()

    if "case_control_std" in work.columns:
        case_counts = (
            work.loc[work["case_control_std"] == 1]
            .groupby("site_facility", dropna=False)
            .size()
            .rename("n_cases")
            .reset_index()
        )
        control_counts = (
            work.loc[work["case_control_std"] == 0]
            .groupby("site_facility", dropna=False)
            .size()
            .rename("n_controls")
            .reset_index()
        )
        out = out.merge(case_counts, on="site_facility", how="left")
        out = out.merge(control_counts, on="site_facility", how="left")
    else:
        out["n_cases"] = pd.NA
        out["n_controls"] = pd.NA

    if "child_id" in work.columns:
        child_counts = (
            work.groupby("site_facility", dropna=False)["child_id"]
            .nunique(dropna=True)
            .rename("n_unique_child_ids")
            .reset_index()
        )
        out = out.merge(child_counts, on="site_facility", how="left")
    else:
        out["n_unique_child_ids"] = pd.NA

    if "episode_id" in work.columns:
        episode_counts = (
            work.groupby("site_facility", dropna=False)["episode_id"]
            .nunique(dropna=True)
            .rename("n_unique_episode_ids")
            .reset_index()
        )
        out = out.merge(episode_counts, on="site_facility", how="left")
    else:
        out["n_unique_episode_ids"] = pd.NA

    if "country_std" in work.columns:
        n_countries = (
            work.groupby("site_facility", dropna=False)["country_std"]
            .nunique(dropna=True)
            .rename("n_distinct_countries_within_site")
            .reset_index()
        )
        country_preview = (
            work.groupby("site_facility", dropna=False)["country_std"]
            .apply(lambda s: " | ".join(sorted([x for x in s.dropna().astype(str).unique()])))
            .rename("countries_observed")
            .reset_index()
        )
        out = out.merge(n_countries, on="site_facility", how="left")
        out = out.merge(country_preview, on="site_facility", how="left")
    else:
        out["n_distinct_countries_within_site"] = pd.NA
        out["countries_observed"] = pd.NA

    out["pct_of_total"] = (out["n_records"] / total_n * 100).round(3)
    out = out.sort_values(["n_records", "site_facility"], ascending=[False, True]).reset_index(drop=True)
    out = out.rename(columns={"site_facility": "site"})
    out["site"] = out["site"].fillna("__MISSING_SITE__")

    return out


def build_country_site_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crosstab between country and facility-level site.
    This is descriptive only, not an error detector.
    """
    work = df.copy()

    if "site_facility" not in work.columns or "country_std" not in work.columns:
        raise ValueError("Both 'site_facility' and 'country_std' are required.")

    work["site_facility"] = normalize_string_series(work["site_facility"]).fillna("__MISSING_SITE__")
    work["country_std"] = normalize_string_series(work["country_std"]).fillna("__MISSING_COUNTRY__")

    ct = pd.crosstab(work["country_std"], work["site_facility"], dropna=False)
    ct["__row_total__"] = ct.sum(axis=1)
    total_row = pd.DataFrame([ct.sum(axis=0)], index=["__column_total__"])
    ct = pd.concat([ct, total_row], axis=0)

    ct = ct.reset_index().rename(columns={"index": "country"})
    if "country_std" in ct.columns:
        ct = ct.rename(columns={"country_std": "country"})
    return ct


def build_country_site_inconsistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revised inconsistency logic:
    - FAIL only when one site maps to multiple countries
    - WARN for missing site/country rows
    - DO NOT flag country -> multiple sites
    """
    work = df.copy()

    if "site_facility" not in work.columns or "country_std" not in work.columns:
        raise ValueError("Both 'site_facility' and 'country_std' are required.")

    work["site_facility"] = normalize_string_series(work["site_facility"])
    work["country_std"] = normalize_string_series(work["country_std"])

    rows = []

    valid = work.loc[
        work["site_facility"].notna() & work["country_std"].notna()
    ].copy()

    site_map = (
        valid.groupby("site_facility")["country_std"]
        .agg(lambda s: sorted(set(s.dropna().astype(str))))
        .reset_index()
    )
    site_map["n_countries"] = site_map["country_std"].apply(len)

    for _, r in site_map.iterrows():
        if r["n_countries"] > 1:
            rows.append({
                "issue_type": "site_maps_to_multiple_countries",
                "key": r["site_facility"],
                "n_distinct_values": int(r["n_countries"]),
                "observed_values": " | ".join(r["country_std"]),
                "status": "FAIL"
            })

    missing_site = int(work["site_facility"].isna().sum())
    missing_country = int(work["country_std"].isna().sum())

    if missing_site > 0:
        rows.append({
            "issue_type": "missing_site_rows",
            "key": "__missing__",
            "n_distinct_values": missing_site,
            "observed_values": "",
            "status": "WARN"
        })

    if missing_country > 0:
        rows.append({
            "issue_type": "missing_country_rows",
            "key": "__missing__",
            "n_distinct_values": missing_country,
            "observed_values": "",
            "status": "WARN"
        })

    if len(rows) == 0:
        rows.append({
            "issue_type": "no_inconsistency_found",
            "key": "__all_checked__",
            "n_distinct_values": 0,
            "observed_values": "",
            "status": "PASS"
        })

    out = pd.DataFrame(rows)
    return out


def build_site_distribution_broad_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Country-level broad grouping check.
    Since current site is facility-level, broad check summarizes by country (= recommended site_7).
    """
    work = df.copy()

    if "country_std" not in work.columns:
        raise ValueError("Column 'country_std' not found.")

    total_n = len(work)

    grp = work.groupby("country_std", dropna=False)
    out = grp.size().rename("n_records").reset_index()

    if "case_control_std" in work.columns:
        case_counts = (
            work.loc[work["case_control_std"] == 1]
            .groupby("country_std", dropna=False)
            .size()
            .rename("n_cases")
            .reset_index()
        )
        control_counts = (
            work.loc[work["case_control_std"] == 0]
            .groupby("country_std", dropna=False)
            .size()
            .rename("n_controls")
            .reset_index()
        )
        out = out.merge(case_counts, on="country_std", how="left")
        out = out.merge(control_counts, on="country_std", how="left")
    else:
        out["n_cases"] = pd.NA
        out["n_controls"] = pd.NA

    if "site_facility" in work.columns:
        n_facility = (
            work.groupby("country_std", dropna=False)["site_facility"]
            .nunique(dropna=True)
            .rename("n_unique_facility_sites")
            .reset_index()
        )
        facility_examples = (
            work.groupby("country_std", dropna=False)["site_facility"]
            .apply(lambda s: " | ".join(sorted([x for x in s.dropna().astype(str).unique()][:10])))
            .rename("facility_site_examples_up_to_10")
            .reset_index()
        )
        out = out.merge(n_facility, on="country_std", how="left")
        out = out.merge(facility_examples, on="country_std", how="left")
    else:
        out["n_unique_facility_sites"] = pd.NA
        out["facility_site_examples_up_to_10"] = pd.NA

    out["suggested_site_7"] = out["country_std"]
    out["broad_level_interpretation"] = "Use country as broad site grouping for main analysis"
    out["pct_of_total"] = (out["n_records"] / total_n * 100).round(3)

    # simple status rule
    out["broad_check_status"] = out["country_std"].isna().map(lambda x: "WARN" if x else "PASS")

    out = out.rename(columns={"country_std": "country"})
    out["country"] = out["country"].fillna("__MISSING_COUNTRY__")
    out = out.sort_values(["n_records", "country"], ascending=[False, True]).reset_index(drop=True)

    return out


def build_site_7_mapping_framework(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a framework table to preserve:
    - current facility-level site
    - merged country
    - suggested broad site_7
    """
    work = df.copy()

    if "site_facility" not in work.columns:
        raise ValueError("Column 'site_facility' not found.")
    if "country_std" not in work.columns:
        raise ValueError("Column 'country_std' not found.")

    grp_cols = ["site_facility", "country_std"]
    out = (
        work.groupby(grp_cols, dropna=False)
        .size()
        .rename("n_records")
        .reset_index()
    )

    if "case_control_std" in work.columns:
        case_counts = (
            work.loc[work["case_control_std"] == 1]
            .groupby(grp_cols, dropna=False)
            .size()
            .rename("n_cases")
            .reset_index()
        )
        control_counts = (
            work.loc[work["case_control_std"] == 0]
            .groupby(grp_cols, dropna=False)
            .size()
            .rename("n_controls")
            .reset_index()
        )
        out = out.merge(case_counts, on=grp_cols, how="left")
        out = out.merge(control_counts, on=grp_cols, how="left")
    else:
        out["n_cases"] = pd.NA
        out["n_controls"] = pd.NA

    out["site_7_suggested"] = out["country_std"]
    out["mapping_basis"] = "country_derived_broad_grouping"
    out["site_granularity_detected"] = "facility_level"
    out["recommended_analysis_use"] = (
        "Keep site_facility for sensitivity or fine-grained exploration; "
        "use site_7_suggested for broad grouping in main models"
    )

    out["manual_review_flag"] = out["country_std"].isna().map(lambda x: "YES" if x else "NO")
    out["mapping_status"] = out["manual_review_flag"].map(
        {"YES": "NEEDS_REVIEW", "NO": "AUTO_SUGGESTED"}
    )

    out = out.rename(columns={
        "site_facility": "site",
        "country_std": "country"
    })

    out["site"] = out["site"].fillna("__MISSING_SITE__")
    out["country"] = out["country"].fillna("__MISSING_COUNTRY__")
    out["site_7_suggested"] = out["site_7_suggested"].fillna("__MISSING_COUNTRY__")

    out = out.sort_values(["country", "n_records", "site"], ascending=[True, False, True]).reset_index(drop=True)
    return out


# =========================================================
# main
# =========================================================
def main():
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to project_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    project_root = Path(cfg.get("project_root", ".")).resolve()
    derived_cfg = cfg.get("derived", {})

    cleaned_dir = project_root / derived_cfg.get("cleaned_dir", "02_intermediate/cleaned")
    qc_dir = project_root / derived_cfg.get("qc_dir", "02_intermediate/qc")
    ensure_dir(qc_dir)

    input_path = pick_existing_input(cleaned_dir)
    logging.info(f"Using input file: {input_path}")

    df = pd.read_parquet(input_path)
    logging.info(f"Loaded rows: {len(df):,} | cols: {df.shape[1]}")

    df = harmonize_required_columns(df)
    df = add_broad_grouping_columns(df)

    required_present = {
        "site": "site" in df.columns,
        "country": "country" in df.columns,
        "site_facility": "site_facility" in df.columns,
        "country_std": "country_std" in df.columns,
        "site_7": "site_7" in df.columns,
        "case_control": "case_control" in df.columns,
        "case_control_std": "case_control_std" in df.columns,
        "child_id": "child_id" in df.columns,
        "episode_id": "episode_id" in df.columns,
    }
    logging.info(f"Field presence: {required_present}")

    # paths
    site_dist_path = qc_dir / "qc_site_distribution.csv"
    crosstab_path = qc_dir / "qc_country_site_crosstab.csv"
    inconsistency_path = qc_dir / "qc_country_site_inconsistency.csv"
    broad_check_path = qc_dir / "qc_site_distribution_broad_check.csv"
    site7_map_path = qc_dir / "qc_site_7_mapping_framework.csv"
    manifest_path = qc_dir / "qc_site_country_manifest.json"

    # 1) facility-level site distribution
    site_dist = build_site_distribution(df)
    site_dist.to_csv(site_dist_path, index=False, encoding="utf-8-sig")
    logging.info(f"Wrote: {site_dist_path}")

    # 2) descriptive crosstab
    crosstab = build_country_site_crosstab(df)
    crosstab.to_csv(crosstab_path, index=False, encoding="utf-8-sig")
    logging.info(f"Wrote: {crosstab_path}")

    # 3) revised inconsistency
    inconsistency = build_country_site_inconsistency(df)
    inconsistency.to_csv(inconsistency_path, index=False, encoding="utf-8-sig")
    logging.info(f"Wrote: {inconsistency_path}")

    # 4) broad-level check
    broad_check = build_site_distribution_broad_check(df)
    broad_check.to_csv(broad_check_path, index=False, encoding="utf-8-sig")
    logging.info(f"Wrote: {broad_check_path}")

    # 5) site_7 mapping framework
    site7_map = build_site_7_mapping_framework(df)
    site7_map.to_csv(site7_map_path, index=False, encoding="utf-8-sig")
    logging.info(f"Wrote: {site7_map_path}")

    manifest = {
        "input_file_used": str(input_path),
        "n_rows_input": int(len(df)),
        "n_cols_input": int(df.shape[1]),
        "field_presence": required_present,
        "interpretation_note": (
            "Current site field is treated as facility-level. "
            "Broad grouping (site_7) is auto-suggested from country."
        ),
        "outputs": {
            "qc_site_distribution_csv": str(site_dist_path),
            "qc_country_site_crosstab_csv": str(crosstab_path),
            "qc_country_site_inconsistency_csv": str(inconsistency_path),
            "qc_site_distribution_broad_check_csv": str(broad_check_path),
            "qc_site_7_mapping_framework_csv": str(site7_map_path),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logging.info(f"Wrote: {manifest_path}")
    logging.info("03c_qc_site_country.py completed successfully.")


if __name__ == "__main__":
    main()
