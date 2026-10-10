from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ============================================================
# Logging
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ============================================================
# Config / paths
# ============================================================

def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_project_root(config_path: Path, cfg: dict) -> Path:
    """
    Try config-defined root first; otherwise use parent of 09_configs.
    """
    candidate_keys = [
        "project_root",
        "project_dir",
        "root_dir",
        "base_dir",
    ]
    for key in candidate_keys:
        if key in cfg and cfg[key]:
            p = Path(cfg[key]).expanduser()
            if p.exists():
                return p.resolve()

    # default: /.../project_root/09_configs/project_config.yaml -> parent.parent
    return config_path.resolve().parent.parent


def build_paths(project_root: Path) -> Dict[str, Path]:
    paths = {
        "project_root": project_root,
        "qc_dir": project_root / "02_intermediate" / "qc",
        "final_tables_dir": project_root / "03_features" / "final_tables",
        "climate_features_dir": project_root / "03_features" / "climate_features",
        "pathogen_features_dir": project_root / "03_features" / "pathogen_labels",
    }
    for k in ["qc_dir", "final_tables_dir", "climate_features_dir", "pathogen_features_dir"]:
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def find_ml_master_file(paths: Dict[str, Path]) -> Path:
    candidates = [
        paths["final_tables_dir"] / "ml_master_wide.parquet",
        paths["final_tables_dir"] / "ml_master_table.parquet",
        paths["final_tables_dir"] / "ml_master.parquet",
        paths["final_tables_dir"] / "ml_master_wide_with_features.parquet",
    ]
    found = first_existing_path(candidates)
    if found is None:
        raise FileNotFoundError(
            "Could not find ML master table. Expected one of:\n"
            + "\n".join(str(p) for p in candidates)
        )
    return found


# ============================================================
# Utilities
# ============================================================

def normalize_name(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def make_unique_columns(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        base = normalize_name(c)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def ensure_unique_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_columns(list(df.columns))
    return df


def coerce_binary_series(s: pd.Series) -> pd.Series:
    """
    Convert common binary encodings to pandas Int64 with values {0,1,<NA>}.
    """
    if s is None:
        return pd.Series(pd.array([pd.NA] * 0, dtype="Int64"))

    # numeric direct
    if pd.api.types.is_numeric_dtype(s):
        out = pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)
        out.loc[s == 1] = 1
        out.loc[s == 0] = 0
        return out

    s2 = s.astype("string").str.strip().str.lower()

    positive_tokens = {
        "1", "y", "yes", "true", "t", "positive", "pos", "detected",
        "present", "det", "p", "reactive"
    }
    negative_tokens = {
        "0", "n", "no", "false", "f", "negative", "neg", "not detected",
        "absent", "nd", "nonreactive", "non_reactive"
    }
    missing_tokens = {
        "", "na", "n/a", "nan", "null", "none", ".", "missing", "unknown"
    }

    out = pd.Series(pd.array([pd.NA] * len(s2), dtype="Int64"), index=s2.index)
    out.loc[s2.isin(positive_tokens)] = 1
    out.loc[s2.isin(negative_tokens)] = 0
    out.loc[s2.isin(missing_tokens)] = pd.NA
    return out


def standardize_case_control(s: pd.Series) -> pd.Series:
    """
    Return string series with values {'case','control', <NA>}.
    """
    if s is None:
        return pd.Series(pd.array([pd.NA] * 0, dtype="string"))

    if pd.api.types.is_numeric_dtype(s):
        out = pd.Series(pd.array([pd.NA] * len(s), dtype="string"), index=s.index)
        out.loc[s == 1] = "case"
        out.loc[s == 0] = "control"
        return out

    s2 = s.astype("string").str.strip().str.lower()

    case_tokens = {
        "case", "cases", "1", "diarrhea", "diarrhoea", "patient", "ill", "episode_case"
    }
    control_tokens = {
        "control", "controls", "0", "healthy_control", "healthy", "comparison"
    }

    out = pd.Series(pd.array([pd.NA] * len(s2), dtype="string"), index=s2.index)
    out.loc[s2.isin(case_tokens)] = "case"
    out.loc[s2.isin(control_tokens)] = "control"
    return out


def pct(n: int, d: int) -> float:
    if d == 0:
        return float("nan")
    return round((n / d) * 100.0, 4)


def safe_to_int(x):
    if pd.isna(x):
        return None
    try:
        return int(x)
    except Exception:
        return None


# ============================================================
# Column detection
# ============================================================

PATHOGENS = [
    "rotavirus",
    "shigella",
    "cryptosporidium",
    "st_etec",
    "norovirus",
]

PATHOGEN_ALIASES = {
    "rotavirus": [
        "rotavirus",
        "rota",
    ],
    "shigella": [
        "shigella",
    ],
    "cryptosporidium": [
        "cryptosporidium",
        "crypto",
    ],
    "st_etec": [
        "st_etec",
        "stetec",
        "st_etec",
        "etec_st",
        "st_etec",
        "st_etec",
        "st_etec",
        "st_etec",
        "st_etec",
        "st_etec",
        "etec",
    ],
    "norovirus": [
        "norovirus",
        "noro",
    ],
}


def detect_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_key_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    keys = {}

    keys["row_id"] = detect_first_existing(df, [
        "row_id", "record_id", "index_id"
    ])
    keys["child_id"] = detect_first_existing(df, [
        "child_id", "participant_id", "subject_id", "childid"
    ])
    keys["episode_id"] = detect_first_existing(df, [
        "episode_id", "case_id", "visit_id", "episodeid"
    ])
    keys["household_id"] = detect_first_existing(df, [
        "household_id", "hh_id", "householdid"
    ])
    keys["case_control"] = detect_first_existing(df, [
        "case_control", "casecontrol", "case_status", "status_case_control"
    ])
    keys["event_date"] = detect_first_existing(df, [
        "event_date", "visit_date", "date_event", "onset_date", "sample_date"
    ])
    keys["site"] = detect_first_existing(df, [
        "site", "site_facility", "facility_site"
    ])
    keys["site_7"] = detect_first_existing(df, [
        "site_7", "study_site", "site7", "broad_site", "country"
    ])
    keys["country"] = detect_first_existing(df, [
        "country", "country_name"
    ])
    keys["age_months"] = detect_first_existing(df, [
        "age_months", "age_mo", "agemonths"
    ])
    keys["sex"] = detect_first_existing(df, [
        "sex", "gender"
    ])

    return keys


def score_pathogen_label_column(col: str, pathogen: str) -> int:
    """
    Prefer main label_*_positive_any style columns.
    Higher = better.
    """
    col_n = normalize_name(col)
    aliases = PATHOGEN_ALIASES[pathogen]

    score = 0
    if any(a in col_n for a in aliases):
        score += 20

    if col_n.startswith("label_"):
        score += 30
    if "positive_any" in col_n:
        score += 50
    if "main_label" in col_n:
        score += 20

    # preferred
    if re.fullmatch(rf"label_{pathogen}_positive_any", col_n):
        score += 100

    # de-prioritize auxiliary/non-primary fields
    bad_tokens = [
        "attributable",
        "result",
        "positive_result",
        "child",
        "source",
        "ct",
        "assay",
        "direct",
        "fallback",
        "sample_level",
    ]
    if any(tok in col_n for tok in bad_tokens):
        score -= 40

    return score


def detect_pathogen_label_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for pathogen in PATHOGENS:
        scored = []
        for c in df.columns:
            score = score_pathogen_label_column(c, pathogen)
            if score > 0:
                scored.append((c, score))
        if not scored:
            mapping[pathogen] = None
        else:
            scored = sorted(scored, key=lambda x: (-x[1], x[0]))
            mapping[pathogen] = scored[0][0]
    return mapping


def choose_preferred_climate_core_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Select a minimal set of climate core feature columns for analytic-eligibility checks.
    Prefer lag0_7; fallback to lag0_3; then first matching summary column.
    """
    cols = set(df.columns)

    preferences = {
        "precip_core": [
            "precip_mm_sum_lag0_7",
            "precip_mm_sum_lag0_3",
            "precip_mm_sum_lag0_1",
        ],
        "tmean_core": [
            "tmean_c_mean_lag0_7",
            "tmean_c_mean_lag0_3",
            "tmean_c_mean_lag0_1",
        ],
        "dewpoint_core": [
            "dewpoint_c_mean_lag0_7",
            "dewpoint_c_mean_lag0_3",
            "dewpoint_c_mean_lag0_1",
        ],
        "rh_core": [
            "relative_humidity_pct_mean_lag0_7",
            "relative_humidity_pct_mean_lag0_3",
            "relative_humidity_pct_mean_lag0_1",
        ],
    }

    out: Dict[str, Optional[str]] = {}

    # exact preferred candidates
    for grp, cand_list in preferences.items():
        chosen = None
        for c in cand_list:
            if c in cols:
                chosen = c
                break
        out[grp] = chosen

    # regex fallback if not found
    regex_fallbacks = {
        "precip_core": r"^precip_mm_sum_lag\d+_\d+$",
        "tmean_core": r"^tmean_c_mean_lag\d+_\d+$",
        "dewpoint_core": r"^dewpoint_c_mean_lag\d+_\d+$",
        "rh_core": r"^relative_humidity_pct_mean_lag\d+_\d+$",
    }

    for grp, rgx in regex_fallbacks.items():
        if out[grp] is None:
            matches = sorted([c for c in cols if re.search(rgx, c)])
            out[grp] = matches[0] if matches else None

    return out


# ============================================================
# Main logic
# ============================================================

def build_climate_core_nonmissing_flag(
    df: pd.DataFrame,
    climate_cols: Dict[str, Optional[str]]
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Return boolean flag: all selected climate core columns non-missing.
    If no climate core columns are found, return all False.
    """
    selected = [c for c in climate_cols.values() if c is not None]
    n_total = len(df)

    if len(selected) == 0:
        return pd.Series([False] * n_total, index=df.index), {
            "n_selected_climate_core_columns": 0,
            "n_rows_non_missing_all_selected": 0,
        }

    mask = pd.Series([True] * n_total, index=df.index)
    for c in selected:
        mask &= df[c].notna()

    info = {
        "n_selected_climate_core_columns": len(selected),
        "n_rows_non_missing_all_selected": int(mask.sum()),
    }
    return mask, info


def build_reason_counts(
    total_rows: int,
    has_label: pd.Series,
    has_case_control: pd.Series,
    has_climate: pd.Series
) -> Dict[str, int]:
    excluded_any = (~(has_label & has_case_control & has_climate)).sum()

    return {
        "n_total_rows": int(total_rows),
        "n_missing_main_label": int((~has_label).sum()),
        "n_missing_case_control": int((~has_case_control).sum()),
        "n_missing_climate_core": int((~has_climate).sum()),
        "n_excluded_any_reason": int(excluded_any),
        "n_included_analytic": int((has_label & has_case_control & has_climate).sum()),
    }


def create_empty_string_series(index) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * len(index), dtype="string"), index=index)


def build_analytic_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns:
    - full_with_flags
    - analytic_wide
    - qc_summary
    - qc_missing_reasons
    - qc_balance
    - manifest_dict
    """
    df = df.copy()

    # Standardize columns first
    df = ensure_unique_normalized_columns(df)

    # Detect keys
    key_cols = detect_key_columns(df)
    if key_cols["row_id"] is None:
        df["row_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
        key_cols["row_id"] = "row_id"

    # Standardize case-control
    if key_cols["case_control"] is None:
        df["_case_control_std"] = pd.Series(pd.array([pd.NA] * len(df), dtype="string"), index=df.index)
    else:
        df["_case_control_std"] = standardize_case_control(df[key_cols["case_control"]])

    # Fallback site_7 to country if absent
    if key_cols["site_7"] is None and key_cols["country"] is not None:
        df["_site_7_for_qc"] = df[key_cols["country"]].astype("string")
        key_cols["site_7"] = "_site_7_for_qc"
    elif key_cols["site_7"] is None and key_cols["site"] is not None:
        df["_site_7_for_qc"] = df[key_cols["site"]].astype("string")
        key_cols["site_7"] = "_site_7_for_qc"

    # Climate core selection
    climate_cols = choose_preferred_climate_core_columns(df)
    climate_nonmissing, climate_info = build_climate_core_nonmissing_flag(df, climate_cols)

    # Detect pathogen main label columns
    pathogen_label_cols = detect_pathogen_label_columns(df)

    total_rows = len(df)

    summary_rows = []
    missing_rows = []
    balance_rows = []

    flag_columns_for_output = []

    for pathogen in PATHOGENS:
        label_col = pathogen_label_cols.get(pathogen)

        # coerce main label
        if label_col is None:
            label_bin = pd.Series(pd.array([pd.NA] * total_rows, dtype="Int64"), index=df.index)
        else:
            label_bin = coerce_binary_series(df[label_col])

        has_label = label_bin.notna()
        has_case_control = df["_case_control_std"].isin(["case", "control"])
        has_climate = climate_nonmissing

        analytic_flag = has_label & has_case_control & has_climate
        test_available_flag = has_label
        eligible_case_control_flag = has_case_control
        non_missing_climate_core_flag = has_climate

        outcome = pd.Series(pd.array([pd.NA] * total_rows, dtype="Int64"), index=df.index)
        outcome.loc[analytic_flag] = label_bin.loc[analytic_flag].astype("Int64")

        is_case = df["_case_control_std"].eq("case")
        is_control = df["_case_control_std"].eq("control")

        n_positive_any = int((label_bin == 1).sum())
        n_negative_any = int((label_bin == 0).sum())
        n_missing_positive_any = int(label_bin.isna().sum())
        n_case_positive_main_label = int(((label_bin == 1) & is_case).sum())

        n_analytic = int(analytic_flag.sum())
        n_case_analytic = int((analytic_flag & is_case).sum())
        n_control_analytic = int((analytic_flag & is_control).sum())

        # attach columns
        analytic_col = f"analytic_{pathogen}"
        outcome_col = f"outcome_{pathogen}"
        test_avail_col = f"pathogen_test_available_{pathogen}"
        eligible_cc_col = f"eligible_case_control_{pathogen}"
        climate_ok_col = f"non_missing_climate_core_{pathogen}"
        label_source_col = f"main_label_column_{pathogen}"

        df[analytic_col] = analytic_flag.astype("Int64")
        df[outcome_col] = outcome.astype("Int64")
        df[test_avail_col] = test_available_flag.astype("Int64")
        df[eligible_cc_col] = eligible_case_control_flag.astype("Int64")
        df[climate_ok_col] = non_missing_climate_core_flag.astype("Int64")
        df[label_source_col] = label_col if label_col is not None else pd.NA

        flag_columns_for_output.extend([
            analytic_col, outcome_col, test_avail_col, eligible_cc_col,
            climate_ok_col, label_source_col
        ])

        summary_rows.append({
            "pathogen": pathogen,
            "main_label_column": label_col,
            "n_rows_total_master": total_rows,
            "n_positive_any": n_positive_any,
            "n_negative_any": n_negative_any,
            "n_missing_positive_any": n_missing_positive_any,
            "n_case_positive_main_label": n_case_positive_main_label,
            "n_analytic_rows_pathogen_specific": n_analytic,
            "n_case_rows_in_analytic_set": n_case_analytic,
            "n_control_rows_in_analytic_set": n_control_analytic,
            "pct_rows_in_analytic_set": pct(n_analytic, total_rows),
            "pct_case_rows_in_analytic_set": pct(n_case_analytic, total_rows),
            "pct_control_rows_in_analytic_set": pct(n_control_analytic, total_rows),
            "selected_precip_core_col": climate_cols.get("precip_core"),
            "selected_tmean_core_col": climate_cols.get("tmean_core"),
            "selected_dewpoint_core_col": climate_cols.get("dewpoint_core"),
            "selected_rh_core_col": climate_cols.get("rh_core"),
        })

        reason_counts = build_reason_counts(
            total_rows=total_rows,
            has_label=has_label,
            has_case_control=has_case_control,
            has_climate=has_climate
        )
        missing_rows.append({
            "pathogen": pathogen,
            "main_label_column": label_col,
            **reason_counts,
            "pct_missing_main_label": pct(reason_counts["n_missing_main_label"], total_rows),
            "pct_missing_case_control": pct(reason_counts["n_missing_case_control"], total_rows),
            "pct_missing_climate_core": pct(reason_counts["n_missing_climate_core"], total_rows),
            "pct_excluded_any_reason": pct(reason_counts["n_excluded_any_reason"], total_rows),
            "pct_included_analytic": pct(reason_counts["n_included_analytic"], total_rows),
        })

        # balance rows
        def add_balance_row(subset_name: str, mask: pd.Series):
            n_rows = int(mask.sum())
            n_pos = int(((outcome == 1) & mask).sum())
            n_neg = int(((outcome == 0) & mask).sum())
            balance_rows.append({
                "pathogen": pathogen,
                "subset": subset_name,
                "n_rows": n_rows,
                "n_positive_outcome": n_pos,
                "n_negative_outcome": n_neg,
                "positivity_rate_pct": pct(n_pos, n_rows) if n_rows > 0 else np.nan,
            })

        add_balance_row("analytic_all", analytic_flag)
        add_balance_row("analytic_cases", analytic_flag & is_case)
        add_balance_row("analytic_controls", analytic_flag & is_control)

        # optional site_7 breakdown
        if key_cols.get("site_7") is not None:
            site_7_col = key_cols["site_7"]
            tmp = pd.DataFrame({
                "site_7": df[site_7_col].astype("string"),
                "analytic": analytic_flag,
                "outcome": outcome,
            })
            tmp = tmp.loc[tmp["analytic"] == True].copy()
            if not tmp.empty:
                grp = (
                    tmp.groupby("site_7", dropna=False)
                    .agg(
                        n_rows=("analytic", "size"),
                        n_positive_outcome=("outcome", lambda x: int((x == 1).sum())),
                        n_negative_outcome=("outcome", lambda x: int((x == 0).sum())),
                    )
                    .reset_index()
                )
                for _, r in grp.iterrows():
                    n_rows = int(r["n_rows"])
                    n_pos = int(r["n_positive_outcome"])
                    n_neg = int(r["n_negative_outcome"])
                    balance_rows.append({
                        "pathogen": pathogen,
                        "subset": f"analytic_site_7::{r['site_7']}",
                        "n_rows": n_rows,
                        "n_positive_outcome": n_pos,
                        "n_negative_outcome": n_neg,
                        "positivity_rate_pct": pct(n_pos, n_rows) if n_rows > 0 else np.nan,
                    })

    qc_summary = pd.DataFrame(summary_rows)
    qc_missing = pd.DataFrame(missing_rows)
    qc_balance = pd.DataFrame(balance_rows)

    # Wide analytic table = keep IDs + new flags
    preferred_output_cols = [
        key_cols.get("row_id"),
        key_cols.get("child_id"),
        key_cols.get("household_id"),
        key_cols.get("episode_id"),
        key_cols.get("case_control"),
        key_cols.get("event_date"),
        key_cols.get("site"),
        key_cols.get("site_7"),
        key_cols.get("country"),
        key_cols.get("age_months"),
        key_cols.get("sex"),
    ]
    preferred_output_cols = [c for c in preferred_output_cols if c is not None and c in df.columns]

    climate_selected_cols = [c for c in climate_cols.values() if c is not None and c in df.columns]

    analytic_wide_cols = preferred_output_cols + climate_selected_cols + flag_columns_for_output
    analytic_wide_cols = list(dict.fromkeys(analytic_wide_cols))
    analytic_wide = df[analytic_wide_cols].copy()

    manifest = {
        "n_rows_input_master": total_rows,
        "detected_key_columns": key_cols,
        "detected_pathogen_label_columns": pathogen_label_cols,
        "selected_climate_core_columns": climate_cols,
        "climate_core_selection_info": climate_info,
        "pathogens": PATHOGENS,
    }

    return df, analytic_wide, qc_summary, qc_missing, qc_balance, manifest


# ============================================================
# Write outputs
# ============================================================

def write_outputs(
    full_df: pd.DataFrame,
    analytic_wide: pd.DataFrame,
    qc_summary: pd.DataFrame,
    qc_missing: pd.DataFrame,
    qc_balance: pd.DataFrame,
    manifest: dict,
    paths: Dict[str, Path],
    input_master_path: Path
) -> None:
    out_full = paths["final_tables_dir"] / "ml_master_wide_with_analytic_sets.parquet"
    out_wide = paths["final_tables_dir"] / "pathogen_analytic_sets_wide.parquet"

    out_qc_summary = paths["qc_dir"] / "qc_pathogen_analytic_set_summary.csv"
    out_qc_missing = paths["qc_dir"] / "qc_pathogen_analytic_missing_reasons.csv"
    out_qc_balance = paths["qc_dir"] / "qc_pathogen_analytic_case_control_balance.csv"
    out_manifest = paths["qc_dir"] / "qc_pathogen_analytic_manifest.json"

    full_df.to_parquet(out_full, index=False)
    analytic_wide.to_parquet(out_wide, index=False)

    qc_summary.to_csv(out_qc_summary, index=False)
    qc_missing.to_csv(out_qc_missing, index=False)
    qc_balance.to_csv(out_qc_balance, index=False)

    manifest_out = {
        **manifest,
        "input_master_path": str(input_master_path),
        "outputs": {
            "ml_master_wide_with_analytic_sets": str(out_full),
            "pathogen_analytic_sets_wide": str(out_wide),
            "qc_pathogen_analytic_set_summary": str(out_qc_summary),
            "qc_pathogen_analytic_missing_reasons": str(out_qc_missing),
            "qc_pathogen_analytic_case_control_balance": str(out_qc_balance),
        }
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, ensure_ascii=False, indent=2)

    logging.info("Wrote: %s", out_full)
    logging.info("Wrote: %s", out_wide)
    logging.info("Wrote: %s", out_qc_summary)
    logging.info("Wrote: %s", out_qc_missing)
    logging.info("Wrote: %s", out_qc_balance)
    logging.info("Wrote: %s", out_manifest)


# ============================================================
# Main
# ============================================================

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Build pathogen-specific analytic sets from ML master table.")
    parser.add_argument("--config", required=True, help="Path to YAML config, e.g. 09_configs/project_config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = read_yaml(config_path)
    project_root = resolve_project_root(config_path, cfg)
    paths = build_paths(project_root)

    logging.info("Project root: %s", project_root)

    master_path = find_ml_master_file(paths)
    logging.info("Loading ML master table: %s", master_path)

    df = pd.read_parquet(master_path)
    logging.info("Loaded ML master: %s rows x %s cols", df.shape[0], df.shape[1])

    full_df, analytic_wide, qc_summary, qc_missing, qc_balance, manifest = build_analytic_sets(df)

    write_outputs(
        full_df=full_df,
        analytic_wide=analytic_wide,
        qc_summary=qc_summary,
        qc_missing=qc_missing,
        qc_balance=qc_balance,
        manifest=manifest,
        paths=paths,
        input_master_path=master_path
    )

    logging.info("07b_build_pathogen_analytic_sets.py completed successfully.")


if __name__ == "__main__":
    main()
