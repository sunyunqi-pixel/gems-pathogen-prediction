import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml


# =========================================================
# 基础函数
# =========================================================

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_col(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def path_to_posix(path_obj):
    return str(path_obj).replace("\\", "/")


def safe_string_series(s):
    x = s.astype(str).str.strip()
    x = x.replace({
        "": pd.NA, "nan": pd.NA, "None": pd.NA,
        "NA": pd.NA, "na": pd.NA, "NaT": pd.NA
    })
    return x


def safe_string_strip(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = safe_string_series(df[c])
    return df


def coerce_numeric_if_exists(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================================================
# 自动读取文本表
# =========================================================

def detect_text_delimiter(file_path, encodings=("utf-8", "utf-8-sig", "latin1")):
    candidates = ["\t", "|", ",", ";"]
    best_delim = "\t"
    best_score = -1

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                lines = []
                for _ in range(30):
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        lines.append(line.rstrip("\n"))
            if not lines:
                continue

            for delim in candidates:
                counts = [len(line.split(delim)) for line in lines if line]
                if not counts:
                    continue
                score = (sum(counts) / len(counts)) - (max(counts) - min(counts))
                if score > best_score:
                    best_score = score
                    best_delim = delim
        except Exception:
            continue

    return best_delim


def read_table_flexible(file_path):
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext in [".txt", ".csv", ".tsv"]:
        delim = detect_text_delimiter(file_path)
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(
                    file_path,
                    sep=delim,
                    dtype=str,
                    encoding=enc,
                    engine="python"
                )
            except Exception:
                continue
        raise ValueError(f"Failed to read text table: {file_path}")

    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, dtype=str)

    elif ext == ".dta":
        return pd.read_stata(file_path, convert_categoricals=False)

    elif ext == ".sav":
        return pd.read_spss(file_path)

    elif ext == ".parquet":
        return pd.read_parquet(file_path)

    elif ext == ".feather":
        return pd.read_feather(file_path)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


# =========================================================
# Registry
# =========================================================

def load_registry(project_root, cfg):
    reg_path = project_root / cfg["derived"]["inventory_dir"] / "extracted_file_registry.csv"
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry not found: {reg_path}")
    return pd.read_csv(reg_path)


def get_registry_subsets(reg_df):
    participants = reg_df[
        (reg_df["dataset_family"] == "case_control") &
        (reg_df["table_hint"] == "participants")
    ].copy()

    households = reg_df[
        (reg_df["dataset_family"] == "case_control") &
        (reg_df["table_hint"] == "households")
    ].copy()

    samples = reg_df[
        (reg_df["dataset_family"] == "case_control") &
        (reg_df["table_hint"] == "samples")
    ].copy()

    return participants, households, samples


# =========================================================
# 强制优先映射
# =========================================================

PARTICIPANT_PRIORITY_MAP = {
    "child_id": [
        "participant_id",
        "child_id",
        "subject_id",
        "study_child_id",
    ],
    "household_id": [
        "household_id",
        "hhid",
        "hh_id",
    ],
    "episode_id": [
        "case_id_eupath_0010049",
        "case_id",
        "visit_id",
        "episode_id",
        "record_id",
    ],
    "case_control": [
        "case_or_control_participant_eupath_0010375",
        "case_control",
        "casecontrol",
        "case_control_sub_classification_eupath_0033155",
    ],
    "site": [
        "study_center_eupath_0010047",
        "study_center",
        "site",
        "site_name",
        "site_id",
        "siteid",
    ],
    "event_date": [
        "enrollment_date_eupath_0000151",
        "enrollment_date_matched_case_eupath_0010042",
        "control_eligibility_check_date_eupath_0010119",
        "eligibility_form_date_eupath_0015426",
        "registration_date_eupath_0015440",
        "event_date",
        "visit_date",
        "enroll_date",
        "date",
    ],
    "age_months": [
        "age_at_eligibility_form_completion_months_eupath_0010368",
        "age_months",
        "agemo",
        "age_mo",
        "age_month",
        "agemonths",
        "age_matched_case_months_eupath_0010120",
    ],
    "sex": [
        "sex_pato_0000047",
        "sex",
        "gender",
        "sex_matched_case_eupath_0010405",
    ],
}

HOUSEHOLD_PRIORITY_MAP = {
    "household_id": [
        "household_id",
        "hhid",
        "hh_id",
    ],
    "country": [
        "country_envo_00000009",
        "country",
        "country_name",
        "nation",
    ],
    # 如果后面发现 households 里真有 site，可以继续加
    "site": [
        "site",
        "site_name",
        "site_id",
        "siteid",
        "study_site",
    ],
}

SAMPLE_PRIORITY_MAP = {
    "child_id": [
        "participant_id",
        "child_id",
        "subject_id",
        "study_child_id",
    ],
    "household_id": [
        "household_id",
        "hhid",
        "hh_id",
    ],
    "episode_id": [
        "case_id_eupath_0010049",
        "case_id",
        "sample_id",
        "visit_id",
        "episode_id",
        "record_id",
    ],
    "sample_date": [
        "collection_date",
        "sample_date",
        "date",
        "sample_collection_date",
        "collection_date_eupath_0015000", 
        "enrollment_date_eupath_0000151",
    ],
    "site": [
        "study_center_eupath_0010047",
        "study_center",
        "site",
        "site_name",
        "siteid",
    ],
}


def apply_priority_mapping(df, priority_map):
    """
    根据优先级，把真实列映射到标准列。
    返回：
    - 新 df
    - 命中报告 list
    """
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]

    hits = []
    for standard_field, candidates in priority_map.items():
        matched = None
        method = "unmatched"

        # 若标准字段已存在，直接保留
        if standard_field in df.columns:
            matched = standard_field
            method = "already_exists"
        else:
            for cand in candidates:
                cand_norm = normalize_col(cand)
                if cand_norm in df.columns:
                    df = df.rename(columns={cand_norm: standard_field})
                    matched = cand_norm
                    method = "priority_exact"
                    break

        hits.append({
            "standard_field": standard_field,
            "matched_column": matched,
            "method": method,
            "matched_flag": int(matched is not None),
        })

    return df, hits


# =========================================================
# metadata
# =========================================================

def add_metadata_columns(df, registry_row):
    df = df.copy()
    df["study_phase"] = registry_row["study_phase"]
    df["dataset_family"] = registry_row["dataset_family"]
    df["table_hint"] = registry_row["table_hint"]
    df["source_zip_relative_path"] = registry_row["source_zip_relative_path"]
    df["source_extracted_relative_path"] = registry_row["extracted_relative_path"]
    return df


def load_registry_files(project_root, subset_df, priority_map, kind_name):
    dfs = []
    source_rows = []
    field_hit_rows = []
    column_rows = []

    for _, r in subset_df.iterrows():
        file_path = project_root / r["extracted_relative_path"]
        logging.info(f"Reading {kind_name}: {file_path}")

        raw_df = read_table_flexible(file_path)

        for c in raw_df.columns:
            column_rows.append({
                "kind": kind_name,
                "study_phase": r["study_phase"],
                "dataset_family": r["dataset_family"],
                "table_hint": r["table_hint"],
                "source_zip_relative_path": r["source_zip_relative_path"],
                "extracted_relative_path": r["extracted_relative_path"],
                "raw_column_name": c,
                "normalized_column_name": normalize_col(c),
            })

        df = raw_df.copy()
        df.columns = [normalize_col(c) for c in df.columns]
        df, hits = apply_priority_mapping(df, priority_map)
        df = add_metadata_columns(df, r)

        for h in hits:
            field_hit_rows.append({
                "kind": kind_name,
                "study_phase": r["study_phase"],
                "dataset_family": r["dataset_family"],
                "table_hint": r["table_hint"],
                "source_zip_relative_path": r["source_zip_relative_path"],
                "extracted_relative_path": r["extracted_relative_path"],
                **h
            })

        source_rows.append({
            "kind": kind_name,
            "study_phase": r["study_phase"],
            "dataset_family": r["dataset_family"],
            "table_hint": r["table_hint"],
            "source_zip_relative_path": r["source_zip_relative_path"],
            "extracted_relative_path": r["extracted_relative_path"],
            "n_rows": raw_df.shape[0],
            "n_cols": raw_df.shape[1],
        })

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()
    source_meta = pd.DataFrame(source_rows)
    field_hits = pd.DataFrame(field_hit_rows)
    columns_meta = pd.DataFrame(column_rows)

    return combined, source_meta, field_hits, columns_meta


# =========================================================
# 值标准化
# =========================================================

def parse_case_control(series):
    """
    专门适配：
    - Case / Control
    - 1 / 0
    """
    s_raw = safe_string_series(series)
    s = s_raw.astype(str).str.strip().str.lower()

    case_values = {
        "case", "1", "yes", "true",
        "msd case", "lsd case"
    }
    control_values = {
        "control", "0", "no", "false"
    }

    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out[s.isin(case_values)] = 1
    out[s.isin(control_values)] = 0

    # 数值
    num = pd.to_numeric(s_raw, errors="coerce")
    out[num == 1] = 1
    out[num == 0] = 0

    # 对 subclassification 不做主 case/control 字段，避免误判
    return pd.to_numeric(out, errors="coerce")


def standardize_sex(series):
    s = safe_string_series(series).astype(str).str.strip().str.lower()

    male_vals = {"m", "male", "boy", "1"}
    female_vals = {"f", "female", "girl", "2"}

    out = pd.Series("unknown", index=s.index, dtype="object")
    out[s.isin(male_vals)] = "male"
    out[s.isin(female_vals)] = "female"
    out[s.isin({"", "nan", "none", "na"})] = "unknown"

    return out


def standardize_site(series):
    """
    目前只做轻清洗，不做复杂映射。
    """
    s = safe_string_series(series)
    return s


def standardize_country(series):
    s = safe_string_series(series)
    return s


# =========================================================
# 合并辅助 lookup
# =========================================================

def build_household_lookup(households_df):
    if households_df.empty:
        return None

    df = households_df.copy()

    for c in ["study_phase", "household_id", "country", "site"]:
        if c not in df.columns:
            df[c] = pd.NA

    df = safe_string_strip(df, ["study_phase", "household_id", "country", "site"])
    df["country"] = standardize_country(df["country"])
    df["site"] = standardize_site(df["site"])

    df["_has_country"] = df["country"].notna().astype(int)
    df["_has_site"] = df["site"].notna().astype(int)

    lookup = (
        df[["study_phase", "household_id", "country", "site", "_has_country", "_has_site"]]
        .dropna(subset=["study_phase", "household_id"])
        .sort_values(
            ["study_phase", "household_id", "_has_country", "_has_site"],
            ascending=[True, True, False, False]
        )
        .drop_duplicates(subset=["study_phase", "household_id"], keep="first")
        .drop(columns=["_has_country", "_has_site"])
    )

    return lookup


def build_sample_lookups(samples_df):
    if samples_df.empty:
        return None, None, None

    df = samples_df.copy()

    for c in ["study_phase", "child_id", "household_id", "episode_id", "sample_date"]:
        if c not in df.columns:
            df[c] = pd.NA

    df = safe_string_strip(df, ["study_phase", "child_id", "household_id", "episode_id"])
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")

    by_episode = (
        df[["study_phase", "episode_id", "sample_date"]]
        .dropna(subset=["study_phase", "episode_id"])
        .sort_values(["study_phase", "episode_id", "sample_date"])
        .drop_duplicates(subset=["study_phase", "episode_id"], keep="first")
    ) if df["episode_id"].notna().any() else None

    by_child = (
        df[["study_phase", "child_id", "episode_id", "sample_date"]]
        .dropna(subset=["study_phase", "child_id"])
        .sort_values(["study_phase", "child_id", "sample_date"])
        .drop_duplicates(subset=["study_phase", "child_id"], keep="first")
    ) if df["child_id"].notna().any() else None

    by_household = (
        df[["study_phase", "household_id", "episode_id", "sample_date"]]
        .dropna(subset=["study_phase", "household_id"])
        .sort_values(["study_phase", "household_id", "sample_date"])
        .drop_duplicates(subset=["study_phase", "household_id"], keep="first")
    ) if df["household_id"].notna().any() else None

    return by_episode, by_child, by_household


# =========================================================
# episode_id 生成与去重
# =========================================================

def build_episode_id(df):
    df = df.copy()

    if "episode_id" not in df.columns:
        df["episode_id"] = pd.NA

    df["episode_id"] = safe_string_series(df["episode_id"])

    # 对已有 ID 加 phase 前缀，避免 GEMS1/GEMS1A 冲突
    has_orig = df["episode_id"].notna()
    df.loc[has_orig, "episode_id"] = (
        df.loc[has_orig, "study_phase"].astype(str) + "_" +
        df.loc[has_orig, "episode_id"].astype(str)
    )

    # 对没有 ID 的记录生成
    miss = df["episode_id"].isna()
    date_part = pd.to_datetime(df["event_date"], errors="coerce").dt.strftime("%Y%m%d").fillna("nodate")
    child_part = safe_string_series(df["child_id"]).fillna("nochild")
    site_part = safe_string_series(df["site"]).fillna("nosite")
    phase_part = safe_string_series(df["study_phase"]).fillna("nophase")

    gen = phase_part + "_" + child_part + "_" + site_part + "_" + date_part
    df.loc[miss, "episode_id"] = gen.loc[miss]

    # 若仍重复，加序号
    dup_index = df.groupby("episode_id").cumcount()
    dup_mask = df["episode_id"].duplicated(keep=False)
    df.loc[dup_mask, "episode_id"] = (
        df.loc[dup_mask, "episode_id"].astype(str) + "_dup" + dup_index.loc[dup_mask].astype(str)
    )

    return df


def completeness_score(df):
    score = pd.Series(0, index=df.index, dtype="int64")
    for c in ["site", "country", "event_date", "age_months"]:
        if c in df.columns:
            score += df[c].notna().astype(int)
    if "sex" in df.columns:
        score += (df["sex"] != "unknown").astype(int)
    if "episode_id" in df.columns:
        score += df["episode_id"].notna().astype(int)
    return score


# =========================================================
# 主逻辑
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to project_config.yaml")
    args = parser.parse_args()

    setup_logger()
    cfg = load_yaml(args.config)

    project_root = Path(cfg["project_root"]).resolve()
    cleaned_dir = project_root / cfg["derived"]["cleaned_dir"]
    qc_dir = project_root / cfg["derived"]["qc_dir"]

    ensure_dir(cleaned_dir)
    ensure_dir(qc_dir)

    # -----------------------------------------------------
    # 1) 读取 registry
    # -----------------------------------------------------
    reg_df = load_registry(project_root, cfg)
    part_reg, hh_reg, samp_reg = get_registry_subsets(reg_df)

    if part_reg.empty:
        raise ValueError("No case_control + participants files found in extracted_file_registry.csv")

    # -----------------------------------------------------
    # 2) 读取 participants / households / samples
    # -----------------------------------------------------
    participants_df, part_source_meta, part_field_hits, part_columns_meta = load_registry_files(
        project_root=project_root,
        subset_df=part_reg,
        priority_map=PARTICIPANT_PRIORITY_MAP,
        kind_name="participants"
    )

    households_df, hh_source_meta, hh_field_hits, hh_columns_meta = load_registry_files(
        project_root=project_root,
        subset_df=hh_reg,
        priority_map=HOUSEHOLD_PRIORITY_MAP,
        kind_name="households"
    ) if not hh_reg.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    samples_df, samp_source_meta, samp_field_hits, samp_columns_meta = load_registry_files(
        project_root=project_root,
        subset_df=samp_reg,
        priority_map=SAMPLE_PRIORITY_MAP,
        kind_name="samples"
    ) if not samp_reg.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # 输出列与命中报告
    source_meta = pd.concat([part_source_meta, hh_source_meta, samp_source_meta], ignore_index=True, sort=False)
    field_hits = pd.concat([part_field_hits, hh_field_hits, samp_field_hits], ignore_index=True, sort=False)
    columns_meta = pd.concat([part_columns_meta, hh_columns_meta, samp_columns_meta], ignore_index=True, sort=False)

    source_meta.to_csv(qc_dir / "qc_episode_master_source_files.csv", index=False)
    field_hits.to_csv(qc_dir / "qc_episode_master_field_hits.csv", index=False)
    columns_meta.to_csv(qc_dir / "qc_episode_master_columns_by_source.csv", index=False)

    field_hit_summary = (
        field_hits.groupby(["kind", "standard_field", "method"], dropna=False)
        .size()
        .reset_index(name="n_files")
        .sort_values(["kind", "standard_field", "method"])
    )
    field_hit_summary.to_csv(qc_dir / "qc_episode_master_field_hit_summary.csv", index=False)

    # -----------------------------------------------------
    # 3) participants 主表标准化
    # -----------------------------------------------------
    df = participants_df.copy()

    # 确保关键列存在
    for c in [
        "child_id", "household_id", "episode_id", "case_control", "site",
        "country", "event_date", "age_months", "sex"
    ]:
        if c not in df.columns:
            df[c] = pd.NA

    df = safe_string_strip(df, [
        "child_id", "household_id", "episode_id", "case_control",
        "site", "country", "sex", "study_phase"
    ])

    df["case_control"] = parse_case_control(df["case_control"])
    df["site"] = standardize_site(df["site"])
    df["country"] = standardize_country(df["country"])
    df["sex"] = standardize_sex(df["sex"])
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["age_months"] = pd.to_numeric(df["age_months"], errors="coerce")

    # -----------------------------------------------------
    # 4) households merge: 只主补 country，site 仅作为次级备用
    # -----------------------------------------------------
    before_site_missing = df["site"].isna().sum()
    before_country_missing = df["country"].isna().sum()

    if not households_df.empty:
        hh_lookup = build_household_lookup(households_df)
        if hh_lookup is not None and len(hh_lookup):
            df = df.merge(
                hh_lookup.rename(columns={
                    "site": "_site_from_household",
                    "country": "_country_from_household"
                }),
                on=["study_phase", "household_id"],
                how="left"
            )
            # site 以 participants 为主，只在 participants 缺失时才用 household 补
            df["site"] = df["site"].fillna(df["_site_from_household"])
            df["country"] = df["country"].fillna(df["_country_from_household"])
        else:
            df["_site_from_household"] = pd.NA
            df["_country_from_household"] = pd.NA
    else:
        df["_site_from_household"] = pd.NA
        df["_country_from_household"] = pd.NA

    after_site_missing = df["site"].isna().sum()
    after_country_missing = df["country"].isna().sum()

    # -----------------------------------------------------
    # 5) samples merge：补 episode_id / event_date
    # -----------------------------------------------------
    before_event_missing = df["event_date"].isna().sum()
    before_epi_missing = df["episode_id"].isna().sum()

    if not samples_df.empty:
        by_episode, by_child, by_household = build_sample_lookups(samples_df)

        # 先按 child_id 补 episode_id
        if by_child is not None:
            df = df.merge(
                by_child.rename(columns={
                    "episode_id": "_episode_id_from_sample_child",
                    "sample_date": "_sample_date_from_child"
                }),
                on=["study_phase", "child_id"],
                how="left"
            )
        else:
            df["_episode_id_from_sample_child"] = pd.NA
            df["_sample_date_from_child"] = pd.NaT

        # 再按 household_id 补 episode_id
        if by_household is not None:
            df = df.merge(
                by_household.rename(columns={
                    "episode_id": "_episode_id_from_sample_household",
                    "sample_date": "_sample_date_from_household"
                }),
                on=["study_phase", "household_id"],
                how="left"
            )
        else:
            df["_episode_id_from_sample_household"] = pd.NA
            df["_sample_date_from_household"] = pd.NaT

        # 先补 episode_id
        df["episode_id"] = df["episode_id"].fillna(df["_episode_id_from_sample_child"])
        df["episode_id"] = df["episode_id"].fillna(df["_episode_id_from_sample_household"])

        # 再按 episode_id 补 event_date
        if by_episode is not None:
            df = df.merge(
                by_episode.rename(columns={"sample_date": "_sample_date_from_episode"}),
                on=["study_phase", "episode_id"],
                how="left"
            )
        else:
            df["_sample_date_from_episode"] = pd.NaT

        df["event_date"] = df["event_date"].fillna(df["_sample_date_from_episode"])
        df["event_date"] = df["event_date"].fillna(df["_sample_date_from_child"])
        df["event_date"] = df["event_date"].fillna(df["_sample_date_from_household"])

    else:
        df["_episode_id_from_sample_child"] = pd.NA
        df["_episode_id_from_sample_household"] = pd.NA
        df["_sample_date_from_episode"] = pd.NaT
        df["_sample_date_from_child"] = pd.NaT
        df["_sample_date_from_household"] = pd.NaT

    after_event_missing = df["event_date"].isna().sum()
    after_epi_missing = df["episode_id"].isna().sum()

    # -----------------------------------------------------
    # 6) 衍生时间字段
    # -----------------------------------------------------
    df["event_year"] = pd.to_datetime(df["event_date"], errors="coerce").dt.year
    df["event_month"] = pd.to_datetime(df["event_date"], errors="coerce").dt.month
    df["event_day"] = pd.to_datetime(df["event_date"], errors="coerce").dt.day

    # -----------------------------------------------------
    # 7) 构造 episode_id
    # -----------------------------------------------------
    df = build_episode_id(df)

    # -----------------------------------------------------
    # 8) enriched 输出（宽松版）
    # -----------------------------------------------------
    enriched_keep_cols = [
        "child_id",
        "household_id",
        "episode_id",
        "case_control",
        "site",
        "country",
        "event_date",
        "event_year",
        "event_month",
        "event_day",
        "age_months",
        "sex",
        "study_phase",
        "dataset_family",
        "table_hint",
        "source_zip_relative_path",
        "source_extracted_relative_path",
        "_site_from_household",
        "_country_from_household",
        "_episode_id_from_sample_child",
        "_episode_id_from_sample_household",
        "_sample_date_from_episode",
        "_sample_date_from_child",
        "_sample_date_from_household",
    ]
    for c in enriched_keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    episode_master_enriched = df[enriched_keep_cols].copy()

    # 类型
    episode_master_enriched["case_control"] = pd.to_numeric(episode_master_enriched["case_control"], errors="coerce").astype("Int64")
    episode_master_enriched["event_year"] = pd.to_numeric(episode_master_enriched["event_year"], errors="coerce").astype("Int64")
    episode_master_enriched["event_month"] = pd.to_numeric(episode_master_enriched["event_month"], errors="coerce").astype("Int64")
    episode_master_enriched["event_day"] = pd.to_numeric(episode_master_enriched["event_day"], errors="coerce").astype("Int64")
    episode_master_enriched["age_months"] = pd.to_numeric(episode_master_enriched["age_months"], errors="coerce")

    episode_master_enriched.to_parquet(cleaned_dir / "episode_master_enriched.parquet", index=False)

    # -----------------------------------------------------
    # 9) 宽松版 episode_master
    # 只要求 child_id + case_control
    # -----------------------------------------------------
    loose = episode_master_enriched.copy()

    before_loose_filter_n = len(loose)
    loose = loose[~loose["child_id"].isna()].copy()
    loose = loose[~loose["case_control"].isna()].copy()
    after_loose_filter_n = len(loose)

    # 去重规则
    # 1. completeness score 高优先
    # 2. 原始 episode_id 存在优先（通过 _dup 判断前已经反映）
    # 3. event_date 最早优先
    loose["_completeness_score"] = completeness_score(loose)
    loose["_has_country"] = loose["country"].notna().astype(int)
    loose["_has_site"] = loose["site"].notna().astype(int)
    loose["_has_event_date"] = loose["event_date"].notna().astype(int)
    loose["_has_age"] = loose["age_months"].notna().astype(int)
    loose["_has_sex"] = (loose["sex"] != "unknown").astype(int)

    loose = loose.sort_values(
        by=[
            "episode_id",
            "_completeness_score",
            "_has_site",
            "_has_country",
            "_has_event_date",
            "_has_age",
            "_has_sex",
            "event_date"
        ],
        ascending=[True, False, False, False, False, False, False, True]
    )

    dup_mask = loose.duplicated(subset=["episode_id"], keep=False)
    loose.loc[dup_mask].to_csv(qc_dir / "qc_episode_master_duplicates.csv", index=False)

    before_dedup_n = len(loose)
    loose = loose.drop_duplicates(subset=["episode_id"], keep="first").copy()
    after_dedup_n = len(loose)

    loose = loose.drop(columns=[
        "_completeness_score", "_has_country", "_has_site",
        "_has_event_date", "_has_age", "_has_sex"
    ])

    final_keep_cols = [
        "child_id",
        "household_id",
        "episode_id",
        "case_control",
        "site",
        "country",
        "event_date",
        "event_year",
        "event_month",
        "event_day",
        "age_months",
        "sex",
        "study_phase",
        "dataset_family",
        "table_hint",
        "source_zip_relative_path",
        "source_extracted_relative_path",
    ]
    episode_master = loose[final_keep_cols].copy()
    episode_master.to_parquet(cleaned_dir / "episode_master.parquet", index=False)

    # -----------------------------------------------------
    # 10) 严格建模版
    # -----------------------------------------------------
    modeling_ready = episode_master.copy()
    modeling_ready = modeling_ready[
        (~modeling_ready["child_id"].isna()) &
        (~modeling_ready["case_control"].isna()) &
        (~modeling_ready["site"].isna()) &
        (~modeling_ready["event_date"].isna())
    ].copy()
    modeling_ready.to_parquet(cleaned_dir / "episode_master_modeling_ready.parquet", index=False)

    # -----------------------------------------------------
    # 11) QC 输出
    # -----------------------------------------------------
    site_counts = (
        episode_master
        .groupby(["study_phase", "site", "case_control"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["study_phase", "site", "case_control"])
    )
    site_counts.to_csv(qc_dir / "qc_episode_master_site_counts.csv", index=False)

    missingness = pd.DataFrame({
        "column": episode_master.columns,
        "missing_n": [episode_master[c].isna().sum() for c in episode_master.columns],
        "missing_rate": [episode_master[c].isna().mean() for c in episode_master.columns],
        "dtype": [str(episode_master[c].dtype) for c in episode_master.columns],
    })
    missingness.to_csv(qc_dir / "qc_episode_master_missingness.csv", index=False)

    priority_mapping_report = pd.DataFrame([
        {"table": "participants", "standard_field": k, "priority_candidates": " | ".join(v)}
        for k, v in PARTICIPANT_PRIORITY_MAP.items()
    ] + [
        {"table": "households", "standard_field": k, "priority_candidates": " | ".join(v)}
        for k, v in HOUSEHOLD_PRIORITY_MAP.items()
    ] + [
        {"table": "samples", "standard_field": k, "priority_candidates": " | ".join(v)}
        for k, v in SAMPLE_PRIORITY_MAP.items()
    ])
    priority_mapping_report.to_csv(qc_dir / "qc_episode_master_priority_mapping_report.csv", index=False)

    qc_summary = pd.DataFrame([
        {"metric": "n_case_control_participant_files", "value": int(len(part_reg))},
        {"metric": "n_case_control_household_files", "value": int(len(hh_reg))},
        {"metric": "n_case_control_sample_files", "value": int(len(samp_reg))},

        {"metric": "participants_rows_raw", "value": int(len(participants_df))},
        {"metric": "households_rows_raw", "value": int(len(households_df))},
        {"metric": "samples_rows_raw", "value": int(len(samples_df))},

        {"metric": "site_missing_before_household_merge", "value": int(before_site_missing)},
        {"metric": "site_missing_after_household_merge", "value": int(after_site_missing)},
        {"metric": "country_missing_before_household_merge", "value": int(before_country_missing)},
        {"metric": "country_missing_after_household_merge", "value": int(after_country_missing)},

        {"metric": "episode_id_missing_before_sample_backfill", "value": int(before_epi_missing)},
        {"metric": "episode_id_missing_after_sample_backfill", "value": int(after_epi_missing)},
        {"metric": "event_date_missing_before_sample_backfill", "value": int(before_event_missing)},
        {"metric": "event_date_missing_after_sample_backfill", "value": int(after_event_missing)},

        {"metric": "rows_before_loose_filter", "value": int(before_loose_filter_n)},
        {"metric": "rows_after_loose_filter", "value": int(after_loose_filter_n)},
        {"metric": "rows_before_dedup", "value": int(before_dedup_n)},
        {"metric": "rows_after_dedup", "value": int(after_dedup_n)},

        {"metric": "n_cases", "value": int((episode_master["case_control"] == 1).sum())},
        {"metric": "n_controls", "value": int((episode_master["case_control"] == 0).sum())},
        {"metric": "n_unique_child_id", "value": int(episode_master["child_id"].nunique(dropna=True))},
        {"metric": "n_unique_episode_id", "value": int(episode_master["episode_id"].nunique(dropna=True))},

        {"metric": "modeling_ready_n_rows", "value": int(len(modeling_ready))},
        {"metric": "modeling_ready_n_cases", "value": int((modeling_ready["case_control"] == 1).sum())},
        {"metric": "modeling_ready_n_controls", "value": int((modeling_ready["case_control"] == 0).sum())},

        {"metric": "min_event_date", "value": str(episode_master["event_date"].min())},
        {"metric": "max_event_date", "value": str(episode_master["event_date"].max())},
    ])
    qc_summary.to_csv(qc_dir / "qc_episode_master_summary.csv", index=False)

    manifest = {
        "output_files": {
            "episode_master_enriched": str(cleaned_dir / "episode_master_enriched.parquet"),
            "episode_master": str(cleaned_dir / "episode_master.parquet"),
            "episode_master_modeling_ready": str(cleaned_dir / "episode_master_modeling_ready.parquet"),
        },
        "n_rows_enriched": int(len(episode_master_enriched)),
        "n_rows_episode_master": int(len(episode_master)),
        "n_rows_modeling_ready": int(len(modeling_ready)),
        "columns_episode_master": list(episode_master.columns),
    }
    with open(qc_dir / "qc_episode_master_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved: {cleaned_dir / 'episode_master_enriched.parquet'}")
    logging.info(f"Saved: {cleaned_dir / 'episode_master.parquet'}")
    logging.info(f"Saved: {cleaned_dir / 'episode_master_modeling_ready.parquet'}")
    logging.info("Saved QC files.")
    logging.info("Done.")


if __name__ == "__main__":
    main()
