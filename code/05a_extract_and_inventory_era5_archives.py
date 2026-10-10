#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05a_extract_and_inventory_era5_archives.py

Purpose
-------
Extract all ERA5 zip archives under:
    00_raw/ERA5/era5_downloads

And generate:
1) zip inventory
2) extracted file registry
3) extraction summary / manifest

Main outputs
------------
1) 02_intermediate/qc/qc_era5_zip_inventory.csv
2) 02_intermediate/qc/qc_era5_extracted_registry.csv
3) 02_intermediate/qc/qc_era5_extraction_summary.csv
4) 02_intermediate/qc/qc_era5_extraction_manifest.json

Extraction target
-----------------
01_extracted/ERA5/<zip_stem>/...

Design notes
------------
- Safe extraction with zip-slip protection
- Idempotent enough for reruns:
  existing files are not overwritten unless --overwrite is specified
- Preserves internal folder structure inside each zip
"""

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import List, Tuple

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


def resolve_paths(cfg: dict) -> dict:
    project_root = Path(cfg.get("project_root", ".")).resolve()
    raw = cfg.get("raw", {})
    derived = cfg.get("derived", {})

    extracted_root = project_root / derived.get("extracted_dir", "01_extracted")
    era5_extracted_dir = extracted_root / "ERA5"

    return {
        "project_root": project_root,
        "era5_zip_dir": project_root / raw.get("era5_dir", "00_raw/ERA5/era5_downloads"),
        "qc_dir": project_root / derived.get("qc_dir", "02_intermediate/qc"),
        "era5_extracted_dir": era5_extracted_dir,
    }


# =========================================================
# zip scanning
# =========================================================
def scan_era5_zip_files(era5_zip_dir: Path) -> pd.DataFrame:
    if not era5_zip_dir.exists():
        raise FileNotFoundError(f"ERA5 zip directory not found: {era5_zip_dir}")

    rows = []
    for p in era5_zip_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".zip":
            continue

        rows.append({
            "zip_path": str(p),
            "zip_name": p.name,
            "zip_stem": p.stem,
            "parent_dir": str(p.parent),
            "file_size_bytes": p.stat().st_size,
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=[
            "zip_path", "zip_name", "zip_stem", "parent_dir", "file_size_bytes"
        ])

    out = out.sort_values(["zip_name", "zip_path"]).reset_index(drop=True)
    return out


def build_unique_extract_dir(base_dir: Path, zip_stem: str, existing_names: set) -> Path:
    stem = zip_stem
    candidate = stem
    i = 1
    while candidate in existing_names:
        candidate = f"{stem}__dup{i}"
        i += 1
    existing_names.add(candidate)
    return base_dir / candidate


# =========================================================
# safe extraction
# =========================================================
def is_within_directory(base_dir: Path, target_path: Path) -> bool:
    try:
        base_resolved = base_dir.resolve()
        target_resolved = target_path.resolve()
        return str(target_resolved).startswith(str(base_resolved))
    except Exception:
        return False


def safe_member_target(extract_root: Path, member_name: str) -> Path:
    """
    Compute safe target path and reject zip-slip attempts.
    """
    member_path = Path(member_name)
    target = extract_root / member_path
    if not is_within_directory(extract_root, target):
        raise ValueError(f"Unsafe zip member path detected: {member_name}")
    return target


def classify_extracted_file_type(file_name: str) -> str:
    ext = Path(file_name).suffix.lower()

    if ext in [".nc", ".nc4", ".cdf"]:
        return "netcdf"
    if ext in [".grib", ".grib2"]:
        return "grib"
    if ext in [".csv", ".txt", ".tsv"]:
        return "text_table"
    if ext in [".parquet"]:
        return "parquet"
    if ext in [".xlsx", ".xls"]:
        return "excel"
    if ext in [".json"]:
        return "json"
    if ext in [".zip"]:
        return "nested_zip"
    return "other"


def extract_one_zip(
    zip_path: Path,
    extract_dir: Path,
    project_root: Path,
    overwrite: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """
    Returns:
    - extracted_registry_rows (one row per member)
    - zip_summary_dict (one row summary)
    """
    registry_rows = []
    summary = {
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "extract_dir": str(extract_dir),
        "status": "started",
        "n_members_total": 0,
        "n_files_members": 0,
        "n_dirs_members": 0,
        "n_extracted_files": 0,
        "n_existing_files_skipped": 0,
        "n_failed_members": 0,
        "error_message": "",
    }

    try:
        ensure_dir(extract_dir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            summary["n_members_total"] = len(members)

            for info in members:
                member_name = info.filename

                is_dir = info.is_dir()
                if is_dir:
                    summary["n_dirs_members"] += 1
                else:
                    summary["n_files_members"] += 1

                row = {
                    "source_zip_path": str(zip_path),
                    "source_zip_name": zip_path.name,
                    "zip_member_name": member_name,
                    "zip_member_is_dir": is_dir,
                    "zip_member_compressed_size": int(info.compress_size),
                    "zip_member_uncompressed_size": int(info.file_size),
                    "extract_root": str(extract_dir),
                    "extracted_relative_path": "",
                    "extracted_path": "",
                    "extracted_file_ext": Path(member_name).suffix.lower(),
                    "extracted_file_type": classify_extracted_file_type(member_name),
                    "status": "",
                    "error_message": "",
                }

                try:
                    target_path = safe_member_target(extract_dir, member_name)

                    if is_dir:
                        ensure_dir(target_path)
                        rel = target_path.relative_to(project_root)
                        row["extracted_relative_path"] = str(rel)
                        row["extracted_path"] = str(target_path)
                        row["status"] = "directory_created"
                        registry_rows.append(row)
                        continue

                    ensure_dir(target_path.parent)

                    if target_path.exists() and not overwrite:
                        summary["n_existing_files_skipped"] += 1
                        rel = target_path.relative_to(project_root)
                        row["extracted_relative_path"] = str(rel)
                        row["extracted_path"] = str(target_path)
                        row["status"] = "existing_skipped"
                        registry_rows.append(row)
                        continue

                    with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                        dst.write(src.read())

                    summary["n_extracted_files"] += 1
                    rel = target_path.relative_to(project_root)
                    row["extracted_relative_path"] = str(rel)
                    row["extracted_path"] = str(target_path)
                    row["status"] = "extracted"
                    registry_rows.append(row)

                except Exception as member_e:
                    summary["n_failed_members"] += 1
                    row["status"] = "failed_member"
                    row["error_message"] = str(member_e)[:500]
                    registry_rows.append(row)

        if summary["n_failed_members"] == 0:
            summary["status"] = "success"
        else:
            summary["status"] = "partial_success"

    except Exception as e:
        summary["status"] = f"failed_{type(e).__name__}"
        summary["error_message"] = str(e)[:500]

    return pd.DataFrame(registry_rows), summary


# =========================================================
# summary builders
# =========================================================
def build_zip_inventory_with_member_counts(
    zip_inventory: pd.DataFrame,
    extraction_summary_df: pd.DataFrame
) -> pd.DataFrame:
    if len(zip_inventory) == 0:
        return zip_inventory

    out = zip_inventory.copy()
    if len(extraction_summary_df) > 0:
        merge_cols = [
            "zip_path", "status", "n_members_total", "n_files_members", "n_dirs_members",
            "n_extracted_files", "n_existing_files_skipped", "n_failed_members",
            "extract_dir", "error_message"
        ]
        out = out.merge(
            extraction_summary_df[merge_cols],
            on="zip_path",
            how="left"
        )
    return out


# =========================================================
# main
# =========================================================
def main():
    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to project_config.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If provided, overwrite already extracted files."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)

    ensure_dir(paths["qc_dir"])
    ensure_dir(paths["era5_extracted_dir"])

    zip_inventory = scan_era5_zip_files(paths["era5_zip_dir"])
    if len(zip_inventory) == 0:
        raise RuntimeError(f"No zip files found under {paths['era5_zip_dir']}")

    logging.info(f"ERA5 zip files found: {len(zip_inventory)}")

    existing_extract_names = set()
    all_registry_rows = []
    extraction_summaries = []

    for _, row in zip_inventory.iterrows():
        zip_path = Path(row["zip_path"])
        extract_dir = build_unique_extract_dir(
            base_dir=paths["era5_extracted_dir"],
            zip_stem=row["zip_stem"],
            existing_names=existing_extract_names
        )

        logging.info(f"Extracting: {zip_path.name} -> {extract_dir}")

        registry_df, summary_dict = extract_one_zip(
            zip_path=zip_path,
            extract_dir=extract_dir,
            project_root=paths["project_root"],
            overwrite=args.overwrite
        )

        summary_dict["zip_path"] = str(zip_path)
        summary_dict["zip_name"] = zip_path.name
        summary_dict["extract_dir"] = str(extract_dir)

        all_registry_rows.append(registry_df)
        extraction_summaries.append(summary_dict)

    extracted_registry_df = (
        pd.concat(all_registry_rows, axis=0, ignore_index=True, sort=False)
        if len(all_registry_rows) > 0 else
        pd.DataFrame(columns=[
            "source_zip_path", "source_zip_name", "zip_member_name", "zip_member_is_dir",
            "zip_member_compressed_size", "zip_member_uncompressed_size", "extract_root",
            "extracted_relative_path", "extracted_path", "extracted_file_ext",
            "extracted_file_type", "status", "error_message"
        ])
    )

    extraction_summary_df = pd.DataFrame(extraction_summaries)
    zip_inventory_qc_df = build_zip_inventory_with_member_counts(zip_inventory, extraction_summary_df)

    # output paths
    zip_inventory_path = paths["qc_dir"] / "qc_era5_zip_inventory.csv"
    extracted_registry_path = paths["qc_dir"] / "qc_era5_extracted_registry.csv"
    extraction_summary_path = paths["qc_dir"] / "qc_era5_extraction_summary.csv"
    manifest_path = paths["qc_dir"] / "qc_era5_extraction_manifest.json"

    zip_inventory_qc_df.to_csv(zip_inventory_path, index=False, encoding="utf-8-sig")
    extracted_registry_df.to_csv(extracted_registry_path, index=False, encoding="utf-8-sig")
    extraction_summary_df.to_csv(extraction_summary_path, index=False, encoding="utf-8-sig")

    manifest = {
        "era5_zip_dir": str(paths["era5_zip_dir"]),
        "era5_extracted_dir": str(paths["era5_extracted_dir"]),
        "overwrite_mode": bool(args.overwrite),
        "n_zip_files_found": int(len(zip_inventory)),
        "n_zip_success": int((extraction_summary_df["status"] == "success").sum()) if len(extraction_summary_df) > 0 else 0,
        "n_zip_partial_success": int((extraction_summary_df["status"] == "partial_success").sum()) if len(extraction_summary_df) > 0 else 0,
        "n_zip_failed": int(
            extraction_summary_df["status"].astype(str).str.startswith("failed").sum()
        ) if len(extraction_summary_df) > 0 else 0,
        "n_registry_rows": int(len(extracted_registry_df)),
        "output_files": {
            "qc_era5_zip_inventory_csv": str(zip_inventory_path),
            "qc_era5_extracted_registry_csv": str(extracted_registry_path),
            "qc_era5_extraction_summary_csv": str(extraction_summary_path),
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logging.info(f"Wrote: {zip_inventory_path}")
    logging.info(f"Wrote: {extracted_registry_path}")
    logging.info(f"Wrote: {extraction_summary_path}")
    logging.info(f"Wrote: {manifest_path}")
    logging.info("05a_extract_and_inventory_era5_archives.py completed successfully.")


if __name__ == "__main__":
    main()
