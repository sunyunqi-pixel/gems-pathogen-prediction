from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/Users/aikeyan/Desktop/Project-GEMS-IDP/Shigella_Optimization")
OUT_CSV = ROOT / "table5_recomputed_ci.csv"

PATHOGENS = ["shigella", "st_etec", "rotavirus", "cryptosporidium", "norovirus"]


def choose_unique(candidates, what):
    candidates = sorted(set(candidates))
    if len(candidates) == 0:
        raise FileNotFoundError(f"没有找到 {what}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"{what} 找到多个候选，请手动确认：\n" + "\n".join(str(x) for x in candidates)
        )
    return candidates[0]


def find_parquet(root: Path):
    files = [p for p in root.rglob("*") if p.is_file()]
    cands = [p for p in files if p.name == "gems_master_full_v4.parquet"]
    return choose_unique(cands, "gems_master_full_v4.parquet")


def find_site_clean_oof(root: Path, pathogen: str):
    files = [p for p in root.rglob("*") if p.is_file()]
    exact_names = {
        f"{pathogen}_site_clean_oof.csv",
        f"{pathogen}_site_site_clean_oof.csv",
    }
    cands = [p for p in files if p.name.lower() in exact_names]
    return choose_unique(cands, f"{pathogen} site-clean OOF csv")


def bootstrap_auc_ci(y, p, groups, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    groups = np.asarray(groups).astype(str)

    uniq = np.unique(groups)
    aucs = []

    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])

        yy = y[idx]
        pp = p[idx]

        if len(np.unique(yy)) < 2:
            continue

        aucs.append(roc_auc_score(yy, pp))

    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


PARQUET = find_parquet(ROOT)
print(f"[FOUND] parquet: {PARQUET}")

cols = [
    "case_control", "site", "ctx_female_pct",
    "analytic_shigella", "outcome_shigella",
    "analytic_st_etec", "outcome_st_etec",
    "analytic_rotavirus", "outcome_rotavirus",
    "analytic_cryptosporidium", "outcome_cryptosporidium",
    "analytic_norovirus", "outcome_norovirus",
]

df = pd.read_parquet(PARQUET, columns=cols)

rows = []

for p in PATHOGENS:
    oof_path = find_site_clean_oof(ROOT, p)
    print(f"[FOUND] {p} oof: {oof_path}")

    mask = (
        (df[f"analytic_{p}"] == 1)
        & df[f"outcome_{p}"].notna()
        & df["ctx_female_pct"].notna()
    )

    sub = df.loc[mask, ["case_control", "site", f"outcome_{p}"]].copy().reset_index(drop=True)
    oof = pd.read_csv(oof_path)

    aligned = (
        len(sub) == len(oof)
        and (sub[f"outcome_{p}"].astype(int).values == oof["y_true"].values).all()
    )
    if not aligned:
        raise RuntimeError(f"{p}: subgroup rows do not align to OOF")

    mixed_auc = roc_auc_score(oof["y_true"], oof["oof_weighted"])

    for subgroup_value, subgroup_name in [(1, "case"), (0, "control")]:
        idx = np.where(sub["case_control"].astype(int).values == subgroup_value)[0]

        yy = oof.loc[idx, "y_true"].values
        pp = oof.loc[idx, "oof_weighted"].values
        gg = sub.loc[idx, "site"].astype(str).values

        auc = roc_auc_score(yy, pp)
        lo, hi = bootstrap_auc_ci(yy, pp, gg, n_boot=2000, seed=42)

        rows.append({
            "pathogen": p,
            "subgroup": subgroup_name,
            "n": int(len(idx)),
            "positive_n": int(yy.sum()),
            "prevalence": float(yy.mean()),
            "auc": float(auc),
            "auc_ci_low": lo,
            "auc_ci_high": hi,
            "mixed_auc": float(mixed_auc),
            "delta_auc_vs_mixed": float(auc - mixed_auc),
            "oof_path": str(oof_path),
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

print("\n===== TABLE 5 RECOMPUTED CI =====")
print(out.to_string(index=False))
print(f"\nSaved to: {OUT_CSV}")
