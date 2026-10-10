import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import statsmodels.api as sm

from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT_DIR = Path(r"/Users/aikeyan/Desktop/Project-GEMS-IDP/Shigella_Optimization")
OUT_DIR = ROOT_DIR / "final_shap_calibration_outputs"
OUT_SHAP_DIR = OUT_DIR / "shap"
OUT_CAL_DIR = OUT_DIR / "calibration"

PATHOGENS = [
    "shigella",
    "st_etec",
    "rotavirus",
    "cryptosporidium",
    "norovirus",
]

DISPLAY_NAME = {
    "shigella": "Shigella",
    "st_etec": "ST-ETEC",
    "rotavirus": "Rotavirus",
    "cryptosporidium": "Cryptosporidium",
    "norovirus": "Norovirus",
}

COLORS = {
    "shigella": "#1f77b4",
    "st_etec": "#d62728",
    "rotavirus": "#2ca02c",
    "cryptosporidium": "#9467bd",
    "norovirus": "#ff7f0e",
}


SITE_FEATURE_LABEL = "site"


MANUAL_RESULTS_JSON = {
    # "shigella": ROOT_DIR / "results_all" / "shigella_site_clean_results.json.txt",
}
MANUAL_OOF_CSV = {
    # "shigella": ROOT_DIR / "oof_clean_raw" / "shigella_site_clean_oof.csv",
}
MANUAL_PARQUET = None
# 例如：ROOT_DIR / "final_input_probe" / "gems_master_full_v4.parquet"

BEESWARM_MAX_N = 1000
SHAP_BATCH_SIZE = 500
N_BINS = 10
RUN_BOOTSTRAP = True
N_BOOT = 2000
SEED = 42


# =========================
# 2. 基础工具
# =========================
def ensure_dirs():
    OUT_SHAP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CAL_DIR.mkdir(parents=True, exist_ok=True)


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1 - p))


def print_resolved(label, path):
    print(f"[RESOLVED] {label}: {path}", flush=True)


# =========================
# 3. 递归查找文件
# =========================
def all_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file()]


def require_root_exists(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在：{root}")


def choose_unique(candidates, what: str):
    uniq = sorted(set(candidates))
    if len(uniq) == 0:
        raise FileNotFoundError(f"没有找到 {what}")
    if len(uniq) > 1:
        msg = "\n".join(str(x) for x in uniq)
        raise RuntimeError(f"{what} 匹配到多个候选，请手动指定：\n{msg}")
    return uniq[0]


def locate_parquet(root: Path) -> Path:
    if MANUAL_PARQUET is not None:
        path = Path(MANUAL_PARQUET)
        if not path.exists():
            raise FileNotFoundError(f"手动指定的 parquet 不存在：{path}")
        print_resolved("parquet(manual)", path)
        return path

    files = all_files(root)
    exact = [p for p in files if p.name == "gems_master_full_v4.parquet"]
    path = choose_unique(exact, "gems_master_full_v4.parquet")
    print_resolved("parquet", path)
    return path


def locate_results_json(root: Path, pathogen: str) -> Path:
    if pathogen in MANUAL_RESULTS_JSON:
        path = Path(MANUAL_RESULTS_JSON[pathogen])
        if not path.exists():
            raise FileNotFoundError(f"手动指定的 results json 不存在：{path}")
        print_resolved(f"results_json(manual)-{pathogen}", path)
        return path

    files = all_files(root)

    exact_names = [
        f"{pathogen}_site_clean_results.json.txt",
        f"{pathogen}_site_clean_results.json",
        f"{pathogen}_site_site_clean_results.json.txt",
        f"{pathogen}_site_site_clean_results.json",
    ]
    for exact_name in exact_names:
        hit = [p for p in files if p.name.lower() == exact_name]
        if len(hit) == 1:
            print_resolved(f"results_json-{pathogen}", hit[0])
            return hit[0]
        if len(hit) > 1:
            return choose_unique(hit, f"{pathogen} site-clean results json")

    preferred_general = []
    for p in files:
        name = p.name.lower()
        if not (name.endswith(".json") or name.endswith(".json.txt") or name.endswith(".txt")):
            continue
        if not name.startswith(f"{pathogen}_"):
            continue
        stem = name
        if stem.endswith(".json.txt"):
            stem = stem[:-9]
        elif stem.endswith(".json"):
            stem = stem[:-5]
        elif stem.endswith(".txt"):
            stem = stem[:-4]
        suffix = stem[len(pathogen) + 1:]
        if suffix in {"site_clean_results", "site_site_clean_results"}:
            preferred_general.append(p)

    path = choose_unique(preferred_general, f"{pathogen} site-clean results json")
    print_resolved(f"results_json-{pathogen}", path)
    return path


def locate_oof_csv(root: Path, pathogen: str) -> Path:
    if pathogen in MANUAL_OOF_CSV:
        path = Path(MANUAL_OOF_CSV[pathogen])
        if not path.exists():
            raise FileNotFoundError(f"手动指定的 OOF csv 不存在：{path}")
        print_resolved(f"oof_csv(manual)-{pathogen}", path)
        return path

    files = all_files(root)

    exact_names = [
        f"{pathogen}_site_clean_oof.csv",
        f"{pathogen}_site_site_clean_oof.csv",
    ]
    for exact_name in exact_names:
        hit = [p for p in files if p.name.lower() == exact_name]
        if len(hit) == 1:
            print_resolved(f"oof_csv-{pathogen}", hit[0])
            return hit[0]
        if len(hit) > 1:
            return choose_unique(hit, f"{pathogen} site-clean OOF csv")

    preferred_general = []
    for p in files:
        name = p.name.lower()
        if not name.endswith(".csv"):
            continue
        if not name.startswith(f"{pathogen}_"):
            continue
        stem = name[:-4]
        suffix = stem[len(pathogen) + 1:]
        if suffix in {"site_clean_oof", "site_site_clean_oof"}:
            preferred_general.append(p)

    path = choose_unique(preferred_general, f"{pathogen} site-clean OOF csv")
    print_resolved(f"oof_csv-{pathogen}", path)
    return path


# =========================
# 4. SHAP 部分
# =========================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "event_month" in df.columns:
        m = pd.to_numeric(df["event_month"], errors="coerce")
        df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    for c in [
        "hh_persons_living_in_house_count",
        "hh_sleeping_rooms_in_dwelling_count",
        "hh_persons_5_years_in_household_count",
        "hh_household_wealth_index_numerical",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def feature_family(feature_name: str) -> str:
    c = feature_name.lower()

    if c.startswith("tmean_c") or "hot_day" in c or "cold_day" in c or "dewpoint" in c:
        return "temperature"
    if c.startswith("precip_mm") or "heavy_rain" in c or "dry_day" in c:
        return "precipitation"
    if c.startswith("relative_humidity") or "_rh_" in c or c.startswith("flag_high_rh") or c.startswith("flag_low_rh"):
        return "humidity"
    if c in ("month_sin", "month_cos", "event_month"):
        return "seasonality"
    if c in ("site", "country"):
        return "geography"
    if c in (
        "age_months",
        "sex",
        "caregiver_education",
        "breastfeeding_status",
        "ever_breastfed",
        "breastfed_week_before",
    ):
        return "demographic_child"
    if c in (
        "licensed_practitioner",
        "unlicensed_practitioner",
        "pharmacy",
        "traditional_healer",
        "seek_outside_care",
    ) or c.startswith((
        "ctx_bought_remedy",
        "ctx_pharmacy",
        "ctx_traditional_healer",
        "ctx_unlicensed",
        "ctx_treatment_cost",
        "ctx_not_perceived",
        "ctx_travel_cost",
        "ctx_distance_barrier",
        "ctx_transport_barrier",
        "ctx_work_time",
    )):
        return "care_seeking"
    if c.startswith("hh_") or c.startswith("ctx_") or "water" in c or "wash" in c or c == "given_stored_water":
        return "household_wash_ses"

    return "other"


def feature_display_name(feature_name: str) -> str:
    if feature_name == "site":
        return SITE_FEATURE_LABEL
    return feature_name


def load_result_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_matrix_for_pathogen(parquet_path: Path, pathogen: str, feature_list: list[str]):
    required_base_cols = [
        f"analytic_{pathogen}",
        f"outcome_{pathogen}",
        "ctx_female_pct",
        "event_month",
    ]

    engineered_cols = {"month_sin", "month_cos"}

    needed_cols = []
    seen = set()
    for c in required_base_cols + [x for x in feature_list if x not in engineered_cols]:
        if c not in seen:
            needed_cols.append(c)
            seen.add(c)

    df = pd.read_parquet(parquet_path, columns=needed_cols, engine="pyarrow")

    mask = (
        (df[f"analytic_{pathogen}"] == 1)
        & df["ctx_female_pct"].notna()
        & df[f"outcome_{pathogen}"].notna()
    )

    sub = df.loc[mask].copy().reset_index(drop=True)
    sub = engineer_features(sub)

    y = sub[f"outcome_{pathogen}"].astype(int).reset_index(drop=True)
    X = sub[feature_list].copy().reset_index(drop=True)

    cat_features = []
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = (
                X[c].astype("string")
                .fillna("missing")
                .replace({"nan": "missing", "None": "missing"})
            )
            cat_features.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    del df, sub
    gc.collect()

    return X, y, cat_features


def full_mean_abs_shap(model, X, y, cat_features, batch_size=500):
    n = len(X)
    p = X.shape[1]
    abs_sum = np.zeros(p, dtype=float)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)

        X_chunk = X.iloc[start:stop].copy()
        y_chunk = y.iloc[start:stop].copy()

        pool = Pool(X_chunk, label=y_chunk, cat_features=cat_features)
        shap_values = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
        abs_sum += np.abs(shap_values).sum(axis=0)

        del X_chunk, y_chunk, pool, shap_values
        gc.collect()

    return abs_sum / n


def make_beeswarm(model, X, y, cat_features, pathogen):
    rng = np.random.default_rng(SEED)
    n = len(X)
    take = min(BEESWARM_MAX_N, n)

    idx = rng.choice(n, size=take, replace=False)
    idx.sort()

    Xs = X.iloc[idx].copy()
    ys = y.iloc[idx].copy()

    pool = Pool(Xs, label=ys, cat_features=cat_features)
    shap_values = model.get_feature_importance(pool, type="ShapValues")[:, :-1]

    explanation = shap.Explanation(
        values=shap_values,
        data=Xs,
        feature_names=[feature_display_name(x) for x in X.columns],
    )

    plt.figure(figsize=(8.5, 6.0))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title(f"{DISPLAY_NAME[pathogen]}: SHAP beeswarm", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_SHAP_DIR / f"{pathogen}_site_clean_shap_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    del Xs, ys, pool, shap_values, explanation
    gc.collect()


def run_shap(root: Path):
    ensure_dirs()
    parquet_path = locate_parquet(root)

    family_rows = []
    top_feature_rows = []
    top15_rows = []
    path_manifest = []

    for pathogen in PATHOGENS:
        print(f"[SHAP] start: {pathogen}", flush=True)
        result_json_path = locate_results_json(root, pathogen)
        path_manifest.append({
            "pathogen": pathogen,
            "type": "results_json",
            "path": str(result_json_path),
        })

        res = load_result_json(result_json_path)
        feature_list = res["features"]
        params = res["best_params"]["catboost"].copy()

        X, y, cat_features = load_matrix_for_pathogen(parquet_path, pathogen, feature_list)

        params.update({
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": SEED,
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": 1,
        })

        model = CatBoostClassifier(**params)
        model.fit(X, y, cat_features=cat_features if len(cat_features) > 0 else None)

        mean_abs = full_mean_abs_shap(model, X, y, cat_features, batch_size=SHAP_BATCH_SIZE)

        imp = pd.DataFrame({
            "pathogen": pathogen,
            "pathogen_label": DISPLAY_NAME[pathogen],
            "feature_raw": feature_list,
            "feature_label": [feature_display_name(x) for x in feature_list],
            "family": [feature_family(x) for x in feature_list],
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        imp.to_csv(OUT_SHAP_DIR / f"{pathogen}_site_clean_shap_importance.csv", index=False)

        top1 = imp.iloc[0]
        top_feature_rows.append({
            "pathogen": pathogen,
            "pathogen_label": DISPLAY_NAME[pathogen],
            "n": len(X),
            "positive_n": int(y.sum()),
            "n_features": len(feature_list),
            "top_feature_raw": top1["feature_raw"],
            "top_feature_label": top1["feature_label"],
            "top_feature_family": top1["family"],
            "top_feature_mean_abs_shap": top1["mean_abs_shap"],
        })

        top15 = imp.head(15).copy()
        top15_rows.append(top15)

        plot_df = top15.iloc[::-1].copy()
        plt.figure(figsize=(8.2, 5.4))
        plt.barh(plot_df["feature_label"], plot_df["mean_abs_shap"], color=COLORS[pathogen])
        plt.xlabel("Mean absolute SHAP value")
        plt.title(f"{DISPLAY_NAME[pathogen]}: top 15 features")
        plt.tight_layout()
        plt.savefig(OUT_SHAP_DIR / f"{pathogen}_site_clean_shap_top15.png", dpi=220)
        plt.close()

        fam = (
            imp.groupby(["pathogen", "pathogen_label", "family"], as_index=False)["mean_abs_shap"]
            .sum()
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        fam["family_pct"] = fam["mean_abs_shap"] / fam["mean_abs_shap"].sum() * 100.0
        family_rows.append(fam)

        make_beeswarm(model, X, y, cat_features, pathogen)

        del X, y, model, imp, fam, top15
        gc.collect()

        print(f"[SHAP] done: {pathogen}", flush=True)

    family_long = pd.concat(family_rows, ignore_index=True)
    family_long.to_csv(OUT_SHAP_DIR / "shap_family_summary_long.csv", index=False)

    top_feature_summary = pd.DataFrame(top_feature_rows)
    top_feature_summary.to_csv(OUT_SHAP_DIR / "shap_top_feature_summary.csv", index=False)

    top15_long = pd.concat(top15_rows, ignore_index=True)
    top15_long.to_csv(OUT_SHAP_DIR / "shap_top15_long.csv", index=False)

    family_order = [
        "demographic_child",
        "seasonality",
        "temperature",
        "precipitation",
        "humidity",
        "geography",
        "household_wash_ses",
        "care_seeking",
        "other",
    ]

    heat = (
        family_long
        .pivot(index="family", columns="pathogen_label", values="family_pct")
        .fillna(0.0)
    )
    heat = heat.reindex([x for x in family_order if x in heat.index])

    plt.figure(figsize=(8.6, 5.8))
    im = plt.imshow(heat.values, aspect="auto", cmap="Blues")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            plt.text(j, i, f"{heat.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, label="Share of total mean absolute SHAP (%)")
    plt.title("Feature-family SHAP contribution by pathogen")
    plt.tight_layout()
    plt.savefig(OUT_SHAP_DIR / "shap_family_heatmap.png", dpi=220)
    plt.close()

    path_manifest.append({"pathogen": "all", "type": "parquet", "path": str(parquet_path)})
    pd.DataFrame(path_manifest).to_csv(OUT_SHAP_DIR / "resolved_input_paths_shap.csv", index=False)

    print("[SHAP] all done", flush=True)
    print(f"[SHAP] outputs: {OUT_SHAP_DIR}", flush=True)


# =========================
# 5. Calibration 部分
# =========================
def calibration_intercept_offset(y, p):
    y = np.asarray(y).astype(float)
    lp = logit(p)
    X = np.ones((len(y), 1), dtype=float)

    fit = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        offset=lp,
    ).fit()

    return float(fit.params[0])


def calibration_joint(y, p):
    y = np.asarray(y).astype(float)
    lp = logit(p)
    X = sm.add_constant(lp)

    fit = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
    ).fit()

    intercept = float(fit.params[0])
    slope = float(fit.params[1])
    return intercept, slope


def compute_calibration_metrics(df, prob_col="oof_weighted", true_col="y_true"):
    y = df[true_col].to_numpy(dtype=float)
    p = clip_prob(df[prob_col].to_numpy(dtype=float))

    prevalence = float(y.mean())
    auc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p))

    null_pred = np.repeat(prevalence, len(y))
    null_brier = float(brier_score_loss(y, null_pred))
    brier_skill_score = float(1.0 - (brier / null_brier))

    intercept_offset1 = calibration_intercept_offset(y, p)
    intercept_joint, slope_joint = calibration_joint(y, p)

    return {
        "n": int(len(df)),
        "positive_n": int(y.sum()),
        "prevalence": prevalence,
        "auc": auc,
        "ap": ap,
        "brier": brier,
        "null_brier": null_brier,
        "brier_skill_score": brier_skill_score,
        "calibration_intercept_offset1": intercept_offset1,
        "calibration_intercept_joint": intercept_joint,
        "calibration_slope_joint": slope_joint,
    }


def cluster_bootstrap_ci(df, group_col="group_site", prob_col="oof_weighted", true_col="y_true", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)

    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)

    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    out_rows = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        bs = df.iloc[idx].copy().reset_index(drop=True)

        if bs[true_col].nunique() < 2:
            continue

        try:
            out_rows.append(compute_calibration_metrics(bs, prob_col=prob_col, true_col=true_col))
        except Exception:
            continue

    return pd.DataFrame(out_rows)


def summarize_bootstrap_ci(bs_df):
    cols = [
        "auc",
        "ap",
        "brier",
        "null_brier",
        "brier_skill_score",
        "calibration_intercept_offset1",
        "calibration_intercept_joint",
        "calibration_slope_joint",
    ]

    out = {}
    for c in cols:
        out[f"{c}_ci_low"] = float(np.percentile(bs_df[c], 2.5))
        out[f"{c}_ci_high"] = float(np.percentile(bs_df[c], 97.5))
    return out


def make_single_calibration_plot(df, pathogen, prob_col="oof_weighted", true_col="y_true"):
    y = df[true_col].to_numpy(dtype=float)
    p = clip_prob(df[prob_col].to_numpy(dtype=float))

    frac_pos, mean_pred = calibration_curve(
        y,
        p,
        n_bins=N_BINS,
        strategy="quantile",
    )

    bin_df = pd.DataFrame({
        "pathogen": pathogen,
        "pathogen_label": DISPLAY_NAME[pathogen],
        "bin": np.arange(1, len(frac_pos) + 1),
        "pred_mean": mean_pred,
        "obs_rate": frac_pos,
    })
    bin_df.to_csv(OUT_CAL_DIR / f"{pathogen}_binned_calibration.csv", index=False)

    fig = plt.figure(figsize=(5.4, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.plot(mean_pred, frac_pos, marker="o", color=COLORS[pathogen], lw=1.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(DISPLAY_NAME[pathogen])

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(p, bins=25, color=COLORS[pathogen], alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    fig.suptitle("Calibration under health-centre-held-out validation", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_CAL_DIR / f"{pathogen}_calibration.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return bin_df


def make_panel_plot(all_bins):
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.6))
    axes = axes.flatten()

    for i, pathogen in enumerate(PATHOGENS):
        ax = axes[i]
        df = all_bins[all_bins["pathogen"] == pathogen].copy()

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(df["pred_mean"], df["obs_rate"], marker="o", color=COLORS[pathogen], lw=1.5)
        ax.set_title(DISPLAY_NAME[pathogen])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")

    axes[-1].axis("off")
    fig.suptitle("Calibration plots: weighted ensemble, site-clean OOF", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_CAL_DIR / "calibration_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_calibration(root: Path):
    ensure_dirs()
    path_manifest = []
    summary_rows = []
    all_bins = []

    for pathogen in PATHOGENS:
        print(f"[CAL] start: {pathogen}", flush=True)
        oof_path = locate_oof_csv(root, pathogen)
        path_manifest.append({
            "pathogen": pathogen,
            "type": "oof_csv",
            "path": str(oof_path),
        })

        df = pd.read_csv(oof_path)

        if "y_true" not in df.columns:
            raise ValueError(f"{oof_path} 缺少列 y_true")
        if "oof_weighted" not in df.columns:
            raise ValueError(f"{oof_path} 缺少列 oof_weighted")
        if "group_site" not in df.columns:
            raise ValueError(f"{oof_path} 缺少列 group_site")

        if df["y_true"].isna().any():
            raise ValueError(f"{pathogen}: y_true 有缺失值")
        if df["oof_weighted"].isna().any():
            raise ValueError(f"{pathogen}: oof_weighted 有缺失值")
        if (df["oof_weighted"] < 0).any() or (df["oof_weighted"] > 1).any():
            raise ValueError(f"{pathogen}: oof_weighted 超出 [0,1]")

        metrics = compute_calibration_metrics(df, prob_col="oof_weighted", true_col="y_true")
        metrics["pathogen"] = pathogen
        metrics["pathogen_label"] = DISPLAY_NAME[pathogen]
        summary_rows.append(metrics)

        bin_df = make_single_calibration_plot(df, pathogen, prob_col="oof_weighted", true_col="y_true")
        all_bins.append(bin_df)

        if RUN_BOOTSTRAP:
            bs_df = cluster_bootstrap_ci(
                df,
                group_col="group_site",
                prob_col="oof_weighted",
                true_col="y_true",
                n_boot=N_BOOT,
                seed=SEED,
            )
            bs_df.to_csv(OUT_CAL_DIR / f"{pathogen}_bootstrap_draws.csv", index=False)
            ci = summarize_bootstrap_ci(bs_df)
            summary_rows[-1].update(ci)

        print(f"[CAL] done: {pathogen}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_CAL_DIR / "calibration_summary_site_clean.csv", index=False)

    bins_long = pd.concat(all_bins, ignore_index=True)
    bins_long.to_csv(OUT_CAL_DIR / "calibration_bins_long.csv", index=False)

    make_panel_plot(bins_long)
    pd.DataFrame(path_manifest).to_csv(OUT_CAL_DIR / "resolved_input_paths_calibration.csv", index=False)

    print("[CAL] all done", flush=True)
    print(f"[CAL] outputs: {OUT_CAL_DIR}", flush=True)


# =========================
# 6. 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["shap", "calibration", "both"], default="both")
    args = parser.parse_args()

    require_root_exists(ROOT_DIR)
    ensure_dirs()

    if args.mode in ("shap", "both"):
        run_shap(ROOT_DIR)
    if args.mode in ("calibration", "both"):
        run_calibration(ROOT_DIR)

    print("done", flush=True)


if __name__ == "__main__":
    main()
