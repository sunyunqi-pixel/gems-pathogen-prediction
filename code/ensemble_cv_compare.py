import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
)

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
DATA_PATH = "gems_master_full_v4.parquet"
OUTPUT_DIR = Path("./ensemble_results_cvcompare")
OUTPUT_DIR.mkdir(exist_ok=True)

PATHOGENS = ["shigella", "rotavirus", "cryptosporidium", "st_etec", "norovirus"]
N_SPLITS = 5
CV_SEED = 42

# ============================================================================
# FEATURE SELECTION
# ============================================================================
CLIMATE_FLAG_PREFIXES = (
    'flag_cold_day_', 'flag_dry_day_', 'flag_heavy_rain_',
    'flag_high_rh_', 'flag_hot_day_', 'flag_low_rh_',
    'flag_very_heavy_rain_', 'flag_very_hot_day_',
)
CLIMATE_EXACT = {'precip_mm', 'tmean_c', 'dewpoint_c', 'relative_humidity_pct'}
CLIMATE_PREFIXES = ('precip_mm_', 'tmean_c_', 'dewpoint_c_', 'relative_humidity_pct_')

CTX_VARS = [
    'ctx_female_pct', 'ctx_age_months_mean', 'ctx_age_months_median',
    'ctx_diarrhea_in_past_2_weeks_pct', 'ctx_treatment_cost_barrier_pct',
    'ctx_travel_cost_barrier_pct', 'ctx_distance_barrier_pct',
    'ctx_transport_barrier_pct', 'ctx_work_time_barrier_pct',
    'ctx_not_perceived_need_pct', 'ctx_bought_remedy_pct',
    'ctx_pharmacy_pct', 'ctx_traditional_healer_pct',
    'ctx_unlicensed_practitioner_pct',
]
OLD_HH_VARS = [
    'hh_household_wealth_index_numerical', 'hh_persons_living_in_house_count',
    'hh_sleeping_rooms_in_dwelling_count', 'hh_floor_material',
    'hh_drinking_water_container', 'hh_water_availability', 'hh_usual_hand_washing',
]
HEALTHCARE_VARS = [
    'licensed_practitioner', 'unlicensed_practitioner', 'pharmacy',
    'traditional_healer', 'seek_outside_care',
]
NEW_VARS = [
    'breastfeeding_status', 'caregiver_education', 'given_stored_water',
    'ever_breastfed', 'breastfed_week_before',
    'hh_piped_water', 'hh_nonpiped_tap', 'hh_nonpiped_notap', 'hh_rainwater',
    'hh_other_water', 'hh_waterguard', 'hh_watermaker', 'hh_soap',
    'hh_water_covered', 'hh_water_time',
    'hh_wash_eating', 'hh_wash_cooking', 'hh_wash_defecation',
    'hh_wash_child_def', 'hh_wash_animals', 'hh_wash_nursing',
    'hh_wash_other', 'hh_never_wash',
    'hh_electricity', 'hh_radio', 'hh_telephone', 'hh_motorcycle',
    'hh_bicycle', 'hh_car', 'hh_cart', 'hh_boat_motor',
    'hh_cow', 'hh_goat', 'hh_animal_dung',
    'hh_floor_raw', 'hh_wealth_num', 'hh_wealth_cat', 'hh_rooms',
    'hh_usual_wash', 'hh_water_avail', 'hh_wash_location',
    'hh_water_container', 'hh_water_observed', 'hh_fetch_daily',
    'hh_water_trips_day', 'hh_agri_land', 'hh_agri_residue',
]

# +++ SENSITIVITY +++ (1/6) 发病后求医行为变量组 + 存疑的社区腹泻负担变量
# 个体求医行为(Yes/No，含 "Never had diarrhea" 取值 -> 与患病状态耦合)
# 及其 site 层面的 ctx_*_pct 聚合，均为发病后信息，存在反向因果/泄漏风险。
BEHAVIOR_LEAKAGE = [
    'seek_outside_care', 'licensed_practitioner', 'unlicensed_practitioner',
    'pharmacy', 'traditional_healer',
    'ctx_bought_remedy_pct', 'ctx_pharmacy_pct', 'ctx_traditional_healer_pct',
    'ctx_treatment_cost_barrier_pct', 'ctx_not_perceived_need_pct',
    'ctx_unlicensed_practitioner_pct',
    'ctx_travel_cost_barrier_pct', 'ctx_distance_barrier_pct',
    'ctx_transport_barrier_pct', 'ctx_work_time_barrier_pct',
]
# 时序存疑：社区近期腹泻比例（可能是合法暴露，也可能含本例/循环定义）
COMMUNITY_BURDEN = ['ctx_diarrhea_in_past_2_weeks_pct']
# +++ END SENSITIVITY +++

LEAKAGE_PATTERNS = [
    'positive_in_controls', 'positive_in_cases', 'positive_any',
    'attributable', 'positive_result', '_result$', '_ct$', '_ct_',
    'assay', 'sample_', 'direct_', 'fallback', 'test_available',
    '^case_control$', '_case_control', 'target_pathogen_positive',
    'n_target_pathogens',
]
LEAKAGE_PREFIXES = (
    'label_', 'path_', 'analytic_', 'outcome_',
    'eligible_case_control_', 'non_missing_climate_core_',
    'pathogen_test_available_', 'main_label_column_',
)
HARD_BLOCK = {
    'case_control', '_case_control_std', 'any_target_pathogen_positive',
    'n_target_pathogens_positive', 'child_id', 'household_id', 'episode_id',
    'ml_master_row_id', 'row_id', 'event_date', 'date',
}


def is_leakage(col):
    c = col.lower()
    if col in HARD_BLOCK or c in HARD_BLOCK:
        return True
    for p in LEAKAGE_PREFIXES:
        if c.startswith(p):
            return True
    for pat in LEAKAGE_PATTERNS:
        if re.search(pat, c):
            return True
    for p in PATHOGENS:
        if p in c:
            labelish = ['positive', 'label', 'result', 'attributable',
                        'pathogen', 'test', 'detected']
            if any(tok in c for tok in labelish):
                return True
    return False


def engineer_features(df):
    """无泄漏特征工程：周期性月份 + 数值列类型修正。"""
    df = df.copy()
    if 'event_month' in df.columns:
        m = pd.to_numeric(df['event_month'], errors='coerce')
        df['month_sin'] = np.sin(2 * np.pi * m / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * m / 12.0)
    to_numeric_cols = [
        'hh_persons_living_in_house_count',
        'hh_sleeping_rooms_in_dwelling_count',
        'hh_persons_5_years_in_household_count',
        'hh_household_wealth_index_numerical',
    ]
    for c in to_numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# +++ SENSITIVITY +++ (2/6) select_features 新增 drop_behavior / drop_burden 开关
def select_features(df, use_site=True, drop_behavior=False, drop_burden=False):
    features = ['age_months', 'sex', 'event_month', 'country',
                'month_sin', 'month_cos',
                'caregiver_education',
                'hh_wealth_num', 'hh_wealth_cat', 'hh_rooms']
    if use_site:
        features.insert(4, 'site')

    for c in df.columns:
        cl = c.lower()
        if cl in CLIMATE_EXACT:
            features.append(c)
        elif cl.startswith(CLIMATE_PREFIXES):
            features.append(c)
        elif cl.startswith(CLIMATE_FLAG_PREFIXES):
            features.append(c)

    features += [c for c in CTX_VARS if c in df.columns]
    features += [c for c in OLD_HH_VARS if c in df.columns]
    features += [c for c in HEALTHCARE_VARS if c in df.columns]
    features += [c for c in NEW_VARS if c in df.columns]

    features = list(dict.fromkeys(features))
    features = [c for c in features if not is_leakage(c)]

    # +++ SENSITIVITY +++ 敏感性分析：剔除发病后行为 / 社区负担变量
    if drop_behavior:
        features = [c for c in features if c not in BEHAVIOR_LEAKAGE]
    if drop_burden:
        features = [c for c in features if c not in COMMUNITY_BURDEN]
    # +++ END SENSITIVITY +++

    keep = []
    for c in features:
        if c not in df.columns:
            continue
        if df[c].isna().mean() <= 0.95 and df[c].nunique(dropna=True) > 1:
            keep.append(c)
    return keep


# +++ SENSITIVITY +++ (3/6) prepare_data 透传 drop_behavior / drop_burden
def prepare_data(df, pathogen, common_support_only=True, use_site=True,
                 drop_behavior=False, drop_burden=False):
    analytic_col = f'analytic_{pathogen}'
    outcome_col = f'outcome_{pathogen}'
    mask = (df[analytic_col] == 1) & df[outcome_col].notna()
    if common_support_only and 'ctx_female_pct' in df.columns:
        mask = mask & df['ctx_female_pct'].notna()

    sub = df.loc[mask].copy()
    sub = engineer_features(sub)
    y = sub[outcome_col].astype(int).reset_index(drop=True)
    groups_country = sub['country'].astype(str).reset_index(drop=True)

    # +++ site 分组（供 cv-mode=site 使用）。若无 site 列则回退为 country +++
    if 'site' in sub.columns:
        groups_site = sub['site'].astype(str).reset_index(drop=True)
    else:
        print("WARNING: 'site' 列不存在，cv-mode=site 将回退为 country 分组")
        groups_site = groups_country.copy()

    feature_cols = select_features(sub, use_site=use_site,
                                   drop_behavior=drop_behavior, drop_burden=drop_burden)
    X = sub[feature_cols].copy().reset_index(drop=True)

    cat_features = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = (X[c].astype(str)
                        .replace({'nan': 'missing', 'None': 'missing'})
                        .fillna('missing'))
            cat_features.append(c)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')

    X_cat = X.copy()
    X_enc = X.copy()
    for c in cat_features:
        X_enc[c] = X_enc[c].astype('category')

    return X_cat, X_enc, y, groups_country, groups_site, feature_cols, cat_features


def scale_pos_weight(y):
    pos = float((y == 1).sum()); neg = float((y == 0).sum())
    return neg / pos if pos > 0 else 1.0


# +++ 唯一核心改动：make_folds 支持三种 cv_mode +++
def make_folds(X, y, groups_country, groups_site, cv_mode="country"):
    if cv_mode == "country":
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
        return list(sgkf.split(X, y, groups_country))
    elif cv_mode == "site":
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
        return list(sgkf.split(X, y, groups_site))
    elif cv_mode == "random":
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
        return list(skf.split(X, y))
    else:
        raise ValueError(f"未知 cv_mode: {cv_mode}")


# ============================================================================
# OOF 生成
# ============================================================================
def oof_catboost(params, X, y, folds, cat_features):
    oof = np.zeros(len(y))
    p = dict(params)
    p.update({'loss_function': 'Logloss', 'eval_metric': 'AUC',
              'random_seed': 42, 'verbose': 0, 'allow_writing_files': False,
              'cat_features': cat_features if cat_features else None})
    for tr, va in folds:
        m = CatBoostClassifier(**p)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]),
              early_stopping_rounds=100, use_best_model=True)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def oof_xgboost(params, X, y, folds):
    oof = np.zeros(len(y))
    p = dict(params)
    p.update({'objective': 'binary:logistic', 'eval_metric': 'auc',
              'tree_method': 'hist', 'enable_categorical': True,
              'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
              'early_stopping_rounds': 100})
    for tr, va in folds:
        m = XGBClassifier(**p)
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def oof_lightgbm(params, X, y, folds):
    import lightgbm as lgb
    oof = np.zeros(len(y))
    p = dict(params)
    p.update({'objective': 'binary', 'metric': 'auc',
              'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
    for tr, va in folds:
        m = LGBMClassifier(**p)
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def oof_logreg(params, X, y, folds, cat_features):
    num_cols = [c for c in X.columns if c not in cat_features]
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('oh', OneHotEncoder(handle_unknown='ignore',
                                               min_frequency=20))]), cat_features),
    ])
    oof = np.zeros(len(y))
    for tr, va in folds:
        pipe = Pipeline([('pre', pre),
                         ('lr', LogisticRegression(max_iter=2000,
                                                   class_weight='balanced', **params))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return oof


# ============================================================================
# 调参
# ============================================================================
def tune_catboost(X, y, folds, cat_features, spw, n_trials, timeout):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 800, 3000, step=100),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.5),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'border_count': trial.suggest_int('border_count', 64, 254),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100, log=True),
            'auto_class_weights': trial.suggest_categorical(
                'auto_class_weights', ['None', 'Balanced', 'SqrtBalanced']),
        }
        return roc_auc_score(y, oof_catboost(params, X, y, folds, cat_features))
    return _run_study("CatBoost", objective, n_trials, timeout)


def tune_xgboost(X, y, folds, spw, n_trials, timeout):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 4000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 30.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 200.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 10.0),
            'scale_pos_weight': trial.suggest_categorical(
                'scale_pos_weight', [1.0, spw, float(np.sqrt(spw))]),
        }
        return roc_auc_score(y, oof_xgboost(params, X, y, folds))
    return _run_study("XGBoost", objective, n_trials, timeout)


def tune_lightgbm(X, y, folds, spw, n_trials, timeout):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 4000, step=100),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 30.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200, log=True),
            'scale_pos_weight': trial.suggest_categorical(
                'scale_pos_weight', [1.0, spw, float(np.sqrt(spw))]),
        }
        return roc_auc_score(y, oof_lightgbm(params, X, y, folds))
    return _run_study("LightGBM", objective, n_trials, timeout)


def tune_logreg(X, y, folds, cat_features, n_trials, timeout):
    def objective(trial):
        params = {'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
                  'penalty': 'l2', 'solver': 'lbfgs'}
        return roc_auc_score(y, oof_logreg(params, X, y, folds, cat_features))
    return _run_study("LogReg", objective, min(n_trials, 40), timeout)


def _run_study(name, objective, n_trials, timeout):
    print(f"\n--- Tuning {name} ({n_trials} trials) ---")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    print(f"{name} best OOF AUC: {study.best_value:.4f}  ({time.time()-t0:.0f}s)")
    return study.best_params, study.best_value


# ============================================================================
# 融合
# ============================================================================
def optimize_weights(M, y):
    n = M.shape[1]
    def neg_auc(w):
        w = np.clip(w, 0, None); s = w.sum()
        return 0.0 if s == 0 else -roc_auc_score(y, M @ (w / s))
    best = None
    for _ in range(30):
        res = minimize(neg_auc, np.random.dirichlet(np.ones(n)),
                       method='Nelder-Mead',
                       options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-5})
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(best.x, 0, None); return w / w.sum()


def metrics(y, pred):
    return {'auc': float(roc_auc_score(y, pred)),
            'ap': float(average_precision_score(y, pred)),
            'brier': float(brier_score_loss(y, pred)),
            'log_loss': float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6)))}


# ============================================================================
# 主流程
# ============================================================================
# +++ SENSITIVITY +++ (4/6) run_pathogen 新增 drop_behavior / drop_burden / tag_suffix
def run_pathogen(df, pathogen, n_trials, timeout, common_support, use_site, cv_mode,
                 drop_behavior=False, drop_burden=False, tag_suffix=''):
    print(f"\n{'='*70}\nCV-COMPARE — {pathogen.upper()}  "
          f"(site={'ON' if use_site else 'OFF'}, cv_mode={cv_mode}, "
          f"drop_behavior={drop_behavior}, drop_burden={drop_burden})\n{'='*70}")

    X_cat, X_enc, y, groups_country, groups_site, feature_cols, cat_features = prepare_data(
        df, pathogen, common_support_only=common_support, use_site=use_site,
        drop_behavior=drop_behavior, drop_burden=drop_burden)

    leakage = [c for c in feature_cols if is_leakage(c)]
    assert not leakage, f"LEAKAGE DETECTED: {leakage}"

    # +++ SENSITIVITY +++ 打印实际被剔除的行为/负担变量，便于日志核对
    if drop_behavior or drop_burden:
        dropped = [c for c in (BEHAVIOR_LEAKAGE + COMMUNITY_BURDEN)
                   if c not in feature_cols]
        print(f"[SENSITIVITY] dropped {len(dropped)} vars: {dropped}")

    spw = scale_pos_weight(y)
    print(f"N={len(y):,}  prevalence={y.mean():.4f}  "
          f"features={len(feature_cols)} (cat={len(cat_features)})  spw={spw:.2f}")
    print(f"cv_mode={cv_mode}  "
          f"n_country_groups={groups_country.nunique()}  "
          f"n_site_groups={groups_site.nunique()}")
    print(f"site in features: {'site' in feature_cols}  "
          f"month_sin in features: {'month_sin' in feature_cols}")

    folds = make_folds(X_enc, y, groups_country, groups_site, cv_mode=cv_mode)

    cb_params, cb_auc = tune_catboost(X_cat, y, folds, cat_features, spw, n_trials, timeout)
    xgb_params, xgb_auc = tune_xgboost(X_enc, y, folds, spw, n_trials, timeout)
    lgb_params, lgb_auc = tune_lightgbm(X_enc, y, folds, spw, n_trials, timeout)
    lr_params, lr_auc = tune_logreg(X_cat, y, folds, cat_features, n_trials, timeout)

    print("\nRegenerating OOF with best params...")
    oof_cb = oof_catboost(cb_params, X_cat, y, folds, cat_features)
    oof_xgb = oof_xgboost(xgb_params, X_enc, y, folds)
    oof_lgb = oof_lightgbm(lgb_params, X_enc, y, folds)
    oof_lr = oof_logreg(lr_params, X_cat, y, folds, cat_features)

    M = np.column_stack([oof_cb, oof_xgb, oof_lgb, oof_lr])
    names = ['catboost', 'xgboost', 'lightgbm', 'logreg']

    print("\nSingle-model OOF AUC:")
    for n_, col in zip(names, M.T):
        print(f"  {n_:<10}: {roc_auc_score(y, col):.4f}")

    m_simple = metrics(y, M.mean(axis=1))
    w = optimize_weights(M, y); weighted = M @ w
    m_weighted = metrics(y, weighted)
    print("\nWeights: " + ", ".join(f"{n_}={wi:.3f}" for n_, wi in zip(names, w)))

    stack_oof = np.zeros(len(y))
    for tr, va in folds:
        meta = LogisticRegression(max_iter=1000, C=1.0)
        meta.fit(M[tr], y.iloc[tr]); stack_oof[va] = meta.predict_proba(M[va])[:, 1]
    m_stack = metrics(y, stack_oof)

    print(f"\n{'='*70}\nRESULTS — {pathogen.upper()}  (cv_mode={cv_mode})\n{'='*70}")
    print(f"{'Method':<22}{'AUC':>8}{'AP':>9}{'Brier':>9}")
    print("-" * 48)
    for n_, col in zip(names, M.T):
        mm = metrics(y, col)
        print(f"{n_:<22}{mm['auc']:>8.4f}{mm['ap']:>9.4f}{mm['brier']:>9.4f}")
    print(f"{'simple_average':<22}{m_simple['auc']:>8.4f}{m_simple['ap']:>9.4f}{m_simple['brier']:>9.4f}")
    print(f"{'weighted_average':<22}{m_weighted['auc']:>8.4f}{m_weighted['ap']:>9.4f}{m_weighted['brier']:>9.4f}")
    print(f"{'stacking_logreg':<22}{m_stack['auc']:>8.4f}{m_stack['ap']:>9.4f}{m_stack['brier']:>9.4f}")

    best = max([('simple_average', m_simple), ('weighted_average', m_weighted),
                ('stacking_logreg', m_stack)], key=lambda t: t[1]['auc'])
    print(f"\n>>> BEST ENSEMBLE: {best[0]}  AUC={best[1]['auc']:.4f}")

    site_tag = 'site' if use_site else 'nosite'
    # +++ SENSITIVITY +++ (5/6) tag 追加 tag_suffix，防止覆盖主结果
    tag = f"{cv_mode}_{site_tag}{tag_suffix}"
    results = {
        'pathogen': pathogen, 'use_site': use_site, 'cv_mode': cv_mode,
        'drop_behavior': drop_behavior, 'drop_burden': drop_burden,
        'tag_suffix': tag_suffix,
        'n': int(len(y)), 'prevalence': float(y.mean()),
        'n_country_groups': int(groups_country.nunique()),
        'n_site_groups': int(groups_site.nunique()),
        'n_features': len(feature_cols), 'features': feature_cols,
        'single_model_auc': {'catboost': float(cb_auc), 'xgboost': float(xgb_auc),
                             'lightgbm': float(lgb_auc), 'logreg': float(lr_auc)},
        'ensemble': {'simple_average': m_simple, 'weighted_average': m_weighted,
                     'stacking_logreg': m_stack},
        'weights': {n_: float(wi) for n_, wi in zip(names, w)},
        'best_method': best[0], 'best_auc': best[1]['auc'],
        'best_params': {'catboost': cb_params, 'xgboost': xgb_params,
                        'lightgbm': lgb_params, 'logreg': lr_params},
    }
    with open(OUTPUT_DIR / f"{pathogen}_{tag}_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    pd.DataFrame({'y_true': y.values,
                  'group_country': groups_country.values,
                  'group_site': groups_site.values,
                  'oof_catboost': oof_cb, 'oof_xgboost': oof_xgb,
                  'oof_lightgbm': oof_lgb, 'oof_logreg': oof_lr,
                  'oof_weighted': weighted, 'oof_stacking': stack_oof}
                 ).to_csv(OUTPUT_DIR / f"{pathogen}_{tag}_oof.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR}/{pathogen}_{tag}_results.json + _oof.csv")
    return results


def main():
    ap = argparse.ArgumentParser(description="CV strategy comparison (based on ensemble_v3)")
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--all-pathogens', action='store_true')
    ap.add_argument('--pathogen', type=str, default='shigella')
    ap.add_argument('--data', type=str, default=DATA_PATH)
    ap.add_argument('--use-all-rows', action='store_true')
    ap.add_argument('--no-site', action='store_true', help="对照：不加入 site 特征")
    ap.add_argument('--cv-mode', choices=['country', 'site', 'random'],
                    default='country',
                    help="CV 分组策略：country=外推新国家(最保守)；"
                         "site=外推新站点；random=分层随机(within-site,最乐观)")
    # +++ SENSITIVITY +++ (6/6) 新增命令行开关
    ap.add_argument('--drop-behavior', action='store_true',
                    help="敏感性分析：剔除发病后求医行为变量组(潜在反向因果/泄漏)")
    ap.add_argument('--drop-burden', action='store_true',
                    help="敏感性分析：额外剔除社区腹泻负担变量(时序存疑)")
    ap.add_argument('--tag-suffix', type=str, default='',
                    help="输出文件名追加后缀，防止覆盖主结果(如 _clean)")
    # +++ END SENSITIVITY +++
    args = ap.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data, engine='pyarrow')
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"CV MODE = {args.cv_mode}")

    common_support = not args.use_all_rows
    use_site = not args.no_site
    pathogens = PATHOGENS if args.all_pathogens else [args.pathogen]

    all_results = {}
    for p in pathogens:
        try:
            all_results[p] = run_pathogen(df, p, args.trials, args.timeout,
                                          common_support, use_site, args.cv_mode,
                                          args.drop_behavior, args.drop_burden,
                                          args.tag_suffix)
        except Exception as e:
            print(f"\nERROR for {p}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*70}\nFINAL SUMMARY (cv_mode={args.cv_mode})\n{'='*70}")
    print(f"{'Pathogen':<18}{'best_method':<20}{'best_AUC':>10}")
    print("-" * 48)
    for p, r in all_results.items():
        print(f"{p:<18}{r['best_method']:<20}{r['best_auc']:>10.4f}")


if __name__ == "__main__":
    main()
