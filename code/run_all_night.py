import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE = os.path.join(BASE, "ensemble_cv_compare.py")
RECOMPUTE = os.path.join(BASE, "recompute_ensemble_foldwise.py")
OUTDIR = os.path.join(BASE, "ensemble_results_cvcompare")

PATHOGENS = ["shigella", "st_etec", "rotavirus", "cryptosporidium", "norovirus"]

CONFIGS = [
    ("random",  "_clean",     True,  False, "clean random  (272/269 features)"),
    ("country", "_clean",     True,  False, "clean country (272/269 features)"),
    ("random",  "_cleanplus", True,  True,  "cleanplus random  (271/268 features)"),
    ("country", "_cleanplus", True,  True,  "cleanplus country (271/268 features)"),
]


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def oof_path(pathogen, cv_mode, tag):
    # 文件名与 recompute_ensemble_foldwise.py 的 load_oof 完全一致
    return os.path.join(OUTDIR, f"{pathogen}_{cv_mode}_site{tag}_oof.csv")


def config_done(cv_mode, tag):
    """该配置的 5 个病原 OOF 是否都已存在且非空。"""
    return all(
        os.path.exists(oof_path(p, cv_mode, tag)) and
        os.path.getsize(oof_path(p, cv_mode, tag)) > 0
        for p in PATHOGENS
    )


def run_ensemble(cfg, trials, timeout):
    cv_mode, tag, drop_beh, drop_bur, label = cfg
    log(f"{'='*72}")
    log(f"STEP: {label}  (cv_mode={cv_mode}, tag='{tag}')")
    log(f"{'='*72}")
    cmd = [
        sys.executable, ENSEMBLE,
        "--all-pathogens",
        "--cv-mode", cv_mode,
        "--trials", str(trials),
        "--timeout", str(timeout),
    ]
    if drop_beh:
        cmd.append("--drop-behavior")
    if drop_bur:
        cmd.append("--drop-burden")
    cmd.append("--tag-suffix")
    cmd.append(tag)

    log(f"$ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=BASE)
    dt = time.time() - t0
    log(f"   -> exit code {proc.returncode}  (elapsed {dt/60:.1f} min)")

    missing = [p for p in PATHOGENS if not os.path.exists(oof_path(p, cv_mode, tag))]
    if missing:
        log(f"   [WARN] 以下病原未生成 OOF: {missing}")
    else:
        log(f"   [OK] 5/5 病原 OOF 已生成")
    return proc.returncode


def run_recompute(tag):
    log(f"{'='*72}")
    log(f"STEP: recompute foldwise weights  (tag='{tag}')")
    log(f"{'='*72}")
    cmd = [sys.executable, RECOMPUTE, "--tag-suffix", tag]
    log(f"$ {' '.join(cmd)}")
    t0 = time.time()
    with open(os.path.join(BASE, f"recompute_{tag.strip('_')}.log"), "w") as f:
        proc = subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    log(f"   -> exit code {proc.returncode}  (elapsed {dt/60:.1f} min)")
    log(f"   (完整输出见 recompute_{tag.strip('_')}.log)")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="通宵补跑 clean/cleanplus 的 random+country CV")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--skip-cleanplus", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    log(f"项目目录: {BASE}")
    log(f"参数: trials={args.trials}, timeout={args.timeout}s, "
        f"skip_cleanplus={args.skip_cleanplus}, force={args.force}")
    if not os.path.exists(ENSEMBLE):
        log(f"[FATAL] 找不到 {ENSEMBLE} —— 请把本脚本放到与 ensemble_cv_compare.py 同级目录")
        sys.exit(1)

    os.makedirs(OUTDIR, exist_ok=True)
    t_start = time.time()
    ok, skipped, failed = [], [], []

    for cfg in CONFIGS:
        cv_mode, tag, *_ = cfg
        if args.skip_cleanplus and tag == "_cleanplus":
            log(f"[SKIP-CONFIG] cleanplus 已被 --skip-cleanplus 跳过")
            continue
        if config_done(cv_mode, tag) and not args.force:
            log(f"[SKIP-DONE] {cv_mode}{tag} 的 5 病原 OOF 已存在, 跳过 (加 --force 可重跑)")
            skipped.append(f"{cv_mode}{tag}")
            continue
        rc = run_ensemble(cfg, args.trials, args.timeout)
        if config_done(cv_mode, tag):
            ok.append(f"{cv_mode}{tag}")
        else:
            failed.append(f"{cv_mode}{tag}")

    # 重算修正后数字(只在有 OOF 的 tag 上跑)
    recompute_tags = []
    for tag in ["_clean", "_cleanplus"]:
        if not args.skip_cleanplus or tag == "_clean":
            if any(config_done(cv, tag) for cv in ["random", "site", "country"]):
                recompute_tags.append(tag)
    for tag in recompute_tags:
        run_recompute(tag)

    elapsed = (time.time() - t_start) / 60
    summary_lines = [
        f"=== RUN ALL NIGHT SUMMARY ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===",
        f"总耗时: {elapsed:.1f} 分钟",
        f"本次完成: {ok if ok else '无'}",
        f"跳过(已完成): {skipped if skipped else '无'}",
        f"失败: {failed if failed else '无'}",
        f"",
        f"产物清单:",
        *[f"  - {p}" for p in sorted(os.listdir(OUTDIR))
          if p.endswith('_clean_oof.csv') or p.endswith('_cleanplus_oof.csv')],
        f"",
        f"修正后数字:",
        *[f"  - recompute_{tag.strip('_')}.log / recomputed_ensemble{tag}.csv"
          for tag in recompute_tags],
        f"",
        f"提示: 若有失败, 重跑本脚本即可自动续跑失败/未完成的配置。",
    ]
    summary_path = os.path.join(
        BASE, f"summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    log(f"\n{'='*72}")
    log("\n".join(summary_lines))
    log(f"\n摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
