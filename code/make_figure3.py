import os
import sys
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'Liberation Sans',           # Arial metric-compatible
    'font.size': 9, 'axes.linewidth': 0.9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
    'pdf.fonttype': 42, 'ps.fonttype': 42,      # embed editable TrueType
})

PATH_ORDER = ['Shigella', 'ST-ETEC', 'Rotavirus', 'Cryptosporidium', 'Norovirus']
KEY = {'Shigella': 'shigella', 'ST-ETEC': 'st_etec', 'Rotavirus': 'rotavirus',
       'Cryptosporidium': 'cryptosporidium', 'Norovirus': 'norovirus'}
COL = {'Shigella': '#4C72B0', 'ST-ETEC': '#C44E52', 'Rotavirus': '#55A868',
       'Cryptosporidium': '#8172B2', 'Norovirus': '#DD8452'}
GREY = '#555555'; INK = '#1A1A1A'

SKIP = {'venv', '.venv', 'env', '.git', 'site-packages', '__pycache__',
        'node_modules', '.mypy_cache', '.ipynb_checkpoints'}


def find_file(target, roots):
    """Search each root upward, recursively, for a file named `target`."""
    seen = set()
    for root in roots:
        d = os.path.abspath(root)
        while True:
            if d in seen:
                break
            seen.add(d)
            for dirpath, dirnames, filenames in os.walk(d):
                dirnames[:] = [x for x in dirnames if x not in SKIP and not x.startswith('.')]
                if target in filenames:
                    return os.path.join(dirpath, target)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


def find_dir(contains_name, roots):
    """Find a directory that contains a file matching contains_name (exact)."""
    seen = set()
    for root in roots:
        d = os.path.abspath(root)
        while True:
            if d in seen:
                break
            seen.add(d)
            for dirpath, dirnames, filenames in os.walk(d):
                dirnames[:] = [x for x in dirnames if x not in SKIP and not x.startswith('.')]
                if contains_name in filenames:
                    return dirpath
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


def resolve_inputs(args):
    roots = [OUT, os.getcwd()]

    cal = args.cal or os.environ.get('GEMS_CAL_CSV')
    if not cal:
        cal = find_file('calibration_summary_site_clean.csv', roots)
    if not cal or not os.path.isfile(cal):
        sys.exit(
            "ERROR: could not find calibration_summary_site_clean.csv.\n"
            "Pass it explicitly:  python make_figure3.py --cal /path/to/calibration_summary_site_clean.csv\n"
            "or set GEMS_CAL_CSV=/path/to/that/file.csv")

    binned_dir = args.binned_dir or os.environ.get('GEMS_BINNED_DIR')
    if not binned_dir:
        # locate the folder that holds the per-pathogen binned-calibration CSVs
        binned_dir = find_file('shigella_site_clean_binned_calibration.csv', roots)
        if binned_dir:
            binned_dir = os.path.dirname(binned_dir)
    if not binned_dir or not os.path.isdir(binned_dir):
        sys.exit(
            "ERROR: could not find the folder containing "
            "<pathogen>_site_clean_binned_calibration.csv.\n"
            "Pass it explicitly:  python make_figure3.py --binned-dir /path/to/final_results_plots\n"
            "or set GEMS_BINNED_DIR=/path/to/that/folder")

    return cal, binned_dir


def save(fig, name):
    p1 = os.path.join(OUT, name + '.png'); p2 = os.path.join(OUT, name + '.pdf')
    fig.savefig(p1); fig.savefig(p2); plt.close(fig)
    print('  [ok] %-28s png %8d B   pdf %8d B' % (name, os.path.getsize(p1), os.path.getsize(p2)))


def figure3(cal_path, binned_dir):
    cal = pd.read_csv(cal_path).set_index('pathogen_label')

    fig = plt.figure(figsize=(11.0, 10.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.72], hspace=0.38, wspace=0.28,
                          left=0.072, right=0.985, top=0.965, bottom=0.075)

    for i, p in enumerate(PATH_ORDER):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        d = pd.read_csv(os.path.join(binned_dir, '%s_site_clean_binned_calibration.csv' % KEY[p]))
        ax.plot([0, 1], [0, 1], ls=(0, (5, 4)), color='#A8A8A8', lw=1.0, zorder=1)
        ax.plot(d.pred_mean, d.obs_rate, '-o', color=COL[p], lw=1.8, ms=4.4,
                mec='white', mew=0.7, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        b = cal.loc[p]
        ax.set_title('%s\nAUC %.4f  |  slope %.2f' % (p, b.auc, b.calibration_slope_joint),
                     fontsize=8.6, color=INK, pad=6)
        ax.tick_params(labelsize=7.2, colors='#333333')
        if i >= 3: ax.set_xlabel('Predicted probability', fontsize=7.9, color=INK)
        if i % 3 == 0: ax.set_ylabel('Observed proportion', fontsize=7.9, color=INK)

    # Key panel (legend only; no descriptive text box -> it lives in the caption)
    axk = fig.add_subplot(gs[1, 2]); axk.axis('off')
    axk.text(0.02, 0.94, 'Key', fontsize=9.4, fontweight='bold', va='top', color=INK,
             transform=axk.transAxes)
    for j, p in enumerate(PATH_ORDER):
        yv = 0.80 - j * 0.115
        axk.plot(0.075, yv, marker='o', color=COL[p], ms=8.0, mec='white', mew=0.8,
                 transform=axk.transAxes)
        axk.text(0.175, yv, p, fontsize=8.4, va='center', color=INK, transform=axk.transAxes)

    # Brier skill score panel
    axb = fig.add_subplot(gs[2, :])
    b = cal.loc[PATH_ORDER]
    vals = b.brier_skill_score.values; lo = b.brier_skill_score_ci_low.values
    hi = b.brier_skill_score_ci_high.values
    ys = np.arange(len(PATH_ORDER))[::-1]
    for y_, v, l_, h_, p in zip(ys, vals, lo, hi, PATH_ORDER):
        axb.barh(y_, v, height=0.56, color=COL[p], edgecolor='white', linewidth=0.8, zorder=2)
        axb.plot([l_, h_], [y_, y_], color='#3F3F3F', lw=1.3, zorder=3)
        for xc in (l_, h_):
            axb.plot([xc, xc], [y_ - 0.12, y_ + 0.12], color='#3F3F3F', lw=1.3, zorder=3)
        axb.annotate('%+.3f (%.3f to %.3f)' % (v, l_, h_), (max(v, h_, 0.0), y_),
                     textcoords='offset points', xytext=(13, 0), va='center', ha='left',
                     fontsize=7.3, color=INK, zorder=5)
    axb.axvline(0, color='#666666', lw=1.0, ls=(0, (4, 3)), zorder=1)
    axb.set_yticks(ys); axb.set_yticklabels(PATH_ORDER, fontsize=8.6, color=INK)
    axb.set_xlim(-0.62, 0.60); axb.set_ylim(-0.7, len(PATH_ORDER) - 0.3)
    axb.set_xlabel('Brier skill score (versus prevalence-only reference)', fontsize=8.6, color=INK)
    axb.tick_params(labelsize=7.4, colors='#333333')
    axb.spines['left'].set_visible(False); axb.tick_params(axis='y', length=0)
    axb.set_title('Brier skill score', fontsize=9.8, fontweight='bold', loc='left',
                  color=INK, pad=8)

    # No in-figure title and no bottom footer: all descriptive text is in the
    # LaTeX caption, so nothing is duplicated on the artwork.
    save(fig, 'figure3_calibration')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build Figure 3 (calibration).')
    ap.add_argument('--cal', default=None,
                    help='path to calibration_summary_site_clean.csv')
    ap.add_argument('--binned-dir', dest='binned_dir', default=None,
                    help='folder containing <pathogen>_site_clean_binned_calibration.csv')
    args = ap.parse_args()

    cal_path, binned_dir = resolve_inputs(args)
    print('calibration summary :', cal_path)
    print('binned calibration  :', binned_dir)
    figure3(cal_path, binned_dir)
    print('done ->', OUT)
