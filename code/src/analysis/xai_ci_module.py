"""
xai_ci_module.py
----------------
Trains a Random Forest regressor mapping physiological vitals → CI score
on the Kaggle IoMT vital-signs dataset (Barman 2024,
doi:10.34740/KAGGLE/DSV/7736523; stored under data/Mendeley-IoMT/ for
historical reasons). Applies SHAP TreeExplainer for feature importance.
The CI target is threshold-derived from the dataset's alert columns
(_build_ci_target); the JSON records that construction, the split, the RF
parameters, the package versions and the input file's sha256.

Outputs:
  results/shap_feature_importance.json
  latex/figures/fig_shap_beeswarm.pdf
  latex/figures/fig_shap_bar.pdf
  latex/figures/fig_ci_distribution.pdf
"""

from __future__ import annotations

import sys
import os

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# IEEE single-column figure style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS_RAW = [
    'Heart Rate (bpm)',
    'SpO2 Level (%)',
    'Systolic Blood Pressure (mmHg)',
    'Diastolic Blood Pressure (mmHg)',
    'Body Temperature (°C)',
]
FEATURE_NAMES = [
    'Heart Rate',
    'SpO2 Level',
    'Systolic BP',
    'Diastolic BP',
    'Body Temperature',
]

ALERT_COL_KEYWORDS = {
    'hr_alert':   'Heart Rate Alert',
    'spo2_alert': 'SpO2 Level Alert',
    'bp_alert':   'Blood Pressure Alert',
    'temp_alert': 'Temperature Alert',
}

ALERT_SCORE_MAP = {
    'High':     0.70,
    'Low':      0.75,   # SpO2 Low is clinically dangerous
    'Abnormal': 0.60,
    'Normal':   0.10,
}

DISEASE_BOOST_MAP = {
    'Arrhythmia':  0.20,
    'Hypertension': 0.10,
    'Asthma':       0.12,
    'Diabetes':     0.08,
    'Normal':       0.00,
}

# Feature groups for colour coding in bar chart
FEATURE_GROUPS = {
    'Heart Rate':      'Cardiac',
    'SpO2 Level':      'Respiratory',
    'Systolic BP':     'Cardiac',
    'Diastolic BP':    'Cardiac',
    'Body Temperature': 'Thermal',
}
GROUP_COLORS = {
    'Cardiac':      '#D62728',
    'Respiratory':  '#1F77B4',
    'Thermal':      '#FF7F0E',
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _find_column(df_columns, keyword: str) -> Optional[str]:
    """Case-insensitive substring search for a column."""
    kw_lower = keyword.lower()
    for col in df_columns:
        if kw_lower in col.lower():
            return col
    return None


def _resolve_feature_columns(df_columns):
    """
    Try exact match first; fall back to substring search.
    Returns dict: clean_name -> actual_col_name or None.
    """
    resolved = {}
    pairs = zip(FEATURE_COLS_RAW, FEATURE_NAMES)
    for raw_col, clean_name in pairs:
        # Exact match (handles BOM and encoding variants)
        for col in df_columns:
            if col.strip('﻿').strip() == raw_col.strip():
                resolved[clean_name] = col
                break
        else:
            # Substring fallback
            keywords = {
                'Heart Rate': 'heart rate',
                'SpO2 Level': 'spo2',
                'Systolic BP': 'systolic',
                'Diastolic BP': 'diastolic',
                'Body Temperature': 'temperature',
            }
            kw = keywords.get(clean_name, clean_name.lower())
            found = _find_column(df_columns, kw)
            resolved[clean_name] = found
    return resolved


def _resolve_alert_column(df_columns, keyword: str) -> Optional[str]:
    """Find alert column by keyword."""
    for col in df_columns:
        col_clean = col.strip('﻿').strip()
        if keyword.lower() in col_clean.lower():
            return col
    return None


def _alert_score(value: str) -> float:
    """Map alert string to numeric CI contribution."""
    v = str(value).strip()
    return ALERT_SCORE_MAP.get(v, 0.10)


def _disease_boost(value: str) -> float:
    """Map disease name to CI boost."""
    v = str(value).strip()
    for disease, boost in DISEASE_BOOST_MAP.items():
        if disease.lower() in v.lower():
            return boost
    return 0.0


def _build_ci_target(df: pd.DataFrame) -> np.ndarray:
    """
    Construct CI target vector from alert and disease columns.

    CI = max(hr_alert_score, spo2_alert_score, bp_alert_score, temp_alert_score)
         + disease_boost
    Clipped to [0, 1].
    """
    n = len(df)
    alert_scores = np.full((n, 4), 0.10)

    alert_kws = [
        'Heart Rate Alert',
        'SpO2 Level Alert',
        'Blood Pressure Alert',
        'Temperature Alert',
    ]
    for j, kw in enumerate(alert_kws):
        col = _resolve_alert_column(df.columns, kw)
        if col is not None:
            alert_scores[:, j] = df[col].apply(_alert_score).values

    base_ci = alert_scores.max(axis=1)

    disease_col = None
    for kw in ['Predicted Disease', 'Disease', 'Diagnosis']:
        disease_col = _find_column(df.columns, kw)
        if disease_col is not None:
            break

    boost = np.zeros(n)
    if disease_col is not None:
        boost = df[disease_col].apply(_disease_boost).values

    ci = np.clip(base_ci + boost, 0.0, 1.0)
    return ci.astype(float)


# ---------------------------------------------------------------------------
# SHAP / fallback importance
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> Dict[str, str]:
    from importlib import metadata
    out = {}
    for pkg in ('shap', 'scikit-learn', 'numpy', 'pandas', 'openpyxl'):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = 'not installed'
    return out


def _try_import_shap():
    try:
        import shap
        return shap
    except ImportError:
        return None


def _compute_shap_values(model: RandomForestRegressor,
                         X_train: np.ndarray,
                         X_test: np.ndarray,
                         shap_mod):
    """Compute SHAP values using TreeExplainer; returns (shap_values, explainer)."""
    explainer = shap_mod.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values, explainer


def _compute_permutation_importance(model, X_test, y_test,
                                    feature_names, seed=42):
    """Fallback: permutation importance when SHAP is unavailable."""
    result = permutation_importance(model, X_test, y_test,
                                    n_repeats=30, random_state=seed)
    importances = result.importances_mean
    return importances


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _fig_shap_beeswarm(shap_values, X_test_df, figures_dir: Path,
                       shap_mod=None, mean_abs_shap=None,
                       feature_names=None):
    """
    Figure 1: SHAP beeswarm or fallback importance bars.
    IEEE single column: 3.5 × 3.0 inches.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    if shap_mod is not None and shap_values is not None:
        # Use SHAP's built-in beeswarm via matplotlib backend
        try:
            plt.close(fig)  # Close the blank figure we opened
            fig_shap = plt.figure(figsize=(3.5, 3.0))
            shap_mod.summary_plot(
                shap_values, X_test_df,
                plot_type='dot',
                show=False,
                plot_size=(3.5, 3.0),
                color_bar_label='Feature value',
            )
            plt.gcf().set_size_inches(3.5, 3.0)
            plt.title('SHAP Feature Importances for Criticality Index Prediction',
                      fontsize=8, pad=4)
            plt.tight_layout()
            out_pdf = figures_dir / 'fig_shap_beeswarm.pdf'
            out_png = figures_dir / 'fig_shap_beeswarm.png'
            plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close('all')
            print(f'[XAI] Saved {out_pdf}')
            return
        except Exception as exc:
            warnings.warn(f'[XAI] SHAP beeswarm failed ({exc}); using bar fallback.')
            plt.close('all')
            fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Fallback: horizontal bar chart of mean absolute importances
    if mean_abs_shap is None:
        mean_abs_shap = np.zeros(len(feature_names or FEATURE_NAMES))

    names = feature_names or FEATURE_NAMES
    order = np.argsort(mean_abs_shap)
    colors = [GROUP_COLORS[FEATURE_GROUPS.get(n, 'Cardiac')] for n in np.array(names)[order]]

    ax.barh(np.array(names)[order], mean_abs_shap[order],
            color=colors, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Mean |SHAP value|', fontsize=8)
    ax.set_title('SHAP Feature Importances for Criticality\nIndex Prediction',
                 fontsize=8, pad=4)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_pdf = figures_dir / 'fig_shap_beeswarm.pdf'
    out_png = figures_dir / 'fig_shap_beeswarm.png'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[XAI] Saved {out_pdf}')


def _fig_shap_bar(mean_abs_shap: np.ndarray, feature_names,
                  figures_dir: Path):
    """
    Figure 2: Mean absolute SHAP values horizontal bar chart.
    Coloured by feature group. IEEE single column: 3.5 × 2.8 inches.
    """
    names = feature_names or FEATURE_NAMES
    order = np.argsort(mean_abs_shap)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bar_colors = [GROUP_COLORS[FEATURE_GROUPS.get(n, 'Cardiac')]
                  for n in np.array(names)[order]]

    bars = ax.barh(np.array(names)[order], mean_abs_shap[order],
                   color=bar_colors, edgecolor='black', linewidth=0.5, height=0.6)

    # Value labels on bars
    for bar, val in zip(bars, mean_abs_shap[order]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=6.5)

    ax.set_xlabel('Mean |SHAP value|', fontsize=8)
    ax.set_title('Feature Contribution to CI Prediction', fontsize=8, pad=4)
    ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend for feature groups
    legend_patches = [
        mpatches.Patch(color=col, label=grp)
        for grp, col in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc='lower right',
              fontsize=6, framealpha=0.7, edgecolor='gray')

    plt.tight_layout()
    out_pdf = figures_dir / 'fig_shap_bar.pdf'
    out_png = figures_dir / 'fig_shap_bar.png'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[XAI] Saved {out_pdf}')


def _fig_ci_distribution(ci_values: np.ndarray, figures_dir: Path):
    """
    Figure 3: Histogram of CI scores with threshold annotations
    and dual y-axis for count and cumulative %.
    IEEE single column: 3.5 × 2.8 inches.
    """
    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))

    # Histogram
    n_bins = 30
    counts, bin_edges, patches = ax1.hist(
        ci_values, bins=n_bins, range=(0.0, 1.0),
        color='#4C72B0', edgecolor='white', linewidth=0.3, alpha=0.85,
        label='Count'
    )

    # Colour patches by CI tier
    for patch, left in zip(patches, bin_edges[:-1]):
        if left >= 0.7:
            patch.set_facecolor('#D62728')   # High
        elif left >= 0.4:
            patch.set_facecolor('#FF7F0E')   # Medium
        else:
            patch.set_facecolor('#2CA02C')   # Low

    ax1.set_xlabel('Criticality Index (CI)', fontsize=8)
    ax1.set_ylabel('Patient Count', fontsize=8, color='#333333')
    ax1.tick_params(axis='y', labelcolor='#333333')

    # Threshold vertical lines
    thresholds = [(0.4, 'Low|Med', '#2CA02C'), (0.7, 'Med|High', '#D62728')]
    for thr, label, color in thresholds:
        ax1.axvline(thr, color=color, linestyle='--', linewidth=0.8, alpha=0.8)
        ax1.text(thr + 0.01, counts.max() * 0.92, label,
                 color=color, fontsize=6, rotation=90, va='top')

    # Dual y-axis: cumulative percentage
    ax2 = ax1.twinx()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cumulative_pct = np.cumsum(counts) / counts.sum() * 100.0
    ax2.plot(bin_centers, cumulative_pct, color='black', linewidth=1.0,
             linestyle='-', marker='', label='Cumulative %')
    ax2.set_ylabel('Cumulative %', fontsize=8, color='black')
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='y', labelcolor='black')

    # Legend patches for CI tiers
    low_patch = mpatches.Patch(color='#2CA02C', label='Low CI (<0.4)')
    med_patch = mpatches.Patch(color='#FF7F0E', label='Med CI (0.4–0.7)')
    hi_patch  = mpatches.Patch(color='#D62728', label='High CI (>0.7)')
    cum_line  = plt.Line2D([0], [0], color='black', linewidth=1.0, label='Cumul. %')
    ax1.legend(handles=[low_patch, med_patch, hi_patch, cum_line],
               loc='upper left', fontsize=6, framealpha=0.7, edgecolor='gray')

    ax1.set_title('CI Score Distribution (Mendeley IoMT)', fontsize=8, pad=4)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax1.spines['top'].set_visible(False)

    plt.tight_layout()
    out_pdf = figures_dir / 'fig_ci_distribution.pdf'
    out_png = figures_dir / 'fig_ci_distribution.png'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[XAI] Saved {out_pdf}')


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def run_xai_analysis(data_dir: str, results_dir: str, figures_dir: str) -> Dict:
    """
    Train a SHAP-explainable CI predictor on the Mendeley IoMT dataset.

    Parameters
    ----------
    data_dir    : str  — root data directory (contains Mendeley-IoMT/)
    results_dir : str  — directory for JSON output
    figures_dir : str  — directory for PDF/PNG figure output

    Returns
    -------
    dict with keys:
        r2_score, feature_importances, shap_available, n_samples
    """
    data_path   = Path(data_dir)
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)

    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Mendeley XLSX directly
    # ------------------------------------------------------------------
    xlsx_candidates = [
        data_path / 'Mendeley-IoMT' / 'patients_data_with_alerts.xlsx',
        data_path / 'patients_data_with_alerts.xlsx',
    ]
    xlsx_path = None
    for p in xlsx_candidates:
        if p.exists():
            xlsx_path = p
            break

    if xlsx_path is None:
        raise FileNotFoundError(
            f'[XAI] Mendeley XLSX not found. Searched:\n'
            + '\n'.join(f'  {p}' for p in xlsx_candidates)
        )

    print(f'[XAI] Reading: {xlsx_path}')
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    print(f'[XAI] Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns')
    print(f'[XAI] Columns: {list(df.columns)}')

    # ------------------------------------------------------------------
    # 2. Resolve feature columns
    # ------------------------------------------------------------------
    col_map = _resolve_feature_columns(df.columns)
    print(f'[XAI] Feature column mapping: {col_map}')

    available_features = [(cn, ac) for cn, ac in col_map.items() if ac is not None]
    if not available_features:
        raise RuntimeError('[XAI] No vital-sign feature columns found in dataset.')

    clean_names  = [cn for cn, _ in available_features]
    actual_cols  = [ac for _, ac in available_features]

    # ------------------------------------------------------------------
    # 3. Build feature matrix and CI target
    # ------------------------------------------------------------------
    X_raw = df[actual_cols].copy()

    # Coerce to numeric, fill missing with column median
    for col in actual_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
    X_raw.fillna(X_raw.median(numeric_only=True), inplace=True)

    ci_full = _build_ci_target(df)
    print(f'[XAI] CI scores — min: {ci_full.min():.3f}, '
          f'max: {ci_full.max():.3f}, mean: {ci_full.mean():.3f}')

    # Drop rows still containing NaN
    valid_mask = X_raw.notna().all(axis=1)
    X_arr = X_raw[valid_mask].values.astype(float)
    y_arr = ci_full[valid_mask]
    n_samples = X_arr.shape[0]
    print(f'[XAI] Usable samples: {n_samples}')

    # ------------------------------------------------------------------
    # 4. Train/test split and RF training
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.20, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'[XAI] RandomForest R² on test set: {r2:.4f}')

    # ------------------------------------------------------------------
    # 5. SHAP or permutation importance
    # ------------------------------------------------------------------
    shap_mod = _try_import_shap()
    shap_values_arr = None
    mean_abs_shap = None
    shap_available = False

    X_test_df = pd.DataFrame(X_test, columns=clean_names)

    if shap_mod is not None:
        print('[XAI] SHAP available — computing TreeExplainer values...')
        try:
            shap_values_arr, explainer = _compute_shap_values(
                rf, X_train, X_test, shap_mod
            )
            mean_abs_shap = np.abs(shap_values_arr).mean(axis=0)
            shap_available = True
            print(f'[XAI] SHAP mean |values|: '
                  + ', '.join(f'{n}={v:.4f}'
                              for n, v in zip(clean_names, mean_abs_shap)))
        except Exception as exc:
            warnings.warn(f'[XAI] SHAP computation failed: {exc}. '
                          'Falling back to permutation importance.')
            shap_mod = None
    else:
        print('[XAI] SHAP not installed — using permutation importance.')

    if not shap_available:
        perm_imp = _compute_permutation_importance(
            rf, X_test, y_test, clean_names
        )
        mean_abs_shap = perm_imp
        print(f'[XAI] Permutation importances: '
              + ', '.join(f'{n}={v:.4f}'
                          for n, v in zip(clean_names, mean_abs_shap)))

    # ------------------------------------------------------------------
    # 6. Save feature importances JSON
    # ------------------------------------------------------------------
    fi_dict = {
        'method': 'shap' if shap_available else 'permutation_importance',
        'r2_score': float(r2),
        'n_samples': n_samples,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_importances': {
            name: float(val)
            for name, val in zip(clean_names, mean_abs_shap)
        },
        'rf_feature_importances': {
            name: float(val)
            for name, val in zip(clean_names, rf.feature_importances_)
        },
        'dataset': {
            'citation': 'Barman, IoMT Dataset for ML-Based Health Monitoring, '
                        'Kaggle 2024, doi:10.34740/KAGGLE/DSV/7736523',
            'file': xlsx_path.name,
            'sha256': _sha256(xlsx_path),
            'n_rows': int(df.shape[0]),
            'feature_columns': dict(zip(clean_names, actual_cols)),
        },
        'label_construction': {
            'formula': 'CI = clip(max_j alert_score(alert column j) + '
                       'disease_boost(disease column), 0, 1)',
            'alert_columns': list(ALERT_COL_KEYWORDS.values()),
            'alert_score_map': ALERT_SCORE_MAP,
            'alert_score_default': 0.10,
            'disease_columns_searched': ['Predicted Disease', 'Disease', 'Diagnosis'],
            'disease_boost_map': DISEASE_BOOST_MAP,
            'note': 'the target is derived from the dataset threshold alerts, '
                    'not from clinical outcomes',
            'ci_target_summary': {
                'min': float(ci_full.min()), 'max': float(ci_full.max()),
                'mean': float(ci_full.mean()),
                'frac_low_lt_0.3': float((ci_full < 0.3).mean()),
                'frac_high_ge_0.7': float((ci_full >= 0.7).mean()),
            },
        },
        'model': {'estimator': 'RandomForestRegressor', 'n_estimators': 100,
                  'max_depth': 8, 'random_state': 42,
                  'split': 'train_test_split(test_size=0.20, random_state=42)'},
        'shap_explainer': 'TreeExplainer on the test split' if shap_available else None,
        'versions': _versions(),
    }
    fi_path = results_path / 'shap_feature_importance.json'
    with open(fi_path, 'w', encoding='utf-8') as fh:
        json.dump(fi_dict, fh, indent=2)
    print(f'[XAI] Saved {fi_path}')

    # ------------------------------------------------------------------
    # 7. Generate figures
    # ------------------------------------------------------------------
    _fig_shap_beeswarm(
        shap_values=shap_values_arr,
        X_test_df=X_test_df,
        figures_dir=figures_path,
        shap_mod=shap_mod if shap_available else None,
        mean_abs_shap=mean_abs_shap,
        feature_names=clean_names,
    )

    _fig_shap_bar(mean_abs_shap, clean_names, figures_path)

    _fig_ci_distribution(ci_full, figures_path)

    print('[XAI] XAI analysis complete.')
    return {
        'r2_score': r2,
        'feature_importances': fi_dict['feature_importances'],
        'shap_available': shap_available,
        'n_samples': n_samples,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent  # code/ -> src/ -> analysis/

    parser = argparse.ArgumentParser(description='XAI CI Module')
    parser.add_argument('--data-dir',    default=str(project_root / 'data'))
    parser.add_argument('--results-dir', default=str(project_root / 'results'))
    parser.add_argument('--figures-dir', default=str(project_root / 'latex' / 'figures'))
    args = parser.parse_args()

    out = run_xai_analysis(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
    print('\n[XAI] Summary:')
    for k, v in out.items():
        if isinstance(v, dict):
            print(f'  {k}:')
            for kk, vv in v.items():
                print(f'    {kk}: {vv:.4f}')
        else:
            print(f'  {k}: {v}')
