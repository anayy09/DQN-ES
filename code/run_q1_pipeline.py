"""
run_q1_pipeline.py — Q1 elevation orchestrator (Fix.md + Fix2.md).

Fix.md steps (already implemented):
  Step 1  Full Monte Carlo 30 runs, 5 scales (Fixes 1, 2, 3, 5, 7)
  Step 2  Bonferroni-corrected pairwise tests (Fix 4)
  Step 3  CI weight ablation — mixed-CI workload (Fix 6)
  Step 4  Privacy Guard validation on MedSec-25 (Fix 8)
  Step 5  MIT-BIH real-trace evaluation (Fix 10)
  Step 6  Figures

Fix2.md steps (new):
  Step 7  PSO+DQN is included in the Monte Carlo registry (Fix A) — automatic
  Step 8  All-high-CI weight ablation ICU scenario (Fix B)
  Step 9  Scheduling overhead / wall-clock timing (Fix C)
  Step 10 Privacy Guard re-run with sklearn AUC (Fix D) — same as Step 4
  Step 11 Latency decomposition tx/queue/compute (Fix F)
  Step 12 DQN-only routing distribution (Fix G)
  Step 13 Framing note (Fix A result assessment)

Steps can be skipped individually via --skip-* flags.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))   # code/
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'
FIGURES_DIR  = PROJECT_ROOT / 'latex' / 'figures'
DATA_DIR     = PROJECT_ROOT / 'data'


def _header(msg):
    print('\n' + '=' * 72)
    print(f'  {msg}')
    print('=' * 72)


def main():
    p = argparse.ArgumentParser(
        description='DQN-ES Q1 elevation pipeline (Fix.md + Fix2.md).'
    )
    p.add_argument('--n_runs',       type=int, default=30)
    p.add_argument('--scales',       type=int, nargs='+', default=None)
    p.add_argument('--skip-mc',      action='store_true',
                   help='Skip Monte Carlo (Step 1)')
    p.add_argument('--skip-stats',   action='store_true',
                   help='Skip pairwise statistical tests (Step 2)')
    p.add_argument('--skip-weight',  action='store_true',
                   help='Skip mixed-CI weight ablation (Step 3)')
    p.add_argument('--skip-privacy', action='store_true',
                   help='Skip Privacy Guard / AUC validation (Step 4)')
    p.add_argument('--skip-mitbih',  action='store_true',
                   help='Skip MIT-BIH real-trace evaluation (Step 5)')
    p.add_argument('--skip-figures', action='store_true',
                   help='Skip figure generation (Step 6)')
    p.add_argument('--skip-highci',  action='store_true',
                   help='Skip all-high-CI weight ablation (Step 8)')
    p.add_argument('--skip-overhead', action='store_true',
                   help='Skip scheduling overhead analysis (Step 9)')
    p.add_argument('--skip-decomp',  action='store_true',
                   help='Skip latency decomposition (Step 11)')
    p.add_argument('--skip-routing', action='store_true',
                   help='Skip DQN-only routing analysis (Step 12)')
    p.add_argument('--workers',      type=int, default=None,
                   help='Worker processes for parallel steps')
    args = p.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Monte Carlo (Fixes 1-3, 5, 7; Fix A: PSO+DQN auto-included)
    # ------------------------------------------------------------------
    if not args.skip_mc:
        _header('Step 1 — Monte Carlo (all algorithms incl. PSO+DQN, 30 runs)')
        from src.analysis.run_full_experiments import run_full
        from src.config import TASK_SCALES
        run_full(
            task_scales=args.scales or TASK_SCALES,
            n_runs=args.n_runs,
            results_dir=RESULTS_DIR,
            workers=args.workers,
        )

    # ------------------------------------------------------------------
    # Step 2: Paired signed-rank + Holm over the declared families (E10)
    # ------------------------------------------------------------------
    if not args.skip_stats:
        _header('Step 2 — paired signed-rank (exact) + Holm, HL diff + CI, rank-biserial r')
        from src.analysis.statistical_tests import run_pairwise_tests
        exp_summary = RESULTS_DIR / 'mc_exp_summary.json'
        families = ['main'] + (['decomposition'] if exp_summary.exists() else [])
        run_pairwise_tests(RESULTS_DIR / 'mc_full_summary.json', RESULTS_DIR,
                           workers=args.workers, families=families,
                           extra_summaries=[exp_summary])

    # ------------------------------------------------------------------
    # Step 3: Mixed-CI weight ablation (Fix 6)
    # ------------------------------------------------------------------
    if not args.skip_weight:
        _header('Step 3 — CI weight ablation (mixed-CI workload, Fix 6)')
        from src.analysis.weight_ablation import run_ablation
        from src.config import PRIMARY_SCALE
        run_ablation(PRIMARY_SCALE, args.n_runs, RESULTS_DIR, ci_distribution='mixed', workers=args.workers)

    # ------------------------------------------------------------------
    # Step 4: Privacy Guard AUC validation (Fix 8; Fix D: sklearn AUC)
    # ------------------------------------------------------------------
    if not args.skip_privacy:
        _header('Step 4 — Privacy Guard on MedSec-25 (Fix 8, Fix D: sklearn AUC)')
        from src.analysis.privacy_guard import run_validation
        run_validation(DATA_DIR, RESULTS_DIR, FIGURES_DIR, workers=args.workers)

    # ------------------------------------------------------------------
    # Step 5: MIT-BIH real-trace evaluation (Fix 10)
    # ------------------------------------------------------------------
    if not args.skip_mitbih:
        _header('Step 5 — MIT-BIH real-trace evaluation (Fix 10)')
        from src.analysis.mitbih_trace_eval import run_mitbih_trace
        run_mitbih_trace(DATA_DIR, RESULTS_DIR,
                         workers=args.workers)

    # ------------------------------------------------------------------
    # Step 6: Publication figures (Fix 5, 7, 9; Fix A, B, D)
    # ------------------------------------------------------------------
    if not args.skip_figures:
        _header('Step 6 — Publication figures')
        from src.analysis.figures_q1 import (
            _load_summary,
            fig_energy_sla_vs_scale,
            fig_epsilon_convergence,
            fig_latency_vs_scale,
            fig_metric_bars,
            fig_mitbih_trace,
            fig_pareto_energy_latency,
            fig_pareto_latency_privacy,
            fig_privacy_guard_roc,
            fig_shap_summary,
            fig_weight_ablation,
        )
        from src.config import PRIMARY_SCALE

        mc_path = RESULTS_DIR / 'mc_full_summary.json'
        if mc_path.exists():
            summary = _load_summary(mc_path)
            scales = sorted(summary.keys())
            fig_latency_vs_scale(summary, FIGURES_DIR, scales=scales)
            fig_energy_sla_vs_scale(summary, FIGURES_DIR, scales=scales)
            fig_metric_bars(summary, FIGURES_DIR, ref_scale=PRIMARY_SCALE)
            fig_pareto_energy_latency(summary, FIGURES_DIR, ref_scale=PRIMARY_SCALE)
            fig_pareto_latency_privacy(summary, FIGURES_DIR, ref_scale=PRIMARY_SCALE)
        else:
            print(f'[SKIP] {mc_path} not found — skipping MC-dependent figures')

        fig_epsilon_convergence(RESULTS_DIR / 'epsilon_trajectory.json', FIGURES_DIR)
        # Fix B: two-panel weight ablation
        fig_weight_ablation(
            RESULTS_DIR / 'weight_ablation_raw.json',
            FIGURES_DIR,
            highci_path=RESULTS_DIR / 'weight_ablation_highci_raw.json',
        )
        fig_privacy_guard_roc(RESULTS_DIR / 'privacy_guard_metrics.json', FIGURES_DIR)
        fig_shap_summary(RESULTS_DIR / 'shap_feature_importance.json', FIGURES_DIR)
        fig_mitbih_trace(RESULTS_DIR / 'mitbih_trace_raw.json', FIGURES_DIR)

    # ------------------------------------------------------------------
    # Step 8: All-high-CI weight ablation (Fix B)
    # ------------------------------------------------------------------
    if not args.skip_highci:
        _header('Step 8 — All-high-CI weight ablation ICU scenario (Fix B)')
        from src.analysis.weight_ablation import run_ablation
        from src.config import PRIMARY_SCALE
        run_ablation(PRIMARY_SCALE, args.n_runs, RESULTS_DIR, ci_distribution='all_high', workers=args.workers)

    # ------------------------------------------------------------------
    # Step 9: Scheduling overhead (Fix C)
    # ------------------------------------------------------------------
    if not args.skip_overhead:
        _header('Step 9 — Scheduling overhead analysis (Fix C)')
        from src.analysis.scheduling_overhead import run_overhead_analysis
        run_overhead_analysis(RESULTS_DIR, n_runs=args.n_runs,
                              workers=args.workers)

    # ------------------------------------------------------------------
    # Step 11: Latency decomposition (Fix F)
    # ------------------------------------------------------------------
    if not args.skip_decomp:
        _header('Step 11 — Latency decomposition tx/queue/compute (Fix F)')
        from src.analysis.latency_decomposition import run_decomposition
        run_decomposition(RESULTS_DIR, n_runs=args.n_runs,
                          workers=args.workers)

    # ------------------------------------------------------------------
    # Step 12: DQN-only routing distribution (Fix G)
    # ------------------------------------------------------------------
    if not args.skip_routing:
        _header('Step 12 — DQN-only routing distribution (Fix G)')
        from src.analysis.dqn_routing_analysis import run_routing_analysis
        run_routing_analysis(RESULTS_DIR, n_runs=args.n_runs,
                             workers=args.workers)

    # ------------------------------------------------------------------
    # Step 13: Framing note (Fix A — assess PSO+DQN vs DQN-ES on privacy)
    # ------------------------------------------------------------------
    _write_framing_note(RESULTS_DIR)

    dt = time.time() - t0
    _header(f'Q1 pipeline complete in {dt/60:.1f} min')
    print('  Deliverables:')
    print(f'   results/mc_full_summary.json           — MC with PSO+DQN')
    print(f'   results/table3_n1000_with_pvals.csv    — main table (paired, Holm over 20)')
    print(f'   results/table5_weight_ablation.csv     — Table IV (mixed-CI)')
    print(f'   results/table6_highci_weights.csv      — Table VI (all-high-CI)')
    print(f'   results/table5_mitbih_trace.csv        — Table V (real-trace)')
    print(f'   results/privacy_guard_metrics.json     — AUC, TPR, FPR, F1')
    print(f'   results/scheduling_overhead*.csv       — Fix C timing')
    print(f'   results/latency_decomposition.csv      — Fix F breakdown')
    print(f'   results/dqn_only_routing_dist.csv      — Fix G routing')
    print(f'   results/framing_note.txt               — Fix A assessment')
    print(f'   latex/figures/fig*.pdf                 — All publication figures')


def _write_framing_note(results_dir: Path) -> None:
    """
    Fix A: generate framing_note.txt assessing PSO+DQN vs DQN-ES result.
    Reads from mc_full_summary.json if available; otherwise writes a stub.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    note_path = results_dir / 'framing_note.txt'

    mc_path = results_dir / 'mc_full_summary.json'
    if not mc_path.exists():
        with open(note_path, 'w', encoding='utf-8') as fh:
            fh.write(
                'FRAMING NOTE (Fix A — PSO+DQN vs DQN-ES assessment)\n'
                '=====================================================\n\n'
                'mc_full_summary.json not yet available.\n'
                'Re-run after Step 1 (Monte Carlo) completes.\n'
            )
        return

    import json
    import numpy as np
    from src.analysis.paired_stats import holm_adjust, paired_comparison, samples_by_run

    with open(mc_path, 'r', encoding='utf-8') as fh:
        summary = json.load(fh)

    scale_key = '1000'
    cell = summary.get(scale_key, {})
    bbodrl  = cell.get('DQN-ES',  {})
    psodqn  = cell.get('PSO+DQN', {})

    metrics = ['avg_latency_ms', 'avg_energy_mj',
               'avg_privacy_risk', 'sla_violation_pct']

    lines = [
        'FRAMING NOTE (Fix A — PSO+DQN vs DQN-ES assessment)',
        '=====================================================',
        '',
        'Purpose: Determine whether the BBO inner-loop specifically contributes',
        'to DQN-ES\'s advantages, or whether any DRL+bio-inspired hybrid',
        '(e.g., PSO+DQN) achieves the same result.',
        '',
        f'Reference scale: N={scale_key} tasks, 30 MC replicates.',
        '',
        'Results:',
    ]

    # Paired by replicate (same seeds); exact signed-rank, Holm over the 4
    # metrics.  At K=3 PSO+DQN returns the enumeration argmin by
    # construction, so with per-replicate seeds the runs should be identical.
    pso_matches = []
    comps = {}
    for metric in metrics:
        if not bbodrl.get(metric, {}).get('samples') or \
                not psodqn.get(metric, {}).get('samples'):
            continue
        comps[metric] = paired_comparison(samples_by_run(bbodrl[metric]),
                                          samples_by_run(psodqn[metric]),
                                          n_boot=2000)
    adj = dict(zip(comps, holm_adjust([c['p_exact'] for c in comps.values()])))
    for metric in metrics:
        if metric not in comps:
            lines.append(f'  {metric}: insufficient data')
            continue
        c = comps[metric]
        if c['identical']:
            lines.append(f'  {metric}:  DQN-ES={c["mean_ref"]:.3f}  '
                         f'PSO+DQN={c["mean_cmp"]:.3f}  identical on all '
                         f'{c["n_pairs"]} paired replicates')
            pso_matches.append(True)
            continue
        sig = 'SIGNIFICANT' if adj[metric] < 0.05 else 'not significant'
        lines.append(
            f'  {metric}:'
            f'  DQN-ES={c["mean_ref"]:.3f}  PSO+DQN={c["mean_cmp"]:.3f}'
            f'  HL d={c["hl_diff"]:+.4f} [{c["ci_lo"]:+.4f}, {c["ci_hi"]:+.4f}]'
            f'  p_holm={adj[metric]:.3e}  [{sig}]'
        )
        pso_matches.append(adj[metric] >= 0.05)

    lines += ['']

    # Overall framing recommendation
    privacy_bbo = float(bbodrl.get('avg_privacy_risk', {}).get('mean', 0))
    privacy_pso = float(psodqn.get('avg_privacy_risk', {}).get('mean', 0))

    lines.append('FRAMING RECOMMENDATION:')
    if privacy_bbo < privacy_pso * 0.95:
        lines.append(
            '  DQN-ES outperforms PSO+DQN on privacy risk by >5%.'
            '  The BBO-specific contribution is defensible.'
            '  Main claim: "BBO provides superior exploration for privacy-aware'
            '  routing vs PSO in the DRL+bio-inspired hybrid framework."'
        )
    elif abs(privacy_bbo - privacy_pso) / max(privacy_pso, 1e-12) < 0.05:
        lines.append(
            '  DQN-ES and PSO+DQN are statistically indistinguishable on privacy (<5% diff).'
            '  REFRAMING REQUIRED: The contribution is the DRL+bio-inspired coupling'
            '  design, not BBO specifically.  Remove any claims that BBO uniquely'
            '  drives the privacy advantage.  Retitle contribution as:'
            '  "Hybrid DRL + bio-inspired inner search for CI-adaptive offloading."'
        )
    else:
        lines.append(
            '  Mixed result — review per-metric findings above and decide framing.'
        )

    weight_path = results_dir / 'weight_ablation_highci_raw.json'
    lines += ['', 'NON-LINEAR WEIGHT CONTRIBUTION:']
    if weight_path.exists():
        with open(weight_path, 'r', encoding='utf-8') as fh:
            wdata = json.load(fh)
        nl = wdata.get('nonlinear', {})
        flat = wdata.get('flat', {})
        if nl and flat:
            priv_nl   = nl.get('avg_privacy_risk', {}).get('mean', 0)
            priv_flat = flat.get('avg_privacy_risk', {}).get('mean', 0)
            if priv_nl < priv_flat * 0.95:
                lines.append(
                    '  Non-linear weights separate from flat on all-high-CI workload.'
                    '  Contribution claim is supported. Keep in paper.'
                )
            else:
                lines.append(
                    '  Non-linear weights do NOT separate from flat on all-high-CI workload.'
                    '  REMOVE non-linear CI weight functions from contributions list.'
                    '  Report full negative result in Section V-F.'
                )
        else:
            lines.append('  Insufficient data in weight_ablation_highci_raw.json.')
    else:
        lines.append('  weight_ablation_highci_raw.json not found — run Step 8.')

    with open(note_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'[FRAMING] Saved {note_path}')


if __name__ == '__main__':
    main()
