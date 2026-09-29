"""
run_q1_pipeline.py — orchestrator for every experiment step.

Each step is declared up front (driver, arms, scales, replicates, condition,
output directory) and the whole list is written to
<results>/manifests/declared_arms.json before anything runs.  Each step then
runs inside a manifest (src/analysis/manifest.py: commit, config hash, CLI,
seeds, host, start/end time).  The run ends with `verify`, which checks
every declared step against its manifest and summary.

Steps (skip flags in brackets):
  mc_main              main registry, all scales                     [--skip-mc]
  mc_experiments       E1-E3 arms at the primary scale               [--skip-experiments]
  mc_experiments_long  DQN-ES, ES-only, q-mixed curve at N=5000      [--skip-experiments]
  stats                paired tests: main + decomposition families   [--skip-stats]
  payload_10kb         D15 sensitivity, main registry                [--skip-sensitivity]
  result_size_*        E9 sweep 1/4/16/64 KB                         [--skip-sensitivity]
  scaling_M*           E4 fog-node counts 8/16/32                    [--skip-scaling]
  workload_*           E7 load targets and MMPP-2                    [--skip-workload]
  channel_*            E7b Rayleigh fading and ARQ packet loss       [--skip-channel]
  warm_start_*         E12 N_pre 500/2000                            [--skip-warm]
  ci_noise_*           E14 Gaussian sigma and misclassification      [--skip-cinoise]
  weight_mixed         weight-scheme ablation, mixed CI              [--skip-weight]
  weight_highci        weight-scheme ablation, all-high CI           [--skip-highci]
  mitbih               MIT-BIH trace, main + experiment arms         [--skip-mitbih]
  privacy_guard        Privacy Guard on MedSec-25                    [--skip-privacy]
  overhead             E4/E13 decision-time split (serial)           [--skip-overhead]
  decomposition        latency decomposition                         [--skip-decomp]
  routing              DQN-only routing distribution                 [--skip-routing]
  model_checks         closed-form component checks                  [--skip-checks]
  figures              figures                                       [--skip-figures]
  framing_note, verify (always)

--only NAME [NAME ...] runs just the steps whose name starts with one of
the given prefixes.
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


def _header(msg):
    print('\n' + '=' * 72)
    print(f'  {msg}')
    print('=' * 72, flush=True)


def build_steps(args, results_dir: Path, figures_dir: Path, data_dir: Path) -> list:
    """Declare every step: a list of dicts with a `run` callable."""
    from src.config import (
        CI_MISCLASS_PROBS, CI_NOISE_SIGMAS, ECG_PAYLOAD_SENSITIVITY_BITS,
        LOAD_RHO_TARGETS, PRIMARY_SCALE, Q_MIX_SWEEP, RESULT_SIZE_SENSITIVITY_BITS,
        ROBUSTNESS_ARMS, SCALING_FOG_COUNTS, SCALING_TIMING_RUNS, TASK_SCALES,
        WARM_START_SWEEP, get_experiment_registry, get_full_algorithm_registry,
    )
    from src.simulation.replicate import overrides_tag

    main_arms = list(get_full_algorithm_registry())
    exp_arms = list(get_experiment_registry())
    primary = args.scales[0] if args.scales else PRIMARY_SCALE
    long_n = args.scales[-1] if args.scales else 5000
    all_scales = args.scales or TASK_SCALES
    n, w = args.n_runs, args.workers
    long_arms = ['DQN-ES', 'ES-only'] + [f'q-mixed[q={q:g}]' for q in Q_MIX_SWEEP]
    rs_arms = ['DQN-ES', 'ES-only', 'Static-Tier']

    def mc(name, skip, out, registry, arms, scales, cond=None, n_fog=None,
           desc=''):
        prefix = {'main': 'mc_full', 'experiments': 'mc_exp', 'all': 'mc_all'}[registry]

        def run():
            from src.analysis.run_full_experiments import run_full
            kw = {} if n_fog is None else {'n_fog_nodes': n_fog}
            run_full(scales, n, results_dir / out, workers=w,
                     registry_name=registry,
                     algorithms=None if registry == 'main' else arms,
                     task_overrides=cond, **kw)
        return {'name': name, 'skip': skip, 'driver': 'run_full_experiments',
                'description': desc, 'registry': registry, 'arms': arms,
                'scales': scales, 'n_runs': n, 'condition': cond or {},
                'n_fog_nodes': n_fog or 3, 'out': str(out).replace('\\', '/'),
                'summary': f'{str(out).replace(chr(92), "/")}/{prefix}_summary.json'.lstrip('./'),
                'run': run}

    steps = [
        mc('mc_main', args.skip_mc, Path('.'), 'main', main_arms, all_scales,
           desc='main comparison, all scales'),
        mc('mc_experiments', args.skip_experiments, Path('experiments'),
           'experiments', exp_arms, [primary], desc='E1-E3 decomposition arms'),
        mc('mc_experiments_long', args.skip_experiments,
           Path('experiments_n5000'), 'all', long_arms, [long_n],
           desc='steady state visible: DQN-ES, ES-only, q-mixed curve'),
    ]

    def stats():
        from src.analysis.statistical_tests import run_pairwise_tests
        run_pairwise_tests(results_dir / 'mc_full_summary.json', results_dir,
                           scale=primary, families=['main', 'decomposition'],
                           extra_summaries=[results_dir / 'experiments' / 'mc_exp_summary.json'])
        try:
            from src.analysis.statistical_tests import run_matched_latency
        except ImportError:
            return
        run_matched_latency(results_dir / 'mc_full_summary.json',
                            [results_dir / 'experiments' / 'mc_exp_summary.json'],
                            results_dir, scale=primary)
    steps.append({'name': 'stats', 'skip': args.skip_stats,
                  'driver': 'statistical_tests', 'scales': [primary],
                  'families': ['main', 'decomposition', 'matched_latency'],
                  'run': stats})

    cond = {'ecg_payload_bits': ECG_PAYLOAD_SENSITIVITY_BITS}
    steps.append(mc('payload_10kb', args.skip_sensitivity,
                    Path('sensitivity') / overrides_tag(cond), 'main',
                    main_arms, [primary], cond, desc='D15 10 KB ECG payload'))
    for b in RESULT_SIZE_SENSITIVITY_BITS:
        cond = {'result_size_bits': b}
        steps.append(mc(f'result_size_{b // 8000}kb', args.skip_sensitivity,
                        Path('sensitivity') / overrides_tag(cond), 'all',
                        rs_arms, [primary], cond, desc='E9 result-size sweep'))
    for m in SCALING_FOG_COUNTS:
        if m == 3:
            continue   # M = 3 is the main configuration (mc_main / mc_experiments)
        steps.append(mc(f'scaling_M{m}', args.skip_scaling,
                        Path('scaling') / f'M{m}', 'all', ROBUSTNESS_ARMS,
                        [primary], n_fog=m, desc='E4 fog-node count'))
    workload = ([{'load_rho': r} for r in LOAD_RHO_TARGETS]
                + [{'arrival_process': 'mmpp2'}]
                + [{'arrival_process': 'mmpp2', 'load_rho': r} for r in LOAD_RHO_TARGETS])
    for cond in workload:
        steps.append(mc(f'workload_{overrides_tag(cond)}', args.skip_workload,
                        Path('sensitivity') / overrides_tag(cond), 'all',
                        ROBUSTNESS_ARMS, [primary], cond, desc='E7 workload'))
    for n_pre in WARM_START_SWEEP:
        cond = {'warm_start_tasks': n_pre}
        steps.append(mc(f'warm_start_{n_pre}', args.skip_warm,
                        Path('sensitivity') / overrides_tag(cond), 'all',
                        ROBUSTNESS_ARMS, sorted({100, primary}), cond,
                        desc='E12 warm start'))
    for cond in ([{'ci_noise_sigma': s} for s in CI_NOISE_SIGMAS]
                 + [{'ci_misclass_prob': p} for p in CI_MISCLASS_PROBS]):
        steps.append(mc(f'ci_noise_{overrides_tag(cond)}', args.skip_cinoise,
                        Path('sensitivity') / overrides_tag(cond), 'all',
                        ROBUSTNESS_ARMS, [primary], cond, desc='E14 CI noise'))

    from src.config import PACKET_LOSS_SWEEP
    channel = ([{'fading': 'rayleigh'}]
               + [{'packet_loss': p} for p in PACKET_LOSS_SWEEP]
               + [{'fading': 'rayleigh', 'packet_loss': max(PACKET_LOSS_SWEEP)}])
    for cond in channel:
        steps.append(mc(f'channel_{overrides_tag(cond)}', args.skip_channel,
                        Path('sensitivity') / overrides_tag(cond), 'all',
                        ROBUSTNESS_ARMS, [primary], cond,
                        desc='E7b fading / ARQ (Tier 3)'))

    def weight(ci):
        def run():
            from src.analysis.weight_ablation import run_ablation
            run_ablation(primary, n, results_dir, ci_distribution=ci, workers=w)
        return run
    steps += [
        {'name': 'weight_mixed', 'skip': args.skip_weight, 'driver': 'weight_ablation',
         'arms': ['flat', 'step', 'linear', 'nonlinear'], 'scales': [primary],
         'n_runs': n, 'summary': 'weight_ablation_raw.json', 'run': weight('mixed')},
        {'name': 'weight_highci', 'skip': args.skip_highci, 'driver': 'weight_ablation',
         'arms': ['flat', 'step', 'linear', 'nonlinear'], 'scales': [primary],
         'n_runs': n, 'summary': 'weight_ablation_highci_raw.json',
         'run': weight('all_high')},
    ]

    def mitbih():
        from src.analysis.mitbih_trace_eval import run_mitbih_trace
        run_mitbih_trace(data_dir, results_dir, n_runs=n, workers=w,
                         max_tasks=args.mitbih_max_tasks,
                         algorithms=main_arms + exp_arms)
    steps.append({'name': 'mitbih', 'skip': args.skip_mitbih,
                  'driver': 'mitbih_trace_eval', 'arms': main_arms + exp_arms,
                  'n_runs': n, 'max_tasks': args.mitbih_max_tasks,
                  'summary': 'mitbih_trace_raw.json', 'data': True, 'run': mitbih})

    def privacy_guard():
        from src.analysis.privacy_guard import run_validation
        run_validation(data_dir, results_dir, figures_dir, workers=w)
    steps.append({'name': 'privacy_guard', 'skip': args.skip_privacy,
                  'driver': 'privacy_guard', 'run': privacy_guard})

    def overhead():
        from src.analysis.scheduling_overhead import run_overhead_analysis
        run_overhead_analysis(results_dir, n_runs=min(SCALING_TIMING_RUNS, n),
                              n_tasks=primary, workers=1)
    steps.append({'name': 'overhead', 'skip': args.skip_overhead,
                  'driver': 'scheduling_overhead', 'arms': ROBUSTNESS_ARMS,
                  'fog_counts': SCALING_FOG_COUNTS, 'scales': [primary],
                  'n_runs': min(SCALING_TIMING_RUNS, n), 'workers': 1,
                  'run': overhead})

    def decomp():
        from src.analysis.latency_decomposition import run_decomposition
        run_decomposition(results_dir, n_runs=n, n_tasks=primary, workers=w)

    def routing():
        from src.analysis.dqn_routing_analysis import run_routing_analysis
        run_routing_analysis(results_dir, n_runs=n, n_tasks=primary, workers=w)

    def checks():
        import subprocess
        r = subprocess.run([sys.executable, str(Path(BASE) / 'run_model_checks.py'),
                            '--results-dir', str(results_dir)])
        if r.returncode:
            raise RuntimeError('model checks failed')
    steps += [
        {'name': 'decomposition', 'skip': args.skip_decomp,
         'driver': 'latency_decomposition', 'scales': [primary], 'n_runs': n,
         'run': decomp},
        {'name': 'routing', 'skip': args.skip_routing,
         'driver': 'dqn_routing_analysis', 'scales': [primary], 'n_runs': n,
         'run': routing},
        {'name': 'model_checks', 'skip': args.skip_checks,
         'driver': 'run_model_checks', 'run': checks},
        {'name': 'figures', 'skip': args.skip_figures, 'driver': 'figures_q1',
         'figures_dir': str(figures_dir),
         'run': lambda: _figures(results_dir, figures_dir, primary)},
    ]
    if args.only:
        for st in steps:
            st['skip'] = st['skip'] or not any(st['name'].startswith(o)
                                               for o in args.only)
    return steps


def _figures(results_dir: Path, figures_dir: Path, ref_scale: int) -> None:
    from src.analysis.figures_q1 import (
        _load_summary, fig_energy_sla_vs_scale, fig_epsilon_convergence,
        fig_latency_vs_scale, fig_metric_bars, fig_mitbih_trace,
        fig_pareto_energy_latency, fig_pareto_latency_privacy,
        fig_privacy_guard_roc, fig_shap_summary, fig_weight_ablation,
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    mc_path = results_dir / 'mc_full_summary.json'
    if mc_path.exists():
        summary = _load_summary(mc_path)
        scales = sorted(summary.keys())
        ref = ref_scale if ref_scale in summary else scales[0]
        fig_latency_vs_scale(summary, figures_dir, scales=scales)
        fig_energy_sla_vs_scale(summary, figures_dir, scales=scales)
        fig_metric_bars(summary, figures_dir, ref_scale=ref)
        fig_pareto_energy_latency(summary, figures_dir, ref_scale=ref)
        fig_pareto_latency_privacy(summary, figures_dir, ref_scale=ref)
    else:
        print(f'[SKIP] {mc_path} not found; MC figures skipped')
    fig_epsilon_convergence(results_dir / 'epsilon_trajectory.json', figures_dir)
    fig_weight_ablation(results_dir / 'weight_ablation_raw.json', figures_dir,
                        highci_path=results_dir / 'weight_ablation_highci_raw.json')
    fig_privacy_guard_roc(results_dir / 'privacy_guard_metrics.json', figures_dir)
    fig_shap_summary(results_dir / 'shap_feature_importance.json', figures_dir)
    fig_mitbih_trace(results_dir / 'mitbih_trace_raw.json', figures_dir)


def main():
    p = argparse.ArgumentParser(description='DQN-ES experiment pipeline.')
    p.add_argument('--n_runs', type=int, default=30)
    p.add_argument('--scales', type=int, nargs='+', default=None,
                   help='Main-MC scales; the first is also the scale of every '
                        'other step, the last the long-episode scale')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--results-dir', type=str, default=str(PROJECT_ROOT / 'results'))
    p.add_argument('--figures-dir', type=str,
                   default=str(PROJECT_ROOT / 'latex' / 'figures'))
    p.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data'))
    p.add_argument('--mitbih-max-tasks', type=int, default=None,
                   help='Truncate the MIT-BIH trace (checks only)')
    p.add_argument('--only', nargs='+', default=None,
                   help='Run only steps whose name starts with these prefixes')
    p.add_argument('--declare-only', action='store_true',
                   help='Write declared_arms.json and exit')
    for flag in ('mc', 'experiments', 'stats', 'sensitivity', 'scaling',
                 'workload', 'channel', 'warm', 'cinoise', 'weight', 'highci', 'mitbih',
                 'privacy', 'overhead', 'decomp', 'routing', 'checks', 'figures'):
        p.add_argument(f'--skip-{flag}', action='store_true')
    args = p.parse_args()

    from src.analysis.manifest import Manifest, declare, verify
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    data_dir = Path(args.data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    steps = build_steps(args, results_dir, figures_dir, data_dir)
    decl = declare(results_dir, [{k: v for k, v in s.items() if k != 'run'}
                                 for s in steps if not s['skip']])
    print(f'[PIPELINE] declared {sum(not s["skip"] for s in steps)} steps -> {decl}')
    if args.declare_only:
        return

    t0 = time.time()
    for st in steps:
        if st['skip']:
            continue
        _header(f"{st['name']}  ({st.get('description') or st['driver']})")
        params = {k: v for k, v in st.items() if k not in ('run', 'skip')}
        with Manifest(results_dir, st['name'], params, n_runs=st.get('n_runs'),
                      scales=st.get('scales') or [],
                      data_dir=data_dir if st.get('data') else None):
            st['run']()

    _write_framing_note(results_dir)
    problems = verify(results_dir)
    _header(f'Pipeline complete in {(time.time() - t0) / 60:.1f} min')
    print('verify:', 'ok' if not problems else '')
    for pr in problems:
        print('  ' + pr)


def _write_framing_note(results_dir: Path) -> None:
    """
    Write framing_note.txt: PSO+DQN vs DQN-ES per metric (paired), and the
    weight-scheme contrast on the all-high-CI workload.
    Reads from mc_full_summary.json if available; otherwise writes a stub.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    note_path = results_dir / 'framing_note.txt'

    mc_path = results_dir / 'mc_full_summary.json'
    if not mc_path.exists():
        with open(note_path, 'w', encoding='utf-8') as fh:
            fh.write(
                'FRAMING NOTE (PSO+DQN vs DQN-ES assessment)\n'
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

    from src.config import PRIMARY_SCALE
    scale_key = (str(PRIMARY_SCALE) if str(PRIMARY_SCALE) in summary
                 else sorted(summary, key=int)[0])
    cell = summary.get(scale_key, {})
    dqnes  = cell.get('DQN-ES',  {})
    psodqn  = cell.get('PSO+DQN', {})

    metrics = ['avg_latency_ms', 'avg_energy_mj',
               'avg_privacy_risk', 'sla_violation_pct']

    lines = [
        'FRAMING NOTE (PSO+DQN vs DQN-ES assessment)',
        '=====================================================',
        '',
        'PSO+DQN replaces the enumeration over the K candidates with PSO; at',
        'K = 3 it returns the same argmin by construction.',
        '',
        f'Reference scale: N={scale_key} tasks.',
        '',
        'Results:',
    ]

    # Paired by replicate (same seeds); exact signed-rank, Holm over the 4
    # metrics.  At K=3 PSO+DQN returns the enumeration argmin by
    # construction, so with per-replicate seeds the runs should be identical.
    pso_matches = []
    comps = {}
    for metric in metrics:
        if not dqnes.get(metric, {}).get('samples') or \
                not psodqn.get(metric, {}).get('samples'):
            continue
        comps[metric] = paired_comparison(samples_by_run(dqnes[metric]),
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

    # All-high-CI weight schemes: paired numbers only (no recommendation;
    # interpretation belongs to the manuscript gate, not this script).
    weight_path = results_dir / 'weight_ablation_highci_raw.json'
    lines += ['WEIGHT SCHEMES, all-high-CI workload (nonlinear vs flat, paired):']
    if weight_path.exists():
        with open(weight_path, 'r', encoding='utf-8') as fh:
            wdata = json.load(fh)
        nl, flat = wdata.get('nonlinear', {}), wdata.get('flat', {})
        for metric in metrics:
            if metric in nl and metric in flat:
                c = paired_comparison(samples_by_run(nl[metric]),
                                      samples_by_run(flat[metric]), n_boot=2000)
                lines.append(f'  {metric}: nonlinear={c["mean_ref"]:.4f} '
                             f'flat={c["mean_cmp"]:.4f}  HL d={c["hl_diff"]:+.4f} '
                             f'[{c["ci_lo"]:+.4f}, {c["ci_hi"]:+.4f}]  '
                             f'p_exact={c["p_exact"]:.3e}  n={c["n_pairs"]}')
    else:
        lines.append('  weight_ablation_highci_raw.json not found.')
    lines += ['', 'Numbers only; see statistical_tests outputs for the declared families.']

    with open(note_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'[FRAMING] Saved {note_path}')


if __name__ == '__main__':
    main()
