"""
post_run.py — post-run analyses on the final raw logs and summaries.

Analysis only: nothing here runs the simulator.  Every command reuses the
functions of the declared analyses unchanged (e5_adversary, e8_energy_profiles,
e11_exploration, statistical_tests) and runs inside a manifest
(analysis/manifest.py), so each result is logged like a pipeline step.  The
drivers load one scheduler's logs at a time, which keeps memory bounded when
all ~30 arms are analysed.

Commands (from code/):
  python -m src.analysis.post_run e5  --raw-dirs DIR... --out DIR [--workers W]
      Acuity adversary for every scheduler found (same samples, features,
      classifiers, split and outputs as e5_adversary.run), plus
        adversary_auc_ci.csv            mean per-test-replicate AUC with a
                                        replicate-bootstrap CI, every
                                        feature set x classifier
        adversary_auc_by_replicate_all.json
        trace_identity.json             schedulers whose destination traces
                                        are identical on every replicate
      then statistical_tests.run_privacy_inference on the primary cell.
  python -m src.analysis.post_run corr --e5-dir DIR --summaries F... --scale N --out DIR
      Exploratory: Spearman rank correlation across schedulers between R_P
      (all-task and steady-state) and adversary AUC / I(dest; tier), with a
      permutation p and a test-replicate bootstrap CI.
  python -m src.analysis.post_run e8  --raw-dirs DIR... --out DIR
  python -m src.analysis.post_run e11 --raw-dirs DIR... --out DIR --algorithms A...
  python -m src.analysis.post_run dominance --summaries F... --scale N --out DIR
      Exploratory: paired DQN-ES vs Static-Tier, ES-only[lP], q-mixed and
      Random-K on every metric (HL difference, bootstrap CI, exact p, Holm
      over this table only).
  python -m src.analysis.post_run paired --pairs 'A|B'... --tag T --out DIR
      Exploratory: per-arm means (bootstrap CI) and paired HL differences for
      each pair in every condition of the d17a list.
  python -m src.analysis.post_run e5cv30 --raw-dirs DIR... --out DIR
      Secondary: adversary primary config, 5-fold CV grouped by replicate over
      all replicates (n = 30 test AUCs per scheduler) + the privacy-inference
      tests on those.
  python -m src.analysis.post_run aucmatched --e5-dir DIR --summaries F... --scale N --out DIR
      Secondary: matched-latency form with adversary AUC (declared split, CV30);
      --scale 0 reads flat {arm: cell} summaries (MIT-BIH).
  python -m src.analysis.post_run lambdamech --raw-dirs DIR... --out DIR
      Secondary: destination distribution by CI tier, lambda_P arms.
  python -m src.analysis.post_run d17a --out DIR
      Exploratory: matched-latency privacy excess in every condition where
      DQN-ES and a frontier exist, with the reason for each exclusion.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

_ROOT = Path(_CODE_DIR).parent


# --------------------------------------------------------------------------
# Log index: which files belong to which scheduler, read one scheduler at a time
# --------------------------------------------------------------------------

def index_logs(raw_dirs: List[Path]) -> Dict[str, Dict[int, Path]]:
    """{algorithm: {run_id: path}}, from the first row of each raw log."""
    from src.analysis.e5_adversary import LOG_RE
    from src.simulation.episode_log import _long_path
    idx: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for d in raw_dirs:
        d = _long_path(Path(d))
        if not d.exists():
            print(f'[POST] missing {d}')
            continue
        for f in sorted(d.iterdir()):
            m = LOG_RE.match(f.name)
            if not m:
                continue
            with gzip.open(f, 'rt', encoding='utf-8') as fh:
                row = next(csv.DictReader(fh), None)
            if row is not None:
                idx[row['algorithm']][int(m.group('run'))] = f
    return idx


# Columns the adversary features, labels and MI need (keeps MIT-BIH-size logs in memory)
E5_COLUMNS = ('algorithm', 'device_id', 'timestamp', 'assigned_node', 'ci_tier',
              'latency_ms', 'payload_bits')


def load_runs(files: Dict[int, Path], columns=None) -> Dict[int, List[dict]]:
    out = {}
    for run_id, f in sorted(files.items()):
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            rd = csv.DictReader(fh)
            if columns:
                rows = [{k: r[k] for k in columns} for r in rd]
            else:
                rows = list(rd)
        if rows:
            out[run_id] = rows
    return out


def _vocab_and_hash(files: Dict[int, Path]):
    """Destinations used and a hash of the destination trace per run."""
    from src.analysis.e5_adversary import _dest
    runs = load_runs(files, ('device_id', 'assigned_node'))
    v = set()
    h = hashlib.sha256()
    for run_id, rows in sorted(runs.items()):
        seq = [_dest(r) for r in rows]
        v.update(seq)
        h.update(f'{run_id}:'.encode() + ','.join(seq).encode() + b';')
    return v, h.hexdigest()


def _manifest(results_dir: Path, step: str, params: dict):
    from src.analysis.manifest import Manifest
    return Manifest(results_dir, step, params)


# --------------------------------------------------------------------------
# Acuity-inference adversary
# --------------------------------------------------------------------------

def _e5_worker(job):
    alg, files, vocab = job
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    from src.analysis.e5_adversary import (FEATURE_SETS, _cfg, build_samples,
                                           evaluate, mi_by_run)
    cfg = _cfg()
    runs = load_runs(files, E5_COLUMNS)
    wins = build_samples(runs, cfg['context'], vocab)
    res = []
    for feat in FEATURE_SETS:
        for clf in ('hgb', 'logreg'):
            r = evaluate(wins, feat, clf, cfg['seed'])
            per = r.pop('per_run_auc', {})
            res.append((feat, clf, r, {int(k): v for k, v in per.items()}))
    meta = {'n_samples': len(wins), 'n_runs': len(runs),
            'pos_rate': float(np.mean([w['y'] for w in wins])) if wins else None}
    return alg, res, mi_by_run(runs), meta


def cmd_e5(raw_dirs: List[Path], out_dir: Path, workers: int,
           algorithms: Optional[List[str]] = None) -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from src.analysis.e5_adversary import FEATURE_SETS, _cfg, mean_bootstrap_ci
    from src.config import STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    from src.simulation.episode_log import _long_path
    cfg = _cfg()
    idx = index_logs(raw_dirs)
    if algorithms:
        idx = {a: f for a, f in idx.items() if a in algorithms}
    algs = sorted(idx)

    # vocabulary over every log (as e5_adversary.dest_vocabulary) + trace hashes
    vocab_set, hashes = set(), {}
    for a in algs:
        v, h = _vocab_and_hash(idx[a])
        vocab_set |= v
        hashes[a] = h
    vocab = ['local'] + sorted(d for d in vocab_set if d != 'local')
    print(f'[E5] schedulers={algs} destinations={vocab} unit=task '
          f'context={cfg["context"]}', flush=True)

    results = {}
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for alg, res, mi, meta in ex.map(_e5_worker,
                                         [(a, idx[a], vocab) for a in algs]):
            results[alg] = (res, mi, meta)
            print(f'  [E5] done {alg}', flush=True)

    out_dir = Path(out_dir)
    _long_path(out_dir).mkdir(parents=True, exist_ok=True)
    rows, primary, meta_all, ci_rows, per_all = [], {}, {}, [], {}
    for alg in algs:
        res, _, meta = results[alg]
        meta_all[alg] = meta
        for feat, clf, r, per in res:
            ok = [v for v in per.values() if not math.isnan(v)]
            rows.append({'scheduler': alg, 'features': feat, 'classifier': clf,
                         **r, 'per_run_auc_median': float(np.median(ok)) if ok else float('nan'),
                         'n_test_runs_with_auc': len(ok)})
            ids = sorted(k for k, v in per.items() if not math.isnan(v))
            per_all.setdefault(alg, {})[f'{feat}|{clf}'] = {
                'samples': [per[k] for k in ids], 'run_ids': ids}
            if feat == cfg['primary_features'] and clf == cfg['primary_classifier']:
                primary[alg] = {'adversary_auc': {'samples': [per[k] for k in ids],
                                                  'run_ids': ids}}
            lo, hi = mean_bootstrap_ci([per[k] for k in ids], STAT_BOOT_N,
                                       STAT_CI_LEVEL, STAT_BOOT_SEED + len(ci_rows))
            ci_rows.append({'scheduler': alg, 'features': feat, 'classifier': clf,
                            'n_test_runs': len(ids),
                            'auc_mean_per_run': float(np.mean([per[k] for k in ids])) if ids else float('nan'),
                            'ci_lo': lo, 'ci_hi': hi,
                            'pooled_test_auc': r['pooled_test_auc'],
                            'cv_auc_mean': r.get('cv_auc_mean', float('nan')),
                            'cv_auc_sd': r.get('cv_auc_sd', float('nan')),
                            'macro3_auc': r.get('macro3_auc', '')})

    # outputs of e5_adversary.run, same names and fields
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in (
        'scheduler', 'features', 'classifier'), k))
    _csv(out_dir / 'adversary_auc_summary.csv', rows, fields)
    _long_path(out_dir / 'adversary_auc_by_replicate.json').write_text(
        json.dumps({'primary': {k: cfg[k] for k in ('unit', 'primary_features',
                                                    'primary_classifier', 'context',
                                                    'train_runs', 'test_runs')},
                    'cells': primary}, indent=1), encoding='utf-8')
    mi_rows = []
    for i, alg in enumerate(algs):
        v = list(results[alg][1].values())
        lo, hi = mean_bootstrap_ci(v, STAT_BOOT_N, STAT_CI_LEVEL, STAT_BOOT_SEED + i)
        mi_rows.append({'scheduler': alg, 'n_runs': len(v),
                        'mi_bits_mean': float(np.mean(v)), 'ci_lo': lo, 'ci_hi': hi})
    _csv(out_dir / 'mutual_information.csv', mi_rows)
    _long_path(out_dir / 'e5_meta.json').write_text(
        json.dumps({'vocab': vocab, 'schedulers': meta_all,
                    'raw_dirs': [str(d) for d in raw_dirs]}, indent=1), encoding='utf-8')

    # additions: CIs for every configuration, per-run values, MI per run, identities
    _csv(out_dir / 'adversary_auc_ci.csv', ci_rows)
    _long_path(out_dir / 'adversary_auc_by_replicate_all.json').write_text(
        json.dumps(per_all, indent=1), encoding='utf-8')
    _long_path(out_dir / 'mutual_information_by_replicate.json').write_text(
        json.dumps({a: {str(k): v for k, v in results[a][1].items()} for a in algs},
                   indent=1), encoding='utf-8')
    groups = defaultdict(list)
    for a in algs:
        groups[hashes[a]].append(a)
    _long_path(out_dir / 'trace_identity.json').write_text(json.dumps(
        {'note': 'schedulers in one group have identical destination traces on '
                 'every replicate (hence identical E5 features and AUCs)',
         'groups': sorted([g for g in groups.values() if len(g) > 1])},
        indent=1), encoding='utf-8')

    for r in ci_rows:
        if r['features'] == cfg['primary_features'] and r['classifier'] == 'hgb':
            print(f"  {r['scheduler']:<18} AUC {r['auc_mean_per_run']:.3f} "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] pooled {r['pooled_test_auc']:.3f}")

    from src.analysis.statistical_tests import run_privacy_inference
    run_privacy_inference(out_dir / 'adversary_auc_by_replicate.json', out_dir)
    print(f'[E5] wrote {out_dir}')


def _csv(path: Path, rows: List[dict], fields: Optional[List[str]] = None) -> None:
    from src.simulation.episode_log import _long_path
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(_long_path(path), 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, restval='')
        wr.writeheader()
        wr.writerows(rows)


# --------------------------------------------------------------------------
# Rank correlation R_P vs adversary AUC (exploratory)
# --------------------------------------------------------------------------

def _spearman(x, y) -> float:
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_corr(e5_dir: Path, summaries: List[Path], scale: int, out_dir: Path,
             n_perm: int = 100_000) -> None:
    from src.analysis.paired_stats import samples_by_run
    from src.analysis.statistical_tests import _load_cells
    from src.config import E5_ADVERSARY, STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    from src.simulation.episode_log import _long_path
    cell = _load_cells(summaries, scale)
    per_all = json.loads(_long_path(e5_dir / 'adversary_auc_by_replicate_all.json')
                         .read_text(encoding='utf-8'))
    mi_all = json.loads(_long_path(e5_dir / 'mutual_information_by_replicate.json')
                        .read_text(encoding='utf-8'))
    ident = json.loads(_long_path(e5_dir / 'trace_identity.json')
                       .read_text(encoding='utf-8'))['groups']
    # one representative per identical-trace group: the name in `cell` that
    # sorts first among main-registry names, else the first name
    from src.config import get_full_algorithm_registry
    main_names = set(get_full_algorithm_registry())
    drop = set()
    for g in ident:
        keep = sorted(g, key=lambda a: (a not in main_names, a))[0]
        drop |= set(g) - {keep}
    test_runs = E5_ADVERSARY['test_runs']
    algs_all = sorted(a for a in per_all if a in cell and a not in drop)
    sets = {'distinct_policies': algs_all,
            'listed_nine': [a for a in ['DQN-ES', 'ES-only', 'Random-K[K=3]',
                                        'q-mixed[q=0.25]', 'q-mixed[q=0.5]',
                                        'Static-Tier', 'ES-only[lP=3]',
                                        'Local-Only', 'Cloud-Only'] if a in algs_all],
            'excl_single_destination': [a for a in algs_all
                                        if a not in ('Local-Only', 'Cloud-Only')]}
    ys = {f'auc[{k}]': v for k, v in
          {'dest': 'dest|hgb', 'dest+timing': 'dest+timing|hgb',
           'dest+timing+size': 'dest+timing+size|hgb'}.items()}
    rng = np.random.default_rng(STAT_BOOT_SEED)
    rows, points = [], []
    for sname, algs in sets.items():
        for xm in ('avg_privacy_risk', 'avg_privacy_risk_ss'):
            X = {a: samples_by_run(cell[a][xm]) for a in algs}
            for yname, key in [*ys.items(), ('mi_dest_tier', None)]:
                if key is None:
                    Y = {a: {int(k): v for k, v in mi_all[a].items()} for a in algs}
                else:
                    Y = {a: dict(zip(per_all[a][key]['run_ids'], per_all[a][key]['samples']))
                         for a in algs}
                runs = [r for r in test_runs
                        if all(r in X[a] and r in Y[a] for a in algs)]
                xv = np.array([np.mean([X[a][r] for r in runs]) for a in algs])
                yv = np.array([np.mean([Y[a][r] for r in runs]) for a in algs])
                x30 = np.array([np.mean(list(X[a].values())) for a in algs])
                rho = _spearman(xv, yv)
                perm = np.array([_spearman(xv, rng.permutation(yv))
                                 for _ in range(n_perm // 10)])
                p = float((np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (perm.size + 1))
                boots = []
                bw = np.random.default_rng(STAT_BOOT_SEED + len(rows))
                for _ in range(STAT_BOOT_N // 5):
                    rr = bw.choice(runs, size=len(runs), replace=True)
                    bx = [np.mean([X[a][r] for r in rr]) for a in algs]
                    by = [np.mean([Y[a][r] for r in rr]) for a in algs]
                    boots.append(_spearman(bx, by))
                boots = np.array([b for b in boots if not math.isnan(b)])
                a_ = (1 - STAT_CI_LEVEL) / 2
                rows.append({'scheduler_set': sname, 'n_schedulers': len(algs),
                             'x_metric': xm, 'y_metric': yname,
                             'n_test_runs': len(runs), 'spearman_rho': rho,
                             'ci_lo': float(np.quantile(boots, a_)) if boots.size else float('nan'),
                             'ci_hi': float(np.quantile(boots, 1 - a_)) if boots.size else float('nan'),
                             'p_perm_two_sided': p,
                             'spearman_rho_x_all30': _spearman(x30, yv),
                             'label': 'exploratory (not declared in D17)'})
                if sname == 'distinct_policies':
                    for a, xa, ya, xa30 in zip(algs, xv, yv, x30):
                        points.append({'scheduler': a, 'x_metric': xm, 'y_metric': yname,
                                       'x_test_runs': xa, 'x_all30': xa30, 'y_test_runs': ya})
                print(f"  {sname:<24} {xm:<20} {yname:<24} rho {rho:+.3f} "
                      f"[{rows[-1]['ci_lo']:+.3f}, {rows[-1]['ci_hi']:+.3f}] p {p:.4f} "
                      f"(n={len(algs)})", flush=True)
    out_dir = Path(out_dir)
    _long_path(out_dir).mkdir(parents=True, exist_ok=True)
    _csv(out_dir / 'rp_vs_auc_correlation.csv', rows)
    _csv(out_dir / 'rp_vs_auc_points.csv', points)
    _long_path(out_dir / 'rp_vs_auc_meta.json').write_text(json.dumps(
        {'dropped_as_identical': sorted(drop), 'sets': sets, 'test_runs': test_runs,
         'n_perm': n_perm // 10, 'n_boot': STAT_BOOT_N // 5,
         'bootstrap': 'resample test replicates (shared across schedulers), '
                      'recompute scheduler means of x and y, Spearman',
         'label': 'exploratory (not declared in D17)'}, indent=1), encoding='utf-8')


# --------------------------------------------------------------------------
# Energy profiles (same computation as e8_energy_profiles.run, one scheduler at a time)
# --------------------------------------------------------------------------

def cmd_e8(raw_dirs: List[Path], out_dir: Path,
           algorithms: Optional[List[str]] = None) -> None:
    from src.analysis.e5_adversary import mean_bootstrap_ci
    from src.analysis.e8_energy_profiles import PROFILES, _model_powers, task_energy_mj
    from src.config import STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    idx = index_logs(raw_dirs)
    if algorithms:
        idx = {a: f for a, f in idx.items() if a in algorithms}
    model = _model_powers()
    per_run = {p: {} for p in PROFILES}
    for alg in sorted(idx):
        runs = load_runs(idx[alg])
        for pname, prof in PROFILES.items():
            per_run[pname][alg] = [np.mean([task_energy_mj(r, prof, model) for r in rs])
                                   for _, rs in sorted(runs.items())]
        print(f'  [E8] {alg}', flush=True)
    rows = []
    for pname in PROFILES:
        per_alg = {}
        for alg, v in sorted(per_run[pname].items()):
            lo, hi = mean_bootstrap_ci(v, STAT_BOOT_N, STAT_CI_LEVEL, STAT_BOOT_SEED)
            per_alg[alg] = (float(np.mean(v)), lo, hi, len(v))
        order = sorted(per_alg, key=lambda a: per_alg[a][0])
        for alg, (m, lo, hi, n) in per_alg.items():
            rows.append({'profile': pname, 'scheduler': alg, 'n_runs': n,
                         'energy_mj_mean': m, 'ci_lo': lo, 'ci_hi': hi,
                         'rank_lowest_first': order.index(alg) + 1})
    from src.simulation.episode_log import _long_path
    _long_path(Path(out_dir)).mkdir(parents=True, exist_ok=True)
    _csv(Path(out_dir) / 'energy_profiles.csv', rows)
    for r in rows:
        print(f"  {r['profile']:<14} {r['scheduler']:<18} {r['energy_mj_mean']:9.3f} mJ "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] rank {r['rank_lowest_first']}")


# --------------------------------------------------------------------------
# Matched-latency privacy excess, robustness (exploratory)
# --------------------------------------------------------------------------

def _conditions() -> List[dict]:
    """Every result cell where DQN-ES and at least one frontier can be formed."""
    R = _ROOT / 'results'
    c = [{'condition': 'main N=1000 (declared)', 'summaries': [R / 'mc_full_summary.json',
          R / 'experiments' / 'mc_exp_summary.json'], 'scale': 1000},
         {'condition': 'N=5000', 'summaries': [R / 'experiments_n5000' / 'mc_all_summary.json'],
          'scale': 5000},
         {'condition': 'MIT-BIH trace', 'summaries': [R / 'mitbih_trace_raw.json'],
          'scale': None},
         {'condition': 'ECG payload 10 KB', 'summaries': [R / 'sensitivity' / 'ecg80000'
                                                          / 'mc_full_summary.json'],
          'scale': 1000}]
    for tag in ('rho0.3', 'rho0.6', 'rho0.85', 'mmpp2', 'mmpp2_rho0.3', 'mmpp2_rho0.6',
                'mmpp2_rho0.85', 'cin0.05', 'cin0.1', 'cin0.2', 'cim0.1', 'cim0.2',
                'rayleigh', 'rayleigh_loss0.01', 'rayleigh_loss0.05'):
        c.append({'condition': f'sensitivity/{tag}',
                  'summaries': [R / 'sensitivity' / tag / 'mc_all_summary.json'],
                  'scale': 1000})
    for tag in ('warm500', 'warm2000'):
        for n in (100, 1000):
            c.append({'condition': f'sensitivity/{tag} N={n}',
                      'summaries': [R / 'sensitivity' / tag / 'mc_all_summary.json'],
                      'scale': n})
    for m in (8, 16, 32):
        c.append({'condition': f'scaling M={m}',
                  'summaries': [R / 'scaling' / f'M{m}' / 'mc_all_summary.json'],
                  'scale': 1000})
    return c


def _load_condition(cond: dict) -> dict:
    from src.analysis.statistical_tests import _load_cells
    if cond['scale'] is None:            # MIT-BIH: {alg: {metric: cell}}
        cell = {}
        for p in cond['summaries']:
            cell.update(json.loads(Path(p).read_text(encoding='utf-8')))
        return cell
    return _load_cells(cond['summaries'], cond['scale'])


def _exclusion_reasons(cell, ref, arms, x_metric, y_metric) -> Dict[int, str]:
    """Why each replicate was excluded (mirrors matched_latency_excess)."""
    from src.analysis.paired_stats import samples_by_run
    ref_x = samples_by_run(cell[ref][x_metric])
    arms = [a for a in arms if a in cell and y_metric in cell[a]]
    fx = {a: samples_by_run(cell[a][x_metric]) for a in arms}
    out = {}
    for r in sorted(ref_x):
        xs = sorted({fx[a][r] for a in arms if r in fx[a]})
        if len(xs) < 2:
            out[r] = 'fewer than two frontier points'
        elif ref_x[r] < xs[0]:
            out[r] = (f'DQN-ES latency {ref_x[r]:.2f} ms below the fastest frontier '
                      f'point {xs[0]:.2f} ms')
        elif ref_x[r] > xs[-1]:
            out[r] = (f'DQN-ES latency {ref_x[r]:.2f} ms above the slowest frontier '
                      f'point {xs[-1]:.2f} ms')
    return out


def _identity_gap(cell, a, b, metrics=('avg_latency_ms', 'avg_privacy_risk',
                                         'avg_privacy_risk_ss')) -> Optional[float]:
    from src.analysis.paired_stats import samples_by_run
    if a not in cell or b not in cell:
        return None
    gap = 0.0
    for m in metrics:
        sa, sb = samples_by_run(cell[a][m]), samples_by_run(cell[b][m])
        gap = max(gap, max(abs(sa[r] - sb[r]) for r in set(sa) & set(sb)))
    return gap


def cmd_d17a(out_dir: Path) -> None:
    import numpy as np  # noqa: F811
    from src.analysis.statistical_tests import matched_latency_excess, median_bootstrap_ci
    from src.config import MATCHED_LATENCY as ML, Q_MIX_SWEEP, STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    q_full = ML['frontiers']['q_mixed']
    lam = ML['frontiers']['lambda_p']
    rows = []
    for cond in _conditions():
        if not all(Path(p).exists() for p in cond['summaries']):
            rows.append({'condition': cond['condition'], 'status': 'summary missing'})
            continue
        cell = _load_condition(cond)
        ref = ML['reference']
        # q-mixed: the declared 6 points when present; otherwise the identities
        # q=0 == ES-only and q=1 == Random-K[K=3] complete what exists
        alias = {}
        if 'q-mixed[q=0]' not in cell and 'ES-only' in cell:
            alias['q-mixed[q=0]'] = 'ES-only'
        if 'q-mixed[q=1]' not in cell and 'Random-K[K=3]' in cell:
            alias['q-mixed[q=1]'] = 'Random-K[K=3]'
        cell2 = dict(cell)
        for k, v in alias.items():
            cell2[k] = cell[v]
        q_present = [a for a in q_full if a in cell2]
        frontiers = [('q_mixed', q_full)]
        if cond['condition'].startswith('main'):
            frontiers.append(('q_mixed_3pt', ['q-mixed[q=0]', 'q-mixed[q=0.5]',
                                              'q-mixed[q=1]']))
        frontiers.append(('lambda_p', lam))
        k = 0
        for fname, arms in frontiers:
            src_cell = cell2 if fname.startswith('q_mixed') else cell
            present = [a for a in arms if a in src_cell]
            for y in ML['y_metrics']:
                row = {'condition': cond['condition'], 'scale': cond['scale'] or 'trace',
                       'frontier': fname, 'y_metric': y,
                       'frontier_points': len(present),
                       'arms_used': ';'.join(present),
                       'aliases': ';'.join(f'{a}={b}' for a, b in alias.items()
                                           if a in present) if fname.startswith('q_mixed') else '',
                       'label': ('declared (D17a), reproduced' if cond['condition'].startswith('main')
                                 and fname != 'q_mixed_3pt' else 'exploratory')}
                seed = STAT_BOOT_SEED + k
                k += 1
                if ref not in cell:
                    row['status'] = 'no DQN-ES'
                elif y not in cell[ref]:
                    row['status'] = f'{y} not in the summary at this scale'
                elif len(present) < 2:
                    row['status'] = ('not run: frontier arms absent'
                                     if not present else
                                     f'not computable: only {present[0]}'
                                     f'{" (= " + alias[present[0]] + ")" if present[0] in alias else ""}'
                                     ' present')
                else:
                    res = matched_latency_excess(src_cell, ref, arms, ML['x_metric'], y)
                    ex = list(res['excess_by_run'].values())
                    lo, hi = median_bootstrap_ci(ex, STAT_BOOT_N, STAT_CI_LEVEL, seed)
                    reasons = _exclusion_reasons(src_cell, ref, arms, ML['x_metric'], y)
                    row.update(status='ok', n_replicates=len(ex),
                               n_excluded=len(res['excluded_runs']),
                               median_excess=float(np.median(ex)) if ex else float('nan'),
                               ci_lo=lo, ci_hi=hi,
                               excluded_runs=';'.join(map(str, res['excluded_runs'])),
                               exclusion_reasons=' | '.join(f'run {r}: {reasons[r]}'
                                                             for r in res['excluded_runs']))
                rows.append(row)
        # record the identities the aliases rely on, where both names exist
        for a, b in (('q-mixed[q=0]', 'ES-only'), ('q-mixed[q=1]', 'Random-K[K=3]')):
            g = _identity_gap(cell, a, b)
            if g is not None:
                rows.append({'condition': cond['condition'], 'frontier': 'identity check',
                             'y_metric': f'{a} vs {b}', 'status': f'max |diff| = {g:.3g}'})
    fields = ['condition', 'scale', 'frontier', 'y_metric', 'label', 'status',
              'frontier_points', 'n_replicates', 'n_excluded', 'median_excess',
              'ci_lo', 'ci_hi', 'arms_used', 'aliases', 'excluded_runs',
              'exclusion_reasons']
    from src.simulation.episode_log import _long_path
    _long_path(Path(out_dir)).mkdir(parents=True, exist_ok=True)
    _csv(Path(out_dir) / 'matched_latency_robustness.csv', rows, fields)
    for r in rows:
        if r.get('status') == 'ok':
            print(f"  {r['condition']:<30} {r['frontier']:<12} {r['y_metric']:<20} "
                  f"{r['median_excess']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                  f"n={r['n_replicates']} excl={r['n_excluded']}")
        else:
            print(f"  {r['condition']:<30} {r.get('frontier', ''):<12} "
                  f"{r.get('y_metric', ''):<20} {r['status']}")


# --------------------------------------------------------------------------
# Paired DQN-ES vs frontier arms (exploratory; not a declared family)
# --------------------------------------------------------------------------

DOMINANCE_ARMS = ['Static-Tier', 'ES-only[lP=1.5]', 'ES-only[lP=2]', 'ES-only[lP=3]',
                  'q-mixed[q=0.25]', 'q-mixed[q=0.5]', 'Random-K[K=3]']
DOMINANCE_METRICS = ['avg_latency_ms', 'avg_energy_mj', 'avg_privacy_risk',
                     'avg_privacy_risk_ss', 'sla_violation_pct']


def cmd_dominance(summaries: List[Path], scale: int, out_dir: Path) -> None:
    from src.analysis.paired_stats import holm_adjust, paired_comparison, samples_by_run
    from src.analysis.statistical_tests import ROW_FIELDS, _load_cells
    from src.config import STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    cell = _load_cells(summaries, scale)
    rows = []
    for comp in DOMINANCE_ARMS:
        for m in DOMINANCE_METRICS:
            if comp not in cell or m not in cell[comp]:
                continue
            res = paired_comparison(samples_by_run(cell['DQN-ES'][m]),
                                    samples_by_run(cell[comp][m]), n_boot=STAT_BOOT_N,
                                    level=STAT_CI_LEVEL, seed=STAT_BOOT_SEED + len(rows))
            rows.append({'family': 'exploratory_dominance', 'metric': m,
                         'reference': 'DQN-ES', 'comparator': comp,
                         'status': 'identical' if res['identical'] else 'tested', **res})
    adj = holm_adjust([r.get('p_exact', 1.0) for r in rows])
    for r, pa in zip(rows, adj):
        r['family_size'] = len(rows)
        r['p_holm'] = float(pa)
        r['reject_holm'] = bool(pa <= 0.05)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _csv(out_dir / 'dominance_vs_dqn_es.csv',
         [{k: r.get(k, '') for k in ROW_FIELDS if k not in ('tost_margin', 'p_tost')}
          | {'label': 'exploratory (not declared); Holm over this table only'}
          for r in rows])
    for r in rows:
        print(f"  {r['comparator']:<16} {r['metric']:<20} d={r['hl_diff']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p_holm={r['p_holm']:.2e}")


# --------------------------------------------------------------------------
# Paired arm-vs-arm differences in every condition (exploratory)
# --------------------------------------------------------------------------

PAIRED_ALIASES = {'Random-K[K=3]': 'q-mixed[q=1]', 'ES-only': 'q-mixed[q=0]'}


def cmd_paired(pairs: List[str], out_dir: Path, tag: str) -> None:
    """
    For each condition of _conditions() and each 'A|B' pair present there:
    per-arm mean with a replicate-bootstrap CI and the paired HL difference
    A - B with its CI and exact p (no multiplicity correction; exploratory).
    Random-K[K=3] / ES-only fall back to their identities q-mixed[q=1] / [q=0].
    """
    from src.analysis.e5_adversary import mean_bootstrap_ci
    from src.analysis.paired_stats import paired_comparison, samples_by_run
    from src.config import STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    rows = []
    for cond in _conditions():
        if not all(Path(p).exists() for p in cond['summaries']):
            continue
        cell = _load_condition(cond)
        for pair in pairs:
            a, b = pair.split('|')
            names = []
            for x in (a, b):
                if x not in cell and PAIRED_ALIASES.get(x) in cell:
                    x = PAIRED_ALIASES[x]
                names.append(x)
            if not all(x in cell for x in names):
                continue
            for m in DOMINANCE_METRICS:
                if m not in cell[names[0]] or m not in cell[names[1]]:
                    continue
                sa, sb = samples_by_run(cell[names[0]][m]), samples_by_run(cell[names[1]][m])
                seed = STAT_BOOT_SEED + len(rows)
                la, ha = mean_bootstrap_ci(list(sa.values()), STAT_BOOT_N, STAT_CI_LEVEL, seed)
                lb, hb = mean_bootstrap_ci(list(sb.values()), STAT_BOOT_N, STAT_CI_LEVEL, seed + 1)
                res = paired_comparison(sa, sb, n_boot=STAT_BOOT_N, level=STAT_CI_LEVEL, seed=seed)
                rows.append({'condition': cond['condition'], 'scale': cond['scale'] or 'trace',
                             'a': names[0], 'b': names[1], 'metric': m,
                             'n_pairs': res['n_pairs'],
                             'mean_a': float(np.mean(list(sa.values()))), 'mean_a_ci_lo': la,
                             'mean_a_ci_hi': ha,
                             'mean_b': float(np.mean(list(sb.values()))), 'mean_b_ci_lo': lb,
                             'mean_b_ci_hi': hb,
                             'hl_diff_a_minus_b': res['hl_diff'], 'ci_lo': res['ci_lo'],
                             'ci_hi': res['ci_hi'], 'p_exact': res.get('p_exact'),
                             'identical': res['identical'],
                             'label': 'exploratory (not declared; no multiplicity correction)'})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _csv(out_dir / f'paired_{tag}.csv', rows)
    print(f'[PAIRED] {len(rows)} rows -> {out_dir / f"paired_{tag}.csv"}')


# --------------------------------------------------------------------------
# Adversary secondary analyses (not pre-declared)
# --------------------------------------------------------------------------

def _e5_cv30_worker(job):
    """Primary features/classifier; 5-fold GroupKFold over every replicate."""
    alg, files, vocab = job
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    from sklearn.model_selection import GroupKFold
    from src.analysis.e5_adversary import _auc, _cfg, _make, build_samples
    cfg = _cfg()
    runs = load_runs(files, E5_COLUMNS)
    wins = build_samples(runs, cfg['context'], vocab)
    feat = cfg['primary_features']
    X = np.array([w['x'][feat] for w in wins], dtype=float)
    y = np.array([w['y'] for w in wins])
    g = np.array([w['run_id'] for w in wins])
    per, fold_of = {}, {}
    pooled_y, pooled_p = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        m = _make(cfg['primary_classifier'], cfg['seed']).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        pooled_y += y[te].tolist()
        pooled_p += p.tolist()
        for r in sorted(set(g[te].tolist())):
            sel = g[te] == r
            per[int(r)] = _auc(y[te][sel], p[sel])
            fold_of[int(r)] = k
    return alg, per, fold_of, _auc(pooled_y, pooled_p)


def cmd_e5_cv30(raw_dirs: List[Path], out_dir: Path, workers: int,
                algorithms: Optional[List[str]] = None) -> None:
    """
    Secondary (not declared): adversary primary configuration with 5-fold
    cross-validation grouped by replicate over all replicates, so every
    replicate contributes one test AUC (n = 30 per scheduler). Then the
    privacy-inference family on those AUCs as a sensitivity check.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from src.analysis.e5_adversary import mean_bootstrap_ci
    from src.analysis.statistical_tests import _write_rows, run_family_tests
    from src.config import STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    from src.simulation.episode_log import _long_path
    idx = index_logs(raw_dirs)
    if algorithms:
        idx = {a: f for a, f in idx.items() if a in algorithms}
    algs = sorted(idx)
    vocab_set = set()
    for a in algs:
        vocab_set |= _vocab_and_hash(idx[a])[0]
    vocab = ['local'] + sorted(d for d in vocab_set if d != 'local')
    res = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
        for alg, per, fold_of, pooled in ex.map(_e5_cv30_worker,
                                                [(a, idx[a], vocab) for a in algs]):
            res[alg] = (per, fold_of, pooled)
            print(f'  [E5-CV30] done {alg}', flush=True)
    out_dir = Path(out_dir)
    _long_path(out_dir).mkdir(parents=True, exist_ok=True)
    cells, rows = {}, []
    for i, a in enumerate(algs):
        per, fold_of, pooled = res[a]
        ids = sorted(k for k, v in per.items() if not math.isnan(v))
        v = [per[k] for k in ids]
        cells[a] = {'adversary_auc': {'samples': v, 'run_ids': ids}}
        lo, hi = mean_bootstrap_ci(v, STAT_BOOT_N, STAT_CI_LEVEL, STAT_BOOT_SEED + i)
        rows.append({'scheduler': a, 'n_test_runs': len(ids),
                     'auc_mean_per_run': float(np.mean(v)), 'ci_lo': lo, 'ci_hi': hi,
                     'pooled_cv_auc': pooled,
                     'label': 'secondary (P2b): 5-fold GroupKFold over all replicates'})
        print(f"  {a:<18} AUC {np.mean(v):.3f} [{lo:.3f}, {hi:.3f}] n={len(ids)}")
    _long_path(out_dir / 'adversary_auc_cv30_by_replicate.json').write_text(json.dumps(
        {'config': 'E5 primary features/classifier; GroupKFold(5) by replicate, '
                   'every replicate tested once (secondary, P2b)',
         'folds': {a: res[a][1] for a in algs}, 'cells': cells}, indent=1), encoding='utf-8')
    _csv(out_dir / 'adversary_auc_cv30_ci.csv', rows)
    fam = run_family_tests(cells, 'privacy_inference')
    for r in fam:
        r['family'] = 'privacy_inference_cv30 (sensitivity, not declared)'
    _write_rows(fam, out_dir / 'stat_tests_privacy_inference_cv30.csv')
    for r in fam:
        if r['status'] == 'tested':
            print(f"  D17b-CV30 {r['comparator']:<14} d={r['hl_diff']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p_holm={r['p_holm']:.4f} n={r['n_pairs']}")


def cmd_auc_matched(e5_dir: Path, summaries: List[Path], scale: int, out_dir: Path) -> None:
    """
    Secondary: the matched-latency form with adversary AUC as y. Per replicate,
    interpolate the q-mixed and lambda_P curves' AUC at DQN-ES's latency;
    median over replicates with a replicate-bootstrap CI. Positive = DQN-ES
    AUC above the frontier. Run on the declared test split (n <= 10) and on
    the CV30 AUCs (n <= 30) when present. scale 0 = flat {arm: cell} files.
    """
    from src.analysis.statistical_tests import (_load_cells, matched_latency_excess,
                                                median_bootstrap_ci)
    from src.config import MATCHED_LATENCY as ML, STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    from src.simulation.episode_log import _long_path
    if scale:
        base = _load_cells(summaries, scale)
    else:
        base = {}
        for p in summaries:
            base.update(json.loads(Path(p).read_text(encoding='utf-8')))
    sources = [('declared_split', 'adversary_auc_by_replicate.json')]
    if _long_path(e5_dir / 'adversary_auc_cv30_by_replicate.json').exists():
        sources.append(('cv30', 'adversary_auc_cv30_by_replicate.json'))
    rows = []
    for sname, fn in sources:
        auc = json.loads(_long_path(e5_dir / fn).read_text(encoding='utf-8'))['cells']
        cell = {a: dict(base[a], adversary_auc=auc[a]['adversary_auc'])
                for a in base if a in auc}
        if 'q-mixed[q=0]' not in cell and 'ES-only' in cell:
            cell['q-mixed[q=0]'] = cell['ES-only']
        if 'q-mixed[q=1]' not in cell and 'Random-K[K=3]' in cell:
            cell['q-mixed[q=1]'] = cell['Random-K[K=3]']
        for fname, arms in ML['frontiers'].items():
            present = [a for a in arms if a in cell]
            row = {'auc_source': sname, 'frontier': fname, 'y_metric': 'adversary_auc',
                   'frontier_points': len(present), 'arms_used': ';'.join(present),
                   'label': 'secondary (P2b), not pre-declared'}
            if 'DQN-ES' not in cell or len(present) < 2:
                row['status'] = 'not computable (frontier arms absent)'
                rows.append(row)
                continue
            res = matched_latency_excess(cell, 'DQN-ES', arms, ML['x_metric'], 'adversary_auc')
            ex = list(res['excess_by_run'].values())
            lo, hi = median_bootstrap_ci(ex, STAT_BOOT_N, STAT_CI_LEVEL,
                                         STAT_BOOT_SEED + len(rows))
            reasons = _exclusion_reasons(cell, 'DQN-ES', arms, ML['x_metric'], 'adversary_auc')
            row.update(status='ok', n_replicates=len(ex), n_excluded=len(res['excluded_runs']),
                       median_excess=float(np.median(ex)) if ex else float('nan'),
                       ci_lo=lo, ci_hi=hi,
                       excluded_runs=';'.join(map(str, res['excluded_runs'])),
                       exclusion_reasons='; '.join(
                           f'run {r}: {reasons.get(r, "no test AUC for this replicate")}'
                           for r in res['excluded_runs']))
            rows.append(row)
            print(f"  {sname:<15} {fname:<9} AUC excess {row['median_excess']:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] n={len(ex)} excl={row['n_excluded']}")
    _long_path(Path(out_dir)).mkdir(parents=True, exist_ok=True)
    _csv(Path(out_dir) / 'auc_matched_latency.csv', rows)


LAMBDA_MECH_ARMS = ['ES-only[lP=0.5]', 'ES-only[lP=1]', 'ES-only[lP=3]', 'ES-only[lP=10]']


def cmd_lambda_mech(raw_dirs: List[Path], out_dir: Path) -> None:
    """
    Secondary: destination distribution by true CI tier and the pooled
    I(dest; tier) for the lambda_P arms, over all replicates.
    """
    from src.analysis.e5_adversary import TIERS, _dest, mutual_information
    from src.simulation.episode_log import _long_path
    idx = index_logs(raw_dirs)
    rows = []
    for a in LAMBDA_MECH_ARMS:
        if a not in idx:
            continue
        runs = load_runs(idx[a], ('device_id', 'assigned_node', 'ci_tier'))
        cnt = defaultdict(lambda: defaultdict(int))
        dests, tiers = [], []
        for rs in runs.values():
            for r in rs:
                d = _dest(r)
                cnt[r['ci_tier']][d] += 1
                dests.append(d)
                tiers.append(r['ci_tier'])
        mi_pooled = mutual_information(dests, tiers)
        allds = sorted({d for t in cnt for d in cnt[t]})
        for t in TIERS:
            n = sum(cnt[t].values())
            rows.append({'scheduler': a, 'ci_tier': t, 'n_tasks': n,
                         **{f'p_{d}': cnt[t][d] / n if n else 0.0 for d in allds},
                         'mi_pooled_bits': mi_pooled,
                         'label': 'secondary (P2b), not pre-declared'})
            print(f"  {a:<16} {t:<6} n={n:<6} " + ' '.join(
                f"{d}={cnt[t][d] / n:.3f}" for d in allds) + f"  MI(pooled)={mi_pooled:.4f}")
    _long_path(Path(out_dir)).mkdir(parents=True, exist_ok=True)
    _csv(Path(out_dir) / 'lambda_p_destination_by_tier.csv', rows)

# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='post-run analyses (P2)')
    p.add_argument('command', choices=['e5', 'corr', 'e8', 'e11', 'd17a', 'dominance', 'paired',
                                          'e5cv30', 'aucmatched', 'lambdamech'])
    p.add_argument('--raw-dirs', nargs='+', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--algorithms', nargs='+', default=None)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--e5-dir', type=str, default=None)
    p.add_argument('--summaries', nargs='+', default=[])
    p.add_argument('--scale', type=int, default=1000)
    p.add_argument('--pairs', nargs='+', default=[], help="'A|B' arm pairs (paired)")
    p.add_argument('--tag', type=str, default='pairs')
    p.add_argument('--step', type=str, default=None,
                   help='manifest step name (default post_<command>)')
    p.add_argument('--results-dir', type=str, default=str(_ROOT / 'results'))
    p.add_argument('--label', type=str, default=None,
                   help="recorded in the manifest, e.g. 'secondary (P2b), not pre-declared'")
    a = p.parse_args()
    out = Path(a.out)
    def rel(x):
        x = Path(x).resolve()
        try:
            return x.relative_to(_ROOT).as_posix()
        except ValueError:
            return x.as_posix()
    params = {'command': a.command,
              'raw_dirs': [rel(d) for d in a.raw_dirs],
              'out': rel(out),
              'algorithms': a.algorithms, 'e5_dir': a.e5_dir,
              'summaries': [rel(s) for s in a.summaries], 'scale': a.scale,
              'pairs': a.pairs, 'tag': a.tag, 'label': a.label}
    with _manifest(Path(a.results_dir), a.step or f'post_{a.command}', params):
        if a.command == 'e5':
            cmd_e5([Path(d) for d in a.raw_dirs], out, a.workers, a.algorithms)
        elif a.command == 'corr':
            cmd_corr(Path(a.e5_dir), [Path(s) for s in a.summaries], a.scale, out)
        elif a.command == 'e8':
            cmd_e8([Path(d) for d in a.raw_dirs], out, a.algorithms)
        elif a.command == 'e11':
            from src.analysis.e11_exploration import run as e11_run
            e11_run(a.raw_dirs, out, a.algorithms)
        elif a.command == 'd17a':
            cmd_d17a(out)
        elif a.command == 'dominance':
            cmd_dominance([Path(s) for s in a.summaries], a.scale, out)
        elif a.command == 'paired':
            cmd_paired(a.pairs, out, a.tag)
        elif a.command == 'e5cv30':
            cmd_e5_cv30([Path(d) for d in a.raw_dirs], out, a.workers, a.algorithms)
        elif a.command == 'aucmatched':
            cmd_auc_matched(Path(a.e5_dir), [Path(s) for s in a.summaries], a.scale, out)
        elif a.command == 'lambdamech':
            cmd_lambda_mech([Path(d) for d in a.raw_dirs], out)


if __name__ == '__main__':
    main()
