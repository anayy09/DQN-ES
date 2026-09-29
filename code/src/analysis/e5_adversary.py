"""
e5_adversary.py — learned acuity-inference adversary on routing traces (plan E5).

Reads per-run raw logs only (simulation/episode_log.py format).

Threat model.  A passive on-path observer sees, per device, the sequence of
destinations it offloads to, when tasks leave the device, how large they
are, and how long each takes to complete.  It wants to know whether the
patient is in a high-acuity period.

Samples and labels (ruling D20).  One sample per task.  Binary label: the
task's true CI tier is 'high'; 3-tier label: the tier.  (The synthetic
generator draws tiers i.i.d. per task, so window-majority labels do not
occur; the task is the unit at which acuity varies.)

Features (cumulative ablations).
  dest          the task's destination (one-hot) and the device's histogram
                over its preceding E5_ADVERSARY['context'] (50) destinations
                (plus how many preceding decisions exist, up to 50)
  dest+timing   + inter-arrival time since the device's previous task and the
                task's response time (latency)
  dest+timing+size
                + payload size

Adversary.  Adaptive: a separate classifier is trained on each scheduler's
own traces, so it knows the policy it attacks.  HistGradientBoosting
(primary) and logistic regression on standardised features (check).

Evaluation.
  split         train on replicates E5_ADVERSARY['train_runs'], test on
                E5_ADVERSARY['test_runs'] (replicate-disjoint); AUC on the
                pooled test tasks and per test replicate (the unit of the
                D17(b) 'privacy_inference' family)
  cv            5-fold cross-validation grouped by replicate
  macro3        one-vs-rest macro AUC for the 3-tier label (primary config)
  MI            plug-in mutual information I(destination; CI tier) in bits
                per task, per replicate; mean with a replicate-bootstrap CI

Outputs (<out>/):
  adversary_auc_summary.csv     scheduler x feature set x classifier
  adversary_auc_by_replicate.json
                {scheduler: {'adversary_auc': {'samples', 'run_ids'}}} for
                the primary configuration -> statistical_tests family
                'privacy_inference'
  mutual_information.csv
  e5_meta.json                  sample counts and label balance

Usage (from code/):
  python -m src.analysis.e5_adversary --raw-dirs ../results/raw/mc_full/n1000 \\
      ../results/experiments/raw/mc_exp/n1000 --out ../results/e5_adversary
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

TIERS = ['low', 'medium', 'high']
FEATURE_SETS = ['dest', 'dest+timing', 'dest+timing+size']
LOG_RE = re.compile(r'^(?P<alg>.+)__run(?P<run>\d+)\.csv\.gz$')


def _cfg() -> dict:
    from src.config import E5_ADVERSARY
    return E5_ADVERSARY


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_logs(raw_dirs: List[Path], algorithms: Optional[List[str]] = None
              ) -> Dict[str, Dict[int, List[dict]]]:
    """{algorithm: {run_id: [rows]}} from every *__runRR.csv.gz found."""
    from src.simulation.episode_log import _long_path
    out: Dict[str, Dict[int, List[dict]]] = defaultdict(dict)
    for d in raw_dirs:
        d = _long_path(Path(d))
        if not d.exists():
            print(f'[E5] missing {d}')
            continue
        for f in sorted(d.iterdir()):
            m = LOG_RE.match(f.name)
            if not m:
                continue
            with gzip.open(f, 'rt', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                continue
            alg = rows[0]['algorithm']
            if algorithms and alg not in algorithms:
                continue
            out[alg][int(m.group('run'))] = rows
    return out


# --------------------------------------------------------------------------
# Windows and features
# --------------------------------------------------------------------------

def _dest(row: dict) -> str:
    return 'local' if row['assigned_node'] == row['device_id'] else f"n{row['assigned_node']}"


def _q(a, qs):
    return list(np.quantile(a, qs)) if len(a) else [0.0] * len(qs)


def build_samples(runs: Dict[int, List[dict]], context: int,
                  dest_vocab: List[str]):
    """One sample per task: run_id, labels, and features by feature set."""
    k = len(dest_vocab)
    idx = {d: i for i, d in enumerate(dest_vocab)}
    out = []
    for run_id, rows in runs.items():
        by_dev = defaultdict(list)
        for r in rows:
            by_dev[r['device_id']].append(r)
        for dev, rs in by_dev.items():
            rs.sort(key=lambda r: float(r['timestamp']))
            hist = []                                # preceding destinations
            prev_t = None
            for r in rs:
                d = idx[_dest(r)]
                one_hot = [0.0] * k
                one_hot[d] = 1.0
                h = np.bincount(hist[-context:], minlength=k).astype(float)
                n_prev = min(len(hist), context)
                h = list(h / n_prev) if n_prev else [0.0] * k
                t = float(r['timestamp'])
                ia = (t - prev_t) if prev_t is not None else float('nan')
                dest = one_hot + h + [n_prev / context]
                timing = dest + [ia, float(r['latency_ms'])]
                size = timing + [float(r['payload_bits'])]
                tier = r['ci_tier']
                out.append({'run_id': run_id, 'y': int(tier == 'high'),
                            'y3': TIERS.index(tier),
                            'x': {'dest': dest, 'dest+timing': timing,
                                  'dest+timing+size': size}})
                hist.append(d)
                prev_t = t
    return out


def dest_vocabulary(all_runs: Dict[str, Dict[int, List[dict]]]) -> List[str]:
    v = set()
    for runs in all_runs.values():
        for rows in runs.values():
            v.update(_dest(r) for r in rows)
    return ['local'] + sorted(d for d in v if d != 'local')


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------

def _make(clf: str, seed: int):
    if clf == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(max_iter=2000))


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float('nan')


def evaluate(wins: List[dict], feat: str, clf: str, seed: int) -> dict:
    cfg = _cfg()
    X = np.array([w['x'][feat] for w in wins], dtype=float)
    y = np.array([w['y'] for w in wins])
    y3 = np.array([w['y3'] for w in wins])
    g = np.array([w['run_id'] for w in wins])
    tr = np.isin(g, cfg['train_runs'])
    te = np.isin(g, cfg['test_runs'])
    res = {'n_samples': int(len(wins)), 'n_train': int(tr.sum()),
           'n_test': int(te.sum()), 'pos_rate': float(y.mean()) if len(y) else float('nan')}
    if tr.sum() == 0 or te.sum() == 0 or len(set(y[tr].tolist())) < 2:
        res.update(pooled_test_auc=float('nan'), per_run_auc={})
    else:
        m = _make(clf, seed).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        res['pooled_test_auc'] = _auc(y[te], p)
        per = {}
        for r in sorted(set(g[te].tolist())):
            sel = g[te] == r
            per[int(r)] = _auc(y[te][sel], p[sel])
        res['per_run_auc'] = per
        if clf == 'hgb' and len(set(y3[tr].tolist())) == 3 and len(set(y3[te].tolist())) == 3:
            from sklearn.metrics import roc_auc_score
            m3 = _make(clf, seed).fit(X[tr], y3[tr])
            res['macro3_auc'] = float(roc_auc_score(y3[te], m3.predict_proba(X[te]),
                                                    multi_class='ovr', average='macro'))
    # 5-fold grouped by replicate
    groups = sorted(set(g.tolist()))
    if len(groups) >= 5 and len(set(y.tolist())) == 2:
        from sklearn.model_selection import GroupKFold
        aucs = []
        for a, b in GroupKFold(n_splits=5).split(X, y, g):
            if len(set(y[a].tolist())) < 2:
                continue
            p = _make(clf, seed).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
            aucs.append(_auc(y[b], p))
        aucs = [a for a in aucs if not math.isnan(a)]
        res['cv_auc_mean'] = float(np.mean(aucs)) if aucs else float('nan')
        res['cv_auc_sd'] = float(np.std(aucs)) if aucs else float('nan')
    return res


# --------------------------------------------------------------------------
# Mutual information I(destination; CI tier)
# --------------------------------------------------------------------------

def mutual_information(dest: List[str], tier: List[str]) -> float:
    """Plug-in MI in bits between two discrete sequences."""
    n = len(dest)
    if n == 0:
        return float('nan')
    joint = defaultdict(int)
    for a, b in zip(dest, tier):
        joint[(a, b)] += 1
    pa, pb = defaultdict(int), defaultdict(int)
    for (a, b), c in joint.items():
        pa[a] += c
        pb[b] += c
    return float(sum(c / n * math.log2(c * n / (pa[a] * pb[b]))
                     for (a, b), c in joint.items()))


def mi_by_run(runs: Dict[int, List[dict]]) -> Dict[int, float]:
    return {r: mutual_information([_dest(x) for x in rows], [x['ci_tier'] for x in rows])
            for r, rows in runs.items()}


def mean_bootstrap_ci(values, n_boot: int, level: float, seed: int):
    from src.analysis.eval_stats import multinomial_weights, percentile_ci
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return (float(v[0]), float(v[0])) if v.size else (float('nan'),) * 2
    W = multinomial_weights(v.size, n_boot, np.random.default_rng(seed))
    return percentile_ci((W @ v) / v.size, 1.0 - level)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(raw_dirs: List[Path], out_dir: Path, algorithms: Optional[List[str]] = None,
        feature_sets: List[str] = FEATURE_SETS, classifiers=('hgb', 'logreg')) -> dict:
    from src.config import STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    cfg = _cfg()
    from src.simulation.episode_log import _long_path
    out_dir = Path(out_dir)
    _long_path(out_dir).mkdir(parents=True, exist_ok=True)
    logs = load_logs(raw_dirs, algorithms)
    if not logs:
        raise SystemExit('[E5] no raw logs found')
    vocab = dest_vocabulary(logs)
    print(f'[E5] schedulers={sorted(logs)} destinations={vocab} '
          f'unit=task context={cfg["context"]}')

    rows, primary, meta = [], {}, {}
    for alg in sorted(logs):
        wins = build_samples(logs[alg], cfg['context'], vocab)
        meta[alg] = {'n_samples': len(wins), 'n_runs': len(logs[alg]),
                     'pos_rate': float(np.mean([w['y'] for w in wins])) if wins else None}
        for feat in feature_sets:
            for clf in classifiers:
                r = evaluate(wins, feat, clf, cfg['seed'])
                per = r.pop('per_run_auc', {})
                ok = [v for v in per.values() if not math.isnan(v)]
                rows.append({'scheduler': alg, 'features': feat, 'classifier': clf,
                             **r, 'per_run_auc_median': float(np.median(ok)) if ok else float('nan'),
                             'n_test_runs_with_auc': len(ok)})
                if feat == cfg['primary_features'] and clf == cfg['primary_classifier']:
                    ids = sorted(k for k, v in per.items() if not math.isnan(v))
                    primary[alg] = {'adversary_auc': {'samples': [per[k] for k in ids],
                                                      'run_ids': ids}}
                print(f"  {alg:<16} {feat:<17} {clf:<6} pooled AUC "
                      f"{r['pooled_test_auc']:.3f}  cv {r.get('cv_auc_mean', float('nan')):.3f}"
                      f"  tasks={r['n_samples']}")

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in (
        'scheduler', 'features', 'classifier'), k))
    with open(_long_path(out_dir / 'adversary_auc_summary.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, restval='')
        wr.writeheader()
        wr.writerows(rows)
    _long_path(out_dir / 'adversary_auc_by_replicate.json').write_text(
        json.dumps({'primary': {k: cfg[k] for k in ('unit', 'primary_features',
                                                    'primary_classifier', 'context',
                                                    'train_runs', 'test_runs')},
                    'cells': primary}, indent=1), encoding='utf-8')

    mi_rows = []
    for i, alg in enumerate(sorted(logs)):
        mi = mi_by_run(logs[alg])
        v = list(mi.values())
        lo, hi = mean_bootstrap_ci(v, STAT_BOOT_N, STAT_CI_LEVEL, STAT_BOOT_SEED + i)
        mi_rows.append({'scheduler': alg, 'n_runs': len(v),
                        'mi_bits_mean': float(np.mean(v)), 'ci_lo': lo, 'ci_hi': hi})
        print(f'  MI {alg:<16} {np.mean(v):.4f} bits [{lo:.4f}, {hi:.4f}]')
    with open(_long_path(out_dir / 'mutual_information.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(mi_rows[0]))
        wr.writeheader()
        wr.writerows(mi_rows)
    _long_path(out_dir / 'e5_meta.json').write_text(
        json.dumps({'vocab': vocab, 'schedulers': meta,
                    'raw_dirs': [str(d) for d in raw_dirs]}, indent=1), encoding='utf-8')
    print(f'[E5] wrote {out_dir}')
    return {'rows': rows, 'primary': primary, 'mi': mi_rows}


def main():
    p = argparse.ArgumentParser(description='E5 learned acuity-inference adversary')
    p.add_argument('--raw-dirs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--algorithms', nargs='+', default=None)
    a = p.parse_args()
    run([Path(d) for d in a.raw_dirs], Path(a.out), a.algorithms)


if __name__ == '__main__':
    main()
