"""
statistical_tests.py — paired replicate statistics for the comparison tables
(plan E10).

For each declared family in src.config.STAT_FAMILIES, every (comparator,
metric) pair is tested against the family's reference on the per-run means:

  unit            replicate (paired by run_id; same seeds for every algorithm)
  difference      d = reference - comparator  (negative = reference lower)
  test            Wilcoxon signed-rank, exact p (zeros dropped)
  multiplicity    Holm across all tests of the family
  effect size     Hodges-Lehmann paired difference, replicate-bootstrap CI
                  (STAT_BOOT_N resamples, STAT_CI_LEVEL), matched-pairs
                  rank-biserial r
  equivalence     TOST with STAT_TOST_MARGINS, only if STAT_TOST_ENABLED

A comparator whose per-run values equal the reference's on every replicate is
reported as identical (p = 1, d = 0) rather than tested for equivalence.

Outputs (default names kept for the pipeline):
  <out>/table3_stat_tests.csv          one row per test of the 'main' family
  <out>/table3_n1000_with_pvals.csv    mean ± sd per algorithm, with the HL
                                       difference [CI] and Holm p vs DQN-ES
  <out>/stat_tests_<family>.csv        any other family that was run
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.analysis.paired_stats import holm_adjust, paired_comparison, samples_by_run
from src.config import (
    PRIMARY_SCALE,
    STAT_ALPHA,
    STAT_BOOT_N,
    STAT_BOOT_SEED,
    STAT_CI_LEVEL,
    STAT_FAMILIES,
    STAT_TOST_ENABLED,
    STAT_TOST_MARGINS,
)


def run_family_tests(cell: Dict[str, dict], family: str,
                     n_boot: int = STAT_BOOT_N,
                     tost_enabled: bool = STAT_TOST_ENABLED) -> List[dict]:
    """
    cell: {algorithm: {metric: {'samples': [...], 'run_ids': [...]}}}.
    Returns one row per (comparator, metric) of the declared family with
    Holm-adjusted p-values.  Comparators absent from `cell` are reported as
    missing rather than silently dropped from the family.
    """
    spec = STAT_FAMILIES[family]
    ref_name = spec['reference']
    if ref_name not in cell:
        raise KeyError(f'reference {ref_name!r} not in summary')

    rows: List[dict] = []
    for comp in spec['comparators']:
        for metric in spec['metrics']:
            row = {'family': family, 'metric': metric,
                   'reference': ref_name, 'comparator': comp}
            if comp not in cell or metric not in cell[comp]:
                row['status'] = 'missing'
                rows.append(row)
                continue
            margin = STAT_TOST_MARGINS.get(metric) if tost_enabled else None
            seed = STAT_BOOT_SEED + len(rows)
            res = paired_comparison(samples_by_run(cell[ref_name][metric]),
                                    samples_by_run(cell[comp][metric]),
                                    n_boot=n_boot, level=STAT_CI_LEVEL,
                                    seed=seed, tost_margin=margin)
            row.update(res)
            row['status'] = 'identical' if res['identical'] else 'tested'
            rows.append(row)

    # Holm over every test the family declares; a missing test counts as
    # p = 1 so the family size is never reduced after the fact.
    ps = [r.get('p_exact', 1.0) if r['status'] != 'missing' else 1.0
          for r in rows]
    ps = [1.0 if (p is None or (isinstance(p, float) and math.isnan(p))) else p
          for p in ps]
    adj = holm_adjust(ps)
    for r, pa in zip(rows, adj):
        r['family_size'] = len(rows)
        r['p_holm'] = float(pa)
        r['reject_holm'] = bool(r['status'] == 'tested' and pa <= STAT_ALPHA)
    return rows


ROW_FIELDS = [
    'family', 'metric', 'reference', 'comparator', 'status', 'n_pairs',
    'mean_ref', 'mean_cmp', 'hl_diff', 'ci_lo', 'ci_hi', 'w_plus',
    'n_nonzero', 'p_exact', 'family_size', 'p_holm', 'reject_holm',
    'rank_biserial_r', 'tost_margin', 'p_tost',
]


def _write_rows(rows: List[dict], path: Path) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction='ignore',
                            restval='')
        wr.writeheader()
        wr.writerows(rows)
    print(f'[STAT] Saved {path}')


def _fmt_p(p: float) -> str:
    return f'{p:.2e}' if p < 1e-3 else f'{p:.4f}'


def _load_cells(summary_paths: Iterable[Path], scale: int) -> Dict[str, dict]:
    """Merge the scale's cells from several summaries (main + experiments)."""
    cell: Dict[str, dict] = {}
    for p in summary_paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p, 'r', encoding='utf-8') as fh:
            summary = json.load(fh)
        cell.update(summary.get(str(scale), {}))
    return cell


def run_pairwise_tests(summary_path: Path, out_dir: Path,
                       scale: int = PRIMARY_SCALE,
                       workers: Optional[int] = None,
                       families: Iterable[str] = ('main',),
                       extra_summaries: Iterable[Path] = (),
                       n_boot: int = STAT_BOOT_N) -> dict:
    """
    Run the declared families on one scale.  `workers` is accepted for the
    pipeline's call signature and unused (the tests are cheap).
    Returns {family: [rows]}.
    """
    cell = _load_cells([summary_path, *extra_summaries], scale)
    if not cell:
        raise KeyError(f'scale {scale} not in {summary_path}')
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    for fam in families:
        rows = run_family_tests(cell, fam, n_boot=n_boot)
        out[fam] = rows
        name = 'table3_stat_tests.csv' if fam == 'main' else f'stat_tests_{fam}.csv'
        _write_rows(rows, out_dir / name)
        _print_family(fam, rows, scale)

    if 'main' in out:
        _write_formatted_table(cell, out['main'], out_dir, scale)
    return out


def _write_formatted_table(cell: dict, rows: List[dict], out_dir: Path,
                           scale: int) -> None:
    spec = STAT_FAMILIES['main']
    ref = spec['reference']
    by = {(r['comparator'], r['metric']): r for r in rows}
    path = out_dir / f'table3_n{scale}_with_pvals.csv'
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(['algorithm', *[f'{m}: mean ± sd; HL d [{int(STAT_CI_LEVEL*100)}% CI]; p_holm'
                                    for m in spec['metrics']]])
        for alg in [ref] + list(spec['comparators']):
            if alg not in cell:
                continue
            out_row = [alg]
            for m in spec['metrics']:
                d = cell[alg][m]
                txt = f"{d['mean']:.3f} ± {d['std']:.3f}"
                r = by.get((alg, m))
                if alg != ref and r and r['status'] == 'tested':
                    star = '*' if r['reject_holm'] else ''
                    txt += (f"; d={r['hl_diff']:.3f} [{r['ci_lo']:.3f}, "
                            f"{r['ci_hi']:.3f}]; p={_fmt_p(r['p_holm'])}{star}")
                elif alg != ref and r and r['status'] == 'identical':
                    txt += '; identical decisions'
                out_row.append(txt)
            wr.writerow(out_row)
        wr.writerow([])
        wr.writerow([
            f"Footnote: d = {ref} - comparator, Hodges-Lehmann median of the "
            f"paired per-replicate differences with a {int(STAT_CI_LEVEL*100)}% "
            f"percentile bootstrap CI ({STAT_BOOT_N} resamples of replicates); "
            f"p = exact Wilcoxon signed-rank, Holm-adjusted over the declared "
            f"family ({len(spec['comparators'])} comparators x "
            f"{len(spec['metrics'])} metrics = {len(rows)} tests); "
            f"* p_holm <= {STAT_ALPHA}. Replicates paired by seed (run_id)."
        ])
    print(f'[STAT] Saved {path}')


def _print_family(fam: str, rows: List[dict], scale: int) -> None:
    print(f'\n[STAT] family={fam} scale={scale} tests={len(rows)} '
          f'(paired signed-rank, exact; Holm)')
    for r in rows:
        if r['status'] == 'missing':
            print(f"  {r['comparator']:<16} {r['metric']:<18} MISSING")
            continue
        if r['status'] == 'identical':
            print(f"  {r['comparator']:<16} {r['metric']:<18} identical "
                  f"(n={r['n_pairs']})")
            continue
        tag = ' *' if r['reject_holm'] else '  '
        print(f"  {r['comparator']:<16} {r['metric']:<18} n={r['n_pairs']:<3d} "
              f"d={r['hl_diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
              f"r={r['rank_biserial_r']:+.2f} p_holm={_fmt_p(r['p_holm'])}{tag}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=str, default=None,
                   help='Path to mc_full_summary.json')
    p.add_argument('--extra-summary', type=str, nargs='*', default=[],
                   help='More summaries to merge (e.g. mc_exp_summary.json)')
    p.add_argument('--families', nargs='+', default=['main'],
                   choices=sorted(STAT_FAMILIES))
    p.add_argument('--scale', type=int, default=PRIMARY_SCALE)
    p.add_argument('--n-boot', type=int, default=STAT_BOOT_N)
    p.add_argument('--output', type=str, default=None,
                   help='Output directory for CSVs')
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    summary_path = (Path(args.summary) if args.summary
                    else project_root / 'results' / 'mc_full_summary.json')
    out_dir = (Path(args.output) if args.output
               else project_root / 'results')
    run_pairwise_tests(summary_path, out_dir, scale=args.scale,
                       families=args.families,
                       extra_summaries=[Path(x) for x in args.extra_summary],
                       n_boot=args.n_boot)


if __name__ == '__main__':
    main()
