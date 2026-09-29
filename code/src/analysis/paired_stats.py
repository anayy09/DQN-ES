"""
Paired replicate statistics (plan E10).

Unit of analysis = replicate.  Replicate r of every algorithm uses the same
seeds (tasks, topology, environment; src.config.replicate_seeds), so the
per-run means of two algorithms form n paired observations and the paired
difference d_r = reference_r - comparator_r is the quantity tested.

  paired_signed_rank()   Wilcoxon signed-rank on d (zeros dropped, 'wilcox'),
                         exact p from the permutation (sign-flip) distribution
                         of W+ computed by dynamic programming over mid-ranks,
                         so p stays exact with tied |d|.
  hodges_lehmann()       median of the Walsh averages (d_i + d_j)/2, i <= j.
  hl_bootstrap_ci()      percentile CI of the HL estimate, resampling
                         replicates (eval_stats.multinomial_weights with one
                         group per replicate; eval_stats.percentile_ci).
  rank_biserial()        matched-pairs rank-biserial r = (W+ - W-)/(W+ + W-).
  holm_adjust()          Holm step-down adjustment (same procedure as
                         eval_stats.cmd_holm) over a declared family.
  tost()                 optional paired TOST: two one-sided exact signed-rank
                         tests on d + delta > 0 and d - delta < 0.

No scipy dependency here; eval_stats imports scipy only if available.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analysis.eval_stats import multinomial_weights, percentile_ci


# --------------------------------------------------------------------------
# Signed-rank machinery
# --------------------------------------------------------------------------

def _midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _nonzero(d: np.ndarray, tol: float) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    return d[np.abs(d) > tol]


def _signed_rank_null(ranks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact null distribution of W+ given (mid)ranks: support and probs."""
    twice = np.rint(2.0 * ranks).astype(int)          # mid-ranks are k/2
    total = int(twice.sum())
    counts = np.zeros(total + 1, dtype=float)
    counts[0] = 1.0
    for r in twice:                                    # each sign +/- w.p. 1/2
        shifted = np.zeros_like(counts)
        shifted[r:] = counts[:len(counts) - r]
        counts = 0.5 * (counts + shifted)
    support = np.arange(total + 1) / 2.0
    return support, counts


def _w_plus(d: np.ndarray) -> Tuple[float, float, np.ndarray]:
    ranks = _midranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    return w_plus, w_minus, ranks


def signed_rank_p(d: Sequence[float], alternative: str = 'two-sided',
                  tol: float = 1e-12) -> Tuple[float, float, int]:
    """
    Exact Wilcoxon signed-rank p-value.  Returns (p, W+, n_nonzero).
    alternative: 'two-sided' | 'greater' (median d > 0) | 'less'.
    """
    dz = _nonzero(np.asarray(d, dtype=float), tol)
    n = len(dz)
    if n == 0:
        return 1.0, 0.0, 0
    w_plus, _, ranks = _w_plus(dz)
    support, prob = _signed_rank_null(ranks)
    eps = 1e-9
    if alternative == 'greater':
        p = prob[support >= w_plus - eps].sum()
    elif alternative == 'less':
        p = prob[support <= w_plus + eps].sum()
    else:
        mean = ranks.sum() / 2.0
        dev = abs(w_plus - mean)
        p = prob[np.abs(support - mean) >= dev - eps].sum()
    return float(min(1.0, p)), w_plus, n


def rank_biserial(d: Sequence[float], tol: float = 1e-12) -> float:
    """Matched-pairs rank-biserial r in [-1, 1] (sign of d = ref - cmp)."""
    dz = _nonzero(np.asarray(d, dtype=float), tol)
    if len(dz) == 0:
        return 0.0
    w_plus, w_minus, _ = _w_plus(dz)
    return float((w_plus - w_minus) / (w_plus + w_minus))


# --------------------------------------------------------------------------
# Hodges-Lehmann and its replicate bootstrap
# --------------------------------------------------------------------------

def hodges_lehmann(d: Sequence[float]) -> float:
    d = np.asarray(d, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def hl_bootstrap_ci(d: Sequence[float], n_boot: int = 10_000,
                    level: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Percentile CI of the HL paired difference, resampling replicates."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2 or np.all(d == d[0]):
        return float(d[0]) if n else float('nan'), float(d[0]) if n else float('nan')
    rng = np.random.default_rng(seed)
    W = multinomial_weights(n, n_boot, rng).astype(int)   # one group per replicate
    iu, ju = np.triu_indices(n)       # each resample has exactly n replicates
    stats = np.empty(n_boot)
    for b in range(n_boot):
        s = np.repeat(d, W[b])
        stats[b] = np.median((s[iu] + s[ju]) / 2.0)
    return percentile_ci(stats, 1.0 - level)


# --------------------------------------------------------------------------
# Multiplicity and equivalence
# --------------------------------------------------------------------------

def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (procedure of eval_stats.cmd_holm)."""
    ps = np.asarray(pvals, dtype=float)
    m = len(ps)
    order = np.argsort(ps, kind='mergesort')
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def tost(d: Sequence[float], margin: float) -> Dict[str, float]:
    """
    Paired TOST with equivalence margin +/- margin on the median difference:
    H01: median d <= -margin  (test d + margin > 0)
    H02: median d >= +margin  (test d - margin < 0)
    p_tost = max of the two exact one-sided signed-rank p-values.
    """
    d = np.asarray(d, dtype=float)
    p_low, _, _ = signed_rank_p(d + margin, alternative='greater')
    p_high, _, _ = signed_rank_p(d - margin, alternative='less')
    return {'margin': float(margin), 'p_lower': p_low, 'p_upper': p_high,
            'p_tost': max(p_low, p_high)}


# --------------------------------------------------------------------------
# One paired comparison
# --------------------------------------------------------------------------

def paired_comparison(ref: Dict[int, float], cmp: Dict[int, float],
                      n_boot: int = 10_000, level: float = 0.95,
                      seed: int = 0,
                      tost_margin: Optional[float] = None) -> dict:
    """
    ref / cmp map run_id -> per-run mean.  Only run_ids present in both are
    paired.  d = ref - cmp.
    """
    ids = sorted(set(ref) & set(cmp))
    d = np.array([ref[i] - cmp[i] for i in ids], dtype=float)
    out = {
        'n_pairs': len(ids),
        'mean_ref': float(np.mean([ref[i] for i in ids])) if ids else float('nan'),
        'mean_cmp': float(np.mean([cmp[i] for i in ids])) if ids else float('nan'),
        'identical': bool(len(ids) > 0 and np.all(np.abs(d) <= 1e-12)),
    }
    if not ids:
        out.update(hl_diff=float('nan'), ci_lo=float('nan'), ci_hi=float('nan'),
                   w_plus=float('nan'), n_nonzero=0, p_exact=float('nan'),
                   rank_biserial_r=float('nan'))
        return out
    p, w_plus, n_nz = signed_rank_p(d)
    lo, hi = hl_bootstrap_ci(d, n_boot=n_boot, level=level, seed=seed)
    out.update(hl_diff=hodges_lehmann(d), ci_lo=lo, ci_hi=hi, w_plus=w_plus,
               n_nonzero=n_nz, p_exact=p, rank_biserial_r=rank_biserial(d))
    if tost_margin is not None:
        t = tost(d, tost_margin)
        out.update(tost_margin=t['margin'], p_tost=t['p_tost'])
    return out


def samples_by_run(cell: dict) -> Dict[int, float]:
    """
    {run_id: value} from a summary cell {'samples': [...], 'run_ids': [...]}.
    Summaries written before run_ids were recorded fall back to list order.
    """
    samples = cell.get('samples', [])
    run_ids = cell.get('run_ids') or list(range(len(samples)))
    if len(run_ids) != len(samples):
        raise ValueError('samples and run_ids differ in length')
    return {int(r): float(v) for r, v in zip(run_ids, samples)}
