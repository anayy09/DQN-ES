#!/usr/bin/env python3
"""Statistics for comparing your own models: clustered bootstrap intervals,
paired comparisons, calibration, and selective prediction.

Everything here defaults to resampling at the group level (patient, subject,
stay). Item-level resampling on clustered data understates the standard error,
often by a large factor, and no downstream analysis can repair it.

Subcommands
-----------
  ci           clustered bootstrap interval for one model's metric
  compare      paired clustered bootstrap difference between two models
  mcnemar      exact or corrected McNemar test on paired correctness
  delong       DeLong AUROC comparison (independent items only)
  calibration  ECE under both binning schemes, plus reliability bins
  selective    risk-coverage curve, AURC, coverage at risk, risk at coverage
  holm         Holm-Bonferroni correction across a family of p-values

Input is a CSV with one row per prediction. Requires numpy; pandas and scipy are
used when available but are not required.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _sps  # type: ignore
except ImportError:  # pragma: no cover
    _sps = None


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def read_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} is empty")
    cols: Dict[str, np.ndarray] = {}
    for k in rows[0]:
        raw = [r[k] for r in rows]
        try:
            cols[k] = np.array([float(v) for v in raw])
        except ValueError:
            cols[k] = np.array(raw, dtype=object)
    return cols


def need(cols: Dict[str, np.ndarray], name: str, what: str) -> np.ndarray:
    if name not in cols:
        sys.exit(f"column '{name}' ({what}) not found. available: {sorted(cols)}")
    return cols[name]


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def to_hard(pred: np.ndarray, classes: np.ndarray, threshold: float) -> np.ndarray:
    """Accept either predicted labels or scores. Scores are thresholded only in
    the binary case, where the threshold is a real modeling choice and should be
    stated in the paper rather than left implicit at 0.5."""
    if pred.dtype == object:
        return pred
    uniq = np.unique(pred)
    looks_like_scores = uniq.size > len(classes) or (
        uniq.min() >= 0.0 and uniq.max() <= 1.0 and uniq.size > 2)
    if looks_like_scores:
        if len(classes) != 2:
            sys.exit("continuous predictions with more than two classes: pass "
                     "predicted labels, not scores")
        return np.where(pred >= threshold, classes[1], classes[0])
    return pred


def auroc_weighted(y_pos: np.ndarray, order: np.ndarray, block: np.ndarray,
                   n_blocks: int, w: np.ndarray) -> float:
    """Weighted Mann-Whitney AUROC with tie handling.

    y_pos, order, block are precomputed once; w carries the bootstrap
    multiplicity of each item, so a resample costs one pass instead of a sort.
    """
    ws = w[order]
    pos = y_pos[order]
    neg_w = np.where(pos, 0.0, ws)
    pos_w = np.where(pos, ws, 0.0)

    block_neg = np.bincount(block, weights=neg_w, minlength=n_blocks)
    block_pos = np.bincount(block, weights=pos_w, minlength=n_blocks)
    # negative weight strictly below each tie block
    below = np.concatenate(([0.0], np.cumsum(block_neg)[:-1]))
    W_pos = block_pos.sum()
    W_neg = block_neg.sum()
    if W_pos == 0 or W_neg == 0:
        return float("nan")
    num = np.sum(block_pos * (below + 0.5 * block_neg))
    return float(num / (W_pos * W_neg))


class GroupStats:
    """Per-group confusion counts, so that ratio-of-sums metrics (accuracy,
    balanced accuracy, macro F1) can be bootstrapped by reweighting groups
    instead of rebuilding index arrays. This is exact, not an approximation."""

    def __init__(self, y: np.ndarray, p: np.ndarray, groups: np.ndarray):
        self.classes = np.unique(y)
        self.gids, self.g_index = np.unique(groups, return_inverse=True)
        G, C = len(self.gids), len(self.classes)
        self.tp = np.zeros((G, C))
        self.fp = np.zeros((G, C))
        self.fn = np.zeros((G, C))
        self.n_true = np.zeros((G, C))
        for ci, c in enumerate(self.classes):
            yt, pt = (y == c), (p == c)
            self.tp[:, ci] = np.bincount(self.g_index, weights=(yt & pt).astype(float),
                                         minlength=G)
            self.fp[:, ci] = np.bincount(self.g_index, weights=(~yt & pt).astype(float),
                                         minlength=G)
            self.fn[:, ci] = np.bincount(self.g_index, weights=(yt & ~pt).astype(float),
                                         minlength=G)
            self.n_true[:, ci] = np.bincount(self.g_index, weights=yt.astype(float),
                                             minlength=G)

    @property
    def n_groups(self) -> int:
        return len(self.gids)

    def metric(self, name: str, W: np.ndarray) -> np.ndarray:
        """W has shape (B, G) of group multiplicities. Returns (B,) metric values."""
        tp, fp, fn, nt = W @ self.tp, W @ self.fp, W @ self.fn, W @ self.n_true
        with np.errstate(invalid="ignore", divide="ignore"):
            if name == "accuracy":
                return tp.sum(axis=1) / nt.sum(axis=1)
            if name == "balanced_accuracy":
                recall = np.where(nt > 0, tp / np.maximum(nt, 1e-12), np.nan)
                return np.nanmean(recall, axis=1)
            if name == "macro_f1":
                f1 = np.where((2 * tp + fp + fn) > 0,
                              2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), np.nan)
                return np.nanmean(f1, axis=1)
        sys.exit(f"metric '{name}' is not a ratio-of-sums metric; use auroc path")


RATIO_METRICS = {"accuracy", "balanced_accuracy", "macro_f1"}


def multinomial_weights(n_groups: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster bootstrap: draw G groups with replacement. The resulting per-group
    counts are exactly multinomial, so this is the same procedure as resampling
    group index lists, computed in one step."""
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups),
                           size=n_boot).astype(float)


def percentile_ci(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(s, 100 * alpha / 2)),
            float(np.percentile(s, 100 * (1 - alpha / 2))))


def design_effect(n_items: int, n_groups: int) -> str:
    if n_groups <= 1:
        return ""
    ratio = n_items / n_groups
    return (f"  items/group      : {ratio:,.1f}  "
            f"(effective n is closer to {n_groups:,} than {n_items:,})")


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def _prep(a: argparse.Namespace, pred_cols: Sequence[str]):
    cols = read_csv(a.csv)
    y = need(cols, a.label, "true label")
    groups = need(cols, a.group, "grouping variable") if a.group else np.arange(len(y))
    if not a.group:
        print("WARNING: no --group given, resampling at the item level. If items "
              "are patches, windows, or repeated measures from the same subject, "
              "the intervals below are too narrow.", file=sys.stderr)
    classes = np.unique(y)
    preds = {c: need(cols, c, "prediction") for c in pred_cols}
    return cols, y, groups, classes, preds


def cmd_ci(a: argparse.Namespace) -> int:
    cols, y, groups, classes, preds = _prep(a, [a.pred])
    raw = preds[a.pred]
    rng = np.random.default_rng(a.seed)

    if a.metric == "auroc":
        if len(classes) != 2:
            sys.exit("auroc requires a binary label column")
        y_pos = (y == classes[1])
        order = np.argsort(raw, kind="mergesort")
        _, block = np.unique(raw[order], return_inverse=True)
        n_blocks = int(block.max()) + 1
        gids, gidx = np.unique(groups, return_inverse=True)
        W = multinomial_weights(len(gids), a.n_boot, rng)
        point = auroc_weighted(y_pos, order, block, n_blocks,
                               np.ones(len(y), dtype=float))
        boots = np.array([auroc_weighted(y_pos, order, block, n_blocks, W[b][gidx])
                          for b in range(a.n_boot)])
        n_groups = len(gids)
    else:
        p = to_hard(raw, classes, a.threshold)
        gs = GroupStats(y, p, groups)
        W = multinomial_weights(gs.n_groups, a.n_boot, rng)
        point = float(gs.metric(a.metric, np.ones((1, gs.n_groups)))[0])
        boots = gs.metric(a.metric, W)
        n_groups = gs.n_groups

    lo, hi = percentile_ci(boots, a.alpha)
    print(f"metric           : {a.metric}")
    print(f"point estimate   : {point:.4f}")
    print(f"{int((1-a.alpha)*100)}% CI           : [{lo:.4f}, {hi:.4f}]  "
          f"(percentile, {a.n_boot:,} resamples)")
    print(f"items            : {len(y):,}")
    print(f"groups (resampled): {n_groups:,}")
    print(design_effect(len(y), n_groups))
    if n_groups < 20:
        print("NOTE: fewer than 20 groups. The percentile bootstrap is unreliable "
              "here; report the group count prominently and treat the interval as "
              "indicative.")
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    cols, y, groups, classes, preds = _prep(a, [a.a, a.b])
    rng = np.random.default_rng(a.seed)
    gids, gidx = np.unique(groups, return_inverse=True)
    W = multinomial_weights(len(gids), a.n_boot, rng)

    if a.metric == "auroc":
        if len(classes) != 2:
            sys.exit("auroc requires a binary label column")
        y_pos = (y == classes[1])
        packs = {}
        for name in (a.a, a.b):
            s = preds[name]
            order = np.argsort(s, kind="mergesort")
            _, block = np.unique(s[order], return_inverse=True)
            packs[name] = (order, block, int(block.max()) + 1)
        ones = np.ones(len(y))
        pa = auroc_weighted(y_pos, *packs[a.a], ones)
        pb = auroc_weighted(y_pos, *packs[a.b], ones)
        ba = np.array([auroc_weighted(y_pos, *packs[a.a], W[i][gidx]) for i in range(a.n_boot)])
        bb = np.array([auroc_weighted(y_pos, *packs[a.b], W[i][gidx]) for i in range(a.n_boot)])
    else:
        ga = GroupStats(y, to_hard(preds[a.a], classes, a.threshold), groups)
        gb = GroupStats(y, to_hard(preds[a.b], classes, a.threshold), groups)
        one = np.ones((1, ga.n_groups))
        pa = float(ga.metric(a.metric, one)[0])
        pb = float(gb.metric(a.metric, one)[0])
        # The same W for both models: pairing removes shared item difficulty,
        # which is exactly the nuisance that makes marginal intervals misleading.
        ba = ga.metric(a.metric, W)
        bb = gb.metric(a.metric, W)

    diff = bb - ba
    point = pb - pa
    lo, hi = percentile_ci(diff, a.alpha)
    d = diff[np.isfinite(diff)]
    p_two = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    p_two = min(1.0, max(p_two, 1.0 / (len(d) + 1)))  # bounded below by resolution

    print(f"metric           : {a.metric}")
    print(f"{a.a:<16} : {pa:.4f}")
    print(f"{a.b:<16} : {pb:.4f}")
    print(f"paired difference: {point:+.4f}  ({a.b} minus {a.a})")
    print(f"{int((1-a.alpha)*100)}% CI of diff   : [{lo:+.4f}, {hi:+.4f}]")
    print(f"bootstrap p      : {p_two:.4g}  (two-sided, {a.n_boot:,} resamples)")
    print(f"items            : {len(y):,}    groups: {len(gids):,}")
    print(design_effect(len(y), len(gids)))
    if lo <= 0 <= hi:
        print("\nThe interval includes zero. Report this as no detected difference "
              "at this sample size, not as equivalence: absence of evidence needs "
              "an equivalence margin to become evidence of absence.")
    else:
        print("\nReport the difference and its interval, not the two marginal "
              "intervals. Overlapping marginal intervals are compatible with this "
              "result and do not contradict it.")
    return 0


def cmd_mcnemar(a: argparse.Namespace) -> int:
    cols = read_csv(a.csv)
    y = need(cols, a.label, "true label")
    classes = np.unique(y)
    ca = to_hard(need(cols, a.a, "prediction A"), classes, a.threshold) == y
    cb = to_hard(need(cols, a.b, "prediction B"), classes, a.threshold) == y

    n01 = int(np.sum(ca & ~cb))   # A right, B wrong
    n10 = int(np.sum(~ca & cb))   # B right, A wrong
    n = n01 + n10

    print(f"A correct, B wrong : {n01:,}")
    print(f"B correct, A wrong : {n10:,}")
    print(f"discordant pairs   : {n:,}  (concordant pairs carry no information)")
    if n == 0:
        print("The two models make identical errors. No test is possible.")
        return 0

    if n < 25 or a.exact:
        if _sps is not None:
            p = float(_sps.binomtest(n10, n, 0.5).pvalue)
        else:
            p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
        print(f"exact binomial p   : {p:.4g}")
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n
        p = 2 * (1 - normal_cdf(math.sqrt(chi2)))
        print(f"chi2 (continuity)  : {chi2:.4f}")
        print(f"p                  : {p:.4g}")

    print("\nMcNemar assumes independent pairs. With patches or windows from the "
          "same subject that assumption fails and the p-value is too small; use "
          "`compare` with --group instead.")
    return 0


def _fast_delong(scores: np.ndarray, y_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sun and Xu (2014) fast DeLong. scores has shape (k, n)."""
    order = np.argsort(-y_pos.astype(int), kind="mergesort")
    s = scores[:, order]
    m = int(y_pos.sum())
    n = s.shape[1] - m

    def midrank(x: np.ndarray) -> np.ndarray:
        J = np.argsort(x, kind="mergesort")
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(N)
        out[J] = T
        return out

    k = s.shape[0]
    tx = np.vstack([midrank(s[r, :m]) for r in range(k)])
    ty = np.vstack([midrank(s[r, m:]) for r in range(k)])
    tz = np.vstack([midrank(s[r, :]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def cmd_delong(a: argparse.Namespace) -> int:
    cols = read_csv(a.csv)
    y = need(cols, a.label, "true label")
    classes = np.unique(y)
    if len(classes) != 2:
        sys.exit("DeLong requires a binary label")
    y_pos = (y == classes[1])
    sa = need(cols, a.a, "score A").astype(float)
    sb = need(cols, a.b, "score B").astype(float)

    aucs, cov = _fast_delong(np.vstack([sa, sb]), y_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[1] - aucs[0]
    z = diff / math.sqrt(var) if var > 0 else float("nan")
    p = 2 * (1 - normal_cdf(abs(z))) if math.isfinite(z) else float("nan")

    print(f"AUROC {a.a:<12}: {aucs[0]:.4f}")
    print(f"AUROC {a.b:<12}: {aucs[1]:.4f}")
    print(f"difference       : {diff:+.4f}")
    print(f"z                : {z:.3f}")
    print(f"p (DeLong)       : {p:.4g}")
    print("\nDeLong assumes independent observations. If rows are patches, windows, "
          "or otherwise clustered within subjects, this p-value is too small. "
          "Use `compare --metric auroc --group <subject>` instead and report that.")
    return 0


def _ece(y_pos: np.ndarray, prob: np.ndarray, n_bins: int, scheme: str) -> Tuple[float, List[tuple]]:
    if scheme == "equal_mass":
        edges = np.quantile(prob, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(prob, edges[1:-1], right=False)
    rows, ece = [], 0.0
    N = len(prob)
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        conf = float(prob[sel].mean())
        acc = float(y_pos[sel].mean())
        ece += n / N * abs(acc - conf)
        rows.append((b, n, conf, acc, acc - conf))
    return ece, rows


def cmd_calibration(a: argparse.Namespace) -> int:
    cols = read_csv(a.csv)
    y = need(cols, a.label, "true label")
    prob = need(cols, a.prob, "predicted probability").astype(float)
    classes = np.unique(y)
    if len(classes) != 2:
        sys.exit("calibration here is binary; for multiclass, pass the "
                 "max-probability column and a correctness label")
    y_pos = (y == classes[1]).astype(float)
    if prob.min() < 0 or prob.max() > 1:
        sys.exit("probability column is outside [0,1]")

    brier = float(np.mean((prob - y_pos) ** 2))
    eps = 1e-12
    nll = float(-np.mean(y_pos * np.log(np.clip(prob, eps, 1)) +
                         (1 - y_pos) * np.log(np.clip(1 - prob, eps, 1))))

    print(f"items            : {len(prob):,}")
    print(f"base rate        : {y_pos.mean():.4f}")
    print(f"mean confidence  : {prob.mean():.4f}")
    print(f"Brier            : {brier:.4f}   (proper score, no binning)")
    print(f"NLL              : {nll:.4f}   (proper score, no binning)")

    for scheme in ("equal_width", "equal_mass"):
        ece, rows = _ece(y_pos, prob, a.bins, scheme)
        print(f"\nECE ({scheme}, {a.bins} bins): {ece:.4f}")
        if a.show_bins:
            print("  bin      n     conf      acc      gap")
            for b, n, conf, acc, gap in rows:
                print(f"  {b:>3} {n:>7,} {conf:>8.3f} {acc:>8.3f} {gap:>+8.3f}")

    # Bin-count sensitivity: ECE falls with fewer bins for reasons unrelated to
    # calibration, so a single reported value invites the objection.
    sens = [(k, _ece(y_pos, prob, k, "equal_mass")[0]) for k in (5, 10, 15, 20, 50)]
    print("\nECE vs bin count (equal mass): " +
          ", ".join(f"{k}:{v:.4f}" for k, v in sens))
    print("Report Brier or NLL alongside ECE. They need no binning and cannot be "
          "improved by choosing a favorable bin count.")
    return 0


def cmd_selective(a: argparse.Namespace) -> int:
    cols = read_csv(a.csv)
    y = need(cols, a.label, "true label")
    classes = np.unique(y)
    pred = to_hard(need(cols, a.pred, "prediction"), classes, a.threshold)
    conf = need(cols, a.confidence, "confidence score").astype(float)
    correct = (pred == y).astype(float)

    order = np.argsort(-conf, kind="mergesort")
    c = correct[order]
    n = len(c)
    cum_err = np.cumsum(1.0 - c)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cum_err / k
    aurc = float(np.trapezoid(risk, coverage)) if hasattr(np, "trapezoid") \
        else float(np.trapz(risk, coverage))

    full_risk = float(1.0 - correct.mean())
    # Optimal-ranking AURC for the same error count, i.e. all errors deferred last.
    n_err = int(round(cum_err[-1]))
    ideal = np.maximum(0.0, (k - (n - n_err)) / k)
    aurc_opt = float(np.trapezoid(ideal, coverage)) if hasattr(np, "trapezoid") \
        else float(np.trapz(ideal, coverage))

    print(f"items                 : {n:,}")
    print(f"risk at full coverage : {full_risk:.4f}")
    print(f"AURC                  : {aurc:.4f}")
    print(f"AURC (optimal ranking): {aurc_opt:.4f}")
    print(f"E-AURC (excess)       : {aurc - aurc_opt:.4f}   "
          f"(0 means the confidence ranking is perfect)")

    print("\ncoverage   risk    n_reviewed")
    for cov in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        i = max(1, int(round(cov * n))) - 1
        print(f"  {cov:>5.0%}  {risk[i]:>7.4f}  {i + 1:>10,}")

    ok = np.where(risk <= a.target_risk)[0]
    if ok.size:
        i = ok[-1]
        print(f"\nmax coverage at risk <= {a.target_risk:.1%}: {coverage[i]:.1%} "
              f"({i + 1:,} items auto-handled, {n - i - 1:,} deferred)")
        print("For a triage framing, state this as workload: the fraction of cases "
              "the system handles without review at a clinically acceptable error "
              "rate. That is the quantity a reader cares about, not AURC.")
    else:
        print(f"\nNo coverage level reaches risk <= {a.target_risk:.1%}. The system "
              f"cannot operate at that error rate; report the achievable rate "
              f"instead of the target.")

    if not a.group:
        print("\nNOTE: no --group given. Add it to bootstrap these operating points "
              "at the subject level; a coverage estimate without an interval is a "
              "point on one draw of the test set.")
    else:
        groups = need(cols, a.group, "grouping variable")
        rng = np.random.default_rng(a.seed)
        gids, gidx = np.unique(groups, return_inverse=True)
        W = multinomial_weights(len(gids), a.n_boot, rng)
        covs = []
        for b in range(a.n_boot):
            w = W[b][gidx][order]
            tot = w.sum()
            if tot == 0:
                continue
            cw = np.cumsum(w)
            ce = np.cumsum(w * (1.0 - c))
            with np.errstate(invalid="ignore", divide="ignore"):
                rb = np.where(cw > 0, ce / np.maximum(cw, 1e-12), 0.0)
            good = np.where(rb <= a.target_risk)[0]
            covs.append(cw[good[-1]] / tot if good.size else 0.0)
        covs = np.array(covs)
        lo, hi = percentile_ci(covs, a.alpha)
        print(f"\ncoverage at risk <= {a.target_risk:.1%}: "
              f"{np.mean(covs):.1%} [{lo:.1%}, {hi:.1%}] "
              f"(clustered bootstrap over {len(gids):,} groups)")
    return 0


def z_quantile(p: float) -> float:
    """Inverse standard normal CDF. scipy when present, else Acklam's
    rational approximation (relative error below 1.2e-9)."""
    if _sps is not None:
        return float(_sps.norm.ppf(p))
    if not 0.0 < p < 1.0:
        return float("nan")
    a_ = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b_ = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c_ = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d_ = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c_[0] * q + c_[1]) * q + c_[2]) * q + c_[3]) * q + c_[4]) * q + c_[5]) / \
               ((((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a_[0] * r + a_[1]) * r + a_[2]) * r + a_[3]) * r + a_[4]) * r + a_[5]) * q / \
               (((((b_[0] * r + b_[1]) * r + b_[2]) * r + b_[3]) * r + b_[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c_[0] * q + c_[1]) * q + c_[2]) * q + c_[3]) * q + c_[4]) * q + c_[5]) / \
           ((((d_[0] * q + d_[1]) * q + d_[2]) * q + d_[3]) * q + 1)


def paired_boot_diff(a: argparse.Namespace, y, groups, classes, preds, name_a: str, name_b: Optional[str]):
    """Bootstrap replicates of a metric for arm A, and of B minus A when B is
    given, under one shared group resample (the paired design)."""
    rng = np.random.default_rng(a.seed)
    gids, gidx = np.unique(groups, return_inverse=True)
    W = multinomial_weights(len(gids), a.n_boot, rng)
    if a.metric == "auroc":
        y_pos = (y == classes[1])

        def pack(s):
            order = np.argsort(s, kind="mergesort")
            _, block = np.unique(s[order], return_inverse=True)
            return order, block, int(block.max()) + 1

        pa = pack(preds[name_a])
        ba = np.array([auroc_weighted(y_pos, *pa, W[i][gidx]) for i in range(a.n_boot)])
        bb = None
        if name_b:
            pb = pack(preds[name_b])
            bb = np.array([auroc_weighted(y_pos, *pb, W[i][gidx]) for i in range(a.n_boot)])
    else:
        ga = GroupStats(y, to_hard(preds[name_a], classes, a.threshold), groups)
        ba = ga.metric(a.metric, W)
        bb = None
        if name_b:
            gb = GroupStats(y, to_hard(preds[name_b], classes, a.threshold), groups)
            bb = gb.metric(a.metric, W)
    return ba, bb, len(gids)


def cmd_mde(a: argparse.Namespace) -> int:
    """Minimum detectable effect for a paired comparison at this sample size:
    (z_{1-alpha/2} + z_{power}) times the bootstrap standard error of the
    paired difference. The standard error comes from the difference between
    two arms under one shared resample, never from an arm bootstrapped
    against itself: with a shared resample and the same signal on both sides
    the sampling variance cancels exactly, and what is left is tie-break
    noise, which understates the MDE by a large factor."""
    cols, y, groups, classes, preds = _prep(a, [a.a] + ([a.b] if a.b else []))
    if a.metric == "auroc" and len(classes) != 2:
        sys.exit("auroc requires a binary label column")
    ba, bb, n_groups = paired_boot_diff(a, y, groups, classes, preds, a.a, a.b)
    if bb is not None:
        se = float(np.nanstd(bb - ba, ddof=1))
        basis = f"paired difference {a.b} minus {a.a}, one shared group resample"
    else:
        se = float(np.nanstd(ba, ddof=1))
        basis = (f"marginal SE of {a.a} (no --b given); the paired SE is usually smaller, "
                 f"so this MDE is conservative")
    z_a = z_quantile(1 - a.alpha / 2)
    z_b = z_quantile(a.power)
    mde = (z_a + z_b) * se
    print(f"metric           : {a.metric}")
    print(f"SE basis         : {basis}")
    print(f"bootstrap SE     : {se:.5f}  ({a.n_boot:,} resamples over {n_groups:,} groups)")
    print(f"alpha, power     : {a.alpha}, {a.power}   (z = {z_a:.3f} + {z_b:.3f})")
    print(f"MDE              : {mde:.4f}")
    print(f"items            : {len(y):,}    groups: {n_groups:,}")
    print(design_effect(len(y), n_groups))
    print("\nAn interval that contains zero from a design whose MDE is larger than the "
          "smallest effect worth reporting is 'not distinguishable at this sample size', "
          "not evidence of equivalence. State the MDE beside every null.")
    return 0


def self_test() -> int:
    import contextlib
    import io
    import tempfile
    ok = True

    def check(desc: str, cond: bool) -> None:
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + desc)
        ok = ok and cond

    def run(fn, **kw) -> str:
        buf = io.StringIO()
        ns = argparse.Namespace(**kw)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            fn(ns)
        return buf.getvalue()

    def grab(text: str, key: str) -> List[float]:
        line = next(ln for ln in text.splitlines() if ln.startswith(key))
        return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line.split(":", 1)[1].replace(",", ""))]

    import re
    rng = np.random.default_rng(1)
    n_pat, per = 120, 5
    pid = np.repeat(np.arange(n_pat), per)
    y = rng.integers(0, 2, size=n_pat * per)
    good = np.where(rng.random(len(y)) < 0.85, y, 1 - y)          # ~85 percent accuracy
    same = good.copy()
    worse = np.where(rng.random(len(y)) < 0.75, y, 1 - y)          # ~75 percent
    prob = np.where(y == 1, 0.9, 0.1)                              # near-perfect calibration
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/p.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y", "pid", "good", "same", "worse", "prob"])
            for i in range(len(y)):
                w.writerow([y[i], pid[i], good[i], same[i], worse[i], prob[i]])
        base = dict(csv=path, label="y", group="pid", n_boot=400, alpha=0.05, seed=0, threshold=0.5)
        out = run(cmd_ci, pred="good", metric="accuracy", **base)
        lo, hi = grab(out, "95% CI")[:2]
        pt = grab(out, "point estimate")[0]
        check("ci: interval brackets the point estimate near 0.85", lo <= pt <= hi and 0.78 < pt < 0.92)
        out = run(cmd_compare, a="good", b="same", metric="accuracy", **base)
        lo, hi = grab(out, "95% CI of diff")[:2]
        check("compare: identical arms give a zero difference with an interval containing zero", lo <= 0 <= hi and abs(grab(out, "paired difference")[0]) < 1e-9)
        out = run(cmd_compare, a="worse", b="good", metric="accuracy", **base)
        lo, hi = grab(out, "95% CI of diff")[:2]
        check("compare: a real 10-point gap excludes zero", lo > 0)
        out = run(cmd_mde, a="worse", b="good", metric="accuracy", power=0.8, **base)
        mde = grab(out, "MDE")[0]
        se = grab(out, "bootstrap SE")[0]
        check("mde: (1.96 + 0.84) * SE, positive, from the paired difference", abs(mde - 2.8016 * se) < 0.002 and 0 < mde < 0.2)
        out2 = run(cmd_mde, a="good", b="same", metric="accuracy", power=0.8, **base)
        check("mde: identical arms give SE 0 (the paired SE cancels only when the arms really are identical)", grab(out2, "bootstrap SE")[0] < 1e-9)
        out3 = run(cmd_mde, a="good", b=None, metric="accuracy", power=0.8, **base)
        check("mde: single arm reports the marginal basis and a positive MDE", "marginal" in out3 and grab(out3, "MDE")[0] > 0)
        out = run(cmd_mcnemar, csv=path, label="y", a="good", b="same", threshold=0.5, exact=False)
        check("mcnemar: identical arms, no discordant pairs", "0" in out.split("A correct, B wrong")[1].split("\n")[0])
        prob_cal = rng.random(4000)
        y_cal = rng.random(4000) < prob_cal
        ece, _ = _ece(y_cal, prob_cal, 10, "equal_width")
        ece_bad, _ = _ece(y_cal, np.clip(prob_cal * 0.5, 0, 1), 10, "equal_width")
        check("calibration: labels drawn from the probabilities give a small ECE, halved probabilities a large one", ece < 0.05 < ece_bad)
        out = run(cmd_holm, p="0.004,0.031,0.048,0.220", alpha=0.05)
        check("holm: smallest p adjusted by family size, largest by one", "0.016" in out and "0.22" in out)
        check("z quantile: 0.975 -> 1.96, 0.8 -> 0.8416", abs(z_quantile(0.975) - 1.95996) < 1e-4 and abs(z_quantile(0.8) - 0.84162) < 1e-4)
    print("\nself-test " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_holm(a: argparse.Namespace) -> int:
    ps = [float(x) for x in a.p.split(",") if x.strip()]
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * ps[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(1.0, running)
    print(f"family size: {m}   alpha: {a.alpha}")
    print("  raw p      adjusted   reject")
    for i, p in enumerate(ps):
        print(f"  {p:<10.4g} {adj[i]:<10.4g} {'yes' if adj[i] <= a.alpha else 'no'}")
    print("\nThe family must be the set of comparisons declared in advance. "
          "Correcting only the tests that were reported, after dropping the ones "
          "that failed, does not control anything.")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, group_default=True):
        q.add_argument("--csv", required=True)
        q.add_argument("--label", required=True, help="true label column")
        q.add_argument("--group", default="", help="subject/patient column to resample")
        q.add_argument("--n-boot", type=int, default=2000)
        q.add_argument("--alpha", type=float, default=0.05)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--threshold", type=float, default=0.5,
                       help="threshold if predictions are binary scores")

    q = sub.add_parser("ci", help="clustered bootstrap interval for one model")
    common(q)
    q.add_argument("--pred", required=True)
    q.add_argument("--metric", default="balanced_accuracy",
                   choices=sorted(RATIO_METRICS | {"auroc"}))
    q.set_defaults(func=cmd_ci)

    q = sub.add_parser("compare", help="paired clustered bootstrap difference")
    common(q)
    q.add_argument("--a", required=True, help="reference model column")
    q.add_argument("--b", required=True, help="treatment model column")
    q.add_argument("--metric", default="balanced_accuracy",
                   choices=sorted(RATIO_METRICS | {"auroc"}))
    q.set_defaults(func=cmd_compare)

    q = sub.add_parser("mcnemar", help="paired test on correctness")
    q.add_argument("--csv", required=True)
    q.add_argument("--label", required=True)
    q.add_argument("--a", required=True)
    q.add_argument("--b", required=True)
    q.add_argument("--threshold", type=float, default=0.5)
    q.add_argument("--exact", action="store_true")
    q.set_defaults(func=cmd_mcnemar)

    q = sub.add_parser("delong", help="AUROC comparison, independent items only")
    q.add_argument("--csv", required=True)
    q.add_argument("--label", required=True)
    q.add_argument("--a", required=True)
    q.add_argument("--b", required=True)
    q.set_defaults(func=cmd_delong)

    q = sub.add_parser("calibration", help="ECE, Brier, NLL, reliability bins")
    q.add_argument("--csv", required=True)
    q.add_argument("--label", required=True)
    q.add_argument("--prob", required=True)
    q.add_argument("--bins", type=int, default=15)
    q.add_argument("--show-bins", action="store_true")
    q.set_defaults(func=cmd_calibration)

    q = sub.add_parser("selective", help="risk-coverage, AURC, operating points")
    common(q)
    q.add_argument("--pred", required=True)
    q.add_argument("--confidence", required=True)
    q.add_argument("--target-risk", type=float, default=0.05)
    q.set_defaults(func=cmd_selective)

    q = sub.add_parser("holm", help="Holm-Bonferroni across a family")
    q.add_argument("--p", required=True, help="comma separated p-values")
    q.add_argument("--alpha", type=float, default=0.05)
    q.set_defaults(func=cmd_holm)

    q = sub.add_parser("mde", help="minimum detectable effect for a paired comparison at this sample size")
    common(q)
    q.add_argument("--a", required=True, help="reference model column")
    q.add_argument("--b", default=None, help="treatment model column; omit for a single-arm (marginal, conservative) MDE")
    q.add_argument("--metric", default="balanced_accuracy",
                   choices=sorted(RATIO_METRICS | {"auroc"}))
    q.add_argument("--power", type=float, default=0.8)
    q.set_defaults(func=cmd_mde)

    q = sub.add_parser("self-test", help="check the estimators on synthetic data (no files needed)")
    q.set_defaults(func=lambda a: self_test())

    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()
    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
