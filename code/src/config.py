"""
Central hyperparameter configuration for the Q1 DQN-ES paper.

All Monte Carlo runs, ablations, and statistical analyses import from
this single file so that the experimental protocol is reproducible and
unambiguous.

Hyperparameter values are also surfaced in Table II of the manuscript
and in the supplementary complexity_analysis.md.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED:      int = 42


def replicate_seed(run_id: int, n_tasks: int = 0) -> int:
    """
    Base seed of replicate r at episode length N:  s = 42 + 1000*r + N.

    Every driver derives all of a replicate's randomness from this one value,
    so replicate r is identical (tasks, topology, environment, scheduler
    seed) across algorithms and the per-run comparison is paired.
    Trace-driven drivers (MIT-BIH) pass n_tasks=0.
    """
    return GLOBAL_SEED + run_id * 1000 + n_tasks


def replicate_seeds(run_id: int, n_tasks: int = 0) -> dict:
    """
    Per-stream seeds for one replicate, all derived from replicate_seed().

      task       base seed (task generator; unchanged from the round-1 rule)
      scheduler  base seed (passed as seed= to every stochastic scheduler)
      topology   independent stream: device/fog placement per replicate
      env        independent stream: attack-burst draws in the environment

    topology and env are spawned with numpy SeedSequence so that they do not
    replay the task generator's random.Random(base) stream.  (With a shared
    seed the environment's burst draws reused the task generator's CI-tier
    draws, so every burst landed on a high-CI task.)
    """
    import numpy as _np
    base = replicate_seed(run_id, n_tasks)
    # Child i of a SeedSequence depends only on i, so adding streams never
    # changes the existing ones.
    kids = _np.random.SeedSequence(base).spawn(5)
    s = lambda k: int(kids[k].generate_state(1)[0])
    return {
        'base':      base,
        'task':      base,
        'scheduler': base,
        'topology':  s(0),
        'env':       s(1),
        'arrival':   s(2),   # MMPP-2 state/arrival draws (E7)
        'ci_noise':  s(3),   # scheduler-visible CI perturbation (E14)
        'pretrain':  s(4),   # warm-start pre-training stream (E12)
    }


def seed_global_rngs(seed: int) -> None:
    """Seed the process-global `random` and `numpy.random` states."""
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed % (2 ** 32))


# Legacy alias (round-1 name); prefer replicate_seed().
PER_RUN_SEED_FN = replicate_seed


# ---------------------------------------------------------------------------
# Monte Carlo protocol
# ---------------------------------------------------------------------------
N_RUNS:          int        = 30
TASK_SCALES:     list[int]  = [100, 500, 1000, 2000, 5000]
PRIMARY_SCALE:   int        = 1000          # scale used for Table III, ablations
N_WEARABLES:     int        = 10
N_FOG_NODES:     int        = 3


# ---------------------------------------------------------------------------
# DQN-ES hyperparameters (explicit epsilon decay schedule)
# ---------------------------------------------------------------------------
# Epsilon-greedy exploration schedule:
#   epsilon(t+1) = max(epsilon_min, epsilon(t) * epsilon_decay)
# i.e., a discrete-step geometric (exponential) decay applied once per
# scheduling decision.  Closed-form: epsilon(t) = epsilon_0 * decay^t
# (clipped at epsilon_min).
EPSILON_INIT:        float = 1.0
EPSILON_DECAY:       float = 0.995          # per scheduling decision
EPSILON_MIN:         float = 0.05
# Pre-computed convergence task counts under this schedule
# (epsilon_0 * decay^t = target  ->  t = log(target/epsilon_0) / log(decay))
#   t(eps=0.10) = log(0.10 / 1.0) / log(0.995) = 460
#   t(eps=0.05) = log(0.05 / 1.0) / log(0.995) = 598
EPSILON_T_AT_0_10:   int   = 460
EPSILON_T_AT_0_05:   int   = 598

# DQN architecture / training
DQN_HIDDEN_DIM:      int   = 64
DQN_LR:              float = 0.001
DQN_GAMMA:           float = 0.95
DQN_BATCH_SIZE:      int   = 32
DQN_REPLAY_CAPACITY: int   = 10_000
DQN_TARGET_SYNC:     int   = 50              # steps between target-net updates

# DQN top-K candidate set.  K ranges over 1..(M+2) network destinations
# (edge, M fog nodes, cloud); local execution is not an action.
DQN_TOP_K:           int   = 3


# ---------------------------------------------------------------------------
# Decomposition experiments (plan E1-E3; experiment registry below)
# ---------------------------------------------------------------------------
K_SWEEP:             list[int]   = [1, 2, 3, 4, 5]                   # E1 DQN-ES K-sweep
RANDOM_K:            int         = DQN_TOP_K                          # E2 Random-K subset size
Q_MIX_SWEEP:         list[float] = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]   # E2 q-mixed
LAMBDA_P_SWEEP:      list[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]  # E3 ES-only privacy scale

# E4 scalability: fog-node counts M (action set = M + 2 network
# destinations; DQN state dim = 2 + 4 (M + 2)).  Decision-time split is
# measured serially (no concurrent workers) on SCALING_TIMING_RUNS replicates.
SCALING_FOG_COUNTS:  list[int]   = [3, 8, 16, 32]
SCALING_TIMING_RUNS: int         = 5
ROBUSTNESS_ARMS:     list[str]   = ['DQN-ES', 'ES-only', 'Random-K[K=3]',
                                    'q-mixed[q=0.5]', 'Static-Tier']   # E4/E7/E12/E14 arms


# ---------------------------------------------------------------------------
# CI-adaptive weight functions (the weight ablation compares four schemes)
# ---------------------------------------------------------------------------
# Default ("proposed") non-linear weights are defined analytically in
# core/cost_function.py with constants ALPHA_E, BETA_L, GAMMA_P.
ALPHA_E:             float = 3.0
BETA_L:              float = 4.0
GAMMA_P:             float = 2.0

# Threshold separating high-criticality vs low-criticality regime for the
# step-weight ablation condition.
STEP_CI_THRESHOLD:   float = 0.5

# CI tier labels for logs and the acuity adversary (plan E5).  Matches the
# synthetic generator's tiers: low [0, 0.3), medium [0.3, 0.7), high [0.7, 1].
CI_TIER_BOUNDS:      tuple = (0.3, 0.7)


def ci_tier(ci: float) -> str:
    lo, hi = CI_TIER_BOUNDS
    return 'low' if ci < lo else ('medium' if ci < hi else 'high')


# ---------------------------------------------------------------------------
# ECG payload  (D15; single source for core/task.py TASK_PROFILES, the
# synthetic generator and the MIT-BIH trace)
# ---------------------------------------------------------------------------
# Main configuration: a heavy 5 MB (SI) payload per ECG analysis task.
# Sensitivity: 10 KB, the order of one raw 10 s, 360 Hz MIT-BIH window
# (2 leads x 11 bit x 3600 samples ~ 9.9 KB).
ECG_PAYLOAD_BITS:             int = 40_000_000     # 5 MB
ECG_PAYLOAD_SENSITIVITY_BITS: int = 80_000         # 10 KB


# ---------------------------------------------------------------------------
# Result return / downlink  (plan E9, D7; core/offload_model.py)
# ---------------------------------------------------------------------------
# Every offloaded task returns a result (class label, confidence, timestamp)
# to the wearable.  Latency adds the return propagation delay plus
# RESULT_SIZE_BITS / R_dl; the wearable pays RX power for the download.
RESULT_SIZE_BITS:              int        = 8_000          # 1 KB (SI) result
RESULT_SIZE_SENSITIVITY_BITS:  list[int]  = [8_000, 32_000, 128_000, 512_000]  # 1, 4, 16, 64 KB

# Downlink rate uses the uplink channel model on the wearable's registered
# link (reciprocal channel, wearable's 20 MHz channel) with the serving side
# transmitting at DOWNLINK_TX_POWER_W.  Assumed value, set at the regulatory
# maximum: ETSI EN 300 328 V2.2.2 (2019-07) cl. 4.3.2.2.3 limits 2.4 GHz
# wideband (non-FHSS) equipment to 20 dBm e.i.r.p.; not measured.
DOWNLINK_TX_POWER_W:           float      = 0.100


# ---------------------------------------------------------------------------
# Wearable power model (G1-3; core/hardware_profiles.py WEARABLE_ESP32)
# ---------------------------------------------------------------------------
# All values from the Espressif ESP32-S3 Series Datasheet, Version 2.2
# (2026-03-05).  Radiated power (the rate equation) and battery draw (the
# energy terms) are separate constants.  Draw = datasheet current x supply.

# Supply: Table 5-2 (Recommended Operating Conditions), VDD typ 3.3 V
# (3.0-3.6 V); the section 5.6.1 currents are measured at 3.3 V, 25 C.
WEARABLE_SUPPLY_V:             float      = 3.3

# Radiated TX power (uplink SNR only): Table 6-2, 802.11n HT20 MCS7, typ
# 18.5 dBm -- the HT20 output whose current Table 5-7 gives.  Below the
# ETSI EN 300 328 cl. 4.3.2.2.3 bound of 20 dBm e.i.r.p.
WEARABLE_TX_RADIATED_DBM:      float      = 18.5
WEARABLE_TX_RADIATED_W:        float      = 10.0 ** (WEARABLE_TX_RADIATED_DBM / 10.0) / 1000.0

# Battery draw while transmitting: Table 5-7, TX 802.11n HT20 MCS7 @ 18.5 dBm,
# 283 mA (peak, 100 % duty cycle).
WEARABLE_TX_CURRENT_A:         float      = 0.283
WEARABLE_TX_DRAW_W:            float      = WEARABLE_TX_CURRENT_A * WEARABLE_SUPPLY_V

# Battery draw while receiving the result: Table 5-7, RX 802.11b/g/n HT20,
# 88 mA.
WEARABLE_RX_CURRENT_A:         float      = 0.088
WEARABLE_RX_POWER_W:           float      = WEARABLE_RX_CURRENT_A * WEARABLE_SUPPLY_V

# Battery draw while awaiting the result: Table 5-9 (Modem-sleep), 240 MHz,
# WAITI (dual core idle), Typ2 = all peripheral clocks enabled, 47.6 mA.
WEARABLE_WAIT_CURRENT_A:       float      = 0.0476
WEARABLE_WAIT_DRAW_W:          float      = WEARABLE_WAIT_CURRENT_A * WEARABLE_SUPPLY_V

# Battery draw during local computation: Table 5-9, 240 MHz, single core
# running 32-bit data access instructions, other core idle, Typ2 (peripheral
# clocks enabled), 65.9 mA.  Local energy = draw x C_i / f_w (the single-core
# latency model); the CMOS kappa model is not used for the wearable.
WEARABLE_COMPUTE_CURRENT_A:    float      = 0.0659
WEARABLE_COMPUTE_DRAW_W:       float      = WEARABLE_COMPUTE_CURRENT_A * WEARABLE_SUPPLY_V

# CPU clock: datasheet p. 5, "up to 240 MHz".
WEARABLE_CPU_FREQ_HZ:          float      = 240e6


# ---------------------------------------------------------------------------
# Workload realism (plan E7; data_ingestion/event_generator.py)
# ---------------------------------------------------------------------------
# Default arrivals: Poisson with rate N / 300 s.  MMPP-2 option: two states
# with rate multipliers MMPP2_RATE_MULTIPLIERS (renormalised so the long-run
# rate equals the Poisson rate); after each arrival the state switches with
# probability MMPP2_SWITCH_PROBS[state] (low->high, high->low).  Assumed
# values, chosen to give short bursts at ~4x the mean rate.
MMPP2_RATE_MULTIPLIERS: tuple = (0.5, 4.0)
MMPP2_SWITCH_PROBS:     tuple = (0.02, 0.10)
# Load knob: target *offered* edge utilisation rho = lambda * E[C_i] / f_edge
# (the utilisation if every task ran on the edge), which sets the absolute
# arrival rate lambda independent of N.  The achieved utilisation depends on
# the scheduler and is reported per run (edge_utilisation).
LOAD_RHO_TARGETS:       list  = [0.3, 0.6, 0.85]

# Warm start (plan E12): pre-train DQN schedulers on N_pre synthetic tasks
# from the replicate's disjoint 'pretrain' seed stream before evaluation.
WARM_START_SWEEP:       list  = [500, 2000]

# CI noise (plan E14): perturbs only the Phi the scheduler sees; logs and
# labels keep the true Phi.  Gaussian sigma (clipped to [0, 1]) and a
# tier-misclassification probability (Phi redrawn uniformly inside one of
# the other two CI_TIER_BOUNDS tiers).  Misclassification is applied first
# when both are set.
CI_NOISE_SIGMAS:        list  = [0.05, 0.1, 0.2]
CI_MISCLASS_PROBS:      list  = [0.1, 0.2]


# ---------------------------------------------------------------------------
# Adversarial attack bursts (environment.py)
# ---------------------------------------------------------------------------
# On every task arrival the environment draws an attack burst with
# probability ATTACK_BURST_PROB; during a burst the task's p_atk is set to
# ATTACK_BURST_INTENSITY (otherwise it keeps its generated value, 0 for
# synthetic and MIT-BIH tasks).  p_atk enters only the DQN state vector.
# Applies to every scheduler and every run; set ATTACK_BURST_PROB = 0 to
# disable.
ATTACK_BURST_PROB:      float = 0.05
ATTACK_BURST_INTENSITY: float = 0.8


# ---------------------------------------------------------------------------
# Privacy guard (entropy threshold)
# ---------------------------------------------------------------------------
# A flow is classified as a "traffic-analysis attack" when its empirical
# offload-entropy ratio H/H_max falls below this threshold for the source
# device.  Threshold tuned to maximise F1 on a 20% calibration split.
PRIVACY_ENTROPY_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# MIT-BIH trace-driven evaluation
# ---------------------------------------------------------------------------
MITBIH_N_RUNS:       int   = 30
MITBIH_PAYLOAD_BITS: int   = None  # set below to ECG_PAYLOAD_BITS (single source, D15)
MITBIH_DEADLINE_S:   float = 0.500                   # 500 ms ECG SLA (matches paper's stated SLA)
MITBIH_RHO:          float = 0.9
MITBIH_PAYLOAD_BITS = ECG_PAYLOAD_BITS


# ---------------------------------------------------------------------------
# Statistical testing  (plan E10; analysis/statistical_tests.py, paired_stats.py)
# ---------------------------------------------------------------------------
# Unit = replicate (per-run mean); replicates are paired by run_id because
# every algorithm sees the same seeds.  Test: Wilcoxon signed-rank on the
# paired differences d = reference - comparator, exact p; Holm over each
# family declared here (before the runs); effect sizes: Hodges-Lehmann
# paired difference with a replicate-bootstrap CI and matched-pairs
# rank-biserial r.  Cohen's d is not reported.
STAT_ALPHA:          float = 0.05
STAT_METRICS:        list[str] = [
    'avg_latency_ms', 'avg_energy_mj',
    'avg_privacy_risk', 'sla_violation_pct',
]
STAT_BASELINES:      list[str] = [
    'PSO', 'ACO', 'HS-HHO', 'ES-only', 'DQN-only',
]
STAT_BOOT_N:         int   = 10_000
STAT_BOOT_SEED:      int   = GLOBAL_SEED
STAT_CI_LEVEL:       float = 0.95

# Declared comparison families.  Holm is applied within a family across all
# (comparator x metric) tests.  PSO+DQN is not tested: at K=3 it makes the
# same decisions as DQN-ES by construction (reported as "identical").
STAT_FAMILIES: dict = {
    'main': {                                   # 5 x 4 = 20 tests
        'reference':   'DQN-ES',
        'comparators': STAT_BASELINES,
        'metrics':     STAT_METRICS,
    },
    'decomposition': {                          # E2: 2 x 4 = 8 tests
        'reference':   'DQN-ES',
        'comparators': ['Random-K[K=3]', 'Static-Tier'],
        'metrics':     STAT_METRICS,
    },
    'weight_ablation': {                        # E15: 3 x 4 = 12 tests
        'reference':   'nonlinear',
        'comparators': ['flat', 'step', 'linear'],
        'metrics':     STAT_METRICS,
    },
    'privacy_inference': {                      # E5 / D17(b): 3 x 1 = 3 tests
        'reference':   'DQN-ES',
        'comparators': ['ES-only', 'Random-K[K=3]', 'Static-Tier'],
        'metrics':     ['adversary_auc'],       # per test replicate (analysis/e5_adversary.py)
    },
}

# E5 adversary (analysis/e5_adversary.py), task-level (ruling D20: the
# generator draws CI tiers i.i.d. per task, so a window-majority label never
# occurs).  Primary configuration for the 'privacy_inference' family,
# declared before the freeze: one sample per task, label = the task's true
# CI tier is 'high'; features = the task's destination, the device's
# histogram over its preceding `context` destinations, inter-arrival since
# the device's previous task, response time and payload size;
# HistGradientBoosting; train on replicates 0-19, test on 20-29
# (replicate-disjoint); the unit is the per-test-replicate AUC.
E5_ADVERSARY: dict = {
    'unit':               'task',
    'context':            50,
    'train_runs':         list(range(0, 20)),
    'test_runs':          list(range(20, 30)),
    'primary_features':   'dest+timing+size',
    'primary_classifier': 'hgb',
    'seed':               GLOBAL_SEED,
}

# D17(a): E2/E3 primary statistic, "privacy excess at matched latency".
# Per replicate r: sort the frontier's points (latency_r, R_P_r) by latency,
# interpolate R_P linearly at DQN-ES's latency_r, and take
# excess_r = R_P(DQN-ES)_r - R_P(frontier at latency_r).  A replicate whose
# DQN-ES latency lies outside that replicate's frontier latency range is not
# extrapolated: it is excluded and the count is reported.  Report the median
# of excess_r with a replicate-bootstrap CI (STAT_BOOT_N, STAT_CI_LEVEL),
# for all-task and for steady-state R_P.  Negative = DQN-ES below the frontier.
MATCHED_LATENCY: dict = {
    'reference': 'DQN-ES',
    'x_metric':  'avg_latency_ms',
    'y_metrics': ['avg_privacy_risk', 'avg_privacy_risk_ss'],
    'frontiers': {
        'q_mixed':  [f'q-mixed[q={q:g}]' for q in Q_MIX_SWEEP],
        'lambda_p': [f'ES-only[lP={lam:g}]' for lam in LAMBDA_P_SWEEP],
    },
    'scale':     PRIMARY_SCALE,
}

# TOST (equivalence) only where equivalence is claimed; off by default.
# Margins are declared here, before the runs, in the metric's units.
STAT_TOST_ENABLED:   bool  = False
STAT_TOST_MARGINS:   dict  = {
    'avg_privacy_risk':  0.01,
    'avg_latency_ms':    2.0,
}


# ---------------------------------------------------------------------------
# Algorithm registries
# ---------------------------------------------------------------------------
def get_full_algorithm_registry():
    """
    Return the main 9-algorithm registry, including PSO+DQN,
    ES-only and DQN-only ablations.
    Imported lazily to avoid circular imports at module load.
    """
    from src.algorithms.aco import ACOScheduler
    from src.algorithms.dqn_es import DQNESScheduler
    from src.algorithms.es_only import ESOnlyScheduler
    from src.algorithms.cloud_only import CloudOnlyScheduler
    from src.algorithms.dqn_only import DQNOnlyScheduler
    from src.algorithms.hs_hho import HSHHOScheduler
    from src.algorithms.local_only import LocalOnlyScheduler
    from src.algorithms.pso import PSOScheduler
    from src.algorithms.pso_dqn import PSODQNScheduler

    return {
        'DQN-ES':     DQNESScheduler,
        'PSO+DQN':    PSODQNScheduler,
        'ES-only':    ESOnlyScheduler,
        'DQN-only':   DQNOnlyScheduler,
        'PSO':        PSOScheduler,
        'ACO':        ACOScheduler,
        'HS-HHO':     HSHHOScheduler,
        'Local-Only': LocalOnlyScheduler,
        'Cloud-Only': CloudOnlyScheduler,
    }


def get_experiment_registry():
    """
    Decomposition arms (plan E1-E3), kept out of the main comparison table.
    Values are functools.partial(SchedulerClass, **params); make_scheduler()
    and the drivers accept them wherever a class is accepted.

      DQN-ES[K=k]       E1 K-sweep (K=3 is DQN-ES, K=1 is DQN-only's policy,
                        K=5 enumerates every destination = ES-only decisions)
      Random-K[K=k]     E2 random K-subset + argmin F, no learning
      q-mixed[q=..]     E2 random K-subset w.p. q, else full enumeration
      Static-Tier       E2 ECG -> edge, all other task types local
      ES-only[lP=..]    E3 reweighted greedy (privacy weight x lambda_P)
    """
    from functools import partial
    from src.algorithms.dqn_es import DQNESScheduler
    from src.algorithms.es_only import ESOnlyScheduler
    from src.algorithms.random_k import QMixedScheduler, RandomKScheduler
    from src.algorithms.static_tier import StaticTierScheduler

    reg = {}
    for k in K_SWEEP:
        reg[f'DQN-ES[K={k}]'] = partial(DQNESScheduler, n_candidate_nodes=k)
    reg[f'Random-K[K={RANDOM_K}]'] = partial(RandomKScheduler,
                                             n_candidate_nodes=RANDOM_K)
    for q in Q_MIX_SWEEP:
        reg[f'q-mixed[q={q:g}]'] = partial(QMixedScheduler, q=q,
                                           n_candidate_nodes=RANDOM_K)
    reg['Static-Tier'] = StaticTierScheduler
    for lam in LAMBDA_P_SWEEP:
        reg[f'ES-only[lP={lam:g}]'] = partial(ESOnlyScheduler,
                                              privacy_weight_scale=lam)
    return reg


def get_registry(name: str = 'main'):
    """'main' (9-algorithm comparison), 'experiments', or 'all'."""
    if name == 'main':
        return get_full_algorithm_registry()
    if name == 'experiments':
        return get_experiment_registry()
    if name == 'all':
        return {**get_full_algorithm_registry(), **get_experiment_registry()}
    raise ValueError(f'unknown registry {name!r}')


def make_scheduler(sched_cls, topology, seed: int, **kwargs):
    """
    Construct a scheduler for one replicate.  `seed` is passed to every
    scheduler whose constructor accepts it (all stochastic ones do), so no
    driver can silently fall back to a class default seed.  `sched_cls` may
    be a class or a functools.partial of one (experiment registry).
    """
    import functools
    import inspect
    if isinstance(sched_cls, functools.partial):
        kwargs = {**sched_cls.keywords, **kwargs}
        sched_cls = sched_cls.func
    params = inspect.signature(sched_cls.__init__).parameters
    if 'seed' in params:
        kwargs['seed'] = seed
    return sched_cls(topology, **kwargs)


def summary() -> str:
    """Return a human-readable summary of all hyperparameters."""
    return (
        f"DQN-ES Q1 hyperparameter summary\n"
        f"  Monte Carlo: N_RUNS={N_RUNS}  scales={TASK_SCALES}\n"
        f"  Epsilon: init={EPSILON_INIT} decay={EPSILON_DECAY} "
        f"min={EPSILON_MIN}\n"
        f"           eps<0.10 after {EPSILON_T_AT_0_10} tasks; "
        f"eps<0.05 after {EPSILON_T_AT_0_05} tasks\n"
        f"  DQN: hidden={DQN_HIDDEN_DIM}  lr={DQN_LR}  gamma={DQN_GAMMA}  "
        f"batch={DQN_BATCH_SIZE}\n"
        f"  DQN top-K: K={DQN_TOP_K}  K-sweep={K_SWEEP}\n"
        f"  Random-K={RANDOM_K}  q-mix={Q_MIX_SWEEP}  lambda_P={LAMBDA_P_SWEEP}\n"
        f"  CI weights (non-linear, default): "
        f"alpha_E={ALPHA_E} beta_L={BETA_L} gamma_P={GAMMA_P}\n"
        f"  Privacy guard entropy threshold: {PRIVACY_ENTROPY_THRESHOLD}\n"
        f"  Statistical tests: paired signed-rank (exact), Holm, alpha={STAT_ALPHA}; "
        f"families={ {k: len(v['comparators']) * len(v['metrics']) for k, v in STAT_FAMILIES.items()} }\n"
    )


if __name__ == '__main__':
    print(summary())
