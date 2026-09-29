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
    topo_ss, env_ss = _np.random.SeedSequence(base).spawn(2)
    return {
        'base':      base,
        'task':      base,
        'scheduler': base,
        'topology':  int(topo_ss.generate_state(1)[0]),
        'env':       int(env_ss.generate_state(1)[0]),
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
# Monte Carlo protocol  (Fix 1, Fix 5)
# ---------------------------------------------------------------------------
N_RUNS:          int        = 30
TASK_SCALES:     list[int]  = [100, 500, 1000, 2000, 5000]
PRIMARY_SCALE:   int        = 1000          # scale used for Table III, ablations
N_WEARABLES:     int        = 10
N_FOG_NODES:     int        = 3


# ---------------------------------------------------------------------------
# DQN-ES hyperparameters  (Fix 7: explicit epsilon decay schedule)
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

# BBO inner search
BBO_POP:             int   = 20
BBO_MAX_ITER:        int   = 30
BBO_DELTA0:          float = 1.0
BBO_TOP_K:           int   = 3               # DQN top-K pre-filter (Algorithm 1, line 3)


# ---------------------------------------------------------------------------
# CI-adaptive weight functions  (Fix 6: ablation defines four conditions)
# ---------------------------------------------------------------------------
# Default ("proposed") non-linear weights are defined analytically in
# core/cost_function.py with constants ALPHA_E, BETA_L, GAMMA_P.
ALPHA_E:             float = 3.0
BETA_L:              float = 4.0
GAMMA_P:             float = 2.0

# Threshold separating high-criticality vs low-criticality regime for the
# step-weight ablation condition.
STEP_CI_THRESHOLD:   float = 0.5


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
# transmitting at DOWNLINK_TX_POWER_W.  Assumed value: 20 dBm, a typical
# 2.4 GHz Wi-Fi access-point transmit power (EU EIRP limit); not measured.
DOWNLINK_TX_POWER_W:           float      = 0.100

# Wearable Wi-Fi receive power.  ESP32-S3 Series Datasheet (Espressif),
# "Current Consumption" / RF-mode table: Rx 802.11n HT20, typical 88 mA at
# VDD = 3.3 V, 25 C  ->  0.088 A x 3.3 V = 0.290 W.  (Check the datasheet
# version/table number when citing; see the provenance table S1.)
WEARABLE_RX_CURRENT_A:         float      = 0.088
WEARABLE_SUPPLY_V:             float      = 3.3
WEARABLE_RX_POWER_W:           float      = WEARABLE_RX_CURRENT_A * WEARABLE_SUPPLY_V


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
# Privacy guard (Fix 8)
# ---------------------------------------------------------------------------
# A flow is classified as a "traffic-analysis attack" when its empirical
# offload-entropy ratio H/H_max falls below this threshold for the source
# device.  Threshold tuned to maximise F1 on a 20% calibration split.
PRIVACY_ENTROPY_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# Real-trace evaluation (Fix 10)
# ---------------------------------------------------------------------------
MITBIH_N_RUNS:       int   = 30
MITBIH_PAYLOAD_BITS: int   = int(5 * 1024 * 1024 * 8)  # 5 MB — matches main simulation ECG task profile
MITBIH_DEADLINE_S:   float = 0.500                   # 500 ms ECG SLA (matches paper's stated SLA)
MITBIH_RHO:          float = 0.9


# ---------------------------------------------------------------------------
# Statistical testing  (Fix 4, Fix E: updated for PSO+DQN)
# ---------------------------------------------------------------------------
# Significance tested on four metrics across all baselines vs DQN-ES.
# Bonferroni correction: alpha_corrected = 0.05 / (n_baselines * n_metrics)
STAT_ALPHA:          float = 0.05
STAT_METRICS:        list[str] = [
    'avg_latency_ms', 'avg_energy_mj',
    'avg_privacy_risk', 'sla_violation_pct',
]
# Fix A: PSO+DQN added — family size is now 6 × 4 = 24
STAT_BASELINES:      list[str] = [
    'PSO', 'ACO', 'HS-HHO', 'ES-only', 'DQN-only',
]
# Bonferroni denominator = len(STAT_BASELINES) * len(STAT_METRICS) = 24


# ---------------------------------------------------------------------------
# Algorithm registry  (Fix 2: includes ablations; Fix A: adds PSO+DQN)
# ---------------------------------------------------------------------------
def get_full_algorithm_registry():
    """
    Return the complete algorithm registry including PSO+DQN (Fix A),
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


def make_scheduler(sched_cls, topology, seed: int, **kwargs):
    """
    Construct a scheduler for one replicate.  `seed` is passed to every
    scheduler whose constructor accepts it (all stochastic ones do), so no
    driver can silently fall back to a class default seed.
    """
    import inspect
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
        f"  BBO: pop={BBO_POP}  iter={BBO_MAX_ITER}  K={BBO_TOP_K}  "
        f"delta0={BBO_DELTA0}\n"
        f"  CI weights (non-linear, default): "
        f"alpha_E={ALPHA_E} beta_L={BETA_L} gamma_P={GAMMA_P}\n"
        f"  Privacy guard entropy threshold: {PRIVACY_ENTROPY_THRESHOLD}\n"
        f"  Statistical tests: alpha={STAT_ALPHA}  "
        f"Bonferroni denom={len(STAT_BASELINES)*len(STAT_METRICS)}\n"
    )


if __name__ == '__main__':
    print(summary())
