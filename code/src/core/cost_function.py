"""
Multi-objective cost function for bio-inspired task offloading.

Mathematical model:
  F(x) = ŵ_E · Ê + ŵ_L · L̂ + ŵ_P · R_P

where weights are CI-adaptive (Criticality Index Φ ∈ [0,1]):
  w_E(Φ) = exp(-α_E · Φ)                          — energy weight decays with CI
  w_L(Φ) = (exp(β_L·Φ) - 1) / (exp(β_L) - 1)    — latency weight grows with CI
  w_P(Φ) = (1-Φ)^γ_P                              — privacy relaxed in emergencies

Energy model: CMOS dynamic power  E = κ · C · f²
Latency model: L = t_tx + t_prop + t_queue + t_proc + t_dl
  (t_dl = return propagation + result download; see core/offload_model.py)
Privacy risk: R_P = ρ · (1 - H(u_i) / H_max)
"""

import math
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CI-to-weight mapping constants
# ---------------------------------------------------------------------------
ALPHA_E: float = 3.0   # energy weight decay rate
BETA_L: float = 4.0    # latency weight growth rate
GAMMA_P: float = 2.0   # privacy weight decay exponent


# ---------------------------------------------------------------------------
# CI-adaptive weight functions
# ---------------------------------------------------------------------------

def weight_energy(ci: float) -> float:
    """
    w_E(Φ) = exp(-α_E · Φ)

    Energy conservation is least important when CI is high
    (emergency tasks must be processed regardless of battery cost).
    """
    return math.exp(-ALPHA_E * ci)


def weight_latency(ci: float) -> float:
    """
    w_L(Φ) = (exp(β_L·Φ) - 1) / (exp(β_L) - 1)

    Normalised exponential growth — latency urgency rises sharply with CI.
    At Φ=0: w_L=0; at Φ=1: w_L=1.
    """
    numerator = math.exp(BETA_L * ci) - 1.0
    denominator = math.exp(BETA_L) - 1.0
    return numerator / denominator


def weight_privacy(ci: float) -> float:
    """
    w_P(Φ) = (1 - Φ)^γ_P

    Privacy constraints are relaxed in clinical emergencies (high CI).
    At Φ=0: w_P=1; at Φ=1: w_P=0.
    """
    return (1.0 - ci) ** GAMMA_P


def compute_normalized_weights(ci: float) -> Tuple[float, float, float]:
    """
    Return (ŵ_E, ŵ_L, ŵ_P) normalised so that ŵ_E + ŵ_L + ŵ_P = 1.

    Dispatches to the currently active weight mode (default: 'nonlinear',
    the paper's proposed CI-adaptive non-linear scheme).  Mode can be
    overridden globally via `set_weight_mode()` for the weight ablation.
    """
    ci = float(max(0.0, min(1.0, ci)))
    mode = WEIGHT_MODE
    if mode == 'flat':
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    if mode == 'step':
        if ci > 0.5:
            return 0.1, 0.8, 0.1
        return 0.7, 0.15, 0.15
    if mode == 'linear':
        wl = ci
        we = (1.0 - ci) / 2.0
        wp = (1.0 - ci) / 2.0
        total = we + wl + wp
        return we / total, wl / total, wp / total
    # default: 'nonlinear' (proposed)
    we = weight_energy(ci)
    wl = weight_latency(ci)
    wp = weight_privacy(ci)
    total = we + wl + wp
    if total < 1e-12:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return we / total, wl / total, wp / total


# ---------------------------------------------------------------------------
# Weight-mode switch  (ablation of CI-adaptive weight design)
# ---------------------------------------------------------------------------
WEIGHT_MODE: str = 'nonlinear'   # 'flat' | 'step' | 'linear' | 'nonlinear'


def set_weight_mode(mode: str) -> None:
    """
    Switch the global CI-weight scheme.  Allowed values:
      'nonlinear' — proposed exponential / power weights (paper's design)
      'flat'      — equal 1/3 weights regardless of CI
      'step'      — piecewise constant at CI=0.5 threshold
      'linear'    — w_L=Φ, w_E=w_P=(1-Φ)/2  (then normalised)
    """
    global WEIGHT_MODE
    if mode not in ('flat', 'step', 'linear', 'nonlinear'):
        raise ValueError(f"Unknown weight mode: {mode}")
    WEIGHT_MODE = mode


def get_weight_mode() -> str:
    return WEIGHT_MODE


# ---------------------------------------------------------------------------
# Latency models
# ---------------------------------------------------------------------------

def compute_local_latency(cpu_cycles: int, cpu_freq_hz: float) -> float:
    """
    Local processing latency: L_local = C_i / f  [seconds]

    Parameters
    ----------
    cpu_cycles : int    — C_i, required instruction count
    cpu_freq_hz : float — f, processor clock frequency in Hz

    Returns
    -------
    Latency in seconds.
    """
    if cpu_freq_hz <= 0:
        return float('inf')
    return cpu_cycles / cpu_freq_hz


def compute_offload_latency(
    data_size_bits: int,
    cpu_cycles: int,
    uplink_rate_bps: float,
    propagation_delay_s: float,
    queue_delay_s: float,
    node_cpu_freq_hz: float,
) -> float:
    """
    Total offloading latency:
      L_off = t_tx + t_prop + t_queue + t_proc

    where:
      t_tx   = D_i / R         (transmission time)
      t_prop = propagation delay (distance / signal speed)
      t_queue = M/M/1 queue sojourn time at remote node
      t_proc = C_i / f_node    (remote processing time)

    Parameters
    ----------
    data_size_bits   : int   — D_i in bits
    cpu_cycles       : int   — C_i
    uplink_rate_bps  : float — R in bits/s (Shannon capacity)
    propagation_delay_s : float — t_prop
    queue_delay_s    : float — t_queue (M/M/1 sojourn)
    node_cpu_freq_hz : float — f_node

    Returns
    -------
    Total offload latency in seconds.
    """
    if uplink_rate_bps <= 0:
        return float('inf')
    if node_cpu_freq_hz <= 0:
        return float('inf')

    t_tx = data_size_bits / uplink_rate_bps
    t_proc = cpu_cycles / node_cpu_freq_hz
    return t_tx + propagation_delay_s + queue_delay_s + t_proc


# ---------------------------------------------------------------------------
# Energy models
# ---------------------------------------------------------------------------

def compute_local_energy(
    cpu_cycles: int,
    cpu_freq_hz: float,
    kappa: float,
) -> float:
    """
    CMOS dynamic energy model:
      E_local = κ · C_i · f²

    Parameters
    ----------
    cpu_cycles  : int   — C_i
    cpu_freq_hz : float — f (Hz)
    kappa       : float — effective switched capacitance (F·cycle^{-2})

    Returns
    -------
    Energy in joules.
    """
    return kappa * cpu_cycles * (cpu_freq_hz ** 2)


def compute_local_energy_draw(
    cpu_cycles: int,
    cpu_freq_hz: float,
    active_power_w: float,
) -> float:
    """
    Battery energy of local execution from a measured supply draw:
      E_local = P_active * C_i / f

    The wearable uses this with the ESP32-S3 datasheet current
    (config.WEARABLE_COMPUTE_DRAW_W); the CMOS kappa model above is kept only
    for profiles without a datasheet draw.
    """
    if cpu_freq_hz <= 0:
        return float('inf')
    return active_power_w * cpu_cycles / cpu_freq_hz


def compute_offload_energy(
    data_size_bits: int,
    uplink_rate_bps: float,
    total_latency_s: float,
    tx_power_w: float,
    idle_power_w: float,
    rx_time_s: float = 0.0,
    rx_power_w: float = 0.0,
) -> float:
    """
    Wearable energy during offloading:
      E_off = P_tx · t_tx + P_rx · t_rx + P_idle · (L_off - t_tx - t_rx)

    The wearable transmits the payload at TX power, waits in idle/listening
    mode, and receives the result at RX power (t_rx = result download time;
    0 reproduces the earlier model without result return).

    Parameters
    ----------
    data_size_bits  : int   — D_i in bits
    uplink_rate_bps : float — R in bits/s
    total_latency_s : float — L_off (end-to-end)
    tx_power_w      : float — P_tx (wearable battery draw while transmitting;
                              not the radiated power of the rate equation)
    idle_power_w    : float — P_idle (wearable during waiting)
    rx_time_s       : float — t_rx, time spent receiving the result
    rx_power_w      : float — P_rx (wearable receive power)

    Returns
    -------
    Energy in joules.
    """
    if uplink_rate_bps <= 0:
        return float('inf')

    t_tx = min(data_size_bits / uplink_rate_bps, total_latency_s)
    t_rx = min(max(0.0, rx_time_s), max(0.0, total_latency_s - t_tx))
    t_idle = max(0.0, total_latency_s - t_tx - t_rx)
    return tx_power_w * t_tx + rx_power_w * t_rx + idle_power_w * t_idle


# ---------------------------------------------------------------------------
# Privacy risk model
# ---------------------------------------------------------------------------

def compute_privacy_entropy(offload_counts: Dict[int, int]) -> float:
    """
    Shannon entropy over offloading distribution:
      H(u_i) = -Σ_j  p_{i,j} · log2(p_{i,j})

    where p_{i,j} = count_j / total — fraction of tasks sent to node j.

    Parameters
    ----------
    offload_counts : dict mapping node_id → number of tasks sent there

    Returns
    -------
    Entropy in bits.  Returns 0 if total count is 0.
    """
    total = sum(offload_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in offload_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def compute_privacy_risk(
    privacy_sensitivity: float,
    offload_counts: Dict[int, int],
    n_available_nodes: int,
) -> float:
    """
    Privacy risk:
      R_P = ρ_i · (1 - H(u_i) / H_max)

    Higher entropy (more distributed offloading) → lower risk.
    If traffic is concentrated on one node, risk is maximised.

    Parameters
    ----------
    privacy_sensitivity : float — ρ_i ∈ [0, 1]
    offload_counts      : dict  — {node_id: task_count}
    n_available_nodes   : int   — number of nodes (determines H_max)

    Returns
    -------
    Privacy risk ∈ [0, 1].
    """
    h = compute_privacy_entropy(offload_counts)
    # n_available_nodes passed here is the number of remote candidate nodes
    # We restrict entropy calculation to network-facing decisions
    total_destinations = n_available_nodes
    h_max = math.log2(total_destinations) if total_destinations > 1 else 1.0
    concentration = 1.0 - h / max(h_max, 1e-12)
    return float(privacy_sensitivity * max(0.0, min(1.0, concentration)))


# ---------------------------------------------------------------------------
# Full multi-objective cost
# ---------------------------------------------------------------------------

def compute_cost(
    latency_s: float,
    energy_j: float,
    privacy_risk: float,
    ci: float,
    latency_bounds: Tuple[float, float],
    energy_bounds: Tuple[float, float],
    privacy_scale: float = 1.0,
) -> float:
    """
    Full normalised multi-objective cost:
      F(x) = ŵ_E · Ê + ŵ_L · L̂ + ŵ_P · R_P

    Latency and energy are min-max normalised to [0, 1] using the
    supplied bounds (estimated from the feasible solution space).

    Parameters
    ----------
    latency_s       : float — end-to-end latency (local or offload)
    energy_j        : float — energy consumption in joules
    privacy_risk    : float — R_P ∈ [0, 1]
    ci              : float — Criticality Index Φ ∈ [0, 1]
    latency_bounds  : (l_min, l_max) in seconds
    energy_bounds   : (e_min, e_max) in joules
    privacy_scale   : float — lambda_P; the privacy weight is multiplied by
                      lambda_P and the three weights renormalised to sum 1
                      (reweighted-greedy frontier).  1 = default.

    Returns
    -------
    Scalar cost F(x) ∈ [0, 1] (approximately).
    """
    we, wl, wp = compute_normalized_weights(ci)
    if privacy_scale != 1.0:
        wp = wp * privacy_scale
        total = we + wl + wp
        if total > 1e-12:
            we, wl, wp = we / total, wl / total, wp / total

    l_min, l_max = latency_bounds
    e_min, e_max = energy_bounds

    lat_norm = (latency_s - l_min) / (l_max - l_min + 1e-10)
    eng_norm = (energy_j - e_min) / (e_max - e_min + 1e-10)

    # Clamp to [0, 1] (handles out-of-range inputs gracefully)
    lat_norm = float(max(0.0, min(1.0, lat_norm)))
    eng_norm = float(max(0.0, min(1.0, eng_norm)))

    return we * eng_norm + wl * lat_norm + wp * privacy_risk


def estimate_bounds(
    latencies: list,
    energies: list,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Helper: derive min-max normalisation bounds from a list of candidate values.
    Adds a 5 % margin to avoid degenerate [0,0] ranges.

    Returns
    -------
    (latency_bounds, energy_bounds) each as (min, max) tuples.
    """
    lat_arr = np.array(latencies, dtype=float)
    eng_arr = np.array(energies, dtype=float)

    lat_min, lat_max = float(lat_arr.min()), float(lat_arr.max())
    eng_min, eng_max = float(eng_arr.min()), float(eng_arr.max())

    margin_lat = max(0.05 * (lat_max - lat_min), 1e-6)
    margin_eng = max(0.05 * (eng_max - eng_min), 1e-12)

    return (
        (lat_min - margin_lat, lat_max + margin_lat),
        (eng_min - margin_eng, eng_max + margin_eng),
    )
