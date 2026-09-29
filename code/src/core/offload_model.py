"""
Per-(task, destination) latency and wearable-energy model.

Single implementation used by the scheduler cost (BaseScheduler.evaluate_node),
the normalisation bounds (BaseScheduler.estimate_feasible_bounds) and the
realised metrics (OffloadingEnvironment.step), so every scheduler optimises
and is scored against the same model.

Local execution (destination = the task's own wearable):
    L = C_i / f_local,            E = kappa C_i f_local^2
Offload to node j:
    L = t_tx + t_prop + t_queue + t_proc + t_dl
        t_tx   = D_i / R_ul                      (uplink)
        t_prop = d / v                           (one way)
        t_queue: M/M/1 waiting time at j
        t_proc = C_i / f_j
        t_dl   = t_prop + S_res / R_dl           (result return; plan E9)
    E = P_tx t_tx + P_rx (S_res / R_dl) + P_idle (L - t_tx - S_res / R_dl)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import (
    DOWNLINK_TX_POWER_W,
    RESULT_SIZE_BITS,
    WEARABLE_RX_POWER_W,
)
from src.core.cost_function import (
    compute_local_energy,
    compute_local_latency,
    compute_offload_energy,
)


@dataclass
class OffloadOutcome:
    latency_s: float
    energy_j: float
    is_local: bool
    # latency components (seconds)
    t_tx: float = 0.0
    t_prop: float = 0.0
    t_queue: float = 0.0
    t_proc: float = 0.0
    t_dl: float = 0.0          # return propagation + result download
    # wearable energy components (joules)
    e_tx: float = 0.0
    e_idle: float = 0.0
    e_rx: float = 0.0
    e_compute: float = 0.0     # local execution only


def result_size_bits(task) -> int:
    size = getattr(task, 'result_size_bits', None)
    return RESULT_SIZE_BITS if size is None else int(size)


def offload_outcome(topology, task, node_id: int,
                    latency_cap_s: Optional[float] = None) -> OffloadOutcome:
    """
    Latency and wearable energy of running `task` on `node_id`.

    latency_cap_s caps L before the idle-energy term is computed (the
    environment uses 999 s to keep an overloaded queue finite).
    """
    src = topology.get_node(task.device_id)

    if node_id == task.device_id:
        lat = compute_local_latency(task.cpu_cycles, src.hardware.cpu_freq_hz)
        if latency_cap_s is not None:
            lat = min(lat, latency_cap_s)
        eng = compute_local_energy(task.cpu_cycles, src.hardware.cpu_freq_hz,
                                   src.hardware.kappa)
        return OffloadOutcome(latency_s=lat, energy_j=eng, is_local=True,
                              t_proc=lat, e_compute=eng)

    dst = topology.get_node(node_id)
    r_ul = topology.get_uplink_rate(task.device_id, node_id)
    t_prop = topology.get_propagation_delay(task.device_id, node_id)
    t_queue = topology.get_queue_delay(node_id)
    if r_ul <= 0 or dst.hardware.cpu_freq_hz <= 0:
        inf = float('inf')
        return OffloadOutcome(latency_s=inf, energy_j=inf, is_local=False)

    t_tx = task.data_size_bits / r_ul
    t_proc = task.cpu_cycles / dst.hardware.cpu_freq_hz

    s_res = result_size_bits(task)
    if s_res > 0:
        r_dl = topology.get_downlink_rate(task.device_id, node_id,
                                          DOWNLINK_TX_POWER_W)
        t_rx = s_res / r_dl
        t_dl = t_prop + t_rx
    else:
        t_rx = 0.0
        t_dl = 0.0

    lat = t_tx + t_prop + t_queue + t_proc + t_dl
    if latency_cap_s is not None:
        lat = min(lat, latency_cap_s)

    hw = src.hardware
    eng = compute_offload_energy(task.data_size_bits, r_ul, lat,
                                 hw.tx_power_w, hw.idle_power_w,
                                 rx_time_s=t_rx, rx_power_w=WEARABLE_RX_POWER_W)
    # Components, clipped exactly as in compute_offload_energy
    tx_time = min(t_tx, lat)
    rx_time = min(t_rx, max(0.0, lat - tx_time))
    e_tx = hw.tx_power_w * tx_time
    e_rx = WEARABLE_RX_POWER_W * rx_time
    e_idle = hw.idle_power_w * max(0.0, lat - tx_time - rx_time)

    return OffloadOutcome(latency_s=lat, energy_j=eng, is_local=False,
                          t_tx=t_tx, t_prop=t_prop, t_queue=t_queue,
                          t_proc=t_proc, t_dl=t_dl,
                          e_tx=e_tx, e_idle=e_idle, e_rx=e_rx)
