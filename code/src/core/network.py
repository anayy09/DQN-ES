"""
IoT-Edge-Cloud network model.

Models:
  - Shannon capacity for uplink rates
  - M/M/1 queueing for node queue delays
  - Propagation delay based on physical distance

References:
  - Goldsmith, A. (2005). Wireless Communications. Cambridge University Press.
  - Gross, D. et al. (2008). Fundamentals of Queueing Theory. Wiley.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.hardware_profiles import HardwareProfile


# ---------------------------------------------------------------------------
# Default channel parameters
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_FIBER = 2e8       # m/s (signal speed in optical fiber / copper)
BOLTZMANN_K = 1.38e-23           # J/K
TEMPERATURE_K = 290.0            # standard temperature (290 K â‰ˆ 17 Â°C)
REFERENCE_DISTANCE_M = 1.0       # d_0 for path-loss reference
REFERENCE_GAIN_H0 = 1.0          # h_0 at d_0 (unit gain reference)
AVG_CPU_CYCLES_PER_TASK = 6_000_000  # used in M/M/1 Î¼ computation


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NetworkNode:
    """Represents a compute/communication node in the topology."""
    node_id: int
    name: str
    node_type: str            # 'wearable' | 'edge' | 'fog' | 'cloud'
    hardware: HardwareProfile
    position_km: Tuple[float, float] = (0.0, 0.0)   # (x, y) in kilometres
    current_load: int = 0     # number of tasks currently in queue
    arrival_rate: float = 0.0  # Î» â€” tasks/second arriving at this node


@dataclass
class NetworkLink:
    """Directed link from source to destination node."""
    source_id: int
    dest_id: int
    distance_km: float
    path_loss_exponent: float = 3.0   # Î±: 2.0 LOS, 3.5 NLOS, 3.0 typical indoor

    @property
    def distance_m(self) -> float:
        return self.distance_km * 1000.0


class NetworkTopology:
    """
    Manages nodes and directed links; provides delay/rate query methods.
    """

    def __init__(self):
        self.nodes: Dict[int, NetworkNode] = {}
        self.links: Dict[Tuple[int, int], NetworkLink] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: NetworkNode) -> None:
        self.nodes[node.node_id] = node

    def add_link(self, link: NetworkLink) -> None:
        self.links[(link.source_id, link.dest_id)] = link

    def get_node(self, node_id: int) -> NetworkNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not in topology.")
        return self.nodes[node_id]

    def get_link(self, src_id: int, dst_id: int) -> NetworkLink:
        """
        Return the NetworkLink for (src_id, dst_id).
        Falls back to a default long-haul link (50 km, Î±=2.0) for
        wearableâ†’cloud paths not explicitly registered (e.g. multi-hop).
        """
        if (src_id, dst_id) in self.links:
            return self.links[(src_id, dst_id)]
        # Default: treat as long-haul WAN / cloud link
        return NetworkLink(
            source_id=src_id,
            dest_id=dst_id,
            distance_km=50.0,
            path_loss_exponent=2.0,
        )

    # ------------------------------------------------------------------
    # Physical-layer uplink rate (Shannon capacity)
    # ------------------------------------------------------------------

    def get_uplink_rate(
        self,
        src_id: int,
        dst_id: int,
        channel_noise_dbm: float = -100.0,
    ) -> float:
        """
        Shannon capacity R = B Â· log2(1 + SNR)  [bits/s]

        Path-loss channel model:
            h = h_0 Â· (d_0 / d)^Î±
        where h_0 = 1 (0 dB at d_0 = 1 m), Î± = link.path_loss_exponent.

        SNR = P_tx Â· h / ÏƒÂ²   (no interference: I = 0)

        If src node has tx_power_w = 0 (receiver node), we use the
        destination node's bandwidth and a nominal SNR representing
        wired / fibre backhaul (very high SNR).
        """
        src_node = self.get_node(src_id)
        dst_node = self.get_node(dst_id)
        link = self.get_link(src_id, dst_id)

        # Bandwidth: use src node's channel bandwidth (transmitter sets channel)
        B = src_node.hardware.bandwidth_hz

        p_tx = src_node.hardware.tx_power_w

        if p_tx <= 0.0:
            # Wired / fibre link â€” use destination bandwidth, assume high SNR
            B = dst_node.hardware.bandwidth_hz
            # Assume 30 dB SNR for wired link
            snr = 1000.0
            return B * math.log2(1.0 + snr)

        return self._shannon_rate(p_tx, B, link, channel_noise_dbm)

    @staticmethod
    def _shannon_rate(
        p_tx: float,
        B: float,
        link: 'NetworkLink',
        channel_noise_dbm: float = -100.0,
    ) -> float:
        """R = B log2(1 + P h / sigma^2) over `link` (path-loss, I = 0)."""
        # Noise power: sigma^2 = k T B (thermal floor) or the given noise
        # floor, whichever is higher, in watts
        noise_thermal_w = BOLTZMANN_K * TEMPERATURE_K * B
        noise_dbm_w = 10.0 ** ((channel_noise_dbm - 30.0) / 10.0)
        sigma_sq = max(noise_thermal_w, noise_dbm_w)

        # Path-loss: h = (d_0/d)^alpha  (free-space + log-distance model)
        d_m = max(link.distance_m, REFERENCE_DISTANCE_M)   # avoid d=0
        alpha = link.path_loss_exponent
        h = REFERENCE_GAIN_H0 * (REFERENCE_DISTANCE_M / d_m) ** alpha

        snr = (p_tx * h) / sigma_sq
        snr = max(snr, 1e-9)   # floor to avoid log(0)

        rate_bps = B * math.log2(1.0 + snr)
        return max(rate_bps, 1e3)   # at least 1 kbps to prevent division errors

    def get_downlink_rate(
        self,
        device_id: int,
        node_id: int,
        tx_power_w: float,
        channel_noise_dbm: float = -100.0,
    ) -> float:
        """
        Result-return rate node -> wearable, from the same channel model as
        the uplink: the wearable's registered link to `node_id` is used as a
        reciprocal channel (same distance and path-loss exponent), on the
        wearable's channel bandwidth, with the serving side transmitting at
        `tx_power_w`.
        """
        wearable = self.get_node(device_id)
        link = self.get_link(device_id, node_id)
        return self._shannon_rate(tx_power_w, wearable.hardware.bandwidth_hz,
                                  link, channel_noise_dbm)

    # ------------------------------------------------------------------
    # M/M/1 queue delay
    # ------------------------------------------------------------------

    def get_queue_delay(
        self,
        node_id: int,
        avg_cpu_cycles: float = AVG_CPU_CYCLES_PER_TASK,
    ) -> float:
        """
        M/M/1 queueing delay W = Î» / (Î¼ Â· (Î¼ - Î»))  seconds.

        Î¼ (service rate) = (max_mips Ã— 10^6) / avg_cpu_cycles  [tasks/s]
        Î» (arrival rate) = node.arrival_rate  [tasks/s]

        Returns 999.0 if the node is overloaded (Î» â‰¥ Î¼).
        """
        node = self.get_node(node_id)

        # Service rate: tasks per second based on CPU capacity
        mu = (node.hardware.max_mips * 1e6) / max(avg_cpu_cycles, 1)

        lam = node.arrival_rate

        if lam <= 0.0:
            return 0.0   # no load — no queue delay

        if lam >= mu:
            return 999.0  # overloaded / unstable queue

        # M/M/1 mean waiting time in queue: W_q = ρ / (μ(1-ρ))
        # The service time (t_proc) is added separately in compute_offload_latency.
        rho = lam / mu
        W_q = rho / (mu * (1.0 - rho))
        return W_q

    # ------------------------------------------------------------------
    # Propagation delay
    # ------------------------------------------------------------------

    def get_propagation_delay(self, src_id: int, dst_id: int) -> float:
        """
        Propagation delay = distance / signal_speed

        Uses SPEED_OF_LIGHT_FIBER (2Ã—10^8 m/s) for both wireless and
        wired segments as a conservative average.
        """
        link = self.get_link(src_id, dst_id)
        return link.distance_m / SPEED_OF_LIGHT_FIBER

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_compute_nodes(self) -> List[NetworkNode]:
        """Return all non-wearable nodes that can execute tasks."""
        return [n for n in self.nodes.values() if n.node_type != 'wearable']

    def get_all_candidate_nodes(self, include_wearable: bool = True) -> List[int]:
        """Return sorted list of all node IDs available for scheduling."""
        if include_wearable:
            return sorted(self.nodes.keys())
        return sorted(n.node_id for n in self.nodes.values() if n.node_type != 'wearable')

    def update_load(self, node_id: int, delta: int = 1) -> None:
        """Increment (or decrement) queue load for a node."""
        self.nodes[node_id].current_load = max(
            0, self.nodes[node_id].current_load + delta
        )

    def update_arrival_rate(self, node_id: int, rate: float) -> None:
        """Update the estimated arrival rate Î» for M/M/1 computation."""
        self.nodes[node_id].arrival_rate = max(0.0, rate)

    def __repr__(self) -> str:
        return (
            f"NetworkTopology("
            f"nodes={len(self.nodes)}, links={len(self.links)})"
        )

