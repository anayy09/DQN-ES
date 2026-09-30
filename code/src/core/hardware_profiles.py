"""
Hardware profiles based on published datasheets.
ESP32-S3: https://documentation.espressif.com/esp32-s3_datasheet_en.pdf (v2.2)
Raspberry Pi 4: https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-datasheet.pdf
"""

from dataclasses import dataclass

from src.config import (
    WEARABLE_COMPUTE_DRAW_W,
    WEARABLE_CPU_FREQ_HZ,
    WEARABLE_TX_DRAW_W,
    WEARABLE_TX_RADIATED_W,
    WEARABLE_WAIT_DRAW_W,
)


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    cpu_freq_hz: float        # cycles per second
    kappa: float              # effective switched capacitance (F/cycle^2), CMOS model
    tx_power_w: float         # transmission power in watts (0 if not wireless transmitter)
    idle_power_w: float       # idle/listening power in watts
    ram_bytes: int            # available RAM
    bandwidth_hz: float       # channel bandwidth
    max_mips: float           # compute capacity in millions of instructions per second
    # Battery draw (W) of a node that runs on a battery (the wearable only);
    # 0 = not modelled.  tx_power_w above is the *radiated* power used by the
    # rate equation; these are the supply-side powers used by the energy terms.
    tx_draw_w: float = 0.0        # while transmitting
    active_power_w: float = 0.0   # while computing locally (E = P * C_i / f)


# ---------------------------------------------------------------------------
# ESP32-S3: 240 MHz dual-core Xtensa LX7, Wi-Fi 802.11b/g/n
# ESP32-S3 Series Datasheet v2.2 (2026-03-05); values and table numbers in
# src/config.py ("Wearable power model"):
#   - radiated TX 18.5 dBm (Table 6-2, 802.11n HT20 MCS7), rate equation only
#   - TX draw 283 mA x 3.3 V (Table 5-7), RX 88 mA x 3.3 V (Table 5-7)
#   - waiting: modem-sleep 240 MHz WAITI 47.6 mA x 3.3 V (Table 5-9)
#   - local compute: 65.9 mA x 3.3 V x C_i / f (Table 5-9, 240 MHz, one core)
#   - channel bandwidth: 20 MHz (802.11n HT20)
# kappa is not used for the wearable (the datasheet gives the draw directly).
# ---------------------------------------------------------------------------
WEARABLE_ESP32 = HardwareProfile(
    name="ESP32-S3 Wearable",
    cpu_freq_hz=WEARABLE_CPU_FREQ_HZ,
    kappa=0.0,                # unused: local energy = active_power_w * C_i / f
    tx_power_w=WEARABLE_TX_RADIATED_W,    # radiated, 18.5 dBm = 70.8 mW
    idle_power_w=WEARABLE_WAIT_DRAW_W,    # 47.6 mA x 3.3 V = 157 mW
    ram_bytes=512 * 1024,     # 512 KB SRAM (datasheet p. 5)
    bandwidth_hz=20e6,        # 20 MHz Wi-Fi HT20 channel
    max_mips=240.0,           # single-core effective MIPS ≈ clock (simple IPC=1 model)
    tx_draw_w=WEARABLE_TX_DRAW_W,         # 283 mA x 3.3 V = 934 mW
    active_power_w=WEARABLE_COMPUTE_DRAW_W,  # 65.9 mA x 3.3 V = 217 mW
)

# ---------------------------------------------------------------------------
# Raspberry Pi 4 Model B: 1.5 GHz Cortex-A72, BCM2711
# Datasheet / measured power:
#   - Idle draw: ~3.4 W (measured, RPi Foundation)
#   - Gateway receives; wearable transmits → tx_power_w = 0
#   - 5G NR channel (via USB dongle): 100 MHz bandwidth
# ---------------------------------------------------------------------------
EDGE_GATEWAY_RPI4 = HardwareProfile(
    name="Raspberry Pi 4 Edge Gateway",
    cpu_freq_hz=1500e6,
    kappa=1e-28,              # CMOS model, kept: no datasheet draw; not in any reported metric
    tx_power_w=0.0,           # gateway is the receiver; wearable pays TX energy
    idle_power_w=3.4,         # measured idle draw at 3.4 W (RPi4 with 8 GB)
    ram_bytes=8 * 1024 * 1024 * 1024,   # 8 GB LPDDR4
    bandwidth_hz=100e6,       # 5G NR sub-6 GHz, 100 MHz channel
    max_mips=1500.0,
)

# ---------------------------------------------------------------------------
# Mid-tier fog compute node: server-grade ARM or x86, ~2.2 GHz
# Modelled after an NVIDIA Jetson AGX Orin / small rack server
# ---------------------------------------------------------------------------
FOG_NODE = HardwareProfile(
    name="Fog Compute Node",
    cpu_freq_hz=2200e6,
    kappa=1e-28,              # CMOS model, kept: no datasheet draw; not in any reported metric
    tx_power_w=0.0,           # fog node is receiver
    idle_power_w=5.0,         # ~5 W idle for compact server
    ram_bytes=16 * 1024 * 1024 * 1024,  # 16 GB
    bandwidth_hz=100e6,       # 5G NR backhaul
    max_mips=2200.0,
)

# ---------------------------------------------------------------------------
# Cloud server: abstracted as unlimited capacity resource
# Energy not charged to the wearable (metered by cloud provider separately)
# Modelled as a 3.2 GHz instance with 1 Gbps NIC
# ---------------------------------------------------------------------------
CLOUD_SERVER = HardwareProfile(
    name="Cloud Server",
    cpu_freq_hz=3200e6,
    kappa=0.0,                # cloud energy not attributed to wearable budget
    tx_power_w=0.0,
    idle_power_w=0.0,         # abstracted; not charged to IoT device
    ram_bytes=int(1e12),      # effectively unlimited (1 TB)
    bandwidth_hz=1e9,         # 1 Gbps NIC
    max_mips=3200.0,
)
