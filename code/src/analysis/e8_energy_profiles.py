"""
e8_energy_profiles.py — wearable energy under alternative ESP32-S3 power
profiles, recomputed post hoc from the per-task raw logs (plan E8).

The simulator logs, per task, the wearable's energy by phase under the model
powers (energy_tx_mj, energy_rx_mj, energy_idle_mj, energy_compute_mj).  The
time spent in each phase is recovered exactly as energy / model power, and
the energy is recomputed under each profile:

  E = P_tx t_tx + P_rx t_rx + P_wait t_wait + E_wake (per offloaded task)
      + E_compute (local execution, CMOS model, unchanged)

Profiles (power = current x 3.3 V).  Current values are from the ESP32-S3
Series Datasheet (Espressif), RF current-consumption and low-power-mode
tables; verify the datasheet version and table numbers before citing them
(parameter provenance table S1).  Wake-up energy is an assumption, not a
datasheet value.

  model          the simulator's powers: TX 178 mW (22.5 dBm output power,
                 used as the drawn power), RX 88 mA, wait 10 mA (33 mW)
  radio_on_wait  TX 340 mA (802.11b 1 Mbps, 21 dBm), RX 88 mA
                 (802.11n HT20), waiting with the receiver on (88 mA)
  light_sleep    TX 283 mA (802.11n HT20 MCS7), RX 88 mA, waiting in light
                 sleep (240 uA); assumed wake-up of 1 ms at 66 mA per
                 offloaded task

Output: <out>/energy_profiles.csv  scheduler x profile: mean wearable energy
per task (mean over replicates, replicate-bootstrap 95 % CI) and the rank of
each scheduler within the profile.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

V = 3.3
PROFILES = {
    'model':         {'tx': None, 'rx': None, 'wait': None, 'wake_j': 0.0},
    'radio_on_wait': {'tx': 0.340 * V, 'rx': 0.088 * V, 'wait': 0.088 * V, 'wake_j': 0.0},
    'light_sleep':   {'tx': 0.283 * V, 'rx': 0.088 * V, 'wait': 240e-6 * V,
                      'wake_j': 0.066 * V * 1e-3},
}


def _model_powers():
    from src.config import WEARABLE_RX_POWER_W
    from src.core.hardware_profiles import WEARABLE_ESP32
    return {'tx': WEARABLE_ESP32.tx_power_w, 'rx': WEARABLE_RX_POWER_W,
            'wait': WEARABLE_ESP32.idle_power_w}


def task_energy_mj(row: dict, prof: dict, model: dict) -> float:
    e_tx, e_rx = float(row['energy_tx_mj']), float(row['energy_rx_mj'])
    e_idle, e_cmp = float(row['energy_idle_mj']), float(row['energy_compute_mj'])
    if row['assigned_node'] == row['device_id']:          # local execution
        return e_cmp
    t_tx, t_rx, t_wait = e_tx / model['tx'], e_rx / model['rx'], e_idle / model['wait']
    p = {k: (model[k] if prof[k] is None else prof[k]) for k in ('tx', 'rx', 'wait')}
    return p['tx'] * t_tx + p['rx'] * t_rx + p['wait'] * t_wait + prof['wake_j'] * 1e3


def run(raw_dirs, out_dir: Path, algorithms=None) -> list:
    from src.analysis.e5_adversary import load_logs, mean_bootstrap_ci
    from src.config import STAT_BOOT_N, STAT_BOOT_SEED, STAT_CI_LEVEL
    from src.simulation.episode_log import _long_path
    logs = load_logs([Path(d) for d in raw_dirs], algorithms)
    model = _model_powers()
    rows = []
    for pname, prof in PROFILES.items():
        per_alg = {}
        for alg, runs in sorted(logs.items()):
            per_run = [np.mean([task_energy_mj(r, prof, model) for r in rs])
                       for _, rs in sorted(runs.items())]
            lo, hi = mean_bootstrap_ci(per_run, STAT_BOOT_N, STAT_CI_LEVEL,
                                       STAT_BOOT_SEED)
            per_alg[alg] = (float(np.mean(per_run)), lo, hi, len(per_run))
        order = sorted(per_alg, key=lambda a: per_alg[a][0])
        for alg, (m, lo, hi, n) in per_alg.items():
            rows.append({'profile': pname, 'scheduler': alg, 'n_runs': n,
                         'energy_mj_mean': m, 'ci_lo': lo, 'ci_hi': hi,
                         'rank_lowest_first': order.index(alg) + 1})
    out_dir = Path(out_dir)
    _long_path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(_long_path(out_dir / 'energy_profiles.csv'), 'w', newline='',
              encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(f"  {r['profile']:<14} {r['scheduler']:<16} {r['energy_mj_mean']:9.3f} mJ "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] rank {r['rank_lowest_first']}")
    return rows


def main():
    p = argparse.ArgumentParser(description='E8 energy-profile sensitivity')
    p.add_argument('--raw-dirs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--algorithms', nargs='+', default=None)
    a = p.parse_args()
    run(a.raw_dirs, Path(a.out), a.algorithms)


if __name__ == '__main__':
    main()
