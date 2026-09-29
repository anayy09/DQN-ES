"""
Host description for timing reports and run manifests (plan E4, E13).

Avoids the `platform` module: on Windows with Python 3.12+ platform.system()
and platform.processor() can block for ~2 min on a WMI query.
"""

from __future__ import annotations

import os
import sys


def cpu_name() -> str:
    if sys.platform.startswith('win'):
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            name, _ = winreg.QueryValueEx(key, 'ProcessorNameString')
            return str(name).strip()
        except OSError:
            return os.environ.get('PROCESSOR_IDENTIFIER', 'unknown')
    if sys.platform == 'darwin':
        try:
            import subprocess
            return subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        except Exception:
            return 'unknown'
    try:
        with open('/proc/cpuinfo', encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return 'unknown'


def host_info() -> dict:
    import numpy as np
    return {
        'cpu': cpu_name(),
        'logical_cpus': os.cpu_count(),
        'os': sys.platform,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
    }


def dqn_cost(state_dim: int, n_actions: int, hidden: int, batch: int,
             replay_capacity: int) -> dict:
    """
    Analytic size and cost of the two-hidden-layer NumPy DQN.
      params         weights + biases of one network
      forward_flops  2 x multiply-adds of one forward pass (activations ignored)
      update_flops   one minibatch update ~ online forward + target forward +
                     backward (~2 x forward) = 4 x batch x forward_flops
      memory_bytes   online + target networks (float64) + full replay buffer
                     (two states + action + reward + done per transition)
    """
    macs = state_dim * hidden + hidden * hidden + hidden * n_actions
    params = macs + 2 * hidden + n_actions
    fwd = 2 * macs
    return {
        'state_dim': state_dim,
        'n_actions': n_actions,
        'params': params,
        'forward_flops': fwd,
        'update_flops_approx': 4 * batch * fwd,
        'memory_bytes_approx': 2 * params * 8 + replay_capacity * (2 * state_dim + 3) * 8,
    }
