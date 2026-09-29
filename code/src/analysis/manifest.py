"""
Run manifests and the declared-arm registry (experiment-ledger discipline).

Every driver invocation (one pipeline step, or a standalone CLI run) writes
<results>/manifests/<step>__<run_id>.json and appends one line to
<results>/manifests/runs.jsonl when it finishes.  A manifest pins:

  code         git commit, dirty paths (reported, not suppressed)
  config       sha256 of every resolved UPPER_CASE constant in src.config
               (canonical JSON), plus the step parameters and CLI
  seeds        replicate seed rule and the base seeds of the replicates run
  data         MIT-BIH file-list hash when the step reads the trace
  env          host CPU, Python and package versions
  determinism  what is and is not claimed
  timing       start / end (UTC) and wall seconds

run_id = sha256(canonical(step params + config hash) + commit)[:16]:
identical inputs collide, any change produces a new id.  Timestamps and
output paths are excluded from the id.

declare() writes <results>/manifests/declared_arms.json before a launch: every
step with its driver, arms, scales, replicates, condition and output
directory, plus the declared statistics (STAT_FAMILIES, MATCHED_LATENCY,
TOST setting).  verify() checks each declared step against the manifests and
the arms present in its summary.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

_CODE_DIR = Path(__file__).resolve().parents[2]
_ROOT = _CODE_DIR.parent


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=repr)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def git_state() -> dict:
    def run(*args):
        return subprocess.run(['git', *args], cwd=_ROOT, capture_output=True,
                              text=True).stdout
    try:
        commit = run('rev-parse', 'HEAD').strip()
        dirty = [line[3:] for line in run('status', '--porcelain',
                                          '--untracked-files=no').splitlines()
                 if line.strip()]
        tag = run('describe', '--tags', '--exact-match').strip() or None
    except OSError:
        commit, dirty, tag = 'unknown', [], None
    return {'git_commit': commit, 'git_tag': tag, 'git_dirty': bool(dirty),
            'git_dirty_paths': dirty}


def resolved_config() -> dict:
    import src.config as C
    out = {}
    for k in sorted(dir(C)):
        if k.isupper():
            v = getattr(C, k)
            if isinstance(v, (int, float, str, bool, list, tuple, dict, type(None))):
                out[k] = v
    return out


def config_hash() -> str:
    return _sha(_canonical(resolved_config()))


def _packages() -> dict:
    from importlib import metadata
    out = {}
    for p in ('numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'wfdb'):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def mitbih_data_hash(data_dir: Path) -> dict:
    d = Path(data_dir) / 'MIT-BIH-Arrhythmia'
    if not d.exists():
        return {'path': str(d), 'present': False}
    files = sorted((p.name, p.stat().st_size) for p in d.iterdir() if p.is_file())
    return {'path': str(d), 'present': True, 'n_files': len(files),
            'file_list_sha256': _sha(_canonical(files))}


def replicate_seed_list(n_runs: int, scales: Iterable[int]) -> dict:
    from src.config import replicate_seed
    return {str(n): [replicate_seed(r, n) for r in range(n_runs)] for n in scales}


class Manifest:
    """Context manager: writes the manifest on entry and completes it on exit."""

    def __init__(self, results_dir: Path, step: str, params: dict,
                 n_runs: Optional[int] = None, scales: Iterable[int] = (),
                 data_dir: Optional[Path] = None, outputs: Iterable[str] = ()):
        from src.analysis.machine_info import host_info
        from src.simulation.episode_log import _long_path
        self.dir = _long_path(Path(results_dir) / 'manifests')
        self.dir.mkdir(parents=True, exist_ok=True)
        git = git_state()
        cfg = config_hash()
        self.run_id = _sha(_canonical({'step': step, 'params': params,
                                       'config_sha256': cfg})
                           + git['git_commit'])[:16]
        self.path = self.dir / f'{step}__{self.run_id}.json'
        self.m = {
            'run_id': self.run_id,
            'step': step,
            'code': {**git, 'entrypoint': ' '.join(sys.argv)},
            'config': {'resolved_sha256': cfg, 'params': params},
            'seeds': {
                'rule': 'replicate_seed(r, N) = 42 + 1000 r + N for tasks and '
                        'scheduler; topology/env/arrival/ci_noise/pretrain '
                        'streams = SeedSequence(replicate_seed(r, N)).spawn(5)',
                'base_seeds': (replicate_seed_list(n_runs, scales)
                               if n_runs else {}),
            },
            'data': mitbih_data_hash(data_dir) if data_dir else {},
            'env': {**host_info(), 'packages': _packages()},
            'determinism': {
                'claimed_deterministic': True,
                'note': 'all RNGs seeded per replicate; bitwise equality is '
                        'expected on the same machine and library versions; '
                        'BLAS reduction order may differ across machines',
            },
            'outputs': list(outputs),
            'started_utc': None, 'finished_utc': None, 'wall_s': None,
            'status': 'declared',
        }

    def __enter__(self):
        self.m['started_utc'] = _utc()
        self.m['status'] = 'running'
        self._t0 = time.time()
        self._write()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.m['finished_utc'] = _utc()
        self.m['wall_s'] = round(time.time() - self._t0, 1)
        self.m['status'] = 'failed' if exc_type else 'finished'
        if exc_type:
            self.m['error'] = f'{exc_type.__name__}: {exc}'
        self._write()
        from src.simulation.episode_log import _long_path
        with open(_long_path(self.dir / 'runs.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write(_canonical({k: self.m[k] for k in
                                 ('run_id', 'step', 'status', 'started_utc',
                                  'finished_utc', 'wall_s')}
                                | {'git_commit': self.m['code']['git_commit'],
                                   'config_sha256': self.m['config']['resolved_sha256']})
                     + '\n')
        return False

    def _write(self):
        from src.simulation.episode_log import _long_path
        _long_path(self.path).write_text(json.dumps(self.m, indent=1, default=repr),
                             encoding='utf-8')


# --------------------------------------------------------------------------
# Declared-arm registry
# --------------------------------------------------------------------------

def declare(results_dir: Path, steps: list) -> Path:
    """Write the declaration of every step before launch."""
    from src.config import (MATCHED_LATENCY, STAT_FAMILIES, STAT_TOST_ENABLED,
                            STAT_TOST_MARGINS)
    from src.simulation.episode_log import _long_path
    d = _long_path(Path(results_dir) / 'manifests')
    d.mkdir(parents=True, exist_ok=True)
    doc = {
        'declared_utc': _utc(),
        'code': git_state(),
        'config_sha256': config_hash(),
        'steps': steps,
        'statistics': {
            'families': STAT_FAMILIES,
            'matched_latency': MATCHED_LATENCY,
            'tost_enabled': STAT_TOST_ENABLED,
            'tost_margins': STAT_TOST_MARGINS,
        },
    }
    path = _long_path(d / 'declared_arms.json')
    path.write_text(json.dumps(doc, indent=1, default=repr), encoding='utf-8')
    return path


def verify(results_dir: Path) -> list:
    """
    Problems found: declared steps without a finished manifest, manifests
    whose config hash differs from the declaration, and declared arms missing
    from a step's summary.
    """
    from src.simulation.episode_log import _long_path
    results_dir = _long_path(Path(results_dir))
    decl_path = results_dir / 'manifests' / 'declared_arms.json'
    if not decl_path.exists():
        return [f'no declaration at {decl_path}']
    decl = json.loads(decl_path.read_text(encoding='utf-8'))
    finished = {}
    for p in (results_dir / 'manifests').glob('*__*.json'):
        m = json.loads(p.read_text(encoding='utf-8'))
        if m.get('status') == 'finished':
            finished.setdefault(m['step'], []).append(m)
    problems = []
    for st in decl['steps']:
        ms = finished.get(st['name'])
        if not ms:
            problems.append(f"{st['name']}: no finished manifest")
            continue
        if all(m['config']['resolved_sha256'] != decl['config_sha256'] for m in ms):
            problems.append(f"{st['name']}: config hash differs from the declaration")
        summ = st.get('summary')
        if summ and st.get('arms'):
            sp = results_dir / summ
            if not sp.exists():
                problems.append(f"{st['name']}: summary {summ} missing")
                continue
            s = json.loads(sp.read_text(encoding='utf-8'))
            cells = s if 'avg_latency_ms' in next(iter(s.values()), {}) else None
            for scale in st.get('scales') or [None]:
                cell = cells if cells is not None else s.get(str(scale), {})
                missing = [a for a in st['arms'] if a not in cell]
                if missing:
                    problems.append(f"{st['name']} N={scale}: arms missing {missing}")
    return problems


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['verify'])
    p.add_argument('--results-dir', type=str, default=str(_ROOT / 'results'))
    a = p.parse_args()
    probs = verify(Path(a.results_dir))
    print('\n'.join(probs) if probs else 'verify: all declared steps finished '
          'with their declared arms')
    sys.exit(1 if probs else 0)


if __name__ == '__main__':
    sys.path.insert(0, str(_CODE_DIR))
    main()
