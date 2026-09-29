# DQN-ES: DQN candidate restriction with exhaustive enumeration for privacy-aware task offloading in IoMT networks

Simulation code for a scheduler that offloads Internet of Medical Things (IoMT) tasks from wearables to an edge gateway, fog nodes or the cloud. A deep Q-network (DQN) ranks the network destinations, and the top-K candidates are enumerated exhaustively under a cost that weighs wearable energy, latency and a routing-entropy privacy proxy. The weights depend on a patient criticality index (CI).

---

## Repository layout

```
code/
  src/
    algorithms/       schedulers (main comparison + decomposition arms)
    analysis/         experiment drivers, statistics, figures, XAI
    core/             task profiles, cost function, offload model, network/queue model
    data_ingestion/   dataset parsers and synthetic task generator
    simulation/       environment, topology, replicate builder, raw logs
    config.py         hyperparameters, seeds, registries, declared statistics
  run_q1_pipeline.py  orchestrator for all experiment steps
  run_model_checks.py closed-form checks of the simulator's model components
  run_test.py         quick integration test (no datasets needed)
  run_data_ingestion.py
  requirements.txt
configs/simulation_config.yaml   descriptive only; not read by the code
results/                         outputs (per-run raw logs under results/raw/ are not committed)
```

---

## Schedulers

Main comparison (`config.get_full_algorithm_registry()`):

| Name | File | Description |
|------|------|-------------|
| DQN-ES | `dqn_es.py` | DQN ranks the destinations; exhaustive enumeration over the top-K (K = 3) |
| PSO+DQN | `pso_dqn.py` | Same DQN; PSO searches the K candidates (identical decisions at K = 3) |
| ES-only | `es_only.py` | Exhaustive enumeration over all destinations, no DQN |
| DQN-only | `dqn_only.py` | DQN-ES with K = 1 (argmax-Q) |
| PSO | `pso.py` | Per-task particle swarm |
| ACO | `aco.py` | Ant colony optimisation |
| HS-HHO | `hs_hho.py` | Hybrid slime-mould / Harris-hawks optimiser |
| Local-Only | `local_only.py` | Every task on its wearable |
| Cloud-Only | `cloud_only.py` | Every task to the cloud |

Decomposition arms (`config.get_experiment_registry()`): `DQN-ES[K=1..5]`, `Random-K[K=3]` (random K-subset, no learning), `q-mixed[q=...]` (random K-subset with probability q, otherwise full enumeration), `Static-Tier` (ECG to the edge, everything else local), and `ES-only[lP=...]` (privacy weight scaled by λ_P).

The DQN is written in NumPy; there is no PyTorch or TensorFlow dependency.

---

## Dependencies

Python 3.10 or later.

```bash
pip install -r code/requirements.txt
pip install scikit-learn shap      # XAI module and the adversary analysis
```

---

## Data

The datasets are not redistributed. Place them under `data/` with these folder names, which the parsers expect:

```
data/
  MIT-BIH-Arrhythmia/   48 WFDB records (.dat/.hea/.atr)
                        PhysioNet MIT-BIH Arrhythmia Database, https://physionet.org/content/mitdb/1.0.0/
  Mendeley-IoMT/        patients_data_with_alerts.xlsx
                        Barman, "IoMT Dataset for ML-Based Health Monitoring", Kaggle, 2024,
                        doi:10.34740/KAGGLE/DSV/7736523 (the folder name is historical)
  CICIoMT2024/          https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
  MedSec-25/            MedSec-25.csv
```

```bash
python code/run_data_ingestion.py    # writes data/processed/
```

The synthetic Monte Carlo experiments need no dataset. The MIT-BIH trace-driven evaluation needs `MIT-BIH-Arrhythmia/`, the CI module needs the Kaggle IoMT file, and the Privacy Guard step needs MedSec-25.

---

## Quick checks

```bash
python code/run_test.py          # every registered scheduler on 500 synthetic tasks
python code/run_model_checks.py  # model components and the FIFO queue vs closed-form expressions
```

`run_test.py` prints one row per scheduler (average latency, energy, routing-predictability risk R_P and SLA violations) and ends with "All schedulers completed successfully." Values depend on the model version; the test checks that every scheduler runs. `run_model_checks.py` exits with status 1 if any component deviates from its equation.

---

## Experiment pipeline

`code/run_q1_pipeline.py` runs the steps below in order. Each step can be skipped with its flag.

```bash
python code/run_q1_pipeline.py --n_runs 30 --workers 8          # full run (several hours)
python code/run_q1_pipeline.py --n_runs 2 --scales 100           # quick end-to-end check
```

| Flag | Step | Main output |
|------|------|-------------|
| `--skip-mc` | Monte Carlo, main registry, all scales | `results/mc_full_summary.json`, `table3_n1000.csv` |
| `--skip-stats` | Paired signed-rank tests, Holm within declared families | `table3_stat_tests.csv`, `table3_n1000_with_pvals.csv` |
| `--skip-weight` | CI weight-scheme ablation (mixed CI) | `table5_weight_ablation.csv` |
| `--skip-privacy` | Privacy Guard on MedSec-25 | `privacy_guard_metrics.json` |
| `--skip-mitbih` | MIT-BIH trace-driven evaluation | `table5_mitbih_trace.csv` |
| `--skip-figures` | Figures | figures directory |
| `--skip-highci` | Weight-scheme ablation, all-high-CI workload | `table6_highci_weights.csv` |
| `--skip-overhead` | Decision-time split vs fog-node count | `scheduling_overhead_summary.csv` |
| `--skip-decomp` | Latency decomposition | `latency_decomposition.csv` |
| `--skip-routing` | DQN-only routing distribution | `dqn_only_routing_summary.json` |

Multi-run steps use a `spawn` process pool (Windows-safe); `--workers N` sets its size.

### Reproducibility

Replicate r at episode length N uses base seed 42 + 1000·r + N for the task stream and the scheduler; the topology, environment, arrival, CI-noise and warm-start streams are derived from it with `numpy.random.SeedSequence` (`config.replicate_seeds`). Replicate r is therefore identical for every algorithm, and comparisons are paired by replicate.

---

## License

MIT License, see `LICENSE`. The datasets are subject to their own terms (PhysioNet for MIT-BIH, Kaggle for the IoMT vital-signs data, CIC terms for CICIoMT2024, Kaggle for MedSec-25). Do not redistribute the dataset files.
