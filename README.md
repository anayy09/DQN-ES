# DQN-ES: DQN candidate restriction and acuity leakage in privacy-aware IoMT task offloading

Simulation and analysis code for the article *Deep Q-Network Candidate Restriction and Acuity Leakage in Privacy-Aware Healthcare IoT Task Offloading* (A. Sinhal, A. Sinhal, A. Sinhal; submitted to *Discover Internet of Things*).

Wearables in an Internet of Medical Things (IoMT) network offload tasks to an edge gateway, fog servers or the cloud. DQN-ES lets a deep Q-network (DQN) rank the network destinations, keeps the top K, and picks the destination that minimises a cost of wearable energy, latency and a routing-entropy term R_P, with weights that depend on a criticality index Φ. The study compares DQN-ES with per-task enumeration, random candidate restriction, entropy-reweighted greedy scheduling and a criticality-agnostic static rule, and trains an adversary on each policy's flow metadata to measure how well patient acuity can be inferred. The article reports that R_P does not track that inference.

---

## Reproducing the article

Every number in the article and its Supplementary Information comes from files under `results/`. They were produced in two stages.

1. **Simulation.** The tag `round2-freeze-2` ran the full pipeline: 39 declared steps, 30 paired replicates per configuration. Release `v2.0.0` differs from that tag only in analysis code that reads logs, comments and documentation, so it produces the same simulation outputs.
2. **Post-run analyses.** The adversary, energy profiles, exploration diagnostics, matched-latency robustness and paired tables read the per-task raw logs and never rerun the simulator.

Each run wrote a manifest to `results/manifests/<step>__<run_id>.json`, recording the commit, config hash, parameters, seeds, host and timings, plus a line in `results/manifests/runs.jsonl`.

### 1. Environment

Python 3.10 or later; the runs used Python 3.14 and NumPy 2.4 on Windows 11 (Intel Core Ultra 7 266V).

```bash
pip install -r code/requirements.txt
pip install scikit-learn shap      # adversary, CI module
```

### 2. Quick checks (about 1 minute)

```bash
python code/run_test.py           # every scheduler on 500 synthetic tasks; ends "All schedulers completed successfully."
python code/run_model_checks.py   # latency, energy and FIFO-queue components against closed-form values (17 checks)
```

### 3. Full simulation (entry command; about 6.5 h of compute on 7 workers)

```bash
python code/run_q1_pipeline.py --n_runs 30 --workers 7 --results-dir results --figures-dir results/figures_freeze2
cd code && python -m src.analysis.manifest verify --results-dir ../results
```

The pipeline declares all steps in `results/manifests/declared_arms.json` before it runs them, and `verify` checks that every declared step finished with its declared arms. The MIT-BIH step needs the PhysioNet records (see Data); the CI-module step needs the Kaggle file. To check the pipeline end to end in a few minutes, run `--n_runs 2 --scales 100 --mitbih-max-tasks 500` into scratch directories.

### 4. Post-run analyses (from `code/`, after step 3)

```bash
python -m src.analysis.post_run e5 --raw-dirs ../results/raw/mc_full/n1000 ../results/experiments/raw/mc_exp/n1000 --out ../results/e5_adversary
python -m src.analysis.post_run corr --e5-dir ../results/e5_adversary --summaries ../results/mc_full_summary.json ../results/experiments/mc_exp_summary.json --scale 1000 --out ../results/e5_adversary
python -m src.analysis.post_run e5cv30 --raw-dirs ../results/raw/mc_full/n1000 ../results/experiments/raw/mc_exp/n1000 --out ../results/e5_adversary
python -m src.analysis.post_run aucmatched --e5-dir ../results/e5_adversary --summaries ../results/mc_full_summary.json ../results/experiments/mc_exp_summary.json --scale 1000 --out ../results/e5_adversary
python -m src.analysis.post_run lambdamech --raw-dirs ../results/experiments/raw/mc_exp/n1000 --out ../results/e5_adversary
python -m src.analysis.post_run e8 --raw-dirs ../results/raw/mc_full/n1000 ../results/experiments/raw/mc_exp/n1000 --out ../results/e8_energy
python -m src.analysis.post_run e8 --raw-dirs ../results/sensitivity/ecg80000/raw/mc_full/n1000 --out ../results/e8_energy_10kb
python -m src.analysis.post_run e11 --raw-dirs ../results/raw/mc_full/n1000 ../results/experiments/raw/mc_exp/n1000 --algorithms DQN-ES DQN-only DQN-ES[K=2] DQN-ES[K=4] DQN-ES[K=5] --out ../results/e11_exploration
python -m src.analysis.post_run d17a --out ../results/d17a_robustness
python -m src.analysis.post_run dominance --summaries ../results/mc_full_summary.json ../results/experiments/mc_exp_summary.json --scale 1000 --out ../results/exploratory
python -m src.analysis.post_run paired --pairs 'DQN-ES[K=1]|DQN-ES[K=2]' 'DQN-ES[K=2]|DQN-ES[K=3]' 'DQN-ES[K=3]|DQN-ES[K=4]' 'DQN-ES[K=4]|DQN-ES[K=5]' --tag ksweep --out ../results/exploratory
python -m src.analysis.post_run paired --pairs 'DQN-ES|Random-K[K=3]' 'DQN-ES|ES-only' 'DQN-ES|Static-Tier' 'DQN-ES|q-mixed[q=0.5]' --tag e2_dqnes_vs_randomk --out ../results/exploratory
python -m src.analysis.post_run paired --pairs 'DQN-ES|ES-only[lP=2]' 'DQN-ES|ES-only[lP=3]' 'DQN-ES|q-mixed[q=0.25]' 'DQN-ES|Local-Only' 'DQN-ES|Cloud-Only' 'DQN-ES|DQN-only' --tag headline --out ../results/exploratory
python -m src.analysis.post_run e5 --raw-dirs ../results/experiments_n5000/raw/mc_all/n5000 --out ../results/e5_adversary_n5000
python -m src.analysis.post_run aucmatched --e5-dir ../results/e5_adversary_n5000 --summaries ../results/experiments_n5000/mc_all_summary.json --scale 5000 --out ../results/e5_adversary_n5000
python -m src.analysis.post_run e5 --raw-dirs ../results/raw/mitbih/n8640 --out ../results/e5_adversary_mitbih
python -m src.analysis.post_run aucmatched --e5-dir ../results/e5_adversary_mitbih --summaries ../results/mitbih_trace_raw.json --scale 0 --out ../results/e5_adversary_mitbih
python -m src.analysis.post_run e5 --raw-dirs ../results/sensitivity/warm500/raw/mc_all/n1000 --out ../results/e5_adversary_warm500
python -m src.analysis.post_run e5 --raw-dirs ../results/sensitivity/warm2000/raw/mc_all/n1000 --out ../results/e5_adversary_warm2000
```

The adversary runs take about 5 to 20 minutes each. Instead of rerunning step 3, you can run these commands on the archived raw logs (see Raw logs).

### Where the numbers are

| Article content | File(s) under `results/` |
|---|---|
| Main comparison, N = 1000 (means, CIs) | `mc_full_summary.json`, `experiments/mc_exp_summary.json`, `table3_n1000.csv` |
| Declared paired tests (main, decomposition) | `table3_stat_tests.csv`, `stat_tests_decomposition.csv` |
| Privacy excess at matched latency | `matched_latency.csv`; every condition: `d17a_robustness/` |
| Acuity-inference adversary (AUC, MI, declared tests, correlation, mechanism) | `e5_adversary/` (`adversary_auc_ci.csv`, `stat_tests_privacy_inference.csv`, `rp_vs_auc_correlation.csv`, `auc_matched_latency.csv`, `lambda_p_destination_by_tier.csv`) |
| Adversary at N = 5000, on MIT-BIH, after warm start | `e5_adversary_n5000/`, `e5_adversary_mitbih/`, `e5_adversary_warm500/`, `e5_adversary_warm2000/` |
| Episode length N = 100 to 5000 | `mc_full_summary.json`, `experiments_n5000/` |
| Fog-server count M = 8, 16, 32 | `scaling/M<M>/` |
| Load, MMPP-2, fading/ARQ, 10 KB payload, result size, warm start, CI noise | `sensitivity/<tag>/` |
| Decision time, parameters, FLOPs, memory | `scheduling_overhead.json`, `scheduling_overhead_summary.csv` |
| Energy profiles | `e8_energy/`, `e8_energy_10kb/` |
| Exploration diagnostics | `e11_exploration/` |
| Exploratory paired tables | `exploratory/` |
| MIT-BIH trace-driven simulation | `table5_mitbih_trace.csv`, `mitbih_trace_raw.json` |
| Weight-scheme ablation | `table5_weight_ablation.csv`, `table6_highci_weights.csv` |
| CI module (R², SHAP) | `shap_feature_importance.json` |
| Closed-form model checks | `model_checks.json` |

File names do not follow the article's table numbers. Naming differences: the scheduler the article calls SMA-HHO is `HS-HHO` in the code and result files. It makes the same decision as ES-only on every task. `DQN-ES[K=3]` is DQN-ES; `q-mixed[q=0]` and `ES-only[lP=1]` are ES-only; `q-mixed[q=1]` is `Random-K[K=3]`.

Two pipeline steps are not reported in the article. `privacy_guard` is an attack-detection module whose evaluation was withdrawn; it needs the MedSec-25 file. `figures` draws diagnostic figures to `--figures-dir`.

### Raw logs

The pipeline writes per-task logs (gzip CSV, one file per scheduler and replicate) to `raw/` inside each output directory: `results/raw/<mc_full|mitbih>/n<N>/` and `results/<sub-experiment>/raw/...`. They are not in the repository; they total several hundred MB. The logs behind the article are archived separately on Zenodo with the same directory layout: [raw-log archive DOI].

---

## Repository layout

```
code/
  src/
    algorithms/       schedulers (main comparison + decomposition arms)
    analysis/         experiment drivers, statistics, adversary, figures, manifests
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
results/                         outputs and manifests (per-task raw logs not committed)
```

`code/README.md` describes the model as implemented, the seed streams and how to add a scheduler.

---

## Schedulers

Main comparison (`config.get_full_algorithm_registry()`):

| Name | File | Description |
|------|------|-------------|
| DQN-ES | `dqn_es.py` | DQN ranks the destinations; exhaustive enumeration over the top K (K = 3) |
| PSO+DQN | `pso_dqn.py` | Same DQN; PSO searches the K candidates (identical decisions at K = 3) |
| ES-only | `es_only.py` | Exhaustive enumeration over all destinations, no DQN |
| DQN-only | `dqn_only.py` | DQN-ES with K = 1 (argmax-Q) |
| PSO | `pso.py` | Per-task particle swarm |
| ACO | `aco.py` | Ant colony optimisation |
| HS-HHO | `hs_hho.py` | Slime-mould / Harris-hawks hybrid (SMA-HHO in the article) |
| Local-Only | `local_only.py` | Every task on its wearable |
| Cloud-Only | `cloud_only.py` | Every task to the cloud |

Decomposition arms (`config.get_experiment_registry()`): `DQN-ES[K=1..5]`, `Random-K[K=3]` (random K-subset, no learning), `q-mixed[q=...]` (random K-subset with probability q, otherwise full enumeration), `Static-Tier` (ECG to the edge, everything else local), and `ES-only[lP=...]` (privacy weight scaled by λ_P).

The DQN is written in NumPy; there is no PyTorch or TensorFlow dependency.

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
  MedSec-25/            MedSec-25.csv (only for the unreported privacy_guard step)
```

```bash
python code/run_data_ingestion.py    # writes data/processed/
```

The synthetic experiments need no dataset. The MIT-BIH trace-driven simulation needs `MIT-BIH-Arrhythmia/`, and the CI module needs the Kaggle file.

---

## Reproducibility

Replicate r at episode length N uses base seed 42 + 1000·r + N for the task stream and the scheduler. The topology, environment, arrival, CI-noise, warm-start and fading streams are derived from it with `numpy.random.SeedSequence` (`config.replicate_seeds`). Replicate r is therefore the same draw for every algorithm, and comparisons are paired by replicate. The statistical families, the matched-latency statistic and the adversary configuration are declared in `src/config.py` (`STAT_FAMILIES`, `MATCHED_LATENCY`, `E5_ADVERSARY`).

---

## Citation

If you use this code, please cite the article and the archived release. `CITATION.cff` gives the metadata, and the Zenodo DOI is shown on the repository's release page.

## License

MIT License, see `LICENSE`. The datasets are subject to their own terms (PhysioNet for MIT-BIH, Kaggle for the IoMT vital-signs data and for MedSec-25). Do not redistribute the dataset files.
