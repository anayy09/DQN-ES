# Code reference

Technical notes for the simulation code: data flow, the model as implemented, reproducibility, and how to add a scheduler.

---

## Data flow

```
data_ingestion/event_generator.py  (synthetic tasks)   or   analysis/mitbih_trace_eval.py (MIT-BIH windows)
        |  SimulationTask -> core.task.HealthcareTask
        v
simulation/replicate.py            per-replicate seeds, topology, task stream, scheduler construction
        v
simulation/environment.py          OffloadingEnvironment.step(): advance clock, scheduler.select_node(task),
        |                          realised latency/energy/R_P/SLA, queue reservation, per-task row
        v
simulation/episode_log.py          per-run raw logs (gzip CSV), steady-state R_P, queue metrics
        v
analysis/*                         drivers -> results/*.json, *.csv  ->  analysis/statistical_tests.py, figures_q1.py
```

`src/config.py` holds every hyperparameter, the seed helpers, both scheduler registries and the declared statistics families. `configs/simulation_config.yaml` is descriptive only.

---

## Model as implemented

**Network (`core/network.py`, `simulation/topology.py`).** The topology has N wearables within 50 m of an edge gateway, M fog nodes 1 to 5 km away, and a cloud 50 km away. Each wearable has a direct link to every destination. The uplink rate is R = B log2(1 + P h / N0), with h = (d0/d)^α and no interference. The result-return downlink uses the same link as a reciprocal channel, with the serving side transmitting at `DOWNLINK_TX_POWER_W`.

**Queues.** Every node, including a wearable executing locally, is a FIFO server. A task that reaches node j at time a waits max(0, busy_until_j − a) and is then served for C_i / f_j, using its own C_i. The node stays occupied until service ends; `advance_time()` releases it at completion time. Schedulers see the current backlog when they evaluate a node, and this prediction equals the realised wait.

**Latency and energy (`core/offload_model.py`).** A single function serves the scheduler cost, the normalisation bounds and the realised metrics.
- Offload latency: L = t_tx + t_prop + t_queue + t_proc + t_dl, where t_dl = t_prop + S_res / R_dl.
- Wearable offload energy: E = P_tx·t_tx + P_rx·t_rx + P_wait·(L − t_tx − t_rx), with battery draws from the ESP32-S3 datasheet (TX 283 mA, RX 88 mA, wait 47.6 mA, all at 3.3 V; `WEARABLE_TX_DRAW_W`, `WEARABLE_RX_POWER_W`, `WEARABLE_WAIT_DRAW_W` in `src/config.py`). The radiated power used in the rate equation (18.5 dBm) is a separate constant.
- Local execution: L = t_queue + C_i / f_w and E = P_cmp·C_i / f_w (compute draw 65.9 mA at 3.3 V).
- Optional Rayleigh block fading and ARQ packet loss (`fading`, `packet_loss`): schedulers plan with the expected channel and the environment realises the sampled one.

**Cost (`core/cost_function.py`).** F = ŵ_E Ê + ŵ_L L̂ + ŵ_P R_P. The CI-adaptive weights are exp(−α_E Φ), (e^{β_L Φ} − 1)/(e^{β_L} − 1) and (1 − Φ)^{γ_P}, renormalised to sum 1. R_P = ρ (1 − H/H_max), where H is the entropy of the device's last W = 50 destinations. `set_weight_mode()` switches the weight scheme for the ablation, and `privacy_scale` (λ_P) rescales w_P.

**DQN-ES (`algorithms/dqn_es.py`).**
- Network: a two-hidden-layer NumPy network (64 units) over a (2 + 4(M+2))-dimensional state: CI, p_atk, and per destination the uplink rate, load, RTT and a node-type constant.
- Candidate selection: with probability ε a uniformly random K-subset is taken; otherwise the top-K by Q. The executed action is argmin F within that set.
- Learning: transitions are stored when the next decision arrives, with a replay buffer and a target network.

**Workload (`data_ingestion/event_generator.py`).** Four task types have profiles in `core/task.py:TASK_PROFILES` (the single source). Arrivals are Poisson by default. Options: MMPP-2 arrivals, an absolute rate set from an offered edge utilisation, attack bursts in the environment, and the payload, result-size and CI-noise knobs in `simulation/replicate.py`.

---

## Reproducibility

| Stream | Seed |
|---|---|
| task generator, scheduler (DQN init, exploration, replay sampling) | `replicate_seed(r, N)` = 42 + 1000 r + N |
| topology, environment (attack bursts), MMPP arrivals, CI noise, warm-start pre-training, fading/ARQ | children of `SeedSequence(replicate_seed(r, N))` |

`config.make_scheduler()` passes `seed=` to every scheduler that accepts it. All drivers build replicates through `simulation/replicate.py`, so replicate r is the same draw for every algorithm. `analysis/paired_stats.py` pairs algorithms by `run_id`.

---

## Checks

- `python code/run_test.py`: every scheduler in both registries runs on 500 tasks.
- `python code/run_model_checks.py`: uplink, propagation, compute, downlink, total latency and energy, and local execution are compared against their equations. The FIFO queue is compared against M/M/1 and M/D/1 mean waits. The script exits with status 1 on failure.

---

## Adding a scheduler

1. Subclass `BaseScheduler` in `src/algorithms/` and implement `select_node(task) -> node_id`. Use `self.estimate_feasible_bounds(task)` and `self.evaluate_node(task, node_id, lat_bounds, eng_bounds)`, which returns `(cost, latency_s, energy_j, privacy_risk)`. Call `self.record_decision(task.device_id, node_id)` before returning. Accept a `seed` argument if the scheduler is stochastic.
2. Register it in `get_full_algorithm_registry()` or `get_experiment_registry()` in `src/config.py`.
3. Run `python code/run_test.py`.
