"""
parse_mendeley.py
-----------------
Parse the Kaggle IoMT dataset (patients_data_with_alerts.xlsx; the folder and
module names are historical) and convert
each patient record into a simulation task event 4-tuple:
    (D_i, C_i, T_max_i, rho_i)  +  Criticality Index Phi_i

Output: results/mendeley_events.json
Public API: load_mendeley_events(data_dir) -> list[dict]
"""

import os
import json
import random
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Task-type parameter profiles: single source in src/core/task.py
# ---------------------------------------------------------------------------
from src.core.task import TASK_PROFILES  # noqa: E402

# ---------------------------------------------------------------------------
# Alert-level → CI base range
# Handles: 'High', 'Low', 'Normal', 'Abnormal', 'critical', 'warning', etc.
# 'Low' SpO2 is clinically dangerous → treat as critical
# ---------------------------------------------------------------------------
ALERT_CI_MAP = {
    "critical":   (0.80, 1.00),
    "emergency":  (0.80, 1.00),
    "high":       (0.55, 0.85),   # 'High' HR or BP
    "low":        (0.65, 0.90),   # 'Low' SpO2 — most dangerous
    "abnormal":   (0.45, 0.75),   # 'Abnormal' temperature
    "warning":    (0.40, 0.70),
    "moderate":   (0.35, 0.65),
    "normal":     (0.00, 0.25),
    "benign":     (0.00, 0.20),
}

# Disease-specific base CI boost (added to alert-derived CI)
DISEASE_CI_BOOST = {
    "arrhythmia":        0.20,
    "hypertension":      0.10,
    "asthma":            0.12,
    "diabetes mellitus": 0.08,
    "diabetes":          0.08,
    "heart failure":     0.25,
    "normal":            0.00,
}


def _normalise_alert(value: str) -> str:
    """Return a canonical alert tier from a raw string."""
    v = str(value).strip().lower()
    for key in ALERT_CI_MAP:
        if key in v:
            return key
    try:
        num = float(v)
        if num >= 2:
            return "critical"
        elif num >= 1:
            return "warning"
        else:
            return "normal"
    except ValueError:
        pass
    return "normal"


def _ci_from_alert(alert_str: str, rng: random.Random) -> float:
    """Convert an alert level string to a CI score with jitter."""
    tier = _normalise_alert(alert_str)
    lo, hi = ALERT_CI_MAP.get(tier, (0.0, 0.25))
    return round(lo + rng.random() * (hi - lo), 4)


def _disease_boost(disease_str: str) -> float:
    """Return a CI boost based on predicted disease severity."""
    d = str(disease_str).strip().lower()
    for key, boost in DISEASE_CI_BOOST.items():
        if key in d:
            return boost
    return 0.05   # unknown disease: small boost


def _detect_columns(df_columns):
    """
    Detect relevant column names using case-insensitive substring matching.
    Returns a dict with canonical keys mapped to actual column names.
    Handles columns like 'Heart Rate (bpm)', 'SpO2 Level Alert', etc.
    """
    # Build lowercase → original mapping for substring search
    cols_lower = [(c.lower(), c) for c in df_columns]

    def find_col(keywords):
        for kw in keywords:
            kw_l = kw.lower()
            for lc, orig in cols_lower:
                if kw_l in lc:
                    return orig
        return None

    mapping = {}

    # Patient / row identifier
    col = find_col(["patient number", "patient_id", "patient id", "patientid"])
    if col:
        mapping["patient_id"] = col

    # Timestamp
    col = find_col(["timestamp", "datetime", "date_time", "date"])
    if col:
        mapping["timestamp"] = col

    # Primary alert column (highest severity wins):
    # Dataset has: 'Heart Rate Alert', 'SpO2 Level Alert', 'Blood Pressure Alert', 'Temperature Alert'
    # Use the most critical one; we'll aggregate all alert columns in parse logic
    alert_cols = []
    for kw in ["alert"]:
        for lc, orig in cols_lower:
            if kw in lc and orig not in alert_cols:
                alert_cols.append(orig)
    if alert_cols:
        mapping["alert_columns"] = alert_cols   # all alert columns
        mapping["alert"] = alert_cols[0]         # primary alert

    # Predicted disease / condition
    col = find_col(["predicted disease", "disease", "condition", "diagnosis", "prediction"])
    if col:
        mapping["disease"] = col

    # SpO2
    col = find_col(["spo2", "sp02", "oxygen saturation", "oxygen_saturation", "sao2"])
    if col:
        mapping["spo2"] = col

    # Heart rate / ECG
    col = find_col(["heart rate", "heartrate", "heart_rate", "bpm", "ecg", "rr interval"])
    if col:
        mapping["ecg"] = col

    # Blood pressure (systolic preferred)
    col = find_col(["systolic", "sbp", "blood pressure", "blood_pressure"])
    if col:
        mapping["bp"] = col

    # Temperature
    col = find_col(["temperature", "body temp", "temp"])
    if col:
        mapping["temperature"] = col

    # Fall detection
    col = find_col(["fall"])
    if col:
        mapping["fall"] = col

    return mapping


def _infer_task_type(col_mapping: dict, row, rng: random.Random) -> str:
    """
    Decide the task type from which vital-sign columns are present.
    If all three vital groups (ECG, SpO2, BP) are present (as in this dataset),
    use a probability mix to reflect realistic IoMT distributions.
    """
    has_ecg  = "ecg"  in col_mapping
    has_spo2 = "spo2" in col_mapping
    has_bp   = "bp"   in col_mapping
    has_temp = "temperature" in col_mapping

    active = sum([has_ecg, has_spo2, has_bp])

    # When all vitals are present (multi-sensor device), use weighted random
    # to produce realistic task-type distribution (not all "multi_vital")
    if active >= 3 or (active >= 2 and has_temp):
        return rng.choices(
            ["ecg_analysis", "spo2_monitoring", "bp_analysis", "multi_vital"],
            weights=[0.30, 0.40, 0.20, 0.10],
        )[0]
    if active == 2:
        return "multi_vital"
    if has_ecg:
        return "ecg_analysis"
    if has_spo2:
        return "spo2_monitoring"
    if has_bp:
        return "bp_analysis"

    return rng.choices(
        ["ecg_analysis", "spo2_monitoring", "bp_analysis", "multi_vital"],
        weights=[0.30, 0.40, 0.20, 0.10],
    )[0]


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------
def parse_mendeley(data_dir: str) -> list:
    """
    Parse patients_data_with_alerts.xlsx and return a list of simulation
    event dicts.

    Parameters
    ----------
    data_dir : str
        Path to the directory that contains Mendeley-IoMT/
        patients_data_with_alerts.xlsx

    Returns
    -------
    list of dict
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required: pip install pandas openpyxl"
        )

    base = Path(data_dir)
    xlsx_path = base / "Mendeley-IoMT" / "patients_data_with_alerts.xlsx"

    if not xlsx_path.exists():
        raise FileNotFoundError(f"[Mendeley] File not found: {xlsx_path}")

    print(f"\n[Mendeley] Reading: {xlsx_path}")
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"[Mendeley] Failed to read XLSX: {exc}") from exc

    print(f"[Mendeley] Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[Mendeley] Columns detected:\n  {list(df.columns)}\n")

    col_map = _detect_columns(df.columns)
    print(f"[Mendeley] Canonical column mapping: {col_map}")

    rng = random.Random(42)
    events = []

    for idx, row in df.iterrows():
        # --- patient_id ---
        if "patient_id" in col_map:
            pid = str(row[col_map["patient_id"]])
        else:
            pid = f"P{idx:05d}"

        # --- timestamp ---
        if "timestamp" in col_map:
            ts_raw = row[col_map["timestamp"]]
            try:
                ts = float(pd.Timestamp(ts_raw).timestamp())
            except Exception:
                ts = float(idx)
        else:
            ts = float(idx)

        # --- CI score: aggregate all available alert columns + disease boost ---
        alert_cols = col_map.get("alert_columns", [col_map["alert"]] if "alert" in col_map else [])
        disease_boost = 0.0
        if "disease" in col_map:
            disease_boost = _disease_boost(str(row[col_map["disease"]]))
        if alert_cols:
            ci_scores = [_ci_from_alert(str(row[c]), rng) for c in alert_cols if c in row.index]
            base_ci = max(ci_scores) if ci_scores else rng.uniform(0.0, 0.3)
            ci = round(min(1.0, base_ci + disease_boost), 4)
        else:
            # Derive from numeric vitals: out-of-range → higher CI
            ci_components = []
            if "spo2" in col_map:
                try:
                    spo2 = float(row[col_map["spo2"]])
                    # normal SpO2 >= 95; below → critical
                    ci_components.append(max(0.0, (95 - spo2) / 10))
                except (ValueError, TypeError):
                    pass
            if "ecg" in col_map:
                try:
                    hr = float(row[col_map["ecg"]])
                    # normal HR 60-100
                    deviation = max(0, hr - 100) / 100 + max(0, 60 - hr) / 60
                    ci_components.append(min(1.0, deviation))
                except (ValueError, TypeError):
                    pass
            if ci_components:
                ci = round(min(1.0, sum(ci_components) / len(ci_components))
                           + rng.gauss(0, 0.05), 4)
                ci = max(0.0, min(1.0, ci))
            else:
                ci = round(rng.uniform(0.0, 0.5), 4)

        # --- task type ---
        task_type = _infer_task_type(col_map, row, rng)

        # --- parameters from profile ---
        profile = TASK_PROFILES[task_type]

        events.append({
            "patient_id":          pid,
            "timestamp":           ts,
            "ci_score":            ci,
            "task_type":           task_type,
            "data_size_bits":      profile["data_size_bits"],
            "cpu_cycles":          profile["cpu_cycles"],
            "max_delay_s":         profile["max_delay_s"],
            "privacy_sensitivity": profile["privacy_sensitivity"],
        })

    print(f"[Mendeley] Generated {len(events)} simulation events.")
    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_mendeley_events(data_dir: str) -> list:
    """
    Load (or parse and cache) Kaggle IoMT simulation events.

    Tries to read results/mendeley_events.json first; falls back to
    parsing the raw XLSX if the JSON does not exist.

    Parameters
    ----------
    data_dir : str
        Root data directory (contains Mendeley-IoMT/ subdirectory).

    Returns
    -------
    list of dict
    """
    results_dir = Path(data_dir).parent / "results"
    cache_path = results_dir / "mendeley_events.json"

    if cache_path.exists():
        print(f"[Mendeley] Loading cached events from {cache_path}")
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[Mendeley] Cache read failed ({exc}); re-parsing raw data.")

    events = parse_mendeley(data_dir)
    return events


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    script_dir = Path(__file__).resolve().parent          # src/data_ingestion/
    project_root = script_dir.parent.parent               # project root
    data_dir = str(project_root / "data")
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Mendeley IoMT Data Ingestion")
    print("=" * 60)

    try:
        events = parse_mendeley(data_dir)
    except Exception as exc:
        print(f"[Mendeley] ERROR: {exc}")
        sys.exit(1)

    out_path = results_dir / "mendeley_events.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(events, fh, indent=2, default=str)
        print(f"[Mendeley] Saved {len(events)} events to {out_path}")
    except Exception as exc:
        print(f"[Mendeley] Failed to save JSON: {exc}")
        sys.exit(1)

    # Quick sanity print
    if events:
        print("\n[Mendeley] Sample event (index 0):")
        for k, v in events[0].items():
            print(f"  {k}: {v}")
