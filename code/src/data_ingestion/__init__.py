"""
src/data_ingestion/__init__.py
------------------------------
Public API for the data ingestion layer of the Bio-Inspired Adaptive Task
Offloading simulation.

Exposed names
-------------
SimulationTask          – dataclass representing one simulation task
load_mendeley_events    – parse/cache Kaggle IoMT XLSX
load_mitbih_events      – parse/cache MIT-BIH Arrhythmia WFDB records
load_ciciot_events      – parse/cache CICIoMT2024 CSV attack flows
load_medsec_events      – parse/cache MedSec-25 CSV attack flows
generate_event_stream   – combine all sources into a unified task stream
generate_synthetic_tasks – generate purely synthetic SimulationTask objects
"""

from .parse_mendeley import load_mendeley_events
from .parse_mitbih import load_mitbih_events
from .parse_ciciot import load_ciciot_events
from .parse_medsec import load_medsec_events
from .event_generator import (
    SimulationTask,
    generate_event_stream,
    generate_synthetic_tasks,
)

__all__ = [
    "SimulationTask",
    "load_mendeley_events",
    "load_mitbih_events",
    "load_ciciot_events",
    "load_medsec_events",
    "generate_event_stream",
    "generate_synthetic_tasks",
]
