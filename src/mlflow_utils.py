"""Acceso al registro local de MLflow."""
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from common import ROOT

EXPERIMENT = "defiende-el-servidor"


def client() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    return mlflow.MlflowClient()


def runs_by_experiment_id() -> dict:
    """Último run FINISHED por configuración (tag experiment_id_local)."""
    c = client()
    exp = c.get_experiment_by_name(EXPERIMENT)
    runs = c.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'",
                         order_by=["attributes.start_time DESC"])
    out = {}
    for r in runs:
        out.setdefault(r.data.tags["experiment_id_local"], r)
    return out


def artifact(run_id: str, path: str) -> Path:
    return Path(client().download_artifacts(run_id, path))


def val_scores(run_id: str) -> pd.DataFrame:
    return pd.read_csv(artifact(run_id, "validation/val_scores.csv.gz"))
