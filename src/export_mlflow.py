"""Exporta los runs de MLflow a evidencias/mlflow/ y genera evidencias/experimentos.csv.

Uso: python src/export_mlflow.py
"""
from __future__ import annotations

import json
import shutil

import pandas as pd
import yaml

from common import ROOT, write_json
from mlflow_utils import artifact, runs_by_experiment_id

OUT = ROOT / "evidencias" / "mlflow"


def main() -> None:
    conclusions = yaml.safe_load((ROOT / "configs/experiments/conclusiones.yaml").read_text(encoding="utf-8"))
    runs = runs_by_experiment_id()
    rows, index = [], []
    for exp_id in sorted(runs):
        r = runs[exp_id]
        rid = r.info.run_id
        d = OUT / f"{exp_id}__{rid}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        record = {
            "run_id": rid,
            "run_name": r.info.run_name,
            "experiment_id": r.info.experiment_id,
            "status": r.info.status,
            "start_time_ms": r.info.start_time,
            "end_time_ms": r.info.end_time,
            "artifact_uri": r.info.artifact_uri,
            "params": r.data.params,
            "metrics": r.data.metrics,
            "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.log-model")},
        }
        write_json(d / "run.json", record)
        for p in ("config", "validation/val_metrics.json", "validation/val_scores.csv.gz"):
            src = artifact(rid, p)
            dst = d / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        m, t = r.data.metrics, r.data.tags
        rows.append({
            "id": exp_id,
            "hipotesis": t["hypothesis"],
            "cambio": t["change"],
            "familia": t["family"],
            "parametros": json.dumps({k: v for k, v in r.data.params.items() if k.startswith(("model.", "preprocess."))},
                                     ensure_ascii=False),
            "config_path": f"configs/experiments/{exp_id}.yaml",
            "semilla": r.data.params["seed"],
            "umbral": r.data.params["threshold"],
            "git_commit": t["git_commit"],
            "dvc_prepared_md5": t["dvc_prepared_md5"],
            "dvc_raw_md5": t["dvc_raw_md5"],
            "run_id": rid,
            "model_sha256": t["model_sha256"],
            "val_recall_attack": round(m["val_recall_attack"], 6),
            "val_precision_attack": round(m["val_precision_attack"], 6),
            "val_fpr": round(m["val_fpr"], 6),
            "val_macro_f1": round(m["val_macro_f1"], 6),
            "val_accuracy": round(m["val_accuracy"], 6),
            "val_tp": int(m["val_tp"]), "val_fp": int(m["val_fp"]),
            "val_tn": int(m["val_tn"]), "val_fn": int(m["val_fn"]),
            "fit_seconds": round(m["fit_seconds"], 1),
            "conclusion_validacion": conclusions[exp_id],
        })
        index.append({"id": exp_id, "run_id": rid, "folder": d.relative_to(ROOT).as_posix()})
    pd.DataFrame(rows).to_csv(ROOT / "evidencias/experimentos.csv", index=False)
    write_json(OUT / "index.json", {
        "tracking": "MLflow local, backend sqlite:///mlflow.db (no versionado); esta carpeta es la exportación",
        "experiment_name": "defiende-el-servidor",
        "runs": index,
    })
    print(pd.DataFrame(rows)[["id", "run_id", "val_recall_attack", "val_fpr", "val_macro_f1"]])


if __name__ == "__main__":
    main()
