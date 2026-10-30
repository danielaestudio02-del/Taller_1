"""Entrena una configuración sobre las particiones fijas y la registra en MLflow.

Uso: python src/train.py configs/experiments/exp1_logreg_base.yaml
"""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn
import yaml

from common import (ROOT, TARGET, apply_threshold, binary_metrics, data_config, dvc_ref,
                    feature_columns, git_commit, git_is_dirty, numeric_columns, sha256_file)
from pipeline import build_pipeline

EXPERIMENT = "defiende-el-servidor"


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--allow-dirty", action="store_true", help="solo para pruebas; no trazable")
    args = ap.parse_args()

    if git_is_dirty() and not args.allow_dirty:
        raise SystemExit("Hay cambios sin commit: haz commit antes de entrenar para que el run sea trazable.")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg = data_config()
    manifest = json.loads((ROOT / dcfg["prepared"]["manifest"]).read_text())

    train = pd.read_parquet(ROOT / dcfg["prepared"]["train"])
    val = pd.read_parquet(ROOT / dcfg["prepared"]["val"])
    feats = feature_columns(train)
    pipe = build_pipeline(cfg, numeric_columns(train))

    setup_mlflow()
    raw_ref, prep_ref = dvc_ref("data/raw.dvc"), dvc_ref("data/prepared.dvc")
    with mlflow.start_run(run_name=cfg["id"]) as run:
        t0 = time.time()
        pipe.fit(train[feats], train[TARGET])
        fit_seconds = time.time() - t0
        scores = pipe.predict_proba(val[feats])[:, 1]
        metrics = binary_metrics(val[TARGET], apply_threshold(scores, cfg["threshold"]))

        mlflow.set_tags({
            "experiment_id_local": cfg["id"],
            "family": cfg["family"],
            "hypothesis": cfg["hypothesis"].strip(),
            "change": cfg["change"],
            "git_commit": git_commit(),
            "git_dirty": str(git_is_dirty()),
            "dvc_raw_md5": raw_ref["md5"],
            "dvc_prepared_md5": prep_ref["md5"],
            "dvc_prepared_git_commit": prep_ref["git_commit"],
            "train_content_sha256": manifest["splits"]["train"]["content_sha256"],
            "val_content_sha256": manifest["splits"]["val"]["content_sha256"],
            "positive_class": "1=ataque",
            "zero_division_policy": "metricas con denominador 0 se reportan como 0.0",
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        })
        mlflow.log_params(flatten({k: v for k, v in cfg.items() if k not in ("hypothesis", "change")}))
        mlflow.log_metrics({**{f"val_{k}": v for k, v in metrics.items()}, "fit_seconds": fit_seconds,
                            "train_rows": len(train), "val_rows": len(val)})

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            joblib.dump(pipe, tmp / "model.joblib")
            mlflow.set_tag("model_sha256", sha256_file(tmp / "model.joblib"))
            mlflow.log_artifact(str(tmp / "model.joblib"), "model")
            mlflow.log_artifact(args.config, "config")
            pd.DataFrame({"row_id": val["row_id"], "y_true": val[TARGET], "score_attack": scores}).to_csv(
                tmp / "val_scores.csv.gz", index=False, float_format="%.10g")
            mlflow.log_artifact(str(tmp / "val_scores.csv.gz"), "validation")
            (tmp / "val_metrics.json").write_text(json.dumps(
                {"threshold": cfg["threshold"], **metrics}, indent=2))
            mlflow.log_artifact(str(tmp / "val_metrics.json"), "validation")
        print(run.info.run_id, cfg["id"], json.dumps({k: round(v, 4) for k, v in metrics.items()}))


if __name__ == "__main__":
    main()
