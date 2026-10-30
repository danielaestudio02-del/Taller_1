"""Evaluación ÚNICA de la versión congelada (v2) sobre el archivo de test completo.

Se niega a correr si evidencias/seleccion.json no está en un commit anterior.
Escribe evidencias/metricas-test.json y evidencias/predicciones-test.csv.

Uso: python src/evaluate_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from common import (ROOT, TARGET, binary_metrics, data_config, frame_sha256, git_commit, sha256_file,
                    write_json)

sys.path.insert(0, str(ROOT / "src"))
from predict import load_package, predict_frame  # noqa: E402


def main() -> None:
    freeze_commit = subprocess.check_output(
        ["git", "log", "-n", "1", "--format=%H", "--", "evidencias/seleccion.json"], cwd=ROOT, text=True).strip()
    if not freeze_commit:
        raise SystemExit("seleccion.json no está versionado: congela la selección antes de evaluar test")
    sel = json.loads((ROOT / "evidencias/seleccion.json").read_text())
    pkg = ROOT / "models/release" / sel["version"]
    model, cfg = load_package(pkg)
    assert cfg["model_sha256"] == sel["model_sha256"] and cfg["threshold"] == sel["threshold"]

    dcfg = data_config()
    test = pd.read_parquet(ROOT / dcfg["prepared"]["test"])
    pred = predict_frame(model, cfg, test)
    pred.insert(1, "y_true", test[TARGET].values)
    pred = pred[["row_id", "y_true", "score_attack", "y_pred"]]
    pred.to_csv(ROOT / "evidencias/predicciones-test.csv", index=False, float_format="%.10g")

    m = binary_metrics(pred["y_true"], pred["y_pred"])
    write_json("evidencias/metricas-test.json", {
        "version": sel["version"],
        "experiment": sel["experiment"],
        "mlflow_run_id": sel["run_id"],
        "model_sha256": cfg["model_sha256"],
        "model_package": pkg.relative_to(ROOT).as_posix(),
        "huggingface": sel["references"]["huggingface"],
        "selection_freeze_commit": freeze_commit,
        "evaluation_code_commit": git_commit(),
        "dataset": {
            "raw_file": dcfg["raw"]["test"],
            "raw_sha256": sha256_file(ROOT / dcfg["raw"]["test"]),
            "prepared_file": dcfg["prepared"]["test"],
            "prepared_content_sha256": frame_sha256(test),
            "rows_evaluated": int(len(test)),
            "rows_in_published_file": dcfg["expected_rows"]["test"],
        },
        "threshold": cfg["threshold"],
        "decision_rule": cfg["decision_rule"],
        "positive_class": "1 = ataque",
        "zero_division_policy": "denominador cero -> 0.0",
        "confusion_matrix": {"labels": ["normal(0)", "ataque(1)"], "rows": "real", "cols": "predicho",
                             "matrix": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]},
        "metrics": m,
        "predictions_file": "evidencias/predicciones-test.csv",
        "predictions_sha256": sha256_file(ROOT / "evidencias/predicciones-test.csv"),
        "recompute": "python src/recompute_metrics.py evidencias/predicciones-test.csv",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(json.dumps({k: round(v, 4) for k, v in m.items()}))


if __name__ == "__main__":
    main()
