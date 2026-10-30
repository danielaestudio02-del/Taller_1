"""Selecciona modelo y umbral de v2 usando SOLO validación (reglas en configs/selection.yaml).

Escribe evidencias/seleccion.json y evidencias/seleccion_umbral.csv. Debe ejecutarse y
versionarse (commit) antes de evaluar test.

Uso: python src/select_v2.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import ROOT, apply_threshold, binary_metrics, dvc_ref, git_commit, load_yaml, write_json
from mlflow_utils import runs_by_experiment_id, val_scores


def main() -> None:
    rules = load_yaml("configs/selection.yaml")
    runs = runs_by_experiment_id()
    ranking = sorted(
        ({"id": e, "run_id": runs[e].info.run_id, "val_macro_f1_at_0.5": runs[e].data.metrics["val_macro_f1"],
          "val_fpr_at_0.5": runs[e].data.metrics["val_fpr"]} for e in rules["candidates"]),
        key=lambda r: (-r["val_macro_f1_at_0.5"], r["val_fpr_at_0.5"]),
    )
    best = ranking[0]
    run = runs[best["id"]]
    scores = val_scores(best["run_id"])

    tr = rules["threshold_rule"]
    g = tr["grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 4)
    curve = pd.DataFrame([{"threshold": t, **binary_metrics(scores["y_true"], apply_threshold(scores["score_attack"], t))}
                          for t in grid])
    curve.to_csv(ROOT / "evidencias/seleccion_umbral.csv", index=False)
    feasible = curve[curve[tr["constraint"]["metric"]] <= tr["constraint"]["max"]]
    chosen = feasible.sort_values(["recall_attack", "threshold"], ascending=[False, True]).iloc[0]
    threshold = float(chosen["threshold"])

    at_05 = binary_metrics(scores["y_true"], apply_threshold(scores["score_attack"], 0.5))
    selection = {
        "version": "v2",
        "experiment": best["id"],
        "run_id": best["run_id"],
        "model_sha256": run.data.tags["model_sha256"],
        "threshold": threshold,
        "decision_rule": "ataque (1) si score_attack >= threshold",
        "positive_class": "1 = ataque",
        "references": {
            "git_commit_training": run.data.tags["git_commit"],
            "git_commit_selection_code": git_commit(),
            "git_tag": "v2",
            "dvc_prepared": dvc_ref("data/prepared.dvc"),
            "dvc_raw": dvc_ref("data/raw.dvc"),
            "huggingface": {"repo_id": None, "revision_sha": None,
                            "nota": "se completa al publicar; publicar no cambia model_sha256"},
        },
        "reason": (
            f"{best['id']} tiene el mayor macro-F1 de validación con umbral 0.5 "
            f"({best['val_macro_f1_at_0.5']:.4f}). Con FPR <= {tr['constraint']['max']} como límite de falsas "
            f"alarmas, el umbral {threshold} maximiza el recall de ataque en validación "
            f"({chosen['recall_attack']:.4f}, FPR {chosen['fpr']:.4f})."
        ),
        "rules": rules,
        "model_ranking_validation": ranking,
        "validation_metrics_at_threshold": {k: (float(v) if isinstance(v, float) else int(v))
                                            for k, v in chosen.drop("threshold").items()},
        "validation_metrics_at_0.5": at_05,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_before_test": True,
        "freeze_evidence": "commit Git que agrega este archivo (anterior al commit de evidencias de test)",
    }
    write_json("evidencias/seleccion.json", selection)
    print(selection["reason"])


if __name__ == "__main__":
    main()
