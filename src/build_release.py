"""Construye el paquete de una versión (v1/v2) para Hugging Face en models/release/<versión>/.

Contenido: model.joblib (pipeline completo), config.json (umbral, variables, referencias),
requirements.txt, predict.py y README.md (model card).

Uso: python src/build_release.py v1   |   python src/build_release.py v2
"""
from __future__ import annotations

import json
import platform
import shutil
import sys

import sklearn

from common import ROOT, dvc_ref, load_yaml, sha256_file, write_json
from mlflow_utils import artifact, runs_by_experiment_id, val_scores
from common import apply_threshold, binary_metrics


def main(version: str) -> None:
    runs = runs_by_experiment_id()
    rules = load_yaml("configs/selection.yaml")
    if version == "v1":
        exp_id, threshold = rules["v1"]["experiment"], float(rules["v1"]["threshold"])
        reason = rules["v1"]["reason"]
    elif version == "v2":
        sel = json.loads((ROOT / "evidencias/seleccion.json").read_text())
        exp_id, threshold, reason = sel["experiment"], float(sel["threshold"]), sel["reason"]
    else:
        raise SystemExit("versión debe ser v1 o v2")
    run = runs[exp_id]
    rid = run.info.run_id
    out = ROOT / "models" / "release" / version
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(artifact(rid, "model/model.joblib"), out / "model.joblib")
    model_sha = sha256_file(out / "model.joblib")
    assert model_sha == run.data.tags["model_sha256"], "el artefacto no coincide con el registrado en MLflow"
    shutil.copy2(ROOT / "src/predict.py", out / "predict.py")
    exp_cfg = load_yaml(f"configs/experiments/{exp_id}.yaml")
    manifest = json.loads((ROOT / "data/prepared/manifest.json").read_text())
    s = val_scores(rid)
    val = binary_metrics(s["y_true"], apply_threshold(s["score_attack"], threshold))
    req = [l for l in (ROOT / "requirements.txt").read_text().splitlines()
           if l.split("==")[0] in ("scikit-learn", "pandas", "numpy", "scipy", "joblib")]
    (out / "requirements.txt").write_text(f"# Python {platform.python_version()}\n" + "\n".join(req) + "\n")

    cfg = {
        "version": version,
        "model_file": "model.joblib",
        "model_sha256": model_sha,
        "threshold": threshold,
        "decision_rule": "y_pred = 1 (ataque) si score_attack >= threshold; score_attack = predict_proba[:, 1]",
        "positive_class": {"value": 1, "name": "ataque"},
        "features": manifest["features"],
        "excluded_columns": ["id", "attack_cat", "label"],
        "experiment": exp_id,
        "family": exp_cfg["family"],
        "preprocess": exp_cfg["preprocess"],
        "model": exp_cfg["model"],
        "seed": exp_cfg["seed"],
        "selection_reason": reason,
        "validation_metrics": val,
        "references": {
            "github_repo": "https://github.com/danielaestudio02-del/Taller_1",
            "git_tag": version,
            "git_commit_training": run.data.tags["git_commit"],
            "dvc_raw": dvc_ref("data/raw.dvc"),
            "dvc_prepared": dvc_ref("data/prepared.dvc"),
            "mlflow_experiment": "defiende-el-servidor",
            "mlflow_run_id": rid,
        },
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    write_json(out / "config.json", cfg)
    (out / "README.md").write_text(model_card(cfg, manifest), encoding="utf-8")
    print(version, exp_id, rid, model_sha, threshold)


def model_card(c: dict, manifest: dict) -> str:
    v, r, m = c["version"], c["references"], c["validation_metrics"]
    sp = manifest["splits"]
    return f"""---
license: other
library_name: sklearn
tags: [tabular-classification, intrusion-detection, unsw-nb15, mlops-course]
pipeline_tag: tabular-classification
---

# Defiende el servidor · modelo {v}

Clasificador binario de conexiones de red **normal (0) / ataque (1)** entrenado con
UNSW-NB15 para el taller evaluable de MLOps (Universidad Externado de Colombia, 2026-II).
Es material académico: **no** está validado para proteger una red real.

## Propósito
Distinguir conexiones normales de posibles ataques y demostrar trazabilidad y recuperación
de versiones (código → datos → experimento → modelo).

## Modelo
- Experimento: `{c['experiment']}` (familia {c['family']}), semilla {c['seed']}
- Preprocesamiento: `{json.dumps(c['preprocess'])}`
- Estimador: `{json.dumps(c['model'])}`
- Archivo: `model.joblib` (pipeline sklearn completo: imputación, codificación/escalado y estimador)
- SHA-256 del artefacto: `{c['model_sha256']}`
- **Umbral: {c['threshold']}** → `y_pred = 1` si `score_attack >= {c['threshold']}`
- Razón de esta versión: {c['selection_reason']}

## Datos
- Fuente: UNSW-NB15, particiones publicadas `UNSW_NB15_training-set.csv` (175.341 filas) y
  `UNSW_NB15_testing-set.csv` (82.332), página oficial https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Entrenamiento/validación salen solo del archivo de entrenamiento (StratifiedGroupKFold, semilla 42;
  las filas con entradas idénticas nunca se reparten entre ambos):
  train {sp['train']['rows']} filas, validación {sp['val']['rows']} filas.
- Variables: {len(c['features'])} predictoras; se excluyen `id`, `attack_cat` (revela la respuesta) y `label`.

## Métricas de validación (umbral {c['threshold']}, ataque = positiva)
| recall ataque | precision ataque | FPR | macro-F1 | accuracy |
|---|---|---|---|---|
| {m['recall_attack']:.4f} | {m['precision_attack']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |

Matriz de confusión (validación): TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}.
Denominadores cero se reportan como 0.0.

## Uso
```bash
pip install -r requirements.txt
python predict.py --model-dir . --input filas.csv --output predicciones.csv
```
```python
from huggingface_hub import snapshot_download
d = snapshot_download(repo_id="<repo_id>", revision="<SHA completo de la revisión>")
```
`predict.py` verifica el SHA-256 de `model.joblib` antes de cargarlo. La entrada debe traer las
columnas de `config.json["features"]` (mismo formato que los CSV de UNSW-NB15).

## Limitaciones
- Dataset histórico (2015) generado en laboratorio; no demuestra eficacia sobre una red actual.
- 42 % de las filas de entrenamiento son entradas duplicadas y hay grupos con etiquetas contradictorias.
- Cerca del 10 % de las entradas de test aparecen idénticas en train/val: el test es optimista.
- Niveles de `state` no vistos en entrenamiento se codifican como desconocidos.
- Cargar `model.joblib` ejecuta pickle: usar solo revisiones verificadas por SHA.

## Trazabilidad
- Código: {r['github_repo']} · tag `{r['git_tag']}` · commit de entrenamiento `{r['git_commit_training']}`
- Datos DVC: originales `data/raw.dvc` md5 `{r['dvc_raw']['md5']}` (commit `{r['dvc_raw']['git_commit']}`);
  preparados `data/prepared.dvc` md5 `{r['dvc_prepared']['md5']}` (commit `{r['dvc_prepared']['git_commit']}`)
- MLflow: experimento `{r['mlflow_experiment']}`, run `{r['mlflow_run_id']}`
- Entorno: Python {c['python']}, scikit-learn {c['sklearn']}
"""


if __name__ == "__main__":
    main(sys.argv[1])
