---
license: other
library_name: sklearn
tags: [tabular-classification, intrusion-detection, unsw-nb15, mlops-course]
pipeline_tag: tabular-classification
---

# Defiende el servidor · modelo v1

Clasificador binario de conexiones de red **normal (0) / ataque (1)** entrenado con
UNSW-NB15 para el taller evaluable de MLOps (Universidad Externado de Colombia, 2026-II).
Es material académico: **no** está validado para proteger una red real.

## Propósito
Distinguir conexiones normales de posibles ataques y demostrar trazabilidad y recuperación
de versiones (código → datos → experimento → modelo).

## Modelo
- Experimento: `exp1_logreg_base` (familia lineal), semilla 42
- Preprocesamiento: `{"kind": "linear", "log1p": false, "min_frequency": 20}`
- Estimador: `{"type": "logistic_regression", "params": {"C": 1.0, "max_iter": 2000}}`
- Archivo: `model.joblib` (pipeline sklearn completo: imputación, codificación/escalado y estimador)
- SHA-256 del artefacto: `2b350cdc7d82316f448ed5b1a498a43798201a6a9654ecd7a42191aa8ab09c6c`
- **Umbral: 0.5** → `y_pred = 1` si `score_attack >= 0.5`
- Razón de esta versión: primera solución funcional (referencia lineal) con el umbral por defecto

## Datos
- Fuente: UNSW-NB15, particiones publicadas `UNSW_NB15_training-set.csv` (175.341 filas) y
  `UNSW_NB15_testing-set.csv` (82.332), página oficial https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Entrenamiento/validación salen solo del archivo de entrenamiento (StratifiedGroupKFold, semilla 42;
  las filas con entradas idénticas nunca se reparten entre ambos):
  train 140274 filas, validación 35067 filas.
- Variables: 42 predictoras; se excluyen `id`, `attack_cat` (revela la respuesta) y `label`.

## Métricas de validación (umbral 0.5, ataque = positiva)
| recall ataque | precision ataque | FPR | macro-F1 | accuracy |
|---|---|---|---|---|
| 0.9886 | 0.9223 | 0.1776 | 0.9225 | 0.9355 |

Matriz de confusión (validación): TP=23595 FP=1989 TN=9211 FN=272.
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
- Código: https://github.com/danielaestudio02-del/Taller_1 · tag `v1` · commit de entrenamiento `9a17cd225142061547ee474b9720e1113bfbea92`
- Datos DVC: originales `data/raw.dvc` md5 `fe93051ae209ffb0128a31e7a478c009.dir` (commit `1b826c1f1700cad8881f4c046b83ed0c47938f73`);
  preparados `data/prepared.dvc` md5 `53545c819eb4a624ddf5da61d08666c7.dir` (commit `9a17cd225142061547ee474b9720e1113bfbea92`)
- MLflow: experimento `defiende-el-servidor`, run `d918b87582934fdb85c0567161272c9c`
- Entorno: Python 3.13.17, scikit-learn 1.9.1
