# Entrega · MLOps — Defiende el servidor

> Los campos marcados **PENDIENTE** se completan al publicar en Hugging Face (`src/publish_hf.py`).

## Integrantes y repositorio
- Integrantes: **PENDIENTE (nombres completos)** · Grupo: **PENDIENTE (A/B)**
- Repositorio de código: https://github.com/danielaestudio02-del/Taller_1 (rama `ccr-1ea24a3d-l2kztl`)
- Commit final de la entrega: **PENDIENTE** (el último commit de la rama al enviar)

## Versiones
| | v1 | v2 |
|---|---|---|
| Qué es | Regresión logística base (one-hot + escalado), umbral 0.5 | HistGradientBoosting, umbral 0.65 elegido en validación |
| Commit de código de la solución | `330015e9da6c114e870f444c720d585952fbfaac` (tag `v1`) | `2ae6479c9528c7dcc2342908618d4b432c8131fb` (tag `v2`) |
| Commit en que se entrenó | `9a17cd225142061547ee474b9720e1113bfbea92` | `9a17cd225142061547ee474b9720e1113bfbea92` |
| Datos DVC (preparados) | `data/prepared.dvc` md5 `53545c819eb4a624ddf5da61d08666c7.dir` | ídem |
| Datos DVC (originales) | `data/raw.dvc` md5 `fe93051ae209ffb0128a31e7a478c009.dir` | ídem |
| Run ID MLflow | `d918b87582934fdb85c0567161272c9c` | `a5ca54703ec744fba37ff99212277e71` |
| SHA-256 de `model.joblib` | `2b350cdc7d82316f448ed5b1a498a43798201a6a9654ecd7a42191aa8ab09c6c` | `ac60460c5b0508bea7d5504a8d0dff24cda1d2fafca94825b4ff1e369d5f2b62` |
| Umbral | 0.50 | 0.65 |
| Hugging Face repo | **PENDIENTE** | **PENDIENTE** |
| Revisión HF (SHA completo) | **PENDIENTE** | **PENDIENTE** |
| Validación recall / precision / FPR / macro-F1 | 0.9886 / 0.9223 / 0.1776 / 0.9225 | 0.9540 / 0.9760 / 0.0499 / 0.9463 |

Los tags Git `v1` y `v2` se crearon sobre esos commits. Si no aparecen en GitHub, los commits completos
de la tabla identifican cada solución.

Los datos v1 (originales) se fijaron en el commit `1b826c1f1700cad8881f4c046b83ed0c47938f73`.
Los datos v2 (preparados y particiones) se fijaron en el commit `9a17cd225142061547ee474b9720e1113bfbea92`.

## Evidencias
| Evidencia | Ubicación |
|---|---|
| Auditoría de datos y particiones | `evidencias/auditoria.json`, `data/prepared/manifest.json` (vía DVC) |
| Tabla de experimentos | `evidencias/experimentos.csv` |
| Exportación MLflow (4 runs) | `evidencias/mlflow/` (`index.json` + `run.json`, config y puntuaciones de validación por run) |
| Selección y congelación | `evidencias/seleccion.json`, curva `evidencias/seleccion_umbral.csv`; congelado en el commit `2ae6479c…` |
| Test final | `evidencias/metricas-test.json`, `evidencias/predicciones-test.csv` (commit `701a209c…`) |
| Referencias Hugging Face | `evidencias/huggingface.json` |
| Recuperación | `evidencias/recuperacion/` (muestra, predicciones originales/recuperadas, log, resultado) |
| Paquetes de modelo publicados | `models/release/v1`, `models/release/v2` |

## Almacenamiento DVC y acceso
- Remote `hfstore`, un repositorio *dataset* de Hugging Face: **PENDIENTE** (URL en `.dvc/config` y `evidencias/huggingface.json`).
- Acceso: público para lectura, o invitación al docente. No se comparten tokens.

## Comandos
Están completos en el `README.md`:
- Instalación: `pip install -r requirements.txt`
- Recuperación de datos: `dvc pull`
- Entrenamiento: `python src/train.py configs/experiments/<exp>.yaml`
- Predicción: `python predict.py --model-dir <carpeta HF> --input <csv> --output <csv>`
- Prueba de recuperación: `bash scripts/recuperacion.sh <commit>`

## Tiempo y recursos medidos
Máquina de 4 vCPU, 15 GB de RAM, sin GPU.
- `prepare.py`: ~22 s.
- Entrenamiento con 140.274 filas: exp1 9 s, exp2 6 s, exp3 7 s, exp4 4 s.
- Evaluación en test: < 5 s.

## Limitaciones conocidas
- La FPR en test (0.181) es mucho mayor que en validación (0.050). Se conserva el resultado y se explica
  en el informe.
- Alrededor del 10 % de las entradas de test aparecen idénticas en train/val.
- El dataset es histórico y de laboratorio: no demuestra eficacia en una red actual.

## Uso de IA, código externo y contribuciones
- Se usó Claude Code (asistente de IA de Anthropic) para escribir el código, ejecutar los
  experimentos y redactar la documentación. **PENDIENTE: describir qué verificó cada integrante**
  (p. ej., revisión de particiones, recálculo de métricas, prueba de recuperación).
- Sin código externo copiado. Las bibliotecas usadas son scikit-learn, pandas, MLflow, DVC y
  huggingface_hub, con sus licencias respectivas.
- Contribución de cada integrante: **PENDIENTE**.
