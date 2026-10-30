# Entrega · MLOps — Defiende el servidor

## Integrantes y repositorio
- Integrante: **Daniela González Ortiz** (trabajo individual) · Grupo: **A**
- Repositorio de código: https://github.com/danielaestudio02-del/Taller_1 (rama `ccr-1ea24a3d-l2kztl`)
- Commit final de la entrega: el último commit de la rama `ccr-1ea24a3d-l2kztl`; su SHA completo se indica en el mensaje de entrega (un archivo no puede contener el hash de su propio commit).

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
| Hugging Face repo | https://huggingface.co/Daniela201000/defiende-el-servidor | ídem |
| Revisión HF (SHA completo) | `691479df2d72c01e99bd1b32c9d5b82368ff8b70` (tag `v1`) | `e2f7bfdd932f383d7cf40c5413d5d407408889ba` (tag `v2`) |
| Validación recall / precision / FPR / macro-F1 | 0.9886 / 0.9223 / 0.1776 / 0.9225 | 0.9540 / 0.9760 / 0.0499 / 0.9463 |

La revisión `45323bdaff1232c44ca6f4c8273b12a7de78426c` del repo de modelos solo agrega el resultado de test a la
model card de v2: `model.joblib` es idéntico al de la revisión evaluada `e2f7bfdd…`.
Los tags Git `v1` y `v2` se crearon sobre los commits de la tabla. Si no aparecen en GitHub, los commits
completos identifican cada solución.

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
| Recuperación | `evidencias/recuperacion/`: `muestra_validacion.csv` (25 filas), `pred_v*_original.csv`, `pred_v*_recuperado.csv`, `log_recuperacion.txt`, `recuperacion_resultado.json`. **Resultado: OK** (30-10-2026, commit `3da339d1`; etiquetas idénticas, diferencia máxima de score 4,9e-11 ≤ tolerancia 1e-9) |
| Paquetes de modelo publicados | `models/release/v1`, `models/release/v2` |

## Almacenamiento DVC y acceso
- Remote `hfstore` (por defecto en `.dvc/config`): https://huggingface.co/datasets/Daniela201000/defiende-el-servidor-dvc,
  ruta `dvcstore/`, commit `7e9779311b8530acc76a454728321492f254c88a`.
- Acceso: repositorio **público**. `dvc pull` funciona sin credenciales. No se comparten tokens.

## Comandos
Están completos en el `README.md`:
- Instalación: `pip install -r requirements.txt`
- Recuperación de datos: `dvc pull` (~1 min)
- Entrenamiento: `python src/train.py configs/experiments/<exp>.yaml`
- Predicción: `python predict.py --model-dir <carpeta HF> --input <csv> --output <csv>`
- Prueba de recuperación: `bash scripts/recuperacion.sh <commit>` (~8 min, la mayor parte en instalar dependencias)

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
  experimentos, publicar en Hugging Face y redactar la documentación y el informe. La verificación
  quedó automatizada y registrada: huellas SHA-256 contra el anexo, ausencia de grupos compartidos
  train/val, recálculo de métricas de test (`recompute_metrics.py`) y prueba de recuperación en un
  entorno nuevo (resultado OK).
- Sin código externo copiado. Las bibliotecas usadas son scikit-learn, pandas, MLflow, DVC y
  huggingface_hub, con sus licencias respectivas.
- Contribución: trabajo individual de Daniela González Ortiz. Definió el alcance, creó y configuró
  las cuentas y credenciales (GitHub, Hugging Face), configuró el entorno, dirigió el trabajo con el
  asistente y revisó las decisiones y los resultados.
