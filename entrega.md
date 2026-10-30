# 📦 Entrega · MLOps — Defiende el servidor

Este archivo es el índice de mi entrega. Aquí reuní en un solo lugar todos los identificadores
(commits, versiones de datos, runs y revisiones de los modelos) y dónde está cada evidencia, para que
sea fácil revisar el trabajo de punta a punta.

## 👩‍💻 Quién soy y dónde está el código
- **Integrante:** Daniela González Ortiz (trabajé sola) · **Grupo A**
- **Repositorio de código:** https://github.com/danielaestudio02-del/Taller_1 (rama `ccr-1ea24a3d-l2kztl`)
- **Commit final de la entrega:** es el último commit de esa rama. Pongo su SHA completo en el mensaje de
  entrega, porque un archivo no puede contener el hash del commit que lo guarda.

## 🔢 Mis dos versiones: v1 y v2
Esta tabla tiene todo lo necesario para seguir la historia de cada versión, desde el código hasta el
modelo que se descarga:

| | v1 | v2 |
|---|---|---|
| **Qué es** | Regresión logística básica (one-hot + escalado), umbral 0,50 | HistGradientBoosting, umbral 0,65 elegido con validación |
| **Commit de la solución** | `330015e9da6c114e870f444c720d585952fbfaac` (tag `v1`) | `2ae6479c9528c7dcc2342908618d4b432c8131fb` (tag `v2`) |
| **Commit en que se entrenó** | `9a17cd225142061547ee474b9720e1113bfbea92` | `9a17cd225142061547ee474b9720e1113bfbea92` |
| **Datos preparados (DVC)** | `data/prepared.dvc` · md5 `53545c819eb4a624ddf5da61d08666c7.dir` | el mismo |
| **Datos originales (DVC)** | `data/raw.dvc` · md5 `fe93051ae209ffb0128a31e7a478c009.dir` | el mismo |
| **Run de MLflow** | `d918b87582934fdb85c0567161272c9c` | `a5ca54703ec744fba37ff99212277e71` |
| **Repo en Hugging Face** | https://huggingface.co/Daniela201000/defiende-el-servidor | el mismo |
| **Revisión en Hugging Face (SHA)** | `691479df2d72c01e99bd1b32c9d5b82368ff8b70` (tag `v1`) | `e2f7bfdd932f383d7cf40c5413d5d407408889ba` (tag `v2`) |
| **Huella del modelo (SHA-256 de `model.joblib`)** | `2b350cdc7d82316f448ed5b1a498a43798201a6a9654ecd7a42191aa8ab09c6c` | `ac60460c5b0508bea7d5504a8d0dff24cda1d2fafca94825b4ff1e369d5f2b62` |
| **Umbral** | 0,50 | 0,65 |
| **Validación: recall / precision / FPR / macro-F1** | 0,9886 / 0,9223 / 0,1776 / 0,9225 | 0,9540 / 0,9760 / 0,0499 / 0,9463 |

Algunas aclaraciones:
- En Hugging Face hay una revisión más, `45323bdaff1232c44ca6f4c8273b12a7de78426c`. En ella solo agregué
  el resultado de test a la model card de v2. El modelo no cambió: `model.joblib` es idéntico al de la
  revisión evaluada (`e2f7bfdd…`).
- Creé los tags `v1` y `v2` en Git sobre los commits de la tabla. Si no aparecen en GitHub, los commits
  completos bastan para identificar cada versión.
- Los datos originales (v1 de datos) quedaron fijados en el commit `1b826c1f1700cad8881f4c046b83ed0c47938f73`,
  y los datos preparados con las particiones (v2 de datos), en el commit `9a17cd225142061547ee474b9720e1113bfbea92`.

## 🗂️ Dónde está cada evidencia
| Evidencia | Dónde encontrarla |
|---|---|
| Auditoría de datos y particiones | `evidencias/auditoria.json` y `data/prepared/manifest.json` (este último llega con `dvc pull`) |
| Tabla de los 4 experimentos | `evidencias/experimentos.csv` |
| Exportación de MLflow | `evidencias/mlflow/`: un `index.json` y una carpeta por run con su `run.json`, la configuración y las puntuaciones de validación |
| Selección de v2 y congelación | `evidencias/seleccion.json` y la curva de umbrales `evidencias/seleccion_umbral.csv`. La congelé en el commit `2ae6479c…`, antes de mirar el test |
| Test final | `evidencias/metricas-test.json` y `evidencias/predicciones-test.csv` (commit `701a209c…`) |
| Análisis de errores por tipo de ataque | `evidencias/errores_validacion_por_categoria.json` |
| Referencias de Hugging Face | `evidencias/huggingface.json` |
| Prueba de recuperación | `evidencias/recuperacion/` (detalle abajo) |
| Paquetes de los modelos publicados | `models/release/v1` y `models/release/v2` |

**Sobre la prueba de recuperación:** la corrí el 30-10-2026 sobre el commit `3da339d1`, en una carpeta y un
entorno nuevos, sin cachés. **Salió OK**: las etiquetas de v1 y v2 coincidieron en las 25 filas, y la
diferencia máxima en las probabilidades fue de 4,9e-11, muy por debajo de la tolerancia de 1e-9. En la
carpeta están la muestra fija (`muestra_validacion.csv`), las predicciones antes de publicar
(`pred_v*_original.csv`) y después de recuperar (`pred_v*_recuperado.csv`), el log
(`log_recuperacion.txt`) y el resultado (`recuperacion_resultado.json`).

## 💾 Dónde están los datos (DVC) y cómo acceder
- Guardé el almacenamiento de DVC en un repositorio de datasets de Hugging Face:
  https://huggingface.co/datasets/Daniela201000/defiende-el-servidor-dvc (carpeta `dvcstore/`, commit
  `7e9779311b8530acc76a454728321492f254c88a`). Ya está configurado como remote por defecto (`hfstore`)
  en `.dvc/config`.
- Es **público**, así que `dvc pull` funciona sin pedir credenciales. No compartí ningún token.

## ⌨️ Comandos principales
Todos están explicados paso a paso en el `README.md`. En resumen:
- **Instalar:** `pip install -r requirements.txt`
- **Bajar los datos:** `dvc pull` (más o menos 1 minuto)
- **Entrenar un experimento:** `python src/train.py configs/experiments/<experimento>.yaml`
- **Predecir:** `python predict.py --model-dir <carpeta descargada de HF> --input <csv> --output <csv>`
- **Prueba de recuperación:** `bash scripts/recuperacion.sh <commit>` (unos 8 minutos, casi todo es la
  instalación de dependencias)

## ⏱️ Tiempo y recursos que medí
Todo corrió en una máquina con 4 vCPU, 15 GB de RAM y sin GPU:
- Preparar los datos (`prepare.py`): unos 22 segundos.
- Entrenar con 140.274 filas: exp1 9 s, exp2 6 s, exp3 7 s y exp4 4 s.
- Evaluar en test: menos de 5 segundos.

## ⚠️ Limitaciones que conozco
- En test, la tasa de falsas alarmas (FPR) subió a 0,181, frente a 0,050 en validación. No cambié nada
  para mejorarla: la dejé como salió y la explico en el informe.
- Cerca del 10 % de las filas de test ya aparecían idénticas en entrenamiento o validación, así que el
  resultado de test es algo optimista.
- El dataset es de 2015 y se hizo en laboratorio, así que no demuestra que el modelo funcione en una red real
  de hoy.

## 🤖 Uso de IA, código externo y mi contribución
- **Uso de IA:** usé Claude Code (asistente de IA de Anthropic) para escribir el código, correr los
  experimentos, publicar en Hugging Face y redactar la documentación y el informe. Las verificaciones
  quedaron automatizadas y registradas en las evidencias: comparé las huellas SHA-256 con las del anexo,
  confirmé que ninguna fila idéntica quedó repartida entre entrenamiento y validación, recalculé las
  métricas de test con `recompute_metrics.py` e hice la prueba de recuperación en un entorno nuevo (OK).
- **Código externo:** no copié código de otras fuentes. Usé las bibliotecas scikit-learn, pandas, MLflow,
  DVC y huggingface_hub, cada una con su licencia.
- **Mi contribución:** fue un trabajo individual. Definí el alcance, creé y configuré las cuentas y
  credenciales (GitHub y Hugging Face), preparé el entorno de trabajo, dirigí el trabajo con el asistente
  y revisé las decisiones y los resultados.
