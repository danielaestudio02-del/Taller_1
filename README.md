# Defiende el servidor · MLOps 2026-II

Clasificación binaria **normal (0) / ataque (1)** con UNSW-NB15. El foco es la trazabilidad
y la recuperación de versiones:

```
Código (commit Git) → Datos (archivo .dvc / md5) → Experimento (run ID MLflow) → Modelo (SHA en Hugging Face)
```

El índice de la entrega, con todos los identificadores, está en [`entrega.md`](entrega.md).

## Requisitos
- Python **3.13** (probado con 3.13.17), Linux/macOS, CPU (4 núcleos, 15 GB de RAM bastan).
- Dependencias fijadas en `requirements.txt`.

## Instalación
```bash
git clone https://github.com/danielaestudio02-del/Taller_1.git
cd Taller_1
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Recuperar los datos (DVC)
El almacenamiento DVC es un repositorio *dataset* de Hugging Face (remote `hfstore`, configurado en
`.dvc/config`). Si el repositorio es público, basta con:
```bash
dvc pull                      # data/raw (CSV originales) y data/prepared (particiones)
sha256sum data/raw/*.csv      # deben coincidir con configs/data.yaml
dvc status                    # "Data and pipelines are up to date."
```
Si el repositorio es privado, primero agrega tu token **solo en la configuración local**, que no se versiona:
```bash
dvc remote modify --local hfstore auth custom
dvc remote modify --local hfstore custom_auth_header Authorization
dvc remote modify --local hfstore password "Bearer $HF_TOKEN"
```
Alternativa sin DVC remoto: descargar los CSV del mirror de `configs/data.yaml` en `data/raw/` y ejecutar
`python src/prepare.py`. El resultado es idéntico (`dvc status` no muestra cambios).

## Reproducir los experimentos
```bash
python src/prepare.py   # verifica SHA-256 y crea data/prepared (particiones fijas, semilla 42)
python src/audit.py     # evidencias/auditoria.json
python src/train.py configs/experiments/exp1_logreg_base.yaml          # ~10 s
python src/train.py configs/experiments/exp2_logreg_log_c01.yaml
python src/train.py configs/experiments/exp3_random_forest.yaml
python src/train.py configs/experiments/exp4_hist_gradient_boosting.yaml
python src/export_mlflow.py   # evidencias/mlflow/ y evidencias/experimentos.csv
mlflow ui --backend-store-uri sqlite:///mlflow.db   # opcional: explorar los runs
```
`train.py` se niega a correr con cambios sin commit, para que el commit de cada run sea exacto.

## Selección, paquetes y test
```bash
python src/build_release.py v1   # models/release/v1 (modelo, config + umbral, deps, predict.py, model card)
python src/select_v2.py          # elige modelo y umbral SOLO con validación -> evidencias/seleccion.json
python src/build_release.py v2
# (commit de seleccion.json = congelación)  luego, una sola vez:
python src/evaluate_test.py      # evidencias/metricas-test.json y predicciones-test.csv
python src/recompute_metrics.py evidencias/predicciones-test.csv --threshold 0.65
```

## Descargar cada modelo de Hugging Face y predecir
Los SHA completos de v1 y v2 están en `evidencias/huggingface.json` y en `entrega.md`.
```bash
REPO=$(python -c "import json;print(json.load(open('evidencias/huggingface.json'))['model_repo']['repo_id'])")
SHA_V1=$(python -c "import json;print(json.load(open('evidencias/huggingface.json'))['v1']['revision_sha'])")
hf download "$REPO" --revision "$SHA_V1" --local-dir modelo_v1
python modelo_v1/predict.py --model-dir modelo_v1 \
  --input evidencias/recuperacion/muestra_validacion.csv --output pred_v1.csv
```
Haz lo mismo para v2 con `['v2']['revision_sha']`. `predict.py` verifica el SHA-256 de `model.joblib`
y aplica la regla `ataque si score_attack >= umbral`, con el umbral de `config.json`.

## Prueba de recuperación en un entorno nuevo
```bash
bash scripts/recuperacion.sh <commit-final>
```
El script clona el repositorio en `recovery_workspace/`, crea un venv nuevo, usa una caché DVC y un
`HF_HOME` nuevos, ejecuta `dvc pull` y verifica las huellas. Luego descarga v1 y v2 por SHA, predice
sobre `muestra_validacion.csv` (25 filas de validación) y compara contra las predicciones guardadas.
El log y el resultado quedan en `evidencias/recuperacion/`.

## Publicación (la hace el autor, una sola vez)
Con `HF_TOKEN` (permiso de escritura), `HF_REPO_ID` y `HF_DVC_REPO_ID` definidos en el entorno:
```bash
python src/publish_hf.py dvc        # dvc push al repo dataset y fija el remote hfstore
python src/publish_hf.py models     # v1 y v2 como dos commits distintos (+ tags v1/v2)
python src/publish_hf.py card-test  # agrega el resultado de test a la model card de v2
```

## Estructura
```
configs/      data.yaml, selection.yaml, experiments/*.yaml (+ conclusiones.yaml)
src/          prepare, audit, pipeline, train, export_mlflow, select_v2, build_release,
              evaluate_test, recompute_metrics, predict, publish_hf
data/         raw.dvc y prepared.dvc (los datos los trae dvc pull)
models/release/v1, v2   paquetes publicados en Hugging Face
evidencias/   auditoria.json, experimentos.csv, seleccion.json, seleccion_umbral.csv,
              metricas-test.json, predicciones-test.csv, huggingface.json, mlflow/, recuperacion/
scripts/      recuperacion.sh
```
