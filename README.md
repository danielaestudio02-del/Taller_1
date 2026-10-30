# 🛡️ Defiende el servidor

**Taller evaluable de MLOps · 2026-II · Universidad Externado de Colombia**
Daniela González Ortiz · Grupo A

En este proyecto entrené un modelo que clasifica conexiones de red como **normales (0)** o **ataques (1)**
con el dataset UNSW-NB15. Pero lo más importante del taller no es la métrica: es demostrar que puedo
**versionar todo** (código, datos, experimentos y modelos) y **recuperar cualquier versión** aunque pierda
mi carpeta de trabajo.

> 📌 Si buscas los identificadores exactos (commits, SHA, run IDs, hashes), están en [`entrega.md`](entrega.md).

---

## 🔗 Cómo se conecta todo

La idea que guió todo el proyecto es que cada pieza apunte a la siguiente con un identificador, no con un nombre:

```
Código (commit Git) → Datos (versión DVC) → Experimento (run de MLflow) → Modelo (revisión en Hugging Face)
```

| Herramienta | Qué versiono ahí | Dónde está |
|---|---|---|
| **Git + GitHub** | Código, configuraciones, documentación y los archivos `.dvc` | Este repositorio |
| **DVC** | Datos originales y datos preparados (con las particiones) | [Hugging Face Dataset](https://huggingface.co/datasets/Daniela201000/defiende-el-servidor-dvc) |
| **MLflow** | Los 4 experimentos: hipótesis, parámetros, métricas y modelos | Local, exportado en `evidencias/mlflow/` |
| **Hugging Face Hub** | Las dos versiones del modelo (v1 y v2), cada una con su model card | [Repo de modelos](https://huggingface.co/Daniela201000/defiende-el-servidor) |

---

## 📊 Resultados en pocas palabras

Probé cuatro configuraciones de dos familias de modelos. Todas usan las mismas particiones, y las
métricas son de **validación**:

| Experimento | Modelo | Recall ataque | Precision | FPR | Macro-F1 |
|---|---|---|---|---|---|
| exp1 | Regresión logística (referencia) | 0,989 | 0,922 | 0,178 | 0,922 |
| exp2 | Regresión logística + log1p | 0,977 | 0,935 | 0,144 | 0,927 |
| exp3 | Random Forest | 0,981 | 0,955 | 0,099 | 0,948 |
| exp4 | HistGradientBoosting | 0,976 | 0,961 | 0,084 | 0,950 |

**Mis dos versiones:**

- **v1:** regresión logística con umbral 0,5. Detecta casi todos los ataques, pero da falsa alarma en
  ~18 % del tráfico normal.
- **v2:** HistGradientBoosting con umbral 0,65, elegido **solo con validación**. Baja las falsas alarmas
  al 5 % a cambio de perder algo de recall (sobre todo en ataques tipo *Fuzzers*).

En el **test final** (evaluado una sola vez), v2 obtuvo un recall de 0,970, pero la FPR subió a 0,181. No
lo maquillé ni cambié el umbral: el tráfico normal del archivo de test es distinto al de entrenamiento, y
eso lo explico en el informe.

---

## 🚀 Cómo correr el proyecto

### 1. Lo que necesitas
- Python **3.13** (yo usé 3.13.17)
- Linux o macOS. Con un computador normal basta: no hace falta GPU.

### 2. Instalar
```bash
git clone https://github.com/danielaestudio02-del/Taller_1.git
cd Taller_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Bajar los datos con DVC
Los datos no están en GitHub (pesan mucho). Se descargan desde Hugging Face con un solo comando:
```bash
dvc pull
```
Para comprobar que llegaron bien:
```bash
sha256sum data/raw/*.csv   # tienen que coincidir con los de configs/data.yaml
dvc status                 # debe decir "Data and pipelines are up to date."
```

<details>
<summary>¿No funciona <code>dvc pull</code>? Hay un plan B</summary>

Descarga los dos CSV del mirror que aparece en `configs/data.yaml`, guárdalos en `data/raw/` y corre:
```bash
python src/prepare.py
```
Sale exactamente lo mismo (lo comprobé: `dvc status` no detecta cambios).
</details>

### 4. Repetir los experimentos
```bash
python src/prepare.py   # revisa los hashes y arma las particiones (semilla 42)
python src/audit.py     # genera evidencias/auditoria.json

python src/train.py configs/experiments/exp1_logreg_base.yaml
python src/train.py configs/experiments/exp2_logreg_log_c01.yaml
python src/train.py configs/experiments/exp3_random_forest.yaml
python src/train.py configs/experiments/exp4_hist_gradient_boosting.yaml

python src/export_mlflow.py   # exporta los runs y arma evidencias/experimentos.csv
```
Cada entrenamiento tarda unos segundos. Si quieres ver los runs en la interfaz de MLflow:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
💡 `train.py` no te deja entrenar si tienes cambios sin commit. Lo hice así a propósito, para que cada
run quede amarrado a un commit exacto.

### 5. Descargar un modelo de Hugging Face y predecir
Cada versión se descarga por su **SHA**, que es su identificador fijo:
```bash
# v1
hf download Daniela201000/defiende-el-servidor \
  --revision 691479df2d72c01e99bd1b32c9d5b82368ff8b70 --local-dir modelo_v1

python modelo_v1/predict.py --model-dir modelo_v1 \
  --input evidencias/recuperacion/muestra_validacion.csv --output pred_v1.csv
```
Para v2 es igual, solo cambia el SHA: `e2f7bfdd932f383d7cf40c5413d5d407408889ba`.

`predict.py` revisa que el hash del modelo sea el correcto antes de cargarlo y marca como ataque toda
conexión con `score_attack >= umbral` (el umbral viene en `config.json`).

### 6. Hacer la prueba de recuperación
Este script hace todo desde cero en una carpeta nueva: clona el repo, crea un entorno nuevo, baja los
datos con DVC, descarga v1 y v2 por su SHA y compara las predicciones con las que guardé antes:
```bash
bash scripts/recuperacion.sh <commit>
```
✅ Cuando la corrí, el resultado fue **OK**: las predicciones salieron idénticas. El log está en
`evidencias/recuperacion/`.

---

## 🧭 El flujo que seguí (para reproducir v1, v2 y el test)
```bash
python src/build_release.py v1   # arma el paquete de v1 (modelo, umbral, dependencias y model card)
python src/select_v2.py          # elige modelo y umbral de v2 SOLO con validación
python src/build_release.py v2
# ...commit de la selección (así queda "congelada" antes de mirar el test)...
python src/evaluate_test.py      # evaluación única en test
python src/recompute_metrics.py evidencias/predicciones-test.csv --threshold 0.65
```

Para publicar en Hugging Face usé `src/publish_hf.py` (`dvc`, `models` y `card-test`), con mi token
guardado como secreto, nunca dentro del repo.

---

## 📁 Qué hay en cada carpeta
```
configs/         configuración de datos, de los 4 experimentos y la regla de selección
src/             todo el código (preparar, auditar, entrenar, seleccionar, evaluar, predecir, publicar)
data/            solo los archivos .dvc; los datos reales llegan con dvc pull
models/release/  los paquetes de v1 y v2 que subí a Hugging Face
evidencias/      auditoría, experimentos, selección, test, exportación de MLflow y recuperación
informe/         el informe en PDF (y su HTML)
scripts/         prueba de recuperación, generación del PDF y empaquetado del ZIP
```

---

## ⚠️ Limitaciones (para ser honesta)
- UNSW-NB15 es de 2015 y se generó en laboratorio, así que esto **no** demuestra que el modelo funcione
  en una red real de hoy.
- Cerca del 10 % de las filas de test ya aparecían idénticas en entrenamiento, así que el test es un
  poco optimista.
- v2 se equivoca más con los ataques tipo *Fuzzers*, que se parecen mucho al tráfico normal.

## 🤖 Uso de IA
Usé Claude Code (un asistente de IA) para escribir el código, correr los experimentos y redactar la
documentación. Los detalles de qué hice yo y qué se verificó están en [`entrega.md`](entrega.md).

## 📚 Fuente de los datos
Moustafa, N. y Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.*
Página oficial: https://research.unsw.edu.au/projects/unsw-nb15-dataset
