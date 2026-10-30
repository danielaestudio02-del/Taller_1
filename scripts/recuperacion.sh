#!/usr/bin/env bash
# Prueba de recuperación en carpeta y entorno NUEVOS (sin cachés de la ejecución original).
# Uso: bash scripts/recuperacion.sh <commit-o-rama> 
# Lee evidencias/huggingface.json del repo clonado para obtener repo y SHAs de v1/v2.
set -euo pipefail
REV="${1:?indica el commit Git a recuperar}"
REPO_URL="${REPO_URL:-https://github.com/danielaestudio02-del/Taller_1.git}"
BASE="$(cd "$(dirname "$0")/.." && pwd)"
WS="${WS:-$BASE/recovery_workspace}"
OUT="$BASE/evidencias/recuperacion"
LOG="$OUT/log_recuperacion.txt"
rm -rf "$WS"; mkdir -p "$WS" "$OUT"
exec > >(tee "$LOG") 2>&1
run() { echo "+ $*"; "$@"; }
echo "== Prueba de recuperación $(date -u +%FT%TZ) en $WS"
cd "$WS"
run git clone --quiet "$REPO_URL" proyecto
cd proyecto
run git checkout --quiet "$REV"
echo "commit: $(git rev-parse HEAD)"
run python3 -m venv .venv
. .venv/bin/activate
run pip install --quiet --no-cache-dir -r requirements.txt
# caché DVC nueva dentro de la carpeta nueva
run dvc config --local cache.dir "$WS/dvc-cache"
run dvc pull
run sha256sum data/raw/UNSW_NB15_training-set.csv data/raw/UNSW_NB15_testing-set.csv
run dvc status
export HF_HOME="$WS/hf-home"   # caché de Hugging Face nueva
python - <<'PY'
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from common import data_config, sha256_file, frame_sha256
from predict import load_package, predict_frame
from huggingface_hub import snapshot_download

cfg = data_config()
checks = {"raw_sha256": {}, "prepared_content_sha256": {}}
for k in ("train", "test"):
    d = sha256_file(cfg["raw"][k]); checks["raw_sha256"][k] = {"sha256": d, "ok": d == cfg["expected_sha256"][k]}
man = json.loads(Path(cfg["prepared"]["manifest"]).read_text())
for k in ("train", "val", "test"):
    d = frame_sha256(pd.read_parquet(cfg["prepared"][k]))
    checks["prepared_content_sha256"][k] = {"sha256": d, "ok": d == man["splits"][k]["content_sha256"]}
hf = json.loads(Path("evidencias/huggingface.json").read_text())
sample = pd.read_csv("evidencias/recuperacion/muestra_validacion.csv")
result = {"data_checks": checks, "tolerance_abs_score": 1e-9, "versions": {}}
for v in ("v1", "v2"):
    sha = hf[v]["revision_sha"]
    d = snapshot_download(repo_id=hf["model_repo"]["repo_id"], revision=sha)
    model, mcfg = load_package(d)  # verifica SHA-256 de model.joblib
    pred = predict_frame(model, mcfg, sample)
    pred.to_csv(f"evidencias/recuperacion/pred_{v}_recuperado.csv", index=False, float_format="%.10g")
    orig = pd.read_csv(f"evidencias/recuperacion/pred_{v}_original.csv")
    diff = float(np.abs(orig["score_attack"].values - pred["score_attack"].values).max())
    same = bool((orig["y_pred"].values == pred["y_pred"].values).all() and (orig["row_id"] == pred["row_id"]).all())
    result["versions"][v] = {"hf_revision_sha": sha, "model_sha256": mcfg["model_sha256"],
                             "model_sha256_expected": hf[v]["model_sha256"], "threshold": mcfg["threshold"],
                             "rows": len(pred), "labels_equal": same, "max_abs_score_diff": diff,
                             "ok": same and diff <= 1e-9 and mcfg["model_sha256"] == hf[v]["model_sha256"]}
    print(v, result["versions"][v])
result["ok"] = all(x["ok"] for x in result["versions"].values()) and all(
    c["ok"] for g in checks.values() for c in g.values())
Path("recuperacion_resultado.json").write_text(json.dumps(result, indent=2))
print("RESULTADO:", "OK" if result["ok"] else "FALLA")
PY
cp recuperacion_resultado.json evidencias/recuperacion/pred_v*_recuperado.csv "$OUT/"
echo "== Fin $(date -u +%FT%TZ)"
