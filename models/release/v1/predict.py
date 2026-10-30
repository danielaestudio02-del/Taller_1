"""Predicción con un paquete de modelo (carpeta local o descargada de Hugging Face).

Autocontenido: se copia tal cual dentro de cada revisión del modelo.

Uso:
  python predict.py --model-dir <carpeta> --input filas.csv --output predicciones.csv
La entrada necesita las variables listadas en config.json["features"]; si trae row_id se conserva.
Regla: y_pred = 1 (ataque) si score_attack >= threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_package(model_dir: str | Path) -> tuple[object, dict]:
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model_path = model_dir / cfg["model_file"]
    digest = sha256_file(model_path)
    if digest != cfg["model_sha256"]:
        raise SystemExit(f"Hash del modelo no coincide: {digest} != {cfg['model_sha256']}")
    return joblib.load(model_path), cfg


def predict_frame(model, cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    scores = model.predict_proba(df[cfg["features"]])[:, 1]
    out = pd.DataFrame({"row_id": df["row_id"] if "row_id" in df else np.arange(len(df))})
    out["score_attack"] = scores
    out["y_pred"] = (scores >= cfg["threshold"]).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=".")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    model, cfg = load_package(args.model_dir)
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    out = predict_frame(model, cfg, df)
    out.to_csv(args.output, index=False, float_format="%.10g")
    print(f"{cfg['version']} sha256={cfg['model_sha256'][:12]} umbral={cfg['threshold']} filas={len(out)} "
          f"ataques={int(out['y_pred'].sum())}")


if __name__ == "__main__":
    main()
