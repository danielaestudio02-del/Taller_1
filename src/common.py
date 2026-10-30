"""Utilidades compartidas: rutas, configuración, hashes, métricas y trazabilidad."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

TARGET = "label"
# Columnas que nunca entran al modelo: id identifica filas, attack_cat revela la respuesta.
EXCLUDED = ["id", "attack_cat", "label"]
CATEGORICAL = ["proto", "service", "state"]
POSITIVE_CLASS = 1  # 1 = ataque


def load_yaml(path: str | Path) -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_config() -> dict:
    return load_yaml("configs/data.yaml")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(path: str | Path) -> pd.DataFrame:
    # Los CSV publicados traen BOM UTF-8 en la cabecera.
    return pd.read_csv(ROOT / path, encoding="utf-8-sig")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDED and c not in ("row_id", "group_id", "split")]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(df) if c not in CATEGORICAL]


def row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Huella por fila de las variables predictoras (sin id ni etiquetas)."""
    return pd.util.hash_pandas_object(df[cols], index=False).astype("uint64")


def frame_sha256(df: pd.DataFrame) -> str:
    """Huella de contenido de un DataFrame, independiente del formato de archivo."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_is_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    )
    return bool(out.strip())


def dvc_ref(dvc_file: str) -> dict:
    """Referencia de datos DVC: md5 del archivo .dvc y commit Git que lo fija."""
    meta = yaml.safe_load((ROOT / dvc_file).read_text())
    out = meta["outs"][0]
    commit = subprocess.check_output(
        ["git", "log", "-n", "1", "--format=%H", "--", dvc_file], cwd=ROOT, text=True
    ).strip()
    return {"dvc_file": dvc_file, "md5": out["md5"], "path": out["path"], "git_commit": commit}


def safe_div(num: float, den: float) -> float:
    """Denominador cero -> 0.0 (se documenta en el informe y en cada registro)."""
    return float(num) / float(den) if den else 0.0


def binary_metrics(y_true, y_pred) -> dict:
    """Métricas con ataque (1) como clase positiva. Denominador cero -> 0.0."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = safe_div(tp, tp + fn)
    precision = safe_div(tp, tp + fp)
    f1_attack = safe_div(2 * precision * recall, precision + recall)
    # F1 de la clase normal: misma fórmula con normal como positiva.
    recall_n = safe_div(tn, tn + fp)
    precision_n = safe_div(tn, tn + fn)
    f1_normal = safe_div(2 * precision_n * recall_n, precision_n + recall_n)
    return {
        "recall_attack": recall,
        "precision_attack": precision,
        "fpr": safe_div(fp, fp + tn),
        "f1_attack": f1_attack,
        "macro_f1": (f1_attack + f1_normal) / 2,
        "accuracy": safe_div(tp + tn, len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def apply_threshold(scores, threshold: float) -> np.ndarray:
    """Regla de decisión: ataque si score_attack >= umbral."""
    return (np.asarray(scores) >= threshold).astype(int)


def write_json(path: str | Path, obj) -> None:
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
