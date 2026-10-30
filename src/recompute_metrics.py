"""Recalcula métricas desde un CSV de predicciones (row_id, y_true, score_attack, y_pred).

Uso: python src/recompute_metrics.py evidencias/predicciones-test.csv [--threshold 0.65]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common import apply_threshold, binary_metrics

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--threshold", type=float, help="si se da, recalcula y_pred desde score_attack")
args = ap.parse_args()
df = pd.read_csv(args.csv)
y_pred = apply_threshold(df["score_attack"], args.threshold) if args.threshold is not None else df["y_pred"]
if args.threshold is not None:
    print("y_pred coincide con score>=umbral:", bool((y_pred == df["y_pred"]).all()))
print(json.dumps({"rows": len(df), "unique_row_ids": int(df["row_id"].nunique()),
                  **binary_metrics(df["y_true"], y_pred)}, indent=2))
