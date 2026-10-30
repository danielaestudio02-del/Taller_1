"""Prepara los datos: verifica huellas, crea grupos por entradas idénticas y particiona
entrenamiento/validación solo a partir del archivo de entrenamiento.

Uso: python src/prepare.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import (ROOT, TARGET, data_config, feature_columns, frame_sha256, read_raw,
                    row_hashes, sha256_file, write_json)


def load_verified(cfg: dict, name: str) -> pd.DataFrame:
    path = ROOT / cfg["raw"][name]
    digest = sha256_file(path)
    if digest != cfg["expected_sha256"][name]:
        raise SystemExit(f"SHA-256 inesperado en {path}: {digest}")
    df = read_raw(path)
    if len(df) != cfg["expected_rows"][name]:
        raise SystemExit(f"{path}: {len(df)} filas, se esperaban {cfg['expected_rows'][name]}")
    if not df["id"].is_unique:
        raise SystemExit(f"{path}: la columna id no es única")
    return df


def main() -> None:
    cfg = data_config()
    train = load_verified(cfg, "train")
    test = load_verified(cfg, "test")
    feats = feature_columns(train)

    # row_id identifica archivo + id original (único dentro de cada archivo).
    train.insert(0, "row_id", "train:" + train["id"].astype(str))
    test.insert(0, "row_id", "test:" + test["id"].astype(str))
    train["group_id"] = row_hashes(train, feats)
    test["group_id"] = row_hashes(test, feats)

    sgkf = StratifiedGroupKFold(
        n_splits=cfg["split"]["n_splits"], shuffle=True, random_state=cfg["split"]["seed"]
    )
    folds = list(sgkf.split(train[feats], train[TARGET], groups=train["group_id"]))
    _, val_idx = folds[cfg["split"]["val_fold"]]
    train["split"] = "train"
    train.loc[train.index[val_idx], "split"] = "val"
    test["split"] = "test"

    tr = train[train["split"] == "train"].reset_index(drop=True)
    va = train[train["split"] == "val"].reset_index(drop=True)
    shared = set(tr["group_id"]) & set(va["group_id"])
    if shared:
        raise SystemExit(f"Fuga: {len(shared)} grupos repartidos entre train y val")

    out = cfg["prepared"]
    (ROOT / out["dir"]).mkdir(parents=True, exist_ok=True)
    for name, df in (("train", tr), ("val", va), ("test", test)):
        df.to_parquet(ROOT / out[name], index=False)

    manifest = {
        "transformation": (
            "Se leen los CSV publicados (utf-8-sig), se verifica SHA-256 y filas, se agrega "
            "row_id = <archivo>:<id>, group_id = hash de las variables predictoras y split. "
            "Train/val salen solo del archivo de entrenamiento con StratifiedGroupKFold "
            "(estratifica label, agrupa entradas idénticas). Test es el archivo publicado completo. "
            "No se eliminan filas: duplicados y contradicciones se conservan dentro de un solo split."
        ),
        "seed": cfg["split"]["seed"],
        "features": feats,
        "splits": {
            name: {
                "file": out[name],
                "rows": int(len(df)),
                "attack_rows": int(df[TARGET].sum()),
                "groups": int(df["group_id"].nunique()),
                "content_sha256": frame_sha256(df),
                "row_ids_sha256": frame_sha256(df[["row_id"]]),
            }
            for name, df in (("train", tr), ("val", va), ("test", test))
        },
        "shared_groups_train_val": 0,
    }
    write_json(out["manifest"], manifest)
    print({k: v["rows"] for k, v in manifest["splits"].items()})


if __name__ == "__main__":
    main()
