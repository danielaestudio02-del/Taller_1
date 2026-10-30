"""Auditoría de datos crudos y particiones -> evidencias/auditoria.json

Uso: python src/audit.py   (requiere haber ejecutado src/prepare.py)
"""
from __future__ import annotations

import json

import pandas as pd

from common import (CATEGORICAL, ROOT, TARGET, data_config, feature_columns, read_raw,
                    row_hashes, sha256_file, write_json)


def describe_file(path: str, expected_sha: str) -> tuple[dict, pd.DataFrame]:
    df = read_raw(path)
    feats = feature_columns(df)
    digest = sha256_file(ROOT / path)
    h = row_hashes(df, feats)
    groups = df.assign(_h=h).groupby("_h")[TARGET].agg(["size", "nunique"])
    contradictory = groups[groups["nunique"] > 1]
    info = {
        "file": path,
        "size_bytes": (ROOT / path).stat().st_size,
        "sha256": digest,
        "sha256_matches_annex": digest == expected_sha,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v},
        "label_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "attack_cat_counts": {str(k): int(v) for k, v in df["attack_cat"].value_counts().items()},
        "id_unique": bool(df["id"].is_unique),
        "duplicates_by_features": {
            "unique_inputs": int(len(groups)),
            "rows_in_duplicated_groups": int(groups.loc[groups["size"] > 1, "size"].sum()),
            "duplicate_rows_beyond_first": int(len(df) - len(groups)),
            "duplicate_rate": round(1 - len(groups) / len(df), 6),
        },
        "contradictions": {
            "groups_with_both_labels": int(len(contradictory)),
            "rows_in_contradictory_groups": int(contradictory["size"].sum()),
        },
        "categorical_levels": {c: int(df[c].nunique()) for c in CATEGORICAL},
    }
    return info, df.assign(_h=h)


def main() -> None:
    cfg = data_config()
    train_info, train_raw = describe_file(cfg["raw"]["train"], cfg["expected_sha256"]["train"])
    test_info, test_raw = describe_file(cfg["raw"]["test"], cfg["expected_sha256"]["test"])
    manifest = json.loads((ROOT / cfg["prepared"]["manifest"]).read_text())

    tr = pd.read_parquet(ROOT / cfg["prepared"]["train"], columns=["row_id", "group_id"])
    va = pd.read_parquet(ROOT / cfg["prepared"]["val"], columns=["row_id", "group_id"])
    te = pd.read_parquet(ROOT / cfg["prepared"]["test"], columns=["row_id", "group_id"])
    g_tr, g_va, g_te = set(tr["group_id"]), set(va["group_id"]), set(te["group_id"])

    def overlap(df: pd.DataFrame, groups: set) -> dict:
        n = int(df["group_id"].isin(groups).sum())
        return {"rows": n, "fraction_of_rows": round(n / len(df), 6)}

    train_cats = pd.read_parquet(ROOT / cfg["prepared"]["train"], columns=CATEGORICAL)
    unseen = {c: sorted(set(test_raw[c].unique()) - set(train_cats[c].unique())) for c in CATEGORICAL}

    audit = {
        "source": cfg["source"],
        "files": {"train": train_info, "test": test_info},
        "target": {"column": TARGET, "encoding": {"0": "normal", "1": "ataque"}, "positive_class": 1},
        "features_used": manifest["features"],
        "features_excluded": {
            "id": "identificador de fila; solo auditoría y row_id",
            "attack_cat": "categoría del ataque; revela la respuesta (fuga)",
            "label": "variable objetivo",
        },
        "cleaning_policy": {
            "missing": "No hay faltantes en los archivos publicados; el pipeline incluye imputación "
                       "(mediana / más frecuente) ajustada solo con train por robustez.",
            "duplicates": "Se conservan (representan frecuencia real de tráfico) pero todas las filas "
                          "con entradas idénticas quedan en el mismo split (StratifiedGroupKFold por "
                          "hash de entradas).",
            "contradictions": "Grupos con entradas idénticas y etiquetas distintas se conservan: son ruido "
                              "irreducible que acota el error mínimo; no se reetiquetan ni se eliminan, y "
                              "quedan dentro de un único split.",
            "test": "Archivo de test completo, sin eliminar filas. Solo se audita esquema, duplicados y cruces.",
        },
        "split": {
            "method": "StratifiedGroupKFold(n_splits=5, shuffle=True), pliegue 0 = validación",
            "seed": manifest["seed"],
            "sizes_and_hashes": manifest["splits"],
            "shared_groups_train_val": len(g_tr & g_va),
        },
        "cross_partition_overlap": {
            "description": "Filas cuyas entradas (variables predictoras) aparecen idénticas en otra partición",
            "test_rows_seen_in_train": overlap(te, g_tr),
            "test_rows_seen_in_val": overlap(te, g_va),
            "test_rows_seen_in_train_or_val": overlap(te, g_tr | g_va),
            "val_rows_seen_in_train": overlap(va, g_tr),
        },
        "test_categorical_levels_unseen_in_train_split": {k: [str(x) for x in v] for k, v in unseen.items()},
    }
    write_json("evidencias/auditoria.json", audit)
    print(json.dumps({k: audit[k] for k in ("cross_partition_overlap",)}, indent=2))
    print("dups train", train_info["duplicates_by_features"], train_info["contradictions"])
    print("dups test", test_info["duplicates_by_features"], test_info["contradictions"])


if __name__ == "__main__":
    main()
