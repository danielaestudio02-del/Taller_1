"""Publica en Hugging Face Hub: modelos v1/v2 (repo de modelos) y almacenamiento DVC (repo dataset).

Requiere variables de entorno (nunca escribir el token en archivos):
  HF_TOKEN       token con permiso de escritura (opcional si el entorno lo inyecta como
                 "secreto de red" para huggingface.co)
  HF_REPO_ID     repo de modelos, p. ej. usuario/defiende-el-servidor
  HF_DVC_REPO_ID repo dataset para DVC, p. ej. usuario/defiende-el-servidor-dvc
  HF_PRIVATE     "1" para crear repos privados (por defecto públicos)

Uso:
  python src/publish_hf.py dvc        # dvc push + sube el almacenamiento y fija el remote http
  python src/publish_hf.py models     # sube v1 y v2 como revisiones distintas (+ tags v1/v2)
  python src/publish_hf.py card-test  # agrega el resultado de test a la model card (no toca el modelo)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from huggingface_hub import HfApi

from common import ROOT, sha256_file, write_json

HF_JSON = ROOT / "evidencias/huggingface.json"


def api() -> HfApi:
    # Sin HF_TOKEN se asume que el proxy del entorno agrega la credencial (secreto de red).
    return HfApi(token=os.environ.get("HF_TOKEN") or False)


def private() -> bool:
    return os.environ.get("HF_PRIVATE") == "1"


def load_state() -> dict:
    return json.loads(HF_JSON.read_text()) if HF_JSON.exists() else {}


def publish_dvc() -> None:
    repo = os.environ["HF_DVC_REPO_ID"]
    a = api()
    a.create_repo(repo, repo_type="dataset", private=private(), exist_ok=True)
    store = ROOT / "dvcstore"
    subprocess.run(["dvc", "remote", "add", "-f", "localstore", str(store)], cwd=ROOT, check=True)
    subprocess.run(["dvc", "push", "-r", "localstore"], cwd=ROOT, check=True)
    subprocess.run(["dvc", "remote", "remove", "localstore"], cwd=ROOT, check=True)
    info = a.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(store), path_in_repo="dvcstore",
                           commit_message="Almacenamiento DVC: datos originales y preparados (UNSW-NB15)")
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/dvcstore"
    subprocess.run(["dvc", "remote", "add", "-d", "-f", "hfstore", url], cwd=ROOT, check=True)
    state = load_state()
    state["dvc_storage"] = {"repo_id": repo, "repo_type": "dataset", "url": url, "commit_sha": info.oid,
                            "private": private()}
    write_json(HF_JSON, state)
    print("DVC remote:", url, info.oid)


def publish_models() -> None:
    repo = os.environ["HF_REPO_ID"]
    a = api()
    a.create_repo(repo, repo_type="model", private=private(), exist_ok=True)
    state = load_state()
    state["model_repo"] = {"repo_id": repo, "url": f"https://huggingface.co/{repo}", "private": private()}
    for v in ("v1", "v2"):
        folder = ROOT / "models/release" / v
        cfg = json.loads((folder / "config.json").read_text())
        assert sha256_file(folder / "model.joblib") == cfg["model_sha256"]
        info = a.upload_folder(repo_id=repo, folder_path=str(folder), delete_patterns="*",
                               commit_message=f"{v}: {cfg['experiment']} umbral {cfg['threshold']} "
                                              f"(run {cfg['references']['mlflow_run_id']})")
        try:
            a.create_tag(repo, tag=v, revision=info.oid, tag_message=f"Solución {v}")
        except Exception as e:  # el tag ya existe en una republicación
            print("tag", v, "no creado:", e)
        state[v] = {"revision_sha": info.oid, "model_sha256": cfg["model_sha256"], "threshold": cfg["threshold"],
                    "mlflow_run_id": cfg["references"]["mlflow_run_id"], "tag": v}
        print(v, info.oid)
    write_json(HF_JSON, state)
    for f in ("evidencias/seleccion.json", "evidencias/metricas-test.json"):
        d = json.loads((ROOT / f).read_text())
        refs = d["references"] if "references" in d else d
        refs["huggingface"] = {"repo_id": repo, "revision_sha": state["v2"]["revision_sha"], "tag": "v2",
                               "nota": "publicado después de congelar; publicar no cambia model_sha256"}
        write_json(ROOT / f, d)


def publish_card_test() -> None:
    state = load_state()
    repo, v2 = state["model_repo"]["repo_id"], state["v2"]
    t = json.loads((ROOT / "evidencias/metricas-test.json").read_text())
    assert t["model_sha256"] == v2["model_sha256"]
    m = t["metrics"]
    card = (ROOT / "models/release/v2/README.md").read_text(encoding="utf-8")
    card += f"""
## Resultado final en test (evaluación única)
Evaluada la revisión `{v2['revision_sha']}` (artefacto SHA-256 `{t['model_sha256']}`), umbral {t['threshold']},
sobre las {t['dataset']['rows_evaluated']} filas de `UNSW_NB15_testing-set.csv`
(SHA-256 `{t['dataset']['raw_sha256']}`).

| recall ataque | precision ataque | FPR | macro-F1 | accuracy |
|---|---|---|---|---|
| {m['recall_attack']:.4f} | {m['precision_attack']:.4f} | {m['fpr']:.4f} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |

Matriz: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}. La FPR en test (0.18) es mucho mayor que en
validación (0.05): el tráfico normal del archivo de test difiere del de entrenamiento. Este commit solo
actualiza documentación; `model.joblib` es idéntico al de la revisión evaluada.
"""
    info = api().upload_file(repo_id=repo, path_or_fileobj=card.encode(), path_in_repo="README.md",
                             commit_message="Model card v2: agrega resultado final de test (modelo sin cambios)")
    state["v2_card_with_test"] = {"revision_sha": info.oid, "evaluated_revision_sha": v2["revision_sha"]}
    write_json(HF_JSON, state)
    print("card", info.oid)


if __name__ == "__main__":
    {"dvc": publish_dvc, "models": publish_models, "card-test": publish_card_test}[sys.argv[1]]()
