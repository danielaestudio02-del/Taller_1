#!/usr/bin/env bash
# Crea entrega/proyecto.zip con el contenido versionado del commit actual (sin datos, venv ni cachés).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p entrega
git archive --format=zip --prefix=proyecto/ -o entrega/proyecto.zip HEAD
echo "entrega/proyecto.zip  commit $(git rev-parse HEAD)"; unzip -l entrega/proyecto.zip | tail -1
