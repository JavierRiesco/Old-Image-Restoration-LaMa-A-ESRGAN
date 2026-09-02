#!/usr/bin/env bash
# Publica en el repo los datos del análisis del TFM:
#   1. Copia métricas y figuras de la evaluación (Fase 6) a results/
#   2. Empaqueta dataset sintético y checkpoints y los sube al Release v1.0
#   3. Hace commit de results/ a nombre de Javier Riesco (sin co-autoría)
#
# REQUISITOS previos (descárgalos tú desde tu Google Drive):
#   - Carpeta  MyDrive/TFM/Fase6/            -> $FASE6_DIR
#   - Fichero  MyDrive/TFM/data/lama_synthetic/  (carpeta) -> $DATASET_DIR
#   - Fichero  MyDrive/TFM/checkpoints/lama_finetuned/lama_best.pth      -> $LAMA_PTH
#   - Fichero  MyDrive/TFM/checkpoints/aesrgan_finetuned/net_g_latest.pth -> $AESRGAN_PTH
#   - Variable GITHUB_PERSONAL_ACCESS_TOKEN exportada (scope repo)
#
# Uso:  bash scripts/publicar_datos.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OWNER="JavierRiesco"
REPO="Old-Image-Restoration-LaMa-A-ESRGAN"
TAG="v1.0"

# ---- AJUSTA ESTAS RUTAS ----
FASE6_DIR="${FASE6_DIR:-$HOME/Downloads/Fase6}"
DATASET_DIR="${DATASET_DIR:-$HOME/Downloads/lama_synthetic}"
LAMA_PTH="${LAMA_PTH:-$HOME/Downloads/lama_best.pth}"
AESRGAN_PTH="${AESRGAN_PTH:-$HOME/Downloads/net_g_latest.pth}"
# ----------------------------

: "${GITHUB_PERSONAL_ACCESS_TOKEN:?exporta GITHUB_PERSONAL_ACCESS_TOKEN}"
cd "$REPO_DIR"

echo "==> 1. Copiando métricas y figuras a results/"
DST="results/fase6_evaluacion_historica"
mkdir -p "$DST/figuras"
# métricas: json + csv (no imágenes)
find "$FASE6_DIR/metricas" -maxdepth 1 -type f \( -name '*.json' -o -name '*.csv' \) -exec cp {} "$DST/" \;
# figuras: png
find "$FASE6_DIR/figuras" -maxdepth 1 -type f -name '*.png' -exec cp {} "$DST/figuras/" \;
echo "    $(find "$DST" -type f | wc -l) archivos copiados"

echo "==> 2. Empaquetando assets"
TMP="$(mktemp -d)"
( cd "$(dirname "$DATASET_DIR")" && zip -qr "$TMP/lama_synthetic.zip" "$(basename "$DATASET_DIR")" )
zip -qj "$TMP/lama_finetuned.zip"    "$LAMA_PTH"
zip -qj "$TMP/aesrgan_finetuned.zip" "$AESRGAN_PTH"
ls -lh "$TMP"

echo "==> 3. Subiendo assets al Release $TAG"
REL_ID=$(curl -s -H "Authorization: Bearer $GITHUB_PERSONAL_ACCESS_TOKEN" \
  "https://api.github.com/repos/$OWNER/$REPO/releases/tags/$TAG" | python -c "import sys,json;print(json.load(sys.stdin)['id'])")
for f in lama_synthetic.zip lama_finetuned.zip aesrgan_finetuned.zip; do
  echo "    subiendo $f ..."
  curl -s -X POST \
    -H "Authorization: Bearer $GITHUB_PERSONAL_ACCESS_TOKEN" \
    -H "Content-Type: application/zip" \
    --data-binary @"$TMP/$f" \
    "https://uploads.github.com/repos/$OWNER/$REPO/releases/$REL_ID/assets?name=$f" \
    | python -c "import sys,json;d=json.load(sys.stdin);print('     ->',d.get('browser_download_url', d))"
done
rm -rf "$TMP"

echo "==> 4. Commit de results/ a nombre de Javier Riesco"
git add results/
git -c user.name="Javier Riesco" \
    -c user.email="324204402+JavierRiesco@users.noreply.github.com" \
    commit -m "Añade resultados de la evaluación (Fase 6: fotografía histórica)"
git push origin main

echo "==> Hecho. Revisa https://github.com/$OWNER/$REPO/releases/tag/$TAG"
