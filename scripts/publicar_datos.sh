#!/usr/bin/env bash
# Empaqueta los artefactos pesados del TFM y los sube como assets del Release v1.0.
#
# Las métricas y figuras de la Fase 6 YA están versionadas en results/ — este script
# solo sube los binarios que no caben en git.
#
# REQUISITOS previos (descárgalos de tu Google Drive y ajusta las rutas abajo):
#   - Carpeta  data/lama_synthetic/          (train/val/test)      -> $DATASET_DIR
#   - Fichero  checkpoints/lama_finetuned/lama_best.pth            -> $LAMA_PTH
#   - Fichero  checkpoints/aesrgan_finetuned/net_g_latest.pth      -> $AESRGAN_PTH
#   - Carpeta  datasets/Vintage_Degraded/    (54 fotografías)      -> $VINTAGE_DIR
#   - Variable GITHUB_PERSONAL_ACCESS_TOKEN exportada (scope repo o contents:write)
#
# Uso:  bash scripts/publicar_datos.sh
set -euo pipefail

OWNER="JavierRiesco"
REPO="Old-Image-Restoration-LaMa-A-ESRGAN"
TAG="v1.0"

# ---- AJUSTA ESTAS RUTAS ----
DATASET_DIR="${DATASET_DIR:-$HOME/Downloads/lama_synthetic}"
LAMA_PTH="${LAMA_PTH:-$HOME/Downloads/lama_best.pth}"
AESRGAN_PTH="${AESRGAN_PTH:-$HOME/Downloads/net_g_latest.pth}"
VINTAGE_DIR="${VINTAGE_DIR:-$HOME/Downloads/Vintage_Degraded}"
# ----------------------------

: "${GITHUB_PERSONAL_ACCESS_TOKEN:?exporta GITHUB_PERSONAL_ACCESS_TOKEN}"

echo "==> 1. Empaquetando assets"
TMP="$(mktemp -d)"
( cd "$(dirname "$DATASET_DIR")" && zip -qr "$TMP/lama_synthetic.zip"   "$(basename "$DATASET_DIR")" )
( cd "$(dirname "$VINTAGE_DIR")" && zip -qr "$TMP/vintage_degraded.zip" "$(basename "$VINTAGE_DIR")" )
zip -qj "$TMP/lama_finetuned.zip"    "$LAMA_PTH"
zip -qj "$TMP/aesrgan_finetuned.zip" "$AESRGAN_PTH"
ls -lh "$TMP"

echo "==> 2. Subiendo assets al Release $TAG"
REL_ID=$(curl -s -H "Authorization: Bearer $GITHUB_PERSONAL_ACCESS_TOKEN" \
  "https://api.github.com/repos/$OWNER/$REPO/releases/tags/$TAG" | python -c "import sys,json;print(json.load(sys.stdin)['id'])")
for f in lama_synthetic.zip lama_finetuned.zip aesrgan_finetuned.zip vintage_degraded.zip; do
  echo "    subiendo $f ..."
  curl -s -X POST \
    -H "Authorization: Bearer $GITHUB_PERSONAL_ACCESS_TOKEN" \
    -H "Content-Type: application/zip" \
    --data-binary @"$TMP/$f" \
    "https://uploads.github.com/repos/$OWNER/$REPO/releases/$REL_ID/assets?name=$f" \
    | python -c "import sys,json;d=json.load(sys.stdin);print('     ->',d.get('browser_download_url', d))"
done
rm -rf "$TMP"

echo "==> Hecho. Revisa https://github.com/$OWNER/$REPO/releases/tag/$TAG"
