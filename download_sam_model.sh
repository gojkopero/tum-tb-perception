#!/bin/bash
#
# Downloads the MobileSAM checkpoint to the models/ directory.
# Source: https://github.com/ChaoningZhang/MobileSAM
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"
CHECKPOINT="mobile_sam.pt"
URL="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_DIR}/${CHECKPOINT}" ]; then
    echo "[INFO] MobileSAM checkpoint already exists: ${MODEL_DIR}/${CHECKPOINT}"
    echo "[INFO] To re-download, delete the file and run this script again."
    exit 0
fi

echo "[INFO] Downloading MobileSAM checkpoint..."
echo "[INFO] URL: ${URL}"
echo "[INFO] Destination: ${MODEL_DIR}/${CHECKPOINT}"

wget -q --show-progress -O "${MODEL_DIR}/${CHECKPOINT}" "${URL}"

if [ $? -eq 0 ]; then
    echo "[INFO] Download complete."
    ls -lh "${MODEL_DIR}/${CHECKPOINT}"
else
    echo "[ERROR] Download failed!"
    rm -f "${MODEL_DIR}/${CHECKPOINT}"
    exit 1
fi
