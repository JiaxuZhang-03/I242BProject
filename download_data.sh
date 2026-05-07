#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Download and unzip the food image dataset into data/.

Usage:
  DATA_URL=<direct-zip-url> ./download_data.sh
  ./download_data.sh <direct-zip-url>

Optional environment variables:
  DATA_URL       Direct URL to the dataset zip file.
                 Default: Kaggle API URL for mimiyh/healthy-and-unhealthy-food-image-dataset
  DATA_DIR       Directory where data should be stored. Default: data
  DATA_ZIP       Local zip path. Default: data/healthy-and-unhealthy-food-image-dataset.zip
  FORCE_DOWNLOAD Set to 1 to re-download even if DATA_ZIP already exists. Default: 0

Examples:
  ./download_data.sh
  DATA_URL=https://example.com/healthy-and-unhealthy-food-image-dataset.zip ./download_data.sh
  ./download_data.sh https://example.com/healthy-and-unhealthy-food-image-dataset.zip

If the zip already exists locally, the script can unzip and validate it without DATA_URL:
  DATA_ZIP=data/healthy-and-unhealthy-food-image-dataset.zip ./download_data.sh
USAGE
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${DATA_DIR:-data}"
DATA_ZIP="${DATA_ZIP:-${DATA_DIR}/healthy-and-unhealthy-food-image-dataset.zip}"
DEFAULT_DATA_URL="https://www.kaggle.com/api/v1/datasets/download/mimiyh/healthy-and-unhealthy-food-image-dataset"
DATA_URL="${1:-${DATA_URL:-${DEFAULT_DATA_URL}}}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

find_dataset_root() {
  find "${DATA_DIR}" -type d -name train -print | while IFS= read -r train_dir; do
    root_dir="$(dirname "${train_dir}")"
    if [[ -d "${root_dir}/val" && -d "${root_dir}/test" \
       && -d "${root_dir}/train/healthy" && -d "${root_dir}/train/unhealthy" \
       && -d "${root_dir}/val/healthy" && -d "${root_dir}/val/unhealthy" \
       && -d "${root_dir}/test/healthy" && -d "${root_dir}/test/unhealthy" ]]; then
      printf '%s\n' "${root_dir}"
      return 0
    fi
  done
}

require_command curl
require_command unzip
require_command find

mkdir -p "${DATA_DIR}"

if [[ "${FORCE_DOWNLOAD}" == "1" || ! -f "${DATA_ZIP}" ]]; then
  if [[ -z "${DATA_URL}" ]]; then
    cat >&2 <<EOF
Dataset zip not found at ${DATA_ZIP}, and DATA_URL was not provided.

Please pass a direct zip download URL:
  DATA_URL=<direct-zip-url> ./download_data.sh

Or place the zip at:
  ${DATA_ZIP}
and rerun:
  ./download_data.sh
EOF
    exit 1
  fi

  echo "Downloading dataset zip..."
  echo "URL: ${DATA_URL}"
  echo "Output: ${DATA_ZIP}"
  tmp_zip="${DATA_ZIP}.part"
  curl -L --fail --retry 3 --retry-delay 3 -o "${tmp_zip}" "${DATA_URL}"
  mv "${tmp_zip}" "${DATA_ZIP}"
else
  echo "Using existing dataset zip: ${DATA_ZIP}"
fi

echo "Unzipping dataset into ${DATA_DIR}/"
unzip -q -o "${DATA_ZIP}" -d "${DATA_DIR}"

dataset_root="$(find_dataset_root || true)"
if [[ -z "${dataset_root}" ]]; then
  cat >&2 <<EOF
Could not find the expected dataset structure after unzipping.

Expected to find a folder under ${DATA_DIR}/ with:
  train/healthy
  train/unhealthy
  val/healthy
  val/unhealthy
  test/healthy
  test/unhealthy

Please check the downloaded archive or DATA_URL.
EOF
  exit 1
fi

echo "Dataset ready."
echo "Dataset root: ${dataset_root}"
