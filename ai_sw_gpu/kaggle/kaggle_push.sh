#!/bin/bash
# Build the code bundle, publish it as a Kaggle dataset, and push the kernel
# that trains on it.
#
#   bash kaggle_push.sh <KAGGLE_USER> [CIRCUIT] [OUT_NAME] [RESUME_FROM]
#
#   KAGGLE_USER   your Kaggle username (the owner slug)
#   CIRCUIT       Spa (default), Monza, ...
#   OUT_NAME      checkpoint basename, default <CIRCUIT>_v26d
#   RESUME_FROM   a previous kernel slug to continue from, e.g.
#                 <user>/aisw-spa-v26d -- its output is attached as an input
#                 so the run picks up the weights AND the replay buffer
#
# Needs ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).
set -eu

USER_SLUG="${1:?usage: kaggle_push.sh <KAGGLE_USER> [CIRCUIT] [OUT_NAME] [RESUME_FROM]}"
CIRCUIT="${2:-Spa}"
OUT_NAME="${3:-${CIRCUIT}_v26d}"
RESUME_FROM="${4:-}"

REPO="/mnt/c/Users/user/Desktop/WorkSpace/Dongari/STEM2026"
STAGE="$HOME/aisw_kaggle"
DS_SLUG="aisw-code"
K_SLUG="aisw-$(echo "$OUT_NAME" | tr '[:upper:]_' '[:lower:]-')"

export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

echo "=== bundling the code ==="
rm -rf "$STAGE"; mkdir -p "$STAGE/data" "$STAGE/kernel"
tar czf "$STAGE/data/ai_sw_bench.tar.gz" -C "$REPO" \
  ai_sw/game/*.py ai_sw/tools/train_iqn.py ai_sw/assets/racelines \
  "Neural_Network_NEAT-master/new/f1tenth_racetracks-main/$CIRCUIT" \
  ai_sw_gpu/gpuenv ai_sw_gpu/tests ai_sw_gpu/train_iqn_gpu.py
ls -la "$STAGE/data/ai_sw_bench.tar.gz"

cat > "$STAGE/data/dataset-metadata.json" <<JSON
{
  "title": "ai_sw code bundle",
  "id": "$USER_SLUG/$DS_SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

echo "=== publishing the dataset ==="
if kaggle datasets status "$USER_SLUG/$DS_SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/data" -m "code $(date '+%Y-%m-%d %H:%M')" --dir-mode zip
else
  kaggle datasets create -p "$STAGE/data" --dir-mode zip
fi

echo "=== writing the kernel ==="
# The circuit/out-name are baked in here because a script kernel takes no
# arguments; kernel_run.py reads them from the environment it sets itself.
{
  echo "import os"
  echo "os.environ['AISW_CIRCUIT'] = '$CIRCUIT'"
  echo "os.environ['AISW_OUT'] = '$OUT_NAME'"
  echo "os.environ['AISW_EXTRA'] = '--ddqn --stop-after-stale 600'"
  cat "$REPO/ai_sw_gpu/kaggle/kernel_run.py"
} > "$STAGE/kernel/aisw_train.py"

SOURCES=""
[ -n "$RESUME_FROM" ] && SOURCES="\"$RESUME_FROM\""

cat > "$STAGE/kernel/kernel-metadata.json" <<JSON
{
  "id": "$USER_SLUG/$K_SLUG",
  "title": "aisw $OUT_NAME",
  "code_file": "aisw_train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["$USER_SLUG/$DS_SLUG"],
  "competition_sources": [],
  "kernel_sources": [$SOURCES],
  "model_sources": []
}
JSON
cat "$STAGE/kernel/kernel-metadata.json"

echo "=== pushing (this starts the run) ==="
kaggle kernels push -p "$STAGE/kernel"

echo
echo "kernel: $USER_SLUG/$K_SLUG"
echo "  watch:  bash kaggle_watch.sh $USER_SLUG/$K_SLUG"
echo "  fetch:  bash kaggle_fetch.sh $USER_SLUG/$K_SLUG $OUT_NAME"
