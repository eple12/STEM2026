#!/bin/bash
# Bring a finished kernel's checkpoints back into the repo.
#
#   bash kaggle_fetch.sh <user>/<kernel-slug> <OUT_NAME>
#
# Lands <OUT_NAME>.npz / _best.npz / _state.pt in ai_sw/assets/policies,
# which is where the trace script and the game expect them.
set -eu
KERNEL="${1:?usage: kaggle_fetch.sh <user>/<kernel-slug> <OUT_NAME>}"
OUT_NAME="${2:?}"
export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

REPO="/mnt/c/Users/user/Desktop/WorkSpace/Dongari/STEM2026"
DEST="$REPO/ai_sw/assets/policies"
TMP="$HOME/aisw_kaggle/out"
rm -rf "$TMP"; mkdir -p "$TMP"

echo "=== downloading $KERNEL output ==="
kaggle kernels output "$KERNEL" -p "$TMP"

found=0
while IFS= read -r f; do
  cp "$f" "$DEST/$(basename "$f")"
  echo "  -> $(basename "$f")  $(du -h "$f" | cut -f1)"
  found=1
done < <(find "$TMP" -name "$OUT_NAME*" -type f)

[ "$found" -eq 1 ] || { echo "no $OUT_NAME* in the output"; exit 1; }

echo
echo "trace it with, from the ai_sw directory:"
echo "  python <scratchpad>/trace_v24.py $DEST/${OUT_NAME}_best.npz out.png <Circuit>"
