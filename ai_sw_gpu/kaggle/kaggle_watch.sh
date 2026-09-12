#!/bin/bash
# Follow a running Kaggle kernel.
#
#   bash kaggle_watch.sh <user>/<kernel-slug>          # stream the log live
#   bash kaggle_watch.sh <user>/<kernel-slug> status   # just the status
#
# `kernels logs -f` tails the running session the way tail -f does, so there
# is nothing to poll and no log file to parse.
set -u
KERNEL="${1:?usage: kaggle_watch.sh <user>/<kernel-slug> [status]}"
MODE="${2:-follow}"
export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

kaggle kernels status "$KERNEL"

if [ "$MODE" = "follow" ]; then
  echo "=== streaming (Ctrl-C to stop watching; the run keeps going) ==="
  kaggle kernels logs "$KERNEL" -f
fi
