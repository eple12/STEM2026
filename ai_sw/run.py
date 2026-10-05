#!/usr/bin/env python
"""Convenience launcher so you don't have to remember `python -m game`.

    python run.py                 # Monza, 3 laps
    python run.py --track Spa
    python run.py --track Silverstone --laps 5 --fullscreen
"""
import os

# The game's numpy work is small matrices, many times a frame. OpenBLAS's
# thread pool costs more to wake than those products take (the AI's forward
# pass runs 2.5x faster on one thread), and idle-spinning workers compete with
# the render thread. Must be set before numpy is first imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

if __name__ == "__main__":
    # Imported here, not at the top: the grand prix runs its field in a
    # worker process, and on Windows a worker re-imports this file -- it
    # must not pull the whole renderer in with it.
    from game.app import main
    main()
