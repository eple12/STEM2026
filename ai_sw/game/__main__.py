import os

# See run.py: one BLAS thread, set before numpy is imported. (`python -m game`
# imports game/__init__.py first, which imports nothing.)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from .app import main  # noqa: E402

if __name__ == "__main__":
    main()
