"""The body of the Kaggle script kernel: unpack the code, train, leave the
checkpoints in /kaggle/working.

Kaggle runs this non-interactively for up to 9 hours on a GPU, which is long
enough for a whole 5580-iteration run -- so unlike the Colab path there is
no session to lose and no replay buffer to ferry between machines.

Inputs (attached through kernel-metadata.json):
  /kaggle/input/<code dataset>/ai_sw_bench.tar.gz   the code + track data
  /kaggle/input/<previous kernel>/                  optional, to resume

Output: everything under /kaggle/working, which is where the trainer's
checkpoints land, so they come back with `kaggle kernels output`.
"""
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path("/kaggle/working/work")
POLICIES = WORK / "ai_sw" / "assets" / "policies"
INPUT = Path("/kaggle/input")

# Read by kaggle_push.sh when it writes the kernel; edit there, not here.
CIRCUIT = os.environ.get("AISW_CIRCUIT", "Spa")
OUT_NAME = os.environ.get("AISW_OUT", "Spa_v26d")
EXTRA = os.environ.get(
    "AISW_EXTRA",
    # v26k, confirmed 2026-09-13: EDGE_MARGIN=0 / the off-track-cost ramp /
    # 42-action table live in config.py and rlpolicy.py, not here -- nothing
    # circuit-specific to pass. --eval-backlog 1 added the same day after
    # Silverstone's 2-lap-sized episode window (rlenv.py) roughly doubled
    # how long one eval takes, which made the old drop-on-busy eval
    # scheduling skip most --eval-every requests outright; see the long
    # comment above --eval-backlog in train_iqn_gpu.py. --revive-after 150
    # (an exploration boost for exactly the "stale on a tier-0 best" plateau
    # widening produces) was tried 2026-09-14 and reverted the same day in
    # favour of a reward-side fix instead -- see RL_POST_WIDEN_RAMP in
    # config.py and the long comment above it. --n-step 3 -> 6 the same day:
    # RL_SHAPE_TO_RACELINE is now off (config.py) so nothing hands the
    # network a fixed speed target any more, only lap-time reward -- a
    # decision now has to have its consequence show up in a LONGER raw
    # reward window for multi-corner trade-offs (brake a bit less here, gain
    # it back at the next exit) to be visible without relying entirely on
    # bootstrapped value estimates. RL_POST_WIDEN_RAMP is OFF (flat) while
    # this is tested, to isolate its effect from the ramp's.
    #
    # 2026-09-15: a run of experiments on top of this (staged widening via
    # EPISODE_WIDEN_STAGES, --widen-after-perfects gating the widen trigger,
    # a zero-tolerance clean-lap bonus, a continuous off-step decay
    # replacing it, a dense per-lap progress penalty) were tried in sequence
    # chasing "PERFECT doesn't reliably recur past the post-widen bar" --
    # none beat the 91.30 s Monza record this exact recipe (n-step 6,
    # RL_SHAPE_TO_RACELINE off, the 43-dim geometry-only observation, single-
    # stage widen, flat ramp) already produced, and the last one diverged
    # outright. Explicit request: with GPU budget nearly gone and no
    # confirmed replacement, stop guessing and lock in the recipe already
    # proven to work rather than risk it further. --widen-after-perfects and
    # RL_LAP_OFF_DECAY_CAP/RL_DIRTY_LAP_PROGRESS_MULT's code paths are still
    # there (the former defaults to 1 = original behaviour if unused; the
    # latter two were removed outright, see config.py) in case a properly
    # isolated retest is worth it later, just not defaulted on here.
    #
    # 2026-09-19: --stop-if-no-perfect 900 added. --stop-after-stale only
    # counts once a run has had a PERFECT, so a run that never gets one
    # (Melbourne: tier-1 full laps from it675, best at it1000, then eroding)
    # ran on to the last iteration. This only ends such runs; it cannot
    # touch one that has a PERFECT.
    "--ddqn --line-k 0 --stop-after-stale 600 --eval-backlog 1 "
    "--n-step 6 --stop-if-no-perfect 900").split()

print("=" * 64, flush=True)
# Diagnostics must never be able to kill the run: nvidia-smi is absent from
# the image when no accelerator is attached, and an unguarded call to it
# took a whole kernel down before it trained a single step.
try:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)
except FileNotFoundError:
    print("no nvidia-smi on this image", flush=True)

import torch
CUDA = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda={CUDA}  "
      f"{torch.cuda.get_device_name(0) if CUDA else 'CPU ONLY'}", flush=True)
if not CUDA:
    print("WARNING: no GPU attached. Check that the kernel metadata asked "
          "for one and that the account may use accelerators; this run would "
          "take days on CPU.", flush=True)
elif CUDA and torch.cuda.get_device_capability(0) < (7, 0):
    # This build's cu128 wheel only ships kernels for sm_70+, so a Pascal
    # card (P100 = sm_60, sometimes assigned instead of the requested T4)
    # cannot actually run anything -- every op raises at first use. Stop
    # here with a clear reason rather than burning the GPU-hour quota on a
    # traceback from deep inside the first matmul.
    sys.exit(f"GPU {torch.cuda.get_device_name(0)} has compute capability "
             f"{torch.cuda.get_device_capability(0)}, below what this torch "
             f"build supports (7.0+). This happens when Kaggle assigns a "
             f"P100 despite --accelerator NvidiaTeslaT4 -- push again, or "
             f"check quota with `kaggle quota` and try later.")

# Diagnostic dump, unconditional: which datasets are actually mounted and
# under what path. Kaggle has mounted the same dataset at a different path
# across two otherwise-identical pushes here (once as /kaggle/input/aisw-code,
# once as /kaggle/input/datasets), so print the real layout every run instead
# of only on failure.
print("--- /kaggle/input ---", flush=True)
for p in sorted(INPUT.glob("*")):
    kids = sorted(x.name for x in p.iterdir())[:8] if p.is_dir() else []
    print(f"  {p}  {'-> ' + ', '.join(kids) if kids else ''}", flush=True)
print("=" * 64, flush=True)

# -- get the code into place ----------------------------------------------
# Kaggle expands an uploaded archive when it publishes a dataset, so the
# attachment is usually the extracted tree rather than the tarball we sent.
# Take whichever turns up.
WORK.mkdir(parents=True, exist_ok=True)
# Search a few levels deep rather than assuming one: the same dataset has
# mounted at /kaggle/input/<slug>/... on one push and
# /kaggle/input/datasets/<slug>/... on another.
tarballs = list(INPUT.glob("**/ai_sw_bench.tar.gz"))
# .parent is ai_sw_gpu; .parent.parent is the bundle root that holds
# ai_sw/, ai_sw_gpu/ and the track data side by side.
# A resumed / warm-started run attaches a PREVIOUS kernel's output, and that
# output carries its own copy of ai_sw_gpu/ (under work/). Taking whichever
# copy globs first ran the old trainer instead of the freshly bundled one --
# unrecognised new flags, or worse, silently old behaviour. The code dataset
# is the only source of truth for code, so it wins; anything found under a
# kernel-output "work/" directory is used only if nothing else exists.
_all_trees = [p.parent.parent for p in INPUT.glob("**/ai_sw_gpu/train_iqn_gpu.py")]
trees = ([t for t in _all_trees if "aisw-code" in str(t)]
         or [t for t in _all_trees if "/work" not in str(t)]
         or _all_trees)

if tarballs:
    with tarfile.open(tarballs[0]) as tf:
        tf.extractall(WORK)
    print(f"unpacked {tarballs[0]}", flush=True)
elif trees:
    for item in trees[0].iterdir():
        dst = WORK / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy(item, dst)
    print(f"copied the extracted tree from {trees[0]}", flush=True)
else:
    print("attached inputs:", [str(p) for p in INPUT.glob("*")], flush=True)
    sys.exit("no ai_sw code among the attached datasets")
POLICIES.mkdir(parents=True, exist_ok=True)

# -- carry a previous run forward, if one is attached ----------------------
# A kernel's own output shows up as an input on the next run, so a run that
# hit the 9 h wall continues with its weights AND its replay buffer rather
# than restarting -- the distinction that cost the Monza run its policy.
resumed = []
for src in INPUT.glob(f"*/work/ai_sw/assets/policies/{OUT_NAME}*"):
    shutil.copy(src, POLICIES / src.name)
    resumed.append(src.name)
for src in INPUT.glob(f"*/{OUT_NAME}*"):          # flattened output layout
    if src.is_file():
        shutil.copy(src, POLICIES / src.name)
        resumed.append(src.name)
print(f"resuming from: {sorted(set(resumed)) or 'nothing, fresh run'}", flush=True)

# --init-from NAME warm-starts from NAME.npz. Its file lives in an attached
# kernel's output (kernel_sources), under a name that is not OUT_NAME's, so the
# resume globs above never pick it up -- fetch it explicitly.
for i, tok in enumerate(EXTRA):
    if tok == "--init-from" and i + 1 < len(EXTRA):
        found = sorted(INPUT.glob(f"**/{EXTRA[i + 1]}.npz"))
        if not found:
            sys.exit(f"--init-from {EXTRA[i + 1]}: no {EXTRA[i + 1]}.npz among "
                     f"the attached inputs {[str(p) for p in INPUT.glob('*')]}")
        shutil.copy(found[0], POLICIES / found[0].name)
        print(f"init checkpoint: {found[0]}", flush=True)

# -- train -----------------------------------------------------------------
# The eval now runs as real separate PROCESSES (multiprocessing, not
# threads -- see train_iqn_gpu.py), so it genuinely uses however many cores
# are handed a worker each, unlike the old thread-based version where the
# GIL serialised them regardless of core count. Training itself is
# GPU-bound; leave the main process one core and give the rest to eval.
n_eval_workers = max(1, (os.cpu_count() or 4) - 1)
print(f"cpu_count={os.cpu_count()} -> --eval-workers {n_eval_workers}", flush=True)

cmd = [sys.executable, "-u", str(WORK / "ai_sw_gpu" / "train_iqn_gpu.py"),
       "--circuit", CIRCUIT, "--steps", "15000000",
       "--envs", "112", "--rollout", "24",
       "--start-at-line", "0.20", "--target-sync", "48",
       "--eval-every", "25",
       "--device", "cuda" if CUDA else "cpu", "--eval-device", "cpu",
       "--async-eval", "--eval-workers", str(n_eval_workers),
       "--save-every", "5", "--state-every", "50", "--resume",
       *EXTRA, "--out-name", OUT_NAME]
print("launching:", " ".join(cmd[1:]), flush=True)

p = subprocess.Popen(cmd, cwd=str(WORK / "ai_sw_gpu"),
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
for line in p.stdout:
    print(line.rstrip(), flush=True)
rc = p.wait()
print(f"\ntrainer exited rc={rc}", flush=True)

for f in sorted(POLICIES.glob(f"{OUT_NAME}*")):
    print(f"  output: {f.name}  {f.stat().st_size / 1e6:.1f} MB", flush=True)
