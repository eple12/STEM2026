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
EXTRA = os.environ.get("AISW_EXTRA", "--ddqn --stop-after-stale 600").split()

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
print("=" * 64, flush=True)

# -- get the code into place ----------------------------------------------
# Kaggle expands an uploaded archive when it publishes a dataset, so the
# attachment is usually the extracted tree rather than the tarball we sent.
# Take whichever turns up.
WORK.mkdir(parents=True, exist_ok=True)
tarballs = list(INPUT.glob("*/ai_sw_bench.tar.gz"))
# .parent is ai_sw_gpu; .parent.parent is the bundle root that holds
# ai_sw/, ai_sw_gpu/ and the track data side by side.
trees = [p.parent.parent for p in INPUT.glob("*/ai_sw_gpu/train_iqn_gpu.py")]

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

# -- train -----------------------------------------------------------------
cmd = [sys.executable, "-u", str(WORK / "ai_sw_gpu" / "train_iqn_gpu.py"),
       "--circuit", CIRCUIT, "--steps", "15000000",
       "--envs", "112", "--rollout", "24",
       "--start-at-line", "0.20", "--target-sync", "48",
       "--eval-every", "25",
       "--device", "cuda" if CUDA else "cpu", "--eval-device", "cpu",
       "--async-eval", "--eval-workers", "2",
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
