"""Publish the race-AI code bundle to Kaggle and push the CPU kernel that trains it.

    python raceai_push.py ppo-v1                              # a fresh run
    python raceai_push.py ppo-v2 --resume-from mskimdev/aisw-raceai-ppo-v1
    python raceai_push.py ppo-v1 --extra "--kl 0.2 --T 1.3" --hours 11

What runs is ``ai_sw/tools/ppo_raceai.py`` (stage 3 of the learned race driver,
see ai_sw/README.md). It is a **CPU** kernel on purpose: the cost is the Python
simulation, not the network (59 -> 128 -> 128 -> 21), and a GPU session has two
cores where a CPU one has four -- a GPU would make it slower. CPU sessions run
12 h and have no weekly quota; the kernel stops itself at ``--hours`` so its
output is kept.

The bundle is ~35 MB: the game's ``.py`` files, the trainer, the behaviour-cloned
start (``policy/raceai/bc.npz``), the solved raceline plans (``*.npz`` only) and
the circuit data. Everything is private to the account in ``~/.kaggle``.

Outputs come back with ``kaggle kernels output <slug> -p <dir>``: ``ppo/ppo_best.npz``
(the policy that scored best on the held-out evaluation), ``ppo_last.npz``,
``ppo_log.jsonl`` and ``ppo_state.pt`` (so ``--resume-from`` continues a run).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DS_SLUG = "aisw-raceai-code"
TRACKS = "Neural_Network_NEAT-master/new/f1tenth_racetracks-main"

KERNEL = r'''
import os, shutil, subprocess, sys, tarfile
from pathlib import Path

HOURS = {hours}
NAME = {name!r}
EXTRA = {extra!r}
INPUT = Path("/kaggle/input")
WORK = Path("/tmp/aisw")
OUT = Path("/kaggle/working/ppo")

print("=" * 60, flush=True)
print("/kaggle/input layout:", flush=True)
for p in sorted(INPUT.glob("*"))[:20]:
    print("  ", p, flush=True)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cpus", os.cpu_count(), flush=True)

# The dataset mounts at a different path from one push to the next, and Kaggle
# may or may not have expanded the archive: take whichever is there.
tars = list(INPUT.glob("**/raceai_bundle.tar.gz"))
trees = [p.parent.parent.parent for p in INPUT.glob("**/ai_sw/tools/ppo_raceai.py")]
if WORK.exists():
    shutil.rmtree(WORK)
if tars:
    WORK.mkdir(parents=True)
    with tarfile.open(tars[0]) as tf:
        tf.extractall(WORK)
    print("unpacked", tars[0], flush=True)
elif trees:
    shutil.copytree(trees[0], WORK)
    print("copied", trees[0], flush=True)
else:
    sys.exit("no code bundle found under /kaggle/input")

OUT.mkdir(parents=True, exist_ok=True)
resume = []
prev = [p for p in INPUT.glob("**/ppo/ppo_state.pt")]
if prev:
    for f in prev[0].parent.iterdir():
        if f.is_file():
            shutil.copy(f, OUT / f.name)
    resume = ["--resume"]
    print("resuming from", prev[0].parent, flush=True)

cmd = [sys.executable, "tools/ppo_raceai.py", "--bc", "policy/raceai/bc.npz", "--out", str(OUT),
       "--jobs", "4", "--iters", "100000", "--max-hours", str(HOURS), "--eval-every", "10",
       "--eval-seeds", "3"] + resume + EXTRA.split()
print("running:", " ".join(cmd), flush=True)
sys.stdout.flush()
rc = subprocess.call(cmd, cwd=str(WORK / "ai_sw"))
print("trainer exited with", rc, flush=True)
(OUT / "_cur.npz").unlink(missing_ok=True)
sys.exit(rc)
'''


def kaggle(*args, check=True, capture=False):
    cmd = ["kaggle", *args]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, check=check, capture_output=capture, text=True)
    return r


def user() -> str:
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        sys.exit("no Kaggle username: set up ~/.kaggle first")
    return m.group(1)


def bundle(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "raceai_bundle.tar.gz"
    files = []
    files += sorted((REPO / "ai_sw" / "game").glob("*.py"))
    files += [REPO / "ai_sw" / "tools" / n for n in ("ppo_raceai.py", "train_raceai.py")]
    files += [REPO / "ai_sw" / "policy" / "raceai" / "bc.npz"]
    files += sorted((REPO / "ai_sw" / "assets" / "racelines").glob("*.npz"))
    files += sorted(p for p in (REPO / TRACKS).rglob("*") if p.is_file())
    with tarfile.open(path, "w:gz") as tf:
        for f in files:
            tf.add(f, arcname=str(f.relative_to(REPO)).replace("\\", "/"))
    print(f"bundle: {len(files)} files, {path.stat().st_size / 1e6:.1f} MB", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="run name; the kernel is aisw-raceai-<name>")
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--extra", default="", help="more flags for ppo_raceai.py")
    ap.add_argument("--resume-from", default=None, help="a previous kernel slug")
    ap.add_argument("--no-dataset", action="store_true", help="reuse the published bundle")
    args = ap.parse_args()

    me = user()
    slug = f"aisw-raceai-{re.sub('[^a-z0-9]+', '-', args.name.lower()).strip('-')}"
    stage = Path(tempfile.mkdtemp(prefix="raceai_kaggle_"))
    try:
        if not args.no_dataset:
            data = stage / "data"
            bundle(data)
            (data / "dataset-metadata.json").write_text(json.dumps({
                "title": "aisw raceai code bundle", "id": f"{me}/{DS_SLUG}",
                "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
            exists = subprocess.run(["kaggle", "datasets", "status", f"{me}/{DS_SLUG}"],
                                    capture_output=True, text=True)
            if exists.returncode == 0 and exists.stdout.strip():
                kaggle("datasets", "version", "-p", str(data), "-m",
                       f"raceai {time.strftime('%Y-%m-%d %H:%M')}", "--dir-mode", "zip")
            else:
                kaggle("datasets", "create", "-p", str(data), "--dir-mode", "zip")
            print("waiting for the dataset", end="", flush=True)
            for _ in range(90):
                st = subprocess.run(["kaggle", "datasets", "status", f"{me}/{DS_SLUG}"],
                                    capture_output=True, text=True).stdout.strip()
                if st == "ready":
                    print(" -> ready", flush=True)
                    break
                print(".", end="", flush=True)
                time.sleep(10)
            else:
                sys.exit("\nthe dataset never became ready")

        k = stage / "kernel"
        k.mkdir()
        (k / "raceai_train.py").write_text(
            KERNEL.format(hours=args.hours, name=args.name, extra=args.extra), encoding="utf-8")
        sources = [json.dumps(args.resume_from)] if args.resume_from else []
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{me}/{slug}", "title": f"aisw raceai {args.name}",
            "code_file": "raceai_train.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "false", "enable_internet": "false",
            "dataset_sources": [f"{me}/{DS_SLUG}"], "competition_sources": [],
            "kernel_sources": [json.loads(s) for s in sources], "model_sources": []},
            indent=1), encoding="utf-8")
        print((k / "kernel-metadata.json").read_text(encoding="utf-8"))
        kaggle("kernels", "push", "-p", str(k))
        print(f"\nkernel: {me}/{slug}")
        print(f"  status: kaggle kernels status {me}/{slug}")
        print(f"  output: kaggle kernels output {me}/{slug} -p <dir>")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
