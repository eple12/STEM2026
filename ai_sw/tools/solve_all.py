"""Solve every circuit's grip ladder for the grand prix field, unattended.

The field (``game/grandprix.py``) needs, per circuit, the minimum-time plan
at each grip in ``teams.PLAN_GRIPS`` (``tools/mintime.py``). This runs them
all: per circuit the fastest first (from the min-curvature line, the slow
cold solve), then each lower grip warm-started from the one above it -- a
nearby solution converges in a few minutes where a cold one takes twenty.
Each solve is its own process, so a solver crash costs one plan, not the
run; a solve that does not converge is retried once with more iterations
and a smoother reference, and a grip that still fails is skipped (the field
maps drivers onto the grips that exist).

After a circuit's plans: one flying lap on the fastest plan in the game's
own physics, as a check, and the qualifying ghost laps for every difficulty
level (``replay.level_lap``).

Memory is what limits this on a laptop, not cores: one solve's sparse
factorisation takes a gigabyte or two, and three at once alongside a browser
ran the machine out (every one then failed in seconds). So by default one
circuit at a time (``--jobs`` to raise it), each solve starts only once
there is ``--min-free`` GB free, and a solve that died for want of memory is
waited out and run again rather than counted as a failure. The whole run
sits at below-normal priority, so the game stays smooth beside it.

Progress goes to ``sim_preview/solve_all.log`` and a status table to
``assets/racelines/solve_status.json``. Already-solved plans are skipped, so
the run can be stopped and started again.

    python tools/solve_all.py
    python tools/solve_all.py --circuit Spa --circuit Silverstone --jobs 2
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game import config, teams  # noqa: E402
from game.menu import available_circuits  # noqa: E402
from game.mintime_driver import path_file  # noqa: E402

PY = sys.executable
LOG = ROOT / "sim_preview" / "solve_all.log"
STATUS = config.RACELINE_DIR / "solve_status.json"
_lock = threading.Lock()


def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    with _lock:
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def status_update(circuit: str, **kw):
    with _lock:
        data = {}
        if STATUS.is_file():
            try:
                data = json.loads(STATUS.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data.setdefault(circuit, {}).update(kw)
        STATUS.write_text(json.dumps(data, indent=1), encoding="utf-8")


def free_gb() -> float:
    """Physical memory free right now, GB (Windows; elsewhere: plenty)."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 2 ** 30
    except Exception:
        return 1e9


#: Phrases in a solver's output that mean it ran out of memory, not that the
#: problem has no solution.
MEMORY_ERRORS = ("Insufficient_Memory", "bad_alloc", "Memory allocation",
                 "MemoryError", "Unable to allocate", "out of memory")


def wait_for_memory(need_gb: float, what: str):
    waited = 0
    while free_gb() < need_gb:
        if waited % 300 == 0:
            log(f"{what}: waiting for memory ({free_gb():.1f} GB free, "
                f"want {need_gb:.1f})")
        time.sleep(10)
        waited += 10


def tag_of(g: float) -> str:
    return f"g{round(g * 100):02d}"


def solve(circuit: str, grip: float, warm_from: str | None, max_iter: int,
          sigma: float, timeout: float, min_free: float = 3.0) -> bool:
    """One solve; True if it converged. One that ran out of memory is
    waited out and retried (up to three times) rather than failed."""
    for attempt in range(4):
        wait_for_memory(min_free, f"{circuit} {tag_of(grip)}")
        ok, oom = _solve_once(circuit, grip, warm_from, max_iter, sigma, timeout)
        if ok or not oom:
            return ok
        log(f"{circuit} {tag_of(grip)}: ran out of memory -- trying again "
            f"once there is room")
        time.sleep(60)
    return False


def _solve_once(circuit, grip, warm_from, max_iter, sigma, timeout):
    tag = tag_of(grip)
    cmd = [PY, str(ROOT / "tools" / "mintime.py"), "--circuit", circuit,
           "--grip", f"{grip}", "--tag", tag, "--max-iter", str(max_iter),
           "--sigma", f"{sigma}"]
    if warm_from:
        cmd += ["--warm-from", warm_from]
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             cwd=str(ROOT))
        text = (out.stdout or "") + (out.stderr or "")
    except subprocess.TimeoutExpired:
        log(f"{circuit} {tag}: timed out after {timeout / 60:.0f} min")
        return False, False
    ok = path_file(circuit, tag).exists()
    oom = (not ok) and any(m in text for m in MEMORY_ERRORS)
    lap = next((ln for ln in text.splitlines() if ln.startswith(f"{circuit}:")), "")
    log(f"{circuit} {tag}: {'ok' if ok else 'FAILED'}  {lap.strip()}  "
        f"({(time.perf_counter() - t0) / 60:.1f} min"
        + (f", warm from {warm_from}" if warm_from else ", cold") + ")")
    if not ok and not oom:
        tail = "\n".join(text.splitlines()[-6:])
        log(f"{circuit} {tag}: last output:\n{tail}")
    if not ok:
        # Never leave an unconverged iterate lying about.
        for f in (path_file(circuit, tag + "_failed"),
                  path_file(circuit, tag + "_failed").with_suffix(".png")):
            f.unlink(missing_ok=True)
    return ok, oom


def check_lap(circuit: str) -> str:
    """One flying lap on the fastest plan, in the game's physics."""
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from game.mintime_driver import MinTimeDriver, path_file\n"
        "from game.surface import Surface\n"
        "from game.trackdata import load_track\n"
        "from game.vehicle import Vehicle\n"
        "t = load_track('%s'); s = Surface(t)\n"
        "d = MinTimeDriver(t, s, path_file('%s', '%s'))\n"
        "v = Vehicle(); p, y = t.start_pose(); v.place(p, y); v.frozen = False\n"
        "n = t.count; armed = False; tt = 0.0; start = None; laps = []; off = 0\n"
        "while len(laps) < 1 and tt < 400:\n"
        "    v.step(d.controls(v), 1/120, s); tt += 1/120\n"
        "    i, _ = s.progress(v.pos)\n"
        "    if start is not None and not v.on_track: off += 1\n"
        "    if 0.4*n <= i <= 0.6*n: armed = True\n"
        "    elif armed and i < 0.1*n:\n"
        "        armed = False\n"
        "        if start is not None: laps.append(tt - start)\n"
        "        start = tt\n"
        "print('LAP', laps[0] if laps else -1, off)\n"
    ) % (ROOT, circuit, circuit, tag_of(teams.PLAN_GRIPS[0]))
    try:
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                             timeout=900, cwd=str(ROOT))
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("LAP")), "")
        return line[4:] if line else "no lap"
    except subprocess.TimeoutExpired:
        return "timed out"


def ghosts(circuit: str) -> str:
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from game import replay, teams\n"
        "from game.trackdata import load_track\n"
        "t = load_track('%s')\n"
        "out = []\n"
        "for lv in sorted(teams.DIFFICULTY):\n"
        "    rec = replay.level_lap('%s', lv, t)\n"
        "    out.append(f'L{lv} ' + (f'{rec.lap_time:.2f}' if rec else 'none'))\n"
        "print('GHOSTS', '  '.join(out))\n"
    ) % (ROOT, circuit, circuit)
    try:
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                             timeout=1800, cwd=str(ROOT))
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("GHOSTS")), "")
        if line:
            return line[7:]
        err = (out.stderr or "").strip().splitlines()
        return "failed: " + (err[-1] if err else "no output")
    except subprocess.TimeoutExpired:
        return "timed out"


def run_circuit(circuit: str, args) -> None:
    grips = list(teams.PLAN_GRIPS)
    done = [g for g in grips if path_file(circuit, tag_of(g)).exists()]
    log(f"{circuit}: start ({len(done)}/{len(grips)} plans already solved)")
    status_update(circuit, state="solving")
    prev = None
    solved = []
    for g in grips:
        tag = tag_of(g)
        if path_file(circuit, tag).exists():
            prev = tag
            solved.append(tag)
            continue
        ok = solve(circuit, g, prev, args.max_iter, args.sigma,
                   args.timeout * 60, args.min_free)
        if not ok:
            # Once more, harder: more iterations, a smoother reference, and
            # (for a lower grip) from the fastest plan rather than the last.
            ok = solve(circuit, g, solved[0] if solved else None,
                       args.max_iter * 2, args.sigma + 3.0, args.timeout * 60,
                       args.min_free)
        if ok:
            prev = tag
            solved.append(tag)
        status_update(circuit, plans=solved)
    if not solved:
        log(f"{circuit}: no plan converged -- the grand prix there races the "
            f"single AI ghost")
        status_update(circuit, state="failed")
        return
    lap = check_lap(circuit) if tag_of(grips[0]) in solved else "no g97"
    log(f"{circuit}: fastest plan driven in the game's physics: lap, off-track "
        f"ticks = {lap}")
    gh = ghosts(circuit)
    log(f"{circuit}: qualifying ghost laps {gh}")
    status_update(circuit, state="done", check=lap, ghosts=gh)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--circuit", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--min-free", type=float, default=3.0,
                    help="GB of free memory a solve waits for before it starts")
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--sigma", type=float, default=7.0)
    ap.add_argument("--timeout", type=float, default=75.0,
                    help="minutes one solve may take")
    args = ap.parse_args()
    LOG.parent.mkdir(exist_ok=True)
    try:
        # Below normal: the game (and anything else) comes first. Solver
        # processes started from here inherit it.
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    names = args.circuit or available_circuits()
    # Circuits nearest done first, so the field reaches the most tracks soon.
    def missing(c):
        return sum(not path_file(c, tag_of(g)).exists() for g in teams.PLAN_GRIPS)
    names = sorted(names, key=missing)
    log(f"solve_all: {len(names)} circuits, {args.jobs} at a time: "
        + ", ".join(names))
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(lambda c: run_circuit(c, args), names))
    log("solve_all: finished")


if __name__ == "__main__":
    main()
