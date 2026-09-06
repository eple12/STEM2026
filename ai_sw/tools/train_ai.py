"""Learn a fast AI driver: separable CMA-ES on simulated lap time, in parallel.

Why the objective is a simulated lap and not an analytic one
-----------------------------------------------------------
``raceline.lap_time`` costs 20 ms against a simulated lap's 14 s, which looked
like the whole game. It is not usable as a search objective: it scored the
centreline at 129.65 s where the car actually laps 126.79, and the optimised
line at 112.56 where the car actually laps 125.68. Not even *ordered* the same
way as reality -- and ``min_curvature`` minimises exactly what that model
measures, so the search was rewarded for exploiting the estimator. Simulating
removes all of it.

Why the line is refined coarse-to-fine
--------------------------------------
The first version used 24 control points. On Monza that is one every 241 m,
each spanning 241 m either side -- while the corners have a median length of
72 m and the shortest is 25 m. The most lateral movement it could produce was
5.9 m per 100 m of track; taking a 150 m chicane out-in-out needs about 15.7.
It was not that the search failed to find out-in-out. **It could not represent
it.** So the line is refined in stages, each starting from the last upsampled,
down to a control point every ~18 m.

That resolution puts the parameter count in the hundreds, where full CMA-ES
spends O(d^3) an eigendecomposition and O(d^2) memory on a covariance whose
off-diagonal terms are mostly meaningless anyway -- an offset at one corner
does not interact with an offset two kilometres away. Separable CMA-ES keeps
the diagonal only and costs O(d).

What is learned
---------------
* the racing line, at increasing resolution;
* a per-sample multiplier on the planned speed, which is what lets the car stop
  braking so early and get back on the throttle out of a chicane;
* the planner's twelve constants, and the grip it plans for.

Not a neural policy. For one deterministic circuit the fastest lap is a single
trajectory, and a position-indexed parameterisation represents it exactly;
a state-conditioned network has to learn to reproduce it through a much harder
credit assignment, and its generalisation buys nothing when there is only one
track and no noise.

    python tools/train_ai.py --circuit Monza
    python tools/train_ai.py --circuit Monza --stages 24,64,160,320 --generations 40
"""
import os

# Before numpy, and therefore before OpenBLAS builds its thread pool. Each
# worker process gets its own pool sized to the whole machine, so fifteen
# workers asked for fifteen times the machine's threads and the first run died
# on "Memory allocation still failed after 10 retries" without completing a
# single generation. A lap simulation is scalar numpy -- BLAS threading buys it
# nothing and costs it everything.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game import config, raceline
from game.autopilot import TUNING, Autopilot
from game.surface import Surface
from game.trackdata import load_track
from game.vehicle import Vehicle

DT = 1.0 / 120.0
KEYS = sorted(TUNING)
#: How far each constant may move, as a multiplier. Wide enough to find a
#: different driver, narrow enough that the search cannot wander into settings
#: where the car does not finish.
# Widened from 0.7 after Monza's run pinned nine of the twelve constants
# against it. A search that ends on its bounds has not converged, it has been
# capped, and the value it reports is the cap rather than the answer. The
# objective is a real lap with an off-track cost, so a setting the car cannot
# actually drive is rejected on its own.
SPREAD = 1.0
FAIL = 1e4
#: The dimension sigma was tuned at, for the sqrt(d) scaling above.
SIGMA_REF_DIM = 49

_CACHE = {}


def _track(name):
    if name not in _CACHE:
        tr = load_track(name)
        _CACHE[name] = (tr, Surface(tr))
    return _CACHE[name]


def drive(name, offset, tune, pace, speed=None, limit=400.0):
    """One flying lap on this line with these constants. None if it fails.

    Started already moving, on the line, at the speed the profile asks for
    there, so a single lap is representative and the run costs one lap rather
    than a warm-up plus one.
    """
    track, surf = _track(name)
    # Reuse the process's Surface and just rewind its search hint. Building a
    # new one per lap rebuilt the barrier index -- a segments-by-samples argmin
    # -- every time, and fifteen workers each doing that ran the machine out of
    # memory before a single generation finished.
    line = raceline.Line(track, offset) if offset is not None else None
    surf.hint = 0
    pilot = Autopilot(track, surf, pace=pace, line=line, tuning=tune,
                      speed_scale=speed)
    geo = pilot.geo

    v = Vehicle()
    v.frozen = False
    i0 = 0
    yaw = math.atan2(geo.tangent[i0, 0], geo.tangent[i0, 1])
    v.place(geo.center[i0], yaw)
    speed = float(min(pilot.v_profile[i0], config.MAX_SPEED))
    v.vel = np.array([math.sin(yaw), math.cos(yaw)]) * speed

    n = track.count
    t, armed, off_track = 0.0, False, 0
    steps = 0
    while t < limit:
        v.step(pilot.controls(v), DT, surf)
        t += DT
        steps += 1
        if not np.isfinite(v.pos).all():
            return None
        i, _ = surf.progress(v.pos)
        # The vehicle already judged this at its four wheels on this very
        # step, which is the rule the game plays by. Asking the surface again
        # without a yaw silently used the centre-of-mass rule instead, so the
        # objective was stricter than the physics and the search left a
        # body-width of legal track unused at every corner.
        if not v.on_track:
            off_track += 1
        if 0.4 * n <= i <= 0.6 * n:
            armed = True
        elif armed and i < 0.1 * n:
            # Penalise time spent off the asphalt rather than forbidding it:
            # a line that uses the kerb is legitimate, one that cuts the grass
            # is not, and a soft cost tells the two apart without a rule about
            # which corners allow what.
            return t + 3.0 * (off_track / max(steps, 1)) * t
    return None


#: Units per unit of search space, so one step size suits every group. Without
#: these a sigma that moves the line a sensible 0.35 m also moves every speed
#: target by 42% and every constant by a third, and the first generations are
#: spent throwing the car off the road rather than exploring. sep-CMA-ES would
#: eventually learn per-coordinate variances, but it starts from all-ones and
#: pays for the whole descent.
LINE_SCALE = 1.00      # metres of lateral offset
SPEED_SCALE = 0.06     # log speed multiplier
TUNE_SCALE = 0.25      # log multiplier on a planner constant
PACE_SCALE = 0.10


def _decode(vec, B, Bv, lo, hi, base):
    """Split a parameter vector into (line offsets, speed scale, tuning, pace).

    Each stage searches a **correction** on the stage before it, not a line of
    its own, so a vector of zeros decodes to exactly what the previous stage
    finished with. Projecting the old solution onto the new basis instead --
    which is what this did -- has to be paid for at every handover: going from
    64 to 160 control points cost 24.5 s of lap time, more than the whole stage
    then won back. A correction that starts at zero cannot lose anything.
    """
    base_off, base_speed = base
    k, kv = B.shape[1], Bv.shape[1]
    offset = np.clip(base_off + B @ (vec[:k] * LINE_SCALE), lo, hi)
    # Speed corrections live in log space so the search steps are symmetric
    # between "carry 10% more" and "carry 10% less".
    speed = np.clip(base_speed * np.exp(Bv @ (vec[k:k + kv] * SPEED_SCALE)),
                    0.55, 1.80)
    u = np.clip(vec[k + kv:k + kv + len(KEYS)] * TUNE_SCALE, -SPREAD, SPREAD)
    tune = {key: TUNING[key] * math.exp(float(x)) for key, x in zip(KEYS, u)}
    pace = float(np.clip(vec[-1] * PACE_SCALE, 0.70, 1.60))
    return offset, speed, tune, pace


def _job(args):
    name, vec, B, Bv, lo, hi, base = args
    offset, speed, tune, pace = _decode(np.asarray(vec), B, Bv, lo, hi, base)
    got = drive(name, offset, tune, pace, speed)
    return FAIL if got is None else got


def _checkpoint(name, vec, B, Bv, lo, hi, base, best_f, gen):
    """Write the best-so-far as soon as it improves.

    Two earlier runs were lost to a stopped session before they printed
    anything. A search that only reports at the end has no partial result.
    """
    track, _ = _track(name)
    # Never overwrite something better. best_f starts at FAIL inside a run, so
    # its first generation always "improves" -- and a resumed run whose first
    # generation happened to be worse than the solution it started from wrote
    # that over the top of it. A 111.850 s line was lost to a 124.283 s one
    # exactly this way.
    path = config.RACELINE_DIR / f"{track.name}_tuning.json"
    if path.exists():
        try:
            if json.loads(path.read_text()).get("lap", FAIL) <= best_f:
                return
        except (ValueError, OSError):
            pass

    offset, speed, tune, pace = _decode(np.asarray(vec), B, Bv, lo, hi, base)
    raceline.save(track, offset)
    np.save(config.RACELINE_DIR / f"{track.name}_speed.npy", speed)
    (config.RACELINE_DIR / f"{track.name}_tuning.json").write_text(
        json.dumps({"pace": pace, "tuning": tune, "lap": best_f,
                    "generation": gen}, indent=2))


def sep_cma(name, x0, B, Bv, lo, hi, base, gens, workers, seed, sigma0,
            on_best):
    """Separable CMA-ES: the covariance is a diagonal, so everything is O(d).

    Same algorithm as the full version otherwise -- weighted recombination, an
    evolution path for the step size, a second one for the variances.
    """
    d = len(x0)
    rng = np.random.default_rng(seed)
    lam = max(12, 4 + int(3 * math.log(d)))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mu_eff = 1.0 / np.sum(w ** 2)

    c_s = (mu_eff + 2) / (d + mu_eff + 3)
    d_s = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_s
    c_c = 4.0 / (d + 4.0)
    # The separable variants of the rank-1 and rank-mu rates, scaled up by
    # (d + 2) / 3 as Ros & Hansen give them: a diagonal has d free parameters
    # rather than d^2, so it can be learned that much faster.
    c_1 = (2.0 / ((d + 1.3) ** 2 + mu_eff)) * (d + 2) / 3.0
    c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)
                         / ((d + 2) ** 2 + mu_eff)) * (d + 2) / 3.0)
    chi = math.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))

    # sigma is a *per-coordinate* step, but what reaches the car is the whole
    # displacement, and that grows as sqrt(d). Holding the per-coordinate value
    # fixed across the refinement stages means the finest one -- 493 parameters
    # against the coarsest one's 49 -- moves three times as far in aggregate,
    # on the stage whose job is to refine rather than explore. Scaled to keep
    # the total step the size it was at the coarse stage.
    sigma = sigma0 * math.sqrt(SIGMA_REF_DIM / d)

    x = x0.copy()
    p_s, p_c, var = np.zeros(d), np.zeros(d), np.ones(d)
    pool = Pool(workers) if workers > 1 else None
    try:
        # Evaluate where we start. The search only ever samples x + sigma*y, so
        # without this the seed's own score is never known and a run can drift
        # away from a good solution while reporting the best of its own noise.
        best_x = x.copy()
        best_f = _job((name, x, B, Bv, lo, hi, base))
        print(f"    seed    {best_f:7.3f} s   sigma {sigma:5.3f}", flush=True)
        if best_f < FAIL:
            on_best(best_x, best_f, 0)
        for g in range(gens):
            sd = np.sqrt(var)
            z = rng.standard_normal((lam, d))
            y = z * sd
            pop = x + sigma * y
            jobs = [(name, p, B, Bv, lo, hi, base) for p in pop]
            fit = np.array(pool.map(_job, jobs) if pool
                           else [_job(j) for j in jobs])
            order = np.argsort(fit)
            if fit[order[0]] < best_f:
                best_f, best_x = float(fit[order[0]]), pop[order[0]].copy()
                on_best(best_x, best_f, g + 1)

            y_w = w @ y[order[:mu]]
            x = x + sigma * y_w
            p_s = ((1 - c_s) * p_s
                   + math.sqrt(c_s * (2 - c_s) * mu_eff) * (y_w / sd))
            h = (np.linalg.norm(p_s) / math.sqrt(1 - (1 - c_s) ** (2 * (g + 1)))
                 / chi < 1.4 + 2 / (d + 1))
            p_c = (1 - c_c) * p_c + h * math.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            rank_mu = w @ (y[order[:mu]] ** 2)
            var = ((1 - c_1 - c_mu) * var
                   + c_1 * (p_c ** 2 + (not h) * c_c * (2 - c_c) * var)
                   + c_mu * rank_mu)
            var = np.maximum(var, 1e-12)
            sigma = float(np.clip(
                sigma * math.exp((c_s / d_s) * (np.linalg.norm(p_s) / chi - 1)),
                1e-3, 3.0))
            ok = int((fit < FAIL).sum())
            print(f"    gen {g + 1:3d}/{gens}  best {best_f:7.3f} s   "
                  f"gen {fit[order[0]]:7.3f}   ok {ok:2d}/{lam}   "
                  f"sigma {sigma:5.3f}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return best_x, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--stages", default="24,64,160,320",
                    help="line control points per refinement stage")
    ap.add_argument("--generations", type=int, default=40,
                    help="generations per stage")
    # Each worker holds a track, its wall offsets and its barrier index, so
    # the ceiling here is memory rather than cores.
    ap.add_argument("--workers", type=int,
                    default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--pace", type=float, default=config.GHOST_PACE)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="start from the saved line, speed profile and "
                         "constants for this circuit")
    ap.add_argument("--from-centreline", action="store_true",
                    help="start from the centreline instead of the "
                         "least-squares line")
    args = ap.parse_args()

    track, _ = _track(args.circuit)
    lo, hi = raceline.bounds(track)
    stages = [int(v) for v in args.stages.split(",")]

    base_lap = drive(args.circuit, None, dict(TUNING), args.pace)
    print(f"{args.circuit}: {track.length:.0f} m, baseline (centreline, "
          f"hand-picked constants) {base_lap:.3f} s")

    # Carried between stages in *sample* space, not control space: the bases
    # differ, so the handover has to go through the thing they both describe.
    # Which is also what makes resuming possible -- the checkpoint on disk is
    # in that same sample space, so a run killed mid-search can be picked up
    # by any stage, at any resolution.
    speed = np.ones(track.count)
    extra = np.concatenate([np.zeros(len(KEYS)), [args.pace / PACE_SCALE]])
    if args.resume and raceline.load(track) is not None:
        offset = raceline.load(track).offset
        saved_speed = raceline.load_speed(track)
        if saved_speed is not None:
            speed = saved_speed
        saved_pace, saved_tune = raceline.load_tuning(track)
        if saved_pace is not None:
            extra = np.array(
                [math.log(saved_tune[k] / TUNING[k]) / TUNE_SCALE for k in KEYS]
                + [saved_pace / PACE_SCALE])
        print(f"  resuming from the saved line "
              f"({np.abs(offset).max():.2f} m widest, "
              f"speed x{speed.min():.2f}..{speed.max():.2f})")
    else:
        offset = (np.zeros(track.count) if args.from_centreline
                  else raceline.min_curvature(track))

    t0 = time.perf_counter()
    best_f = FAIL
    for stage, k in enumerate(stages, 1):
        kv = max(8, k // 2)
        B = raceline._control_matrix(track.count, k)
        Bv = raceline._control_matrix(track.count, kv)
        # Zeros for the line and speed corrections: the stage starts exactly
        # where the last one finished, with nothing lost to a projection.
        base = (offset.copy(), speed.copy())
        x0 = np.concatenate([np.zeros(k + kv), extra])
        print()
        print(f"stage {stage}/{len(stages)}: {k} line points "
              f"(one every {track.length / k:.0f} m) + {kv} speed points, "
              f"{len(x0)} parameters, {args.generations} generations")

        def on_best(vec, f, gen, _B=B, _Bv=Bv, _base=base):
            _checkpoint(args.circuit, vec, _B, _Bv, lo, hi, _base, f, gen)

        # Later stages start from a good answer, but only its *coarse* part:
        # more than half of a fine stage's control points are new degrees of
        # freedom, projected from a curve that had nothing to say about them.
        # Halving the step each stage left the finest one exploring 4 cm of
        # lateral movement -- resolution handed over and then the step size to
        # use it taken away, which is why the chicanes still came out flat.
        # A gentle decay keeps roughly 20 cm at the finest, about the scale of
        # the detail being added.
        sigma = args.sigma * (0.8 ** (stage - 1))
        best_x, best_f = sep_cma(args.circuit, x0, B, Bv, lo, hi, base,
                                 args.generations, args.workers,
                                 args.seed + stage, sigma, on_best)
        offset, speed, tune, pace = _decode(best_x, B, Bv, lo, hi, base)
        extra = best_x[-(len(KEYS) + 1):]
        print(f"  stage {stage} -> {best_f:.3f} s")

    mins = (time.perf_counter() - t0) / 60.0
    print()
    print(f"{args.circuit}: {base_lap:.3f} -> {best_f:.3f} s "
          f"({best_f - base_lap:+.3f}, "
          f"{100 * (best_f - base_lap) / base_lap:+.1f}%) "
          f"in {mins:.1f} min")
    print(f"  pace {pace:.3f}   speed multiplier "
          f"{speed.min():.2f}..{speed.max():.2f} (mean {speed.mean():.2f})")
    for key in KEYS:
        print(f"  {key:22s} {TUNING[key]:8.3f} -> {tune[key]:8.3f}"
              f"  ({tune[key] / TUNING[key]:.2f}x)")


if __name__ == "__main__":
    main()
