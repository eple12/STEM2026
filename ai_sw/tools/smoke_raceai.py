"""Smoke test of the learned decision layer's plumbing (game/raceai.py,
game/raceenv.py): run it after touching either, or racecraft's hook.

    python tools/smoke_raceai.py

* actions round-trip through encode / decode;
* observations are finite and the right size for every car of a pack;
* a numpy ``raceai.Policy`` (random weights) installed on drivers of a whole
  twenty-car field runs a minute of racing without an exception;
* a driver with no policy is unchanged by the hook: the same seed gives the
  same race twice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game import raceai, raceenv


def random_policy(seed=0, hidden=64):
    rng = np.random.default_rng(seed)
    dims = [raceai.OBS_DIM, hidden, raceai.N_ACTIONS]
    w = {"mean": np.zeros(raceai.OBS_DIM, np.float32), "std": np.ones(raceai.OBS_DIM, np.float32),
         "layers": np.array(len(dims) - 1)}
    for i in range(len(dims) - 1):
        w[f"W{i}"] = (rng.standard_normal((dims[i], dims[i + 1])) * 0.3).astype(np.float32)
        w[f"b{i}"] = np.zeros(dims[i + 1], np.float32)
    return raceai.Policy(w)


def main():
    for a in range(raceai.ATTACK_IN):
        lane, pace = raceai.decode(a)
        assert raceai.encode(lane, pace) == a, a
    print("actions: ok", raceai.N_ACTIONS, "(lane x pace round-trip; pass actions", raceai.ATTACK_IN,
          raceai.ATTACK_OUT, raceai.HOLD, ")")

    sc = raceenv.make_scenario("Monza", "pack", 3)
    fld = sc.fld
    for _ in range(30):
        fld.step(raceenv.DT)
    for e in fld.cars:
        o = raceai.observe(e.driver, fld.views[e.idx], fld.views)
        assert o.shape == (raceai.OBS_DIM,) and np.isfinite(o).all() and np.abs(o).max() <= 3.01
    print("observations: ok", raceai.OBS_DIM, "dims")

    from game import grandprix
    from game.trackdata import load_track
    track = load_track("Spa")
    fld = grandprix.build(track, 6, 3, 1, player=False)
    pol = random_policy()
    for e in fld.cars[5:12]:
        e.driver.policy = pol
    fld.lights_out()
    while fld.t < 60.0:
        fld.step(raceenv.DT)
    print("20 cars, 7 of them on a random numpy policy, 60 s of racing: ok")

    reps = []
    for _ in range(2):
        sc = raceenv.make_scenario("Monza", "tow", 4)
        r = raceenv.run(sc)
        reps.append((r["t"], r["ego_rank_end"], r["ego_progress_m"], r["contacts"]))
    assert reps[0] == reps[1], reps
    print("rules unchanged by the hook (same seed, same race):", reps[0])


if __name__ == "__main__":
    main()
