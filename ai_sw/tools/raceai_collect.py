"""Record what the rule-based race drivers decide, as (observation, action)
pairs, for teaching a policy to do the same (tools/train_raceai.py --bc).

    python tools/raceai_collect.py --out data/raceai_rules.npz
    python tools/raceai_collect.py --circuits Monza Spa --seeds 4 --races 2

Every driver in every scenario (game/raceenv.py) and every driver in a whole
twenty-car race logs once per plan tick: ``raceai.observe`` -- what a policy
would see -- and ``raceai.rule_action`` -- the nearest action to what the rules
chose. Long stretches of one car alone on the line with nothing near are kept
only now and then, so the pairs are mostly traffic.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "policy" / "raceai"
#: Keep this share of the "alone on the line" decisions.
KEEP_ALONE = 0.15


class _Logger:
    """``before`` takes the observation ahead of the rules' decision, ``__call__``
    pairs it with what they decided."""

    def __init__(self, rng, obs_list, act_list, tag_list, tag):
        from game import raceai
        self.raceai = raceai
        self.n0 = raceai.OBS_NAMES.index("n0_valid")
        self.rng, self.obs, self.act, self.tags, self.tag = rng, obs_list, act_list, tag_list, tag

    def before(self, driver, me, field):
        return self.raceai.observe(driver, me, field)

    def __call__(self, driver, me, field, pace_ratio, before=None):
        if driver.mode != "race" or before is None:
            return                            # recovery is not a decision of this layer
        ra = self.raceai
        a = ra.rule_action(driver, me, pace_ratio, before)
        if before[self.n0] == 0.0 and a == ra.DEFAULT_ACTION and self.rng.random() > KEEP_ALONE:
            return
        self.obs.append(before)
        self.act.append(a)
        self.tags.append(self.tag)


def _logger(rng, obs_list, act_list, tag_list, tag):
    return _Logger(rng, obs_list, act_list, tag_list, tag)


def _pack(obs, acts, tags):
    if not obs:
        return None
    return np.stack(obs), np.asarray(acts, np.int16), np.asarray(tags, np.int16)


def _scenario(args):
    circuit, kind, seed = args
    from game import raceenv
    rng = np.random.default_rng(seed + 17)
    obs, acts, tags = [], [], []
    sc = raceenv.make_scenario(circuit, kind, seed)
    raceenv.run(sc, watch=_logger(rng, obs, acts, tags, raceenv.KINDS.index(kind)))
    return _pack(obs, acts, tags)


def _race(args):
    circuit, seed, laps = args
    from game import raceenv
    rng = np.random.default_rng(seed + 29)
    obs, acts, tags = [], [], []
    raceenv.run_race(circuit, seed, laps,
                     watch=_logger(rng, obs, acts, tags, len(raceenv.KINDS)))
    return _pack(obs, acts, tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", nargs="*", default=None,
                    help="default: every circuit with solved plans")
    ap.add_argument("--seeds", type=int, default=6, help="scenario seeds per circuit and kind")
    ap.add_argument("--seed0", type=int, default=1000,
                    help="first seed; the evaluation uses 0.. so these never overlap")
    ap.add_argument("--races", type=int, default=0)
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "rules_pairs.npz"))
    args = ap.parse_args()

    from game import raceenv
    circuits = args.circuits or raceenv.circuits()
    tasks = [(c, k, args.seed0 + s) for c in circuits for k in raceenv.KINDS
             for s in range(args.seeds)]
    t0 = time.time()
    chunks = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(_scenario, tasks, chunksize=4)):
            if r is not None:
                chunks.append(r)
            if (i + 1) % 100 == 0:
                print(f"  scenarios {i + 1}/{len(tasks)}  {sum(len(c[1]) for c in chunks)} pairs  "
                      f"{time.time() - t0:.0f}s", flush=True)
        for r in ex.map(_race, [(circuits[i % len(circuits)], args.seed0 + i, args.laps)
                                for i in range(args.races)]):
            if r is not None:
                chunks.append(r)
                print(f"  race: {len(r[1])} pairs  {time.time() - t0:.0f}s", flush=True)
    obs = np.concatenate([c[0] for c in chunks])
    act = np.concatenate([c[1] for c in chunks])
    tag = np.concatenate([c[2] for c in chunks])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=obs, act=act, tag=tag)
    print(f"{len(act)} pairs -> {args.out}   ({time.time() - t0:.0f}s)")
    from game import raceai
    hist = np.bincount(act, minlength=raceai.N_ACTIONS)
    print("action histogram (lane rows x pace cols 0.94/1.0/1.03):")
    for li, lane in enumerate(raceai.LANES):
        print(f"  lane {lane:+5.2f}: " + "  ".join(f"{hist[li * raceai.N_PACES + pi]:7d}"
                                                     for pi in range(raceai.N_PACES)))


if __name__ == "__main__":
    main()
