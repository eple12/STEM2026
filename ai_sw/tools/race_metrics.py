"""Measure the race driver's passing, defending and merging -- the baseline a
learned decision layer (game/raceai.py) has to beat.

    python tools/race_metrics.py                       # scenarios, rules drive
    python tools/race_metrics.py --seeds 12 --jobs 8
    python tools/race_metrics.py --races 4             # + whole 20-car races
    python tools/race_metrics.py --policy path.npz     # a trained policy drives the ego

Scenarios (game/raceenv.py) are short, built on purpose and repeatable from a
seed, on circuits with solved plans. Each is scored pass/fail on what a spectator
would call a good outcome, and the referee's counts are averaged:

* tow     the ego sits behind a slower car before a braking zone: success is
          being ahead after the corner, with no contact and no penalty.
* sbs     side by side into the braking zone: success is no contact, no penalty.
* defend  a faster car right behind: success is still being ahead, no penalty.
* merge   a stranded car at the road's edge, traffic coming: the ego is the
          first car coming up behind it; success is passing it with no contact
          and no penalty.
* pack    eight cars in sixty metres: success is no contact and no penalty
          for the ego.

Output goes to ``policy/raceai/<name>.json`` and a table on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CIRCUITS = ("Monza", "Spa", "Austin", "Silverstone", "Catalunya", "Sepang",
            "Hockenheim", "Montreal")
OUT = Path(__file__).resolve().parents[1] / "policy" / "raceai"


def _episode(args):
    circuit, kind, seed, policy_path, all_cars = args
    from game import raceai, raceenv
    sc = raceenv.make_scenario(circuit, kind, seed)
    pol = raceai.Policy.load(policy_path) if policy_path else None
    rep = raceenv.run(sc, pol, all_cars=all_cars)
    rep["ok"] = _success(kind, rep)
    return rep


def _race(args):
    circuit, seed, laps, policy_path, all_cars = args
    from game import raceai, raceenv
    pol = raceai.Policy.load(policy_path) if policy_path else None
    which = None
    if pol is not None:
        which = {i: pol for i in (range(20) if all_cars else [0])}
    return raceenv.run_race(circuit, seed, laps, policy_for=which)


def _success(kind: str, r: dict) -> bool:
    from game import raceenv
    return raceenv.success(kind, r)


def summarise(reps: list[dict]) -> dict:
    out = {}
    for kind in sorted({r["kind"] for r in reps}):
        rs = [r for r in reps if r["kind"] == kind]
        n = len(rs)
        m = lambda key: sum(r[key] for r in rs) / n
        out[kind] = {
            "episodes": n,
            "success_rate": round(sum(r["ok"] for r in rs) / n, 3),
            "contact_rate": round(sum(r["contacts"] > 0 for r in rs) / n, 3),
            "ego_penalised_rate": round(sum(r["ego_penalty_s"] > 0 for r in rs) / n, 3),
            "recoveries_per_ep": round(m("recoveries"), 3),
            "ego_passes": round(m("passes_by_ego"), 3),
            "passes_on_ego": round(m("passes_on_ego"), 3),
            "side_by_side_s": round(m("ego_side_by_side_s"), 2),
            "toward_follower_moves": round(m("defence_moves"), 2),
            "ego_lane_moves_m": round(m("ego_lane_moves_m"), 1),
            "ego_queue_s": round(m("ego_queue_s"), 2),
            "ego_kerb_s": round(m("ego_kerb_s"), 2),
        }
    return out


def summarise_races(reps: list[dict]) -> dict:
    n = len(reps)
    laps = sum(r["laps"] for r in reps)
    zone = [z for r in reps for z in r["pass_zone_offsets"] if z == z]
    return {
        "races": n, "laps": laps, "cars": reps[0]["cars"],
        "passes_per_lap": round(sum(r["passes"] for r in reps) / laps, 2),
        "contacts_per_race": round(sum(r["hits"] for r in reps) / n / 2, 2),
        "penalised_cars_per_race": round(sum(r["n_penalised"] for r in reps) / n, 2),
        "penalty_s_per_race": round(sum(r["total_penalty_s"] for r in reps) / n, 1),
        "recoveries_per_race": round(sum(r["recoveries"] for r in reps) / n, 2),
        "queue_pct": round(sum(r["queue_pct"] for r in reps) / n, 2),
        "kerb_pct": round(sum(r["kerb_pct"] for r in reps) / n, 2),
        "toward_follower_moves_per_lap": round(sum(r["defence_moves"] for r in reps) / laps, 2),
        "passes_before_braking_zone": round(sum(1 for z in zone if -400 < z < 0) / max(len(zone), 1), 2),
        "passes_in_or_after_zone": round(sum(1 for z in zone if z >= 0) / max(len(zone), 1), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--circuits", nargs="*", default=list(CIRCUITS))
    ap.add_argument("--kinds", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--races", type=int, default=0, help="also run this many whole races")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--all-cars", action="store_true",
                    help="the policy drives every car, not just the ego")
    ap.add_argument("--name", default="baseline_rules")
    args = ap.parse_args()

    from game import raceenv
    kinds = args.kinds or list(raceenv.KINDS)
    tasks = [(c, k, s, args.policy, args.all_cars) for c in args.circuits for k in kinds
             for s in range(args.seeds)]
    t0 = time.time()
    result = {"policy": args.policy or "rules", "circuits": args.circuits,
              "seeds": args.seeds}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        reps = list(ex.map(_episode, tasks, chunksize=4))
        result["scenarios"] = summarise(reps)
        result["episodes"] = reps
        if args.races:
            races = [(args.circuits[i % len(args.circuits)], 100 + i, args.laps, args.policy, args.all_cars)
                     for i in range(args.races)]
            rr = list(ex.map(_race, races))
            result["races"] = summarise_races(rr)
            result["race_reports"] = rr
    result["wall_s"] = round(time.time() - t0, 1)

    print(f"\n{len(tasks)} scenarios in {result['wall_s']} s  ({result['policy']})")
    head = f"{'kind':7s}{'n':>4s}{'ok':>7s}{'contact':>9s}{'ego pen':>9s}{'recov':>7s}{'passes':>8s}{'on ego':>8s}{'s-by-s':>8s}{'moves→':>8s}"
    print(head)
    for k, v in result["scenarios"].items():
        print(f"{k:7s}{v['episodes']:4d}{v['success_rate']:7.2f}{v['contact_rate']:9.2f}"
              f"{v['ego_penalised_rate']:9.2f}{v['recoveries_per_ep']:7.2f}{v['ego_passes']:8.2f}"
              f"{v['passes_on_ego']:8.2f}{v['side_by_side_s']:8.2f}{v['toward_follower_moves']:8.1f}")
    if "races" in result:
        print("\nwhole races:", json.dumps(result["races"], indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.name}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("->", path)


if __name__ == "__main__":
    main()
