"""Run a whole grand prix headless and measure the racing.

Twenty AI cars (``game/teams.py``) on one circuit with the race driver
(``game/racecraft.py``) and the field (``game/field.py``): contact,
slipstream, passing, recovery. Prints the classification and what happened --
passes, contacts, recoveries, mistakes -- and can draw it.

    python tools/race_sim.py --circuit Monza --laps 3
    python tools/race_sim.py --laps 2 --gif start.gif --gif-span 0 25
    python tools/race_sim.py --calibrate          # lap time per difficulty

Needs the circuit's solved plans (``tools/mintime.py --grip 0.XX --tag gXX``);
drivers are mapped onto whichever of ``teams.PLAN_GRIPS`` exist.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game import config, grandprix, teams
from game.field import Entrant, Field
from game.racecraft import RaceDriver, TrackFrame
from game.rules import TrackLimits
from game.surface import Surface
from game.trackdata import load_track
from game.vehicle import Vehicle

DT = 1.0 / 60.0


def build_field(track, level: int, laps: int, seed: int, n_cars: int = 20,
                grid: str = "rating"):
    if not grandprix.ready(track.name):
        raise SystemExit(f"no solved plans for {track.name}: run tools/mintime.py")
    return grandprix.build(track, level, laps, seed, player=False,
                           n_cars=n_cars, grid=grid)


def run(fld: Field, record_every: int = 6, limit: float | None = None,
        verbose: bool = True):
    """Lights out to the flag. Returns snapshots [(t, xy, yaw)]."""
    fld.lights_out()
    snaps = []
    tick = 0
    wall = time.perf_counter()
    limit = limit or 140.0 * (fld.total_laps + 1)
    while not fld.finished() and fld.t < limit:
        fld.step(DT)
        if fld.leader_done is not None and fld.t > fld.leader_done + 60.0:
            break
        if tick % record_every == 0:
            snaps.append((fld.t,
                          np.array([e.vehicle.pos for e in fld.cars]),
                          np.array([e.vehicle.yaw for e in fld.cars])))
        tick += 1
        if verbose and tick % (60 * 30) == 0:
            lead = fld.order()[0]
            print(f"  t {fld.t:6.1f} s  leader {lead.name:13s} lap {lead.laps + 1}"
                  f"  passes {fld.overtakes}  wall {time.perf_counter() - wall:5.0f} s",
                  flush=True)
    return snaps


def report(fld: Field):
    order = fld.order()
    lead_t = fld.total_time(order[0])
    print(f"\n{'pos':>3} {'driver':13s} {'team':14s} {'grid':>4} {'laps':>4} "
          f"{'gap':>9} {'best':>7} {'hits':>4} {'rec':>3} {'err':>3} {'att':>3} {'pen':>4}")
    for p, e in enumerate(order, 1):
        if e.finish_t is not None and lead_t is not None:
            if e.laps == order[0].laps:
                gap = "leader" if p == 1 else f"+{fld.total_time(e) - lead_t:7.2f}"
            else:
                gap = f"+{order[0].laps - e.laps} lap"
        else:
            gap = "running"
        best = f"{e.best:7.2f}" if e.best else "   -   "
        d = e.driver
        print(f"{p:3d} {e.name:13s} {e.team:14s} {e.grid_slot + 1:4d} {e.laps:4d} "
              f"{gap:>9} {best} {e.hits:4d} {d.recoveries:3d} {d.mistakes_made:3d} "
              f"{d.attacks:3d} {fld.rc.penalty(e.idx):4.0f}")
    hits = sum(e.hits for e in fld.cars) // 2
    rec = sum(e.driver.recoveries for e in fld.cars)
    print(f"\npasses on track {fld.overtakes}   contacts {hits}   recoveries {rec}   "
          f"track-limit incidents {sum(e.limits.incidents for e in fld.cars)}")
    kinds = {}
    for m in fld.rc.messages:
        kinds[m.kind] = kinds.get(m.kind, 0) + 1
    print("race control:", kinds)
    for m in fld.rc.messages:
        if m.kind in ("pen", "flag"):
            print(f"  {m.t:6.1f} s  {fld.cars[m.car].tla}  {m.text}")


def lap_chart(fld: Field, snaps, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Position every second, from the race-order the snapshots imply.
    t_all = [s[0] for s in snaps]
    L = fld.track.length
    prog = np.zeros((len(snaps), len(fld.cars)))
    # Unwrapped progress from snapshot positions via the frame.
    last = None
    surf = [Surface(fld.track) for _ in fld.cars]
    for j, (t, xy, _) in enumerate(snaps):
        s = np.array([fld.frame.locate(surf[c], xy[c])[1] for c in range(len(fld.cars))])
        if last is None:
            base = np.where(s > 0.5 * L, s - L, s)
        else:
            d = (s - last[1] + 0.5 * L) % L - 0.5 * L
            base = last[0] + d
        prog[j] = base
        last = (base, s)
    rank = np.argsort(np.argsort(-prog, axis=1), axis=1) + 1
    fig, ax = plt.subplots(figsize=(14, 7))
    for c, e in enumerate(fld.cars):
        ax.plot(t_all, rank[:, c], color=e.color, lw=1.6,
                ls="-" if c % 2 == 0 else "--")
        ax.text(t_all[-1] + 1, rank[-1, c], e.name, fontsize=8, va="center",
                color=e.color)
        ax.text(t_all[0] - 1, rank[0, c], e.name, fontsize=8, va="center",
                ha="right", color=e.color)
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(fld.cars) + 1))
    ax.set_xlabel("race time (s)")
    ax.set_ylabel("position")
    ax.set_title(f"{fld.track.name}: running order ({fld.overtakes} passes on track)")
    ax.set_xlim(t_all[0] - 12, t_all[-1] + 14)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def animate(fld: Field, snaps, out: Path, t0: float, t1: float,
            span: float = 170.0, fps: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    t = fld.track
    frames = [s for s in snaps if t0 <= s[0] <= t1]
    left = t.center - t.normal * t.w_left[:, None]
    right = t.center + t.normal * t.w_right[:, None]
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#203020")
    ax.set_facecolor("#2f4f2f")
    poly = np.vstack([left, right[::-1]])
    ax.fill(poly[:, 0], poly[:, 1], color="#555555", zorder=0)
    ax.plot(*np.vstack([left, left[:1]]).T, color="white", lw=0.8)
    ax.plot(*np.vstack([right, right[:1]]).T, color="white", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    hl, hw = config.BODY_TO_FRONT, config.BODY_HALF_WIDTH
    tl = config.BODY_TO_REAR
    shapes = []
    for e in fld.cars:
        p = Polygon(np.zeros((4, 2)), closed=True, fc=e.color, ec="black",
                    lw=0.5, zorder=3)
        ax.add_patch(p)
        shapes.append(p)
    title = ax.set_title("", color="white")
    # Follow the middle of the pack, not the leader: that is where it happens.
    def draw(j):
        tt, xy, yaw = frames[j]
        for c, p in enumerate(shapes):
            f = np.array([math.sin(yaw[c]), math.cos(yaw[c])])
            r = np.array([math.cos(yaw[c]), -math.sin(yaw[c])])
            q = xy[c]
            p.set_xy(np.array([q + f * hl + r * hw, q + f * hl - r * hw,
                               q - f * tl - r * hw, q - f * tl + r * hw]))
        mid = np.median(xy, axis=0)
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        title.set_text(f"{t.name}  t = {tt:5.1f} s")
        return shapes
    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def calibrate(track, seed: int = 0):
    """Lap time of a driver alone, per difficulty level and rating."""
    grips = grandprix.solved_grips(track.name)
    plans = grandprix.PlanCache(track.name)
    ref = plans(f"g{round(max(grips) * 100):02d}")
    frame = TrackFrame(track, ref)
    print(f"plans available: {grips}")
    best_team = teams.TEAMS[0]
    print(f"{'level':14s} " + " ".join(f"rating {r:4.2f}" for r in (1.0, 0.8, 0.6)))
    for lv, lvl in teams.DIFFICULTY.items():
        row = []
        for rating in (1.0, 0.8, 0.6):
            d = teams.Driver("cal", rating)
            skill, power = teams.skill_for(d, best_team, lvl, grips)
            skill.mistakes = 0.0
            skill.consistency = 0.0
            v = Vehicle()
            v.power_scale = power
            drv = RaceDriver(track, frame, plans(skill.plan), skill,
                             np.random.default_rng(seed))
            drv.idx = 0
            e = Entrant(idx=0, name="cal", team=best_team.name, color=(1, 1, 1),
                        vehicle=v, surface=Surface(track), driver=drv,
                        limits=TrackLimits(track))
            fld = Field(track, frame, [e], laps=3)
            fld.grid()
            fld.lights_out()
            while not fld.finished() and fld.t < 500:
                fld.step(DT)
            row.append(min(e.lap_times[1:]) if len(e.lap_times) > 1 else math.nan)
        print(f"{lv} {lvl.name:12s} " + " ".join(f"{x:11.2f}" for x in row),
              flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--cars", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", default="rating", choices=("rating", "reverse"))
    ap.add_argument("--chart", default=None, help="write a lap chart PNG")
    ap.add_argument("--gif", default=None, help="write a top-down GIF")
    ap.add_argument("--gif-span", type=float, nargs=2, default=(0.0, 25.0),
                    metavar=("T0", "T1"))
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args(argv)
    track = load_track(args.circuit)
    if args.calibrate:
        calibrate(track, args.seed)
        return
    fld = build_field(track, args.level, args.laps, args.seed, args.cars, args.grid)
    t0 = time.perf_counter()
    snaps = run(fld)
    print(f"simulated {fld.t:.0f} s of racing in {time.perf_counter() - t0:.0f} s")
    report(fld)
    if args.chart:
        lap_chart(fld, snaps, Path(args.chart))
    if args.gif:
        animate(fld, snaps, Path(args.gif), *args.gif_span)


if __name__ == "__main__":
    main()
