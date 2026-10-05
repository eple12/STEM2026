"""Check the session rules headlessly: grand prix penalties, qualifying laps.

    python tools/session_check.py                    # both
    python tools/session_check.py --only limits      # the penalty maths, no window
    python tools/session_check.py --only quali --circuit Monza

``limits`` drives TrackLimits along the centreline by hand and hands each
excursion to race control (``racecontrol.py``): an honest trip wide is a
track-limits warning and costs nothing, a cut across the infield costs more
than the time it could have saved, and a trip off right after a contact is
let go.

``quali`` runs a qualifying session with a slower autopilot on a circuit that
has a recorded ghost lap and checks the ghost's life cycle -- hidden on the out
lap, launched from the line with each flying lap, gone at the line when it
beats the player -- and that a moment off the track deletes the lap.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game import config

DT = 1.0 / config.PHYSICS_HZ


def check_limits(circuit: str):
    from game.racecontrol import RaceControl
    from game.rules import TrackLimits
    from game.trackdata import load_track

    track = load_track(circuit)
    c, nrm, n = track.center, track.normal, track.count
    ds = track.length / n
    speed = 40.0

    def drive(lim, start, stop, off_from=None, off_to=None, t0=0.0,
              lateral=0.0):
        """Step along samples [start, stop) at *speed*, off for a range."""
        t = t0
        steps = int(round(ds / speed / DT))
        for k in range(start, stop):
            for _ in range(max(1, steps)):
                off = off_from is not None and off_from <= k < off_to
                p = c[k % n] + (nrm[k % n] * lateral if off else 0.0)
                lim.update(DT, k % n, p, off, t, speed)
                t += DT
        return t

    def judge(lim, rc, car=0):
        while lim.pending:
            rc.excursion(car, lim.pending.pop(0))

    # 1. Running wide: off for ~60 m of lap on the outside, nothing gained:
    #    a warning, no time.
    lim = TrackLimits(track, config.GP_OFFTRACK_REJOIN)
    rc = RaceControl(1)
    k0 = n // 4
    m = int(60.0 / ds)
    t = drive(lim, k0, k0 + 3 * m, off_from=k0 + m, off_to=k0 + 2 * m,
              lateral=track.w_right.max() + 4.0)
    t = drive(lim, k0 + 3 * m, k0 + 3 * m + int(80 / ds), t0=t)
    judge(lim, rc)
    print(f"  run wide : {lim.incidents} incident, {rc.penalty(0):.2f} s, "
          f"strikes {rc.cars[0].strikes}: {[m.text for m in rc.messages]}")
    assert lim.incidents == 1, "one excursion counted as several"
    assert rc.penalty(0) == 0.0, "running wide was penalised"
    assert rc.cars[0].strikes == 1

    # ...and the same trip just after a contact: nothing at all.
    lim = TrackLimits(track, config.GP_OFFTRACK_REJOIN)
    rc = RaceControl(2)
    rc.cars[0].last_contact = 0.0
    drive(lim, k0, k0 + 3 * m, off_from=k0 + m, off_to=k0 + 2 * m,
          lateral=track.w_right.max() + 4.0)
    judge(lim, rc)
    assert rc.penalty(0) == 0.0 and rc.cars[0].strikes == 0, "pushed off, yet judged"
    print("  pushed off: excused")

    # 2. Cutting. The circuit's best shortcut in a band of lap distance --
    #    the two points furthest apart round the lap for how close they are
    #    in a straight line -- driven straight across, off the road, at the
    #    same speed. A chicane-sized one, then the biggest on the circuit.
    from game.surface import Surface

    arc = track.arclen
    surf = Surface(track)

    def best_shortcut(lo_m, hi_m):
        step_k = max(1, n // 400)
        best = (0.0, 0, 0)
        for a in range(0, n, step_k):
            for b in range(a + 1, n, step_k):
                along = float(arc[b] - arc[a])
                if along > hi_m:
                    break
                if along < lo_m:
                    continue
                chord = float(np.hypot(*(c[b] - c[a])))
                if along - chord > best[0]:
                    best = (along - chord, a, b)
        return best[1], best[2]

    def cut(label, lo_m, hi_m):
        ka, kb = best_shortcut(lo_m, hi_m)
        along = float(arc[kb] - arc[ka])
        chord = float(np.hypot(*(c[kb] - c[ka])))
        lim = TrackLimits(track, config.GP_OFFTRACK_REJOIN)
        rc = RaceControl(1)
        t = drive(lim, ka - 10, ka + 1)
        m_steps = max(1, int(chord / speed / DT))
        for st in range(1, m_steps + 1):
            p = c[ka] + (c[kb] - c[ka]) * (st / m_steps)
            i, _ = surf.progress(p)
            lim.update(DT, i, p, True, t, speed)
            t += DT
        drive(lim, kb + 1, kb + 1 + int(80 / ds), t0=t)
        judge(lim, rc)
        saved = (along - chord) / speed
        print(f"  {label:9s}: {along:.0f} m of lap in a {chord:.0f} m line "
              f"saves {saved:.2f} s at {speed:.0f} m/s; penalty "
              f"{rc.penalty(0):.2f} s")
        assert lim.incidents == 1
        assert rc.penalty(0) > saved, "cutting the track paid for itself"
        return rc.penalty(0)

    cut("chicane", 80.0, 260.0)
    cut("big cut", 260.0, 0.5 * track.length)

    # 3. Progress unwraps across the line.
    lim = TrackLimits(track, config.GP_OFFTRACK_REJOIN)
    drive(lim, n - 20, n + 20)
    assert abs(lim.progress - float(track.arclen[19])) < 1.0, lim.progress
    print(f"  progress : {lim.progress:.1f} m after crossing the line")
    print("limits OK")


def check_quali(circuit: str, pace: float, limit: float):
    from ursina import Ursina, time as utime
    from panda3d.core import loadPrcFileData

    loadPrcFileData('', 'sync-video 0')
    from game import app as ga
    from game.autopilot import Autopilot
    from game.replay import ReplayGhost

    Ursina(size=(320, 240), vsync=False, development_mode=False)
    ga._build_race(circuit, 3, True, mode="quali")
    game = ga.GAME
    g = game.ghost
    assert isinstance(g, ReplayGhost), f"no ghost lap for {circuit}"
    assert game.lap_num == 0 and game._start_lights() == -1

    pilot = Autopilot(game.track, game.surface, pace=pace)
    game.read_controls = lambda: pilot.controls(game.vehicle)
    game.state = 1                                        # RACING
    game.vehicle.frozen = False

    seen = dict(visible_on_out=False, launched=[], vanished=[],
                deltas=[])
    kicked = False
    t, last_lap = 0.0, 0
    while t < limit and game.lap_num < 4:
        utime.dt = 1.0 / 60.0
        # Lap 2: shove the car sideways past the white line for a moment,
        # keeping its speed, then put it back.
        kick = (game.lap_num == 2 and not kicked
                and game.session_time - game.lap_start > 20.0)
        if kick:
            i, _ = game.surface.progress(game.vehicle.pos)
            back = game.vehicle.pos.copy()
            game.vehicle.pos = back + game.track.normal[i] * (
                game.track.w_right[i] + 6.0)
        was_visible = g.visible
        game.update()
        if kick:
            kicked = True
            assert game.lap_invalid, "leaving the track did not delete the lap"
            game.vehicle.pos = back
        t += 1.0 / 60.0

        if game.lap_num == 0 and g.visible:
            seen["visible_on_out"] = True
        if game.lap_num != last_lap:
            seen["launched"].append((game.lap_num, g.visible))
            last_lap = game.lap_num
            if game.lap_num == 3:
                assert game.last_invalid, "the kicked lap was not deleted"
                assert game.best_t is not None and abs(
                    game.best_t - lap1) < 1e-9, "a deleted lap set the best"
            if game.lap_num == 2:
                lap1 = game.last_t
                assert not game.last_invalid, "a clean lap was deleted"
        elif was_visible and not g.visible:
            seen["vanished"].append((game.lap_num,
                                     game.session_time - game.lap_start))
        i, _ = game.surface.progress(game.vehicle.pos)
        if game.lap_num >= 1:
            d = g.delta(i, game.session_time - game.lap_start)
            if d is not None:
                seen["deltas"].append(d)

    print(f"  ghost lap {g.rec.lap_time:.3f} s   player laps "
          f"{game.last_t and round(game.last_t, 3)} (best {game.best_t:.3f})")
    print(f"  launches (lap, visible): {seen['launched']}")
    print(f"  vanished (lap, player lap time): "
          f"{[(k, round(s, 2)) for k, s in seen['vanished']]}")
    assert not seen["visible_on_out"], "the ghost ran during the out lap"
    assert all(vis for lap, vis in seen["launched"]), (
        "the ghost did not launch with a flying lap")
    assert len(seen["vanished"]) >= 2, "the ghost never cleared the line"
    for _lap, at in seen["vanished"]:
        assert abs(at - g.rec.lap_time) < 0.5, (
            f"the ghost vanished {at:.2f} s into the lap, not at its lap time")
    assert np.median(seen["deltas"]) > 0, "a slower player read as ahead"
    rows = game._standings(0, None)
    assert rows[0]["tla"] == "AI" and rows[0]["gap"] == "POLE", rows
    print("quali OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--only", choices=("limits", "quali"))
    ap.add_argument("--player-pace", type=float, default=0.84)
    ap.add_argument("--limit", type=float, default=900.0)
    args = ap.parse_args()
    if args.only in (None, "limits"):
        check_limits(args.circuit)
    if args.only in (None, "quali"):
        check_quali(args.circuit, args.player_pace, args.limit)


if __name__ == "__main__":
    main()
