"""Drive a whole race headlessly and check the ghost, the out lap and the gap.

The physics smoke test cannot cover any of this: the ghost owns an Ursina
entity, so it needs a window, and the out lap lives in the race's state machine
rather than in the vehicle model.

    python tools/ghost_check.py --circuit Monza --laps 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from panda3d.core import loadPrcFileData

loadPrcFileData('', 'sync-video 0')

from ursina import Ursina, time as utime

from game import app as ga
from game import config
from game.autopilot import Autopilot

DT = 1.0 / 60.0


def _cam_gap(game):
    """(distance to the player's car, distance to the AI's) from the camera.

    Which car the camera is on, measured rather than read off the flag: the
    flag is only what the key sets, and everything downstream of it is what
    can actually break.
    """
    from ursina import camera

    cam = np.array([camera.world_position.x, camera.world_position.z])
    return (float(np.hypot(*(cam - game.vehicle.pos))),
            float(np.hypot(*(cam - game.ghost.vehicle.pos))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--laps", type=int, default=2)
    # Slower than the ghost's, so the gap has a sign the test can check.
    ap.add_argument("--player-pace", type=float, default=0.84)
    ap.add_argument("--limit", type=float, default=700.0,
                    help="seconds of simulated time before giving up")
    args = ap.parse_args()

    Ursina(size=(320, 240), vsync=False, development_mode=False)
    ga.SESSION.update(laps=args.laps, mute=True, track=args.circuit)
    ga._build_race(args.circuit, args.laps, True)
    game = ga.GAME
    assert game.ghost is not None, "the ghost was not built"

    apart = float(np.hypot(*(game.ghost.vehicle.pos - game.vehicle.pos)))
    print(f"{args.circuit}: grid separation {apart:.2f} m")
    assert apart > config.CAR_BODY_LENGTH, "the two cars start inside each other"
    assert game.lap_num == 1, "a grand prix must be timed from the lights"

    pilot = Autopilot(game.track, game.surface, pace=args.player_pace)
    game.read_controls = lambda: pilot.controls(game.vehicle)
    game.state = 1                       # RACING
    game.vehicle.frozen = False
    game.ghost.start()

    deltas = []
    laps_seen = []
    watch = []                 # (label, distance from camera to each car)
    t = 0.0
    while t < args.limit and game.state != 2:      # 2 == FINISHED
        utime.dt = DT
        game.update()
        t += DT

        # Spectator view. Toggled twice, once the two cars are far enough
        # apart that "which one is the camera on" has an unambiguous answer:
        # measuring it while they are still side by side would pass whatever
        # the key did.
        apart = float(np.hypot(*(game.ghost.vehicle.pos - game.vehicle.pos)))
        if apart > 40.0 and len(watch) < 3:
            if not watch:
                watch.append(("before", _cam_gap(game)))
                game.on_key("g")
            elif len(watch) == 1:
                watch.append(("watching", _cam_gap(game)))
                game.on_key("g")
            else:
                watch.append(("after", _cam_gap(game)))
        i, _ = game.surface.progress(game.vehicle.pos)
        d = game.ghost.delta(i, game.session_time - game.lap_start)
        if d is not None:
            deltas.append(d)
        if game.lap_num not in laps_seen:
            laps_seen.append(game.lap_num)

    print(f"laps seen {laps_seen}   player best "
          f"{game.best_t:.2f} s   ghost best "
          f"{game.ghost.best_t:.2f} s" if game.best_t and game.ghost.best_t
          else f"laps seen {laps_seen}")
    print(f"gap samples {len(deltas)}   range "
          f"{min(deltas):+.3f} .. {max(deltas):+.3f} s")

    assert laps_seen[:2] == [1, 2], f"lap numbering started at {laps_seen[:2]}"
    assert game.state == 2, "the race never finished"
    assert game.best_t is not None and game.ghost.best_t is not None
    # The gap is a lap-relative delta, so it resets every lap rather than
    # growing without bound. A run of it that never comes back near zero means
    # the splits are not being reset with the lap.
    assert min(abs(np.array(deltas))) < 0.5, "the gap never reset at the line"
    assert max(np.abs(deltas)) < 60.0, "the gap grew past a plausible bound"
    # The slower car must end up slower.
    assert game.best_t > game.ghost.best_t, (
        f"the slower driver set the faster lap "
        f"({game.best_t:.2f} vs {game.ghost.best_t:.2f})")

    assert len(watch) == 3, "the two cars never separated enough to test G"
    for label, (to_player, to_ghost) in watch:
        print(f"  G {label:9s}: camera is {to_player:6.1f} m from the player, "
              f"{to_ghost:6.1f} m from the AI")
    assert watch[0][1][0] < watch[0][1][1], "the camera did not start on the player"
    assert watch[1][1][1] < watch[1][1][0], "G did not move the camera to the AI"
    assert watch[2][1][0] < watch[2][1][1], "G again did not give the camera back"
    print("\nOK")


if __name__ == "__main__":
    main()
