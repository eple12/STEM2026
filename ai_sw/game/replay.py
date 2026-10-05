"""Qualifying ghost: the AI's fastest clean lap, recorded once and replayed.

In a grand prix the AI is a live driver (ghost.py). In qualifying it is a
lap time to beat, and a lap time is one lap: the same car, on the same line,
every time round, so a player can learn exactly where it is quicker. So the AI
drives a handful of flying laps offline, the fastest one that never put all
four wheels past the white line or touched a wall is kept -- a PERFECT lap in
the training tools' terms -- and the game plays that back like a video.

    python tools/record_ghost.py --circuit Monza

The recording is the car's state at every physics step from one crossing of
the line to the next, so it replays at any frame rate and needs nothing about
the driver that made it: no policy, no planner, not even the same physics
rate (playback interpolates on time, not on step count).
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .surface import Surface
from .trackdata import Track
from .vehicle import Controls, Vehicle

#: Column order of ``frames``.
COLS = ("x", "z", "yaw", "vx", "vz", "yaw_rate", "steer_angle", "steer_input",
        "lat_accel", "long_accel", "downforce", "throttle", "brake", "tc_cut",
        "esc_cut", "index")
C = {name: k for k, name in enumerate(COLS)}


def lap_path(circuit: str, level: int | None = None):
    suffix = "" if level is None else f"_L{level}"
    return config.GHOST_LAP_DIR / f"{circuit}{suffix}.npz"


class Recording:
    def __init__(self, frames, hz, lap_time, splits, sectors, driver):
        self.frames = frames
        self.hz = float(hz)
        self.lap_time = float(lap_time)
        self.splits = splits
        self.sectors = [float(s) for s in sectors]
        self.driver = driver


def load(circuit: str, track: Track | None = None,
         level: int | None = None) -> Recording | None:
    """The circuit's ghost lap, or None if it has none (or a stale one)."""
    path = lap_path(circuit, level)
    if not path.is_file():
        return None
    try:
        d = np.load(path, allow_pickle=False)
        rec = Recording(d["frames"], d["hz"], d["lap_time"], d["splits"],
                        d["sectors"], str(d["driver"]))
    except Exception as exc:                   # a truncated or old file
        print(f"ghost lap {path.name}: unreadable ({exc})")
        return None
    if track is not None and len(rec.splits) != track.count:
        # Recorded against a different sampling of the circuit; its indices
        # and splits would point at the wrong places.
        print(f"ghost lap {path.name}: recorded for a different track build")
        return None
    return rec


def lap_time(circuit: str) -> float | None:
    """Just the lap time, for the menu. Cheap: no frames are decompressed."""
    path = lap_path(circuit)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            return float(d["lap_time"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
def _frame(v: Vehicle, ctl: Controls, i: int) -> list[float]:
    t = v.tele
    return [v.pos[0], v.pos[1], v.yaw, v.vel[0], v.vel[1], v.yaw_rate,
            v.steer_angle, v.steer_input, t.lat_accel, t.long_accel,
            t.downforce, ctl.throttle, ctl.brake, v.tc_cut, v.esc_cut, i]


def record(track: Track, laps: int = 5, limit: float | None = None,
           verbose: bool = True, start: dict | None = None,
           accept=None, starts: list | None = None, driver_factory=None,
           power_scale: float = 1.0):
    """Drive *laps* flying laps and return (best clean Recording or None,
    [(lap time, clean) for every flying lap]).

    ``start`` (default: the standing grid start) is a rolling start instead:
    ``{"back_m", "speed", "lateral"}`` places the car that many metres before
    the line, moving at ``speed`` m/s and ``lateral`` m off the centreline,
    so the first crossing already starts a timed lap. The state at the line
    differs a little from attempt to attempt, so the laps differ a little
    too -- which is what lets a best-of-N pick a faster clean one.

    Every lap's crossing state (speed, signed offset from the centreline,
    heading error) is appended to ``starts`` when given. ``accept(state)``,
    when given, decides whether a clean lap may become the returned best:
    a lap that began faster or better placed than the AI can reach from an
    ordinary flying lap would be a lap time it cannot really drive."""
    from .ghost import make_driver

    dt = 1.0 / config.PHYSICS_HZ
    surface = Surface(track)
    v = Vehicle()
    v.power_scale = power_scale
    pos, yaw = track.start_pose()
    v.place(pos, yaw)
    v.frozen = False
    if driver_factory is not None:
        # A given driver (a difficulty level's pole sitter, say) rather than
        # the ghost's own.
        driver = driver_factory(surface)
    else:
        driver, _pilot, _line, _pace = make_driver(track, surface)
    name = type(driver).__name__

    n = track.count
    b1, b2 = track.sector_bounds()
    limit = limit if limit is not None else 300.0 * (laps + 1)
    armed = False
    if start is not None:
        sp = track.length / track.count
        j = (n - max(int(round(start["back_m"] / sp)), 1)) % n
        pos = track.center[j] + track.normal[j] * float(start.get("lateral", 0.0))
        yaw = float(math.atan2(track.tangent[j, 0], track.tangent[j, 1]))
        v.place(pos, yaw)
        v.vel = float(start["speed"]) * np.array([math.sin(yaw), math.cos(yaw)])
        if hasattr(driver, "_on_grid"):
            driver._on_grid = False
        armed = True
    last_i, _ = surface.progress(v.pos)
    buf: list | None = None            # None until the first flying lap
    start_state: dict | None = None
    splits = np.full(n, np.nan)
    dirty = False
    laps_seen = []
    best: Recording | None = None
    steps = 0

    def fill(after, upto, value):
        for k in range(after + 1, upto + 1):
            splits[k % n] = value

    while len(laps_seen) < laps and steps * dt < limit:
        ctl = driver.controls(v)
        v.step(ctl, dt, surface)
        steps += 1
        i, _ = surface.progress(v.pos)
        off = (not v.on_track) or v.hit_wall
        if buf is not None:
            dirty = dirty or off

        crossing = False
        if 0.4 * n <= i <= 0.6 * n:
            armed = True
        elif armed and i < 0.1 * n:
            fwd = np.array([math.sin(v.yaw), math.cos(v.yaw)])
            crossing = float(np.dot(v.vel, fwd)) > 0.0

        frame = _frame(v, ctl, i)
        if crossing:
            armed = False
            if buf is not None:
                buf.append(frame)
                t = (len(buf) - 1) * dt
                fill(last_i, n - 1, t)
                clean = not dirty and bool(np.all(np.isfinite(splits)))
                laps_seen.append((t, clean))
                ok = accept is None or accept(start_state)
                if starts is not None:
                    starts.append(dict(start_state, time=t, clean=clean,
                                       accepted=bool(clean and ok)))
                if verbose:
                    print(f"  lap {len(laps_seen)}: {t:8.3f} s  "
                          f"{'PERFECT' if clean else 'dirty'}"
                          f"{'' if ok else '  (start state out of range)'}")
                if clean and ok and (best is None or t < best.lap_time):
                    s1, s2 = float(splits[b1]), float(splits[b2])
                    best = Recording(np.asarray(buf, dtype=np.float32),
                                     config.PHYSICS_HZ, t,
                                     splits.astype(np.float32).copy(),
                                     (s1, s2 - s1, t - s2), name)
            buf = [frame]
            tan = track.tangent[i]
            start_state = {
                "speed": float(np.linalg.norm(v.vel)),
                "lateral": float(np.dot(v.pos - track.center[i],
                                        track.normal[i])),
                "heading": float((math.atan2(tan[0], tan[1]) - v.yaw + math.pi)
                                 % (2 * math.pi) - math.pi),
            }
            splits = np.full(n, np.nan)
            fill(-1, i, 0.0)
            dirty = off
        elif buf is not None:
            buf.append(frame)
            gap = (i - last_i) % n
            if 0 < gap < n // 4:
                fill(last_i, last_i + gap, (len(buf) - 1) * dt)
        last_i = i
    return best, laps_seen


def save(circuit: str, rec: Recording, level: int | None = None):
    config.GHOST_LAP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(lap_path(circuit, level), frames=rec.frames, hz=rec.hz,
                        lap_time=rec.lap_time, splits=rec.splits,
                        sectors=np.asarray(rec.sectors), driver=rec.driver)


def level_lap(circuit: str, level: int, track: Track,
              progress=None) -> Recording | None:
    """Qualifying's lap to beat at a difficulty level: the pole sitter's.

    The fastest AI driver at that level (``grandprix.quali_table``) drives
    its solved plan, at its pace and with its engine, for a flying lap, and
    the lap is kept in ``assets/ghosts/<circuit>_L<level>.npz``. Recorded
    the first time it is asked for -- a few seconds behind the loading card
    -- or ahead of time by ``tools/solve_all.py``. None where the circuit
    has no solved plans.
    """
    rec = load(circuit, track, level)
    if rec is not None:
        return rec
    rec = record_level(circuit, level, track, progress=progress)
    if rec is not None:
        save(circuit, rec, level)
    return rec


def record_level(circuit: str, level: int, track: Track,
                 progress=None) -> Recording | None:
    from . import grandprix
    from .mintime_driver import MinTimeDriver, path_file
    if not grandprix.ready(circuit):
        return None
    pole = grandprix.quali_table(circuit, level)[0]
    plan = path_file(circuit, pole.skill.plan)
    tick = [0]

    def factory(surface):
        drv = MinTimeDriver(track, surface, plan, pace=pole.skill.pace)
        if progress is None:
            return drv
        # Keep the loading card's bar moving while the lap is driven.
        inner = drv.controls

        def controls(vehicle):
            tick[0] += 1
            if tick[0] % 600 == 0:
                progress()
            return inner(vehicle)
        drv.controls = controls
        return drv

    best, _laps = record(track, laps=1, verbose=False, driver_factory=factory,
                         power_scale=pole.power)
    if best is not None:
        best.driver = f"{pole.driver.name} (L{level})"
    return best


# ---------------------------------------------------------------------------
class ReplayGhost:
    """Plays a Recording back, looking to the race like a live ``Ghost``.

    It is only on screen while it has a lap to show: ``launch`` puts it on
    the line as the player starts a flying lap, and it vanishes as it reaches
    the line again. If the player is quicker, their next lap launches it
    again before it gets there.
    """

    def __init__(self, track: Track, rec: Recording):
        from .ghost import ghost_car

        self.track = track
        self.rec = rec
        self.vehicle = Vehicle()
        self.vehicle.frozen = True

        # The Ghost interface the HUD and the timing tower read.
        self.splits = rec.splits.astype(float)
        self.prev_splits = self.splits
        self.best_t = self.last_t = rec.lap_time
        self.best_sectors: list[float | None] = list(rec.sectors)
        self.sectors: list[float | None] = [None, None, None]
        self.sector = 0
        self.lap_num = 0
        self.lap_time = 0.0
        self.controls = Controls()
        self._last_i = int(rec.frames[0, C["index"]])
        self.visible = False
        self._t = 0.0

        self._apply(0.0)
        self.sector = 0
        self.vehicle.prev_pos = self.vehicle.pos.copy()
        self.vehicle.prev_yaw = self.vehicle.yaw
        self.car = ghost_car(self.vehicle)
        self.car.enabled = False

    # -- playback -------------------------------------------------------
    def launch(self):
        """Start the lap over from the line, visible."""
        self._t = 0.0
        self.lap_num += 1
        self.sectors = [None, None, None]
        self._apply(0.0)
        v = self.vehicle
        v.prev_pos, v.prev_yaw = v.pos.copy(), v.yaw
        self.visible = True
        self.car.enabled = True

    def step(self, dt: float):
        if not self.visible:
            return
        self._t += dt
        if self._t >= self.rec.lap_time:
            # At the line: gone until the player starts another lap.
            self.visible = False
            self.car.enabled = False
            self.sectors = list(self.rec.sectors)
            return
        v = self.vehicle
        v.prev_pos, v.prev_yaw = v.pos.copy(), v.yaw
        self._apply(self._t)

    def _apply(self, t: float):
        f = self.rec.frames
        x = min(t * self.rec.hz, len(f) - 1.0)
        k = int(x)
        k1 = min(k + 1, len(f) - 1)
        a = x - k
        row = f[k] + (f[k1] - f[k]) * a
        # Yaw across the +-pi seam is interpolated the short way round.
        dy = (float(f[k1, C["yaw"]]) - float(f[k, C["yaw"]]) + math.pi) \
            % (2 * math.pi) - math.pi
        v = self.vehicle
        v.pos = np.array([row[C["x"]], row[C["z"]]], dtype=float)
        v.yaw = float(f[k, C["yaw"]]) + dy * a
        v.vel = np.array([row[C["vx"]], row[C["vz"]]], dtype=float)
        v.yaw_rate = float(row[C["yaw_rate"]])
        v.steer_angle = float(row[C["steer_angle"]])
        v.steer_input = float(row[C["steer_input"]])
        v.tc_cut = float(row[C["tc_cut"]])
        v.esc_cut = float(row[C["esc_cut"]])
        v.tele.lat_accel = float(row[C["lat_accel"]])
        v.tele.long_accel = float(row[C["long_accel"]])
        v.tele.downforce = float(row[C["downforce"]])
        v.track = self.track
        v.on_track = True
        self.controls = Controls(throttle=float(row[C["throttle"]]),
                                 brake=float(row[C["brake"]]),
                                 steer=v.steer_input)
        i = int(f[k, C["index"]])
        self._last_i = i
        self.lap_time = t
        sec = self.track.sector_of(i)
        if sec != self.sector:
            if sec == self.sector + 1:
                self.sectors[self.sector] = self.rec.sectors[self.sector]
            self.sector = sec

    def sync(self, dt: float, alpha: float):
        if self.visible:
            self.car.sync(dt=dt, alpha=alpha)

    # -- the gap --------------------------------------------------------
    def delta(self, index: int, player_lap_time: float | None) -> float | None:
        """Seconds the player is behind the ghost lap at the player's position.

        The recorded lap is complete, so this has an answer everywhere on the
        lap, whether or not the ghost is still on screen.
        """
        if player_lap_time is None:
            return None
        t = self.splits[index]
        return None if not np.isfinite(t) else float(player_lap_time - t)

    # -- lifecycle ------------------------------------------------------
    def start(self):
        """The race's go signal. A replay waits for the player's first
        flying lap instead, so there is nothing to do."""

    def destroy(self):
        from .ui import destroy_tree

        destroy_tree(self.car)
