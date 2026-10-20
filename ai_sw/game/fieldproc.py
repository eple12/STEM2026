"""The grand-prix field in a worker process.

Nineteen AI drivers and their cars cost several milliseconds a physics tick
in Python -- more than a 60 fps frame can spare next to the renderer. So the
field (``field.py``) runs in a process of its own, on its own core, and the
game talks to it once a frame:

* **game -> worker** (``FieldClient.tick``): "advance to session time T",
  with the player's car as it is at T (its physics stays in the game, at the
  game's rate, so the player's controls never wait on anything), plus the
  contact kicks the game has applied so far and any events (lights out, the
  player reset onto the track).
* **worker -> game**: a snapshot -- every car's pose and motion at the
  field's time, the timing, the stewards' new messages, and what contacts
  have done to the player's car since the last one (``kick``).

The game never blocks on the worker. It draws the AI cars from the newest
snapshot it has, carried forward by each car's own velocity to the moment
being drawn, which is at most a frame or two -- a millimetre-scale guess.

Inside the worker, the player's car is placed at every field tick by
interpolating between the two states the game last sent; contact pushes it
like any other car, and the change is sent back for the game to apply to
the real one. Until the game confirms it has (cumulative ``kick_applied``),
the worker keeps adding the difference to the states it is sent, so one
contact is not felt twice.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import time
import traceback

import numpy as np

#: Snapshot columns per car.
COLS = ("x", "z", "yaw", "vx", "vz", "yaw_rate", "steer", "lat_accel",
        "long_accel", "downforce", "braking", "ghost", "drs", "racing",
        "progress", "laps", "finish_t", "best", "last", "lap_start",
        "penalty", "on_track", "hits", "strikes", "pos", "interval", "gap",
        "recover", "s", "total", "stricken")
C = {name: k for k, name in enumerate(COLS)}

DT = 1.0 / 60.0


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------
def _snapshot(fld, kick_total, messages_from: int):
    rows = np.zeros((len(fld.cars), len(COLS)), np.float64)
    for e in fld.cars:
        v = e.vehicle
        r = rows[e.idx]
        r[C["x"]], r[C["z"]] = v.pos
        r[C["yaw"]] = v.yaw
        r[C["vx"]], r[C["vz"]] = v.vel
        r[C["yaw_rate"]] = v.yaw_rate
        r[C["steer"]] = v.steer_angle
        r[C["lat_accel"]] = v.tele.lat_accel
        r[C["long_accel"]] = v.tele.long_accel
        r[C["downforce"]] = v.tele.downforce
        r[C["braking"]] = float(e.braking)
        r[C["ghost"]] = float(e.ghost)
        r[C["drs"]] = float(e.drs)
        view = fld.views[e.idx] if fld.views else None
        r[C["racing"]] = float(view.racing) if view is not None else 0.0
        p = e.limits.progress
        r[C["progress"]] = p if p is not None else -1e9
        r[C["laps"]] = e.laps
        r[C["finish_t"]] = e.finish_t if e.finish_t is not None else np.nan
        r[C["best"]] = e.best if e.best is not None else np.nan
        r[C["last"]] = e.lap_times[-1] if e.lap_times else np.nan
        r[C["lap_start"]] = e.lap_start
        r[C["penalty"]] = fld.rc.penalty(e.idx)
        r[C["on_track"]] = float(v.on_track)
        r[C["hits"]] = e.hits
        r[C["strikes"]] = fld.rc.cars[e.idx].strikes
        r[C["stricken"]] = float(fld._stricken(e)) if fld.started else 0.0
    # The running order and the gaps on the tower: interval to the car
    # ahead and gap to the leader, both measured at the same point of track
    # (Field.gap); at the flag, the classification by total time.
    order = fld.order()
    lead = order[0]
    for p, e in enumerate(order):
        r = rows[e.idx]
        r[C["pos"]] = p + 1
        tot = fld.total_time(e)
        r[C["total"]] = tot if tot is not None else np.nan
        if p == 0:
            r[C["interval"]] = r[C["gap"]] = 0.0
            continue
        if e.finish_t is not None and order[p - 1].finish_t is not None:
            r[C["interval"]] = tot - fld.total_time(order[p - 1])
            r[C["gap"]] = tot - fld.total_time(lead) if lead.finish_t is not None else np.nan
            continue
        g = fld.gap(e, order[p - 1])
        r[C["interval"]] = g if g is not None else np.nan
        g = fld.gap(e, lead)
        r[C["gap"]] = g if g is not None else np.nan
    for e in fld.cars:
        rows[e.idx, C["recover"]] = float(e.driver is not None
                                          and e.driver.mode == "recover")
        rows[e.idx, C["s"]] = fld.views[e.idx].s if fld.views else 0.0
    msgs = [(m.t, m.car, m.text, m.kind, m.other)
            for m in fld.rc.messages[messages_from:]]
    return {"t": fld.t, "rows": rows, "kick": list(kick_total), "msgs": msgs,
            "leader_done": fld.leader_done, "started": fld.started,
            "yellow": list(fld.rc.yellow), "overload": fld.frame.think_every > 2}


def _lerp_state(a, b, w):
    """Player state between two sent states (shortest way round in yaw)."""
    dy = (b["yaw"] - a["yaw"] + math.pi) % (2 * math.pi) - math.pi
    return dict(
        pos=(a["pos"][0] + (b["pos"][0] - a["pos"][0]) * w,
             a["pos"][1] + (b["pos"][1] - a["pos"][1]) * w),
        yaw=a["yaw"] + dy * w,
        vel=(a["vel"][0] + (b["vel"][0] - a["vel"][0]) * w,
             a["vel"][1] + (b["vel"][1] - a["vel"][1]) * w),
        yaw_rate=a["yaw_rate"] + (b["yaw_rate"] - a["yaw_rate"]) * w,
        steer=b["steer"], on_track=b["on_track"], braking=b["braking"])


def _worker(conn, setup: dict):
    import gc

    try:
        from . import grandprix, settings
        from .trackdata import load_track

        settings.apply(setup.get("settings"))

        track = load_track(setup["circuit"])
        # Spectating: twenty AI drivers, nobody driven from the game.
        watch = bool(setup.get("spectate"))
        fld = grandprix.build(track, setup["level"], setup["laps"],
                              seed=setup.get("seed", 0), player=not watch,
                              player_time=setup.get("player_time"),
                              player_grid=setup.get("player_grid"))
        me = next((e.idx for e in fld.cars if e.external), None)
        info = [dict(idx=e.idx, name=e.name, tla=e.tla, team=e.team,
                     color=tuple(e.color), grid=e.grid_slot, quali=e.quali,
                     external=e.external,
                     pos=tuple(e.vehicle.pos), yaw=e.vehicle.yaw)
                for e in fld.cars]
        conn.send(("ready", {"cars": info, "player": me}))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    # Everything built so far lives for the whole race: keep the collector
    # from walking it every few thousand allocations.
    gc.collect()
    gc.freeze()

    prev = None                    # (time, player state) last sent
    cur = None
    kick_applied = np.zeros(5)
    sent_msgs = 0
    wall_ratio = 0.0               # wall seconds per simulated second, smoothed
    last_snap = None
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        # Take only the newest tick if several are queued: catch up, do not
        # replay the backlog frame by frame.
        latest = None
        events = []
        while True:
            kind = msg[0]
            if kind == "quit":
                return
            if kind == "tick":
                latest = msg
                events += msg[1].get("events", ())
            if not conn.poll():
                break
            msg = conn.recv()
        if latest is None:
            continue
        body = latest[1]
        target = body["t"]
        state = body["player"]
        kick_applied = np.asarray(body.get("kick_applied", kick_applied), float)
        for ev in events:
            if ev == "go" and not fld.started:
                fld.lights_out()
                prev = None
            elif ev == "reset" and me is not None:
                fld.reset_external(me)
                prev = None
        cur = (target, state)
        if prev is None:
            prev = cur
        e_me = fld.cars[me] if me is not None else None
        t0 = time.perf_counter()
        sim0 = fld.t
        if e_me is None:
            # Nobody external: just run the field up to the game's clock.
            if not fld.started:
                fld.step(DT)
            while fld.started and fld.t + 0.5 * DT < target:
                fld.step(DT)
        elif not fld.started:
            # On the grid: the player's car is where the game says; the
            # others sit in their slots.
            st = state
            pend = np.asarray(e_me.kick) - kick_applied
            fld.set_external(me, st["pos"], st["yaw"], st["vel"], st["yaw_rate"],
                             st["steer"], st["on_track"], st["braking"])
            fld.step(DT)
        else:
            while fld.t + 0.5 * DT < target:
                ta, sa = prev
                tb, sb = cur
                w = 1.0 if tb <= ta else min(max((fld.t + DT - ta) / (tb - ta), 0.0), 1.0)
                st = _lerp_state(sa, sb, w)
                # Contacts the game has not applied yet still happened.
                pend = np.asarray(e_me.kick) - kick_applied
                pos = (st["pos"][0] + pend[0], st["pos"][1] + pend[1])
                vel = (st["vel"][0] + pend[2], st["vel"][1] + pend[3])
                fld.set_external(me, pos, st["yaw"], vel, st["yaw_rate"] + pend[4],
                                 st["steer"], st["on_track"], st["braking"])
                fld.step(DT)
        prev = cur
        sim = fld.t - sim0
        if sim > 0.0:
            ratio = (time.perf_counter() - t0) / sim
            wall_ratio = ratio if wall_ratio == 0.0 else 0.9 * wall_ratio + 0.1 * ratio
            # Running at more than ~80% of real time: thin out the thinking
            # until there is headroom again.
            if wall_ratio > 0.8:
                fld.overload(True)
            elif wall_ratio < 0.55:
                fld.overload(False)
        snap = _snapshot(fld, e_me.kick if e_me is not None else [0.0] * 5,
                         sent_msgs)
        sent_msgs = len(fld.rc.messages)
        snap["wall_ratio"] = wall_ratio
        last_snap = snap
        try:
            conn.send(("snap", last_snap))
        except (EOFError, OSError, BrokenPipeError):
            return


def _entry(conn, setup):
    """Process entry point: one BLAS thread, like the game itself."""
    import os
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _worker(conn, setup)


# ---------------------------------------------------------------------------
# game side
# ---------------------------------------------------------------------------
class FieldClient:
    """The game's end of the worker. Never blocks once the race is on."""

    def __init__(self, circuit: str, level: int, laps: int, seed: int = 0,
                 player_time: float | None = None, spectate: bool = False,
                 player_grid: int | None = None):
        from . import settings
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe(duplex=True)
        setup = dict(circuit=circuit, level=level, laps=laps, seed=seed,
                     player_time=player_time, spectate=spectate,
                     player_grid=player_grid, settings=settings.to_dict())
        self.proc = ctx.Process(target=_entry, args=(child, setup), daemon=True,
                                name="aisw-field")
        self.proc.start()
        child.close()
        self.info = None
        self.player = -1
        self.snap = None
        self.kick_applied = np.zeros(5)
        self._events: list = []
        self.error: str | None = None
        self.msgs: list = []           # every steward message so far

    def wait_ready(self, timeout: float = 60.0, pump=None) -> bool:
        """Block (pumping the loading card) until the worker has built the
        field. False on failure, with ``error`` set."""
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            if self.conn.poll(0.05):
                kind, body = self.conn.recv()
                if kind == "ready":
                    self.info = body["cars"]
                    self.player = body["player"]
                    return True
                self.error = body
                return False
            if not self.proc.is_alive():
                self.error = "field worker exited"
                return False
            if pump is not None:
                pump()
        self.error = "field worker timed out"
        return False

    def event(self, name: str):
        self._events.append(name)

    def tick(self, t: float, vehicle, braking: bool):
        """Ask for the field at session time *t*, with the player's car as
        it is now."""
        st = dict(pos=(float(vehicle.pos[0]), float(vehicle.pos[1])),
                  yaw=float(vehicle.yaw),
                  vel=(float(vehicle.vel[0]), float(vehicle.vel[1])),
                  yaw_rate=float(vehicle.yaw_rate),
                  steer=float(vehicle.steer_angle),
                  on_track=bool(vehicle.on_track), braking=bool(braking))
        body = {"t": float(t), "player": st,
                "kick_applied": self.kick_applied.tolist(),
                "events": self._events}
        self._events = []
        try:
            self.conn.send(("tick", body))
        except (EOFError, OSError, BrokenPipeError):
            self.error = self.error or "field worker gone"

    def poll(self):
        """The newest snapshot (or None if nothing new), and the kick to
        apply to the player's car (dx, dz, dvx, dvz, dyaw_rate)."""
        new = None
        try:
            while self.conn.poll():
                kind, body = self.conn.recv()
                if kind == "snap":
                    new = body
                    self.msgs.extend(body["msgs"])
        except (EOFError, OSError):
            self.error = self.error or "field worker gone"
            return None, None
        if new is None:
            return None, None
        self.snap = new
        total = np.asarray(new["kick"], float)
        kick = total - self.kick_applied
        self.kick_applied = total
        return new, kick

    def close(self):
        try:
            self.conn.send(("quit",))
        except Exception:
            pass
        try:
            self.proc.join(timeout=1.0)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()
