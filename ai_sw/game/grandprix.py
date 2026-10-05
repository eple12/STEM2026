"""Setting up a grand prix: who drives what, where they start.

Shared by the game (through ``fieldproc``) and ``tools/race_sim.py``, so a
race simulated headless is the race the player gets.

* Each AI driver's car follows from the difficulty level (``teams.py``):
  which solved plan, at what pace, with which engine.
* The grid is a qualifying order: every AI driver's flying-lap time is
  estimated from its plan (``teams.expected_lap``) with a little noise, and
  the player slots in by their own qualifying lap if they set one on this
  circuit at this level -- otherwise at ``config.GP_PLAYER_GRID``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, teams
from .field import Entrant, Field
from .mintime_driver import Plan, available, path_file
from .racecontrol import RaceControl
from .racecraft import RaceDriver, TrackFrame
from .rules import TrackLimits
from .surface import Surface
from .vehicle import Vehicle


_POLICY = {}
_DRIVE = {}


def drive_net():
    """The learned steering and pedals (drivenet.DriveNet) if ``config.DRIVE_AI``
    asks for them and the weights are there, else None -- the hand-built
    follower drives. Loaded once per process."""
    if config.DRIVE_AI != "rl":
        return None
    path = config.DRIVE_AI_POLICY
    if path not in _DRIVE:
        try:
            from .drivenet import DriveNet
            _DRIVE[path] = DriveNet.load(path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"drive AI: no network at {path} ({exc.__class__.__name__}); the follower drives")
            _DRIVE[path] = None
    return _DRIVE[path]


def race_policy():
    """The learned decision layer (raceai.Policy) if ``config.RACE_AI`` asks for
    it and its weights are there, else None -- the rules decide. Loaded once per
    process."""
    if config.RACE_AI != "rl":
        return None
    path = config.RACE_AI_POLICY
    if path not in _POLICY:
        try:
            from .raceai import Policy
            _POLICY[path] = Policy.load(path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"race AI: no policy at {path} ({exc.__class__.__name__}); the rules decide")
            _POLICY[path] = None
    return _POLICY[path]


def solved_grips(circuit: str) -> tuple:
    """The plan grips solved for a circuit, fastest first."""
    return tuple(g for g in teams.PLAN_GRIPS
                 if available(circuit, f"g{round(g * 100):02d}"))


def ready(circuit: str) -> bool:
    """Can a full field race here? Needs at least the fastest plan."""
    return len(solved_grips(circuit)) > 0


class PlanCache:
    def __init__(self, circuit):
        self.circuit = circuit
        self.plans = {}

    def __call__(self, tag):
        if tag not in self.plans:
            self.plans[tag] = Plan(path_file(self.circuit, tag))
        return self.plans[tag]


@dataclass
class Seat:
    team: teams.Team
    driver: teams.Driver
    player: bool
    skill: object = None
    power: float = 1.0
    quali: float = 0.0          # qualifying lap, estimated (AI) or set (player)


def seats(circuit: str, level: int, seed: int = 0,
          player_time: float | None = None,
          player_grid: int | None = None) -> list[Seat]:
    """All twenty seats with their qualifying times, in grid order. The
    player starts where *player_grid* (1 = pole) says, else by their
    qualifying lap, else at ``config.GP_PLAYER_GRID``."""
    grips = solved_grips(circuit)
    if not grips:
        raise RuntimeError(f"no solved plans for {circuit}")
    lvl = teams.DIFFICULTY[level]
    plans = PlanCache(circuit)
    rng = np.random.default_rng(seed + 7919 * level)
    out = []
    for tm, d, is_player in teams.lineup():
        s = Seat(tm, d, is_player)
        if not is_player:
            s.skill, s.power = teams.skill_for(d, tm, lvl, grips)
            t = teams.expected_lap(plans(s.skill.plan).lap_time, s.skill, s.power)
            # A qualifying lap is a good lap, not a perfect one: a spread of
            # a couple of tenths, wider for the less consistent.
            s.quali = t * (1.0 + abs(rng.normal(0.0, 0.0012 + s.skill.consistency * 0.2)))
        out.append(s)
    ai = sorted((s for s in out if not s.player), key=lambda s: s.quali)
    me = next(s for s in out if s.player)
    if player_grid is not None:
        slot = min(max(int(player_grid) - 1, 0), len(ai))
        me.quali = float("nan")
        ordered = ai[:slot] + [me] + ai[slot:]
    elif player_time is not None:
        me.quali = player_time
        ordered = sorted(ai + [me], key=lambda s: s.quali)
    else:
        slot = min(max(config.GP_PLAYER_GRID - 1, 0), len(ai))
        me.quali = float("nan")
        ordered = ai[:slot] + [me] + ai[slot:]
    return ordered


def build(track, level: int, laps: int, seed: int = 0, player: bool = True,
          player_time: float | None = None, n_cars: int = 20,
          grid: str = "quali", player_grid: int | None = None) -> Field:
    """The field on the grid, ready for lights out. With ``player`` one car
    is external (``Entrant.external``) and has no driver."""
    circuit = track.name
    grips = solved_grips(circuit)
    plans = PlanCache(circuit)
    ref = plans(f"g{round(max(grips) * 100):02d}")
    frame = TrackFrame(track, ref)
    order = seats(circuit, level, seed, player_time, player_grid)
    if not player:
        order = [s for s in order if not s.player]
        if grid == "reverse":
            order = order[::-1]
    order = order[:n_cars]
    lvl = teams.DIFFICULTY[level]
    # Race-day form and a few more mistakes than the level alone gives (see
    # config.GP_FORM_SPREAD). After the grid was set: qualifying is untouched.
    form = np.random.default_rng(seed * 31 + level + 5)
    for s in order:
        if not s.player:
            s.skill.pace *= 1.0 - abs(form.normal(0.0, config.GP_FORM_SPREAD))
            s.skill.mistakes += config.GP_EXTRA_MISTAKES
            s.skill.consistency += config.GP_LAP_VARIATION
    entrants = []
    for idx, s in enumerate(order):
        v = Vehicle()
        surf = Surface(track)
        if s.player:
            drv = None
            name, tla = teams.PLAYER_NAME, teams.PLAYER_TLA
        else:
            v.power_scale = s.power
            drv = RaceDriver(track, frame, plans(s.skill.plan), s.skill,
                             np.random.default_rng(seed * 100 + idx))
            drv.idx = idx
            drv.policy = race_policy()
            drv.follow.drive = drive_net()
            name, tla = s.driver.name, s.driver.tla
        e = Entrant(idx=idx, name=name, team=s.team.name, color=s.team.color,
                    vehicle=v, surface=surf, driver=drv,
                    limits=TrackLimits(track, config.GP_OFFTRACK_REJOIN),
                    tla=tla, external=s.player)
        e.quali = s.quali
        entrants.append(e)
    rc = RaceControl(len(entrants), lvl.strike_pen, lvl.minor_repeat_pen)
    fld = Field(track, frame, entrants, laps, rc)
    fld.grid()
    return fld


def quali_table(circuit: str, level: int, seed: int = 0) -> list[Seat]:
    """The AI's qualifying times at a level, fastest first (no player)."""
    return [s for s in seats(circuit, level, seed) if not s.player]
