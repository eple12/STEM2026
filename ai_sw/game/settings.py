"""Session settings: what the rules and aids are for the race about to start.

Set on the settings screen (main menu or circuit list, key O), never during a
race. The race's AI and stewards run in a worker process (fieldproc.py), which
gets a copy at start (``to_dict`` / ``apply``); everything that reads a setting
reads ``current`` at the moment it needs it, so the module-level object has to
be the one that was applied.

The choices are kept between runs in ``~/.formula-ai/settings.json``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PATH = Path.home() / ".formula-ai" / "settings.json"


@dataclass
class Settings:
    #: The assist steers the car along the racing line; the player picks the
    #: lane (A / D) and works the pedals.
    auto_steer: bool = False
    #: The assist works the pedals too, at the pace of the racing line and
    #: behind any car ahead; with auto_steer on, only the lane is left to the
    #: player.
    auto_pedals: bool = False
    #: Yellow flags: a stricken car brings out a yellow, AI cars slow down and
    #: stewards watch for overtaking and speeding in the zone.
    yellow_flags: bool = True
    #: Drag reduction: within a second or so of the car ahead on a straight.
    drs: bool = True
    #: The tow behind another car.
    slipstream: bool = True
    #: Time penalties, by kind. Off: no penalty and no warning of that kind.
    pen_track_limits: bool = True     # cutting a corner, running wide
    pen_collision: bool = True        # causing a collision
    pen_yellow: bool = True           # overtaking or speeding under yellow
    #: Manual gearbox for the player's car (paddles / E and Z); off = automatic.
    manual_gearbox: bool = False
    #: Steering wheel sensitivity: 1.0 = lock to lock as the wheel is set in
    #: Pit House; higher reaches full steering with less turn of the wheel.
    steer_sens: float = 1.0


#: (field, title, what it does) in the order of the settings screen.
ROWS = (
    ("manual_gearbox", "MANUAL GEARBOX", "shift yourself: wheel paddles or E / Z; the engine cuts out at the rev limiter"),
    ("steer_sens", "STEERING SENSITIVITY", "wheel only: more = less turn of the wheel for the same steering (0.5x - 4.0x)"),
    ("auto_steer", "AUTO STEERING", "the car steers along the line; steer yourself (A / D) any time to take over, Q back to the line"),
    ("auto_pedals", "AUTO PEDALS", "throttle and brake are automatic until you press W / S yourself"),
    ("yellow_flags", "YELLOW FLAGS", "a stricken car brings out a yellow and everyone slows down"),
    ("drs", "DRS", "drag reduction within a second of the car ahead on a straight"),
    ("slipstream", "SLIPSTREAM", "the tow behind another car"),
    ("pen_track_limits", "PENALTY · TRACK LIMITS", "time penalties for cutting and running wide"),
    ("pen_collision", "PENALTY · COLLISIONS", "time penalties for causing a collision"),
    ("pen_yellow", "PENALTY · YELLOW FLAGS", "time penalties for overtaking or speeding under yellow"),
)

current = Settings()


def to_dict() -> dict:
    return asdict(current)


def apply(d: dict | None) -> None:
    """Take the values in *d* (unknown keys ignored) as the current settings."""
    for f in fields(Settings):
        if d and f.name in d:
            v = d[f.name]
            setattr(current, f.name,
                    min(4.0, max(0.5, float(v))) if f.type in (float, "float") else bool(v))


def load() -> None:
    try:
        apply(json.loads(PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass


def save() -> None:
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(to_dict(), indent=1), encoding="utf-8")
    except OSError:
        pass
