"""The grid: ten teams of two, and how hard they race you.

Colours follow the shape of a modern F1 grid; the team and driver names are
made up on purpose (real teams' names and liveries are trademarks, real
drivers' names are real people) -- edit freely.

A driver is a ``rating`` from 0 (back of the grid) to 1 (the best on it), and
a team a ``power`` (its engine, against the 1.0 every plan was solved for).
What those turn into on track depends on the DIFFICULTY level:

* which solved plan the driver drives -- a plan solved at less grip brakes
  earlier and corners slower everywhere, the way a driver who commits less
  does, and it is still a physically consistent line, not a slowed-down copy
  of the fast one;
* a pace multiplier under that, for steps finer than the plans;
* how consistent they are lap to lap, and how often they make a mistake;
* and how strict race control is (``racecontrol.py``).

Levels are calibrated in lap time (``tools/race_sim.py --calibrate``), so
"level 3" means a measured number of seconds off the AI's best, not a guess.

The player takes one seat: the second Rosso Corsa car, whose red is the
player's own car's.
"""
from __future__ import annotations

from dataclasses import dataclass

from .racecraft import Skill


@dataclass(frozen=True)
class Driver:
    name: str
    rating: float          # 0..1

    @property
    def tla(self) -> str:
        """Three-letter abbreviation, as on a timing tower: the surname's
        first three letters."""
        return self.name.split()[-1][:3].upper()


@dataclass(frozen=True)
class Team:
    name: str
    color: tuple           # (r, g, b) 0..1 -- the car's paint, the tower bar
    power: float           # engine, x the reference car
    drivers: tuple
    #: The livery's secondary (engine cover, nose tip) and accent (the line
    #: down the flank, the helmet), (r, g, b) 0..1.
    secondary: tuple = (0.07, 0.07, 0.08)
    accent: tuple = (0.95, 0.95, 0.96)


def _rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


TEAMS = (
    Team("Papaya Racing", _rgb("FF8000"), 1.000,
         (Driver("A. Kerr", 1.00), Driver("J. Moreau", 0.98)),
         _rgb("1A1A1E"), _rgb("47C7FC")),
    Team("Rosso Corsa", _rgb("DC0000"), 0.997,
         (Driver("T. Lindqvist", 0.96), Driver("R. Okafor", 0.92)),
         _rgb("101012"), _rgb("FFF200")),
    Team("Argento", _rgb("27F4D2"), 0.998,
         (Driver("D. Santos", 0.95), Driver("S. Han", 0.86)),
         _rgb("101214"), _rgb("C8CCCE")),
    Team("Notte Blu", _rgb("1E41FF"), 0.996,
         (Driver("M. Kowalczyk", 0.99), Driver("H. Tanaka", 0.80)),
         _rgb("0B1640"), _rgb("E10600")),
    Team("Verde Scuro", _rgb("229971"), 0.990,
         (Driver("S. Novak", 0.85), Driver("E. Brandt", 0.66)),
         _rgb("0E2A1F"), _rgb("CEDC00")),
    Team("Bleu Alpin", _rgb("0093CC"), 0.986,
         (Driver("P. Ruiz", 0.72), Driver("K. Larsen", 0.62)),
         _rgb("FF87BC"), _rgb("FFFFFF")),
    Team("Steel Stripe", _rgb("B6BABD"), 0.988,
         (Driver("I. Varga", 0.74), Driver("L. Dubois", 0.70)),
         _rgb("1A1A1A"), _rgb("E6002B")),
    Team("Junior Azure", _rgb("6692FF"), 0.993,
         (Driver("N. Ferreira", 0.78), Driver("C. Moretti", 0.68)),
         _rgb("FFFFFF"), _rgb("1A1A6E")),
    Team("Grove Blue", _rgb("64C4FF"), 0.992,
         (Driver("B. Osei", 0.82), Driver("J. Yoon", 0.84)),
         _rgb("041E42"), _rgb("FFFFFF")),
    Team("Volt Green", _rgb("52E252"), 0.984,
         (Driver("F. Lund", 0.76), Driver("O. Walsh", 0.64)),
         _rgb("101010"), _rgb("FFFFFF")),
)

#: The seat the player takes: (team index, driver index).
PLAYER_SEAT = (1, 1)
PLAYER_NAME = "PLAYER"
PLAYER_TLA = "YOU"

#: Plans solved per circuit (``mintime.py --grip 0.XX --tag gXX``), fastest
#: first. A driver's grip is mapped onto the nearest one at or under it.
PLAN_GRIPS = (0.97, 0.94, 0.90, 0.86, 0.82, 0.78, 0.74)


@dataclass(frozen=True)
class Level:
    name: str
    top_grip: float        # grip the best driver on the grid gets
    spread: float          # grip from best (rating 1) to worst (rating 0)
    power: float           # engine scale on top of the team's
    consistency: float     # per-lap pace std-dev
    mistakes: float        # chance per lap of a late-braking error
    aggression: float      # added to every driver's (0..1)
    #: Race control: seconds per track-limits strike past the third (0:
    #: warnings only), and whether a second light collision is penalised.
    strike_pen: float = 5.0
    minor_repeat_pen: bool = True
    blurb: str = ""


#: 1 = an easy afternoon, 6 = the AI at full stretch. Calibrated by
#: ``tools/race_sim.py --calibrate``; the lap times in the blurbs are Monza's
#: best AI driver (the solved optimum there is 87.4 s).
DIFFICULTY = {
    1: Level("Novice", 0.70, 0.04, 0.85, 0.012, 0.30, -0.4, 0.0, False,
             "A gentle field that brakes early and slips up often. "
             "Track limits are warnings only."),
    2: Level("Rookie", 0.78, 0.04, 0.88, 0.010, 0.25, -0.3, 1.0, False,
             "Steady but beatable. Track limits cost 1 s after three strikes."),
    3: Level("Amateur", 0.82, 0.05, 0.92, 0.008, 0.15, -0.2, 1.0, True,
             "Quick in the corners, still makes mistakes."),
    4: Level("Club", 0.86, 0.06, 0.96, 0.006, 0.08, -0.1, 2.0, True,
             "A proper race. Track limits strikes cost 2 s after three."),
    5: Level("Pro", 0.94, 0.07, 0.99, 0.004, 0.03, 0.0, 2.0, True,
             "Close to the limit, rarely wrong."),
    6: Level("Legend", 0.97, 0.07, 1.00, 0.003, 0.0, 0.1, 2.0, True,
             "The solved optimum, driven flat out."),
}
DEFAULT_LEVEL = 2


def plan_tag(grip: float, available=PLAN_GRIPS) -> tuple[str, float]:
    """(plan tag, pace) for a wanted grip: the nearest solved plan at or
    ABOVE it, driven at a pace that takes off the remainder.

    Always from above: a plan driven a little slower than it was solved for
    has grip to spare and tracks cleanly, while a plan driven faster than it
    was solved for has none. Corner speed goes as sqrt(grip), hence the pace.
    """
    over = [g for g in available if g >= grip - 1e-9]
    g = min(over) if over else max(available)
    pace = min((grip / g) ** 0.5, 1.0)
    return f"g{round(g * 100):02d}", pace


def skill_for(driver: Driver, team: Team, level: Level,
              available=PLAN_GRIPS) -> tuple[Skill, float]:
    """(Skill, engine power scale) for one driver at one difficulty."""
    grip = level.top_grip - level.spread * (1.0 - driver.rating)
    tag, pace = plan_tag(grip, available)
    aggression = min(max(0.35 + 0.4 * driver.rating + level.aggression, 0.0), 1.0)
    return (Skill(plan=tag, pace=pace,
                  consistency=level.consistency * (1.5 - 0.5 * driver.rating),
                  mistakes=level.mistakes * (1.4 - 0.6 * driver.rating),
                  reaction=0.32 - 0.14 * driver.rating,
                  aggression=aggression),
            team.power * level.power)


#: Lap time lost per unit of engine power missing, as a share of the lap
#: (fitted to the --calibrate table: Monza, the most power-hungry circuit).
POWER_COST = 0.25


def expected_lap(plan_time: float, skill: Skill, power: float) -> float:
    """A driver's flying lap, estimated without driving it: the plan's own
    time at the driver's pace, plus what the missing power costs."""
    return plan_time / max(skill.pace, 1e-3) * (1.0 + POWER_COST * (1.0 - power))


def lineup():
    """[(team, driver, is_player)] for the twenty seats, in team order."""
    out = []
    for ti, tm in enumerate(TEAMS):
        for di, d in enumerate(tm.drivers):
            out.append((tm, d, (ti, di) == PLAYER_SEAT))
    return out
