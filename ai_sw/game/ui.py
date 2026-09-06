"""Shared UI kit: the F1-broadcast look, used by both the menu and the HUD.

Kept separate so the two screens cannot drift apart. The look rests on three
things the real thing does: a near-black ground with a single saturated red,
angular shapes that all lean the same way, and condensed uppercase type with
wide tracking.
"""
from __future__ import annotations

import math
from pathlib import Path

from ursina import Mesh, Vec3

from . import config
from . import palette as pal

# --- brand ---------------------------------------------------------------
RED = pal.rgb(225, 6, 0)          # F1 red
INK = pal.rgb(21, 21, 30)         # near-black ground
PANEL = pal.rgb(31, 31, 43)
PANEL_HI = pal.rgb(42, 42, 56)
WHITE = pal.rgb(255, 255, 255)
GREY = pal.rgb(150, 152, 162)
GREY_DIM = pal.rgb(96, 98, 108)
# Timing colours, straight off a timing tower: purple is the session best,
# green a personal best, yellow a sector slower than your own best, amber the
# lap in progress.
PURPLE = pal.rgb(180, 60, 220)
GREEN = pal.rgb(60, 210, 90)
AMBER = pal.rgb(245, 170, 40)
YELLOW = pal.rgb(240, 212, 48)

# "Team" colours, the vertical bar beside a driver's name on the tower. The
# player is amber -- the same as their dot on the map -- and the AI is the red
# of the car it drives.
TEAM_YOU = AMBER
TEAM_AI = RED

# Shift lights on the telemetry widget, in the order a steering wheel's LEDs
# come on: green, then red, then blue for the shift.
LED_GREEN = pal.rgb(70, 220, 90)
LED_RED = pal.rgb(235, 40, 30)
LED_BLUE = pal.rgb(80, 130, 255)
LED_OFF = pal.rgb(44, 44, 56)
# The start gantry.
LAMP_OFF = pal.rgb(58, 16, 16)
LAMP_ON = pal.rgb(255, 28, 18)
KEYCAP = pal.rgb(64, 66, 82)

GEARS = 6


def gear_of(speed_frac: float) -> tuple[int, float]:
    """(gear, revs within it 0..1) for a speed as a fraction of the maximum.

    One fake gearbox for the whole program: the engine note and the gear on
    the telemetry widget have to agree, or the number changes without the
    sound and the eye notices.
    """
    frac = min(1.0, max(0.0, speed_frac))
    g = min(GEARS - 1, int(frac * GEARS))
    return g + 1, frac * GEARS - g

# Panels over a bright track need to stay readable without hiding it. 190 was
# not enough: over the start/finish chequer the readouts washed out almost to
# nothing. Legibility over the worst surface on the circuit sets this, not how
# it looks over asphalt.
GLASS = pal.rgb(18, 18, 26, 226)
GLASS_HI = pal.rgb(48, 48, 62, 226)

SLANT_DEG = 15.0                  # every angled edge leans by this much


# Folder name -> (list label, full circuit name, country code).
#
# The folder names are identifiers, not captions: "MOSCOWRACEWAY", "SAOPAULO"
# and "IMS" all read as filenames on screen. The label is what a broadcast puts
# on the timing tower, the full name is the circuit's actual name.
CIRCUITS = {
    "Austin": ("AUSTIN", "Circuit of the Americas", "USA"),
    "BrandsHatch": ("BRANDS HATCH", "Brands Hatch Circuit", "GBR"),
    "Budapest": ("BUDAPEST", "Hungaroring", "HUN"),
    "Catalunya": ("BARCELONA", "Circuit de Barcelona-Catalunya", "ESP"),
    "Hockenheim": ("HOCKENHEIM", "Hockenheimring", "GER"),
    "IMS": ("INDIANAPOLIS", "Indianapolis Motor Speedway", "USA"),
    "Melbourne": ("MELBOURNE", "Albert Park Circuit", "AUS"),
    "Mexico City": ("MEXICO CITY", "Autodromo Hermanos Rodriguez", "MEX"),
    "Montreal": ("MONTREAL", "Circuit Gilles Villeneuve", "CAN"),
    "Monza": ("MONZA", "Autodromo Nazionale Monza", "ITA"),
    "MoscowRaceway": ("MOSCOW", "Moscow Raceway", "RUS"),
    "Nuerburgring": ("NURBURGRING", "Nurburgring GP-Strecke", "GER"),
    "Oschersleben": ("OSCHERSLEBEN", "Motorsport Arena Oschersleben", "GER"),
    "Sakhir": ("SAKHIR", "Bahrain International Circuit", "BRN"),
    "SaoPaulo": ("INTERLAGOS", "Autodromo Jose Carlos Pace", "BRA"),
    "Sepang": ("SEPANG", "Sepang International Circuit", "MYS"),
    "Shanghai": ("SHANGHAI", "Shanghai International Circuit", "CHN"),
    "Silverstone": ("SILVERSTONE", "Silverstone Circuit", "GBR"),
    "Sochi": ("SOCHI", "Sochi Autodrom", "RUS"),
    "Spa": ("SPA", "Circuit de Spa-Francorchamps", "BEL"),
    "Spielberg": ("SPIELBERG", "Red Bull Ring", "AUT"),
    "YasMarina": ("YAS MARINA", "Yas Marina Circuit", "UAE"),
    "Zandvoort": ("ZANDVOORT", "Circuit Zandvoort", "NED"),
}


def caption(key: str) -> tuple[str, str, str]:
    """(label, full name, country code) for a circuit folder, known or not."""
    return CIRCUITS.get(key, (key.upper(), key, "---"))


def pick_font() -> str | None:
    """Name of the most F1-looking font available, or None for Ursina's default.

    Ursina resolves font names only inside its own asset folders and raises on
    an absolute path, so this also points ``fonts_folder`` at whichever
    directory won. The built-in default still resolves, from the internal
    fonts folder, and so does the fallback if nothing here exists.
    """
    from ursina import application

    # A bundled font wins, so dropping Titillium Web (SIL OFL, the face the
    # real F1 typeface derives from) into assets/fonts upgrades the look
    # without touching this file. Nothing is bundled by default -- the system
    # fonts below are Microsoft's and are not ours to redistribute.
    places = [
        (config.ASSET_DIR / "fonts",
         ("Formula1-Bold.ttf", "TitilliumWeb-Bold.ttf", "TitilliumWeb-SemiBold.ttf")),
        (Path("C:/Windows/Fonts"), ("bahnschrift.ttf", "seguisb.ttf", "tahoma.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu"), ("DejaVuSans-Bold.ttf",)),
    ]
    for folder, names in places:
        for name in names:
            if (folder / name).is_file():
                application.fonts_folder = folder
                return name
    return None


def spaced(s: str, gap: str = " ") -> str:
    """Letter-spacing, the only way to get it out of Ursina's Text."""
    return gap.join(s.upper())


def skew_quad(w: float, h: float, angle: float = SLANT_DEG) -> Mesh:
    """A parallelogram leaning right by *angle* degrees.

    The lean is derived from the height rather than being a fixed offset, so
    every shape leans by the same visual angle. A constant offset looks right
    on a 0.05-tall row and turns a 0.7-tall panel into a wedge.

    Wound counter-clockwise. Listing the corners in reading order (top-left,
    top-right, bottom-right) is clockwise on screen, which is the *back* face,
    and every shape built that way is silently culled -- the same trap that
    made the car's ground shadow invisible.
    """
    s = math.tan(math.radians(angle)) * h / 2
    v = [Vec3(-w / 2 + s, h / 2, 0), Vec3(w / 2 + s, h / 2, 0),
         Vec3(w / 2 - s, -h / 2, 0), Vec3(-w / 2 - s, -h / 2, 0)]
    return Mesh(vertices=v, triangles=[(0, 2, 1), (0, 3, 2)], mode="triangle")


def lap_time(t: float | None) -> str:
    if t is None:
        return "--:--.---"
    m, s = divmod(t, 60)
    return f"{int(m):d}:{s:06.3f}"


def delta_time(t: float | None) -> str:
    """Signed gap, the way a timing tower writes it."""
    if t is None:
        return ""
    return f"{'+' if t >= 0 else '-'}{abs(t):.3f}"
