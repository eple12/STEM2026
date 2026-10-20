"""Steering wheel + pedals (MOZA R3 and other DirectInput wheels) via winmm.

No third-party package: the Windows multimedia joystick API (``joyGetPosEx``)
is read through ctypes. PitHouse-configured wheels show up there as an ordinary
joystick with up to 6 axes (X Y Z R U V), a POV hat and 32 buttons.

Mapping lives in ``ai_sw/wheel.json`` (written with defaults on first use; edit
it or run ``python -m ai_sw.tools.wheel_probe`` to see which axis is which):

    {"steer": "X", "throttle": "Z", "brake": "R",
     "invert": {"throttle": false, "brake": false, "steer": false},
     "deadzone": 0.02, "range": 1.0, "buttons": {"handbrake": 1, ...}}

Pedals are calibrated automatically: the lowest and highest value each pedal
has shown since start-up are its released / pressed ends (``rest`` in the json
pins the released end if the pedal starts pressed).
"""
from __future__ import annotations

import ctypes
import json
import sys
from ctypes import wintypes
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "wheel.json"
AXES = ("X", "Y", "Z", "R", "U", "V")
DEFAULT = {
    "device": 0,
    "steer": "X", "throttle": "Z", "brake": "R",
    "invert": {"steer": False, "throttle": False, "brake": False},
    "deadzone": 0.02,
    "steer_range": 1.0,          # fraction of full lock that gives full steer
    "buttons": {"handbrake": 1},  # game action -> button index
    "pedal_rest": {},
    "leds": False,                # drive the wheel RPM LEDs (close Pit House first)             # e.g. {"throttle": 0.0} to pin the released end
}

JOY_RETURNALLCAPS = 0xFF  # JOY_RETURNALL
_JOY_ALL = 0xFF


class JOYINFOEX(ctypes.Structure):
    _fields_ = [(n, wintypes.DWORD) for n in (
        "dwSize", "dwFlags", "dwXpos", "dwYpos", "dwZpos", "dwRpos", "dwUpos",
        "dwVpos", "dwButtons", "dwButtonNumber", "dwPOV", "dwReserved1",
        "dwReserved2")]


_ATTR = {"X": "dwXpos", "Y": "dwYpos", "Z": "dwZpos",
         "R": "dwRpos", "U": "dwUpos", "V": "dwVpos"}


def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT))
    try:
        user = json.loads(CONFIG_PATH.read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    except (OSError, ValueError):
        try:
            CONFIG_PATH.write_text(json.dumps(DEFAULT, indent=2))
        except OSError:
            pass
    return cfg


class Wheel:
    """Poll with ``read()`` once per frame; ``connected`` says whether a
    device answered."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.connected = False
        self.raw: dict[str, float] = {}      # axes 0..1 as the OS reports them
        self.buttons = 0
        self.pov = -1
        self._lo: dict[str, float] = {}
        self._hi: dict[str, float] = {}
        self._winmm = ctypes.windll.winmm if sys.platform == "win32" else None
        self._dev = int(self.cfg["device"])
        self._info = JOYINFOEX()
        self._info.dwSize = ctypes.sizeof(JOYINFOEX)
        self._info.dwFlags = _JOY_ALL
        self._next_scan = 0

    def _poll(self, dev: int) -> bool:
        return self._winmm.joyGetPosEx(dev, ctypes.byref(self._info)) == 0

    def read(self) -> bool:
        if self._winmm is None:
            return False
        if not self._poll(self._dev):
            self.connected = False
            # hot-plug: look through the other slots now and then
            self._next_scan -= 1
            if self._next_scan <= 0:
                self._next_scan = 120
                for d in range(16):
                    if self._poll(d):
                        self._dev = d
                        break
                else:
                    return False
            else:
                return False
        self.connected = True
        i = self._info
        self.raw = {a: getattr(i, _ATTR[a]) / 65535.0 for a in AXES}
        self.buttons = i.dwButtons
        self.pov = -1 if i.dwPOV == 0xFFFF else i.dwPOV // 100
        return True

    # -- processed values -------------------------------------------------
    def _pedal(self, action: str) -> float:
        axis = self.cfg[action]
        v = self.raw.get(axis, 0.0)
        if self.cfg["invert"].get(action):
            v = 1.0 - v
        rest = self.cfg["pedal_rest"].get(action)
        lo = self._lo[action] = min(self._lo.get(action, v), v)
        hi = self._hi[action] = max(self._hi.get(action, v), v)
        if rest is not None:
            lo = rest
        span = hi - lo
        if span < 0.25:        # not moved yet: do not amplify noise
            return 0.0
        x = (v - lo) / span
        return 0.0 if x < self.cfg["deadzone"] else min(1.0, x)

    @property
    def steer(self) -> float:
        v = (self.raw.get(self.cfg["steer"], 0.5) - 0.5) * 2.0
        if self.cfg["invert"].get("steer"):
            v = -v
        v /= max(self.cfg["steer_range"], 1e-3)
        dz = self.cfg["deadzone"]
        if abs(v) < dz:
            return 0.0
        return max(-1.0, min(1.0, v))

    @property
    def throttle(self) -> float:
        return self._pedal("throttle")

    @property
    def brake(self) -> float:
        return self._pedal("brake")

    def pressed(self, action: str) -> bool:
        idx = self.cfg["buttons"].get(action)
        return idx is not None and bool(self.buttons >> int(idx) & 1)


#: wheel button index -> the key it presses (menus, camera, pause...).
KEY_BUTTONS = {0: "enter", 2: "x", 3: "tab", 12: "gear down", 13: "gear up",
               4: "up arrow", 5: "right arrow", 6: "down arrow", 7: "left arrow",
               24: "escape"}

#: Hold CAM (21) and press another button for the keys the wheel has no button
#: of its own for. A CAM tap with no second button is the camera key, "c".
LAYER_BUTTON = 21
LAYER_KEYS = {0: "r",        # reset car
              1: "m",        # mute
              2: "h",        # hide HUD
              3: "t",        # driver aids on/off
              4: "o",        # settings (menus)
              5: "g",        # watch next car
              6: "f",        # free camera
              7: "p",        # hide pause card
              12: "q",       # back to the racing line (auto steering)
              13: "f9"}      # recorder

_shared: Wheel | None = None


def shared() -> Wheel:
    global _shared
    if _shared is None:
        _shared = Wheel()
    return _shared


def poll_keys(menu: bool = False) -> list[str]:
    """Once per frame: read the wheel and return the keys its buttons just pressed."""
    w = shared()
    prev = getattr(w, "_prev_buttons", 0)
    if not w.read():
        w._prev_buttons = 0
        return []
    w._prev_buttons = w.buttons
    new = w.buttons & ~prev
    layer = bool(w.buttons >> LAYER_BUTTON & 1)
    keys = []
    if layer:
        keys += [k for i, k in LAYER_KEYS.items() if new >> i & 1]
        if new & ~(1 << LAYER_BUTTON):
            w._layer_used = True
        if new >> LAYER_BUTTON & 1:
            w._layer_used = bool(new & ~(1 << LAYER_BUTTON))
        return keys
    if prev >> LAYER_BUTTON & 1 and not getattr(w, "_layer_used", False):
        keys.append("c")
    w._layer_used = False
    keys += [k for i, k in KEY_BUTTONS.items() if new >> i & 1]
    if menu and new >> 1 & 1:        # B is the handbrake in a race, "back" in menus
        keys.append("escape")
    return keys
