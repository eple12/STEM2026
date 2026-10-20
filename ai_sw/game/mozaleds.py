"""MOZA wheel RPM indicator LEDs, driven over the base's USB serial port.

Frame format and command ids come from Boxflat's reverse-engineered protocol
(github.com/Lawstorant/boxflat, data/serial.yml):

    0x7E, len, group, device, id..., payload..., checksum
    len      = len(id) + len(payload)
    checksum = (13 + sum of all bytes before it) % 256

Wheel (device 23, can differ) commands used:
    [28, 0]  telemetry-mode   write group 63 / read group 64:  0 off, 1 telemetry, 2 static
    [26, 0]  send-rpm-telemetry, 2 bytes, bit i (little endian) = LED i lit

The wheel is put into telemetry mode while the game runs and left in
``restore_mode`` on exit (the previous mode cannot be read without a reader). The port is opened as a plain file (no pyserial).
"""
from __future__ import annotations

import atexit
import threading
import time

START, MAGIC = 0x7E, 13
WRITE_GROUP, READ_GROUP = 63, 64
LEDS = 10
MODE_ID, RPM_ID = (28, 0), (26, 0)


def frame(group: int, device: int, ident, payload: bytes) -> bytes:
    body = bytes([START, len(ident) + len(payload), group, device, *ident]) + payload
    return body + bytes([(MAGIC + sum(body)) % 256])


def find_port() -> str | None:
    """COM port of the MOZA base: the USB serial device of vendor 346E."""
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Enum\USB")
        for i in range(winreg.QueryInfoKey(root)[0]):
            dev = winreg.EnumKey(root, i)
            if not dev.upper().startswith("VID_346E"):
                continue
            k = winreg.OpenKey(root, dev)
            for j in range(winreg.QueryInfoKey(k)[0]):
                try:
                    p = winreg.OpenKey(k, winreg.EnumKey(k, j) + r"\Device Parameters")
                    return winreg.QueryValueEx(p, "PortName")[0]
                except OSError:
                    continue
    except (OSError, ImportError):
        pass
    return None


class MozaLeds:
    def __init__(self, port: str | None = None, device: int = 23, restore_mode: int = 1):
        self.ok = False
        self.device = device
        self._f = None
        self._want = 0
        self._sent = -1
        self._restore = restore_mode     # mode the wheel is left in on exit
        self._stop = threading.Event()
        port = port or find_port()
        if not port:
            return
        try:
            self._f = open(rf"\\.\{port}", "r+b", buffering=0)
        except OSError as e:           # e.g. Pit House holds the port
            print("MOZA LEDs: cannot open", port, e)
            return
        self.ok = True
        self._send(WRITE_GROUP, MODE_ID, bytes([1]))
        atexit.register(self.close)
        threading.Thread(target=self._run, daemon=True).start()

    def _send(self, group, ident, payload):
        try:
            self._f.write(frame(group, self.device, ident, payload))
        except OSError:
            self.ok = False

    def set_fraction(self, frac: float, blink: bool = False) -> None:
        n = max(0, min(LEDS, int(frac * LEDS + 0.5)))
        mask = (1 << n) - 1
        if blink and int(time.time() * 8) % 2:
            mask = 0
        self._want = mask

    def _run(self):
        while not self._stop.is_set() and self.ok:
            if self._want != self._sent:
                m = self._want
                self._send(WRITE_GROUP, RPM_ID, bytes([m & 255, m >> 8]))
                self._sent = m
            time.sleep(0.02)

    def close(self):
        if self._f is None:
            return
        self._stop.set()
        try:
            self._send(WRITE_GROUP, RPM_ID, bytes([0, 0]))
            # leave the wheel in the mode it normally runs in (wheel.json: led_restore_mode)
            self._send(WRITE_GROUP, MODE_ID, bytes([self._restore]))
            self._f.close()
        except OSError:
            pass
        self._f = None
        self.ok = False


_shared: MozaLeds | None = None
_tried = False


def shared() -> MozaLeds | None:
    """The LED driver if wheel.json has "leds": true and the port could be opened."""
    global _shared, _tried
    if not _tried:
        _tried = True
        from . import wheel
        cfg = wheel.load_config()
        if cfg.get("leds"):
            led = MozaLeds(cfg.get("leds_port"), int(cfg.get("leds_device", 23)),
                           int(cfg.get("leds_restore_mode", 1)))
            _shared = led if led.ok else None
    return _shared
