"""In-game screen recording: one key (``config.REC_KEY``, F9) starts and stops it.

Recording from inside the renderer would have to read every finished frame back
from the GPU -- a stall per frame on this laptop -- and then encode it on the
CPU that the game's Python and Panda's cull and draw threads already fill.
That is what makes a plain 60 fps recording lag. So the game does none of it:
it starts ``ffmpeg`` on the side and lets it do what OBS's display capture
does, only without OBS and with the encoder on the GPU:

* **Capture** is Windows' Desktop Duplication (``ddagrab``): the finished
  desktop image, already in GPU memory, cropped to the game's window (the whole
  monitor when the window is fullscreen).
* **Encode** is the GPU's fixed-function H.264 block -- Intel Quick Sync here
  (``h264_qsv``), else NVENC or AMF -- fed straight from that texture. The CPU
  never sees a pixel. Where none of them will start, CPU x264 at its fastest
  preset is the last resort.

The recording is what is on screen, so the window must be in front (fullscreen
is ideal) and the HUD, and the small red REC marker, are in it
(``config.REC_INDICATOR``). No sound: the game's audio is not captured.

The file is a fragmented MP4, so a crash or a power cut still leaves a playable
video up to the last second.
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from . import config

#: ffmpeg is looked for here when it is not on PATH.
FFMPEG_FALLBACKS = (r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    r"C:\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe")
#: An encoder that is still running this long after it started is working.
PROBE_T = 2.5


def find_ffmpeg() -> str | None:
    if config.REC_FFMPEG and Path(config.REC_FFMPEG).is_file():
        return config.REC_FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        return found
    return next((p for p in FFMPEG_FALLBACKS if Path(p).is_file()), None)


def out_dir() -> Path:
    d = Path(config.REC_DIR) if config.REC_DIR else Path.home() / "Videos" / "FORMULA-AI"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dpi_scale() -> float:
    """Desktop pixels per window pixel: 1 unless the game is DPI-unaware on a
    scaled display, where Panda's sizes are in scaled-down units."""
    try:
        import ctypes
        aware = ctypes.c_int(0)
        ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(aware))
        if aware.value != 0:
            return 1.0
        scale = ctypes.c_int(100)
        ctypes.windll.shcore.GetScaleFactorForDevice(0, ctypes.byref(scale))
        return max(scale.value, 100) / 100.0
    except Exception:
        return 1.0


def capture_rect():
    """(DXGI output index, x, y, w, h) of the game window, in desktop pixels
    relative to its monitor; w and h even (the encoders need it). None if it
    cannot be worked out."""
    import builtins

    props = builtins.base.win.get_properties()
    try:
        from screeninfo import get_monitors
        mons = list(get_monitors())
    except Exception:
        mons = []
    k = _dpi_scale()
    ox, oy = props.get_x_origin() * k, props.get_y_origin() * k
    w, h = props.get_x_size() * k, props.get_y_size() * k
    if not mons:
        return 0, int(ox), int(oy), int(w) // 2 * 2, int(h) // 2 * 2
    cx, cy = ox + w / 2, oy + h / 2
    mon = next((m for m in mons if m.x <= cx < m.x + m.width
                and m.y <= cy < m.y + m.height), mons[0])
    # DXGI lists the primary monitor first, the rest in the order found.
    others = [m for m in mons if m is not mon and not m.is_primary]
    idx = 0 if mon.is_primary else 1 + others.index(mon) if mon in others else 0
    x = max(0, int(ox - mon.x))
    y = max(0, int(oy - mon.y))
    w = min(int(w), mon.width - x) // 2 * 2
    h = min(int(h), mon.height - y) // 2 * 2
    return idx, x, y, w, h


def commands(ffmpeg: str, rect, fps: int, mbps: float, path: Path):
    """The ffmpeg command lines to try, best first: [(name, argv)]."""
    idx, x, y, w, h = rect
    src = (f"ddagrab=output_idx={idx}:framerate={fps}:draw_mouse=0"
           f":video_size={w}x{h}:offset_x={x}:offset_y={y}")
    head = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"]
    tail = ["-movflags", "frag_keyframe+empty_moov+default_base_moof", "-y", str(path)]
    b = f"{mbps:g}M"
    mx = f"{mbps * 1.3:g}M"
    cpu = src + ",hwdownload,format=bgra,format=yuv420p"
    return [
        ("Quick Sync", head + [src + ",hwmap=derive_device=qsv,format=qsv",
                               "-c:v", "h264_qsv", "-b:v", b, "-maxrate", mx,
                               "-preset", "veryfast", "-look_ahead", "0",
                               "-bf", "0"] + tail),
        ("NVENC", head + [cpu, "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll",
                          "-b:v", b, "-maxrate", mx, "-bf", "0"] + tail),
        ("AMF", head + [cpu, "-c:v", "h264_amf", "-quality", "speed",
                        "-b:v", b, "-bf", "0"] + tail),
        ("x264", head + [cpu, "-c:v", "libx264", "-preset", "ultrafast",
                         "-b:v", b, "-maxrate", mx, "-bf", "0"] + tail),
    ]


class Recorder:
    def __init__(self):
        self.proc = None
        self.path: Path | None = None
        self.t0 = 0.0
        self._cmds: list = []
        self._name = ""
        self._err = None
        self._ind = None
        self._toast = None
        self._toast_until = 0.0
        self._pending_toast: tuple[str, float] | None = None
        self._last_sec = -1
        atexit.register(self.stop, wait=True)

    @property
    def recording(self) -> bool:
        return self.proc is not None

    # -- control ---------------------------------------------------------
    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

    def start(self):
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            self.say("RECORDING NEEDS FFMPEG  ·  NOT FOUND", 4.0)
            return
        rect = capture_rect()
        if rect is None or rect[3] < 64 or rect[4] < 64:
            self.say("RECORDING: COULD NOT FIND THE WINDOW", 4.0)
            return
        self.path = out_dir() / datetime.now().strftime("formula-ai_%Y%m%d_%H%M%S.mp4")
        self._cmds = commands(ffmpeg, rect, int(config.REC_FPS),
                              float(config.REC_BITRATE_MBPS), self.path)
        self._launch()

    def _launch(self):
        self._name, argv = self._cmds.pop(0)
        if self._err is not None:
            self._err.close()
        self._err = tempfile.TemporaryFile()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=self._err,
                                     creationflags=flags)
        self.t0 = time.time()
        self._last_sec = -1

    def stop(self, wait: bool = False):
        proc, path = self.proc, self.path
        if proc is None:
            return
        self.proc = None
        self._set_indicator(None)

        def finish():
            try:
                if proc.poll() is None:
                    proc.stdin.write(b"q")      # ffmpeg's own "stop and finalise"
                    proc.stdin.flush()
                    proc.wait(timeout=15)
            except Exception:
                proc.kill()
            ok = path is not None and path.is_file() and path.stat().st_size > 4096
            self._pending_toast = (f"SAVED  ·  {path.name}" if ok
                                   else "RECORDING FAILED  ·  NO FILE", 4.0)
        if wait:
            finish()
        else:
            threading.Thread(target=finish, daemon=True).start()

    # -- per frame -------------------------------------------------------
    def tick(self):
        if self._pending_toast is not None:
            text, secs = self._pending_toast
            self._pending_toast = None
            self.say(text, secs)
        if self._toast is not None and time.time() > self._toast_until:
            self._destroy(self._toast)
            self._toast = None
        proc = self.proc
        if proc is None:
            return
        age = time.time() - self.t0
        if proc.poll() is not None:
            # It died. Early on that means this encoder is not available here:
            # try the next, the way a recorder with a fallback list would.
            if age < PROBE_T + 3.0 and self._cmds:
                self._launch()
                return
            self.proc = None
            self._set_indicator(None)
            self.say("RECORDING FAILED  ·  " + self._why(), 5.0)
            return
        if age < PROBE_T:
            return
        sec = int(age)
        if sec != self._last_sec:
            self._last_sec = sec
            self._set_indicator(f"●  REC  {sec // 60:02d}:{sec % 60:02d}")

    def _why(self) -> str:
        try:
            self._err.seek(0)
            lines = [ln.strip() for ln in self._err.read().decode("utf-8", "replace")
                     .splitlines() if ln.strip()]
            return lines[-1][:70].upper() if lines else "ENCODER EXITED"
        except Exception:
            return "ENCODER EXITED"

    # -- on-screen marks --------------------------------------------------
    def say(self, text: str, secs: float):
        from ursina import Text, camera, color
        if self._toast is not None:
            self._destroy(self._toast)
        self._toast = Text(text=text, parent=camera.ui, origin=(0, 0),
                           position=(0, 0.44), scale=1.1, z=-5,
                           color=color.rgba(255, 255, 255, 235), eternal=True)
        self._toast_until = time.time() + secs

    def _set_indicator(self, text: str | None):
        if text is None or not config.REC_INDICATOR:
            if self._ind is not None:
                self._destroy(self._ind)
                self._ind = None
            return
        from ursina import Text, Vec2, camera, color, window
        if self._ind is None or getattr(self._ind, "_destroyed_", False):
            self._ind = Text(text=text, parent=camera.ui, origin=(0.5, 0.5),
                             position=window.top_right - Vec2(0.018, 0.018),
                             scale=0.9, z=-5, color=color.rgb32(235, 40, 40),
                             eternal=True)
        else:
            self._ind.text = text

    @staticmethod
    def _destroy(e):
        try:
            from ursina import destroy
            destroy(e)
        except Exception:
            pass


RECORDER = Recorder()
