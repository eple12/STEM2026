"""Synthesised engine note (Phase A: no downloaded audio).

A looping waveform is written to a .wav at start-up and played back with its
pitch driven by engine speed. Sound is a surprisingly large part of perceived
speed -- a rising note reads as acceleration even when the picture doesn't.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
BASE_HZ = 55.0          # fundamental of the loop at pitch 1.0
LOOP_SECONDS = 0.5


def _write_wav(path: Path, samples: np.ndarray) -> None:
    data = np.clip(samples, -1.0, 1.0)
    pcm = (data * 32000).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _engine_wave() -> np.ndarray:
    """A few harmonics plus combustion noise, seamless over the loop.

    Weighted low: the half-order and first two harmonics carry most of the
    energy and the upper partials are kept quiet, which is what makes a big
    engine read as heavy rather than buzzy.
    """
    # an exact number of cycles keeps the loop click-free
    cycles = round(BASE_HZ * LOOP_SECONDS)
    n = int(SAMPLE_RATE * LOOP_SECONDS)
    phase = np.linspace(0.0, 2 * math.pi * cycles, n, endpoint=False)

    sig = np.zeros(n)
    for harmonic, amp in ((0.5, 0.42), (1, 0.68), (2, 0.40), (3, 0.16),
                          (4, 0.09), (6, 0.035)):
        sig += amp * np.sin(phase * harmonic)

    # lopey V8 beat: a shaped half-order component
    sig += 0.22 * np.sin(phase * 0.5 + 0.7) ** 3

    # intake/exhaust rumble -- low-passed harder than before so it is body,
    # not hiss
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(n)
    k = 90
    noise = np.convolve(np.concatenate([noise[-k:], noise, noise[:k]]),
                        np.ones(k) / k, mode="same")[k:-k]
    sig += 0.30 * noise / (np.max(np.abs(noise)) or 1.0)

    sig /= np.max(np.abs(sig)) or 1.0
    # gentle saturation adds low-order harmonics and glues it together
    sig = np.tanh(sig * 1.5) / math.tanh(1.5)
    return sig * 0.85


class EngineSound:
    """Maps road speed + throttle onto a pitch and volume.

    ``idle_hz``/``max_hz`` are the *played fundamental*, so they say directly
    how the engine sits in the register: ~60 Hz is a heavy idle rumble and
    ~250 Hz is a big engine at the limiter. (Pushed much above that it starts
    to sound like a small, busy engine instead.)
    """

    def __init__(self, asset_dir: Path, idle_hz=60.0, max_hz=250.0):
        self.enabled = False
        self.idle_hz = idle_hz
        self.max_hz = max_hz
        self._rpm = 0.0
        try:
            path = asset_dir / "audio" / "engine_loop.wav"
            if not path.exists():
                _write_wav(path, _engine_wave())
            from ursina import Audio
            # Pass a Path, not a str: Ursina globs string names inside its own
            # asset folder and would never find ours.
            self.audio = Audio(path, loop=True, autoplay=True, volume=0.0)
            self.enabled = True
        except Exception as exc:  # pragma: no cover - audio is optional
            print("engine sound unavailable:", exc)
            self.audio = None

    def update(self, speed: float, max_speed: float, throttle: float,
               dt: float, muted: bool = False) -> None:
        if not self.enabled:
            return
        # Fake a gearbox: revs climb through a gear then drop on the shift.
        # The same gearbox the HUD's gear readout uses (see ui.gear_of).
        from .ui import gear_of
        _, within = gear_of(speed / max_speed)
        target = 0.22 + 0.78 * (0.30 + 0.70 * within) + 0.18 * throttle
        self._rpm += (target - self._rpm) * min(1.0, 7.0 * dt)

        hz = self.idle_hz + (self.max_hz - self.idle_hz) * self._rpm
        # the loop was rendered at BASE_HZ, so this ratio *is* the pitch
        self.audio.pitch = max(0.35, hz / BASE_HZ)
        self.audio.volume = 0.0 if muted else 0.16 + 0.30 * self._rpm

    def stop(self) -> None:
        """Silence the engine and give the Audio back.

        Ursina's ``Audio`` is an Entity, so muting it leaves a node in the
        scene graph. Racing, returning to the menu and racing again then piles
        up one dead Audio per race -- measured as exactly +1 scene child per
        menu/race cycle. It has to be destroyed, not just quietened.
        """
        if self.audio is not None:
            from ursina import destroy
            self.audio.volume = 0.0
            self.audio.stop(destroy=False)
            destroy(self.audio)
            self.audio = None
        self.enabled = False
