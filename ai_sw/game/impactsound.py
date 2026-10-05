"""Contact sounds: car against barrier, car against car, a car grinding along
a wall.

No recordings: each sound is put together from what a racing-car impact is
made of, the way a foley artist layers one --

* **crack**: a few milliseconds of broadband noise, the carbon itself
  fracturing (high-passed, so it snaps rather than thuds);
* **thump**: the mass of the car stopping, a low sine falling in pitch over
  a tenth of a second -- big for a heavy hit, absent for a tap;
* **panel modes**: the bodywork and wings ringing at a handful of inharmonic
  frequencies, each dying at its own rate (that ring is what tells carbon
  from steel);
* **debris**: for anything harder than a tap, a crunch of tiny grains spread
  over a few hundred milliseconds and thinning out, and a late tinkle of
  bits landing.

Three weights (light, medium, heavy) with three variations each, written to
``assets/audio`` once and played with a little random pitch, so no two hits
sound alike. The scrape is a loop: band-passed noise in stick-slip bursts,
its volume following the speed while the car is against the wall.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

SR = 44100
WEIGHTS = ("light", "medium", "heavy")
VARIANTS = 3


def _band(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Band-pass by masking the spectrum (fine for short one-shots)."""
    spec = np.fft.rfft(sig)
    f = np.fft.rfftfreq(len(sig), 1.0 / SR)
    mask = np.clip((f - lo * 0.7) / (lo * 0.3 + 1e-9), 0, 1) * \
        np.clip((hi * 1.3 - f) / (hi * 0.3), 0, 1)
    return np.fft.irfft(spec * mask, len(sig))


def _impact(weight: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = {"light": 0.0, "medium": 0.5, "heavy": 1.0}[weight]
    dur = 0.35 + 0.65 * w
    n = int(SR * dur)
    t = np.arange(n) / SR
    out = np.zeros(n)

    # Crack.
    crack = rng.normal(0, 1, n) * np.exp(-t / (0.004 + 0.006 * w))
    out += _band(crack, 1500.0, 12000.0) * (0.9 + 0.4 * w)

    # Thump: falling low sine.
    if w > 0.0:
        f0, f1 = 115.0 - 30.0 * w, 48.0
        freq = f1 + (f0 - f1) * np.exp(-t / 0.05)
        phase = 2 * math.pi * np.cumsum(freq) / SR
        out += np.sin(phase) * np.exp(-t / (0.05 + 0.10 * w)) * (0.6 + 0.9 * w)

    # Panel modes.
    for _ in range(5 + int(3 * w)):
        f = rng.uniform(300.0, 3200.0)
        tau = rng.uniform(0.015, 0.06) * (1.0 + w)
        a = rng.uniform(0.08, 0.30)
        out += a * np.sin(2 * math.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / tau)

    # Debris: grains thinning out over time, then a few late tinkles.
    if w > 0.0:
        grains = int(18 + 45 * w)
        for _ in range(grains):
            at = rng.exponential(0.06 + 0.12 * w)
            if at > dur - 0.03:
                continue
            k0 = int(at * SR)
            m = int(SR * rng.uniform(0.002, 0.007))
            g = rng.normal(0, 1, m) * np.exp(-np.arange(m) / (m * 0.3))
            out[k0:k0 + m] += g[:max(0, min(m, n - k0))] * rng.uniform(0.1, 0.5) \
                * math.exp(-at / (0.25 + 0.2 * w))
        for _ in range(int(4 + 8 * w)):
            at = rng.uniform(0.12, dur - 0.05)
            k0 = int(at * SR)
            m = int(SR * 0.03)
            f = rng.uniform(2500.0, 6000.0)
            tt = np.arange(m) / SR
            out[k0:k0 + m] += 0.06 * np.sin(2 * math.pi * f * tt) * np.exp(-tt / 0.008)

    # A soft attack (1 ms) so it does not click, and normalised.
    out[: int(0.001 * SR)] *= np.linspace(0, 1, int(0.001 * SR))
    out /= np.max(np.abs(out)) or 1.0
    return np.tanh(out * 1.4) / math.tanh(1.4) * 0.9


def _scrape() -> np.ndarray:
    rng = np.random.default_rng(91)
    n = SR * 2
    t = np.arange(n) / SR
    noise = _band(rng.normal(0, 1, n), 400.0, 4500.0)
    # Stick-slip: rough bursts at a few tens of hertz, unevenly.
    env = 0.55 + 0.45 * np.sign(np.sin(2 * math.pi * 37.0 * t + 3 * np.sin(2 * math.pi * 3.1 * t)))
    env *= 0.7 + 0.3 * np.sin(2 * math.pi * 7.3 * t) ** 2
    sig = noise * env
    # Loop seam: crossfade the end into the start.
    k = int(0.05 * SR)
    sig[:k] = sig[:k] * np.linspace(0, 1, k) + sig[-k:] * np.linspace(1, 0, k)
    sig = sig[:-k]
    sig /= np.max(np.abs(sig)) or 1.0
    return sig * 0.8


def _write(path: Path, sig: np.ndarray):
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(sig, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


class ImpactSound:
    """One-shots for hits, a loop for scraping. Silent if audio fails."""

    #: Change of speed (m/s) at which a hit becomes medium, and heavy.
    MEDIUM_DV, HEAVY_DV = 2.0, 6.0

    def __init__(self, asset_dir: Path):
        self.ok = False
        self.muted = False
        self._shots = {}
        self._next = {}
        self._scrape = None
        try:
            from ursina import Audio
            d = asset_dir / "audio"
            for wname in WEIGHTS:
                pool = []
                for v in range(VARIANTS):
                    p = d / f"impact_{wname}_{v}.wav"
                    if not p.exists():
                        _write(p, _impact(wname, 1000 * WEIGHTS.index(wname) + v))
                    # Two players per sound, so two hits close together can
                    # overlap rather than cut each other off.
                    pool += [Audio(p, autoplay=False, loop=False)
                             for _ in range(2)]
                self._shots[wname] = pool
                self._next[wname] = 0
            p = d / "scrape_loop.wav"
            if not p.exists():
                _write(p, _scrape())
            self._scrape = Audio(p, loop=True, autoplay=True, volume=0.0)
            self.ok = True
        except Exception as exc:  # pragma: no cover - audio is optional
            print("impact sounds unavailable:", exc)

    def hit(self, dv: float, gain: float = 1.0):
        """A hit that changed a car's speed by *dv* m/s; *gain* for distance."""
        if not self.ok or self.muted or dv < 0.4 or gain < 0.03:
            return
        wname = ("heavy" if dv >= self.HEAVY_DV else
                 "medium" if dv >= self.MEDIUM_DV else "light")
        pool = self._shots[wname]
        k = self._next[wname]
        self._next[wname] = (k + 1) % len(pool)
        a = pool[(k + random.randrange(len(pool))) % len(pool)]
        a.volume = min(1.0, (0.30 + dv / 9.0)) * gain
        a.pitch = random.uniform(0.88, 1.12)
        a.play()

    def scrape(self, speed: float, touching: bool, dt: float):
        """While a car grinds along a wall: louder and brighter with speed."""
        if self._scrape is None:
            return
        want = 0.0 if (self.muted or not touching or speed < 3.0) else \
            min(0.85, 0.15 + speed / 60.0)
        v = float(self._scrape.volume or 0.0)
        # Quick to come in, a touch slower to stop.
        k = min(1.0, (25.0 if want > v else 10.0) * dt)
        self._scrape.volume = v + (want - v) * k
        self._scrape.pitch = 0.8 + min(0.5, speed / 120.0)

    def stop(self):
        for pool in self._shots.values():
            for a in pool:
                a.stop()
        if self._scrape is not None:
            self._scrape.stop()
