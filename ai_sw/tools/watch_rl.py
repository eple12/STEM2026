"""Watch the policy drive, live, while it is still training.

The still picture from ``trace_rl`` answers what one episode looked like. This
answers what the policy is doing *now*: it drives episode after episode in a
window, and picks up each new checkpoint as the trainer writes it, so the
driving visibly changes as training goes on.

Nothing here writes anything. It re-reads the checkpoint file and runs its own
copy of the environment, so it cannot disturb a run and can be opened and
closed at will.

    python -m tools.watch_rl --circuit Monza

Keys: space pause, r restart the episode, n exploration noise on/off,
      1/2/3 simulation speed, q quit.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

import matplotlib
# TkAgg for the window; MPLBACKEND wins when it is set, which is what lets
# this be exercised headlessly.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from game import config, rlpolicy
from game.rlenv import RaceEnv

BG, PANEL, GRID, EDGE = "#12141a", "#1c1f27", "#2a2e38", "#3a3f4b"
FG, DIM, LINE_C = "#e8eaf0", "#8a90a0", "#ff7a3c"

#: Metres of history kept in the rolling panels.
WINDOW = 900.0
#: Path segments kept on the map. Long enough to show a whole sector.
TRAIL = 1400


class Checkpoint:
    """The trained weights, reloaded whenever the file on disk changes.

    The trainer rewrites the file every iteration, and a read that lands
    mid-write gets a truncated archive. That is not worth locking against for
    a viewer -- it just keeps the previous weights and tries again on the next
    frame.
    """

    def __init__(self, circuit: str):
        self.path = rlpolicy.policy_path(circuit)
        self.mtime = 0.0
        self.weights = None
        self.mean = self.std = None
        self.generation = 0
        self.reload()

    def reload(self) -> bool:
        try:
            m = self.path.stat().st_mtime
        except OSError:
            return False
        if m == self.mtime:
            return False
        try:
            d = np.load(self.path, allow_pickle=False)
            w = {k: d[k] for k in d.files if k[0] in "wb" and k[1:].isdigit()}
            mean, std = d["obs_mean"], d["obs_std"]
        except Exception:
            return False           # half-written; the old weights still drive
        self.weights, self.mean, self.std = w, mean, std
        self.mtime = m
        self.generation += 1
        return True

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.mtime)


class Viewer:
    def __init__(self, circuit: str, noise: float, rate: int):
        self.circuit = circuit
        self.env = RaceEnv(circuit, seed=0, randomise_start=False)
        self.ckpt = Checkpoint(circuit)
        if self.ckpt.weights is None:
            raise SystemExit(f"no checkpoint at {self.ckpt.path}")
        self.rng = np.random.default_rng(0)
        self.noise_amt = noise
        self.noise_on = False
        self.rate = rate
        self.paused = False
        self.hold = 0                      # frames to linger on a crash
        self.episode = 0
        self.best = 0.0
        self.last_reason = ""
        self._build()
        self.reset()

    # ------------------------------------------------------------------ setup
    def _build(self):
        t = self.env.track
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            f"watch_rl - {self.circuit}")
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.3, 1],
                                   hspace=0.45, wspace=0.14,
                                   left=0.02, right=0.97, top=0.93,
                                   bottom=0.07)

        ax = self.fig.add_subplot(gs[:, 0])
        self.ax_map = ax
        ax.set_facecolor(BG)
        er = t.center + t.normal * t.w_right[:, None]
        el = t.center - t.normal * t.w_left[:, None]
        ax.fill(*np.vstack([er, el[::-1]]).T, color=PANEL, zorder=0)
        for e in (er, el):
            ax.plot(*np.vstack([e, e[:1]]).T, color=EDGE, lw=1.0, zorder=1)
        g = self.env.line.center
        ax.plot(*np.vstack([g, g[:1]]).T, color=LINE_C, lw=1.0,
                ls=(0, (5, 4)), zorder=2)

        self.trail = LineCollection([], cmap="turbo",
                                    norm=plt.Normalize(0, 340), lw=2.8,
                                    zorder=3)
        ax.add_collection(self.trail)
        self.car, = ax.plot([], [], "o", ms=9, color="#ffffff", zorder=6)
        self.crash, = ax.plot([], [], "X", ms=14, color="#ff2d55", zorder=6)
        cb = self.fig.colorbar(self.trail, ax=ax, fraction=0.028, pad=0.01)
        cb.set_label("km/h", color=FG)
        cb.ax.tick_params(colors=DIM)
        cb.outline.set_edgecolor(EDGE)
        ax.set_aspect("equal")
        ax.axis("off")

        self.title = ax.set_title("", color=FG, fontsize=13, pad=12)
        self.readout = self.fig.text(
            0.025, 0.055, "", color=FG, fontsize=11, family="monospace",
            va="bottom", linespacing=1.7)

        def panel(row, title, ylabel):
            a = self.fig.add_subplot(gs[row, 1])
            a.set_facecolor(PANEL)
            a.set_title(title, color=FG, fontsize=10, loc="left")
            a.set_ylabel(ylabel, color=DIM, fontsize=9)
            a.tick_params(colors=DIM, labelsize=8)
            for sp in a.spines.values():
                sp.set_color(EDGE)
            a.grid(color=GRID, lw=0.6)
            return a

        self.ax_v = panel(0, "speed vs reference", "km/h")
        self.l_ref, = self.ax_v.plot([], [], color=LINE_C, lw=1.3,
                                     label="reference")
        self.l_spd, = self.ax_v.plot([], [], color="#4cc9f0", lw=1.6,
                                     label="policy")
        self.ax_v.legend(facecolor=BG, edgecolor=EDGE, labelcolor=FG,
                         fontsize=8, loc="lower left")
        self.ax_v.set_ylim(0, 350)

        self.ax_o = panel(1, "overspeed  (red = too fast for here)", "km/h")
        self.ax_o.axhline(0, color=DIM, lw=0.8)
        self.l_over, = self.ax_o.plot([], [], color="#ff2d55", lw=1.4)
        self.fill_o = None
        self.ax_o.set_ylim(-120, 200)

        self.ax_x = panel(2, "distance from the reference line", "m")
        self.ax_x.axhline(0, color=LINE_C, lw=0.9)
        for y in (-config.RL_LINE_TOLERANCE, config.RL_LINE_TOLERANCE):
            self.ax_x.axhline(y, color=DIM, lw=0.7, ls=":")
        self.l_x, = self.ax_x.plot([], [], color="#c77dff", lw=1.5)
        self.ax_x.set_xlabel("distance driven (m)", color=DIM, fontsize=9)
        self.ax_x.set_ylim(-16, 16)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # ------------------------------------------------------------- episode
    def reset(self):
        self.obs = self.env.reset()
        self.path = deque(maxlen=TRAIL)
        self.spd = deque(maxlen=TRAIL)
        self.hist = deque()             # (dist, speed, v_ref, cross)
        self.ret = 0.0
        self.episode += 1
        self.crash.set_data([], [])
        v = self.env.vehicle
        self.path.append(v.pos.copy())
        self.spd.append(v.speed * 3.6)

    def sim_step(self) -> bool:
        """One physics step. Returns False when the episode ended."""
        a = rlpolicy.mlp_forward(
            self.ckpt.weights,
            rlpolicy.normalise(self.obs[None], self.ckpt.mean,
                               self.ckpt.std))[0]
        if self.noise_on:
            a = a + self.rng.normal(0.0, self.noise_amt, a.shape)
        self.obs, r, done, info = self.env.step(np.clip(a, -1.0, 1.0))
        self.ret += r
        v = self.env.vehicle
        i = self.env._index()
        self.path.append(v.pos.copy())
        self.spd.append(v.speed * 3.6)
        self.hist.append((self.env.progress, v.speed * 3.6,
                          float(self.env.v_ref[i]) * 3.6,
                          float(np.dot(v.pos - self.env.line.center[i],
                                       self.env.line.normal[i]))))
        while self.hist and self.env.progress - self.hist[0][0] > WINDOW:
            self.hist.popleft()
        if done:
            self.last_reason = info["reason"]
            self.best = max(self.best, self.env.progress)
            self.crash.set_data([v.pos[0]], [v.pos[1]])
            return False
        return True

    # ---------------------------------------------------------------- frame
    def update(self, _):
        self.ckpt.reload()
        if self.hold > 0:
            self.hold -= 1
            if self.hold == 0:
                self.reset()
        elif not self.paused:
            for _ in range(self.rate):
                if not self.sim_step():
                    self.hold = 45          # ~1.5 s on the crash
                    break

        p = np.asarray(self.path)
        if len(p) > 1:
            self.trail.set_segments(np.stack([p[:-1], p[1:]], axis=1))
            self.trail.set_array(np.asarray(self.spd)[:-1])
        self.car.set_data([p[-1, 0]], [p[-1, 1]])

        if self.hist:
            h = np.asarray(self.hist)
            d, s, w, x = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
            self.l_ref.set_data(d, w)
            self.l_spd.set_data(d, s)
            self.l_over.set_data(d, s - w)
            self.l_x.set_data(d, x)
            if self.fill_o is not None:
                self.fill_o.remove()
            self.fill_o = self.ax_o.fill_between(
                d, 0, s - w, where=(s >= w), color="#ff2d55", alpha=0.7)
            lo, hi = d[0], max(d[-1], d[0] + 50.0)
            for a in (self.ax_v, self.ax_o, self.ax_x):
                a.set_xlim(lo, hi)

        e = self.env
        v = e.vehicle
        i = e._index()
        want = float(e.v_ref[i]) * 3.6
        off = abs(float(np.dot(v.pos - e.line.center[i], e.line.normal[i])))
        state = ("PAUSED" if self.paused else
                 f"ended: {self.last_reason}" if self.hold else "running")
        self.title.set_text(
            f"{self.circuit}   episode {self.episode}   {state}"
            f"   -   best {self.best:.0f} m")
        self.readout.set_text(
            f"distance   {e.progress:7.0f} m      best {self.best:7.0f} m\n"
            f"speed      {v.speed * 3.6:7.1f} km/h   reference {want:6.1f}\n"
            f"line error {off:7.2f} m      return {self.ret:8.1f}\n"
            f"checkpoint #{self.ckpt.generation}, {self.ckpt.age:.0f} s old"
            f"   noise {'on' if self.noise_on else 'off'}   {self.rate}x")
        return ()

    # ----------------------------------------------------------------- keys
    def on_key(self, ev):
        if ev.key == " ":
            self.paused = not self.paused
        elif ev.key == "r":
            self.hold = 0
            self.reset()
        elif ev.key == "n":
            self.noise_on = not self.noise_on
        elif ev.key in "123":
            self.rate = {"1": 1, "2": 2, "3": 6}[ev.key]
        elif ev.key == "q":
            plt.close(self.fig)

    def run(self):
        # Held on the instance: a FuncAnimation that is only a local goes out
        # of scope and stops drawing without saying so.
        self.anim = FuncAnimation(self.fig, self.update, interval=33,
                                  blit=False, cache_frame_data=False)
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--noise", type=float, default=0.4,
                    help="exploration noise applied when it is toggled on")
    ap.add_argument("--rate", type=int, default=2,
                    help="physics steps per frame; 2 is about real time")
    args = ap.parse_args()
    Viewer(args.circuit, args.noise, args.rate).run()


if __name__ == "__main__":
    main()
