"""Draw what the current policy actually does, as a picture.

Training reports a distance and a failure reason, which says where an episode
ended but nothing about how it got there. This runs the checkpoint on the spot
-- the trainer rewrites it every iteration, so it is always the live policy --
and plots the path over the track: coloured by speed, against the reference
line it is being asked to follow, with the place it went wrong marked.

Run it whenever, including mid-training. It reads the checkpoint and touches
nothing, so it cannot disturb the run.

    python -m tools.trace_rl --circuit Monza --out trace.png

Pass --policy to trace an arbitrary checkpoint file instead of the deployed
<circuit>.npz -- a freshly-fetched Kaggle output, say, before deciding
whether to deploy it. The episode window is widened to a few laps (not just
the deployed policy's normal single-attempt length) so a genuine clean lap
has room to close and ``best_lap_time`` means something.
"""
from __future__ import annotations

import argparse
import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from game import config, rlpolicy
from game.rlenv import RaceEnv


def run(env: RaceEnv, weights, mean, std, noisy: float, seed: int):
    """One episode. Returns the path, speeds, reference speeds and the end."""
    rng = np.random.default_rng(seed)
    obs = env.reset()
    path, speed, want, over = [], [], [], []
    info = {"reason": "time"}
    while True:
        q = rlpolicy.iqn_q(weights, rlpolicy.normalise(obs[None], mean, std))[0]
        if noisy > 0.0 and rng.random() < noisy:
            a = int(rng.integers(rlpolicy.N_ACTIONS))
        else:
            a = int(np.argmax(q))
        obs, _, done, info = env.step(a)
        v = env.vehicle
        i = env._index()
        path.append(v.pos.copy())
        speed.append(v.speed * 3.6)
        want.append(float(env.v_ref[i]) * 3.6)
        over.append(float(np.dot(v.pos - env.line.center[i],
                                 env.line.normal[i])))
        if done:
            break
    return (np.asarray(path), np.asarray(speed), np.asarray(want),
            np.asarray(over), info, env.progress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--out", default="trace.png")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="exploration noise; 0 is the policy as it will drive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=None,
                    help="trace this checkpoint file instead of the "
                         "deployed assets/policies/<circuit>.npz")
    ap.add_argument("--lap-time-file", default=None,
                    help="write the best lap time (seconds, or empty if no "
                         "clean lap closed) to this file -- lets a calling "
                         "script (train_track.bat) pick it up without "
                         "parsing the printed summary line")
    args = ap.parse_args()

    # The reported best lap must be a genuinely clean one: the training
    # reward tolerates a few off-track steps per lap (RL_RL_LAP_OFF_TOL), which
    # is not what "clean" means for a ghost lap or a record.
    config.RL_RL_LAP_OFF_TOL = config.RL_OFF_PERFECT_TOL
    env = RaceEnv(args.circuit, seed=args.seed, randomise_start=False)
    env.episode_seconds = 300.0  # room for a couple of clean laps to close
    policy_path = args.policy or rlpolicy.policy_path(args.circuit)
    d = np.load(policy_path, allow_pickle=False)
    w = {k: d[k] for k in d.files if "." in k}
    path, speed, want, cross, info, dist = run(
        env, w, d["obs_mean"], d["obs_std"], args.noise, args.seed)

    t = env.track
    fig = plt.figure(figsize=(15, 8.5), facecolor="#12141a")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.42,
                          wspace=0.16)

    # ---- the track, and the path over it ---------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("#12141a")
    edge_r = t.center + t.normal * t.w_right[:, None]
    edge_l = t.center - t.normal * t.w_left[:, None]
    for e in (edge_r, edge_l):
        ax.plot(*np.vstack([e, e[:1]]).T, color="#3a3f4b", lw=1.0)
    ax.fill(*np.vstack([edge_r, edge_l[::-1]]).T, color="#1c1f27", zorder=0)
    ax.plot(*np.vstack([env.line.center, env.line.center[:1]]).T,
            color="#ff7a3c", lw=1.1, ls=(0, (5, 4)), label="reference line",
            zorder=2)

    seg = np.stack([path[:-1], path[1:]], axis=1)
    lc = LineCollection(seg, cmap="turbo", norm=plt.Normalize(0, 340),
                        lw=2.6, zorder=3)
    lc.set_array(speed[:-1])
    ax.add_collection(lc)
    ax.scatter(*path[0], s=70, c="#4cd964", zorder=5, label="start",
               edgecolors="none")
    ax.scatter(*path[-1], s=110, c="#ff2d55", marker="X", zorder=5,
               edgecolors="none", label=f"end: {info['reason']}")
    cb = fig.colorbar(lc, ax=ax, fraction=0.030, pad=0.01)
    cb.set_label("km/h", color="#c8ccd6")
    cb.ax.tick_params(colors="#8a90a0")
    cb.outline.set_edgecolor("#3a3f4b")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", facecolor="#1c1f27", edgecolor="#3a3f4b",
              labelcolor="#c8ccd6", fontsize=9)
    ax.set_title(f"{args.circuit} - {dist:.0f} m, ended: {info['reason']}",
                 color="#e8eaf0", fontsize=13, pad=14)

    # ---- speed against the reference, and line error ----------------------
    s_ax = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(path, axis=0), axis=1)])

    def panel(row, title, ylabel):
        a = fig.add_subplot(gs[row, 1])
        a.set_facecolor("#1c1f27")
        a.set_title(title, color="#e8eaf0", fontsize=10, loc="left")
        a.set_ylabel(ylabel, color="#8a90a0", fontsize=9)
        a.tick_params(colors="#8a90a0", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#3a3f4b")
        a.grid(color="#2a2e38", lw=0.6)
        return a

    a = panel(0, "speed vs reference", "km/h")
    a.plot(s_ax, want, color="#ff7a3c", lw=1.2, label="reference")
    a.plot(s_ax, speed, color="#4cc9f0", lw=1.4, label="policy")
    a.legend(facecolor="#12141a", edgecolor="#3a3f4b", labelcolor="#c8ccd6",
             fontsize=8)

    a = panel(1, "overspeed (positive = too fast for here)", "km/h")
    a.axhline(0, color="#8a90a0", lw=0.8)
    a.fill_between(s_ax, 0, speed - want, where=(speed >= want),
                   color="#ff2d55", alpha=0.75)
    a.fill_between(s_ax, 0, speed - want, where=(speed < want),
                   color="#4cd964", alpha=0.45)

    a = panel(2, "distance from the reference line", "m")
    a.axhline(0, color="#ff7a3c", lw=0.9)
    for y in (-config.RL_LINE_TOLERANCE, config.RL_LINE_TOLERANCE):
        a.axhline(y, color="#8a90a0", lw=0.7, ls=":")
    a.plot(s_ax, cross, color="#c77dff", lw=1.3)
    a.set_xlabel("distance driven (m)", color="#8a90a0", fontsize=9)

    fig.savefig(args.out, dpi=115, facecolor=fig.get_facecolor())
    lap_note = (f", best lap {env.best_lap_time:.2f}s"
                if env.best_lap_time > 0.0 else ", no clean lap closed")
    print(f"{args.circuit}: {dist:.0f} m, ended {info['reason']!r}, "
          f"mean {speed.mean():.1f} km/h, "
          f"mean line error {np.abs(cross).mean():.2f} m{lap_note} -> {args.out}")
    if args.lap_time_file:
        with open(args.lap_time_file, "w", encoding="utf-8") as f:
            f.write(f"{env.best_lap_time:.2f}" if env.best_lap_time > 0.0 else "")


if __name__ == "__main__":
    main()
