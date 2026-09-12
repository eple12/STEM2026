"""Does the batched env reproduce the scalar one?

Runs N scalar ``RaceEnv``s and one ``VecRaceEnv`` of the same size from the
same deterministic grid starts, feeds both the identical action stream, and
reports the largest divergence in observation, reward and car state.

Three modes, because no single one reaches every branch:

* ``random``  -- random actions. Off the road and into a recovery within
  seconds, so the off-track fee, the patience window and the stall path all
  fire; a policy would never go there.
* ``wall``    -- the car is seeded a couple of metres inside a barrier chord
  facing out, so the collision test and the contact-arm impulse run on the
  very first tick. Random driving almost never gets that far: the off-track
  recovery fires first.
* ``policy``  -- greedy rollout of a trained checkpoint, long enough to close
  laps. This is the one that exercises lap arming, the clean-lap bonus and
  the ordinary on-the-limit driving the trainer actually sees.

    python tests/parity.py --mode random [--steps 400] [--envs 4]
    python tests/parity.py --mode wall
    python tests/parity.py --mode policy --steps 2200

Run in float64 (the default here) so the only differences left are the ones
that matter; the trainer runs float32.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai_sw"))

torch.set_default_dtype(torch.float64)

from game.rlenv import RaceEnv                             # noqa: E402
from gpuenv.vecenv import VecRaceEnv                       # noqa: E402

LAUNCHES = (0.0, 0.30, 0.55, 0.90)


def _seed_at_barrier(cpu_envs, vec):
    """Put every car a couple of metres inside a barrier chord, facing out.

    The collision test then runs on the first tick, which is the only
    reliable way to reach it: driven randomly the car is recovered for being
    off the track long before it travels the width of the run-off.
    """
    track = vec.t.cpu_track
    right, left = track.barrier_lines()
    segs = np.concatenate([s for s in (right, left) if len(s)], axis=0)
    n = len(cpu_envs)
    pick = np.linspace(0, len(segs) - 1, n).astype(int)

    pos = np.zeros((n, 2))
    yaw = np.zeros(n)
    vel = np.zeros((n, 2))
    idx = np.zeros(n, dtype=np.int64)
    for k, j in enumerate(pick):
        mid = segs[j].mean(axis=0)
        i = int(np.argmin(((track.center - mid) ** 2).sum(axis=1)))
        nrm = track.normal[i]
        side = 1.0 if float((mid - track.center[i]) @ nrm) > 0 else -1.0
        out = nrm * side
        pos[k] = mid - out * 2.5
        yaw[k] = math.atan2(out[0], out[1])
        vel[k] = np.array([math.sin(yaw[k]), math.cos(yaw[k])]) * 45.0
        idx[k] = i

    for k, e in enumerate(cpu_envs):
        e.vehicle.place(pos[k], yaw[k])
        e.vehicle.frozen = False
        e.vehicle.vel = vel[k].copy()
        e.surface.hint = int(idx[k])
        e._last_s = float(e.line.arclen[idx[k]])
        e.start_s = e._last_s
        e.off_time = 0.0
        e.stall_time = 0.0

    m = torch.ones(n, dtype=torch.bool)
    vec.vehicle.place(m, torch.as_tensor(pos), torch.as_tensor(yaw),
                      torch.as_tensor(vel))
    vec.surface.hint = torch.as_tensor(idx)
    vec._last_s = vec.t.l_arclen[torch.as_tensor(idx)].clone()
    vec.start_s = vec._last_s.clone()
    vec.off_time = torch.zeros(n, dtype=torch.float64)
    vec.stall_time = torch.zeros(n, dtype=torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--envs", type=int, default=len(LAUNCHES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--mode", choices=("random", "wall", "policy"),
                    default="random")
    ap.add_argument("--policy", default="Monza",
                    help="checkpoint basename under ai_sw/assets/policies")
    args = ap.parse_args()

    n = args.envs
    fracs = [LAUNCHES[k % len(LAUNCHES)] for k in range(n)]

    cpu_envs = [RaceEnv("Monza", seed=7 + k, randomise_start=False)
                for k in range(n)]
    cpu_obs = np.stack([e.reset_grid(f) for e, f in zip(cpu_envs, fracs)])

    vec = VecRaceEnv("Monza", n_env=n, randomise_start=False, device="cpu")
    vec_obs = vec.reset_grid(torch.tensor(fracs, dtype=torch.float64))

    if args.mode == "wall":
        _seed_at_barrier(cpu_envs, vec)
        cpu_obs = np.stack([e.observe() for e in cpu_envs])
        vec_obs = vec.observe()

    W = mean = std = None
    if args.mode == "policy":
        import game.config as gcfg
        from game.rlpolicy import iqn_q, normalise
        d = np.load(gcfg.RL_POLICY / f"{args.policy}.npz", allow_pickle=False)
        W = {k: d[k] for k in d.files if "." in k}
        mean, std = d["obs_mean"], d["obs_std"]
        print(f"greedy rollout of {args.policy}.npz")

    d0 = np.abs(cpu_obs - vec_obs.numpy()).max()
    print(f"reset obs   max|diff| {d0:.3e}")

    rng = np.random.default_rng(args.seed)
    worst = {"obs": 0.0, "rew": 0.0, "pos": 0.0, "yaw": 0.0, "spd": 0.0}
    first_bad = None

    for t in range(args.steps):
        if args.mode == "policy":
            from game.rlpolicy import iqn_q, normalise
            act = iqn_q(W, normalise(cpu_obs, mean, std)).argmax(1)
        elif args.mode == "wall":
            act = np.full(n, 3, dtype=np.int64)      # straight, full throttle
        else:
            act = rng.integers(vec.act_dim, size=n)

        cpu_r, cpu_d, nxt = [], [], []
        for k, e in enumerate(cpu_envs):
            o, r, d, _ = e.step(int(act[k]))
            nxt.append(o)
            cpu_r.append(r)
            cpu_d.append(d)
        cpu_obs = np.stack(nxt)

        vo, vr, vd, _ = vec.step(torch.as_tensor(act))
        vec_obs = vo.numpy()

        do = np.abs(cpu_obs - vec_obs).max()
        dr = np.abs(np.asarray(cpu_r) - vr.numpy()).max()
        dp = max(np.abs(e.vehicle.pos - vec.vehicle.pos[k].numpy()).max()
                 for k, e in enumerate(cpu_envs))
        dy = max(abs(e.vehicle.yaw - float(vec.vehicle.yaw[k]))
                 for k, e in enumerate(cpu_envs))
        ds = max(abs(e.vehicle.speed - float(vec.vehicle.speed[k]))
                 for k, e in enumerate(cpu_envs))
        worst["obs"] = max(worst["obs"], do)
        worst["rew"] = max(worst["rew"], dr)
        worst["pos"] = max(worst["pos"], dp)
        worst["yaw"] = max(worst["yaw"], dy)
        worst["spd"] = max(worst["spd"], ds)
        if first_bad is None and max(do, dr, dp) > args.tol:
            first_bad = (t, do, dr, dp)

        assert list(vd.numpy()) == cpu_d, f"done mismatch at step {t}"

        counters = {
            "off_steps": ([e.off_steps for e in cpu_envs],
                          vec.off_steps.tolist()),
            "recoveries": ([e.recoveries for e in cpu_envs],
                           vec.recoveries.tolist()),
            "wall_steps": ([e.wall_steps for e in cpu_envs],
                           vec.wall_steps.tolist()),
            "laps": ([e.laps for e in cpu_envs], vec.laps.tolist()),
        }
        for name, (a, b) in counters.items():
            if a != b:
                print(f"  step {t}: {name} CPU {a} != GPU {b}")
                return 1

        if any(cpu_d):
            m = torch.as_tensor(cpu_d)
            for k, e in enumerate(cpu_envs):
                if cpu_d[k]:
                    cpu_obs[k] = e.reset()
            vec.reset_done(m)
            # Resets draw from different RNG streams, so re-sync the scalar
            # envs onto the batched placement and keep comparing.
            for k, e in enumerate(cpu_envs):
                if not cpu_d[k]:
                    continue
                e.vehicle.pos = vec.vehicle.pos[k].numpy().copy()
                e.vehicle.yaw = float(vec.vehicle.yaw[k])
                e.vehicle.vel = vec.vehicle.vel[k].numpy().copy()
                e.surface.hint = int(vec.surface.hint[k])
                e._last_s = float(vec._last_s[k])
                e.start_s = float(vec.start_s[k])
                cpu_obs[k] = e.observe()

    print(f"steps {args.steps}, envs {n}, mode {args.mode}")
    print(f"  coverage: wall {sum(e.wall_steps for e in cpu_envs)}  "
          f"rec {sum(e.recoveries for e in cpu_envs)}  "
          f"off {sum(e.off_steps for e in cpu_envs)}  "
          f"laps {sum(e.laps for e in cpu_envs)}")
    for k, v in worst.items():
        print(f"  max|diff| {k:4s} {v:.3e}")
    if first_bad:
        t, do, dr, dp = first_bad
        print(f"  first above {args.tol:g}: step {t} "
              f"(obs {do:.2e}, rew {dr:.2e}, pos {dp:.2e})")
        return 1
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
