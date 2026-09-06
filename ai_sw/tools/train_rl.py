"""PPO on the driving task, with the f1tenth racing line given as the route.

    python tools/train_rl.py --circuit Monza --steps 5000000
    python tools/train_rl.py --circuit Monza --workers 8 --rollout 2048

How the parallelism is arranged
-------------------------------
Not one env per process stepped in lockstep from the parent: the simulator runs
at about 1,100 steps a second, so a step costs 0.9 ms and a round trip to ask
the parent for an action would be a large fraction of that again.

Instead each worker owns its environments *and a copy of the policy*, runs a
whole rollout segment, and sends back the trajectory. The workers do their
forward passes in numpy from exported weights, so torch is only ever loaded in
the parent, and the inter-process traffic is one weight blob out and one
trajectory back per iteration rather than two messages per step.

That is also why the game needs no torch: the same numpy forward pass that runs
in the workers is what runs in the car.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import collections
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game import config
from game.rlenv import RaceEnv
from game.rlpolicy import mlp_forward, normalise

HIDDEN = (128, 128)


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
_ENVS = {}


def _envs(circuit, count, seed, start_at_line):
    key = (circuit, seed)
    if key not in _ENVS:
        _ENVS[key] = [RaceEnv(circuit, seed=seed * 977 + i)
                      for i in range(count)]
        for e in _ENVS[key]:
            e.reset()
    for e in _ENVS[key]:
        e.start_at_line = start_at_line
    return _ENVS[key]


def _rollout(args):
    """One trajectory segment from this worker's environments."""
    (circuit, seed, n_env, steps, weights, log_std, mean, std,
     start_at_line) = args
    envs = _envs(circuit, n_env, seed, start_at_line)
    rng = np.random.default_rng(seed * 7919 + 13)
    scale = np.exp(log_std)

    obs_buf = np.zeros((steps, n_env, envs[0].obs_dim), dtype=np.float32)
    act_buf = np.zeros((steps, n_env, 2), dtype=np.float32)
    logp_buf = np.zeros((steps, n_env), dtype=np.float32)
    rew_buf = np.zeros((steps, n_env), dtype=np.float32)
    done_buf = np.zeros((steps, n_env), dtype=np.float32)
    # Separate from done_buf: a time-limit truncation resets the environment
    # but must still be bootstrapped, or the value function learns that the
    # world ends every forty-five seconds and stops valuing anything past it.
    term_buf = np.zeros((steps, n_env), dtype=np.float32)
    stats = []

    cur = np.stack([e.observe() for e in envs])
    for t in range(steps):
        norm = normalise(cur, mean, std)
        mu = mlp_forward(weights, norm)
        act = mu + scale * rng.standard_normal(mu.shape)
        # Gaussian log-density of the *unclipped* sample, which is what the
        # ratio in the PPO objective has to be consistent with.
        logp = (-0.5 * (((act - mu) / scale) ** 2).sum(axis=1)
                - np.log(scale).sum() - 0.5 * 2 * math.log(2 * math.pi))

        obs_buf[t], act_buf[t], logp_buf[t] = norm, act, logp
        for k, e in enumerate(envs):
            o, r, d, info = e.step(act[k])
            rew_buf[t, k], done_buf[t, k] = r, float(d)
            term_buf[t, k] = float(d and not info.get("truncated"))
            if d:
                stats.append((e.progress, e.laps, e.lap_time,
                              e.off_steps / max(e.steps, 1),
                              e.wall_steps / max(e.steps, 1),
                              e.speed_sum / max(e.steps, 1),
                              info["reason"]))
                o = e.reset()
            cur[k] = o
    last = normalise(cur, mean, std)
    # Live episodes count too. The furthest a car got this iteration is the
    # readout of how much of the lap the policy has learned, and an episode
    # still running when the segment ends is often the best one.
    live = [e.progress for e in envs]
    return (obs_buf, act_buf, logp_buf, rew_buf, done_buf, term_buf, last,
            stats, live)



# ---------------------------------------------------------------------------
def _bc_pretrain(actor, critic, obs_dim, args, torch, nn):
    """Fit the actor to the scripted planner before PPO, and seed the critic
    and the observation statistics from the same data.

    The planner drives the racing line -- including the chicane the last four
    reward designs never got a policy through -- so this hands PPO a starting
    point that already gets round the lap. PPO then only has to find the time
    in it, which is a search it can actually run: every rollout is a full lap,
    not a sequence of crashes at turn one.

    Collection injects a little action noise and records the planner's clean
    action, so the set contains the planner correcting from states slightly
    off its own trajectory. Straight behaviour cloning without that drifts:
    the policy makes a small error, lands somewhere the planner never was, and
    has no label for how to get back.
    """
    from game.autopilot import Autopilot

    n_env = 16
    envs = [RaceEnv(args.circuit, seed=args.seed * 17 + i)
            for i in range(n_env)]
    for e in envs:
        e.uniform_start = True
        e.reset()
    aps = [Autopilot(e.track, e.surface, line=e.line) for e in envs]
    rng = np.random.default_rng(args.seed)
    per = max(1, args.bc_steps // n_env)

    O = np.zeros((per, n_env, obs_dim), dtype=np.float32)
    A = np.zeros((per, n_env, 2), dtype=np.float32)
    Rw = np.zeros((per, n_env), dtype=np.float32)
    Dn = np.zeros((per, n_env), dtype=np.float32)
    print(f"  collecting {per * n_env:,} planner transitions "
          f"({n_env} envs)", flush=True)
    t_bc = time.perf_counter()
    for t in range(per):
        for k, (e, ap) in enumerate(zip(envs, aps)):
            O[t, k] = e.observe()
            c = ap.controls(e.vehicle)
            clean = np.array([c.steer, c.throttle - c.brake], dtype=np.float32)
            A[t, k] = clean
            noisy = np.clip(clean + rng.normal(0.0, args.bc_noise, 2), -1, 1)
            _, r, d, _ = e.step(noisy)
            Rw[t, k], Dn[t, k] = r, float(d)
            if d:
                e.reset()
        if (t + 1) % max(1, per // 8) == 0:
            done_n = (t + 1) * n_env
            rate = done_n / (time.perf_counter() - t_bc)
            print(f"    {done_n:>8,} / {per * n_env:,}  "
                  f"({rate:,.0f}/s)", flush=True)

    # Discounted return-to-go, reset at each terminal, for the critic.
    rtg = np.zeros_like(Rw)
    acc = np.zeros(n_env, dtype=np.float32)
    for t in reversed(range(per)):
        acc = Rw[t] + args.gamma * acc * (1.0 - Dn[t])
        rtg[t] = acc

    flat_o = O.reshape(-1, obs_dim)
    flat_a = A.reshape(-1, 2)
    flat_r = rtg.reshape(-1)
    mean = flat_o.mean(0).astype(np.float64)
    var = flat_o.var(0).astype(np.float64) + 1e-6
    norm = ((flat_o - mean) / np.sqrt(var)).astype(np.float32)

    ten_o = torch.as_tensor(norm)
    ten_a = torch.as_tensor(flat_a)
    ten_r = torch.as_tensor(flat_r)
    bc_opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    n = len(ten_o)
    print(f"  behaviour cloning: {n:,} planner transitions", flush=True)
    for epoch in range(12):
        idx = torch.randperm(n)
        a_loss = v_loss = 0.0
        for start in range(0, n, 4096):
            sl = idx[start:start + 4096]
            pred_a = actor(ten_o[sl])
            pred_v = critic(ten_o[sl]).squeeze(-1)
            la = ((pred_a - ten_a[sl]) ** 2).mean()
            lv = ((pred_v - ten_r[sl]) ** 2).mean()
            bc_opt.zero_grad()
            (la + lv).backward()
            bc_opt.step()
            a_loss += la.item(); v_loss += lv.item()
        nb = math.ceil(n / 4096)
        print(f"    epoch {epoch + 1:2d}/12  action mse {a_loss / nb:.4f}  "
              f"value mse {v_loss / nb:9.1f}", flush=True)
    # A frozen snapshot of the cloned actor. PPO is anchored to this during
    # the first part of training so that its incremental steps -- each of
    # which locally pays to carry more speed into the opening sector -- cannot
    # quietly walk the policy away from the one part it cannot rediscover on
    # its own: braking for the chicane and steering through it. Twice now the
    # warm start got round the whole lap at iteration 15 and had forgotten the
    # chicane by iteration 60.
    import copy
    ref = copy.deepcopy(actor)
    for pp in ref.parameters():
        pp.requires_grad_(False)
    return mean, var, float(n), ref


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int,
                    default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    # The objective is distance covered in a fixed 150 s, so the value
    # function has to be able to see most of that. 0.999 is a 16 s / 780 m
    # horizon -- it reaches the braking zone for a corner but not the reward
    # for having got through it, which is the next 6 km of lap. Every earlier
    # run optimised the corner myopically for that reason: at this discount
    # "brake, make the chicane, keep going" and "stay flat, crash at the
    # chicane" have almost the same discounted return, because the 6 km that
    # separates them is beyond the horizon. 0.9998 is 83 s / 3.9 km, far
    # enough that finishing the lap is worth more than the straight before it.
    ap.add_argument("--gamma", type=float, default=0.9998)
    ap.add_argument("--lam", type=float, default=0.97)
    ap.add_argument("--clip", type=float, default=0.2)
    # No entropy bonus. PPO optimises the return of the *noisy* policy, so the
    # exploration noise is part of the objective: at std 0.45 -- half of full
    # steering lock, resampled every step -- driving fast goes off the road
    # often enough that the best mean policy is a slow one. The first run
    # learned exactly that, 60 km/h round Monza with the throttle at 0.09 and
    # full throttle on 4% of steps, and the entropy term was what held the
    # noise up there while the advantage tried to bring it down.
    ap.add_argument("--entropy", type=float, default=0.0)
    # Warm-started from the planner, so the noise starts low: 0.45 rad is half
    # of full steering lock resampled every step and would shake a competent
    # policy apart before it learned anything.
    ap.add_argument("--std-start", type=float, default=0.15)
    ap.add_argument("--std-end", type=float, default=0.05,
                    help="exploration noise is annealed between these, so the "
                         "policy is optimised against less and less of it")
    ap.add_argument("--bc-steps", type=int, default=250_000,
                    help="transitions of scripted-planner driving to imitate "
                         "before PPO starts. 0 skips it and PPO trains from "
                         "noise")
    ap.add_argument("--bc-noise", type=float, default=0.12,
                    help="action noise during BC collection, so the dataset "
                         "includes the planner recovering from small errors")
    ap.add_argument("--bc-anchor", type=float, default=0.6,
                    help="weight of the penalty keeping the policy near the "
                         "cloned one; the mean action may not drift from the "
                         "clone by more than the advantage is worth")
    ap.add_argument("--bc-anchor-frac", type=float, default=0.4,
                    help="fraction of training over which the anchor decays "
                         "linearly to zero, after which PPO is unconstrained")
    ap.add_argument("--start-at-line", type=float, default=1.0,
                    help="fraction of resets placed on the grid. 1.0 means "
                         "every lap is a real attempt from the line -- the "
                         "warm start already covers the whole circuit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    probe = RaceEnv(args.circuit)
    obs_dim, act_dim = probe.obs_dim, probe.act_dim

    def block(out):
        layers, last = [], obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers += [nn.Linear(last, out)]
        return nn.Sequential(*layers)

    actor, critic = block(act_dim), block(1)
    # A small last layer keeps the first actions near zero, which for this task
    # means "hold what you have" rather than a random lurch off the road.
    for net, gain in ((actor, 0.01), (critic, 1.0)):
        with torch.no_grad():
            net[-1].weight *= gain
            net[-1].bias.zero_()
    log_std = nn.Parameter(torch.full((act_dim,),
                                     math.log(args.std_start)))
    params = list(actor.parameters()) + list(critic.parameters()) + [log_std]
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)

    # Observations are on wildly different scales -- a curvature of 0.02 next
    # to a speed fraction of 0.9 -- so they are standardised by running
    # statistics. Without it the first layer spends its capacity on units.
    obs_mean = np.zeros(obs_dim, dtype=np.float64)
    obs_var = np.ones(obs_dim, dtype=np.float64)
    obs_n = 1e-4

    ref_actor = None
    if args.bc_steps > 0:
        obs_mean, obs_var, obs_n, ref_actor = _bc_pretrain(
            actor, critic, obs_dim, args, torch, nn)

    def export():
        w = {}
        i = 0
        for layer in actor:
            if isinstance(layer, nn.Linear):
                w[f"w{i}"] = layer.weight.detach().numpy().T.astype(np.float32)
                w[f"b{i}"] = layer.bias.detach().numpy().astype(np.float32)
                i += 1
        return w

    n_env = args.workers * args.envs_per_worker
    batch = args.rollout * n_env
    iters = max(1, args.steps // batch)
    print(f"{args.circuit}: PPO, {obs_dim}-dim observation, "
          f"{n_env} environments on {args.workers} workers, "
          f"{batch} steps per iteration, {iters} iterations "
          f"({iters * batch:,} steps)")

    config.RL_POLICY.mkdir(parents=True, exist_ok=True)
    out = config.RL_POLICY / f"{args.circuit}.npz"
    print(f"  parent pid {os.getpid()}", flush=True)
    pool = Pool(args.workers)

    # A stopped run used to leave its eight worker processes alive on Windows,
    # spinning on nothing and stealing a core each from whatever ran next.
    # Tear the pool down on the way out however that happens.
    import atexit
    import signal

    def _shutdown(*_):
        pool.terminate()
        pool.join()
        os._exit(0)

    atexit.register(lambda: (pool.terminate(), pool.join()))
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _shutdown)
        except (ValueError, OSError):
            pass
    best_dist = 0.0
    t0 = time.perf_counter()
    try:
        for it in range(iters):
            # The noise ceiling comes down on a schedule rather than being
            # left to the advantage to argue down against an entropy bonus.
            # The policy may sit below it; it may not sit above.
            frac = it / max(iters - 1, 1)
            at_line = args.start_at_line
            beta = (args.bc_anchor * max(0.0, 1.0 - frac / args.bc_anchor_frac)
                    if ref_actor is not None else 0.0)
            ceiling = math.log(args.std_start
                               + (args.std_end - args.std_start) * frac)
            with torch.no_grad():
                log_std.clamp_(max=ceiling)
            weights = export()
            std = np.sqrt(obs_var)
            jobs = [(args.circuit, args.seed * 131 + w, args.envs_per_worker,
                     args.rollout, weights, log_std.detach().numpy(),
                     obs_mean.astype(np.float32), std.astype(np.float32),
                     at_line)
                    for w in range(args.workers)]
            parts = pool.map(_rollout, jobs)

            obs = np.concatenate([p[0] for p in parts], axis=1)
            act = np.concatenate([p[1] for p in parts], axis=1)
            logp_old = np.concatenate([p[2] for p in parts], axis=1)
            rew = np.concatenate([p[3] for p in parts], axis=1)
            done = np.concatenate([p[4] for p in parts], axis=1)
            term = np.concatenate([p[5] for p in parts], axis=1)
            last = np.concatenate([p[6] for p in parts], axis=0)
            stats = [row for p in parts for row in p[7]]
            live = [d for p in parts for d in p[8]]
            reach = max([r[0] for r in stats] + live + [0.0])
            best_dist = max(best_dist, reach)

            # Running observation statistics, from the *unnormalised* samples
            # this batch would have produced. Kept in the parent so every
            # worker sees the same ones.
            flat = obs.reshape(-1, obs_dim) * std + obs_mean
            b_mean, b_var, b_n = flat.mean(0), flat.var(0), len(flat)
            delta = b_mean - obs_mean
            tot = obs_n + b_n
            obs_mean += delta * b_n / tot
            obs_var = (obs_var * obs_n + b_var * b_n
                       + delta ** 2 * obs_n * b_n / tot) / tot
            obs_n = tot

            with torch.no_grad():
                val = critic(torch.as_tensor(obs)).squeeze(-1).numpy()
                last_val = critic(torch.as_tensor(last)).squeeze(-1).numpy()

            adv = np.zeros_like(rew)
            run = np.zeros(n_env, dtype=np.float32)
            nxt = last_val
            for t in reversed(range(args.rollout)):
                mask = 1.0 - term[t]
                delta_t = rew[t] + args.gamma * nxt * mask - val[t]
                run = delta_t + args.gamma * args.lam * mask * run
                adv[t] = run
                nxt = val[t]
            ret = adv + val
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            b_obs = torch.as_tensor(obs.reshape(-1, obs_dim))
            b_act = torch.as_tensor(act.reshape(-1, act_dim))
            b_logp = torch.as_tensor(logp_old.reshape(-1))
            b_adv = torch.as_tensor(adv.reshape(-1))
            b_ret = torch.as_tensor(ret.reshape(-1))

            idx = np.arange(len(b_obs))
            for _ in range(args.epochs):
                np.random.shuffle(idx)
                for start in range(0, len(idx), args.minibatch):
                    sl = torch.as_tensor(idx[start:start + args.minibatch])
                    mu = actor(b_obs[sl])
                    std_t = log_std.exp()
                    lp = (-0.5 * (((b_act[sl] - mu) / std_t) ** 2).sum(-1)
                          - log_std.sum() - act_dim * 0.5 * math.log(2 * math.pi))
                    ratio = (lp - b_logp[sl]).exp()
                    a = b_adv[sl]
                    pl = -torch.min(ratio * a,
                                    ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    vl = ((critic(b_obs[sl]).squeeze(-1) - b_ret[sl]) ** 2).mean()
                    ent = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
                    if beta > 0.0:
                        with torch.no_grad():
                            ref_mu = ref_actor(b_obs[sl])
                        anchor = ((mu - ref_mu) ** 2).mean()
                    else:
                        anchor = torch.zeros(())
                    loss = pl + 0.5 * vl - args.entropy * ent + beta * anchor
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 0.5)
                    opt.step()
                    with torch.no_grad():
                        log_std.clamp_(max=ceiling)

            mins = (time.perf_counter() - t0) / 60.0
            if stats:
                arr = np.array([r[:6] for r in stats], dtype=float)
                crashed = sum(1 for r in stats if r[6] != "time")
                why = collections.Counter(r[6] or "-" for r in stats)
                worst = "  ".join(f"{k}:{v}" for k, v in why.most_common(3))
                # Distance per episode, not lap time: an episode is 45 s and
                # Monza is 5.8 km, so a lap cannot fit inside one. Mean speed
                # is the quantity being maximised anyway -- the reward is
                # distance covered in a fixed time.
                lap_times = [r[2] / r[1] for r in stats if r[1] > 0]
                print(f"  it {it + 1:4d}/{iters}  {(it + 1) * batch:>9,} steps  "
                      f"rew {rew.mean():+6.3f}  "
                      f"speed {arr[:, 5].mean() * 3.6:5.1f} km/h  "
                      f"dist {arr[:, 0].mean():5.0f} m  "
                      f"reach {reach:5.0f}/{best_dist:5.0f} m  "
                      f"off {100 * arr[:, 3].mean():4.1f}%  "
                      f"wall {100 * arr[:, 4].mean():4.1f}%  "
                      f"crashed {crashed:2d}/{len(stats):2d} [{worst}]  "
                      f"std {np.exp(log_std.detach().numpy()).mean():.3f}  "
                      f"anc {beta:.2f}  "
                      f"{mins:5.1f} min", flush=True)
            else:
                print(f"  it {it + 1:4d}/{iters}  {(it + 1) * batch:>9,} steps  "
                      f"rew {rew.mean():+6.3f}  "
                      f"reach {reach:5.0f}/{best_dist:5.0f} m  (none ended)  "
                      f"std {np.exp(log_std.detach().numpy()).mean():.3f}  "
                      f"{mins:5.1f} min", flush=True)

            np.savez(out, **export(), log_std=log_std.detach().numpy(),
                     obs_mean=obs_mean.astype(np.float32),
                     obs_std=np.sqrt(obs_var).astype(np.float32),
                     circuit=args.circuit)
    finally:
        pool.close()
        pool.join()
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
