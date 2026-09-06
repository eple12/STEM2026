"""IQN on the driving task -- a port of the Linesight (Trackmania RL) approach.

    python tools/train_iqn.py --circuit Monza --steps 6000000

Distributional value learning with a discrete action set, an off-policy replay
buffer, n-step returns, a target network, and epsilon-greedy exploration. The
point of moving off PPO: with a Gaussian policy the exploration noise is part
of the objective, so the best noisy policy is a cautious one. Argmax over
Q-values has no such coupling -- the greedy policy can sit on the limit while
epsilon-greedy explores separately.

Parallelism mirrors ``train_rl.py``: workers own their environments and a numpy
copy of the network, run whole rollout segments with epsilon-greedy actions,
and ship back n-step transitions. torch lives only in the parent.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import math
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import config
from game.rlenv import RaceEnv
from game.rlpolicy import (IQN_EMBED, IQN_K, N_ACTIONS, OBS_DIM, iqn_q,
                           normalise)

# --------------------------------------------------------------------------
# schedules  (step counts are decisions, not physics ticks)
# --------------------------------------------------------------------------
def _interp(schedule, x):
    xs = [p[0] for p in schedule]
    ys = [p[1] for p in schedule]
    return float(np.interp(x, xs, ys))


# The interesting learning is front-loaded. Once epsilon and gamma have
# finished annealing the run is pure exploitation and slow lap-time polish, so
# there is no reason to stretch the anneal over Linesight's millions of steps
# -- our physics is faster and the buffer smaller. Exploitation by ~700 k
# decisions (~40 min), then let it run as long as there is time.
# gamma ceiling is 0.9997, not 1.0. The 9-action run peaked at iteration ~1000
# and then the value function slowly diverged -- loss climbed 0.12 -> 1.2 over
# the rest of the run and the policy came off its best. gamma == 1.0 with
# bootstrapping has no contraction to damp that. 0.9997 is a 3300-step / 165 s
# horizon, longer than a 130 s episode, so it still sees the whole lap, but the
# Bellman operator stays a contraction.
GAMMA_SCHED = [(0, 0.999), (250_000, 0.999), (600_000, 0.9997)]
EPS_SCHED = [(0, 1.0), (25_000, 1.0), (200_000, 0.1), (750_000, 0.03)]
EPS_BOLTZ_SCHED = [(0, 0.15), (750_000, 0.02)]
LR_SCHED = [(0, 1e-3), (800_000, 3e-4), (4_000_000, 5e-5)]
TAU_BOLTZ = 0.03


# --------------------------------------------------------------------------
# workers
#
# One env-set per worker *process*, built once by the pool initializer and
# advanced in place on every rollout. Keying the envs by a per-job seed was a
# bug: pool.map does not pin a job to a worker, so a worker handed a new seed
# would build a fresh env-set and reset it, and no episode ever ran long
# enough to reach its 130 s truncation.
# --------------------------------------------------------------------------
_W = {}


def _init_worker(circuit, n_env, seed_base, start_at_line, n_step):
    seed = seed_base * 100003 + os.getpid()
    envs = [RaceEnv(circuit, seed=seed + i, start_at_line=start_at_line)
            for i in range(n_env)]
    # Stagger the first episode length per env so the 130 s truncations do not
    # all land in the same iteration forever (they never desync on their own --
    # every episode is exactly EPISODE_SECONDS).
    rng0 = np.random.default_rng(seed)
    for e in envs:
        e.reset()
        e.lap_time = float(rng0.uniform(0.0, 120.0))
    _W["envs"] = envs
    _W["obs"] = np.stack([e.observe() for e in envs])
    _W["hist"] = [collections.deque(maxlen=n_step) for _ in range(n_env)]
    _W["rng"] = np.random.default_rng(seed * 6151 + 1)
    _W["n_step"] = n_step


def _rollout(job):
    (steps, gamma, eps, eps_boltz, weights, mean, std) = job
    envs = _W["envs"]
    obs = _W["obs"]
    hist = _W["hist"]
    rng = _W["rng"]
    n_step = _W["n_step"]
    n_env = len(envs)
    tr_obs, tr_act, tr_ret, tr_next, tr_gamma, tr_done = [], [], [], [], [], []
    stats = []
    gpow = gamma ** np.arange(n_step)

    for _ in range(steps):
        q = iqn_q(weights, normalise(obs, mean, std))          # (n_env, A)
        greedy = q.argmax(1)
        roll = rng.random(n_env)
        rand_a = rng.integers(N_ACTIONS, size=n_env)
        boltz = (q + eps_boltz * TAU_BOLTZ
                 * rng.standard_normal(q.shape)).argmax(1)
        act = np.where(roll < eps, rand_a,
                       np.where(roll < eps + eps_boltz, boltz, greedy))

        for k, e in enumerate(envs):
            nobs, rew, done, info = e.step(int(act[k]))
            hist[k].append((obs[k].copy(), int(act[k]), float(rew)))
            if len(hist[k]) == n_step:
                o0, a0, _ = hist[k][0]
                ret = float(np.dot(gpow, [h[2] for h in hist[k]]))
                tr_obs.append(o0); tr_act.append(a0); tr_ret.append(ret)
                tr_next.append(nobs.copy())
                tr_gamma.append(gamma ** n_step)
                tr_done.append(0.0)                # truncation -> bootstrap
            if done:
                # flush the partial n-step tails as truncated transitions
                for j in range(1, len(hist[k])):
                    oj, aj, _ = hist[k][j]
                    tail = [h[2] for h in list(hist[k])[j:]]
                    ret = float(np.dot(gamma ** np.arange(len(tail)), tail))
                    tr_obs.append(oj); tr_act.append(aj); tr_ret.append(ret)
                    tr_next.append(nobs.copy())
                    tr_gamma.append(gamma ** len(tail))
                    tr_done.append(0.0)
                stats.append((e.progress, e.recoveries, e.laps,
                              e.speed_sum / max(e.steps * 3, 1),
                              e.off_steps / max(e.steps * 3, 1)))
                hist[k].clear()
                nobs = e.reset()
            obs[k] = nobs

    _W["obs"] = obs
    live = [e.progress for e in envs]
    return (np.asarray(tr_obs, np.float32), np.asarray(tr_act, np.int64),
            np.asarray(tr_ret, np.float32), np.asarray(tr_next, np.float32),
            np.asarray(tr_gamma, np.float32), np.asarray(tr_done, np.float32),
            stats, live)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--workers", type=int,
                    default=min(7, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--rollout", type=int, default=192,
                    help="decisions per env per iteration")
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--grad-steps", type=int, default=48,
                    help="learner updates per iteration")
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=25_000)
    ap.add_argument("--out-name", default=None,
                    help="basename under assets/policies for the checkpoint; "
                         "defaults to the circuit. Use a different name to "
                         "train without clobbering a policy the game is using")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="iterations between greedy evaluations that decide "
                         "the *_best checkpoint")
    ap.add_argument("--target-sync", type=int, default=1_500,
                    help="learner steps between hard target-net copies")
    ap.add_argument("--start-at-line", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - args.workers))

    A, E = IQN_EMBED, 256

    class IQN(nn.Module):
        def __init__(self):
            super().__init__()
            act = nn.LeakyReLU
            self.ff = nn.Sequential(nn.Linear(OBS_DIM, E), act(),
                                    nn.Linear(E, E), act())
            self.phi = nn.Sequential(nn.Linear(A, E), act())
            self.A = nn.Sequential(nn.Linear(E, E), act(), nn.Linear(E, N_ACTIONS))
            self.V = nn.Sequential(nn.Linear(E, E), act(), nn.Linear(E, 1))

        def forward(self, x, nq):
            b = x.shape[0]
            h = self.ff(x)                                  # (b, E)
            tau = torch.rand(b, nq, 1)
            ar = torch.arange(1, A + 1, dtype=torch.float32)
            emb = self.phi(torch.cos(ar * math.pi * tau))   # (b, nq, E)
            mixed = h.unsqueeze(1) * emb                    # (b, nq, E)
            a = self.A(mixed)                               # (b, nq, n_act)
            v = self.V(mixed)                               # (b, nq, 1)
            q = v + a - a.mean(-1, keepdim=True)
            return q, tau                                   # (b,nq,n_act),(b,nq,1)

    online, target = IQN(), IQN()
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=LR_SCHED[0][1], eps=1e-4)

    def export():
        s = online.state_dict()
        w = {
            "ff0.w": s["ff.0.weight"].numpy().T, "ff0.b": s["ff.0.bias"].numpy(),
            "ff1.w": s["ff.2.weight"].numpy().T, "ff1.b": s["ff.2.bias"].numpy(),
            "iqn.w": s["phi.0.weight"].numpy().T, "iqn.b": s["phi.0.bias"].numpy(),
            "A0.w": s["A.0.weight"].numpy().T, "A0.b": s["A.0.bias"].numpy(),
            "A1.w": s["A.2.weight"].numpy().T, "A1.b": s["A.2.bias"].numpy(),
            "V0.w": s["V.0.weight"].numpy().T, "V0.b": s["V.0.bias"].numpy(),
            "V1.w": s["V.2.weight"].numpy().T, "V1.b": s["V.2.bias"].numpy(),
        }
        return {k: v.astype(np.float32) for k, v in w.items()}

    # replay buffer (numpy ring)
    cap = args.buffer
    b_obs = np.zeros((cap, OBS_DIM), np.float32)
    b_act = np.zeros(cap, np.int64)
    b_ret = np.zeros(cap, np.float32)
    b_next = np.zeros((cap, OBS_DIM), np.float32)
    b_gam = np.zeros(cap, np.float32)
    b_done = np.zeros(cap, np.float32)
    b_fill = 0
    b_pos = 0

    obs_mean = np.zeros(OBS_DIM, np.float64)
    obs_var = np.ones(OBS_DIM, np.float64)
    obs_n = 1e-4

    KAPPA = 5e-3
    IQN_N = 8

    def iqn_loss(tgt, out, tau_out):
        # tgt, out: (b, N, 1)   tau_out: (b, N, 1)
        td = tgt[:, :, None, :] - out[:, None, :, :]        # (b, N, N, 1)
        hub = torch.where(td.abs() < KAPPA,
                          0.5 / KAPPA * td ** 2,
                          td.abs() - 0.5 * KAPPA)
        tau = tau_out[:, None, :, :]                        # (b,1,N,1)
        rho = torch.where(td < 0, 1 - tau, tau) * hub
        return rho.sum(2).mean(1).squeeze(-1)               # (b,)

    n_env = args.workers * args.envs_per_worker
    per_iter = args.rollout * n_env
    iters = max(1, args.steps // per_iter)
    print(f"{args.circuit}: IQN, {OBS_DIM}-dim obs, {N_ACTIONS} actions, "
          f"{n_env} envs / {args.workers} workers, {per_iter} decisions/iter, "
          f"{iters} iters ({iters * per_iter:,} decisions)", flush=True)
    print(f"  parent pid {os.getpid()}", flush=True)

    config.RL_POLICY.mkdir(parents=True, exist_ok=True)
    name = args.out_name or args.circuit
    out_path = config.RL_POLICY / f"{name}.npz"
    best_path = config.RL_POLICY / f"{name}_best.npz"

    # Greedy evaluation env, kept in the parent. The training log's speed is a
    # mean over episodes that each include a couple of exploration crashes and
    # their slow recoveries; this is the policy driven straight, which is what
    # actually gets deployed and what the *_best checkpoint tracks.
    eval_env = RaceEnv(args.circuit, seed=99_991, randomise_start=False)

    # Deterministic launch speeds, not RaceEnv's randomised one: v4 was picked
    # as *_best by an eval whose "grid start" still drew a random already-
    # rolling speed each call, so the number that chose the checkpoint was
    # itself noisy, and it never once tested the state the game actually
    # starts from. 0.0 is a dead-stop launch -- Vehicle.place's own standing
    # start -- through 0.90, already at speed; same four launches every time,
    # so a change in the eval number is a change in the policy, not the draw.
    EVAL_LAUNCHES = (0.0, 0.30, 0.60, 0.90)
    #: Off-track physics steps still counted as tier-0 "perfect". Two steps at
    #: the eval's 60 Hz is 0.03 s -- an edge kiss nobody sees -- and holding
    #: out for a literal zero made *_best ratchet up far too slowly.
    OFF_PERFECT_TOL = 2

    def greedy_eval(weights, mean, std):
        """(mean reach, tier, per-launch detail). Tier 0 is the checkpoint
        that never once puts a wheel on the grass across all four launches --
        off_steps == 0, not just recoveries == 0. A car can run the kerb's
        edge for a while without ever staying off long enough to trigger a
        recovery, and that is exactly the "very slightly off the white line"
        the eye catches that a recoveries-only "clean" flag was blind to.
        Tier 1 tolerates that but not an actual recovery; tier 2 is anything
        that hit a wall or ran off long enough to be teleported back."""
        total = 0.0
        tier = 0
        per_launch = []
        for sf in EVAL_LAUNCHES:
            o = eval_env.reset_grid(sf)
            while True:
                a = int(iqn_q(weights, normalise(o[None], mean, std))[0]
                        .argmax())
                o, _, d, _ = eval_env.step(a)
                if d:
                    break
            total += eval_env.progress
            if eval_env.recoveries > 0:
                tier = max(tier, 2)
            elif eval_env.off_steps > OFF_PERFECT_TOL:
                tier = max(tier, 1)
            per_launch.append((sf, eval_env.progress, eval_env.recoveries,
                              eval_env.off_steps))
        return total / len(EVAL_LAUNCHES), tier, per_launch

    best_eval = 0.0
    best_tier = 3          # worse than any real tier, so the first eval wins

    pool = Pool(args.workers, initializer=_init_worker,
                initargs=(args.circuit, args.envs_per_worker, args.seed,
                          args.start_at_line, args.n_step))

    def _shutdown(*_):
        pool.terminate(); pool.join(); os._exit(0)
    import atexit
    atexit.register(lambda: (pool.terminate(), pool.join()))
    for _s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_s, _shutdown)
        except (ValueError, OSError):
            pass

    learner_steps = 0
    best = 0.0
    recent = collections.deque(maxlen=40)   # last completed episodes
    t0 = time.perf_counter()
    try:
        for it in range(iters):
            seen = it * per_iter
            gamma = _interp(GAMMA_SCHED, seen)
            eps = _interp(EPS_SCHED, seen)
            eps_b = _interp(EPS_BOLTZ_SCHED, seen)
            lr = _interp(LR_SCHED, seen)
            for g in opt.param_groups:
                g["lr"] = lr

            std = np.sqrt(obs_var)
            w = export()
            job = (args.rollout, gamma, eps, eps_b, w,
                   obs_mean.astype(np.float32), std.astype(np.float32))
            parts = pool.map(_rollout, [job] * args.workers)

            o = np.concatenate([p[0] for p in parts])
            ac = np.concatenate([p[1] for p in parts])
            rt = np.concatenate([p[2] for p in parts])
            nx = np.concatenate([p[3] for p in parts])
            gm = np.concatenate([p[4] for p in parts])
            dn = np.concatenate([p[5] for p in parts])
            stats = [s for p in parts for s in p[6]]
            live = [d for p in parts for d in p[7]]

            m = len(o)
            # running obs stats from the raw (un-normalised) observations
            bm, bv = o.mean(0), o.var(0)
            delta = bm - obs_mean
            tot = obs_n + m
            obs_mean += delta * m / tot
            obs_var = (obs_var * obs_n + bv * m
                       + delta ** 2 * obs_n * m / tot) / tot
            obs_n = tot

            idx = (b_pos + np.arange(m)) % cap
            b_obs[idx] = o; b_act[idx] = ac; b_ret[idx] = rt
            b_next[idx] = nx; b_gam[idx] = gm; b_done[idx] = dn
            b_pos = (b_pos + m) % cap
            b_fill = min(cap, b_fill + m)

            reach = max([s[0] for s in stats] + live + [0.0])
            best = max(best, reach)

            loss_acc = 0.0
            if b_fill >= args.warmup:
                mean_t = torch.as_tensor(obs_mean, dtype=torch.float32)
                std_t = torch.as_tensor(std, dtype=torch.float32)
                for _ in range(args.grad_steps):
                    sel = np.random.randint(0, b_fill, args.batch)
                    so = (torch.as_tensor(b_obs[sel]) - mean_t) / std_t.clamp(min=1e-4)
                    sn = (torch.as_tensor(b_next[sel]) - mean_t) / std_t.clamp(min=1e-4)
                    sa = torch.as_tensor(b_act[sel])
                    sr = torch.as_tensor(b_ret[sel])
                    sg = torch.as_tensor(b_gam[sel])
                    sd = torch.as_tensor(b_done[sel])
                    with torch.no_grad():
                        # Double DQN: the *online* net picks the next action,
                        # the *target* net values it. Picking and valuing with
                        # the same net is what lets a lucky over-estimate feed
                        # itself; that is the overestimation that diverged the
                        # first run.
                        qn_online, _ = online(sn, IQN_N)
                        best_a = qn_online.mean(1).argmax(1)   # (b,)
                        qn_target, _ = target(sn, IQN_N)       # (b,N,n_act)
                        qn_sel = qn_target.gather(
                            2, best_a[:, None, None].expand(-1, IQN_N, 1)
                        ).squeeze(-1)                          # (b,N)
                        tgt = (sr[:, None]
                               + sg[:, None] * (1 - sd[:, None]) * qn_sel)
                        tgt = tgt.unsqueeze(-1)                # (b,N,1)
                    q, tau = online(so, IQN_N)                 # (b,N,n_act)
                    q_sel = q.gather(
                        2, sa[:, None, None].expand(-1, IQN_N, 1))  # (b,N,1)
                    loss = iqn_loss(tgt, q_sel, tau).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(online.parameters(), 30.0)
                    opt.step()
                    loss_acc += float(loss.detach())
                    learner_steps += 1
                    if learner_steps % args.target_sync == 0:
                        target.load_state_dict(online.state_dict())

            w_now = export()
            np.savez(out_path, **w_now,
                     obs_mean=obs_mean.astype(np.float32),
                     obs_std=std.astype(np.float32), circuit=args.circuit)

            eval_note = ""
            if b_fill >= args.warmup and (it + 1) % args.eval_every == 0:
                ev, tier, per_launch = greedy_eval(
                    w_now, obs_mean.astype(np.float32), std.astype(np.float32))
                # Tier first, distance only to break a tie within the same
                # tier: a checkpoint that never touches the grass beats one
                # that merely covers more ground while wandering onto it.
                better = tier < best_tier or (
                    tier == best_tier and ev > best_eval)
                TIER_TAG = {0: "PERFECT", 1: "off-track", 2: "wall/recover"}
                worst_off = max(os_ for _, _, _, os_ in per_launch)
                bad = ",".join(f"{sf:.2f}" for sf, _, rc, os_ in per_launch
                               if rc > 0 or os_ > OFF_PERFECT_TOL)
                tag = TIER_TAG[tier] + (f"@{bad}" if tier and bad
                                        else f"(off{worst_off})")
                if better:
                    best_eval, best_tier = ev, tier
                    np.savez(best_path, **w_now,
                             obs_mean=obs_mean.astype(np.float32),
                             obs_std=std.astype(np.float32),
                             circuit=args.circuit)
                    eval_note = f"  eval {ev:5.0f}m {tag} *BEST*"
                else:
                    eval_note = (f"  eval {ev:5.0f}m {tag} "
                                f"(best {best_eval:.0f}T{best_tier})")

            recent.extend(stats)
            mins = (time.perf_counter() - t0) / 60.0
            # Rolling mean over the last ~40 completed episodes, so every line
            # is informative even when this iteration's 2688 decisions happened
            # to contain no 130 s truncation. `n` is how many episodes the
            # averages are over; `+k` how many of them ended just now.
            if recent:
                r = np.asarray([s[:5] for s in recent], dtype=float)
                dist, rc, _, sp, of = r.mean(0)
                laps = r[:, 2].max()
                print(f"  it {it+1:4d}/{iters}  {seen+per_iter:>9,}  "
                      f"buf {b_fill:>7,}  reach {reach:5.0f}/{best:5.0f} m  "
                      f"dist {dist:5.0f}  spd {sp*3.6:5.1f}  rec {rc:4.1f}  "
                      f"off {of*100:4.1f}%  laps {laps:.0f}  "
                      f"eps {eps:.2f}  g {gamma:.4f}  "
                      f"loss {loss_acc/max(args.grad_steps,1):.3f}  "
                      f"n{len(recent):2d}+{len(stats):d}  {mins:5.1f}m"
                      f"{eval_note}", flush=True)
            else:
                print(f"  it {it+1:4d}/{iters}  {seen+per_iter:>9,}  "
                      f"buf {b_fill:>7,}  reach {reach:5.0f}/{best:5.0f} m  "
                      f"warming up  eps {eps:.2f}  {mins:5.1f}m", flush=True)
    finally:
        pool.terminate()
        pool.join()
    print(f"\nsaved {out_path}  (best greedy {best_eval:.0f} m -> "
          f"{best_path})", flush=True)


if __name__ == "__main__":
    main()
