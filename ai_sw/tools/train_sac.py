"""Soft Actor-Critic on the driving task -- continuous [steer, pedal] control.

    python tools/train_sac.py --circuit Monza --steps 8000000

After Fuchs et al., "Super-Human Performance in Gran Turismo Sport Using Deep
Reinforcement Learning" (the SAC predecessor of Sony's Sophy). The discrete
IQN driver beat the scripted planner but topped out at one or two wall
contacts a lap and could not modulate a bang-bang pedal. SAC gives a
continuous, entropy-regularised policy: exploration is a tunable temperature,
not baked into the objective, and the actor puts the pedal exactly where it
wants it.

What is taken from that paper: the twin-Q SAC with a squashed-Gaussian actor
and automatic temperature, the rangefinder observation (see
``rlpolicy.observe_sac``), the ``progress - c_w * v^2`` wall term (in
``rlenv``), 5-step returns "to stabilise training", and a modest replay ratio
(they did far fewer gradient steps than environment steps).

Parallelism and the best-checkpoint / worker-pinning machinery mirror
``train_iqn.py``.
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
from game.rlpolicy import (OBS_DIM_SAC, SAC_ACT_DIM, SAC_HIDDEN, normalise,
                           sac_forward)


def _interp(sched, x):
    return float(np.interp(x, [p[0] for p in sched], [p[1] for p in sched]))


# GT-Sport SAC used gamma ~0.98 at 10 Hz (a ~5 s horizon) and leaned on the
# wall term for the "brake for the corner" signal. At our 20 Hz decision rate
# 0.99 is the same ~5 s; it rises to 0.997 (~16 s) once the critic is stable.
# Never higher -- the discrete IQN runs all diverged with gamma pinned at the
# ceiling under a near-1 discount, and SAC's bootstrap has the same failure.
GAMMA_SCHED = [(0, 0.995), (400_000, 0.995), (1_200_000, 0.997)]
LR_SCHED = [(0, 3e-4), (5_000_000, 1e-4)]
#: Floor on the Gaussian action noise the workers inject, on top of the
#: policy's own std -- keeps the buffer fresh through the collapse-prone
#: early phase.
EXPL_SCHED = [(0, 0.15), (100_000, 0.10), (1_000_000, 0.05)]

#: Temperature. Fixed at 0.2 through the phase where the first two SAC attempts
#: collapsed (do-nothing, and auto-alpha eating the entropy before the policy
#: found the throttle), then auto-tuned toward a target entropy with a floor so
#: it cannot die again.
ALPHA_FIXED = 0.20
ALPHA_AUTO_AFTER = 1_500_000       # decisions
ALPHA_TARGET_ENTROPY = -1.0        # per the 2-D action; less negative = more explore
ALPHA_FLOOR = 0.03
ALPHA_LR = 1e-4


# --------------------------------------------------------------------------
_W = {}


def _init_worker(circuit, n_env, seed_base, start_at_line, n_step):
    seed = seed_base * 100003 + os.getpid()
    envs = [RaceEnv(circuit, seed=seed + i, start_at_line=start_at_line,
                    continuous=True) for i in range(n_env)]
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
    steps, gamma, expl, weights, mean, std = job
    envs = _W["envs"]
    obs = _W["obs"]
    hist = _W["hist"]
    rng = _W["rng"]
    n_step = _W["n_step"]
    n_env = len(envs)
    o_b, a_b, r_b, no_b, g_b, d_b = [], [], [], [], [], []
    stats = []

    for _ in range(steps):
        m, ls = sac_forward(weights, normalise(obs, mean, std))
        u = m + np.maximum(np.exp(ls), expl) * rng.standard_normal(m.shape)
        act = np.tanh(u)
        for k, e in enumerate(envs):
            nobs, rew, done, info = e.step(act[k])
            # A crash ends the episode with a true terminal (no bootstrap off
            # the post-crash state); the 130 s time limit is a truncation
            # (bootstrap Q(s') as usual).
            term = 1.0 if info.get("terminal") else 0.0
            hist[k].append((obs[k].copy(), act[k].copy(), float(rew)))
            if len(hist[k]) == n_step:
                o0, a0, _ = hist[k][0]
                ret = sum(gamma ** j * hist[k][j][2] for j in range(n_step))
                o_b.append(o0); a_b.append(a0); r_b.append(ret)
                no_b.append(nobs.copy()); g_b.append(gamma ** n_step)
                d_b.append(term)
            if done:
                for j in range(1, len(hist[k])):
                    oj, aj, _ = hist[k][j]
                    tail = [hist[k][t][2] for t in range(j, len(hist[k]))]
                    ret = sum(gamma ** t * tail[t] for t in range(len(tail)))
                    o_b.append(oj); a_b.append(aj); r_b.append(ret)
                    no_b.append(nobs.copy()); g_b.append(gamma ** len(tail))
                    d_b.append(term)
                # col 2 is now the crash rate (1.0 = ended on a wall/stall/
                # wrong-way, 0.0 = survived to the time limit), not a recovery
                # count -- the continuous run has no teleport-recover.
                stats.append((e.progress, term, e.laps,
                              e.speed_sum / max(e.steps * 3, 1),
                              e.off_steps / max(e.steps * 3, 1)))
                hist[k].clear()
                nobs = e.reset()
            obs[k] = nobs
    _W["obs"] = obs
    return (np.asarray(o_b, np.float32), np.asarray(a_b, np.float32),
            np.asarray(r_b, np.float32), np.asarray(no_b, np.float32),
            np.asarray(g_b, np.float32), np.asarray(d_b, np.float32),
            stats, [e.progress for e in envs])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--workers", type=int,
                    default=min(7, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--envs-per-worker", type=int, default=2)
    ap.add_argument("--rollout", type=int, default=192)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--grad-steps", type=int, default=32)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--warmup", type=int, default=25_000)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--start-at-line", type=float, default=0.25)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--init-from", default=None,
                    help="warm-start the actor + obs stats from this policy npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - args.workers))

    O, A, H = OBS_DIM_SAC, SAC_ACT_DIM, SAC_HIDDEN
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def mlp(sizes):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = mlp([O, H, H])
            self.mu = nn.Linear(H, A)
            self.ls = nn.Linear(H, A)

        def forward(self, x):
            h = torch.relu(self.body(x))
            mu = self.mu(h)
            ls = self.ls(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
            return mu, ls

        def sample(self, x):
            mu, ls = self.forward(x)
            std = ls.exp()
            n = torch.randn_like(mu)
            u = mu + std * n
            a = torch.tanh(u)
            # log prob with tanh correction
            logp = (-0.5 * (n ** 2) - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
            logp -= torch.log(1 - a ** 2 + 1e-6).sum(-1)
            return a, logp

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = mlp([O + A, H, H, 1])

        def forward(self, x, a):
            return self.net(torch.cat([x, a], -1)).squeeze(-1)

    actor = Actor()
    # The first SAC run collapsed to the do-nothing action: mean pedal ~0,
    # coast from the start speed to a stop, 880 m in 130 s. Starting the actor
    # *on the throttle* means it begins by driving and SAC refines a real
    # policy, instead of having to discover from a standstill that throttle is
    # good while the entropy term is already shrinking.
    with torch.no_grad():
        # Start the actor *on the throttle*: mean pedal ~1.0, near-zero steer,
        # tiny weights so it begins by driving straight and SAC refines from a
        # real policy instead of discovering throttle from a standstill while
        # the entropy term is already shrinking.
        actor.mu.bias[:] = torch.tensor([0.0, 1.3])
        actor.mu.weight.mul_(0.1)
    q1, q2 = Q(), Q()
    q1t, q2t = Q(), Q()
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    for p in list(q1t.parameters()) + list(q2t.parameters()):
        p.requires_grad_(False)

    pi_opt = torch.optim.Adam(actor.parameters(), lr=LR_SCHED[0][1])
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                             lr=LR_SCHED[0][1])
    # Temperature: fixed ALPHA_FIXED until ALPHA_AUTO_AFTER decisions, then
    # gradient-tuned toward ALPHA_TARGET_ENTROPY but clamped at ALPHA_FLOOR.
    log_alpha = torch.tensor(math.log(ALPHA_FIXED), requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=ALPHA_LR)

    def export():
        s = actor.state_dict()
        w = {
            "pi0.w": s["body.0.weight"].numpy().T, "pi0.b": s["body.0.bias"].numpy(),
            "pi1.w": s["body.2.weight"].numpy().T, "pi1.b": s["body.2.bias"].numpy(),
            "pi_mu.w": s["mu.weight"].numpy().T, "pi_mu.b": s["mu.bias"].numpy(),
            "pi_ls.w": s["ls.weight"].numpy().T, "pi_ls.b": s["ls.bias"].numpy(),
        }
        return {k: v.astype(np.float32) for k, v in w.items()}

    cap = args.buffer
    b_o = np.zeros((cap, O), np.float32)
    b_a = np.zeros((cap, A), np.float32)
    b_r = np.zeros(cap, np.float32)
    b_no = np.zeros((cap, O), np.float32)
    b_g = np.zeros(cap, np.float32)
    b_d = np.zeros(cap, np.float32)
    b_fill = b_pos = 0

    obs_mean = np.zeros(O, np.float64)
    obs_var = np.ones(O, np.float64)
    obs_n = 1e-4

    if args.init_from:
        ip = Path(args.init_from)
        if not ip.exists():
            ip = config.RL_POLICY / f"{args.init_from}.npz"
        d0 = np.load(ip, allow_pickle=False)
        amap = {"body.0": "pi0", "body.2": "pi1", "mu": "pi_mu", "ls": "pi_ls"}
        asd = actor.state_dict()
        for tk, nk in amap.items():
            asd[f"{tk}.weight"] = torch.as_tensor(
                np.ascontiguousarray(d0[f"{nk}.w"].T), dtype=torch.float32)
            asd[f"{tk}.bias"] = torch.as_tensor(
                np.ascontiguousarray(d0[f"{nk}.b"]), dtype=torch.float32)
        actor.load_state_dict(asd)
        if "obs_mean" in d0.files and len(d0["obs_mean"]) == O:
            obs_mean[:] = d0["obs_mean"]
            obs_var[:] = np.square(d0["obs_std"].astype(np.float64))
            obs_n = 1e5
        print(f"warm-started actor from {ip.name}", flush=True)

    n_env = args.workers * args.envs_per_worker
    per_iter = args.rollout * n_env
    iters = max(1, args.steps // per_iter)
    name = args.out_name or args.circuit
    out_path = config.RL_POLICY / f"{name}.npz"
    best_path = config.RL_POLICY / f"{name}_best.npz"
    print(f"{args.circuit}: SAC, {O}-dim obs, {A} continuous actions, "
          f"{n_env} envs / {args.workers} workers, {per_iter} decisions/iter, "
          f"{iters} iters ({iters * per_iter:,} decisions)", flush=True)
    print(f"  parent pid {os.getpid()}", flush=True)
    config.RL_POLICY.mkdir(parents=True, exist_ok=True)

    eval_env = RaceEnv(args.circuit, seed=99_991, randomise_start=False,
                       continuous=True)

    # Deterministic grid launches, mean action, no noise -- exactly what the
    # game deploys. Tier 0 (PERFECT) never puts a wheel off across all seven;
    # tier 1 tolerates an off-track excursion but no recovery; tier 2 hit a
    # wall or had to be teleported back. The *_best checkpoint tracks tier
    # first, distance only to break a tie -- ported from train_iqn.
    EVAL_LAUNCHES = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
    OFF_PERFECT_TOL = 2

    def greedy_eval(weights, mean, std):
        total = 0.0
        tier = 0
        per_launch = []
        for sf in EVAL_LAUNCHES:
            o = eval_env.reset_grid(sf)
            info = {"reason": "time"}
            while True:
                m, _ = sac_forward(weights, normalise(o[None], mean, std))
                o, _, d, info = eval_env.step(np.tanh(m[0]))
                if d:
                    break
            total += eval_env.progress
            reason = info.get("reason", "")
            # No teleport-recover in the continuous run, so the tier comes from
            # how the episode *ended*: a wall / stall / wrong-way is tier 2, an
            # off-track excursion is tier 1, and surviving the full time limit
            # is tier 0 -- unless it grazed the white line on the way (off_steps
            # over the tolerance), which is also tier 1.
            if reason in ("wall", "stalled", "wrong way"):
                tier = max(tier, 2)
            elif reason == "off track" or eval_env.off_steps > OFF_PERFECT_TOL:
                tier = max(tier, 1)
            per_launch.append((sf, eval_env.progress, reason,
                               eval_env.off_steps))
        return total / len(EVAL_LAUNCHES), tier, per_launch

    best_eval = 0.0
    best_tier = 3

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

    recent = collections.deque(maxlen=40)
    t0 = time.perf_counter()
    try:
        for it in range(iters):
            seen = it * per_iter
            gamma = _interp(GAMMA_SCHED, seen)
            lr = _interp(LR_SCHED, seen)
            expl = _interp(EXPL_SCHED, seen)
            for opt in (pi_opt, q_opt):
                for gp in opt.param_groups:
                    gp["lr"] = lr

            std = np.sqrt(obs_var)
            job = (args.rollout, gamma, expl, export(),
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
            bm, bv = o.mean(0), o.var(0)
            delta = bm - obs_mean
            tot = obs_n + m
            obs_mean += delta * m / tot
            obs_var = (obs_var * obs_n + bv * m
                       + delta ** 2 * obs_n * m / tot) / tot
            obs_n = tot

            idx = (b_pos + np.arange(m)) % cap
            b_o[idx] = o; b_a[idx] = ac; b_r[idx] = rt
            b_no[idx] = nx; b_g[idx] = gm; b_d[idx] = dn
            b_pos = (b_pos + m) % cap
            b_fill = min(cap, b_fill + m)

            reach = max([s[0] for s in stats] + live + [0.0])

            auto_alpha = seen >= ALPHA_AUTO_AFTER
            q_loss_acc = pi_loss_acc = alpha_acc = 0.0
            if b_fill >= args.warmup:
                mt = torch.as_tensor(obs_mean, dtype=torch.float32)
                st = torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-4)
                for _ in range(args.grad_steps):
                    sel = np.random.randint(0, b_fill, args.batch)
                    so = (torch.as_tensor(b_o[sel]) - mt) / st
                    sn = (torch.as_tensor(b_no[sel]) - mt) / st
                    sa = torch.as_tensor(b_a[sel])
                    sr = torch.as_tensor(b_r[sel])
                    sg = torch.as_tensor(b_g[sel])
                    sd = torch.as_tensor(b_d[sel])
                    if auto_alpha:
                        alpha = float(log_alpha.detach().exp().clamp(
                            min=ALPHA_FLOOR))
                    else:
                        alpha = ALPHA_FIXED

                    with torch.no_grad():
                        na, nlogp = actor.sample(sn)
                        tq = torch.min(q1t(sn, na), q2t(sn, na)) - alpha * nlogp
                        y = sr + sg * (1 - sd) * tq

                    q1l = ((q1(so, sa) - y) ** 2).mean()
                    q2l = ((q2(so, sa) - y) ** 2).mean()
                    q_opt.zero_grad(set_to_none=True)
                    (q1l + q2l).backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(q1.parameters()) + list(q2.parameters()), 10.0)
                    q_opt.step()

                    a_new, logp = actor.sample(so)
                    qpi = torch.min(q1(so, a_new), q2(so, a_new))
                    pil = (alpha * logp - qpi).mean()
                    pi_opt.zero_grad(set_to_none=True)
                    pil.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    pi_opt.step()

                    if auto_alpha:
                        alpha_loss = -(log_alpha
                                       * (logp.detach() + ALPHA_TARGET_ENTROPY)
                                       ).mean()
                        alpha_opt.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_opt.step()

                    with torch.no_grad():
                        for p, pt in zip(q1.parameters(), q1t.parameters()):
                            pt.mul_(1 - args.tau).add_(args.tau * p)
                        for p, pt in zip(q2.parameters(), q2t.parameters()):
                            pt.mul_(1 - args.tau).add_(args.tau * p)

                    q_loss_acc += float(q1l.detach())
                    pi_loss_acc += float(pil.detach())
                    alpha_acc += alpha

            w_now = export()
            np.savez(out_path, **w_now, obs_mean=obs_mean.astype(np.float32),
                     obs_std=std.astype(np.float32), circuit=args.circuit)

            eval_note = ""
            if b_fill >= args.warmup and (it + 1) % args.eval_every == 0:
                ev, tier, per_launch = greedy_eval(
                    w_now, obs_mean.astype(np.float32),
                    std.astype(np.float32))
                better = tier < best_tier or (tier == best_tier
                                              and ev > best_eval)
                TIER_TAG = {0: "PERFECT", 1: "off-track", 2: "wall/stall"}
                bad = ",".join(
                    f"{sf:.2f}({rs or 'off'})" for sf, _, rs, os_ in per_launch
                    if (rs and rs != "time") or os_ > OFF_PERFECT_TOL)
                tag = TIER_TAG[tier] + (f"@{bad}" if tier and bad else "")
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
            gs = max(args.grad_steps, 1)
            if recent:
                r = np.asarray([s[:5] for s in recent], dtype=float)
                dist, cr, _, sp, of = r.mean(0)
                print(f"  it {it+1:4d}/{iters}  {seen+per_iter:>9,}  "
                      f"buf {b_fill:>7,}  reach {reach:5.0f} m  "
                      f"dist {dist:5.0f}  spd {sp*3.6:5.1f}  crash {cr*100:3.0f}%  "
                      f"off {of*100:4.1f}%  g {gamma:.4f}  a {alpha_acc/gs:.3f}  "
                      f"qL {q_loss_acc/gs:6.2f}  piL {pi_loss_acc/gs:+6.2f}  "
                      f"n{len(recent):2d}+{len(stats):d}  {mins:5.1f}m"
                      f"{eval_note}", flush=True)
            else:
                print(f"  it {it+1:4d}/{iters}  {seen+per_iter:>9,}  "
                      f"buf {b_fill:>7,}  reach {reach:5.0f} m  warming up  "
                      f"{mins:5.1f}m", flush=True)
    finally:
        pool.terminate()
        pool.join()
    print(f"\nsaved {out_path}  (best greedy {best_eval:.0f} m)", flush=True)


if __name__ == "__main__":
    main()
