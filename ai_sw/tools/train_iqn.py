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


# v18: the Linesight schedules verbatim (Linesight-RL/linesight
# config_files/config.py). Every prior run compressed these to <1 M decisions
# and capped gamma below 1.0 to stop the value function diverging; Linesight
# stretches the anneal over 3 M and takes gamma all the way to 1.0, paired with
# a *soft* target update (tau 0.02) and plain (non-double) IQN. This run keeps
# that exactly and only adds our off-track fee + PERFECT eval on top.
GAMMA_SCHED = [(0, 0.999), (1_500_000, 0.999), (2_500_000, 1.0)]
EPS_SCHED = [(0, 1.0), (50_000, 1.0), (300_000, 0.1), (3_000_000, 0.03)]
# Warm-start variant kept for --init-from (unused by v18's from-scratch run).
EPS_SCHED_WARM = [(0, 0.20), (150_000, 0.10), (700_000, 0.04)]
EPS_BOLTZ_SCHED = [(0, 0.15), (3_000_000, 0.03)]
LR_SCHED = [(0, 1e-3), (3_000_000, 5e-5), (12_000_000, 5e-5), (15_000_000, 1e-5)]
TAU_BOLTZ = 0.01                          # Linesight epsilon-Boltzmann tau
SOFT_TARGET_TAU = 0.02                    # Linesight soft_update_tau

# Periodic exploration revival: a sawtooth on top of EPS_SCHED -- every
# EPS_REVIVE_PERIOD decisions eps jumps to EPS_REVIVE_PEAK and re-anneals over
# EPS_REVIVE_DECAY. DISABLED (peak 0.0): the v12 from-scratch run showed the
# 600 k spike landing mid-learning, before the policy had a clean lap to
# escape from, and it only slowed the initial climb. If a plateau reappears
# after PERFECT, re-enable with the period set past the PERFECT milestone
# (~1.5 M decisions), not before it.
EPS_REVIVE_PERIOD = 600_000
EPS_REVIVE_PEAK = 0.0
EPS_REVIVE_DECAY = 150_000

# Curriculum on the per-second off-track fee (config.RL_OFF_TRACK_COST). Cheap
# while the policy is still learning to string a lap together, then tightened
# so the last white-line clips get sanded off. Lives in config so RaceEnv and
# this schedule cannot drift apart.
OFF_COST_SCHED = [tuple(p) for p in config.RL_OFF_TRACK_COST_RAMP]
LAP_W_SCHED = [tuple(p) for p in config.RL_LAP_W_RAMP]
# Keyed to decisions since episode widening, not decisions since training
# start -- only train_iqn_gpu.py (the GPU trainer) tracks that event and
# applies this; see RL_POST_WIDEN_RAMP's own long comment in config.py.
POST_WIDEN_SCHED = [tuple(p) for p in config.RL_POST_WIDEN_RAMP]

# v24: curriculum on the dense-speed-reward target pace. The policy first
# converges clean against an attainable profile (pace mult 1.0 -- this is
# essentially the v19 setup, which reached PERFECT by it ~300), then the target
# ramps up so it is continuously pulled toward a faster line rather than
# settling at one comfortable speed (v19 stalled at 181 km/h for 1100 iters,
# gamma=1.0 included). By ~2.5 M decisions the target sits ~28 % above the
# baseline profile -- past what is cleanly holdable everywhere, so the bell
# keeps a live "faster here" gradient without ever demanding a specific
# unclean speed.
PACE_SCHED = [(0, 1.0), (15_000_000, 1.0)]   # v25: retired. The v24 ramp
# inflated corner speed targets past what is cleanly reachable, which switched
# off the bell's overspeed penalty in corners and eroded the clean policy after
# ~it 1300. v25 removes the speed plateau with a linear reward term instead
# (config.RL_RL_SPEED_LIN), not by inflating the target. Kept flat at 1.0 so
# the pace_mult plumbing stays harmless rather than being ripped out.


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


def _init_worker(circuit, n_env, seed_base, start_at_line, n_step,
                 focus_window=None, adaptive_reset=False):
    seed = seed_base * 100003 + os.getpid()
    envs = [RaceEnv(circuit, seed=seed + i, start_at_line=start_at_line,
                    focus_window=focus_window, adaptive_reset=adaptive_reset)
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
    (steps, gamma, eps, eps_boltz, weights, mean, std,
     off_scale, pace_mult, lap_w_mult) = job
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
            nobs, rew, done, info = e.step(int(act[k]), off_scale, pace_mult,
                                           lap_w_mult)
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
    ap.add_argument("--n-step", type=int, default=3,
                    help="Linesight n_steps = 3")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--grad-steps", type=int, default=48,
                    help="learner updates per iteration")
    ap.add_argument("--buffer", type=int, default=150_000,
                    help="Linesight memory_size final = 150k")
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--out-name", default=None,
                    help="basename under assets/policies for the checkpoint; "
                         "defaults to the circuit. Use a different name to "
                         "train without clobbering a policy the game is using")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="iterations between greedy evaluations that decide "
                         "the *_best checkpoint")
    ap.add_argument("--target-sync", type=int, default=1,
                    help="learner steps between SOFT target updates "
                         "(Polyak, SOFT_TARGET_TAU). Linesight soft-updates "
                         "roughly once per rollout; 1 = every learner step, "
                         "which with tau 0.02 is a slower effective track.")
    ap.add_argument("--ddqn", action="store_true",
                    help="use Double-DQN action selection. Linesight is plain "
                         "IQN (use_ddqn=False), which is the default here.")
    ap.add_argument("--start-at-line", type=float, default=0.15)
    ap.add_argument("--focus-window", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="see train_iqn_gpu.py's --focus-window -- same "
                         "fraction-of-lap scattered-reset restriction, "
                         "mirrored here for CPU-side sector-focused runs.")
    ap.add_argument("--adaptive-reset", action="store_true",
                    help="see train_iqn_gpu.py's --adaptive-reset -- same "
                         "difficulty-weighted scattered-reset mechanism, "
                         "mirrored here (each worker process tracks its own "
                         "bucket EMA independently from its own envs' "
                         "telemetry -- not synchronised across workers, an "
                         "acceptable approximation since this trainer isn't "
                         "the one actually used for the timed Kaggle runs).")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights + obs stats from this checkpoint "
                         "(basename under assets/policies, or a path to an .npz)")
    ap.add_argument("--resume", action="store_true",
                    help="if <out-name>.npz exists, continue that run: restore "
                         "weights, obs stats, the decisions-seen counter (so "
                         "gamma/eps/lr schedules pick up where they left off) "
                         "and the best-eval trackers. For an auto-restart loop "
                         "on a machine that keeps killing the process.")
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

    # Warm start: load weights and obs statistics from an existing checkpoint.
    # Used to carry v11's robust PERFECT policy into v12 rather than relearning
    # a clean lap from scratch -- v12 only has to find speed. The obs stats are
    # seeded as if from 1e5 samples so the running estimate does not lurch on
    # the first batch while the raceline reference shifts the distribution a
    # little.
    if args.init_from:
        ip = Path(args.init_from)
        if not ip.exists():
            ip = config.RL_POLICY / f"{args.init_from}.npz"
        d0 = np.load(ip, allow_pickle=False)
        name_map = {"ff.0": "ff0", "ff.2": "ff1", "phi.0": "iqn",
                    "A.0": "A0", "A.2": "A1", "V.0": "V0", "V.2": "V1"}
        sd = online.state_dict()
        for tk, nk in name_map.items():
            sd[f"{tk}.weight"] = torch.as_tensor(
                np.ascontiguousarray(d0[f"{nk}.w"].T), dtype=torch.float32)
            sd[f"{tk}.bias"] = torch.as_tensor(
                np.ascontiguousarray(d0[f"{nk}.b"]), dtype=torch.float32)
        online.load_state_dict(sd)
        target.load_state_dict(online.state_dict())
        if "obs_mean" in d0.files:
            obs_mean[:] = d0["obs_mean"]
            obs_var[:] = np.square(d0["obs_std"].astype(np.float64))
            obs_n = 1e5
        print(f"warm-started from {ip.name}", flush=True)

    # --resume: continue this run's own checkpoint. Unlike --init-from this
    # restores the schedule position (resume_seen) and the best trackers, and
    # uses the normal (not warm) schedules. The replay buffer is not saved, so
    # there is a short re-warmup after each restart.
    _rname = args.out_name or args.circuit
    out_path = config.RL_POLICY / f"{_rname}.npz"
    best_path = config.RL_POLICY / f"{_rname}_best.npz"
    resume_seen = 0
    resume_best = (0.0, 3, 0.0)
    if args.resume and out_path.exists():
        dR = np.load(out_path, allow_pickle=False)
        sd = online.state_dict()
        _nm = {"ff.0": "ff0", "ff.2": "ff1", "phi.0": "iqn",
               "A.0": "A0", "A.2": "A1", "V.0": "V0", "V.2": "V1"}
        for tk, nk in _nm.items():
            sd[f"{tk}.weight"] = torch.as_tensor(
                np.ascontiguousarray(dR[f"{nk}.w"].T), dtype=torch.float32)
            sd[f"{tk}.bias"] = torch.as_tensor(
                np.ascontiguousarray(dR[f"{nk}.b"]), dtype=torch.float32)
        online.load_state_dict(sd)
        target.load_state_dict(online.state_dict())
        if "obs_mean" in dR.files:
            obs_mean[:] = dR["obs_mean"]
            obs_var[:] = np.square(dR["obs_std"].astype(np.float64))
            obs_n = 1e6
        resume_seen = int(dR["seen"]) if "seen" in dR.files else 0
        # The authoritative best trackers live in best_path (only overwritten
        # on a genuine improvement); fall back to out_path.
        _bsrc = dR
        if best_path.exists():
            _bd = np.load(best_path, allow_pickle=False)
            if "best_eval" in _bd.files:
                _bsrc = _bd
        if "best_eval" in _bsrc.files:
            resume_best = (float(_bsrc["best_eval"]), int(_bsrc["best_tier"]),
                          float(_bsrc["best_lap"]) if "best_lap" in _bsrc.files else 0.0)
        print(f"resumed {out_path.name} at {resume_seen:,} decisions "
              f"(best {resume_best[0]:.0f} T{resume_best[1]})", flush=True)
    elif args.resume:
        print(f"--resume: no {out_path.name} yet, starting fresh", flush=True)

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
    # start -- through 0.90, already at speed; same launches every time, so a
    # change in the eval number is a change in the policy, not the draw. Seven
    # not four: v10 hit PERFECT on a 4-launch eval and lost it two evals later,
    # so a single 4/4 was too thin a reed to pick *_best from. 7/7 clean is a
    # policy that actually has margin.
    EVAL_LAUNCHES = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
    #: Off-track physics steps still counted as tier-0 "perfect". See
    #: config.RL_OFF_PERFECT_TOL for the current value and why.
    OFF_PERFECT_TOL = config.RL_OFF_PERFECT_TOL

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
        laps = []
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
            if eval_env.best_lap_time > 0.0:
                laps.append(eval_env.best_lap_time)
            per_launch.append((sf, eval_env.progress, eval_env.recoveries,
                              eval_env.off_steps))
        best_lap = min(laps) if laps else 0.0
        return total / len(EVAL_LAUNCHES), tier, per_launch, best_lap

    best_eval, best_tier, best_lap_saved = resume_best  # (0.0, 3, 0.0) unless --resume restored it
    best_it = resume_seen // per_iter    # iteration the current best was set at

    pool = Pool(args.workers, initializer=_init_worker,
                initargs=(args.circuit, args.envs_per_worker, args.seed,
                          args.start_at_line, args.n_step,
                          tuple(args.focus_window) if args.focus_window
                          else None, args.adaptive_reset))

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
    eps_sched = EPS_SCHED_WARM if args.init_from else EPS_SCHED
    lr_sched = ([(0, 3e-4), (2_000_000, 1e-4), (5_000_000, 5e-5)]
                if args.init_from else LR_SCHED)
    t0 = time.perf_counter()
    try:
        for it in range(resume_seen // per_iter, iters):
            seen = it * per_iter
            gamma = _interp(GAMMA_SCHED, seen)
            eps = _interp(eps_sched, seen)
            if seen >= EPS_REVIVE_PERIOD:
                phase = seen % EPS_REVIVE_PERIOD
                if phase < EPS_REVIVE_DECAY:
                    eps = max(eps, EPS_REVIVE_PEAK
                              * (1.0 - phase / EPS_REVIVE_DECAY))
            eps_b = _interp(EPS_BOLTZ_SCHED, seen)
            lr = _interp(lr_sched, seen)
            off_scale = _interp(OFF_COST_SCHED, seen)
            pace_mult = _interp(PACE_SCHED, seen)
            lap_w_mult = _interp(LAP_W_SCHED, seen)
            for g in opt.param_groups:
                g["lr"] = lr

            std = np.sqrt(obs_var)
            w = export()
            job = (args.rollout, gamma, eps, eps_b, w,
                   obs_mean.astype(np.float32), std.astype(np.float32),
                   float(off_scale), float(pace_mult), float(lap_w_mult))
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
                        qn_target, _ = target(sn, IQN_N)       # (b,N,n_act)
                        if args.ddqn:
                            # Double DQN: online picks the next action, target
                            # values it.
                            qn_online, _ = online(sn, IQN_N)
                            best_a = qn_online.mean(1).argmax(1)
                        else:
                            # Plain IQN (Linesight use_ddqn=False): target both
                            # picks and values.
                            best_a = qn_target.mean(1).argmax(1)
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
                        # Polyak soft update (Linesight soft_update_tau).
                        with torch.no_grad():
                            for pt, po in zip(target.parameters(),
                                              online.parameters()):
                                pt.mul_(1.0 - SOFT_TARGET_TAU).add_(
                                    SOFT_TARGET_TAU * po)

            w_now = export()
            np.savez(out_path, **w_now,
                     obs_mean=obs_mean.astype(np.float32),
                     obs_std=std.astype(np.float32), circuit=args.circuit,
                     seen=np.int64(seen + per_iter),
                     best_eval=np.float32(best_eval),
                     best_tier=np.int64(best_tier),
                     best_lap=np.float32(best_lap_saved))

            eval_note = ""
            if b_fill >= args.warmup and (it + 1) % args.eval_every == 0:
                ev, tier, per_launch, best_lap = greedy_eval(
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
                lap_s = f" lap{best_lap:5.1f}s" if best_lap > 0.0 else ""
                if better:
                    best_eval, best_tier, best_lap_saved = ev, tier, best_lap
                    best_it = it + 1
                    np.savez(best_path, **w_now,
                             obs_mean=obs_mean.astype(np.float32),
                             obs_std=std.astype(np.float32),
                             circuit=args.circuit,
                             seen=np.int64(seen + per_iter),
                             best_eval=np.float32(best_eval),
                             best_tier=np.int64(best_tier),
                             best_lap=np.float32(best_lap_saved))
                    eval_note = f"  eval {ev:5.0f}m {tag}{lap_s} *BEST*"
                else:
                    best_lap_note = f" {best_lap_saved:.1f}s" if best_lap_saved > 0.0 else ""
                    eval_note = (f"  eval {ev:5.0f}m {tag}{lap_s} "
                                f"(best {best_eval:.0f}T{best_tier}{best_lap_note} @it{best_it})")

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
                      f"eps {eps:.2f}  oS {off_scale:.1f}  pc {pace_mult:.2f}  "
                      f"lw {lap_w_mult:.2f}  "
                      f"g {gamma:.4f}  "
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
