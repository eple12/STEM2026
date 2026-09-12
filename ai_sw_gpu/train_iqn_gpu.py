"""IQN on the batched driving environment -- the GPU twin of
``ai_sw/tools/train_iqn.py``.

    python train_iqn_gpu.py --circuit Monza --steps 15000000 --envs 2048

Same algorithm, same schedules, same checkpoint format. The difference is
where the rollout happens: the CPU trainer forks worker processes that each
step a couple of environments in Python, and tops out around 250 decisions a
second; here every environment is a row of one tensor, so the rollout is the
same handful of kernels whether there are 14 of them or 4096.

The schedules are IMPORTED from the CPU file rather than copied, so the two
runs cannot drift apart. Defaults reproduce the CPU run exactly (14
environments x 192 decisions = 2688 decisions an iteration); ``--envs`` is
the throughput lever.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
AI_SW = ROOT.parent / "ai_sw"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AI_SW))

# The CPU trainer pins BLAS to one thread for its worker pool; this run has
# no pool, so claim the threads back before importing it.
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

from game import config                                    # noqa: E402
from game.rlpolicy import IQN_EMBED, IQN_K, N_ACTIONS, OBS_DIM   # noqa: E402
from gpuenv.vecenv import VecRaceEnv                       # noqa: E402
from gpuenv.trackgpu import TrackGPU                       # noqa: E402


def _load_cpu_trainer():
    """Import ``ai_sw/tools/train_iqn.py`` for its schedules."""
    spec = importlib.util.spec_from_file_location(
        "cpu_train_iqn", AI_SW / "tools" / "train_iqn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CPU = _load_cpu_trainer()
GAMMA_SCHED = CPU.GAMMA_SCHED
EPS_SCHED = CPU.EPS_SCHED
EPS_SCHED_WARM = CPU.EPS_SCHED_WARM
EPS_BOLTZ_SCHED = CPU.EPS_BOLTZ_SCHED
LR_SCHED = CPU.LR_SCHED
TAU_BOLTZ = CPU.TAU_BOLTZ
SOFT_TARGET_TAU = CPU.SOFT_TARGET_TAU
OFF_COST_SCHED = CPU.OFF_COST_SCHED
PACE_SCHED = CPU.PACE_SCHED
_interp = CPU._interp

EVAL_LAUNCHES = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
OFF_PERFECT_TOL = 2
KAPPA = 5e-3
IQN_N = 8
A_EMB, E = IQN_EMBED, 256


class IQN(nn.Module):
    """Identical to the CPU trainer's network, weight for weight."""

    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU
        self.ff = nn.Sequential(nn.Linear(OBS_DIM, E), act(),
                                nn.Linear(E, E), act())
        self.phi = nn.Sequential(nn.Linear(A_EMB, E), act())
        self.A = nn.Sequential(nn.Linear(E, E), act(), nn.Linear(E, N_ACTIONS))
        self.V = nn.Sequential(nn.Linear(E, E), act(), nn.Linear(E, 1))

    def forward(self, x, nq):
        b = x.shape[0]
        h = self.ff(x)
        tau = torch.rand(b, nq, 1, device=x.device, dtype=x.dtype)
        ar = torch.arange(1, A_EMB + 1, device=x.device, dtype=x.dtype)
        emb = self.phi(torch.cos(ar * math.pi * tau))
        mixed = h.unsqueeze(1) * emb
        a = self.A(mixed)
        v = self.V(mixed)
        return v + a - a.mean(-1, keepdim=True), tau

    def q_mean(self, x, k: int = IQN_K):
        """Mean Q per action on the FIXED quantile grid.

        This is the torch twin of ``rlpolicy.iqn_q`` -- the deterministic tau
        grid, not ``forward``'s random draw -- so a greedy action here is the
        same greedy action the exported checkpoint takes in the game.
        """
        h = self.ff(x)
        tau = torch.linspace(0.5 / k, 1.0 - 0.5 / k, k,
                             device=x.device, dtype=x.dtype)
        ar = torch.arange(1, A_EMB + 1, device=x.device, dtype=x.dtype)
        emb = self.phi(torch.cos(ar[None, :] * math.pi * tau[:, None]))
        mixed = h[:, None, :] * emb[None, :, :]
        a = self.A(mixed)
        v = self.V(mixed)
        q = v + a - a.mean(-1, keepdim=True)
        return q.mean(1)


def iqn_loss(tgt, out, tau_out):
    td = tgt[:, :, None, :] - out[:, None, :, :]
    hub = torch.where(td.abs() < KAPPA, 0.5 / KAPPA * td ** 2,
                      td.abs() - 0.5 * KAPPA)
    tau = tau_out[:, None, :, :]
    rho = torch.where(td < 0, 1 - tau, tau) * hub
    return rho.sum(2).mean(1).squeeze(-1)


def export(net):
    s = net.state_dict()
    w = {
        "ff0.w": s["ff.0.weight"].cpu().numpy().T,
        "ff0.b": s["ff.0.bias"].cpu().numpy(),
        "ff1.w": s["ff.2.weight"].cpu().numpy().T,
        "ff1.b": s["ff.2.bias"].cpu().numpy(),
        "iqn.w": s["phi.0.weight"].cpu().numpy().T,
        "iqn.b": s["phi.0.bias"].cpu().numpy(),
        "A0.w": s["A.0.weight"].cpu().numpy().T,
        "A0.b": s["A.0.bias"].cpu().numpy(),
        "A1.w": s["A.2.weight"].cpu().numpy().T,
        "A1.b": s["A.2.bias"].cpu().numpy(),
        "V0.w": s["V.0.weight"].cpu().numpy().T,
        "V0.b": s["V.0.bias"].cpu().numpy(),
        "V1.w": s["V.2.weight"].cpu().numpy().T,
        "V1.b": s["V.2.bias"].cpu().numpy(),
    }
    return {k: v.astype(np.float32) for k, v in w.items()}


_NAME_MAP = {"ff.0": "ff0", "ff.2": "ff1", "phi.0": "iqn",
             "A.0": "A0", "A.2": "A1", "V.0": "V0", "V.2": "V1"}


def savez_atomic(path, **kw):
    """Write the checkpoint, then move it into place.

    A checkpoint is copied off the machine while training continues, so a
    reader must never catch a half-written file. ``os.replace`` is atomic on
    the same filesystem, so the file at ``path`` is always a complete one.
    """
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def load_into(net, d, device):
    sd = net.state_dict()
    for tk, nk in _NAME_MAP.items():
        sd[f"{tk}.weight"] = torch.as_tensor(
            np.ascontiguousarray(d[f"{nk}.w"].T), dtype=torch.float32,
            device=device)
        sd[f"{tk}.bias"] = torch.as_tensor(
            np.ascontiguousarray(d[f"{nk}.b"]), dtype=torch.float32,
            device=device)
    net.load_state_dict(sd)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--steps", type=int, default=15_000_000)
    ap.add_argument("--envs", type=int, default=14,
                    help="parallel environments; 14 reproduces the CPU run")
    ap.add_argument("--rollout", type=int, default=192,
                    help="decisions per env per iteration")
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--grad-steps", type=int, default=48)
    ap.add_argument("--buffer", type=int, default=150_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--target-sync", type=int, default=48)
    ap.add_argument("--ddqn", action="store_true")
    ap.add_argument("--start-at-line", type=float, default=0.15)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--async-eval", action="store_true",
                    help="run the eval in a worker thread so the GPU keeps "
                         "training through it. The result lands a few "
                         "hundred iterations late and is logged with the "
                         "iteration it actually measured.")
    ap.add_argument("--save-every", type=int, default=1,
                    help="iterations between running-checkpoint saves")
    ap.add_argument("--resume-warmup", type=int, default=None,
                    help="on --resume, collect this many transitions before "
                         "learning again (default: half the buffer). The "
                         "replay buffer does not survive a restart, and "
                         "resuming a good policy straight into a nearly "
                         "empty, highly correlated buffer at gamma 1.0 is "
                         "how a resumed run throws away its progress. Pass "
                         "0 to learn immediately.")
    ap.add_argument("--eval-device", default="cpu",
                    help="where the 7-launch greedy eval runs. Seven "
                         "environments stepped 2600 times sequentially is a "
                         "latency problem, not a throughput one, so the CPU "
                         "wins; pass the training device to keep it there.")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    online, target = IQN().to(device), IQN().to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=LR_SCHED[0][1], eps=1e-4)

    n_env = args.envs
    per_iter = args.rollout * n_env
    iters = max(1, args.steps // per_iter)

    track = TrackGPU(args.circuit, device)
    env = VecRaceEnv(args.circuit, n_env, seed=args.seed,
                     start_at_line=args.start_at_line, device=device,
                     track=track)

    # The eval is seven launches driven to their 130 s truncation: 2600
    # SEQUENTIAL steps over seven environments. That shape is the opposite of
    # what a GPU is for -- the kernels are trivial and the wall clock is all
    # launch latency and host syncs, which measured ~7x slower per step than
    # the same code at 16384 environments. Seven environments of numpy-sized
    # work belong on the CPU, so the eval keeps its own CPU track and a CPU
    # copy of the network.
    eval_device = torch.device(args.eval_device)
    eval_track = (track if eval_device == device
                  else TrackGPU(args.circuit, eval_device))
    eval_env = VecRaceEnv(args.circuit, len(EVAL_LAUNCHES), seed=99_991,
                          randomise_start=False, device=eval_device,
                          track=eval_track)
    eval_net = IQN().to(eval_device) if eval_device != device else None

    # Stagger the first episode so the 130 s truncations do not all land in
    # the same iteration forever -- they never desync on their own.
    obs = env.reset_done(torch.ones(n_env, dtype=torch.bool, device=device))
    env.lap_time = torch.rand(n_env, generator=env.gen, device=device) * 120.0

    # -- replay buffer, on the device -------------------------------------
    cap = args.buffer
    b_obs = torch.zeros(cap, OBS_DIM, device=device)
    b_act = torch.zeros(cap, dtype=torch.long, device=device)
    b_ret = torch.zeros(cap, device=device)
    b_next = torch.zeros(cap, OBS_DIM, device=device)
    b_gam = torch.zeros(cap, device=device)
    b_done = torch.zeros(cap, device=device)
    b_fill = 0
    b_pos = 0

    obs_mean = torch.zeros(OBS_DIM, dtype=torch.float64, device=device)
    obs_var = torch.ones(OBS_DIM, dtype=torch.float64, device=device)
    obs_n = 1e-4

    config.RL_POLICY.mkdir(parents=True, exist_ok=True)
    name = args.out_name or args.circuit
    out_path = config.RL_POLICY / f"{name}.npz"
    best_path = config.RL_POLICY / f"{name}_best.npz"

    if args.init_from:
        ip = Path(args.init_from)
        if not ip.exists():
            ip = config.RL_POLICY / f"{args.init_from}.npz"
        d0 = np.load(ip, allow_pickle=False)
        load_into(online, d0, device)
        target.load_state_dict(online.state_dict())
        if "obs_mean" in d0.files:
            obs_mean[:] = torch.as_tensor(d0["obs_mean"], device=device)
            obs_var[:] = torch.as_tensor(
                np.square(d0["obs_std"].astype(np.float64)), device=device)
            obs_n = 1e5
        print(f"warm-started from {ip.name}", flush=True)

    resume_seen = 0
    warmup_now = args.warmup
    best_eval, best_tier = 0.0, 3
    if args.resume and out_path.exists():
        dR = np.load(out_path, allow_pickle=False)
        load_into(online, dR, device)
        target.load_state_dict(online.state_dict())
        if "obs_mean" in dR.files:
            obs_mean[:] = torch.as_tensor(dR["obs_mean"], device=device)
            obs_var[:] = torch.as_tensor(
                np.square(dR["obs_std"].astype(np.float64)), device=device)
            obs_n = 1e6
        resume_seen = int(dR["seen"]) if "seen" in dR.files else 0
        bsrc = dR
        if best_path.exists():
            bd = np.load(best_path, allow_pickle=False)
            if "best_eval" in bd.files:
                bsrc = bd
        if "best_eval" in bsrc.files:
            best_eval, best_tier = float(bsrc["best_eval"]), int(bsrc["best_tier"])
        # The replay buffer is not saved, so a resumed run starts with an
        # empty one. Learning from a nearly empty, highly correlated buffer
        # at gamma 1.0 is how a resumed run undoes its own progress, so
        # refill a decent fraction of it before touching the weights.
        warmup_now = (args.buffer // 2 if args.resume_warmup is None
                      else args.resume_warmup)
        warmup_now = max(warmup_now, args.warmup)
        print(f"resumed {out_path.name} at {resume_seen:,} decisions "
              f"(best {best_eval:.0f} T{best_tier}); refilling the buffer to "
              f"{warmup_now:,} before learning", flush=True)
    elif args.resume:
        print(f"--resume: no {out_path.name} yet, starting fresh", flush=True)

    print(f"{args.circuit}: IQN on {device}, {OBS_DIM}-dim obs, "
          f"{N_ACTIONS} actions, {n_env} envs, {per_iter} decisions/iter, "
          f"{iters} iters ({iters * per_iter:,} decisions)", flush=True)

    # -- greedy evaluation -------------------------------------------------
    def greedy_eval(sd_cpu, mean_e, std_e):
        """Mean reach and the cleanliness tier over the seven fixed launches.

        All seven run in one batch: every launch truncates at the same
        ``EPISODE_SECONDS``, so they finish on the same step and nothing has
        to be frozen while the others catch up.

        Takes a detached CPU snapshot of the weights rather than reading
        ``online``, because with ``--async-eval`` this runs in a worker
        thread while the learner keeps changing them. Everything it touches
        -- ``eval_net``, ``eval_env`` -- belongs to that thread alone.
        """
        if eval_net is not None:
            eval_net.load_state_dict(sd_cpu)
            net = eval_net
        else:
            net = online

        fr = torch.tensor(EVAL_LAUNCHES, device=eval_device)
        o = eval_env.reset_grid(fr)
        with torch.no_grad():
            while True:
                x = ((o - mean_e) / std_e.clamp(min=1e-4)).clamp(-10.0, 10.0)
                a = net.q_mean(x).argmax(1)
                o, _, d, _ = eval_env.step(a)
                if bool(d.all()):
                    break
        reach = eval_env.progress
        rec = eval_env.recoveries
        off = eval_env.off_steps
        tier = 0
        if bool((rec > 0).any()):
            tier = 2
        elif bool((off > OFF_PERFECT_TOL).any()):
            tier = 1
        laps = eval_env.best_lap_time[eval_env.best_lap_time > 0]
        best_lap = float(laps.min()) if laps.numel() else 0.0
        return (float(reach.mean()), tier, rec.cpu().numpy(),
                off.cpu().numpy(), best_lap)

    # -- n-step history ----------------------------------------------------
    n_step = args.n_step
    h_obs = torch.zeros(n_step, n_env, OBS_DIM, device=device)
    h_act = torch.zeros(n_step, n_env, dtype=torch.long, device=device)
    h_rew = torch.zeros(n_step, n_env, device=device)
    h_cnt = torch.zeros(n_env, dtype=torch.long, device=device)

    # An eval is ~54 s of CPU work on a snapshot of the weights, so there is
    # no reason for the GPU to sit idle through it: hand it to a worker
    # thread and collect the answer whenever it turns up. The torch CPU ops
    # release the GIL, so it really does overlap. One at a time -- if the
    # previous eval is still running the next is skipped rather than queued,
    # because a backlog of stale evaluations helps nobody.
    eval_pool = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval")
                 if args.async_eval else None)
    pending = None              # (future, weights snapshot, iteration, seen)

    def submit_eval(w_snapshot, it_at, seen_at):
        sd_cpu = {k: v.detach().to(eval_device, copy=True)
                  for k, v in online.state_dict().items()}
        mean_e = obs_mean.to(torch.float32).to(eval_device)
        std_e = torch.sqrt(obs_var).to(torch.float32).to(eval_device)
        if eval_pool is None:
            return (greedy_eval(sd_cpu, mean_e, std_e), w_snapshot, it_at,
                    seen_at)
        return (eval_pool.submit(greedy_eval, sd_cpu, mean_e, std_e),
                w_snapshot, it_at, seen_at)

    def harvest(entry, force=False):
        """Turn a finished eval into the *_best ratchet and a log note.

        The checkpoint written is the snapshot that was evaluated, not
        whatever the learner has moved on to since.
        """
        nonlocal best_eval, best_tier
        fut, w_snap, it_at, seen_at = entry
        if hasattr(fut, "result"):                  # a Future
            if not (force or fut.done()):
                return None, entry
            ev, tier, rec_e, off_e, best_lap = fut.result()
        else:                                       # already a result tuple
            ev, tier, rec_e, off_e, best_lap = fut
        better = tier < best_tier or (tier == best_tier and ev > best_eval)
        TIER_TAG = {0: "PERFECT", 1: "off-track", 2: "wall/recover"}
        bad = ",".join(f"{sf:.2f}" for sf, rc, os_ in
                       zip(EVAL_LAUNCHES, rec_e, off_e)
                       if rc > 0 or os_ > OFF_PERFECT_TOL)
        tag = TIER_TAG[tier] + (f"@{bad}" if tier and bad
                                else f"(off{int(off_e.max())})")
        lap_s = f" lap{best_lap:5.1f}s" if best_lap > 0.0 else ""
        at = f"@it{it_at}" if hasattr(fut, "result") else ""
        if better:
            best_eval, best_tier = ev, tier
            w, mean_np, std_np = w_snap
            savez_atomic(best_path, **w, obs_mean=mean_np, obs_std=std_np,
                     circuit=args.circuit, seen=np.int64(seen_at),
                     best_eval=np.float32(best_eval),
                     best_tier=np.int64(best_tier))
            return f"  eval{at} {ev:5.0f}m {tag}{lap_s} *BEST*", None
        return (f"  eval{at} {ev:5.0f}m {tag}{lap_s} "
                f"(best {best_eval:.0f}T{best_tier})"), None

    # Ctrl-C / SIGTERM finishes the current iteration and then runs the
    # shutdown eval, rather than dropping the run wherever it happened to be.
    stop = {"asked": False}

    def _on_signal(_sig, _frame):
        if stop["asked"]:                          # second one: go now
            raise KeyboardInterrupt
        stop["asked"] = True
        print("\n[stop requested -- finishing this iteration, then a final "
              "eval of the weights we are stopping with]", flush=True)

    for _s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_s, _on_signal)
        except (ValueError, OSError):
            pass

    learner_steps = 0
    last_it = last_seen = 0
    best = 0.0
    recent = []                                   # last ~40 completed episodes
    eps_sched = EPS_SCHED_WARM if args.init_from else EPS_SCHED
    lr_sched = ([(0, 3e-4), (2_000_000, 1e-4), (5_000_000, 5e-5)]
                if args.init_from else LR_SCHED)
    t0 = time.perf_counter()

    for it in range(resume_seen // per_iter, iters):
        seen = it * per_iter
        gamma = _interp(GAMMA_SCHED, seen)
        eps = _interp(eps_sched, seen)
        eps_b = _interp(EPS_BOLTZ_SCHED, seen)
        lr = _interp(lr_sched, seen)
        off_scale = _interp(OFF_COST_SCHED, seen)
        pace_mult = _interp(PACE_SCHED, seen)
        for g in opt.param_groups:
            g["lr"] = lr

        std = torch.sqrt(obs_var)
        mean_t = obs_mean.to(torch.float32)
        std_t = std.to(torch.float32)
        gpow = gamma ** torch.arange(n_step, device=device, dtype=torch.float32)

        e_obs, e_act, e_ret, e_next, e_gam = [], [], [], [], []
        stats = []

        with torch.no_grad():
            for _ in range(args.rollout):
                x = ((obs - mean_t) / std_t.clamp(min=1e-4)).clamp(-10.0, 10.0)
                q = online.q_mean(x)
                greedy = q.argmax(1)
                roll = torch.rand(n_env, device=device)
                rand_a = torch.randint(N_ACTIONS, (n_env,), device=device)
                boltz = (q + eps_b * TAU_BOLTZ
                         * torch.randn_like(q)).argmax(1)
                act = torch.where(roll < eps, rand_a,
                                  torch.where(roll < eps + eps_b, boltz,
                                              greedy))

                nobs, rew, done, _ = env.step(act, float(off_scale),
                                              float(pace_mult))

                # push onto the rolling n-step window
                h_obs = torch.roll(h_obs, -1, dims=0)
                h_act = torch.roll(h_act, -1, dims=0)
                h_rew = torch.roll(h_rew, -1, dims=0)
                h_obs[-1], h_act[-1], h_rew[-1] = obs, act, rew
                h_cnt = (h_cnt + 1).clamp(max=n_step)

                full = h_cnt == n_step
                if bool(full.any()):
                    ret = (h_rew * gpow[:, None]).sum(0)
                    sel = full.nonzero(as_tuple=True)[0]
                    e_obs.append(h_obs[0, sel])
                    e_act.append(h_act[0, sel])
                    e_ret.append(ret[sel])
                    e_next.append(nobs[sel])
                    e_gam.append(torch.full((sel.numel(),), gamma ** n_step,
                                            device=device))

                if bool(done.any()):
                    # flush the partial n-step tails as truncated transitions
                    for c in range(2, n_step + 1):
                        m = done & (h_cnt == c)
                        if not bool(m.any()):
                            continue
                        sel = m.nonzero(as_tuple=True)[0]
                        for j in range(1, c):
                            tail = h_rew[n_step - c + j:, sel]
                            gp = gamma ** torch.arange(
                                tail.shape[0], device=device,
                                dtype=torch.float32)
                            e_obs.append(h_obs[n_step - c + j, sel])
                            e_act.append(h_act[n_step - c + j, sel])
                            e_ret.append((tail * gp[:, None]).sum(0))
                            e_next.append(nobs[sel])
                            e_gam.append(torch.full(
                                (sel.numel(),), gamma ** tail.shape[0],
                                device=device))

                    ds = done.nonzero(as_tuple=True)[0]
                    steps3 = (env.steps[ds] * 3).clamp(min=1).to(torch.float32)
                    stats.append(torch.stack([
                        env.progress[ds],
                        env.recoveries[ds].to(torch.float32),
                        env.laps[ds].to(torch.float32),
                        env.speed_sum[ds] / steps3,
                        env.off_steps[ds].to(torch.float32) / steps3,
                    ], dim=1))
                    h_cnt = torch.where(done, torch.zeros_like(h_cnt), h_cnt)
                    nobs = env.reset_done(done)
                obs = nobs

        o = torch.cat(e_obs) if e_obs else torch.zeros(0, OBS_DIM, device=device)
        m = o.shape[0]
        if m:
            ac = torch.cat(e_act)
            rt = torch.cat(e_ret)
            nx = torch.cat(e_next)
            gm = torch.cat(e_gam)

            # running obs stats from the raw (un-normalised) observations
            bm = o.mean(0).to(torch.float64)
            bv = o.var(0, unbiased=False).to(torch.float64)
            delta = bm - obs_mean
            tot = obs_n + m
            obs_mean += delta * m / tot
            obs_var = (obs_var * obs_n + bv * m
                       + delta ** 2 * obs_n * m / tot) / tot
            obs_n = tot

            idx = (b_pos + torch.arange(m, device=device)) % cap
            b_obs[idx] = o
            b_act[idx] = ac
            b_ret[idx] = rt
            b_next[idx] = nx
            b_gam[idx] = gm
            b_done[idx] = 0.0
            b_pos = (b_pos + m) % cap
            b_fill = min(cap, b_fill + m)

        stat = torch.cat(stats) if stats else None
        reach = float(torch.maximum(
            env.progress.max(),
            stat[:, 0].max() if stat is not None
            else torch.zeros((), device=device)))
        best = max(best, reach)

        loss_acc = 0.0
        if b_fill >= warmup_now:
            std = torch.sqrt(obs_var)
            mean_l = obs_mean.to(torch.float32)
            std_l = std.to(torch.float32).clamp(min=1e-4)
            for _ in range(args.grad_steps):
                sel = torch.randint(0, b_fill, (args.batch,), device=device)
                so = (b_obs[sel] - mean_l) / std_l
                sn = (b_next[sel] - mean_l) / std_l
                sa, sr, sg, sd = b_act[sel], b_ret[sel], b_gam[sel], b_done[sel]
                with torch.no_grad():
                    qn_target, _ = target(sn, IQN_N)
                    if args.ddqn:
                        qn_online, _ = online(sn, IQN_N)
                        best_a = qn_online.mean(1).argmax(1)
                    else:
                        best_a = qn_target.mean(1).argmax(1)
                    qn_sel = qn_target.gather(
                        2, best_a[:, None, None].expand(-1, IQN_N, 1)
                    ).squeeze(-1)
                    tgt = (sr[:, None]
                           + sg[:, None] * (1 - sd[:, None]) * qn_sel)
                    tgt = tgt.unsqueeze(-1)
                q, tau = online(so, IQN_N)
                q_sel = q.gather(2, sa[:, None, None].expand(-1, IQN_N, 1))
                loss = iqn_loss(tgt, q_sel, tau).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 30.0)
                opt.step()
                loss_acc += float(loss.detach())
                learner_steps += 1
                if learner_steps % args.target_sync == 0:
                    with torch.no_grad():
                        for pt, po in zip(target.parameters(),
                                          online.parameters()):
                            pt.mul_(1.0 - SOFT_TARGET_TAU).add_(
                                SOFT_TARGET_TAU * po)

        eval_note = ""
        due_eval = b_fill >= warmup_now and (it + 1) % args.eval_every == 0
        last_iter = it == iters - 1
        # export() walks every weight back to the host, so at a quarter of a
        # second per iteration it is worth doing only when something needs it.
        need_w = (due_eval or last_iter
                  or (it + 1) % args.save_every == 0)
        if need_w:
            std_np = torch.sqrt(obs_var).to(torch.float32).cpu().numpy()
            mean_np = obs_mean.to(torch.float32).cpu().numpy()
            w_now = export(online)
            savez_atomic(out_path, **w_now, obs_mean=mean_np, obs_std=std_np,
                     circuit=args.circuit, seen=np.int64(seen + per_iter),
                     best_eval=np.float32(best_eval),
                     best_tier=np.int64(best_tier))

        if pending is not None:
            eval_note, pending = harvest(pending)
            eval_note = eval_note or ""
        if due_eval and pending is None:
            pending = submit_eval((w_now, mean_np, std_np), it + 1,
                                  seen + per_iter)
            if eval_pool is None:
                eval_note, pending = harvest(pending)

        if stat is not None:
            recent.append(stat.cpu().numpy())
            allr = np.concatenate(recent)[-40:]
            recent = [allr]
        mins = (time.perf_counter() - t0) / 60.0
        if recent:
            r = recent[0]
            dist, rc, _, sp, of = r.mean(0)
            laps = r[:, 2].max()
            print(f"  it {it+1:4d}/{iters}  {seen+per_iter:>9,}  "
                  f"buf {b_fill:>7,}  reach {reach:5.0f}/{best:5.0f} m  "
                  f"dist {dist:5.0f}  spd {sp*3.6:5.1f}  rec {rc:4.1f}  "
                  f"off {of*100:4.1f}%  laps {laps:.0f}  "
                  f"eps {eps:.2f}  oS {off_scale:.1f}  pc {pace_mult:.2f}  "
                  f"g {gamma:.4f}  "
                  f"loss {loss_acc/max(args.grad_steps,1):.3f}  "
                  f"n{len(r):2d}+{0 if stat is None else len(stat):d}  "
                  f"{mins:5.1f}m{eval_note}", flush=True)
        else:
            print(f"  it {it+1:4d}/{iters}  {seen+per_iter:>9,}  "
                  f"buf {b_fill:>7,}  reach {reach:5.0f}/{best:5.0f} m  "
                  f"warming up  eps {eps:.2f}  {mins:5.1f}m", flush=True)

        last_it, last_seen = it + 1, seen + per_iter
        if stop["asked"]:
            break

    # Collect whatever eval was in flight...
    if pending is not None:
        note, _ = harvest(pending, force=True)
        print(f"  (in flight){note}", flush=True)
    if eval_pool is not None:
        eval_pool.shutdown(wait=True)

    # ...and then evaluate the weights we are ACTUALLY stopping with. Async
    # evals are skipped while one is running, so without this the last
    # hundreds of iterations could go unmeasured and *_best would be left
    # describing a policy from well before the stop.
    if b_fill >= warmup_now:
        print("  final eval of the stopping weights...", flush=True)
        std_np = torch.sqrt(obs_var).to(torch.float32).cpu().numpy()
        mean_np = obs_mean.to(torch.float32).cpu().numpy()
        w_final = export(online)
        savez_atomic(out_path, **w_final, obs_mean=mean_np, obs_std=std_np,
                     circuit=args.circuit, seen=np.int64(last_seen),
                     best_eval=np.float32(best_eval),
                     best_tier=np.int64(best_tier))
        sd_cpu = {k: v.detach().to(eval_device, copy=True)
                  for k, v in online.state_dict().items()}
        res = greedy_eval(sd_cpu,
                          torch.as_tensor(mean_np, device=eval_device),
                          torch.as_tensor(std_np, device=eval_device))
        note, _ = harvest((res, (w_final, mean_np, std_np), last_it, last_seen))
        print(f"  (stop){note}", flush=True)

    print(f"\nsaved {out_path}  (best greedy {best_eval:.0f} m -> "
          f"{best_path})", flush=True)


if __name__ == "__main__":
    main()
