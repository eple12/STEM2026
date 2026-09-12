"""End-to-end training-iteration cost: rollout AND learner.

    python tests/bench_train.py [--device cuda] [--envs 14 2048]

``bench.py`` times environment stepping alone, which is what the CPU trainer
is bottlenecked on. Once the rollout is batched that bottleneck moves, so
this times one whole training iteration the way ``train_iqn_gpu.py`` runs it
-- ``--rollout`` decisions per environment, then ``--grad-steps`` gradient
steps of ``--batch`` -- and reports the two halves separately.

The CPU baseline to compare against is measured from the training logs:
``train_monza_v26.log`` reaches it 1324 at 237.0 min, i.e. **10.7 s per
iteration** of 2688 decisions.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai_sw"))

from game.rlpolicy import N_ACTIONS, OBS_DIM                # noqa: E402
from gpuenv.trackgpu import TrackGPU                        # noqa: E402
from gpuenv.vecenv import VecRaceEnv                        # noqa: E402
from train_iqn_gpu import IQN, IQN_N, iqn_loss              # noqa: E402

CPU_ITER_S = 10.7            # s/iteration, from train_monza_v26.log
CPU_ITER_DECISIONS = 2688


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[14, 512, 2048])
    ap.add_argument("--rollout", type=int, default=192)
    ap.add_argument("--grad-steps", type=int, default=48)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--buffer", type=int, default=150_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device {device}  rollout {args.rollout}/env  "
          f"grad-steps {args.grad_steps} x batch {args.batch}")
    print(f"CPU trainer baseline: {CPU_ITER_S:.1f} s/iter "
          f"for {CPU_ITER_DECISIONS} decisions\n")

    online, target = IQN().to(device), IQN().to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=1e-3, eps=1e-4)

    # A full replay buffer of noise: the learner's cost does not depend on
    # what is in it, only on its shape.
    cap = args.buffer
    b_obs = torch.randn(cap, OBS_DIM, device=device)
    b_next = torch.randn(cap, OBS_DIM, device=device)
    b_act = torch.randint(N_ACTIONS, (cap,), device=device)
    b_ret = torch.randn(cap, device=device)
    b_gam = torch.full((cap,), 0.999 ** 3, device=device)

    def learner_pass():
        for _ in range(args.grad_steps):
            sel = torch.randint(0, cap, (args.batch,), device=device)
            so, sn = b_obs[sel], b_next[sel]
            sa, sr, sg = b_act[sel], b_ret[sel], b_gam[sel]
            with torch.no_grad():
                qn_target, _ = target(sn, IQN_N)
                best_a = qn_target.mean(1).argmax(1)
                qn_sel = qn_target.gather(
                    2, best_a[:, None, None].expand(-1, IQN_N, 1)).squeeze(-1)
                tgt = (sr[:, None] + sg[:, None] * qn_sel).unsqueeze(-1)
            q, tau = online(so, IQN_N)
            q_sel = q.gather(2, sa[:, None, None].expand(-1, IQN_N, 1))
            loss = iqn_loss(tgt, q_sel, tau).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 30.0)
            opt.step()

    # The learner is independent of the environment count, so time it once.
    for _ in range(3):
        learner_pass()
    sync(device)
    t0 = time.perf_counter()
    learner_pass()
    sync(device)
    learn_s = time.perf_counter() - t0
    print(f"learner: {learn_s:6.2f} s per iteration "
          f"({args.grad_steps} steps x batch {args.batch})\n")

    track = TrackGPU("Monza", device)
    hdr = f"{'envs':>6} {'decisions':>10} {'rollout s':>10} {'total s':>8} " \
          f"{'s/1k dec':>9} {'vs CPU':>9}"
    print(hdr)
    print("-" * len(hdr))

    for n in args.envs:
        env = VecRaceEnv("Monza", n, device=device, track=track)
        obs = env.reset_done(torch.ones(n, dtype=torch.bool, device=device))
        mean = torch.zeros(OBS_DIM, device=device)
        std = torch.ones(OBS_DIM, device=device)

        def rollout(steps):
            nonlocal obs
            with torch.no_grad():
                for _ in range(steps):
                    x = ((obs - mean) / std).clamp(-10.0, 10.0)
                    act = online.q_mean(x).argmax(1)
                    obs, _, done, _ = env.step(act)
                    if bool(done.any()):
                        obs = env.reset_done(done)

        rollout(3)
        sync(device)
        t0 = time.perf_counter()
        rollout(args.rollout)
        sync(device)
        roll_s = time.perf_counter() - t0

        dec = args.rollout * n
        total = roll_s + learn_s
        per_1k = total / (dec / 1000.0)
        cpu_per_1k = CPU_ITER_S / (CPU_ITER_DECISIONS / 1000.0)
        print(f"{n:6d} {dec:10,} {roll_s:10.2f} {total:8.2f} "
              f"{per_1k:9.3f} {cpu_per_1k / per_1k:8.1f}x")

    print("\n'vs CPU' is decisions/second end to end, against the CPU "
          "trainer's 10.7 s per 2688 decisions.")


if __name__ == "__main__":
    main()
