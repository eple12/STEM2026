# ai_sw_gpu — batched (GPU) twin of the ai_sw driving environment

The CPU project in `../ai_sw` is untouched and stays authoritative. This is
the same environment and the same IQN training recipe with every environment
turned into a row of a tensor, so a rollout is a handful of kernels instead
of a Python loop per car.

**Why.** The CPU trainer is not GPU-starved, it is rollout-starved: seven
worker processes each step two `RaceEnv`s in Python and the whole run manages
roughly 250 decisions a second, so a 15 M-decision schedule takes 15–25
hours. The network is small; moving only it to CUDA would change little.
Batching the *environment* is the lever.

## Layout

| file | what it is |
| --- | --- |
| `gpuenv/trackgpu.py` | static circuit data, built by the CPU code and frozen into tensors (including a padded per-sample barrier-neighbourhood table) |
| `gpuenv/surface.py` | batched `game.surface.Surface` — nearest-sample search, four-wheel grip, barrier penetration |
| `gpuenv/physics.py` | batched `game.vehicle.Vehicle` — Pacejka tyres, friction circle, ABS/TC/ESC, wall impulse |
| `gpuenv/vecenv.py` | batched `game.rlenv.RaceEnv` — reward, recovery, lap arming, potential shaping |
| `train_iqn_gpu.py` | the trainer; schedules are **imported** from `ai_sw/tools/train_iqn.py`, not copied |
| `tests/parity.py` | scalar-vs-batched trajectory comparison |
| `tests/bench.py` | decisions/second at several batch sizes |

Geometry is never re-derived: `TrackGPU` calls the CPU `load_track`,
`reference_line`, `shaping_line` and `speed_profile` and converts the result,
so the two implementations cannot disagree about the circuit. Checkpoints are
written in the same `.npz` layout, so anything trained here loads straight
into the game.

## Parity

`tests/parity.py` runs N scalar `RaceEnv`s and one `VecRaceEnv` from the same
deterministic grid starts, feeds both the identical action stream and
compares observation, reward, car state and every counter at every step. It
runs in float64 so the only differences left are real ones.

Three modes, because no single one reaches every branch:

```bash
python tests/parity.py --mode random --steps 900 --envs 8   # off-track, recovery, stall
python tests/parity.py --mode wall   --steps 120 --envs 8   # barrier collision + contact-arm impulse
python tests/parity.py --mode policy --steps 2200 --envs 4  # real driving, closed laps, lap bonus
```

Measured (2026-09-12), all three `PARITY OK`, counters identical at every
step (`off_steps`, `recoveries`, `wall_steps`, `laps`):

| mode | coverage | max abs diff (pos / reward) |
| --- | --- | --- |
| random, 900 steps × 8 | 76 recoveries, 1364 off-steps | 1.5e-13 / 4.7e-14 |
| wall, 120 steps × 8 | 8 wall contacts | 3.6e-15 / 6.9e-18 |
| policy, 2200 steps × 4 | 4 completed laps | 1.4e-11 / 4.0e-12 |

The residual observation difference of ~5e-7 is not an error: the CPU
`observe()` returns float32, so that is its own quantisation.

## Throughput

```bash
python tests/bench.py --sizes 16 128 1024
```

On this laptop — **CPU only, no NVIDIA GPU**, and with the v26 CPU training
run occupying most cores:

| envs | decisions/s | vs CPU trainer |
| --- | --- | --- |
| 16 | 329 | 1.3× |
| 128 | 883 | 3.5× |
| 1024 | 3,939 | 15.8× |

On a **Colab Tesla T4** (16 GB, torch 2.11+cu128), measured 2026-09-12 via
the Colab CLI:

| envs | decisions/s | vs CPU trainer | ms/step |
| --- | --- | --- | --- |
| 16 | 479 | 1.9× | 33.4 |
| 128 | 3,873 | 15.5× | 33.0 |
| 1024 | 30,346 | 121× | 33.7 |
| 4096 | 87,248 | 349× | 46.9 |
| 16384 | 421,848 | **1687×** | 38.8 |

Note the ms/step column: a step costs about the same whether it advances 16
cars or 16384. The step is latency-bound — a long chain of small kernels plus
the handful of host syncs in the stale-hint fallback — so throughput scales
almost linearly with the batch until something else gives. Reproduce with
`tests/bench.py --device cuda`.

**This measures environment stepping only**, which is what the CPU trainer is
bottlenecked on. It is not a training-time speedup: the learner still runs 48
gradient steps of batch 512 per iteration, and once the rollout is this cheap
the learner is the bottleneck instead. That side has not been measured.

## Training

Defaults reproduce the CPU run exactly (14 environments × 192 decisions =
2688 decisions an iteration, same schedules, same eval):

```bash
python train_iqn_gpu.py --circuit Monza --steps 15000000 --out-name Monza_gpu
```

`--envs` is the throughput lever; raising it changes the decisions per
iteration, so it is a different experiment, not just a faster one:

```bash
python train_iqn_gpu.py --circuit Monza --steps 15000000 --envs 2048 --device cuda --out-name Monza_gpu
```

`--resume`, `--init-from`, `--ddqn`, `--start-at-line` and the eval/tier logic
behave exactly as in the CPU trainer.
