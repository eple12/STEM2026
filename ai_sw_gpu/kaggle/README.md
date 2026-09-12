# Training on Kaggle

Colab reclaims a runtime after about an hour, so a 5580-iteration run has to
survive several restarts and ferry a 77 MB replay buffer between machines to
avoid losing its policy. Kaggle runs a kernel non-interactively for up to
**9 hours on a GPU**, which holds a whole run — the interruption problem
disappears rather than being managed.

| | Colab | Kaggle |
| --- | --- | --- |
| session | ~1 h observed, three times | up to 9 h (GPU), 12 h (CPU) |
| weekly GPU | undocumented; 503 after ~4 h | 30 h, stated |
| GPU | T4 | T4 x2 or P100 |
| model | interactive session + exec | batch: push a kernel, collect output |

## One-time setup

1. Kaggle → Settings → API → **Create New Token**, which downloads
   `kaggle.json`.
2. Put it where the CLI looks and lock it down:

```bash
mkdir -p ~/.kaggle && mv /mnt/c/Users/<you>/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. The CLI itself is already installed (`uv tool install kaggle`).

## Running

```bash
bash kaggle_push.sh <kaggle-user>                    # Spa, fresh
bash kaggle_push.sh <kaggle-user> Monza Monza_v26d   # another circuit
```

`kaggle_push.sh` bundles the code and track data, publishes it as the
private dataset `<user>/aisw-code` (a new version each push), writes the
script kernel, and pushes it — which starts the run.

```bash
bash kaggle_watch.sh <kaggle-user>/aisw-spa-v26d     # streams the log live
bash kaggle_fetch.sh <kaggle-user>/aisw-spa-v26d Spa_v26d
```

`kaggle_fetch.sh` drops `<OUT_NAME>.npz`, `_best.npz` and `_state.pt` into
`ai_sw/assets/policies`, where the trace script and the game expect them.

## Continuing a run that hit the wall

A kernel's own output is attachable as an input to the next one, so pass the
previous kernel as the fourth argument:

```bash
bash kaggle_push.sh <user> Spa Spa_v26d <user>/aisw-spa-v26d
```

`kernel_run.py` copies `<OUT_NAME>*` out of the attached input, and the
trainer's `--resume` picks up the weights **and** `_state.pt` — the replay
buffer, target network, optimiser and n-step window. That is what makes it a
continuation rather than a restart; without the state file a resume drops a
trained policy into an empty buffer at gamma 1.0 and undoes its own
progress.

## Accelerators need phone verification

`enable_gpu` in the metadata, `machine_shape`, and `kaggle kernels push
--accelerator` were all set, the stored kernel metadata read them back
correctly — and the run still came up on `torch 2.10.0+cpu` with no
`nvidia-smi`. Kaggle gates GPU and TPU behind **SMS phone verification**
(Settings → Phone Verification); until that is done the accelerator request
is silently ignored rather than refused, so a kernel looks like it started
fine and is simply running on a CPU. VoIP numbers are commonly rejected, so
use a carrier-backed one.

`kernel_run.py` checks `torch.cuda.is_available()` and says so loudly rather
than quietly spending hours of quota on a CPU.

## Notes

- `enable_internet` is off. Kaggle requires phone verification to turn it
  on, and nothing here needs it: torch and numpy are in the image and the
  code arrives through the dataset.
- Everything under `/kaggle/working` becomes the kernel output, and the
  trainer writes its checkpoints there, so a run killed at the 9 h wall
  still returns whatever it had reached.
- `machine_shape` picks the accelerator (`NvidiaTeslaT4`, `NvidiaL4`, ...);
  availability depends on the account.
