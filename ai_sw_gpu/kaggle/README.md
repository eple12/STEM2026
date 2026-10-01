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

## One-shot: push, wait, fetch, trace, deploy

`train_track.bat` (Windows) wraps the whole loop below into one command:

```bat
train_track.bat Spa
train_track.bat Monza Monza_v26k_retry
```

It runs the parity check, pushes the kernel with no extra flags (the fixed
recipe below lives in `config.py`/`rlpolicy.py`, not in a CLI flag), polls
Kaggle every 2 minutes until `--stop-after-stale` ends the run on its own,
fetches the checkpoint, traces it and prints the flying lap time, then asks
before touching the live `assets/policies/<circuit>.npz` — it never deploys
silently. Either way the checkpoint and Kaggle log are filed under
`ai_sw/policy/<circuit>/` and `ai_sw/logs/<circuit>/`.

### The confirmed recipe (v26k, 2026-09-13)

Locked in after Spa converged to a **121.30 s** flying lap (previous deployed
best: 121.95 s) and validated head-to-head against a dozen-plus reward-tuning
variants that all landed 122.6–123.0 s or worse. Nothing here is
circuit-specific, so it is the default for any new track:

- `RL_EDGE_MARGIN = 0.0` -- no pre-emptive potential-shaping penalty for
  using the full legal track width (config.py).
- `--line-k 0` -- no lateral pull toward a reference line; the policy finds
  its own (kernel_run.py's own default EXTRA flags already pass this).
- `--ddqn` -- Double-DQN action selection (also a kernel_run.py default).
- `RL_OFF_TRACK_COST = 0.20` with `RL_OFF_TRACK_COST_RAMP` ending by 2M
  decisions -- softened and front-loaded so the off-track penalty is flat
  well before a policy typically reaches its first clean lap, instead of
  still escalating for a million decisions after (config.py; see the long
  comment there for the loss-divergence story that motivated this).
- `RL_RL_LAP_OFF_TOL = 6`, `RL_OFF_PERFECT_TOL = 0` -- a lap tolerates a
  couple of physics-tick grazes and still counts as clean for the training
  bonus, but `*_best` selection only ever accepts a literal zero-off-track
  eval (config.py).
- 42-action table: 7 steer levels x 6 pedal levels (full/half throttle,
  coast, light/medium/full brake) -- a binary brake or a binary throttle
  each independently showed the same "slams full, overshoots, corrects"
  failure at corner entry and exit; both got a middle rung (rlpolicy.py).
- Everything else is the untouched v26 baseline (envs 112, rollout 24,
  target-sync 48, start-at-line 0.20, `--stop-after-stale 600`).
- 2026-09-14: the episode-widening curriculum below (`LONG_EPISODE_RATIO`,
  `RL_POST_WIDEN_RAMP`, `--post-widen-start-at-line`) is folded into the
  recipe too, confirmed on Monza (93.30 s, essentially matching the 93.60 s
  record it was retrained from) after the ramp-window fix -- see "Episode
  widening and `_bestwide.npz`" below for the full story. All defaults, so
  still nothing circuit-specific to pass.

Tried and **rejected** on top of this (all worse, or eroded): a steeper
post-PERFECT lap-time bonus (`RL_LAP_W_RAMP`), resuming a stale-stopped run
from its saved state to keep improving (erodes regardless of whether the
optimiser state is carried over or reset), per-sector specialist training
spliced together, and a single-session adaptive reset that reweights toward
whichever track segment is currently failing most. The mechanisms for the
last two still exist (`--focus-window`, `--adaptive-reset` on
`train_iqn_gpu.py`) in case a future circuit's failure mode looks different,
but neither is part of the default recipe.

### Episode widening and `_bestwide.npz`

`train_iqn_gpu.py` starts every run with a short (~1 lap) episode window for
reset-frequency/exposure diversity, and widens it once -- to ~2 laps -- the
first time an eval comes back genuinely PERFECT (`LONG_EPISODE_RATIO` in
`rlenv.py`; see Silverstone's 104.15 s, the recipe's proof case). The catch:
after widening, "PERFECT" means clean for the *entire* 2-lap window, a much
harder bar than the 1-lap PERFECT that triggered the widening. `*_best.npz`'s
ratchet only ever accepts an equal-or-better tier, so nothing weaker can ever
unseat that first, pre-widening checkpoint -- even while the policy keeps
improving afterwards, if it never happens to re-clear the full 2-lap bar
before `--stop-after-stale` fires, `*_best.npz` is left describing a policy
that had barely just learned one clean lap. Confirmed on Monza (2026-09-13):
only 2 PERFECTs the whole run, the run stale-stopped still holding the
pre-widening one, and it traced at 97.45 s -- worse than the already-deployed
93.60 s record.

Fix (safety net): `train_iqn_gpu.py` now also tracks `*_bestwide.npz`, a
separate, looser ratchet that accepts the best tier<=1 (not necessarily fully
clean) result seen since widening, so that in-progress improvement is never
silently lost just because it hasn't re-cleared the full PERFECT bar yet.
`train_track.bat` traces both files when `_bestwide.npz` exists and lets you
pick which to deploy -- always worth comparing both rather than trusting
`_best.npz` alone on a run where PERFECT was rare.

Root cause, and two attempts at it: the Monza log showed *why* PERFECT stayed
rare after widening -- eval reach jumped from 7,711 m (the pre-widening
PERFECT) to a steady 13,000-13,700 m (genuinely ~2 laps at pace) within 100
iterations of widening, but every single one of those evals still came back
off-track or wall/recover, right through all 600 stale iterations. The
policy was plainly capable of covering the extra distance -- it just
couldn't yet do it clean.

- Tried and **reverted** (2026-09-14): `--revive-after`, a plateau-gated
  Boltzmann exploration boost already built for exactly "reached a clean
  tier-0 checkpoint, then gone stale on it" (see the comment above the
  revive block in `train_iqn_gpu.py`), but never actually turned on by
  `kernel_run.py`'s default `EXTRA`. Reverted the same day in favour of a
  reward-side fix -- an exploration boost changes what states get *visited*,
  not what the policy is actually rewarded for, and the Monza log's own eval
  numbers argue the policy already visits post-widening states plenty (13k+
  m of clean-looking distance every attempt); it just isn't priced to stay
  clean there.
- `RL_POST_WIDEN_RAMP` in `config.py` -- scales `RL_OFF_TRACK_COST` and
  `RL_RL_LAP_W` by the *same* growing factor, keyed to decisions since
  widening (not decisions since training start), so "clean vs fast" never
  gets relatively cheaper (which is why `RL_LAP_W_RAMP` alone made things
  worse, see its own comment) while the combined clean-and-fast signal still
  gets sharper as training continues past widening.
- Tested on Monza 2026-09-14 (with `--post-widen-start-at-line` below) and
  it worked -- 4 widened PERFECTs (it1075/1150/1175/1200, best lap 93.7 s,
  essentially matching the 93.60 s record) where the un-fixed run got zero.
  But it also reproduced the exact moving-target failure
  `RL_OFF_TRACK_COST_RAMP`'s own comment already diagnosed once: all 4 hits
  landed while the ramp (window then 1,000,000 decisions) was still only
  ~1.3x-1.6x of the way to its 1.8x ceiling, and the policy went off-track
  on every single eval for the remaining 575 stale-stop iterations -- right
  where the ramp kept climbing the rest of the way. Shortened the window to
  300,000 decisions so it is flat well before ~400k, roughly where that
  run's first widened PERFECT appeared.
- Retested on Monza with the 300k window: **93.30 s**, beating the 93.60 s
  record outright. Confirmed as part of the standard recipe 2026-09-14 (see
  the note in "The confirmed recipe" above) -- this is the whole
  episode-widening curriculum's first clean win end to end: triggered once,
  re-converged reliably post-widening, and actually improved on the
  pre-widening record rather than just matching it.
- 300k decisions is itself a Monza-tuned number, though, the same trap
  `EPISODE_SECONDS_BY_CIRCUIT`/`LONG_EPISODE_RATIO` already got caught in
  once for `episode_seconds` -- a track that takes meaningfully longer or
  shorter to re-clear the harder post-widening PERFECT bar has no reason to
  share Monza's timing. Rather than guess a per-circuit formula with one
  data point, `train_iqn_gpu.py` now **freezes** `RL_POST_WIDEN_RAMP` at
  whatever value it has reached the moment a post-widening PERFECT is first
  actually seen, instead of trusting the window to land before that
  happens on its own. The window in `config.py` now only controls how fast
  the ramp climbs while still searching, not when it has to be flat by --
  freezing does that job per run, adapting to whatever a given circuit's
  timing actually is.

Also changed alongside it (2026-09-14): before widening, most resets scatter
round the whole lap (`--start-at-line`, default 0.15 of resets are a grid
start) so no corner is bottlenecked behind mastering an earlier one -- but
eval (`reset_grid`) and the actual game both *always* start from the grid,
and once PERFECT proves the whole lap is already clean, scattering most
resets elsewhere no longer buys much while starving grid-only states (a
standing-start launch only ever occurs right after a grid reset, never
mid-episode) of exposure. `--post-widen-start-at-line` (default 0.75) swaps
`start_at_line` to this the same moment widening triggers, same reasoning as
`RL_POST_WIDEN_RAMP` above: match the training distribution to what is
actually evaluated once there is no more sequential-bottleneck reason not
to. Tested alongside `RL_POST_WIDEN_RAMP` on the same 2026-09-14 Monza run
(the 4-widened-PERFECT one above) -- can't be separated from the ramp's own
effect in that result, but nothing about it looked harmful, so keeping it on
for the ramp-window retest rather than re-isolating it first.

### Self-discovered line: dropping the raceline crutch, and pure geometry obs

`RL_SHAPE_TO_RACELINE = False` (2026-09-14): the saved raceline it was
pulling `v_ref`/`model_lap`/`episode_seconds` from turned out to be flat-out
broken (CMA-ES refine converging to something *worse* than the deterministic
min-curvature stage -- see the "raceline reliability" investigation this
session), so it was quietly setting the wrong speed target and lap-time
break-even for however long that had gone unnoticed. Off means all three
fall back to the plain centreline, which is always honest even if slower.

Once that was fixed, the network still had the raceline's ghost in its own
INPUT: the "lookahead" observation's tail included the target speed at each
of 16 lookahead points (`v_ref[js]/MAX_SPEED`), not just their geometry --
telling the network how fast to go there, not just where "there" is. Removed
2026-09-14 (`OBS_DIM` 59 -> 43, see rlpolicy.py's `observe()`): the only hint
left about pace is the lap-time reward itself, so a fast line -- and how
fast to take it -- is entirely the policy's own to find. Breaks warm-starting
from any pre-existing checkpoint (different input width); every run from
here on is from scratch.

`RL_POST_WIDEN_RAMP` is OFF (flat, config.py) and `--n-step` 3 -> 6
(kernel_run.py) while this is being tested, specifically so a change in
result can be attributed to one thing -- re-enable the ramp separately once
n-step's own effect is read. n-step's reasoning: with no more fixed speed
target anywhere (reward or observation), a corner-linkage trade-off (brake a
little less here, gain it back at the next exit) now depends entirely on
lap-time credit reaching backward through the decisions that caused it --
n-step=3 leaves most of that reliant on bootstrapped value estimates rather
than raw reward a network actually saw. Monza with the raceline-shaping
fix alone (ramp/n-step both still at their old defaults): 94.8 s, essentially
matching the 93.30/93.8 s cluster -- inconclusive on its own, hence isolating
these two next rather than reading anything into that number yet.

n-step=6 alone (ramp still off): 91.30 s, a new record, and the log
(`Monza_v26k_scratch`) shows the line itself got genuinely faster -- but
also shows the curriculum is still fragile: widened at it700, ONE
post-widening PERFECT at it725 (froze the ramp at 1.00x, i.e. it never
actually engaged -- this run predates re-enabling it), then 600 straight
iterations with zero PERFECT before stale-stop. The good lap time was one
lucky early roll of a 2-lap x 7-launch x zero-tolerance compound bar, not a
mechanism reliably reproducing it. `RL_POST_WIDEN_RAMP` was re-enabled
afterward on the theory that it would help close that gap, but with no run
where it was actually active yet, that is still unconfirmed.

2026-09-15, tried and **reverted** same day: `--start-widened` (skip the
curriculum entirely, `episode_seconds` at the wide value for the whole run)
-- asked whether the curriculum was worth its own complexity at all. Never
actually run; reverted in favour of attacking the real question underneath
it instead -- not "is the curriculum necessary" but "why does PERFECT stop
recurring once episode_seconds gets harder", which starting already-wide
does not answer either way.

2026-09-15, tried and reverted same day: staged widening,
`EPISODE_WIDEN_STAGES = (1.3, 1.6, LONG_EPISODE_RATIO)` -- a 3-step climb
instead of one jump, on the theory that a smaller per-step difficulty
increase would make each individual PERFECT re-clear more likely. Monza
result read as a small, possibly-noise improvement over the one-shot jump,
not clearly worth the extra complexity. Reverted to a single entry (GPU
budget is tight -- explicit request to keep the curriculum surface at just
"widen" rather than keep tuning its shape) -- `EPISODE_WIDEN_STAGES` is a
tuple so this is a one-line change, not a code revert; multi-stage still
works if revisited.

2026-09-15, a run of experiments chasing "PERFECT doesn't reliably recur
past the post-widen bar", in order tried:

1. `--widen-after-perfects 3` (train_iqn_gpu.py) -- require 3 cumulative
   tier-0 evals under the pre-widen window before actually widening,
   instead of firing on the first one, so the trigger reflects a reliably
   clean policy rather than a lucky one-off. Found and fixed a real bug
   along the way: the PERFECT that finally satisfies the gate is not
   necessarily a `*_best` win under the strict eval-distance ratchet (an
   earlier confirmation can have higher `ev`, same tier, and win the
   ratchet instead), so `stale["since"]` stayed anchored to that earlier
   eval -- `--stop-after-stale`'s clock was then already mostly spent by
   the time the harder post-widen bar even started (observed: anchored at
   it725, widening didn't fire until it1100, leaving ~225 of the intended
   600 iterations). Fixed: `stale["since"]` now resets to the actual widen
   event. Confirmed the fix works (a rerun got the full 600 post-widen
   iterations) -- and with that confirmed, PERFECT still never recurred
   once in that window, reach held around 13-14k m the whole time if
   anything trending down. Strong evidence the bottleneck was never budget.
2. `RL_RL_LAP_OFF_TOL` (training's clean-lap-bonus tolerance) taken to 0,
   matching `RL_OFF_PERFECT_TOL` (eval's selection tolerance, also 0) --
   the two had drifted apart since 2026-09-13 (train "clean" <= 6 ticks,
   eval PERFECT == 0), so training had no gradient rewarding zero over six.
   Result: statistically indistinguishable from not making the change --
   eval reach within 100 m of a matching earlier run at every iteration
   compared.
3. `RL_LAP_OFF_DECAY_CAP` -- replaced the binary tolerance with a
   continuous falloff (`max(0, 1 - off_steps/12)`) instead of a threshold
   anywhere, on the theory that ANY fixed cutoff (0, 4, or 6) has the same
   all-or-nothing-bet problem, just relocated. Also statistically
   indistinguishable from (2) and from doing nothing.
4. `RL_DIRTY_LAP_PROGRESS_MULT` -- a DENSE version, multiplying the
   per-step progress reward (not just the once-per-lap bonus) by 0.3 once
   a lap had gone off-track even once, on the theory that the sparse lap
   bonus was simply too rare an event to shape steering/braking decisions
   either way (consistent with (2) and (3) both doing ~nothing). This one
   diverged -- worse than the confirmed recipe, not better.

**Reverted all four (2026-09-15), explicit request**: GPU budget was
nearly gone and none had a confirmed positive result, one made things
worse -- rather than keep guessing at reward shapes with no budget left to
isolate each one properly, config.py/rlenv.py/vecenv.py/kernel_run.py are
back to the exact recipe that produced the current record. The
`--widen-after-perfects` CLI flag and its stale-clock-reset fix stay in the
code (default 1 = inert, matches original behaviour) in case a properly
isolated retest is worth it with a future GPU budget; `RL_LAP_OFF_DECAY_CAP`
and `RL_DIRTY_LAP_PROGRESS_MULT` were removed outright rather than left
wired in unused.

**Confirmed recipe, locked in 2026-09-15**: n-step 6, `RL_SHAPE_TO_RACELINE`
off, the 43-dim geometry-only observation (no reference speed), single-stage
widen on first PERFECT, `RL_POST_WIDEN_RAMP` flat/off, `RL_RL_LAP_OFF_TOL`
back to 6. This is exactly what produced Monza's 91.30 s record
(`Monza_v26k_scratch`) -- nothing else has beaten it as of 2026-09-15.

## Running by hand

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
