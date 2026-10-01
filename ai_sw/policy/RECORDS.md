# Track records

The current best flying-lap time per circuit, and what produced it. Update
this file whenever a new checkpoint is deployed (replaces `assets/policies/
<Circuit>.npz`) -- whether that's done by `train_track.bat`'s own deploy
prompt or by hand.

Recipe/settings are only noted here when they differ from the standard v26k
recipe (see `ai_sw_gpu/kaggle/README.md`). "Checkpoint" points into
`ai_sw/policy/<Circuit>/`, where the full `.npz`/`_state.pt` and the Kaggle
log live.

| Circuit | Best lap | Checkpoint | Date | Notes |
|---|---|---|---|---|
| Melbourne | 96.40 s | `Melbourne/Melbourne_v26_event20_best.npz` | 2026-09-20 | v26k + --off-event-cost 20 + eps anneal (--anneal-iters 200 --anneal-eps 0.005 --anneal-lr -1) |
| Montreal | 81.70 s | `Montreal/Montreal_v26_best_best.npz` | 2026-09-19 | v26k recipe |
| MexicoCity | 85.00 s | `MexicoCity/MexicoCity_v26_ramp20_best.npz` | 2026-09-19 | v26k recipe |
| Hockenheim | 86.40 s | `Hockenheim/Hockenheim_v26_best_best.npz` | 2026-09-19 | v26k recipe |
| Catalunya | 91.70 s | `Catalunya/Catalunya_v26_best_best.npz` | 2026-09-19 | v26k recipe |
| Budapest | 88.15 s | `Budapest/Budapest_v26_best_best.npz` | 2026-09-19 | v26k recipe |
| BrandsHatch | 67.00 s | `BrandsHatch/BrandsHatch_v26_best_best.npz` | 2026-09-19 | v26k recipe |
| Austin | 110.45 s | `Austin/Austin_v26_best_best.npz` | 2026-09-19 | v26k recipe |
| IMS | 46.60 s | `IMS/IMS_v26_best_best.npz` | 2026-09-19 | v26k recipe |
| Nuerburgring | 99.65 s | `Nuerburgring/Nuerburgring_v26k_best_best.npz` | 2026-09-16 | v26k recipe |
| Spa | 119.60 s | `Spa/Spa_v26k_best_best.npz` | 2026-09-15 | v26k recipe |
| Monza | 91.30 s | `Monza/Monza_v26k_scratch_best.npz` | 2026-09-14 | v26k recipe |
| Silverstone | 101.75 s | `Silverstone/Silverstone_v26k_best_best.npz` | 2026-09-16 | v26k recipe |

## How to update

1. Run `ai_sw_gpu\kaggle\train_track.bat <Circuit>` (or push/fetch/trace by
   hand -- see the Kaggle README).
2. If the traced lap time beats the row above, deploy it (the .bat asks
   before touching `assets/policies/<Circuit>.npz`).
3. Update that circuit's row here: lap time, checkpoint filename (under
   `ai_sw/policy/<Circuit>/`), today's date, and a one-line note **only** if
   something about the recipe was non-default for this run (a config.py/
   rlpolicy.py change, a non-default CLI flag, etc.) -- if it was the plain
   v26k recipe, leave Notes as "v26k recipe" or blank.
4. This table is the CURRENT-best snapshot, not a history -- the full
   experiment history for a circuit is in its Kaggle log
   (`ai_sw/logs/<Circuit>/`) and, for Spa specifically, in this project's
   memory file (`ai_sw_gpu/kaggle/README.md`'s recipe section covers what
   was tried and rejected).
