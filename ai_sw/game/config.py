"""Central tuning block for the FORMULA-AI racing prototype.

Physics follows the classic dynamic *bicycle model* described in Marco
Monster's "Car Physics for Games" and implemented in spacejack/carphysics2d,
with parameter magnitudes cross-checked against the CommonRoad / f1tenth_gym
single-track model. Units are SI (metres, seconds, newtons, kilograms).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT = Path(__file__).resolve().parents[1]
#: The circuits (f1tenth_racetracks, MIT): shipped with the game under
#: ``data/``; the old place beside the NEAT project is kept as a fallback, and
#: ``FORMULA_AI_TRACKS`` points anywhere else.
TRACK_DB = next((p for p in (
    *([Path(os.environ["FORMULA_AI_TRACKS"])] if os.environ.get("FORMULA_AI_TRACKS") else []),
    _PROJECT / "data" / "f1tenth_racetracks",
    REPO_ROOT / "Neural_Network_NEAT-master" / "new" / "f1tenth_racetracks-main",
) if p.is_dir()), _PROJECT / "data" / "f1tenth_racetracks")
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"

DEFAULT_TRACK = "Monza"

# ---------------------------------------------------------------------------
# World / track geometry
# ---------------------------------------------------------------------------
# The F1TENTH database is a scaled model of the real circuits. Fitting stored
# centreline length against published lap distances over 10 circuits gives a
# median factor of 12.67 -- at this scale Monza comes out 5799 m against the
# real 5793 m, Silverstone 5885 m against 5891 m.
TRACK_SCALE = 12.67

# Per-circuit refinement: each entry is (published lap length) / (stored
# centreline length), so every track comes out at its real distance and lap
# times are directly comparable with the real thing.
TRACK_SCALE_BY_NAME = {
    "Austin": 13.094, "BrandsHatch": 10.969, "Budapest": 10.882,
    "Catalunya": 11.218, "Hockenheim": 12.711, "IMS": 13.726,
    "Melbourne": 11.129, "MexicoCity": 12.067, "Montreal": 15.299,
    "Monza": 12.986, "MoscowRaceway": 12.179, "Nuerburgring": 11.540,
    "Oschersleben": 14.177, "Sakhir": 12.247, "SaoPaulo": 12.502,
    "Sepang": 11.382, "Shanghai": 10.954, "Silverstone": 12.865,
    "Sochi": 12.609, "Spa": 12.632, "Spielberg": 12.577,
    "YasMarina": 13.268, "Zandvoort": 10.978,
}

# Width is NOT taken from the scale above: the source data uses generous
# margins meant for 1:10 RC cars, which would give a ~28 m wide circuit. Real
# F1 tracks run 12-15 m, and a narrow track is the single strongest visual cue
# for speed, so the stored width profile is renormalised onto this range.
TRACK_WIDTH_MEAN = 13.5
TRACK_WIDTH_VARIATION = 0.30   # how much of the stored width variation to keep

RUNOFF_WIDTH = 13.0            # asphalt edge -> wall, on a straight
# A corner gets more, and gets it towards the exit -- that is the direction
# a car leaves the circuit in, and where a real one has acres of asphalt.
# Nothing clamps this any more: where two legs of a chicane both ask for
# acres, their run-off simply merges and the barrier between them is dropped
# (Track.wall_valid), which is how a real chicane is fenced -- one wall round
# the whole complex, open asphalt inside it.
RUNOFF_CORNER_EXTRA = 105.0    # extra metres on the outside of the worst bend
RUNOFF_CORNER_RADIUS = 320.0   # what counts as a bend for run-off
RUNOFF_SHARP_RADIUS = 60.0     # a bend this tight gets the full extra
RUNOFF_ENTRY_LOOK = 320.0      # metres of approach that set the entry speed
RUNOFF_ENTRY_RADIUS = 900.0    # an approach this open counts as flat out
KERB_WIDTH = 1.1
KERB_CURVATURE_RADIUS = 400.0  # a kerb is drawn where the radius drops below this

# ---------------------------------------------------------------------------
# Vehicle: chassis
# ---------------------------------------------------------------------------
CAR_MASS = 1180.0
CG_TO_FRONT = 1.25             # lf
CG_TO_REAR = 1.40              # lr
WHEELBASE = CG_TO_FRONT + CG_TO_REAR
CG_HEIGHT = 0.42               # low, sports-car

# Body box, used for wall contact. Measured off the rendered shell (4.45 x
# 2.41 m at race scale); the overhangs are whatever the wheelbase does not
# cover, split evenly, so changing WHEELBASE cannot leave these behind.
CAR_BODY_LENGTH = 4.45
CAR_BODY_WIDTH = 2.41
BODY_OVERHANG = (CAR_BODY_LENGTH - WHEELBASE) / 2.0
BODY_TO_FRONT = CG_TO_FRONT + BODY_OVERHANG     # 2.15 m ahead of the CG
BODY_TO_REAR = CG_TO_REAR + BODY_OVERHANG       # 2.30 m behind it
BODY_HALF_WIDTH = CAR_BODY_WIDTH / 2.0
# Half the axle track. The tyres sit inboard of the bodywork, and this is what
# track limits are judged at -- the rule is "a wheel inside the line", not "the
# bodywork inside the line".
WHEEL_HALF_TRACK = 0.86 * BODY_HALF_WIDTH
# Yaw inertia. f1tenth's fitted car sits at I / (m * L^2) = 0.185.
YAW_INERTIA = 0.19 * CAR_MASS * WHEELBASE ** 2

GRAVITY = 9.81

# ---------------------------------------------------------------------------
# Vehicle: tyres
# ---------------------------------------------------------------------------
# CORNER_STIFFNESS sets where each axle's tyre peaks: peak slip = mu / C.
# Rear stiffer than front => the rear peaks later => mild, stable understeer.
#
# Racing slicks are far stiffer than the road-car figures the reference
# implementations use (carphysics2d 5.0/5.2, f1tenth 4.7/5.5). A stiffer tyre
# reaches a given lateral force at a smaller slip angle, so the car both
# responds sooner AND slides less -- raising these improved turn-in time and
# peak sideslip at the same time, rather than trading one for the other.
CORNER_STIFFNESS_FRONT = 8.4
CORNER_STIFFNESS_REAR = 10.2
TYRE_GRIP = 2               # mu, racing slick

# --- Pacejka "Magic Formula" tyre ---------------------------------------
#   F = D sin(C atan(B a - E (B a - atan(B a))))
# Pacejka's formula is published mathematics, so using it carries none of the
# licensing baggage of lifting code out of a GPL simulator.
#
# It replaces the previous linear-then-clamped curve. That one rose straight to
# the grip limit and then stayed flat for ever, so exceeding the limit cost
# nothing and a slide had no distinct feel. A real tyre peaks and then gives
# force *back* -- which is what makes the limit findable and a slide something
# you feel rather than read off a number.
#
# B is derived so the peak lands at exactly mu / CORNER_STIFFNESS, i.e. where
# the old model's peak was. That keeps every downstream formula that reasons
# about the peak (steer_limit, the steering assist, the autopilot) valid, and
# changes only the shape of the curve.
PACEJKA = True
PACEJKA_C = 1.45               # shape factor (lateral: 1.3-1.8)
PACEJKA_E = 0.0                # curvature; < 1 or there is no real peak
# A bicycle model with equal front/rear grip is exactly neutral-steering -- it
# sits on the knife edge between under- and oversteer and spins at the smallest
# provocation. Real GT cars run wider rear tyres for the same reason; this
# buys a stable understeer bias, which is what an exhibition car wants.
REAR_GRIP_BIAS = 1.06
HANDBRAKE_GRIP_SCALE = 0.42    # rear grip multiplier while the handbrake is down
OFF_TRACK_GRIP_SCALE = 0.42    # grass
KERB_GRIP_SCALE = 0.88

WEIGHT_TRANSFER = 0.22         # how much longitudinal accel shifts axle load

# Slip angles are meaningless at a crawl (atan2 with vx ~ 0 explodes), so the
# model falls back to kinematic steering. A hard switch makes the car twitch as
# it crosses the threshold, so blend across a window instead.
BLEND_SPEED_LO = 0.8           # m/s, pure kinematic below
BLEND_SPEED_HI = 6.0           # m/s, pure dynamic above

# ---------------------------------------------------------------------------
# Vehicle: aero
# ---------------------------------------------------------------------------
# Drag: F = DRAG_COEFF * v^2   (0.5 * Cd * A * rho, Cd~0.70 A~1.8 m^2)
DRAG_COEFF = 0.5
ROLL_RESIST = 12.0             # F = ROLL_RESIST * v
# Downforce: F = DOWNFORCE_COEFF * v^2, split front/rear. This is what makes a
# racing car planted at speed and loose at low speed.
DOWNFORCE_COEFF = 2.6
DOWNFORCE_FRONT_BIAS = 0.42

# ---------------------------------------------------------------------------
# Vehicle: drivetrain
# ---------------------------------------------------------------------------
# Power-limited above the traction limit: F = min(F_max, P / v). Gives the
# realistic "pull falls away with speed" instead of constant acceleration.
ENGINE_POWER = 430_000.0       # W (~575 hp)
ENGINE_FORCE_MAX = 11_500.0    # N, low-speed traction/torque ceiling
REVERSE_FORCE = 4_000.0

# --- traction control -------------------------------------------------
# Without it, a 575 hp rear-drive car at full throttle puts the rear tyres on
# their traction limit, which by the friction circle leaves ~10% of their grip
# for cornering -- so it snaps into a spin below about 100 km/h and never
# recovers. Every GT3 car runs TC for exactly this reason. It reserves part of
# the rear friction budget for steering, and cuts further once the back steps
# out.
TRACTION_CONTROL = True
TC_SAFETY = 0.90               # fraction of the *spare* rear grip TC will use
TC_SLIP_DEG = 6.0              # rear slip angle where TC starts cutting power
TC_SLIP_FULL_DEG = 15.0        # ...and where it cuts hardest
TC_MIN_POWER = 0.25            # never cut below this fraction

# --- ABS ---------------------------------------------------------------
# The mirror image of TC. Full braking otherwise spends the front axle's whole
# friction budget, leaving almost none for cornering, so the car refuses to
# turn while slowing down.
ABS_ENABLED = True
ABS_SAFETY = 0.92              # fraction of the *spare* axle grip ABS will use
# ...but never give cornering the whole budget. Computing brake force as
# "whatever is left after the lateral demand" drops it to exactly zero once a
# tyre is near its cornering limit, so braking into a corner stopped slowing
# the car at all and it just washed wide -- which from the driver's seat reads
# as the steering not working. A real ABS modulates; it does not switch off.
#
# The floor is not a constant, because a constant silently decides how the car
# trail-brakes and leaves the driver out of it. Measured with a fixed floor at
# 200 km/h, brake and steering both pinned: 0.40 gave 0.85 g of decel against
# 1.75 g lateral (turns beautifully, barely slows), 0.70 gave 1.26 g / 1.35 g.
# Both are defensible; neither is something a single number should be choosing.
#
# So the reservation follows what the driver is actually asking for. Hard on
# the brakes with a small steering input reserves most of the circle for
# slowing; a dab of brake mid-corner reserves almost none and lets the tyre
# corner. The friction circle still caps the total either way -- this only
# decides who gets the scarce grip when both inputs want it.
ABS_BRAKE_SHARE_MIN = 0.22     # floor when the driver is mostly steering
ABS_BRAKE_SHARE_MAX = 0.88     # floor when the driver is mostly braking

# --- steering assist ---------------------------------------------------
# A keyboard is always at full deflection, so the front axle would otherwise
# sit permanently past its grip peak and wash out whenever a direction key is
# held. The assist eases the commanded angle back towards the peak.
STEER_ASSIST = True
# The assist must never fight a correction. It keys off total front slip, which
# is large whenever the chassis is sliding *regardless* of steering, so without
# this it removed authority exactly when the driver was counter-steering -- the
# wheel sawed back and forth and the slide never ended.
ASSIST_SKIP_WHEN_CORRECTING = True

# --- stability control -------------------------------------------------
# Damps yaw the car is carrying beyond what the steering actually asked for,
# which is what makes a slide decay instead of settling into a steady spin.
# Only oversteer is damped; understeer is left alone, because inventing yaw the
# tyres are not generating would be a lie.
ESC_ENABLED = True
ESC_GAIN = 3.2                 # 1/s, how fast excess yaw is bled off
ESC_DEADBAND = 1.15            # act above this multiple of the commanded yaw
ASSIST_SLIP_ALLOW = 0.95       # slip allowed, as a multiple of the peak
ASSIST_STRENGTH = 0.60         # how much of the excess angle is removed

BRAKE_FORCE = 21_000.0         # ~1.8 g unloaded, more with downforce
BRAKE_BIAS_FRONT = 0.62        # share of braking at the front axle
HANDBRAKE_FORCE = 7_000.0
IDLE_DRAG = 900.0              # engine braking when off throttle

# ---------------------------------------------------------------------------
# Vehicle: steering
# ---------------------------------------------------------------------------
# Generous at a crawl: a race car's rack is ~30 deg, but this is also the
# lock available for manoeuvring, spinning round after a mistake, and threading
# a slow chicane -- all of which want more. It is only reachable below
# STEER_LIMIT_FREE_SPEED; above that the physics-derived limit takes over.
MAX_STEER = math.radians(45.0)
# The steering *input* is rate limited, so a keyboard tap can no longer snap
# the wheels to full lock. This is the main cure for twitchy handling.
# This was the single largest source of steering lag: at 2.0 it took a full
# half-second of input before the wheel could even reach its usable angle. It
# was set that low back when full stick meant ~13x the angle the tyres could
# use; now that the lock is capped by STEER_LIMIT below, a fast stick simply
# reaches the grip-optimal angle sooner and cannot make the car twitchy.
STEER_INPUT_RATE = 5.7         # units/s towards the held direction
STEER_RETURN_RATE = 8.5        # units/s back to centre when released
# ...but wound on more gently the faster you are going, so a held key gives
# fine, progressive input at speed instead of arriving at the stop at once,
# while slow corners keep the quick rate above.
#
# The falloff is deliberately NOT linear in speed. Linear starts taking input
# away immediately, so the car already feels dulled at 100 km/h where it should
# still be sharp, and then has nothing left to give when it really matters.
# Instead the rate is untouched up to the knee and then falls away on a
# smoothstep, which is flat at both ends and steep in the middle -- so the
# change is something you drive into rather than a threshold you cross.
STEER_RATE_KNEE = 130 / 3.6    # m/s, full rate at or below this
STEER_RATE_FULL = 290 / 3.6    # m/s, the drop below is fully applied here
STEER_RATE_SPEED_DROP = 0.64   # fraction of the rate removed at STEER_RATE_FULL
# Rate limit on the front wheels themselves (steering rack speed).
STEER_RACK_RATE = 5.2          # rad/s

# Available lock is derived from physics rather than from a hand-picked curve:
# at speed v the tyres can only use delta ~= atan(L * a_lat_max / v^2), so full
# stick is mapped onto that angle. Without this, full lock at 200 km/h asks for
# 13x more steering than the front axle can deliver -- the first few percent of
# input does everything and the rest is scrub, which is exactly what "too
# sensitive" feels like.
# Kept close to 1.0: with a keyboard the stick is always fully deflected, so a
# generous over-range means the front is permanently past its grip peak.
STEER_LIMIT_MARGIN = 1.12      # >1 so you can still provoke understeer/slides
# ...and more margin than that below the knee, on the same smoothstep the input
# rate uses, so the two speed-dependent effects share one shape.
#
# This is worth doing only because the grip-optimal lock sits on a broad
# plateau: measured at 100 km/h, 6 / 8 / 10 degrees give 43.4 / 44.3 / 44.2
# deg/s of yaw. So handing back a quarter more lock in the mid range costs
# under 1% of turn rate and buys a steering feel that stays sharp to 130 and
# then falls away, instead of collapsing before 100 and staying flat.
# The extra is gone by STEER_MARGIN_FULL, so nothing changes at racing speed.
STEER_MARGIN_LOW_EXTRA = 0.30
STEER_MARGIN_KNEE = 130 / 3.6  # m/s, full extra at or below this
STEER_MARGIN_FULL = 230 / 3.6  # m/s, no extra at or above this
STEER_LIMIT_FLOOR = math.radians(1.6)
STEER_LIMIT_FREE_SPEED = 10.0  # m/s below which full lock is always available

MAX_SPEED = 88.0               # m/s, used only for normalising the above
LOW_SPEED_CUTOFF = 0.6         # below this the car is parked

# ---------------------------------------------------------------------------
# The AI opponent, and the out lap
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The racing line (game/raceline.py)
# ---------------------------------------------------------------------------
# Lines are solved offline and loaded at race start: the least-squares solve
# alone takes tens of seconds, which is not something to do while someone is
# waiting at an exhibition machine.
RACELINE_DIR = ASSET_DIR / "racelines"
# Metres of track the imported f1tenth line is smoothed over, to take out the
# noise its source track widths carry.
F1TENTH_SMOOTH = 30.0

# ---------------------------------------------------------------------------
# Reinforcement learning (game/rlenv.py, tools/train_rl.py)
# ---------------------------------------------------------------------------
# Costs are per second of it happening, scaled by speed where speed is what
# makes it bad. Progress along the line is the reward; these only have to make
# the shortcuts less attractive than the road.
# Charged once when an episode ends in failure rather than at the time limit.
# Losing the rest of the episode is the real cost, but the discount horizon is
# 16 s and an episode is 45, so without this the tail of a crash is invisible.
SHOW_LINE_MARKERS = False   # debug: draw the centreline and racing line on the road
SHOW_DISTANCE_HUD = False   # debug: 'N m / total m' readout, for lining up with training logs
SHOW_KEY_HINTS = False      # the keycap legend along the bottom edge in-race
HUD_MARGIN = 0.022     # clear space between a panel and the screen edge

#: Which observation the discrete IQN driver uses -- see rlpolicy.RL_OBS.
#: "rangefinder": white-line rangefinders + curvature + reference speed ahead
#: (v14, the direct "how much room do I have" signal). "lookahead": the
#: original centreline points in the car frame.
RL_OBS = "lookahead"      # v14 rangefinder obs was worse (rec 8/ep, 170 km/h); reverted

RL_OFF_TRACK_COST = 0.20       # per second beyond track limits, times speed.
                              # v17: the clean-lap bonus (zeroed for a dirty
                              # lap) does most of the work of forcing PERFECT;
                              # this is a moderate backstop with a gentle ramp.
                              # 2026-09-13: 0.35 -> 0.20. This fee is small
                              # next to a lap's gross reward components, but
                              # RL_RL_LAP_OFF_TOL disqualifies the ENTIRE
                              # clean-lap bonus (up to ~65 on Spa) the moment
                              # a lap has more than a couple of off-track
                              # ticks -- so the marginal cost of a single
                              # braking-precision mistake (this fee + the lost
                              # bonus) already exceeds a whole lap's net
                              # reward, before scale even ramps up. That
                              # all-or-nothing structure, not this per-second
                              # fee, is what should carry the "must be
                              # PERFECT" pressure; lowering the fee itself
                              # softens the redundant, doubly-harsh part so a
                              # near-limit braking experiment that clips the
                              # edge isn't punished twice as hard as it needs
                              # to be to still net-lose against a clean lap.
RL_OFF_TRACK_COST_RAMP = ((0, 1.0), (500_000, 1.0), (2_000_000, 1.6))
                              # 2026-09-13: was ((0,1),(300k,1),(1.5M,2.0)),
                              # then briefly (500k,1.0),(3.5M,1.6) -- that
                              # v26e4 run showed loss trough right at first
                              # PERFECT (it950, ~2.55M decisions, loss 0.123)
                              # and then climb back to 0.18 by it1650 while
                              # rec/off stayed flat near zero -- a moving-
                              # target artifact, not a policy regression: the
                              # ramp was STILL rising (1.5->1.6) a full 1M
                              # decisions after the policy had already found
                              # its clean line, so the value network kept
                              # re-chasing a shifting reward floor for no
                              # benefit. Ending the ramp at 2.0M -- before the
                              # ~2.5M where PERFECT typically first appears --
                              # means the reward is flat by the time there is
                              # anything to disturb, so loss should hold
                              # rather than climb post-PERFECT. With
                              # RL_EDGE_MARGIN=0 the car runs the legal width
                              # right up to the paint on entry -- a late-
                              # braking error has no lateral margin to absorb
                              # it and lands on this fee at full scale -- but
                              # RL_RL_LAP_OFF_TOL / the clean-lap bonus still
                              # carry the "must be clean" pressure, this ramp
                              # only needs to hold still once that is found.
#: Multiplier on RL_OFF_TRACK_COST, keyed to decisions seen. train_iqn.py reads
#: this and passes the scale per rollout; RaceEnv.step multiplies it in. Flat
#: ((0,1),(1,1)) disables the ramp.

# How hard the reference line pulls. The reference is the *centreline*, and a
# racing line is by definition several metres off it, so any pull at all is
# pulling against the optimum -- it is here to stop the car drifting to a
# barrier for no reason, not to say where to drive. At this weight the car
# keeps 85% of its progress even at the edge of the road, which is far less
# than a proper line is worth, so the search can still find one.
#
# The imported f1tenth line was tried as the reference and was worse than
# nothing: solved for a track twice as wide, it came out *tighter* than the
# centreline through 40% of the corners on every circuit, and multiplying the
# reward by accuracy against it paid the policy to drive a bad line well.
RL_LINE_TOLERANCE = 2.0        # (kept for the debug overlays only)

# v12: point the potential attractor (and the observation's reference-speed
# profile) at this project's *own* optimised raceline -- assets/racelines/
# <Circuit>.npy, from raceline.refine's CMA-ES on simulated lap time -- rather
# than the centreline. Unlike the f1tenth line above this one is scale-correct
# and bounds-constrained, and it is pulled inward by RL_RACELINE_SAFETY so the
# whole car body clears the white line: verified on Monza, all four wheels
# inside with margin. This turns the line term from a headwind (pull to centre,
# away from the fast line) into a tailwind, without letting it prescribe a line
# that clips a limit. The edge / off-track terms still measure from the
# centreline, where the widths are defined. False -> centreline, as before.
RL_SHAPE_TO_RACELINE = False
RL_RACELINE_SAFETY = 0.20     # metres the saved raceline is pulled in so every
                              #  wheel stays inside the white line by this much

# --- Linesight-style reward -------------------------------------------------
# Reward per metre advanced along the centreline, and time penalty per second.
# Lifted straight from Linesight (5/500 and 6/5000 per ms = 1.2 per s). Over a
# fixed-length episode the time term is nearly constant, so the return tracks
# distance covered, i.e. average speed.
RL_PROGRESS_W = 5.0 / 500.0
RL_TIME_W = 6.0 / 5000.0 * 1000.0     # per second

# 2026-09-15, tried and reverted: RL_DIRTY_LAP_PROGRESS_MULT, a multiplier
# on RL_PROGRESS_W once a lap had gone off-track even one tick (dense,
# every decision, on the theory that the sparse once-per-lap clean-lap
# bonus below was too rare to actually shape steering/braking). Monza
# diverged worse under it, not better -- reverted rather than left wired in
# at some inert value, since no confirmed replacement exists yet.

# --- v17 raceline task -----------------------------------------------------
# "linesight": the v10-v16 reward -- sparse-ish progress minus time, the policy
# has to *discover* how fast to take each corner, and the fast/clean balance is
# tuned through the off-track fee (unstable: plateaus at 96.5 s clean or goes
# ~90 s dirty). "raceline": add two things the sparse reward lacked --
#   * a DENSE per-step speed-tracking reward against an ambitious speed profile
#     (the deployed raceline's curvature at RL_RL_SPEED_PACE): "be going THIS
#     fast here", a gradient from step one for braking zones and straights;
#   * a LAP-TIME BONUS awarded only on a clean finish-line crossing
#     (RL_RL_LAP_BASE - RL_RL_LAP_W * seconds, zero if the lap had more than
#     RL_RL_LAP_OFF_TOL off-track steps) -- makes "fast AND clean" the literal
#     payoff, not an emergent balance.
# The centreline stays the reference for progress and the observation; only the
# reward changes.
RL_TASK = "raceline"         # v20: v18's EXACT Linesight reward (progress -
                            #  time + centre-pull shaping, all restored below)
                            #  PLUS just one thing -- a steep clean-lap-TIME
                            #  bonus. v19 tried zeroing the centre-pull + a
                            #  dense speed target and converged clean-but-SLOW
                            #  (181 km/h, stuck): the dense target became a
                            #  comfortable attractor. v20 removes that
                            #  (RL_RL_SPEED = 0) and instead makes "once clean,
                            #  every second faster is worth RL_RL_LAP_W" an
                            #  unambiguous, large per-lap reward.
RL_RL_SPEED = 0.0            # v26: BELL OFF entirely. v21/v22/v25b proved any
                            #  monotone (no-overspeed-penalty) dense speed term
                            #  diverges the IQN at gamma->1; v19/v24's two-sided
                            #  bell converges clean but caps at the 30%-
                            #  conservative v_ref (~105 s). v26 keeps v18's
                            #  EXACT reward (progress-time + full shaping, which
                            #  is self-limiting and gave 94.45 s) and adds ONLY
                            #  the small linear term below as a gentle "a bit
                            #  faster is a bit better" nudge on the proven base.
RL_RL_SPEED_SIG = 9.0        # v25b: width of the ONE-SIDED-BELOW bell (m/s).
                            #  Below v_ref the reward falls off over ~9 m/s;
                            #  at or above v_ref it is flat at RL_RL_SPEED (no
                            #  overspeed penalty -- v_ref is a 30%-conservative
                            #  profile, not a real limit, see rlenv note).
RL_RL_SPEED_CAP = 1.0        # (legacy, unused)
RL_RL_SPEED_LIN = 0.022      # v26-retry (2026-09-12): re-enabled. The original
                            #  v26 verdict ("diverges, rec stuck ~7, loss
                            #  0.07->0.21") was reached at it 373 using an
                            #  invalid absolute-loss threshold -- v27 (v18's
                            #  exact reward, run to completion) later proved
                            #  that v18 ITSELF shows rec 6-7 / loss up to 0.46
                            #  during the normal gamma-ramp rough patch
                            #  (it~558-930), self-correcting only after gamma
                            #  hits 1.0 (~it930+). v26 was killed at it373 --
                            #  well before that window even opens. Retrying to
                            #  completion, judged by direct comparison against
                            #  v18's own log at matching iterations (not an
                            #  absolute loss number). If it still diverges past
                            #  v18's resolution point (~it1100-1150), the
                            #  original v26 verdict stands confirmed for real.
RL_RL_SPEED_LIN_CAP = 1.0    # v/MAX_SPEED rarely exceeds 1; clip in case
RL_RL_SPEED_PACE = 1.00      # v25: v_ref = the physics grip-limited profile,
                            #  no inflation. The bell's peak sits exactly at
                            #  the grip limit so overspeeding a corner is a
                            #  real penalty. The v24 pace ramp is retired (it
                            #  inflated corner targets past reachability and
                            #  killed the corner discipline); PACE_SCHED in
                            #  train_iqn is now flat at 1.0.
# Clean-lap-time bonus, added once on a finish-line crossing that closed a
# whole lap with <= RL_RL_LAP_OFF_TOL off-steps and no recovery. v20 makes it
# STEEP: every second cut off the lap is worth RL_RL_LAP_W = 2.5 -- about
# twice the base per-second time penalty (1.2), so once the policy is clean,
# going faster is a clear, large net gain rather than the marginal gamble it
# was for v18 (progress-minus-time only).
#
# The BASE is computed per circuit (rlenv.RaceEnv / gpuenv.VecRaceEnv set it
# from the track's own analytic modelled lap time, RL_RL_LAP_W * model_lap),
# not read from here. 2026-09-13: found that a flat BASE=300 gave a
# break-even of exactly 300/2.5=120.0 s -- which is Monza's OWN modelled lap
# time (120.6 s) to within 0.5 %, not a coincidence but an un-generalised
# calibration. It meant the bonus was healthy for Monza (94 s clean laps,
# well under 120) but effectively ZERO for Spa, whose clean laps (~122 s)
# sit ABOVE Spa's fixed 120 s break-even even though they are well inside
# Spa's own modelled lap (148.9 s) -- the "make fast-and-clean the literal
# payoff" mechanism that drove v27 past v18 on Monza was providing no
# gradient on Spa at all. Setting BASE = W * model_lap replicates the exact
# same design (break-even at the conservative modelled pace, grows as the
# policy beats it) for whichever circuit is actually loaded.
RL_RL_LAP_W = 2.5
#: Multiplier on RL_RL_LAP_W, keyed to decisions seen -- same interpolation
#: mechanism as RL_OFF_TRACK_COST_RAMP (train_iqn.py's LAP_W_SCHED / _interp),
#: passed into RaceEnv.step()/VecRaceEnv.step() as lap_w_mult. 2026-09-13:
#: added because once a policy is reliably clean the flat W=2.5 rate gives a
#: very diffuse "shave more time" gradient -- a lap a few tenths faster is a
#: small fraction of the total per-lap return at gamma~1, easy to lose in
#: value-estimation noise. Held at 1.0 (i.e. inert) through the same window
#: the off-track-cost ramp needs to find a clean line, then stepped up to
#: sharpen the reward for marginal speed gains once that is already found --
#: it should never fire before a policy can plausibly be clean, since a
#: steeper lap bonus with no clean baseline yet is just a bigger dangling
#: carrot for a policy that cannot reach it. Flat ((0,1),(1,1)) disables it.
RL_LAP_W_RAMP = ((0, 1.0), (1, 1.0))
                              # 2026-09-13: tried ((0,1),(2.0M,1),(2.4M,1.8))
                              # on v26e5 -- best (9256T0) came in BELOW
                              # v26e4's 9273 and kept eroding after the boost
                              # (9256->9249->9234->9184->9138->9121->9106
                              # ->9071 as it approached its own stale-stop),
                              # so a steeper post-PERFECT lap bonus made
                              # things worse here, not better. Flattened back
                              # to inert pending a different idea -- see
                              # RL_POST_WIDEN_RAMP below for that idea.

# 2026-09-14: the "different idea". RL_LAP_W_RAMP's failure above raised only
# the reward for speed, not the price of risk -- so the policy pushed closer
# to the track limit for a bigger bonus, and the all-or-nothing clean-lap
# eval (any off-track tick zeroes the whole bonus) punished the resulting
# off-track ticks harder than the extra speed paid for. Motivated by Monza's
# episode-widening run (LONG_EPISODE_RATIO, rlenv.py; see the Kaggle README's
# episode-widening section): eval reach jumped from 7,711 m (the pre-widening
# PERFECT) to a steady 13,000-13,700 m within 100 iterations of widening --
# so the policy plainly COULD cover a full second lap at pace, it just
# couldn't yet do it clean, every single post-widening eval came back
# off-track or wall/recover. RL_POST_WIDEN_RAMP scales RL_OFF_TRACK_COST and
# RL_RL_LAP_W by the SAME factor instead of RL_RL_LAP_W alone, so the
# relative price of "clean" vs "fast" never shifts (whatever policy was
# already optimal stays optimal) -- only their combined signal grows
# relative to the unscaled per-step progress/time reward (RL_PROGRESS_W /
# RL_TIME_W), sharpening the "stay clean, get faster" gradient without
# incentivising more risk-taking near the limit. Keyed to decisions SINCE
# episode widening (train_iqn_gpu.py tracks the exact seen-count of the eval
# that triggered it), not decisions since training start, since widening is
# exactly the moment "clean" gets categorically harder (1 lap -> 2) and is
# when this extra push is wanted -- inert (1.0) the entire time before that,
# including through the ordinary pre-widening PERFECT climb. Flat
# ((0,1),(1,1)) disables it. Only wired into train_iqn_gpu.py (the GPU
# trainer used for real runs), not train_iqn.py.
#
# 2026-09-14: first guess was (0,1.0),(1_000_000,1.8) -- tested on Monza
# (with --post-widen-start-at-line, below) and it reproduced EXACTLY the
# moving-target failure mode RL_OFF_TRACK_COST_RAMP's own comment already
# diagnosed once: all 4 widened PERFECTs (it1075/1150/1175/1200, best lap
# 93.7 s) landed while this ramp was still climbing (roughly 1.3x-1.6x of
# its way to 1.8x, well short of flat), and the policy then went off-track
# on every single eval for the remaining 575 stale-stop iterations -- almost
# exactly the window where the ramp kept climbing the rest of the way to its
# 1.8x ceiling. First fix: shortened the window to 300k decisions, so it
# reaches flat well before ~400k (roughly where that run's first widened
# PERFECT appeared). Retested and confirmed on Monza: 93.30 s, beating the
# 93.60 s record outright.
#
# 2026-09-14, second pass: a fixed decision count is itself circuit-
# specific in exactly the way EPISODE_SECONDS_BY_CIRCUIT/LONG_EPISODE_RATIO
# already flagged once for episode_seconds (rlenv.py) -- 300k was tuned to
# ONE circuit's observed timing, and a track that takes meaningfully longer
# (or shorter) than Monza to re-clear the harder post-widening PERFECT bar
# has no reason to share it: too short just means extra sharpening tops out
# early (harmless, the same flat-and-safe state as before widening even
# starts), but too long reproduces the exact moving-target problem this was
# meant to fix, just later. So train_iqn_gpu.py now FREEZES the ramp at
# whatever value it has reached the moment a post-widening PERFECT is first
# actually seen (see post_widen_ramp_freeze in harvest()), rather than
# trusting a fixed window to land before that happens. The window below only
# controls how fast the ramp climbs while still searching -- it no longer
# needs to be calibrated per circuit, since freezing does the actual job now.
# 2026-09-14: re-enabled after --n-step=6 alone showed a genuinely faster
# line but zero widened-PERFECTs -- except the run that actually tested it
# (Monza_v26k_scratch, 91.30 s) turned out to have been pushed to Kaggle
# BEFORE this re-enable landed, so its one widened PERFECT (it725, 67,200
# decisions after widening) froze the ramp at 1.00x -- i.e. it was never
# actually active. That result says nothing about whether the ramp helps;
# it says n-step + the cleaner reward/observation alone can find a fast
# line, and that PERFECT recurring even once in 600 iterations can still be
# luck rather than a solved mechanism (this run had exactly one hit, then
# stale-stopped). 2026-09-15: OFF again (flat) while EPISODE_WIDEN_STAGES
# (rlenv.py) is tried instead -- a different lever on the same "PERFECT
# doesn't reliably recur past a hard difficulty jump" problem, but one that
# changes the TASK (smaller jumps) rather than the incentive (steeper
# reward). Keeping both off at once isolates which one, if either, is
# actually doing the work.
RL_POST_WIDEN_RAMP = ((0, 1.0), (1, 1.0))
# 2026-09-13: 3 -> 6 (0.05s -> 0.10s at 60Hz). Still well under
# RL_OFF_PERFECT_TOL=8 at the time (the eval-selection tolerance), so a
# genuinely dirty lap still couldn't count as clean or get selected -- this
# only stops a single-tick graze from disqualifying the whole clean-lap
# bonus outright, which was making a near-limit braking attempt an
# all-or-nothing bet against ~65 points of bonus for a 0.02s error.
# 2026-09-15: RL_OFF_PERFECT_TOL was separately cut to 0 that same day
# (2026-09-13), leaving this drifted from it (train "clean" <= 6 ticks, eval
# PERFECT == 0 ticks) -- tried closing that gap three different ways
# (6 -> 4, 6 -> 0, then a continuous off_steps decay replacing the binary
# cutoff entirely) and, separately, a dense per-lap progress penalty
# (RL_DIRTY_LAP_PROGRESS_MULT). None beat the recipe already confirmed to
# work (Monza's 91.30 s record, Monza_v26k_scratch) -- two were statistically
# indistinguishable from doing nothing and one diverged outright. Reverted
# to the exact confirmed value rather than keep guessing at reward shapes
# with no budget left to properly isolate each one.
RL_RL_LAP_OFF_TOL = 6        # off-steps a lap may have and still count as clean

#: The trainers' *_best selection tolerance: off-track physics ticks (60 Hz)
#: a 7-launch eval may show and still count as tier-0 PERFECT. This is a
#: SELECTION criterion only -- it never touches the training reward above,
#: so changing it cannot destabilise learning, only change which already-
#: trained checkpoint gets written out as *_best. 2026-09-13: raised from 2
#: to 8, then back to 0 -- explicit request to only ever select a checkpoint
#: with a literal zero-off-track eval as *_best, no tolerance. This was
#: already the case during the confirmed 91.30 s Monza run, so it stays.
RL_OFF_PERFECT_TOL = 0

# Potential-based line term: Phi = -K * clip(|offset|, LO, HI). Linesight uses
# K = 0.1 and accepts that it pulls the car off the racing line ("any pull at
# all is pulling against the optimum") because Trackmania's runoff is narrow.
# v19 zeroed this hoping for a wider racing line; instead the policy wandered
# and converged clean-but-SLOW (181 km/h). v20: back to Linesight's 0.10 --
# turns out this mild pull keeps the car on a consistent efficient line, which
# is what let v18 reach 215 km/h. The lap-time bonus is the speed driver now.
#: Of grid-start resets, the fraction drawn from a near-zero launch-speed
#: band instead of the full uniform(0, 0.95) range, and how wide that band
#: is. 2026-09-13: raised from the ~1-in-20 a plain uniform draw gave, after
#: the deployed policy was observed being visibly conservative off a literal
#: standing start -- plausibly undertrained on that exact narrow state.
RL_LAUNCH_STOP_FRAC = 0.35
RL_LAUNCH_STOP_MAX = 0.15

# --------------------------------------------------------------------------
# Adaptive (difficulty-weighted) scattered resets.
# --------------------------------------------------------------------------
# 2026-09-13: an alternative to training separate per-sector specialists and
# splicing them (tried, parked -- 2x-3x the GPU for a seam-consistency
# problem that has to be solved anyway). This keeps ONE policy training on
# the WHOLE lap the entire time -- no splice, no seam -- but skews where
# scattered resets land toward whichever stretch the policy is CURRENTLY
# failing on, tracked live from real off-track telemetry rather than a
# circuit-specific hardcoded sector. Generalises to any track for free: a
# circuit this has never seen starts uniform and re-weights itself as soon
# as the policy starts failing somewhere in particular.
RL_ADAPTIVE_BUCKETS = 40      # scattered resets are drawn from this many
                             #  equal-arclength bins instead of the raw
                             #  per-sample index -- fine enough to localise
                             #  a single corner, coarse enough that each bin
                             #  still gets visited often enough to keep its
                             #  failure-rate estimate from being pure noise.
RL_ADAPTIVE_EMA = 0.98       # per-iteration decay on each bin's tracked
                             #  off-track rate. High = slow to react but
                             #  stable; the bin only reweights once a
                             #  genuine, sustained pattern shows up, not one
                             #  unlucky rollout.
RL_ADAPTIVE_FLOOR = 0.30     # minimum share of the reset-weight mass kept
                             #  UNIFORM across all bins regardless of their
                             #  tracked difficulty, so an easy stretch never
                             #  drops to zero visits and quietly regresses
                             #  while training chases the hard one.

RL_LINE_K = 0.10           # v25: RESTORED to v18/Linesight. v24 (K=0) drove a
                           #  wider line that eval'd 10 s slower than v18 (105 s
                           #  vs 94.45 s) at the same cleanliness -- on this
                           #  Monza layout the tight line IS the fast line.
                           #  This mild pull toward shape_line is most of that
                           #  10 s. Bell + linear term supply SPEED; this
                           #  supplies the LINE; they don't overlap.
RL_LINE_LO = 2.0
RL_LINE_HI = 25.0

# Edge-margin term, also potential-based (same Ng-Harada-Russell guarantee, so
# it cannot move the optimum -- only front-loads the lesson). Phi rises with
# how much room the car has to the *nearest white line* and saturates once it
# is comfortably inside. Unlike the centreline term it is width-aware, and it
# keeps giving a gradient through the last metre before the edge -- exactly
# where the centreline term is already clipped flat and where "very slightly
# off the line" actually happens.
RL_EDGE_K = 0.090          # v26: back to v18's value -- v26 is "pure v18 reward
                           #  + linear term", full v18 shaping. Moot with
                           #  MARGIN 0 below (nothing left for it to scale),
                           #  kept only so a future non-zero margin has its
                           #  slope ready-configured.
RL_EDGE_MARGIN = 0.0      # 2026-09-13: 2.4 (v18) -> 1.0 -> 0.0. clip(edge-off,
                           #  0, MARGIN) is identically 0 at MARGIN=0 for any
                           #  position still on the track, so this term now
                           #  costs NOTHING anywhere inside the white line --
                           #  no shaping penalty for using any part of the
                           #  legally available track, full width, right up
                           #  to the line. RL_EDGE_OUT_K/CAP and the
                           #  off-track/recovery costs below are UNCHANGED:
                           #  those fire only once a wheel is actually past
                           #  the line, which is what still induces PERFECT.
# Beyond the white line the edge term above is flat zero, so there is no
# potential gradient pulling a car that has *just* stepped out back onto the
# road -- only the centreline term and the per-second fee, both weak in that
# first metre. This adds one: Phi keeps falling, linearly, with how far past
# the line the car is, capped so a big off nets a bounded penalty. Still a
# function of state alone -> still optimum-preserving. This is the term aimed
# squarely at "0 track-limit steps".
RL_EDGE_OUT_K = 0.40         # v14: back on (v11 value). v13 ran without it and
                              # drifted off the limit like v5.
RL_EDGE_OUT_CAP = 8.0         # metres past the line at which the pull saturates

# Soft barrier inside the free corridor (2026-09-19). The off-track test is
# "all four wheels past the white line", so the car CENTRE may sit up to ~1.04
# m past the line at no cost, and the fee above only exists beyond that cliff.
# Measured on Melbourne: a fast policy uses that corridor to within 0.0-0.3 m
# of the cliff, so ordinary lap-to-lap jitter (+-0.2 m) drops it over and
# PERFECT is intermittent -- and raising RL_OFF_TRACK_COST only moved the
# cushion from ~0.1 to ~0.4 m. This adds a REAL (not potential-based) cost that
# starts RL_EDGE_WALL_START metres past the white line and grows as the square
# of the extra depth, in the same per-second x speed units as the off-track
# fee: cost = COST * ((depth - START) / WIDTH)^2 * speed * dt per physics step.
# Nothing inside START of the line is touched, so the racing line up to the
# white line (and a little beyond) is unchanged. 0.0 disables it.
RL_EDGE_WALL_COST = 0.0

# Per-excursion off-track cost (2026-09-20). RL_OFF_TRACK_COST is per second x
# speed, so a one- or two-tick cut costs almost nothing (0.37/tick at ramp 2.0,
# ~0.04 s of lap time) while the corridor beyond the white line is worth ~0.5 s
# per 0.1 m of cushion -- brushing the line for a moment is simply the optimum.
# Scaling the per-tick fee up until a brief cut hurts (x40) makes a 20-tick
# excursion cost >100 and inflates the value scale. This charges a FLAT amount
# at the first tick of every excursion (a wheel-set going from on-track to
# all-four-wheels-off), independent of how long it lasts; the per-tick fee stays
# as it is. 0.0 disables. The ramp holds it at zero until the policy can string
# a lap together, then brings it to full over the same window as the tick fee.
RL_OFF_EVENT_COST = 0.0
RL_OFF_EVENT_RAMP = ((0, 0.0), (500_000, 0.0), (2_000_000, 1.0))

# Failure-state replay (GPU trainer only, --hard-states). While the policy is
# already nearly clean, every rollout decision that puts a wheel-set past the
# line (or hits a wall) saves the vehicle state from RL_HARD_LOOKBACK decisions
# EARLIER -- the approach to whatever corner it just failed at -- into a bank,
# and RL_HARD_FRAC of all episode restarts are then restored from that bank
# instead of the usual grid / scattered start. The hard corners get trained on
# far more often than one visit per lap, from the states that actually lead
# into them, on ANY track, with no corner named anywhere.
RL_HARD_FRAC = 0.20           # share of the parallel ENV SLOTS that do nothing but
                              # short failure-state episodes. Slots, not restarts:
                              # a restart-probability share is diluted by episode
                              # length (a 6 s hard episode vs a 270 s normal one
                              # would make it ~1 % of the samples); a slot share
                              # is exactly that share of the samples.
RL_HARD_MIN = 64              # bank size before the hard slots switch over
RL_EDGE_WALL_START = 0.3      # metres of car-centre depth past the white line that stay free
RL_EDGE_WALL_WIDTH = 0.5      # depth beyond START at which the cost equals COST per (m/s)
RL_EDGE_WALL_CAP = 2.0        # the ratio depth/WIDTH saturates here, so the wedge stays bounded (~4x COST);
                              # anything deeper is the plain off-track fee's job

# Speed (as a fraction of the local reference speed) a recovered car is given
# back after a mistake. Low enough that the mistake really costs time.
RL_RECOVER_FRAC = 0.20

# Wall-contact penalty, per (m/s)^2 of speed carried into the wall. Only the
# SAC/continuous reward uses it (the GTS-SAC c_w term); the discrete run relied
# on the recovery alone. GTS-SAC used 5e-4 with a similar speed range.
RL_WALL_KE_COST = 4.0e-4
RL_OFF_COURSE_COST = 0.02      # per second off the asphalt, times speed (legacy;
                              #  the v15 SAC reward uses RL_SAC_OFF_* below)

# ---- v15 continuous SAC reward -------------------------------------------
#: Off-track fee, continuous run. BASE is the flat per-second-off-times-speed
#: charge (like the old RL_OFF_COURSE_COST); DEPTH multiplies metres-past-the-
#: white-line-times-speed on top, so a wheel on the line is nearly free and a
#: two-metre excursion is a steep loss. Tuned so a lap's worth of small
#: excursions clearly loses to a clean lap (~58 progress reward at Monza).
RL_SAC_OFF_BASE = 0.12
RL_SAC_OFF_DEPTH = 0.25       # v15b: 0.35 -> 0.25. The first run over-braked
                             #  into caution -- small-excursion penalty was
                             #  beating the per-step progress reward.
#: Penalty on ||a_t - a_{t-1}||^2 (each component of [steer, pedal] in [-1,1]),
#: once per decision. Kills the steering chatter a squashed-Gaussian actor
#: falls into without changing where it wants the wheel on average.
RL_SAC_ACT_SMOOTH = 0.02      # v15b: 0.05 -> 0.02, it was also suppressing the
                             #  throttle/brake modulation a fast lap needs.
#: Flat time penalty per second, continuous run. v15's first attempt dropped
#: this entirely (GT Sophy style) and collapsed toward a slow, safe policy:
#: at gamma 0.99 the cost of not finishing the lap is beyond the horizon, so
#: "brake now" always won locally. A third of the IQN figure (1.2) makes
#: "every second slow is a loss" a *local* signal without swamping progress.
RL_SAC_TIME_W = 0.40
#: v15c: for the continuous run a crash ENDS the episode (no teleport-recover
#: -- that poisoned the twin-Q critic, which regressed the flood of
#: "action -> teleport -> random state" transitions to a flat low value and
#: flatlined). Charged once at the terminal: a flat part plus one scaled by
#: the speed carried in, so arriving at a wall slow beats arriving fast, and
#: braking for the corner beats both. ~58 progress reward for a Monza lap.
RL_SAC_CRASH_COST = 10.0
RL_SAC_CRASH_SPEED_COST = 25.0
# Charged once per recovery (wall / off-track / stall / wrong-way). Flat part
# plus a part scaled by the speed carried in. A Monza lap earns ~58 in progress
# reward (5793 m * RL_PROGRESS_W), so 2.5 + up to 4.5 per crash makes one or
# two contacts a lap a real cost without making standing still attractive.
# Raised from 0.5 / 1.0: v9 drove clean at its peak then drifted back to ~2
# wall contacts a lap because a recovery only cost the respawn time and a
# token fee. With the runoff now much wider a genuine wall contact is rare, so
# the few that remain should clearly hurt -- 1.2 flat + up to 3.0 by entry
# speed, against ~58 progress a lap.
RL_RECOVER_COST = 1.2         # v14: back to v11 (v13 tried v5's 0.5)
RL_RECOVER_SPEED_COST = 3.0   # v14: back to v11 (v13 tried v5's 1.0)

# Metres the centreline curvature is smoothed over before its speed profile is
# built (the profile feeds the observation, not the reward).
RL_CURVATURE_SMOOTH = 25.0
# Charged once when an episode ends in a mistake rather than at the time limit.
# Graded by the speed the car carried into it, because arriving at a barrier at
# 180 km/h and brushing it at walking pace are not the same mistake and a flat
# figure scored them identically. It also supplies a gradient where there was
# none: while every episode ends against the same wall, a flat cost makes an
# early lift worth nothing, and this makes a slower arrival strictly better
# even when the car still crashes.
RL_CRASH_COST = 8.0
RL_CRASH_SPEED_COST = 24.0     # added at MAX_SPEED, scaled linearly below it
RL_POLICY = ASSET_DIR / "policies"
# "rl" makes the ghost drive a trained policy where one exists for the circuit,
# falling back to the planner where it does not. "planner" always uses the
# planner. The two are interchangeable at the wheel: both answer
# controls(vehicle).
# "planner" drives the tuned racing line with the scripted controller -- on
# Monza that is a clean 102.7 s lap. "rl" swaps in a trained policy where one
# exists (assets/policies/<circuit>.npz), falling back to the planner where it
# does not. Monza uses the trained IQN policy (Monza_v7_best).
GHOST_DRIVER = os.environ.get("AISW_GHOST_DRIVER", "rl")
                              # override with AISW_GHOST_DRIVER=planner to
                              # watch the CMA-ES raceline itself (game/
                              # raceline.py) drive, unmixed with a trained
                              # policy -- e.g. to judge the raceline's own
                              # quality before blaming the RL policy for a
                              # line that was never good to begin with.
RACELINE_EDGE_MARGIN = 0.25    # metres of asphalt left beyond the body
RACELINE_KERB = 0.55           # fraction of the kerb the line may use
RACELINE_LSQ_ITERS = 60
RACELINE_CONTROLS = 48         # control points round the lap
RACELINE_GENERATIONS = 200
RACELINE_SIGMA = 1.2           # initial CMA-ES step, in metres

GHOST_ENABLED = True
# Car models -- the Blender F1 car (blender/f1_car.py), baked to .bam by
# tools/build_blender_f1.py. Three other sources are still wired up in car.py
# and picked by name alone, so switching is a one-line edit here:
#   "bl_red" / "bl_white"           the Blender car        (assets/models/f1)
#   "rb_red" / "rb_white"           the baked fp04rb asset (assets/models/f1)
#   "rc_red" / "rc_white"           assets/for+race.blend as modelled, 894k tris
#   "rl_red" / "rl_white"           the same, decimated to 91k tris
#   "f1red" / "f1white"             procedural, f1car.py
#   "raceCarRed" / "raceCarWhite"   Kenney CC0 kit         (assets/models/kenney)
# rc_/rl_ are built by blender/export_race_car.py + tools/build_blender_race.py,
# fitted to the same 4.45 x 2.41 m box as bl_. 1080p, twenty-car grand prix on
# the dev laptop, back to back: bl_ 72 fps, rl_ 66 (at 160k tris: 61 against bl_'s
# 80); rc_ 27 (qualifying, two cars: 95 / 84 / 69). rl_ is indistinguishable
# from rc_ under the game's lighting.
PLAYER_MODEL = "rl_red"
GHOST_MODEL = "rl_white"
# Fraction of the tyre's grip the AI commits to. This is the difficulty dial,
# and it is a physical quantity rather than a fudge factor: the planner works
# out a corner speed from the friction circle, and this says how much of the
# circle it is willing to use.
GHOST_PACE = 0.88
GHOST_ALPHA = 0.55             # how solid the ghost looks
GHOST_GRID_OFFSET = 3.2        # metres beside the player's grid slot
GHOST_GRID_BACK = 5.0          # ...and behind it

# --- session modes -----------------------------------------------------------
# Qualifying always starts with an out lap (not counted, not timed): a standing
# start folded into a lap time is not a lap time. A grand prix has none -- the
# race is timed from lights out, the way a real one is.
#
# Grand prix track limits: rules.py measures every trip with all four wheels
# past the white line, racecontrol.py decides what it deserves (a warning, a
# penalty that outweighs any time gained, or nothing when the car was pushed
# off or had already lost time) -- see the table in racecontrol.py.
GP_OFFTRACK_REJOIN = 1.0       # seconds back on track that end an excursion
# Grid slot (1 = pole) for a player who has not qualified on the circuit at
# the chosen level this session: mid-pack, so there is racing both ways.
GP_PLAYER_GRID = 10
# Race-day form: each AI driver's race pace is a little slower than their
# qualifying lap implies, by an amount drawn per race (|N(0, spread)|), so the
# grid order is not the finishing order and there is something to pass.
# Qualifying itself is unaffected.
GP_FORM_SPREAD = 0.020
# Chance per lap, added to every driver's own, of a late-braking error.
GP_EXTRA_MISTAKES = 0.06
# Added to every driver's lap-to-lap pace variation (std-dev, slower only): a
# constant handicap sorts the field by pace in a lap or two and then nothing
# changes; a handicap that is drawn afresh every lap keeps it shuffling.
GP_LAP_VARIATION = 0.015
# The twenty-car field (fieldproc.py) on circuits that have solved plans;
# False races the single AI ghost everywhere, as before.
GP_FIELD = True
# Who decides the AI drivers' passing, defending and queueing: "rules"
# (racecraft's hand-written layer) or "rl" (the policy trained by
# tools/ppo_raceai.py and kept in assets/policies/raceai.npz -- the rules, if
# that file is not there). The plan following, the room rule, yellow flags and
# recovery are the rules' either way. See README, "레이스 AI의 학습 판단 층".
RACE_AI = "rl"
RACE_AI_POLICY = ASSET_DIR / "policies" / "raceai.npz"
# Who steers and works the pedals: "rl" (the network of game/drivenet.py, weights
# in assets/policies/drivenet.npz -- the follower, if that file is not there) or
# "rules" (mintime_driver.PlanFollower, the hand-built tracker).
DRIVE_AI = "rules"
DRIVE_AI_POLICY = ASSET_DIR / "policies" / "drivenet.npz"
# Seconds a race control message about the player stays up (others: 60%).
# Under a yellow flag every car, the player's included, keeps below this speed
# for the stretch the flag covers (racecontrol.py judges it, racecraft.py drives
# to it, and the HUD says it).
YELLOW_SPEED_KMH = 100.0
RC_MESSAGE_T = 4.0
# Race control's messages queue and take turns: never less than this on
# screen however many are waiting, and news of other cars that has waited
# this long is dropped (the player's own always gets its turn).
RC_MESSAGE_MIN_T = 1.6
RC_STALE_T = 12.0
# Qualifying ghost laps: the AI's fastest clean flying lap, recorded by
# tools/record_ghost.py and replayed frame for frame.
GHOST_LAP_DIR = ASSET_DIR / "ghosts"
# Fraction of the autopilot's throttle used after the flag. The result card
# comes up straight away, over a car that is still moving, and the car goes on
# lapping at this pace until the session is left.
COOLDOWN_PACE = 0.45

WALL_RESTITUTION = 0.35
# Coulomb scrub along the barrier, as a fraction of the normal impulse. This
# replaced a flat WALL_SPEED_KEEP that multiplied the whole velocity on every
# contact: it killed as much speed for a glancing scrape as for a head-on hit,
# which is backwards. Capping the tangential impulse by the normal one makes a
# square impact expensive and a graze cheap, on its own.
WALL_FRICTION = 0.55

# ---------------------------------------------------------------------------
# Camera -- tuned for perceived speed, not for a pretty static shot
# ---------------------------------------------------------------------------
# Camera 1 is the broadcast onboard -- the T-cam on top of the airbox, just
# behind and above the halo, which is the shot an F1 viewer knows the car
# from. Camera 2 is the close chase, and a session opens on it (CAM_DEFAULT):
# the chase is the view you can place a car from, and the onboard is the one
# you switch to once you know where the circuit goes. The cycle runs from the
# tightest view outwards, so it does not start at its own first entry.
# The bonnet camera is disabled for now -- it is still built and still
# supported everywhere (put "hood" back in this tuple to have it again), it
# just does not deserve a stop in the cycle.
CAM_MODES = ("onboard", "chase", "far")
CAM_DEFAULT = "chase"
# The car is 0.98 m to the top of its airbox and 2.15 m from the centre of
# gravity to the nose, so the camera sits just clear of the airbox and a
# little behind it, and looks slightly down -- that is what puts the halo and
# the length of the nose in the lower half of the shot instead of an empty
# road.
# Height is set by the near plane, not by taste: the car is 0.98 m to the top
# of its airbox, so a camera at 1.40 m leaves 0.42 m of clearance under it,
# and CLIP_NEAR_ONBOARD below stays inside that. Clearance is what stops the
# bodywork being sliced open by the near plane -- the one thing this view
# cannot do, since the car is its subject.
CAM_ONBOARD_OFFSET = (0.0, 1.40, -0.40)
CAM_ONBOARD_AIM_Y = 0.68
# A real onboard is a wide-angle camera at any speed, and this one has to be:
# the FOV below is 60 deg at rest, which is narrow enough that the nose falls
# out of the bottom of the frame -- the car appears only once the FOV has
# opened with speed. A floor under it keeps the same shot standing still.
CAM_ONBOARD_FOV_MIN = 88.0
# Close behind and above, looking down onto the car: the ground rushes past,
# the car is big enough in frame to place on the road, and the road surface
# either side of it is visible rather than edge-on. 6.3 m rather than the
# 12.5 m this started at -- at racing speed the FOV has opened to ~87
# degrees, and at 12.5 m that left the car half the size it is here.
CAM_CHASE_OFFSET = (0.0, 4.3, -8.3)
CAM_HOOD_OFFSET = (0.0, 1.9, 1.35)
CAM_FAR_OFFSET = (0.0, 7.5, -17.5)
CAM_LOOKAHEAD = 12.0
CAM_POS_LERP = 7.0
CAM_AIM_LERP = 9.0
# A plain lerp settles at a lag of v/k metres, so at 250 km/h the camera would
# trail an extra 10 m and the car would shrink to a speck. Allow a little trail
# (it reads as acceleration) but no more than this.
CAM_MAX_LAG = 2.2
# FOV opens up with speed: the periphery stretches and the world rushes by.
# --- framing: hold the car's size on screen as the FOV opens ---------------
# The FOV below runs from 60 deg at rest to past 110 flat out. That alone
# makes the car 23% bigger standing still than at 80 km/h -- big enough that
# a stopped car sinks behind the telemetry widget, while the same offset
# frames it properly on the move. So the chase cameras are pulled back by the
# same ratio the FOV opens by: the world still stretches with speed, the car
# stays put in the frame.
#
# REF_FOV is the FOV the offsets are authored at (~80 km/h, a normal corner
# speed). LOCK is how much of the change to cancel -- 1.0 holds the car
# exactly, 0.0 is the old behaviour. The clamp stops the camera diving into
# the car at 300 km/h, where full compensation would ask for 0.5x the
# distance.
# ...and the second half of the same job: the look-ahead shortens at low
# speed (CAM_LOOKAHEAD below), which tips the chase camera's aim up and drops
# the car towards the bottom of the frame -- standing still it sank behind
# the telemetry widget. Aiming lower as the car slows cancels that, so the
# car keeps its place in the frame from a standstill to racing speed. The
# aim is back to its normal height by CAM_CHASE_AIM_SPEED.
# The chase rises with speed, above CAM_CHASE_RISE_FROM. Frame-locking the
# offset (below) scales the whole vector, height included, so at 270 km/h the
# camera sat a metre lower than at 80 and the road ahead was squashed into
# the middle of the frame. A small rise opens the view down the track again
# without turning the chase into a helicopter shot -- 2.0 m was that, and
# read as the camera climbing away from the car. It starts at the reference
# speed, so the framing that camera 2 was tuned at is untouched.
CAM_CHASE_RISE = 0.6           # metres added by MAX_SPEED
CAM_CHASE_RISE_FROM = 22.0     # m/s where it starts (~80 km/h)

CAM_CHASE_AIM_Y = 1.35         # aim height at speed, as every other camera
CAM_CHASE_AIM_LOW = 0.85       # ...and standing still
CAM_CHASE_AIM_SPEED = 22.0     # m/s (~80 km/h) where the two meet

CAM_FRAME_LOCK_MODES = ("chase", "far")
CAM_FRAME_REF_FOV = 70.0
CAM_FRAME_LOCK = 1.0
CAM_FRAME_SCALE = (0.85, 1.35)

# Horizontal degrees (Ursina's camera.fov is the horizontal angle): opens
# from 60 to 132 flat out. (62 -> 78 and 62 -> 92 were tried on 2026-10-04, and
# 60 -> 96 with a heavier POST_SPEED_BLUR on 2026-10-05, to stop cars reading as
# squashed; all put back: this is the rush of speed the game is meant to have.
# The squashed look was the car model's proportions, fixed in the model.)
CAM_FOV_BASE = 60.0
CAM_FOV_GAIN = 72.0            # added at MAX_SPEED (-> 132 deg flat out)

# --- depth buffer ------------------------------------------------------
# A 24-bit depth buffer's precision is dominated by the NEAR plane: the
# smallest resolvable separation at distance d is roughly d^2 / (near * 2^24).
# Ursina defaults to near=0.1 / far=10000, which resolves only 54 mm at 300 m
# and 380 mm at 800 m -- far coarser than the ~15 mm by which road markings sit
# above the asphalt, so they z-fought and vanished into the distance. Every
# depth-offset hack in this project existed to paper over that.
#
# near=1.5 gives 15x the precision at every range (3.6 mm at 300 m), which is
# finer than the real height separations below, so the ordering is simply
# correct and no polygon offsets are needed anywhere.
CLIP_NEAR = 1.5
CLIP_FAR = 6000.0
# ...except on the onboard camera, which is mounted on the car: at 1.5 m the
# near plane cuts away the halo and the airbox the shot exists to show, and
# leaves the nose floating in front of nothing.
#
# 0.35 m, against 0.42 m of clearance to the airbox (CAM_ONBOARD_OFFSET), so
# the bodywork stays outside the near plane rather than being clipped by it.
# The margin is only 7 cm, less than the speed shake (+/- 5.5 cm at 300 km/h,
# 17.6 cm off track), which is exactly what tore the airbox open. The shake
# stays -- it belongs to the view -- but in this camera it is never allowed
# to point *downwards* (see _update_camera): shaken up, sideways and along,
# the clearance under the camera cannot shrink, and no shot is missing for
# it.
#
# The precision this costs (about 15 mm at 300 m, against 9 mm at 0.6) is
# only ever paid while this camera is selected.
CLIP_NEAR_ONBOARD = 0.35

# Road surface heights, in metres above the asphalt. These are real geometric
# separations, not render tricks; each gap is comfortably larger than the depth
# buffer can resolve out to ~600 m, beyond which fog hides everything anyway.
# 38 mm below the apron was not a real separation at all: past a couple of
# hundred metres the depth buffer cannot tell them apart, and the ground
# plane's own quads punch up through the run-off as rectangles of grass
# inside the barrier. Almost a third of a metre resolves out to the fog, and
# the step is only ever seen at the outer rim of the apron, where that apron
# has already faded to grass and the ground beyond is grass too.
Y_GRASS = -0.32
# The run-off apron sits between the two: above the grass plane so it is
# the surface you see beside the track, below the asphalt so the edge line
# and the kerbs still win where they overlap it.
Y_RUNOFF = -0.012
# Metres per cell of the grid the run-off apron is triangulated on. The apron
# is the interior of the drivable region, filled cell by cell against the same
# contour the barrier is drawn on -- see trackdata.runoff_fill. Four metres is
# a couple of car lengths, which is finer than any gradient painted on it, and
# it keeps the whole lap's ground under fifty thousand triangles.
RUNOFF_CELL = 4.0
# Metres the ground is carried on *past* the barrier before it meets the flat
# plane behind it. Without this the run-off stops dead at the fence and the
# plane picks up a third of a metre lower, which is a step running the whole
# length of every barrier on the circuit.
RUNOFF_SKIRT = 10.0
# Metres of rise per metre out across the apron, and its cap. Real run-off is
# not a billiard table and a corner's gravel trap is big enough to show it.
# There is no side bias any more: the apron is one mesh covering the whole
# region, so there are no longer two of them to keep apart.
RUNOFF_RISE = 0.004
RUNOFF_RISE_MAX = 0.12
# How far the apron sinks below Y_RUNOFF directly under the centreline. It
# fills the whole region, road included, and 12 mm of clearance is not enough
# to keep it out of the asphalt at the far end of a straight. Kept a little
# above the grass plane so those two do not trade places instead -- both are
# under opaque asphalt there, but there is no reason to add a second fight.
# ...and how far it sits below the road and the kerb it runs beside. Tapered
# back to nothing over fourteen metres of run-off rather than over one, which
# is the difference between a run-off that is slightly lower than the circuit
# and a trench dug round the kerb.
RUNOFF_UNDER_ROAD = 0.10
# ...and how much further it drops where the road is cambered, ramped in over
# this much bank. The apron is triangulated on its own grid and the road on
# the circuit's samples, so on a steep bank the two descriptions of the same
# surface disagree by a few centimetres -- enough for a cell of apron to lift
# through the kerb beside it. The clearance is scaled by the camber rather
# than set to a constant so that a flat circuit keeps its run-off flush with
# the white line, and it varies only along the lap, never across it, so it
# adds no step for the eye to catch.
RUNOFF_BANK_SINK = 0.16
RUNOFF_BANK_SINK_REF = 6.0     # degrees of camber at which the full drop applies
Y_ASPHALT = 0.0
Y_SEAM = 0.020
Y_LINE = 0.035
Y_KERB = 0.025
Y_START = 0.045
Y_SHADOW = 0.060
# How far the view and the body lean when cornering. Both were tuned for
# drama and read as the picture shaking every time you touch a direction key,
# so they are dialled back here rather than buried as literals.
CAM_LEAN = 0.18                # camera roll per rad/s of yaw (was 0.30)
# Body attitude. These are presentation, not physics: the hull leans on the
# telemetry so the car reads as loaded, and the wheels stay where the
# suspension put them (see car.py).
#
# They are deliberately smaller than they "should" be, because the model
# sits *on* the ground -- its lowest point is y = 0 -- so every degree of
# lean and every millimetre of squat puts bodywork under the road surface.
# At the old values a hard entry dipped a wing through the asphalt.
BODY_ROLL_GAIN = 0.20          # hull roll per m/s^2 of lateral accel (0.28)
BODY_ROLL_MAX = 3.2            # degrees (4.5)
BODY_PITCH_GAIN = 0.12         # hull pitch per m/s^2 of long. accel (0.20)
BODY_PITCH_MAX = 2.0           # degrees (3.5)
BODY_SQUAT_MAX = 0.03          # metres of ride-height drop under downforce
                               #  (0.07, which alone sank the floor into the
                               #  road before any lean was added)

# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------
# Everything below is LINEAR light (see shaders.py): the sun is several times
# brighter than the sky, as it is outdoors, and post.py's tone curve brings the
# result down to a screen. Colours picked in sRGB elsewhere (vertex colours,
# textures) are converted to linear in the shader, so they keep meaning what
# they meant.
#
# Two looks, picked by LIGHTING_PRESET:
#   "day"     race-day afternoon: a high, near-white sun, a deep blue sky with
#             cloud, the broadcast look every modern racing game is graded to.
#   "sunset"  the old golden hour, retuned for linear light.
LIGHTING_PRESET = "day"

# The azimuth is derived from the circuit rather than fixed, because a fixed
# one is flattering on some tracks and puts the sun behind the main grandstand
# on others. Light comes across the start/finish line from the paddock side, so
# the stands are front-lit and their shadows fall away from the track, raked
# along it for a long diagonal rather than a flat side-light.
# Raked well along the straight on purpose. Whatever stands on the sun side
# throws its shadow across the circuit, and the main grandstand is 10 m tall
# some 23 m from the centreline: at 22 degrees elevation its shadow is 25 m
# long, so a side-on sun lays it right over the racing line. Raking it to 65
# leaves only 11 m of that across the track, which stops in the run-off.
SUN_RAKE = 65.0                # degrees the beam is turned along the straight
SUN_SIDE = -1                  # -1: sun over the grandstands, +1: over the paddock
SUN_AZIMUTH = 205.0            # fallback when there is no track to derive from
SUN_DISC_DEG = 0.55            # angular radius of the drawn disc (real: 0.27)

LIGHT_SUN_WRAP = 0.08              # softens the terminator a touch
# Global scale on what surfaces reflect of the sky.
LIGHT_ENV = 1.0
# Glow strength round the sun, in the sky and on faces turned towards it.
LIGHT_GLOW_STRENGTH = 1.0

# Cloud layer: coverage threshold, edge softness, scale, brightness.
CLOUD_SHAPE = (0.56, 0.26, 0.055, 1.15)
CLOUD_DRIFT = (0.0016, 0.0007)  # texture units per second

HAZE_DENSITY = 0.85
HAZE_START = 160.0
# Far enough out that the range keeps most of its own colour. At 3200 the
# hills came out the same value as the sky behind them and simply vanished.
HAZE_END = 5200.0

_LIGHTING = {
    "day": dict(
        # High enough that the light is white, low enough that everything
        # still throws a shadow with some length to it.
        SUN_ELEVATION=38.0,
        LIGHT_SUN=(3.30, 3.12, 2.88),      # irradiance of the beam
        LIGHT_SKY=(0.46, 0.60, 0.86),      # sky irradiance on an upward face
        LIGHT_BOUNCE=(0.16, 0.16, 0.12),   # off the ground, on a downward one
        LIGHT_GLOW=(1.10, 0.95, 0.75),     # forward scatter round the sun
        SKY_ZENITH=(0.10, 0.24, 0.62),
        SKY_HORIZON=(0.62, 0.76, 0.95),
        SKY_GROUND=(0.10, 0.11, 0.09),     # the world below the horizon
    ),
    "sunset": dict(
        # 8.5 degrees was too low to drive under: a 10 m grandstand throws a
        # 67 m shadow at that angle, which covers the whole main straight.
        SUN_ELEVATION=22.0,
        LIGHT_SUN=(3.40, 1.75, 0.80),
        LIGHT_SKY=(0.30, 0.36, 0.55),
        LIGHT_BOUNCE=(0.20, 0.13, 0.08),
        LIGHT_GLOW=(2.20, 0.95, 0.40),
        SKY_ZENITH=(0.05, 0.08, 0.22),
        SKY_HORIZON=(1.15, 0.55, 0.30),
        SKY_GROUND=(0.08, 0.06, 0.05),
    ),
}
globals().update(_LIGHTING[LIGHTING_PRESET])

# --- camera chain (post.py) ----------------------------------------------------
POST_ENABLED = True
# Panda's threading model: "Cull/Draw" puts culling and drawing on threads of
# their own, so they overlap the next frame's Python instead of following it
# (a grand prix went 23 -> 17 ms a frame on the dev laptop). "" for the old
# single-threaded renderer.
RENDER_THREADING = "Cull/Draw"
# Render scale: the 3D scene is drawn at full resolution up to this many
# pixels, and above it at the size that keeps it to this many, stretched to
# the window by the post pass (with RENDER_SHARPEN to restore edges); the HUD
# is always drawn at the window's own resolution. The GPU, not Python, is
# what a bigger window costs: on the dev laptop's Iris Xe a race ran 98 fps
# at 1280x720 and 51 at 1920x1080. RENDER_SCALE multiplies the result (1.0:
# no further scaling); RENDER_MAX_PIXELS 0 turns the cap off.
RENDER_MAX_PIXELS = 1600 * 900           # off: full resolution at any size (sharp)
RENDER_SCALE = 1.0
RENDER_SHARPEN = 0.35
# Samples on the HDR buffer; 0 for none. 2, not 4: with the pipelined
# renderer (RENDER_THREADING) the GPU became the slowest stage, and 4x cost
# ~3.2 ms a frame at 1600x900 on the Iris Xe against ~0.6 ms for 2x.
POST_MSAA = 2
POST_BITS = 11                 # 16: RGBA16F, 11: R11G11B10F, 8: RGBA8
POST_BLOOM = 0.10              # how much of the over-bright light bleeds
POST_BLOOM_THRESHOLD = 1.6     # linear level the glow starts from
POST_BLOOM_KNEE = 0.8
POST_GRAIN = 0.010
POST_SPEED_BLUR = 0.7          # radial smear at full speed
POST_SPEED_FROM = 0.45         # fraction of MAX_SPEED where it starts
POST_GRADE = dict(
    exposure=0.82,
    contrast=1.05,
    saturation=1.06,
    vignette=0.22,
    # Split tone: shadows a hair cool, highlights a hair warm.
    shadow_tint=(0.97, 1.0, 1.04),
    high_tint=(1.03, 1.0, 0.96),
)

# One shadow map, focused on a box around the car. Stretched over a whole
# circuit it would be ~3 m per texel; over 140 m it is 7 cm.
# Static shadows are rendered once, over the whole circuit, into their own
# map; the map that follows the car then only has to carry the car. Nothing
# outside the car moves and the sun does not either, so redrawing the roadside
# into a depth buffer sixty times a second was work with no output.
#
# The trade is texel size. The following map is 2048 over 140 m (7 cm); the
# baked one is 4096 over the whole circuit, which for Monza is about 50 cm --
# coarse for the shadow of a roof on the steps beneath it, but it reaches the
# whole lap instead of stopping 67 m from the car.
BAKE_SHADOWS = True
# 8192 halves the texel (0.55 m -> 0.27 m), which is what makes a single
# baked map good enough to be the *only* source of static shadows -- see
# BAKE_REPLACES_LIVE. If the buffer cannot be allocated, bake() says so
# and the roadside falls back to live shadows.
BAKE_RESOLUTION = 8192
BAKE_MARGIN = 260.0            # metres of roadside beyond the track's bounds
BAKE_SAMPLES = 2               # PCF taps per axis; the map is coarse already
BAKE_BLUR = 0.0009
# In metres, and converted to normalised depth once the film's depth range is
# known. A raw normalised figure is meaningless on its own -- the same number
# is centimetres of slack on the following map and metres on this one -- and
# too little slack at this texel size makes a surface shadow itself.
# Most of the acne is dealt with by offsetting the lookup along the surface
# normal instead, so this only has to cover what is left.
BAKE_SLACK = 0.18              # halved with the texel
# Multiples of a baked texel to push the lookup out along the normal.
BAKE_NORMAL_OFFSET = 1.6
# Record the casters' back faces rather than their front ones.
BAKE_BACKFACE = True
# True hands the roadside's shadows entirely to the baked map, which takes it
# out of the per-frame shadow pass.
#
# Now the default, and not for the saving. Keeping both meant every static
# shadow existed twice, at eleven times the resolution near the car and once
# past the following map's edge, cross-faded between 32 m and 67 m. The two
# never matched: a grandstand's shadow faded in as you approached it, and the
# join between the crisp one and the coarse one was a visible line sweeping
# over the ground. One source has no join to show. At 8192 the baked texel is
# 27 cm, which a building-sized caster does not need beating.
BAKE_REPLACES_LIVE = True
BAKE_TOP = 30.0                # tallest thing that casts, for fitting the film

SHADOW_RESOLUTION = (1024, 1024)   # legacy follow map (unused, see below)
# The car's shadow has a map of its own, drawn every frame by a camera that
# rides with the car and sees nothing else (lighting.CarShadow). Because the
# camera moves *with* the car rather than across a grid, the car rasterises
# into it identically from one frame to the next: nothing for its edges to
# crawl over, and it is placed from the same interpolated pose the car is
# drawn at, so it cannot trail the car either. The static world's shadows all
# come from the baked map.
CAR_SHADOW_RES = 1024         # 9 mm texels over the 9 m film; 2048 cost ~1.8 ms of GPU a frame
CAR_SHADOW_FILM = 9.0          # metres across: the car at any heading, plus its shadow
CAR_SHADOW_DEPTH = 12.0        # metres either side of the car along the beam
CAR_SHADOW_BIAS_M = 0.012      # depth slack, metres
CAR_SHADOW_SOFT_M = 0.035      # penumbra width, metres
CAR_SHADOW_NORMAL_M = 0.012    # lookup pushed out along the normal, metres
# The other cars' shadows (lighting.FieldShadow): one wider map round the car
# on camera, pushed ahead of it along the view, faded out at its edge.
FIELD_SHADOW_RES = 2048
FIELD_SHADOW_FILM = 90.0       # metres across (4.4 cm texels)
FIELD_SHADOW_DEPTH = 40.0      # metres either side along the beam
FIELD_SHADOW_AHEAD = 22.0      # film centre this far ahead of the car, metres
FIELD_SHADOW_BIAS_M = 0.05
FIELD_SHADOW_SOFT_M = 0.09
FIELD_SHADOW_NORMAL_M = 0.05
SHADOW_AREA = 140.0            # metres across the shadow film
SHADOW_DEPTH = 300.0           # how far along the beam the film reaches
SHADOW_HEIGHT = 60.0           # where the light node sits above the car
# The bias is in NORMALISED depth, so its world-space meaning is
# bias * (far - near) = bias * 2 * SHADOW_DEPTH. At 0.0022 over a 800 m span
# that was 1.76 m of slack -- larger than most of the features that would
# self-shadow (a wing over a deck, a roof over seats), so they were swallowed
# whole. 0.0004 over 600 m is 24 cm.
# Where the following map hands over to the baked one, as fractions of
# SHADOW_AREA. The band is wide on purpose: the two maps differ by eleven
# times in texel size, and a narrow crossfade shows that as a line on the
# ground. Spread over 35 m it reads as the shadows softening with distance,
# which is what a viewer expects anyway.
SHADOW_FADE_START = 0.23
SHADOW_FADE_END = 0.48

SHADOW_BIAS = 0.0004
SHADOW_BLUR = 0.0018
SHADOW_SAMPLES = 3              # PCF taps per axis near the car (9; 16 cost ~1 ms more)

CAM_SHAKE = 0.055              # metres of jitter at MAX_SPEED
CAM_SHAKE_OFFTRACK = 3.2       # multiplier on grass/kerb

# ---------------------------------------------------------------------------
# Scenery: roadside objects give the parallax reference that sells speed
# ---------------------------------------------------------------------------
MARKER_SPACING = 42.0          # metres between marker posts
# Barrier modules are 0.83 m long; lining a 6 km circuit on both sides at that
# pitch would be fourteen thousand copies. Each is stretched to the pitch
# instead, which on a barrier profile is what a longer run really looks like.
# Modules are built greedily: one run continues while the edge stays straight,
# up to MAX_RUN, and breaks where it bends by more than MAX_BEND. A fixed pitch
# would have to be short enough for the tightest corner and would then spend
# that resolution on every straight -- and copies are what the build time is
# made of.
BARRIER_STEP = 4.0             # metres between edge samples the runs are cut from
BARRIER_MAX_RUN = 24.0         # longest single stretched module
BARRIER_MAX_BEND = 3.0         # degrees of bend that ends a run
# --- roadside layout -----------------------------------------------------
# Furniture goes where the real thing would put it, which is emphatically not
# "evenly, everywhere". The old layout covered 68 per cent of both sides of the
# lap in grandstands; the result read as wallpaper, because with stands
# everywhere there is nothing for the eye to arrive at. These pick out the
# handful of places that matter and leave the rest as open country.
FURNITURE_CORNER_RADIUS = 300.0   # what counts as a corner for furniture
FURNITURE_CORNER_MIN_LEN = 45.0   # metres; shorter than this is a kink, not a corner
STAND_CORNERS = 5              # corners that get a grandstand, slowest first
TYRE_WALL_CORNERS = 8          # corners that get a tyre wall on the outside
BOARD_CORNERS = 6              # corners that get a 150/100/50 countdown
# How far a distance board is turned from square-to-the-fence towards the
# oncoming car. 0 faces straight across the track, 1 faces straight back
# down it; a real board sits between the two so it reads on the approach.
BOARD_AIM = 0.62
BOARD_STANDOFF = 0.55         # metres the board hangs clear of the fence
STAND_MIN_RUN = 34.0           # metres of straight worth a run of stands
# What fraction of a straight a run of stands takes. Low on purpose: the
# treeline is what fills this circuit now, and stands are punctuation.
STAND_STRAIGHT_FRACTION = 0.34
# The Kenney grandstand at kit scale is 3.3 m wide; this is what brings it up
# to something a Formula 1 car looks small in front of.
GRANDSTAND_SCALE = 7.5

# --- barrier line --------------------------------------------------------
# Armco on posts with a debris fence behind it, in place of the kit's solid
# wall module. Posts are placed at a pitch rather than stretched -- a post
# stretched along its length is a wall, which is what was being replaced.
GUARDRAIL_POST_PITCH = 4.0     # metres between Armco posts
FENCE_POST_PITCH = 5.0         # metres between debris-fence posts
FENCE_SETBACK = 1.1            # metres behind the rail; a fence is not a barrier

# --- the treeline --------------------------------------------------------
# (from the barrier, to, spacing) in metres. The near band is tight enough to
# have no sky through it and is most of what closes the world in; the far one
# is for depth. Spacing is also the jitter, so the grid never shows.
# Tight enough that there is no sky through the near band -- that is the
# whole job. A treeline you can see the horizon through is a scattering of
# trees, not a treeline, and it leaves the circuit as open as bare grass did.
FOREST_BANDS = ((2.0, 28.0, 6.5), (28.0, 118.0, 24.0))
# Big. A 12 m tree scaled to 1.0-2.0 stands 12-24 m, and fewer large trees
# close a horizon than many small ones -- at a tenth of the triangles.
FOREST_SCALE = (0.95, 2.05)    # random size multiplier per tree
# Metres of the world in one forest batch. 0 batches by species instead --
# one node per species spanning the whole lap, whose bounding volume contains
# the camera wherever it stands, so nothing is ever culled and all 7800 trees
# are submitted every frame.
#
# There is a real optimum and it is not "as small as possible": tight cells
# cull more but every node costs its own cull test and draw call. Measured on
# Monza, 600 frames, mean fps / 1% low:
#
#     off  36.3 / 26.0      130 m  28.3 / 16.3
#   240 m  39.7 / 29.0      400 m  41.9 / 29.1      600 m  39.1 / 28.2
#
# 130 m is *worse than not culling at all*, which is the same wall an earlier
# attempt at cell-batching the roadside hit. Re-measure before changing this.
FOREST_CELL = 400.0
#: Painted card trees (foliage.py) instead of the faceted Blender models.
FOREST_CARDS = True
FOREST_CLEAR = 1.8             # metres a tree keeps from the barrier line
FOREST_PROP_CLEAR = 6.0        # ...and from anything already built there
# The grandstand is swept along the wall in scenery.py rather than placed as
# copies of a model, so its size lives here. Eleven tiers of 1.9 m tread put
# the back row 21 m up and 27 m back -- an actual Formula 1 main grandstand,
# which is three or four times the length of the car parked in front of it.
# Set back far enough that a 21 m building does not lean over the car. At a
# chicane the barrier line comes in to 9 m from the centreline, so a small
# setback there puts the front row directly above the kerb.
STAND_SETBACK = 15.0           # metres from the barrier to the front wall
# ...but the barrier line is not a constant distance from the track. On a
# straight it sits 20 m out; through a chicane the medial-axis walk brings
# it in to 9, so a setback measured from it alone puts a 21 m building
# twice as close at exactly the corner where it looms most. This is the
# floor, measured from the centreline, and it is what actually governs.
STAND_MIN_FRONT = 34.0         # metres from the centreline to the front wall
STAND_TIERS = 11               # visible seating tiers
PIT_SIDE = +1                  # which side of the main straight the pits are on
PIT_SETBACK = 7.0              # metres from the barrier to the pit wall
PIT_WALL_CLEAR = 1.5           # metres a garage keeps clear of the barrier
PIT_GARAGES = 20               # bays; ~250 m of building, like the real thing
HOARDING_MIN_RUN = 8.0         # metres of straight wall worth a hoarding
TYRE_WALL_MIN_RUN = 5.0
MARSHAL_SPACING = 430.0        # metres between marshal posts
# How far outside the asphalt edge a structure that straddles the circuit
# puts its legs. Not "just outside the barrier", which is what it used to be:
# the wall is the outline of the run-off, and now that a corner's run-off
# opens out to fifty metres a gantry stretched to reach it stood with its
# feet a cricket pitch apart and its beam a hundred metres long. Measured
# from the road instead, and clamped inside the wall, so it reads as a gantry
# over the circuit wherever it lands.
# The cinematic that plays before the countdown: two corners, an aerial over
# the grid, then a look down the main straight. Ten seconds all told -- long
# enough to say where you are, short enough that nobody reaches for the skip
# key on the second lap of the evening.
# --- banking ---------------------------------------------------------
# The stored circuits are a centreline and two widths; there is no elevation
# in them and no camber, so the banking is synthesised from curvature. A
# corner leans as much as its radius asks for, capped, and the cap is raised
# per circuit for the ones that are actually famous for it.
# Toggle the whole synthesised-camber system on or off. Off: every circuit is
# dead flat (trackdata.bank returns zeros) and the run-off tilt, the cone/kerb
# grounding and the pit/scenery height sampling all follow, because they read
# the same bank profile. Kept off by default -- the camber here is guessed from
# curvature, not measured, and a wrong guess reads worse than a flat corner.
BANKING_ENABLED = False
BANK_MAX_DEG = 5.0             # default cap, anywhere on any circuit
# Radius at which the cap is reached, and how sharply the lean falls away
# above it. Squared, and keyed to a genuinely tight corner: keyed to 90 m and
# linear, Zandvoort came out banked over more than half its lap, which is not
# a circuit with two banked corners, it is a bowl.
BANK_RADIUS = 40.0
BANK_FALLOFF = 2.0
BANK_SMOOTH = 90.0             # metres of lap the profile is averaged over
#: circuit -> cap in degrees. Zandvoort's Hugenholtzbocht and Arie Luyendyk
#: are banked at about 18 degrees, which is most of what the circuit is known
#: for and is worth having even though the source data cannot know it.
BANK_CIRCUIT_MAX = {"Zandvoort": 18.0, "Monza": 6.0, "YasMarina": 4.0}
#: Metres past the asphalt edge over which the tilt fades out. The road is
#: banked; forty metres of run-off tilted with it would drive one edge of the
#: apron underground and the other into the air.
BANK_RUNOFF_FADE = 14.0

INTRO_ENABLED = True
INTRO_CORNER_TIME = 4.6
INTRO_AERIAL_TIME = 3.0
INTRO_STRAIGHT_TIME = 2.2
GANTRY_LEG_CLEAR = 4.0
BRIDGE_LEG_CLEAR = 8.0
BRIDGE_COUNT = 3               # spectator bridges round the lap (one is skipped
                               # -- the start line already has the gantry)
# The Kenney kit is authored for a smaller world than a 4.5 m GT car, so what
# is left of it has to be scaled up to the size the track and car imply. The
# stands and tents it used to supply are gone: seating is the purpose-built
# stretchable part in blender/circuit_kit.py, which follows the wall instead of
# stepping along it in fixed modules.
LIGHTPOST_SCALE = 3.0          # -> 7.7 m
# Lamp posts stand hard against the *outside* face of the barrier, at this gap
# and no other. Placed off the nearest chord and then checked back against
# every chord on both sides: a post two metres behind one wall can be a metre
# inside another where the circuit doubles back, and one that drifts is a lamp
# growing out of the run-off. Anything outside the tolerance is dropped rather
# than nudged -- a missing lamp reads as a gap in a row, a wrong one reads as
# a bug.
LIGHTPOST_WALL_GAP = 1.6       # metres outside the barrier line
LIGHTPOST_GAP_TOL = 0.7        # how far from that gap a post may still stand
# Clearance a grandstand module's footprint must keep from the barrier, on top
# of standing wholly outside the corridor.
STAND_WALL_CLEAR = 2.0
TREE_SCALE = (1.00, 1.80)      # random range -> 5.0 to 9.0 m
# The barrier line is Track.wall_offsets(): a walk outward along each normal,
# as far as the run-off asks for, minus the stretches Track.wall_valid() finds
# already swallowed by another leg's run-off. That is the same corridor the
# collision test uses, so the wall you see and the wall you hit are one line.
# The module is 0.43 m tall against a 1.1 m wall mesh, so at kit scale the
# barrier read as skirting board with grey wall looming over it. Stretched to
# cover the wall it is what you actually see at the edge of the circuit.
BARRIER_HEIGHT_SCALE = 2.90    # -> 1.25 m, comfortably over the 1.1 m wall mesh
BARRIER_DEPTH_SCALE = 1.5      # -> 0.62 m, a believable guardrail section
BARRIER_OUTSET = 0.20          # metres outside the wall line, to cover the mesh
BARRIER_MODULE_DEPTH = 0.40    # the asset's own depth, in metres
# Half the placed rail's thickness: the car should stop at the face it can see,
# not at the line through the middle of the barrier. Derived, so rescaling the
# module cannot leave the collision behind.
BARRIER_HALF_DEPTH = BARRIER_MODULE_DEPTH * BARRIER_DEPTH_SCALE / 2.0
# How far through a barrier the car can be and still be pushed back out. A
# step at 90 m/s covers 1.5 m, so nothing legitimate ever reaches this. It is
# there to disown chords the car is nowhere near: a chord round the far side
# of a bend has its outward face pointing back across the circuit, and
# without a depth limit the road 40 m away counts as "through" it.
BARRIER_MAX_PENETRATION = 6.0
# The drivable region is sampled onto a grid this fine before its outline is
# taken. A metre resolves the waist of a chicane without the sampling costing
# a noticeable part of the load.
BARRIER_GRID = 1.0
BARRIER_SMOOTH = 9             # smoothing passes over the raw contour
BARRIER_RESAMPLE_SMOOTH = 5    # samples in the second pass, after resampling
# Metres along the lap, either side of the car, to search for barrier segments.
# One segment can be BARRIER_MAX_RUN long and is keyed by its midpoint, so this
# has to cover half of that plus the car plus room to spare.
BARRIER_QUERY_RANGE = 45.0

WALL_GAP = 0.6                 # metres left short of the medial axis
# The strip of ground a barrier between two legs of the circuit needs, and
# the width below which there is no room for one and the run-off of the two
# legs is allowed to merge into a single wrapped complex. WALL_MERGE_SPREAD
# widens the merged stretch so it cannot flicker on and off sample by sample.
WALL_MERGE_GAP = 9.0
# The wall never comes closer to the road than this, however tight the gap
# between two legs. Below it the chords cut inside their own arc and stand on
# the track.
WALL_MIN_MARGIN = 3.5
WALL_MERGE_FLOOR = 8.0
WALL_MERGE_SPREAD = 6
# Samples over which the merged/not-merged decision is cross-faded. A hard
# switch is a step of tens of metres in the wall's radius inside one sample,
# and the fence then leaves a chicane complex through most of a right angle --
# which is exactly the sharp bend where the contour smoothing, the chord walk
# and the run-off apron all stopped agreeing with each other.
# The mask is already dilated by WALL_MERGE_SPREAD, so a window this size
# leaves the middle of the shortest merged run at full weight and only ramps
# its ends -- widening it instead swallows the walls that separate a circuit's
# infield from itself, which is Zandvoort's whole layout.
WALL_MERGE_BLEND = 9
# Metres of lap between two legs before they stop counting as one complex.
# A chicane doubles back within a few seconds; two straights that run side by
# side are half a lap apart and must keep the wall between them.
WALL_MERGE_LAP = 260.0
WALL_MAX_SLOPE = 0.30          # how fast the wall may open out or close in
WALL_SMOOTH = 5                # samples in the final averaging window
# Grass embankment (hills) along the track outside barriers (True: enabled, False: disabled)
SPECTATOR_BANK_ENABLED = False
TREE_SPACING = 34.0            # denser: the treeline behind the bank is what
                               # closes the view where the bank runs out
TREE_BAND = (78.0, 190.0)      # clear of the stands, whose backs now reach
                               # ~38 m past the barrier line      # distance from the track edge -- clear of the
                               # stands, whose backs now reach ~25 m past the wall
FOG_DENSITY = 0.0026   # dense enough to hide the far clip plane at CLIP_FAR

# ---------------------------------------------------------------------------
# Race / session
# ---------------------------------------------------------------------------
# Start sequence, F1 lights. After the loading card lifts: a beat, then the
# gantry drops in from the top of the screen, the five lamps come on one by
# one (0.8 s apart), they hold lit for a randomised pause, all five go dark
# (that is the go signal -- the car is released here), and the gantry whips
# back up and away while the race is already under way.
START_WAIT = 0.7
START_GANTRY_DROP = 0.55
START_LIGHTS_BUILD = 4.0
START_HOLD_MIN = 1.2
START_HOLD_MAX = 2.6
START_LIGHTS_OUT = 0.25
START_GANTRY_RISE = 0.42
# Laps in a session: the grand prix distance, and qualifying's timed laps
# (after the out lap, which is not timed).
TOTAL_LAPS = 5
# --- surrounding landform ------------------------------------------------
# A circuit on an endless flat plane has no sense of place. These ring the
# track far enough out that they cannot intrude on it whatever its shape.
# Distant mountain/hill ring around the circuit (True: enabled, False: disabled)
MOUNTAINS_ENABLED = True
MOUNTAIN_GAP = 400.0           # metres clear of the track's bounding circle
MOUNTAIN_DEPTH = 2100.0        # radial thickness of the range
# Sized from how tall the range should *look* from the track, not from a
# number that sounds like a mountain: the peak line sits about 2.3 km out, so
# 750 m subtends ~18 degrees -- a range you notice rather than a distant smudge.
MOUNTAIN_HEIGHT = 750.0        # tallest peak
MOUNTAIN_RAMP = 0.30           # fraction of the band spent climbing
MOUNTAIN_SEGMENTS = 128        # around
MOUNTAIN_RINGS = 14            # outward

MINIMAP_POINTS = 180           # centreline samples drawn in the HUD minimap

WINDOW_SIZE = (1280, 720)
FULLSCREEN = False

# --- recording (recorder.py) -----------------------------------------------------
# F9 starts and stops a screen recording of the game window: ffmpeg's desktop
# capture into the GPU's H.264 encoder, so the game itself does no work for it.
# The REC marker (top right) is part of the picture; turn it off for a clean
# video if you will remember that you are recording.
REC_KEY = "f9"
REC_FPS = 60
REC_BITRATE_MBPS = 30.0        # 1080p60 of this game: ~220 MB a minute
REC_INDICATOR = True
REC_DIR = None                 # default: Videos\FORMULA-AI
REC_FFMPEG = ""                # default: ffmpeg on PATH, else Program Files/ffmpeg
PHYSICS_HZ = 120
