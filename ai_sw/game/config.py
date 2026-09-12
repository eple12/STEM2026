"""Central tuning block for the FORMULA-AI racing prototype.

Physics follows the classic dynamic *bicycle model* described in Marco
Monster's "Car Physics for Games" and implemented in spacejack/carphysics2d,
with parameter magnitudes cross-checked against the CommonRoad / f1tenth_gym
single-track model. Units are SI (metres, seconds, newtons, kilograms).
"""
from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK_DB = REPO_ROOT / "Neural_Network_NEAT-master" / "new" / "f1tenth_racetracks-main"
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
    "Melbourne": 11.129, "Mexico City": 12.067, "Montreal": 15.299,
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

RL_OFF_TRACK_COST = 0.35       # per second beyond track limits, times speed.
                              # v17: the clean-lap bonus (zeroed for a dirty
                              # lap) does most of the work of forcing PERFECT;
                              # this is a moderate backstop with a gentle ramp.
RL_OFF_TRACK_COST_RAMP = ((0, 1.0), (300_000, 1.0), (1_500_000, 2.0))
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
RL_SHAPE_TO_RACELINE = True
RL_RACELINE_SAFETY = 0.20     # metres the saved raceline is pulled in so every
                              #  wheel stays inside the white line by this much

# --- Linesight-style reward -------------------------------------------------
# Reward per metre advanced along the centreline, and time penalty per second.
# Lifted straight from Linesight (5/500 and 6/5000 per ms = 1.2 per s). Over a
# fixed-length episode the time term is nearly constant, so the return tracks
# distance covered, i.e. average speed.
RL_PROGRESS_W = 5.0 / 500.0
RL_TIME_W = 6.0 / 5000.0 * 1000.0     # per second

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
# STEEP: base 300, so a 94 s clean lap is +64 and a 130 s crawl is 0 (clamped),
# and every second cut off the lap is worth RL_RL_LAP_W = 2.5 -- about twice
# the base per-second time penalty (1.2), so once the policy is clean, going
# faster is a clear, large net gain rather than the marginal gamble it was for
# v18 (progress-minus-time only). A Monza lap earns ~58 in progress reward, so
# the bonus is comparable in size, not swamping.
RL_RL_LAP_BASE = 300.0
RL_RL_LAP_W = 2.5
RL_RL_LAP_OFF_TOL = 3        # off-steps a lap may have and still count as clean

# Potential-based line term: Phi = -K * clip(|offset|, LO, HI). Linesight uses
# K = 0.1 and accepts that it pulls the car off the racing line ("any pull at
# all is pulling against the optimum") because Trackmania's runoff is narrow.
# v19 zeroed this hoping for a wider racing line; instead the policy wandered
# and converged clean-but-SLOW (181 km/h). v20: back to Linesight's 0.10 --
# turns out this mild pull keeps the car on a consistent efficient line, which
# is what let v18 reach 215 km/h. The lap-time bonus is the speed driver now.
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
                           #  + linear term", full v18 shaping.
RL_EDGE_MARGIN = 2.4      # v26: back to v18.
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
GHOST_DRIVER = "rl"
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
#   "f1red" / "f1white"             procedural, f1car.py
#   "raceCarRed" / "raceCarWhite"   Kenney CC0 kit         (assets/models/kenney)
PLAYER_MODEL = "bl_red"
GHOST_MODEL = "bl_white"
# Fraction of the tyre's grip the AI commits to. This is the difficulty dial,
# and it is a physical quantity rather than a fudge factor: the planner works
# out a corner speed from the friction circle, and this says how much of the
# circle it is willing to use.
GHOST_PACE = 0.88
GHOST_ALPHA = 0.55             # how solid the ghost looks
GHOST_GRID_OFFSET = 3.2        # metres beside the player's grid slot
GHOST_GRID_BACK = 5.0          # ...and behind it

# The first lap is an out lap: not counted, not timed. A standing start folded
# into a lap time is not a lap time, and both cars need a lap to get up to
# speed before the comparison means anything.
OUT_LAP = True
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
CAM_MODES = ("chase", "hood", "far")
CAM_CHASE_OFFSET = (0.0, 5, -12.5)    # low + close => ground rushes past
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
CAM_FOV_BASE =60.0
CAM_FOV_GAIN = 72.0            # added at MAX_SPEED (-> 120 deg flat out)

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
BODY_ROLL_GAIN = 0.28          # hull roll per m/s^2 of lateral accel (was 0.42)
BODY_ROLL_MAX = 4.5            # degrees (was 7.0)

# ---------------------------------------------------------------------------
# Lighting -- golden hour
# ---------------------------------------------------------------------------
# The whole look is one relationship: the low beam is warm because the blue has
# been scattered out of it, and that scattered blue is what fills the shadows.
# Warm key, cool fill. Reverse it and the scene is just noon with an orange
# filter over it.
# 8.5 degrees was too low to drive under: a 10 m grandstand throws a 67 m
# shadow at that angle, which covers the whole width of the main straight and
# puts the racing line in the dark on every circuit.
SUN_ELEVATION = 22.0           # degrees above the horizon
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

LIGHT_SUN = (1.55, 0.86, 0.44)     # direct beam, over 1.0 so lit faces glow
LIGHT_SKY = (0.42, 0.50, 0.65)     # ambient from the sky: cool
LIGHT_BOUNCE = (0.26, 0.20, 0.15)  # ambient from the ground: warm, dim
LIGHT_SUN_WRAP = 0.35              # softens the terminator, as grazing light does
# The sky around the sun is far brighter than the rest of it. Without this the
# hemispheric ambient gives every vertical face the same value and unlit
# objects read as flat cut-outs.
LIGHT_GLOW = (0.55, 0.30, 0.16)    # horizon glow picked up by faces turned to it
LIGHT_GLOW_STRENGTH = 1.0

# Sky dome gradient, bottom to top.
SKY_HORIZON = (1.00, 0.60, 0.33)
SKY_MID = (0.62, 0.45, 0.55)
SKY_ZENITH = (0.10, 0.16, 0.36)
SKY_GLOW_BAND = 0.22           # fraction of the dome the horizon glow occupies

HAZE_COLOR = (0.94, 0.63, 0.42)
HAZE_DENSITY = 0.92
HAZE_START = 140.0
# Far enough out that the range keeps most of its own colour. At 3200 the
# hills came out the same value as the sky behind them and simply vanished.
HAZE_END = 4800.0

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

SHADOW_RESOLUTION = (2048, 2048)
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
SHADOW_SAMPLES = 3

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
FOREST_BANDS = ((2.0, 28.0, 8.5), (28.0, 118.0, 24.0))
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
TOTAL_LAPS = 3
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
PHYSICS_HZ = 120
