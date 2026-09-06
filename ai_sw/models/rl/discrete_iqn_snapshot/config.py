"""Central tuning block for the AI-SW racing prototype.

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

RUNOFF_WIDTH = 13.0            # asphalt edge -> wall
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
HUD_MARGIN = 0.022     # clear space between a panel and the screen edge
RL_OFF_TRACK_COST = 3.0        # per second beyond track limits, times speed

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

# --- Linesight-style reward -------------------------------------------------
# Reward per metre advanced along the centreline, and time penalty per second.
# Lifted straight from Linesight (5/500 and 6/5000 per ms = 1.2 per s). Over a
# fixed-length episode the time term is nearly constant, so the return tracks
# distance covered, i.e. average speed.
RL_PROGRESS_W = 5.0 / 500.0
RL_TIME_W = 6.0 / 5000.0 * 1000.0     # per second

# Potential-based line term: Phi = -K * clip(|offset|, LO, HI). Potential-based
# so it cannot change which policy is optimal -- only speeds learning toward
# the middle of the road. Linesight: K = 0.1, clip [2, 25].
RL_LINE_K = 0.10
RL_LINE_LO = 2.0
RL_LINE_HI = 25.0

# Speed (as a fraction of the local reference speed) a recovered car is given
# back after a mistake. Low enough that the mistake really costs time.
RL_RECOVER_FRAC = 0.30

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
# does not. The RL work is parked mid-experiment: every policy so far fails
# the first chicane, so the planner is the ghost until that is solved.
GHOST_DRIVER = "rl"
RACELINE_EDGE_MARGIN = 0.25    # metres of asphalt left beyond the body
RACELINE_KERB = 0.55           # fraction of the kerb the line may use
RACELINE_LSQ_ITERS = 60
RACELINE_CONTROLS = 48         # control points round the lap
RACELINE_GENERATIONS = 200
RACELINE_SIGMA = 1.2           # initial CMA-ES step, in metres

GHOST_ENABLED = True
GHOST_MODEL = "raceCarRed"
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
CAM_CHASE_OFFSET = (0.0, 2.55, -8.2)    # low + close => ground rushes past
CAM_HOOD_OFFSET = (0.0, 1.18, 1.35)
CAM_FAR_OFFSET = (0.0, 4.6, -13.5)
CAM_LOOKAHEAD = 14.0
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
Y_GRASS = -0.05
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
BAKE_RESOLUTION = 4096
BAKE_MARGIN = 260.0            # metres of roadside beyond the track's bounds
BAKE_SAMPLES = 2               # PCF taps per axis; the map is coarse already
BAKE_BLUR = 0.0009
# In metres, and converted to normalised depth once the film's depth range is
# known. A raw normalised figure is meaningless on its own -- the same number
# is centimetres of slack on the following map and metres on this one -- and
# too little slack at this texel size makes a surface shadow itself.
# Most of the acne is dealt with by offsetting the lookup along the surface
# normal instead, so this only has to cover what is left.
BAKE_SLACK = 0.35
# Multiples of a baked texel to push the lookup out along the normal.
BAKE_NORMAL_OFFSET = 2.0
# Record the casters' back faces rather than their front ones.
BAKE_BACKFACE = True
# True hands the roadside's shadows entirely to the baked map, which takes it
# out of the per-frame shadow pass. Cheaper, and visibly coarser everywhere
# near the car; the default keeps both and uses the baked one only past the
# following map's edge.
BAKE_REPLACES_LIVE = False
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
# Stands are laid out as straight runs along both sides, covering roughly
# STAND_COVERAGE of the lap. A grandstand is a straight building: one heading
# and one origin per run, modules exactly a width apart along it. Placing each
# module by aiming it at the nearest centreline point gave every one a slightly
# different heading, which on a straight reads as a row that does not line up.
# Measured over Monza / Spa / Zandvoort / Shanghai, these come out at 69 / 66 /
# 55 / 59 per cent of the wall covered, with zero footprint overlaps and every
# module in a run sharing its heading to 0.000 degrees.
STAND_COVERAGE = 0.68          # fraction of each side that carries stands
STAND_SAMPLE_STEP = 4.0        # metres between wall samples the runs are cut from
STAND_STRAIGHT_TOL = 6.0       # degrees of bend that ends a run
STAND_PERP_TOL = 6.0           # metres the wall may stray from the run's line
STAND_MIN_MODULES = 1          # shortest run worth placing, in modules
STAND_MIN_GAP = 8.0            # metres between consecutive runs
# The Kenney kit is authored for a smaller world than a 4.5 m GT car: at true
# kit scale a grandstand module is 3.3 m wide and 3.9 m tall, i.e. shorter than
# the car parked in front of it. These bring the built environment up to the
# size the track and car imply.
GRANDSTAND_SCALE = 5.2         # -> 17.2 m wide, 20.4 m tall
TENT_SCALE = 3.0               # -> 9.9 m square, 7.0 m tall
LIGHTPOST_SCALE = 3.0          # -> 7.7 m
TREE_SCALE = (1.00, 1.80)      # random range -> 5.0 to 9.0 m
# The barrier line is Track.wall_offsets(): a walk outward along each normal,
# stopping where another part of the centreline becomes the nearest one. That
# is the same corridor the collision test uses, so the wall you see and the
# wall you hit are one line.
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
# Metres along the lap, either side of the car, to search for barrier segments.
# One segment can be BARRIER_MAX_RUN long and is keyed by its midpoint, so this
# has to cover half of that plus the car plus room to spare.
BARRIER_QUERY_RANGE = 45.0

WALL_GAP = 0.6                 # metres left short of the medial axis
WALL_MAX_SLOPE = 0.06          # how fast the wall may close in, as a gradient
WALL_SMOOTH = 5                # samples in the final averaging window
TREE_SPACING = 70.0
TREE_BAND = (34.0, 120.0)      # distance from the track edge, beyond the wall
FOG_DENSITY = 0.0026   # dense enough to hide the far clip plane at CLIP_FAR

# ---------------------------------------------------------------------------
# Race / session
# ---------------------------------------------------------------------------
COUNTDOWN_SECONDS = 3
TOTAL_LAPS = 3
# --- surrounding landform ------------------------------------------------
# A circuit on an endless flat plane has no sense of place. These ring the
# track far enough out that they cannot intrude on it whatever its shape.
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
