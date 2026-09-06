"""A detached camera you can fly round the circuit.

Bound to **F**. The race carries on underneath -- physics, the AI, the clock --
so this is for looking at the scene, not for pausing it: the whole point is
being able to go and stand somewhere the chase camera never goes and see what
the roadside actually looks like from there.

    F                 toggle on / off
    W A S D           move on the ground plane, relative to where you look
    R / Space         up            F is taken, so R rises
    C / Shift-hold    down / faster
    hold right mouse  look around
    arrow keys        look around, if a mouse is awkward
    [ / ]             slower / faster base speed

Held-key movement rather than a click-and-drag orbit because a circuit is six
kilometres long: an orbit control is right for inspecting one object and wrong
for travelling.
"""
from __future__ import annotations

import math

from ursina import Vec3, camera, held_keys, mouse

#: metres per second at the default speed step
BASE_SPEED = 34.0
SPEED_STEPS = (6.0, 14.0, 34.0, 80.0, 190.0)
LOOK_SENS = 90.0            # degrees per unit of mouse velocity
KEY_LOOK = 90.0             # degrees per second on the arrow keys
PITCH_LIMIT = 88.0


class FreeCam:
    """Drives ``camera`` directly while enabled."""

    def __init__(self):
        self.enabled = False
        self.pos = Vec3(0.0, 12.0, 0.0)
        self.yaw = 0.0
        self.pitch = 12.0
        self.step = SPEED_STEPS.index(BASE_SPEED)
        self._was_locked = False

    # -- control ------------------------------------------------------
    def toggle(self) -> bool:
        """Flip state. Entering, it takes over from wherever the camera is, so
        it never starts by teleporting you somewhere unrelated."""
        self.enabled = not self.enabled
        if self.enabled:
            self.pos = Vec3(camera.world_position)
            self.yaw = camera.world_rotation_y
            self.pitch = _wrap180(camera.world_rotation_x)
            self._was_locked = mouse.locked
        else:
            camera.rotation_z = 0.0
            mouse.locked = self._was_locked
        return self.enabled

    def on_key(self, key: str) -> bool:
        """Consume a key press. Returns True if it was ours."""
        if key == "[":
            self.step = max(0, self.step - 1)
            return True
        if key == "]":
            self.step = min(len(SPEED_STEPS) - 1, self.step + 1)
            return True
        return False

    # -- per frame ----------------------------------------------------
    def update(self, dt: float):
        # Look: mouse while the right button is down, arrows otherwise. Using
        # mouse.velocity rather than absolute position means it does not matter
        # where the pointer happens to be sitting when you grab it.
        if mouse.right:
            self.yaw += mouse.velocity[0] * LOOK_SENS
            self.pitch -= mouse.velocity[1] * LOOK_SENS
        self.yaw += (held_keys["right arrow"] - held_keys["left arrow"]) * KEY_LOOK * dt
        self.pitch += (held_keys["down arrow"] - held_keys["up arrow"]) * KEY_LOOK * dt
        self.pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, self.pitch))

        speed = SPEED_STEPS[self.step] * (4.0 if held_keys["shift"] else 1.0)
        fwd = held_keys["w"] - held_keys["s"]
        right = held_keys["d"] - held_keys["a"]
        up = (held_keys["r"] + held_keys["space"]) - held_keys["c"]

        # Ground-plane movement: WASD keeps its height whatever the pitch is,
        # which is what makes flying along a straight feel like flying rather
        # than diving. Height is its own axis.
        rad = math.radians(self.yaw)
        f = Vec3(math.sin(rad), 0.0, math.cos(rad))
        r = Vec3(math.cos(rad), 0.0, -math.sin(rad))
        self.pos += (f * fwd + r * right) * speed * dt
        self.pos += Vec3(0.0, 1.0, 0.0) * up * speed * dt

        camera.world_position = self.pos
        camera.world_rotation = Vec3(self.pitch, self.yaw, 0.0)

    def hud_line(self) -> str:
        return (f"FREECAM  {SPEED_STEPS[self.step]:.0f} m/s  "
                f"[ ] speed | WASD move | R/C up-down | shift boost | "
                f"RMB or arrows look | F exit")


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0
