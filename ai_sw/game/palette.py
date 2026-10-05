"""Ursina 8's ``color.rgb`` takes 0..1 floats; this wraps 0..255 ints."""
from __future__ import annotations

from ursina import color


def rgb(r: float, g: float, b: float, a: float = 255) -> color.Color:
    return color.rgba(r / 255.0, g / 255.0, b / 255.0, a / 255.0)


# named
SKY = rgb(158, 186, 214)
FOG = rgb(176, 196, 214)
ASPHALT = rgb(68, 70, 76)
ASPHALT_EDGE = rgb(58, 60, 66)
BARRIER = rgb(52, 54, 60)
GRASS = rgb(76, 102, 50)

# roadside furniture
POST_A = rgb(206, 48, 44)
POST_B = rgb(238, 238, 238)
BOARD = rgb(232, 226, 96)
TYRE = rgb(34, 34, 38)
TRUNK = rgb(88, 66, 48)
CROWN_A = rgb(48, 104, 52)
CROWN_B = rgb(62, 122, 60)
CAR_RED = rgb(210, 32, 40)
CAR_DARK = rgb(28, 30, 36)
CAR_BLACK = rgb(18, 18, 20)
CAR_GLASS = rgb(46, 54, 66)
HEADLIGHT = rgb(226, 232, 238)
WHEEL_MARK = rgb(150, 150, 156)
BRAKE_OFF = rgb(80, 8, 8)
BRAKE_ON = rgb(245, 40, 30)
SHADOW = rgb(0, 0, 0, 125)
