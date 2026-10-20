"""Print the wheel's axes and buttons live: turn the wheel / press each pedal and
see which of X Y Z R U V moves, then put that in ai_sw/wheel.json."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_sw.game.wheel import AXES, Wheel  # noqa: E402

w = Wheel()
print("Ctrl+C to stop")
try:
    while True:
        if w.read():
            ax = " ".join(f"{a}={w.raw[a]:.2f}" for a in AXES)
            btn = [i for i in range(32) if w.buttons >> i & 1]
            print(f"\r{ax} POV={w.pov:>3} btn={btn}        ", end="", flush=True)
        else:
            print("\rno joystick device found      ", end="", flush=True)
        time.sleep(0.03)
except KeyboardInterrupt:
    pass
