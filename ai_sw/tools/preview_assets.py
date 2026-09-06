"""Look at every model in ``assets/models/`` without launching the game.

Two modes:

    python tools/preview_assets.py            # contact sheet of everything
    python tools/preview_assets.py --live     # orbit one model interactively

The contact sheet is the one to reach for when deciding what to build scenery
from: it renders each asset at a consistent camera distance *relative to its
own bounding box*, so shapes are comparable, and prints the real-world size
underneath. Size is the thing you actually need -- a tree asset that turns out
to be 3 m tall looks fine in isolation and absurd beside a 4.5 m car.

``--live`` opens a window with mouse orbit for checking a specific model up
close (holes in the hull, how the wheels sit, which way it faces).
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ursina import (AmbientLight, DirectionalLight, Entity, Ursina, Vec3,
                    application, camera, held_keys, mouse, scene, window)

from game import palette as pal
from game.car import load_gltf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "assets" / "models"
OUT = ROOT / "asset_preview"
SIZE = 420          # per-cell render size
COLS = 4


def find_models() -> list[Path]:
    found = sorted(p for p in MODELS.rglob("*")
                   if p.suffix.lower() in (".glb", ".gltf"))
    if not found:
        sys.exit(f"no .glb/.gltf found under {MODELS}")
    return found


def kit_scale(paths: list[Path]) -> float:
    """Units-to-metres factor for the whole kit, taken from the car.

    The GLBs are authored in arbitrary units -- a car measures ~1.3 long -- so
    raw bounds are the wrong thing to plan scenery against. ``car.py`` already
    solves this for the car by scaling its axle spacing onto WHEELBASE; every
    model in one Kenney kit shares that authoring scale, so the same factor
    converts them all. Falls back to 1.0 (and says so) if no car is present.
    """
    from game.car import WHEEL_NODES
    from game import config

    car = next((p for p in paths if "racecar" in p.stem.lower()), None)
    if car is None:
        return 0.0
    holder = load_gltf(car)
    axles = [holder.find(f"**/{n}") for n in WHEEL_NODES]
    zs = sorted(a.get_pos().z for a in axles if not a.is_empty())
    from ursina import destroy
    destroy(holder)
    if len(zs) < 2:
        return 0.0
    return config.WHEELBASE / max(abs(zs[-1] - zs[0]), 1e-6)


def measure(holder: Entity):
    """Bounding box of a loaded model, as (lo, hi, size) in asset units."""
    lo, hi = holder.get_tight_bounds()
    return lo, hi, (hi.x - lo.x, hi.z - lo.z, hi.y - lo.y)   # w, d, h in Panda axes


def frame_camera(lo, hi, azimuth=38.0, elevation=26.0, fill=1.05):
    """Place the camera so the model fills the frame whatever its size.

    Distance comes from the bounding sphere and the current FOV rather than a
    hand-tuned constant, which is what makes a 1 m pylon and a 20 m grandstand
    directly comparable on the same sheet.
    """
    centre = Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
    radius = max((hi - lo).length() / 2, 1e-3)
    dist = radius * fill / math.tan(math.radians(camera.fov) / 2)

    a, e = math.radians(azimuth), math.radians(elevation)
    offset = Vec3(math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e))
    camera.position = centre + offset * dist
    camera.look_at(centre)
    camera.rotation_z = 0          # look_at rolls the camera when off-axis
    return centre, radius


def build_sheet(shots: list[tuple[str, str, Path]]) -> Path:
    """Stitch the per-asset PNGs into one labelled contact sheet."""
    from PIL import Image, ImageDraw, ImageFont

    pad, label_h = 14, 42
    cell_w, cell_h = SIZE, SIZE + label_h
    rows = math.ceil(len(shots) / COLS)
    sheet = Image.new("RGB",
                      (COLS * cell_w + pad * (COLS + 1),
                       rows * cell_h + pad * (rows + 1)),
                      (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    try:
        f_name = ImageFont.truetype("arialbd.ttf", 19)
        f_size = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        f_name = f_size = ImageFont.load_default()

    for i, (name, dims, path) in enumerate(shots):
        r, c = divmod(i, COLS)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        img = Image.open(path).convert("RGB")
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
        sheet.paste(img, (x, y))
        draw.text((x + 6, y + SIZE + 4), name, font=f_name, fill=(238, 240, 244))
        draw.text((x + 6, y + SIZE + 24), dims, font=f_size, fill=(150, 156, 166))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "contact_sheet.png"
    sheet.save(out)
    return out


def run_sheet(paths: list[Path]):
    app = Ursina(title="asset preview", size=(SIZE, SIZE), vsync=False,
                 development_mode=False, borderless=False)
    window.color = pal.rgb(120, 126, 134)
    AmbientLight(color=pal.rgb(170, 173, 180))
    sun = DirectionalLight(color=pal.rgb(255, 250, 240))
    sun.look_at(Vec3(0.5, -1.0, 0.35))
    camera.fov = 34
    OUT.mkdir(parents=True, exist_ok=True)

    state = {"frame": 0, "idx": 0, "cur": None, "shots": [], "k": 0.0}

    def _save(name):
        from panda3d.core import Filename
        p = OUT / f"{name}.png"
        base.win.saveScreenshot(Filename.fromOsSpecific(str(p)))  # noqa: F821
        return p

    # Three frames per asset: load, then let the camera move settle, then
    # capture. Panda only has the finished framebuffer a frame after the change.
    def update():
        f = state["frame"]
        state["frame"] += 1
        if f < 4:
            return
        idx, phase = divmod(f - 4, 3)
        if idx >= len(paths):
            sheet = build_sheet(state["shots"])
            print(f"\nSHEET {sheet}")
            application.quit()
            return

        path = paths[idx]
        if phase == 0:
            if state["cur"] is not None:
                from ursina import destroy
                destroy(state["cur"])
            if not state["k"]:
                state["k"] = kit_scale(paths) or 1.0
                print(f"kit scale: 1 unit = {state['k']:.2f} m "
                      f"(from the car's axle spacing vs WHEELBASE)\n")
            k = state["k"]
            state["cur"] = load_gltf(path)
            lo, hi, (w, d, h) = measure(state["cur"])
            frame_camera(lo, hi)
            state["dims"] = (f"{w * k:.1f} x {d * k:.1f} m  |  {h * k:.1f} m tall")
            print(f"   {path.stem:22s} {state['dims']}")
        elif phase == 2:
            state["shots"].append((path.stem, state["dims"], _save(path.stem)))

    print(f"rendering {len(paths)} assets from {MODELS}\n")
    import __main__
    __main__.update = update
    app.run()


def run_live(path: Path):
    app = Ursina(title=f"asset preview - {path.stem}", size=(1100, 760),
                 vsync=True, development_mode=False)
    window.color = pal.rgb(120, 126, 134)
    AmbientLight(color=pal.rgb(170, 173, 180))
    sun = DirectionalLight(color=pal.rgb(255, 250, 240))
    sun.look_at(Vec3(0.5, -1.0, 0.35))
    camera.fov = 34

    k = kit_scale(find_models()) or 1.0
    holder = load_gltf(path)
    lo, hi, (w, d, h) = measure(holder)
    centre, radius = frame_camera(lo, hi)
    print(f"{path.stem}: {w * k:.2f} x {d * k:.2f} m footprint, "
          f"{h * k:.2f} m tall  (1 unit = {k:.2f} m)")
    print("drag = orbit, scroll = zoom, G = ground plane, ESC = quit")

    ground = Entity(parent=scene, model="plane", scale=radius * 12,
                    color=pal.ASPHALT, y=lo.y - 0.001, enabled=False)
    orbit = {"az": 38.0, "el": 26.0, "dist_k": 1.05, "g": False, "g_held": False}

    def update():
        if mouse.left:
            orbit["az"] -= mouse.velocity[0] * 260
            orbit["el"] = max(-85, min(85, orbit["el"] + mouse.velocity[1] * 260))
        # held_keys goes to 0 on release, so this edge-detects a tap
        g = held_keys["g"] > 0
        if g and not orbit["g_held"]:
            orbit["g"] = not orbit["g"]
            ground.enabled = orbit["g"]
        orbit["g_held"] = g
        frame_camera(lo, hi, orbit["az"], orbit["el"], orbit["dist_k"])

    def input(key):
        if key == "scroll up":
            orbit["dist_k"] = max(0.25, orbit["dist_k"] * 0.9)
        elif key == "scroll down":
            orbit["dist_k"] = min(6.0, orbit["dist_k"] * 1.1)
        elif key == "escape":
            application.quit()

    import __main__
    __main__.update = update
    __main__.input = input
    app.run()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="orbit one model in a window instead of rendering a sheet")
    ap.add_argument("--model", default=None,
                    help="model name (stem) to view; defaults to the first found")
    args = ap.parse_args()

    paths = find_models()
    if args.live or args.model:
        if args.model:
            match = [p for p in paths if p.stem.lower() == args.model.lower()]
            if not match:
                sys.exit(f"no such model: {args.model}\n  have: "
                         + ", ".join(p.stem for p in paths))
            paths = match
        run_live(paths[0])
    else:
        run_sheet(paths)


if __name__ == "__main__":
    main()
