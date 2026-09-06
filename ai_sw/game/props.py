"""Place many copies of a glTF asset without paying for many draw calls.

The roadside used to be procedural boxes merged into one mesh per class -- one
draw call for a thousand marker posts. Swapping in real models must not throw
that away, so each asset is loaded once as a detached template, stamped with
``copy_to`` for every placement, and the whole batch is then run through
Panda3D's ``flatten_strong()``, which merges compatible geometry back down to a
handful of Geoms. One entity per asset type reaches the scene graph.

Coordinate note: Ursina's ``position`` setter passes (right, up, forward)
straight to Panda's ``setPos``, so on a raw NodePath here **y is up and z is
forward**, and a yaw of ``rotation_y`` degrees is ``set_h(-rotation_y)``
(``Entity.rotation_directions`` is (-1, -1, 1)). The Kenney models face -z, the
same as the car asset, so anything with a front gets 180 degrees added.
"""
from __future__ import annotations

import math
from pathlib import Path

from ursina import Entity, scene

from . import config
from .car import WHEEL_NODES, _bake_material_colors

MODEL_DIR = config.ASSET_DIR / "models"


class PropLibrary:
    """Loads asset templates and stamps them into flattened batches."""

    def __init__(self):
        #: Set by build_scenery; applied to each batch *before* flattening.
        self.shader = None
        self._templates: dict[str, object] = {}
        self._heights: dict[str, float] = {}
        self._scale: float | None = None

    # -- kit scale ------------------------------------------------------
    def kit_scale(self) -> float:
        """Metres per asset unit, recovered from the car's axle spacing.

        The GLBs are authored in arbitrary units -- a car measures about 1.3
        long -- so nothing can be placed at a believable size without this.
        ``car.py`` already solves it for the car; every model in one kit shares
        the authoring scale, so the same factor converts them all.
        """
        if self._scale is not None:
            return self._scale
        self._scale = 1.0
        car = self._find("raceCarWhite")
        if car is not None:
            node = self._raw(car)
            axles = [node.find(f"**/{n}") for n in WHEEL_NODES]
            zs = sorted(a.get_pos().z for a in axles if not a.is_empty())
            if len(zs) >= 2:
                self._scale = config.WHEELBASE / max(abs(zs[-1] - zs[0]), 1e-6)
            node.remove_node()
        return self._scale

    # -- templates ------------------------------------------------------
    @staticmethod
    def _find(name: str) -> Path | None:
        for ext in (".glb", ".gltf", ".bam"):
            hits = list(MODEL_DIR.rglob(name + ext))
            if hits:
                return hits[0]
        return None

    @staticmethod
    def _raw(path: Path):
        import builtins

        from panda3d.core import Filename

        return builtins.base.loader.loadModel(
            Filename.fromOsSpecific(str(path.resolve())))

    def template(self, name: str, face_forward: bool = False):
        """A detached, normalised NodePath: metres, centred in x/z, base at y=0.

        Normalisation is baked into a child so the template's own transform is
        identity and a copy can be posed freely.
        """
        key = (name, face_forward)
        if key in self._templates:
            return self._templates[key]

        from panda3d.core import NodePath

        path = self._find(name)
        if path is None:
            print(f"props: no model named {name}")
            self._templates[key] = None
            return None

        holder = NodePath(f"prop_{name}")
        node = self._raw(path)
        node.reparent_to(holder)
        if path.suffix == ".bam":
            # Built for this game by tools/build_blender_scenery.py: metres,
            # game axes, vertex colours, and already facing +z. Its authored
            # origin is load-bearing -- a hoarding sits at barrier height and a
            # fence post starts 1.25 m up -- so normalising it onto y = 0 would
            # drop both of them on the floor. Left exactly as baked.
            if face_forward:
                node.set_h(180.0)
        else:
            _bake_material_colors(node)
            node.set_scale(self.kit_scale())
            if face_forward:
                node.set_h(180.0)           # the kit assets face -z, the game +z
            lo, hi = holder.get_tight_bounds()
            node.set_pos(-(lo.x + hi.x) / 2, -lo.y, -(lo.z + hi.z) / 2)

        # Collapse the template to a single Geom before anyone copies it.
        # panda3d-gltf wraps each mesh in a ModelNode whose transform is marked
        # preserved, and flatten_strong will not merge across those -- which is
        # why batching 1500 barriers produced 1500 GeomNodes and took 8.5 s.
        # clear_model_nodes() drops that flag; doing it here means every copy
        # is cheap and the batch below can actually merge.
        holder.clear_model_nodes()
        holder.flatten_strong()

        lo, hi = holder.get_tight_bounds()
        self._heights[name] = float(hi.y - lo.y)
        self._templates[key] = holder
        return holder

    def height(self, name: str) -> float:
        self.template(name)
        return self._heights.get(name, 0.0)

    def footprint(self, name: str) -> tuple[float, float]:
        """(width, depth) in metres, so placements can be derived from the
        asset rather than from constants that go stale when it is rescaled."""
        t = self.template(name)
        if t is None:
            return (0.0, 0.0)
        lo, hi = t.get_tight_bounds()
        return (float(hi.x - lo.x), float(hi.z - lo.z))

    # -- batching -------------------------------------------------------
    def batch(self, name: str, placements, face_forward: bool = False,
              parent=None, shader=None) -> Entity | None:
        """Stamp *placements* of one asset into a single flattened entity.

        ``placements`` yields ``(x, z, yaw_deg)`` or ``(x, z, yaw_deg, scale)``.

        *shader* must be given here rather than assigned to the result later.
        ``flatten_strong`` composes the node states down onto the geoms, and
        that includes the shader in force at the time -- so a shader assigned
        afterwards is overridden by the baked one and never runs. Every
        roadside batch was rendering with Ursina's default because of this: no
        sun, no shadows, just flat vertex colour, which is exactly what "the
        stands never get a shadow from their own roof" looked like.
        """
        tmpl = self.template(name, face_forward)
        if tmpl is None:
            return None
        placements = list(placements)
        if not placements:
            return None

        root = Entity(parent=parent if parent is not None else scene)
        shader = shader if shader is not None else self.shader
        if shader is not None:
            root.shader = shader
        for p in placements:
            x, z, yaw = p[0], p[1], p[2]
            s = p[3] if len(p) > 3 else 1.0
            copy = tmpl.copy_to(root)
            copy.set_pos(float(x), 0.0, float(z))
            # A scalar scales uniformly; a triple stretches per axis, which is
            # how one 0.8 m barrier module becomes a run of wall without
            # needing thousands of copies.
            if isinstance(s, (tuple, list)):
                copy.set_scale(float(s[0]), float(s[1]), float(s[2]))
            else:
                copy.set_scale(float(s))
            copy.set_h(-float(yaw))
        # Merges the copies back into a few Geoms. Without it a few hundred
        # props cost a few hundred draw calls and the frame rate halves.
        #
        # Tried and reverted: flattening per 250 m cell so the nodes have tight
        # bounds and can be culled. It cost 9 ms a frame. With a 6 km far plane
        # on a flat circuit almost the whole track is inside the frustum
        # anyway, so there was nothing to cull, and the extra nodes were pure
        # overhead in both the camera pass and the shadow pass.
        root.clear_model_nodes()
        root.flatten_strong()
        return root

    def batch_many(self, groups, parent=None, shader=None) -> Entity | None:
        """One flattened entity holding copies of *several* assets.

        ``groups`` maps asset name -> placements, exactly as ``batch`` takes
        them one at a time. The point is spatial: batching by asset type gives
        one node per type spanning the whole circuit, whose bounding volume
        therefore contains the camera at all times and can never be culled.
        Batching a *neighbourhood* of mixed assets instead gives nodes with
        tight bounds, which is what lets Panda throw away the two thirds of a
        forest that is behind the car.
        """
        root = Entity(parent=parent if parent is not None else scene)
        shader = shader if shader is not None else self.shader
        if shader is not None:
            root.shader = shader
        used = False
        for name, placements in groups.items():
            tmpl = self.template(name)
            if tmpl is None:
                continue
            for p in placements:
                x, z, yaw = p[0], p[1], p[2]
                s = p[3] if len(p) > 3 else 1.0
                copy = tmpl.copy_to(root)
                copy.set_pos(float(x), 0.0, float(z))
                if isinstance(s, (tuple, list)):
                    copy.set_scale(float(s[0]), float(s[1]), float(s[2]))
                else:
                    copy.set_scale(float(s))
                copy.set_h(-float(yaw))
                used = True
        if not used:
            from ursina import destroy
            destroy(root)
            return None
        root.clear_model_nodes()
        root.flatten_strong()
        return root

    def dispose(self):
        for t in self._templates.values():
            if t is not None:
                t.remove_node()
        self._templates.clear()


def tri_count(entity: Entity) -> int:
    """Triangles under an entity -- for keeping the roadside within budget."""
    from panda3d.core import GeomNode

    total = 0
    for gn in entity.find_all_matches("**/+GeomNode"):
        node = gn.node()
        for i in range(node.get_num_geoms()):
            total += node.get_geom(i).get_num_primitives() and sum(
                node.get_geom(i).get_primitive(k).get_num_faces()
                for k in range(node.get_geom(i).get_num_primitives()))
    return total


def yaw_at(track, i: int) -> float:
    """Heading of the centreline at sample *i*, in degrees (Ursina rotation_y)."""
    t = track.tangent[i]
    return math.degrees(math.atan2(t[0], t[1]))


def yaw_towards(direction) -> float:
    """rotation_y that points an unrotated model's +z along *direction*.

    Use this rather than adding 90 to a track heading and guessing the sign.
    The light posts were placed that way and ended up lighting the track
    lengthways instead of across it -- their arm reaches along the model's z,
    which a heading-plus-90 rule silently got backwards on one side.
    """
    return math.degrees(math.atan2(direction[0], direction[1]))
