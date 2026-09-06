"""Small geometry toolkit shared by the Blender asset scripts.

Everything here emits meshes with explicit, recalculated normals and no
modifier stack, so what an OBJ export sees is what the viewport shows.

Convention for the circuit kit (``circuit_kit.py``)
    -Y is the front -- the side that faces the track. +X is right, +Z is up,
    and the part sits on z = 0. That is Blender's own "front view" convention,
    so a part looks the right way round in the viewport with no mental
    gymnastics, and the baker turns it into the game's +z-forward space with a
    single proper rotation.

Anything meant to be *stretched* along a run (barriers, tyre walls, hoardings,
grandstand seating) must be built with ``extrude_x``: a cross-section in the
YZ plane swept along X. Stretching that along X only makes it longer, whereas
stretching a part built from boxes and cylinders makes it fat.
"""

import math

import bmesh
import bpy

TAU = math.tau


def sgn(v):
    return (v > 0) - (v < 0)


# ---------------------------------------------------------------- scene
def clear_scene():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.objects):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_material(name):
    """Materials carry no colour here -- the name *is* the paint role, and the
    baker maps it to a game colour. Keeping colour out of Blender means a
    livery change is a one-line edit on the game side, not a re-export."""
    mat = bpy.data.materials.get(name)
    return mat if mat else bpy.data.materials.new(name)


def assign(obj, mat_name):
    obj.data.materials.clear()
    obj.data.materials.append(make_material(mat_name))
    return obj


def new_object(name, verts, faces, mat=None, smooth=False):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if mat:
        assign(obj, mat)
    return obj


# ---------------------------------------------------------------- solids
def _rotate(p, rx, ry, rz):
    x, y, z = p
    if rx:
        c, s = math.cos(rx), math.sin(rx)
        y, z = y * c - z * s, y * s + z * c
    if ry:
        c, s = math.cos(ry), math.sin(ry)
        x, z = x * c + z * s, -x * s + z * c
    if rz:
        c, s = math.cos(rz), math.sin(rz)
        x, y = x * c - y * s, x * s + y * c
    return (x, y, z)


BOX_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 2, 6, 4),
             (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6)]


def box(name, center, size, mat=None, rx=0.0, ry=0.0, rz=0.0):
    """Axis box at *center* with *size*, optionally rotated about its centre.

    Rotations are in radians and applied X then Y then Z. ``ry`` is what a
    diagonal brace in the XZ plane needs; ``rx`` is what a fence post that
    kinks back over the track needs.
    """
    cx, cy, cz = center
    hx, hy, hz = (s * 0.5 for s in size)
    verts = []
    for dx in (-hx, hx):
        for dy in (-hy, hy):
            for dz in (-hz, hz):
                x, y, z = _rotate((dx, dy, dz), rx, ry, rz)
                verts.append((cx + x, cy + y, cz + z))
    return new_object(name, verts, BOX_FACES, mat)


def slab(name, x0, x1, y0, y1, z0, z1, mat=None):
    """A box given by its bounds -- easier to read than centre-plus-size when
    a part is being fitted between two known faces."""
    return box(name, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
               (x1 - x0, y1 - y0, z1 - z0), mat)


def extrude_x(name, profile, x0, x1, mat=None, smooth=False):
    """Sweep a closed YZ cross-section along X. The stretch-safe primitive."""
    n = len(profile)
    verts = [(x0, y, z) for y, z in profile] + [(x1, y, z) for y, z in profile]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(range(n, 2 * n)))
    return new_object(name, verts, faces, mat, smooth)


def circle_profile(cy, cz, r, n=12):
    """A closed circle in the YZ plane -- swept by extrude_x it becomes a
    horizontal cylinder, which is what a row of stacked tyres looks like."""
    return [(cy + r * math.cos(TAU * i / n), cz + r * math.sin(TAU * i / n))
            for i in range(n)]


def cylinder_z(name, center, radius, height, mat=None, n=12):
    """Upright cylinder -- posts, masts, floodlight columns."""
    cx, cy, cz = center
    ring = [(cx + radius * math.cos(TAU * i / n),
             cy + radius * math.sin(TAU * i / n)) for i in range(n)]
    verts = [(x, y, cz) for x, y in ring] + [(x, y, cz + height) for x, y in ring]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(range(n, 2 * n)))
    return new_object(name, verts, faces, mat, smooth=True)


def stair_profile(rows, run, rise, plinth=1.0):
    """Closed YZ profile of a raked seating deck, front (y=0) to back.

    Swept along X this is a grandstand tier that stretches to any run length
    without deforming -- which is the whole point: a circuit needs hundreds of
    metres of seating, and copying a fixed module instead gives a row of
    identical blocks with a visible seam at every join.
    """
    pts = [(0.0, 0.0), (0.0, plinth)]         # up the front face
    y, z = 0.0, plinth
    for _ in range(rows):                     # tread, then riser
        y += run
        pts.append((y, z))
        z += rise
        pts.append((y, z))
    pts.append((y, 0.0))                      # down the back; closes along z=0
    return pts


def ico(name, center, radius, mat=None, subdiv=1, scale=(1.0, 1.0, 1.0)):
    """Faceted blob. One subdivision is 80 flat triangles, which is what makes
    a low-poly canopy read as leaves rather than as a beach ball."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=radius,
                                          location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign(obj, mat) if mat else obj


def cone(name, center, r_low, r_high, depth, mat=None, verts=8):
    """Truncated cone, base at *center* z. Stacked, these are a conifer."""
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r_low, radius2=r_high,
                                    depth=depth,
                                    location=(center[0], center[1],
                                              center[2] + depth * 0.5))
    obj = bpy.context.object
    obj.name = name
    return assign(obj, mat) if mat else obj


def taper_z(name, center, r_low, r_high, height, mat=None, verts=8):
    """A trunk: a short tapered cylinder standing on *center*."""
    return cone(name, center, r_low, r_high, height, mat, verts)


# ---------------------------------------------------------------- export
def export_obj(path, apply_modifiers=False):
    """Write every mesh in the scene as one OBJ in Blender's own axes.

    forward_axis='Y', up_axis='Z' is the identity, so the file carries the
    convention documented at the top of this module and the baker has exactly
    one transform to reason about.
    """
    bpy.ops.object.select_all(action='SELECT')
    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not objs:
        raise RuntimeError("nothing to export: " + path)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.wm.obj_export(filepath=path,
                          export_selected_objects=True,
                          forward_axis='Y', up_axis='Z',
                          apply_modifiers=apply_modifiers,
                          export_materials=True,
                          export_normals=True,
                          export_uv=False,
                          export_triangulated_mesh=True,
                          path_mode='COPY')
    return objs
