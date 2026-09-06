"""
F1 스타일 레이싱카 절차적 생성 스크립트 (Blender 3.x ~ 5.x)

사용법
  1) Blender 실행 -> 상단 Scripting 탭 -> Open 으로 이 파일 열고 Run Script (Alt+P)
  2) 커맨드라인 렌더:
       blender -b -P f1_car.py -- --render hero.png
  3) 커맨드라인 내보내기:
       blender -b -P f1_car.py -- --export car.glb

좌표계
  노즈 끝이 x=0, 리어 윙이 x~5.4에 오도록 뒤로 눕혀 놓았다. 즉 +X는 차의
  뒤쪽이고 차가 "바라보는" 방향은 -X다 -- 이 모델을 엔진에 실을 때 여기를
  +X 전방으로 잘못 읽으면 차가 180도 돌아간 채로 달린다.
  +Y = 좌측, +Z = 위. 전장 약 5.7m로 실제 F1 규격에 근사.

설계 메모
  차체 단면은 BODY_PROFILE 테이블 하나로 정의되고, 서스펜션 부착점은
  body_section()으로 그 테이블에서 역산한다. 따라서 차체 실루엣을 바꿔도
  암 끝이 허공에 뜨지 않는다.
"""

import bpy
import bmesh
import math
import os
import sys
from mathutils import Vector

# ---------------------------------------------------------------- 설정
CLEAR_SCENE = True                 # 기존 씬 오브젝트 삭제
JOIN_PARTS = False                 # True면 모든 파트를 하나로 합침
LIVERY = (0.75, 0.05, 0.06)        # 메인 컬러 (기본: 레드)

TAU = math.tau
AIM = Vector((2.4, 0.0, 0.45))     # 카메라/조명이 바라보는 지점

# 차체 단면: (x, 반폭, 반높이, 중심 z) -- 노즈 팁 -> 테일
BODY_PROFILE = [
    (-0.03, 0.020, 0.0238, 0.3038),  # 둥근 노즈 팁
    (0.00, 0.055, 0.0594, 0.3144),
    (0.35, 0.090, 0.0924, 0.3224),
    (0.80, 0.135, 0.1320, 0.3520),
    (1.25, 0.190, 0.1782, 0.4032),
    (1.70, 0.270, 0.2376, 0.4576),   # 프런트 벌크헤드
    (2.15, 0.340, 0.2838, 0.4988),
    (2.55, 0.375, 0.3102, 0.5252),   # 콕핏
    (3.00, 0.385, 0.3300, 0.5500),
    (3.45, 0.370, 0.3564, 0.5864),   # 엔진
    (3.95, 0.320, 0.3300, 0.5800),
    (4.45, 0.240, 0.2640, 0.5440),
    (4.90, 0.150, 0.1716, 0.4916),
    (5.20, 0.080, 0.0990, 0.4440),   # 테일
]

# 콕핏 개구부 (상면도 기준 슈퍼타원)
COCKPIT_CX, COCKPIT_RX, COCKPIT_RY = 2.60, 0.38, 0.175
COCKPIT_FLOOR = 0.50               # 차체를 파낼 커터의 바닥 높이
COCKPIT_SEAT = 0.63                # 어두운 라이너의 바닥(시트가 놓이는 면)


def sgn(v):
    return (v > 0) - (v < 0)


def body_section(x):
    """BODY_PROFILE을 선형 보간해 임의의 x에서 (반폭, 반높이, 중심 z)를 얻는다."""
    if x <= BODY_PROFILE[0][0]:
        return BODY_PROFILE[0][1:]
    if x >= BODY_PROFILE[-1][0]:
        return BODY_PROFILE[-1][1:]
    for (x0, w0, h0, z0), (x1, w1, h1, z1) in zip(BODY_PROFILE, BODY_PROFILE[1:]):
        if x0 <= x <= x1:
            f = (x - x0) / (x1 - x0)
            return (w0 + (w1 - w0) * f, h0 + (h1 - h0) * f, z0 + (z1 - z0) * f)
    return BODY_PROFILE[-1][1:]


def chassis_anchor(x, side, z_frac):
    """차체 표면보다 확실히 안쪽인 부착점. 서스펜션 암이 허공에서 끊기지 않게 한다."""
    w, h, cz = body_section(x)
    return (x, side * w * 0.80, cz + h * z_frac)


# ---------------------------------------------------------------- 유틸
def clear_scene():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.curves):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_material(name, color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
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


def bevel(obj, width=0.02, segments=2):
    mod = obj.modifiers.new("Bevel", 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(40)
    return obj


def auto_smooth(obj, angle_deg=35.0):
    """스무스 셰이딩 + 각진 모서리는 분리 -- 타이어 숄더 같은 경계가 뭉개지지 않는다."""
    try:
        mod = obj.modifiers.new("EdgeSplit", 'EDGE_SPLIT')
        mod.split_angle = math.radians(angle_deg)
        mod.use_edge_angle = True
        mod.use_edge_sharp = False
    except TypeError:
        pass
    return obj


def mirror_y(obj):
    mod = obj.modifiers.new("Mirror", 'MIRROR')
    mod.use_axis = (False, True, False)
    return obj


def boolean_cut(target, cutter):
    """cutter 모양만큼 target에서 파낸다. cutter는 숨기되 모디파이어는 계속 참조한다."""
    mod = target.modifiers.new("Cut_" + cutter.name, 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = cutter
    try:
        mod.solver = 'EXACT'
    except (AttributeError, TypeError):
        pass
    cutter.display_type = 'WIRE'
    cutter.hide_render = True
    cutter.hide_viewport = True
    return target


# ---------------------------------------------------------------- 형상 헬퍼
def superellipse(t, half_w, half_h, e=2.6):
    p = 2.0 / e
    c, s = math.cos(t), math.sin(t)
    return (half_w * sgn(c) * (abs(c) ** p), half_h * sgn(s) * (abs(s) ** p))


def superellipse_ring(x, half_w, half_h, cz, n=20, e=2.6):
    """YZ 평면 슈퍼타원 단면. e=2면 정확한 타원, 값이 클수록 사각형에 가까워진다."""
    pts = []
    for i in range(n):
        y, dz = superellipse(TAU * i / n, half_w, half_h, e)
        pts.append((x, y, cz + dz))
    return pts


def loft(sections, name, mat=None, smooth=True):
    """sections: 정점 수가 같은 링들의 리스트(앞->뒤). 튜브를 만들고 양 끝을 막는다."""
    n = len(sections[0])
    verts = [v for ring in sections for v in ring]
    faces = []
    for r in range(len(sections) - 1):
        a, b = r * n, (r + 1) * n
        for j in range(n):
            k = (j + 1) % n
            faces.append((a + j, a + k, b + k, b + j))
    faces.append(tuple(range(n - 1, -1, -1)))
    base = (len(sections) - 1) * n
    faces.append(tuple(range(base, base + n)))
    return new_object(name, verts, faces, mat, smooth)


def revolve(name, profile, center, segments, mat, smooth=True):
    """단면(반지름, Y오프셋) 폐곡선을 Y축 둘레로 회전 -- 구멍 뚫린 타이어/림용."""
    cx, cy, cz = center
    p = len(profile)
    verts = []
    for s in range(segments):
        t = TAU * s / segments
        c, si = math.cos(t), math.sin(t)
        for r, dy in profile:
            verts.append((cx + r * c, cy + dy, cz + r * si))
    faces = []
    for s in range(segments):
        s2 = (s + 1) % segments
        for i in range(p):
            j = (i + 1) % p
            faces.append((s * p + i, s * p + j, s2 * p + j, s2 * p + i))
    return new_object(name, verts, faces, mat, smooth)


def prism(name, outline, z0, z1, mat=None):
    """XY 외곽선을 Z로 수직 압출한 닫힌 솔리드 (불리언 커터/콕핏 터브용)."""
    n = len(outline)
    verts = [(x, y, z0) for x, y in outline] + [(x, y, z1) for x, y in outline]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(range(n, 2 * n)))
    return new_object(name, verts, faces, mat)


def box(name, center, size, mat=None, rot_y=0.0):
    """중심/크기로 박스 생성. rot_y는 Y축 회전(윙 받음각, 휠 스포크)."""
    cx, cy, cz = center
    sx, sy, sz = (s * 0.5 for s in size)
    verts = []
    for dx in (-sx, sx):
        for dy in (-sy, sy):
            for dz in (-sz, sz):
                x, z = dx, dz
                if rot_y:
                    c, s = math.cos(rot_y), math.sin(rot_y)
                    x, z = dx * c + dz * s, -dx * s + dz * c
                verts.append((cx + x, cy + dy, cz + z))
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 2, 6, 4),
             (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6)]
    return new_object(name, verts, faces, mat)


def disc(name, center, radius, depth, mat, segments=32):
    """Y축을 축으로 하는 원판/원기둥. 캡은 플랫 셰이딩."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=segments, radius=radius, depth=depth,
                                        location=center,
                                        rotation=(math.radians(90), 0, 0))
    obj = bpy.context.object
    obj.name = name
    for poly in obj.data.polygons:
        poly.use_smooth = abs(poly.normal.z) < 0.9
    return assign(obj, mat)


def rod(a, b, radius, mat, verts=10):
    """두 점을 잇는 원기둥 -- 서스펜션 암, 필러용."""
    a, b = Vector(a), Vector(b)
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius,
                                        depth=d.length, location=(a + b) / 2)
    obj = bpy.context.object
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = d.to_track_quat('Z', 'Y')
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return assign(obj, mat)


def tube(points, radius, name, mat, resolution=6):
    """점 목록을 따라가는 매끈한 튜브. 원기둥을 이어붙일 때 생기는 이음매가 없다."""
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = radius
    curve.bevel_resolution = resolution
    curve.use_fill_caps = True
    spline = curve.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for pt, (x, y, z) in zip(spline.points, points):
        pt.co = (x, y, z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(mat)
    return obj


def cockpit_outline(scale=1.0, n=28):
    pts = []
    for i in range(n):
        dx, dy = superellipse(TAU * i / n, COCKPIT_RX * scale,
                              COCKPIT_RY * scale, e=3.0)
        pts.append((COCKPIT_CX + dx, dy))
    return pts


# ---------------------------------------------------------------- 파트
def build_body(mat):
    """노즈 -> 모노코크 -> 엔진커버 -> 테일."""
    rings = [superellipse_ring(x, w, h, cz) for x, w, h, cz in BODY_PROFILE]
    return loft(rings, "Body_Monocoque", mat)


def build_airbox(mat):
    """드라이버 뒤 에어박스 인테이크에서 엔진커버 핀으로 이어지는 형상."""
    profile = [
        (3.04, 0.118, 0.108, 0.96),   # 콕핏 개구부(x<=2.98) 뒤에서 시작
        (3.20, 0.158, 0.146, 0.94),
        (3.55, 0.148, 0.130, 0.88),
        (4.10, 0.100, 0.092, 0.81),
        (4.70, 0.056, 0.056, 0.74),
        (5.05, 0.030, 0.034, 0.68),
    ]
    rings = [superellipse_ring(x, w, h, cz, e=3.0) for x, w, h, cz in profile]
    return loft(rings, "Airbox", mat)


def build_sidepod(mat):
    """좌측 사이드포드만 만들고 미러 모디파이어로 우측 생성."""
    profile = [  # (x, 반폭, 반높이, 중심 z, 좌측 오프셋)
        (2.05, 0.120, 0.1782, 0.4032, 0.46),   # 인테이크 입구
        (2.35, 0.225, 0.2508, 0.4208, 0.46),
        (2.80, 0.245, 0.2508, 0.4108, 0.45),
        (3.30, 0.205, 0.2178, 0.4028, 0.42),
        (3.80, 0.130, 0.1650, 0.3800, 0.36),   # 코크보틀로 조여든다
        (4.20, 0.070, 0.1122, 0.3672, 0.28),
    ]
    rings = []
    for x, w, h, cz, off in profile:
        ring = superellipse_ring(x, w, h, cz, e=3.4)
        rings.append([(px, py + off, pz) for px, py, pz in ring])
    return mirror_y(loft(rings, "Sidepod", mat))


def build_floor(mat):
    """플로어 + 뒤로 갈수록 들리는 디퓨저."""
    z = 0.09
    outline = [
        (1.55,  0.30, z), (1.55, -0.30, z),
        (2.20,  0.70, z), (2.20, -0.70, z),
        (4.10,  0.70, z), (4.10, -0.70, z),
        (4.60,  0.62, z), (4.60, -0.62, z),
        (5.15,  0.55, z + 0.26), (5.15, -0.55, z + 0.26),
    ]
    n = len(outline)
    verts = outline + [(x, y, zz + 0.03) for x, y, zz in outline]
    faces = []
    for i in range(0, n - 2, 2):
        faces.append((i, i + 2, i + 3, i + 1))
        faces.append((n + i, n + i + 1, n + i + 3, n + i + 2))
        faces.append((i, i + 1, n + i + 1, n + i))
        faces.append((i + 2, n + i + 2, n + i + 3, i + 3))
    faces.append((n - 2, n - 1, 2 * n - 1, 2 * n - 2))
    return new_object("Floor_Diffuser", verts, faces, mat)


def build_front_wing(mat_body, mat_livery):
    """3엘리먼트 프런트 윙 + 엔드플레이트 + 노즈 필러.

    노즈 팁(x=0)이 메인 플레인 바로 위에 오도록 윙을 앞으로 당겨서,
    필러 두 개가 윙 -> 노즈 아랫면을 실제로 이어준다.
    """
    parts = []
    elements = [  # (x, z, 코드, 두께, 받음각)
        (-0.06, 0.105, 0.32, 0.026, math.radians(-4)),
        (0.16, 0.170, 0.27, 0.024, math.radians(-11)),
        (0.34, 0.240, 0.22, 0.022, math.radians(-19)),
    ]
    for i, (x, z, chord, th, ang) in enumerate(elements):
        mat = mat_livery if i == 0 else mat_body
        parts.append(bevel(box("FrontWing_E%d" % i, (x, 0, z),
                               (chord, 1.80, th), mat, rot_y=ang), 0.008, 2))
    for side, tag in ((1, "L"), (-1, "R")):
        parts.append(bevel(box("FrontWing_EP_" + tag, (0.14, side * 0.90, 0.215),
                               (0.68, 0.022, 0.44), mat_livery), 0.012, 2))
        # 필러: 메인 플레인 내부(x=0.00) -> 노즈 아랫면 내부(x=0.28)
        parts.append(rod((0.00, side * 0.085, 0.125),
                         (0.28, side * 0.060, 0.292), 0.018, mat_body))
    return parts


def build_rear_wing(mat_body, mat_livery):
    """2엘리먼트 리어 윙 + 엔드플레이트 + 빔 윙 + 중앙 필러."""
    parts = [
        bevel(box("RearWing_Main", (5.15, 0, 1.06), (0.28, 1.05, 0.030),
                  mat_livery, rot_y=math.radians(-12)), 0.008, 2),
        bevel(box("RearWing_Flap", (5.34, 0, 1.17), (0.18, 1.05, 0.026),
                  mat_body, rot_y=math.radians(-30)), 0.008, 2),
        bevel(box("BeamWing", (5.12, 0, 0.60), (0.20, 0.90, 0.026),
                  mat_body, rot_y=math.radians(-14)), 0.008, 2),
        rod((5.05, 0.0, 0.50), (5.18, 0.0, 1.05), 0.030, mat_body),
    ]
    for side, tag in ((1, "L"), (-1, "R")):
        parts.append(bevel(box("RearWing_EP_" + tag, (5.20, side * 0.53, 0.96),
                               (0.55, 0.024, 0.58), mat_livery), 0.012, 2))
    return parts


def build_cockpit(body, mat_dark, mat_body):
    """차체에 실제로 구멍을 뚫고(불리언) 그 안에 어두운 터브/시트/스티어링을 넣는다."""
    cutter = prism("Cockpit_Cutter", cockpit_outline(1.00), COCKPIT_FLOOR, 1.55)
    boolean_cut(body, cutter)

    # 라이너 -- 속을 파낸 어두운 그릇. 잘린 차체 단면(리버리 색)을 가려서
    # 개구부가 얕은 홈이 아니라 실제로 뚫린 구멍으로 읽히게 한다.
    liner = prism("Cockpit_Liner", cockpit_outline(0.985), 0.30, 0.760, mat_dark)
    liner_cut = prism("Cockpit_LinerCut", cockpit_outline(0.88), COCKPIT_SEAT, 1.35)
    boolean_cut(liner, liner_cut)

    # 스티어링 휠 -- 드라이버를 향해 세운 링
    bpy.ops.mesh.primitive_torus_add(major_radius=0.100, minor_radius=0.017,
                                     major_segments=24, minor_segments=8,
                                     location=(2.40, 0, 0.710),
                                     rotation=(0, math.radians(68), 0))
    wheel = bpy.context.object
    wheel.name = "SteeringWheel"
    assign(wheel, mat_dark)

    return [
        cutter, liner, liner_cut,
        bevel(box("Seat", (2.84, 0, 0.720), (0.13, 0.30, 0.20), mat_dark), 0.04, 2),
        # 헤드레스트 -- 개구부 뒤쪽을 감싸는 어두운 패딩
        bevel(box("Headrest", (2.92, 0, 0.735), (0.20, 0.40, 0.17), mat_dark), 0.05, 3),
        wheel,
        rod((2.52, 0, 0.650), (2.41, 0, 0.710), 0.020, mat_dark),
    ]


def build_halo(mat):
    """헤일로 -- 아크 양 끝을 차체 안쪽까지 밀어 넣어 실제로 섀시에 물리게 한다."""
    steps = 32
    arc = []
    for i in range(steps + 1):
        t = -math.pi / 2 + math.pi * i / steps      # 우 -> 앞 -> 좌
        arc.append((2.30 + 0.60 * (1 - math.cos(t)),
                    0.44 * math.sin(t),
                    1.03 + 0.05 * math.cos(t)))
    # 양 끝을 콕핏 뒤 차체 측면 속으로 내려 꽂는다
    arc = ([(3.10, -0.28, 0.66), (3.02, -0.38, 0.86)] + arc
           + [(3.02, 0.38, 0.86), (3.10, 0.28, 0.66)])
    return [
        tube(arc, 0.033, "Halo_Ring", mat),
        # 중앙 필러도 개구부 앞쪽 차체 내부(x=2.14)까지 내려 꽂는다
        tube([(2.14, 0.0, 0.68), (2.22, 0.0, 0.86), (2.30, 0.0, 1.03)],
             0.036, "Halo_Pillar", mat),
    ]


def build_wheel(x, y, radius, width, mat_tire, mat_rim, mat_dark, name):
    """단면 회전으로 만든 링 타이어 + 배럴 / 스포크 / 허브 / 브레이크 디스크.

    바깥면이 밋밋한 원판이 되지 않도록 타이어에 실제 보어를 뚫고
    그 안쪽에 깊이가 있는 휠 구조를 넣는다.
    """
    side = 1 if y > 0 else -1
    hw = width / 2.0
    bore = radius * 0.64
    sh = 0.075                                   # 숄더 라운드
    parts = []

    tire_profile = [
        (bore, -hw),
        (radius - sh, -hw),
        (radius - sh * 0.28, -hw + sh * 0.55),
        (radius, -hw + sh),
        (radius, hw - sh),
        (radius - sh * 0.28, hw - sh * 0.55),
        (radius - sh, hw),
        (bore, hw),
    ]
    parts.append(auto_smooth(revolve("Tire_" + name, tire_profile,
                                     (x, y, radius), 56, mat_tire)))

    barrel = [(bore, -hw * 0.94), (bore, hw * 0.94),
              (bore - 0.028, hw * 0.94), (bore - 0.028, -hw * 0.94)]
    parts.append(auto_smooth(revolve("RimBarrel_" + name, barrel,
                                     (x, y, radius), 40, mat_rim)))

    # 안쪽으로 들어간 어두운 페이스 -- 스포크 사이로 깊이가 보인다
    parts.append(disc("RimFace_" + name, (x, y + side * (hw - 0.09), radius),
                      bore - 0.02, 0.025, mat_dark, 32))
    # 반대편에서 보이는 브레이크 디스크
    parts.append(disc("Brake_" + name, (x, y - side * (hw - 0.06), radius),
                      bore * 0.86, 0.035, mat_dark, 28))

    spoke_len = bore - 0.085
    rm = (bore + 0.085) * 0.5
    for i in range(5):
        a = TAU * i / 5 + 0.3
        parts.append(box("Spoke_%s_%d" % (name, i),
                         (x + rm * math.cos(a), y + side * (hw - 0.045),
                          radius + rm * math.sin(a)),
                         (spoke_len, 0.045, 0.05), mat_rim, rot_y=-a))

    parts.append(disc("Hub_" + name, (x, y + side * (hw - 0.02), radius),
                      0.075, 0.10, mat_rim, 12))
    return parts


def build_suspension(x_axle, y_track, hub_z, mat, reach=0.26):
    """더블 위시본 근사. 섀시 쪽 끝은 body_section()으로 차체 안쪽에 고정한다."""
    parts = []
    y_hub = y_track - 0.17
    for side in (1, -1):
        upper_hub = (x_axle, side * y_hub, hub_z + 0.09)
        lower_hub = (x_axle, side * y_hub, hub_z - 0.14)
        # 업라이트 -- 암 두 개가 허공에서 만나지 않도록 휠 허브를 잇는다
        parts.append(bevel(box("Upright_%.0f_%d" % (x_axle * 100, side),
                               (x_axle, side * (y_hub - 0.01), hub_z - 0.025),
                               (0.10, 0.085, 0.30), mat), 0.02, 2))
        for dx in (-reach, reach):
            parts.append(rod(upper_hub, chassis_anchor(x_axle + dx, side, 0.35),
                             0.022, mat))
            parts.append(rod(lower_hub, chassis_anchor(x_axle + dx, side, -0.55),
                             0.024, mat))
        # 푸시로드
        parts.append(rod(lower_hub, chassis_anchor(x_axle - reach * 0.6, side, 0.55),
                         0.020, mat))
    return parts


# ---------------------------------------------------------------- 조립
def build_car():
    carbon = make_material("Carbon", (0.035, 0.035, 0.040), 0.35, 0.35)
    livery = make_material("Livery", LIVERY, 0.25, 0.28)
    rubber = make_material("Rubber", (0.020, 0.020, 0.022), 0.0, 0.85)
    metal = make_material("Metal", (0.62, 0.63, 0.66), 1.0, 0.25)
    dark = make_material("Interior", (0.012, 0.012, 0.014), 0.0, 0.9)

    body = build_body(livery)
    parts = [body, build_airbox(carbon), build_sidepod(livery), build_floor(carbon)]
    parts += build_front_wing(carbon, livery)
    parts += build_rear_wing(carbon, livery)
    parts += build_cockpit(body, dark, livery)
    parts += build_halo(carbon)

    front_r, rear_r = 0.355, 0.375
    parts += build_wheel(1.30, 0.75, front_r, 0.32, rubber, metal, dark, "FL")
    parts += build_wheel(1.30, -0.75, front_r, 0.32, rubber, metal, dark, "FR")
    parts += build_wheel(4.35, 0.72, rear_r, 0.42, rubber, metal, dark, "RL")
    parts += build_wheel(4.35, -0.72, rear_r, 0.42, rubber, metal, dark, "RR")
    parts += build_suspension(1.30, 0.75, front_r, carbon)
    parts += build_suspension(4.35, 0.72, rear_r, carbon)

    if JOIN_PARTS:
        joinable = [p for p in parts if p.type == 'MESH' and not p.hide_render]
        bpy.ops.object.select_all(action='DESELECT')
        for p in joinable:
            p.select_set(True)
        bpy.context.view_layer.objects.active = joinable[0]
        bpy.ops.object.join()
        joinable[0].name = "F1_Car"
    return parts


def setup_scene():
    """바닥 + 3점 조명 + 프런트 3/4 히어로 앵글."""
    bpy.ops.mesh.primitive_plane_add(size=40, location=(2.7, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    assign(ground, make_material("Ground", (0.16, 0.17, 0.18), 0.0, 0.6))

    bpy.ops.object.empty_add(location=AIM)
    target = bpy.context.object
    target.name = "CameraTarget"

    bpy.ops.object.camera_add(location=(-6.2, -9.0, 2.9))
    cam = bpy.context.object
    cam.data.lens = 72
    track = cam.constraints.new('TRACK_TO')
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam

    for loc, energy, size in (((6, -6, 8), 3000, 6),
                              ((-6, 5, 6), 1200, 8),
                              ((2, 8, 3), 700, 6)):
        bpy.ops.object.light_add(type='AREA', location=loc)
        light = bpy.context.object
        light.data.energy = energy
        light.data.size = size
        light.rotation_euler = (AIM - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()


def export(path):
    path = os.path.abspath(path)
    bpy.ops.object.select_all(action='SELECT')
    low = path.lower()
    if low.endswith((".glb", ".gltf")):
        bpy.ops.export_scene.gltf(filepath=path)
    elif low.endswith(".fbx"):
        bpy.ops.export_scene.fbx(filepath=path)
    elif low.endswith(".obj"):
        try:
            bpy.ops.wm.obj_export(filepath=path)      # Blender 4.x+
        except AttributeError:
            bpy.ops.export_scene.obj(filepath=path)   # Blender 3.x
    elif low.endswith(".blend"):
        bpy.ops.wm.save_as_mainfile(filepath=path)
    else:
        raise ValueError("지원하지 않는 확장자: " + path)
    print("[f1_car] exported -> " + path)


WHEEL_TAGS = ("FL", "FR", "RL", "RR")
#: 불리언 커터 -- 형상에는 기여하지만 그 자체는 내보내지 않는다.
CUTTERS = ("Cockpit_Cutter", "Cockpit_LinerCut")


def export_game_obj(out_dir):
    """게임 엔진용 OBJ 파트 5개(body + 휠 4개)를 Blender 원좌표 그대로 내보낸다.

    forward_axis='Y', up_axis='Z'가 Blender 네이티브(변환 없음)라서 좌표는
    위 docstring 그대로 나간다 -- 노즈가 -X 쪽, +Y=좌, +Z=상, 미터 단위.
    휠을 따로 빼는 이유는 게임이 휠 노드를 찾아 조향/회전시키기 때문이다.
    """
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    groups = dict((tag, []) for tag in WHEEL_TAGS)
    body = []
    for obj in bpy.context.scene.objects:
        if obj.type not in {'MESH', 'CURVE'} or obj.name in CUTTERS:
            continue
        if obj.name == "Ground":
            continue
        hit = set(obj.name.split("_")) & set(WHEEL_TAGS)
        (groups[hit.pop()] if hit else body).append(obj)

    written = []
    for name, objs in [("body", body)] + [(t, groups[t]) for t in WHEEL_TAGS]:
        if not objs:
            raise RuntimeError("내보낼 오브젝트가 없다: " + name)
        bpy.ops.object.select_all(action='DESELECT')
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        path = os.path.join(out_dir, name + ".obj")
        bpy.ops.wm.obj_export(filepath=path,
                              export_selected_objects=True,
                              forward_axis='Y', up_axis='Z',
                              apply_modifiers=True,
                              export_eval_mode='DAG_EVAL_RENDER',
                              export_materials=True,
                              export_normals=True,
                              export_uv=False,
                              export_triangulated_mesh=True,
                              path_mode='COPY')
        written.append((name, len(objs), os.path.getsize(path)))
    for name, count, size in written:
        print("[f1_car] %-6s %2d objects  %6.1f KB" % (name, count, size / 1024.0))
    print("[f1_car] game OBJ parts -> " + out_dir)


def render(path, samples=64):
    path = os.path.abspath(path)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1000
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("[f1_car] rendered -> " + path)


def main():
    if CLEAR_SCENE:
        clear_scene()
    build_car()
    setup_scene()

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--export" in argv:
        export(argv[argv.index("--export") + 1])
    if "--game-obj" in argv:
        export_game_obj(argv[argv.index("--game-obj") + 1])
    if "--render" in argv:
        render(argv[argv.index("--render") + 1])


if __name__ == "__main__":
    main()
