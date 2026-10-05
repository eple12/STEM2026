"""A leaner stand-in for Ursina's per-frame task.

Ursina's own ``_update`` visits every entity in the scene every frame -- some
350 in a grand prix -- and for each one reads ``enabled`` and ``ignore``
through its property machinery, walks up the parents (``has_disabled_
ancestor``) and probes for ``update``, ``scripts`` and continuous shader
inputs. Almost none of them have any of those. Profiled in a race that loop
cost more than the whole game's own update.

Worse, the default shader (``unlit_with_fog_shader``) declares
``camera_world_position`` as a *continuous* input, so ~30 entities had a
fresh shader input set every frame: a new ShaderAttrib and RenderState each
time, and Panda's state garbage collector (``garbageCollectStates``, 1.2 ms
a frame) cleaning them up.

``install()`` replaces the task with one that does the same things for the
entities that need them: the candidates (an ``update`` method or scripts)
are found when the entity list changes and re-checked once a second, and
the camera position goes to every default-shader entity through one shared
array written in place. Nothing else about Ursina changes.
"""
from __future__ import annotations

import __main__

from panda3d.core import LVecBase3f, PTA_LVecBase3f

#: The camera's world position, shared by every entity on Ursina's default
#: shader (see install).
CAMERA_POS = PTA_LVecBase3f.empty_array(1)
_RESCAN_FRAMES = 60

_state = {"key": None, "cands": [], "frames": 0}


def prepare():
    """Before the window opens: Panda's pipelined renderer (config.
    RENDER_THREADING -- cull and draw on their own threads, overlapping the
    Python frame instead of following it) and the shared camera input."""
    from panda3d.core import loadPrcFileData

    from . import config
    if config.RENDER_THREADING:
        loadPrcFileData("", f"threading-model {config.RENDER_THREADING}")
    patch_default_shader()


def patch_default_shader():
    """Before any entity exists: the default shader's per-frame input
    becomes one shared array (set on each entity once, as its default
    input) instead of a value pushed to each entity every frame."""
    from ursina.shaders.unlit_with_fog_shader import unlit_with_fog_shader as sh
    sh.default_input["camera_world_position"] = CAMERA_POS
    sh.continuous_input.pop("camera_world_position", None)


def _rescan(scene):
    cands = []
    for e in scene.entities:
        try:
            upd = getattr(e, "update", None)
            scripts = getattr(e, "scripts", None)
            sh = getattr(e, "shader", None)
        except Exception:
            continue
        cont = getattr(sh, "continuous_input", None) if sh is not None else None
        if callable(upd) or scripts or cont:
            cands.append(e)
    _state["cands"] = cands
    _state["key"] = (id(scene.entities), len(scene.entities))
    _state["frames"] = 0


def install(app):
    """Swap Ursina's per-frame task for the lean one."""
    import builtins

    from ursina import application, camera, mouse, scene
    from ursina import time as utime
    from ursina.audio import _audio_manager
    from panda3d.core import ClockObject

    clock = ClockObject.get_global_clock()
    # Mouse picking -- a ray cast through the UI and the world every frame,
    # then a walk over every entity to clear hover flags -- is for clickable
    # entities, and nothing here is clicked: the game is driven from the
    # keyboard. The pointer's position and velocity (the free camera looks
    # with them) are still tracked; only the picking never comes round.
    mouse.update_step = 1 << 30
    render = builtins.render
    patch_default_shader()

    def _update(task=None):
        if application.calculate_dt:
            utime.dt_unscaled = clock.get_dt()
            utime.dt = utime.dt_unscaled * application.time_scale
        mouse.update()
        p = camera.get_pos(render)
        CAMERA_POS[0] = LVecBase3f(p[0], p[1], p[2])

        main_update = getattr(__main__, "update", None)
        if main_update and not application.paused:
            main_update()

        for seq in application.sequences:
            seq.update()

        if scene._entities_marked_for_removal:
            gone = scene._entities_marked_for_removal
            scene.entities = [e for e in scene.entities if e not in gone]
            gone.clear()

        key = (id(scene.entities), len(scene.entities))
        _state["frames"] += 1
        if key != _state["key"] or _state["frames"] >= _RESCAN_FRAMES:
            _rescan(scene)

        removed = scene._entities_marked_for_removal
        for e in _state["cands"]:
            if not e or e in removed:
                continue
            if not e.enabled or e.ignore:
                continue
            if application.paused and e.ignore_paused is False:
                continue
            if e.has_disabled_ancestor():
                continue
            upd = getattr(e, "update", None)
            if callable(upd):
                upd()
            if not e:
                continue
            for script in getattr(e, "scripts", ()):
                if script.enabled and callable(getattr(script, "update", None)):
                    script.update()
            if not e:
                continue
            sh = e.shader
            cont = getattr(sh, "continuous_input", None) if sh else None
            if cont:
                for k, value in cont.items():
                    e.set_shader_input(k, value())

        _audio_manager.update()
        return task.cont if task is not None else None

    app.taskMgr.remove(app._update_task)
    app._update_task = app.taskMgr.add(_update, "update")
