#!/usr/bin/env python3
"""
blend_to_stl.py - export the donut objects from a .blend to STL. Runs INSIDE Blender.

    blender -b <file.blend> -P blend_to_stl.py -- <out_full.stl> <out_plain.stl>

Called by fetch_showcase.py; not useful standalone.

Two things here are not what a naive export would do:

1. Sprinkles are particle instances, not objects. Blender's tutorial scatters six
   sprinkle meshes over the Icing with a particle system, so the evaluated Icing mesh
   does NOT contain them and a plain object-by-object export silently produces a bald
   donut. They have to be collected from `depsgraph.object_instances`, which yields
   one entry per realised instance with its own world matrix.

2. Everything else in the scene is set dressing. The .blend also holds a cup, saucer,
   plate, liquid, countertop and wall. Only `Donut` and `Icing` (plus instances
   emitted by them) are wanted.

Normals are recomputed from triangle winding rather than trusted from the source,
since the point of the exercise is to feed a mesh to a segmenter that reads normals.
"""

import math
import struct
import sys

import bpy
import mathutils

KEEP = {"Donut", "Icing"}
TARGET_SPAN_MM = 90.0   # a real donut is about this across; keeps tolerances sane


def gather(with_sprinkles: bool):
    dg = bpy.context.evaluated_depsgraph_get()
    tris = []
    for inst in dg.object_instances:
        ob = inst.object
        if ob is None or ob.type != "MESH":
            continue
        if inst.is_instance:
            par = inst.parent
            pname = par.original.name if (par and par.original) else None
            if not with_sprinkles or pname not in KEEP:
                continue
        elif ob.original.name not in KEEP:
            continue

        me = ob.to_mesh()
        if me is None:
            continue
        me.calc_loop_triangles()
        mw = inst.matrix_world.copy()
        for lt in me.loop_triangles:
            p = [mw @ me.vertices[i].co for i in lt.vertices]
            n = (p[1] - p[0]).cross(p[2] - p[0])
            length = n.length
            n = (n / length) if length > 1e-20 else mathutils.Vector((0.0, 0.0, 1.0))
            tris.append((tuple(n), tuple(p[0]), tuple(p[1]), tuple(p[2])))
        ob.to_mesh_clear()
    return tris


def write_stl(path: str, tris, scale: float) -> None:
    with open(path, "wb") as fh:
        fh.write(b"Blender donut (adcimon/blender-donut, MIT) - held-out eval asset".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        for n, a, b, c in tris:
            fh.write(struct.pack(
                "<12fH", *n,
                *(x * scale for x in a), *(x * scale for x in b), *(x * scale for x in c), 0))


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 2:
        print("usage: ... -P blend_to_stl.py -- <out_full.stl> <out_plain.stl>")
        return
    for out, sprinkles in ((argv[0], True), (argv[1], False)):
        tris = gather(sprinkles)
        if not tris:
            print(f"ERROR no geometry for {out}")
            continue
        pts = [v for t in tris for v in t[1:]]
        span = max(max(p[i] for p in pts) - min(p[i] for p in pts) for i in (0, 1))
        scale = TARGET_SPAN_MM / span if span else 1.0
        dims = tuple((max(p[i] for p in pts) - min(p[i] for p in pts)) * scale for i in (0, 1, 2))
        write_stl(out, tris, scale)
        print(f"WROTE {out} tris={len(tris)} mm=({dims[0]:.1f},{dims[1]:.1f},{dims[2]:.1f})")


main()
