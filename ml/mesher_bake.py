#!/usr/bin/env python3
"""
mesher_bake.py - re-mesh baked ABC parts with other meshers, in bake format.

WHY THIS EXISTS

The converter's input is a mesh from anywhere, not an OCCT mesh. Measured on 150
validation parts, the seeded-fill model reads OCCT's triangulation pattern more than the
surface. Identical torus faces are predicted Torus 84% of the time when OCCT meshes them
as tori. After the whole part is NURBS-converted and OCCT meshes it again, that drops to
3%, and membership IoU falls from 0.866 to 0.730. The cue is mostly relative triangle area.
A model trained only on OCCT meshes of ABC is partly a detector of OCCT's meshing of each
stored type. These variants give training and evaluation other meshers' patterns, over
the same parts and labels.

VARIANTS

  nurbs   BRepBuilderAPI_NurbsConvert on the whole shape, then OCCT's BRepMesh at the
          usual density ladder. The geometry is identical, the triangulation is a
          knot-grid pattern, and the face order is preserved, so labels are exact.
  gmsh    gmsh's own OCC import and Frontal-Delaunay surface mesher, sized from curvature
          with a maximum element size. It is a different meshing algorithm entirely, with
          near-isotropic triangles rather than a UV grid. gmsh surfaces are matched to
          OCCT faces one-to-one by centroid and area. They are not matched by order,
          because TopExp_Explorer can visit a shared face twice, and not by bounding box,
          because the two kernels pad boxes differently (gaps of up to 4e-3 of the
          diagonal were measured). Triangles are flipped
          where gmsh's winding disagrees with the face's outward normal, since the model
          and the fitters both read facet normals.

OUTPUT

  <out>/<model_id[:5]>/<model_id>_<part>.npz, with the same keys as stage1_bake.py:
  {density}_positions, {density}_face_id and {density}_face_type, plus model_id, part,
  diag and mesher.
  face_id indexes the ORIGINAL shape's TopExp_Explorer faces, so the relabel sidecars
  (canonical types) apply unchanged. face_type is the original STORED type; NURBS
  conversion would otherwise report every face as BSpline.
  Parts are drawn from the main bake's own split (--split val|train|all), so a variant
  validation set contains exactly the main validation parts.

Every part runs in an isolated worker (isolated_pool.py): NurbsConvert segfaults on
some shapes, and gmsh can stall.

Usage:
    python mesher_bake.py --variant gmsh --split val --out D:/abc-data/bake_gmsh --jobs 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_TRIS = 32
MAX_TRIS = 400_000

# (density name, MeshSizeFromCurvature = elements per 2*pi of curvature, max size / diag).
# Calibrated so gmsh's median triangle count per part tracks OCCT's ladder, so that a
# difference between meshers is a difference in pattern rather than in density.
# Measured median gmsh/OCCT triangle ratios on 20 validation parts: (6, 0.25) is 1.59 x coarse,
# (8, 0.15) is 1.10 x medium, (12, 0.08) is 0.70 x fine and (16, 0.05) is 1.21 x fine.
# Those settings still gave 1.5-1.9 x OCCT on a 40-part bake (p90 5 x), and widening the
# maximum size changed nothing: gmsh extends each curved edge's small elements into the
# face interior (Mesh.MeshSizeExtendFromBoundary, on by default). With extension OFF, a
# 12-part sweep gave (5, 0.5) 1.03 x coarse (p90 2.0), (12, 0.08) 1.06 x medium (p90 2.7)
# and (20, 0.05) 0.85 x fine (p90 1.6); fine is interpolated to about 1.0.
GMSH_EXTEND_FROM_BOUNDARY = 0
GMSH_LADDER = (
    ("coarse", 5, 0.5),
    ("medium", 12, 0.08),
    ("fine", 22, 0.045),
)


def _explorer_faces(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    out = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        out.append(TopoDS.Face_s(ex.Current()))
        ex.Next()
    return out


def _bake_nurbs(shape, faces, diag):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
    from occ_tessellate import DENSITY_LADDER, tessellate

    conv = BRepBuilderAPI_NurbsConvert(shape, True).Shape()
    if len(_explorer_faces(conv)) != len(faces):
        return None, "face_count_changed"
    arrays = {}
    for name, lin, ang in DENSITY_LADDER:
        t = tessellate(conv, lin, ang, diag)
        if MIN_TRIS <= len(t) <= MAX_TRIS:
            arrays[f"{name}_positions"] = t.positions
            arrays[f"{name}_face_id"] = t.face_id
    return arrays, "ok"


def _bake_gmsh(step_path, faces, diag):
    import gmsh
    from OCP.BRep import BRep_Tool
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_REVERSED

    occ_c, occ_a = [], []
    for fc in faces:
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(fc, props)
        c = props.CentreOfMass()
        occ_c.append((c.X(), c.Y(), c.Z()))
        occ_a.append(props.Mass())
    occ_c, occ_a = np.array(occ_c), np.array(occ_a)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        surfs = [t for _, t in gmsh.model.getEntities(2)]
        if not surfs:
            return None, "gmsh_no_surfaces"
        g_c = np.array([gmsh.model.occ.getCenterOfMass(2, s) for s in surfs])
        g_a = np.array([gmsh.model.occ.getMass(2, s) for s in surfs])
        # one-to-one greedy match on centroid distance (relative to the diagonal) plus
        # relative area difference; ties go to the lower explorer index, which is the
        # first visit of a face TopExp_Explorer reports twice
        cd = np.linalg.norm(g_c[:, None, :] - occ_c[None, :, :], axis=2) / diag
        ad = np.abs(g_a[:, None] - occ_a[None, :]) / np.maximum(occ_a[None, :], 1e-30)
        score = cd + ad
        face_of = np.full(len(surfs), -1, np.int64)
        used = np.zeros(len(faces), bool)
        for flat in np.lexsort((np.tile(np.arange(len(faces)), len(surfs)), score.ravel())):
            si, fi = divmod(int(flat), len(faces))
            if face_of[si] >= 0 or used[fi]:
                continue
            if cd[si, fi] <= 1e-5 and ad[si, fi] <= 1e-3:
                face_of[si], used[fi] = fi, True
        mapped = face_of >= 0

        # outward orientation per gmsh surface, decided once from the coarsest mesh
        flip = {}
        arrays = {}
        for name, curv, mx in GMSH_LADDER:
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", GMSH_EXTEND_FROM_BOUNDARY)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mx * diag)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 1e-4 * diag)
            gmsh.model.mesh.generate(2)
            ntags, coords, _ = gmsh.model.mesh.getNodes()
            lookup = np.zeros(int(ntags.max()) + 1, np.int64)
            lookup[ntags.astype(np.int64)] = np.arange(ntags.size)
            xyz = coords.reshape(-1, 3)

            pos_chunks, fid_chunks = [], []
            for s, fi, ok in zip(surfs, face_of, mapped):
                if not ok:
                    continue
                types, _, nodes = gmsh.model.mesh.getElements(2, s)
                tri = [n.reshape(-1, 3) for et, n in zip(types, nodes) if et == 2]
                if not tri or tri[0].size == 0:
                    continue
                V = xyz[lookup[tri[0].astype(np.int64)]]          # (nt, 3, 3)
                if s not in flip:
                    fc = faces[fi]
                    srf = BRep_Tool.Surface_s(fc)
                    votes = 0
                    for k in np.linspace(0, len(V) - 1, min(5, len(V))).astype(int):
                        a, b, c = V[k]
                        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*((a + b + c) / 3)), srf)
                        if proj.NbPoints() == 0:
                            continue
                        u, v = proj.LowerDistanceParameters()
                        p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
                        srf.D1(u, v, p, du, dv)
                        ns = np.cross([du.X(), du.Y(), du.Z()], [dv.X(), dv.Y(), dv.Z()])
                        if fc.Orientation() == TopAbs_REVERSED:
                            ns = -ns
                        votes += 1 if float(np.cross(b - a, c - a) @ ns) > 0 else -1
                    flip[s] = votes < 0
                if flip[s]:
                    V = V[:, [0, 2, 1]]
                area2 = np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1)
                V = V[area2 > 2e-12 * diag * diag]
                if len(V):
                    pos_chunks.append(V.reshape(-1, 9).astype(np.float32))
                    fid_chunks.append(np.full(len(V), fi, np.int32))
            if pos_chunks:
                P = np.concatenate(pos_chunks)
                if MIN_TRIS <= len(P) <= MAX_TRIS:
                    arrays[f"{name}_positions"] = P
                    arrays[f"{name}_face_id"] = np.concatenate(fid_chunks)
        status = "ok" if mapped.all() else f"ok_unmapped_{int((~mapped).sum())}"
        return arrays, status
    finally:
        gmsh.finalize()


def bake_variant(task):
    variant, model_id, part, step_path, out_path = task
    rec = {"model_id": model_id, "part": part, "status": "ok", "tris": 0}
    try:
        from occ_tessellate import TYPE_INDEX, bbox_diagonal, face_type_name, read_step
        if Path(out_path).exists():
            rec["status"] = "exists"
            return rec
        shape = read_step(step_path)
        if shape is None:
            rec["status"] = "unreadable"
            return rec
        diag = bbox_diagonal(shape)
        faces = _explorer_faces(shape)
        stored = np.array([TYPE_INDEX[face_type_name(f)] for f in faces], np.uint8)
        if variant == "nurbs":
            arrays, status = _bake_nurbs(shape, faces, diag)
        elif variant == "gmsh":
            arrays, status = _bake_gmsh(step_path, faces, diag)
        else:
            raise ValueError(variant)
        if arrays is None or not arrays:
            rec["status"] = status if arrays is None else "no_usable_density"
            return rec
        for key in [k for k in arrays if k.endswith("_face_id")]:
            arrays[key.replace("_face_id", "_face_type")] = stored[arrays[key]]
            rec["tris"] += int(arrays[key].size)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, model_id=model_id, part=part, diag=np.float64(diag),
                 mesher=variant, **arrays)
        rec["status"] = status
    except Exception as exc:
        rec["status"] = f"error:{type(exc).__name__}"
    return rec


def main(argv=None):
    from isolated_pool import run_isolated
    from relabel_sidecar import val_split
    from stage1_bake import find_step

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=("nurbs", "gmsh"))
    ap.add_argument("--split", default="val", choices=("val", "train", "all"))
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"),
                    help="main bake whose files and split define the parts")
    ap.add_argument("--step-root", type=Path, default=Path("D:/abc-data/step"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    val, train = val_split(args.bake)
    files = {"val": val, "train": train, "all": val + train}[args.split]
    tasks = []
    for f in files:
        rel = f.relative_to(args.bake)
        d = np.load(f)
        mid, part = str(d["model_id"]), str(d["part"])
        sp = find_step(args.step_root, mid)
        if sp is not None:
            tasks.append((args.variant, mid, part, str(sp), str(args.out / rel)))
        if args.limit and len(tasks) >= args.limit:
            break
    print(f"{args.variant}: {len(tasks):,} parts from the {args.split} split -> {args.out}", flush=True)

    t0 = time.time()
    status, tris = Counter(), 0
    fail = lambda task, st: {"model_id": task[1], "part": task[2], "status": st, "tris": 0}
    for i, rec in enumerate(run_isolated(bake_variant, tasks, args.jobs, args.timeout, fail), 1):
        status[rec["status"]] += 1
        tris += rec["tris"]
        if i % 250 == 0:
            print(f"  {i:,}/{len(tasks):,}  {i / (time.time() - t0):.1f}/s", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    print("status: " + ", ".join(f"{k} {v:,}" for k, v in status.most_common()))
    print(f"triangles across all densities: {tris:,}")


if __name__ == "__main__":
    raise SystemExit(main())
