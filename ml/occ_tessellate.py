#!/usr/bin/env python3
"""
occ_tessellate.py - turn a STEP file into labelled triangles via OCCT.

The core of Stage 1. Reads a STEP, meshes it at a chosen deflection, and emits raw
triangles with, for each one, the B-rep face it came from and that face's surface type.

WHY LABELS COME FROM OCCT AND NOT FROM ABC'S FEATURE FILE

OCCT's `GeomAbs_SurfaceType` maps one-to-one onto ABC's taxonomy - Plane, Cylinder,
Cone, Sphere, Torus, SurfaceOfRevolution, SurfaceOfExtrusion, BSpline/Bezier, and the
offset/other catch-alls. Since we have to open the STEP anyway to re-tessellate it, the
type comes free per face, and BRepMesh already keys its triangulation by face. So there
is no matching step at all: no nearest-surface test, no index join, no tolerance to
tune. The feature file becomes an independent CROSS-CHECK rather than the label source
(see --verify), which is strictly better than trusting either one alone.

Verified on ABC's published sample model 00000050: OCCT reports Plane 20 / Cylinder 23,
exactly matching the feature file, which is itself 43 = the count of PLANE and
CYLINDRICAL_SURFACE records in the raw STEP text.

WHY RAW TRIANGLES AND NOT WELDED VERTICES

The converter reads STL, and an STL has no shared vertices - `js/meshCore.js` takes a
flat array of 9 floats per triangle and welds by quantised position in
`buildAdjacency(positions, facets, quant = 1e4)`. OCCT also triangulates each face
independently, so its output has the same shape. Emitting raw triangles means training
data goes through the exact same welding the tool performs at inference; pre-welding
here with a different tolerance would be a silent train/test mismatch at precisely the
place segmentation is decided - face boundaries.

DENSITY IS A CONTROLLED VARIABLE

`BRepTools.Clean_s` before each mesh is load-bearing. BRepMesh caches triangulation on
the shape, so without it a second call at a finer deflection returns very nearly the
first result (measured: 1,608 vs 1,636 triangles across a 3x deflection change) and the
density ladder silently collapses. With the clean, the same model gives 1,502 / 2,554 /
5,024 / 9,388 triangles across the ladder below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

# Label vocabulary. Index is what gets baked; keep this ordering stable forever, since
# trained weights encode it.
TYPE_NAMES = ("Plane", "Cylinder", "Cone", "Sphere", "Torus",
              "Revolution", "Extrusion", "BSpline", "Other")
TYPE_INDEX = {n: i for i, n in enumerate(TYPE_NAMES)}

_GA = GeomAbs_SurfaceType
_OCC_TO_NAME = {
    _GA.GeomAbs_Plane: "Plane",
    _GA.GeomAbs_Cylinder: "Cylinder",
    _GA.GeomAbs_Cone: "Cone",
    _GA.GeomAbs_Sphere: "Sphere",
    _GA.GeomAbs_Torus: "Torus",
    _GA.GeomAbs_SurfaceOfRevolution: "Revolution",
    _GA.GeomAbs_SurfaceOfExtrusion: "Extrusion",
    # ABC has no Bezier category; it files them under BSpline, so we do too.
    _GA.GeomAbs_BSplineSurface: "BSpline",
    _GA.GeomAbs_BezierSurface: "BSpline",
    _GA.GeomAbs_OffsetSurface: "Other",
    _GA.GeomAbs_OtherSurface: "Other",
}

# (linear deflection as a fraction of bbox diagonal, angular deflection in radians).
# Chosen to bracket what real STLs look like: the coarse end is slicer-export chunky,
# the fine end is what a CAD package emits when asked for a good mesh.
DENSITY_LADDER = (
    ("coarse", 0.02, 0.5),
    ("medium", 0.005, 0.3),
    ("fine", 0.001, 0.15),
)


@dataclass
class Tessellation:
    """One meshing of one model."""
    positions: np.ndarray   # (ntri, 9) float32 - STL-style, vertices NOT shared
    face_id: np.ndarray     # (ntri,)  int32   - index of the owning B-rep face
    face_type: np.ndarray   # (ntri,)  uint8   - TYPE_INDEX of that face's surface
    n_faces: int
    diag: float
    degenerate: int         # triangles dropped for near-zero area

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def read_step(path: str):
    """Returns a TopoDS_Shape, or None if the file will not load."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        return None
    try:
        reader.TransferRoots()
        shape = reader.OneShape()
    except Exception:
        return None
    return None if shape is None or shape.IsNull() else shape


def bbox_diagonal(shape) -> float:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        return 0.0
    xm, ym, zm, xM, yM, zM = box.Get()
    return math.dist((xm, ym, zm), (xM, yM, zM))


def face_type_name(face) -> str:
    try:
        return _OCC_TO_NAME.get(BRepAdaptor_Surface(face).GetType(), "Other")
    except Exception:
        return "Other"


def tessellate(shape, lin_frac: float, ang: float, diag: float | None = None,
               min_area_frac: float = 1e-12) -> Tessellation:
    """Mesh `shape` and return labelled raw triangles.

    lin_frac is relative to the bounding-box diagonal so that models of wildly
    different scale - ABC spans millimetre parts to metre assemblies - get comparable
    triangle counts rather than comparable absolute deflection.
    """
    if diag is None:
        diag = bbox_diagonal(shape)
    if diag <= 0:
        return Tessellation(np.zeros((0, 9), np.float32), np.zeros(0, np.int32),
                            np.zeros(0, np.uint8), 0, 0.0, 0)

    BRepTools.Clean_s(shape)   # see module header - without this the ladder collapses
    BRepMesh_IncrementalMesh(shape, diag * lin_frac, False, ang, True)

    min_area = min_area_frac * diag * diag
    pos_chunks: list[np.ndarray] = []
    fid_chunks: list[np.ndarray] = []
    fty_chunks: list[np.ndarray] = []
    degenerate = 0
    n_faces = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        fidx = n_faces
        n_faces += 1
        tname = face_type_name(face)
        reversed_ = face.Orientation() == TopAbs_REVERSED

        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        exp.Next()
        if tri is None or tri.NbTriangles() == 0:
            continue

        trsf = loc.Transformation()
        nn = tri.NbNodes()
        nodes = np.empty((nn, 3), np.float64)
        for i in range(1, nn + 1):
            p = tri.Node(i).Transformed(trsf)
            nodes[i - 1] = (p.X(), p.Y(), p.Z())

        nt = tri.NbTriangles()
        idx = np.empty((nt, 3), np.int32)
        for i in range(1, nt + 1):
            a, b, c = tri.Triangle(i).Get()
            # A reversed face carries its triangulation in the underlying surface's
            # winding, so flip to keep every emitted normal pointing outward. The
            # converter fits on geometric normals, so a flipped facet is not cosmetic.
            idx[i - 1] = (a - 1, c - 1, b - 1) if reversed_ else (a - 1, b - 1, c - 1)

        verts = nodes[idx]                                   # (nt, 3, 3)
        u = verts[:, 1] - verts[:, 0]
        v = verts[:, 2] - verts[:, 0]
        area2 = np.linalg.norm(np.cross(u, v), axis=1)
        keep = area2 > (2.0 * min_area)
        degenerate += int((~keep).sum())
        if not keep.any():
            continue
        verts = verts[keep]

        pos_chunks.append(verts.reshape(-1, 9).astype(np.float32))
        fid_chunks.append(np.full(verts.shape[0], fidx, np.int32))
        fty_chunks.append(np.full(verts.shape[0], TYPE_INDEX[tname], np.uint8))

    if not pos_chunks:
        return Tessellation(np.zeros((0, 9), np.float32), np.zeros(0, np.int32),
                            np.zeros(0, np.uint8), n_faces, diag, degenerate)

    return Tessellation(
        np.concatenate(pos_chunks), np.concatenate(fid_chunks),
        np.concatenate(fty_chunks), n_faces, diag, degenerate,
    )


def face_type_histogram(shape) -> dict[str, int]:
    """Per-face surface types without meshing - used by --verify."""
    hist: dict[str, int] = {}
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        name = face_type_name(TopoDS.Face_s(exp.Current()))
        hist[name] = hist.get(name, 0) + 1
        exp.Next()
    return hist
