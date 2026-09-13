#!/usr/bin/env python3
"""
test_meshcore.py - check the Python port against the real js/meshCore.js.

Run:  python test_meshcore.py            (needs node on PATH, and numpy)

This is not a unit test of the Python in isolation. It runs BOTH implementations over
the same meshes and compares results, because the only property that matters is that
training data is built exactly the way the converter builds it at inference. A port
that is merely "correct" but disagrees with the JS is a silent train/test mismatch at
face boundaries - precisely where segmentation is decided.

Meshes are chosen to exercise the cases where a reimplementation would plausibly drift:
open boundaries (edges with one triangle), non-manifold edges (three triangles), exact
.5 quantisation ties (JS Math.round is half-up, numpy rint is half-to-even), and
vertices that are close but must NOT weld.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshcore import build_adjacency, build_facets  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

JS_DRIVER = r"""
import { buildFacets, buildAdjacency } from '%s';
import { readFileSync } from 'node:fs';
const positions = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const f = buildFacets(positions);
const adj = buildAdjacency(positions, f);
const pairs = [];
for (const [t, list] of adj) for (const e of list) if (t < e.neighbor) pairs.push([t, e.neighbor, e.angle]);
pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
process.stdout.write(JSON.stringify({
  count: f.count,
  centroids: Array.from(f.centroids),
  normals: Array.from(f.normals),
  areas: Array.from(f.areas),
  pairs,
}));
"""


def run_js(positions: np.ndarray) -> dict:
    mesh_js = (REPO / "js" / "meshCore.js").as_uri()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "pos.json").write_text(json.dumps([float(x) for x in positions.ravel()]))
        (td / "drv.mjs").write_text(JS_DRIVER % mesh_js)
        out = subprocess.run(
            ["node", str(td / "drv.mjs"), str(td / "pos.json")],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            raise RuntimeError(f"node failed:\n{out.stderr}")
        return json.loads(out.stdout)


def tri(*pts) -> list[float]:
    return [c for p in pts for c in p]


# --- meshes ---------------------------------------------------------------------

def mesh_tetra() -> np.ndarray:
    """Closed, every edge shared by exactly two triangles."""
    a, b, c, d = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
    return np.array([tri(a, c, b), tri(a, b, d), tri(a, d, c), tri(b, c, d)], np.float64)


def mesh_open_strip() -> np.ndarray:
    """Two triangles sharing one edge: four boundary edges contribute no adjacency."""
    return np.array([
        tri((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        tri((1, 0, 0), (1, 1, 0), (0, 1, 0)),
    ], np.float64)


def mesh_nonmanifold() -> np.ndarray:
    """Three triangles on one edge - the JS skips edges without exactly two owners."""
    e0, e1 = (0, 0, 0), (1, 0, 0)
    return np.array([
        tri(e0, e1, (0, 1, 0)),
        tri(e0, e1, (0, 0, 1)),
        tri(e0, e1, (0, -1, 1)),
    ], np.float64)


def mesh_half_ties() -> np.ndarray:
    """Coordinates landing exactly on .5 after scaling by quant=1e4.

    0.00005 * 1e4 == 0.5 exactly. JS Math.round gives 1, numpy rint gives 0. If the
    port used rint these two triangles would weld differently and the adjacency would
    disagree.
    """
    z = 0.00005
    return np.array([
        tri((0, 0, 0), (1, 0, 0), (0, 1, z)),
        tri((1, 0, 0), (1, 1, z), (0, 1, z)),
    ], np.float64)


def mesh_near_but_not_welded() -> np.ndarray:
    """Vertices 2e-4 apart: above the 1e-4 quantum, so they must NOT weld."""
    return np.array([
        tri((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        tri((1.0002, 0, 0), (2, 0, 0), (1.0002, 1, 0)),
    ], np.float64)


def mesh_random_closed(seed: int = 3) -> np.ndarray:
    """Icosphere-ish: subdivided octahedron, closed and manifold."""
    rng = np.random.default_rng(seed)
    verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    v = [np.array(p, np.float64) for p in verts]
    for _ in range(2):
        newf, mid = [], {}
        for a, b, c in faces:
            def m(i, j):
                k = (min(i, j), max(i, j))
                if k not in mid:
                    p = (v[i] + v[j]) / 2
                    v.append(p / np.linalg.norm(p))
                    mid[k] = len(v) - 1
                return mid[k]
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            newf += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = newf
    pts = np.array(v) * (1.0 + 0.02 * rng.standard_normal((len(v), 1)))
    return np.array([tri(pts[a], pts[b], pts[c]) for a, b, c in faces], np.float64)


MESHES = {
    "tetra": mesh_tetra(),
    "open_strip": mesh_open_strip(),
    "nonmanifold": mesh_nonmanifold(),
    "half_ties": mesh_half_ties(),
    "near_not_welded": mesh_near_but_not_welded(),
    "random_closed": mesh_random_closed(),
}


def main() -> int:
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except Exception:
        print("node not found on PATH - cannot cross-check against meshCore.js",
              file=sys.stderr)
        return 2

    failures = 0
    for name, positions in MESHES.items():
        js = run_js(positions)
        f = build_facets(positions)
        pairs, angles = build_adjacency(positions, f)

        order = np.lexsort((pairs[:, 1], pairs[:, 0])) if pairs.shape[0] else np.zeros(0, int)
        py_pairs = pairs[order]
        py_ang = angles[order]
        js_pairs = np.array([[p[0], p[1]] for p in js["pairs"]], np.int32).reshape(-1, 2)
        js_ang = np.array([p[2] for p in js["pairs"]], np.float64)

        problems = []
        if f["count"] != js["count"]:
            problems.append(f'count {f["count"]} != {js["count"]}')
        for key, got in (("centroids", f["centroids"]), ("normals", f["normals"]),
                         ("areas", f["areas"])):
            want = np.array(js[key], np.float64).reshape(got.shape)
            if not np.allclose(got, want, rtol=0, atol=1e-12):
                problems.append(f"{key} max|d|={np.abs(got - want).max():.3e}")
        if py_pairs.shape != js_pairs.shape or not np.array_equal(py_pairs, js_pairs):
            problems.append(f"adjacency pairs differ: py={py_pairs.shape[0]} js={js_pairs.shape[0]}")
        elif py_ang.size and not np.allclose(py_ang, js_ang, rtol=0, atol=1e-9):
            problems.append(f"angles max|d|={np.abs(py_ang - js_ang).max():.3e}")

        status = "ok" if not problems else "MISMATCH"
        print(f"  {name:<18} tris={f['count']:>5}  edges={py_pairs.shape[0]:>5}  {status}")
        for p in problems:
            print(f"      {p}")
            failures += 1

    print("\nPASS" if not failures else f"\n{failures} mismatch(es)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
