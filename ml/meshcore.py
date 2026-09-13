#!/usr/bin/env python3
"""
meshcore.py - vectorised port of js/meshCore.js `buildFacets` / `buildAdjacency`.

Training data must go through the SAME facet construction and welding the converter
performs at inference, or the model learns a mesh topology the tool will never hand it.
Face boundaries are exactly where segmentation is decided, so a different weld tolerance
here would be a silent train/test mismatch in the worst possible place.

This is therefore a deliberate duplicate of JS logic, not a reimplementation with
improvements. Two details are load-bearing and must not be "fixed":

  - Welding quantises positions with round(x * 1e4) and compares the integer triple.
    That is an ABSOLUTE quantum, not relative to model size, so it behaves differently
    on a 10 mm part and a 10 m assembly. The tool does this; so do we.
  - Only edges shared by EXACTLY TWO triangles produce adjacency. Boundary edges and
    non-manifold edges (3+ triangles) contribute nothing, so facets across them are not
    neighbours at all.
  - Zero-area facets are DROPPED before the edge map is built, not merely skipped
    later. A sliver with vertices (A, B, B) contributes the edge A-B twice, so an
    ordinary 2-manifold edge reads as 4-manifold and the rule above then rejects it.

`test_meshcore.py` checks this port against the JS by running both over the same mesh
and comparing facet arrays and the full adjacency structure.

JS rounding note: JavaScript's Math.round is round-half-UP (towards +Infinity), while
numpy's rint is round-half-to-EVEN. They differ only for exact .5 cases, which is rare
for float coordinates but not impossible, so `_jsround` reproduces the JS behaviour
rather than using rint.
"""

from __future__ import annotations

import numpy as np

QUANT = 1e4


def _jsround(a: np.ndarray) -> np.ndarray:
    """JavaScript Math.round: floor(x + 0.5), i.e. half away from zero towards +inf."""
    return np.floor(a + 0.5)


def build_facets(positions: np.ndarray) -> dict:
    """positions: (ntri, 9) or flat, STL-style. Returns centroids, normals, areas.

    Normals are GEOMETRIC (recomputed from winding), matching the JS. The module header
    there explains why: vertex normals from an STL header are frequently zero or
    unnormalised, and raw vertices weight dense regions more heavily.
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3, 3)
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1)
    areas = 0.5 * length
    normals = np.zeros_like(n)
    ok = length > 1e-20
    normals[ok] = n[ok] / length[ok, None]
    return {
        "count": int(p.shape[0]),
        "centroids": p.mean(axis=1),
        "normals": normals,
        "areas": areas,
        # Zero-area facets have no meaningful normal and are not surface. Real STLs
        # carry them - 3DBenchy ships 563 of 225,706 - and leaving them in inflates
        # edge multiplicity, because a sliver with vertices (A, B, B) contributes the
        # edge A-B twice and an ordinary 2-manifold edge then reads as 4-manifold.
        "degenerate": ~ok,
    }


def build_adjacency(positions: np.ndarray, facets: dict, quant: float = QUANT):
    """Returns (pairs, angles_deg).

    pairs:  (nedge, 2) int32 - facet index pairs sharing a welded edge
    angles: (nedge,)   float64 - dihedral angle in degrees, as the JS computes it
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3, 3)
    ntri = p.shape[0]
    if ntri == 0:
        return np.zeros((0, 2), np.int32), np.zeros(0, np.float64)

    # Degenerate facets are excluded, not merely ignored - see build_facets.
    live = ~facets["degenerate"] if "degenerate" in facets else np.ones(ntri, bool)

    # Weld: quantise every vertex, then map identical integer triples to one id.
    q = _jsround(p.reshape(-1, 3) * quant).astype(np.int64)
    _, vid = np.unique(q, axis=0, return_inverse=True)
    vid = vid.reshape(ntri, 3)

    # Three edges per triangle, each as an unordered pair of welded vertex ids.
    e0 = vid[:, [0, 1]]
    e1 = vid[:, [1, 2]]
    e2 = vid[:, [2, 0]]
    edges = np.concatenate([e0, e1, e2], axis=0)
    edges = np.sort(edges, axis=1)
    owner = np.tile(np.arange(ntri, dtype=np.int64), 3)
    keep_live = np.tile(live, 3)
    edges = edges[keep_live]
    owner = owner[keep_live]
    if edges.shape[0] == 0:
        return np.zeros((0, 2), np.int32), np.zeros(0, np.float64)

    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owner = owner[order]

    # Group identical edges; keep only those shared by exactly two triangles.
    new_group = np.empty(edges.shape[0], bool)
    new_group[0] = True
    np.not_equal(edges[1:], edges[:-1]).any(axis=1, out=new_group[1:])
    gid = np.cumsum(new_group) - 1
    counts = np.bincount(gid)
    keep = counts == 2
    if not keep.any():
        return np.zeros((0, 2), np.int32), np.zeros(0, np.float64)

    starts = np.flatnonzero(new_group)[keep]
    # Sort within each pair: the JS only ever records t < neighbor, and the two owners
    # arrive here in edge-sort order, which is not the same thing.
    pairs = np.sort(np.stack([owner[starts], owner[starts + 1]], axis=1), axis=1).astype(np.int32)

    n = facets["normals"]
    d = np.clip(np.einsum("ij,ij->i", n[pairs[:, 0]], n[pairs[:, 1]]), -1.0, 1.0)
    return pairs, np.degrees(np.arccos(d))


def adjacency_lists(pairs: np.ndarray, ntri: int) -> list[list[int]]:
    """Expand edge pairs into a per-facet neighbour list (both directions)."""
    out: list[list[int]] = [[] for _ in range(ntri)]
    for a, b in pairs:
        out[a].append(int(b))
        out[b].append(int(a))
    return out


def adjacency_csr(pairs: np.ndarray, ntri: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR form of the dual graph: (indptr, indices). Used for fast k-ring BFS."""
    if pairs.shape[0] == 0:
        return np.zeros(ntri + 1, np.int64), np.zeros(0, np.int32)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.zeros(ntri + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=ntri), out=indptr[1:])
    return indptr, dst.astype(np.int32)


def k_ring(seed: int, indptr: np.ndarray, indices: np.ndarray,
           max_facets: int, max_rings: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """BFS outward from `seed` over the dual graph.

    Returns (facet_ids, ring_index), truncated at `max_facets`. This is the same
    adjacency BFS `neighbourhood()` uses in regionGrow.js - which is exactly why
    paintFrom fails on faces small relative to their neighbours: the patch crosses the
    boundary. The model's job is to label which of these belong to the seed's surface,
    so the BFS only has to supply a superset.
    """
    ntri = indptr.shape[0] - 1
    if ntri == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int16)
    seen = np.zeros(ntri, bool)
    seen[seed] = True
    out = [np.array([seed], np.int32)]
    rings = [np.zeros(1, np.int16)]
    total = 1
    frontier = out[0]
    for r in range(1, max_rings + 1):
        if total >= max_facets or frontier.size == 0:
            break
        nbrs = np.concatenate([indices[indptr[f]:indptr[f + 1]] for f in frontier]) \
            if frontier.size else np.zeros(0, np.int32)
        if nbrs.size == 0:
            break
        nbrs = np.unique(nbrs)
        nbrs = nbrs[~seen[nbrs]]
        if nbrs.size == 0:
            break
        if total + nbrs.size > max_facets:
            nbrs = nbrs[: max_facets - total]
        seen[nbrs] = True
        out.append(nbrs)
        rings.append(np.full(nbrs.size, min(r, 32767), np.int16))
        total += nbrs.size
        frontier = nbrs
    return np.concatenate(out), np.concatenate(rings)
