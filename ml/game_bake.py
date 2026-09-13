#!/usr/bin/env python3
"""
game_bake.py - degrade baked ABC parts the way game-art meshes are degraded, keeping labels.

WHY THIS EXISTS

Game assets, cosplay props and other hand-built meshes are a target
(docs/game-assets-scope.md), and they have no ground truth. ABC does. Every operation
here changes something a game mesh changes - triangle pattern, density, welds, shell
structure - while each triangle's B-rep face id stays true. The output can therefore be
trained on and scored with the existing tools, unchanged:

    python train.py ... --extra-bake D:/abc-data/bake_game
    python rescore.py --bake D:/abc-data/bake_game --intersect D:/abc-data/bake ...

It is the far end of the axis mesher_bake.py opened. OCCT, NURBS-then-OCCT and gmsh are
all algorithms; a hand-modelled mesh is not, and the model already fails on the second
algorithm it meets (Cylinder 0.877 -> 0.043 on gmsh).

OPERATIONS, applied in this order, each label-exact

  lowpoly   re-tessellate the STEP at game-like deflection (GAME_LADDER). Cylinders get a
            handful of sides. Vertices stay ON the true surface, so the face is still a
            cylinder - the model just sees what looks like a prism, which is exactly the
            ambiguity a low-poly asset poses. Without this op, the other ops are applied
            to the source bake's own densities.
  quadflip  pair triangles across same-face edges and swap the shared diagonal at random.
            Artists model in quads and the exporter picks the diagonal; this is that
            pattern, and no vertex moves.
  jitter    move each vertex tangentially by a fraction of its shortest edge, as hand
            placement does. The only op that moves geometry off the surface. Not in the
            default op list.
  unweld    open random seams between adjacent faces by pulling one side's boundary
            vertices a small gap back INTO that face, within its tangent plane. Rips carry
            these along UV and mirror seams; masterchief.stl has 756 open edges.
            The first version pushed them along the face normal instead. That is not a
            seam, it is a warp: OCCT meshes planes as long slivers, so lifting one corner
            by the gap tilts the whole triangle, and planar faces stopped being planar.
  shells    translate random whole faces rigidly along their mean normal, so they become
            separate shells floating over or sunk into the part, like armour plates over a
            body; masterchief.stl is 49 shells. A rigid move keeps a surface's type.

GAPS ARE ABSOLUTE-AWARE. meshCore welds on round(x * 1e4), an ABSOLUTE quantum, so a gap
set relative to the diagonal alone would round back onto its twin on small parts. Every
gap is max(rel * diag, MIN_GAP), and MIN_GAP is three quanta.

OUTPUT matches mesher_bake.py: <out>/<model_id[:5]>/<model_id>_<part>.npz with
{density}_positions / _face_id / _face_type, model_id, part, diag, and
mesher="game:<ops>". face_type is the STORED type and face_id indexes the original
TopExp_Explorer faces, so relabel sidecars apply unchanged. Parts come from the main
bake's own split, so a game validation set holds exactly the main validation parts.

Usage:
    python game_bake.py --split val --limit 20 --out D:/abc-data/bake_game_smoke --jobs 2
    python game_bake.py --ops quadflip,unweld,shells --split val --out D:/abc-data/bake_game_nolow
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MIN_TRIS = 32
MAX_TRIS = 400_000
WELD_QUANT = 1e4       # meshCore.buildAdjacency's default; see GAPS ARE ABSOLUTE-AWARE
MIN_GAP = 3e-4         # three weld quanta

OPS = ("lowpoly", "quadflip", "jitter", "unweld", "shells")
DEFAULT_OPS = "lowpoly,quadflip,unweld,shells"

# (name, linear deflection / diag, angular deflection in radians). Named like
# occ_tessellate.DENSITY_LADDER so rescore.py --density and --intersect work unchanged.
# Where angular deflection governs, a full circle gets about ceil(2*pi / ang) segments:
# 7 at coarse, 8 at medium, 11 at fine.
GAME_LADDER = (
    ("coarse", 0.10, 1.0),
    ("medium", 0.05, 0.8),
    ("fine", 0.02, 0.6),
)


# ── Mesh helpers ─────────────────────────────────────────────────────────────

def weld(P: np.ndarray):
    """Corner -> welded vertex id on meshCore's quantum, plus one representative corner."""
    keys = np.round(P.reshape(-1, 3) * WELD_QUANT).astype(np.int64)
    _, first, vid = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return vid.reshape(-1, 3), first


def tri_normals(P: np.ndarray):
    A, B, C = P[:, 0:3], P[:, 3:6], P[:, 6:9]
    n = np.cross(B - A, C - A)
    ln = np.linalg.norm(n, axis=1)
    return n / np.maximum(ln, 1e-30)[:, None], 0.5 * ln


def shared_edges(T: np.ndarray):
    """Manifold edges as (t1, t2, local edge in t1, local edge in t2)."""
    n = len(T)
    tri = np.repeat(np.arange(n), 3)
    loc = np.tile(np.arange(3), n)
    a, b = T[tri, loc], T[tri, (loc + 1) % 3]
    base = np.int64(T.max()) + 1
    key = np.minimum(a, b) * base + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    _, start, count = np.unique(key[order], return_index=True, return_counts=True)
    two = start[count == 2]
    e1, e2 = order[two], order[two + 1]
    return tri[e1], tri[e2], loc[e1], loc[e2], key, base


# ── Operations ───────────────────────────────────────────────────────────────

def quadflip(P, F, rng, prob, diag):
    """Swap the diagonal of random same-face triangle pairs. Returns (P, flips)."""
    T, _ = weld(P)
    t1, t2, l1, l2, key, base = shared_edges(T)
    same = F[t1] == F[t2]
    t1, t2, l1, l2 = t1[same], t2[same], l1[same], l2[same]
    existing = set(key.tolist())
    N0, _ = tri_normals(P)
    min_area2 = 4e-12 * diag * diag
    used = np.zeros(len(T), bool)
    out = P.copy()
    flips = 0
    for i in rng.permutation(len(t1)):
        u, v, la, lb = t1[i], t2[i], l1[i], l2[i]
        if used[u] or used[v] or rng.random() >= prob:
            continue
        # u = (a, b, c) along its shared edge a->b; a consistently wound v runs b->a
        if T[v, lb] != T[u, (la + 1) % 3] or T[v, (lb + 1) % 3] != T[u, la]:
            continue
        c, d = T[u, (la + 2) % 3], T[v, (lb + 2) % 3]
        if c == d or min(c, d) * base + max(c, d) in existing:
            continue                                   # would duplicate an edge
        U, W = P[u].reshape(3, 3), P[v].reshape(3, 3)
        pa, pb, pc, pd = U[la], U[(la + 1) % 3], U[(la + 2) % 3], W[(lb + 2) % 3]
        n1 = np.cross(pd - pa, pc - pa)                # (a, d, c)
        n2 = np.cross(pb - pd, pc - pd)                # (d, b, c)
        if n1 @ n1 < min_area2 or n2 @ n2 < min_area2:
            continue
        # Both new triangles must face the same way as BOTH old ones. Checking against the
        # pair's mean normal alone let 9 of 5,024 flips fold on tightly curved faces.
        if min(n1 @ N0[u], n1 @ N0[v], n2 @ N0[u], n2 @ N0[v]) <= 0:
            continue
        out[u] = np.concatenate((pa, pd, pc))
        out[v] = np.concatenate((pd, pb, pc))
        used[u] = used[v] = True
        existing.add(min(c, d) * base + max(c, d))
        flips += 1
    return out, flips


def jitter(P, rng, frac):
    """Tangential vertex noise, `frac` of each vertex's shortest incident edge."""
    T, first = weld(P)
    nv = len(first)
    N, A = tri_normals(P)
    V = P.reshape(-1, 3)[first]                         # one position per welded vertex
    vn = np.zeros((nv, 3))
    np.add.at(vn, T.ravel(), np.repeat(N * A[:, None], 3, axis=0))
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)
    C = P.reshape(-1, 3, 3)
    el = np.linalg.norm(C - np.roll(C, -1, axis=1), axis=2)   # corner k -> k+1
    short = np.full(nv, np.inf)
    np.minimum.at(short, T.ravel(), np.minimum(el, np.roll(el, 1, axis=1)).ravel())
    d = rng.normal(size=(nv, 3))
    d -= (d * vn).sum(1, keepdims=True) * vn
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-30)
    d *= (frac * np.where(np.isfinite(short), short, 0.0))[:, None]
    return (V + d)[T.ravel()].reshape(-1, 9)


def unweld(P, F, rng, frac, gap):
    """Open random seams between adjacent faces. Returns (P, seams opened)."""
    T, _ = weld(P)
    t1, t2, l1, _, _, base = shared_edges(T)
    cross = F[t1] != F[t2]
    t1, t2, l1 = t1[cross], t2[cross], l1[cross]
    if not len(t1):
        return P, 0
    fbase = np.int64(F.max()) + 1
    pair = np.minimum(F[t1], F[t2]) * fbase + np.maximum(F[t1], F[t2])
    pairs = np.unique(pair)
    chosen = pairs[rng.random(len(pairs)) < frac]
    if not len(chosen):
        return P, 0
    side = np.where(rng.random(len(chosen)) < 0.5, chosen // fbase, chosen % fbase)
    side_of = dict(zip(chosen.tolist(), side.tolist()))

    # (moving face, vertex) keys: the moving side's copies of each seam vertex
    moved = set()
    for i in np.nonzero(np.isin(pair, chosen))[0]:
        s = side_of[int(pair[i])]
        moved.add(s * base + int(T[t1[i], l1[i]]))
        moved.add(s * base + int(T[t1[i], (l1[i] + 1) % 3]))
    ck = (F[:, None] * base + T).ravel()
    corners = np.nonzero(np.isin(ck, np.fromiter(moved, np.int64)))[0]

    # One displacement per (face, vertex), so the moving face itself stays closed. The
    # direction is toward the face's own incident triangles, projected into its tangent
    # plane: a planar face stays exactly planar, and a curved one moves O(gap^2 / R).
    N, A = tri_normals(P)
    V = P.reshape(-1, 3)
    tri = corners // 3
    cen = P.reshape(-1, 3, 3).mean(1)
    uk, inv = np.unique(ck[corners], return_inverse=True)
    nrm = np.zeros((len(uk), 3))
    inward = np.zeros((len(uk), 3))
    np.add.at(nrm, inv, N[tri] * A[tri, None])
    np.add.at(inward, inv, (cen[tri] - V[corners]) * A[tri, None])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    inward -= (inward * nrm).sum(1, keepdims=True) * nrm
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-30)

    # A retraction wider than a sliver's altitude would invert it. Revert every
    # (face, vertex) move that touches a flipped or collapsed triangle, until none do.
    move = np.ones(len(uk), bool)
    for _ in range(4):
        Q = V.copy()
        Q[corners] += gap * (inward * move[:, None])[inv]
        N1, A1 = tri_normals(Q.reshape(-1, 9))
        bad = ((np.einsum('ij,ij->i', N, N1) <= 0) | (A1 <= 1e-3 * A)) & (A > 0)
        hit = np.unique(inv[bad[tri]])
        hit = hit[move[hit]]
        if not len(hit):
            break
        move[hit] = False
    return Q.reshape(-1, 9), len(chosen)


def shells(P, F, rng, frac, gap):
    """Translate random whole faces off the part as rigid shells. Returns (P, shells)."""
    N, A = tri_normals(P)
    faces, counts = np.unique(F, return_counts=True)
    candidates = faces[counts <= 0.3 * len(F)]          # never detach most of the part
    chosen = candidates[rng.random(len(candidates)) < frac]
    Q = P.copy()
    for f in chosen:
        m = F == f
        n = (N[m] * A[m, None]).sum(0)
        ln = np.linalg.norm(n)
        if ln < 1e-12 * A[m].sum():                     # closed face, e.g. a full sphere
            n = rng.normal(size=3)
            ln = np.linalg.norm(n)
        Q[m] += np.tile(n / ln * gap * rng.choice((-1.0, 1.0)), 3)
    return Q, len(chosen)


# ── Per part ─────────────────────────────────────────────────────────────────

def bake_game(task):
    model_id, part, src_path, step_path, out_path, ops, cfg = task
    rec = {"model_id": model_id, "part": part, "status": "ok", "tris": 0,
           **{op: 0 for op in ops}}
    try:
        if Path(out_path).exists():
            rec["status"] = "exists"
            return rec
        src = np.load(src_path)
        diag = float(src["diag"])
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(f"{model_id}_{part}".encode())])

        if "lowpoly" in ops:
            from occ_tessellate import read_step, tessellate
            shape = read_step(step_path) if step_path else None
            if shape is None:
                rec["status"] = "unreadable"
                return rec
            layers = {}
            for name, lin, ang in GAME_LADDER:
                t = tessellate(shape, lin, ang, diag)
                layers[name] = (t.positions, t.face_id, t.face_type)
            rec["lowpoly"] = len(layers)
        else:
            layers = {name: (src[f"{name}_positions"], src[f"{name}_face_id"],
                             src[f"{name}_face_type"])
                      for name in ("coarse", "medium", "fine") if f"{name}_positions" in src.files}

        gap = max(cfg["gap"] * diag, MIN_GAP)
        arrays = {}
        for name, (pos, fid, fty) in layers.items():
            P = np.asarray(pos, np.float64)
            F = np.asarray(fid, np.int64)
            if len(P) < MIN_TRIS:
                continue
            if "quadflip" in ops:
                P, k = quadflip(P, F, rng, cfg["flip_prob"], diag)
                rec["quadflip"] += k
            if "jitter" in ops:
                P = jitter(P, rng, cfg["jitter"])
                rec["jitter"] += 1
            if "unweld" in ops:
                P, k = unweld(P, F, rng, cfg["seam_frac"], gap)
                rec["unweld"] += k
            if "shells" in ops:
                P, k = shells(P, F, rng, cfg["shell_frac"], gap * cfg["shell_gap_scale"])
                rec["shells"] += k
            _, area = tri_normals(P)
            keep = area > 1e-12 * diag * diag
            if MIN_TRIS <= keep.sum() <= MAX_TRIS:
                arrays[f"{name}_positions"] = P[keep].astype(np.float32)
                arrays[f"{name}_face_id"] = np.asarray(fid)[keep].astype(np.int32)
                arrays[f"{name}_face_type"] = np.asarray(fty)[keep].astype(np.uint8)
                rec["tris"] += int(keep.sum())
        if not arrays:
            rec["status"] = "no_usable_density"
            return rec
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, model_id=model_id, part=part, diag=np.float64(diag),
                 mesher="game:" + ",".join(ops), **arrays)
    except Exception as exc:
        rec["status"] = f"error:{type(exc).__name__}"
    return rec


def main(argv=None):
    from isolated_pool import run_isolated
    from relabel_sidecar import val_split
    from stage1_bake import find_step

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", default=DEFAULT_OPS, help=f"comma list from {', '.join(OPS)}")
    ap.add_argument("--split", default="val", choices=("val", "train", "all"))
    ap.add_argument("--bake", type=Path, default=Path("D:/abc-data/bake"),
                    help="main bake whose files and split define the parts")
    ap.add_argument("--step-root", type=Path, default=Path("D:/abc-data/step"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flip-prob", type=float, default=0.5)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--seam-frac", type=float, default=0.3,
                    help="fraction of adjacent face pairs whose seam is opened")
    ap.add_argument("--shell-frac", type=float, default=0.1,
                    help="fraction of faces detached as shells")
    ap.add_argument("--gap", type=float, default=1e-3, help="seam gap / diagonal")
    ap.add_argument("--shell-gap-scale", type=float, default=5.0)
    args = ap.parse_args(argv)

    ops = tuple(o for o in args.ops.split(",") if o)
    unknown = set(ops) - set(OPS)
    if unknown:
        ap.error(f"unknown ops: {', '.join(sorted(unknown))}")
    ops = tuple(o for o in OPS if o in ops)              # fixed order, whatever was typed
    cfg = {k: getattr(args, k) for k in
           ("seed", "flip_prob", "jitter", "seam_frac", "shell_frac", "gap", "shell_gap_scale")}

    val, train = val_split(args.bake)
    files = {"val": val, "train": train, "all": val + train}[args.split]
    tasks = []
    for f in files:
        model_id, part = f.stem.rsplit("_", 1)
        sp = find_step(args.step_root, model_id) if "lowpoly" in ops else None
        if "lowpoly" in ops and sp is None:
            continue
        tasks.append((model_id, part, str(f), str(sp) if sp else None,
                      str(args.out / f.relative_to(args.bake)), ops, cfg))
        if args.limit and len(tasks) >= args.limit:
            break
    print(f"game:{','.join(ops)}: {len(tasks):,} parts from the {args.split} split -> {args.out}",
          flush=True)

    t0 = time.time()
    status, totals = Counter(), Counter()
    fail = lambda task, st: {"model_id": task[0], "part": task[1], "status": st, "tris": 0}
    for i, rec in enumerate(run_isolated(bake_game, tasks, args.jobs, args.timeout, fail), 1):
        status[rec["status"]] += 1
        totals.update({k: v for k, v in rec.items() if k in OPS or k == "tris"})
        if i % 250 == 0:
            print(f"  {i:,}/{len(tasks):,}  {i / (time.time() - t0):.1f}/s", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    print("status: " + ", ".join(f"{k} {v:,}" for k, v in status.most_common()))
    print("totals: " + ", ".join(f"{k} {v:,}" for k, v in totals.items()))


if __name__ == "__main__":
    raise SystemExit(main())
