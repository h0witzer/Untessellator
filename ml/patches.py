#!/usr/bin/env python3
"""
patches.py - turn baked meshes into seeded training patches.

One sample is: a seed facet, the adjacency-BFS neighbourhood around it, per-facet
features, and two labels - which of those facets share the seed's B-rep face, and what
surface type the seed's face is. That is exactly the query `paintFrom` makes when a user
clicks, so the training distribution matches the deployed one by construction.

CANONICAL FRAME

Each patch is rotated so the SEED'S NORMAL POINTS ALONG +Z, then rotated randomly about
Z as augmentation. This is not cosmetic. "Which facets lie on the same surface as this
one" is invariant to how the part is oriented in space, so leaving global orientation in
the input forces the network to spend capacity learning an invariance we can simply hand
it. Aligning to the seed normal collapses two of the three rotational degrees of
freedom; randomising the third teaches the remaining one.

Positions are divided by the patch's RMS radius, so a 2 mm fillet and a 2 m flange
present identically. This matters more here than in most 3D tasks because ABC spans
millimetre parts to metre assemblies, and because surface TYPE is scale-free - a
cylinder is a cylinder at any radius.

WHY ONE MESH YIELDS MANY SAMPLES

Deriving facets and adjacency from a baked mesh costs far more than extracting a patch
from it, and the MVP framing means every facet is a legitimate seed. So the dataset
loads a mesh once, derives once, and emits `seeds_per_mesh` patches from it. This is
also what makes the Revolution shortage manageable: 1,642 Revolution models is not 1,642
training examples, it is 1,642 x (facets on revolution faces) seeds.

CLASS BALANCE

Measured on baked ABC: a 512-facet BFS around a random seed averages only ~17% facets
sharing the seed's face. Sampling seeds uniformly would also mirror the corpus type
distribution, which is 55% Plane. `type_weights` lets the sampler bias seed selection
towards rarer surface types; `pos_weight` for the binary head should be set from the
measured positive rate, not guessed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meshcore import (  # noqa: E402
    adjacency_csr, build_adjacency, build_facets, facet_curvature, k_ring,
)
from occ_tessellate import TYPE_NAMES  # noqa: E402

N_TYPES = len(TYPE_NAMES)
N_FEATURES = 12

FEATURE_NAMES = (
    "dx", "dy", "dz",          # position relative to seed, canonical frame, scaled
    "nx", "ny", "nz",          # unit normal in canonical frame
    "dot_seed",                # normal . seed normal
    "log_area",                # log(area / mean patch area)
    "ring",                    # BFS ring index / max ring
    "dist",                    # |position - seed| / scale
    # Principal curvatures, scaled by the patch radius so they are dimensionless like
    # everything else. These are the LOCAL surface-type signal: plane 0/0, cylinder
    # 1/R and 0, sphere both 1/R, torus 1/r around the tube at every point. The network
    # otherwise sees only positions, normals and adjacency, and has to infer local shape
    # from their arrangement.
    "k1s",                     # larger principal curvature * scale
    "k2s",                     # smaller principal curvature * scale
)


def derive(positions: np.ndarray):
    """Facets, adjacency and curvature for one baked mesh, via the checked port."""
    facets = build_facets(positions)
    pairs, angles = build_adjacency(positions, facets)
    indptr, indices = adjacency_csr(pairs, facets["count"])
    curv = facet_curvature(positions, facets)
    return facets, indptr, indices, curv


def _frame_from_normal(n: np.ndarray) -> np.ndarray:
    """Rotation taking `n` to +Z. Any frame will do as long as it is continuous enough
    not to flip wildly for nearby normals; the random spin about Z washes out the
    arbitrary choice of the other two axes."""
    n = n / (np.linalg.norm(n) + 1e-30)
    # Pick the least-aligned world axis to build a stable tangent.
    a = np.eye(3)[int(np.argmin(np.abs(n)))]
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1) + 1e-30
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n])          # rows: maps world -> (t1, t2, n)


def extract_patch(facets: dict, indptr, indices, face_id: np.ndarray,
                  face_type: np.ndarray, seed: int, max_facets: int,
                  rng: np.random.Generator | None = None,
                  spin: bool = True, curv: dict | None = None) -> dict | None:
    """Build one training sample around `seed`. Returns None if the patch is too small."""
    ids, rings = k_ring(seed, indptr, indices, max_facets)
    if ids.size < 8:
        return None

    cen = facets["centroids"][ids]
    nrm = facets["normals"][ids]
    ar = facets["areas"][ids]
    seed_pos = facets["centroids"][seed]
    seed_nrm = facets["normals"][seed]

    R = _frame_from_normal(seed_nrm)
    if spin and rng is not None:
        th = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R

    d = (cen - seed_pos) @ R.T
    n = nrm @ R.T
    scale = float(np.sqrt((d * d).sum(axis=1).mean())) or 1.0
    d = d / scale
    dist = np.linalg.norm(d, axis=1)

    mean_area = float(ar.mean()) or 1.0
    log_area = np.log(np.maximum(ar, 1e-30) / mean_area)
    ring_n = rings.astype(np.float64) / max(1.0, float(rings.max()))
    dot_seed = nrm @ seed_nrm

    # zeros, not empty: the curvature columns are only written when curvature is passed,
    # and the control arm trained with them at exactly zero. np.empty left whatever was
    # in memory there, which never bit training (curvature was always passed) but would
    # have fed garbage to any caller that omitted it.
    feats = np.zeros((ids.size, N_FEATURES), np.float32)
    feats[:, 0:3] = d
    feats[:, 3:6] = n
    feats[:, 6] = dot_seed
    feats[:, 7] = np.clip(log_area, -8, 8)
    feats[:, 8] = ring_n
    feats[:, 9] = dist
    if curv is not None:
        v = curv["valid"][ids]
        feats[:, 10] = np.where(v, np.clip(curv["k1"][ids] * scale, -20, 20), 0.0)
        feats[:, 11] = np.where(v, np.clip(curv["k2"][ids] * scale, -20, 20), 0.0)

    return {
        "features": feats,
        "same": (face_id[ids] == face_id[seed]).astype(np.float32),
        "type": int(face_type[seed]),
        "n": int(ids.size),
        "ids": ids,
        "seed": int(seed),
        "scale": scale,
    }


def collate(samples: list[dict], max_facets: int):
    """Pad to a fixed width and build a validity mask. Padding is masked out of both
    the loss and the global max-pool, so it cannot influence a prediction."""
    b = len(samples)
    x = np.zeros((b, max_facets, N_FEATURES), np.float32)
    y = np.zeros((b, max_facets), np.float32)
    m = np.zeros((b, max_facets), np.float32)
    t = np.zeros(b, np.int64)
    for i, s in enumerate(samples):
        k = min(s["n"], max_facets)
        x[i, :k] = s["features"][:k]
        y[i, :k] = s["same"][:k]
        m[i, :k] = 1.0
        t[i] = s["type"]
    return x, y, m, t


class BakedPatches:
    """Iterates baked .npz files, emitting `seeds_per_mesh` patches from each.

    Not a torch Dataset on purpose - it yields plain numpy so it can be exercised and
    measured without importing torch. train.py wraps it.
    """

    def __init__(self, root: Path, densities=("coarse", "medium", "fine"),
                 max_facets: int = 512, seeds_per_mesh: int = 24,
                 type_weights: np.ndarray | None = None, seed: int = 0,
                 files: list[Path] | None = None, relabel_root: Path | None = None):
        self.root = Path(root)
        self.densities = tuple(densities)
        self.max_facets = max_facets
        self.seeds_per_mesh = seeds_per_mesh
        self.type_weights = type_weights
        self.rng = np.random.default_rng(seed)
        self.files = files if files is not None else sorted(self.root.glob("*/*.npz"))
        # CORRECTED LABELS. When set, a real-bake file takes its face types from the
        # relabel sidecar (canonical.py) instead of the stored B-rep type, and a file
        # WITHOUT a sidecar is skipped rather than silently mixing the two label schemes.
        # Synthetic-bake files already carry canonical types and are recognised by their
        # "kind" field.
        self.relabel_root = Path(relabel_root) if relabel_root else None

    def __len__(self) -> int:
        return len(self.files)

    def mesh_samples(self, path: Path):
        """All patches from one baked file, across every density present."""
        try:
            data = np.load(path)
        except Exception:
            return
        canon = None
        if self.relabel_root is not None and "kind" not in data.files:
            mid, part = str(data["model_id"]), str(data["part"])
            sc = self.relabel_root / mid[:5] / f"{mid}_{part}.npz"
            if not sc.exists():
                return
            try:
                canon = np.load(sc)["canonical"]
            except Exception:
                return
        for dens in self.densities:
            key = f"{dens}_positions"
            if key not in data.files:
                continue
            positions = data[key]
            face_id = data[f"{dens}_face_id"]
            face_type = data[f"{dens}_face_type"] if canon is None else canon[face_id]
            if positions.shape[0] < 32:
                continue
            facets, indptr, indices, curv = derive(positions)
            ntri = facets["count"]

            seeds = self._choose_seeds(face_type, ntri)
            for s in seeds:
                out = extract_patch(facets, indptr, indices, face_id, face_type,
                                    int(s), self.max_facets, self.rng, curv=curv)
                if out is not None:
                    out["density"] = dens
                    yield out

    def _choose_seeds(self, face_type: np.ndarray, ntri: int) -> np.ndarray:
        k = min(self.seeds_per_mesh, ntri)
        if self.type_weights is None:
            return self.rng.choice(ntri, size=k, replace=False)
        w = self.type_weights[face_type]
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            return self.rng.choice(ntri, size=k, replace=False)
        return self.rng.choice(ntri, size=k, replace=False, p=w / total)

    def __iter__(self):
        order = self.rng.permutation(len(self.files))
        for i in order:
            yield from self.mesh_samples(self.files[i])
