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
    adjacency_csr, build_adjacency, build_facets, facet_curvature, k_ring, weld_quantum,
)
from seedfeatures import EXT_FEATURE_NAMES, adjacency_csr_angles, ext_features  # noqa: E402
from occ_tessellate import TYPE_NAMES  # noqa: E402
from surface_targets import N_SURFACE, targets as surface_targets  # noqa: E402

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

# v6: the 12 above plus the extended columns of seedfeatures.py (crease bottleneck, local
# crease, normal-spread eigenvalues, seed-local primitive fits), chosen by ml/feature_screen.py.
FEATURE_NAMES_V6 = FEATURE_NAMES + EXT_FEATURE_NAMES
N_FEATURES_V6 = len(FEATURE_NAMES_V6)


def derive(positions: np.ndarray, with_angles: bool = False):
    """Facets, adjacency and curvature for one baked mesh, via the checked port. With
    `with_angles`, also the dihedral angle of every CSR entry (for the v6 features)."""
    facets = build_facets(positions)
    pairs, angles = build_adjacency(positions, facets, weld_quantum(positions))
    curv = facet_curvature(positions, facets)
    if with_angles:
        indptr, indices, eangles = adjacency_csr_angles(pairs, angles, facets["count"])
        return facets, indptr, indices, curv, eangles
    indptr, indices = adjacency_csr(pairs, facets["count"])
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
                  spin: bool = True, curv: dict | None = None,
                  surface: tuple | None = None, extended: bool = False,
                  eangles: np.ndarray | None = None,
                  excluded: np.ndarray | None = None, gnn: bool = False,
                  hard_rings: int = 0) -> dict | None:
    """Build one training sample around `seed`. Returns None if the patch is too small.

    `surface` is a (kind, params) pair from surface_sidecar.py, indexed by face id. When given,
    the sample also carries the seed face's surface-hypothesis targets in this patch's frame
    ("surface", "surface_mask"), and the frame itself ("R", "q") so a predicted surface can be
    placed back in the world.
    """
    ids, rings = k_ring(seed, indptr, indices, max_facets, excluded=excluded)
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
    if extended:
        ext = ext_features(ids, seed, facets, indptr, indices, eangles, d, n, scale, excluded)
        feats = np.concatenate([feats, ext], axis=1)

    out = {
        "features": feats,
        "same": (face_id[ids] == face_id[seed]).astype(np.float32),
        "type": int(face_type[seed]),
        "n": int(ids.size),
        "ids": ids,
        "seed": int(seed),
        "scale": scale,
    }
    if gnn or hard_rings:
        nbr, eattr = patch_graph(ids, indptr, indices, eangles)
        if gnn:
            out["nbr"], out["eattr"] = nbr, eattr
        if hard_rings:
            out["band"] = boundary_band(out["same"], nbr, eattr, hard_rings)
    if surface is not None:
        kinds, params = surface
        sf = int(face_id[seed])
        tgt = np.zeros(N_SURFACE, np.float32)
        msk = np.zeros(3, np.float32)
        if sf < len(kinds):
            q = np.asarray(seed_pos, np.float64)
            nn = np.asarray(seed_nrm, np.float64)
            nn = nn / (np.linalg.norm(nn) + 1e-30)
            ax, ha, kn, hk, ks, hs = surface_targets(int(kinds[sf]), params[sf], q, nn, R, scale)
            tgt[0:3], tgt[3], tgt[4] = ax, kn, ks
            msk[:] = (ha, hk, hs)
        out.update(surface=tgt, surface_mask=msk, R=R, q=np.asarray(seed_pos, np.float64))
    return out


def patch_graph(ids: np.ndarray, indptr, indices, eangles):
    """Patch-local neighbours of every patch facet, (n, 3) int64 with -1 where the neighbour is
    missing or outside the patch, and each edge's dihedral / 90 (n, 3). Mirrored by
    seedNet.patchNeighbours; the model max-pools over the three, so slot order does not matter."""
    n = ids.size
    nbr = np.full((n, 3), -1, np.int64)
    eattr = np.zeros((n, 3), np.float32)
    order = np.argsort(ids, kind="stable")
    sids = ids[order]
    starts = indptr[ids]
    counts = indptr[ids + 1] - starts
    for k in range(3):
        rows = np.nonzero(counts > k)[0]
        if rows.size == 0:
            continue
        nb = indices[starts[rows] + k]
        pos = np.minimum(np.searchsorted(sids, nb), n - 1)
        found = sids[pos] == nb
        nbr[rows[found], k] = order[pos[found]]
        if eangles is not None:
            eattr[rows[found], k] = eangles[starts[rows[found]] + k] / 90.0
    return nbr, eattr


SMOOTH_BOUNDARY = 30.0 / 90.0


def boundary_band(same: np.ndarray, nbr: np.ndarray, eattr: np.ndarray, rings: int) -> np.ndarray:
    """Patch facets within `rings` of a SMOOTH boundary of the seed's face: an edge between a facet
    on the seed's face and one off it whose dihedral is under 30 degrees. That is where the crease
    barrier is blind and fills spill, and where the membership head must be sharpest."""
    valid = nbr >= 0
    j = np.where(valid, nbr, 0)
    edge = valid & (same[:, None] != same[j]) & (eattr < SMOOTH_BOUNDARY)
    band = edge.any(axis=1)
    for _ in range(rings - 1):
        band = band | (valid & band[j]).any(axis=1)
    return band.astype(np.float32)


def collate(samples: list[dict], max_facets: int):
    """Pad to a fixed width and build a validity mask. Padding is masked out of both
    the loss and the global max-pool, so it cannot influence a prediction."""
    b = len(samples)
    x = np.zeros((b, max_facets, samples[0]["features"].shape[1]), np.float32)
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


def collate_weights(samples: list[dict], max_facets: int, hard_weight: float):
    """Per-facet loss weights: `hard_weight` inside the boundary band, 1 elsewhere, 0 on padding."""
    b = len(samples)
    w = np.zeros((b, max_facets), np.float32)
    for i, s in enumerate(samples):
        k = min(s["n"], max_facets)
        band = s.get("band")
        w[i, :k] = 1.0 if band is None else 1.0 + (hard_weight - 1.0) * band[:k]
    return w


def collate_graph(samples: list[dict], max_facets: int):
    """Neighbour indices and dihedrals, padded with -1 / 0 like the features."""
    b = len(samples)
    nbr = np.full((b, max_facets, 3), -1, np.int64)
    eattr = np.zeros((b, max_facets, 3), np.float32)
    for i, s in enumerate(samples):
        k = min(s["n"], max_facets)
        g = s["nbr"][:k]
        nbr[i, :k] = np.where(g < k, g, -1)
        eattr[i, :k] = s["eattr"][:k]
    return nbr, eattr


def collate_surface(samples: list[dict]):
    """Surface targets and their masks; samples without a sidecar contribute zeros, masked out."""
    b = len(samples)
    tgt = np.zeros((b, N_SURFACE), np.float32)
    msk = np.zeros((b, 3), np.float32)
    for i, s in enumerate(samples):
        if "surface" in s:
            tgt[i] = s["surface"]
            msk[i] = s["surface_mask"]
    return tgt, msk


class BakedPatches:
    """Iterates baked .npz files, emitting `seeds_per_mesh` patches from each.

    Not a torch Dataset on purpose - it yields plain numpy so it can be exercised and
    measured without importing torch. train.py wraps it.
    """

    def __init__(self, root: Path, densities=("coarse", "medium", "fine"),
                 max_facets: int = 512, seeds_per_mesh: int = 24,
                 type_weights: np.ndarray | None = None, seed: int = 0,
                 files: list[Path] | None = None, relabel_root: Path | None = None,
                 surface_root: Path | None = None, extended: bool = False,
                 exclude_prob: float = 0.0, gnn: bool = False, hard_rings: int = 0,
                 boundary_seed_prob: float = 0.0):
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
        # SURFACE TARGETS. When set, patches carry the seed face's analytic surface in their
        # frame (surface_sidecar.py). A file without a sidecar is NOT skipped: its patches
        # still train membership and type, with the surface loss masked out.
        self.surface_root = Path(surface_root) if surface_root else None
        # v6 inputs (seedfeatures.py) and PAINTED-NEIGHBOUR AUGMENTATION: with probability
        # `exclude_prob` a patch is drawn as if some neighbouring faces were already painted into
        # other groups, which the studio removes from the model's patch. Faces of a simpler type
        # than the seed's are removed more often (0.6 against 0.25), matching painting simpler
        # surfaces first. Simulated without retraining, excluding painted facets raised clean
        # fills on gmsh 46 -> 58% but degraded the type head on the holed patches it never saw.
        self.extended = extended
        self.exclude_prob = exclude_prob
        self.gnn = gnn
        # DISCRIMINATIVE TRAINING: a band of `hard_rings` around the seed face's smooth boundaries
        # is marked for loss weighting (train.py --hard-weight), and a share of seeds is drawn from
        # facets next to a smooth face boundary, where fills actually go wrong.
        self.hard_rings = hard_rings
        self.boundary_seed_prob = boundary_seed_prob

    def _excluded_faces(self, s: int, indptr, indices, face_id, face_type):
        ids0, _ = k_ring(s, indptr, indices, self.max_facets)
        fids = face_id[ids0]
        faces, first = np.unique(fids, return_index=True)
        types = face_type[ids0][first]
        keep = faces != face_id[s]
        faces, types = faces[keep], types[keep]
        if faces.size == 0:
            return None
        p = np.where(types < face_type[s], 0.6, 0.25)
        chosen = faces[self.rng.random(faces.size) < p]
        return np.isin(face_id, chosen) if chosen.size else None

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
        surface = None
        if self.surface_root is not None and "model_id" in data.files:
            mid, part = str(data["model_id"]), str(data["part"])
            ss = self.surface_root / mid[:5] / f"{mid}_{part}.npz"
            if ss.exists():
                try:
                    sd = np.load(ss)
                    surface = (sd["kind"], sd["params"])
                except Exception:
                    surface = None
        for dens in self.densities:
            key = f"{dens}_positions"
            if key not in data.files:
                continue
            positions = data[key]
            face_id = data[f"{dens}_face_id"]
            face_type = data[f"{dens}_face_type"] if canon is None else canon[face_id]
            if positions.shape[0] < 32:
                continue
            eangles = None
            hard_rings = getattr(self, "hard_rings", 0)
            bprob = getattr(self, "boundary_seed_prob", 0.0)
            if self.extended or self.exclude_prob > 0 or self.gnn or hard_rings or bprob:
                facets, indptr, indices, curv, eangles = derive(positions, with_angles=True)
            else:
                facets, indptr, indices, curv = derive(positions)
            ntri = facets["count"]

            seeds = self._choose_seeds(face_type, ntri)
            if bprob > 0:
                seeds = self._boundary_seeds(seeds, face_id, face_type, indptr, indices, eangles, bprob)
            for s in seeds:
                excluded = None
                if self.exclude_prob > 0 and self.rng.random() < self.exclude_prob:
                    excluded = self._excluded_faces(int(s), indptr, indices, face_id, face_type)
                out = extract_patch(facets, indptr, indices, face_id, face_type,
                                    int(s), self.max_facets, self.rng, curv=curv,
                                    surface=surface, extended=self.extended,
                                    eangles=eangles, excluded=excluded, gnn=self.gnn,
                                    hard_rings=hard_rings)
                if out is not None:
                    out["density"] = dens
                    yield out

    def _boundary_seeds(self, seeds, face_id, face_type, indptr, indices, eangles, prob):
        """Replace each seed, with probability `prob`, by a facet on a smooth face boundary (an
        edge to another face under 30 degrees), type-weighted like the rest when weights are set."""
        src = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
        smooth = (face_id[src] != face_id[indices]) & (eangles < 30.0)
        cand = np.unique(src[smooth])
        if cand.size == 0:
            return seeds
        seeds = np.array(seeds, copy=True)
        swap = self.rng.random(seeds.size) < prob
        k = int(swap.sum())
        if k == 0:
            return seeds
        p = None
        if self.type_weights is not None:
            w = self.type_weights[face_type[cand]]
            if np.isfinite(w.sum()) and w.sum() > 0:
                p = w / w.sum()
        seeds[swap] = self.rng.choice(cand, size=k, replace=True, p=p)
        return seeds

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
