#!/usr/bin/env python3
"""
abc_freeform.py - ABC's non-analytic surfaces (BSpline, Extrusion, Revolution; Other has no
definition) read from a feature file, and turned into exact nurbs.Surface objects to grade fits
against.

    read_freeform(path)   {surface index: definition dict} for every non-analytic surface. The
                          definitions are JSON-ready (export_abc_obj_eval.py --freeform writes them
                          to <part>.ffs.json)
    truth_surface(defn, pts)
                          the exact surface as a nurbs.Surface, or None (Other, ellipse sweeps):
                            BSpline      as stored (flat knots, possibly unclamped / rational)
                            Extrusion    its curve swept by degree 1 along the direction; the
                                         direction's extent is taken from pts (the face's vertices)
                            Revolution   its curve revolved a full turn (the 9-pole rational circle,
                                         NURBS Book A8.1)
                          Curves may be BSpline or Circle. The Ellipse records in ABC are unusable
                          (both foci at z = -500000 and no centre), so ellipse sweeps get no truth.

The feature files are YAML, but ml/ reads them without a YAML dependency (as export_abc_obj_eval.py
does): _parse_block handles the subset they use - block mappings and sequences, a sequence at its
key's indent, and flow lists that wrap onto continuation lines.
"""

from __future__ import annotations

import json
import math

import numpy as np

import nurbs

FREEFORM = ("Revolution", "Extrusion", "BSpline", "Other")
DROP = ("face_indices", "vert_indices", "vert_parameters")


# ── a YAML subset ────────────────────────────────────────────────────────────

def _logical_lines(text):
    out, buf, depth = [], None, 0
    for raw in text.split("\n"):
        if buf is not None:
            buf += " " + raw.strip()
            depth += raw.count("[") - raw.count("]")
            if depth <= 0:
                out.append(buf)
                buf = None
            continue
        if not raw.strip():
            continue
        d = raw.count("[") - raw.count("]")
        if d > 0:
            buf, depth = raw, d
        else:
            out.append(raw)
    return [[len(l) - len(l.lstrip(" ")), l.strip()] for l in out]


def _scalar(s):
    s = s.strip()
    if s.startswith("["):
        return json.loads(s.replace(".nan", "NaN").replace(".inf", "Infinity"))
    if s in ("true", "false"):
        return s == "true"
    if s in ("null", "~", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s.strip("'\"")


def _parse_block(lines, i):
    ind, text = lines[i]
    if text == "-" or text.startswith("- "):
        seq = []
        while i < len(lines) and lines[i][0] == ind and (lines[i][1] == "-" or lines[i][1].startswith("- ")):
            lines[i] = [ind + 2, lines[i][1][2:]]           # the item's content, one level deeper
            val, i = _parse_block(lines, i)
            seq.append(val)
        return seq, i
    if ":" in text and not text.startswith("["):
        mp = {}
        while i < len(lines) and lines[i][0] == ind and not lines[i][1].startswith("- "):
            key, _, val = lines[i][1].partition(":")
            key, val = key.strip(), val.strip()
            i += 1
            if val:
                mp[key] = _scalar(val)
            elif i < len(lines) and (lines[i][0] > ind or (lines[i][0] == ind and lines[i][1].startswith("- "))):
                mp[key], i = _parse_block(lines, i)
            else:
                mp[key] = None
        return mp, i
    return _scalar(text), i + 1


def read_surfaces_full(path):
    """Every surface of a feature file as a dict (the whole YAML item, face_indices included)."""
    text = open(path, "r", encoding="utf8", errors="replace").read()
    k = text.find("\nsurfaces:")
    if k < 0:
        return []
    body = text[k + len("\nsurfaces:"):]
    end = 0
    for line in body.split("\n")[1:]:
        if line and not line.startswith((" ", "-")):
            break
        end += len(line) + 1
    lines = _logical_lines(body[:end + 1])
    if not lines:
        return []
    val, _ = _parse_block(lines, 0)
    return val or []


def read_freeform(path):
    """{surface index: definition} for the non-analytic surfaces (no face/vertex index lists)."""
    out = {}
    for i, s in enumerate(read_surfaces_full(path)):
        if s.get("type") in FREEFORM:
            d = {k: v for k, v in s.items() if k not in DROP}
            if isinstance(d.get("curve"), dict):
                d["curve"] = {k: v for k, v in d["curve"].items() if k not in DROP}
            out[i] = d
    return out


# ── exact surfaces ───────────────────────────────────────────────────────────

S2 = math.sqrt(0.5)
CIRCLE_XY = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]], float)
CIRCLE_W = np.array([1, S2, 1, S2, 1, S2, 1, S2, 1], float)
CIRCLE_T = [0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1]


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def curve_poles(c):
    """(poles (n,3), weights (n,), flat knots, degree) of a BSpline or Circle curve, or None."""
    if c is None:
        return None
    if c.get("type") == "BSpline":
        P = np.asarray(c["poles"], float)
        W = np.asarray(c["weights"], float) if c.get("weights") is not None else np.ones(len(P))
        if W.shape != (len(P),):
            W = np.ones(len(P))
        return P, W, list(map(float, c["knots"])), int(c["degree"])
    if c.get("type") == "Circle":
        o, x, y, r = (np.asarray(c["location"], float), _unit(c["x_axis"]), _unit(c["y_axis"]), float(c["radius"]))
        P = o + r * (CIRCLE_XY[:, :1] * x + CIRCLE_XY[:, 1:] * y)
        return P, CIRCLE_W.copy(), list(CIRCLE_T), 2
    return None


def truth_surface(defn, pts=None):
    kind = defn.get("type")
    if kind == "BSpline":
        W = defn.get("weights")
        rational = defn.get("u_rational") or defn.get("v_rational")
        return nurbs.Surface(defn["poles"], defn["u_knots"], defn["v_knots"], defn["u_degree"], defn["v_degree"],
                             W if rational else None)
    cp = curve_poles(defn.get("curve"))
    if cp is None:
        return None
    P, W, t, p = cp
    if kind == "Extrusion":
        d = _unit(defn["direction"])
        if pts is None or len(pts) == 0:
            return None
        pd, xd = P @ d, np.asarray(pts, float) @ d
        lo, hi = xd.min() - pd.max(), xd.max() - pd.min()
        m = 0.05 * (hi - lo) + 1e-9
        lo, hi = lo - m, hi + m
        poles = np.stack([P + lo * d, P + hi * d], axis=1)
        return nurbs.Surface(poles, t, [lo, lo, hi, hi], p, 1, np.stack([W, W], axis=1))
    if kind == "Revolution":
        o, a = np.asarray(defn["location"], float), _unit(defn["z_axis"])
        poles = np.zeros((len(P), 9, 3))
        weights = np.zeros((len(P), 9))
        for i, (q, w) in enumerate(zip(P, W)):
            c = o + ((q - o) @ a) * a                      # the pole's foot on the axis
            x = q - c
            r = np.linalg.norm(x)
            if r < 1e-300:
                poles[i] = c
            else:
                x = x / r
                y = np.cross(a, x)
                poles[i] = c + r * (CIRCLE_XY[:, :1] * x + CIRCLE_XY[:, 1:] * y)
            weights[i] = w * CIRCLE_W
        return nurbs.Surface(poles, t, CIRCLE_T, p, 2, weights)
    return None
