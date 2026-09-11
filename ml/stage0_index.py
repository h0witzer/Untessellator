#!/usr/bin/env python3
"""
stage0_index.py - index an ABC corpus by surface type, without parsing YAML.

Stage 0 of the learned-segmenter experiment (docs/ml-experiment-scope.md). It answers
the one question that decides everything downstream: what is the ACTUAL class
distribution of surface types in ABC, and which models are worth the cost of
tessellating in Stage 1?

The ABC paper shows the patch-type distribution as a figure and publishes no numbers,
so this has to be measured. The expected shape - planes and cylinders dominating,
Torus/Revolution/Extrusion/BSpline rare - is exactly the setup where a model scores
well on mean IoU while never predicting the classes that motivated the work.


WHY A LINE SCANNER AND NOT A YAML PARSER

ABC feature files carry full control nets, knot vectors and per-vertex (u,v)
parameters, so they run to megabytes each. A conforming YAML parse materialises all
of that in order to count nine strings. This reads each file once, line by line, in
constant memory, and looks at only `type:` and `sharp:` lines.


WHY SECTION ROLES ARE DETECTED AND NOT HARDCODED

The ABC supplementary documents both taxonomies but not the YAML key names, and the
two vocabularies OVERLAP:

    surfaces: Plane Cylinder Cone Sphere Torus Revolution Extrusion BSpline Other
    curves:   Line Circle Ellipse BSpline Other

`BSpline` and `Other` are valid in both. Counting `type:` globally would silently
merge B-spline surfaces with B-spline curves - and B-spline surfaces are one of the
classes this whole experiment exists to cover. So we track which top-level section
each type line sits in, and decide each section's role from the unambiguous types it
contains (Plane/Cylinder/... => surfaces, Line/Circle/Ellipse => curves). The
detected key names are written to the scan_meta table so the inference is auditable
rather than assumed.

Usage:
    python stage0_index.py --root /path/to/abc/feat --db abc_index.sqlite
    python stage0_index.py --root ... --db ... --limit 5000 --jobs 8

Then:
    python stage0_report.py --db abc_index.sqlite

Standard library only, deliberately - it matches the zero-dependency stance of the
rest of the repo, and Stage 0 needs to run before any ML toolchain is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCHEMA_VERSION = 1

# Taxonomies, verbatim from the ABC supplementary material (section 2.3).
SURFACE_TYPES = (
    "Plane", "Cylinder", "Cone", "Sphere", "Torus",
    "Revolution", "Extrusion", "BSpline", "Other",
)
CURVE_TYPES = ("Line", "Circle", "Ellipse", "BSpline", "Other")

# Types that appear in exactly one taxonomy, and therefore identify a section's role.
SURFACE_ONLY = frozenset({"Plane", "Cylinder", "Cone", "Sphere", "Torus", "Revolution", "Extrusion"})
CURVE_ONLY = frozenset({"Line", "Circle", "Ellipse"})

# A top-level YAML key: no indentation, nothing after the colon.
_SECTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(?:#.*)?$")
# A `type:` line. Group 1 is leading whitespace, group 2 the dash (if any), group 3 the value.
_TYPE_RE = re.compile(r"^([ \t]*)(-[ \t]+)?type:[ \t]*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?[ \t]*(?:#.*)?$")
_SHARP_RE = re.compile(r"^[ \t]*(?:-[ \t]+)?sharp:[ \t]*([A-Za-z]+)[ \t]*(?:#.*)?$")

# ABC filenames look like 00000050_80d90bfdd2e74e709a56641e_features_001.yml.
# The trailing index is a PART number: one model can contain several solids, each with
# its own feature file. Keying on the 8-digit id alone would silently drop parts.
_NAME_RE = re.compile(r"^(\d{8})_[0-9a-f]+_features?_(\d+)", re.IGNORECASE)
_ID_RE = re.compile(r"^(\d{8})")


# --------------------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------------------

def parse_identity(path: Path) -> tuple[str, str]:
    """Return (model_id, part) for an ABC feature file.

    Falls back progressively so that a renamed or re-chunked corpus still indexes
    rather than colliding every row onto one primary key.
    """
    name = path.name
    m = _NAME_RE.match(name)
    if m:
        return m.group(1), m.group(2).lstrip("0") or "0"
    m = _ID_RE.match(name)
    if m:
        return m.group(1), "0"
    # Some layouts put the id on the containing directory instead.
    if _ID_RE.match(path.parent.name):
        return path.parent.name[:8], path.stem
    return path.stem, "0"


def scan_file(path: Path) -> dict:
    """Scan one feature file. Returns raw per-section type counts; roles are not
    assigned here so that the worker stays pure and roles can be decided once, in the
    parent, with corpus-wide evidence.

    Counts are keyed by (section, effective column) so that a nested `type:` field
    cannot be mistaken for a list entry.

    "Effective column" is the column the KEY starts at, counting the dash as
    indentation: `- type:` at indent 0 and `  type:` at indent 2 are both at column 2
    and belong to the same entry level. That distinction is load-bearing on real ABC
    data, where keys are emitted in ALPHABETICAL order, so the dash usually carries
    `location:` and `type:` arrives undashed a few lines later:

        curves:
        - location: [0.0, -3.175, 0.0]
          radius: 17.5145949
          sharp: true
          type: Circle          <- indent 2, no dash

    but an entry whose other keys all sort after "type" puts it on the dash instead.
    Both forms are the same level, and a child object's `type:` is strictly deeper,
    so the reduction below simply keeps the shallowest column per section.
    """
    result = {
        "path": str(path),
        "file_bytes": 0,
        "n_sharp": 0,
        "sections": {},       # section -> {(has_dash, indent) -> Counter}
        "flags": set(),
        "error": None,
    }
    try:
        result["file_bytes"] = path.stat().st_size
    except OSError:
        pass

    section = None
    buckets: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))

    try:
        # errors="replace": a handful of ABC files carry stray bytes. A mangled comment
        # must not abort a 10k-model chunk.
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if line[0] not in " \t-":
                    m = _SECTION_RE.match(line.rstrip("\n"))
                    if m:
                        section = m.group(1)
                        continue
                m = _TYPE_RE.match(line.rstrip("\n"))
                if m:
                    indent = len(m.group(1).expandtabs(2))
                    dash = m.group(2)
                    column = indent + (len(dash.expandtabs(2)) if dash else 0)
                    buckets[section or "<root>"][column][m.group(3)] += 1
                    continue
                m = _SHARP_RE.match(line.rstrip("\n"))
                if m and m.group(1).lower() in ("true", "yes"):
                    result["n_sharp"] += 1
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["flags"].add("read_error")
        return result

    # Reduce each section to its entry level: the shallowest effective column.
    # Anything deeper is a field of an entry, not an entry.
    for name, levels in buckets.items():
        if not levels:
            continue
        result["sections"][name] = dict(levels[min(levels)])

    if not result["sections"]:
        result["flags"].add("no_sections")
    result["flags"] = sorted(result["flags"])
    return result


def classify_sections(sections: dict[str, dict[str, int]],
                      learned: dict[str, str] | None = None) -> tuple[dict, dict, set]:
    """Split scanned sections into (surface_hist, curve_hist, flags).

    In-file evidence wins: a section containing Plane/Cylinder/... is the surface list
    regardless of what it is called. `learned` supplies the name->role map inferred
    from the corpus sample, and is consulted only when a file is internally ambiguous
    (for example a part whose only patches are B-splines).
    """
    learned = learned or {}
    surface_hist: Counter = Counter()
    curve_hist: Counter = Counter()
    flags: set[str] = set()

    for name, hist in sections.items():
        types = set(hist)
        if types & SURFACE_ONLY:
            role = "surfaces"
        elif types & CURVE_ONLY:
            role = "curves"
        else:
            role = learned.get(name)
            if role is None:
                flags.add("ambiguous_sections")
                continue
        (surface_hist if role == "surfaces" else curve_hist).update(hist)

    known = SURFACE_TYPES if surface_hist else ()
    if any(t not in SURFACE_TYPES for t in surface_hist):
        flags.add("unknown_type")
    if any(t not in CURVE_TYPES for t in curve_hist):
        flags.add("unknown_type")
    return dict(surface_hist), dict(curve_hist), flags


def learn_section_roles(paths: list[Path]) -> tuple[dict[str, str], dict]:
    """Infer the name->role map from a sample, by unambiguous-type evidence only.

    Returns (roles, evidence) where evidence records the vote counts so the inference
    can be audited in scan_meta rather than trusted.
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    for path in paths:
        res = scan_file(path)
        for name, hist in res["sections"].items():
            types = set(hist)
            if types & SURFACE_ONLY:
                votes[name]["surfaces"] += 1
            elif types & CURVE_ONLY:
                votes[name]["curves"] += 1
    roles = {}
    evidence = {}
    for name, counter in votes.items():
        role, n = counter.most_common(1)[0]
        roles[name] = role
        evidence[name] = dict(counter)
        if len(counter) > 1:
            # Both roles seen under one key: the corpus does not match our assumptions.
            evidence[name]["_conflict"] = True
    return roles, evidence


def build_row(res: dict, root: Path, roles: dict[str, str]) -> dict:
    """Turn a raw scan result into the row written to SQLite."""
    path = Path(res["path"])
    model_id, part = parse_identity(path)
    surface_hist, curve_hist, flags = classify_sections(res["sections"], roles)
    flags = set(flags) | set(res["flags"])

    n_patches = sum(surface_hist.values())
    n_curves = sum(curve_hist.values())

    if res["error"]:
        flags.add("parse_error")
    if n_patches == 0:
        flags.add("no_surfaces")
    elif n_patches == 1:
        # "models that are just made of single primitives such as a plane" - ABC paper.
        flags.add("single_primitive")

    # Cheap structural fingerprint for deduplication. ABC contains many duplicate
    # models; this catches structurally identical parts without any geometry, which is
    # all Stage 0 can afford. Geometric dedup belongs in Stage 1.
    finger = json.dumps(
        [n_patches, n_curves, sorted(surface_hist.items()), sorted(curve_hist.items())],
        separators=(",", ":"),
    )
    struct_hash = hashlib.sha1(finger.encode("utf-8")).hexdigest()[:16]

    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)

    row = {
        "model_id": model_id,
        "part": part,
        "rel_path": rel,
        "file_bytes": res["file_bytes"],
        "n_patches": n_patches,
        "n_curves": n_curves,
        "n_sharp": res["n_sharp"],
        "struct_hash": struct_hash,
        "flags": ",".join(sorted(flags)),
        "error": res["error"],
    }
    for t in SURFACE_TYPES:
        row[f"s_{t.lower()}"] = surface_hist.get(t, 0)
    for t in CURVE_TYPES:
        row[f"c_{t.lower()}"] = curve_hist.get(t, 0)

    unknown = [("surfaces", t, n) for t, n in surface_hist.items() if t not in SURFACE_TYPES]
    unknown += [("curves", t, n) for t, n in curve_hist.items() if t not in CURVE_TYPES]
    row["_unknown"] = unknown
    return row


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------

SURFACE_COLS = "".join(f"  s_{t.lower()} INTEGER NOT NULL DEFAULT 0,\n" for t in SURFACE_TYPES)
CURVE_COLS = "".join(f"  c_{t.lower()} INTEGER NOT NULL DEFAULT 0,\n" for t in CURVE_TYPES)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS parts (
  model_id   TEXT NOT NULL,
  part       TEXT NOT NULL,
  rel_path   TEXT NOT NULL,
  file_bytes INTEGER NOT NULL DEFAULT 0,
  n_patches  INTEGER NOT NULL DEFAULT 0,
  n_curves   INTEGER NOT NULL DEFAULT 0,
  n_sharp    INTEGER NOT NULL DEFAULT 0,
{SURFACE_COLS}{CURVE_COLS}  struct_hash TEXT NOT NULL DEFAULT '',
  flags      TEXT NOT NULL DEFAULT '',
  error      TEXT,
  PRIMARY KEY (model_id, part)
);

CREATE TABLE IF NOT EXISTS unknown_types (
  section   TEXT NOT NULL,
  type_name TEXT NOT NULL,
  n         INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (section, type_name)
);

CREATE TABLE IF NOT EXISTS scan_meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE INDEX IF NOT EXISTS idx_parts_hash    ON parts(struct_hash);
CREATE INDEX IF NOT EXISTS idx_parts_flags   ON parts(flags);
CREATE INDEX IF NOT EXISTS idx_parts_torus   ON parts(s_torus);
CREATE INDEX IF NOT EXISTS idx_parts_bspline ON parts(s_bspline);
CREATE INDEX IF NOT EXISTS idx_parts_rev     ON parts(s_revolution);
CREATE INDEX IF NOT EXISTS idx_parts_ext     ON parts(s_extrusion);
"""

ROW_COLS = (
    ["model_id", "part", "rel_path", "file_bytes", "n_patches", "n_curves", "n_sharp"]
    + [f"s_{t.lower()}" for t in SURFACE_TYPES]
    + [f"c_{t.lower()}" for t in CURVE_TYPES]
    + ["struct_hash", "flags", "error"]
)
INSERT_SQL = (
    f"INSERT OR REPLACE INTO parts ({','.join(ROW_COLS)}) "
    f"VALUES ({','.join('?' for _ in ROW_COLS)})"
)


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO scan_meta (key, value) VALUES (?, ?)",
        (key, value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True)),
    )


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def discover(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    out.sort()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="directory of extracted ABC feature files (searched recursively)")
    ap.add_argument("--db", required=True, type=Path, help="SQLite index to create or extend")
    ap.add_argument("--pattern", default="*.yml,*.yaml",
                    help="comma-separated filename globs (default: *.yml,*.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="index at most N files (0 = all)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="worker processes")
    ap.add_argument("--sample", type=int, default=200,
                    help="files used to infer section roles before the main pass")
    ap.add_argument("--batch", type=int, default=2000, help="rows per SQLite transaction")
    ap.add_argument("--force", action="store_true", help="re-scan files already in the index")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: --root is not a directory: {args.root}", file=sys.stderr)
        return 2

    patterns = tuple(p.strip() for p in args.pattern.split(",") if p.strip())
    print(f"scanning {args.root} for {'/'.join(patterns)} ...", flush=True)
    paths = discover(args.root, patterns)
    if not paths:
        print("error: no feature files found. ABC chunks ship as .7z - extract them first.",
              file=sys.stderr)
        return 2
    print(f"found {len(paths):,} files")

    conn = open_db(args.db)
    if not args.force:
        known = {r[0] for r in conn.execute("SELECT rel_path FROM parts")}
        if known:
            before = len(paths)
            paths = [p for p in paths
                     if str(p.relative_to(args.root) if p.is_relative_to(args.root) else p) not in known]
            print(f"resuming: {before - len(paths):,} already indexed, {len(paths):,} to go")
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("nothing to do")
        return 0

    # Phase 1: infer which section is which, from unambiguous types only.
    sample = paths[: min(args.sample, len(paths))]
    print(f"inferring section roles from {len(sample)} files ...", flush=True)
    roles, evidence = learn_section_roles(sample)
    if roles:
        print("  detected: " + ", ".join(f"{k} -> {v}" for k, v in sorted(roles.items())))
    else:
        print("  WARNING: no unambiguous types found in the sample; role inference failed.")
        print("  Files whose sections cannot be resolved will be flagged ambiguous_sections.")
    for name, ev in evidence.items():
        if ev.get("_conflict"):
            print(f"  WARNING: section '{name}' voted both roles: {ev}")

    # Phase 2: the real pass.
    t0 = time.time()
    unknown_total: Counter = Counter()
    flag_total: Counter = Counter()
    done = 0
    batch: list[tuple] = []

    def flush() -> None:
        if batch:
            conn.executemany(INSERT_SQL, batch)
            conn.commit()
            batch.clear()

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for res in pool.map(scan_file, paths, chunksize=32):
            row = build_row(res, args.root, roles)
            for section, tname, n in row.pop("_unknown"):
                unknown_total[(section, tname)] += n
            for f in row["flags"].split(","):
                if f:
                    flag_total[f] += 1
            batch.append(tuple(row[c] for c in ROW_COLS))
            done += 1
            if len(batch) >= args.batch:
                flush()
                rate = done / max(1e-6, time.time() - t0)
                print(f"  {done:,}/{len(paths):,}  ({rate:,.0f} files/s)", flush=True)
    flush()

    for (section, tname), n in unknown_total.items():
        conn.execute(
            "INSERT INTO unknown_types (section, type_name, n) VALUES (?,?,?) "
            "ON CONFLICT(section, type_name) DO UPDATE SET n = n + excluded.n",
            (section, tname, n),
        )

    elapsed = time.time() - t0
    set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    set_meta(conn, "section_roles", roles)
    set_meta(conn, "section_role_evidence", evidence)
    set_meta(conn, "last_scan_root", str(args.root))
    set_meta(conn, "last_scan_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    set_meta(conn, "last_scan_files", str(done))
    set_meta(conn, "last_scan_seconds", f"{elapsed:.1f}")
    conn.commit()

    print(f"\nindexed {done:,} files in {elapsed:.1f}s ({done / max(1e-6, elapsed):,.0f}/s)")
    if flag_total:
        print("flags: " + ", ".join(f"{k}={v:,}" for k, v in flag_total.most_common()))
    if unknown_total:
        print("UNKNOWN TYPES (format drift - investigate before trusting the histogram):")
        for (section, tname), n in unknown_total.most_common(20):
            print(f"  {section}: {tname} x{n:,}")
    print(f"\nwrote {args.db}\nnext: python stage0_report.py --db {args.db}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
