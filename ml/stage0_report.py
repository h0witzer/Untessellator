#!/usr/bin/env python3
"""
stage0_report.py - read the Stage 0 index and answer the questions that decide Stage 1.

Produces four things:

  1. The measured surface-type distribution. The ABC paper shows this only as a figure
     (Figure 3) with no numbers, so it has to be measured. Reported two ways, because
     they say different things:
       - share of PATCHES  : how much of the corpus, by count, is each type
       - share of MODELS   : how many models contain at least one of each type
     The second is what stratified sampling needs. A type can be 2% of patches and
     still appear in 40% of models, which makes it cheap to sample for.

  2. Corpus hygiene. The ABC paper is explicit that the dataset contains broken
     boundaries, duplicate models, and "models that are just made of single primitives
     such as a plane". This reports how many are lost to each filter.

  3. Rare-class exposure. Torus/Revolution/Extrusion/BSpline are the classes the whole
     experiment exists to cover, and the ones most likely to be too rare to learn.

  4. A stratified sampling plan, emitted as a manifest for Stage 1. Selection is
     greedy by rarity weight, so models covering several rare types are taken first
     and the tessellation budget is not spent on the ten-thousandth plane-and-cylinder
     bracket.

Usage:
    python stage0_report.py --db abc_index.sqlite
    python stage0_report.py --db abc_index.sqlite --target 25000 --min-per-type 2000 \
        --emit-manifest stage1_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from stage0_index import SURFACE_TYPES, CURVE_TYPES

# Flags that disqualify a part from training data. Kept here rather than in the
# indexer so the index stays a faithful record and the policy stays visible.
REJECT_FLAGS = (
    "parse_error", "read_error", "no_sections", "no_surfaces",
    "single_primitive", "ambiguous_sections", "unknown_type",
)

# The classes this experiment exists to cover, which the published benchmarks drop.
RARE_TYPES = ("Torus", "Revolution", "Extrusion", "BSpline")

# Types not worth sampling FOR, though models containing them are still fine to use.
#
# `Other` in ABC is usually a nontrivial fillet out of Onshape, but it is a catch-all
# and can be almost anything, so each distinct element under it would be a separate
# classification fight. Chasing it to quota costs ~43 chunks against ~19 for
# Revolution - more than double the corpus for the least useful class.
#
# The decision is to defer it, and it is a deferral rather than an exclusion: `Other`
# patches occur inside models we train on regardless, so they cannot be avoided, only
# left untargeted. The model therefore needs sane FALLBACK behaviour for them rather
# than a prediction - which the converter already has, in the NURBS fallback path that
# surfaceFitter.js uses when no primitive fits. Treat `Other` as "route to fallback",
# not as a class, and revisit these edge cases after something useful ships.
NO_QUOTA_DEFAULT = "Other"


def bar(frac: float, width: int = 28) -> str:
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def fmt_pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):5.2f}%" if total else "    - "


def section(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--target", type=int, default=25000,
                    help="models to select for Stage 1 tessellation (default 25000)")
    ap.add_argument("--min-per-type", type=int, default=2000,
                    help="models containing each surface type to try to reach")
    ap.add_argument("--min-patches", type=int, default=4,
                    help="reject parts with fewer patches than this")
    ap.add_argument("--no-quota", default=NO_QUOTA_DEFAULT,
                    help=f"comma-separated types to not sample FOR, though models "
                         f"containing them are still used (default: {NO_QUOTA_DEFAULT}; "
                         f"pass '' to target every type)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="do not collapse parts sharing a structural fingerprint")
    ap.add_argument("--emit-manifest", type=Path,
                    help="write the selected models as JSON Lines for Stage 1")
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"error: no such index: {args.db}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    total_parts = conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
    if not total_parts:
        print("error: index is empty. Run stage0_index.py first.", file=sys.stderr)
        return 2
    total_models = conn.execute("SELECT COUNT(DISTINCT model_id) FROM parts").fetchone()[0]

    meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM scan_meta")}

    section("Corpus")
    print(f"  indexed parts       {total_parts:,}")
    print(f"  distinct models     {total_models:,}")
    print(f"  scanned from        {meta.get('last_scan_root', '?')}")
    print(f"  scanned at          {meta.get('last_scan_utc', '?')}")
    roles = json.loads(meta.get("section_roles", "{}"))
    print(f"  detected sections   {', '.join(f'{k}->{v}' for k, v in sorted(roles.items())) or '(none)'}")

    unknown = conn.execute("SELECT section, type_name, n FROM unknown_types ORDER BY n DESC").fetchall()
    if unknown:
        print("\n  WARNING - type names outside the documented taxonomy were found.")
        print("  The histogram below is not trustworthy until these are explained:")
        for r in unknown:
            print(f"    {r['section']}: {r['type_name']} x{r['n']:,}")

    # ---------------------------------------------------------------- distribution
    section("Surface type distribution")
    sums = conn.execute(
        "SELECT " + ",".join(f"SUM(s_{t.lower()}) AS s_{t.lower()}" for t in SURFACE_TYPES) + " FROM parts"
    ).fetchone()
    contains = {
        t: conn.execute(f"SELECT COUNT(DISTINCT model_id) FROM parts WHERE s_{t.lower()} > 0").fetchone()[0]
        for t in SURFACE_TYPES
    }
    total_patches = sum(sums[f"s_{t.lower()}"] or 0 for t in SURFACE_TYPES)

    print(f"  {'type':<12} {'patches':>12} {'of all':>8}  {'models w/ >=1':>13} {'of models':>9}")
    print(f"  {'-' * 12} {'-' * 12:>12} {'-' * 8:>8}  {'-' * 13:>13} {'-' * 9:>9}")
    for t in SURFACE_TYPES:
        n = sums[f"s_{t.lower()}"] or 0
        share = (n / total_patches) if total_patches else 0.0
        print(f"  {t:<12} {n:>12,} {fmt_pct(n, total_patches):>8}  "
              f"{contains[t]:>13,} {fmt_pct(contains[t], total_models):>9}  {bar(share)}")
    print(f"  {'TOTAL':<12} {total_patches:>12,}")

    print("\n  Read the two percentage columns against each other. A type that is a small")
    print("  share of patches but present in many models is cheap to stratify for; one")
    print("  that is rare in both is the real risk to this experiment.")

    # ---------------------------------------------------------------- hygiene
    section("Hygiene")
    flag_counts: dict[str, int] = {}
    for (flags,) in conn.execute("SELECT flags FROM parts WHERE flags != ''"):
        for f in flags.split(","):
            flag_counts[f] = flag_counts.get(f, 0) + 1
    if flag_counts:
        for f, n in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
            marker = "  (rejected)" if f in REJECT_FLAGS else ""
            print(f"  {f:<22} {n:>10,} {fmt_pct(n, total_parts)}{marker}")
    else:
        print("  no flags set")

    reject_sql = " OR ".join(f"flags LIKE '%{f}%'" for f in REJECT_FLAGS)
    n_reject = conn.execute(f"SELECT COUNT(*) FROM parts WHERE {reject_sql}").fetchone()[0]
    n_small = conn.execute(
        f"SELECT COUNT(*) FROM parts WHERE NOT ({reject_sql}) AND n_patches < ?", (args.min_patches,)
    ).fetchone()[0]
    dup_groups = conn.execute(
        f"SELECT COUNT(*) FROM (SELECT struct_hash FROM parts WHERE NOT ({reject_sql}) "
        f"GROUP BY struct_hash HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    dup_extra = conn.execute(
        f"SELECT COALESCE(SUM(c - 1), 0) FROM (SELECT COUNT(*) AS c FROM parts "
        f"WHERE NOT ({reject_sql}) GROUP BY struct_hash HAVING c > 1)"
    ).fetchone()[0]

    print(f"\n  rejected by flag       {n_reject:>10,} {fmt_pct(n_reject, total_parts)}")
    print(f"  below --min-patches    {n_small:>10,} {fmt_pct(n_small, total_parts)}")
    print(f"  duplicate fingerprints {dup_extra:>10,} {fmt_pct(dup_extra, total_parts)}"
          f"  (in {dup_groups:,} groups)")
    print("\n  Structural fingerprint only: identical patch and curve histograms. It cannot")
    print("  see geometry, so it under-counts near-duplicates. Geometric dedup is Stage 1.")

    # ---------------------------------------------------------------- candidates
    where = f"NOT ({reject_sql}) AND n_patches >= ?"
    rows = conn.execute(
        "SELECT model_id, part, rel_path, n_patches, n_curves, n_sharp, struct_hash, "
        + ",".join(f"s_{t.lower()}" for t in SURFACE_TYPES)
        + f" FROM parts WHERE {where}",
        (args.min_patches,),
    ).fetchall()

    if not args.keep_duplicates:
        seen: set[str] = set()
        deduped = []
        for r in rows:
            if r["struct_hash"] in seen:
                continue
            seen.add(r["struct_hash"])
            deduped.append(r)
        rows = deduped

    section("Candidate pool")
    print(f"  usable parts after filters   {len(rows):,} "
          f"({fmt_pct(len(rows), total_parts)} of indexed)")
    if not rows:
        print("\n  Nothing survives the filters. Loosen --min-patches or check the flags above.")
        return 1

    no_quota = {t.strip() for t in args.no_quota.split(",") if t.strip()}
    pool_contains = {t: sum(1 for r in rows if r[f"s_{t.lower()}"] > 0) for t in SURFACE_TYPES}
    print(f"\n  {'type':<12} {'available':>12}  {'vs --min-per-type':>18}")
    for t in SURFACE_TYPES:
        have = pool_contains[t]
        if t in no_quota:
            verdict = "not targeted"
        elif have >= args.min_per_type:
            verdict = "ok"
        else:
            verdict = f"SHORT by {args.min_per_type - have:,}"
        star = " *" if t in RARE_TYPES else "  "
        print(f"  {t:<12}{star}{have:>11,}  {verdict:>18}")
    print("\n  * = a class the published benchmarks (ParSeNet, HPNet) do not support at all.")
    if no_quota:
        print(f"  not targeted: {', '.join(sorted(no_quota))} - models containing these are")
        print("  still used; they are just not sampled for. See NO_QUOTA_DEFAULT for why.")

    # ---------------------------------------------------------------- selection
    section("Stratified selection")
    weight = {t: 1.0 / max(1, pool_contains[t]) for t in SURFACE_TYPES}

    def score(r) -> float:
        return sum(weight[t] for t in SURFACE_TYPES if r[f"s_{t.lower()}"] > 0)

    ranked = sorted(rows, key=score, reverse=True)
    quota = {t: 0 if t in no_quota else min(args.min_per_type, pool_contains[t])
             for t in SURFACE_TYPES}
    have = {t: 0 for t in SURFACE_TYPES}
    selected: list = []
    taken: set[tuple[str, str]] = set()

    # Pass 1: satisfy per-type quotas, rarest-covering models first.
    for r in ranked:
        if len(selected) >= args.target:
            break
        if all(have[t] >= quota[t] for t in SURFACE_TYPES):
            break
        if not any(r[f"s_{t.lower()}"] > 0 and have[t] < quota[t] for t in SURFACE_TYPES):
            continue
        selected.append(r)
        taken.add((r["model_id"], r["part"]))
        for t in SURFACE_TYPES:
            if r[f"s_{t.lower()}"] > 0:
                have[t] += 1

    # Pass 2: top up to --target with the remaining pool, still rarity-ordered.
    for r in ranked:
        if len(selected) >= args.target:
            break
        if (r["model_id"], r["part"]) in taken:
            continue
        selected.append(r)
        taken.add((r["model_id"], r["part"]))
        for t in SURFACE_TYPES:
            if r[f"s_{t.lower()}"] > 0:
                have[t] += 1

    print(f"  selected {len(selected):,} parts (target {args.target:,})")
    print(f"\n  {'type':<12} {'models':>10} {'quota':>8} {'patches':>14}")
    sel_patches = {t: sum(r[f"s_{t.lower()}"] for r in selected) for t in SURFACE_TYPES}
    sel_total = sum(sel_patches.values())
    for t in SURFACE_TYPES:
        met = "-" if t in no_quota else ("ok" if have[t] >= quota[t] else "SHORT")
        print(f"  {t:<12} {have[t]:>10,} {met:>8} {sel_patches[t]:>14,} "
              f"{fmt_pct(sel_patches[t], sel_total)}")

    print("\n  Per-class loss weighting should be derived from the PATCH column above, not")
    print("  from the corpus-wide histogram - the selection has already rebalanced it.")
    shortfalls = [t for t in SURFACE_TYPES if t not in no_quota and have[t] < quota[t]]
    if shortfalls:
        print(f"\n  Under quota: {', '.join(shortfalls)}. Index more chunks, or lower")
        print("  --min-per-type and accept weaker supervision on those classes.")

    if args.emit_manifest:
        with args.emit_manifest.open("w", encoding="utf-8") as fh:
            for r in selected:
                fh.write(json.dumps({
                    "model_id": r["model_id"],
                    "part": r["part"],
                    "feat_path": r["rel_path"],
                    "n_patches": r["n_patches"],
                    "n_curves": r["n_curves"],
                    "n_sharp": r["n_sharp"],
                    "surfaces": {t: r[f"s_{t.lower()}"] for t in SURFACE_TYPES
                                 if r[f"s_{t.lower()}"] > 0},
                }, separators=(",", ":")) + "\n")
        print(f"\n  wrote manifest: {args.emit_manifest} ({len(selected):,} lines)")
        print("  Stage 1 reads this to decide which STEP files to tessellate.")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
