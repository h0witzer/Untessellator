#!/usr/bin/env python3
"""
fetch_abc.py - download and verify ABC dataset chunks.

ABC ships as 100 chunks of 10,000 models per format, listed in per-format manifests
(`feat_v00.txt`, `step_v00.txt`, ...) that pair a bitstream URL with a filename. The
published `size.yml` and `md5.yml` give the expected size and checksum of every
archive, so a truncated or corrupted download is detectable rather than something you
discover 40 GB into an extraction.

Sizes are the thing to plan around (per chunk, compressed):

    stl2  6.59 GB      feat  3.62 GB      step  1.07 GB
    obj   6.36 GB      para  1.97 GB      stat/meta  < 2 MB

Stage 0 needs only `feat`. Stage 1 adds `step`. Nothing else is required, and the two
biggest formats (`stl2`, `obj`) are the two we do not want - we re-tessellate from
STEP so that mesh density is a controlled variable rather than a fixed property of
the download.

Usage:
    python fetch_abc.py --format feat --chunks 0-2 --out D:/abc-data/archives
    python fetch_abc.py --format step --chunks 0-2 --out D:/abc-data/archives --verify md5
    python fetch_abc.py --format feat --chunks 0-99 --plan-only     # just the budget

Downloads resume, so interrupting is safe; re-running skips archives that already
match their published size.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = "https://deep-geometry.github.io/abc-dataset/data"
HERE = Path(__file__).resolve().parent


def load_yaml_map(path: Path) -> dict[str, str]:
    """These two files are flat `key: value` maps; a YAML parser would be overkill."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([\w.\-]+):\s*(\S+)\s*$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def ensure_aux(name: str, out_dir: Path) -> Path:
    """Fetch a manifest/checksum file next to this script if it is not already there."""
    local = HERE / name
    if local.exists():
        return local
    dest = out_dir / name
    if not dest.exists():
        print(f"fetching {name} ...")
        subprocess.run(["curl", "-sL", f"{BASE}/{name}", "-o", str(dest)], check=True)
    return dest


def parse_chunks(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", default="feat",
                    choices=["feat", "step", "obj", "stl2", "meta", "stat", "para", "ofs"])
    ap.add_argument("--chunks", default="0", help="e.g. 0, 0-2, or 0,5,7")
    ap.add_argument("--out", type=Path, default=Path("abc-archives"))
    ap.add_argument("--verify", choices=["size", "md5"], default="size",
                    help="size is free; md5 re-reads every byte (default: size)")
    ap.add_argument("--plan-only", action="store_true", help="print the budget and exit")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = ensure_aux(f"{args.format}_v00.txt", args.out)
    sizes = load_yaml_map(ensure_aux("size.yml", args.out))
    md5s = load_yaml_map(ensure_aux("md5.yml", args.out))

    entries: list[tuple[str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            entries.append((parts[0], parts[1]))

    wanted = parse_chunks(args.chunks)
    todo = [(url, name) for i, (url, name) in enumerate(entries) if i in wanted]
    if not todo:
        print(f"error: no chunks matched {args.chunks} (manifest has {len(entries)})", file=sys.stderr)
        return 2

    total = sum(int(sizes.get(name, 0)) for _, name in todo)
    print(f"format={args.format}  chunks={len(todo)}  models~{len(todo) * 10000:,}")
    print(f"download budget: {human(total)}")
    print(f"destination:     {args.out.resolve()}")
    if args.plan_only:
        for _, name in todo:
            print(f"  {name:28} {human(int(sizes.get(name, 0)))}")
        return 0

    t_start = time.time()
    got = 0
    for idx, (url, name) in enumerate(todo, 1):
        dest = args.out / name
        expect = int(sizes.get(name, 0))
        if dest.exists() and expect and dest.stat().st_size == expect:
            print(f"[{idx}/{len(todo)}] {name}: already complete ({human(expect)})")
            got += expect
            continue

        print(f"[{idx}/{len(todo)}] {name}: downloading {human(expect)} ...", flush=True)
        t0 = time.time()
        # -C - resumes a partial file; --retry rides out transient archive.nyu.edu 5xx.
        rc = subprocess.run(
            ["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5",
             "--retry-all-errors", "-o", str(dest), url]
        ).returncode
        dt = time.time() - t0
        if rc != 0:
            print(f"  curl exited {rc} - re-run to resume", file=sys.stderr)
            return 1

        actual = dest.stat().st_size
        if expect and actual != expect:
            print(f"  SIZE MISMATCH: got {actual:,}, expected {expect:,}", file=sys.stderr)
            print("  re-run to resume; delete the file to start over", file=sys.stderr)
            return 1
        if args.verify == "md5" and name in md5s:
            print("  verifying md5 ...", flush=True)
            digest = md5_of(dest)
            if digest != md5s[name]:
                print(f"  MD5 MISMATCH: {digest} != {md5s[name]}", file=sys.stderr)
                return 1
            print("  md5 ok")
        got += actual
        rate = actual / max(dt, 1e-6)
        print(f"  done in {dt / 60:.1f} min ({rate / 1e6:.1f} MB/s)")

    elapsed = time.time() - t_start
    print(f"\n{human(got)} in {elapsed / 60:.1f} min "
          f"({got / max(elapsed, 1e-6) / 1e6:.1f} MB/s average)")
    print(f"\nextract with:\n  7z x \"{args.out}/<archive>.7z\" -o<dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
