#!/usr/bin/env python3
"""
gallery_build.py - the verified-results gallery: every part whose exported STEP matches the original CAD
(step_match.py), written as data for the static gallery site.

    python gallery_build.py          (the site is the Untessellator-Gallery checkout beside this repo)

Two corpora:

  one-shot  test/studioExport.mjs output, the studio's Auto-paint then Export STEP with default settings and
            no hand edits, for the meshes in --oneshot-in. The original of a part named <model>_<idx> is
            <originals>/<model>/<model>_<hash>_step_<idx>.step (ABC's layout).
  painted   --painted <dir>/<name>/ holds:
              export.step     the studio's Export STEP
              original.step   the CAD the mesh came from
              report.zip      the studio's report (flag button, "+ picture + mesh"): its part.stl is the input,
                              and its groups say which ones were painted by hand
              input.stl       the mesh, if report.zip has none
              meta.json       optional: {"title", "credit", "link", "license"}

Writes into the site:
  data/gallery.json       the criteria, a tally per corpus, and one row per matching part
  parts/<id>/view.bin     our STEP's faces (triangulated per face) and edges, for the viewer
  parts/<id>/<id>.zip     input.stl, export.step, original.step, match.json
  parts/<id>/thumb.png    drawn by the site's own viewer in headless Chrome (--no-thumbs skips)

and into --work:
  match/<id>.json         step_match's report per part, reused while neither STEP changes
  misses.json             the one-shot parts that did not match, closest first: the queue for painting by hand
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

KINDS = ['plane', 'cylinder', 'cone', 'sphere', 'torus', 'bspline', 'revolution', 'extrusion', 'other']
CHROME = [Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),
          Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')]


# ---------------------------------------------------------------------------------------------- matching

def run_match(job, tol, work):
    """step_match on one (id, ours, original), cached in <work>/match/<id>.json."""
    from step_match import VERSION, match
    pid, ours, original = job
    cache = work / 'match' / f'{pid}.json'
    newest = max(Path(ours).stat().st_mtime, Path(original).stat().st_mtime)
    if cache.exists() and cache.stat().st_mtime > newest:
        rep = json.loads(cache.read_text())
        if rep.get('tol_mm') == tol and rep.get('version') == VERSION:
            return pid, rep
    try:
        rep = match(ours, original, tol)
    except Exception as e:                                  # an OCCT failure is a non-match, not a crash
        rep = {'match': False, 'reason': f'verifier error: {e}'}
    cache.write_text(json.dumps(rep))
    return pid, rep


# ---------------------------------------------------------------------------------------------- view.bin

def view_bin(step_path, path, pairs=()):
    """Our STEP for the viewer: 'SGV1', uint32 header length, JSON header (space-padded to 4 bytes), then
    float32 positions, uint32 triangle indices and float32 edge segments (two points each). Each face has
    its own vertices, so normals are smooth within a face and sharp across edges. Seam edges are left out.
    `pairs` (step_match's per-face rows) give the header's `info`: per face of ours, [distance to its
    partner in mm, the partner's size parameters]."""
    import numpy as np
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from step_match import KIND, read_step, shape_map
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    shape = read_step(step_path)
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    diag = math.dist((x0, y0, z0), (x1, y1, z1))
    BRepMesh_IncrementalMesh(shape, 1e-3 * diag, False, 0.2, True)

    pos, idx, faces = [], [], []
    nv = nt = 0
    for f in shape_map(shape, TopAbs_FACE):
        f = TopoDS.Face_s(f)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        kind = KIND.get(GeomAdaptor_Surface(BRep_Tool.Surface_s(f)).GetType(), 'other')
        if tri is None:
            faces.append([KINDS.index(kind), nv, 0, nt, 0])
            continue
        tr = loc.Transformation()
        p = np.array([[q.X(), q.Y(), q.Z()] for q in
                      (tri.Node(i).Transformed(tr) for i in range(1, tri.NbNodes() + 1))])
        t = np.array([[tri.Triangle(i).Value(k) for k in (1, 2, 3)] for i in range(1, tri.NbTriangles() + 1)]) - 1
        if f.Orientation() == TopAbs_REVERSED:
            t = t[:, [0, 2, 1]]
        faces.append([KINDS.index(kind), nv, len(p), nt, len(t)])
        pos.append(p)
        idx.append(t + nv)
        nv += len(p)
        nt += len(t)

    segs = []
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    for i in range(1, m.Extent() + 1):
        e = TopoDS.Edge_s(m.FindKey(i))
        if BRep_Tool.Degenerated_s(e) or m.FindFromIndex(i).Extent() == 0:
            continue
        f = TopoDS.Face_s(m.FindFromIndex(i).First())
        if BRep_Tool.IsClosed_s(e, f):
            continue
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        poly = BRep_Tool.PolygonOnTriangulation_s(e, tri, loc) if tri is not None else None
        if poly is None:
            continue
        tr = loc.Transformation()
        q = [tri.Node(poly.Node(k)).Transformed(tr) for k in range(1, poly.NbNodes() + 1)]
        q = np.array([[a.X(), a.Y(), a.Z()] for a in q])
        segs.append(np.concatenate([q[:-1], q[1:]], 1))

    P = np.concatenate(pos).astype('<f4') if pos else np.zeros((0, 3), '<f4')
    T = np.concatenate(idx).astype('<u4') if idx else np.zeros((0, 3), '<u4')
    S = np.concatenate(segs).astype('<f4') if segs else np.zeros((0, 6), '<f4')
    info = [None] * len(faces)
    for r in pairs:
        if 0 <= r.get('our_face', -1) < len(info):
            info[r['our_face']] = [float(f'{r["dev_mm"]:.3g}'), {k: round(v, 9) for k, v in r['truth'].items()}]
    head = json.dumps({'kinds': KINDS, 'faces': faces, 'info': info, 'nv': len(P), 'nt': len(T), 'ns': len(S),
                       'box': [x0, y0, z0, x1, y1, z1]}, separators=(',', ':')).encode()
    head += b' ' * (-len(head) % 4)
    with open(path, 'wb') as out:
        out.write(b'SGV1' + struct.pack('<I', len(head)) + head + P.tobytes() + T.tobytes() + S.tobytes())


# ---------------------------------------------------------------------------------------------- thumbnails

THUMBS_PAGE = """<!doctype html><meta charset="utf-8">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"}}</script>
<body style="margin:0;background:#fff"><script type="module">
import { thumbnail } from './js/view.js';
const ids = new URLSearchParams(location.search).get('ids').split(',');
for (const id of ids) {
  try {
    const blob = await thumbnail(`parts/${id}/view.bin`, 720, 540);
    await fetch(`/__thumb/${id}`, { method: 'POST', body: blob });
  } catch (e) { await fetch(`/__thumb/${id}`, { method: 'POST', body: 'error ' + e }); }
}
await fetch('/__done', { method: 'POST' });
</script>"""


def render_thumbs(site, ids, timeout=600):
    """Draws parts/<id>/thumb.png with the site's own viewer (js/view.js `thumbnail`) in headless Chrome,
    which posts each PNG back to this server."""
    chrome = next((c for c in CHROME if c.exists()), None)
    if chrome is None or not ids:
        print('  thumbnails skipped: no Chrome or Edge' if chrome is None else '')
        return
    done = threading.Event()
    got = []

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(site), **k)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith('/__thumbs.html'):
                body = THUMBS_PAGE.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            if self.path.startswith('/__thumb/'):
                pid = self.path.split('/')[-1]
                if data[:4] == b'\x89PNG':
                    (site / 'parts' / pid / 'thumb.png').write_bytes(data)
                    got.append(pid)
                else:
                    print(f'  thumbnail {pid}: {data[:200].decode(errors="replace")}')
            elif self.path == '/__done':
                done.set()
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = tempfile.mkdtemp(prefix='gallery-chrome-')
    url = f'http://127.0.0.1:{server.server_address[1]}/__thumbs.html?ids={",".join(ids)}'
    proc = subprocess.Popen([str(chrome), '--headless=new', f'--user-data-dir={profile}', '--no-first-run',
                             '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--window-size=720,540',
                             url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    done.wait(timeout)
    proc.kill()
    server.shutdown()
    shutil.rmtree(profile, ignore_errors=True)
    print(f'  thumbnails: {len(got)}/{len(ids)}')


# ---------------------------------------------------------------------------------------------- corpora

def stl_triangles(data: bytes):
    if len(data) >= 84 and 84 + 50 * struct.unpack_from('<I', data, 80)[0] == len(data):
        return struct.unpack_from('<I', data, 80)[0]
    return data.count(b'endfacet')


def oneshot_jobs(args):
    """(id, ours, original, input stl, extra row fields) for every one-shot export with a findable original."""
    jobs = []
    for stl in sorted(args.oneshot_in.glob('*.stl')):
        pid = stl.stem
        ours = args.oneshot_out / f'{pid}.step'
        model, _, part_idx = pid.partition('_')
        hits = list((args.originals / model).glob(f'{model}_*_step_{part_idx}.step'))
        if not hits:
            continue
        info = {}
        rj = args.oneshot_out / f'{pid}.json'
        if rj.exists():
            r = json.loads(rj.read_text())
            info = {'paint_secs': r.get('paint_secs'), 'knit_secs': r.get('knit_secs'), 'groups': r.get('groups')}
        jobs.append({'id': pid, 'ours': ours, 'original': hits[0], 'stl': stl,
                     'row': {'corpus': 'oneshot', 'title': f'ABC {model} · part {int(part_idx)}',
                             'credit': 'ABC dataset (Onshape public documents); copyright stays with the creator',
                             'link': 'https://deep-geometry.github.io/abc-dataset/',
                             'mesher': 'Onshape', **info}})
    return jobs


def painted_jobs(args):
    jobs = []
    if not args.painted or not args.painted.exists():
        return jobs
    waiting = 0
    for d in sorted(p for p in args.painted.iterdir() if p.is_dir()):
        ours, original = d / 'export.step', d / 'original.step'
        if not original.exists():
            print(f'  painted/{d.name}: needs original.step')
            continue
        if not ours.exists():
            waiting += 1
            continue
        meta = json.loads((d / 'meta.json').read_text()) if (d / 'meta.json').exists() else {}
        row = {'corpus': 'painted', 'title': meta.get('title', d.name), 'credit': meta.get('credit'),
               'link': meta.get('link'), 'license': meta.get('license'), 'mesher': meta.get('mesher')}
        stl_bytes = (d / 'input.stl').read_bytes() if (d / 'input.stl').exists() else None
        rz = d / 'report.zip'
        if rz.exists():
            with zipfile.ZipFile(rz) as z:
                rep = json.loads(z.read('report.json'))
                if stl_bytes is None and 'part.stl' in z.namelist():
                    stl_bytes = z.read('part.stl')
            groups = rep.get('groups', [])
            row['groups'] = len(groups)
            row['hand_groups'] = sum(1 for g in groups if (g.get('origin') or {}).get('by') == 'stroke')
            row['auto_groups'] = sum(1 for g in groups if (g.get('origin') or {}).get('by') == 'auto-paint')
        if stl_bytes is None:
            print(f'  painted/{d.name}: no mesh (input.stl, or report.zip with the mesh)')
            continue
        jobs.append({'id': 'p-' + ''.join(c if c.isalnum() or c in '-_' else '-' for c in d.name).lower(),
                     'ours': ours, 'original': original, 'stl_bytes': stl_bytes, 'row': row})
    if waiting:
        print(f'  painted: {waiting} folders waiting for export.step')
    return jobs


def row_for(job, rep):
    t, o = rep['truth'], rep['ours']
    stl = job.get('stl_bytes') or job['stl'].read_bytes()
    return {'id': job['id'], **{k: v for k, v in job['row'].items() if v is not None},
            'triangles': stl_triangles(stl), 'faces': t['faces'], 'types': t['types'],
            'edges': [t['edges'], o['edges']], 'vertices': [t['vertices'], o['vertices']],
            'topology_identical': rep['topology_identical'], 'max_dev_mm': rep['max_dev_mm'],
            'vol_rel_err': rep['volume_rel_err'], 'param_err': rep['max_param_err'],
            'diag_mm': rep['diag_mm'], 'volume_mm3': t['volume_mm3']}


def write_part(site, job, rep):
    """parts/<id>/view.bin and <id>.zip. Both are byte-for-byte the same for the same inputs, so a rebuild
    leaves unchanged parts unchanged in the site's git history."""
    d = site / 'parts' / job['id']
    d.mkdir(parents=True, exist_ok=True)
    view_bin(job['ours'], d / 'view.bin', rep.get('faces', ()))
    files = [('input.stl', job.get('stl_bytes') or job['stl'].read_bytes()),
             ('export.step', Path(job['ours']).read_bytes()),
             ('original.step', Path(job['original']).read_bytes()),
             ('match.json', json.dumps(rep, indent=1).encode())]
    with zipfile.ZipFile(d / f"{job['id']}.zip", 'w') as z:
        for name, data in files:
            z.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data,
                       compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return job['id']


def queue_misses(args, misses, n):
    """Tops the painting queue up to n folders: <painted>/<id>/ for the closest one-shot misses that have no
    folder yet, with input.stl, original.step, meta.json and NOTES.txt (what differs). Painting one means
    saving the studio's export.step and report.zip there."""
    args.painted.mkdir(parents=True, exist_ok=True)
    waiting = sum(1 for d in args.painted.iterdir() if d.is_dir() and not (d / 'export.step').exists())
    n -= waiting
    made = 0
    for m in misses:
        if made >= n:
            break
        d = args.painted / m['id']
        if d.exists():
            continue
        rep = json.loads((args.work / 'match' / f"{m['id']}.json").read_text())
        t, o = rep.get('truth', {}).get('types', {}), rep.get('ours', {}).get('types', {})
        rows = [f'  {k:10} original {t.get(k, 0):4}   one-shot {o.get(k, 0):4}' for k in KINDS if t.get(k) or o.get(k)]
        d.mkdir(parents=True)
        shutil.copyfile(m['stl'], d / 'input.stl')
        shutil.copyfile(m['original'], d / 'original.step')
        model, _, part_idx = m['id'].partition('_')
        (d / 'meta.json').write_text(json.dumps({
            'title': f'ABC {model} · part {int(part_idx)}', 'mesher': 'Onshape',
            'credit': 'ABC dataset (Onshape public documents); copyright stays with the creator',
            'link': 'https://deep-geometry.github.io/abc-dataset/'}, indent=1))
        (d / 'NOTES.txt').write_text(
            f"{m['id']}: the one-shot export missed because {rep.get('reason')}.\n\n" + '\n'.join(rows) +
            '\n\nTo paint it: open input.stl in the studio, press Auto-paint, repaint the faces that are wrong,\n'
            'then save Export STEP here as export.step, and the flag button\'s report ("+ picture + mesh") as\n'
            'report.zip. The next gallery build checks it against original.step.\n')
        made += 1
    return made


def miss_row(job, rep):
    t, o = rep.get('truth', {}), rep.get('ours', {})
    off = sum(abs(t.get('types', {}).get(k, 0) - o.get('types', {}).get(k, 0))
              for k in set(t.get('types', {})) | set(o.get('types', {})))
    return {'id': job['id'], 'stl': str(job.get('stl', '')), 'original': str(job['original']),
            'reason': rep.get('reason'), 'faces': [t.get('faces'), o.get('faces')], 'types_off': off,
            'max_dev_mm': rep.get('max_dev_mm')}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[2] / 'Untessellator-Gallery',
                    help='the gallery site checkout (default: Untessellator-Gallery beside this repo)')
    ap.add_argument('--work', type=Path, default=Path('D:/abc-data/lab/gallery'))
    ap.add_argument('--oneshot-in', type=Path, default=Path('D:/abc-data/lab/gallery/in'))
    ap.add_argument('--oneshot-out', type=Path, default=Path('D:/abc-data/lab/gallery/out'))
    ap.add_argument('--originals', type=Path, default=Path('D:/abc-data/step'))
    ap.add_argument('--painted', type=Path, default=Path('D:/abc-data/lab/gallery/painted'))
    ap.add_argument('--tol', type=float, default=1e-3,
                    help="largest face-to-face distance, mm (the originals' declared uncertainty)")
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--queue', type=int, default=10, help='set up this many closest misses for painting by hand')
    ap.add_argument('--no-thumbs', action='store_true')
    ap.add_argument('--thumbs-only', action='store_true', help="redraw every thumbnail of the site's current parts")
    args = ap.parse_args(argv)
    (args.work / 'match').mkdir(parents=True, exist_ok=True)
    if args.thumbs_only:
        render_thumbs(args.site, [r['id'] for r in json.loads((args.site / 'data/gallery.json').read_text())['parts']])
        return 0

    oneshot = [j for j in oneshot_jobs(args)]
    painted = painted_jobs(args)
    exported = [j for j in oneshot if j['ours'].exists()] + painted
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        reps = dict(pool.map(partial(run_match, tol=args.tol, work=args.work),
                             [(j['id'], str(j['ours']), str(j['original'])) for j in exported]))
    print(f'  matched {len(exported)} exports in {time.time() - t0:.0f} s')

    hits = [j for j in exported if reps[j['id']]['match']]
    keep = {j['id'] for j in hits}
    parts = args.site / 'parts'
    if parts.exists():
        for d in parts.iterdir():
            if d.name not in keep:
                shutil.rmtree(d)
    new = [j['id'] for j in hits if not (parts / j['id'] / 'thumb.png').exists()]
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(partial(write_part, args.site), hits, [reps[j['id']] for j in hits]))

    rows = [row_for(j, reps[j['id']]) for j in hits]
    for r in rows:
        r['zip_bytes'] = (parts / r['id'] / f"{r['id']}.zip").stat().st_size
    rows.sort(key=lambda r: (r['corpus'], -r['faces']))

    def why(rep):
        first = rep.get('reason', '').split(';')[0]
        return ('a face is more than the tolerance off' if first.startswith('a face is') else
                'volume differs' if first.startswith('volume') else
                'not the same solids' if first.startswith('export is not') else first or 'unknown')

    def tally(corpus, jobs):
        """A job with no report yet is pending (one-shot: studioExport hasn't written its <id>.json), or a
        miss with 'no export' when it ran and wrote no STEP."""
        misses, pending = {}, 0
        for j in jobs:
            rep = reps.get(j['id'])
            ran = rep is not None or corpus != 'oneshot' or Path(j['ours']).with_suffix('.json').exists()
            if not ran:
                pending += 1
                continue
            key = why(rep) if rep and not rep['match'] else None if rep else 'no export'
            if key:
                misses[key] = misses.get(key, 0) + 1
        return {'tried': len(jobs) - pending, 'pending': pending, 'exported': sum(1 for j in jobs if j['id'] in reps),
                'matched': sum(1 for j in jobs if reps.get(j['id'], {}).get('match')),
                'misses': dict(sorted(misses.items(), key=lambda kv: -kv[1]))}
    gallery = {
        'generated': time.strftime('%Y-%m-%d'),
        'criteria': {'tol_mm': args.tol},
        'corpora': {'oneshot': {**tally('oneshot', oneshot),
                                'set': 'ABC held-out Onshape exports: 300 parts that no model was trained or tuned on'},
                    'painted': tally('painted', painted)},
        'parts': rows,
    }
    (args.site / 'data').mkdir(parents=True, exist_ok=True)
    (args.site / 'data' / 'gallery.json').write_text(json.dumps(gallery, indent=1))

    # Closest first: wrong types are what painting fixes, and fewer wrong faces are quicker to fix.
    misses = [miss_row(j, reps[j['id']]) for j in oneshot if j['id'] in reps and not reps[j['id']]['match']]
    misses.sort(key=lambda m: (not (m['reason'] or '').startswith('face types differ'), m['types_off'],
                               abs((m['faces'][0] or 0) - (m['faces'][1] or 0)), m['faces'][0] or 0))
    (args.work / 'misses.json').write_text(json.dumps(misses, indent=1))
    queued = queue_misses(args, misses, args.queue) if args.queue else 0

    if not args.no_thumbs:
        render_thumbs(args.site, new)
    for c, v in gallery['corpora'].items():
        print(f"  {c:8} matched {v['matched']} of {v['tried']} ({v['exported']} exported)")
    if queued:
        print(f'  {queued} new folders to paint in {args.painted}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
