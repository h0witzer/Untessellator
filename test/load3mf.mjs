/**
 * load3mf.mjs — minimal 3MF reader for the headless harness.
 *
 * 3MF is a ZIP containing 3D/3dmodel.model (XML). This reads the central
 * directory, inflates the model part (and any 3D/Objects/*.model parts its
 * components point into), and returns a flat non-indexed position array,
 * applying each build item's and component's transform.
 *
 * Node-only (uses zlib); the browser path still goes through stlLoader.js.
 */

import { readFileSync } from 'node:fs';
import { inflateRawSync } from 'node:zlib';

const U32_MAX = 0xFFFFFFFF;

/**
 * Pull a ZIP64 extended-information extra field, which supplies any size or
 * offset that overflowed its 32-bit slot. Onshape's 3MF export writes a ZIP64
 * record even for small files, so this is not an edge case here.
 */
function zip64Extra(buf, start, len, need) {
  let p = start;
  const end = start + len;
  while (p + 4 <= end) {
    const id = buf.readUInt16LE(p);
    const size = buf.readUInt16LE(p + 2);
    if (id === 0x0001) {
      let q = p + 4;
      const out = {};
      for (const field of need) {
        if (q + 8 > p + 4 + size) break;
        out[field] = Number(buf.readBigUInt64LE(q));
        q += 8;
      }
      return out;
    }
    p += 4 + size;
  }
  return {};
}

/** Extract one named entry from a ZIP buffer (ZIP64 aware). */
function unzipEntry(buf, wantName) {
  // End of central directory: signature 0x06054b50, scan backwards.
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0 && i > buf.length - 66000; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('not a zip (no EOCD)');

  let count = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);

  // A sentinel value means the real figures live in the ZIP64 EOCD record.
  if (p === U32_MAX || count === 0xFFFF) {
    let z = -1;
    for (let i = eocd - 20; i >= 0; i--) {
      if (buf.readUInt32LE(i) === 0x06064b50) { z = i; break; }
    }
    if (z < 0) throw new Error('zip64 EOCD not found');
    count = Number(buf.readBigUInt64LE(z + 32));
    p = Number(buf.readBigUInt64LE(z + 48));
  }

  for (let i = 0; i < count; i++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error('bad central directory');
    const method = buf.readUInt16LE(p + 10);
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const cmtLen = buf.readUInt16LE(p + 32);
    const name = buf.toString('utf8', p + 46, p + 46 + nameLen);

    let compSize = buf.readUInt32LE(p + 20);
    let localOff = buf.readUInt32LE(p + 42);

    if (compSize === U32_MAX || localOff === U32_MAX) {
      // Fields appear in a fixed order, but only those that overflowed.
      const need = [];
      if (buf.readUInt32LE(p + 24) === U32_MAX) need.push('uncompressed');
      if (compSize === U32_MAX) need.push('compressed');
      if (localOff === U32_MAX) need.push('localOff');
      const z64 = zip64Extra(buf, p + 46 + nameLen, extraLen, need);
      if (z64.compressed !== undefined) compSize = z64.compressed;
      if (z64.localOff !== undefined) localOff = z64.localOff;
    }

    if (name.toLowerCase() === wantName.toLowerCase()) {
      const lnLen = buf.readUInt16LE(localOff + 26);
      const leLen = buf.readUInt16LE(localOff + 28);
      const start = localOff + 30 + lnLen + leLen;
      const raw = buf.subarray(start, start + compSize);
      return method === 0 ? raw : inflateRawSync(raw);
    }
    p += 46 + nameLen + extraLen + cmtLen;
  }
  throw new Error(`entry not found: ${wantName}`);
}

const mul = (m, x, y, z) => [
  m[0] * x + m[3] * y + m[6] * z + m[9],
  m[1] * x + m[4] * y + m[7] * z + m[10],
  m[2] * x + m[5] * y + m[8] * z + m[11],
];

const transformOf = (attrs) => {
  const tm = /\btransform="([^"]+)"/i.exec(attrs);
  return tm ? tm[1].trim().split(/\s+/).map(Number) : null;
};

/** Every <object> in one model part: its inline mesh, if any, and its components, if any. */
function parseObjects(xml) {
  const objects = new Map();
  const objRe = /<object\b([^>]*)>([\s\S]*?)<\/object>/g;
  let m;
  while ((m = objRe.exec(xml))) {
    const id = /\bid="([^"]+)"/.exec(m[1])?.[1];
    if (id === undefined) continue;
    const body = m[2];
    const verts = [];
    const vRe = /<vertex\b[^>]*\bx="([^"]+)"[^>]*\by="([^"]+)"[^>]*\bz="([^"]+)"[^>]*\/>/g;
    let v;
    while ((v = vRe.exec(body))) verts.push([+v[1], +v[2], +v[3]]);

    const tris = [];
    const tRe = /<triangle\b[^>]*\bv1="(\d+)"[^>]*\bv2="(\d+)"[^>]*\bv3="(\d+)"[^>]*\/>/g;
    let t;
    while ((t = tRe.exec(body))) tris.push([+t[1], +t[2], +t[3]]);

    // Bambu Studio and PrusaSlicer keep the mesh in 3D/Objects/*.model and reference it from a
    // component (production extension, p:path).
    const comps = [];
    const cRe = /<component\b([^>]*?)\/?>/g;
    let c;
    while ((c = cRe.exec(body))) {
      const oid = /\bobjectid="([^"]+)"/.exec(c[1])?.[1];
      if (oid !== undefined) comps.push({ id: oid, path: /:path="([^"]+)"/.exec(c[1])?.[1] ?? null, xf: transformOf(c[1]) });
    }
    objects.set(id, { verts, tris, comps });
  }
  return objects;
}

/**
 * @param {string} path  .3mf file
 * @param {{maxBytes?: number}} [opts]  refuse any model part larger than this when inflated
 * @returns {{positions: Float64Array, objects: number}}
 */
export function load3MF(path, { maxBytes = Infinity } = {}) {
  const buf = readFileSync(path);
  const partXml = (name) => {
    const raw = unzipEntry(buf, name);
    if (raw.length > maxBytes) throw new Error(`model part ${name} is ${(raw.length / 1e6).toFixed(0)} MB`);
    return raw.toString('utf8');
  };
  const xml = partXml('3D/3dmodel.model');
  const main = parseObjects(xml);
  const parts = new Map();
  const partObjects = (p) => {
    const key = p.replace(/^\//, '');
    if (!parts.has(key)) parts.set(key, parseObjects(partXml(key)));
    return parts.get(key);
  };

  // Build items reference objects, optionally with a 3x4 row-major transform.
  const items = [];
  const iRe = /<item\b([^>]*?)\/?>/gi;
  let m;
  while ((m = iRe.exec(xml))) {
    const id = /\bobjectid="([^"]+)"/.exec(m[1])?.[1];
    if (id !== undefined) items.push({ id, xf: transformOf(m[1]) });
  }
  if (!items.length) for (const id of main.keys()) items.push({ id, xf: null });

  // Emit an object's own triangles, then its components'. Transforms apply innermost first.
  const out = [];
  let meshes = 0;
  const emit = (objects, id, xfs, depth) => {
    const obj = objects.get(id);
    if (!obj || depth > 8) return;
    if (obj.tris.length) meshes++;
    for (const tri of obj.tris) {
      const ps = tri.map((vi) => obj.verts[vi]);
      if (ps.some((p) => !p)) continue;
      for (let p of ps) {
        for (const xf of xfs) if (xf) p = mul(xf, p[0], p[1], p[2]);
        out.push(p[0], p[1], p[2]);
      }
    }
    for (const comp of obj.comps) emit(comp.path ? partObjects(comp.path) : objects, comp.id, [comp.xf, ...xfs], depth + 1);
  };
  for (const { id, xf } of items) emit(main, id, [xf], 0);
  if (!out.length) throw new Error('no mesh objects in 3MF');
  return { positions: Float64Array.from(out), objects: meshes };
}
