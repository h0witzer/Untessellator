/**
 * load3mf.mjs — minimal 3MF reader for the headless harness.
 *
 * 3MF is a ZIP containing 3D/3dmodel.model (XML). This reads the central
 * directory, inflates the model part, and returns a flat non-indexed position
 * array, applying each build item's transform.
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

/**
 * @param {string} path  .3mf file
 * @returns {{positions: Float64Array, objects: number}}
 */
export function load3MF(path) {
  const xml = unzipEntry(readFileSync(path), '3D/3dmodel.model').toString('utf8');

  // Collect every <object> with a <mesh>.
  const objects = new Map();
  const objRe = /<object\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/object>/g;
  let m;
  while ((m = objRe.exec(xml))) {
    const [, id, body] = m;
    const verts = [];
    const vRe = /<vertex\b[^>]*\bx="([^"]+)"[^>]*\by="([^"]+)"[^>]*\bz="([^"]+)"[^>]*\/>/g;
    let v;
    while ((v = vRe.exec(body))) verts.push([+v[1], +v[2], +v[3]]);

    const tris = [];
    const tRe = /<triangle\b[^>]*\bv1="(\d+)"[^>]*\bv2="(\d+)"[^>]*\bv3="(\d+)"[^>]*\/>/g;
    let t;
    while ((t = tRe.exec(body))) tris.push([+t[1], +t[2], +t[3]]);

    if (verts.length && tris.length) objects.set(id, { verts, tris });
  }
  if (!objects.size) throw new Error('no mesh objects in 3MF');

  // Build items reference objects, optionally with a 3x4 row-major transform.
  const items = [];
  const iRe = /<item\b[^>]*\bobjectid="([^"]+)"([^>]*)\/?>/gi;
  while ((m = iRe.exec(xml))) {
    const tm = /transform="([^"]+)"/i.exec(m[2]);
    items.push({ id: m[1], xf: tm ? tm[1].trim().split(/\s+/).map(Number) : null });
  }
  if (!items.length) for (const id of objects.keys()) items.push({ id, xf: null });

  const out = [];
  for (const { id, xf } of items) {
    const obj = objects.get(id);
    if (!obj) continue;
    for (const [a, b, c] of obj.tris) {
      for (const vi of [a, b, c]) {
        const p = obj.verts[vi];
        if (!p) continue;
        const q = xf ? mul(xf, p[0], p[1], p[2]) : p;
        out.push(q[0], q[1], q[2]);
      }
    }
  }
  return { positions: Float64Array.from(out), objects: objects.size };
}
