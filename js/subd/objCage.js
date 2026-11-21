/**
 * objCage.js — a Catmull-Clark control cage from an OBJ file (Blender, Rhino and Plasticity export their SubD
 * cages this way): vertices and polygon faces, read as written, without triangulating. Returns
 * { vertices, faces } when every face is a quad, else { error } naming why it is not a cage this converter takes.
 */
export function parseQuadCage(text) {
  const vertices = [], faces = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.startsWith('v ')) {
      const [x, y, z] = line.slice(2).trim().split(/\s+/).map(Number);
      vertices.push([x, y, z]);
    } else if (line.startsWith('f ')) {
      const ids = line.slice(2).trim().split(/\s+/).map((tok) => {
        const i = parseInt(tok.split('/')[0], 10);
        return i < 0 ? vertices.length + i : i - 1;     // OBJ indices are 1-based; negative counts back
      });
      faces.push(ids);
    }
  }
  if (!faces.length) return { error: 'no faces' };
  const notQuad = faces.findIndex((f) => f.length !== 4);
  if (notQuad >= 0) return { error: `face ${notQuad + 1} has ${faces[notQuad].length} corners (only quads are supported yet)` };
  return { vertices, faces };
}
