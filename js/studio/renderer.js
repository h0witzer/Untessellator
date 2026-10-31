/**
 * renderer.js — the studio's WebGL2 viewport, drawn with the Form Studio kit's pieces.
 *
 * Taken from the kit as it is: the product's lighting (PART_GLSL, a Phong headlight in eye
 * space), its ambient occlusion pass, its screen-space bands and their styles, the flat #333
 * ground, and the two-pass solid that paints an inverted patch red.
 *
 * Where this app departs from the kit, on purpose:
 *   - colour is PER VERTEX, so every face group is its own colour and unpainted triangles
 *     are a neutral grey rather than the kit's part blue (which would read as a group)
 *   - the mesh is FACETED by default: before fitting, the triangles are the data, and
 *     smoothing would hide exactly the tessellation a fill is judged on
 *   - the active group is marked with the product's amber face border only, never its amber
 *     face fill, which would cover the group's colour; the border's hidden-side copy is
 *     drawn without the stipple, because mesh edges come in pieces too short to dash
 *   - picking is a triangle-ID buffer, rendered only when the view has changed
 *   - a HEAT mode (setHeat) for the deviation view: a second per-corner value, a level, is
 *     interpolated across each triangle and mapped through the HEAT ramp per pixel, so a
 *     triangle whose corners differ shows the ramp between them rather than a blend of two
 *     colors; a corner at NO_LEVEL keeps its color
 */

import {
  compileProgram, BAND_VERTEX, BAND_FRAGMENT, drawBand, bandStyle, multiply4,
  setLightingUniforms, createSsao, occlusionFactor, PART_GLSL,
} from '../../vendor/form-studio-ui/index.js';

/**
 * The heat ramp: stop k is level FROM + k, held flat below the first and above the last. The levels are
 * log10 of a distance over the part diagonal, so each stop is a decade; studio.js labels them.
 */
export const HEAT = { from: -7, stops: ['#2c5fd6', '#22a6c9', '#3fbf5a', '#f2d43a', '#f0612e', '#b3141b'] };
/** A level that means "no value here": the corner keeps its color. */
export const NO_LEVEL = -100;

const glslColor = (hex) => `vec3(${[1, 3, 5].map((i) => (parseInt(hex.slice(i, i + 2), 16) / 255).toFixed(4)).join(', ')})`;
const STOPS = HEAT.stops.length;
const HEAT_GLSL = `
const vec3 HEAT[${STOPS}] = vec3[${STOPS}](${HEAT.stops.map(glslColor).join(', ')});
vec3 heat(float level) {
  float x = clamp(level - (${HEAT.from.toFixed(1)}), 0.0, ${(STOPS - 1).toFixed(1)});
  int i = int(min(floor(x), ${(STOPS - 2).toFixed(1)}));
  return mix(HEAT[i], HEAT[i + 1], x - float(i));
}`;

const SURFACE_VERTEX = `#version 300 es
in vec3 position; in vec3 normal; in vec3 colour; in float level;
uniform mat4 uProjection; uniform mat4 uView;
out vec3 vPositionEye; out vec3 vNormalEye; out vec3 vColour; out float vLevel;
void main() {
  vec4 positionEye = uView * vec4(position, 1.0);
  vPositionEye = positionEye.xyz;
  vNormalEye = mat3(uView) * normal;
  vColour = colour;
  vLevel = level;
  gl_Position = uProjection * positionEye;
}`;

const SURFACE_FRAGMENT = `#version 300 es
precision highp float;
in vec3 vPositionEye; in vec3 vNormalEye; in vec3 vColour; in float vLevel;
uniform bool uHeat;
out vec4 color;
${PART_GLSL}
${HEAT_GLSL}
void main() {
  if (!gl_FrontFacing) { color = vec4(0.45, 0.08, 0.10, 1.0); return; }
  vec3 base = uHeat && vLevel > ${(NO_LEVEL / 2).toFixed(1)} ? heat(vLevel) : vColour;
  color = vec4(onshapeLight(base, vPositionEye, vNormalEye, 0.0), 1.0);
}`;

// Non-indexed geometry, so a vertex's triangle is gl_VertexID / 3 and no id attribute is needed.
const PICK_VERTEX = `#version 300 es
in vec3 position;
uniform mat4 uMvp;
flat out int vTriangle;
void main() { vTriangle = gl_VertexID / 3; gl_Position = uMvp * vec4(position, 1.0); }`;

const PICK_FRAGMENT = `#version 300 es
precision highp float; precision highp int;
flat in int vTriangle;
out vec4 color;
void main() {
  int id = vTriangle + 1;
  color = vec4(float(id & 255), float((id >> 8) & 255), float((id >> 16) & 255), 255.0) / 255.0;
}`;

export function createRenderer(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: true });
  if (!gl) throw new Error('WebGL2 is not available in this browser');

  const surfaceProgram = compileProgram(gl, SURFACE_VERTEX, SURFACE_FRAGMENT);
  const bandProgram = compileProgram(gl, BAND_VERTEX, BAND_FRAGMENT);
  const pickProgram = compileProgram(gl, PICK_VERTEX, PICK_FRAGMENT);
  let ssao = null;
  try {
    ssao = createSsao(gl);
  } catch (error) {
    console.warn('no ambient occlusion:', error.message);
  }

  let mesh = null;
  let heatOn = false;
  const bands = { edges: null, active: null, hover: null };
  let pickTarget = null;
  let pickDirty = true;
  let lastMvp = null;
  let lastSize = [0, 0];

  const arrayBuffer = (data, usage = gl.STATIC_DRAW) => {
    const handle = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, handle);
    gl.bufferData(gl.ARRAY_BUFFER, data, usage);
    return handle;
  };
  const bindAttribute = (program, name, handle, size) => {
    const slot = gl.getAttribLocation(program, name);
    if (slot < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, handle);
    gl.enableVertexAttribArray(slot);
    gl.vertexAttribPointer(slot, size, gl.FLOAT, false, 0, 0);
  };

  function disposeMesh() {
    if (!mesh) return;
    for (const vao of [mesh.surfaceVao, mesh.aoVao, mesh.pickVao]) if (vao) gl.deleteVertexArray(vao);
    for (const handle of [mesh.position, mesh.facet, mesh.smooth, mesh.colour, mesh.level]) gl.deleteBuffer(handle);
    mesh = null;
  }

  function buildBands(ends) {
    const count = ends.length / 6;
    const a = new Float32Array(count * 12);
    const b = new Float32Array(count * 12);
    const corner = new Float32Array(count * 8);
    const arc = new Float32Array(count * 8);
    const indices = new Uint32Array(count * 6);
    for (let s = 0; s < count; s++) {
      const o = s * 6;
      const length = Math.hypot(ends[o + 3] - ends[o], ends[o + 4] - ends[o + 1], ends[o + 5] - ends[o + 2]);
      for (let v = 0; v < 4; v++) {
        const i = s * 4 + v;
        a[i * 3] = ends[o]; a[i * 3 + 1] = ends[o + 1]; a[i * 3 + 2] = ends[o + 2];
        b[i * 3] = ends[o + 3]; b[i * 3 + 1] = ends[o + 4]; b[i * 3 + 2] = ends[o + 5];
        corner[i * 2] = v < 2 ? 0 : 1;
        corner[i * 2 + 1] = v % 2 === 0 ? -1 : 1;
        arc[i * 2 + 1] = length;
      }
      const base = s * 4;
      indices.set([base, base + 1, base + 2, base + 1, base + 3, base + 2], s * 6);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buffers = [
      ['pointA', a, 3], ['pointB', b, 3], ['corner', corner, 2], ['arc', arc, 2],
    ].map(([name, data, size]) => {
      const handle = arrayBuffer(data);
      bindAttribute(bandProgram, name, handle, size);
      return handle;
    });
    const element = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, element);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    buffers.push(element);
    gl.bindVertexArray(null);
    return { vao, count: indices.length, buffers };
  }

  function ensurePickTarget(width, height) {
    if (pickTarget && pickTarget.width === width && pickTarget.height === height) return pickTarget;
    if (pickTarget) {
      gl.deleteFramebuffer(pickTarget.framebuffer);
      gl.deleteRenderbuffer(pickTarget.depth);
      gl.deleteTexture(pickTarget.texture);
    }
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindTexture(gl.TEXTURE_2D, null);
    const depth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    pickTarget = { framebuffer, texture, depth, width, height };
    pickDirty = true;
    return pickTarget;
  }

  return {
    gl,

    /** Upload a mesh from meshModel.buildMeshModel. Colours start black; call updateColours. */
    setMesh(model) {
      disposeMesh();
      const positions = model.positions instanceof Float32Array ? model.positions : Float32Array.from(model.positions);
      mesh = {
        count: model.count,
        position: arrayBuffer(positions),
        facet: arrayBuffer(model.facetNormals),
        smooth: arrayBuffer(model.smoothNormals),
        colour: arrayBuffer(new Float32Array(model.count * 9), gl.DYNAMIC_DRAW),
        level: arrayBuffer(new Float32Array(model.count * 3).fill(NO_LEVEL), gl.DYNAMIC_DRAW),
        smoothOn: false,
      };
      mesh.surfaceVao = gl.createVertexArray();
      gl.bindVertexArray(mesh.surfaceVao);
      bindAttribute(surfaceProgram, 'position', mesh.position, 3);
      bindAttribute(surfaceProgram, 'normal', mesh.facet, 3);
      bindAttribute(surfaceProgram, 'colour', mesh.colour, 3);
      bindAttribute(surfaceProgram, 'level', mesh.level, 1);
      if (ssao) {
        mesh.aoVao = gl.createVertexArray();
        gl.bindVertexArray(mesh.aoVao);
        bindAttribute(ssao.geometryProgram, 'position', mesh.position, 3);
        bindAttribute(ssao.geometryProgram, 'normal', mesh.facet, 3);
      }
      mesh.pickVao = gl.createVertexArray();
      gl.bindVertexArray(mesh.pickVao);
      bindAttribute(pickProgram, 'position', mesh.position, 3);
      gl.bindVertexArray(null);
      pickDirty = true;
    },

    setShading(smooth) {
      if (!mesh) return;
      mesh.smoothOn = smooth;
      const handle = smooth ? mesh.smooth : mesh.facet;
      gl.bindVertexArray(mesh.surfaceVao);
      bindAttribute(surfaceProgram, 'normal', handle, 3);
      if (mesh.aoVao) {
        gl.bindVertexArray(mesh.aoVao);
        bindAttribute(ssao.geometryProgram, 'normal', handle, 3);
      }
      gl.bindVertexArray(null);
    },

    /** Re-upload triangles first..last (inclusive) of a per-corner rgb array, nine numbers per triangle. */
    updateColours(colours, first, last) {
      if (!mesh || last < first) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.colour);
      gl.bufferSubData(gl.ARRAY_BUFFER, first * 9 * 4, colours, first * 9, (last - first + 1) * 9);
    },

    /** Re-upload triangles first..last (inclusive) of a per-corner level array, three numbers per triangle. */
    updateLevels(levels, first, last) {
      if (!mesh || last < first) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.level);
      gl.bufferSubData(gl.ARRAY_BUFFER, first * 3 * 4, levels, first * 3, (last - first + 1) * 3);
    },

    /** Color corners by their level through the HEAT ramp (the deviation view), or by their color. */
    setHeat(on) {
      heatOn = on;
    },

    /** Replace one of the band sets ('edges', 'active', 'hover') with segment endpoints, or clear it. */
    setBands(name, ends) {
      const old = bands[name];
      if (old) {
        gl.deleteVertexArray(old.vao);
        for (const handle of old.buffers) gl.deleteBuffer(handle);
      }
      bands[name] = ends && ends.length ? buildBands(ends) : null;
    },

    draw({ view, projection, width, height, ratio, occlusionPercent, occlusionRadius, edges }) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
      gl.clearColor(0.2, 0.2, 0.2, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);

      const mvp = multiply4(projection, view);
      if (!lastMvp || width !== lastSize[0] || height !== lastSize[1] || mvp.some((v, i) => v !== lastMvp[i])) pickDirty = true;
      lastMvp = mvp;
      lastSize = [width, height];
      if (!mesh) return;
      const vertices = mesh.count * 3;

      const occluding = ssao && occlusionPercent > 0;
      if (occluding) {
        ssao.beginGeometry(width, height, view, projection);
        gl.bindVertexArray(mesh.aoVao);
        gl.drawArrays(gl.TRIANGLES, 0, vertices);
        gl.bindVertexArray(null);
        ssao.compute(projection, { radius: occlusionRadius });
        gl.clearColor(0.2, 0.2, 0.2, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
      }

      // The kit editor's solid: faces pushed back so the lines win, front faces culled-back
      // first, then back faces alone further back, which only win where a patch is inverted.
      gl.enable(gl.POLYGON_OFFSET_FILL);
      gl.polygonOffset(1.0, 1.0);
      gl.useProgram(surfaceProgram);
      gl.uniformMatrix4fv(gl.getUniformLocation(surfaceProgram, 'uProjection'), false, projection);
      gl.uniformMatrix4fv(gl.getUniformLocation(surfaceProgram, 'uView'), false, view);
      gl.uniform1i(gl.getUniformLocation(surfaceProgram, 'uHeat'), heatOn ? 1 : 0);
      setLightingUniforms(gl, surfaceProgram, {
        occlusionFactor: occlusionFactor(occlusionPercent),
        occlusionTexture: occluding ? ssao.texture : null,
        viewportSize: [width, height],
      });
      gl.bindVertexArray(mesh.surfaceVao);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.drawArrays(gl.TRIANGLES, 0, vertices);
      gl.cullFace(gl.FRONT);
      gl.polygonOffset(3.0, 64.0);
      gl.drawArrays(gl.TRIANGLES, 0, vertices);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.POLYGON_OFFSET_FILL);
      gl.bindVertexArray(null);

      const size = [width, height];
      if (edges && bands.edges) drawBand(gl, bandProgram, bands.edges, mvp, size, bandStyle('line', 'edge', { ratio }));
      for (const hidden of [true, false]) {
        for (const [name, tone] of [['active', 'selected'], ['hover', 'preHighlight']]) {
          if (!bands[name]) continue;
          const style = { ...bandStyle(tone, 'faceBorder', { ratio, hidden }), dash: [0, 0] };
          drawBand(gl, bandProgram, bands[name], mvp, size, style);
        }
      }
    },

    /**
     * What is under canvas pixel (x, y), in device pixels from the top left: `centre` is the
     * triangle at that pixel or -1, `ids` every visible triangle within `radius` pixels.
     */
    pick(x, y, radius = 0) {
      const ids = new Set();
      if (!mesh || !lastMvp) return { centre: -1, ids };
      const [width, height] = lastSize;
      const target = ensurePickTarget(width, height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      if (pickDirty) {
        gl.viewport(0, 0, width, height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.useProgram(pickProgram);
        gl.uniformMatrix4fv(gl.getUniformLocation(pickProgram, 'uMvp'), false, lastMvp);
        gl.bindVertexArray(mesh.pickVao);
        gl.drawArrays(gl.TRIANGLES, 0, mesh.count * 3);
        gl.bindVertexArray(null);
        pickDirty = false;
      }
      const r = Math.ceil(radius);
      const cx = Math.round(x);
      const cy = Math.round(y);
      const x0 = Math.max(0, cx - r);
      const y0 = Math.max(0, cy - r);
      const w = Math.min(width, cx + r + 1) - x0;
      const h = Math.min(height, cy + r + 1) - y0;
      let centre = -1;
      if (w > 0 && h > 0) {
        const block = new Uint8Array(w * h * 4);
        gl.readPixels(x0, height - (y0 + h), w, h, gl.RGBA, gl.UNSIGNED_BYTE, block);
        for (let row = 0; row < h; row++) {
          const py = y0 + h - 1 - row;   // readPixels rows run bottom up
          for (let col = 0; col < w; col++) {
            const px = x0 + col;
            if ((px - cx) ** 2 + (py - cy) ** 2 > radius * radius) continue;
            const i = (row * w + col) * 4;
            const id = (block[i] | (block[i + 1] << 8) | (block[i + 2] << 16)) - 1;
            if (id < 0) continue;
            ids.add(id);
            if (px === cx && py === cy) centre = id;
          }
        }
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { centre, ids };
    },
  };
}
