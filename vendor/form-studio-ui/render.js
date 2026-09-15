// SPDX-License-Identifier: LGPL-2.1-or-later
//
// The viewport's rendering pieces, so the next app draws bodies the way the
// product does without writing a shader.
//
// What is here, and where each came from (2026-09-11):
//
//   * The SURFACE program: the product's own `lightFragment` (see
//     viewport.js LIGHTING), a Phong headlight fixed in eye space, plus their
//     selected-face highlight fitted from pixels. A backface is painted a
//     diagnostic red, because a closed body never shows one and any red is an
//     inverted patch.
//   * BANDS: screen-space thick lines, antialiased over one pixel: the body's
//     own 1px black edges (their Line width), edge highlights 5px with a 1px
//     rim at alpha 1 and an interior at 0.78, 2px face borders — the shapes
//     their pre-highlight and highlight take (tools/highlight-extract.mjs) —
//     and, behind the body, the show-through version at 0.18, stippled 10/10.
//   * POINTS: round point sprites with the same border/interior scheme, for
//     vertices (12px at 0.68) and the small ring that marks the picked point
//     on a hovered edge.
//   * SSAO: a screen-space ambient occlusion pass, half resolution and blurred,
//     applied as pow(1 - occlusion, factor) the way theirs is. Their default is
//     37.5% of an effect factor of 1.5.
//   * Projections: perspective at their 45 degree field of view, and the
//     orthographic view their "Perspective view" toggle switches to.
//
// Every draw here takes the camera's view matrix and a projection; nothing is
// derived from angles. The editor is the reference consumer.

import { PART_GLSL, LIGHTING, HIGHLIGHT_STYLE } from './viewport.js';

// ---------------------------------------------------------------------------
// projections
// ---------------------------------------------------------------------------

export function perspective(fovYDegrees, aspect, near, far) {
  const f = 1 / Math.tan((fovYDegrees * Math.PI) / 360);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

/** An orthographic projection showing `halfHeight` world units above and below the centre. */
export function orthographic(halfHeight, aspect, near, far) {
  const out = new Float32Array(16);
  out[0] = 1 / (halfHeight * aspect);
  out[5] = 1 / halfHeight;
  out[10] = -2 / (far - near);
  out[14] = -(far + near) / (far - near);
  out[15] = 1;
  return out;
}

/** The half height an orthographic view needs to match a perspective one at `distance`. */
export function orthographicHalfHeight(fovYDegrees, distance) {
  return distance * Math.tan((fovYDegrees * Math.PI) / 360);
}

export function multiply4(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i += 1) {
    for (let j = 0; j < 4; j += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += a[k * 4 + j] * b[i * 4 + k];
      out[i * 4 + j] = sum;
    }
  }
  return out;
}

export function invert4(m) {
  const inv = new Float32Array(16);
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
  const b00 = a00 * a11 - a01 * a10; const b01 = a00 * a12 - a02 * a10; const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11; const b04 = a01 * a13 - a03 * a11; const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30; const b07 = a20 * a32 - a22 * a30; const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31; const b10 = a21 * a33 - a23 * a31; const b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return inv;
  det = 1 / det;
  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det; inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det; inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det; inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det; inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det; inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det; inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det; inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det; inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return inv;
}

// ---------------------------------------------------------------------------
// programs
// ---------------------------------------------------------------------------

export function compileProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`${type === gl.VERTEX_SHADER ? 'vertex' : 'fragment'} shader: ${gl.getShaderInfoLog(shader)}`);
    }
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

/** The lit surface. Attributes: position, normal, selected. */
export const SURFACE_VERTEX = `#version 300 es
in vec3 position; in vec3 normal; in float selected;
uniform mat4 uProjection; uniform mat4 uView;
out vec3 vPositionEye; out vec3 vNormalEye; out float vSelected;
void main() {
  vec4 positionEye = uView * vec4(position, 1.0);
  vPositionEye = positionEye.xyz;
  // The view matrix is a rotation and a translation, so its upper 3x3 carries
  // normals without an inverse transpose.
  vNormalEye = mat3(uView) * normal;
  vSelected = selected;
  gl_Position = uProjection * positionEye;
}`;

export const SURFACE_FRAGMENT = `#version 300 es
precision highp float;
in vec3 vPositionEye; in vec3 vNormalEye; in float vSelected;
uniform vec3 uAlbedo;
out vec4 color;
${PART_GLSL}
void main() {
  if (!gl_FrontFacing) { color = vec4(0.45, 0.08, 0.10, 1.0); return; }
  vec3 lit = onshapeLight(uAlbedo, vPositionEye, vNormalEye, 0.0);
  if (vSelected > 0.5) lit = onshapeSelectedFace(lit, vNormalEye);
  color = vec4(lit, 1.0);
}`;

/** Plain 1px lines and points, depth-biased. Attributes: position. */
export const LINE_VERTEX = `#version 300 es
in vec3 position;
uniform mat4 uMvp; uniform float uDepthBias; uniform float uPointSize;
void main() {
  gl_Position = uMvp * vec4(position, 1.0);
  gl_Position.z -= uDepthBias * gl_Position.w;
  gl_PointSize = uPointSize;
}`;
export const LINE_FRAGMENT = `#version 300 es
precision highp float;
uniform vec4 uColor; out vec4 color;
void main() { color = uColor; }`;

/**
 * Screen-space bands: a segment expanded to a quad `uHalfWidthPx` wide on
 * each side, with the across-coordinate handed to the fragment shader so the
 * border can be opaque and the interior translucent, as theirs are.
 * Attributes: pointA, pointB (the segment), corner (x: 0 at A / 1 at B, y: side).
 */
export const BAND_VERTEX = `#version 300 es
in vec3 pointA; in vec3 pointB; in vec2 corner; in vec2 arc;
uniform mat4 uMvp; uniform vec2 uViewportSize; uniform float uHalfWidthPx; uniform float uDepthBias;
out float vAcross; out float vAlongPx;
void main() {
  vec4 a = uMvp * vec4(pointA, 1.0);
  vec4 b = uMvp * vec4(pointB, 1.0);
  vec2 halfSize = uViewportSize * 0.5;
  vec2 sa = a.xy / a.w * halfSize;
  vec2 sb = b.xy / b.w * halfSize;
  vec2 dir = sb - sa;
  float len = length(dir);
  dir = len > 1e-4 ? dir / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  // Half a pixel wider than the line, so the fragment shader can antialias
  // the rim across one pixel instead of cutting it hard.
  float quadHalf = uHalfWidthPx + 0.5;
  vec4 p = mix(a, b, corner.x);
  vec2 sp = mix(sa, sb, corner.x) + nrm * corner.y * quadHalf;
  gl_Position = vec4(sp / halfSize * p.w, p.z - uDepthBias * p.w, p.w);
  vAcross = corner.y;
  // Screen-space distance along the polyline, for a stipple: the world arc
  // length at this end scaled by this segment's own screen-per-world ratio.
  float worldLen = max(length(pointB - pointA), 1e-6);
  vAlongPx = mix(arc.x, arc.y, corner.x) * (len / worldLen);
}`;
export const BAND_FRAGMENT = `#version 300 es
precision highp float;
in float vAcross; in float vAlongPx;
uniform vec4 uColor; uniform float uInteriorAlpha; uniform float uBorderPx; uniform float uHalfWidthPx;
uniform vec2 uDash;
out vec4 color;
void main() {
  float quadHalf = uHalfWidthPx + 0.5;
  float d = abs(vAcross) * quadHalf;
  // Coverage of the line's true width by this pixel: 1 inside, a one-pixel ramp at the rim.
  float coverage = clamp(uHalfWidthPx + 0.5 - d, 0.0, 1.0);
  float alpha = d > uHalfWidthPx - uBorderPx ? 1.0 : uInteriorAlpha;
  if (uDash.x > 0.0 && mod(vAlongPx, uDash.x + uDash.y) > uDash.x) discard;
  color = vec4(uColor.rgb, uColor.a * alpha * coverage);
}`;

/** Round point sprites with an opaque rim. Attributes: position. */
export const POINT_VERTEX = LINE_VERTEX;
export const POINT_FRAGMENT = `#version 300 es
precision highp float;
uniform vec4 uColor; uniform float uInteriorAlpha; uniform float uBorderFraction;
out vec4 color;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d = length(c);
  if (d > 1.0) discard;
  float alpha = d > 1.0 - uBorderFraction ? 1.0 : uInteriorAlpha;
  color = vec4(uColor.rgb, uColor.a * alpha);
}`;

// ---------------------------------------------------------------------------
// meshes
// ---------------------------------------------------------------------------

function attribute(gl, program, name, data, size) {
  const slot = gl.getAttribLocation(program, name);
  if (slot < 0) return;
  const handle = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, handle);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(slot);
  gl.vertexAttribPointer(slot, size, gl.FLOAT, false, 0, 0);
}

/**
 * A band mesh from segments: `positions` is a flat xyz array and `segments`
 * pairs of indices into it. Four vertices and two triangles per segment.
 */
export function buildBandMesh(gl, program, positions, segments) {
  const count = segments.length / 2;
  const a = new Float32Array(count * 12);
  const b = new Float32Array(count * 12);
  const corner = new Float32Array(count * 8);
  const arc = new Float32Array(count * 8);
  const indices = new Uint32Array(count * 6);
  // Cumulative world arc length, carried across consecutive segments so a
  // stipple runs along a polyline instead of restarting at every sample.
  let running = 0;
  for (let s = 0; s < count; s += 1) {
    const ia = segments[2 * s] * 3;
    const ib = segments[2 * s + 1] * 3;
    if (s > 0 && segments[2 * s - 1] !== segments[2 * s]) running = 0;
    const length = Math.hypot(positions[ib] - positions[ia], positions[ib + 1] - positions[ia + 1], positions[ib + 2] - positions[ia + 2]);
    for (let v = 0; v < 4; v += 1) {
      const o = (s * 4 + v) * 3;
      a[o] = positions[ia]; a[o + 1] = positions[ia + 1]; a[o + 2] = positions[ia + 2];
      b[o] = positions[ib]; b[o + 1] = positions[ib + 1]; b[o + 2] = positions[ib + 2];
      corner[(s * 4 + v) * 2] = v < 2 ? 0 : 1;
      corner[(s * 4 + v) * 2 + 1] = v % 2 === 0 ? -1 : 1;
      arc[(s * 4 + v) * 2] = running;
      arc[(s * 4 + v) * 2 + 1] = running + length;
    }
    running += length;
    const base = s * 4;
    indices.set([base, base + 1, base + 2, base + 1, base + 3, base + 2], s * 6);
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  attribute(gl, program, 'pointA', a, 3);
  attribute(gl, program, 'pointB', b, 3);
  attribute(gl, program, 'corner', corner, 2);
  attribute(gl, program, 'arc', arc, 2);
  const element = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, element);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, count: indices.length };
}

/** A point mesh from flat xyz positions. */
export function buildPointMesh(gl, program, positions) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  attribute(gl, program, 'position', positions, 3);
  gl.bindVertexArray(null);
  return { vao, count: positions.length / 3 };
}

/**
 * The style a band is drawn in. `bandStyle` gives the product's:
 *
 *   tone 'preHighlight' | 'selected', kind 'edge' | 'faceBorder'  — the highlights
 *   tone 'line'                                                    — the body's own edges,
 *                                                                    1px black (their Line width)
 *
 * and `hidden: true` gives the show-through version the product draws where a
 * highlight is behind the body: the same shape at HIGHLIGHT_HIDDEN_EDGE_OPACITY
 * 0.18, stippled 10 on 10 off (HIGHLIGHT_HIDDEN_LINE_STIPPLE), depth-tested the
 * other way round so it appears only where something is in front of it. Face
 * fills never show through; their borders do.
 */
export function bandStyle(tone, kind = 'edge', { ratio = 1, hidden = false, lineWidthPx = 1 } = {}) {
  const style = HIGHLIGHT_STYLE;
  if (tone === 'line') {
    return { colour: [0, 0, 0], alpha: 1, widthPx: lineWidthPx * ratio, borderPx: 0, interiorAlpha: 1, depthBias: 0, hidden: false, dash: [0, 0] };
  }
  const widthPx = (kind === 'edge' ? style.edgeWidthPx : style.faceBorderPx) * ratio;
  return {
    colour: style[tone].unit,
    alpha: hidden ? style.hiddenAlpha : 1,
    widthPx,
    borderPx: kind === 'edge' ? style.edgeBorderPx * ratio : 0,
    interiorAlpha: kind === 'edge' && !hidden ? style.edgeInteriorAlpha : 1,
    depthBias: 0.0012,
    hidden,
    dash: hidden ? style.hiddenDash.map((v) => v * ratio) : [0, 0],
  };
}

/** Draw a band mesh in a style from `bandStyle`. */
export function drawBand(gl, program, mesh, mvp, viewportSize, style) {
  gl.useProgram(program);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uMvp'), false, mvp);
  gl.uniform2f(gl.getUniformLocation(program, 'uViewportSize'), viewportSize[0], viewportSize[1]);
  gl.uniform1f(gl.getUniformLocation(program, 'uHalfWidthPx'), style.widthPx / 2);
  gl.uniform1f(gl.getUniformLocation(program, 'uDepthBias'), style.depthBias);
  gl.uniform4f(gl.getUniformLocation(program, 'uColor'), style.colour[0], style.colour[1], style.colour[2], style.alpha);
  gl.uniform1f(gl.getUniformLocation(program, 'uInteriorAlpha'), style.interiorAlpha);
  gl.uniform1f(gl.getUniformLocation(program, 'uBorderPx'), style.borderPx);
  gl.uniform2f(gl.getUniformLocation(program, 'uDash'), style.dash[0], style.dash[1]);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  if (style.hidden) {
    // Only where the body is in front: the depth test inverted, and no writes.
    gl.depthFunc(gl.GREATER);
    gl.depthMask(false);
  }
  gl.bindVertexArray(mesh.vao);
  gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0);
  gl.bindVertexArray(null);
  if (style.hidden) {
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
  }
  gl.disable(gl.BLEND);
}

/** Draw vertex highlights, or the pick marker ring, as point sprites. */
export function drawPoints(gl, program, mesh, mvp, { tone, marker = false, ratio = 1, depthBias = 0.0016, hidden = false }) {
  const style = HIGHLIGHT_STYLE;
  const colour = marker ? style.pickMarker.rgb.map((v) => v / 255) : style[tone].unit;
  const size = marker ? style.pickMarker.radiusPx * 2 : style.pointSizePx;
  gl.useProgram(program);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uMvp'), false, mvp);
  gl.uniform1f(gl.getUniformLocation(program, 'uDepthBias'), depthBias);
  gl.uniform1f(gl.getUniformLocation(program, 'uPointSize'), size * ratio);
  gl.uniform4f(gl.getUniformLocation(program, 'uColor'), colour[0], colour[1], colour[2], marker ? style.pickMarker.alpha : hidden ? style.hiddenAlpha : 1);
  gl.uniform1f(gl.getUniformLocation(program, 'uInteriorAlpha'), marker ? 0 : hidden ? 1 : style.pointAlpha);
  gl.uniform1f(gl.getUniformLocation(program, 'uBorderFraction'), marker ? style.pickMarker.ringPx / style.pickMarker.radiusPx : 1 / (size / 2));
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  if (hidden) { gl.depthFunc(gl.GREATER); gl.depthMask(false); }
  gl.bindVertexArray(mesh.vao);
  gl.drawArrays(gl.POINTS, 0, mesh.count);
  gl.bindVertexArray(null);
  if (hidden) { gl.depthFunc(gl.LESS); gl.depthMask(true); }
  gl.disable(gl.BLEND);
}

/** Set the lighting uniforms the surface program (PART_GLSL) declares. */
export function setLightingUniforms(gl, program, { shininess = LIGHTING.shininess, specular = LIGHTING.specular, occlusionFactor = 0, occlusionTexture = null, viewportSize, albedo }) {
  gl.uniform1f(gl.getUniformLocation(program, 'uShininess'), shininess);
  gl.uniform3f(gl.getUniformLocation(program, 'uSpecular'), specular[0], specular[1], specular[2]);
  gl.uniform1f(gl.getUniformLocation(program, 'uOcclusionFactor'), occlusionTexture ? occlusionFactor : 0);
  gl.uniform2f(gl.getUniformLocation(program, 'uViewportSize'), viewportSize[0], viewportSize[1]);
  if (albedo) gl.uniform3f(gl.getUniformLocation(program, 'uAlbedo'), albedo[0], albedo[1], albedo[2]);
  if (occlusionTexture) {
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, occlusionTexture);
    gl.uniform1i(gl.getUniformLocation(program, 'uOcclusionMap'), 1);
    gl.activeTexture(gl.TEXTURE0);
  }
}

// ---------------------------------------------------------------------------
// screen-space ambient occlusion
// ---------------------------------------------------------------------------

const AO_GEOMETRY_VERTEX = `#version 300 es
in vec3 position; in vec3 normal;
uniform mat4 uProjection; uniform mat4 uView;
out vec3 vNormalEye; out float vDepthEye;
void main() {
  vec4 positionEye = uView * vec4(position, 1.0);
  vNormalEye = mat3(uView) * normal;
  vDepthEye = -positionEye.z;
  gl_Position = uProjection * positionEye;
}`;
const AO_GEOMETRY_FRAGMENT = `#version 300 es
precision highp float;
in vec3 vNormalEye; in float vDepthEye;
out vec4 color;
void main() {
  vec3 n = normalize(gl_FrontFacing ? vNormalEye : -vNormalEye);
  color = vec4(n, vDepthEye);
}`;
const FULLSCREEN_VERTEX = `#version 300 es
in vec2 xy; out vec2 vUv;
void main() { vUv = xy * 0.5 + 0.5; gl_Position = vec4(xy, 0.0, 1.0); }`;
const AO_FRAGMENT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uGeometry;
uniform mat4 uProjection; uniform mat4 uInverseProjection;
uniform float uRadius; uniform float uBias; uniform float uIntensity;
uniform vec3 uKernel[16];
out vec4 color;
vec3 eyePosition(vec2 uv, float depth) {
  // Reconstruct the eye-space position from the stored eye depth and the pixel's ray.
  vec4 clip = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
  vec4 ray = uInverseProjection * clip;
  ray /= ray.w;
  // For a perspective projection the ray passes through the origin; for an
  // orthographic one it is parallel. Scale to the stored depth either way.
  vec3 direction = ray.xyz;
  if (abs(uProjection[3][3]) < 0.5) return direction * (depth / max(-direction.z, 1e-5));
  return vec3(direction.xy, -depth);
}
void main() {
  vec4 g = texture(uGeometry, vUv);
  if (g.w <= 0.0) { color = vec4(0.0); return; }
  vec3 n = normalize(g.xyz);
  vec3 p = eyePosition(vUv, g.w);
  // An orthonormal basis around the normal, rotated per pixel to break banding.
  float angle = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
  vec3 tangent = normalize(cross(n, abs(n.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)));
  vec3 bitangent = cross(n, tangent);
  mat3 basis = mat3(tangent * cos(angle) + bitangent * sin(angle), bitangent * cos(angle) - tangent * sin(angle), n);
  float occlusion = 0.0;
  for (int i = 0; i < 16; i += 1) {
    vec3 samplePosition = p + basis * uKernel[i] * uRadius;
    vec4 offset = uProjection * vec4(samplePosition, 1.0);
    vec2 sampleUv = offset.xy / offset.w * 0.5 + 0.5;
    if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0) continue;
    float sceneDepth = texture(uGeometry, sampleUv).w;
    if (sceneDepth <= 0.0) continue;
    float sampleDepth = -samplePosition.z;
    float range = smoothstep(0.0, 1.0, uRadius / max(abs(g.w - sceneDepth), 1e-5));
    occlusion += (sceneDepth < sampleDepth - uBias * uRadius ? 1.0 : 0.0) * range;
  }
  color = vec4(clamp(occlusion / 16.0 * uIntensity, 0.0, 1.0), 0.0, 0.0, 1.0);
}`;
const BLUR_FRAGMENT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSource; uniform vec2 uTexel;
out vec4 color;
void main() {
  float sum = 0.0;
  for (int x = -2; x <= 1; x += 1) for (int y = -2; y <= 1; y += 1) sum += texture(uSource, vUv + vec2(float(x) + 0.5, float(y) + 0.5) * uTexel).r;
  color = vec4(sum / 16.0, 0.0, 0.0, 1.0);
}`;

/**
 * Screen-space ambient occlusion in three passes: geometry (eye normal and
 * depth), occlusion at half resolution, a 4x4 blur. Use:
 *
 *     const ssao = createSsao(gl);
 *     ssao.beginGeometry(width, height, view, projection);   // then draw meshes built for ssao.geometryProgram
 *     ssao.compute(projection, { radius, intensity });        // leaves the default framebuffer bound
 *     ...setLightingUniforms(gl, surface, { occlusionTexture: ssao.texture, occlusionFactor })
 */
export function createSsao(gl) {
  if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float is needed for the occlusion pass');
  const geometryProgram = compileProgram(gl, AO_GEOMETRY_VERTEX, AO_GEOMETRY_FRAGMENT);
  const aoProgram = compileProgram(gl, FULLSCREEN_VERTEX, AO_FRAGMENT);
  const blurProgram = compileProgram(gl, FULLSCREEN_VERTEX, BLUR_FRAGMENT);
  const quad = gl.createVertexArray();
  gl.bindVertexArray(quad);
  attribute(gl, aoProgram, 'xy', new Float32Array([-1, -1, 3, -1, -1, 3]), 2);
  gl.bindVertexArray(null);
  // Their kernel is theirs; this is a hemisphere of 16 samples weighted toward the centre.
  const kernel = new Float32Array(48);
  let seed = 7;
  const random = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let i = 0; i < 16; i += 1) {
    let x = random() * 2 - 1; let y = random() * 2 - 1; let z = random();
    const length = Math.hypot(x, y, z) || 1;
    let scale = (i + 1) / 16;
    scale = 0.1 + 0.9 * scale * scale;
    x = (x / length) * scale; y = (y / length) * scale; z = (z / length) * scale;
    kernel.set([x, y, z], i * 3);
  }

  let size = { width: 0, height: 0 };
  let geometry = null;
  let ao = null;
  let blurred = null;
  const makeTarget = (width, height, internal, format, type, filter) => {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    return { texture, framebuffer, width, height };
  };
  const dispose = (target) => {
    if (!target) return;
    gl.deleteTexture(target.texture);
    gl.deleteFramebuffer(target.framebuffer);
    if (target.depth) gl.deleteRenderbuffer(target.depth);
  };
  const resize = (width, height) => {
    if (size.width === width && size.height === height) return;
    dispose(geometry); dispose(ao); dispose(blurred);
    geometry = makeTarget(width, height, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    geometry.depth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, geometry.depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, geometry.framebuffer);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, geometry.depth);
    const half = { width: Math.max(1, Math.round(width * LIGHTING.occlusion.viewportRatio)), height: Math.max(1, Math.round(height * LIGHTING.occlusion.viewportRatio)) };
    ao = makeTarget(half.width, half.height, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.LINEAR);
    blurred = makeTarget(half.width, half.height, gl.R8, gl.RED, gl.UNSIGNED_BYTE, gl.LINEAR);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    size = { width, height };
  };

  return {
    geometryProgram,
    get texture() { return blurred?.texture ?? null; },
    /** Bind the geometry target and set its uniforms; then draw the meshes. */
    beginGeometry(width, height, view, projection) {
      resize(width, height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, geometry.framebuffer);
      gl.viewport(0, 0, width, height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(geometryProgram);
      gl.uniformMatrix4fv(gl.getUniformLocation(geometryProgram, 'uView'), false, view);
      gl.uniformMatrix4fv(gl.getUniformLocation(geometryProgram, 'uProjection'), false, projection);
    },
    /** Occlusion and blur; leaves the default framebuffer bound with the full viewport. */
    compute(projection, { radius, intensity = LIGHTING.occlusion.intensity, bias = LIGHTING.occlusion.bias }) {
      gl.disable(gl.DEPTH_TEST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, ao.framebuffer);
      gl.viewport(0, 0, ao.width, ao.height);
      gl.useProgram(aoProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, geometry.texture);
      gl.uniform1i(gl.getUniformLocation(aoProgram, 'uGeometry'), 0);
      gl.uniformMatrix4fv(gl.getUniformLocation(aoProgram, 'uProjection'), false, projection);
      gl.uniformMatrix4fv(gl.getUniformLocation(aoProgram, 'uInverseProjection'), false, invert4(projection));
      gl.uniform1f(gl.getUniformLocation(aoProgram, 'uRadius'), radius);
      gl.uniform1f(gl.getUniformLocation(aoProgram, 'uBias'), bias);
      gl.uniform1f(gl.getUniformLocation(aoProgram, 'uIntensity'), intensity);
      gl.uniform3fv(gl.getUniformLocation(aoProgram, 'uKernel'), kernel);
      gl.bindVertexArray(quad);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, blurred.framebuffer);
      gl.useProgram(blurProgram);
      gl.bindTexture(gl.TEXTURE_2D, ao.texture);
      gl.uniform1i(gl.getUniformLocation(blurProgram, 'uSource'), 0);
      gl.uniform2f(gl.getUniformLocation(blurProgram, 'uTexel'), 1 / ao.width, 1 / ao.height);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, size.width, size.height);
      gl.enable(gl.DEPTH_TEST);
    },
  };
}

/** Their default occlusion strength as the exponent the shader takes. */
export function occlusionFactor(percent = LIGHTING.occlusion.percent) {
  return LIGHTING.occlusion.effectFactor * (percent / 100);
}
