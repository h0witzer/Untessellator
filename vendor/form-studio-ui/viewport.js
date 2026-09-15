// SPDX-License-Identifier: LGPL-2.1-or-later
//
// What an Onshape viewport looks like, as numbers a renderer can use.
//
// The chrome can be read out of CSS; this cannot. The colour of a part, the
// shape of its shading, the darkness of an edge line and whether there is a
// gradient behind it are facts about pixels, and they are the facts that decide
// whether a 3D panel next to Onshape's looks like the same application.
//
// EVERY CONSTANT BELOW WAS MEASURED by `tools/look-compare.mjs` on 2026-09-11,
// from a screenshot of a real Part Studio displaying a real body — one this
// project put there, so the geometry is known. The method, and what each number
// would look like wrong, is docs/ONSHAPE_STYLE.md.
//
// THE FOUR THINGS PEOPLE GET WRONG, in the order they are noticed:
//
//   1. A GRADIENT. Almost every CAD-adjacent viewport ever written has a
//      vertical gradient behind the model, because SolidWorks and Inventor do.
//      Onshape's does not: measured flat rgb(51,51,51) from the top of the
//      canvas to the bottom, corner to corner, zero variation. It is the same
//      `--os-background-primary` the rest of the product uses, which is why a
//      gradient reads as a hole cut in the page rather than as depth.
//   2. THE EDGE PASS. Onshape draws topological edges in pure black over the
//      surface. They are about 2% of the body's pixels. A renderer that shades
//      without them looks like a mesh viewer; nothing else about the image has
//      to change for that impression to land.
//   3. THE HUE. The default part is a cool blue at 202.5 degrees and only 24%
//      saturation — far less colourful than it looks, and far more blue than a
//      neutral grey. Getting the lightness right and the hue wrong still reads
//      as somebody else's application.
//   4. THE SPECULAR. White (#999 at full light), exponent 30, and it moves
//      with the camera: the light is fixed in EYE space. See LIGHTING below,
//      which is their shader transcribed, not a fit to a still frame.

/**
 * The background, which is flat and is the product's own surface colour.
 *
 * `verticalSpread` and `cornerSpread` are what was measured, in 0-255 steps,
 * across 21 horizontal bands and the four corners. Both zero.
 */
export const BACKGROUND = Object.freeze({
  dark: Object.freeze({ rgb: [51, 51, 51], hex: '#333333' }),
  light: Object.freeze({ rgb: [255, 255, 255], hex: '#ffffff' }),
  measured: Object.freeze({ verticalSpread: 0, cornerSpread: 0, gradient: false }),
});

/**
 * The default part material, in HSL, as it appears ON SCREEN after lighting.
 *
 * These are medians over every surface pixel of the body, not an albedo: an
 * albedo cannot be measured from a render without knowing the lighting, and the
 * thing worth matching is what the eye receives. `bandLightness` is the median
 * lightness in the top, middle and bottom third of the body's own bounding box,
 * which is the shape of the shading independent of the colour — and it is the
 * number that catches a renderer whose fill light is too strong from below.
 */
export const PART_MATERIAL = Object.freeze({
  hue: 202.5,
  saturation: 23.9,
  lightness: 55.5,
  bandLightness: Object.freeze([68.8, 52.2, 38.4]),
  lightnessRange: Object.freeze([26.1, 100]),
  /** hsl(202.5, 23.9%, 55.5%) resolved, for a renderer that wants one colour. */
  hex: '#7294a9',
  rgb: Object.freeze([114, 148, 169]),
});

/**
 * The highlight. Pure white, and small.
 *
 * `share` is the fraction of the body's surface pixels within half a percent of
 * the brightest. 0.51% is a tight highlight; a broad soft one measures several
 * percent and reads as a different, plasticky material.
 */
export const SPECULAR = Object.freeze({
  rgb: Object.freeze([255, 255, 255]),
  luma: 255,
  saturation: 0,
  share: 0.0051,
});

/**
 * Topological edges, drawn over the surface in pure black.
 *
 * `share` is the fraction of body pixels darker than luma 70, which is the cut
 * that separates a black line antialiased over a lit face from the face itself
 * — Onshape's darkest surface band sits near luma 90, so there is daylight
 * between them.
 */
export const EDGES = Object.freeze({
  rgb: Object.freeze([0, 0, 0]),
  hex: '#000000',
  share: 0.0206,
  medianLuma: 38,
  lumaCut: 70,
});

/**
 * Axis colours, which invert in weight between the themes rather than staying
 * put: saturated and dark on white, pale and light on #333.
 */
export const AXES = Object.freeze({
  dark: Object.freeze({ x: '#dd979b', y: '#a2db7d', z: '#90cef1' }),
  light: Object.freeze({ x: '#cd0000', y: '#009400', z: '#0000d0' }),
});

export const SELECTION = Object.freeze({
  dark: Object.freeze({ accent: '#3e94ff', boxSelect: '#769fcc', highlightOutline: '#dba517' }),
  light: Object.freeze({ accent: '#1651b0', boxSelect: '#1651b0', highlightOutline: '#fab601' }),
});

/**
 * The part's colour: what the API reports as the part's appearance in the
 * probe document (`GET /api/v6/parts/...`, `isGenerated: true`), which is also
 * what the pixels resolve to once the lighting below is divided out:
 * rgb(84,111,127) on screen at a lit factor of 0.535 is (157,207,237) exactly.
 * An app drawing a real part should take the part's own appearance; this is
 * the colour to start from when there is none.
 */
export const PART_COLOR = Object.freeze({
  rgb: Object.freeze([157, 207, 237]),
  hex: '#9dcfed',
  unit: Object.freeze([157 / 255, 207 / 255, 237 / 255]),
});

/**
 * THE LIGHTING MODEL, READ OUT OF THE PRODUCT'S OWN SHADER on 2026-09-11.
 *
 * `lightFragment` in the client bundle (`woolsthorpe.<hash>.js`, module 48070)
 * is a Phong headlight in EYE space:
 *
 *     ambient   = colour * COLOR_AMBIENT_FACTOR (0.5)
 *     diffuse   = colour * COLOR_DIFFUSE_FACTOR (1.0) * lightDiffuse (0.5) * max(N.L, 0)
 *     specular  = uSpecular (DEFAULT_SPECULAR #999) * lightSpecular (1.0) * pow(max(N.H, 0), uShininess (30))
 *     L         = normalize(vec3(1, 1, 1))   in eye space — up, right and toward the viewer
 *     eye       = normalize(-positionEye)    per fragment, not per vertex
 *     result   *= pow(1 - occlusion, uOcclusionFactor)   from a screen-space AO map
 *
 * and nothing else: no gamma, no vertical or world-space term, no rim. Because
 * the light is fixed in eye space THE LIGHTING FOLLOWS THE CAMERA: orbit the
 * body and the bright side stays at the upper right of the screen. The
 * previous version of this file lit in world space from a fixed key, which was
 * fitted to one still frame and came apart the moment the view turned.
 *
 * The per-fragment eye vector is also why a flat face is not flat: the
 * specular term varies across it, brightest where the half vector meets the
 * normal, which is the gradient the earlier "darker at the bottom" note
 * measured and attributed to the wrong cause.
 *
 * The Graphics preferences dialog under the view cube exposes these, and its
 * "Reset to default" was disabled with these values showing:
 *
 *     Ambient occlusion   37.5 %      Field of view   45 deg
 *     Line width          1 px        Shininess       3 %  (= exponent 30)
 *     Specular colour     #999999
 *
 * Their AO constants: AMBIENT_OCCLUSION_EFFECT_FACTOR 1.5, INTENSITY 1, BIAS 0.1,
 * SCALE 24, RADIUS 0.01, VIEWPORT_RATIO 0.5 (computed at half resolution).
 */
export const LIGHTING = Object.freeze({
  ambientFactor: 0.5,
  diffuseFactor: 1.0,
  lightDiffuse: 0.5,
  lightSpecular: 1.0,
  lightDirectionEye: Object.freeze([1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)]),
  specular: Object.freeze([0.6, 0.6, 0.6]),
  shininess: 30,
  /** The dialog's percentage and how it maps to the exponent: 3 % -> 30. */
  shininessPercent: 3,
  shininessPerPercent: 10,
  fieldOfViewDegrees: 45,
  /** "Perspective view" is unchecked in their camera menu: the default projection is ORTHOGRAPHIC. */
  perspectiveByDefault: false,
  occlusion: Object.freeze({
    percent: 37.5,
    effectFactor: 1.5,
    intensity: 1,
    bias: 0.1,
    scale: 24,
    radius: 0.01,
    viewportRatio: 0.5,
  }),
  gammaCorrected: false,
});

/**
 * How the product marks what the cursor is on and what is selected, from 3x
 * frames of the live Part Studio (`tools/highlight-extract.mjs`, 2026-09-11)
 * and the appearance constants behind them.
 *
 *   PRE-HIGHLIGHT (under the cursor)  rgba(255,198,133, 1) border, interior alpha 0
 *     face   its boundary drawn as a 2px line; the face itself unchanged
 *     edge   a 5px band: 1px border at alpha 1 each side, 3px interior at 0.78
 *     vertex a 12px point at 0.68
 *   HIGHLIGHT (selected)               rgba(255,169,10, 0.8) fill, rgba(255,169,10, 1) border
 *     face   filled: measured per pixel, result = mix(lit, orange * (0.73 + 0.26 N.L), 0.88),
 *            with a 2px border along its boundary
 *     edge   the same 5px band in orange
 *   HIDDEN   where a highlighted edge or face border is behind the body it still
 *            shows, at alpha 0.18 and stippled 10 on 10 off; a face's fill does not
 *   PICK     vertex beats edge beats face (their priorities 30, 20, 5) within a
 *            radius that measured 7 CSS px from an edge's centre line
 *            (MOUSE_PICK_RADIUS_PX is 5; the renderer bumps it to 8 on some GPUs)
 */
export const HIGHLIGHT_STYLE = Object.freeze({
  preHighlight: Object.freeze({ rgb: [255, 198, 133], unit: [1, 198 / 255, 133 / 255] }),
  selected: Object.freeze({ rgb: [255, 169, 10], unit: [1, 169 / 255, 10 / 255] }),
  edgeWidthPx: 5,
  edgeInteriorAlpha: 0.78,
  edgeBorderAlpha: 1,
  edgeBorderPx: 1,
  faceBorderPx: 2,
  faceBlend: 0.88,
  faceAmbient: 0.73,
  faceDiffuse: 0.26,
  pointSizePx: 12,
  pointAlpha: 0.68,
  /** Behind the body a highlight shows through at HIGHLIGHT_HIDDEN_EDGE_OPACITY, stippled
   *  HIGHLIGHT_HIDDEN_LINE_STIPPLE "stipple(10, 10, 10, 10)". Face fills do not show through. */
  hiddenAlpha: 0.18,
  hiddenDash: Object.freeze([10, 10]),
  /** DEFAULT_LINE_WIDTH: the body's edges, in CSS pixels, black (LINE_POINT_COLOR). */
  edgeLineWidthPx: 1,
  pickRadiusPx: 7.5,
  priority: Object.freeze(['vertex', 'edge', 'face']),
  /** A small ring marks the picked point on a hovered edge. */
  pickMarker: Object.freeze({ radiusPx: 3.5, ringPx: 1.2, rgb: [255, 255, 255], alpha: 0.85 }),
});

/**
 * The product's lighting, as GLSL, for a fragment shader that has the
 * fragment's EYE-space position and normal.
 *
 * Declares its own uniforms: set `uShininess` (30), `uSpecular` (#999 as
 * 0.6, 0.6, 0.6), `uOcclusionFactor` (0 for none; their default is
 * 1.5 * 0.375) and, when the factor is above 0, `uOcclusionMap` sampled by
 * `gl_FragCoord.xy / uViewportSize`. No gamma step follows this: the product
 * does not apply one.
 */
export const PART_GLSL = `
// Onshape's own lightFragment(), transcribed from the client bundle 2026-09-11;
// see src/ui/viewport.js LIGHTING for every constant and where it came from.
uniform float uShininess;
uniform vec3 uSpecular;
uniform float uOcclusionFactor;
uniform sampler2D uOcclusionMap;
uniform vec2 uViewportSize;

// Their Phong headlight: ambient 0.5, diffuse 0.5 * N.L, white specular at
// uSpecular, all in EYE space so the light turns with the camera.
vec3 onshapeLight(vec3 albedo, vec3 positionEye, vec3 normalEye, float ndotlOut) {
  vec3 n = normalize(normalEye);
  vec3 ambient = albedo * 0.5;
  vec3 diffuse = albedo * 1.0;
  vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
  float diffuseWeight = max(dot(n, lightDirection), 0.0);
  vec3 eyeDirection = normalize(-positionEye);
  vec3 halfVector = normalize(lightDirection + eyeDirection);
  float specularWeight = pow(max(dot(n, halfVector), 0.0), uShininess);
  vec3 lit = ambient + diffuseWeight * 0.5 * diffuse + specularWeight * 1.0 * uSpecular;
  if (uOcclusionFactor > 0.0) {
    float occlusion = texture(uOcclusionMap, gl_FragCoord.xy / uViewportSize).r;
    lit *= pow(1.0 - occlusion, uOcclusionFactor);
  }
  return lit;
}

// N.L for the highlight model, which lights the highlight colour by
// 0.73 + 0.26 N.L (fitted to their pixels) and blends it in at 0.88.
float onshapeDiffuseWeight(vec3 normalEye) {
  return max(dot(normalize(normalEye), normalize(vec3(1.0, 1.0, 1.0))), 0.0);
}
vec3 onshapeSelectedFace(vec3 lit, vec3 normalEye) {
  vec3 orange = vec3(1.0, 169.0 / 255.0, 10.0 / 255.0);
  vec3 highlight = orange * (0.73 + 0.26 * onshapeDiffuseWeight(normalEye));
  return mix(lit, highlight, 0.88);
}
`;

/** Everything in one object, for a page that wants to dump it into a panel. */
export const ONSHAPE_VIEWPORT = Object.freeze({
  BACKGROUND,
  PART_COLOR,
  LIGHTING,
  HIGHLIGHT_STYLE,
  PART_MATERIAL,
  SPECULAR,
  EDGES,
  AXES,
  SELECTION,
  measured: '2026-09-11, tools/look-compare.mjs, from a live Part Studio',
});
