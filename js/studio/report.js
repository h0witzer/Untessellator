/**
 * report.js — a conversion report: this run's numbers as a prefilled GitHub issue.
 *
 * The models ship in models/ and run in the browser, and the studio promises that nothing is uploaded,
 * so the only way to learn how they do on real parts is for people to tell us. The Report button opens
 * the repository's "Conversion report" issue form with the numbers below already filled in; the user
 * reads them, adds what went wrong (and the mesh, only if they may share it), and submits. Nothing leaves
 * the browser until they do. The mesh's file name is left out: they can add it if they want to.
 *
 * The numbers are the ones that separate the failure modes: which model and fill settings ran, how many
 * groups auto-paint made and of what type (hundreds of one-triangle cylinders is over-segmentation), and
 * the export's tier (analytic, hybrid, planar, faceted, or not closed) with its time.
 */

export const REPORT_REPO = 'https://github.com/h0witzer/stlToStep';
/** The issue form in .github/ISSUE_TEMPLATE; the query keys are its field ids. */
const TEMPLATE = 'conversion-report.yml';

const n = (x) => Number(x).toLocaleString('en-US');
const pct = (x) => `${(100 * x).toFixed(x < 0.01 ? 1 : 0)}%`;

/** The browser, briefly: "Google Chrome 141 / Windows", or the start of the user agent. */
function browserName() {
  const data = navigator.userAgentData;
  if (data?.brands?.length) {
    const brand = data.brands.find((b) => !/not.?a.?brand|chromium/i.test(b.brand)) ?? data.brands[0];
    return `${brand.brand} ${brand.version}${data.platform ? ` / ${data.platform}` : ''}`;
  }
  return navigator.userAgent.slice(0, 120);
}

/**
 * The report's numbers as plain lines.
 *   triangles   the open mesh's triangle count (0: none open)
 *   repaired    the repair's changes, as the studio words them
 *   model       the seed model's label
 *   settings    the studio's settings object
 *   autoPaint   { secs, groups, freeform, unpainted, stopped } from the last auto-paint, or null
 *   groups      the face groups now, as { type, count } (type null: untyped, exported as a B-spline or facets)
 *   painted     triangles in any group
 *   exported    the knitter's report from the last export, { failed } when it threw, or null
 */
export function conversionNumbers({ triangles, repaired, model, settings, autoPaint, groups, painted, exported }) {
  const lines = [`browser: ${browserName()}`];
  if (!triangles) return [...lines, 'mesh: none open'].join('\n');
  lines.push(`mesh: ${n(triangles)} triangles${repaired.length ? `; repaired: ${repaired.join(', ')}` : ''}`);
  const on = (x) => (x ? 'on' : 'off');
  lines.push(`fill: model ${model}, method ${settings.autoMethod}, refined fills ${on(settings.refine)}, ` +
    `boundary wall ${on(settings.edgeWall)}, group typer ${on(settings.groupTyper)}, curvature wall ${on(settings.autoTangent)}`);
  lines.push(autoPaint
    ? `auto-paint: ${autoPaint.stopped ? 'stopped' : 'done'} in ${autoPaint.secs.toFixed(0)} s, ${n(autoPaint.groups)} new groups` +
      `${autoPaint.freeform ? ` (${autoPaint.freeform} freeform)` : ''}, ${n(autoPaint.unpainted)} triangles left unpainted`
    : 'auto-paint: not run');
  const byType = new Map();
  let single = 0;
  for (const g of groups) {
    const type = g.type ?? 'untyped';
    byType.set(type, (byType.get(type) ?? 0) + 1);
    if (g.count === 1) single++;
  }
  const types = [...byType].sort((a, b) => b[1] - a[1]).map(([t, c]) => `${t} ${n(c)}`).join(', ');
  lines.push(`groups: ${n(groups.length)}${types ? ` (${types})` : ''}; ${n(single)} of one triangle; ` +
    `${n(painted)} of ${n(triangles)} triangles painted`);
  if (!exported) lines.push('export: not run');
  else if (exported.failed && !exported.tier) lines.push(`export: failed (${exported.failed})`);
  else {
    const facets = exported.facetTris ? `, ${n(exported.facetTris)} facet triangles (${pct(exported.facetArea ?? 0)} of the area)` : '';
    const open = exported.free_edges ? `, ${n(exported.free_edges)} open edges` : '';
    const why = exported.timedOut ? ', timed out' : exported.failed ? `, build failed (${exported.failed})` : '';
    lines.push(`export: tier ${exported.tier}${exported.bestTier && exported.bestTier !== exported.tier ? ` (best ${exported.bestTier})` : ''}` +
      `${facets}${open}${why}, ${exported.secs} s`);
  }
  return lines.join('\n');
}

/** The prefilled issue form: the title says the outcome, the numbers go in the form's `numbers` field. */
export function reportUrl(numbers, { tier, groups } = {}) {
  const outcome = tier ? `tier ${tier}` : 'not exported';
  const title = `Conversion report: ${outcome}${groups ? `, ${groups} groups` : ''}`;
  const q = new URLSearchParams({ template: TEMPLATE, title, numbers });
  return `${REPORT_REPO}/issues/new?${q}`;
}
