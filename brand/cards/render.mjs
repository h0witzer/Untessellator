// render.mjs — draws the link cards and README banners in this folder from their HTML: each <name>.html becomes
// <name>.png at its own size (data-scale on <body>: the banners are 2x). With --preview it also writes
// preview/<name>.phone.png and preview/<name>.feed.png, the image shrunk to a phone feed (350px) and a desktop feed
// (520px), which is how a link card is judged: LinkedIn shows it about that size, and type under ~40px at 1280
// stops reading there.
//
// The live copies are elsewhere: link-card.png is brand/social-preview.png, readme-banner.png is
// brand/readme-banner.png, and the gallery's two are brand/ in Untessellator-Gallery. Copy a new render over them.
//
// Usage: node brand/cards/render.mjs <playwright package dir> [--preview] [name.html ...]
//   (the package dir is the folder holding Playwright's index.mjs; CHROME=<chrome.exe> picks a browser)
import { readdirSync, mkdirSync, readFileSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const pwDir = args.shift();
const preview = args.includes('--preview');
const names = args.filter((a) => a !== '--preview');
const pages = names.length ? names : readdirSync(HERE).filter((f) => f.endsWith('.html'));

const { chromium } = await import(pathToFileURL(join(pwDir, 'index.mjs')).href);
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME, args: ['--allow-file-access-from-files'] });
if (preview) mkdirSync(join(HERE, 'preview'), { recursive: true });
for (const file of pages) {
  const name = basename(file, '.html');
  let page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto(pathToFileURL(join(HERE, file)).href);
  const scale = Number(await page.evaluate(() => document.body.dataset.scale || 1));
  await page.close();
  page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: scale });
  await page.goto(pathToFileURL(join(HERE, file)).href);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(800);
  const png = join(HERE, `${name}.png`);
  await page.locator('.card').screenshot({ path: png });
  await page.close();
  if (preview) {
    for (const [tag, width] of [['phone', 350], ['feed', 520]]) {
      const shot = await browser.newPage({ viewport: { width, height: 600 } });
      await shot.setContent(`<body style="margin:0"><img src="data:image/png;base64,${readFileSync(png).toString('base64')}" style="width:${width}px;display:block">`);
      await shot.waitForTimeout(300);
      await shot.locator('img').screenshot({ path: join(HERE, 'preview', `${name}.${tag}.png`) });
      await shot.close();
    }
  }
  console.log(name, `${scale}x`);
}
await browser.close();
