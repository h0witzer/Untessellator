// Runs test/freeformWorker.html in headless Chromium over the DevTools protocol (no window, no screenshots) and
// prints the worker's result. Needs the dev server (node server.js) and a Chromium headless shell.
// node test/freeformBrowser.mjs [--port 3000] [--chrome <path to chrome-headless-shell.exe>]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const args = process.argv.slice(2);
const flag = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const port = flag('--port', '3000');
const pw = path.join(os.homedir(), 'AppData/Local/ms-playwright');
const guess = fs.existsSync(pw) ? fs.readdirSync(pw).filter((d) => d.startsWith('chromium_headless_shell')).map((d) => path.join(pw, d, 'chrome-headless-shell-win64/chrome-headless-shell.exe')).find(fs.existsSync) : null;
const chrome = flag('--chrome', guess);
if (!chrome) { console.log('no Chromium headless shell found; pass --chrome'); process.exit(1); }
const debugPort = 9300 + Math.floor(Math.random() * 500);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'freeform-chrome-'));
const proc = spawn(chrome, [`--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`, '--no-first-run', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let version = null;
for (let i = 0; i < 50 && !version; i++) { await sleep(100); version = await fetch(`http://127.0.0.1:${debugPort}/json/version`).then((r) => r.json()).catch(() => null); }
const ws = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let next = 0;
const waiting = new Map();
ws.addEventListener('message', ({ data }) => { const m = JSON.parse(data); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } });
const send = (method, params = {}, sessionId) => new Promise((r) => { const id = ++next; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params, sessionId })); });
const { result: { targetId } } = await send('Target.createTarget', { url: `http://localhost:${port}/test/freeformWorker.html` });
const { result: { sessionId } } = await send('Target.attachToTarget', { targetId, flatten: true });
let title = '';
for (let i = 0; i < 600 && !title.startsWith('done'); i++) {
  await sleep(200);
  const r = await send('Runtime.evaluate', { expression: 'document.title', returnByValue: true }, sessionId);
  title = r.result?.result?.value ?? '';
}
console.log(title.startsWith('done') ? JSON.stringify(JSON.parse(title.slice(5)), null, 1) : 'timed out');
ws.close();
proc.kill();
