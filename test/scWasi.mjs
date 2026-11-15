/** scWasi.mjs <sc-bench.wasm> <file.step> [args]: runs the wasm32-wasip1 build of sc-bench under Node's WASI. */
import { WASI } from 'node:wasi';
import { readFileSync } from 'node:fs';
import { dirname, basename } from 'node:path';

const [wasm, file, ...rest] = process.argv.slice(2);
const wasi = new WASI({ version: 'preview1', args: ['sc-bench', `/in/${basename(file)}`, ...rest], preopens: { '/in': dirname(file) } });
const instance = await WebAssembly.instantiate(await WebAssembly.compile(readFileSync(wasm)), wasi.getImportObject());
wasi.start(instance);
