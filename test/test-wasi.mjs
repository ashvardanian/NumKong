// Minimal WASI command runner: `node --experimental-wasi-unstable-preview1 run-wasi.mjs <module.wasm> [args...]`
// Lets CTest use node as a cross-runtime engine for the portable WASI tests, alongside wasmtime and wasmer
// (see cmake/toolchain-wasi.cmake, NK_WASM_RUNTIME=node). Node cannot execute a bare `.wasm` from the CLI,
// so this thin wrapper instantiates it with a WASI import object and forwards the process exit code.
import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';
import { argv, exit } from 'node:process';

const modulePath = argv[2];
if (!modulePath) {
    console.error('usage: node --experimental-wasi-unstable-preview1 run-wasi.mjs <module.wasm> [args...]');
    exit(2);
}
const wasi = new WASI({ version: 'preview1', args: argv.slice(2), returnOnExit: true });
const module = new WebAssembly.Module(readFileSync(modulePath));
const instance = new WebAssembly.Instance(module, wasi.getImportObject());
exit(wasi.start(instance));
