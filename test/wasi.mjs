/**
 *  @file test/wasi.mjs
 *  @author Ash Vardanian
 *  @date June 8, 2026
 *  @brief Minimal WASI command runner: `node wasi.mjs <module.wasm> [args...]`.
 *
 *  Lets CTest use node as a cross-runtime engine for the portable WASI tests, next to wasmtime and
 *  wasmer, via `-DCMAKE_CROSSCOMPILING_EMULATOR="node;<source>/test/wasi.mjs"`. Node cannot
 *  execute a bare `.wasm` from the CLI, so this thin wrapper instantiates it with a WASI import
 *  object and forwards the process exit code.
 */
import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';
import { argv, env, exit } from 'node:process';

const modulePath = argv[2];
if (!modulePath) {
    console.error('usage: node wasi.mjs <module.wasm> [args...]');
    exit(2);
}
const wasi = new WASI({ version: 'preview1', args: argv.slice(2), env, returnOnExit: true });
const module = new WebAssembly.Module(readFileSync(modulePath));
const instance = new WebAssembly.Instance(module, wasi.getImportObject());
exit(wasi.start(instance));
