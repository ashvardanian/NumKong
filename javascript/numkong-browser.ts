/**
 *  @file javascript/numkong-browser.ts
 *  @author Ash Vardanian
 *  @date March 20, 2026
 *  @brief Self-contained browser ESM entry point for NumKong WASM.
 *
 *  Auto-initializes the Emscripten module on import through a top-level await, probing the engine
 *  first. The matching glue and binary, `numkong-wasm32-v128.js` or `numkong-wasm32-v128relaxed.js`
 *  with its `.wasm`, must sit beside this file, in the same directory or CDN prefix.
 *
 *  @example
 *  ```js
 *  import { dot, euclidean } from './numkong.js';
 *  console.log(dot(new Float32Array([1, 2, 3]), new Float32Array([4, 5, 6])));
 *  ```
 *
 *  @packageDocumentation
 */

export {
    TensorBase, VectorBase, VectorView, Vector,
    MatrixBase, Matrix, PackedMatrix,
    DType, TypedArray, KernelFamily,
    dtypeToString, outputDType,
    Float16Array, BFloat16Array, E4M3Array, E5M2Array, BinaryArray,
    isFloat16Array, isBFloat16Array, isE4M3Array, isE5M2Array, isBinaryArray,
} from './types.js';

import { initWasm } from './numkong-wasm.js';
export {
    dot, inner, euclidean, sqeuclidean, angular,
    hamming, jaccard, kullbackleibler, jensenshannon,
    capabilitiesDetected, capabilitiesCompiled, capabilitiesEnabled, capabilitiesEnable, Capability,
    dotsPack, dotsPackedSize,
    dotsPacked, angularsPacked, euclideansPacked,
    dotsSymmetric, angularsSymmetric, euclideansSymmetric,
} from './numkong-wasm.js';
import { detectWasmSimdTier } from './wasm-probes.js';

/** Thrown at import where the engine validates neither SIMD probe; no serial module is shipped. */
export class NumKongWasmSimdError extends Error {
    constructor() {
        super(
            'This engine validates neither the SIMD128 nor the Relaxed SIMD probe, ' +
            'and NumKong ships no serial WebAssembly module',
        );
        this.name = 'NumKongWasmSimdError';
    }
}

/* Auto-initialize: probe the engine, load the matching Emscripten glue relative to this module's
 * URL, instantiate the WASM module, and wire up the wrapper before any export is used. */
const tier = detectWasmSimdTier();
if (tier === 'serial') throw new NumKongWasmSimdError();
const glueUrl = new URL(`./numkong-wasm32-${tier}.js`, import.meta.url);
const { default: NumKongModule } = await import(glueUrl.href);
const wasmInstance = await NumKongModule({
    locateFile: (path: string) => new URL(path, glueUrl).href,
});
initWasm(wasmInstance);
