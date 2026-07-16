/**
 * @brief Self-contained browser ESM entry point for NumKong WASM.
 * @file javascript/numkong-browser.ts
 *
 * Auto-initializes the Emscripten module on import via top-level await.
 * The Emscripten glue (`numkong-emscripten.js`) and binary (`numkong.wasm`)
 * must be co-located with this file (same directory or CDN prefix).
 *
 * Usage:
 *   import { dot, euclidean } from './numkong.js';
 *   console.log(dot(new Float32Array([1,2,3]), new Float32Array([4,5,6])));
 */

export {
    TensorBase, VectorBase, VectorView, Vector,
    MatrixBase, Matrix, PackedMatrix,
    DType, TypedArray, KernelFamily,
    dtypeToString, outputDtype,
    Float16Array, BFloat16Array, E4M3Array, E5M2Array, BinaryArray,
    isFloat16Array, isBFloat16Array, isE4M3Array, isE5M2Array, isBinaryArray,
} from './types.js';

import { initWasm } from './numkong-wasm.js';
export {
    dot, inner, euclidean, sqeuclidean, angular,
    hamming, jaccard, kullbackleibler, jensenshannon,
    getCapabilitiesDetected, getCapabilitiesCompiled, getCapabilitiesAvailable, getCapabilitiesEnabled,
    hasCapability,
    dotsPack, dotsPackedSize,
    dotsPacked, angularsPacked, euclideansPacked,
    dotsSymmetric, angularsSymmetric, euclideansSymmetric,
} from './numkong-wasm.js';

// Auto-initialize: load the Emscripten glue relative to this module's URL,
// instantiate the WASM module, and wire up the wrapper before any export is used.
const glueUrl = new URL('./numkong-emscripten.js', import.meta.url);
const { default: NumKongModule } = await import(glueUrl.href);
const wasmInstance = await NumKongModule({
    locateFile: (path: string) => new URL(path, glueUrl).href,
});
initWasm(wasmInstance);
