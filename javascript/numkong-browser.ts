/**
 *  @file javascript/numkong-browser.ts
 *  @author Ash Vardanian
 *  @date March 20, 2026
 *  @brief Self-contained browser ESM entry point for NumKong WASM.
 *
 *  Auto-initializes the Emscripten module on import through a top-level await, probing the engine
 *  first. Each wasm32 module, `serial`, `v128` or `v128relaxed`, is its own package subpath, so a
 *  bundler splits them and fetches only the one the engine validates.
 *
 *  @example
 *  ```js
 *  import { dot, euclidean } from 'numkong/wasm';
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
    Device, Capability,
    dotsPack, dotsPackedSize,
    dotsPacked, angularsPacked, euclideansPacked,
    dotsSymmetric, angularsSymmetric, euclideansSymmetric,
} from './numkong-wasm.js';
export type { DeviceKind } from './numkong-wasm.js';
import { detectWasmSimdTier, type WasmSimdTier } from './wasm-probes.js';

/** Literal subpaths, which bundlers and Node resolve through the package `exports`. */
const modules: Record<WasmSimdTier, () => Promise<{ default: () => Promise<any> }>> = {
    v128relaxed: () => import('numkong/wasm32-v128relaxed'),
    v128: () => import('numkong/wasm32-v128'),
    serial: () => import('numkong/wasm32-serial'),
};

const { default: NumKongModule } = await modules[detectWasmSimdTier()]();
initWasm(await NumKongModule());
