/**
 *  @file javascript/numkong-browser-extended.ts
 *  @author Ash Vardanian
 *  @date October 31, 2026
 *  @brief Browser ESM entry point for the extended NumKong WASM module, with Memory64, threads and
 *      Relaxed SIMD.
 *
 *  An opt-in twin of `numkong/wasm`: it loads `wasm64-v128relaxed` on import through a top-level
 *  await, where the engine validates Memory64 and Relaxed SIMD and the page can share memory, and
 *  throws otherwise, so a caller can fall back to `numkong/wasm`.
 *
 *  @example
 *  ```js
 *  const numkong = await import('numkong/wasm-extended').catch(() => import('numkong/wasm'));
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
import { detectWasmExtended } from './wasm-probes.js';

if (!detectWasmExtended())
    throw new Error('The extended NumKong module needs Memory64, Relaxed SIMD and a cross-origin isolated page');

const { default: NumKongModule } = await import('numkong/wasm64-v128relaxed');
initWasm(await NumKongModule());
