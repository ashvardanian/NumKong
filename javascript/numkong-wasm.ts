/**
 * @brief WASM wrapper for NumKong providing N-API compatible interface
 * @file javascript/numkong-wasm.ts
 * @date February 6, 2026
 *
 * This module wraps the Emscripten-compiled WASM module to provide the same
 * TypeScript API as the native N-API bindings. It handles:
 * - Zero-copy TensorBase interop for cross-module WASM sharing
 * - TypedArray type detection and dispatch
 * - Result extraction from WASM heap
 * - Error handling
 * - Both wasm32 and wasm64 (memory64) modes
 */

import { TensorBase, Matrix, PackedMatrix, DType, dtypeToString, outputDtype, KernelFamily } from './types.js';

/**
 * Emscripten module interface.
 * In wasm64 (memory64) mode, pointer/size params and returns become bigint.
 * We use `any` for pointer arguments to support both modes uniformly.
 */
interface EmscriptenModule {
  _malloc(size: any): any;
  _free(ptr: any): void;
  wasmMemory: { buffer: ArrayBuffer };

  // Distance functions - all use `any` for pointer/size args to support wasm32 (number) and wasm64 (bigint)
  _nk_dot_f32(a: any, b: any, n: any, result: any): void;
  _nk_angular_f32(a: any, b: any, n: any, result: any): void;
  _nk_sqeuclidean_f32(a: any, b: any, n: any, result: any): void;
  _nk_euclidean_f32(a: any, b: any, n: any, result: any): void;
  _nk_dot_f64(a: any, b: any, n: any, result: any): void;
  _nk_angular_f64(a: any, b: any, n: any, result: any): void;
  _nk_sqeuclidean_f64(a: any, b: any, n: any, result: any): void;
  _nk_euclidean_f64(a: any, b: any, n: any, result: any): void;
  _nk_dot_f16(a: any, b: any, n: any, result: any): void;
  _nk_angular_f16(a: any, b: any, n: any, result: any): void;
  _nk_sqeuclidean_f16(a: any, b: any, n: any, result: any): void;
  _nk_euclidean_f16(a: any, b: any, n: any, result: any): void;
  _nk_dot_bf16(a: any, b: any, n: any, result: any): void;
  _nk_angular_bf16(a: any, b: any, n: any, result: any): void;
  _nk_sqeuclidean_bf16(a: any, b: any, n: any, result: any): void;
  _nk_euclidean_bf16(a: any, b: any, n: any, result: any): void;
  _nk_dot_i8(a: any, b: any, n: any, result: any): void;
  _nk_angular_i8(a: any, b: any, n: any, result: any): void;
  _nk_sqeuclidean_i8(a: any, b: any, n: any, result: any): void;
  _nk_euclidean_i8(a: any, b: any, n: any, result: any): void;
  _nk_dot_u8(a: any, b: any, n: any, result: any): void;
  _nk_angular_u8(a: any, b: any, n: any, result: any): void;
  _nk_sqeuclidean_u8(a: any, b: any, n: any, result: any): void;
  _nk_euclidean_u8(a: any, b: any, n: any, result: any): void;
  _nk_hamming_u1(a: any, b: any, n: any, result: any): void;
  _nk_hamming_u8(a: any, b: any, n: any, result: any): void;
  _nk_jaccard_u1(a: any, b: any, n: any, result: any): void;
  _nk_jaccard_u16(a: any, b: any, n: any, result: any): void;
  _nk_kld_f32(a: any, b: any, n: any, result: any): void;
  _nk_kld_f64(a: any, b: any, n: any, result: any): void;
  _nk_jsd_f32(a: any, b: any, n: any, result: any): void;
  _nk_jsd_f64(a: any, b: any, n: any, result: any): void;
  _nk_capabilities_detected(): any;
  _nk_capabilities_compiled(): any;
  _nk_capabilities_available(): any;
  _nk_capabilities_enabled(): any;

  [key: string]: any;
}

/** Pointer type passed to raw C functions: number in wasm32, bigint in wasm64 */
type WasmPtr = number | bigint;

let Module: EmscriptenModule | null = null;

/**
 * Whether the WASM module uses memory64.
 * In memory64 mode, Emscripten wraps _malloc/_free to accept/return number,
 * but raw C function exports expect BigInt (i64) for pointer parameters.
 * nk_size_t is always i32 (number) in WASM since NK_IS_64BIT_=0.
 */
let isMemory64 = false;

// Pre-allocated 8-byte result buffer (covers f64/f32/i32/u32), allocated once in initWasm()
// Always a number (from Emscripten-wrapped _malloc), converted to WasmPtr for C calls
let resultPtr: number = 0;

// Heap views (created from wasmMemory buffer)
let HEAP32: Int32Array;
let HEAPU8: Uint8Array;
let HEAPU32: Uint32Array;
let HEAPF32: Float32Array;
let HEAPF64: Float64Array;

/**
 * Convert a number (e.g. from _malloc or byteOffset) to the pointer type
 * expected by raw C function exports. In wasm64, pointers are i64 (BigInt).
 */
function toWasmPtr(n: number): WasmPtr {
  return isMemory64 ? BigInt(n) : n;
}

/**
 * Initializes the WASM backend with an Emscripten module instance.
 * @param wasmModule - The Emscripten-compiled WASM module to use.
 */
export function initWasm(wasmModule: EmscriptenModule): void {
  Module = wasmModule;

  // Create heap views from the WASM memory buffer
  const buffer = wasmModule.wasmMemory.buffer;
  HEAP32 = new Int32Array(buffer);
  HEAPU8 = new Uint8Array(buffer);
  HEAPU32 = new Uint32Array(buffer);
  HEAPF32 = new Float32Array(buffer);
  HEAPF64 = new Float64Array(buffer);

  // Detect memory64 mode by probing whether raw C functions expect BigInt pointers.
  // Emscripten wraps _malloc/_free to always use number, but raw C exports use i64
  // (BigInt) for pointers in memory64 mode. We probe by calling a distance function
  // with BigInt(0) args — if it doesn't throw, we're in memory64 mode.
  try {
    const probe = wasmModule._malloc(8);
    wasmModule._nk_dot_f32(BigInt(probe), BigInt(probe), 0, BigInt(probe));
    isMemory64 = true;
    wasmModule._free(probe);
  } catch {
    isMemory64 = false;
  }

  // Pre-allocate an 8-byte result buffer (never freed during module lifetime)
  // _malloc always returns number (Emscripten-wrapped in both modes)
  resultPtr = wasmModule._malloc(8);
}

/**
 * Type information for dispatching
 */
interface TypeInfo {
  dtype: DType;
  bytesPerElement: number;
  heapView: 'HEAP32' | 'HEAPU8' | 'HEAPU32' | 'HEAPF32' | 'HEAPF64';
  resultType: 'f32' | 'f64' | 'i32' | 'u32';
}

/**
 * Detect dtype from TypedArray constructor
 */
function detectType(arr: any): TypeInfo {
  if (arr instanceof Float64Array) {
    return { dtype: DType.F64, bytesPerElement: 8, heapView: 'HEAPF64', resultType: 'f64' };
  } else if (arr instanceof Float32Array) {
    return { dtype: DType.F32, bytesPerElement: 4, heapView: 'HEAPF32', resultType: 'f64' };
  } else if (arr instanceof Int8Array) {
    return { dtype: DType.I8, bytesPerElement: 1, heapView: 'HEAPU8', resultType: 'i32' };
  } else if (arr instanceof Uint8Array) {
    return { dtype: DType.U8, bytesPerElement: 1, heapView: 'HEAPU8', resultType: 'u32' };
  }

  // Check for custom typed arrays from types.ts
  const constructorName = arr.constructor.name;

  if (constructorName === 'Float16Array') {
    return { dtype: DType.F16, bytesPerElement: 2, heapView: 'HEAPU8', resultType: 'f32' };
  } else if (constructorName === 'BFloat16Array') {
    return { dtype: DType.BF16, bytesPerElement: 2, heapView: 'HEAPU8', resultType: 'f32' };
  } else if (constructorName === 'E4M3Array') {
    throw new Error('E4M3 not yet supported in WASM backend');
  } else if (constructorName === 'E5M2Array') {
    throw new Error('E5M2 not yet supported in WASM backend');
  } else if (constructorName === 'BinaryArray') {
    return { dtype: DType.U1, bytesPerElement: 1, heapView: 'HEAPU8', resultType: 'u32' };
  }

  throw new Error(`Unsupported array type: ${constructorName}`);
}

/**
 * Get TypeInfo from a DType enum value.
 */
function typeInfoFromDtype(dtype: DType): TypeInfo {
  switch (dtype) {
    case DType.F64: return { dtype, bytesPerElement: 8, heapView: 'HEAPF64', resultType: 'f64' };
    case DType.F32: return { dtype, bytesPerElement: 4, heapView: 'HEAPF32', resultType: 'f64' };
    case DType.F16: return { dtype, bytesPerElement: 2, heapView: 'HEAPU8', resultType: 'f32' };
    case DType.BF16: return { dtype, bytesPerElement: 2, heapView: 'HEAPU8', resultType: 'f32' };
    case DType.I8: return { dtype, bytesPerElement: 1, heapView: 'HEAPU8', resultType: 'i32' };
    case DType.U8: return { dtype, bytesPerElement: 1, heapView: 'HEAPU8', resultType: 'u32' };
    case DType.U1: return { dtype, bytesPerElement: 1, heapView: 'HEAPU8', resultType: 'u32' };
    default: throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

/**
 * Flat struct carrying the fields needed for distance dispatch,
 * avoiding the VectorView constructor chain for raw TypedArrays.
 */
interface ResolvedInput {
  buffer: ArrayBuffer;
  byteOffset: number;
  length: number;
  byteLength: number;
  typeInfo: TypeInfo;
}

/**
 * Resolve an input that may be a TensorBase or a TypedArray into a uniform
 * ResolvedInput for distance dispatch.
 */
function resolveInput(a: TensorBase | any): ResolvedInput {
  if (a instanceof TensorBase) {
    return {
      buffer: a.buffer, byteOffset: a.byteOffset,
      length: a.length, byteLength: a.byteLength,
      typeInfo: typeInfoFromDtype(a.dtype),
    };
  }
  const typeInfo = detectType(a);
  return {
    buffer: a.buffer, byteOffset: a.byteOffset,
    length: a.length, byteLength: a.length * typeInfo.bytesPerElement,
    typeInfo,
  };
}

/**
 * Allocate WASM memory and copy data into it.
 * Returns a number byte-offset (from Emscripten-wrapped _malloc).
 */
function allocAndCopyResolved(buffer: ArrayBuffer, byteOffset: number, byteLength: number): number {
  if (!Module) throw new Error('WASM module not initialized');
  const ptr = Module._malloc(byteLength);
  const src = new Uint8Array(buffer, byteOffset, byteLength);
  HEAPU8.set(src, ptr);
  return ptr;
}

/**
 * Read result from WASM heap. ptr is a number byte-offset.
 */
function readResult(ptr: number, resultType: 'f32' | 'f64' | 'i32' | 'u32'): number {
  if (!Module) throw new Error('WASM module not initialized');

  switch (resultType) {
    case 'f64':
      return HEAPF64[ptr / 8];
    case 'f32':
      return HEAPF32[ptr / 4];
    case 'i32':
      return HEAP32[ptr / 4];
    case 'u32':
      return HEAPU32[ptr / 4];
  }
}

/**
 * Generic distance function wrapper.
 * Uses zero-copy when arrays already live on the WASM heap.
 */
function distance(metric: string, a: TensorBase | any, b: TensorBase | any): number {
  if (!Module) {
    throw new Error('WASM module not initialized. Call initWasm() first.');
  }

  const resolvedA = resolveInput(a);
  const resolvedB = resolveInput(b);

  if (resolvedA.length !== resolvedB.length) {
    throw new Error(`Array length mismatch: ${resolvedA.length} !== ${resolvedB.length}`);
  }

  const n = resolvedA.length;

  // Zero-copy: if the buffer IS the WASM memory, byteOffset is the pointer (number)
  const isOnHeapA = resolvedA.buffer === Module.wasmMemory.buffer;
  const isOnHeapB = resolvedB.buffer === Module.wasmMemory.buffer;
  const aOff = isOnHeapA ? resolvedA.byteOffset : allocAndCopyResolved(resolvedA.buffer, resolvedA.byteOffset, resolvedA.byteLength);
  const bOff = isOnHeapB ? resolvedB.byteOffset : allocAndCopyResolved(resolvedB.buffer, resolvedB.byteOffset, resolvedB.byteLength);

  try {
    // Call C function
    const fnName = `_nk_${metric}_${dtypeToString(resolvedA.typeInfo.dtype)}` as keyof EmscriptenModule;
    const fn = Module[fnName] as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error(`Function ${fnName} not available in WASM module`);
    }

    // In wasm64, raw C exports expect BigInt for pointer args; nk_size_t is always i32 (number)
    fn(toWasmPtr(aOff), toWasmPtr(bOff), n, toWasmPtr(resultPtr));

    // Read result
    return readResult(resultPtr, resolvedA.typeInfo.resultType);
  } finally {
    // _free is Emscripten-wrapped: always takes number
    if (!isOnHeapA) Module._free(aOff);
    if (!isOnHeapB) Module._free(bOff);
  }
}

/**
 * Computes the squared Euclidean distance between two vectors.
 * @param a - First vector (TypedArray or TensorBase).
 * @param b - Second vector (must match type and length of a).
 * @returns The squared Euclidean distance between a and b.
 */
export function sqeuclidean(a: TensorBase | any, b: TensorBase | any): number {
  return distance('sqeuclidean', a, b);
}

/**
 * Computes the Euclidean distance between two vectors.
 * @param a - First vector (TypedArray or TensorBase).
 * @param b - Second vector (must match type and length of a).
 * @returns The Euclidean distance between a and b.
 */
export function euclidean(a: TensorBase | any, b: TensorBase | any): number {
  return distance('euclidean', a, b);
}

/**
 * Computes the angular distance between two vectors.
 * @param a - First vector (TypedArray or TensorBase).
 * @param b - Second vector (must match type and length of a).
 * @returns The angular distance between a and b.
 */
export function angular(a: TensorBase | any, b: TensorBase | any): number {
  return distance('angular', a, b);
}

/**
 * Computes the dot product of two vectors.
 * @param a - First vector (TypedArray or TensorBase).
 * @param b - Second vector (must match type and length of a).
 * @returns The dot product of a and b.
 */
export function dot(a: TensorBase | any, b: TensorBase | any): number {
  return distance('dot', a, b);
}

/** Alias for {@link dot}. */
export const inner = dot;

/**
 * Computes the bitwise Hamming distance between two vectors.
 *
 * Following N-API behavior, always treats input as u1 (binary/bit-packed),
 * even if passed as Uint8Array. Each byte represents 8 bits.
 *
 * @param a - First bit-packed vector (Uint8Array or TensorBase).
 * @param b - Second bit-packed vector (must match length of a).
 * @returns The Hamming distance (number of differing bits) between a and b.
 */
export function hamming(a: TensorBase | Uint8Array | any, b: TensorBase | Uint8Array | any): number {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  // Extract flat fields; for raw TypedArrays treat as u1 (binary/bit-packed)
  const bufferA = a.buffer as ArrayBuffer, offsetA = a.byteOffset as number, lengthA = a.length as number;
  const bufferB = b.buffer as ArrayBuffer, offsetB = b.byteOffset as number, lengthB = b.length as number;
  const byteLengthA = a instanceof TensorBase ? a.byteLength : lengthA;
  const byteLengthB = b instanceof TensorBase ? b.byteLength : lengthB;

  if (lengthA !== lengthB) {
    throw new Error(`Array length mismatch: ${lengthA} !== ${lengthB}`);
  }

  const isOnHeapA = bufferA === Module.wasmMemory.buffer;
  const isOnHeapB = bufferB === Module.wasmMemory.buffer;
  const aOff = isOnHeapA ? offsetA : allocAndCopyResolved(bufferA, offsetA, byteLengthA);
  const bOff = isOnHeapB ? offsetB : allocAndCopyResolved(bufferB, offsetB, byteLengthB);

  try {
    const fn = Module._nk_hamming_u1 as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error('Function _nk_hamming_u1 not available in WASM module');
    }

    fn(toWasmPtr(aOff), toWasmPtr(bOff), lengthA, toWasmPtr(resultPtr));

    return readResult(resultPtr, 'u32');
  } finally {
    if (!isOnHeapA) Module._free(aOff);
    if (!isOnHeapB) Module._free(bOff);
  }
}

/**
 * Computes the bitwise Jaccard distance between two vectors.
 *
 * Following N-API behavior, always treats input as u1 (binary/bit-packed),
 * even if passed as Uint8Array. Each byte represents 8 bits.
 *
 * @param a - First bit-packed vector (Uint8Array or TensorBase).
 * @param b - Second bit-packed vector (must match length of a).
 * @returns The Jaccard distance (1 - Jaccard similarity) between a and b.
 */
export function jaccard(a: TensorBase | Uint8Array | any, b: TensorBase | Uint8Array | any): number {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  // Extract flat fields; for raw TypedArrays treat as u1 (binary/bit-packed)
  const bufferA = a.buffer as ArrayBuffer, offsetA = a.byteOffset as number, lengthA = a.length as number;
  const bufferB = b.buffer as ArrayBuffer, offsetB = b.byteOffset as number, lengthB = b.length as number;
  const byteLengthA = a instanceof TensorBase ? a.byteLength : lengthA;
  const byteLengthB = b instanceof TensorBase ? b.byteLength : lengthB;

  if (lengthA !== lengthB) {
    throw new Error(`Array length mismatch: ${lengthA} !== ${lengthB}`);
  }

  const isOnHeapA = bufferA === Module.wasmMemory.buffer;
  const isOnHeapB = bufferB === Module.wasmMemory.buffer;
  const aOff = isOnHeapA ? offsetA : allocAndCopyResolved(bufferA, offsetA, byteLengthA);
  const bOff = isOnHeapB ? offsetB : allocAndCopyResolved(bufferB, offsetB, byteLengthB);

  try {
    const fn = Module._nk_jaccard_u1 as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error('Function _nk_jaccard_u1 not available in WASM module');
    }

    fn(toWasmPtr(aOff), toWasmPtr(bOff), lengthA, toWasmPtr(resultPtr));

    return readResult(resultPtr, 'f32');
  } finally {
    if (!isOnHeapA) Module._free(aOff);
    if (!isOnHeapB) Module._free(bOff);
  }
}

/**
 * Computes the Kullback-Leibler divergence between two probability distributions.
 * @param a - First probability distribution (Float32Array, Float64Array, or TensorBase).
 * @param b - Second probability distribution (must match type and length of a).
 * @returns The KL divergence KL(a || b).
 */
export function kullbackleibler(a: TensorBase | Float64Array | Float32Array, b: TensorBase | Float64Array | Float32Array): number {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  const resolvedA = resolveInput(a);
  const resolvedB = resolveInput(b);

  if (resolvedA.length !== resolvedB.length) {
    throw new Error(`Array length mismatch: ${resolvedA.length} !== ${resolvedB.length}`);
  }

  const n = resolvedA.length;
  const isOnHeapA = resolvedA.buffer === Module.wasmMemory.buffer;
  const isOnHeapB = resolvedB.buffer === Module.wasmMemory.buffer;
  const aOff = isOnHeapA ? resolvedA.byteOffset : allocAndCopyResolved(resolvedA.buffer, resolvedA.byteOffset, resolvedA.byteLength);
  const bOff = isOnHeapB ? resolvedB.byteOffset : allocAndCopyResolved(resolvedB.buffer, resolvedB.byteOffset, resolvedB.byteLength);

  try {
    const fnName = `_nk_kld_${dtypeToString(resolvedA.typeInfo.dtype)}` as keyof EmscriptenModule;
    const fn = Module[fnName] as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error(`Function ${fnName} not available in WASM module`);
    }

    fn(toWasmPtr(aOff), toWasmPtr(bOff), n, toWasmPtr(resultPtr));

    return readResult(resultPtr, resolvedA.typeInfo.resultType);
  } finally {
    if (!isOnHeapA) Module._free(aOff);
    if (!isOnHeapB) Module._free(bOff);
  }
}

/**
 * Computes the Jensen-Shannon distance between two probability distributions.
 * @param a - First probability distribution (Float32Array, Float64Array, or TensorBase).
 * @param b - Second probability distribution (must match type and length of a).
 * @returns The Jensen-Shannon distance between a and b.
 */
export function jensenshannon(a: TensorBase | Float64Array | Float32Array, b: TensorBase | Float64Array | Float32Array): number {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }

  const resolvedA = resolveInput(a);
  const resolvedB = resolveInput(b);

  if (resolvedA.length !== resolvedB.length) {
    throw new Error(`Array length mismatch: ${resolvedA.length} !== ${resolvedB.length}`);
  }

  const n = resolvedA.length;
  const isOnHeapA = resolvedA.buffer === Module.wasmMemory.buffer;
  const isOnHeapB = resolvedB.buffer === Module.wasmMemory.buffer;
  const aOff = isOnHeapA ? resolvedA.byteOffset : allocAndCopyResolved(resolvedA.buffer, resolvedA.byteOffset, resolvedA.byteLength);
  const bOff = isOnHeapB ? resolvedB.byteOffset : allocAndCopyResolved(resolvedB.buffer, resolvedB.byteOffset, resolvedB.byteLength);

  try {
    const fnName = `_nk_jsd_${dtypeToString(resolvedA.typeInfo.dtype)}` as keyof EmscriptenModule;
    const fn = Module[fnName] as any;

    if (!fn || typeof fn !== 'function') {
      throw new Error(`Function ${fnName} not available in WASM module`);
    }

    fn(toWasmPtr(aOff), toWasmPtr(bOff), n, toWasmPtr(resultPtr));

    return readResult(resultPtr, resolvedA.typeInfo.resultType);
  } finally {
    if (!isOnHeapA) Module._free(aOff);
    if (!isOnHeapB) Module._free(bOff);
  }
}

/** In wasm64 the capability mask arrives as a bigint; in wasm32 as a number. */
function capabilitiesToBigInt(caps: any): bigint {
  return typeof caps === 'bigint' ? caps : BigInt(caps);
}

function requireModule(): any {
  if (!Module) {
    throw new Error('WASM module not initialized');
  }
  return Module;
}

/**
 * Returns the SIMD capabilities this WASM host supports, as a bitmask.
 * Describes the host only, and says nothing about what was compiled into this module.
 * @returns Bitmask of capability flags (use with Capability constants).
 */
export function getCapabilitiesDetected(): bigint {
  return capabilitiesToBigInt(requireModule()._nk_capabilities_detected());
}

/**
 * Returns the SIMD capabilities whose kernels were compiled into this module, as a bitmask.
 * @returns Bitmask of capability flags (use with Capability constants).
 */
export function getCapabilitiesCompiled(): bigint {
  return capabilitiesToBigInt(requireModule()._nk_capabilities_compiled());
}

/**
 * Returns the SIMD capabilities that can actually execute here, as a bitmask.
 * The intersection of getCapabilitiesDetected() and getCapabilitiesCompiled().
 * @returns Bitmask of capability flags (use with Capability constants).
 */
export function getCapabilitiesAvailable(): bigint {
  return capabilitiesToBigInt(requireModule()._nk_capabilities_available());
}

/**
 * Returns the SIMD capabilities dispatch is currently restricted to, as a bitmask.
 * A subset of getCapabilitiesAvailable().
 * @returns Bitmask of capability flags (use with Capability constants).
 */
export function getCapabilitiesEnabled(): bigint {
  return capabilitiesToBigInt(requireModule()._nk_capabilities_enabled());
}

/**
 * Checks whether a specific SIMD capability can actually execute here.
 * Tests against getCapabilitiesAvailable(), so it is false both when the host lacks the feature
 * and when its kernels were not compiled into this module.
 * @param cap - Capability flag to check (from Capability constants).
 * @returns True if the capability is available.
 */
export function hasCapability(cap: bigint): boolean {
  return (getCapabilitiesAvailable() & cap) !== 0n;
}

// FinalizationRegistry for WASM PackedMatrix cleanup (ES2021+, available in Node 14+)
declare class FinalizationRegistry<T> { constructor(callback: (value: T) => void); register(target: object, value: T): void; }
let packedRegistry: FinalizationRegistry<number> | null = null;

/**
 * WASM-backed PackedMatrix that owns a WASM heap allocation.
 */
class WasmPackedMatrix extends PackedMatrix {
  private _heapPointer: number;
  private _wasmDisposed: boolean = false;

  constructor(heapPointer: number, byteLength: number, width: number, depth: number, dtype: DType) {
    // Create a JS ArrayBuffer that is a copy of the WASM heap region for read access
    const buffer = new ArrayBuffer(byteLength);
    new Uint8Array(buffer).set(HEAPU8.subarray(heapPointer, heapPointer + byteLength));
    super(buffer, width, depth, dtype, byteLength);
    this._heapPointer = heapPointer;

    // Register with FinalizationRegistry as safety net
    if (!packedRegistry && typeof FinalizationRegistry !== 'undefined') {
      packedRegistry = new FinalizationRegistry((ptr: number) => {
        if (Module) Module._free(ptr);
      });
    }
    if (packedRegistry) {
      packedRegistry.register(this, heapPointer);
    }
  }

  get heapPointer(): number { return this._heapPointer; }

  dispose(): void {
    if (!this._wasmDisposed && Module) {
      Module._free(this._heapPointer);
      this._wasmDisposed = true;
    }
    // Call parent dispose to set disposed flag
    super.dispose();
  }
}

/**
 * Allocate WASM heap, copy matrix data, return pointer. Caller must free.
 */
function allocAndCopyMatrix(matrix: Matrix): number {
  const byteLength = matrix.rows * matrix.rowStride;
  return allocAndCopyResolved(matrix.buffer, matrix.byteOffset, byteLength);
}

/**
 * Query the packed buffer byte count for a given matrix shape and dtype.
 */
export function dotsPackedSize(width: number, depth: number, dtype: DType): number {
  if (!Module) throw new Error('WASM module not initialized');

  const fnName = `_nk_dots_packed_size_${dtypeToString(dtype)}`;
  const fn = Module[fnName] as any;
  if (!fn || typeof fn !== 'function') {
    throw new Error(`Function ${fnName} not available in WASM module`);
  }
  return fn(width, depth);
}

/**
 * Pack a Matrix for use with packed GEMM-like operations.
 */
export function dotsPack(matrix: Matrix): PackedMatrix {
  if (!Module) throw new Error('WASM module not initialized');

  const dtypeStr = dtypeToString(matrix.dtype);
  const sizeFnName = `_nk_dots_packed_size_${dtypeStr}`;
  const packFnName = `_nk_dots_pack_${dtypeStr}`;

  const sizeFn = Module[sizeFnName] as any;
  const packFn = Module[packFnName] as any;
  if (!sizeFn || !packFn) {
    throw new Error(`Pack functions not available for dtype ${dtypeStr}`);
  }

  const packedByteCount = sizeFn(matrix.rows, matrix.cols) as number;
  const packedPtr = Module._malloc(packedByteCount);
  const matrixPtr = allocAndCopyMatrix(matrix);

  try {
    packFn(
      toWasmPtr(matrixPtr),
      matrix.rows, matrix.cols,
      matrix.rowStride,
      toWasmPtr(packedPtr),
    );
  } finally {
    Module._free(matrixPtr);
  }

  return new WasmPackedMatrix(packedPtr, packedByteCount, matrix.rows, matrix.cols, matrix.dtype);
}

function wasmPackedOperation(metricPrefix: string, family: KernelFamily, a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  if (!Module) throw new Error('WASM module not initialized');
  if (a.cols !== packed.depth) {
    throw new Error(`Matrix cols (${a.cols}) must match packed depth (${packed.depth})`);
  }

  const outDtype = outputDtype(family, a.dtype);
  if (!out) {
    out = new Matrix(a.rows, packed.width, outDtype);
  }

  const dtypeStr = dtypeToString(a.dtype);
  const fnName = `_nk_${metricPrefix}_${dtypeStr}`;
  const fn = Module[fnName] as any;
  if (!fn || typeof fn !== 'function') {
    throw new Error(`Function ${fnName} not available in WASM module`);
  }

  const outBpe = out.bytesPerElement;
  const resultByteLength = out.rows * out.cols * outBpe;
  const aPtr = allocAndCopyMatrix(a);
  const resultPtr = Module._malloc(resultByteLength);

  // For WasmPackedMatrix, copy the packed data to heap; otherwise use the buffer directly
  let packedPtr: number;
  let packedAllocated = false;
  if (packed instanceof WasmPackedMatrix) {
    // The heap pointer may have been freed if disposed; re-copy from JS buffer
    packedPtr = allocAndCopyResolved(packed.buffer, 0, packed.byteLength);
    packedAllocated = true;
  } else {
    packedPtr = allocAndCopyResolved(packed.buffer, 0, packed.byteLength);
    packedAllocated = true;
  }

  try {
    fn(
      toWasmPtr(aPtr), toWasmPtr(packedPtr), toWasmPtr(resultPtr),
      a.rows, packed.width, a.cols,
      a.rowStride, out.rowStride,
    );

    // Copy result back
    const outArray = new Uint8Array(out.buffer, out.byteOffset, resultByteLength);
    outArray.set(HEAPU8.subarray(resultPtr, resultPtr + resultByteLength));
  } finally {
    Module._free(aPtr);
    Module._free(resultPtr);
    if (packedAllocated) Module._free(packedPtr);
  }

  return out;
}

function wasmSymmetricOperation(metricPrefix: string, family: KernelFamily, vectors: Matrix, out?: Matrix, rowStart = 0, rowCount?: number): Matrix {
  if (!Module) throw new Error('WASM module not initialized');

  const count = rowCount ?? vectors.rows - rowStart;
  const outDtype = outputDtype(family, vectors.dtype);
  if (!out) {
    out = new Matrix(vectors.rows, vectors.rows, outDtype);
  }

  const dtypeStr = dtypeToString(vectors.dtype);
  const fnName = `_nk_${metricPrefix}_${dtypeStr}`;
  const fn = Module[fnName] as any;
  if (!fn || typeof fn !== 'function') {
    throw new Error(`Function ${fnName} not available in WASM module`);
  }

  const resultByteLength = out.rows * out.cols * out.bytesPerElement;
  const vectorsPtr = allocAndCopyMatrix(vectors);
  const resultPtr = Module._malloc(resultByteLength);

  try {
    fn(
      toWasmPtr(vectorsPtr),
      vectors.rows, vectors.cols,
      vectors.rowStride,
      toWasmPtr(resultPtr), out.rowStride,
      rowStart, count,
    );

    // Copy result back
    const outArray = new Uint8Array(out.buffer, out.byteOffset, resultByteLength);
    outArray.set(HEAPU8.subarray(resultPtr, resultPtr + resultByteLength));
  } finally {
    Module._free(vectorsPtr);
    Module._free(resultPtr);
  }

  return out;
}

export function dotsPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return wasmPackedOperation('dots_packed', 'dots', a, packed, out);
}

export function angularsPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return wasmPackedOperation('angulars_packed', 'angulars', a, packed, out);
}

export function euclideansPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return wasmPackedOperation('euclideans_packed', 'euclideans', a, packed, out);
}

export function dotsSymmetric(vectors: Matrix, out?: Matrix, options?: { rowStart?: number; rowCount?: number }): Matrix {
  return wasmSymmetricOperation('dots_symmetric', 'dots', vectors, out, options?.rowStart ?? 0, options?.rowCount);
}

export function angularsSymmetric(vectors: Matrix, out?: Matrix, options?: { rowStart?: number; rowCount?: number }): Matrix {
  return wasmSymmetricOperation('angulars_symmetric', 'angulars', vectors, out, options?.rowStart ?? 0, options?.rowCount);
}

export function euclideansSymmetric(vectors: Matrix, out?: Matrix, options?: { rowStart?: number; rowCount?: number }): Matrix {
  return wasmSymmetricOperation('euclideans_symmetric', 'euclideans', vectors, out, options?.rowStart ?? 0, options?.rowCount);
}
