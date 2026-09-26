/**
 *  @file javascript/numkong.ts
 *  @author Ash Vardanian
 *  @date January 18, 2024
 *  @brief Portable mixed-precision BLAS-like vector math library.
 *
 *  NumKong provides SIMD-accelerated distance metrics and vector operations for x86, ARM, RISC-V,
 *  and WASM platforms. The library automatically detects and uses the best available SIMD
 *  instruction set at runtime.
 *
 *  @example
 *  ```typescript
 *  import { dot, euclidean, Float16Array } from 'numkong';
 *
 *  // Auto-detected types
 *  const a = new Float32Array([1, 2, 3]);
 *  const b = new Float32Array([4, 5, 6]);
 *  dot(a, b);        // 32
 *  euclidean(a, b);  // 5.196...
 *
 *  // Custom types with explicit dtype
 *  const c = new Float16Array([1, 2, 3]);
 *  const d = new Float16Array([4, 5, 6]);
 *  dot(c, d, DType.F16); // 32
 *  ```
 *
 *  @packageDocumentation
 */

import build from "node-gyp-build";
import { createRequire } from "node:module";
import * as path from "node:path";
import { existsSync } from "node:fs";
import { getFileName, getRoot } from "bindings";
import { setConversionFunctions, Float16Array, BFloat16Array, E4M3Array, E5M2Array, BinaryArray, TensorBase, VectorBase, VectorView, Vector, MatrixBase, Matrix, PackedMatrix, DType, dtypeToString, dimensionsToValues, outputDType, KernelFamily } from "./types.js";

function loadNativeAddon(): any {
  // Duplicate-libomp guard. We ship our own `libomp.dylib` next to
  // `numkong.node` in each `@numkong/darwin-*` package, but another OpenMP
  // runtime (e.g. one loaded by another native addon) may already be
  // resident. `KMP_DUPLICATE_LIB_OK=TRUE` tells LLVM libomp / Intel
  // libiomp5 to coexist; it must be in `process.env` before the `require()`
  // below triggers the addon's `dlopen`, since libomp's constructor reads
  // the env during dependency resolution and is too late to influence
  // afterwards. Left unguarded because the variable is harmless on
  // platforms / runtimes (GCC libgomp) that don't recognize it, and a user
  // who set it to something else is respected by `??=`. See
  // `python/numkong/__init__.py` for the Python analog.
  process.env.KMP_DUPLICATE_LIB_OK ??= "TRUE";

  // Tier 1: platform-specific optional dependency (@numkong/<os>-<arch>)
  try {
    const req = createRequire(path.join(getDirName(), "noop.js"));
    return req(`@numkong/${process.platform}-${process.arch}`);
  } catch { }

  // Tier 2: node-gyp-build fallback (local dev, unsupported platform, build-from-source)
  try {
    return build(getBuildDir(getDirName()));
  } catch { }

  return null;
}

let addon: any = loadNativeAddon();

if (addon) {
  setConversionFunctions({
    castF16ToF32: addon.castF16ToF32,
    castF32ToF16: addon.castF32ToF16,
    castBF16ToF32: addon.castBF16ToF32,
    castF32ToBF16: addon.castF32ToBF16,
    castE4M3ToF32: addon.castE4M3ToF32,
    castF32ToE4M3: addon.castF32ToE4M3,
    castE5M2ToF32: addon.castE5M2ToF32,
    castF32ToE5M2: addon.castF32ToE5M2,
    cast: addon.cast,
  });
} else {
  throw new Error(
    "NumKong native addon not found. Install with `npm install numkong` (which fetches " +
    "the prebuilt binary), or build from source with `npm run install`. " +
    "For WASM, import from 'numkong/wasm' instead."
  );
}

export { Float16Array, BFloat16Array, E4M3Array, E5M2Array, BinaryArray, TensorBase, VectorBase, VectorView, Vector, MatrixBase, Matrix, PackedMatrix, outputDType };

/** Convert a single FP16 value, as uint16 bits, to FP32. */
export const castF16ToF32 = addon.castF16ToF32;
/** Convert a single FP32 value to FP16, returning uint16 bits. */
export const castF32ToF16 = addon.castF32ToF16;
/** Convert a single BF16 value, as uint16 bits, to FP32. */
export const castBF16ToF32 = addon.castBF16ToF32;
/** Convert a single FP32 value to BF16, returning uint16 bits. */
export const castF32ToBF16 = addon.castF32ToBF16;
/** Convert a single E4M3 value, as uint8 bits, to FP32. */
export const castE4M3ToF32 = addon.castE4M3ToF32;
/** Convert a single FP32 value to E4M3, returning uint8 bits. */
export const castF32ToE4M3 = addon.castF32ToE4M3;
/** Convert a single E5M2 value, as uint8 bits, to FP32. */
export const castE5M2ToF32 = addon.castE5M2ToF32;
/** Convert a single FP32 value to E5M2, returning uint8 bits. */
export const castF32ToE5M2 = addon.castF32ToE5M2;
/** Bulk conversion between different numeric types; modifies the destination array in place. */
export const cast = addon.cast;

export { DType };

/** Numeric arrays supported by distance metrics with auto-detected dtype.
 *
 *  These standard TypedArrays are auto-detected by the N-API binding. */
export type NumericArray = Float64Array | Float32Array | Int8Array | Uint8Array;

/** Extended array types supported by distance metrics with explicit dtype parameter.
 *
 *  Includes Uint16Array, the backing type for Float16Array and BFloat16Array, in addition to the
 *  auto-detected types. Pass a dtype string as the third argument to distance functions when using
 *  custom types. */
export type DistanceArray = Float64Array | Float32Array | Int8Array | Uint8Array | Uint16Array;

/** Union type for all array types, including custom types for conversions. */
export type NumKongArray =
  | Float64Array
  | Float32Array
  | Float16Array
  | BFloat16Array
  | E4M3Array
  | E5M2Array
  | Int8Array
  | Uint8Array
  | BinaryArray;

/** Extract a TypedArray from a TensorBase for the N-API backend.
 *
 *  The native backend doesn't benefit from zero-copy TensorBase, since Node.js TypedArrays already
 *  share process memory, but accepting TensorBase keeps the API uniform. */
function unwrapTensor(input: TensorBase): { arr: DistanceArray; dtype: DType } {
  switch (input.dtype) {
    case DType.F64: return { arr: new Float64Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    case DType.F32: return { arr: new Float32Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    case DType.F16: case DType.BF16: return { arr: new Uint16Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    case DType.I8: return { arr: new Int8Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    case DType.U8: case DType.U1: return { arr: new Uint8Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
    default: return { arr: new Uint8Array(input.buffer, input.byteOffset, input.length), dtype: input.dtype };
  }
}

/**
 *  Returns the CPU capabilities this machine executes, as a bitmask.
 *
 *  Describes the machine only, and says nothing about whether a kernel was compiled into this build
 *  — a prebuild whose ISA probes failed still reports your CPU's full feature set while containing
 *  no SIMD kernels at all. Prefer {@link capabilitiesEnabled}.
 *
 *  @returns Bitmask of {@link Capability} bits.
 */
export const capabilitiesDetected = (): bigint => addon.capabilitiesDetected();

/**
 *  Returns the CPU capabilities whose kernels were compiled into this binary, as a bitmask.
 *
 *  Decided at build time by the ISA probes, independent of the CPU. The only accessor that can tell
 *  you a prebuild is silently scalar.
 *
 *  @returns Bitmask of {@link Capability} bits.
 */
export const capabilitiesCompiled = (): bigint => addon.capabilitiesCompiled();

/**
 *  Returns the CPU capabilities dispatch uses, as a bitmask.
 *
 *  Both {@link capabilitiesDetected} and {@link capabilitiesCompiled} at once, unless narrowed by
 *  {@link capabilitiesEnable}. Always includes `Capability.serial`.
 *
 *  @returns Bitmask of {@link Capability} bits.
 *
 *  @example
 *  ```ts
 *  import { capabilitiesEnabled, Capability } from 'numkong';
 *
 *  const enabled = capabilitiesEnabled();
 *  if (enabled & Capability.haswell) console.log('AVX2 kernels in use');
 *  ```
 */
export const capabilitiesEnabled = (): bigint => addon.capabilitiesEnabled();

/**
 *  Makes `wanted` the set dispatch uses, clamped to {@link capabilitiesDetected} and
 *  {@link capabilitiesCompiled}. The serial fallback is always kept.
 *
 *  @param wanted - Bitmask of {@link Capability} bits.
 *  @returns The enabled set that took effect.
 *
 *  @example
 *  ```ts
 *  import { capabilitiesEnable, capabilitiesEnabled, Capability } from 'numkong';
 *
 *  capabilitiesEnable(capabilitiesEnabled() & ~Capability.skylake);
 *  ```
 */
export const capabilitiesEnable = (wanted: bigint): bigint => addon.capabilitiesEnable(wanted);

/** Lowercase CPU tier names, like `haswell` or `neon`, mapped to their capability bits. */
export const Capability: Readonly<Record<string, bigint>> = Object.freeze(addon.Capability);

/**
 *  Computes the squared Euclidean distance between two vectors.
 *  @param a - The first vector, as TypedArray or TensorBase.
 *  @param b - The second vector, matching the type and length of `a`.
 *  @param dtype - Optional dtype string for custom types, e.g. 'f16', 'bf16', 'e4m3'.
 *  @returns The squared Euclidean distance between `a` and `b`.
 */
export function sqeuclidean(a: NumericArray, b: NumericArray): number;
export function sqeuclidean(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function sqeuclidean(a: TensorBase, b: TensorBase): number;
export function sqeuclidean(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.sqeuclidean(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.sqeuclidean(a, b, dtypeToString(dtype)) : addon.sqeuclidean(a, b);
}

/**
 *  Computes the Euclidean distance between two vectors.
 *  @param a - The first vector, as TypedArray or TensorBase.
 *  @param b - The second vector, matching the type and length of `a`.
 *  @param dtype - Optional dtype string for custom types, e.g. 'f16', 'bf16', 'e4m3'.
 *  @returns The Euclidean distance between `a` and `b`.
 */
export function euclidean(a: NumericArray, b: NumericArray): number;
export function euclidean(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function euclidean(a: TensorBase, b: TensorBase): number;
export function euclidean(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.euclidean(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.euclidean(a, b, dtypeToString(dtype)) : addon.euclidean(a, b);
}

/**
 *  Computes the angular distance between two vectors.
 *  @param a - The first vector, as TypedArray or TensorBase.
 *  @param b - The second vector, matching the type and length of `a`.
 *  @param dtype - Optional dtype string for custom types, e.g. 'f16', 'bf16', 'e4m3'.
 *  @returns The angular distance between `a` and `b`.
 */
export function angular(a: NumericArray, b: NumericArray): number;
export function angular(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function angular(a: TensorBase, b: TensorBase): number;
export function angular(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.angular(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.angular(a, b, dtypeToString(dtype)) : addon.angular(a, b);
}

/**
 *  Computes the inner product of two vectors, the same as the dot product.
 *  @param a - The first vector, as TypedArray or TensorBase.
 *  @param b - The second vector, matching the type and length of `a`.
 *  @param dtype - Optional dtype string for custom types, e.g. 'f16', 'bf16', 'e4m3'.
 *  @returns The inner product of `a` and `b`.
 */
export function inner(a: NumericArray, b: NumericArray): number;
export function inner(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function inner(a: TensorBase, b: TensorBase): number;
export function inner(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.inner(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.inner(a, b, dtypeToString(dtype)) : addon.inner(a, b);
}

/**
 *  Computes the dot product of two vectors, the same as the inner product.
 *  @param a - The first vector, as TypedArray or TensorBase.
 *  @param b - The second vector, matching the type and length of `a`.
 *  @param dtype - Optional dtype string for custom types, e.g. 'f16', 'bf16', 'e4m3'.
 *  @returns The dot product of `a` and `b`.
 */
export function dot(a: NumericArray, b: NumericArray): number;
export function dot(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function dot(a: TensorBase, b: TensorBase): number;
export function dot(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.dot(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.dot(a, b, dtypeToString(dtype)) : addon.dot(a, b);
}

/**
 *  Computes the bitwise Hamming distance between two vectors.
 *
 *  Both vectors are treated as bit-packed, u1 dtype, where each byte holds 8 bits.
 *
 *  Use {@link toBinary} to convert numeric arrays to bit-packed format.
 *
 *  @param a - The first bit-packed vector, as Uint8Array, BinaryArray, or TensorBase.
 *  @param b - The second bit-packed vector, matching the length of `a`.
 *  @returns The Hamming distance, the number of differing bits, between `a` and `b`.
 */
export const hamming = (a: Uint8Array | BinaryArray | TensorBase, b: Uint8Array | BinaryArray | TensorBase): number => {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.hamming(u.arr, v.arr); }
  return addon.hamming(a, b);
};

/**
 *  Computes the bitwise Jaccard distance between two vectors.
 *
 *  Both vectors are treated as bit-packed, u1 dtype, where each byte holds 8 bits.
 *
 *  Use {@link toBinary} to convert numeric arrays to bit-packed format.
 *
 *  @param a - The first bit-packed vector, as Uint8Array, BinaryArray, or TensorBase.
 *  @param b - The second bit-packed vector, matching the length of `a`.
 *  @returns The Jaccard distance, 1 minus the Jaccard similarity, between `a` and `b`.
 */
export const jaccard = (a: Uint8Array | BinaryArray | TensorBase, b: Uint8Array | BinaryArray | TensorBase): number => {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.jaccard(u.arr, v.arr); }
  return addon.jaccard(a, b);
};

/**
 *  Computes the Kullback-Leibler divergence between two probability distributions.
 *
 *  Both vectors must represent valid probability distributions, non-negative and summing to 1.
 *  Supports f64 and f32, auto-detected, and f16 and bf16 with an explicit dtype.
 *
 *  @param a - The first probability distribution, as Float32Array, Float64Array, or TensorBase.
 *  @param b - The second probability distribution, matching the type and length of `a`.
 *  @param dtype - Optional dtype string for custom types, e.g. 'f16', 'bf16'.
 *  @returns The Kullback-Leibler divergence KL(a || b) = Σ a[i] · log(a[i] / b[i]).
 */
export function kullbackleibler(a: Float64Array | Float32Array, b: Float64Array | Float32Array): number;
export function kullbackleibler(a: Float64Array | Float32Array | Uint16Array, b: Float64Array | Float32Array | Uint16Array, dtype: DType): number;
export function kullbackleibler(a: TensorBase, b: TensorBase): number;
export function kullbackleibler(a: Float64Array | Float32Array | Uint16Array | TensorBase, b: Float64Array | Float32Array | Uint16Array | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.kullbackleibler(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.kullbackleibler(a, b, dtypeToString(dtype)) : addon.kullbackleibler(a, b);
}

/**
 *  Computes the Jensen-Shannon distance between two probability distributions.
 *
 *  Both vectors must represent valid probability distributions, non-negative and summing to 1.
 *  Supports f64 and f32, auto-detected, and f16 and bf16 with an explicit dtype. JSD is the square
 *  root of the symmetrized KL divergence:
 *  d_JS(a, b) = √(0.5 × (KL(a‖m) + KL(b‖m))), where m = (a + b) / 2.
 *
 *  @param a - The first probability distribution, as Float32Array, Float64Array, or TensorBase.
 *  @param b - The second probability distribution, matching the type and length of `a`.
 *  @param dtype - Optional dtype string for custom types, e.g. 'f16', 'bf16'.
 *  @returns The Jensen-Shannon distance between `a` and `b`.
 */
export function jensenshannon(a: Float64Array | Float32Array, b: Float64Array | Float32Array): number;
export function jensenshannon(a: Float64Array | Float32Array | Uint16Array, b: Float64Array | Float32Array | Uint16Array, dtype: DType): number;
export function jensenshannon(a: TensorBase, b: TensorBase): number;
export function jensenshannon(a: Float64Array | Float32Array | Uint16Array | TensorBase, b: Float64Array | Float32Array | Uint16Array | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.jensenshannon(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.jensenshannon(a, b, dtypeToString(dtype)) : addon.jensenshannon(a, b);
}

/**
 *  Quantizes a numeric vector into a bit-packed binary representation.
 *
 *  Converts each element to a single bit, 1 for positive values and 0 for non-positive ones, with
 *  element 0 in the least significant bit of byte 0, matching {@link BinaryArray} and the C
 *  `nk_u1x8_t`. This is the required format for {@link hamming} and {@link jaccard} distance
 *  functions. Dimension count must be a multiple of 8, the dimensions per byte.
 *
 *  @param vector - The vector to quantize and pack.
 *  @returns A bit-packed array where each byte contains 8 binary values.
 *
 *  @example
 *  ```ts
 *  const vec = new Float32Array([1.5, -2.3, 0.0, 3.1, -1.0, 2.0, 0.5, -0.5]);
 *  const binary = toBinary(vec);
 *  // Result: Uint8Array([0b01101001]) = [0x69]
 *  //   bits 0..7: [1, 0, 0, 1, 0, 1, 1, 0] for elements [+, -, 0, +, -, +, +, -]
 *
 *  // Use with Hamming distance
 *  const a = toBinary(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
 *  const b = toBinary(new Float32Array([1, -2, 3, 4, 5, 6, 7, 8]));
 *  const dist = hamming(a, b); // Counts differing bits
 *  ```
 */
export const toBinary = (vector: Float32Array | Float64Array | Int8Array): Uint8Array => {
  const packedVector = new Uint8Array(dimensionsToValues(DType.U1, vector.length));

  for (let i = 0; i < vector.length; i++) {
    if (vector[i] > 0) {
      packedVector[i >>> 3] |= 1 << (i & 7);
    }
  }

  return packedVector;
};

/** Extract a TypedArray from a Matrix for passing to the N-API backend. */
function unwrapMatrix(matrix: Matrix): { array: DistanceArray; dtype: DType } {
  switch (matrix.dtype) {
    case DType.F64: return { array: new Float64Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols), dtype: matrix.dtype };
    case DType.F32: return { array: new Float32Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols), dtype: matrix.dtype };
    case DType.F16: case DType.BF16: return { array: new Uint16Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols), dtype: matrix.dtype };
    case DType.I8: return { array: new Int8Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols), dtype: matrix.dtype };
    case DType.U8: return { array: new Uint8Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols), dtype: matrix.dtype };
    default: return { array: new Uint8Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols), dtype: matrix.dtype };
  }
}

/** Extract a result TypedArray from a Matrix matching its output dtype. */
function unwrapResultMatrix(matrix: Matrix): Float64Array | Float32Array | Int32Array | Uint32Array {
  switch (matrix.dtype) {
    case DType.F64: return new Float64Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols);
    case DType.F32: return new Float32Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols);
    case DType.I32: return new Int32Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols);
    case DType.U32: return new Uint32Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols);
    default: return new Float64Array(matrix.buffer, matrix.byteOffset, matrix.rows * matrix.cols);
  }
}

/**
 *  Queries the packed buffer's byte count for a matrix with the given __[width,depth]__ and dtype,
 *  matching what `dotsPack` would allocate.
 *  @param width - Number of vectors being packed, the matrix rows.
 *  @param depth - Dimensionality per vector, the matrix columns.
 *  @param dtype - Element dtype of the source matrix.
 *  @returns The byte count of the packed buffer.
 */
export function dotsPackedSize(width: number, depth: number, dtype: DType): number {
  return addon.dotsPackedSize(width, depth, dtypeToString(dtype));
}

/**
 *  Packs a Matrix for use with packed GEMM-like operations, such as `dotsPacked`.
 *  @param matrix - The matrix to pack, __[rows,columns]__ shaped.
 *  @returns The packed matrix, sized via `dotsPackedSize`.
 */
export function dotsPack(matrix: Matrix): PackedMatrix {
  const { array, dtype } = unwrapMatrix(matrix);
  const result = addon.dotsPack(array, matrix.rows, matrix.cols, matrix.rowStride, dtypeToString(dtype));
  return new PackedMatrix(result.buffer, result.width, result.depth, matrix.dtype, result.byteLength);
}

function packedOperation(compiledName: string, family: KernelFamily, a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  if (a.cols !== packed.depth) {
    throw new Error(`Matrix cols (${a.cols}) must match packed depth (${packed.depth})`);
  }
  const outDType = outputDType(family, a.dtype);
  if (!out) {
    out = new Matrix(a.rows, packed.width, outDType);
  }
  const aUnwrapped = unwrapMatrix(a);
  const resultArray = unwrapResultMatrix(out);
  (addon as any)[compiledName](
    aUnwrapped.array, packed.buffer, resultArray,
    a.rows, packed.width, a.cols,
    a.rowStride, out.rowStride,
    dtypeToString(a.dtype),
  );
  return out;
}

function symmetricOperation(compiledName: string, family: KernelFamily, vectors: Matrix, out?: Matrix, rowStart = 0, rowCount?: number): Matrix {
  const count = rowCount ?? vectors.rows - rowStart;
  const outDType = outputDType(family, vectors.dtype);
  if (!out) {
    out = new Matrix(vectors.rows, vectors.rows, outDType);
  }
  const vectorsUnwrapped = unwrapMatrix(vectors);
  const resultArray = unwrapResultMatrix(out);
  (addon as any)[compiledName](
    vectorsUnwrapped.array, resultArray,
    vectors.rows, vectors.cols,
    vectors.rowStride, out.rowStride,
    rowStart, count,
    dtypeToString(vectors.dtype),
  );
  return out;
}

/**
 *  Computes the dot products between every row of `a` and every packed vector in `packed`.
 *  @param a - The query matrix, __[rows,columns]__ shaped, columns matching packed's depth.
 *  @param packed - The packed matrix produced by `dotsPack`.
 *  @param out - Optional output matrix to write into, __[a.rows,packed.width]__ shaped.
 *  @returns The distance matrix: `out` when given, otherwise a newly allocated Matrix.
 */
export function dotsPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return packedOperation('dotsPacked', 'dots', a, packed, out);
}

/**
 *  Computes the angular distances between every row of `a` and every packed vector in `packed`.
 *  @param a - The query matrix, __[rows,columns]__ shaped, columns matching packed's depth.
 *  @param packed - The packed matrix produced by `dotsPack`.
 *  @param out - Optional output matrix to write into, __[a.rows,packed.width]__ shaped.
 *  @returns The distance matrix: `out` when given, otherwise a newly allocated Matrix.
 */
export function angularsPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return packedOperation('angularsPacked', 'angulars', a, packed, out);
}

/**
 *  Computes the Euclidean distances between every row of `a` and every packed vector in `packed`.
 *  @param a - The query matrix, __[rows,columns]__ shaped, columns matching packed's depth.
 *  @param packed - The packed matrix produced by `dotsPack`.
 *  @param out - Optional output matrix to write into, __[a.rows,packed.width]__ shaped.
 *  @returns The distance matrix: `out` when given, otherwise a newly allocated Matrix.
 */
export function euclideansPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return packedOperation('euclideansPacked', 'euclideans', a, packed, out);
}

/**
 *  Computes the all-pairs dot products between every row of `vectors` and every other row.
 *  @param vectors - The matrix of vectors, __[rows,columns]__ shaped.
 *  @param out - Optional output matrix to write into, __[vectors.rows,vectors.rows]__ shaped.
 *  @param options - Optional row range: `rowStart` and `rowCount` restrict which rows of the
 *      symmetric matrix are computed, defaulting to all rows.
 *  @returns The distance matrix: `out` when given, otherwise a newly allocated Matrix.
 */
export function dotsSymmetric(vectors: Matrix, out?: Matrix, options?: { rowStart?: number; rowCount?: number }): Matrix {
  return symmetricOperation('dotsSymmetric', 'dots', vectors, out, options?.rowStart ?? 0, options?.rowCount);
}

/**
 *  Computes the all-pairs angular distances between every row of `vectors` and every other row.
 *  @param vectors - The matrix of vectors, __[rows,columns]__ shaped.
 *  @param out - Optional output matrix to write into, __[vectors.rows,vectors.rows]__ shaped.
 *  @param options - Optional row range: `rowStart` and `rowCount` restrict which rows of the
 *      symmetric matrix are computed, defaulting to all rows.
 *  @returns The distance matrix: `out` when given, otherwise a newly allocated Matrix.
 */
export function angularsSymmetric(vectors: Matrix, out?: Matrix, options?: { rowStart?: number; rowCount?: number }): Matrix {
  return symmetricOperation('angularsSymmetric', 'angulars', vectors, out, options?.rowStart ?? 0, options?.rowCount);
}

/**
 *  Computes the all-pairs Euclidean distances between every row of `vectors` and every other row.
 *  @param vectors - The matrix of vectors, __[rows,columns]__ shaped.
 *  @param out - Optional output matrix to write into, __[vectors.rows,vectors.rows]__ shaped.
 *  @param options - Optional row range: `rowStart` and `rowCount` restrict which rows of the
 *      symmetric matrix are computed, defaulting to all rows.
 *  @returns The distance matrix: `out` when given, otherwise a newly allocated Matrix.
 */
export function euclideansSymmetric(vectors: Matrix, out?: Matrix, options?: { rowStart?: number; rowCount?: number }): Matrix {
  return symmetricOperation('euclideansSymmetric', 'euclideans', vectors, out, options?.rowStart ?? 0, options?.rowCount);
}

export default {
  dot,
  inner,
  sqeuclidean,
  euclidean,
  angular,
  hamming,
  jaccard,
  kullbackleibler,
  jensenshannon,
  toBinary,
  Float16Array,
  BFloat16Array,
  E4M3Array,
  E5M2Array,
  BinaryArray,
  TensorBase,
  VectorBase,
  VectorView,
  Vector,
  MatrixBase,
  Matrix,
  PackedMatrix,
  castF16ToF32,
  castF32ToF16,
  castBF16ToF32,
  castF32ToBF16,
  castE4M3ToF32,
  castF32ToE4M3,
  castE5M2ToF32,
  castF32ToE5M2,
  cast,
  dotsPack,
  dotsPacked,
  angularsPacked,
  euclideansPacked,
  dotsSymmetric,
  angularsSymmetric,
  euclideansSymmetric,
  dotsPackedSize,
  outputDType,
};

/**
 *  Finds the directory where the native build of the numkong module is located.
 *  @param dir - The directory to start the search from.
 */
function getBuildDir(dir: string) {
  if (existsSync(path.join(dir, "build"))) return dir;
  if (existsSync(path.join(dir, "prebuilds"))) return dir;
  if (path.basename(dir) === ".next") {
    // special case for next.js on custom node (not vercel)
    const sideways = path.join(dir, "..", "node_modules", "numkong");
    if (existsSync(sideways)) return getBuildDir(sideways);
  }
  if (dir === "/") throw new Error("Could not find native build for numkong");
  return getBuildDir(path.join(dir, ".."));
}

function getDirName() {
  try {
    if (__dirname) return __dirname;
  } catch (e) { }
  // Fall back to cwd, which is typically the project root in dev and CI.
  // This helps runtimes like Deno and Bun where the `bindings` module's
  // V8 stack-trace hack may not resolve correctly.
  try {
    const cwd = process.cwd();
    if (existsSync(path.join(cwd, "build")) || existsSync(path.join(cwd, "prebuilds")))
      return cwd;
  } catch (e) { }
  return getRoot(getFileName());
}
