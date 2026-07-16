/**
 * @fileoverview NumKong - Portable mixed-precision BLAS-like vector math library
 *
 * NumKong provides SIMD-accelerated distance metrics and vector operations for
 * x86, ARM, RISC-V, and WASM platforms. The library automatically detects and uses
 * the best available SIMD instruction set at runtime.
 *
 * @module numkong
 * @author Ash Vardanian
 *
 * @example
 * ```typescript
 * import { dot, euclidean, Float16Array } from 'numkong';
 *
 * // Auto-detected types
 * const a = new Float32Array([1, 2, 3]);
 * const b = new Float32Array([4, 5, 6]);
 * dot(a, b);        // 32
 * euclidean(a, b);  // 5.196...
 *
 * // Custom types with explicit dtype
 * const c = new Float16Array([1, 2, 3]);
 * const d = new Float16Array([4, 5, 6]);
 * dot(c, d, DType.F16); // 32
 * ```
 */

import build from "node-gyp-build";
import { createRequire } from "node:module";
import * as path from "node:path";
import { existsSync } from "node:fs";
import { getFileName, getRoot } from "bindings";
import { setConversionFunctions, Float16Array, BFloat16Array, E4M3Array, E5M2Array, BinaryArray, TensorBase, VectorBase, VectorView, Vector, MatrixBase, Matrix, PackedMatrix, DType, dtypeToString, outputDtype, KernelFamily } from "./types.js";

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

/**
 * CPU capability bit masks in chronological order (by first commercial silicon).
 * Use these with `capabilities.has()` or `capabilities.available()` to check for SIMD support.
 */
export const Capability = {
  SERIAL: 1n << 0n,          // Always: Fallback
  NEON: 1n << 1n,            // 2013: ARM NEON
  HASWELL: 1n << 2n,         // 2013: Intel AVX2
  SKYLAKE: 1n << 3n,         // 2017: Intel AVX-512
  NEONHALF: 1n << 4n,        // 2017: ARM NEON FP16
  NEONSDOT: 1n << 5n,        // 2017: ARM NEON i8 dot
  NEONFHM: 1n << 6n,         // 2018: ARM NEON FP16 FML
  ICELAKE: 1n << 7n,         // 2019: Intel AVX-512 VNNI
  GENOA: 1n << 8n,           // 2020: Intel/AMD AVX-512 BF16
  NEONBFDOT: 1n << 9n,       // 2020: ARM NEON BF16
  SVE: 1n << 10n,            // 2020: ARM SVE
  SVEHALF: 1n << 11n,        // 2020: ARM SVE FP16
  SVESDOT: 1n << 12n,        // 2020: ARM SVE i8 dot
  ALDER: 1n << 13n,          // 2021: Intel AVX2+VNNI
  SVEBFDOT: 1n << 14n,       // 2021: ARM SVE BF16
  SVE2: 1n << 15n,           // 2022: ARM SVE2
  V128RELAXED: 1n << 16n,    // 2022: WASM Relaxed SIMD
  SAPPHIRE: 1n << 17n,       // 2023: Intel AVX-512 FP16
  SAPPHIREAMX: 1n << 18n,    // 2023: Intel Sapphire AMX
  RVV: 1n << 19n,            // 2023: RISC-V Vector
  RVVHALF: 1n << 20n,        // 2023: RISC-V Zvfh
  RVVBF16: 1n << 21n,        // 2023: RISC-V Zvfbfwma
  GRANITEAMX: 1n << 22n,     // 2024: Intel Granite AMX FP16
  TURIN: 1n << 23n,          // 2024: AMD Turin AVX-512 CD
  SME: 1n << 24n,            // 2024: ARM SME
  SME2: 1n << 25n,           // 2024: ARM SME2
  SMEF64: 1n << 26n,         // 2024: ARM SME F64
  SMEFA64: 1n << 27n,        // 2024: ARM SME FA64
  SVE2P1: 1n << 28n,         // 2025+: ARM SVE2.1
  SME2P1: 1n << 29n,         // 2025+: ARM SME2.1
  SMEHALF: 1n << 30n,        // 2025+: ARM SME F16F16
  SMEBF16: 1n << 31n,        // 2025+: ARM SME B16B16
  SMELUT2: 1n << 32n,        // 2025+: ARM SME LUTv2
  RVVBB: 1n << 33n,          // 2025+: RISC-V Zvbb
  SIERRA: 1n << 34n,         // 2024: Intel AVXVNNIINT8
  SMEBI32: 1n << 35n,        // 2025+: ARM SME BI32I32
  LOONGSONASX: 1n << 36n,    // LoongArch LASX 256-bit SIMD
  POWERVSX: 1n << 37n,       // Power VSX 128-bit SIMD
  DIAMOND: 1n << 38n,        // 2025+: Intel AVX10.2
  NEONFP8: 1n << 39n,        // ARM NEON FP8
  DIAMONDAMX: 1n << 40n,     // 2025+: Intel Diamond Rapids AMX
} as const;

export { Float16Array, BFloat16Array, E4M3Array, E5M2Array, BinaryArray, TensorBase, VectorBase, VectorView, Vector, MatrixBase, Matrix, PackedMatrix, outputDtype };

/** Convert a single FP16 value (as uint16 bits) to FP32 */
export const castF16ToF32 = addon.castF16ToF32;
/** Convert a single FP32 value to FP16 (returns uint16 bits) */
export const castF32ToF16 = addon.castF32ToF16;
/** Convert a single BF16 value (as uint16 bits) to FP32 */
export const castBF16ToF32 = addon.castBF16ToF32;
/** Convert a single FP32 value to BF16 (returns uint16 bits) */
export const castF32ToBF16 = addon.castF32ToBF16;
/** Convert a single E4M3 value (as uint8 bits) to FP32 */
export const castE4M3ToF32 = addon.castE4M3ToF32;
/** Convert a single FP32 value to E4M3 (returns uint8 bits) */
export const castF32ToE4M3 = addon.castF32ToE4M3;
/** Convert a single E5M2 value (as uint8 bits) to FP32 */
export const castE5M2ToF32 = addon.castE5M2ToF32;
/** Convert a single FP32 value to E5M2 (returns uint8 bits) */
export const castF32ToE5M2 = addon.castF32ToE5M2;
/** Bulk conversion between different numeric types (modifies destination array in-place) */
export const cast = addon.cast;

export { DType };

/**
 * Numeric arrays supported by distance metrics with auto-detected dtype.
 *
 * These standard TypedArrays are auto-detected by the N-API binding.
 */
export type NumericArray = Float64Array | Float32Array | Int8Array | Uint8Array;

/**
 * Extended array types supported by distance metrics with explicit dtype parameter.
 *
 * Includes Uint16Array (backing type for Float16Array and BFloat16Array) in addition
 * to the auto-detected types. Pass a dtype string as the third argument to distance
 * functions when using custom types.
 */
export type DistanceArray = Float64Array | Float32Array | Int8Array | Uint8Array | Uint16Array;

/**
 * Union type for all array types (including custom types for conversions)
 */
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

/**
 * Extract a TypedArray from a TensorBase for the N-API backend.
 *
 * The native backend doesn't benefit from zero-copy TensorBase (Node.js TypedArrays
 * already share process memory), but accepting TensorBase keeps the API uniform.
 */
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
 * Returns the SIMD capabilities this CPU supports, as a bitmask.
 *
 * Describes the machine only, and says nothing about whether a kernel was compiled into this
 * build — a prebuild whose ISA probes failed still reports your CPU's full feature set while
 * containing no SIMD kernels at all. Prefer getCapabilitiesAvailable().
 *
 * @returns {bigint} Bitmask of capability flags (use with Capability constants)
 */
export const getCapabilitiesDetected = (): bigint => addon.getCapabilitiesDetected();

/**
 * Returns the SIMD capabilities whose kernels were compiled into this binary, as a bitmask.
 *
 * Decided at build time by the ISA probes, independent of the CPU. The only accessor that can
 * tell you a prebuild is silently scalar.
 *
 * @returns {bigint} Bitmask of capability flags (use with Capability constants)
 */
export const getCapabilitiesCompiled = (): bigint => addon.getCapabilitiesCompiled();

/**
 * Returns the SIMD capabilities that can actually execute here, as a bitmask.
 *
 * The intersection of getCapabilitiesDetected() and getCapabilitiesCompiled(). Either axis
 * alone over-reports.
 *
 * @returns {bigint} Bitmask of capability flags (use with Capability constants)
 *
 * @example
 * ```ts
 * import { getCapabilitiesAvailable, Capability } from 'numkong';
 *
 * const caps = getCapabilitiesAvailable();
 * console.log(`Capabilities: 0x${caps.toString(16)}`);
 *
 * if (caps & Capability.HASWELL) {
 *   console.log('AVX2 available');
 * }
 * ```
 */
export const getCapabilitiesAvailable = (): bigint => addon.getCapabilitiesAvailable();

/**
 * Returns the SIMD capabilities dispatch is currently restricted to, as a bitmask.
 * A subset of getCapabilitiesAvailable().
 *
 * @returns {bigint} Bitmask of capability flags (use with Capability constants)
 */
export const getCapabilitiesEnabled = (): bigint => addon.getCapabilitiesEnabled();

/**
 * Restricts dispatch to `caps`, clamped to getCapabilitiesAvailable().
 * The serial fallback is always retained.
 *
 * @param {bigint} caps - Capability mask (from Capability constants)
 */
export const restrictCapabilities = (caps: bigint): void => addon.capabilitiesRestrict(caps);

/**
 * Adds `caps` to getCapabilitiesEnabled(). Anything not available is ignored.
 *
 * @param {bigint} caps - Capability mask (from Capability constants)
 */
export const enableCapabilities = (caps: bigint): void => addon.capabilitiesEnable(caps);

/**
 * Removes `caps` from getCapabilitiesEnabled(). The serial fallback cannot be removed.
 *
 * @param {bigint} caps - Capability mask (from Capability constants)
 */
export const disableCapabilities = (caps: bigint): void => addon.capabilitiesDisable(caps);

/**
 * Checks whether a specific SIMD capability can actually execute here.
 *
 * Tests against getCapabilitiesAvailable(), so it is false both when the CPU lacks the feature
 * and when its kernels were not compiled into this build.
 *
 * @param {bigint} cap - Capability flag to check (from Capability constants)
 * @returns {boolean} True if the capability is available, false otherwise
 *
 * @example
 * ```ts
 * import { hasCapability, Capability } from 'numkong';
 *
 * if (hasCapability(Capability.HASWELL)) {
 *   console.log('Intel AVX2 (Haswell) available');
 * }
 * ```
 */
export const hasCapability = (cap: bigint): boolean => (addon.getCapabilitiesAvailable() & cap) !== 0n;

/**
 * Computes the squared Euclidean distance between two vectors.
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The squared Euclidean distance between vectors a and b.
 */
export function sqeuclidean(a: NumericArray, b: NumericArray): number;
export function sqeuclidean(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function sqeuclidean(a: TensorBase, b: TensorBase): number;
export function sqeuclidean(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.sqeuclidean(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.sqeuclidean(a, b, dtypeToString(dtype)) : addon.sqeuclidean(a, b);
}

/**
 * Computes the Euclidean distance between two vectors.
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The Euclidean distance between vectors a and b.
 */
export function euclidean(a: NumericArray, b: NumericArray): number;
export function euclidean(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function euclidean(a: TensorBase, b: TensorBase): number;
export function euclidean(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.euclidean(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.euclidean(a, b, dtypeToString(dtype)) : addon.euclidean(a, b);
}

/**
 * Computes the angular distance between two vectors.
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The angular distance between vectors a and b.
 */
export function angular(a: NumericArray, b: NumericArray): number;
export function angular(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function angular(a: TensorBase, b: TensorBase): number;
export function angular(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.angular(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.angular(a, b, dtypeToString(dtype)) : addon.angular(a, b);
}

/**
 * Computes the inner product of two vectors (same as dot product).
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The inner product of vectors a and b.
 */
export function inner(a: NumericArray, b: NumericArray): number;
export function inner(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function inner(a: TensorBase, b: TensorBase): number;
export function inner(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.inner(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.inner(a, b, dtypeToString(dtype)) : addon.inner(a, b);
}

/**
 * Computes the dot product of two vectors (same as inner product).
 * @param a - The first vector.
 * @param b - The second vector (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16', 'e4m3').
 * @returns {number} The dot product of vectors a and b.
 */
export function dot(a: NumericArray, b: NumericArray): number;
export function dot(a: DistanceArray, b: DistanceArray, dtype: DType): number;
export function dot(a: TensorBase, b: TensorBase): number;
export function dot(a: DistanceArray | TensorBase, b: DistanceArray | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.dot(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.dot(a, b, dtypeToString(dtype)) : addon.dot(a, b);
}

/**
 * Computes the bitwise Hamming distance between two vectors.
 *
 * Both vectors are treated as bit-packed (u1 dtype), where each byte contains 8 bits.
 * Use toBinary() to convert numeric arrays to bit-packed format.
 *
 * @param {Uint8Array | BinaryArray} a - The first bit-packed vector.
 * @param {Uint8Array | BinaryArray} b - The second bit-packed vector.
 * @returns {number} The Hamming distance (number of differing bits) between vectors a and b.
 */
export const hamming = (a: Uint8Array | BinaryArray | TensorBase, b: Uint8Array | BinaryArray | TensorBase): number => {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.hamming(u.arr, v.arr); }
  return addon.hamming(a, b);
};

/**
 * Computes the bitwise Jaccard distance between two vectors.
 *
 * Both vectors are treated as bit-packed (u1 dtype), where each byte contains 8 bits.
 * Use toBinary() to convert numeric arrays to bit-packed format.
 *
 * @param {Uint8Array | BinaryArray} a - The first bit-packed vector.
 * @param {Uint8Array | BinaryArray} b - The second bit-packed vector.
 * @returns {number} The Jaccard distance (1 - Jaccard similarity) between vectors a and b.
 */
export const jaccard = (a: Uint8Array | BinaryArray | TensorBase, b: Uint8Array | BinaryArray | TensorBase): number => {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.jaccard(u.arr, v.arr); }
  return addon.jaccard(a, b);
};

/**
 * Computes the Kullback-Leibler divergence between two probability distributions.
 *
 * Both vectors must represent valid probability distributions (non-negative, sum to 1).
 * Supports f64, f32 (auto-detected) and f16, bf16 (with explicit dtype).
 *
 * @param a - The first probability distribution.
 * @param b - The second probability distribution (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16').
 * @returns {number} The Kullback-Leibler divergence KL(a || b) = Σ a[i] * log(a[i] / b[i]).
 */
export function kullbackleibler(a: Float64Array | Float32Array, b: Float64Array | Float32Array): number;
export function kullbackleibler(a: Float64Array | Float32Array | Uint16Array, b: Float64Array | Float32Array | Uint16Array, dtype: DType): number;
export function kullbackleibler(a: TensorBase, b: TensorBase): number;
export function kullbackleibler(a: Float64Array | Float32Array | Uint16Array | TensorBase, b: Float64Array | Float32Array | Uint16Array | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.kullbackleibler(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.kullbackleibler(a, b, dtypeToString(dtype)) : addon.kullbackleibler(a, b);
}

/**
 * Computes the Jensen-Shannon distance between two probability distributions.
 *
 * Both vectors must represent valid probability distributions (non-negative, sum to 1).
 * Supports f64, f32 (auto-detected) and f16, bf16 (with explicit dtype).
 * JSD is the square root of the symmetrized KL divergence, forming a true metric.
 *
 * @param a - The first probability distribution.
 * @param b - The second probability distribution (must match the type of a).
 * @param dtype - Optional dtype string for custom types (e.g. 'f16', 'bf16').
 * @returns {number} The Jensen-Shannon distance d_JS(a, b) = √(0.5 × (KL(a‖m) + KL(b‖m))), where m = (a + b) / 2.
 */
export function jensenshannon(a: Float64Array | Float32Array, b: Float64Array | Float32Array): number;
export function jensenshannon(a: Float64Array | Float32Array | Uint16Array, b: Float64Array | Float32Array | Uint16Array, dtype: DType): number;
export function jensenshannon(a: TensorBase, b: TensorBase): number;
export function jensenshannon(a: Float64Array | Float32Array | Uint16Array | TensorBase, b: Float64Array | Float32Array | Uint16Array | TensorBase, dtype?: DType): number {
  if (a instanceof TensorBase) { const u = unwrapTensor(a), v = unwrapTensor(b as TensorBase); return addon.jensenshannon(u.arr, v.arr, dtypeToString(u.dtype)); }
  return dtype !== undefined ? addon.jensenshannon(a, b, dtypeToString(dtype)) : addon.jensenshannon(a, b);
}

/**
 * Quantizes a numeric vector into a bit-packed binary representation.
 *
 * Converts each element to a single bit: 1 for positive values, 0 for non-positive values.
 * The bits are packed into bytes (8 bits per byte) in big-endian bit order within each byte.
 * This is the required format for hamming() and jaccard() distance functions.
 *
 * @param {Float32Array | Float64Array | Int8Array} vector - The vector to quantize and pack.
 * @returns {Uint8Array} A bit-packed array where each byte contains 8 binary values.
 *
 * @example
 * ```ts
 * const vec = new Float32Array([1.5, -2.3, 0.0, 3.1, -1.0, 2.0, 0.5, -0.5]);
 * const binary = toBinary(vec);
 * // Result: Uint8Array([0b10010110]) = [0x96]
 * //   bits: [1, 0, 0, 1, 0, 1, 1, 0] for elements [+, -, 0, +, -, +, +, -]
 *
 * // Use with Hamming distance
 * const a = toBinary(new Float32Array([1, 2, 3]));
 * const b = toBinary(new Float32Array([1, -2, 3]));
 * const dist = hamming(a, b); // Counts differing bits
 * ```
 */
export const toBinary = (vector: Float32Array | Float64Array | Int8Array): Uint8Array => {
  const byteLength = Math.ceil(vector.length / 8);
  const packedVector = new Uint8Array(byteLength);

  for (let i = 0; i < vector.length; i++) {
    if (vector[i] > 0) {
      const byteIndex = Math.floor(i / 8);
      const bitPosition = 7 - (i % 8);
      packedVector[byteIndex] |= (1 << bitPosition);
    }
  }

  return packedVector;
};

/**
 * Extract a TypedArray from a Matrix for passing to the N-API backend.
 */
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

/**
 * Extract a result TypedArray from a Matrix matching its output dtype.
 */
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
 * Query the packed buffer byte count for a given matrix shape and dtype.
 */
export function dotsPackedSize(width: number, depth: number, dtype: DType): number {
  return addon.dotsPackedSize(width, depth, dtypeToString(dtype));
}

/**
 * Pack a Matrix for use with packed GEMM-like operations.
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
  const outDtype = outputDtype(family, a.dtype);
  if (!out) {
    out = new Matrix(a.rows, packed.width, outDtype);
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
  const outDtype = outputDtype(family, vectors.dtype);
  if (!out) {
    out = new Matrix(vectors.rows, vectors.rows, outDtype);
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

export function dotsPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return packedOperation('dotsPacked', 'dots', a, packed, out);
}

export function angularsPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return packedOperation('angularsPacked', 'angulars', a, packed, out);
}

export function euclideansPacked(a: Matrix, packed: PackedMatrix, out?: Matrix): Matrix {
  return packedOperation('euclideansPacked', 'euclideans', a, packed, out);
}

export function dotsSymmetric(vectors: Matrix, out?: Matrix, options?: { rowStart?: number; rowCount?: number }): Matrix {
  return symmetricOperation('dotsSymmetric', 'dots', vectors, out, options?.rowStart ?? 0, options?.rowCount);
}

export function angularsSymmetric(vectors: Matrix, out?: Matrix, options?: { rowStart?: number; rowCount?: number }): Matrix {
  return symmetricOperation('angularsSymmetric', 'angulars', vectors, out, options?.rowStart ?? 0, options?.rowCount);
}

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
  outputDtype,
};

/**
 * Finds the directory where the native build of the numkong module is located.
 * @param {string} dir - The directory to start the search from.
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
