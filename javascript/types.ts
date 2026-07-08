/**
 * @brief Custom TypedArray classes for non-native numeric types.
 * @file javascript/types.ts
 * @author Ash Vardanian
 * @date February 3, 2026
 *
 * This file provides TypedArray wrappers for numeric types not natively supported
 * by JavaScript, using NumKong's SIMD-optimized conversion functions from the C library.
 */

// Import conversion functions from the compiled native module
// These will be defined in numkong.ts after the module is loaded
let conversionFunctions: {
  castF16ToF32: (bits: number) => number;
  castF32ToF16: (value: number) => number;
  castBF16ToF32: (bits: number) => number;
  castF32ToBF16: (value: number) => number;
  castE4M3ToF32: (bits: number) => number;
  castF32ToE4M3: (value: number) => number;
  castE5M2ToF32: (bits: number) => number;
  castF32ToE5M2: (value: number) => number;
  cast: (src: TypedArray, srcType: string, dst: TypedArray, dstType: string) => void;
};

// This will be called by numkong.ts after loading the module
export function setConversionFunctions(fns: typeof conversionFunctions) {
  conversionFunctions = fns;
}

// Type alias for any TypedArray
export type TypedArray = Float64Array | Float32Array | Int32Array | Int8Array | Uint8Array | Uint16Array | Uint32Array;

/** @brief Numeric data type enum — integer switch, compiles to jump table. */
export enum DType {
  F64 = 0,
  F32 = 1,
  F16 = 2,
  BF16 = 3,
  E4M3 = 4,
  E5M2 = 5,
  E2M3 = 6,
  E3M2 = 7,
  I8 = 8,
  U8 = 9,
  U1 = 10,
  I32 = 11,
  U32 = 12,
}

/** @brief O(1) array lookup for DType → string conversion (needed at N-API/WASM boundaries). */
export const DTYPE_STRINGS: readonly string[] = [
  'f64', 'f32', 'f16', 'bf16', 'e4m3', 'e5m2', 'e2m3', 'e3m2', 'i8', 'u8', 'u1', 'i32', 'u32',
];

/** @brief Convert a DType enum value to its string representation. */
export function dtypeToString(d: DType): string { return DTYPE_STRINGS[d]; }

/** @brief Infer the DType from a TypedArray instance. */
function inferDtype(arr: TypedArray): DType {
  if (arr instanceof Float64Array) return DType.F64;
  if (arr instanceof Float32Array) return DType.F32;
  if (arr instanceof Int32Array) return DType.I32;
  if (arr instanceof Int8Array) return DType.I8;
  if (arr instanceof Uint8Array) return DType.U8;
  if (arr instanceof Uint16Array) return DType.F16;
  if (arr instanceof Uint32Array) return DType.U32;
  throw new Error(`Cannot infer dtype from ${(arr as any).constructor.name}`);
}

/**
 * @brief Abstract base class for all tensor types.
 *
 * All fields are embedded — zero dynamic allocation. DType is a numeric enum
 * (integer switch). Mirrors the C++ pattern: buffer + byteOffset + dtype.
 */
export abstract class TensorBase {
  // Mutable so owning subclasses can `reserve()` a larger buffer; `tryResize()`/`clear()` never touch these.
  buffer: ArrayBuffer;
  byteOffset: number;
  readonly dtype: DType;

  protected constructor(buffer: ArrayBuffer, byteOffset: number, dtype: DType) {
    this.buffer = buffer;
    this.byteOffset = byteOffset;
    this.dtype = dtype;
  }

  abstract get length(): number;
  abstract get rank(): number;

  /** @brief Bytes per element for this tensor's dtype (compiles to jump table). */
  get bytesPerElement(): number {
    switch (this.dtype) {
      case DType.F64: return 8;
      case DType.F32: case DType.I32: case DType.U32: return 4;
      case DType.F16: case DType.BF16: return 2;
      default: return 1;
    }
  }

  /** @brief Total byte length of the tensor data. */
  get byteLength(): number { return this.length * this.bytesPerElement; }
}

/**
 * @brief Abstract rank-1 tensor base class.
 */
export abstract class VectorBase extends TensorBase {
  // Mutable so owning Vectors can `tryResize()`/`clear()` within capacity.
  length: number;

  protected constructor(buffer: ArrayBuffer, byteOffset: number, length: number, dtype: DType) {
    super(buffer, byteOffset, dtype);
    this.length = length;
  }

  get rank(): 1 { return 1; }
}

/**
 * @brief Non-owning rank-1 tensor view (like std::span<T>).
 *
 * Zero-copy wrapper for existing memory. Ideal for cross-module WASM interop
 * where data already lives on the WASM heap.
 */
export class VectorView extends VectorBase {
  constructor(buffer: ArrayBuffer, byteOffset: number, length: number, dtype: DType) {
    super(buffer, byteOffset, length, dtype);
  }

  toString(): string {
    return `VectorView(${this.length}, ${dtypeToString(this.dtype)})`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }

  /** @brief Create a VectorView from any TypedArray, inferring or accepting dtype. */
  static from(arr: TypedArray, dtype?: DType): VectorView {
    const d = dtype ?? inferDtype(arr);
    return new VectorView(arr.buffer as ArrayBuffer, arr.byteOffset, arr.length, d);
  }
}

/**
 * @brief Owning rank-1 tensor (like std::vector<T>).
 *
 * Allocates its own ArrayBuffer. Use for storing results or when you need
 * independent ownership of the data.
 */
export class Vector extends VectorBase {
  constructor(length: number, dtype: DType);
  constructor(buffer: ArrayBuffer, length: number, dtype: DType);
  constructor(lengthOrBuffer: number | ArrayBuffer, dtypeOrLength: DType | number, dtype?: DType) {
    if (typeof lengthOrBuffer === 'number') {
      const length = lengthOrBuffer;
      const dt = dtypeOrLength as DType;
      let bpe: number;
      switch (dt) {
        case DType.F64: bpe = 8; break;
        case DType.F32: case DType.I32: case DType.U32: bpe = 4; break;
        case DType.F16: case DType.BF16: bpe = 2; break;
        default: bpe = 1; break;
      }
      super(new ArrayBuffer(length * bpe), 0, length, dt);
    } else {
      super(lengthOrBuffer, 0, dtypeOrLength as number, dtype!);
    }
  }

  toString(): string {
    return `Vector(${this.length}, ${dtypeToString(this.dtype)})`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }

  /** @brief Create an owning Vector by copying data from a TypedArray. */
  static fromTypedArray(arr: TypedArray, dtype?: DType): Vector {
    const d = dtype ?? inferDtype(arr);
    return new Vector((arr.buffer as ArrayBuffer).slice(arr.byteOffset, arr.byteOffset + arr.byteLength), arr.length, d);
  }

  /** @brief Create an owning Vector by copying data from any TensorBase. */
  static fromView(view: TensorBase): Vector {
    return new Vector(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength), view.length, view.dtype);
  }

  /** @brief Return a TypedArray view over this Vector's owned buffer (zero-copy). */
  toTypedArray(): TypedArray {
    switch (this.dtype) {
      case DType.F64: return new Float64Array(this.buffer, 0, this.length);
      case DType.F32: return new Float32Array(this.buffer, 0, this.length);
      case DType.I32: return new Int32Array(this.buffer, 0, this.length);
      case DType.U32: return new Uint32Array(this.buffer, 0, this.length);
      case DType.F16: case DType.BF16: return new Uint16Array(this.buffer, 0, this.length);
      case DType.I8: return new Int8Array(this.buffer, 0, this.length);
      default: return new Uint8Array(this.buffer, 0, this.length);
    }
  }

  /** @brief Allocated element capacity (>= length) — the ceiling `tryResize` honors. */
  get capacity(): number {
    return Math.floor((this.buffer.byteLength - this.byteOffset) / this.bytesPerElement);
  }

  /**
   * @brief Resize within capacity without moving storage — `toTypedArray()` views stay valid.
   * @returns false (length unchanged) if `length` is negative or exceeds `capacity()`.
   */
  tryResize(length: number): boolean {
    if (length < 0 || length > this.capacity) return false;
    this.length = length;
    return true;
  }

  /**
   * @brief Grow the allocated capacity to at least `elements`, reallocating and copying the live
   * elements. A no-op when already large enough. INVALIDATES any TypedArray previously returned by
   * `toTypedArray()` — they keep pointing at the old buffer.
   */
  reserve(elements: number): boolean {
    if (elements <= this.capacity) return true;
    const grown = new ArrayBuffer(elements * this.bytesPerElement);
    new Uint8Array(grown).set(new Uint8Array(this.buffer, this.byteOffset, this.length * this.bytesPerElement));
    this.buffer = grown;
    this.byteOffset = 0;
    return true;
  }

  /** @brief Reset to length 0 while keeping the allocated capacity. */
  clear(): void {
    this.length = 0;
  }
}

/**
 * @brief Abstract rank-2 tensor base class.
 *
 * All 4 dimension fields are embedded — no dynamic allocation.
 * Strides are in bytes to match the C API directly.
 */
export abstract class MatrixBase extends TensorBase {
  // Mutable so owning Matrices can `tryResize()`/`clear()` within capacity.
  rows: number;
  cols: number;
  rowStride: number;
  colStride: number;

  protected constructor(
    buffer: ArrayBuffer, byteOffset: number, dtype: DType,
    rows: number, cols: number, rowStride: number, colStride: number,
  ) {
    super(buffer, byteOffset, dtype);
    this.rows = rows;
    this.cols = cols;
    this.rowStride = rowStride;
    this.colStride = colStride;
  }

  get length(): number { return this.rows * this.cols; }
  get rank(): 2 { return 2; }
}

/**
 * @brief Owning rank-2 tensor (row-major, C-contiguous by default).
 *
 * Strides are byte strides. Default for C-contiguous layout:
 * rowStride = cols * bytesPerElement, colStride = bytesPerElement.
 */
export class Matrix extends MatrixBase {
  constructor(rows: number, cols: number, dtype: DType);
  constructor(buffer: ArrayBuffer, byteOffset: number, dtype: DType, rows: number, cols: number, rowStride?: number, colStride?: number);
  constructor(
    rowsOrBuffer: number | ArrayBuffer,
    colsOrByteOffset: number,
    dtype: DType,
    rows?: number,
    cols?: number,
    rowStride?: number,
    colStride?: number,
  ) {
    if (typeof rowsOrBuffer === 'number') {
      const r = rowsOrBuffer;
      const c = colsOrByteOffset;
      let bpe: number;
      switch (dtype) {
        case DType.F64: bpe = 8; break;
        case DType.F32: case DType.I32: case DType.U32: bpe = 4; break;
        case DType.F16: case DType.BF16: bpe = 2; break;
        default: bpe = 1; break;
      }
      super(new ArrayBuffer(r * c * bpe), 0, dtype, r, c, c * bpe, bpe);
    } else {
      const r = rows!;
      const c = cols!;
      let bpe: number;
      switch (dtype) {
        case DType.F64: bpe = 8; break;
        case DType.F32: case DType.I32: case DType.U32: bpe = 4; break;
        case DType.F16: case DType.BF16: bpe = 2; break;
        default: bpe = 1; break;
      }
      super(rowsOrBuffer, colsOrByteOffset, dtype, r, c, rowStride ?? c * bpe, colStride ?? bpe);
    }
  }

  toString(): string {
    return `Matrix(${this.rows}\u00d7${this.cols}, ${dtypeToString(this.dtype)})`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }

  static fromTypedArray(array: TypedArray, rows: number, cols: number, dtype?: DType): Matrix {
    const d = dtype ?? inferDtype(array);
    const buf = (array.buffer as ArrayBuffer).slice(array.byteOffset, array.byteOffset + array.byteLength);
    return new Matrix(buf, 0, d, rows, cols);
  }

  toTypedArray(): TypedArray {
    switch (this.dtype) {
      case DType.F64: return new Float64Array(this.buffer, this.byteOffset, this.rows * this.cols);
      case DType.F32: return new Float32Array(this.buffer, this.byteOffset, this.rows * this.cols);
      case DType.I32: return new Int32Array(this.buffer, this.byteOffset, this.rows * this.cols);
      case DType.U32: return new Uint32Array(this.buffer, this.byteOffset, this.rows * this.cols);
      case DType.F16: case DType.BF16: return new Uint16Array(this.buffer, this.byteOffset, this.rows * this.cols);
      case DType.I8: return new Int8Array(this.buffer, this.byteOffset, this.rows * this.cols);
      default: return new Uint8Array(this.buffer, this.byteOffset, this.rows * this.cols);
    }
  }

  row(index: number): VectorView {
    return new VectorView(this.buffer, this.byteOffset + index * this.rowStride, this.cols, this.dtype);
  }

  /** @brief Allocated element capacity (>= rows*cols) — the ceiling `tryResize` honors. */
  get capacity(): number {
    return Math.floor((this.buffer.byteLength - this.byteOffset) / this.bytesPerElement);
  }

  /**
   * @brief Resize within capacity without moving storage (`row()`/`toTypedArray()` views stay valid),
   * re-deriving C-contiguous strides.
   * @returns false (unchanged) if either extent is negative or `rows*cols` exceeds `capacity()`.
   */
  tryResize(rows: number, cols: number): boolean {
    if (rows < 0 || cols < 0 || rows * cols > this.capacity) return false;
    this.rows = rows;
    this.cols = cols;
    this.rowStride = cols * this.bytesPerElement;
    this.colStride = this.bytesPerElement;
    return true;
  }

  /**
   * @brief Grow the allocated capacity to at least `elements`, reallocating and copying the live
   * rows*cols elements. A no-op when already large enough. INVALIDATES any prior `toTypedArray()` /
   * `row()` view. Intended for the C-contiguous owning layout.
   */
  reserve(elements: number): boolean {
    if (elements <= this.capacity) return true;
    const grown = new ArrayBuffer(elements * this.bytesPerElement);
    new Uint8Array(grown).set(new Uint8Array(this.buffer, this.byteOffset, this.rows * this.cols * this.bytesPerElement));
    this.buffer = grown;
    this.byteOffset = 0;
    return true;
  }

  /** @brief Reset to an empty 0x0 matrix while keeping the allocated capacity. */
  clear(): void {
    this.rows = 0;
    this.cols = 0;
  }
}

/**
 * @brief Opaque packed matrix container.
 *
 * Packed layout is not indexable — this is a data container for packed GEMM kernels.
 * N-API path: buffer is a V8-managed ArrayBuffer, auto-freed by GC.
 * WASM path: stores a heap pointer, dispose() calls Module._free().
 */
export class PackedMatrix {
  readonly width: number;
  readonly depth: number;
  readonly dtype: DType;
  readonly byteLength: number;
  readonly buffer: ArrayBuffer;
  private _disposed: boolean = false;

  constructor(buffer: ArrayBuffer, width: number, depth: number, dtype: DType, byteLength: number) {
    this.buffer = buffer;
    this.width = width;
    this.depth = depth;
    this.dtype = dtype;
    this.byteLength = byteLength;
  }

  dispose(): void { this._disposed = true; }
  get disposed(): boolean { return this._disposed; }

  toString(): string {
    return `PackedMatrix(${this.width}\u00d7${this.depth}, ${dtypeToString(this.dtype)}, ${this.byteLength} bytes)`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }
}

/** @brief Kernel family identifiers for output dtype resolution. */
export type KernelFamily = 'dots' | 'angulars' | 'euclideans';

/**
 * @brief Determines the output dtype for a given kernel family and input dtype.
 * Mirrors nk_kernel_output_dtype from C.
 */
export function outputDtype(family: KernelFamily, input: DType): DType {
  switch (input) {
    case DType.F64: return DType.F64;
    case DType.F32: return DType.F64;
    case DType.F16: case DType.BF16: case DType.E4M3: case DType.E5M2: case DType.E2M3: case DType.E3M2:
      return DType.F32;
    case DType.I8: return family === 'dots' ? DType.I32 : DType.F32;
    case DType.U8: return family === 'dots' ? DType.U32 : DType.F32;
    default: return DType.F32;
  }
}

/**
 * @brief IEEE 754 Half Precision Float (f16)
 *
 * 16-bit floating point: 1 sign bit, 5 exponent bits, 10 mantissa bits
 * Range: ~±65504, precision: ~3-4 decimal digits
 *
 * Common in GPU inference, model compression, and mixed-precision training.
 * Supported natively on Apple Silicon, NVIDIA GPUs (fp16), AMD GPUs.
 */
export class Float16Array extends Uint16Array {
  constructor(length: number | ArrayLike<number> | ArrayBufferLike, byteOffset?: number, arrayLength?: number) {
    if (typeof length === 'number') {
      super(length);
    } else if (ArrayBuffer.isView(length) || length instanceof ArrayBuffer) {
      super(length as any, byteOffset, arrayLength);
    } else {
      // Convert from array-like of numbers
      const src = length as ArrayLike<number>;
      const arr = new Uint16Array(src.length);
      if (conversionFunctions) {
        for (let i = 0; i < src.length; i++) {
          arr[i] = conversionFunctions.castF32ToF16(src[i]);
        }
      }
      super(arr);
    }
  }

  /**
   * @brief Converts the entire f16 array to f32 (Float32Array).
   * @returns Float32Array with decoded values
   */
  toFloat32Array(): Float32Array {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    const result = new Float32Array(this.length);
    for (let i = 0; i < this.length; i++) {
      result[i] = conversionFunctions.castF16ToF32(this[i]);
    }
    return result;
  }

  /**
   * @brief Gets the f32 value at the specified index.
   * @param index Array index
   * @returns Decoded f32 value
   */
  getFloat32(index: number): number {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    return conversionFunctions.castF16ToF32(this[index]);
  }

  /**
   * @brief Sets the value at the specified index from an f32 value.
   * @param index Array index
   * @param value f32 value to encode and store
   */
  setFloat32(index: number, value: number): void {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    this[index] = conversionFunctions.castF32ToF16(value);
  }

  toString(): string {
    if (!conversionFunctions) return `Float16Array(${this.length})`;
    const limit = Math.min(this.length, 20);
    const parts: string[] = [];
    for (let i = 0; i < limit; i++) {
      const f = conversionFunctions.castF16ToF32(this[i]);
      parts.push(`${f} [0x${this[i].toString(16).padStart(4, '0')}]`);
    }
    const suffix = this.length > 20 ? ', ...' : '';
    return `Float16Array(${this.length}) [${parts.join(', ')}${suffix}]`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }
}

/**
 * @brief Brain Float 16 (bf16)
 *
 * 16-bit floating point: 1 sign bit, 8 exponent bits, 7 mantissa bits
 * Range: same as f32 (~±3.4e38), precision: ~2-3 decimal digits
 *
 * Designed by Google for TPUs, optimized for ML training (wider range than f16).
 * Supported on Google TPUs, Intel Sapphire Rapids, AMD Genoa, ARM Neoverse V2.
 * Truncated f32 (top 16 bits), making conversion very cheap.
 */
export class BFloat16Array extends Uint16Array {
  constructor(length: number | ArrayLike<number> | ArrayBufferLike, byteOffset?: number, arrayLength?: number) {
    if (typeof length === 'number') {
      super(length);
    } else if (ArrayBuffer.isView(length) || length instanceof ArrayBuffer) {
      super(length as any, byteOffset, arrayLength);
    } else {
      const src = length as ArrayLike<number>;
      const arr = new Uint16Array(src.length);
      if (conversionFunctions) {
        for (let i = 0; i < src.length; i++) {
          arr[i] = conversionFunctions.castF32ToBF16(src[i]);
        }
      }
      super(arr);
    }
  }

  toFloat32Array(): Float32Array {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    const result = new Float32Array(this.length);
    for (let i = 0; i < this.length; i++) {
      result[i] = conversionFunctions.castBF16ToF32(this[i]);
    }
    return result;
  }

  getFloat32(index: number): number {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    return conversionFunctions.castBF16ToF32(this[index]);
  }

  setFloat32(index: number, value: number): void {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    this[index] = conversionFunctions.castF32ToBF16(value);
  }

  toString(): string {
    if (!conversionFunctions) return `BFloat16Array(${this.length})`;
    const limit = Math.min(this.length, 20);
    const parts: string[] = [];
    for (let i = 0; i < limit; i++) {
      const f = conversionFunctions.castBF16ToF32(this[i]);
      parts.push(`${f} [0x${this[i].toString(16).padStart(4, '0')}]`);
    }
    const suffix = this.length > 20 ? ', ...' : '';
    return `BFloat16Array(${this.length}) [${parts.join(', ')}${suffix}]`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }
}

/**
 * @brief FP8 E4M3 (4-bit exponent, 3-bit mantissa)
 *
 * 8-bit floating point: 1 sign bit, 4 exponent bits, 3 mantissa bits
 * Range: ~±448, precision: ~1 decimal digit
 *
 * Optimized for forward pass inference with higher precision than E5M2.
 * Supported on NVIDIA Hopper H100 GPUs, AMD Instinct MI300.
 */
export class E4M3Array extends Uint8Array {
  constructor(length: number | ArrayLike<number> | ArrayBufferLike, byteOffset?: number, arrayLength?: number) {
    if (typeof length === 'number') {
      super(length);
    } else if (ArrayBuffer.isView(length) || length instanceof ArrayBuffer) {
      super(length as any, byteOffset, arrayLength);
    } else {
      const src = length as ArrayLike<number>;
      const arr = new Uint8Array(src.length);
      if (conversionFunctions) {
        for (let i = 0; i < src.length; i++) {
          arr[i] = conversionFunctions.castF32ToE4M3(src[i]);
        }
      }
      super(arr);
    }
  }

  toFloat32Array(): Float32Array {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    const result = new Float32Array(this.length);
    for (let i = 0; i < this.length; i++) {
      result[i] = conversionFunctions.castE4M3ToF32(this[i]);
    }
    return result;
  }

  getFloat32(index: number): number {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    return conversionFunctions.castE4M3ToF32(this[index]);
  }

  setFloat32(index: number, value: number): void {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    this[index] = conversionFunctions.castF32ToE4M3(value);
  }

  toString(): string {
    if (!conversionFunctions) return `E4M3Array(${this.length})`;
    const limit = Math.min(this.length, 20);
    const parts: string[] = [];
    for (let i = 0; i < limit; i++) {
      const f = conversionFunctions.castE4M3ToF32(this[i]);
      parts.push(`${f} [0x${this[i].toString(16).padStart(2, '0')}]`);
    }
    const suffix = this.length > 20 ? ', ...' : '';
    return `E4M3Array(${this.length}) [${parts.join(', ')}${suffix}]`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }
}

/**
 * @brief FP8 E5M2 (5-bit exponent, 2-bit mantissa)
 *
 * 8-bit floating point: 1 sign bit, 5 exponent bits, 2 mantissa bits
 * Range: ~±57344, precision: <1 decimal digit
 *
 * Optimized for backward pass training with wider range than E4M3.
 * Supported on NVIDIA Hopper H100 GPUs, AMD Instinct MI300.
 */
export class E5M2Array extends Uint8Array {
  constructor(length: number | ArrayLike<number> | ArrayBufferLike, byteOffset?: number, arrayLength?: number) {
    if (typeof length === 'number') {
      super(length);
    } else if (ArrayBuffer.isView(length) || length instanceof ArrayBuffer) {
      super(length as any, byteOffset, arrayLength);
    } else {
      const src = length as ArrayLike<number>;
      const arr = new Uint8Array(src.length);
      if (conversionFunctions) {
        for (let i = 0; i < src.length; i++) {
          arr[i] = conversionFunctions.castF32ToE5M2(src[i]);
        }
      }
      super(arr);
    }
  }

  toFloat32Array(): Float32Array {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    const result = new Float32Array(this.length);
    for (let i = 0; i < this.length; i++) {
      result[i] = conversionFunctions.castE5M2ToF32(this[i]);
    }
    return result;
  }

  getFloat32(index: number): number {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    return conversionFunctions.castE5M2ToF32(this[index]);
  }

  setFloat32(index: number, value: number): void {
    if (!conversionFunctions) {
      throw new Error('Conversion functions not initialized');
    }
    this[index] = conversionFunctions.castF32ToE5M2(value);
  }

  toString(): string {
    if (!conversionFunctions) return `E5M2Array(${this.length})`;
    const limit = Math.min(this.length, 20);
    const parts: string[] = [];
    for (let i = 0; i < limit; i++) {
      const f = conversionFunctions.castE5M2ToF32(this[i]);
      parts.push(`${f} [0x${this[i].toString(16).padStart(2, '0')}]`);
    }
    const suffix = this.length > 20 ? ', ...' : '';
    return `E5M2Array(${this.length}) [${parts.join(', ')}${suffix}]`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }
}

/**
 * @brief Binary Array (u1) - Bit-packed binary vectors
 *
 * 1-bit per element, packed into bytes (8 bits per byte)
 * Used for binary embeddings, hashing, and Hamming/Jaccard distances.
 *
 * Common in semantic search with binary quantization (Cohere, Voyage).
 */
export class BinaryArray extends Uint8Array {
  private _bitLength: number;

  constructor(bitLength: number) {
    const byteLength = Math.ceil(bitLength / 8);
    super(byteLength);
    this._bitLength = bitLength;
  }

  /**
   * @brief Gets the bit value at the specified index.
   * @param index Bit index (0 to bitLength-1)
   * @returns 0 or 1
   */
  getBit(index: number): number {
    if (index < 0 || index >= this._bitLength) {
      throw new RangeError('Index out of bounds');
    }
    const byteIndex = index >>> 3;  // index / 8
    const bitIndex = index & 7;     // index % 8
    return (this[byteIndex] >>> bitIndex) & 1;
  }

  /**
   * @brief Sets the bit value at the specified index.
   * @param index Bit index (0 to bitLength-1)
   * @param value 0 or 1
   */
  setBit(index: number, value: number): void {
    if (index < 0 || index >= this._bitLength) {
      throw new RangeError('Index out of bounds');
    }
    const byteIndex = index >>> 3;
    const bitIndex = index & 7;
    if (value) {
      this[byteIndex] |= (1 << bitIndex);
    } else {
      this[byteIndex] &= ~(1 << bitIndex);
    }
  }

  /**
   * @brief Returns the logical bit length of the array.
   */
  get bitLength(): number {
    return this._bitLength;
  }

  /**
   * @brief Creates a BinaryArray from a Float32Array (positive values = 1, else 0).
   * @param vector Source floating-point vector
   * @returns Binary array with quantized values
   */
  static fromFloat32Array(vector: Float32Array): BinaryArray {
    const binary = new BinaryArray(vector.length);
    for (let i = 0; i < vector.length; i++) {
      if (vector[i] > 0) {
        binary.setBit(i, 1);
      }
    }
    return binary;
  }

  /**
   * @brief Creates a BinaryArray from a Float64Array (positive values = 1, else 0).
   * @param vector Source floating-point vector
   * @returns Binary array with quantized values
   */
  static fromFloat64Array(vector: Float64Array): BinaryArray {
    const binary = new BinaryArray(vector.length);
    for (let i = 0; i < vector.length; i++) {
      if (vector[i] > 0) {
        binary.setBit(i, 1);
      }
    }
    return binary;
  }

  toString(): string {
    const limit = Math.min(this.length, 20);
    const parts: string[] = [];
    for (let i = 0; i < limit; i++) {
      parts.push(`0b${this[i].toString(2).padStart(8, '0')}`);
    }
    const suffix = this.length > 20 ? ', ...' : '';
    return `BinaryArray(${this._bitLength}) [${parts.join(', ')}${suffix}]`;
  }

  [Symbol.for('nodejs.util.inspect.custom')](): string {
    return this.toString();
  }
}

/**
 * @brief Type guard to check if an object is a Float16Array.
 */
export function isFloat16Array(obj: any): obj is Float16Array {
  return obj instanceof Float16Array;
}

/**
 * @brief Type guard to check if an object is a BFloat16Array.
 */
export function isBFloat16Array(obj: any): obj is BFloat16Array {
  return obj instanceof BFloat16Array;
}

/**
 * @brief Type guard to check if an object is an E4M3Array.
 */
export function isE4M3Array(obj: any): obj is E4M3Array {
  return obj instanceof E4M3Array;
}

/**
 * @brief Type guard to check if an object is an E5M2Array.
 */
export function isE5M2Array(obj: any): obj is E5M2Array {
  return obj instanceof E5M2Array;
}

/**
 * @brief Type guard to check if an object is a BinaryArray.
 */
export function isBinaryArray(obj: any): obj is BinaryArray {
  return obj instanceof BinaryArray;
}
