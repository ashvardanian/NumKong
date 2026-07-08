// Currently the builds are expected to run only on Node.js,
// but Deno tests pass as well.
//
// Bun supports `node:assert`, but not `node:test`.
// Using `require` we can make the tests compatible with Bun.
//
//    const isBun = typeof Bun !== "undefined";
//    let assert, test;
//    if (isBun) {
//      assert = require('node:assert');
//      test = require('bun:test');
//    } else {
//      assert = require('node:assert');
//      test = require('node:test');
//    }
//
// That, however, leads to other issues, like the following:
//
//    require is not defined in ES module scope, you can use import instead
//
// https://bun.sh/docs/runtime/nodejs-apis
// https://bun.sh/guides/util/detect-bun
import test from "node:test";
import assert from "node:assert";

import * as numkong from "../javascript/dist/esm/numkong.js";

function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
  const lowerBound = expected - tolerance;
  const upperBound = expected + tolerance;
  assert(actual >= lowerBound && actual <= upperBound, `Expected ${actual} to be almost equal to ${expected}`);
}

test("Distance from itself", () => {
  const f32s = new Float32Array([1.0, 2.0, 3.0]);
  assertAlmostEqual(numkong.sqeuclidean(f32s, f32s), 0.0, 0.01);
  assertAlmostEqual(numkong.angular(f32s, f32s), 0.0, 0.01);

  const f64s = new Float64Array([1.0, 2.0, 3.0]);
  assertAlmostEqual(numkong.sqeuclidean(f64s, f64s), 0.0, 0.01);
  assertAlmostEqual(numkong.angular(f64s, f64s), 0.0, 0.01);

  const f32sNormalized = new Float32Array([1 / Math.sqrt(14), 2 / Math.sqrt(14), 3 / Math.sqrt(14)]);
  assertAlmostEqual(numkong.inner(f32sNormalized, f32sNormalized), 1.0, 0.01);

  const f32sHistogram = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  assertAlmostEqual(numkong.kullbackleibler(f32sHistogram, f32sHistogram), 0.0, 0.01);
  assertAlmostEqual(numkong.jensenshannon(f32sHistogram, f32sHistogram), 0.0, 0.01);

  const u8s = new Uint8Array([1, 2, 3]);
  assertAlmostEqual(numkong.hamming(u8s, u8s), 0.0, 0.01);
  assertAlmostEqual(numkong.jaccard(u8s, u8s), 0.0, 0.01);
});

test("Squared Euclidean Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.sqeuclidean(f64sOne, f64sTwo), 27.0, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.sqeuclidean(f32sOne, f32sTwo), 27.0, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  assertAlmostEqual(numkong.sqeuclidean(u8sOne, u8sTwo), 27.0, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  assertAlmostEqual(numkong.sqeuclidean(i8sOne, i8sTwo), 27.0, 0.01);
});

test("Euclidean Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.euclidean(f64sOne, f64sTwo), 5.2, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.euclidean(f32sOne, f32sTwo), 5.2, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  assertAlmostEqual(numkong.euclidean(u8sOne, u8sTwo), 5.2, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  assertAlmostEqual(numkong.euclidean(i8sOne, i8sTwo), 5.2, 0.01);
});

test("Inner Product", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.inner(f64sOne, f64sTwo), 32.0, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.inner(f32sOne, f32sTwo), 32.0, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  assertAlmostEqual(numkong.inner(u8sOne, u8sTwo), 32.0, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  assertAlmostEqual(numkong.inner(i8sOne, i8sTwo), 32.0, 0.01);
});

test("Angular Distance", () => {
  const f64sOne = new Float64Array([1.0, 2.0, 3.0]);
  const f64sTwo = new Float64Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.angular(f64sOne, f64sTwo), 0.03, 0.01);

  const f32sOne = new Float32Array([1.0, 2.0, 3.0]);
  const f32sTwo = new Float32Array([4.0, 5.0, 6.0]);
  assertAlmostEqual(numkong.angular(f32sOne, f32sTwo), 0.03, 0.01);

  const u8sOne = new Uint8Array([1, 2, 3]);
  const u8sTwo = new Uint8Array([4, 5, 6]);
  assertAlmostEqual(numkong.angular(u8sOne, u8sTwo), 0.03, 0.01);

  const i8sOne = new Int8Array([1, 2, 3]);
  const i8sTwo = new Int8Array([4, 5, 6]);
  assertAlmostEqual(numkong.angular(i8sOne, i8sTwo), 0.03, 0.01);
});

test("Kullback-Leibler", () => {
  const f64sOne = new Float64Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f64sTwo = new Float64Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  assertAlmostEqual(numkong.kullbackleibler(f64sOne, f64sTwo), 0.16, 0.01);

  const f32sOne = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f32sTwo = new Float32Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  assertAlmostEqual(numkong.kullbackleibler(f32sOne, f32sTwo), 0.16, 0.01);
});

test("Jensen-Shannon", () => {
  const f64sOne = new Float64Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f64sTwo = new Float64Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  assertAlmostEqual(numkong.jensenshannon(f64sOne, f64sTwo), 0.095, 0.01);

  const f32sOne = new Float32Array([1.0 / 6, 2.0 / 6, 3.0 / 6]);
  const f32sTwo = new Float32Array([4.0 / 17, 5.0 / 17, 6.0 / 17]);
  assertAlmostEqual(numkong.jensenshannon(f32sOne, f32sTwo), 0.095, 0.01);
});

test("Matrix construction and toTypedArray", () => {
  const matrix = new numkong.Matrix(2, 3, numkong.DType.F32);
  assert.strictEqual(matrix.rows, 2);
  assert.strictEqual(matrix.cols, 3);
  assert.strictEqual(matrix.rowStride, 3 * 4); // 3 cols * 4 bytes
  const arr = matrix.toTypedArray();
  assert(arr instanceof Float32Array);
  assert.strictEqual(arr.length, 6);
});

test("Matrix.fromTypedArray", () => {
  const data = new Float32Array([1, 2, 3, 4, 5, 6]);
  const matrix = numkong.Matrix.fromTypedArray(data, 2, 3);
  assert.strictEqual(matrix.rows, 2);
  assert.strictEqual(matrix.cols, 3);
  const arr = matrix.toTypedArray();
  assertAlmostEqual(arr[0], 1.0, 0.001);
  assertAlmostEqual(arr[5], 6.0, 0.001);
});

test("Vector resize within capacity, reserve, and clear", () => {
  const v = new numkong.Vector(8, numkong.DType.F32); // capacity 8
  assert.strictEqual(v.capacity, 8);
  assert.strictEqual(v.tryResize(4), true);
  assert.strictEqual(v.length, 4);
  assert.strictEqual(v.capacity, 8); // capacity unchanged by resize
  assert.strictEqual(v.toTypedArray().length, 4);
  assert.strictEqual(v.tryResize(9), false); // beyond capacity, unchanged
  assert.strictEqual(v.length, 4);
  assert.strictEqual(v.reserve(32), true); // grow (may reallocate)
  assert(v.capacity >= 32);
  assert.strictEqual(v.tryResize(32), true);
  assert.strictEqual(v.length, 32);
  v.clear();
  assert.strictEqual(v.length, 0);
  assert(v.capacity >= 32); // capacity retained
});

test("Vector reserve preserves contents", () => {
  const v = numkong.Vector.fromTypedArray(new Float32Array([1, 2, 3, 4]));
  assert.strictEqual(v.reserve(64), true);
  assert(v.capacity >= 64);
  assert.strictEqual(v.length, 4);
  assert.deepStrictEqual(Array.from(v.toTypedArray()), [1, 2, 3, 4]);
});

test("Matrix resize within capacity, reserve, and clear", () => {
  const m = new numkong.Matrix(4, 4, numkong.DType.F32); // capacity 16
  assert.strictEqual(m.capacity, 16);
  assert.strictEqual(m.tryResize(2, 4), true);
  assert.strictEqual(m.rows, 2);
  assert.strictEqual(m.cols, 4);
  assert.strictEqual(m.rowStride, 4 * 4); // C-contiguous strides re-derived
  assert.strictEqual(m.toTypedArray().length, 8);
  assert.strictEqual(m.tryResize(5, 4), false); // 20 > 16, unchanged
  assert.strictEqual(m.rows, 2);
  assert.strictEqual(m.reserve(64), true);
  assert(m.capacity >= 64);
  assert.strictEqual(m.tryResize(8, 8), true);
  assert.strictEqual(m.rows, 8);
  m.clear();
  assert.strictEqual(m.rows, 0);
  assert.strictEqual(m.cols, 0);
});

test("Packed GEMM (dotsPacked)", () => {
  // A is 4x3, B is 5x3 — result should be 4x5 (A @ B.T)
  const aData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const bData = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]);

  const matA = numkong.Matrix.fromTypedArray(aData, 4, 3, numkong.DType.F32);
  const matB = numkong.Matrix.fromTypedArray(bData, 5, 3, numkong.DType.F32);

  const packed = numkong.dotsPack(matB);
  assert.strictEqual(packed.width, 5);
  assert.strictEqual(packed.depth, 3);
  assert.strictEqual(packed.disposed, false);

  const result = numkong.dotsPacked(matA, packed);
  assert.strictEqual(result.rows, 4);
  assert.strictEqual(result.cols, 5);

  const resultArr = result.toTypedArray();

  // Manual A @ B.T computation:
  // Row 0 of A = [1,2,3], dots with each row of B:
  //   [1,0,0]=1, [0,1,0]=2, [0,0,1]=3, [1,1,0]=3, [1,1,1]=6
  assertAlmostEqual(resultArr[0], 1.0, 0.01);
  assertAlmostEqual(resultArr[1], 2.0, 0.01);
  assertAlmostEqual(resultArr[2], 3.0, 0.01);
  assertAlmostEqual(resultArr[3], 3.0, 0.01);
  assertAlmostEqual(resultArr[4], 6.0, 0.01);
  // Row 1 of A = [4,5,6]:
  //   [1,0,0]=4, [0,1,0]=5, [0,0,1]=6, [1,1,0]=9, [1,1,1]=15
  assertAlmostEqual(resultArr[5], 4.0, 0.01);
  assertAlmostEqual(resultArr[6], 5.0, 0.01);
  assertAlmostEqual(resultArr[7], 6.0, 0.01);
  assertAlmostEqual(resultArr[8], 9.0, 0.01);
  assertAlmostEqual(resultArr[9], 15.0, 0.01);
});

test("Packed GEMM with pre-allocated out", () => {
  const aData = new Float32Array([1, 2, 3, 4, 5, 6]);
  const bData = new Float32Array([1, 0, 0, 0, 1, 0]);
  const matA = numkong.Matrix.fromTypedArray(aData, 2, 3, numkong.DType.F32);
  const matB = numkong.Matrix.fromTypedArray(bData, 2, 3, numkong.DType.F32);

  const packed = numkong.dotsPack(matB);
  const out = new numkong.Matrix(2, 2, numkong.DType.F64);
  const result = numkong.dotsPacked(matA, packed, out);
  assert.strictEqual(result, out); // same object returned
  const arr = result.toTypedArray();
  assertAlmostEqual(arr[0], 1.0, 0.01); // [1,2,3]·[1,0,0]
  assertAlmostEqual(arr[1], 2.0, 0.01); // [1,2,3]·[0,1,0]
  assertAlmostEqual(arr[2], 4.0, 0.01); // [4,5,6]·[1,0,0]
  assertAlmostEqual(arr[3], 5.0, 0.01); // [4,5,6]·[0,1,0]
});

test("Symmetric GEMM (dotsSymmetric) — upper triangle", () => {
  // 4x3 matrix of vectors
  const mData = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]);
  const mat = numkong.Matrix.fromTypedArray(mData, 4, 3, numkong.DType.F32);
  const result = numkong.dotsSymmetric(mat);
  assert.strictEqual(result.rows, 4);
  assert.strictEqual(result.cols, 4);

  const arr = result.toTypedArray();
  // Diagonal = self-dot-products
  assertAlmostEqual(arr[0 * 4 + 0], 1.0, 0.01); // [1,0,0]·[1,0,0]
  assertAlmostEqual(arr[1 * 4 + 1], 1.0, 0.01); // [0,1,0]·[0,1,0]
  assertAlmostEqual(arr[2 * 4 + 2], 1.0, 0.01); // [0,0,1]·[0,0,1]
  assertAlmostEqual(arr[3 * 4 + 3], 3.0, 0.01); // [1,1,1]·[1,1,1]

  // Upper triangle only
  assertAlmostEqual(arr[0 * 4 + 1], 0.0, 0.01); // [1,0,0]·[0,1,0]
  assertAlmostEqual(arr[0 * 4 + 2], 0.0, 0.01); // [1,0,0]·[0,0,1]
  assertAlmostEqual(arr[0 * 4 + 3], 1.0, 0.01); // [1,0,0]·[1,1,1]
  assertAlmostEqual(arr[1 * 4 + 2], 0.0, 0.01); // [0,1,0]·[0,0,1]
  assertAlmostEqual(arr[1 * 4 + 3], 1.0, 0.01); // [0,1,0]·[1,1,1]
  assertAlmostEqual(arr[2 * 4 + 3], 1.0, 0.01); // [0,0,1]·[1,1,1]
});

test("Symmetric with rowStart/rowCount partitioning", () => {
  const mData = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]);
  const mat = numkong.Matrix.fromTypedArray(mData, 4, 3, numkong.DType.F32);

  // Compute full result for reference
  const full = numkong.dotsSymmetric(mat);
  const fullArr = full.toTypedArray();

  // Compute rows 0-1 and rows 2-3 separately into same out
  const out = new numkong.Matrix(4, 4, numkong.DType.F64);
  numkong.dotsSymmetric(mat, out, { rowStart: 0, rowCount: 2 });
  numkong.dotsSymmetric(mat, out, { rowStart: 2, rowCount: 2 });
  const outArr = out.toTypedArray();

  // Verify upper triangle matches full computation
  for (let i = 0; i < 4; i++) {
    for (let j = i; j < 4; j++) {
      assertAlmostEqual(outArr[i * 4 + j], fullArr[i * 4 + j], 0.01);
    }
  }
});

test("angularsPacked — valid range", () => {
  const aData = new Float32Array([1, 2, 3, 4, 5, 6]);
  const bData = new Float32Array([1, 0, 0, 0, 1, 0]);
  const matA = numkong.Matrix.fromTypedArray(aData, 2, 3, numkong.DType.F32);
  const matB = numkong.Matrix.fromTypedArray(bData, 2, 3, numkong.DType.F32);
  const packed = numkong.dotsPack(matB);
  const result = numkong.angularsPacked(matA, packed);
  const arr = result.toTypedArray();
  for (let i = 0; i < arr.length; i++) {
    assert(arr[i] >= -0.01, `Angular distance should be non-negative, got ${arr[i]}`);
    assert(arr[i] <= 1.01, `Angular distance should be <= 1, got ${arr[i]}`);
  }
});

test("euclideansPacked — non-negative", () => {
  const aData = new Float32Array([1, 2, 3, 4, 5, 6]);
  const bData = new Float32Array([1, 0, 0, 0, 1, 0]);
  const matA = numkong.Matrix.fromTypedArray(aData, 2, 3, numkong.DType.F32);
  const matB = numkong.Matrix.fromTypedArray(bData, 2, 3, numkong.DType.F32);
  const packed = numkong.dotsPack(matB);
  const result = numkong.euclideansPacked(matA, packed);
  const arr = result.toTypedArray();
  for (let i = 0; i < arr.length; i++) {
    assert(arr[i] >= -0.01, `Euclidean distance should be non-negative, got ${arr[i]}`);
  }
});

test("angularsSymmetric — diagonal zero, upper triangle valid", () => {
  const mData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const mat = numkong.Matrix.fromTypedArray(mData, 3, 3, numkong.DType.F32);
  const result = numkong.angularsSymmetric(mat);
  const arr = result.toTypedArray();
  // Diagonal should be 0 (distance from self)
  assertAlmostEqual(arr[0 * 3 + 0], 0.0, 0.01);
  assertAlmostEqual(arr[1 * 3 + 1], 0.0, 0.01);
  assertAlmostEqual(arr[2 * 3 + 2], 0.0, 0.01);
  // Upper triangle should have valid angular distances [0, 1]
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      assert(
        arr[i * 3 + j] >= -0.01 && arr[i * 3 + j] <= 1.01,
        `Angular symmetric [${i},${j}] = ${arr[i * 3 + j]} out of range`,
      );
    }
  }
});

test("euclideansSymmetric — diagonal zero, upper triangle non-negative", () => {
  const mData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const mat = numkong.Matrix.fromTypedArray(mData, 3, 3, numkong.DType.F32);
  const result = numkong.euclideansSymmetric(mat);
  const arr = result.toTypedArray();
  // Diagonal should be 0
  assertAlmostEqual(arr[0 * 3 + 0], 0.0, 0.01);
  assertAlmostEqual(arr[1 * 3 + 1], 0.0, 0.01);
  assertAlmostEqual(arr[2 * 3 + 2], 0.0, 0.01);
  // Upper triangle should be non-negative
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      assert(arr[i * 3 + j] >= -0.01, `Euclidean symmetric [${i},${j}] = ${arr[i * 3 + j]} should be non-negative`);
    }
  }
});

test("PackedMatrix lifecycle", () => {
  const bData = new Float32Array([1, 0, 0, 0, 1, 0]);
  const matB = numkong.Matrix.fromTypedArray(bData, 2, 3, numkong.DType.F32);
  const packed = numkong.dotsPack(matB);

  assert.strictEqual(packed.disposed, false);
  packed.dispose();
  assert.strictEqual(packed.disposed, true);
  // Double dispose should not throw
  packed.dispose();
  assert.strictEqual(packed.disposed, true);
});

test("Cast compatibility with Matrix.toTypedArray", () => {
  const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
  const matrix = numkong.Matrix.fromTypedArray(data, 2, 2, numkong.DType.F32);
  const arr = matrix.toTypedArray();
  assert(arr instanceof Float32Array);

  // Cast to bf16 via existing cast pipeline
  const target = new Uint16Array(4);
  numkong.cast(arr, "f32", target, "bf16");
  // Cast back to verify round-trip
  const back = new Float32Array(4);
  numkong.cast(target, "bf16", back, "f32");
  assertAlmostEqual(back[0], 1.0, 0.1);
  assertAlmostEqual(back[3], 4.0, 0.1);
});

test("dotsPackedSize", () => {
  const size = numkong.dotsPackedSize(5, 3, numkong.DType.F32);
  assert(size > 0, "Packed size should be positive");
  assert(typeof size === "number", "Packed size should be a number");
});
