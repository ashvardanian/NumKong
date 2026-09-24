/**
 *  @file bench/bench.js
 *  @author Ash Vardanian
 *  @date October 20, 2023
 *  @brief Angular-distance benchmarks of NumKong against pure JavaScript, MathJS and USearch.
 */

const benchmark = require("benchmark");
const math = require("mathjs");
const usearch = require("usearch");
const MetricKind = usearch.MetricKind;
const numkong = require("../javascript/dist/cjs/numkong.js");

/** Angular distance in pure JavaScript, assuming the vectors are of the same length. */
function angular(a, b) {
  let dotProduct = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return 1 - dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));
}

function angularMathJS(a, b) {
  let dotProduct = math.dot(a, b);
  let magA = math.sqrt(math.dot(a, a));
  let magB = math.sqrt(math.dot(b, b));
  return 1 - dotProduct / (magA * magB);
}

/** Random data for testing. */
const dimensions = 1536; // Adjust dimensions as needed
const array1 = Array.from({ length: dimensions }, () => Math.random() * 100);
const array2 = Array.from({ length: dimensions }, () => Math.random() * 100);
const mathm1 = math.matrix(
  Array.from({ length: dimensions }, () => Math.random() * 100)
);
const mathm2 = math.matrix(
  Array.from({ length: dimensions }, () => Math.random() * 100)
);
const floatArray1 = new Float32Array(array1);
const floatArray2 = new Float32Array(array2);
const intArray1 = new Int8Array(array1);
const intArray2 = new Int8Array(array2);

/** Random batch data for testing. */
const batchSize = 1000;
const matrix1 = Array.from(
  { length: dimensions * batchSize },
  () => Math.random() * 100
);
const matrix2 = Array.from(
  { length: dimensions * batchSize },
  () => Math.random() * 100
);
const floatMatrix1 = new Float32Array(matrix1);
const floatMatrix2 = new Float32Array(matrix2);
const intMatrix1 = new Int8Array(matrix1);
const intMatrix2 = new Int8Array(matrix2);

/** Benchmark suites. */
const singleSuite = new benchmark.Suite("Single Vector Processing");
const batchSuite = new benchmark.Suite("Batch Vector Processing");

// Single-vector processing benchmarks
singleSuite

  // Pure JavaScript
  .add("Array of Numbers", () => {
    angular(array1, array2);
  })
  .add("TypedArray of Float32", () => {
    angular(floatArray1, floatArray2);
  })
  .add("TypedArray of Int8", () => {
    angular(intArray1, intArray2);
  })

  // Math JS
  .add("Array of Numbers with MathJS", () => {
    angularMathJS(array1, array2);
  })
  .add("MathMatrix with MathJS", () => {
    angularMathJS(mathm1, mathm2);
  })

  // NumKong
  .add("TypedArray of Float32 with NumKong", () => {
    numkong.angular(floatArray1, floatArray2);
  })
  .add("TypedArray of Int8 with NumKong", () => {
    numkong.angular(intArray1, intArray2);
  })

  .on("cycle", (event) => {
    if (event.target.error) {
      console.error(String(event.target.error));
    } else {
      console.log(String(event.target));
    }
  })
  .on("complete", () => {
    console.log(
      "Fastest Single-Vector Processing is " +
        singleSuite.filter("fastest").map("name")
    );
  })
  .run({
    noCache: true,
    async: false,
  });

// Batch-vector processing benchmarks
batchSuite
  .add("2D Array of Numbers", () => {
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < batchSize; j++) {
        const start = i * dimensions;
        const end = start + dimensions;
        angular(matrix1.slice(start, end), matrix2.slice(start, end));
      }
    }
  })
  .add("2D TypedArray of Float32 with exact USearch", () => {
    usearch.exactSearch(
      floatMatrix1,
      floatMatrix2,
      dimensions,
      1,
      MetricKind.Cos
    );
  })
  .add("2D TypedArray of Int8 with exact USearch", () => {
    usearch.exactSearch(intMatrix1, intMatrix2, dimensions, 1, MetricKind.Cos);
  })
  .on("cycle", (event) => {
    if (event.target.error) {
      console.error(String(event.target.error));
    } else {
      console.log(String(event.target));
    }
  })
  .on("complete", () => {
    console.log(
      "Fastest Batch-Vector Processing is " +
        batchSuite.filter("fastest").map("name")
    );
  })
  .run({
    noCache: true,
    async: false,
  });
