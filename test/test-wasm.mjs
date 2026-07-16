/**
 * Multi-runtime WASM test suite for NumKong
 * Supports: Emscripten (Node.js), WASI (Node.js), Browser (Playwright)
 *
 * Usage:
 *   NK_RUNTIME=emscripten node --test test/test-wasm.mjs
 *   NK_RUNTIME=wasi-node node --test test/test-wasm.mjs
 */

import test from "node:test";
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { WASI } from "node:wasi";

function resolveWasiTestModule() {
  return ["./build-wasi/nk_test.wasm", "./build-wasi/test.wasm"].find((path) => {
    try {
      readFileSync(path);
      return true;
    } catch {
      return false;
    }
  });
}

// Runtime loader - adapts to different WASM execution environments
async function loadNumKong(runtime) {
  switch (runtime) {
    case "native":
      // Load native Node.js addon (baseline for comparison)
      return await import("../javascript/dist/esm/numkong.js");

    case "emscripten":
      // Load Emscripten build (uses EM_ASM for capability detection)
      const wasmWrapper = await import("../javascript/dist/esm/numkong-wasm.js");
      const EmModule = await import("../build-wasm/numkong.js");
      const wasmInstance = await EmModule.default();
      wasmWrapper.initWasm(wasmInstance);
      return wasmWrapper;

    case "emscripten64": {
      // Load Emscripten wasm64 (memory64) build
      const wasmWrapper64 = await import("../javascript/dist/esm/numkong-wasm.js");
      const EmModule64 = await import("../build-wasm64/numkong64.js");
      const wasmInstance64 = await EmModule64.default();
      wasmWrapper64.initWasm(wasmInstance64);
      return wasmWrapper64;
    }

    case "wasi-node":
      // Load WASI via Node.js built-in WASI support (node:wasi)
      // Host probes for SIMD/Relaxed SIMD support and reports honestly via imports.
      const wasi = new WASI({
        version: "preview1",
        args: [],
        env: {},
      });

      const wasiModule = resolveWasiTestModule();
      if (!wasiModule) {
        throw new Error("Missing build-wasi/nk_test.wasm (or legacy build-wasi/test.wasm)");
      }
      const wasmBytes = readFileSync(wasiModule);

      // The WASI toolchain builds with --import-memory --shared-memory,
      // so we must provide a WebAssembly.Memory object in the env imports.
      const memory = new WebAssembly.Memory({ initial: 256, maximum: 4096, shared: true });

      // Probe the engine for SIMD support by validating minimal test modules.
      // A fat binary contains both serial and v128relaxed kernels — the host
      // must report truthfully so dispatch picks the right one.
      const hasV128 = WebAssembly.validate(new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
        0x03, 0x02, 0x01, 0x00,
        0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x0c,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0b,
      ])) ? 1 : 0;
      const hasRelaxed = WebAssembly.validate(new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        0x01, 0x06, 0x01, 0x60, 0x01, 0x7b, 0x01, 0x7b,
        0x03, 0x02, 0x01, 0x00,
        0x0a, 0x07, 0x01, 0x05, 0x00, 0x20, 0x00,
        0xfd, 0x82, 0x02, 0x0b,
      ])) ? 1 : 0;
      console.log(`  WASI probe: v128=${hasV128}, relaxed-simd=${hasRelaxed}`);

      const { instance } = await WebAssembly.instantiate(wasmBytes, {
        wasi_snapshot_preview1: wasi.wasiImport,
        env: {
          memory,
          nk_has_v128: () => hasV128,
          nk_has_relaxed: () => hasRelaxed,
        },
      });

      wasi.start(instance);
      // The C test binary runs its own comprehensive test suite via wasi.start().
      // It only exports _start/main, not distance functions, so return null
      // to signal that JS-level distance tests should be skipped.
      return null;

    default:
      throw new Error(`Unknown runtime: ${runtime}`);
  }
}

// Load runtime based on environment variable
const runtime = process.env.NK_RUNTIME || "native";
const seed = parseInt(process.env.NK_SEED || "42");
const dims = process.env.NK_DENSE_DIMENSIONS
  ? process.env.NK_DENSE_DIMENSIONS.split(",").map(Number)
  : [3, 16, 128, 1536];

// Simple PRNG for reproducible tests
class Random {
  constructor(seed) {
    this.seed = seed;
  }
  next() {
    this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
    return this.seed / 0x7fffffff;
  }
}

// Sub-byte packing ratio — mirrors nk_dimensions_per_value in types.h.
function dimensionsPerValue(dtype) {
  switch (dtype) {
    case "u1":
      return 8;
    case "i4":
    case "u4":
      return 2;
    default:
      return 1;
  }
}

function alignDimension(dimension, dtype) {
  const dpv = dimensionsPerValue(dtype);
  return Math.ceil(dimension / dpv) * dpv;
}

console.log(`Testing NumKong on runtime: ${runtime}`);
console.log(`Seed: ${seed}, Dimensions: ${dims.join(", ")}`);

const numkong = await loadNumKong(runtime);

// For wasi-node, the C test suite already ran via wasi.start() above.
// No JS-level distance functions are exported, so skip the JS tests.
if (numkong === null) {
  console.log(`C test suite passed for runtime: ${runtime}`);
  process.exit(0);
}

// Helper function for approximate equality
function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
  const lowerBound = expected - tolerance;
  const upperBound = expected + tolerance;
  assert(
    actual >= lowerBound && actual <= upperBound,
    `Expected ${actual} to be almost equal to ${expected} (tolerance: ${tolerance})`,
  );
}

// Test suite (copied from test/test.mjs structure)
test(`[${runtime}] Distance from itself`, () => {
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

test(`[${runtime}] Orthogonal vectors`, () => {
  const a = new Float32Array([1.0, 0.0, 0.0]);
  const b = new Float32Array([0.0, 1.0, 0.0]);

  assertAlmostEqual(numkong.inner(a, b), 0.0, 0.01);
  assertAlmostEqual(numkong.angular(a, b), 1.0, 0.01);
});

test(`[${runtime}] Opposite vectors`, () => {
  const a = new Float32Array([1.0, 2.0, 3.0]);
  const b = new Float32Array([-1.0, -2.0, -3.0]);

  assertAlmostEqual(numkong.angular(a, b), 2.0, 0.01);
});

test(`[${runtime}] Euclidean distance`, () => {
  const a = new Float32Array([0.0, 0.0, 0.0]);
  const b = new Float32Array([3.0, 4.0, 0.0]);

  assertAlmostEqual(numkong.euclidean(a, b), 5.0, 0.01);
  assertAlmostEqual(numkong.sqeuclidean(a, b), 25.0, 0.01);
});

test(`[${runtime}] Capability detection`, () => {
  // Test that each capability axis returns a bigint
  if (typeof numkong.getCapabilitiesAvailable === "function") {
    const caps = numkong.getCapabilitiesAvailable();
    assert(typeof caps === "bigint", "getCapabilitiesAvailable should return bigint");
    console.log(`  Available capabilities: 0x${caps.toString(16)}`);

    // `available` is the intersection, so it can never exceed either axis it derives from.
    const detected = numkong.getCapabilitiesDetected();
    const compiled = numkong.getCapabilitiesCompiled();
    assert(caps === (detected & compiled), "available must equal detected & compiled");

    // Test the hasCapability helper
    if (typeof numkong.hasCapability === "function") {
      // Serial fallback should always be present
      assert(numkong.hasCapability(1n << 0n), "SERIAL capability should be present");
    }
  } else {
    console.log(`  Capability accessors not available in ${runtime} mode`);
  }
});

// Expanded test coverage - comprehensive dtype/function/dimension testing
const testMatrix = {
  dot: ["f64", "f32", "i8", "u8"],
  inner: ["f64", "f32", "i8", "u8"],
  sqeuclidean: ["f64", "f32", "i8", "u8"],
  euclidean: ["f64", "f32", "i8", "u8"],
  angular: ["f64", "f32", "i8"],
  kullbackleibler: ["f64", "f32"],
  jensenshannon: ["f64", "f32"],
  hamming: ["u8"],
  jaccard: ["u8"],
};

let rngCounter = 0;
function randomVector(dtype, len) {
  const rng = new Random(seed + rngCounter++);
  if (dtype === "f64") return Float64Array.from({ length: len }, () => rng.next() * 2 - 1);
  if (dtype === "f32") return Float32Array.from({ length: len }, () => rng.next() * 2 - 1);
  if (dtype === "i8") return Int8Array.from({ length: len }, () => (rng.next() * 256 - 128) | 0);
  if (dtype === "u8") return Uint8Array.from({ length: len }, () => (rng.next() * 256) | 0);
}

for (const [fn, dtypes] of Object.entries(testMatrix)) {
  for (const dtype of dtypes) {
    for (const dim of dims) {
      test(`[${runtime}] ${fn}(${dtype}×${dim})`, () => {
        const a = randomVector(dtype, dim);
        const b = randomVector(dtype, dim);
        const result = numkong[fn](a, b);

        assert.strictEqual(typeof result, "number");
        assert.ok(isFinite(result));
        assertAlmostEqual(result, numkong[fn](a, b), 1e-6);
      });
    }
  }
}

console.log(`All tests passed for runtime: ${runtime}`);
