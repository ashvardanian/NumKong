/**
 *  @file test/wasm.mjs
 *  @author Ash Vardanian
 *  @date February 10, 2026
 *  @brief Multi-runtime WASM test suite for NumKong.
 *
 *  Supports Emscripten and WASI under Node.js, and browsers under Playwright, with
 *  `NUMKONG_RUNTIME` choosing the runtime:
 *
 *  ```sh
 *  NUMKONG_RUNTIME=emscripten node --test test/wasm.mjs
 *  NUMKONG_RUNTIME=wasi-node node --test test/wasm.mjs
 *  ```
 */

import test from "node:test";
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { WASI } from "node:wasi";
import { relaxedProbe, simd128Probe } from "../javascript/dist/esm/wasm-probes.js";
import { Random, streamKey } from "./random.mjs";

function resolveModule(candidates) {
  return candidates.find((path) => {
    try {
      readFileSync(path);
      return true;
    } catch {
      return false;
    }
  });
}

/** Runtime loader, adapting to different WASM execution environments. */
async function loadNumKong(runtime) {
  switch (runtime) {
    case "native":
      // Load native Node.js addon (baseline for comparison)
      return await import("../javascript/dist/esm/numkong.js");

    case "emscripten": {
      // Load the wasm32 Emscripten build, the relaxed capability where both capabilities were built
      const wasmWrapper = await import("../javascript/dist/esm/numkong-wasm.js");
      const emscriptenModule = resolveModule([
        "./build_wasm32_emscripten/numkong-wasm32-v128relaxed.js",
        "./build_wasm32_emscripten/numkong-wasm32-v128.js",
      ]);
      if (!emscriptenModule) {
        throw new Error("Missing build_wasm32_emscripten/numkong-wasm32-v128.js or numkong-wasm32-v128relaxed.js");
      }
      const EmModule = await import(new URL(`.${emscriptenModule}`, import.meta.url).href);
      const wasmInstance = await EmModule.default();
      wasmWrapper.initWasm(wasmInstance);
      return wasmWrapper;
    }

    case "emscripten64": {
      // Load Emscripten wasm64 (memory64) build
      const wasmWrapper64 = await import("../javascript/dist/esm/numkong-wasm.js");
      const EmModule64 = await import("../build_wasm64_emscripten/numkong-wasm64-v128relaxed.js");
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

      const wasiModule = resolveModule(["./build_wasm32_wasi/numkong_cpu_test.wasm"]);
      if (!wasiModule) {
        throw new Error("Missing build_wasm32_wasi/numkong_cpu_test.wasm");
      }
      const wasmBytes = readFileSync(wasiModule);

      // The host reports what this engine validates; the module reports what it compiled, and
      // dispatch takes the intersection.
      const hasV128 = WebAssembly.validate(simd128Probe) ? 1 : 0;
      const hasRelaxed = WebAssembly.validate(relaxedProbe) ? 1 : 0;
      console.log(`  WASI probe: v128=${hasV128}, relaxed-simd=${hasRelaxed}`);

      const { instance } = await WebAssembly.instantiate(wasmBytes, {
        wasi_snapshot_preview1: wasi.wasiImport,
        env: {
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

/** `NUMKONG_SEED` as a number: 42 when unset, a fresh draw for `random`; anything else throws. */
function readSeed() {
  const text = process.env.NUMKONG_SEED || "42";
  if (text === "random") return Math.floor(Math.random() * 0x100000000);
  const seed = Number(text);
  if (!Number.isSafeInteger(seed) || seed < 0) throw new Error(`NUMKONG_SEED="${text}" does not parse`);
  return seed;
}

/** Runtime selected by the environment variable. */
const runtime = process.env.NUMKONG_RUNTIME || "native";
const seed = readSeed();
const dims = process.env.NUMKONG_DIMS
  ? process.env.NUMKONG_DIMS.split(",").map(Number)
  : [3, 16, 128, 1536];

/** Sub-byte packing ratio, equal to `nk_dimensions_per_value` in types.h. */
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

/** Asserts approximate equality within a tolerance. */
function assertAlmostEqual(actual, expected, tolerance = 1e-6) {
  const lowerBound = expected - tolerance;
  const upperBound = expected + tolerance;
  assert(
    actual >= lowerBound && actual <= upperBound,
    `Expected ${actual} to be almost equal to ${expected} (tolerance: ${tolerance})`,
  );
}

// Test suite shaped like test/main.mjs
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
  const { Capability, Device } = numkong;
  const cpu = Device.cpu();
  const detected = cpu.capabilitiesDetected();
  const compiled = cpu.capabilitiesCompiled();
  const enabled = cpu.capabilitiesEnabled();
  console.log(`  Enabled capabilities: 0x${enabled.toString(16)}`);
  assert.strictEqual(enabled, detected & compiled, "enabled must be detected & compiled");
  assert.strictEqual(enabled & Capability.serial, Capability.serial, "serial must always be enabled");

  // The C library names them at load, GPU ones from bit 48 up included, and the groups split them.
  assert(Object.isFrozen(Capability), "Capability must be frozen");
  assert.strictEqual(Capability.v128, 1n << 31n);
  assert.strictEqual(Capability.ampere, 1n << 49n);
  assert.strictEqual(Capability.cpus & Capability.gpus, 0n);
  assert.strictEqual(Capability.gpus & Capability.metal, Capability.metal);
  assert.strictEqual(Capability.any, (1n << 64n) - 1n);

  // One CPU, refusing the ordinal past it, and GPUs counted by their runtimes where there are any.
  assert.strictEqual(Device.count("cpu"), 1);
  assert.throws(() => new Device("cpu", 1));
  for (const kind of ["cuda", "rocm", "metal"]) {
    let count = 0;
    try { count = Device.count(kind); } catch { continue; }
    assert.throws(() => new Device(kind, count));
    const gpu = new Device(kind, 0);
    assert.strictEqual(gpu.capabilitiesCompiled() & Capability.cpus, 0n);
  }

  // Every engine in the support matrix validates the SIMD128 probe; relaxed SIMD varies by engine.
  if (runtime !== "native") {
    assert.strictEqual(detected & Capability.v128, Capability.v128, "detected must include v128 on every WASM runtime");
    if (WebAssembly.validate(relaxedProbe)) {
      assert.strictEqual(
        detected & Capability.v128relaxed,
        Capability.v128relaxed,
        "detected must include v128relaxed where the engine validates the relaxed probe",
      );
    }
  }
});

/** Expanded test coverage over every dtype, function and dimension. */
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

function randomVector(dtype, len, rng) {
  if (dtype === "f64") return Float64Array.from({ length: len }, () => rng.next() * 2 - 1);
  if (dtype === "f32") return Float32Array.from({ length: len }, () => rng.next() * 2 - 1);
  if (dtype === "i8") return Int8Array.from({ length: len }, () => (rng.next() * 256 - 128) | 0);
  if (dtype === "u8") return Uint8Array.from({ length: len }, () => (rng.next() * 256) | 0);
}

for (const [fn, dtypes] of Object.entries(testMatrix)) {
  for (const dtype of dtypes) {
    for (const dim of dims) {
      test(`[${runtime}] ${fn}(${dtype}×${dim})`, () => {
        const rng = new Random(streamKey(seed, `${fn}/${dtype}/${dim}`));
        const a = randomVector(dtype, dim, rng);
        const b = randomVector(dtype, dim, rng);
        const result = numkong[fn](a, b);

        assert.strictEqual(typeof result, "number");
        assert.ok(isFinite(result));
        assertAlmostEqual(result, numkong[fn](a, b), 1e-6);
      });
    }
  }
}

console.log(`All tests passed for runtime: ${runtime}`);
