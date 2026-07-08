# NumKong for JavaScript

NumKong's JavaScript package provides vector kernels for Node and Bun-style runtimes, targeting the space between handwritten loops over `TypedArray`s and larger tensor frameworks.
It keeps the JS surface intentionally compact: dense distances, dot products, binary metrics, probability divergences, packed GEMM-like matrix multiplication, symmetric Gram matrices, dtype-tagged low-precision storage, typed views, and runtime capability inspection.

## Quickstart

```ts
import { dot } from "numkong";

const a = new Float32Array([1, 2, 3]);
const b = new Float32Array([4, 5, 6]);
console.log(dot(a, b)); // 32
```

## Highlights

This SDK is deliberately smaller than Python or Rust.
Its job is to make the hot vector kernels easy to use from modern JavaScript runtimes.

__TypedArray-first API.__
Standard `Float32Array`, `Float64Array`, `Int8Array`, and `Uint8Array` work directly.
__DType tags for exotic storage.__
`f16`, `bf16`, fp8, and packed bits stay explicit.
__Owned and borrowed views.__
`Vector`, `VectorView`, and base tensor wrappers preserve dtype metadata.
__Portable runtime story.__
The same package can target native addons and WASM runtimes.
__No fake tensor-framework scope.__
This binding stays centered on the vector families it actually exports.

## Ecosystem Comparison

| Feature                      | NumKong                                                        | [mathjs](https://github.com/josdejong/mathjs) | [tensorflow.js](https://github.com/tensorflow/tfjs) |
| ---------------------------- | -------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------- |
| Operation families           | dots, distances, binary, probability, cast, packed, symmetric  | general arithmetic, matrix ops, statistics    | matmul, elementwise, reductions                     |
| Precision                    | BFloat16 through sub-byte; automatic widening; Kahan summation | Float64 only; standard accuracy               | Float32 primarily; no sub-byte; standard accuracy   |
| Runtime SIMD dispatch        | auto-selects best ISA per-thread across x86, ARM, RISC-V       | none; pure JS                                 | fixed at build time via WASM SIMD or WebGL          |
| Packed matrix, GEMM-like     | `dotsPack` + `dotsPacked`; persistent packing; amortized       | `math.multiply` — no persistent packing       | `tf.matMul` — no persistent packing                 |
| Symmetric kernels, SYRK-like | `dotsSymmetric`; upper triangle only; row-range partitioning   | no duplicate-pair skipping                    | no duplicate-pair skipping                          |
| WASM fallback                | yes — portable, runs in browser without native addon           | yes — pure JS, no native required             | yes — also WebGL/WebGPU                             |
| Bundle size                  | small                                                          | moderate                                      | large                                               |

## Installation

The package targets Node `>= 22`.

```sh
npm install numkong
yarn add numkong
pnpm add numkong
bun add numkong
```

If you build from source, the package uses `node-gyp-build` on install and TypeScript sources under `javascript/`.

## Browser and WASM

The npm package includes a pre-built WASM bundle under `wasm/`.
The simplest way to use it in a browser is via a CDN — no build step required:

```html
<script type="module">
  import { dot, euclidean } from 'https://cdn.jsdelivr.net/npm/numkong@7/wasm/numkong.js';

  const query = new Float32Array([1.0, 2.0, 3.0]);
  const doc = new Float32Array([4.0, 5.0, 6.0]);
  console.log(dot(query, doc));          // 32 — SIMD-accelerated, client-side
  console.log(euclidean(query, doc));    // 5.196...
</script>
```

For self-hosted WASM, download the binaries from a [GitHub Release](https://github.com/ashvardanian/NumKong/releases) and serve them from the same directory:

```html
<script type="module">
  import * as numkong from "./numkong-emscripten.js";
  import NumKongModule from "./numkong.js";

  const wasm = await NumKongModule();
  numkong.initWasm(wasm);

  const a = new Float32Array([1, 2, 3]);
  const b = new Float32Array([4, 5, 6]);
  console.log(numkong.dot(a, b));
</script>
```

Or import the subpath from a bundler or Node.js (without the native addon):

```js
import { dot } from "numkong/wasm";
```

## Dot Products

Dot products are separate from distances because dtype tagging and low-precision storage matter more here.

```ts
import { dot, inner } from "numkong";

const a = new Float32Array([1, 2, 3]);
const b = new Float32Array([4, 5, 6]);

console.log(dot(a, b));
console.log(inner(a, b)); // alias for ecosystem familiarity
```

For non-native numeric layouts, pass an explicit `DType` or wrap the storage in a typed NumKong view.

## Dense Distances

Dense distance entrypoints work directly on the standard numeric `TypedArray` types.

```ts
import { sqeuclidean, euclidean, angular } from "numkong";

const a = new Float32Array([1, 2, 3]);
const b = new Float32Array([4, 5, 6]);

console.log(sqeuclidean(a, b)); // equivalent shape to a manual sum((a[i] - b[i]) ** 2)
console.log(euclidean(a, b));
console.log(angular(a, b));
```

When the storage type is one of the standard JS typed arrays, dtype inference is automatic.

## Binary Metrics

Binary metrics operate on packed storage rather than on generic boolean arrays.

```ts
import { toBinary, hamming, jaccard } from "numkong";

const a = toBinary(new Float32Array([1, -2, 3, -4, 5, -6, 7, -8]));
const b = toBinary(new Float32Array([1, 2, -3, -4, 5, 6, -7, -8]));

console.log(hamming(a, b));
console.log(jaccard(a, b));
```

This is a natural model for sign-quantized embeddings and semantic hashes.

## Probability Metrics

```ts
import { kullbackleibler, jensenshannon } from "numkong";

const p = new Float32Array([0.2, 0.3, 0.5]);
const q = new Float32Array([0.1, 0.3, 0.6]);

console.log(kullbackleibler(p, q));
console.log(jensenshannon(p, q));
```

These call the underlying SIMD divergence kernels directly.

## DType Tags and Low-Precision Arrays

JavaScript has no built-in `f16`, `bf16`, fp8, or packed-bit numeric model.
NumKong handles that with explicit dtype tags and wrapper arrays.

The supported low-precision types and their bit layouts are:

- `Float16Array`: 1 sign + 5 exponent + 10 mantissa bits, 2 bytes per element, range ±65504, supports Inf and NaN
- `BFloat16Array`: 1 sign + 8 exponent + 7 mantissa bits, 2 bytes per element, full `f32` dynamic range, supports Inf and NaN
- `E4M3Array`: 1 sign + 4 exponent + 3 mantissa bits, 1 byte per element, range ±448, no Inf, NaN is 0x7F only
- `E5M2Array`: 1 sign + 5 exponent + 2 mantissa bits, 1 byte per element, range ±57344, supports Inf and NaN
- `BinaryArray`: packed bits in bytes, 8 elements per byte

Use `.byteLength` for the exact payload size.

```ts
import { Float16Array, E4M3Array, DType, dot, angular } from "numkong";

const a16 = new Float16Array([1, 2, 3]);
const b16 = new Float16Array([4, 5, 6]);
console.log(dot(a16, b16, DType.F16));
console.log(a16.byteLength);

const a8 = new E4M3Array([1, 2, 3]);
const b8 = new E4M3Array([4, 5, 6]);
console.log(angular(a8, b8, DType.E4M3));
console.log(a8.byteLength);
```

If the underlying storage is a raw `Uint16Array` or `Uint8Array`, JS itself cannot know whether those bytes mean integers, `f16`, `bf16`, mini-floats, or packed bits.
That is exactly when the `DType` tag becomes mandatory.
You can also pass it to `cast` to convert many values between all supported types:

```ts
numkong.cast(f32Source, "f32", bf16Dest, "bf16");
numkong.cast(f32Source, "f32", bf16Dest, DType.E4M3);
```

## Vector Views and Owned Buffers

The wrapper hierarchy exists to keep dtype and ownership explicit across addon and WASM boundaries.

- `TensorBase` carries `buffer`, `byteOffset`, and `dtype`
- `VectorBase` adds rank-1 semantics
- `VectorView` is a zero-copy borrowed wrapper over existing memory
- `Vector` owns its `ArrayBuffer`

```ts
import { VectorView, Vector, DType, dot } from "numkong";

const raw = new Uint16Array([0x3c00, 0x4000, 0x4200]); // f16 payload, not uint16 values
const view = VectorView.from(raw, DType.F16);

const owned = new Vector(3, DType.F32);
owned.toTypedArray().set(new Float32Array([4, 5, 6]));

console.log(dot(view, view));
console.log(dot(owned, owned));
console.log(owned.byteLength);
```

Use `VectorView` when the bytes already live somewhere else.
Use `Vector` when NumKong should own the storage.

Owned `Vector` and `Matrix` are fixed-capacity resizable.
`capacity` reports the allocated element ceiling.
`tryResize(...)` reshapes within capacity without moving the `ArrayBuffer`, so a `toTypedArray()` result stays valid.
`reserve(elements)` may allocate a new `ArrayBuffer` to grow, which __invalidates__ any `TypedArray` or `row()` handed out earlier.
`clear()` empties the length while keeping capacity.

## Capabilities and Runtime Selection

Capability detection is explicit:

```ts
import { Capability, getCapabilities, hasCapability } from "numkong";

const caps = getCapabilities();

console.log(caps);
console.log(hasCapability(Capability.HASWELL));
console.log(hasCapability(Capability.NEON));
```

The exact bitmask depends on whether you are running the native addon or a WASM runtime.

There is no `configure_thread` call in the JS binding.
Thread configuration is managed internally by the native addon or WASM runtime.

## Native Addon and WASM Runtimes

The top-level package is native-first.
It loads the compiled addon through `node-gyp-build`.

The repository also ships WASM wrappers.
Those are useful for portable or sandboxed environments.
They are not feature-complete replacements for the native SDKs.

Practical guidance:

- Use the native addon for lower host-call latency.
- Use the WASM path when portability matters more than absolute latency.
- Keep your expectations scoped to the vector and matrix-oriented API that this binding actually exports.
