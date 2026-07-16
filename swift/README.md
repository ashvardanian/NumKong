# NumKong for Swift

Apple Silicon is a power-efficient, high-throughput CPU-GPU combination widely used for on-device AI workloads across phones, tablets, and laptops.
NumKong provides hardware-accelerated vector math for Swift without pulling in a full tensor framework.
It gives you collection-based dense metrics, binary set distances, owning tensors, explicit matrix views, reusable packed matrices, symmetric all-pairs kernels, MaxSim late-interaction scoring, geospatial distance helpers, and storage wrappers for low-precision formats that Swift does not model natively.

Swift users usually want one of two things.
They either want ergonomic collection-based scalar metrics.
Or they want a compact matrix API for repeated retrieval-style workloads.
This package targets those two cases directly instead of pretending to be a full tensor framework.

## Quickstart

```swift
import NumKong

let a: [Float32] = [1, 2, 3]
let b: [Float32] = [4, 5, 6]
let dot = a.dot(b) // widened to Float64
print(dot as Any)
```

## Highlights

__Collection-first scalar API.__
Plain `[Float32]`, `[Float16]`, `[Int8]`, `[U1x8]`, and other wrapper arrays work directly.
__Owning tensors.__
`Tensor<T>` owns its storage, produces views and spans without nested pointer closures, and drives the matrix kernel API.
__Explicit matrix views.__
`MatrixView` and `MatrixSpan` make strides and ownership visible.
__Reusable packed matrices.__
`PackedMatrix` owns its internal packed buffer and can be reused across repeated queries.
__Binary metrics.__
`U1x8` packs 8 bits per byte; Hamming and Jaccard kernels operate directly on those packed words.
__MaxSim and ColBERT-style late interaction.__
`MaxSimPackedMatrix` and `.maxSimPack()` cover token-level late-interaction scoring.
__No hidden output allocation.__
You own the result buffers for matrix kernels.
__Low-precision wrappers.__
Storage wrappers preserve exact bits for bf16 and mini-float formats.
__Unaligned caller buffers are fine.__
Packing handles internal layout itself.

## Ecosystem Comparison

| Feature                      | NumKong                                                                                                                | Accelerate/vDSP                                                   | [MLX](https://github.com/ml-explore/mlx)                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| Operation families           | dots, distances, binary, geospatial, MaxSim                                                                            | dots, distances, FFT, some BLAS                                   | matmul, elementwise, reductions, FFT                               |
| Precision                    | BFloat16 through sub-byte — Float8, Float6, packed bits; automatic widening; Kahan summation; 0 ULP in Float32/Float64 | Float32/Float64, limited Float16; no auto-widening; IEEE defaults | Float16/BFloat16/Float32; no Float8 or sub-byte; backend-dependent |
| Runtime SIMD dispatch        | auto-selects best ISA per-thread at runtime across x86, ARM, RISC-V                                                    | Apple-only, no runtime ISA selection                              | GPU dispatch only, no CPU ISA selection                            |
| Packed matrix, GEMM-like     | `PackedMatrix` packs once, reused across query batches                                                                 | BLAS GEMM available                                               | GEMM via graph, implicit caching                                   |
| Symmetric kernels, SYRK-like | `dots_symmetric`, `angulars_symmetric`, etc. skip duplicate pairs, up to 2x speedup                                    | `cblas_ssyrk` available for rank-k updates                        | no duplicate-pair skipping                                         |
| Collection-based API         | works with any `RandomAccessCollection` conforming type                                                                | pointer-based vDSP functions                                      | `MLXArray`-based                                                   |
| Memory model                 | caller-owned buffers; `Tensor`/`PackedMatrix` own their storage                                                        | caller-managed via `UnsafePointer`                                | graph-managed; implicit allocation and caching                     |

## Installation

Add NumKong to `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ashvardanian/NumKong.git", from: "7.0")
]
```

Then add the product to your target:

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "NumKong", package: "NumKong")
    ]
)
```

The root package manifest already exposes the `NumKong` and `CNumKong` targets.
Xcode package integration uses the same URL.

## Collection-Based Dot Products

Dot products follow a collection-first shape.

```swift
import NumKong

let a: [UInt8] = [1, 2, 3, 4]
let b: [UInt8] = [4, 3, 2, 1]

let dot = a.dot(b) // widened to UInt32, not UInt8
print(dot as Any)
```

For `Float32`, the scalar result widens to `Float64`.
For `Float16`, it widens to `Float32`.
For `Int8`, it widens to `Int32`.
For `UInt8`, it widens to `UInt32`.

## Collection-Based Dense Distances

The collection extensions are the lightest entry point.
They are a good fit for per-vector retrieval and ranking work.

```swift
import NumKong

let a: [Float16] = [1, 2, 3, 4]
let b: [Float16] = [4, 3, 2, 1]

let sqeuclidean = a.sqeuclidean(b) // widens to Float32
let euclidean = a.euclidean(b)
let angular = a.angular(b)

print(sqeuclidean as Any, euclidean as Any, angular as Any)
```

The widening is deliberate.
That is the main difference from a naive same-storage implementation.

## Binary Metrics

Binary metrics work on packed words instead of boolean slices.
That is the right model once the workload is "semantic hash" or "binary embedding" rather than "array of booleans".
`U1x8` packs 8 bits into one byte.

```swift
import NumKong

// Each U1x8 holds 8 bits. Two elements = 16 bits total.
let a: [U1x8] = [U1x8(bitPattern: 0b10101010), U1x8(bitPattern: 0b11110000)]
let b: [U1x8] = [U1x8(bitPattern: 0b10101110), U1x8(bitPattern: 0b11000000)]

let hamming = a.hamming(b) // UInt32: count of differing bits
let jaccard = a.jaccard(b) // Float32: Jaccard distance in [0, 1]

print(hamming as Any, jaccard as Any)
```

`Hamming` returns `UInt32` — the count of differing bits across all packed words.
`Jaccard` returns `Float32` — the set-theoretic distance computed on bit populations.

## Owning Tensors and Memory Layout

`Tensor<T>` is the owning two-dimensional type.
It allocates its own buffer, handles deallocation on `deinit`, and produces non-owning views and spans without nesting `withUnsafeBufferPointer` closures.

```swift
import NumKong

// From an existing array:
let t = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6], rows: 2, cols: 3)

// Zero-initialized:
let z = try Tensor<Float32>.zeros(rows: 4, cols: 768)

// Constant fill:
let c = try Tensor<Float32>.full(rows: 4, cols: 768, value: 1.0)

// Subscript access:
let v = t[0, 2] // row 0, col 2

// Row buffer access:
let row1 = t.row(1) // UnsafeBufferPointer<Float32>

// Non-owning views:
let view: MatrixView<Float32>  = t.view()  // immutable
let span: MatrixSpan<Float32>  = t.span()  // mutable
```

`Tensor<T>` is fixed-capacity resizable.
`capacity` reports the allocated element ceiling.
`tryResize(rows:cols:)` reshapes within capacity without moving storage, so an existing `view()`, `span()`, or `row()` stays valid.
`reserve(_:)` may reallocate to grow, which __invalidates__ any view, span, or row pointer obtained earlier.
`clear()` empties the tensor while keeping capacity.

The view/span split is the same aliasing discipline used throughout the binding.
`MatrixView` is non-owning and immutable.
`MatrixSpan` is non-owning and mutable.
Neither allocates.

The ownership model is explicit:

- `MatrixView<Element>` is a non-owning immutable view.
- `MatrixSpan<Element>` is a non-owning mutable view.
- `PackedMatrix<Element>` owns one internal packed buffer and deallocates it on `deinit`.
- `Tensor<Element>` owns its element storage and deallocates it on `deinit`.

`PackedMatrix` allocates its internal payload with `UnsafeMutableRawPointer.allocate(byteCount:alignment:)`.
The alignment is 64 bytes for the owned packed buffer.
That does _not_ mean your source matrix must be aligned.
Packing accepts ordinary Swift-managed buffers and handles the internal layout itself.

The Tensor API eliminates the nested closure structure required when working directly with Swift's `withUnsafeBufferPointer`.
The difference in call-site verbosity is significant for anything more than a single kernel call:

```swift
// Without Tensor — three nested closures just to call one kernel
try a.withUnsafeBufferPointer { aPtr in
    try b.withUnsafeBufferPointer { bPtr in
        try out.withUnsafeMutableBufferPointer { outPtr in
            let aView = MatrixView(baseAddress: aPtr.baseAddress!, rows: 2, cols: 3)
            let bView = MatrixView(baseAddress: bPtr.baseAddress!, rows: 2, cols: 3)
            var cSpan = MatrixSpan(baseAddress: outPtr.baseAddress!, rows: 2, cols: 2)
            let packed = try PackedMatrix<Float32>(packing: bView)
            try dots_packed(aView, packed, &cSpan)
        }
    }
}

// With Tensor — no closures at the call site
let a = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6], rows: 2, cols: 3)
let b = try Tensor<Float32>.fromArray([7, 8, 9, 1, 0, 1], rows: 2, cols: 3)
let packed = try b.packForDots()
let c = try a.dotsPacked(packed) // returns Tensor<Float64>
```

## Matrix Views and Packed Kernels

Packed kernels are the GEMM-like throughput path.
They are useful when the right-hand side is reused across many query batches.

```swift
import NumKong

let a = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6], rows: 2, cols: 3)
let b = try Tensor<Float32>.fromArray([7, 8, 9, 1, 0, 1], rows: 2, cols: 3)

let packed = try b.packForDots()           // PackedMatrix<Float32>, owned
let dots   = try a.dotsPacked(packed)      // Tensor<Float64>, 2x2
let angs   = try a.angularsPacked(packed)  // Tensor<Float64>, 2x2
let eucs   = try a.euclideansPacked(packed)// Tensor<Float64>, 2x2

assert(dots.rows == 2 && dots.cols == 2)
```

The free-function API (`dots_packed`, `angulars_packed`, etc.) accepts `MatrixView`, `PackedMatrix`, and `MatrixSpan` directly for cases where you need manual buffer management — see the verbosity comparison in the Tensors section above.

## Symmetric Matrix Kernels

Symmetric kernels compute self-similarity or self-distance matrices.
They are the right shape for `SYRK`-like workloads and row-window partitioning.

```swift
import NumKong

let vectors = try Tensor<Float32>.fromArray([
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
], rows: 3, cols: 3)

let gram = try vectors.dotsSymmetric()       // Tensor<Float64>, 3x3
let dists = try vectors.euclideansSymmetric()// Tensor<Float64>, 3x3
let angs  = try vectors.angularsSymmetric()  // Tensor<Float64>, 3x3

assert(gram.rows == 3 && gram.cols == 3)
```

The free-function form (`dots_symmetric`, `angulars_symmetric`, etc.) exposes `rowStart` and `rowCount` parameters for external partitioning.

## Set Distance Kernels

Set distance kernels operate on `U1x8` matrices where each row is a packed binary vector.
The same packed and symmetric shapes available for dense metrics exist here.

```swift
import NumKong

// Eight binary vectors, each 16 bits wide (2 x U1x8 per row)
let rows = 8
let cols = 2
var rawBits = [U1x8](repeating: U1x8(bitPattern: 0b10101010), count: rows * cols)

let t = try rawBits.withUnsafeMutableBufferPointer { buf -> Tensor<U1x8> in
    let data = Array(buf)
    return try Tensor<U1x8>.fromArray(data, rows: rows, cols: cols)
}

let packed = try PackedMatrix<U1x8>(packing: t.view())

// Cross-matrix Hamming distances: shape [8, 8]
let hammings = try t.hammingsPacked(packed)  // Tensor<UInt32>

// Symmetric all-pairs Jaccard distances: shape [8, 8]
let jaccards = try t.jaccardsSymmetric()     // Tensor<Float32>

assert(hammings.rows == rows && hammings.cols == rows)
assert(jaccards.rows == rows && jaccards.cols == rows)
```

Free-function forms are also available:

```swift
try hammings_packed(view, packed, &span)
try jaccards_packed(view, packed, &span)
try hammings_symmetric(view, &span, rowStart: 0, rowCount: rows)
try jaccards_symmetric(view, &span, rowStart: 0, rowCount: rows)
```

## MaxSim and ColBERT-Style Late Interaction

MaxSim is the late-interaction primitive used by systems such as [ColBERT](https://arxiv.org/abs/2004.12832).
Each query is a small matrix of token vectors.
Each document is a small matrix of token vectors.
The score between a query and a document is the sum of maximum cosine similarities between each query token and any document token.
That is not a standard matrix multiply.

```swift
import NumKong

// 4 query tokens, each 16-dimensional
let queries = try Tensor<Float32>.full(rows: 4, cols: 16, value: 1.0)

// 8 document tokens, each 16-dimensional
let docs = try Tensor<Float32>.full(rows: 8, cols: 16, value: 1.0)

let queryPacked = try queries.maxSimPack()  // MaxSimPackedMatrix<Float32>
let docPacked   = try docs.maxSimPack()     // MaxSimPackedMatrix<Float32>

let score = queryPacked.score(docPacked)    // Float64
assert(score.isFinite)
```

`MaxSimPackedMatrix` can also be constructed directly from a `MatrixView`:

```swift
let view = queries.view()
let packed = try MaxSimPackedMatrix<Float32>(packing: view)
```

Supported types and their output types:

| Input type | Score output |
| ---------- | ------------ |
| `Float32`  | `Float64`    |
| `BFloat16` | `Float32`    |
| `Float16`  | `Float32`    |

`Float16` support is unavailable on x86-64 targets because Swift's `Float16` type is not available on that architecture.

## Low-Precision Storage Wrappers

Swift has no built-in bf16, mini-float, or packed-bit scalar types.
NumKong ships storage wrappers instead.

- __`BFloat16`__ — 1+8+7 bit layout (sign + exponent + mantissa), 2 bytes.
  Same dynamic range as `Float32` with reduced precision.
  Supports NaN and Inf.
- __`E4M3`__ — 1+4+3 bit layout, 1 byte.
  Range ±448.
  No Inf representation; NaN is encoded only as `0x7F` or `0xFF`.
- __`E5M2`__ — 1+5+2 bit layout, 1 byte.
  Range ±57344.
  Supports Inf and NaN.
- __`E2M3`__ — 1+2+3 bit layout, 1 byte (6 bits used).
  Range ±7.5.
  No Inf, no NaN.
- __`E3M2`__ — 1+3+2 bit layout, 1 byte (6 bits used).
  Range ±28.
  No Inf, no NaN.
- __`U1x8`__ — 8 packed bits per byte.
  Used for binary embeddings and semantic hashing.
  Supports Hamming and Jaccard scalar and matrix kernels.

Every floating-point wrapper provides `init(bitPattern:)`, `init(float:)`, and `var float: Float32`.
All are `@frozen`, `Equatable`, `Hashable`, `Sendable`.
`U1x8` provides `init(bitPattern:)` and exposes its underlying `UInt8` value.
These wrappers are exact-storage types first.
They are there to preserve bits and make the native kernels callable from Swift.
They are not pretending to be standard-library numeric types.

## Scalar Types and Promotions

The output type is intentionally wider than the storage type for most operations.
The table below documents the promotion for scalar collection extensions.

| Input type | `.dot()`  | `.angular()` | `.euclidean()` | `.sqeuclidean()` | `.hamming()` | `.jaccard()` |
| ---------- | --------- | ------------ | -------------- | ---------------- | ------------ | ------------ |
| `Float64`  | `Float64` | `Float64`    | `Float64`      | `Float64`        | —            | —            |
| `Float32`  | `Float64` | `Float64`    | `Float64`      | `Float64`        | —            | —            |
| `Float16`  | `Float32` | `Float32`    | `Float32`      | `Float32`        | —            | —            |
| `BFloat16` | `Float32` | `Float32`    | `Float32`      | `Float32`        | —            | —            |
| `Int8`     | `Int32`   | `Float32`    | `Float32`      | `UInt32`         | —            | —            |
| `UInt8`    | `UInt32`  | `Float32`    | `Float32`      | `UInt32`         | —            | —            |
| `U1x8`     | —         | —            | —              | —                | `UInt32`     | `Float32`    |

The matrix kernel output types follow a similar pattern but vary for the mini-float formats:

| Input type | Dots output | Spatial output | Hamming output | Jaccard output |
| ---------- | ----------- | -------------- | -------------- | -------------- |
| `Float32`  | `Float64`   | `Float64`      | —              | —              |
| `Float64`  | `Float64`   | `Float64`      | —              | —              |
| `Float16`  | `Float32`   | `Float32`      | —              | —              |
| `BFloat16` | `Float32`   | `Float32`      | —              | —              |
| `Int8`     | `Int32`     | `Float32`      | —              | —              |
| `UInt8`    | `UInt32`    | `Float32`      | —              | —              |
| `E4M3`     | `Float32`   | `Float32`      | —              | —              |
| `E5M2`     | `Float32`   | `Float32`      | —              | —              |
| `E2M3`     | `Float32`   | `Float32`      | —              | —              |
| `E3M2`     | `Float32`   | `Float32`      | —              | —              |
| `U1x8`     | `UInt32`    | —              | `UInt32`       | `Float32`      |

## Geospatial Metrics

The Swift geospatial helpers operate on four coordinate arrays.
Inputs are in radians.
Outputs are in meters.

```swift
import NumKong

// Statue of Liberty (40.6892°N, 74.0445°W) → Big Ben (51.5007°N, 0.1246°W)
let libertyLat: [Float64] = [0.7101605100]
let libertyLon: [Float64] = [-1.2923203180]
let bigBenLat: [Float64] = [0.8988567821]
let bigBenLon: [Float64] = [-0.0021746802]

let vincenty = vincenty(aLat: libertyLat, aLon: libertyLon, bLat: bigBenLat, bLon: bigBenLon)   // ≈ [5,589,857] m
let haversine = haversine(aLat: libertyLat, aLon: libertyLon, bLat: bigBenLat, bLon: bigBenLon) // ≈ [5,543,723] m
```

The low-level `UnsafeBufferPointer` static methods on `Float64` and `Float32` remain available for zero-copy use cases.

## Runtime Capabilities and Thread Configuration

Capability detection is exposed directly for diagnostics and tests:

```swift
import NumKong

// `available` is what can actually run: detected on this CPU AND compiled into the binary.
print(Capabilities.has(Capabilities.neon), Capabilities.has(Capabilities.haswell))

// The two raw axes, when you specifically mean one of them:
let onThisCpu = Capabilities.detected
let inThisBinary = Capabilities.compiled
```

| Accessor                 | Meaning                                                      |
| :----------------------- | :----------------------------------------------------------- |
| `Capabilities.detected`  | what this CPU can execute, from CPUID or HWCAP               |
| `Capabilities.compiled`  | what this binary contains, from the ISA probes at build time |
| `Capabilities.available` | the intersection, i.e. what can actually run here            |
| `Capabilities.enabled`   | the subset dispatch is restricted to                         |

Reach for `available` unless you specifically mean one of the raw axes.
`detected` describes the machine and says nothing about whether a kernel was compiled in, so a build whose ISA probes failed still reports your CPU's full feature set while containing no SIMD kernels at all.
`Capabilities.configureThread()` enables per-thread state such as Intel AMX, and `Capabilities.enable` / `.disable` / `.restrict` narrow what dispatch may select.

You usually do not need to branch on this in application code.
The native layer still selects the best enabled kernel automatically.
