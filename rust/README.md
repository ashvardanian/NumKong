# NumKong for Rust

NumKong's Rust crate keeps most of the native kernel surface while expressing it in Rust-native terms.
Rust is a natural fit for NumKong when you want static typing, explicit ownership, and strong container APIs without giving up mixed precision.
Traits cover scalar metric families.
`Tensor`, `Vector`, and packed matrix types cover the higher-level workflows.
Custom allocators, low-precision storage wrappers, and explicit row-contiguity checks stay visible instead of being hidden behind a dynamic runtime.
The crate makes the storage policy and result promotion visible.
That matters for fp16, bf16, fp8, packed bits, and strided reductions.

## Quickstart

```rust
use numkong::{configure_thread, Dot};

fn main() {
    configure_thread();
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let dot = f32::dot(&a, &b).unwrap();
    println!("dot={dot}");
}
```

## Highlights

This is the most fully featured high-level SDK after Python.
It is a good fit if you want most of the native breadth without dropping into a manual FFI layer.

__Trait-first scalar API.__
`Type::operation(&a, &b)` stays compact and predictable.
__Allocator-aware tensors.__
`Tensor`, `PackedMatrix`, and `MaxSimPackedMatrix` can use custom allocators.
__Storage-first low precision.__
`f16`, `bf16`, fp8, fp6, and packed integer wrappers are first-class types.
__Matrix kernels with explicit contracts.__
Packed and symmetric kernels validate shapes and row contiguity.
__No hidden thread pool.__
Parallel helpers remain host-controlled.
__Fork Union support.__
The `parallel` feature is the intended native orchestration layer.

## Ecosystem Comparison

| Feature                      | NumKong                                                                                                             | [nalgebra][nalgebra]                                 | [ndarray][ndarray]                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| Operation families           | dots, distances, binary, probability, geospatial, curved, mesh, sparse, MaxSim, elementwise, reductions, cast, trig | linear algebra, decompositions                       | general n-dimensional arithmetic                     |
| Precision                    | BFloat16 through sub-byte; automatic widening; Kahan summation; 0 ULP in Float32/Float64                            | Float32/Float64 only; no widening; standard accuracy | Float32/Float64 only; no widening; standard accuracy |
| Runtime SIMD dispatch        | auto-selects best ISA per-thread at runtime across x86, ARM, RISC-V                                                 | none                                                 | none                                                 |
| Packed matrix, GEMM-like     | pack once, reuse across query batches                                                                               | standard matmul; no persistent packing               | `dot` for matmul; no persistent packing              |
| Symmetric kernels, SYRK-like | skip duplicate pairs, up to 2x speedup for self-distance                                                            | no duplicate-pair skipping                           | no duplicate-pair skipping                           |
| Memory model                 | Caller-owned; `Tensor`/`PackedMatrix` support custom allocators                                                     | Heap-allocated matrices; custom storage trait        | Heap-allocated; no custom allocator support          |
| Host-side parallelism        | row-range partitioning via reusable `ThreadPool`; no hidden threads                                                 | Rayon-based parallelism possible                     | Rayon-based parallelism possible                     |

[nalgebra]: https://github.com/dimforge/nalgebra
[ndarray]: https://github.com/rust-ndarray/ndarray

NumKong validates `f16` and `bf16` interop against the `half` crate in its own test suite.
That lets you move between ecosystem-standard half types and NumKong's kernel-facing wrappers without ambiguity.

## Installation

Minimal:

```toml
[dependencies]
numkong = "7"
```

With host-side parallel helpers:

```toml
[dependencies]
numkong = { version = "7", features = ["parallel", "std"] }
```

## Compilation and Backend Selection

The crate uses the `cc` build system to compile the C backend with `NK_DYNAMIC_DISPATCH=1` automatically.
All supported backends for the target architecture are compiled into a single binary and selected at runtime.

The two Cargo features are `std`, which enables standard library support, and `parallel`, which adds host-side orchestration via ForkUnion and implies `std`.

Backend selection follows the target architecture.
ARM gets NEON, SVE, and SME, with SME available on Linux, FreeBSD, and macOS.
x86-64 gets Haswell (AVX2), Skylake/Icelake/Sapphire Rapids AVX-512 variants, and AMX on Linux and Windows only.
RISC-V gets RVV backends on Linux and FreeBSD.
WASM gets relaxed v128.

Individual backends can be disabled through environment variables.
Any `NK_TARGET_*` variable set to `0` or `false` disables that backend.
Backends not explicitly disabled are enabled by default for the target platform.

```sh
NK_TARGET_NEON=0 cargo build
NK_TARGET_SVE=0 NK_TARGET_SME=0 cargo build
```

If a backend fails to compile, the build system automatically disables it and retries with the remaining backends.
A warning is emitted for each disabled backend.

## Dynamic Dispatch and Capabilities

`configure_thread` enables CPU-specific acceleration features such as Intel AMX.
It must be called once per thread before using AMX operations.

```rust
use numkong::{available, configure_thread, cap};

let caps = available();
configure_thread();

if caps & cap::SAPPHIREAMX != 0 {
    println!("AMX available");
}
```

Call `configure_thread` at the start of every thread that will use AMX operations.
In a thread-pool setting, each worker thread needs its own call.
The function is idempotent and cheap to call more than once on the same thread.

## Core Traits

The crate root re-exports the main metric families:

- `Dot`, `VDot`, `Angular`, `Euclidean`
- `Hamming`, `Jaccard`
- `KullbackLeibler`, `JensenShannon`
- `Haversine`, `Vincenty`
- `Bilinear`, `Mahalanobis`
- `ReduceMoments`, `ReduceMinMax`
- `EachScale`, `EachSum`, `EachBlend`, `EachFMA`

The standard call shape is:

```rust
use numkong::{Dot, JensenShannon, Jaccard, u1x8};

let a = [1.0_f32, 2.0, 3.0];
let b = [4.0_f32, 5.0, 6.0];
let dot = f32::dot(&a, &b).unwrap();

let bits_a = [u1x8(0b11110000), u1x8(0b00001111)];
let bits_b = [u1x8(0b11110000), u1x8(0b11110000)];
let jaccard = u1x8::jaccard(&bits_a, &bits_b).unwrap();

let p = [0.2_f32, 0.3, 0.5];
let q = [0.1_f32, 0.3, 0.6];
let jsd = f32::jensenshannon(&p, &q).unwrap();

println!("{dot} {jaccard} {jsd}");
```

## Dot Products

Dot products span real, complex, quantized, and packed storage types.

```rust
use numkong::{Dot, VDot, f32c};

let a = [f32c { re: 1.0, im: 2.0 }, f32c { re: 3.0, im: 4.0 }];
let b = [f32c { re: 5.0, im: 6.0 }, f32c { re: 7.0, im: 8.0 }];

let dot = f32c::dot(&a, &b).unwrap();
let vdot = f32c::vdot(&a, &b).unwrap(); // like numpy.vdot, conjugated

println!("{dot:?} {vdot:?}");
```

## Dense Distances

The dense spatial family covers `sqeuclidean`, `euclidean`, and `angular`.
The main value over naive loops is the combination of SIMD and safer accumulation policy.

```rust
use numkong::Euclidean;

let a = [1_i8, 2, 3, 4];
let b = [4_i8, 3, 2, 1];

let distance = i8::euclidean(&a, &b).unwrap(); // widened output, not int8
println!("{distance}");
```

## Scalar Types and Promotions

The scalar wrappers are storage-first types.
They are not decorative aliases over `f32`.

| Type   | Layout        | Bytes | Range        | Inf | NaN |
| ------ | ------------- | ----- | ------------ | --- | --- |
| `f16`  | 1+5+10        | 2     | ±65504       | yes | yes |
| `bf16` | 1+8+7         | 2     | ±3.4×10³⁸    | yes | yes |
| `e4m3` | 1+4+3         | 1     | ±448         | no  | yes |
| `e5m2` | 1+5+2         | 1     | ±57344       | yes | yes |
| `e2m3` | 1+2+3 (6 bit) | 1     | ±7.5         | no  | no  |
| `e3m2` | 1+3+2 (6 bit) | 1     | ±28          | no  | no  |
| `u1x8` | 8 packed bits | 1     | 0–1 per bit  | —   | —   |
| `u4x2` | 2×4-bit uint  | 1     | 0–15 per nib | —   | —   |
| `i4x2` | 2×4-bit int   | 1     | −8–7 per nib | —   | —   |

The trait hierarchy documents intent:

- `StorageElement` — raw storable element type.
- `NumberLike` — adds numeric conversion and ordering.
- `FloatConvertible` — adds unpacking and float-domain conversion.

The output type is intentionally wider than the storage type for many operations.
For example, `i8::dot` returns `i32`.
`f32::dot` returns a wider accumulator type.
Moments reductions widen even more aggressively.

## Set Similarity

Packed-binary metrics work on packed words instead of boolean slices.
That is the right model once the workload is "semantic hash" rather than "array of booleans".

```rust
use numkong::{Hamming, Jaccard, u1x8};

let a = [u1x8(0b10101010), u1x8(0b11110000)];
let b = [u1x8(0b10101110), u1x8(0b11000000)];
let hamming = u1x8::hamming(&a, &b).unwrap();
let jaccard = u1x8::jaccard(&a, &b).unwrap();
```

Integer set Jaccard works on sorted arrays of integer identifiers.

```rust
let set_a = [1_u32, 3, 5, 7, 9];
let set_b = [3_u32, 5, 8, 9, 10];
let jaccard_sets = u32::jaccard(&set_a, &set_b).unwrap();
assert!(jaccard_sets > 0.0 && jaccard_sets < 1.0); // |A ∩ B| / |A ∪ B|
```

## Probability Metrics

```rust
use numkong::{JensenShannon, KullbackLeibler};

let p = [0.2_f32, 0.3, 0.5];
let q = [0.1_f32, 0.3, 0.6];
let kl_forward = f32::kullbackleibler(&p, &q).unwrap();
let kl_reverse = f32::kullbackleibler(&q, &p).unwrap();
assert!(kl_forward != kl_reverse); // KLD is asymmetric

let js_forward = f32::jensenshannon(&p, &q).unwrap();
let js_reverse = f32::jensenshannon(&q, &p).unwrap();
assert!((js_forward - js_reverse).abs() < 1e-6, "JSD is symmetric");
```

## Geospatial Metrics

Inputs are latitudes and longitudes in radians.
Outputs are meters.

```rust
use numkong::{Haversine, Vincenty};

// Statue of Liberty (40.6892°N, 74.0445°W) → Big Ben (51.5007°N, 0.1246°W)
let liberty_lat = [0.7101605100_f64];
let liberty_lon = [-1.2923203180_f64];
let big_ben_lat = [0.8988567821_f64];
let big_ben_lon = [-0.0021746802_f64];
let mut distance = [0.0_f64; 1];
f64::vincenty(&liberty_lat, &liberty_lon, &big_ben_lat, &big_ben_lon, &mut distance).unwrap();  // ≈ 5,589,857 m (ellipsoidal, baseline)
f64::haversine(&liberty_lat, &liberty_lon, &big_ben_lat, &big_ben_lon, &mut distance).unwrap(); // ≈ 5,543,723 m (spherical, ~46 km less)

// Vincenty in f32 — drifts ~2 m from f64
let liberty_lat32 = [0.7101605100_f32];
let liberty_lon32 = [-1.2923203180_f32];
let big_ben_lat32 = [0.8988567821_f32];
let big_ben_lon32 = [-0.0021746802_f32];
let mut distance_f32 = [0.0_f32; 1];
f32::vincenty(&liberty_lat32, &liberty_lon32, &big_ben_lat32, &big_ben_lon32, &mut distance_f32).unwrap(); // ≈ 5,589,859 m (+2 m drift)
```

## Curved Metrics

Curved-space kernels combine vectors with an extra metric tensor or covariance inverse.

```rust
use numkong::{Bilinear, Mahalanobis, f32c};

// Complex bilinear form: aᴴ M b
let a = [f32c { re: 1.0, im: 0.0 }; 16];
let b = [f32c { re: 0.0, im: 1.0 }; 16];
let metric = [f32c { re: 1.0, im: 0.0 }; 16 * 16];
let bilinear = f32c::bilinear(&a, &b, &metric).unwrap();

// Real Mahalanobis distance: √((a−b)ᵀ M⁻¹ (a−b))
let x = [1.0_f32; 32];
let y = [2.0_f32; 32];
let mut inv_cov = vec![0.0_f32; 32 * 32];
for i in 0..32 { inv_cov[i * 32 + i] = 1.0; } // identity matrix
let distance = f32::mahalanobis(&x, &y, &inv_cov).unwrap();
```

## Vectors, Tensors, Views, and Spans

The container model is unusual enough that it needs direct documentation.

- `Vector<T>` owns one-dimensional storage.
- `VectorView<'a, T>` is an immutable borrowed view.
- `VectorSpan<'a, T>` is a mutable borrowed view.
- `Tensor<T, A, MAX_RANK>` owns N-dimensional storage and can use a custom allocator.
- `TensorView` and `TensorSpan` are the borrowed forms.
- `Matrix<T>` is a rank-2 alias over `Tensor<T, _, 2>`.

The allocator story is explicit.
`Tensor` and `PackedMatrix` default to `Global`.
The underlying layout uses `SIMD_ALIGNMENT == 64` for owned allocations.
That does _not_ mean callers must align their source buffers manually.
It means owned outputs and packed payloads are allocated in a SIMD-friendly way when the crate owns them.

```rust
use numkong::{RangeStep, SliceRange, Tensor};

let t = Tensor::<f32>::try_from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 3]).unwrap();

let col = t.slice((.., 1_usize)).unwrap();                  // t[:, 1]  — column 1
let rows = t.slice((0..2_usize, ..)).unwrap();              // t[0:2, :] — first two rows
let tail = t.slice((-2_isize.., ..)).unwrap();              // t[-2:, :] — last two rows
let neg = t.slice((.., -2..-1_isize)).unwrap();             // t[:, -2:-1]
let step = t.slice((.., RangeStep::new(0, 3, 2))).unwrap(); // t[:, ::2]

// Explicit &[SliceRange] syntax also works
let col = t.slice(&[SliceRange::full(), SliceRange::index(1)]).unwrap();
```

Tuple elements implement `SliceArg` — each monomorphized with zero runtime dispatch:

| Rust syntax                     | Meaning                                |
| ------------------------------- | -------------------------------------- |
| `..`                            | all                                    |
| `0_usize` / `-1_isize`          | single index (negative wraps from end) |
| `1..4_usize` / `-3..-1_isize`   | half-open range                        |
| `..3_usize` / `..-1_isize`      | from start                             |
| `1_usize..` / `-2_isize..`      | to end                                 |
| `0..=2_usize` / `-3..=-1_isize` | inclusive range                        |
| `RangeStep::new(0, 6, 2)`       | stepped (no Rust literal)              |

Integer literals default to `i32` — use `_usize` / `_isize` suffixes.
Negative `isize` values wrap from the dimension end, like Python.

Iteration works at the logical-dimension level.
For sub-byte types like `i4x2` (2 nibbles per byte), iterating a 3-element vector yields 6 dimensions.
Immutable iterators (`iter()`) yield `DimRef<T>`, which dereferences to `T::DimScalar`.
Mutable iterators (`iter_mut()`) yield `DimMut<T>`, which writes back on drop — the only way to mutate individual nibbles or bits.

```rust
use numkong::{Vector, i4x2};

let mut nibbles = Vector::<i4x2>::try_zeros(4).unwrap();
for (i, mut dim) in nibbles.iter_mut().enumerate() {
    *dim = i as i8;
}
assert_eq!(nibbles.try_get(0_usize).unwrap(), 0);
assert_eq!(nibbles.try_get(3_usize).unwrap(), 3);
```

Vectors and tensors can be converted between each other without copying:

```rust
use numkong::{Vector, Tensor};

let v = Vector::<f32>::try_from_scalars(&[1.0, 2.0, 3.0]).unwrap();
let t: Tensor<f32, _, 8> = v.try_into_tensor().unwrap();
assert_eq!(t.shape(), &[3]);
let v2 = t.try_into_vector().unwrap();
assert_eq!(v2.dims(), 3);
```

The main layout rules are:

- General slicing and transposition are supported by views.
- Elementwise and many reduction kernels accept strided views.
- Matrix-style kernels require rank-2 inputs with contiguous rows.
- A tensor can be non-contiguous overall and still have contiguous rows.
- Some reductions have SIMD kernels for strided lanes.
- Some backends still fall back depending on alignment and dtype.

Sub-byte types (`i4x2`, `u4x2`, `u1x8`) use logical shapes.
A shape of `[8]` for `i4x2` means 8 nibbles (stored in 4 bytes), not 8 bytes.
The innermost dimension must be divisible by `dimensions_per_value()` (2 for nibble types, 8 for bit types).
Transpose and reshape are not supported for sub-byte types — they return `SubByteUnsupported`.

## Elementwise Operations

Elementwise kernels live on tensors and views.
They are not a promise that every arbitrary strided view gets the same SIMD path on every backend.

```rust
use numkong::Tensor;

let a = Tensor::<f32>::try_from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
let b = Tensor::<f32>::try_full(&[2, 2], 2.0).unwrap();

let blended = a.view().try_blend_tensor(&b.view(), 0.25, 0.75).unwrap();
let sines = blended.sin().unwrap();

assert_eq!(sines.shape(), &[2, 2]);
```

Compound assignment operators work in-place:

```rust
use numkong::Tensor;

let mut t = Tensor::<f32>::try_full(&[4], 1.0).unwrap();
t += 10.0;
t -= 0.5;
t *= 2.0;
```

## Trigonometry

The trigonometric kernels share the tensor and view surface.
They are useful both directly and as a sanity check that the container path is not just about matrix kernels.

```rust
use numkong::Tensor;

let a = Tensor::<f32>::try_from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2]).unwrap();
let c = a.cos().unwrap();
let s = a.sin().unwrap();

assert_eq!(c.shape(), &[2, 2]);
assert_eq!(s.shape(), &[2, 2]);
```

## Moments Reductions

Moments reductions return both sum and sum-of-squares.
That is the right building block for norms and variance-like workflows.

```rust
use numkong::{ReduceMoments, Tensor};

let narrow = Tensor::<u8>::try_full(&[1024], 255).unwrap();
let (sum, sumsq) = narrow.try_moments_all().unwrap();

assert!(sum > 255);      // a naive u8 accumulation would overflow immediately
assert!(sumsq > 255u64); // same for sum-of-squares
```

The important documentation point is not just "wider outputs exist".
It is that the API makes the widened outputs part of the type story.

## Min/Max Reductions

Min/max reductions return a `MinMaxResult` with both the value and its flat index:

```rust
use numkong::Tensor;

let t = Tensor::<f32>::try_from_slice(&[
    3.0, 0.0, 7.0,
    1.0, 2.0, 5.0,
    4.0, -1.0, 6.0,
], &[3, 3]).unwrap();

let second_column = t.slice((.., 1_usize)).unwrap();  // t[:, 1]
let idx = second_column.try_argmin_all().unwrap();

assert_eq!(idx, 2);
```


## Sparse Operations and Intersections

Sparse helpers cover both sorted-index intersection and weighted sparse dot products.

```rust
use numkong::{SparseIntersect, SparseDot};

let a_idx = [1_u32, 3, 5, 7];
let b_idx = [3_u32, 4, 5, 8];
let count = u32::sparse_intersection_size(&a_idx, &b_idx);
assert_eq!(count, 2); // indices 3 and 5

let a_weights = [1.0_f32, 2.0, 3.0, 4.0];
let b_weights = [5.0_f32, 6.0, 7.0, 8.0];
let dot = u32::sparse_dot(&a_idx, &b_idx, &a_weights, &b_weights);
assert!(dot > 0.0); // weighted dot over shared indices
```

## Packed Matrix Kernels for GEMM-Like Workloads

Packed kernels are the main "matrix throughput" path in the crate.
They are `GEMM`-like in workload shape.
They are not a thin BLAS clone.

```rust
use numkong::{PackedMatrix, Tensor};

let a = Tensor::<f32>::try_full(&[1024, 512], 1.0).unwrap();
let b = Tensor::<f32>::try_full(&[256, 512], 1.0).unwrap();

let b_packed = PackedMatrix::try_pack(&b).unwrap();
let c = a.dots_packed(&b_packed);

assert_eq!(c.shape(), &[1024, 256]);
```

The useful economics are:

- pack `B` once
- reuse it across many `A` batches
- convert or pad once during packing instead of on every multiply
- reuse precomputed norms for `angulars_packed` and `euclideans_packed`

The crate checks row contiguity because these kernels assume contiguous rows.
Caller-side source alignment is not required.
The owned packed buffer handles its own aligned allocation internally.

## Symmetric Kernels for SYRK-Like Workloads

Symmetric kernels are for self-similarity and self-distance.
They are `SYRK`-like in shape.
They avoid duplicate `(i, j)` and `(j, i)` work.

```rust
use numkong::Tensor;

let vectors = Tensor::<f32>::try_full(&[100, 768], 1.0).unwrap();
let gram = vectors.view().try_dots_symmetric().unwrap();

assert_eq!(gram.shape(), &[100, 100]);
```

This family is also where row-window partitioning becomes the natural parallel model.
That is structurally different from packed `GEMM`-style work against a shared packed RHS.

## MaxSim and ColBERT-Style Late Interaction

MaxSim is the late-interaction primitive used by systems such as [ColBERT](https://arxiv.org/abs/2004.12832).
It is not "just another matrix multiply".

```rust
use numkong::{MaxSimPackedMatrix, Tensor};

let queries = Tensor::<f32>::try_full(&[4, 16], 1.0).unwrap();
let docs = Tensor::<f32>::try_full(&[8, 16], 1.0).unwrap();

let queries_packed = queries.view().try_maxsim_pack().unwrap();
let docs_packed = docs.view().try_maxsim_pack().unwrap();
let score = queries_packed.score(&docs_packed);

assert!(score.is_finite());
```

## Geometric Mesh Alignment

Mesh alignment returns transforms, scales, and RMSD values.
That is a different API shape from the scalar metric families.

```rust
use numkong::MeshAlignment;

let source = [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
let target = [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

let result = f32::kabsch(&source, &target).unwrap();
assert!(result.rmsd < 1e-6);
assert!((result.scale - 1.0).abs() < 1e-6);

// Umeyama with known 2x scaling
let scaled = [[0.0_f32, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
let result = f32::umeyama(&source, &scaled).unwrap();
assert!(result.rmsd < 1e-6);
assert!((result.scale - 2.0).abs() < 0.01);
```

## Tolerance Comparison

Exact floating-point equality is rarely what you want after arithmetic.
`allclose()` checks every element pair with the formula:

$$
|a - b| \leq \text{atol} + \text{rtol} \cdot |b|
$$

Available on `Vector`, `VectorView`, `VectorSpan`, `Tensor`, `TensorView`, and `TensorSpan`.
For tensors, `allclose` is provided by the `AllCloseOps` trait — import it if calling on a `TensorRef` implementor.
Shape mismatch returns `false`.
The scalar helper `is_close` is re-exported at crate root.

```rust
use numkong::{is_close, Vector, Tensor};

// Scalar check
assert!(is_close(1.0, 1.0 + 1e-8, 1e-6, 0.0));

// Vector tolerance check
let a = Vector::<f32>::try_full(3, 1.0).unwrap();
let b = Vector::<f32>::try_full(3, 1.0 + 1e-7).unwrap();
assert!(a.allclose(&b, 1e-6, 0.0));

// Tensor tolerance check
let ta = Tensor::<f32>::try_full(&[2, 3], 1.0).unwrap();
let tb = Tensor::<f32>::try_full(&[2, 3], 1.0 + 1e-7).unwrap();
assert!(ta.allclose(&tb, 1e-6, 0.0));
```

## Type Casting

The `cast` function performs bulk conversion between contiguous slices.
Any pair of types that implement `CastDtype` (all `NumberLike` scalars) can be converted.

```rust
use numkong::{cast, f16, bf16};

let src: Vec<f32> = vec![1.0, 2.0, 3.0];
let mut dst: Vec<f16> = vec![f16::from(0.0_f32); 3];
cast(&src, &mut dst);
assert!((dst[0].to_f32() - 1.0).abs() < 0.01);
```

`Tensor`, `TensorView`, and `TensorSpan` expose casting via the `CastOps` trait.
`try_cast_dtype()` allocates a new tensor; `try_cast_dtype_into()` writes into a pre-allocated `TensorSpan`.
Strided and non-contiguous views are supported: the implementation scans strides from the innermost dimension outward to find the longest contiguous tail, then walks the outer dimensions and casts each contiguous block in a single kernel call.

```rust
use numkong::{Tensor, f16};

let src = Tensor::<f32>::try_full(&[4, 4], 1.0).unwrap();
let mut dst = Tensor::<f16>::try_zeros(&[4, 4]).unwrap();
src.view().try_cast_dtype_into(&mut dst.span()).unwrap();
```

## Parallelism and ForkUnion

NumKong does not own a thread pool.
The `parallel` feature adds host-side orchestration helpers via [ForkUnion](https://github.com/ashvardanian/ForkUnion), not a hidden scheduler.

```rust
use numkong::{PackedMatrix, Tensor};
use fork_union::ThreadPool;

let a = Tensor::<f32>::try_full(&[4096, 768], 1.0).unwrap();
let b = Tensor::<f32>::try_full(&[8192, 768], 1.0).unwrap();
let mut pool = ThreadPool::try_spawn(4).unwrap();

// GEMM-like: rows of A partitioned across threads, one shared packed B
let b_packed = PackedMatrix::try_pack(&b).unwrap();
let c = a.dots_packed_parallel(&b_packed, &mut pool);
assert_eq!(c.shape(), &[4096, 8192]);

// SYRK-like: row windows of one square output partitioned across threads
let gram = a.dots_symmetric_parallel(&mut pool);
assert_eq!(gram.shape(), &[4096, 4096]);
```

Rayon or a manual thread pool can still work if the rest of your application already depends on them.

## Addressing External Memory

Views wrap raw pointers without ownership, owned containers accept custom allocators, and the scalar trait API works on any `&[T]` regardless of how the memory was allocated.

`VectorView::from_raw_parts` and `TensorView::from_raw_parts` wrap device-accessible or externally allocated memory.
The mutable counterparts `VectorSpan::from_raw_parts` and `TensorSpan::from_raw_parts` work the same way with `*mut T`.

```rust
use numkong::{VectorView, TensorView};

let embeddings_ptr: *const f32 = /* from CUDA, mmap, or FFI */;
let embeddings = unsafe {
    VectorView::from_raw_parts(embeddings_ptr, 1024, std::mem::size_of::<f32>() as isize)
};

let shape = [32, 64];
let strides = [64 * 4, 4]; // row-major f32
let matrix = unsafe { TensorView::<f32>::from_raw_parts(embeddings_ptr, &shape, &strides) };
```

Owned containers accept any allocator.
A CUDA unified memory allocator looks like this:

```rust
use std::alloc::{Allocator, AllocError, Layout};
use std::ptr::NonNull;
use numkong::Vector;

struct CudaAllocator;

unsafe impl Allocator for CudaAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let raw = unsafe { cuda_malloc_managed(layout.size()) };
        let base = NonNull::new(raw).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(base, layout.size()))
    }
    unsafe fn deallocate(&self, block: NonNull<u8>, _layout: Layout) {
        cuda_free(block.as_ptr());
    }
}

let queries = Vector::<f32, CudaAllocator>::try_zeros_in(1024, CudaAllocator).unwrap();
```

The trait-based scalar API works on any `&[T]` — `Vec`, mmap, arena, or pinned buffer:

```rust
use numkong::Dot;

let weights: &[f32] = /* any contiguous slice */;
let similarity = f32::dot(weights, weights).unwrap();
```
