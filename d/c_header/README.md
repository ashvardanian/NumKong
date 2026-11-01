# NumKong for C and C++

NumKong's native SDK is the reference surface for the project.
The plain C ABI exposes every kernel family directly: dot products, dense distances, binary metrics, probability divergences, geospatial solvers, curved-space kernels, sparse intersections, mesh alignment, packed matrix multiplication, symmetric self-similarity, and late-interaction scoring.
The ABI is stable, versioned, and callable from any language that can load a shared library.
There is no runtime overhead: no hidden thread pool, no implicit allocation, no garbage collector interaction.
The C++ layer stays thin, typed, allocator-aware, and close enough to inline through, adding type-level result promotion and owning containers without hiding the dispatch model or the mixed-precision policy.

## Quickstart

```c
#include <numkong/numkong.h>
#include <stdio.h>

int main(void) {
    nk_f32_t a[] = {1, 2, 3};
    nk_f32_t b[] = {4, 5, 6};
    nk_f64_t dot = 0;
    nk_configure_thread(nk_capabilities());
    nk_dot_f32(a, b, 3, &dot); // widened f32 → f64 output
    printf("dot=%f\n", dot);
    return 0;
}
```

## Highlights

This is the primary SDK in the project.
It is the right layer if you want exact control over dtypes, allocators, packed buffers, dispatch, and host-side partitioning.

__Full kernel surface.__
All public operation families are reachable from native code.
__No hidden threading.__
NumKong does not own a thread pool.
__No hidden allocation.__
C APIs take caller-owned buffers, and C++ wrappers make ownership explicit.
__Mixed precision by default.__
Small storage types widen into safer accumulator and output types.
__Allocator-aware containers.__
`vector`, `tensor`, `packed_matrix`, and `packed_maxsim` accept custom allocators.
__Unaligned inputs are fine.__
Packing handles internal layout itself and does not require caller-side alignment.

## Ecosystem Comparison

| Feature                      | NumKong                                                                                                                             | [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)               | [Eigen](https://gitlab.com/libeigen/eigen)                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Operation families           | dots, distances, binary, probability, geospatial, curved, mesh, sparse, MaxSim, elementwise, reductions, cast, trig                 | dense linear algebra only                                         | dense LA, some reductions and elementwise                                               |
| Precision                    | Sub-byte to Float64 dtypes; automatic widening per scalar type; Kahan-compensated summation; 0 ULP Float32/Float64 where applicable | Float32, Float64 only; same-type in/out; no compensated summation | Float16/BFloat16 partial; no Float8 or sub-byte; manual casts; no compensated summation |
| Runtime SIMD dispatch        | per-thread at runtime across x86, ARM, RISC-V                                                                                       | load-time CPU detection; one kernel set per process               | compile-time ISA flags only                                                             |
| Packed matrix, GEMM-like     | `packed_matrix` — pack once, reuse across query batches                                                                             | internal opaque packing per GEMM call; no persistent packed form  | no equivalent packed reuse abstraction                                                  |
| Symmetric kernels, SYRK-like | skips duplicate pairs, up to 2x speedup for self-distance                                                                           | `SSYRK`/`DSYRK` for rank-k updates                                | `.selfadjointView` for rank-k updates                                                   |
| Memory model                 | Caller-owned buffers; C++ adds `tensor<T,A>` with per-container allocators                                                          | Caller-managed buffers; no container abstraction                  | Lazy expression templates avoid most temporaries; `aligned_allocator` provided          |



## Installation

With CMake `FetchContent`:

```cmake
include(FetchContent)

FetchContent_Declare(
    numkong
    GIT_REPOSITORY https://github.com/ashvardanian/NumKong.git
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(numkong)

target_link_libraries(my_target PRIVATE numkong)
```

Vendored:

```cmake
add_subdirectory(external/NumKong)
target_link_libraries(my_target PRIVATE numkong)
```

Header-only C++ usage also works for direct template wrappers.
Most applications should still build the library once and keep `NK_DYNAMIC_DISPATCH=1`.

## The C ABI

The C ABI keeps the operation family, input dtype, and output policy visible in the symbol name.
That makes widening obvious at the call site.

```c
#include <numkong/numkong.h>

nk_i8_t a[1536];
nk_i8_t b[1536];
nk_i32_t dot = 0;     // widened from int8 storage
nk_f32_t l2 = 0;      // widened from int8 storage

nk_dot_i8(a, b, 1536, &dot);
nk_euclidean_i8(a, b, 1536, &l2);
```

If you want runtime-selected kernels without naming a specific ISA, use the punned dispatch layer:

```c
nk_metric_dense_punned_t angular = 0;
nk_capability_t used = nk_cap_serial_k;
nk_find_kernel_punned(nk_kernel_angular_k, nk_f32_k,
    nk_capabilities(), (nk_kernel_punned_t *)&angular, &used);

nk_f32_t a[768], b[768], result = 0;
angular(a, b, 768, &result);
```

That is the lowest-level dynamic path.
The typed C++ wrappers usually read better unless you are building your own dispatch layer.

## The C++ Layer

The C++ wrappers add three things.
They add type-level result promotion.
They add explicit owning and non-owning containers.
They add allocator-aware packed objects for repeated matrix workloads.

```cpp
#include <numkong/numkong.hpp>

namespace nk = ashvardanian::numkong;

int main() {
    nk::f32_t a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
    nk::f64_t dot {};
    nk::dot(a, b, 3, &dot); // default result type is nk::f32_t::dot_result_t == nk::f64_t
}
```

The API is intentionally not STL-shaped.
`vector_view`, `tensor_view`, and `matrix_view` prioritize signed strides, sub-byte storage, and kernel compatibility over resizable-container ergonomics.

## Scalar Types and Promotions

The scalar wrappers in `include/numkong/types.hpp` are storage-first types.
They encode raw layout, default output types, and the kernel function pointer signatures for each family.

| Type        | Layout           | Bytes | Range           | Inf | NaN |
| ----------- | ---------------- | ----- | --------------- | --- | --- |
| `nk_f16_t`  | 1+5+10           | 2     | ±65504          | yes | yes |
| `nk_bf16_t` | 1+8+7            | 2     | ±3.4×10³⁸       | yes | yes |
| `nk_e4m3_t` | 1+4+3            | 1     | ±448            | no  | yes |
| `nk_e5m2_t` | 1+5+2            | 1     | ±57344          | yes | yes |
| `nk_e2m3_t` | 1+2+3            | 1     | ±7.5            | no  | no  |
| `nk_e3m2_t` | 1+3+2            | 1     | ±28             | no  | no  |
| `nk_u1x8_t` | 8 packed bits    | 1     | 0 or 1 per bit  | —   | —   |
| `nk_u4x2_t` | 2x4-bit unsigned | 1     | 0-15 per nibble | —   | —   |
| `nk_i4x2_t` | 2x4-bit signed   | 1     | -8-7 per nibble | —   | —   |

The layout column shows sign, exponent, and mantissa bit counts for floating-point types.
For `nk_f16_t`, 1+5+10 means one sign bit, five exponent bits, and ten mantissa bits, totaling 16 bits stored in 2 bytes.
For `nk_bf16_t`, the wider exponent field (8 bits) gives the same dynamic range as IEEE 754 single precision but with reduced mantissa precision.
The Float8 types `nk_e4m3_t` and `nk_e5m2_t` follow the OFP8 specification.
The narrower `nk_e2m3_t` and `nk_e3m2_t` types are MX-compatible micro-floats.
Sub-byte types `nk_u1x8_t`, `nk_u4x2_t`, and `nk_i4x2_t` pack multiple logical values into a single byte.

Default promotions are encoded on the type.
For example, `f32_t::dot_result_t` is wider than `f32_t`.
`i8_t::dot_result_t` is `i32_t`.
`u1x8_t::dot_result_t` is `u32_t`.

The higher-level templates use `result_type_ = typename in_type_::dot_result_t` and similar defaults.
The fast typed overloads are constrained so that overriding the result type away from the native policy can disable the specialized path and fall back to the more generic one.

When `__cpp_lib_format >= 202110L` for the C++23 `<format>` header support, all NumKong scalar types provide `std::formatter` specializations with similar format specs to the traditional `float`.
For the BFloat16 type, the output for `nk::f16_t::from_f32(3.14f)` will look like:

| Format spec | Output example       | Description                            |
| ----------- | -------------------- | -------------------------------------- |
| `{}`        | `3.140625`           | Clean float value                      |
| `{:#}`      | `3.140625 [0x4248]`  | Annotated with hex bits                |
| `{:.2f}`    | `3.14`               | Precision forwarded to float formatter |
| `{:x}`      | `4248`               | Raw hex bits                           |
| `{:#x}`     | `0x4248`             | Hex with prefix                        |
| `{:X}`      | `4248`               | Uppercase hex                          |
| `{:b}`      | `0100001001001000`   | Binary bits                            |
| `{:#b}`     | `0b0100001001001000` | Binary with prefix                     |

## Dot Products

Dot products are one of the broadest parts of the native SDK.
They include real, complex, packed-binary, mini-float, and mixed-precision forms.

```c
nk_f32c_t a[384];
nk_f32c_t b[384];
nk_f32_t out[2] = {0, 0};

nk_dot_f32c(a, b, 384, out);  // complex inner product
nk_vdot_f32c(a, b, 384, out); // conjugated variant, like numpy.vdot
```

For quantized retrieval pipelines, the storage format often matters more than the nominal math family.
The native SDK lets you keep the compact representation and still get a widened output.

## Dense Distances

The dense spatial kernels cover the SciPy-style `sqeuclidean`, `euclidean`, and `angular` family.
The important difference is that storage type and output type are not forced to match.

```c
nk_f16_t a[768];
nk_f16_t b[768];
nk_f32_t sqeuclidean = 0, euclidean = 0, angular = 0;

// `_Float16` support varies across compilers, and
// auto-vectorization targets `f32` — not `f16`.
nk_sqeuclidean_f16(a, b, 768, &sqeuclidean);
nk_euclidean_f16(a, b, 768, &euclidean);
nk_angular_f16(a, b, 768, &angular);
```

For `i8`, `u8`, `i4`, `u4`, and `u1`, the widening is even more important.
The output type is chosen to avoid the obvious overflow trap of same-width accumulation.

## Set Similarity

Packed-binary metrics operate on packed words, not on byte-wise booleans.
That is why `u1x8_t` exists as a storage type instead of pretending that `bool[8]` is the right primitive.

```c
nk_u1x8_t a[128], b[128];
nk_u32_t hamming = 0;
nk_f32_t jaccard = 0;
nk_hamming_u1(a, b, 128 * 8, &hamming);
nk_jaccard_u1(a, b, 128 * 8, &jaccard);
```

Integer set Jaccard works on sorted arrays of integer identifiers.

```c
nk_u32_t set_a[] = {1, 3, 5, 7, 9}, set_b[] = {3, 5, 8, 9, 10};
nk_f32_t jaccard_sets = 0;
nk_jaccard_u32(set_a, set_b, 5, &jaccard_sets); // |A ∩ B| / |A ∪ B|
assert(jaccard_sets > 0.0f && jaccard_sets < 1.0f && "|A ∩ B| / |A ∪ B| should be in (0, 1)");
```

## Probability Metrics

Probability kernels target divergences directly instead of making you rebuild them from scalar loops.

```c
nk_f32_t p[] = {0.2f, 0.3f, 0.5f}, q[] = {0.1f, 0.3f, 0.6f};
nk_f64_t kl_forward = 0, kl_reverse = 0, js_forward = 0, js_reverse = 0;

nk_kld_f32(p, q, 3, &kl_forward);
nk_kld_f32(q, p, 3, &kl_reverse);
assert(kl_forward != kl_reverse && "KLD is asymmetric");

nk_jsd_f32(p, q, 3, &js_forward);
nk_jsd_f32(q, p, 3, &js_reverse);
assert(js_forward == js_reverse && "JSD is symmetric");
```

These paths are useful once you move below `f64`.
Naive implementations are usually dominated by repeated scalar transcendental calls and weak accumulation policy.

## Geospatial Metrics

The native SDK exposes both the fast spherical approximation and the more accurate ellipsoidal solver.
Inputs are in radians.
Outputs are in meters.

```c
// Statue of Liberty (40.6892°N, 74.0445°W) → Big Ben (51.5007°N, 0.1246°W)
nk_f64_t liberty_lat[] = {0.7101605100}, liberty_lon[] = {-1.2923203180};
nk_f64_t big_ben_lat[] = {0.8988567821}, big_ben_lon[] = {-0.0021746802};

nk_f64_t distance[1];
nk_vincenty_f64(liberty_lat, liberty_lon, big_ben_lat, big_ben_lon, 1, distance);  // ≈ 5,589,857 m (ellipsoidal, baseline)
nk_haversine_f64(liberty_lat, liberty_lon, big_ben_lat, big_ben_lon, 1, distance); // ≈ 5,543,723 m (spherical, ~46 km less)

// Vincenty in f32 — drifts ~2 m from f64
nk_f32_t liberty_lat32[] = {0.7101605100f}, liberty_lon32[] = {-1.2923203180f};
nk_f32_t big_ben_lat32[] = {0.8988567821f}, big_ben_lon32[] = {-0.0021746802f};
nk_f32_t distance_f32[1];
nk_vincenty_f32(liberty_lat32, liberty_lon32, big_ben_lat32, big_ben_lon32, 1, distance_f32); // ≈ 5,589,859 m (+2 m drift)
```

## Curved Metrics

Curved-space kernels are separate from the flat Euclidean family because their dataflow is different.
They combine vectors with an extra metric tensor or covariance inverse.

```c
// Complex bilinear form: aᴴ M b
nk_f32c_t a[32], b[32], metric[32 * 32];
nk_f64c_t result = {0, 0};
nk_bilinear_f32c(a, b, metric, 32, &result);

// Real Mahalanobis distance: √((a−b)ᵀ M⁻¹ (a−b))
nk_f32_t x[64], y[64], inv_cov[64 * 64];
nk_f64_t distance = 0;
nk_mahalanobis_f32(x, y, inv_cov, 64, &distance);
```

## Tensors, Views, and Memory Layout

The native containers are where most integration mistakes happen.
They need to be documented explicitly.

- `vector<T, A>` owns storage and defaults to `aligned_allocator<T, 64>`.
- `vector_view<T>` is a const strided non-owning view.
- `vector_span<T>` is a mutable strided non-owning view.
- `tensor<T, A, R>` owns rank-`R` storage and also defaults to aligned allocation.
- `tensor_view<T>` and `tensor_span<T>` are the view forms.
- `matrix`, `matrix_view`, and `matrix_span` are rank-2 aliases.

The important layout rules are:

- Signed strides are supported by the view types.
- Reversed and sliced views are valid for many elementwise and reduction kernels.
- `reshape` and `flatten` require contiguous layout.
- Matrix-style kernels care about _row contiguity_, not just total tensor contiguity.
- Negative strides are conceptually valid views, but matrix packing and packed matmul workflows are not written around them.

Memory ownership is explicit.
`vector` and `tensor` deallocate through their allocator.
`vector_view`, `tensor_view`, `matrix_view`, and spans never own memory.
And heterogenous index types for `operator[]` enable more interesting access patterns:


```cpp
#include <numkong/numkong.hpp>

namespace nk = ashvardanian::numkong;
using nk::slice, nk::all, nk::f32_t, nk::tensor, nk::tensor_view;

auto t = tensor<f32_t>::try_from({
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
});

f32_t scalar_at_2d_coordinate = t[1, -1];
f32_t scalar_at_global_offset = t[4];
assert(scalar_at_2d_coordinate == scalar_at_global_offset && "same value");

tensor_view<f32_t> scalar_as_tensor = t[1, 1, slice]; 
tensor_view<f32_t> second_row = t[1, slice];
tensor_view<f32_t> second_column = t[all, 1, slice];
assert(second_row[1] == second_column[1] && "same value");
```

You can also use a more traditional syntax with member functions, also leveraging built-in functionality for hardware-accelerated strided reductions and elementwise operations along any axis combination.
Similar to NumPy, but statically typed:

```cpp
auto first_column = t[all, 1, slice];             // strided column view → {2, 5, 8}
auto minimum_index = nk::argmin(first_column);    // index of the minimum in the second column
```

The view types are conceptually close to `std::mdspan` from C++23.
The main differences are sub-byte element support, signed strides, and the kernel dispatch integration that `std::mdspan` does not provide.
If your codebase already uses `std::mdspan`, converting at the NumKong call boundary is straightforward:

```cpp
#include <numkong/numkong.hpp>
#include <mdspan>

namespace nk = ashvardanian::numkong;

// Existing std::mdspan from your codebase
float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
auto md = std::mdspan<float, std::extents<std::size_t, 3, 3>>(data);

// Wrap into a NumKong matrix_view — data pointer, extents, and strides map directly
auto view = nk::matrix_view<nk::f32_t>(
    reinterpret_cast<nk::f32_t const *>(md.data_handle()),
    md.extent(0), md.extent(1));

// Now use any NumKong kernel on it
nk::f64_t dot {};
nk::dot(view.row(0), view.row(1), md.extent(1), &dot);
```

## Iterators and Enumeration

NumKong containers expose random-access iterators for element and row traversal.

- __`dim_iterator`__ — random-access iterator over element values, used by `vector`, `vector_view`, and `vector_span`.
  Supports all standard iterator operations plus `index()` to retrieve the current position.
- __`axis_iterator`__ — random-access iterator over sub-views (rows), used by `tensor_view` and `tensor_span`.
  Also exposes `index()`.
- __`enumerate()`__ — free function returning a lightweight view that yields `{index, value}` pairs from any container with `begin()`/`end()`/`size()`.

```cpp
#include <numkong/numkong.hpp>

namespace nk = ashvardanian::numkong;

nk::vector<nk::f16_t> v(128);
for (auto [i, val] : nk::enumerate(v))
    std::printf("[%zu] = %f\n", i, val.to_f32());

// index() on raw iterators
for (auto it = v.begin(); it != v.end(); ++it)
    std::printf("[%zu] = %f\n", it.index(), (*it).to_f32());
```

Since `tensor.hpp` includes `vector.hpp`, `enumerate()` works on tensor row views too.

Tensors also support range-for over all logical scalar elements, yielding `(position, value)` pairs.
For sub-byte types each dimension is a logical scalar. Use `.dims()` to iterate values without positions.

```cpp
for (auto [pos, val] : matrix)          { /* pos is std::array<size_t, R> */ }
for (auto [pos, ref] : matrix.span())   { ref = nk::f32_t{1}; }
for (auto val : matrix.dims())          { /* scalar only, no position */ }
```

## Packed Matrix Kernels for GEMM-Like Workloads

This is a separate native subsystem from the raw vector kernels.
It is the right tool when the right-hand side is reused many times.

```cpp
#include <numkong/numkong.hpp>

namespace nk = ashvardanian::numkong;

auto a = nk::tensor<nk::f32_t>::try_full({2, 4}, nk::f32_t {1});
auto b = nk::tensor<nk::f32_t>::try_full({3, 4}, nk::f32_t {2});
auto packed = nk::packed_matrix<nk::f32_t>::try_pack(b.as_matrix_view());

// Dot products, angular distances, and Euclidean distances all reuse the same packed B
auto dots = nk::try_dots_packed(a.as_matrix_view(), packed);
auto angulars = nk::try_angulars_packed(a.as_matrix_view(), packed);
auto euclideans = nk::try_euclideans_packed(a.as_matrix_view(), packed);
```

This is GEMM-like in the workload shape, not in the strict BLAS API.
The useful economics are:

- one-time packing of `B`
- one-time type preconversion where needed
- depth padding handled internally
- per-column norm reuse for `angulars_packed` and `euclideans_packed`
- repeated reuse of the same packed RHS across many `A` batches

Caller-side alignment is not required.
Owned `packed_matrix` storage uses its allocator.
The C ABI also exposes `nk_dots_packed_size_*` so you can `malloc` the exact external buffer yourself.

## Symmetric Kernels for SYRK-Like Workloads

The symmetric kernels solve a different problem.
They compute self-similarity or self-distance without paying for both triangles independently.

```cpp
auto vectors = nk::tensor<nk::f32_t>::try_full({100, 768}, nk::f32_t {1});
auto gram = nk::try_dots_symmetric(vectors.as_matrix_view());
auto angular_dists = nk::try_angulars_symmetric(vectors.as_matrix_view());
auto euclidean_dists = nk::try_euclideans_symmetric(vectors.as_matrix_view());
```

This is SYRK-like in the sense that the output is square and symmetric.
The important difference from packed GEMM-style work is the partitioning model.
You typically split by output row windows, not by distinct left batches against a shared packed right-hand side.

The arithmetic advantage is straightforward.
The symmetric kernels avoid recomputing both `(i, j)` and `(j, i)` pairs.
That cuts the pair count almost in half before any micro-kernel details matter.

## Sparse Operations and Intersections

Sparse helpers cover sorted-index intersection and weighted sparse dot products.

```c
nk_u32_t a_idx[] = {1, 3, 5, 7}, b_idx[] = {3, 4, 5, 8};
nk_u32_t intersection[4];
nk_size_t count = 0;
nk_sparse_intersect_u32(a_idx, b_idx, 4, 4, intersection, &count);
assert(count == 2 && "indices 3 and 5");

nk_f32_t a_weights[] = {1.0f, 2.0f, 3.0f, 4.0f};
nk_f32_t b_weights[] = {5.0f, 6.0f, 7.0f, 8.0f};
nk_f64_t result = 0;
nk_sparse_dot_u32f32(a_idx, b_idx, a_weights, b_weights, 4, 4, &result);
assert(result > 0 && "weighted dot over shared indices");
```

This family deserves explicit mention because it is not just sparse dot.
Set intersection itself is often the workload.

## Geometric Mesh Alignment

Mesh alignment returns structured outputs, not just one scalar.
The native API covers `rmsd`, `kabsch`, and `umeyama`.

```c
// Three 3D points, target is source scaled by 2x
nk_f32_t source[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
nk_f32_t target[] = {0, 0, 0, 2, 0, 0, 0, 2, 0};
nk_f32_t a_centroid[3], b_centroid[3], rotation[9];
nk_f32_t scale = 0, rmsd = 0;

nk_umeyama_f32(source, target, 3, a_centroid, b_centroid, rotation, &scale, &rmsd);
assert(rmsd < 1e-6f && "umeyama should recover exact alignment");
assert(scale > 1.99f && scale < 2.01f && "umeyama should recover 2x scale");
```

This family is separate from curved metrics because the output is a transform, not just a distance.

## MaxSim and Late Interaction

MaxSim is the late-interaction primitive used by systems such as [ColBERT](https://arxiv.org/abs/2004.12832).
It is not generic matrix multiplication.
It packs query and document token vectors into a scoring-specific layout and computes a late-interaction score.

```cpp
auto queries = nk::tensor<nk::bf16_t>::try_full({32, 128}, nk::bf16_t::one());
auto docs = nk::tensor<nk::bf16_t>::try_full({192, 128}, nk::bf16_t::one());

auto q = nk::packed_maxsim<nk::bf16_t>::try_pack(queries.as_matrix_view());
auto d = nk::packed_maxsim<nk::bf16_t>::try_pack(docs.as_matrix_view());
auto score = nk::maxsim(q, d);
```

`packed_maxsim` is allocator-aware in the same way as `packed_matrix`.
Its footprint is exposed through `size_bytes()`.

## Runtime Dispatch and Capabilities

Dynamic dispatch is the default recommendation for shipping one binary across many CPU generations.
`nk_configure_thread` configures rounding behavior and enables CPU-specific acceleration features such as Intel AMX.
It must be called once per thread before any kernel invocation and returns 1 on success, 0 on failure.

```c
nk_capability_t caps = nk_capabilities();
nk_configure_thread(caps);
if (caps & nk_cap_sapphireamx_k) { /* AMX available */ }
```

For exact register-level details, see `capabilities.h`.
The C++ wrappers can also call directly into named backends if you want to pin a path for testing or benchmarking.

## Parallelism and ForkUnion

NumKong does not manage its own threads.
That is deliberate.
The library is designed to sit inside a larger scheduler.

GEMM-like packed work is usually partitioned across row ranges of `A` against one shared packed `B`:

```cpp
using nk::range, nk::all, nk::slice;
fork_union.parallel_for(0, worker_count, [&](std::size_t t) {
    auto start = t * rows_per_worker;
    auto stop = std::min(start + rows_per_worker, total_rows);
    auto a_slice = a[range(start, stop), all, slice].as_matrix_view();
    auto c_slice = c[range(start, stop), all, slice].as_matrix_span();
    nk::dots_packed<value_type_>(a_slice, packed, c_slice);
});
```

SYRK-like symmetric work is partitioned by output row windows on one matrix:

```cpp
fork_union.parallel_for(0, worker_count, [&](std::size_t t) {
    auto start = t * rows_per_worker;
    auto count = std::min(rows_per_worker, total_rows - start);
    nk::dots_symmetric<value_type_>(vectors.as_matrix_view(), gram.as_matrix_span(), start, count);
    nk::angulars_symmetric<value_type_>(vectors.as_matrix_view(), angular_dists.as_matrix_span(), start, count);
});
```

We recommend [ForkUnion](https://github.com/ashvardanian/ForkUnion) for that host-side orchestration.
OpenMP is still a reasonable fit if the rest of your application already uses it.
Manual thread pools and task systems also work well because the kernels have explicit row-range interfaces.

The C++26 Executors TS (`std::execution`) is a natural fit here.
NumKong kernels take explicit row-range parameters and do not own threads, so they compose directly with `std::execution::bulk` or any sender/receiver scheduler.
When executors ship in your toolchain, replacing the `parallel_for` lambda above with a `bulk` sender is a one-line change.

## Integration Notes

- The C ABI is the easiest place to integrate with foreign runtimes and custom allocators.
- The C++ layer is the easiest place to express typed packed workflows, tensor slicing, and allocator-aware ownership.
- `aligned_allocator` defaults to 64-byte alignment for owned containers, but unaligned caller inputs are still valid for the kernels that accept raw pointers or views.
- If you override result types away from the scalar defaults, document that choice carefully because it can change both performance and numerical policy.

## CMake Configuration

The main user-facing CMake options are:

- `NK_BUILD_SHARED` builds a shared library, ON by default for standalone builds and OFF when included as a subdirectory.
- `NK_BUILD_TEST` and `NK_BUILD_BENCH` enable precision tests and benchmarks respectively, both OFF by default.
- `NK_DYNAMIC_DISPATCH=1` compiles all backends into one binary and selects at runtime via `nk_capabilities()`, recommended for shipping one binary across CPU generations.
- `NK_COMPARE_TO_BLAS` and `NK_COMPARE_TO_MKL` link benchmarks against a system BLAS or Intel MKL, each accepting `AUTO`, `ON`, or `OFF` with `AUTO` as the default.

The build enforces C99 for the C layer and C++23 for the C++ layer.

```sh
cmake -B build -D CMAKE_BUILD_TYPE=Release -D NK_BUILD_TEST=ON
cmake -B build -D NK_DYNAMIC_DISPATCH=1 -D NK_BUILD_BENCH=ON -D NK_COMPARE_TO_MKL=ON
```

## Cross-Compilation

Toolchain files for cross-compilation live in `cmake/`:

- `cmake/toolchain-aarch64-gnu.cmake` for ARM64 Linux with the GNU toolchain.
- `cmake/toolchain-riscv64-gnu.cmake` for RISC-V 64 Linux with the GNU toolchain.
- `cmake/toolchain-android-arm64.cmake` for Android ARM64 via the NDK.
- `cmake/toolchain-x86_64-llvm.cmake` and `cmake/toolchain-riscv64-llvm.cmake` for Clang/LLD builds.
- `cmake/toolchain-wasm.cmake`, `toolchain-wasm64.cmake`, and `toolchain-wasi.cmake` for WebAssembly targets.

```sh
cmake -B build -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-gnu.cmake
```

## Threading Model

NumKong does not use OpenMP and does not create a hidden thread pool.
Standard pthreads are linked via CMake's `Threads` package.
Parallelism is host-controlled: partition work across row ranges and dispatch through ForkUnion, `std::thread`, or any external scheduler.

## Addressing External Memory

Every kernel takes plain pointers, so any CPU-accessible memory works: mmap, pinned buffers, CUDA unified memory, custom arenas.
C++ views wrap any pointer without ownership.
Owning containers accept any C++ Allocator.

```cpp
template <typename T>
struct cuda_allocator {
    using value_type = T;
    T *allocate(std::size_t n) { T *p;
        cudaMallocManaged(&p, n * sizeof(T), cudaMemAttachGlobal); 
        return p; }
    void deallocate(T *p, std::size_t) noexcept { cudaFree(p); }
};

nk_dot_f32(cuda_managed_ptr, cuda_managed_ptr, 1024, &dot);         // C ABI, any pointer
auto view = nk::tensor_view<nk::f32_t>(mmap_ptr, rows, cols);       // non-owning view
auto v = nk::vector<float, cuda_allocator<float>>::try_zeros(1024); // allocator-aware owning
```
