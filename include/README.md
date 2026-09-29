# NumKong for C and C++

NumKong's native SDK is the reference surface for the project.
The plain C ABI exposes every kernel family directly: dot products, dense distances, binary metrics, probability divergences, geospatial solvers, curved-space kernels, sparse intersections, mesh alignment, packed matrix multiplication, symmetric self-similarity, late-interaction scoring, elementwise arithmetic, reductions, type conversions, scalar math, and trigonometry.
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
    nk_capability_t capabilities = nk_cap_serial_k;
    nk_cpu_capabilities_enabled(&capabilities); // the capabilities this CPU runs and this build holds
    nk_cpu_configure_thread(capabilities);
    nk_status_t status = nk_dot_f32_best(a, b, 3, &dot, capabilities, NULL); // widened f32 → f64 output
    printf("dot=%f, %s\n", dot, nk_status_name(status));
    return status != nk_success_k;
}
```

Every operation has one such dispatch point, named with `_best`, which runs the best capability in the mask it gets; the root README describes [the dispatch model](../README.md#dispatch-points--capability-masks) once.

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

| Feature                      | NumKong                                                                                                                             | [OpenBLAS][openblas]                                              | [Eigen][eigen]                                                                          |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| Operation families           | dots, distances, binary, probability, geospatial, curved, mesh, sparse, MaxSim, elementwise, reductions, cast, trig                 | dense linear algebra only                                         | dense LA, some reductions and elementwise                                               |
| Precision                    | Sub-byte to Float64 dtypes; automatic widening per scalar type; Kahan-compensated summation; 0 ULP Float32/Float64 where applicable | Float32, Float64 only; same-type in/out; no compensated summation | Float16/BFloat16 partial; no Float8 or sub-byte; manual casts; no compensated summation |
| Runtime SIMD dispatch        | per call, by capability mask, across x86, Arm, RISC-V, and GPUs                                                                     | load-time CPU detection; one kernel set per process               | compile-time ISA flags only                                                             |
| Packed matrix, GEMM-like     | `packed_matrix` — pack once, reuse across query batches                                                                             | internal opaque packing per GEMM call; no persistent packed form  | no equivalent packed reuse abstraction                                                  |
| Symmetric kernels, SYRK-like | skips duplicate pairs, up to 2x speedup for self-distance                                                                           | `SSYRK`/`DSYRK` for rank-k updates                                | `.selfadjointView` for rank-k updates                                                   |
| Memory model                 | Caller-owned buffers; C++ adds `tensor<T,A>` with per-container allocators                                                          | Caller-managed buffers; no container abstraction                  | Lazy expression templates avoid most temporaries; `aligned_allocator` provided          |

[openblas]: https://github.com/OpenMathLib/OpenBLAS
[eigen]: https://gitlab.com/libeigen/eigen


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

target_link_libraries(my_target PRIVATE numkong::header) # or numkong::static, numkong::shared
```

Vendored:

```cmake
add_subdirectory(external/NumKong)
target_link_libraries(my_target PRIVATE numkong::header) # or numkong::static, numkong::shared
```

`numkong::static` and `numkong::shared` exist when `NUMKONG_BUILD_SHARED` is on, the default only for a top-level build.
An installed NumKong provides the same targets through `find_package(numkong CONFIG REQUIRED)`.

By default the headers declare every public function and the library defines it, each kernel once, in the unit of its capability.
`numkong::header` sets `NUMKONG_HEADER_ONLY=1` instead: every kernel compiles into your translation unit, and every dispatch point and finder returns `nk_missing_library_k`, so a header-only build calls a capability's kernel, like `nk_dot_f32_haswell`, directly.

## The C ABI

The C ABI keeps the operation family, input dtype, and output policy visible in the symbol name.
That makes widening obvious at the call site.

```c
#include <numkong/numkong.h>

nk_i8_t a[1536];
nk_i8_t b[1536];
nk_i32_t dot = 0;     // widened from int8 storage
nk_f32_t l2 = 0;      // widened from int8 storage

nk_dot_i8_best(a, b, 1536, &dot, capabilities, NULL);
nk_euclidean_i8_best(a, b, 1536, &l2, capabilities, NULL);
```

The examples below pass the `capabilities` of the Quickstart and a null stream, as every CPU call does.
Each capability kernel, like `nk_dot_i8_haswell`, is also callable directly with the same arguments short of the mask.
To resolve a kernel once and call it many times without naming a capability, library builds provide the punned finder:

```c
nk_metric_dense_punned_t angular = 0;
nk_capability_t capability = 0;
nk_find_kernel_punned(nk_kernel_angular_k, nk_f32_k, capabilities, (nk_kernel_punned_t *)&angular, &capability);

nk_f32_t a[768], b[768];
nk_f64_t result = 0; // widened f32 → f64 output
angular(a, b, 768, &result, NULL);
```

That is the lowest-level dynamic path.
The typed C++ wrappers usually read better unless you are building your own dispatch layer.

## The C++ Layer

The C++ wrappers come in three layers.
Raw-pointer overloads mirror the C ABI one to one, with explicit lengths, strides and task windows for sharded launches.
Overloads over the `nk::vector_of` and matrix concepts accept any contiguous run, like `std::vector`, `std::array`, `std::span`, a C array or an `nk::vector_view`, and return their results.
Owning types, like `nk::tensor`, `nk::packed_matrix` and `nk::packed_attention`, allocate through your allocator for repeated workloads.

```cpp
#include <vector>
#include <numkong/numkong.hpp>

namespace nk = ashvardanian::numkong;

int main() {
    std::vector<nk::f32_t> a {1, 2, 3}, b {4, 5, 6};
    auto [dot, status] = nk::dot<nk::f32_t>(a, b); // nk::f32_t::dot_result_t == nk::f64_t
    if (nk::failed(status)) return 1;

    nk::f64_t raw {};
    status = nk::dot(a.data(), b.data(), a.size(), &raw, nk_cap_serial_k); // the raw layer, pinned to serial
    if (nk::failed(status)) return 1;

    auto [matrix, matrix_status] = nk::matrix<nk::f32_t>::full({2, 3}, nk::f32_t(1));
    if (nk::failed(matrix_status)) return 1;
    auto [packed, packed_status] = nk::packed_matrix<nk::f32_t>::make(matrix.view()); // owning, reused across calls
    return nk::succeeded(packed_status) && dot == raw ? 0 : 1;
}
```

Standard containers count storage values, so an `std::vector<nk::u1x8_t>` of 4 bytes holds 32 dimensions, while NumKong's own vectors and views count dimensions.
Runs must fill whole storage values, so odd bit and nibble lengths pad with zero bits up to the next byte, and a view ending mid-value is refused.
Strided views are refused with `nk::status_t::unexpected_dimensions_k`, and `vector_view::values()` turns a contiguous view into an `std::span` or refuses the same way.
Outputs of elementwise kernels, like `nk::scale<nk::f32_t>(a, 2.0f, 1.0f, out)`, are any mutable run of the same size, and the call returns an `nk::status_t`.
Multi-part results come back as small structs: `nk::reduce_moments` returns `{sum, sumsq}`, `nk::reduce_minmax` a `minmax_result`, and `nk::kabsch` a `mesh_result`.

Every wrapper ends in the two arguments of its dispatch point, defaulted to `nk::default_capabilities()` and a null stream, and returns an `nk::status_t`.
That scoped enum mirrors `nk_status_t` value for value, converts to it with `static_cast`, and is tested with `nk::succeeded` and `nk::failed` rather than as a `bool`.
Factories, like `nk::tensor<T>::zeros`, and the allocating overloads return an `nk::expected<T>` holding the `value` and its `status`, which converts to `true` on success and unpacks with structured bindings.
A failed factory leaves its `value` empty, while a zero-volume request succeeds with an empty one.
`nk::default_capabilities()` is the `nk_cpu_capabilities_enabled` mask, and a zero mask runs the C++ reference template instead of any capability's kernel.
`nk::device_t` names one device by its kind and its runtime's ordinal, and asks the matching `nk_<kind>_*` functions for its masks:

```cpp
nk::device_t const cpu = nk::device_t::cpu();
if (auto enabled = cpu.capabilities_enabled()) (void)cpu.configure_thread(enabled.value);
auto const [devices, counted] = nk::device_t::count(nk::device_kind_t::cuda_k);
for (std::size_t ordinal = 0; ordinal != devices; ++ordinal)
    if (auto gpu = nk::device_t::make(nk::device_kind_t::cuda_k, ordinal))
        if (auto mask = gpu.value.capabilities_enabled()) { /* dispatch over `mask.value` on that GPU */ }
```

In header-only builds the dispatch points report `nk_missing_library_k`, so wrappers there take a zero mask or call a capability's kernel.

The API is intentionally not STL-shaped.
`vector_view`, `tensor_view`, and `matrix_view` prioritize signed strides, sub-byte storage, and kernel compatibility over resizable-container ergonomics.

## Scalar Types and Promotions

The scalar wrappers in `include/numkong/types.hpp` are storage-first types.
They encode raw layout, default output types, and the kernel function pointer signatures for each family.

| Type          | Layout           | Bytes |            Range |  Inf  |  NaN  |
| :------------ | :--------------- | ----: | ---------------: | :---: | :---: |
| `nk_f16_t`    | 1+5+10           |     2 |           ±65504 |  yes  |  yes  |
| `nk_bf16_t`   | 1+8+7            |     2 |        ±3.4×10³⁸ |  yes  |  yes  |
| `nk_e4m3_t`   | 1+4+3            |     1 |             ±448 |  no   |  yes  |
| `nk_e5m2_t`   | 1+5+2            |     1 |           ±57344 |  yes  |  yes  |
| `nk_e2m3_t`   | 1+2+3            |     1 |             ±7.5 |  no   |  no   |
| `nk_e3m2_t`   | 1+3+2            |     1 |              ±28 |  no   |  no   |
| `nk_e2m1x2_t` | 2 × (1+2+1)      |     1 |    ±6 per nibble |  no   |  no   |
| `nk_u1x8_t`   | 8 packed bits    |     1 |   0 or 1 per bit |  n/a  |  n/a  |
| `nk_u4x2_t`   | 2 × 4-bit        |     1 |  0…15 per nibble |  n/a  |  n/a  |
| `nk_i4x2_t`   | 2 × 4-bit signed |     1 |  −8…7 per nibble |  n/a  |  n/a  |

The layout column shows sign, exponent, and mantissa bit counts for floating-point types.
For `nk_f16_t`, 1+5+10 means one sign bit, five exponent bits, and ten mantissa bits, totaling 16 bits stored in 2 bytes.
For `nk_bf16_t`, the wider exponent field (8 bits) gives the same dynamic range as IEEE 754 single precision but with reduced mantissa precision.
The Float8 types `nk_e4m3_t` and `nk_e5m2_t` follow the OFP8 specification.
The narrower `nk_e2m3_t` and `nk_e3m2_t` types are MX-compatible micro-floats, and `nk_e2m1x2_t` packs two FP4 values of the MXFP4 and NVFP4 formats.
Sub-byte types `nk_e2m1x2_t`, `nk_u1x8_t`, `nk_u4x2_t`, and `nk_i4x2_t` pack multiple logical values into a single byte.
Element 0 sits in the high nibble or the least significant bit, and every dimension count must be a multiple of the values per byte.

Default promotions are encoded on the type.
For example, `f32_t::dot_result_t` is wider than `f32_t`.
`i8_t::dot_result_t` is `i32_t`.
`u1x8_t::dot_result_t` is `u32_t`.

The higher-level templates use `result_type_ = typename in_type_::dot_result_t` and similar defaults.
The fast typed overloads are constrained so that overriding the result type away from the native policy can disable the specialized path and fall back to the more generic one.

When `__cpp_lib_format >= 202110L` for the C++23 `<format>` header support, all NumKong scalar types provide `std::formatter` specializations with similar format specs to the traditional `float`.
For the Float16 type, the output for `nk::f16_t::from_f32(3.14f)` will look like:

| Format spec | Output example       | Description                            |
| :---------- | :------------------- | :------------------------------------- |
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
nk_f64c_t out = {0, 0};         // widened f32c → f64c output

nk_dot_f32c_best(a, b, 384, &out, capabilities, NULL);   // complex inner product
nk_vdot_f32c_best(a, b, 384, &out, capabilities, NULL);  // conjugated variant, like numpy.vdot
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
nk_sqeuclidean_f16_best(a, b, 768, &sqeuclidean, capabilities, NULL);
nk_euclidean_f16_best(a, b, 768, &euclidean, capabilities, NULL);
nk_angular_f16_best(a, b, 768, &angular, capabilities, NULL);
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
nk_hamming_u1_best(a, b, 128 * 8, &hamming, capabilities, NULL);
nk_jaccard_u1_best(a, b, 128 * 8, &jaccard, capabilities, NULL);
```

`nk_jaccard_u32_best` is a dense word-wise kernel rather than a sorted-set operation.
It walks two arrays of `n` 32-bit words position by position and returns `1 - matches / n`.

```c
nk_u32_t a[] = {1, 3, 5, 7, 9}, b[] = {1, 3, 5, 8, 10};
nk_f32_t jaccard_words = 0;
nk_jaccard_u32_best(a, b, 5, &jaccard_words, capabilities, NULL); // 1 - matches / n
assert(jaccard_words > 0.0f && jaccard_words < 1.0f && "3 of 5 words match");
```

Sorted arrays of integer identifiers are handled by the sparse kernels instead, described below.

## Probability Metrics

Probability kernels target divergences directly instead of making you rebuild them from scalar loops.

```c
nk_f32_t p[] = {0.2f, 0.3f, 0.5f}, q[] = {0.1f, 0.3f, 0.6f};
nk_f64_t kl_forward = 0, kl_reverse = 0, js_forward = 0, js_reverse = 0;

nk_kld_f32_best(p, q, 3, &kl_forward, capabilities, NULL);
nk_kld_f32_best(q, p, 3, &kl_reverse, capabilities, NULL);
assert(kl_forward != kl_reverse && "KLD is asymmetric");

nk_jsd_f32_best(p, q, 3, &js_forward, capabilities, NULL);
nk_jsd_f32_best(q, p, 3, &js_reverse, capabilities, NULL);
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
nk_vincenty_f64_best(liberty_lat, liberty_lon, big_ben_lat, big_ben_lon, 1, distance, capabilities, NULL);  // ≈ 5,589,857 m (ellipsoidal, baseline)
nk_haversine_f64_best(liberty_lat, liberty_lon, big_ben_lat, big_ben_lon, 1, distance, capabilities, NULL); // ≈ 5,543,723 m (spherical, ~46 km less)

// Vincenty in f32 — drifts ~2 m from f64
nk_f32_t liberty_lat32[] = {0.7101605100f}, liberty_lon32[] = {-1.2923203180f};
nk_f32_t big_ben_lat32[] = {0.8988567821f}, big_ben_lon32[] = {-0.0021746802f};
nk_f32_t distance_f32[1];
nk_vincenty_f32_best(liberty_lat32, liberty_lon32, big_ben_lat32, big_ben_lon32, 1, distance_f32, capabilities, NULL); // ≈ 5,589,859 m (+2 m drift)
```

## Curved Metrics

Curved-space kernels are separate from the flat Euclidean family because their dataflow is different.
They combine vectors with an extra metric tensor or covariance inverse.

```c
// Complex bilinear form: aᴴ M b
nk_f32c_t a[32], b[32], metric[32 * 32];
nk_f64c_t result = {0, 0};
nk_bilinear_f32c_best(a, b, metric, 32, &result, capabilities, NULL);

// Real Mahalanobis distance: √((a−b)ᵀ M⁻¹ (a−b))
nk_f32_t x[64], y[64], inv_cov[64 * 64];
nk_f64_t distance = 0;
nk_mahalanobis_f32_best(x, y, inv_cov, 64, &distance, capabilities, NULL);
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

auto [t, status] = tensor<f32_t>::from({
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
});
assert(nk::succeeded(status) && "allocated");

f32_t scalar_at_2d_coordinate = t[1, -1];
f32_t scalar_at_global_offset = t[5];
assert(scalar_at_2d_coordinate == scalar_at_global_offset && "same value");

tensor_view<f32_t> scalar_as_tensor = t[1, 1, slice]; 
tensor_view<f32_t> second_row = t[1, slice];
tensor_view<f32_t> second_column = t[all, 1, slice];
assert(second_row[1] == second_column[1] && "same value");
```

You can also use a more traditional syntax with member functions, also leveraging built-in functionality for hardware-accelerated strided reductions and elementwise operations along any axis combination.
Similar to NumPy, but statically typed:

```cpp
tensor_view<f32_t> second_column = t[all, 1, slice]; // strided column view → {2, 5, 8}
auto minimum_index = nk::argmin(second_column);   // index of the minimum in the second column
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
auto [dot, status] = nk::dot<nk::f32_t>(view.row(0).as_vector(), view.row(1).as_vector());
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

auto [v, v_status] = nk::vector<nk::f16_t>::zeros(128);
if (nk::failed(v_status)) return v_status;
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

auto [a, a_status] = nk::tensor<nk::f32_t>::full({2, 4}, nk::f32_t {1});
auto [b, b_status] = nk::tensor<nk::f32_t>::full({3, 4}, nk::f32_t {2});
if (nk::failed(a_status) || nk::failed(b_status)) return nk::failed(a_status) ? a_status : b_status;
auto [packed, packed_status] = nk::packed_matrix<nk::f32_t>::make(b.as_matrix_view());
if (nk::failed(packed_status)) return packed_status;

// Dot products, angular distances, and Euclidean distances all reuse the same packed B
auto [dots, dots_status] = nk::dots_packed<nk::f32_t>(a.as_matrix_view(), packed); // nk::matrix<nk::f64_t>
auto [angulars, angulars_status] = nk::angulars_packed<nk::f32_t>(a.as_matrix_view(), packed);
auto [euclideans, euclideans_status] = nk::euclideans_packed<nk::f32_t>(a.as_matrix_view(), packed);
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
The C ABI also exposes `nk_dots_pack_size_*_best`, which writes the byte count of the capability the mask picks, so you can `malloc` the exact external buffer yourself.
In C++, `nk::dots_pack_size<nk::f32_t>(rows, depth)` returns that count as an `nk::expected<std::size_t>`.
A packed buffer records the capability that packed it, and a packed kernel of another capability refuses it with `nk_pack_mismatch_k`, so pack and multiply under the same mask.

## Symmetric Kernels for SYRK-Like Workloads

The symmetric kernels solve a different problem.
They compute self-similarity or self-distance without paying for both triangles independently.

```cpp
auto [vectors, vectors_status] = nk::tensor<nk::f32_t>::full({100, 768}, nk::f32_t {1});
if (nk::failed(vectors_status)) return vectors_status;
auto [gram, gram_status] = nk::dots_symmetric<nk::f32_t>(vectors.as_matrix_view()); // nk::matrix<nk::f64_t>
auto [angular_dists, angular_status] = nk::angulars_symmetric<nk::f32_t>(vectors.as_matrix_view());
auto [euclidean_dists, euclidean_status] = nk::euclideans_symmetric<nk::f32_t>(vectors.as_matrix_view());
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
nk_sparse_intersect_u32_best(a_idx, b_idx, 4, 4, intersection, &count, capabilities, NULL);
assert(count == 2 && "indices 3 and 5");

nk_f32_t a_weights[] = {1.0f, 2.0f, 3.0f, 4.0f};
nk_f32_t b_weights[] = {5.0f, 6.0f, 7.0f, 8.0f};
nk_f64_t result = 0;
nk_sparse_dot_u32f32_best(a_idx, b_idx, a_weights, b_weights, 4, 4, &result, capabilities, NULL);
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
nk_f32_t scale = 0;
nk_f64_t rmsd = 0; // widened f32 → f64 output

nk_umeyama_f32_best(source, target, 3, a_centroid, b_centroid, rotation, &scale, &rmsd, capabilities, NULL);
assert(rmsd < 1e-6 && "umeyama should recover exact alignment");
assert(scale > 1.99f && scale < 2.01f && "umeyama should recover 2x scale");
```

In C++, the concept overloads return all of it at once:

```cpp
std::array<nk::f32_t, 9> source {0, 0, 0, 1, 0, 0, 0, 1, 0}, target {0, 0, 0, 2, 0, 0, 0, 2, 0};
auto [fit, status] = nk::umeyama<nk::f32_t>(source, target); // fit.a_centroid, .b_centroid, .rotation, .scale, .rmsd
```

This family is separate from curved metrics because the output is a transform, not just a distance.

## MaxSim and Late Interaction

MaxSim is the late-interaction primitive used by systems such as [ColBERT](https://arxiv.org/abs/2004.12832).
It is not generic matrix multiplication.
It packs query and document token vectors into a scoring-specific layout and computes a late-interaction score.

```cpp
auto [queries, queries_status] = nk::tensor<nk::bf16_t>::full({32, 128}, nk::bf16_t::one());
auto [docs, docs_status] = nk::tensor<nk::bf16_t>::full({192, 128}, nk::bf16_t::one());
if (nk::failed(queries_status) || nk::failed(docs_status)) return nk::failed(queries_status) ? queries_status : docs_status;

auto [q, q_status] = nk::packed_maxsim<nk::bf16_t>::make(queries.as_matrix_view());
auto [d, d_status] = nk::packed_maxsim<nk::bf16_t>::make(docs.as_matrix_view());
if (nk::failed(q_status) || nk::failed(d_status)) return nk::failed(q_status) ? q_status : d_status;
auto score = nk::maxsim(q, d);
```

`packed_maxsim` is allocator-aware in the same way as `packed_matrix`.
Its footprint is exposed through `size_bytes()`.

## Attention over Packed Key-Value Caches

Attention packs a ragged batch of keys and values once, then attends to it with any number of query batches.
`packed_attention` keeps the segment offsets it was packed with, so the view overloads run self-attention over `[tokens, heads, depth]` queries without restating the batch, with query heads shared across fewer key-value heads.

```cpp
#include <numkong/attention.hpp>

// Two sequences of 32 and 64 tokens, 8 query heads over 2 key-value heads of depth 64
auto [keys, keys_status] = nk::tensor<nk::bf16_t>::full({96, 2, 64}, nk::bf16_t::one());
auto [values, values_status] = nk::tensor<nk::bf16_t>::full({96, 2, 64}, nk::bf16_t::one());
auto [queries, queries_status] = nk::tensor<nk::bf16_t>::full({96, 8, 64}, nk::bf16_t::one());
auto [output, output_status] = nk::tensor<nk::f32_t>::zeros({96, 8, 64});
for (nk::status_t made : {keys_status, values_status, queries_status, output_status})
    if (nk::failed(made)) return made;
nk_u32_t const offsets[] = {0, 32, 96}, lengths[] = {32, 64};

auto [packed, packed_status] = nk::packed_attention<nk::bf16_t>::make(
    keys.view(), values.view(), nk::vector_view<nk_u32_t>(offsets, 3u), nk::vector_view<nk_u32_t>(lengths, 2u));
if (nk::failed(packed_status)) return packed_status;
nk::status_t status = nk::attention_causal_packed<nk::bf16_t>(queries.view(), packed, output.span(), 0.125f);
auto [fresh, fresh_status] = nk::attention_bidirectional_packed<nk::bf16_t>(queries.view(), packed, 0.125f); // nk::tensor<nk::f32_t>
```

The causal overloads also take an `nk::causal_mask_t`, whose diagonal offset aligns queries to the end of a longer cache and whose window slides.
For example, `nk::attention_causal_packed<nk::bf16_t>(queries.view(), packed, output.span(), 0.125f, {.window = 4096})` attends to at most 4096 keys per row.
The raw-pointer overloads take the query offsets separately, for cross-attention, and a window over the segments × heads task grid, for sharding one launch across workers.

## Capabilities and Devices

Linking the library is the default, and the way to ship one binary across many CPU generations.
Every query writes its answer through a pointer and returns an `nk_status_t`.
CPU capabilities are reported along two independent axes, plus the mask to dispatch with:

- `nk_cpu_capabilities_detected` is what this CPU can execute, from CPUID or HWCAP.
- `nk_cpu_capabilities_compiled` is what this binary contains, from the ISA probes at build time.
- `nk_cpu_capabilities_enabled` is both axes at once, the mask to pass every CPU call, and always holds `nk_cap_serial_k`.

Ask for `enabled` unless you specifically mean one of the raw axes.
The two axes are independent, and conflating them fails quietly rather than loudly: a binary whose ISA probes failed still reports this machine's full `detected` mask while containing no SIMD kernels at all.
To dispatch over fewer capabilities, pass a narrower mask to the call itself; there is no process-wide setting to change.

`nk_cpu_configure_thread` prepares the calling thread for the capabilities it is given, and only those.
Most capabilities need nothing: AMX on Linux costs one `arch_prctl` syscall, which grants tile state to the whole process, and fused BF16 dot products on Arm cost one `FPCR` write per thread.

```c
nk_capability_t enabled = nk_cap_serial_k;
nk_cpu_capabilities_enabled(&enabled);
nk_cpu_configure_thread(enabled);
if (enabled & nk_cap_sapphireamx_k) { /* AMX both detected and compiled in */ }
nk_dot_bf16_best(a, b, n, &dot, enabled & ~nk_cap_sapphireamx_k, NULL); // dispatch without AMX
```

GPUs are more ISAs with a mask of their own, one device at a time.
Each vendor has its own queries, and a device index is that runtime's own ordinal: the one `cudaSetDevice` or `hipSetDevice` takes, or the position in Metal's device list.
`nk_cuda_count_devices` counts the devices, and `nk_cuda_capabilities_detected`, `nk_cuda_capabilities_compiled`, and `nk_cuda_capabilities_enabled` report one device's capabilities the way their CPU twins do; `nk_rocm_*` and `nk_metal_*` are the same for the other vendors.
Each vendor's baseline, `nk_cap_cuda_k`, `nk_cap_rocm_k`, or `nk_cap_metal_k`, plays the role of `nk_cap_serial_k`.
A GPU capability, like `nk_dots_packed_bf16_ampere`, takes its CPU twin's arguments, queues on the trailing stream — a `cudaStream_t`, a `hipStream_t`, or an `nk_metal_queue_t *` from `nk_metal_queue_init` — and returns without waiting.

```c
nk_size_t devices = 0;
nk_capability_t gpu = 0;
if (nk_cuda_count_devices(&devices) == nk_success_k && nk_cuda_capabilities_enabled(0, &gpu) == nk_success_k)
    nk_dots_packed_bf16_best(a, b_packed, c, height, width, depth, a_stride, c_stride, gpu, cuda_stream);
```

`nk_capabilities_name` spells any such mask as the names bindings accept, like "serial,neon,neonhalf", into a buffer of `NUMKONG_CAPABILITIES_NAME_CAPACITY` bytes.

For exact register-level details, see `capabilities.h`.
Capability kernels, like `nk_dot_f32_haswell`, stay callable directly if you want to pin a path for testing or benchmarking.

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
    auto a_slice = a[range(start, stop), all, slice].as_matrix();
    auto c_slice = c[range(start, stop), all, slice].as_matrix();
    statuses[t] = nk::dots_packed<value_type_>(a_slice, packed, c_slice);
});
```

SYRK-like symmetric work is partitioned by output row windows on one matrix:

```cpp
fork_union.parallel_for(0, worker_count, [&](std::size_t t) {
    auto start = t * rows_per_worker;
    auto count = std::min(rows_per_worker, total_rows - start);
    statuses[t] = nk::dots_symmetric<value_type_>(vectors.as_matrix_view(), gram.as_matrix_span(), start, count);
    if (nk::succeeded(statuses[t]))
        statuses[t] = nk::angulars_symmetric<value_type_>(vectors.as_matrix_view(), angular_dists.as_matrix_span(), start, count);
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

## Building and Cross-Compiling

The build enforces C99 for the C layer and C++20 for the C++ layer.
[CONTRIBUTING.md](../CONTRIBUTING.md#building) lists the CMake presets and options, and [its cross-compilation section](../CONTRIBUTING.md#cross-compilation) the toolchain files in `cmake/` with a recipe for each target.
A translation unit linking `numkong::static` or `numkong::shared` sees declarations only, while `numkong::header` defines `NUMKONG_HEADER_ONLY=1` and compiles the kernels inline.

## Threading Model

The C library creates no threads, uses no OpenMP, and keeps no hidden thread pool.
Its shared and static builds link pthreads only for `pthread_once`, which runs capability detection once per process.
Parallelism is host-controlled: partition work across row ranges and dispatch through ForkUnion, `std::thread`, or any external scheduler.
The Python and Node bindings parallelize their batched calls through `c/parallel.c`, which the C library never links: OpenMP on Linux and FreeBSD, Grand Central Dispatch on macOS, and the system thread pool on Windows.

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

nk_dot_f32_best(cuda_managed_ptr, cuda_managed_ptr, 1024, &dot, capabilities, NULL); // C ABI, any pointer
auto view = nk::tensor_view<nk::f32_t>(mmap_ptr, rows, cols);                         // non-owning view
auto v = nk::vector<float, cuda_allocator<float>>::zeros(1024);                       // allocator-aware owning
```
