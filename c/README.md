# NumKong for C

NumKong's native SDK is the reference surface for the project.
The plain C ABI exposes every kernel family directly: dot products, dense distances, binary metrics, probability divergences, geospatial solvers, curved-space kernels, sparse intersections, many-to-many shape distances, packed matrix multiplication, symmetric self-similarity, late-interaction scoring, elementwise arithmetic, reductions, type conversions, scalar math, and trigonometry.
The ABI is stable, versioned, and callable from any language that can load a shared library.
There is no runtime overhead: no hidden thread pool, no implicit allocation, no garbage collector interaction.
The [C++ SDK](../cpp/README.md) is a typed layer over this ABI, adding result promotion and owning containers.

## Quickstart

```c
#include <numkong/numkong.h>
#include <stdio.h>

int main(void) {
    nk_f32_t a[] = {1, 2, 3};
    nk_f32_t b[] = {4, 5, 6};
    nk_f64_t dot = 0;
    nk_capability_t capabilities = nk_cap_serial_k;
    nk_capabilities_enabled_cpu(&capabilities); // the capabilities this CPU runs and this build holds
    nk_configure_thread_cpu(capabilities);
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
Every call takes caller-owned buffers, and batched kernels take a caller's workspace.
__Mixed precision by default.__
Small storage types widen into safer accumulator and output types.
__Unaligned inputs are fine.__
Packing handles internal layout itself and does not require caller-side alignment.

## Ecosystem Comparison

| Feature                      | NumKong                                                                                                                             | [OpenBLAS][openblas]                                              | [Eigen][eigen]                                                                          |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| Operation families           | dots, distances, binary, probability, geospatial, curved, shapes, sparse, MaxSim, elementwise, reductions, cast, trig               | dense linear algebra only                                         | dense LA, some reductions and elementwise                                               |
| Precision                    | Sub-byte to Float64 dtypes; automatic widening per scalar type; Kahan-compensated summation; 0 ULP Float32/Float64 where applicable | Float32, Float64 only; same-type in/out; no compensated summation | Float16/BFloat16 partial; no Float8 or sub-byte; manual casts; no compensated summation |
| Runtime SIMD dispatch        | per call, by capability mask, across x86, Arm, RISC-V, and GPUs                                                                     | load-time CPU detection; one kernel set per process               | compile-time ISA flags only                                                             |
| Packed matrix, GEMM-like     | `nk_dots_pack_*` — pack once, reuse across query batches                                                                            | internal opaque packing per GEMM call; no persistent packed form  | no equivalent packed reuse abstraction                                                  |
| Symmetric kernels, SYRK-like | skips duplicate pairs, up to 2x speedup for self-distance                                                                           | `SSYRK`/`DSYRK` for rank-k updates                                | `.selfadjointView` for rank-k updates                                                   |
| Memory model                 | Caller-owned buffers; the C++ SDK adds containers with per-container allocators                                                     | Caller-managed buffers; no container abstraction                  | Lazy expression templates avoid most temporaries; `aligned_allocator` provided          |

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

target_link_libraries(my_target PRIVATE numkong::header) # or numkong::abi, numkong::static, numkong::shared
```

Vendored:

```cmake
add_subdirectory(external/NumKong)
target_link_libraries(my_target PRIVATE numkong::header) # or numkong::abi, numkong::static, numkong::shared
```

Each target names what it reaches:

- `numkong::abi` is the C headers of `c/`, with nothing to link.
- `numkong::cpp` is the [C++ SDK](../cpp/README.md) of `cpp/`, linking `numkong::abi` and `numkong::static`.
- `numkong::header` is the whole library header-only: the C headers, the kernels of `include/` and the C++ SDK of `cpp/`, compiled into your own translation units.
- `numkong::static` and `numkong::shared` are the compiled library.

`numkong::static` and `numkong::shared` are always declared, except that WASI and Emscripten have no shared library.
`NUMKONG_BUILD_LIBRARIES`, on by default only for a top-level build, builds them by default and installs them; off, they build only for a target that links them.
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
The typed [C++ wrappers](../cpp/README.md) usually read better unless you are building your own dispatch layer.

## Scalar Types

The scalar types in `c/numkong/types.h` are storage-first: each is the raw bits of one value, or of several packed into a byte.

| Type          | Layout           | Bytes |           Range | Inf | NaN |
| :------------ | :--------------- | ----: | --------------: | :-: | :-: |
| `nk_f16_t`    | 1+5+10           |     2 |          ±65504 | yes | yes |
| `nk_bf16_t`   | 1+8+7            |     2 |       ±3.4×10³⁸ | yes | yes |
| `nk_e4m3_t`   | 1+4+3            |     1 |            ±448 | no  | yes |
| `nk_e5m2_t`   | 1+5+2            |     1 |          ±57344 | yes | yes |
| `nk_e2m3_t`   | 1+2+3            |     1 |            ±7.5 | no  | no  |
| `nk_e3m2_t`   | 1+3+2            |     1 |             ±28 | no  | no  |
| `nk_e2m1x2_t` | 2 × (1+2+1)      |     1 |   ±6 per nibble | no  | no  |
| `nk_u1x8_t`   | 8 packed bits    |     1 |  0 or 1 per bit | n/a | n/a |
| `nk_u4x2_t`   | 2 × 4-bit        |     1 | 0…15 per nibble | n/a | n/a |
| `nk_i4x2_t`   | 2 × 4-bit signed |     1 | −8…7 per nibble | n/a | n/a |

The layout column shows sign, exponent, and mantissa bit counts for floating-point types.
For `nk_f16_t`, 1+5+10 means one sign bit, five exponent bits, and ten mantissa bits, totaling 16 bits stored in 2 bytes.
For `nk_bf16_t`, the wider exponent field (8 bits) gives the same dynamic range as IEEE 754 single precision but with reduced mantissa precision.
The Float8 types `nk_e4m3_t` and `nk_e5m2_t` follow the OFP8 specification.
The narrower `nk_e2m3_t` and `nk_e3m2_t` types are MX-compatible micro-floats, and `nk_e2m1x2_t` packs two FP4 values of the MXFP4 and NVFP4 formats.
Sub-byte types `nk_e2m1x2_t`, `nk_u1x8_t`, `nk_u4x2_t`, and `nk_i4x2_t` pack multiple logical values into a single byte.
Element 0 sits in the high nibble or the most significant bit, and every dimension count must be a multiple of the values per byte.

The [C++ SDK](../cpp/README.md#scalar-types) wraps each in a type that carries its default output types and formats itself.

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

The native SDK exposes both the fast spherical approximation and the more accurate ellipsoidal solver, each between every row and every column of two packed point sets.
Inputs are in radians.
Outputs are in meters.

```c
// Statue of Liberty (40.6892°N, 74.0445°W) and Big Ben (51.5007°N, 0.1246°W)
nk_f64_t latitudes[] = {0.7101605100, 0.8988567821}, longitudes[] = {-1.2923203180, -0.0021746802};

// A pack is built once per point set, on an `nk_default_alignment_k` boundary
nk_size_t bytes = 0;
nk_vincentys_pack_size_f64_best(2, capabilities, &bytes);
void *vincentys_pack = aligned_alloc(nk_default_alignment_k, bytes);
nk_vincentys_pack_f64_best(latitudes, longitudes, 2, vincentys_pack, capabilities, NULL);

// Every row against every column, so the diagonal is zero
nk_f64_t distances[2][2];
nk_vincentys_packed_f64_best(vincentys_pack, vincentys_pack, &distances[0][0], 2 * sizeof(nk_f64_t), capabilities,
                             NULL); // distances[0][1] ≈ 5,589,857 m on WGS-84

nk_haversines_pack_size_f64_best(2, capabilities, &bytes);
void *haversines_pack = aligned_alloc(nk_default_alignment_k, bytes);
nk_haversines_pack_f64_best(latitudes, longitudes, 2, haversines_pack, capabilities, NULL);
nk_haversines_packed_f64_best(haversines_pack, haversines_pack, &distances[0][0], 2 * sizeof(nk_f64_t),
                              capabilities, NULL); // distances[0][1] ≈ 5,574,848 m on the mean sphere, 15 km less
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

## Matrix Kernels

This is a separate native subsystem from the raw vector kernels.
It is the right tool when the right-hand side is reused many times.

```c
nk_f32_t a[2 * 4], b[3 * 4]; // two rows against three columns, both of depth 4
nk_f64_t c[2 * 3];           // widened f32 → f64 output
nk_size_t pack_bytes = 0, workspace_bytes = 0;
nk_dots_pack_size_f32_best(3, 4, capabilities, &pack_bytes);
void *b_packed = malloc(pack_bytes);
nk_dots_pack_f32_best(b, 3, 4, 4 * sizeof(nk_f32_t), b_packed, 0, 3, capabilities, NULL);
nk_dots_packed_workspace_size_f32_best(2, 3, 4, capabilities, &workspace_bytes);
void *workspace = workspace_bytes ? malloc(workspace_bytes) : NULL;
nk_dots_packed_f32_best(a, b_packed, c, 2, 3, 4, 4 * sizeof(nk_f32_t), 3 * sizeof(nk_f64_t), workspace, workspace_bytes,
                        capabilities, NULL);
```

This is GEMM-like in the workload shape, not in the strict BLAS API.
The useful economics are:

- one-time packing of `B`
- one-time type preconversion where needed
- depth padding handled internally
- per-column norm reuse for `angulars_packed` and `euclideans_packed`
- repeated reuse of the same packed RHS across many `A` batches

Caller-side alignment is not required.
`nk_dots_pack_size_*_best` writes the byte count of the capability the mask picks, so you can `malloc` the exact external buffer yourself.
A packed buffer records the capability that packed it, and a packed kernel of another capability refuses it with `nk_pack_mismatch_k`, so pack and multiply under the same mask.
The symmetric `nk_dots_symmetric_*`, `nk_angulars_symmetric_*` and `nk_euclideans_symmetric_*` kernels score one matrix against itself, computing each pair once rather than both `(i, j)` and `(j, i)`, over a window of output rows from `rows_begin` to `rows_end`; the [C++ guide](../cpp/README.md#symmetric-kernels-for-syrk-like-workloads) shows them.

## Sparse Operations

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

## Many-to-Many Shape Distances

Shape kernels compare 3D point clouds of any size, with no correspondence between points: Chamfer, Hausdorff, Wasserstein-Procrustes, and Gromov-Wasserstein.
Each cloud is packed once, centered and rotated onto its principal axes, and every kernel scores a list of row packs against a list of column packs.
Packs are built one cloud at a time, so a caller passes whatever packs it holds as an array of pointers.

```c
nk_f32_t const query[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};          // 4 points
nk_f32_t const candidate[] = {0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 1, 1, 1}; // 5 points
nk_size_t const alignment = nk_default_alignment_k; // `aligned_alloc` sizes are whole multiples of it
nk_size_t query_bytes = 0, candidate_bytes = 0;
nk_shapes_pack_size_f32_best(4, capabilities, &query_bytes);
nk_shapes_pack_size_f32_best(5, capabilities, &candidate_bytes);
void *query_packed = aligned_alloc(alignment, (query_bytes + alignment - 1) / alignment * alignment);
void *candidate_packed = aligned_alloc(alignment, (candidate_bytes + alignment - 1) / alignment * alignment);
nk_shapes_pack_f32_best(query, 4, query_packed, capabilities, NULL);
nk_shapes_pack_f32_best(candidate, 5, candidate_packed, capabilities, NULL);

void const *rows[] = {query_packed}, *columns[] = {candidate_packed};
nk_f64_t chamfer = 0, procrustes = 0;
nk_f32_t rotation[9]; // takes the candidate onto the query, through the centroids in the headers
nk_size_t chamfer_bytes = 0;
nk_chamfers_workspace_size_f32_best(4, 5, capabilities, &chamfer_bytes);
void *chamfer_workspace = aligned_alloc(alignment, (chamfer_bytes + alignment - 1) / alignment * alignment);
nk_chamfers_packed_f32_best(rows, 1, columns, 1, &chamfer, sizeof(nk_f64_t), rotation, INFINITY, chamfer_workspace,
                            chamfer_bytes, capabilities, NULL);

nk_size_t workspace_bytes = 0;
nk_wassersteinprocrustes_workspace_size_f32_best(4, 5, capabilities, &workspace_bytes);
void *workspace = aligned_alloc(alignment, (workspace_bytes + alignment - 1) / alignment * alignment);
nk_wassersteinprocrustes_packed_f32_best(rows, 1, columns, 1, &procrustes, sizeof(nk_f64_t), NULL, workspace,
                                         workspace_bytes, capabilities, NULL);

nk_f64_t query_radius = 0, candidate_radius = 0; // |query_radius - candidate_radius| bounds Hausdorff from below
nk_shapes_packed_shape_f32_best(query_packed, NULL, NULL, NULL, NULL, NULL, &query_radius, capabilities, NULL);
nk_shapes_packed_shape_f32_best(candidate_packed, NULL, NULL, NULL, NULL, NULL, &candidate_radius, capabilities, NULL);
nk_f32_t restored[15]; // the candidate again, within one rounding into `f32`
nk_shapes_unpack_f32_best(candidate_packed, restored, capabilities, NULL);
```

A pack is opaque: `nk_shapes_packed_shape_*` reads its points, centroid, principal frame as a quaternion, scale, descending spreads, and largest radius, and `nk_shapes_unpack_*` restores its cloud.
The spreads and radius bound Wasserstein-Procrustes and Hausdorff from below for free, the way vector norms bound dot products.
Gromov-Wasserstein and Wasserstein-Procrustes take a caller's workspace, since a transport plan depends on both clouds and fits in neither pack.

`nk_shapes_pack_normalized_*` takes the same arguments and also divides each cloud by its RMS distance from the centroid, which it keeps as the scale, so every kind ignores size as well as pose.
Each kind compares the packed coordinates, so mixing absolute and normalized packs in one call is the caller's choice.
Between normalized packs, a rotation takes column cloud `b` onto row cloud `a` as aᵢ ≈ (scale_a / scale_b) × R × (bⱼ − b̄) + ā.

In C++, [`nk::shapes_packed`](../cpp/README.md#many-to-many-shape-distances) owns many packs in one allocation.

## Capabilities and Devices

Linking the library is the default, and the way to ship one binary across many CPU generations.
Every query writes its answer through a pointer and returns an `nk_status_t`.
CPU capabilities are reported along two independent axes, plus the mask to dispatch with:

- `nk_capabilities_detected_cpu` is what this CPU can execute, from CPUID or the OS, or where the OS cannot be asked, from what the compiler flags guarantee.
- `nk_capabilities_compiled_cpu` is what this binary contains, from the ISA probes at build time; `nk_capabilities_compiled_cpu_mask()` is the same mask as a constant of the including unit's `NUMKONG_TARGET_*` macros, for consumers picking kits at compile time.
- `nk_capabilities_enabled_cpu` is both axes at once, the mask to pass every CPU call, and always holds `nk_cap_serial_k`.

Ask for `enabled` unless you specifically mean one of the raw axes.
The two axes are independent, and conflating them fails quietly rather than loudly: a binary whose ISA probes failed still reports this machine's full `detected` mask while containing no SIMD kernels at all.
To dispatch over fewer capabilities, pass a narrower mask to the call itself; there is no process-wide setting to change.

| Word       | Question                                                      | When                                                                                                       | Scope                         |
| :--------- | :------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------- | :---------------------------- |
| `probe`    | Can the toolchain build this capability?                      | Configure, as CMake compiles `probes/<capability>.c`, then compile, as `NUMKONG_ARCH_<ARCH>_<CAPABILITY>_` | The build                     |
| `target`   | Is this capability's unit in the build?                       | Compile, as `NUMKONG_TARGET_<CAPABILITY>`                                                                  | The build                     |
| `implied`  | What do the floor's flags already guarantee?                  | Compile, as both macros default where no build writes them, and at runtime where no OS is asked            | The build                     |
| `compiled` | Does the binary carry this capability's code for this device? | Runtime, as `nk_capabilities_compiled_cpu` or `_cuda(ordinal)` report, the latter asking the driver        | The build, or each GPU's code |
| `detected` | Does this device offer the capability?                        | Runtime, as `nk_capabilities_detected_cpu` or `_cuda(ordinal)` report                                      | The device                    |
| `enabled`  | What may dispatch pick?                                       | Runtime, `detected` and `compiled` at once, less a refused AMX grant                                       | The device                    |

A monoversioned build names one machine: one CPU, and at most one GPU architecture per vendor that every GPU in it shares.
It compiles the best capability that machine runs and folds every dispatch row to it, and CMake refuses two CUDA or two ROCm architectures with `NUMKONG_MULTIVERSIONED=OFF`.
On any other machine, or under a mask narrowed below the folded capability, a dispatch point answers `nk_missing_kernel_k`, never a crash or a slower kernel.
The scalar math and element conversions, like `nk_f32_sqrt_best` and `nk_f16_to_f32_best`, return values rather than statuses, so their rows keep every kernel.

`nk_configure_thread_cpu` prepares the calling thread for the capabilities it is given, and only those.
Most capabilities need nothing: AMX on Linux costs one `arch_prctl` syscall, which grants tile state to the whole process, and fused BF16 dot products on Arm cost one `FPCR` write per thread.

```c
nk_capability_t enabled = nk_cap_serial_k;
nk_capabilities_enabled_cpu(&enabled);
nk_configure_thread_cpu(enabled);
if (enabled & nk_cap_sapphireamx_k) { /* AMX both detected and compiled in */ }
nk_dot_bf16_best(a, b, n, &dot, enabled & ~nk_cap_sapphireamx_k, NULL); // dispatch without AMX
```

GPUs are more ISAs with a mask of their own, one device at a time.
Each vendor has its own queries, and a device index is that runtime's own ordinal: the one `cudaSetDevice` or `hipSetDevice` takes, or the position in Metal's device list.
`nk_count_devices_cuda` counts the devices, and `nk_capabilities_detected_cuda`, `nk_capabilities_compiled_cuda`, and `nk_capabilities_enabled_cuda` report one device's capabilities the way their CPU twins do; the `_rocm` and `_metal` functions are the same for the other vendors.
A GPU's `compiled` mask holds only the capabilities whose unit carries code that device runs, as the runtime answers when asked to load one of the unit's kernels there with `cudaFuncGetAttributes` or `hipFuncGetAttributes`.
Projects compiling their own GPU units call the same two functions with their own lists, and need no GPU runtime for it.
Each vendor's baseline, `nk_cap_cuda_k`, `nk_cap_rocm_k`, or `nk_cap_metal_k`, plays the role of `nk_cap_serial_k`.
A GPU capability, like `nk_dots_packed_bf16_ampere`, takes its CPU twin's arguments, queues on the trailing stream — a `cudaStream_t`, a `hipStream_t`, or an `id<MTLCommandQueue>`, with null for the default one — and returns without waiting.
Only the capability queries take an ordinal; the stream names the device for everything else.
`nk_memory_allocate_unified_best` and `nk_memory_free_unified_best` hand out and return memory the host and that device both address, and `nk_stream_synchronize_best` waits for the stream.
Batched kernels that may need scratch, like the packed and symmetric families and attention, take a `workspace, workspace_bytes` pair before the capabilities and never allocate.
Their `*_workspace_size_*` twin reports the bytes from the shapes alone, and a zero-byte workspace runs the scratch-free path where the capability has one.

```c
nk_size_t devices = 0, workspace_bytes = 0;
nk_capability_t gpu = 0;
if (nk_count_devices_cuda(&devices) == nk_success_k && nk_capabilities_enabled_cuda(0, &gpu) == nk_success_k) {
    nk_dots_packed_workspace_size_f64_best(rows, columns, depth, gpu, &workspace_bytes);
    void *workspace = NULL;
    nk_memory_allocate_unified_best(workspace_bytes, &workspace, gpu, cuda_stream);
    nk_dots_packed_f64_best(a, b_packed, c, rows, columns, depth, a_stride, c_stride, workspace, workspace_bytes, gpu,
                            cuda_stream);
    nk_stream_synchronize_best(gpu, cuda_stream);
}
```

`nk_capabilities_name` spells any such mask as the names bindings accept, like "serial,neon,neonhalf", into a buffer of `NUMKONG_CAPABILITIES_NAME_CAPACITY` bytes.

For exact register-level details, see `capabilities.h`.
Capability kernels, like `nk_dot_f32_haswell`, stay callable directly if you want to pin a path for testing or benchmarking.

## Building and Cross-Compiling

The build enforces C99 for the C layer.
[CONTRIBUTING.md](../CONTRIBUTING.md#building) lists the CMake presets and options, and [its cross-compilation section](../CONTRIBUTING.md#cross-compilation) the toolchain files in `cmake/` with a recipe for each target.
A translation unit linking `numkong::cpp`, `numkong::static` or `numkong::shared` sees declarations only, while `numkong::header` defines `NUMKONG_HEADER_ONLY=1` and compiles the kernels inline.
The library compiles every capability whose probe in `probes/` compiles, each scoped to its kernels by target pragmas, and dispatches between them by runtime detection.
LASX, POWER9 and POWER10 scope that way only under GCC 15 or newer; Clang's intrinsics headers stay closed without `-mlasx`, `-mcpu=power9` or `-mcpu=power10`.
Header-only units enable the capabilities their own flags name.

## Threading Model

The C library creates no threads, uses no OpenMP, and keeps no hidden thread pool.
Its shared and static builds link pthreads only for `pthread_once`, which runs capability detection once per process.
Parallelism is host-controlled: partition work across row ranges and dispatch through [ForkUnion](https://github.com/ashvardanian/ForkUnion), `std::thread`, or any external scheduler, as the [C++ guide](../cpp/README.md#parallelism) shows.
The Python and Node bindings parallelize their batched calls through `src/parallel.c`, which the C library never links: OpenMP on Linux and FreeBSD, Grand Central Dispatch on macOS, and the system thread pool on Windows.

## External Memory

Every kernel takes plain pointers, so any CPU-accessible memory works: mmap, pinned buffers, CUDA unified memory, custom arenas.
That makes the C ABI the easiest place to integrate with foreign runtimes and custom allocators.

```c
nk_dot_f32_best(cuda_managed_ptr, cuda_managed_ptr, 1024, &dot, capabilities, NULL); // any pointer
```
