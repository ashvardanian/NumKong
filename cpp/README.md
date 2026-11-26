# NumKong for C++

The C++ SDK is a thin, typed layer over the C ABI of [`c/numkong/`](../c/README.md), adding type-level result promotion and owning containers without hiding the dispatch model or the mixed-precision policy.
It stays allocator-aware and close enough to inline through, and lives in `cpp/numkong/`: `types.hpp`, `memory.hpp`, `random.hpp`, `vector.hpp`, `tensor.hpp` and `matrix.hpp` hold the vocabulary of scalars, allocators, vectors, tensors and packed matrices, and one header per kernel family wraps its C header.
`numkong/numkong.hpp` includes them all except `numkong/attention.hpp`, and everything lives in `ashvardanian::numkong`.

The wrappers come in three layers.
Raw-pointer overloads mirror the C ABI one to one, with explicit lengths, strides and task windows for sharded launches.
Overloads over the `nk::a_vector_of` and matrix concepts accept any contiguous run, like `std::vector`, `std::array`, `std::span`, a C array or an `nk::vector_view`, and return their results.
Owning types, like `nk::tensor`, `nk::packed_matrix`, `nk::packed_maxsim` and `nk::packed_attention`, allocate through your allocator for repeated workloads.

## Installation

```cmake
include(FetchContent)

FetchContent_Declare(
    numkong
    GIT_REPOSITORY https://github.com/ashvardanian/NumKong.git
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(numkong)

target_link_libraries(my_target PRIVATE numkong::cpp) # or numkong::header
```

`numkong::cpp` adds `cpp/` as an include root beside the `c/` of `numkong::abi`, requires C++20, and links `numkong::static`, so the wrappers dispatch through the compiled kernels.
`numkong::header` carries the same SDK with every kernel compiled into your own translation units instead; the [C guide](../c/README.md#installation) lists every target and the header-only trade-off.
An installed NumKong provides the same targets through `find_package(numkong CONFIG REQUIRED)`.

## Quickstart

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
Multi-part results come back as small structs: `nk::reduce_moments` returns `{sum, sumsq}`, `nk::reduce_minmax` a `minmax_result`, and `nk::shapes_packed::make` an owner of many packs.

## Statuses & Devices

Every wrapper ends in the two arguments of its dispatch point, defaulted to `nk::default_capabilities()` and a null stream, and returns an `nk::status_t`.
That scoped enum mirrors `nk_status_t` value for value, converts to it with `static_cast`, and is tested with `nk::succeeded` and `nk::failed` rather than as a `bool`.
Factories, like `nk::tensor<T>::zeros`, and the allocating overloads return an `nk::expected<T>` holding the `value` and its `status`, which converts to `true` on success and unpacks with structured bindings.
A failed factory leaves its `value` empty, while a zero-volume request succeeds with an empty one.
`nk::default_capabilities()` is the `nk_capabilities_enabled_cpu` mask, and a zero mask runs the C++ reference template instead of any capability's kernel.
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
The [C guide](../c/README.md#capabilities-and-devices) describes the masks themselves.

## Scalar Types

The scalar wrappers in `cpp/numkong/types.hpp` are storage-first types over the C scalars the [C guide](../c/README.md#scalar-types) tabulates.
They encode raw layout, default output types, and the kernel function pointer signatures for each family.

Default promotions are encoded on the type.
For example, `f32_t::dot_result_t` is wider than `f32_t`.
`i8_t::dot_result_t` is `i32_t`.
`u1x8_t::dot_result_t` is `u32_t`.

The higher-level templates use `result_type_ = typename in_type_::dot_result_t` and similar defaults.
The fast typed overloads are constrained so that overriding the result type away from the native policy can disable the specialized path and fall back to the more generic one.
If you override result types away from the scalar defaults, document that choice carefully because it can change both performance and numerical policy.

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

## Tensors and Views

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

The API is intentionally not STL-shaped.
`vector_view`, `tensor_view`, and `matrix_view` prioritize signed strides, sub-byte storage, and kernel compatibility over resizable-container ergonomics.

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
For sub-byte types each dimension is a logical scalar.
Use `.dims()` to iterate values without positions.

```cpp
for (auto [pos, val] : matrix)          { /* pos is std::array<size_t, R> */ }
for (auto [pos, ref] : matrix.span())   { ref = nk::f32_t{1}; }
for (auto val : matrix.dims())          { /* scalar only, no position */ }
```

## Matrix Kernels

`nk::packed_matrix` packs the right-hand side once and reuses it across any number of left-hand batches, with the economics the [C guide](../c/README.md#matrix-kernels) lists.

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

Owned `packed_matrix` storage uses its allocator.
To pack into your own buffer instead, `nk::dots_pack_size<nk::f32_t>(rows, depth)` returns its byte count as an `nk::expected<std::size_t>`.

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

## Many-to-Many Shape Distances

`nk::shapes_packed` owns many packs in one allocation and exposes the pointer array the [C shape kernels](../c/README.md#many-to-many-shape-distances) take, and `make_normalized` packs them normalized:

```cpp
nk::f32_t const query[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
nk::f32_t const candidate[] = {0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 1, 1, 1};
std::span<nk::f32_t const> clouds[] = {query, candidate};
auto [packs, status] = nk::shapes_packed<nk::f32_t>::make(clouds);
nk::f64_t distances[4];
status = nk::hausdorffs_packed<nk::f32_t>(packs.data(), 2, packs.data(), 2, distances, 2 * sizeof(nk::f64_t), nullptr,
                                          nk::f64_t::positive_infinity());
```

## MaxSim

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
std::size_t const all_keys = std::numeric_limits<std::size_t>::max(), causal = 0;
nk::status_t status = nk::attention_packed<nk::bf16_t>(queries.view(), packed, output.span(), 0.125f, all_keys, causal);
auto [fresh, fresh_status] = nk::attention_packed<nk::bf16_t>(queries.view(), packed, 0.125f); // nk::tensor<nk::f32_t>
```

Every overload takes the band of keys each query sees, `keys_before` and `keys_after` its position, every key by default.
Queries align to the end of each segment's keys, so `keys_after = 0` is causal for prefill and decode alike, and `keys_before = 4095` with it attends to at most 4096 keys per row.
The raw-pointer overloads take the query offsets separately, for cross-attention, and a window over the query tokens × heads task grid, for sharding one launch across workers.

## Parallelism

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

## External Memory

C++ views wrap any pointer without ownership, and owning containers accept any C++ Allocator.
`aligned_allocator` defaults to 64-byte alignment for owned containers, but unaligned caller inputs are still valid for the kernels that accept raw pointers or views.

```cpp
template <typename T>
struct cuda_allocator {
    using value_type = T;
    T *allocate(std::size_t n) { T *p;
        cudaMallocManaged(&p, n * sizeof(T), cudaMemAttachGlobal); 
        return p; }
    void deallocate(T *p, std::size_t) noexcept { cudaFree(p); }
};

auto view = nk::tensor_view<nk::f32_t>(mmap_ptr, rows, columns);                      // non-owning view
auto v = nk::vector<float, cuda_allocator<float>>::zeros(1024);                       // allocator-aware owning
```
