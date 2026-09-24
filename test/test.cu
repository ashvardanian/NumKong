/**
 *  @file test/test.cu
 *  @author Ash Vardanian
 *  @date April 15, 2026
 *  @brief CUDA test: tensor memcpy round-trips, fp6/fp8 cast conformance, and every CUDA
 *      cross-kernel entry point.
 *
 *  The first half drives @c cudaMemcpy, pitched @c cudaMemcpy2D and @c cudaMemcpy3D round-trips
 *  through @c nk::tensor_view and runs a trivial add-one kernel to prove device-side readability,
 *  while the second half compares @c nk_cast on the CPU bit for bit against CUDA's @c __nv_cvt_*
 *  intrinsics on the GPU, covering every fp32, fp16 and bf16 input against every e4m3, e5m2, e3m2
 *  and e2m3 variant. The cross sections run the dots, spatial and attention scenarios of
 *  `cross.cuh` through @c cuda_backend_t for every family the device runs, and `NK_FILTER=<regex>`
 *  keeps only the sections and kernels whose names match.
 *
 *  The test builds and runs on any Turing-or-newer GPU. CUDA's __nv_cvt_* converters fall back to
 *  software emulation below SM_89 for fp8 and below SM_100 for fp6, and PTX JIT forward-compiles to
 *  newer hardware, so a single binary works from Turing through Blackwell. Override
 *  CMAKE_CUDA_ARCHITECTURES at configure time to target one compute capability, or build a single
 *  fat binary for all.
 */
#include <cinttypes> // `PRIu64`, `PRIx64`
#include <cstdint>   // `std::uint32_t`, `std::uint64_t`
#include <cstdio>    // `std::printf`, `std::fprintf`
#include <cstdlib>   // `std::getenv`
#include <cstring>   // `std::memcpy`

#include <random> // `std::mt19937`
#include <vector> // `std::vector`

#if !defined(_WIN32)
#include <unistd.h> // `isatty`
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "test.cuh"       // `cuda_backend_t`
#include "cross.cuh"      // `test_dots_packed`, `attention_weights_t`

using namespace ashvardanian::numkong::test;

test_config_t nk::test::global_config;
char const *volatile nk::test::nk_test_current_kernel_ = nullptr;

#pragma region Output Helpers

/** Detect whether stdout supports ANSI colors, kept in sync with test/test.cpp:59-70. */
static bool colors_enabled_() {
    static bool const result = [] {
        if (std::getenv("NO_COLOR")) return false;
        if (std::getenv("FORCE_COLOR")) return true;
#if !defined(_WIN32)
        return isatty(fileno(stdout)) != 0;
#else
        return false;
#endif
    }();
    return result;
}

/** Print a filled ● for pass, a hollow ○ for fail. Colored when stdout is a TTY. */
static void print_indicator_(bool on) {
    if (on) std::printf(colors_enabled_() ? "\033[32m\xe2\x97\x8f\033[0m" : "\xe2\x97\x8f");
    else std::printf(colors_enabled_() ? "\033[31m\xe2\x97\x8b\033[0m" : "\xe2\x97\x8b");
}

#pragma endregion

#pragma region CUDA Error Handling

/** Return true on success; on failure, print a diagnostic and return false. */
static bool cuda_check_(cudaError_t error, char const *expression, char const *file, int line) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", expression, file, line, cudaGetErrorString(error));
    return false;
}

/** Early-return assertion: on failure prints a diagnostic and returns @c false from the enclosing
 *  function. Every `test_*_` / `sweep_*_` here returns bool. */
#define nk_cuda_assert_(expression)                                                    \
    do {                                                                               \
        if (!cuda_check_((expression), #expression, __FILE__, __LINE__)) return false; \
    } while (0)

#pragma endregion

#pragma region Tensor Round Trip Tests

/** 1D contiguous round-trip via cudaMemcpy; bit-exact compare. */
static bool test_memcpy_1d_roundtrip_() {
    constexpr std::size_t count = 1 << 14;
    auto host_source = nk::tensor<float>::try_zeros({count});
    auto host_destination = nk::tensor<float>::try_zeros({count});
    if (host_source.empty() || host_destination.empty()) return false;

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-100.0f, 100.0f);
    for (std::size_t i = 0; i < count; ++i) host_source[i] = distribution(generator);

    float *device_pointer = nullptr;
    nk_cuda_assert_(cudaMalloc(&device_pointer, count * sizeof(float)));
    nk_cuda_assert_(cudaMemcpy(device_pointer, host_source.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    nk_cuda_assert_(cudaMemcpy(host_destination.data(), device_pointer, count * sizeof(float), cudaMemcpyDeviceToHost));
    nk_cuda_assert_(cudaFree(device_pointer));

    for (std::size_t i = 0; i < count; ++i)
        if (host_source[i] != host_destination[i]) return false;
    return true;
}

/** 2D round-trip with padded rows: logical @c columns elements per row, but rows pad to
 *  `stride_bytes(0) > columns * sizeof(T)`, which is the host pitch @c cudaMemcpy2D expects. */
static bool test_memcpy_2d_padded_rows_() {
    constexpr std::size_t rows = 32;
    constexpr std::size_t columns = 80;
    constexpr std::size_t row_elements_padded = 128;

    auto host_source = nk::tensor<float>::try_zeros({rows, row_elements_padded});
    auto host_destination = nk::tensor<float>::try_zeros({rows, row_elements_padded});
    if (host_source.empty() || host_destination.empty()) return false;

    std::mt19937 generator(7);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column) host_source(row, column) = distribution(generator);

    std::size_t const host_pitch = static_cast<std::size_t>(host_source.stride_bytes(0));

    float *device_pointer = nullptr;
    std::size_t device_pitch = 0;
    nk_cuda_assert_(cudaMallocPitch(&device_pointer, &device_pitch, columns * sizeof(float), rows));
    nk_cuda_assert_(cudaMemcpy2D(device_pointer, device_pitch, host_source.data(), host_pitch, columns * sizeof(float),
                                 rows, cudaMemcpyHostToDevice));
    nk_cuda_assert_(cudaMemcpy2D(host_destination.data(), host_pitch, device_pointer, device_pitch,
                                 columns * sizeof(float), rows, cudaMemcpyDeviceToHost));
    nk_cuda_assert_(cudaFree(device_pointer));

    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            if (host_source(row, column) != host_destination(row, column)) return false;
    return true;
}

/** 3D batched round-trip via cudaMemcpy3D + cudaPitchedPtr. */
static bool test_memcpy_3d_batched_() {
    constexpr std::size_t batches = 4, rows = 8, columns = 16;
    auto host_source = nk::tensor<float>::try_zeros({batches, rows, columns});
    auto host_destination = nk::tensor<float>::try_zeros({batches, rows, columns});
    if (host_source.empty() || host_destination.empty()) return false;

    std::mt19937 generator(13);
    std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
    for (std::size_t batch = 0; batch < batches; ++batch)
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column)
                host_source(batch, row, column) = distribution(generator);

    cudaPitchedPtr device_pitched;
    cudaExtent const extent = make_cudaExtent(columns * sizeof(float), rows, batches);
    nk_cuda_assert_(cudaMalloc3D(&device_pitched, extent));

    // Tightly-packed host: stride_bytes(1) = row pitch, stride_bytes(0) = slice pitch.
    cudaMemcpy3DParms host_to_device {};
    host_to_device.srcPtr = make_cudaPitchedPtr(host_source.data(), columns * sizeof(float), columns, rows);
    host_to_device.dstPtr = device_pitched;
    host_to_device.extent = extent;
    host_to_device.kind = cudaMemcpyHostToDevice;
    nk_cuda_assert_(cudaMemcpy3D(&host_to_device));

    cudaMemcpy3DParms device_to_host {};
    device_to_host.srcPtr = device_pitched;
    device_to_host.dstPtr = make_cudaPitchedPtr(host_destination.data(), columns * sizeof(float), columns, rows);
    device_to_host.extent = extent;
    device_to_host.kind = cudaMemcpyDeviceToHost;
    nk_cuda_assert_(cudaMemcpy3D(&device_to_host));
    nk_cuda_assert_(cudaFree(device_pitched.ptr));

    for (std::size_t batch = 0; batch < batches; ++batch)
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column)
                if (host_source(batch, row, column) != host_destination(batch, row, column)) return false;
    return true;
}

/** 2D sub-rectangle extraction from a parent tensor via cudaMemcpy2D. The sub-view keeps the parent
 *  row pitch; only extents shrink. */
static bool test_memcpy_2d_subview_() {
    constexpr std::size_t parent_rows = 16, parent_columns = 32;
    constexpr std::size_t row_start = 4, column_start = 8;
    constexpr std::size_t sub_rows = 8, sub_columns = 16;

    auto parent = nk::tensor<float>::try_zeros({parent_rows, parent_columns});
    if (parent.empty()) return false;
    for (std::size_t row = 0; row < parent_rows; ++row)
        for (std::size_t column = 0; column < parent_columns; ++column)
            parent(row, column) = static_cast<float>(row * parent_columns + column);

    float const *sub_source_pointer = &parent(row_start, column_start);
    std::size_t const parent_row_pitch = static_cast<std::size_t>(parent.stride_bytes(0));

    auto host_back = nk::tensor<float>::try_zeros({sub_rows, sub_columns});
    if (host_back.empty()) return false;

    float *device_pointer = nullptr;
    std::size_t device_pitch = 0;
    nk_cuda_assert_(cudaMallocPitch(&device_pointer, &device_pitch, sub_columns * sizeof(float), sub_rows));
    nk_cuda_assert_(cudaMemcpy2D(device_pointer, device_pitch, sub_source_pointer, parent_row_pitch,
                                 sub_columns * sizeof(float), sub_rows, cudaMemcpyHostToDevice));
    nk_cuda_assert_(cudaMemcpy2D(host_back.data(), host_back.stride_bytes(0), device_pointer, device_pitch,
                                 sub_columns * sizeof(float), sub_rows, cudaMemcpyDeviceToHost));
    nk_cuda_assert_(cudaFree(device_pointer));

    for (std::size_t row = 0; row < sub_rows; ++row)
        for (std::size_t column = 0; column < sub_columns; ++column)
            if (host_back(row, column) != parent(row_start + row, column_start + column)) return false;
    return true;
}

/** Trivial add-1.0 kernel used to prove device-side data is readable/writable. */
__global__ void add_one_kernel_(float *data, std::size_t rows, std::size_t columns, std::size_t pitch_bytes) {
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || column >= columns) return;
    float *row_pointer = reinterpret_cast<float *>(reinterpret_cast<char *>(data) + row * pitch_bytes);
    row_pointer[column] += 1.0f;
}

/** Upload, add-one on device, download, verify every element incremented by 1. */
static bool test_device_kernel_usability_() {
    constexpr std::size_t rows = 24, columns = 48;
    auto host_source = nk::tensor<float>::try_zeros({rows, columns});
    auto host_destination = nk::tensor<float>::try_zeros({rows, columns});
    if (host_source.empty() || host_destination.empty()) return false;
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            host_source(row, column) = static_cast<float>((row * columns + column) % 17) * 0.25f;

    float *device_pointer = nullptr;
    std::size_t device_pitch = 0;
    nk_cuda_assert_(cudaMallocPitch(&device_pointer, &device_pitch, columns * sizeof(float), rows));
    nk_cuda_assert_(cudaMemcpy2D(device_pointer, device_pitch, host_source.data(), host_source.stride_bytes(0),
                                 columns * sizeof(float), rows, cudaMemcpyHostToDevice));

    dim3 block(16, 8);
    dim3 grid((columns + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    add_one_kernel_<<<grid, block>>>(device_pointer, rows, columns, device_pitch);
    nk_cuda_assert_(cudaGetLastError());
    nk_cuda_assert_(cudaDeviceSynchronize());

    nk_cuda_assert_(cudaMemcpy2D(host_destination.data(), host_destination.stride_bytes(0), device_pointer,
                                 device_pitch, columns * sizeof(float), rows, cudaMemcpyDeviceToHost));
    nk_cuda_assert_(cudaFree(device_pointer));

    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            if (host_destination(row, column) != host_source(row, column) + 1.0f) return false;
    return true;
}

/** Device kernel exercising the @c constexpr @c nk::tensor_view and @c nk::vector_view surface on
 *  the GPU — this is what proves the abstractions are @c __device__-callable under
 *  `--expt-relaxed-constexpr`. Covers: typed-pointer construction, `operator bool`, 2D
 *  `operator()`, @c slice_leading, `operator[]` on the reduced-rank row, `flatten<1>()`, and
 *  iteration. Writes a checksum so nothing is optimized away. */
__global__ void tensor_view_ops_kernel_(float const *data, std::size_t rows, std::size_t columns, float *out) {
    if (threadIdx.x != 0 || threadIdx.y != 0 || blockIdx.x != 0 || blockIdx.y != 0) return;
    nk::tensor_view<float> view(data, rows, columns); // typed-pointer rank-2 ctor
    float accumulator = 0.0f;
    if (view) { // operator bool
        for (std::size_t row = 0; row < view.extent(0); ++row) {
            auto row_view = view.slice_leading(row); // rank-reducing slice
            for (std::size_t column = 0; column < row_view.extent(0); ++column)
                accumulator += static_cast<float>(row_view[column]); // operator[] on the rank-1 view
        }
        accumulator += static_cast<float>(view(1, 2)); // 2D operator()
        auto flat = view.flatten<1>();                 // flatten<out_rank_>
        accumulator += static_cast<float>(flat.numel());
    }
    nk::vector_view<float> vector(data, rows * columns); // (ptr, count) ctor
    if (vector)
        for (std::size_t i = 0; i < vector.size(); ++i) accumulator += static_cast<float>(vector[i]);
    out[0] = accumulator;
}

/** Run the tensor-abstraction kernel on device, checking its checksum against the identical host
 *  computation, confirming the @c constexpr surface compiles and runs on the GPU. */
static bool test_device_tensor_view_ops_() {
    constexpr std::size_t rows = 8, columns = 16, count = rows * columns;
    auto host = nk::tensor<float>::try_zeros({rows, columns});
    if (host.empty()) return false;
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            host(row, column) = static_cast<float>((row * columns + column) % 13) * 0.5f;

    auto flat = host.view().flatten<1>();
    float sum_all = 0.0f;
    for (std::size_t i = 0; i < count; ++i) sum_all += static_cast<float>(flat[i]);
    float const host_checksum = 2.0f * sum_all + host(1, 2) + static_cast<float>(count);

    float *device_data = nullptr, *device_out = nullptr;
    nk_cuda_assert_(cudaMalloc(&device_data, count * sizeof(float)));
    nk_cuda_assert_(cudaMalloc(&device_out, sizeof(float)));
    nk_cuda_assert_(cudaMemcpy(device_data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    tensor_view_ops_kernel_<<<1, 1>>>(device_data, rows, columns, device_out);
    nk_cuda_assert_(cudaGetLastError());
    nk_cuda_assert_(cudaDeviceSynchronize());
    float device_checksum = 0.0f;
    nk_cuda_assert_(cudaMemcpy(&device_checksum, device_out, sizeof(float), cudaMemcpyDeviceToHost));
    nk_cuda_assert_(cudaFree(device_data));
    nk_cuda_assert_(cudaFree(device_out));
    float const diff = device_checksum > host_checksum ? device_checksum - host_checksum //
                                                       : host_checksum - device_checksum;
    return diff < 1e-3f * (1.0f + host_checksum);
}

#pragma endregion

#pragma region FP8 and FP6 Conversion Kernels

/** Batch fp32 → fp8 conversion via __nv_cvt_float_to_fp8. */
__global__ void fp32_to_fp8_kernel_(float const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    destination[index] = __nv_cvt_float_to_fp8(source[index], saturate, interpretation);
}

/** Batch fp16 → fp8 conversion via __nv_cvt_halfraw_to_fp8. */
__global__ void fp16_to_fp8_kernel_(__half const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw raw;
    std::memcpy(&raw, &source[index], sizeof raw);
    destination[index] = __nv_cvt_halfraw_to_fp8(raw, saturate, interpretation);
}

/** Batch bf16 → fp8 conversion via __nv_cvt_bfloat16raw_to_fp8. */
__global__ void bf16_to_fp8_kernel_(__nv_bfloat16 const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __nv_bfloat16_raw raw;
    std::memcpy(&raw, &source[index], sizeof raw);
    destination[index] = __nv_cvt_bfloat16raw_to_fp8(raw, saturate, interpretation);
}

/** Batch fp8 → fp32 conversion, routed via halfraw since CUDA has no direct fp8 → fp32. */
__global__ void fp8_to_fp32_kernel_(unsigned char const *source, float *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp8_to_halfraw(source[index], interpretation);
    __half half_value;
    std::memcpy(&half_value, &half_raw, sizeof half_value);
    destination[index] = __half2float(half_value);
}

/** Batch fp8 → fp16 conversion via __nv_cvt_fp8_to_halfraw. */
__global__ void fp8_to_fp16_kernel_(unsigned char const *source, __half *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp8_to_halfraw(source[index], interpretation);
    std::memcpy(&destination[index], &half_raw, sizeof half_raw);
}

/** Batch fp8 → bf16 conversion, routed fp8 → halfraw → fp32 → bf16 as CUDA has no direct path. */
__global__ void fp8_to_bf16_kernel_(unsigned char const *source, __nv_bfloat16 *destination, std::size_t count,
                                    __nv_fp8_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp8_to_halfraw(source[index], interpretation);
    __half half_value;
    std::memcpy(&half_value, &half_raw, sizeof half_value);
    destination[index] = __float2bfloat16(__half2float(half_value));
}

/** Batch fp32 → fp6 conversion via __nv_cvt_float_to_fp6. */
__global__ void fp32_to_fp6_kernel_(float const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    destination[index] = __nv_cvt_float_to_fp6(source[index], interpretation, cudaRoundNearest);
}

/** Batch fp16 → fp6 conversion via __nv_cvt_halfraw_to_fp6. */
__global__ void fp16_to_fp6_kernel_(__half const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw raw;
    std::memcpy(&raw, &source[index], sizeof raw);
    destination[index] = __nv_cvt_halfraw_to_fp6(raw, interpretation, cudaRoundNearest);
}

/** Batch bf16 → fp6 conversion via __nv_cvt_bfloat16raw_to_fp6. */
__global__ void bf16_to_fp6_kernel_(__nv_bfloat16 const *source, unsigned char *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __nv_bfloat16_raw raw;
    std::memcpy(&raw, &source[index], sizeof raw);
    destination[index] = __nv_cvt_bfloat16raw_to_fp6(raw, interpretation, cudaRoundNearest);
}

/** Batch fp6 → fp32 conversion, routed via halfraw. */
__global__ void fp6_to_fp32_kernel_(unsigned char const *source, float *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp6_to_halfraw(source[index], interpretation);
    __half half_value;
    std::memcpy(&half_value, &half_raw, sizeof half_value);
    destination[index] = __half2float(half_value);
}

/** Batch fp6 → fp16 conversion via __nv_cvt_fp6_to_halfraw. */
__global__ void fp6_to_fp16_kernel_(unsigned char const *source, __half *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp6_to_halfraw(source[index], interpretation);
    std::memcpy(&destination[index], &half_raw, sizeof half_raw);
}

/** Batch fp6 → bf16 conversion, routed fp6 → halfraw → fp32 → bf16. */
__global__ void fp6_to_bf16_kernel_(unsigned char const *source, __nv_bfloat16 *destination, std::size_t count,
                                    __nv_fp6_interpretation_t interpretation) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    __half_raw half_raw = __nv_cvt_fp6_to_halfraw(source[index], interpretation);
    __half half_value;
    std::memcpy(&half_value, &half_raw, sizeof half_value);
    destination[index] = __float2bfloat16(__half2float(half_value));
}

#pragma endregion

#pragma region NaN Bit Pattern Predicates

/** True iff the 32-bit IEEE-754 pattern encodes any NaN. */
static bool bits_fp32_nan_(std::uint32_t bits) noexcept {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0;
}

/** True iff the 16-bit IEEE-754 half pattern encodes any NaN. */
static bool bits_fp16_nan_(std::uint16_t bits) noexcept { return (bits & 0x7C00u) == 0x7C00u && (bits & 0x03FFu) != 0; }

/** True iff the 16-bit bfloat16 pattern encodes any NaN. */
static bool bits_bf16_nan_(std::uint16_t bits) noexcept { return (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0; }

/** NaN-tolerant bit equality: any NaN encoding matches any other NaN of the same width, but the "is
 *  a NaN" vs "is a number" bit still matters. */
static bool f32_bits_equal_tolerant_nan_(float left, float right) {
    std::uint32_t left_bits, right_bits;
    std::memcpy(&left_bits, &left, sizeof left_bits);
    std::memcpy(&right_bits, &right, sizeof right_bits);
    if (bits_fp32_nan_(left_bits) && bits_fp32_nan_(right_bits)) return true;
    return left_bits == right_bits;
}

/** NaN-tolerant bit equality for 16-bit half. */
static bool f16_bits_equal_tolerant_nan_(__half left, __half right) {
    std::uint16_t left_bits, right_bits;
    std::memcpy(&left_bits, &left, sizeof left_bits);
    std::memcpy(&right_bits, &right, sizeof right_bits);
    if (bits_fp16_nan_(left_bits) && bits_fp16_nan_(right_bits)) return true;
    return left_bits == right_bits;
}

/** NaN-tolerant bit equality for 16-bit bfloat16. */
static bool bf16_bits_equal_tolerant_nan_(__nv_bfloat16 left, __nv_bfloat16 right) {
    std::uint16_t left_bits, right_bits;
    std::memcpy(&left_bits, &left, sizeof left_bits);
    std::memcpy(&right_bits, &right, sizeof right_bits);
    if (bits_bf16_nan_(left_bits) && bits_bf16_nan_(right_bits)) return true;
    return left_bits == right_bits;
}

/** Masked byte equality; fp6 cares only about the low 6 bits. */
static bool bytes_equal_masked_(unsigned char left, unsigned char right, unsigned mask) noexcept {
    return (left & mask) == (right & mask);
}

/** Print a single mismatch line to stderr for diagnostics. */
static void report_mismatch_(char const *label, std::uint64_t source_bits, unsigned char cuda_output,
                             unsigned char numkong_output, std::uint64_t index) {
    std::fprintf(stderr, "  [%s] input #%" PRIu64 " raw=0x%" PRIx64 " → CUDA=0x%02x NumKong=0x%02x\n", label, index,
                 source_bits, cuda_output, numkong_output);
}

#pragma endregion

#pragma region Conversion Sweeps

/** Batch size: 1M elements per launch keeps device buffers under 16 MB even for 16-byte element
 *  types. */
constexpr std::size_t batch_size_ = 1u << 20;

/** Upload a batch, launch the kernel, download the result. */
template <typename source_type_, typename destination_type_, typename kernel_type_>
static bool run_kernel_batch_(source_type_ const *host_source, destination_type_ *host_destination, std::size_t count,
                              kernel_type_ kernel) {
    source_type_ *device_source = nullptr;
    destination_type_ *device_destination = nullptr;
    nk_cuda_assert_(cudaMalloc(&device_source, count * sizeof(source_type_)));
    nk_cuda_assert_(cudaMalloc(&device_destination, count * sizeof(destination_type_)));
    nk_cuda_assert_(cudaMemcpy(device_source, host_source, count * sizeof(source_type_), cudaMemcpyHostToDevice));
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((count + block.x - 1) / block.x));
    kernel(device_source, device_destination, count, grid, block);
    nk_cuda_assert_(cudaGetLastError());
    nk_cuda_assert_(cudaDeviceSynchronize());
    nk_cuda_assert_(
        cudaMemcpy(host_destination, device_destination, count * sizeof(destination_type_), cudaMemcpyDeviceToHost));
    nk_cuda_assert_(cudaFree(device_source));
    nk_cuda_assert_(cudaFree(device_destination));
    return true;
}

/**
 *  @brief Exhaustive fp32 → fp8 sweep (2^32 inputs) in 1M-element batches.
 *
 *  NumKong uses a single rounding mode, RTNE, and one overflow policy per target type: E4M3
 *  saturates with no Inf encoding, E5M2 produces ±Inf. The CUDA @c saturate dial is derived here
 *  from @p interpretation rather than plumbed through, exposing only NumKong's fixed mode.
 */
template <typename launcher_type_>
static bool sweep_fp32_to_fp8_(char const *label, nk_dtype_t destination_dtype,
                               __nv_fp8_interpretation_t interpretation, launcher_type_ launcher) {
    __nv_saturation_t const saturate = (interpretation == __NV_E4M3) ? __NV_SATFINITE : __NV_NOSAT;
    std::vector<float> host_source(batch_size_);
    std::vector<unsigned char> host_cuda(batch_size_);
    std::vector<unsigned char> host_numkong(batch_size_);
    std::size_t mismatches = 0;
    constexpr std::uint64_t total = 1ull << 32;
    for (std::uint64_t base = 0; base < total; base += batch_size_) {
        std::size_t this_batch = static_cast<std::size_t>(std::min<std::uint64_t>(batch_size_, total - base));
        for (std::size_t i = 0; i < this_batch; ++i) {
            std::uint32_t bits = static_cast<std::uint32_t>(base + i);
            std::memcpy(&host_source[i], &bits, sizeof(float));
        }
        auto kernel = [&](float const *device_source, unsigned char *device_destination, std::size_t n, dim3 grid,
                          dim3 block) {
            launcher(device_source, device_destination, n, interpretation, saturate, grid, block);
        };
        if (!run_kernel_batch_(host_source.data(), host_cuda.data(), this_batch, kernel)) return false;
        nk_cast_serial(host_source.data(), nk_f32_k, this_batch, host_numkong.data(), destination_dtype);
        for (std::size_t i = 0; i < this_batch; ++i) {
            std::uint32_t source_bits;
            std::memcpy(&source_bits, &host_source[i], sizeof(float));
            if (bits_fp32_nan_(source_bits)) continue;
            if (!bytes_equal_masked_(host_cuda[i], host_numkong[i], 0xFF)) {
                if (mismatches < 4) report_mismatch_(label, source_bits, host_cuda[i], host_numkong[i], base + i);
                ++mismatches;
            }
        }
    }
    if (mismatches > 0) std::fprintf(stderr, "  [%s] total mismatches: %zu\n", label, mismatches);
    return mismatches == 0;
}

/** Exhaustive fp32 → fp6 sweep; only low 6 bits of each output byte matter. */
template <typename launcher_type_>
static bool sweep_fp32_to_fp6_(char const *label, nk_dtype_t destination_dtype,
                               __nv_fp6_interpretation_t interpretation, launcher_type_ launcher) {
    std::vector<float> host_source(batch_size_);
    std::vector<unsigned char> host_cuda(batch_size_);
    std::vector<unsigned char> host_numkong(batch_size_);
    std::size_t mismatches = 0;
    constexpr std::uint64_t total = 1ull << 32;
    for (std::uint64_t base = 0; base < total; base += batch_size_) {
        std::size_t this_batch = static_cast<std::size_t>(std::min<std::uint64_t>(batch_size_, total - base));
        for (std::size_t i = 0; i < this_batch; ++i) {
            std::uint32_t bits = static_cast<std::uint32_t>(base + i);
            std::memcpy(&host_source[i], &bits, sizeof(float));
        }
        auto kernel = [&](float const *device_source, unsigned char *device_destination, std::size_t n, dim3 grid,
                          dim3 block) { launcher(device_source, device_destination, n, interpretation, grid, block); };
        if (!run_kernel_batch_(host_source.data(), host_cuda.data(), this_batch, kernel)) return false;
        nk_cast_serial(host_source.data(), nk_f32_k, this_batch, host_numkong.data(), destination_dtype);
        for (std::size_t i = 0; i < this_batch; ++i) {
            std::uint32_t source_bits;
            std::memcpy(&source_bits, &host_source[i], sizeof(float));
            if (bits_fp32_nan_(source_bits)) continue;
            if (!bytes_equal_masked_(host_cuda[i], host_numkong[i], 0x3F)) {
                if (mismatches < 4) report_mismatch_(label, source_bits, host_cuda[i], host_numkong[i], base + i);
                ++mismatches;
            }
        }
    }
    if (mismatches > 0) std::fprintf(stderr, "  [%s] total mismatches: %zu\n", label, mismatches);
    return mismatches == 0;
}

/** Exhaustive fp16-or-bf16 16-bit → fp{6,8} sweep over all 2^16 inputs. */
template <typename source_type_, typename launcher_type_>
static bool sweep_16bit_to_8bit_(char const *label, nk_dtype_t source_dtype, nk_dtype_t destination_dtype,
                                 bool (*is_nan_fn)(std::uint16_t), unsigned byte_mask, launcher_type_ launcher) {
    constexpr std::size_t count = 1u << 16;
    std::vector<source_type_> host_source(count);
    std::vector<unsigned char> host_cuda(count);
    std::vector<unsigned char> host_numkong(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t bits = static_cast<std::uint16_t>(i);
        std::memcpy(&host_source[i], &bits, sizeof(source_type_));
    }
    if (!run_kernel_batch_(host_source.data(), host_cuda.data(), count, launcher)) return false;
    nk_cast_serial(host_source.data(), source_dtype, count, host_numkong.data(), destination_dtype);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t source_bits;
        std::memcpy(&source_bits, &host_source[i], sizeof(source_type_));
        if (is_nan_fn(source_bits)) continue;
        if (!bytes_equal_masked_(host_cuda[i], host_numkong[i], byte_mask)) {
            if (mismatches < 4) report_mismatch_(label, source_bits, host_cuda[i], host_numkong[i], i);
            ++mismatches;
        }
    }
    if (mismatches > 0) std::fprintf(stderr, "  [%s] total mismatches: %zu\n", label, mismatches);
    return mismatches == 0;
}

/** Reverse-direction exhaustive sweep: 2^8 fp8 or 2^6 fp6 inputs, comparing the wider output via a
 *  caller-supplied bit-equality predicate. */
template <typename destination_type_, typename launcher_type_, typename equals_type_>
static bool sweep_small_to_wide_(char const *label, nk_dtype_t source_dtype, nk_dtype_t destination_dtype,
                                 std::size_t domain_size, launcher_type_ launcher, equals_type_ equals) {
    std::vector<unsigned char> host_source(domain_size);
    std::vector<destination_type_> host_cuda(domain_size);
    std::vector<destination_type_> host_numkong(domain_size);
    for (std::size_t i = 0; i < domain_size; ++i) host_source[i] = static_cast<unsigned char>(i);
    if (!run_kernel_batch_(host_source.data(), host_cuda.data(), domain_size, launcher)) return false;
    nk_cast_serial(host_source.data(), source_dtype, domain_size, host_numkong.data(), destination_dtype);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < domain_size; ++i) {
        if (!equals(host_cuda[i], host_numkong[i])) {
            if (mismatches < 4) std::fprintf(stderr, "  [%s] input 0x%02zx mismatch\n", label, i);
            ++mismatches;
        }
    }
    if (mismatches > 0) std::fprintf(stderr, "  [%s] total mismatches: %zu\n", label, mismatches);
    return mismatches == 0;
}

#pragma endregion Conversion Sweeps

#pragma region CUDA Capabilities

/** @c nk_capabilities_cuda_detected reports the families the device's major version runs, and zero
 *  past the last. */
static bool test_cuda_capabilities_(int device) {
    int major = 0, devices_count = 0;
    nk_cuda_assert_(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    nk_cuda_assert_(cudaGetDeviceCount(&devices_count));
    nk_capability_t expected = major >= 8 ? nk_cap_ampere_k : 0;
    if (major == 9) expected |= nk_cap_hopper_k;
    if (major == 10) expected |= nk_cap_blackwell_k;
    if (major == 12) expected |= nk_cap_blackwellrtx_k;
    nk_capability_t const reported = nk_capabilities_cuda_detected(device),
                          absent = nk_capabilities_cuda_detected(devices_count);
    [[maybe_unused]] cudaError_t const cleared = cudaGetLastError(); // the probe past the last device leaves an error
    if (reported != expected)
        std::fprintf(stderr, "  capabilities 0x%llx, expected 0x%llx\n", static_cast<unsigned long long>(reported),
                     static_cast<unsigned long long>(expected));
    return reported == expected && absent == 0;
}

#pragma endregion CUDA Capabilities

#pragma region Cross Kernels

/** Every Ampere entry point, on devices whose families include it. */
static void test_cross_ampere([[maybe_unused]] nk_capability_t available) {
#if NK_TARGET_AMPERE
    error_stats_section_t check(available);
    check.section("Cross Ampere", nk_cap_ampere_k);
    check("dots_packed_f64_ampere", test_dots_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_dots_packed_f64_ampere);
    check("dots_pack_f64_ampere", test_dots_pack_layout<f64_t, cuda_backend_t, nk_dots_pack_size_f64_ampere,
                                                        nk_dots_packed_shape_f64_ampere, nk_dots_pack_f64_ampere>);
    check("dots_contract_f64_ampere",
          test_dots_launch_contract<f64_t, cuda_backend_t, nk_dots_pack_size_f64_ampere, nk_dots_packed_f64_ampere,
                                    nk_dots_symmetric_f64_ampere>);
    check("dots_symmetric_f64_ampere", test_dots_symmetric<f64_t, cuda_backend_t>, nk_dots_symmetric_f64_ampere);
    check("dots_packed_f32_ampere", test_dots_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_dots_packed_f32_ampere);
    check("dots_pack_f32_ampere", test_dots_pack_layout<f32_t, cuda_backend_t, nk_dots_pack_size_f32_ampere,
                                                        nk_dots_packed_shape_f32_ampere, nk_dots_pack_f32_ampere>);
    check("dots_contract_f32_ampere",
          test_dots_launch_contract<f32_t, cuda_backend_t, nk_dots_pack_size_f32_ampere, nk_dots_packed_f32_ampere,
                                    nk_dots_symmetric_f32_ampere>);
    check("dots_symmetric_f32_ampere", test_dots_symmetric<f32_t, cuda_backend_t>, nk_dots_symmetric_f32_ampere);
    check("dots_packed_bf16_ampere", test_dots_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_ampere,
          nk_dots_pack_bf16_ampere, nk_dots_packed_bf16_ampere);
    check("dots_pack_bf16_ampere", test_dots_pack_layout<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_ampere,
                                                         nk_dots_packed_shape_bf16_ampere, nk_dots_pack_bf16_ampere>);
    check("dots_contract_bf16_ampere",
          test_dots_launch_contract<bf16_t, cuda_backend_t, nk_dots_pack_size_bf16_ampere, nk_dots_packed_bf16_ampere,
                                    nk_dots_symmetric_bf16_ampere>);
    check("dots_symmetric_bf16_ampere", test_dots_symmetric<bf16_t, cuda_backend_t>, nk_dots_symmetric_bf16_ampere);
    check("dots_packed_f16_ampere", test_dots_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_dots_packed_f16_ampere);
    check("dots_pack_f16_ampere", test_dots_pack_layout<f16_t, cuda_backend_t, nk_dots_pack_size_f16_ampere,
                                                        nk_dots_packed_shape_f16_ampere, nk_dots_pack_f16_ampere>);
    check("dots_contract_f16_ampere",
          test_dots_launch_contract<f16_t, cuda_backend_t, nk_dots_pack_size_f16_ampere, nk_dots_packed_f16_ampere,
                                    nk_dots_symmetric_f16_ampere>);
    check("dots_symmetric_f16_ampere", test_dots_symmetric<f16_t, cuda_backend_t>, nk_dots_symmetric_f16_ampere);
    check("dots_packed_e5m2_ampere", test_dots_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_ampere,
          nk_dots_pack_e5m2_ampere, nk_dots_packed_e5m2_ampere);
    check("dots_pack_e5m2_ampere", test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_ampere,
                                                         nk_dots_packed_shape_e5m2_ampere, nk_dots_pack_e5m2_ampere>);
    check("dots_contract_e5m2_ampere",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_ampere, nk_dots_packed_e5m2_ampere,
                                    nk_dots_symmetric_e5m2_ampere>);
    check("dots_symmetric_e5m2_ampere", test_dots_symmetric<e5m2_t, cuda_backend_t>, nk_dots_symmetric_e5m2_ampere);
    check("dots_packed_e4m3_ampere", test_dots_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_ampere,
          nk_dots_pack_e4m3_ampere, nk_dots_packed_e4m3_ampere);
    check("dots_pack_e4m3_ampere", test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_ampere,
                                                         nk_dots_packed_shape_e4m3_ampere, nk_dots_pack_e4m3_ampere>);
    check("dots_contract_e4m3_ampere",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_ampere, nk_dots_packed_e4m3_ampere,
                                    nk_dots_symmetric_e4m3_ampere>);
    check("dots_symmetric_e4m3_ampere", test_dots_symmetric<e4m3_t, cuda_backend_t>, nk_dots_symmetric_e4m3_ampere);
    check("dots_packed_e3m2_ampere", test_dots_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_ampere,
          nk_dots_pack_e3m2_ampere, nk_dots_packed_e3m2_ampere);
    check("dots_pack_e3m2_ampere", test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_ampere,
                                                         nk_dots_packed_shape_e3m2_ampere, nk_dots_pack_e3m2_ampere>);
    check("dots_contract_e3m2_ampere",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_ampere, nk_dots_packed_e3m2_ampere,
                                    nk_dots_symmetric_e3m2_ampere>);
    check("dots_symmetric_e3m2_ampere", test_dots_symmetric<e3m2_t, cuda_backend_t>, nk_dots_symmetric_e3m2_ampere);
    check("dots_packed_e2m3_ampere", test_dots_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_ampere,
          nk_dots_pack_e2m3_ampere, nk_dots_packed_e2m3_ampere);
    check("dots_pack_e2m3_ampere", test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_ampere,
                                                         nk_dots_packed_shape_e2m3_ampere, nk_dots_pack_e2m3_ampere>);
    check("dots_contract_e2m3_ampere",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_ampere, nk_dots_packed_e2m3_ampere,
                                    nk_dots_symmetric_e2m3_ampere>);
    check("dots_symmetric_e2m3_ampere", test_dots_symmetric<e2m3_t, cuda_backend_t>, nk_dots_symmetric_e2m3_ampere);
    check("dots_packed_e2m1_ampere", test_dots_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_ampere,
          nk_dots_pack_e2m1_ampere, nk_dots_packed_e2m1_ampere);
    check("dots_pack_e2m1_ampere", test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_ampere,
                                                         nk_dots_packed_shape_e2m1_ampere, nk_dots_pack_e2m1_ampere>);
    check("dots_contract_e2m1_ampere",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_ampere, nk_dots_packed_e2m1_ampere,
                                    nk_dots_symmetric_e2m1_ampere>);
    check("dots_symmetric_e2m1_ampere", test_dots_symmetric<e2m1x2_t, cuda_backend_t>, nk_dots_symmetric_e2m1_ampere);
    check("dots_packed_i8_ampere", test_dots_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_dots_packed_i8_ampere);
    check("dots_pack_i8_ampere", test_dots_pack_layout<i8_t, cuda_backend_t, nk_dots_pack_size_i8_ampere,
                                                       nk_dots_packed_shape_i8_ampere, nk_dots_pack_i8_ampere>);
    check("dots_contract_i8_ampere", test_dots_launch_contract<i8_t, cuda_backend_t, nk_dots_pack_size_i8_ampere,
                                                               nk_dots_packed_i8_ampere, nk_dots_symmetric_i8_ampere>);
    check("dots_symmetric_i8_ampere", test_dots_symmetric<i8_t, cuda_backend_t>, nk_dots_symmetric_i8_ampere);
    check("dots_packed_i4_ampere", test_dots_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_dots_packed_i4_ampere);
    check("dots_pack_i4_ampere", test_dots_pack_layout<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_ampere,
                                                       nk_dots_packed_shape_i4_ampere, nk_dots_pack_i4_ampere>);
    check("dots_contract_i4_ampere", test_dots_launch_contract<i4x2_t, cuda_backend_t, nk_dots_pack_size_i4_ampere,
                                                               nk_dots_packed_i4_ampere, nk_dots_symmetric_i4_ampere>);
    check("dots_symmetric_i4_ampere", test_dots_symmetric<i4x2_t, cuda_backend_t>, nk_dots_symmetric_i4_ampere);
    check("dots_packed_u8_ampere", test_dots_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_dots_packed_u8_ampere);
    check("dots_pack_u8_ampere", test_dots_pack_layout<u8_t, cuda_backend_t, nk_dots_pack_size_u8_ampere,
                                                       nk_dots_packed_shape_u8_ampere, nk_dots_pack_u8_ampere>);
    check("dots_contract_u8_ampere", test_dots_launch_contract<u8_t, cuda_backend_t, nk_dots_pack_size_u8_ampere,
                                                               nk_dots_packed_u8_ampere, nk_dots_symmetric_u8_ampere>);
    check("dots_symmetric_u8_ampere", test_dots_symmetric<u8_t, cuda_backend_t>, nk_dots_symmetric_u8_ampere);
    check("dots_packed_u4_ampere", test_dots_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_dots_packed_u4_ampere);
    check("dots_pack_u4_ampere", test_dots_pack_layout<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_ampere,
                                                       nk_dots_packed_shape_u4_ampere, nk_dots_pack_u4_ampere>);
    check("dots_contract_u4_ampere", test_dots_launch_contract<u4x2_t, cuda_backend_t, nk_dots_pack_size_u4_ampere,
                                                               nk_dots_packed_u4_ampere, nk_dots_symmetric_u4_ampere>);
    check("dots_symmetric_u4_ampere", test_dots_symmetric<u4x2_t, cuda_backend_t>, nk_dots_symmetric_u4_ampere);
    check("angulars_packed_f64_ampere", test_angulars_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_angulars_packed_f64_ampere);
    check("angulars_symmetric_f64_ampere", test_angulars_symmetric<f64_t, cuda_backend_t>,
          nk_angulars_symmetric_f64_ampere);
    check("euclideans_packed_f64_ampere", test_euclideans_packed<f64_t, cuda_backend_t>, nk_dots_pack_size_f64_ampere,
          nk_dots_pack_f64_ampere, nk_euclideans_packed_f64_ampere);
    check("euclideans_symmetric_f64_ampere", test_euclideans_symmetric<f64_t, cuda_backend_t>,
          nk_euclideans_symmetric_f64_ampere);
    check("angulars_packed_f32_ampere", test_angulars_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_angulars_packed_f32_ampere);
    check("angulars_symmetric_f32_ampere", test_angulars_symmetric<f32_t, cuda_backend_t>,
          nk_angulars_symmetric_f32_ampere);
    check("euclideans_packed_f32_ampere", test_euclideans_packed<f32_t, cuda_backend_t>, nk_dots_pack_size_f32_ampere,
          nk_dots_pack_f32_ampere, nk_euclideans_packed_f32_ampere);
    check("euclideans_symmetric_f32_ampere", test_euclideans_symmetric<f32_t, cuda_backend_t>,
          nk_euclideans_symmetric_f32_ampere);
    check("angulars_packed_bf16_ampere", test_angulars_packed<bf16_t, cuda_backend_t>, nk_dots_pack_size_bf16_ampere,
          nk_dots_pack_bf16_ampere, nk_angulars_packed_bf16_ampere);
    check("angulars_symmetric_bf16_ampere", test_angulars_symmetric<bf16_t, cuda_backend_t>,
          nk_angulars_symmetric_bf16_ampere);
    check("euclideans_packed_bf16_ampere", test_euclideans_packed<bf16_t, cuda_backend_t>,
          nk_dots_pack_size_bf16_ampere, nk_dots_pack_bf16_ampere, nk_euclideans_packed_bf16_ampere);
    check("euclideans_symmetric_bf16_ampere", test_euclideans_symmetric<bf16_t, cuda_backend_t>,
          nk_euclideans_symmetric_bf16_ampere);
    check("angulars_packed_f16_ampere", test_angulars_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_angulars_packed_f16_ampere);
    check("angulars_symmetric_f16_ampere", test_angulars_symmetric<f16_t, cuda_backend_t>,
          nk_angulars_symmetric_f16_ampere);
    check("euclideans_packed_f16_ampere", test_euclideans_packed<f16_t, cuda_backend_t>, nk_dots_pack_size_f16_ampere,
          nk_dots_pack_f16_ampere, nk_euclideans_packed_f16_ampere);
    check("euclideans_symmetric_f16_ampere", test_euclideans_symmetric<f16_t, cuda_backend_t>,
          nk_euclideans_symmetric_f16_ampere);
    check("angulars_packed_e5m2_ampere", test_angulars_packed<e5m2_t, cuda_backend_t>, nk_dots_pack_size_e5m2_ampere,
          nk_dots_pack_e5m2_ampere, nk_angulars_packed_e5m2_ampere);
    check("angulars_symmetric_e5m2_ampere", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_ampere);
    check("euclideans_packed_e5m2_ampere", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_ampere, nk_dots_pack_e5m2_ampere, nk_euclideans_packed_e5m2_ampere);
    check("euclideans_symmetric_e5m2_ampere", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_ampere);
    check("angulars_packed_e4m3_ampere", test_angulars_packed<e4m3_t, cuda_backend_t>, nk_dots_pack_size_e4m3_ampere,
          nk_dots_pack_e4m3_ampere, nk_angulars_packed_e4m3_ampere);
    check("angulars_symmetric_e4m3_ampere", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_ampere);
    check("euclideans_packed_e4m3_ampere", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_ampere, nk_dots_pack_e4m3_ampere, nk_euclideans_packed_e4m3_ampere);
    check("euclideans_symmetric_e4m3_ampere", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_ampere);
    check("angulars_packed_e3m2_ampere", test_angulars_packed<e3m2_t, cuda_backend_t>, nk_dots_pack_size_e3m2_ampere,
          nk_dots_pack_e3m2_ampere, nk_angulars_packed_e3m2_ampere);
    check("angulars_symmetric_e3m2_ampere", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_ampere);
    check("euclideans_packed_e3m2_ampere", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_ampere, nk_dots_pack_e3m2_ampere, nk_euclideans_packed_e3m2_ampere);
    check("euclideans_symmetric_e3m2_ampere", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_ampere);
    check("angulars_packed_e2m3_ampere", test_angulars_packed<e2m3_t, cuda_backend_t>, nk_dots_pack_size_e2m3_ampere,
          nk_dots_pack_e2m3_ampere, nk_angulars_packed_e2m3_ampere);
    check("angulars_symmetric_e2m3_ampere", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_ampere);
    check("euclideans_packed_e2m3_ampere", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_ampere, nk_dots_pack_e2m3_ampere, nk_euclideans_packed_e2m3_ampere);
    check("euclideans_symmetric_e2m3_ampere", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_ampere);
    check("angulars_packed_e2m1_ampere", test_angulars_packed<e2m1x2_t, cuda_backend_t>, nk_dots_pack_size_e2m1_ampere,
          nk_dots_pack_e2m1_ampere, nk_angulars_packed_e2m1_ampere);
    check("angulars_symmetric_e2m1_ampere", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_ampere);
    check("euclideans_packed_e2m1_ampere", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_ampere, nk_dots_pack_e2m1_ampere, nk_euclideans_packed_e2m1_ampere);
    check("euclideans_symmetric_e2m1_ampere", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_ampere);
    check("angulars_packed_i8_ampere", test_angulars_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_angulars_packed_i8_ampere);
    check("angulars_symmetric_i8_ampere", test_angulars_symmetric<i8_t, cuda_backend_t>,
          nk_angulars_symmetric_i8_ampere);
    check("euclideans_packed_i8_ampere", test_euclideans_packed<i8_t, cuda_backend_t>, nk_dots_pack_size_i8_ampere,
          nk_dots_pack_i8_ampere, nk_euclideans_packed_i8_ampere);
    check("euclideans_symmetric_i8_ampere", test_euclideans_symmetric<i8_t, cuda_backend_t>,
          nk_euclideans_symmetric_i8_ampere);
    check("angulars_packed_i4_ampere", test_angulars_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_angulars_packed_i4_ampere);
    check("angulars_symmetric_i4_ampere", test_angulars_symmetric<i4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_i4_ampere);
    check("euclideans_packed_i4_ampere", test_euclideans_packed<i4x2_t, cuda_backend_t>, nk_dots_pack_size_i4_ampere,
          nk_dots_pack_i4_ampere, nk_euclideans_packed_i4_ampere);
    check("euclideans_symmetric_i4_ampere", test_euclideans_symmetric<i4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_i4_ampere);
    check("angulars_packed_u8_ampere", test_angulars_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_angulars_packed_u8_ampere);
    check("angulars_symmetric_u8_ampere", test_angulars_symmetric<u8_t, cuda_backend_t>,
          nk_angulars_symmetric_u8_ampere);
    check("euclideans_packed_u8_ampere", test_euclideans_packed<u8_t, cuda_backend_t>, nk_dots_pack_size_u8_ampere,
          nk_dots_pack_u8_ampere, nk_euclideans_packed_u8_ampere);
    check("euclideans_symmetric_u8_ampere", test_euclideans_symmetric<u8_t, cuda_backend_t>,
          nk_euclideans_symmetric_u8_ampere);
    check("angulars_packed_u4_ampere", test_angulars_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_angulars_packed_u4_ampere);
    check("angulars_symmetric_u4_ampere", test_angulars_symmetric<u4x2_t, cuda_backend_t>,
          nk_angulars_symmetric_u4_ampere);
    check("euclideans_packed_u4_ampere", test_euclideans_packed<u4x2_t, cuda_backend_t>, nk_dots_pack_size_u4_ampere,
          nk_dots_pack_u4_ampere, nk_euclideans_packed_u4_ampere);
    check("euclideans_symmetric_u4_ampere", test_euclideans_symmetric<u4x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_u4_ampere);
    check("attention_bidirectional_packed_bf16_ampere",
          test_attention_bidirectional_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere,
          nk_attention_bidirectional_packed_bf16_ampere);
    check("attention_causal_packed_bf16_ampere",
          test_attention_causal_packed<bf16_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_bf16_ampere, nk_attention_pack_bf16_ampere, nk_attention_causal_packed_bf16_ampere);
    check("attention_bidirectional_packed_e4m3_ampere",
          test_attention_bidirectional_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere,
          nk_attention_bidirectional_packed_e4m3_ampere);
    check("attention_causal_packed_e4m3_ampere",
          test_attention_causal_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_11_k>,
          nk_attention_pack_size_e4m3_ampere, nk_attention_pack_e4m3_ampere, nk_attention_causal_packed_e4m3_ampere);
    check("attention_bidirectional_packed_i8_ampere",
          test_attention_bidirectional_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_ampere, nk_attention_pack_i8_ampere, nk_attention_bidirectional_packed_i8_ampere);
    check("attention_causal_packed_i8_ampere",
          test_attention_causal_packed<i8_t, cuda_backend_t, attention_weights_t::bits_8_k>,
          nk_attention_pack_size_i8_ampere, nk_attention_pack_i8_ampere, nk_attention_causal_packed_i8_ampere);
#endif // NK_TARGET_AMPERE
}

/** Every Blackwell RTX entry point, on devices whose families include it. */
static void test_cross_blackwellrtx([[maybe_unused]] nk_capability_t available) {
#if NK_TARGET_BLACKWELLRTX
    error_stats_section_t check(available);
    check.section("Cross Blackwell RTX", nk_cap_blackwellrtx_k);
    check("dots_packed_e5m2_blackwellrtx", test_dots_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_dots_packed_e5m2_blackwellrtx);
    check("dots_pack_e5m2_blackwellrtx",
          test_dots_pack_layout<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwellrtx,
                                nk_dots_packed_shape_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx>);
    check("dots_contract_e5m2_blackwellrtx",
          test_dots_launch_contract<e5m2_t, cuda_backend_t, nk_dots_pack_size_e5m2_blackwellrtx,
                                    nk_dots_packed_e5m2_blackwellrtx, nk_dots_symmetric_e5m2_blackwellrtx>);
    check("dots_symmetric_e5m2_blackwellrtx", test_dots_symmetric<e5m2_t, cuda_backend_t>,
          nk_dots_symmetric_e5m2_blackwellrtx);
    check("dots_packed_e4m3_blackwellrtx", test_dots_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_dots_packed_e4m3_blackwellrtx);
    check("dots_pack_e4m3_blackwellrtx",
          test_dots_pack_layout<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwellrtx,
                                nk_dots_packed_shape_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx>);
    check("dots_contract_e4m3_blackwellrtx",
          test_dots_launch_contract<e4m3_t, cuda_backend_t, nk_dots_pack_size_e4m3_blackwellrtx,
                                    nk_dots_packed_e4m3_blackwellrtx, nk_dots_symmetric_e4m3_blackwellrtx>);
    check("dots_symmetric_e4m3_blackwellrtx", test_dots_symmetric<e4m3_t, cuda_backend_t>,
          nk_dots_symmetric_e4m3_blackwellrtx);
    check("dots_packed_e3m2_blackwellrtx", test_dots_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_dots_packed_e3m2_blackwellrtx);
    check("dots_pack_e3m2_blackwellrtx",
          test_dots_pack_layout<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwellrtx,
                                nk_dots_packed_shape_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx>);
    check("dots_contract_e3m2_blackwellrtx",
          test_dots_launch_contract<e3m2_t, cuda_backend_t, nk_dots_pack_size_e3m2_blackwellrtx,
                                    nk_dots_packed_e3m2_blackwellrtx, nk_dots_symmetric_e3m2_blackwellrtx>);
    check("dots_symmetric_e3m2_blackwellrtx", test_dots_symmetric<e3m2_t, cuda_backend_t>,
          nk_dots_symmetric_e3m2_blackwellrtx);
    check("dots_packed_e2m3_blackwellrtx", test_dots_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_dots_packed_e2m3_blackwellrtx);
    check("dots_pack_e2m3_blackwellrtx",
          test_dots_pack_layout<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwellrtx,
                                nk_dots_packed_shape_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx>);
    check("dots_contract_e2m3_blackwellrtx",
          test_dots_launch_contract<e2m3_t, cuda_backend_t, nk_dots_pack_size_e2m3_blackwellrtx,
                                    nk_dots_packed_e2m3_blackwellrtx, nk_dots_symmetric_e2m3_blackwellrtx>);
    check("dots_symmetric_e2m3_blackwellrtx", test_dots_symmetric<e2m3_t, cuda_backend_t>,
          nk_dots_symmetric_e2m3_blackwellrtx);
    check("dots_packed_e2m1_blackwellrtx", test_dots_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_dots_packed_e2m1_blackwellrtx);
    check("dots_pack_e2m1_blackwellrtx",
          test_dots_pack_layout<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwellrtx,
                                nk_dots_packed_shape_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx>);
    check("dots_contract_e2m1_blackwellrtx",
          test_dots_launch_contract<e2m1x2_t, cuda_backend_t, nk_dots_pack_size_e2m1_blackwellrtx,
                                    nk_dots_packed_e2m1_blackwellrtx, nk_dots_symmetric_e2m1_blackwellrtx>);
    check("dots_symmetric_e2m1_blackwellrtx", test_dots_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_dots_symmetric_e2m1_blackwellrtx);
    check("angulars_packed_e5m2_blackwellrtx", test_angulars_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_angulars_packed_e5m2_blackwellrtx);
    check("angulars_symmetric_e5m2_blackwellrtx", test_angulars_symmetric<e5m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e5m2_blackwellrtx);
    check("euclideans_packed_e5m2_blackwellrtx", test_euclideans_packed<e5m2_t, cuda_backend_t>,
          nk_dots_pack_size_e5m2_blackwellrtx, nk_dots_pack_e5m2_blackwellrtx, nk_euclideans_packed_e5m2_blackwellrtx);
    check("euclideans_symmetric_e5m2_blackwellrtx", test_euclideans_symmetric<e5m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e5m2_blackwellrtx);
    check("angulars_packed_e4m3_blackwellrtx", test_angulars_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_angulars_packed_e4m3_blackwellrtx);
    check("angulars_symmetric_e4m3_blackwellrtx", test_angulars_symmetric<e4m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e4m3_blackwellrtx);
    check("euclideans_packed_e4m3_blackwellrtx", test_euclideans_packed<e4m3_t, cuda_backend_t>,
          nk_dots_pack_size_e4m3_blackwellrtx, nk_dots_pack_e4m3_blackwellrtx, nk_euclideans_packed_e4m3_blackwellrtx);
    check("euclideans_symmetric_e4m3_blackwellrtx", test_euclideans_symmetric<e4m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e4m3_blackwellrtx);
    check("angulars_packed_e3m2_blackwellrtx", test_angulars_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_angulars_packed_e3m2_blackwellrtx);
    check("angulars_symmetric_e3m2_blackwellrtx", test_angulars_symmetric<e3m2_t, cuda_backend_t>,
          nk_angulars_symmetric_e3m2_blackwellrtx);
    check("euclideans_packed_e3m2_blackwellrtx", test_euclideans_packed<e3m2_t, cuda_backend_t>,
          nk_dots_pack_size_e3m2_blackwellrtx, nk_dots_pack_e3m2_blackwellrtx, nk_euclideans_packed_e3m2_blackwellrtx);
    check("euclideans_symmetric_e3m2_blackwellrtx", test_euclideans_symmetric<e3m2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e3m2_blackwellrtx);
    check("angulars_packed_e2m3_blackwellrtx", test_angulars_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_angulars_packed_e2m3_blackwellrtx);
    check("angulars_symmetric_e2m3_blackwellrtx", test_angulars_symmetric<e2m3_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m3_blackwellrtx);
    check("euclideans_packed_e2m3_blackwellrtx", test_euclideans_packed<e2m3_t, cuda_backend_t>,
          nk_dots_pack_size_e2m3_blackwellrtx, nk_dots_pack_e2m3_blackwellrtx, nk_euclideans_packed_e2m3_blackwellrtx);
    check("euclideans_symmetric_e2m3_blackwellrtx", test_euclideans_symmetric<e2m3_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m3_blackwellrtx);
    check("angulars_packed_e2m1_blackwellrtx", test_angulars_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_angulars_packed_e2m1_blackwellrtx);
    check("angulars_symmetric_e2m1_blackwellrtx", test_angulars_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_angulars_symmetric_e2m1_blackwellrtx);
    check("euclideans_packed_e2m1_blackwellrtx", test_euclideans_packed<e2m1x2_t, cuda_backend_t>,
          nk_dots_pack_size_e2m1_blackwellrtx, nk_dots_pack_e2m1_blackwellrtx, nk_euclideans_packed_e2m1_blackwellrtx);
    check("euclideans_symmetric_e2m1_blackwellrtx", test_euclideans_symmetric<e2m1x2_t, cuda_backend_t>,
          nk_euclideans_symmetric_e2m1_blackwellrtx);
    check("attention_bidirectional_packed_e4m3_blackwellrtx",
          test_attention_bidirectional_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwellrtx, nk_attention_pack_e4m3_blackwellrtx,
          nk_attention_bidirectional_packed_e4m3_blackwellrtx);
    check("attention_causal_packed_e4m3_blackwellrtx",
          test_attention_causal_packed<e4m3_t, cuda_backend_t, attention_weights_t::bits_4_k>,
          nk_attention_pack_size_e4m3_blackwellrtx, nk_attention_pack_e4m3_blackwellrtx,
          nk_attention_causal_packed_e4m3_blackwellrtx);
#endif // NK_TARGET_BLACKWELLRTX
}

#pragma endregion Cross Kernels

int main() {
    global_config.load_environment();
    global_config.assert_on_failure = true;
    int device = 0;
    if (cudaSetDevice(device) != cudaSuccess) {
        std::fprintf(stderr, "No CUDA device available\n");
        return 1;
    }
    cudaDeviceProp properties {};
    cudaGetDeviceProperties(&properties, device);

    // Hardware intrinsics: fp8 from SM_89 (Ada), fp6 from SM_100 (Blackwell).
    // Below those thresholds `__nv_cvt_*` falls back to software emulation;
    // the test still produces correct results either way.
    bool const fp8_native = properties.major > 8 || (properties.major == 8 && properties.minor >= 9);
    bool const fp6_native = properties.major >= 10;
    std::printf("NumKong CUDA Interop Test\n");
    std::printf("  GPU:        %s (SM %d.%d)\n", properties.name, properties.major, properties.minor);
    std::printf("  FP8 path:   ");
    print_indicator_(fp8_native);
    std::printf("  %s\n", fp8_native ? "native (Ada+)" : "software emulation");
    std::printf("  FP6 path:   ");
    print_indicator_(fp6_native);
    std::printf("  %s\n", fp6_native ? "native (Blackwell+)" : "software emulation");
    std::printf("\n");

    int passed = 0, failed = 0;
    auto run_ = [&](char const *label, bool ok) {
        std::printf("  %-44s ", label);
        print_indicator_(ok);
        std::printf("\n");
        (ok ? passed : failed) += 1;
    };

    if (global_config.should_run("Tensor memcpy round-trips")) {
        std::printf("Tensor memcpy round-trips\n");
        run_("1D round-trip (cudaMemcpy)", test_memcpy_1d_roundtrip_());
        run_("2D round-trip, padded rows (cudaMemcpy2D)", test_memcpy_2d_padded_rows_());
        run_("2D sub-view extraction (cudaMemcpy2D)", test_memcpy_2d_subview_());
        run_("3D batched round-trip (cudaMemcpy3D)", test_memcpy_3d_batched_());
        run_("device kernel usability (add-one)", test_device_kernel_usability_());
        run_("device tensor_view/vector_view ops", test_device_tensor_view_ops_());
    }

    if (global_config.should_run("FP8 conformance: nk_cast vs __nv_cvt_* (bit-exact)")) {
        std::printf("\nFP8 conformance: nk_cast vs __nv_cvt_* (bit-exact)\n");
        run_("fp32 → e4m3 (exhaustive 2^32)",
             sweep_fp32_to_fp8_( //
                 "fp32 → e4m3", nk_e4m3_k, __NV_E4M3,
                 [](float const *device_source, unsigned char *device_destination, std::size_t count,
                    __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate, dim3 grid, dim3 block) {
                     fp32_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, interpretation,
                                                          saturate);
                 }));
        run_("fp32 → e5m2 (exhaustive 2^32)",
             sweep_fp32_to_fp8_( //
                 "fp32 → e5m2", nk_e5m2_k, __NV_E5M2,
                 [](float const *device_source, unsigned char *device_destination, std::size_t count,
                    __nv_fp8_interpretation_t interpretation, __nv_saturation_t saturate, dim3 grid, dim3 block) {
                     fp32_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, interpretation,
                                                          saturate);
                 }));
        run_("fp16 → e4m3 (exhaustive 2^16)",
             sweep_16bit_to_8bit_<__half>( //
                 "fp16 → e4m3", nk_f16_k, nk_e4m3_k, bits_fp16_nan_, 0xFF,
                 [](__half const *device_source, unsigned char *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp16_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3,
                                                          __NV_SATFINITE);
                 }));
        run_("fp16 → e5m2 (exhaustive 2^16)",
             sweep_16bit_to_8bit_<__half>( //
                 "fp16 → e5m2", nk_f16_k, nk_e5m2_k, bits_fp16_nan_, 0xFF,
                 [](__half const *device_source, unsigned char *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp16_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2,
                                                          __NV_NOSAT);
                 }));
        run_("bf16 → e4m3 (exhaustive 2^16)",
             sweep_16bit_to_8bit_<__nv_bfloat16>( //
                 "bf16 → e4m3", nk_bf16_k, nk_e4m3_k, bits_bf16_nan_, 0xFF,
                 [](__nv_bfloat16 const *device_source, unsigned char *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     bf16_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3,
                                                          __NV_SATFINITE);
                 }));
        run_("bf16 → e5m2 (exhaustive 2^16)",
             sweep_16bit_to_8bit_<__nv_bfloat16>( //
                 "bf16 → e5m2", nk_bf16_k, nk_e5m2_k, bits_bf16_nan_, 0xFF,
                 [](__nv_bfloat16 const *device_source, unsigned char *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     bf16_to_fp8_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2,
                                                          __NV_NOSAT);
                 }));
    }

    if (global_config.should_run("FP8 reverse: fp{32,16,bf16} ← e{4m3,5m2} (bit-exact)")) {
        std::printf("\nFP8 reverse: fp{32,16,bf16} ← e{4m3,5m2} (bit-exact)\n");
        run_("e4m3 → fp32 (exhaustive 2^8)",
             sweep_small_to_wide_<float>( //
                 "e4m3 → fp32", nk_e4m3_k, nk_f32_k, 256u,
                 [](unsigned char const *device_source, float *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp8_to_fp32_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3);
                 },
                 f32_bits_equal_tolerant_nan_));
        run_("e5m2 → fp32 (exhaustive 2^8)",
             sweep_small_to_wide_<float>( //
                 "e5m2 → fp32", nk_e5m2_k, nk_f32_k, 256u,
                 [](unsigned char const *device_source, float *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp8_to_fp32_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2);
                 },
                 f32_bits_equal_tolerant_nan_));
        run_("e4m3 → fp16 (exhaustive 2^8)",
             sweep_small_to_wide_<__half>( //
                 "e4m3 → fp16", nk_e4m3_k, nk_f16_k, 256u,
                 [](unsigned char const *device_source, __half *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp8_to_fp16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3);
                 },
                 f16_bits_equal_tolerant_nan_));
        run_("e5m2 → fp16 (exhaustive 2^8)",
             sweep_small_to_wide_<__half>( //
                 "e5m2 → fp16", nk_e5m2_k, nk_f16_k, 256u,
                 [](unsigned char const *device_source, __half *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp8_to_fp16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2);
                 },
                 f16_bits_equal_tolerant_nan_));
        run_("e4m3 → bf16 (exhaustive 2^8)",
             sweep_small_to_wide_<__nv_bfloat16>( //
                 "e4m3 → bf16", nk_e4m3_k, nk_bf16_k, 256u,
                 [](unsigned char const *device_source, __nv_bfloat16 *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp8_to_bf16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E4M3);
                 },
                 bf16_bits_equal_tolerant_nan_));
        run_("e5m2 → bf16 (exhaustive 2^8)",
             sweep_small_to_wide_<__nv_bfloat16>( //
                 "e5m2 → bf16", nk_e5m2_k, nk_bf16_k, 256u,
                 [](unsigned char const *device_source, __nv_bfloat16 *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp8_to_bf16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E5M2);
                 },
                 bf16_bits_equal_tolerant_nan_));
    }

    if (global_config.should_run("FP6 conformance: nk_cast vs __nv_cvt_* (bit-exact)")) {
        std::printf("\nFP6 conformance: nk_cast vs __nv_cvt_* (bit-exact)\n");
        run_("fp32 → e3m2 (exhaustive 2^32)",
             sweep_fp32_to_fp6_( //
                 "fp32 → e3m2", nk_e3m2_k, __NV_E3M2,
                 [](float const *device_source, unsigned char *device_destination, std::size_t count,
                    __nv_fp6_interpretation_t interpretation, dim3 grid, dim3 block) {
                     fp32_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, interpretation);
                 }));
        run_("fp32 → e2m3 (exhaustive 2^32)",
             sweep_fp32_to_fp6_( //
                 "fp32 → e2m3", nk_e2m3_k, __NV_E2M3,
                 [](float const *device_source, unsigned char *device_destination, std::size_t count,
                    __nv_fp6_interpretation_t interpretation, dim3 grid, dim3 block) {
                     fp32_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, interpretation);
                 }));
        run_("fp16 → e3m2 (exhaustive 2^16)",
             sweep_16bit_to_8bit_<__half>( //
                 "fp16 → e3m2", nk_f16_k, nk_e3m2_k, bits_fp16_nan_, 0x3F,
                 [](__half const *device_source, unsigned char *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp16_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                 }));
        run_("fp16 → e2m3 (exhaustive 2^16)",
             sweep_16bit_to_8bit_<__half>( //
                 "fp16 → e2m3", nk_f16_k, nk_e2m3_k, bits_fp16_nan_, 0x3F,
                 [](__half const *device_source, unsigned char *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp16_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                 }));
        run_("bf16 → e3m2 (exhaustive 2^16)",
             sweep_16bit_to_8bit_<__nv_bfloat16>( //
                 "bf16 → e3m2", nk_bf16_k, nk_e3m2_k, bits_bf16_nan_, 0x3F,
                 [](__nv_bfloat16 const *device_source, unsigned char *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     bf16_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                 }));
        run_("bf16 → e2m3 (exhaustive 2^16)",
             sweep_16bit_to_8bit_<__nv_bfloat16>( //
                 "bf16 → e2m3", nk_bf16_k, nk_e2m3_k, bits_bf16_nan_, 0x3F,
                 [](__nv_bfloat16 const *device_source, unsigned char *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     bf16_to_fp6_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                 }));
    }

    if (global_config.should_run("FP6 reverse: fp{32,16,bf16} ← e{3m2,2m3} (bit-exact)")) {
        std::printf("\nFP6 reverse: fp{32,16,bf16} ← e{3m2,2m3} (bit-exact)\n");
        run_("e3m2 → fp32 (exhaustive 2^6)",
             sweep_small_to_wide_<float>( //
                 "e3m2 → fp32", nk_e3m2_k, nk_f32_k, 64u,
                 [](unsigned char const *device_source, float *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp6_to_fp32_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                 },
                 f32_bits_equal_tolerant_nan_));
        run_("e2m3 → fp32 (exhaustive 2^6)",
             sweep_small_to_wide_<float>( //
                 "e2m3 → fp32", nk_e2m3_k, nk_f32_k, 64u,
                 [](unsigned char const *device_source, float *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp6_to_fp32_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                 },
                 f32_bits_equal_tolerant_nan_));
        run_("e3m2 → fp16 (exhaustive 2^6)",
             sweep_small_to_wide_<__half>( //
                 "e3m2 → fp16", nk_e3m2_k, nk_f16_k, 64u,
                 [](unsigned char const *device_source, __half *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp6_to_fp16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                 },
                 f16_bits_equal_tolerant_nan_));
        run_("e2m3 → fp16 (exhaustive 2^6)",
             sweep_small_to_wide_<__half>( //
                 "e2m3 → fp16", nk_e2m3_k, nk_f16_k, 64u,
                 [](unsigned char const *device_source, __half *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp6_to_fp16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                 },
                 f16_bits_equal_tolerant_nan_));
        run_("e3m2 → bf16 (exhaustive 2^6)",
             sweep_small_to_wide_<__nv_bfloat16>( //
                 "e3m2 → bf16", nk_e3m2_k, nk_bf16_k, 64u,
                 [](unsigned char const *device_source, __nv_bfloat16 *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp6_to_bf16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E3M2);
                 },
                 bf16_bits_equal_tolerant_nan_));
        run_("e2m3 → bf16 (exhaustive 2^6)",
             sweep_small_to_wide_<__nv_bfloat16>( //
                 "e2m3 → bf16", nk_e2m3_k, nk_bf16_k, 64u,
                 [](unsigned char const *device_source, __nv_bfloat16 *device_destination, std::size_t count, dim3 grid,
                    dim3 block) {
                     fp6_to_bf16_kernel_<<<grid, block>>>(device_source, device_destination, count, __NV_E2M3);
                 },
                 bf16_bits_equal_tolerant_nan_));
    }

    if (global_config.should_run("CUDA capabilities")) {
        std::printf("\nCUDA capabilities\n");
        run_("nk_capabilities_cuda_detected families", test_cuda_capabilities_(device));
    }

    nk_capability_t const available = nk_capabilities_cuda_available(device);
    test_cross_ampere(available);
    test_cross_blackwellrtx(available);

    passed += static_cast<int>(global_config.kernel_count - global_config.failure_count);
    failed += static_cast<int>(global_config.failure_count);
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
