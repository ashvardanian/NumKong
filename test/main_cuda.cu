/**
 *  @file test/main_cuda.cu
 *  @author Ash Vardanian
 *  @date April 15, 2026
 *  @brief CUDA test: the capability report, tensor memcpy round-trips, fp6/fp8 cast conformance, the
 *      CUDA cross-kernel sections of `cross_cuda.cu`, and the dispatching entry points.
 *
 *  The memcpy sections drive @c cudaMemcpy, pitched @c cudaMemcpy2D and @c cudaMemcpy3D round-trips
 *  through @c nk::tensor_view and run a trivial add-one kernel to prove device-side readability, while
 *  the conformance sections compare @c nk_cast on the CPU bit for bit against CUDA's @c __nv_cvt_*
 *  intrinsics on the GPU, covering every fp32, fp16 and bf16 input against every e4m3, e5m2, e3m2 and
 *  e2m3 variant. `NUMKONG_FILTER=<regex>` keeps only the sections and kernels whose names match.
 *
 *  The test builds and runs on any Turing-or-newer GPU. CUDA's __nv_cvt_* converters fall back to
 *  software emulation below SM_89 for fp8 and below SM_100 for fp6. NUMKONG_CUDA_ARCHITECTURES picks
 *  the codes compiled. Without a device the test exits zero, so a build host with no GPU still passes,
 *  having compiled every kernel for each architecture.
 */
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `stderr`
#include <cstring> // `std::memcpy`

#include <random> // `std::mt19937`
#include <vector> // `std::vector`

#include "harness.hpp" // `error_stats_section_t`
#include "harness.cuh" // `cuda_backend_t`, `device_capabilities`
#include "each.hpp"    // `test_sum`, `test_rmsnorm`, `test_swiglu`

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using namespace ashvardanian::numkong::test;

char const *volatile nk::test::nk_test_current_kernel_ = nullptr;

namespace ashvardanian::numkong::test {

/** Every CUDA capability this build compiled, from @c test/cross_cuda.cu. */
void test_cross_cuda(error_stats_section_t &check);

} // namespace ashvardanian::numkong::test

#pragma region CUDA Error Handling

/** Return true on success; on failure, print a diagnostic and return false. */
static bool cuda_check_(cudaError_t error, char const *expression, char const *file, int line) {
    if (error == cudaSuccess) return true;
    fmt::println(stderr, "CUDA error {} at {}:{}: {}", expression, file, line, cudaGetErrorString(error));
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
    auto [host_source, source_status] = nk::tensor<float>::zeros({count});
    auto [host_destination, destination_status] = nk::tensor<float>::zeros({count});
    if (nk::failed(source_status) || nk::failed(destination_status)) return false;

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

    auto [host_source, source_status] = nk::tensor<float>::zeros({rows, row_elements_padded});
    auto [host_destination, destination_status] = nk::tensor<float>::zeros({rows, row_elements_padded});
    if (nk::failed(source_status) || nk::failed(destination_status)) return false;

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
    auto [host_source, source_status] = nk::tensor<float>::zeros({batches, rows, columns});
    auto [host_destination, destination_status] = nk::tensor<float>::zeros({batches, rows, columns});
    if (nk::failed(source_status) || nk::failed(destination_status)) return false;

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

    auto [parent, parent_status] = nk::tensor<float>::zeros({parent_rows, parent_columns});
    if (nk::failed(parent_status)) return false;
    for (std::size_t row = 0; row < parent_rows; ++row)
        for (std::size_t column = 0; column < parent_columns; ++column)
            parent(row, column) = static_cast<float>(row * parent_columns + column);

    float const *sub_source_pointer = &parent(row_start, column_start);
    std::size_t const parent_row_pitch = static_cast<std::size_t>(parent.stride_bytes(0));

    auto [host_back, host_back_status] = nk::tensor<float>::zeros({sub_rows, sub_columns});
    if (nk::failed(host_back_status)) return false;

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
    auto [host_source, source_status] = nk::tensor<float>::zeros({rows, columns});
    auto [host_destination, destination_status] = nk::tensor<float>::zeros({rows, columns});
    if (nk::failed(source_status) || nk::failed(destination_status)) return false;
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            host_source(row, column) = static_cast<float>((row * columns + column) % 17) * 0.25f;

    float *device_pointer = nullptr;
    std::size_t device_pitch = 0;
    nk_cuda_assert_(cudaMallocPitch(&device_pointer, &device_pitch, columns * sizeof(float), rows));
    nk_cuda_assert_(cudaMemcpy2D(device_pointer, device_pitch, host_source.data(), host_source.stride_bytes(0),
                                 columns * sizeof(float), rows, cudaMemcpyHostToDevice));

    dim3 block(16, 8);
    dim3 grid(nk::divide_round_up(columns, block.x), nk::divide_round_up(rows, block.y));
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
    auto [host, host_status] = nk::tensor<float>::zeros({rows, columns});
    if (nk::failed(host_status)) return false;
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
    fmt::println(stderr, "  [{}] input #{} raw=0x{:x} → CUDA=0x{:02x} NumKong=0x{:02x}", label, index, source_bits,
                 cuda_output, numkong_output);
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
    dim3 grid(static_cast<unsigned>(nk::divide_round_up(count, block.x)));
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
        if (nk_cast_serial(host_source.data(), nk_f32_k, this_batch, host_numkong.data(), destination_dtype, nullptr) !=
            nk_success_k)
            return false;
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
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
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
        if (nk_cast_serial(host_source.data(), nk_f32_k, this_batch, host_numkong.data(), destination_dtype, nullptr) !=
            nk_success_k)
            return false;
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
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
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
    if (nk_cast_serial(host_source.data(), source_dtype, count, host_numkong.data(), destination_dtype, nullptr) !=
        nk_success_k)
        return false;
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
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
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
    if (nk_cast_serial(host_source.data(), source_dtype, domain_size, host_numkong.data(), destination_dtype,
                       nullptr) != nk_success_k)
        return false;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < domain_size; ++i) {
        if (!equals(host_cuda[i], host_numkong[i])) {
            if (mismatches < 4) fmt::println(stderr, "  [{}] input 0x{:02x} mismatch", label, i);
            ++mismatches;
        }
    }
    if (mismatches > 0) fmt::println(stderr, "  [{}] total mismatches: {}", label, mismatches);
    return mismatches == 0;
}

#pragma endregion Conversion Sweeps

#pragma region CUDA Capabilities

/** Element-wise operations on the CUDA baseline, over @c test/each.hpp. */
static void test_each_cuda(error_stats_section_t &check) {
    check.section("Elementwise Operations CUDA", nk_cap_cuda_k);
    check("each_swiglu_f32_cuda", test_swiglu<f32_t, cuda_backend_t>, nk_each_swiglu_f32_cuda);
    check("each_swiglu_bf16_cuda", test_swiglu<bf16_t, cuda_backend_t>, nk_each_swiglu_bf16_cuda);
    check("each_swiglu_e4m3_cuda", test_swiglu<e4m3_t, cuda_backend_t>, nk_each_swiglu_e4m3_cuda);
    check("each_sum_f32_cuda", test_sum<f32_t, cuda_backend_t>, nk_each_sum_f32_cuda);
    check("each_sum_f16_cuda", test_sum<f16_t, cuda_backend_t>, nk_each_sum_f16_cuda);
    check("each_sum_bf16_cuda", test_sum<bf16_t, cuda_backend_t>, nk_each_sum_bf16_cuda);
    check("each_rmsnorm_f32_cuda", test_rmsnorm<f32_t, cuda_backend_t>, nk_each_rmsnorm_f32_cuda);
    check("each_rmsnorm_bf16_cuda", test_rmsnorm<bf16_t, cuda_backend_t>, nk_each_rmsnorm_bf16_cuda);
    check("each_rmsnorm_e4m3_cuda", test_rmsnorm<e4m3_t, cuda_backend_t>, nk_each_rmsnorm_e4m3_cuda);
}

/** @c nk_cuda_capabilities_detected reports the CUDA baseline on every device, and the tensor-core
 *  capabilities the device's compute capability runs. */
static error_stats_t test_cuda_capabilities(settings_t const &, int device) {
    error_stats_t stats(comparison_family_t::exact_k);
    int major = 0, minor = 0;
    stats.expect(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
                     cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess,
                 "the compute capability was unreadable");
    nk_capability_t expected = nk_cap_cuda_k;
    if (major >= 8) expected |= nk_cap_ampere_k;
    if (major * 10 + minor >= 89) expected |= nk_cap_ada_k;
    if (major == 9) expected |= nk_cap_hopper_k;
    if (major == 10) expected |= nk_cap_blackwell_k;
    if (major == 12) expected |= nk_cap_blackwellrtx_k;
    nk_capability_t reported = 0;
    stats.expect(nk_cuda_capabilities_detected(static_cast<nk_size_t>(device), &reported));
    stats.expect(reported == expected, "the capabilities differ from the compute capability's");
    return stats;
}

#pragma endregion CUDA Capabilities

int main(int, char **argv) {
    environment_t const env {read_settings(argv[0]), probe_machine()};
    print(env.machine);
    print(env.settings);
    auto const devices = nk::device_t::count(nk::device_kind_t::cuda_k);
    if (!devices) {
        fmt::println("- CUDA: no device");
        return 0;
    }
    nk_capability_t const capabilities = device_capabilities<cuda_runtime_t>();
    char names[NUMKONG_CAPABILITIES_NAME_CAPACITY];
    nk_capabilities_name(capabilities, names, sizeof(names));
    fmt::println("- CUDA: {} devices, the first running {}", devices.value, names);

    error_stats_section_t check(env.settings, capabilities | nk_cap_serial_k);
    check.section("CUDA capabilities", nk_cap_serial_k); // the report itself is under test, so never gate on it
    check("gpu_capabilities_cuda", test_cuda_capabilities, 0);

    // Hardware intrinsics: fp8 from SM_89 (Ada), fp6 from SM_100 (Blackwell).
    // Below those thresholds `__nv_cvt_*` falls back to software emulation;
    // the test still produces correct results either way.
    bool const fp8_native = (capabilities & nk_cap_ada_k) != 0;
    bool const fp6_native = (capabilities & (nk_cap_blackwell_k | nk_cap_blackwellrtx_k)) != 0;
    fmt::println("\nFP8 path: {}", fp8_native ? "native (Ada+)" : "software emulation");
    fmt::println("FP6 path: {}", fp6_native ? "native (Blackwell+)" : "software emulation");

    int passed = 0, failed = 0;
    auto run_ = [&](char const *label, bool ok) {
        fmt::println("  {:<44} {}", label, ok ? "ok" : "FAILED");
        (ok ? passed : failed) += 1;
    };

    if (env.settings.selects("Tensor memcpy round-trips")) {
        fmt::println("\nTensor memcpy round-trips");
        run_("1D round-trip (cudaMemcpy)", test_memcpy_1d_roundtrip_());
        run_("2D round-trip, padded rows (cudaMemcpy2D)", test_memcpy_2d_padded_rows_());
        run_("2D sub-view extraction (cudaMemcpy2D)", test_memcpy_2d_subview_());
        run_("3D batched round-trip (cudaMemcpy3D)", test_memcpy_3d_batched_());
        run_("device kernel usability (add-one)", test_device_kernel_usability_());
        run_("device tensor_view/vector_view ops", test_device_tensor_view_ops_());
    }

    if (env.settings.selects("FP8 conformance: nk_cast vs __nv_cvt_* (bit-exact)")) {
        fmt::println("\nFP8 conformance: nk_cast vs __nv_cvt_* (bit-exact)");
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

    if (env.settings.selects("FP8 reverse: fp{32,16,bf16} ← e{4m3,5m2} (bit-exact)")) {
        fmt::println("\nFP8 reverse: fp{{32,16,bf16}} ← e{{4m3,5m2}} (bit-exact)");
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

    if (env.settings.selects("FP6 conformance: nk_cast vs __nv_cvt_* (bit-exact)")) {
        fmt::println("\nFP6 conformance: nk_cast vs __nv_cvt_* (bit-exact)");
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

    if (env.settings.selects("FP6 reverse: fp{32,16,bf16} ← e{3m2,2m3} (bit-exact)")) {
        fmt::println("\nFP6 reverse: fp{{32,16,bf16}} ← e{{3m2,2m3}} (bit-exact)");
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

    test_each_cuda(check);
    test_cross_cuda(check);

    passed += static_cast<int>(check.kernel_count - check.failure_count);
    failed += static_cast<int>(check.failure_count);
    fmt::println("\n{} passed, {} failed", passed, failed);
    return failed != 0 && env.settings.on_failure == on_failure_t::exit_k ? 1 : 0;
}
