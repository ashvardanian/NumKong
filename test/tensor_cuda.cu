/**
 *  @file test/tensor_cuda.cu
 *  @author Ash Vardanian
 *  @date April 15, 2026
 *  @brief CUDA tensor transfers and device-side views.
 */
#include "harness.hpp"

namespace ashvardanian::numkong::test {

#if NUMKONG_ARCH_CUDA_

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

    nk_allocator_t policy;
    if (nk_allocator_init_unified_cuda(&policy) != nk_success_k) return false;
    nk::allocator<char> const allocator(policy);
    using bytes_t = nk::vector<char, nk::allocator<char>>;
    auto [device_data, data_status] = bytes_t::uninitialized(count * sizeof(float), allocator);
    auto [device_out, out_status] = bytes_t::uninitialized(sizeof(float), allocator);
    if (nk::failed(data_status) || nk::failed(out_status)) return false;
    std::memcpy(device_data.raw_values_data(), host.data(), count * sizeof(float));
    auto *const out = reinterpret_cast<float *>(device_out.raw_values_data());
    tensor_view_ops_kernel_<<<1, 1>>>(reinterpret_cast<float const *>(device_data.raw_values_data()), rows, columns,
                                      out);
    nk_cuda_assert_(cudaGetLastError());
    if (nk_stream_synchronize_cuda(nullptr) != nk_success_k) return false;
    float const device_checksum = *out;
    float const diff = device_checksum > host_checksum ? device_checksum - host_checksum //
                                                       : host_checksum - device_checksum;
    return diff < 1e-3f * (1.0f + host_checksum);
}

void test_tensor_cuda(error_stats_section_t &check) {
    check.section("CUDA reference checks", nk_cap_cuda_k);

    cuda_device_scope_t device(check.settings.device.ordinal());
    if (!cuda_check_(device.status, "select CUDA reference device", __FILE__, __LINE__)) {
        ++check.failure_count;
        return;
    }

    check("Tensor memcpy round-trips", [&](settings_t const &) {
        error_stats_t stats(comparison_family_t::exact_k);
        stats.expect(test_memcpy_1d_roundtrip_(), "1D round-trip (cudaMemcpy)");
        stats.expect(test_memcpy_2d_padded_rows_(), "2D round-trip, padded rows (cudaMemcpy2D)");
        stats.expect(test_memcpy_2d_subview_(), "2D sub-view extraction (cudaMemcpy2D)");
        stats.expect(test_memcpy_3d_batched_(), "3D batched round-trip (cudaMemcpy3D)");
        stats.expect(test_device_kernel_usability_(), "device kernel usability (add-one)");
        stats.expect(test_device_tensor_view_ops_(), "device tensor_view/vector_view ops");

        return stats;
    });
}

#else // !NUMKONG_ARCH_CUDA_
void test_tensor_cuda(error_stats_section_t &) {}

#endif // NUMKONG_ARCH_CUDA_

} // namespace ashvardanian::numkong::test
