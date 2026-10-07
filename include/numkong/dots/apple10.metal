/**
 *  @file include/numkong/dots/apple10.metal
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Batched Dot Products for Apple GPUs of Metal family 10, on the Neural Accelerators.
 *
 *  @sa include/numkong/dots/apple10.h, which embeds and launches this source after `metal.metal`
 *  @sa include/numkong/dots/ampere.cuh, the CUDA sibling
 *
 *  @c matmul2d exposes the M5 GPU-core Neural Accelerators.
 *  Four simdgroups own a @b [64,64] output tile - the shape that reaches peak within 32 KB of
 *  threadgroup memory - accumulating in registers through a cooperative tensor, so a canary-filled
 *  C is never read and a symmetric tile stores only on and above the diagonal. Operand extents
 *  bound every edge tile, so no row reads past its depth and no store lands in a stride's padding.
 *
 *  The 8-, 6- and 4-bit floats are widened to @c half on the way in, as @c ampere widens them: each
 *  step of 64 dimensions is decoded into staged @b [64,64] tiles, zero-padded past every edge, and
 *  multiplied with accumulation. @c matmul2d takes FP8 and FP4 natively only from Metal 4.1, which
 *  macOS 26 cannot load, and every one of these values is exact in @c half, so the widened products
 *  match what the native formats would give.
 */
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

/** Side of the output tile one threadgroup owns, and the depth one widened step stages. */
constant uint nk_cross_tile_apple10_k = 64;
static_assert(nk_cross_tile_apple10_k == nk_cross_tile_metal_k, "unscaled kernels share the launch tile size");

/** Stores one accumulated tile: through the bounded tensor store off the diagonal, and cell by cell
 *  on and above it, where the symmetric kernels' tiles cross it. */
template <nk_cross_metric_metal_t metric_ = nk_cross_dot_metal_k, typename tile_type_, typename result_type_>
void nk_cross_store_apple10_(thread tile_type_ &tile, device result_type_ *c,
                             constant nk_cross_arguments_metal_t &arguments, uint first_row, uint first_column,
                             threadgroup float const (*norms)[nk_cross_tile_metal_k] = nullptr) {
    bool const on_diagonal = arguments.upper_triangle != 0 && first_row + nk_cross_tile_apple10_k > first_column;
    if constexpr (metric_ == nk_cross_dot_metal_k) {
        if (!on_diagonal) {
            tensor<device result_type_, dextents<int32_t, 2>, tensor_inline> c_cells(
                c, dextents<int32_t, 2>(arguments.column_count, arguments.row_end),
                array<int32_t, 2> {1, (int32_t)arguments.c_stride});
            auto c_tile = c_cells.slice(first_column, first_row);
            tile.store(c_tile);
            return;
        }
    }
#pragma unroll
    for (uint16_t index = 0; index < tile.get_capacity(); ++index) {
        if (!tile.is_valid_element(index)) continue;
        auto const cell = tile.get_multidimensional_index(index);
        uint const column = first_column + (uint)cell[0], row = first_row + (uint)cell[1];
        if (row >= arguments.row_end || column >= arguments.column_count || (arguments.upper_triangle && column < row))
            continue;
        result_type_ value = result_type_(tile[index]);
        if constexpr (metric_ != nk_cross_dot_metal_k)
            value = arguments.upper_triangle && row == column
                        ? 0
                        : nk_cross_distance_metal_<metric_>(float(value), norms[0][row - first_row],
                                                            norms[1][column - first_column]);
        c[row * arguments.c_stride + column] = value;
    }
}

/**
 *  @brief One @b [64,64] tile of C = A × Bᵀ, with B given row-major as @b [columns,depth].
 *  @tparam input_type_ The operands' element type, as @c matmul2d names it.
 *  @tparam result_type_ The accumulator the cooperative tensor holds; stored bit for bit.
 */
template <typename input_type_, typename result_type_, nk_cross_metric_metal_t metric_ = nk_cross_dot_metal_k,
          typename output_type_ = result_type_>
void nk_cross_tile_apple10_(device input_type_ const *a, device input_type_ const *b, device output_type_ *c,
                            constant nk_cross_arguments_metal_t &arguments, uint2 group,
                            threadgroup float const (*norms)[nk_cross_tile_metal_k] = nullptr) {
    uint const first_row = arguments.row_start + group.y * nk_cross_tile_apple10_k;
    uint const first_column = group.x * nk_cross_tile_apple10_k;
    if (arguments.upper_triangle && first_column + nk_cross_tile_apple10_k <= first_row) return; // below the diagonal

    // `matmul2d` rejects `const` element types, so the operands shed it; neither is ever written.
    tensor<device input_type_, dextents<int32_t, 2>, tensor_inline> a_rows(
        const_cast<device input_type_ *>(a), dextents<int32_t, 2>(arguments.depth, arguments.row_end),
        array<int32_t, 2> {1, (int32_t)(arguments.a_stride / sizeof(input_type_))});
    tensor<device input_type_, dextents<int32_t, 2>, tensor_inline> b_columns(
        const_cast<device input_type_ *>(b), dextents<int32_t, 2>(arguments.depth, arguments.column_count),
        array<int32_t, 2> {1, (int32_t)(arguments.b_stride / sizeof(input_type_))});
    auto a_tile = a_rows.slice(0, first_row);
    auto b_tile = b_columns.slice(0, first_column);

    constexpr auto descriptor = matmul2d_descriptor(nk_cross_tile_apple10_k, nk_cross_tile_apple10_k,
                                                    static_cast<int>(dynamic_extent), /*transpose_left=*/false,
                                                    /*transpose_right=*/true, /*relaxed_precision=*/false);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    auto tile =
        multiply.template get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), result_type_>();
#pragma unroll
    for (uint16_t index = 0; index < tile.get_capacity(); ++index)
        if (tile.is_valid_element(index)) tile[index] = 0;
    multiply.run(a_tile, b_tile, tile);
    nk_cross_store_apple10_<metric_>(tile, c, arguments, first_row, first_column, norms);
}

/**
 *  @brief One @b [64,64] tile of C = A × Bᵀ over a format @c matmul2d cannot read, widened to
 *      @c half 64 dimensions at a time through staged threadgroup tiles, accumulated in @c float.
 *
 *  Staging zero-pads past the last row, column and dimension, so every step multiplies full static
 *  tiles and @c matmul2d checks no bounds. Each thread widens aligned 4-byte words, a row's last
 *  one a byte at a time so no load passes a row's bytes, and consecutive threads read consecutive
 *  words of one row.
 */
template <typename dtype_, nk_cross_metric_metal_t metric_ = nk_cross_dot_metal_k>
void nk_cross_widened_tile_apple10_(device uchar const *a, device uchar const *b, device float *c,
                                    constant nk_cross_arguments_metal_t &arguments, uint2 group, uint thread_index,
                                    threadgroup half (*a_stage)[nk_cross_tile_apple10_k],
                                    threadgroup half (*b_stage)[nk_cross_tile_apple10_k],
                                    threadgroup float const (*norms)[nk_cross_tile_metal_k] = nullptr) {
    constexpr uint side = nk_cross_tile_apple10_k, threads = 4 * 32;
    constexpr uint per_value = dtype_::dimensions_per_value, words_per_line = side / 4 / per_value;
    uint const first_row = arguments.row_start + group.y * side;
    uint const first_column = group.x * side;
    if (arguments.upper_triangle && first_column + side <= first_row) return; // below the diagonal, group-uniform

    constexpr auto descriptor = matmul2d_descriptor(side, side, side, /*transpose_left=*/false,
                                                    /*transpose_right=*/true, /*relaxed_precision=*/false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    using stage_t = tensor<threadgroup half, extents<int32_t, side, side>, tensor_inline>;
    stage_t a_tile(&a_stage[0][0], extents<int32_t, side, side>()),
        b_tile(&b_stage[0][0], extents<int32_t, side, side>());
    auto tile = multiply.template get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), float>();
#pragma unroll
    for (uint16_t index = 0; index < tile.get_capacity(); ++index)
        if (tile.is_valid_element(index)) tile[index] = 0;

    for (uint depth_start = 0; depth_start < arguments.depth; depth_start += side) {
        for (uint word = thread_index; word < side * words_per_line; word += threads) {
            uint const line = word / words_per_line, word_in_line = word % words_per_line;
            uint const first_dimension = depth_start + word_in_line * 4 * per_value;
            uint const row = first_row + line, column = first_column + line;
            ulong const byte = (depth_start / per_value) + word_in_line * 4;
            uint const left = first_dimension < arguments.depth ? arguments.depth - first_dimension : 0;
            bool const whole = left >= 4 * per_value;
            device uchar const *a_word = a + row * arguments.a_stride + byte;
            device uchar const *b_word = b + column * arguments.b_stride + byte;
            uint4 a_words = uint4(0), b_words = uint4(0);
            if (row < arguments.row_end)
                a_words = whole ? uint4(*(device uint const *)a_word, 0, 0, 0) : nk_tail_load_<dtype_>(a_word, left);
            if (column < arguments.column_count)
                b_words = whole ? uint4(*(device uint const *)b_word, 0, 0, 0) : nk_tail_load_<dtype_>(b_word, left);
            for (uint part = 0; part != per_value; ++part) {
                uint const dimension = first_dimension + part * 4 - depth_start;
                *(threadgroup half4 *)(&a_stage[line][dimension]) = dtype_::widen(
                    nk_chunk_codes_<dtype_>(a_words, part));
                *(threadgroup half4 *)(&b_stage[line][dimension]) = dtype_::widen(
                    nk_chunk_codes_<dtype_>(b_words, part));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        multiply.run(a_tile, b_tile, tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    nk_cross_store_apple10_<metric_>(tile, c, arguments, first_row, first_column, norms);
}

#if __HAVE_INT4B_FORMAT_TYPE__
using nk_i4_operand_apple10_t = int4b_format;
using nk_u4_operand_apple10_t = uint4b_format;
using nk_i4_storage_apple10_t = uchar;
using nk_u4_storage_apple10_t = uchar;
constant uint nk_cross_int4_pitch_apple10_k = 128;
#else
using nk_i4_operand_apple10_t = int8_t;
using nk_u4_operand_apple10_t = uchar;
using nk_i4_storage_apple10_t = int8_t;
using nk_u4_storage_apple10_t = uchar;
constant uint nk_cross_int4_pitch_apple10_k = 64;
#endif

template <typename dtype_, typename input_type_, typename packed_type_, nk_cross_metric_metal_t metric_, uint side_,
          typename output_type_, typename storage_type_>
void nk_cross_int4_tile_apple10_(device uchar const *a, device uchar const *b, device output_type_ *c,
                                 constant nk_cross_arguments_metal_t &arguments, uint2 group, uint thread_index,
                                 threadgroup input_type_ (*a_stage)[64],
                                 threadgroup storage_type_ (*b_stage)[nk_cross_int4_pitch_apple10_k],
                                 threadgroup float const (*norms)[side_] = nullptr) {
    constexpr uint side = side_, step = 64, threads = 128;
    constexpr bool native_packing = !is_same_v<input_type_, packed_type_>;
    uint const first_row = arguments.row_start + group.y * side, first_column = group.x * side;
    if (arguments.upper_triangle && first_column + side <= first_row) return;
    constexpr auto descriptor = matmul2d_descriptor(side, side, step, /*transpose_left=*/false,
                                                    /*transpose_right=*/true, /*relaxed_precision=*/false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    tensor<threadgroup input_type_, extents<int32_t, step, side>, tensor_inline> a_tile(&a_stage[0][0],
                                                                                        extents<int32_t, step, side>());
    using packed_tensor_t = tensor<threadgroup packed_type_, extents<int32_t, step, side>, tensor_inline>;
    packed_tensor_t b_tile(
        &b_stage[0][0], extents<int32_t, step, side>(),
        array<int32_t, 2> {1, native_packing ? 2 * nk_cross_int4_pitch_apple10_k : nk_cross_int4_pitch_apple10_k});
    auto tile = multiply.template get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), int>();
    for (uint16_t index = 0; index < tile.get_capacity(); ++index)
        if (tile.is_valid_element(index)) tile[index] = 0;
    for (uint depth_start = 0; depth_start < arguments.depth; depth_start += step) {
        for (uint cell = thread_index; cell < side * step; cell += threads) {
            uint const line = cell / step, offset = cell % step, dimension = depth_start + (offset ^ 1);
            uint const row = first_row + line;
            // Metal reads the low nibble first, so permute A's adjacent dimensions to match B.
            a_stage[line][offset] = row < arguments.row_end && dimension < arguments.depth
                                        ? input_type_(dtype_::load(a + row * arguments.a_stride, dimension))
                                        : input_type_(0);
            if constexpr (!native_packing) {
                uint const column = first_column + line;
                b_stage[line][offset] = column < arguments.column_count && dimension < arguments.depth
                                            ? input_type_(dtype_::load(b + column * arguments.b_stride, dimension))
                                            : input_type_(0);
            }
        }
        if constexpr (native_packing)
            for (uint cell = thread_index; cell < side * step / 2; cell += threads) {
                uint const line = cell / (step / 2), offset = cell % (step / 2), byte = depth_start / 2 + offset;
                uint const column = first_column + line;
                b_stage[line][offset] = column < arguments.column_count && byte < arguments.depth / 2
                                            ? b[column * arguments.b_stride + byte]
                                            : uchar(0);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        multiply.run(a_tile, b_tile, tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t index = 0; index < tile.get_capacity(); ++index) {
        if (!tile.is_valid_element(index)) continue;
        auto const cell = tile.get_multidimensional_index(index);
        uint const column = first_column + uint(cell[0]), row = first_row + uint(cell[1]);
        if (row >= arguments.row_end || column >= arguments.column_count || (arguments.upper_triangle && column < row))
            continue;
        typename dtype_::dot_result_t const dot = typename dtype_::dot_result_t(tile[index]);
        output_type_ value = output_type_(dot);
        if constexpr (metric_ != nk_cross_dot_metal_k)
            value = arguments.upper_triangle && row == column
                        ? 0
                        : nk_cross_distance_metal_<metric_>(float(dot), norms[0][row - first_row],
                                                            norms[1][column - first_column]);
        c[row * arguments.c_stride + column] = value;
    }
}

#pragma region Kernels

kernel void nk_dots_i8_apple10_kernel_(device int8_t const *a [[buffer(0)]], device int8_t const *b [[buffer(1)]],
                                       device int *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]]) {
    nk_cross_tile_apple10_<int8_t, int>(a, b, c, arguments, group);
}
kernel void nk_dots_u8_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device int *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]]) {
    nk_cross_tile_apple10_<uchar, int>(a, b, c, arguments, group);
}
kernel void nk_dots_f16_apple10_kernel_(device half const *a [[buffer(0)]], device half const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]]) {
    nk_cross_tile_apple10_<half, float>(a, b, c, arguments, group);
}
kernel void nk_dots_bf16_apple10_kernel_(device bfloat const *a [[buffer(0)]], device bfloat const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]]) {
    nk_cross_tile_apple10_<bfloat, float>(a, b, c, arguments, group);
}
kernel void nk_dots_e4m3_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e4m3_t>(a, b, c, arguments, group, thread_index, a_stage, b_stage);
}
kernel void nk_dots_e5m2_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e5m2_t>(a, b, c, arguments, group, thread_index, a_stage, b_stage);
}
kernel void nk_dots_e3m2_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e3m2_t>(a, b, c, arguments, group, thread_index, a_stage, b_stage);
}
kernel void nk_dots_e2m3_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e2m3_t>(a, b, c, arguments, group, thread_index, a_stage, b_stage);
}
kernel void nk_dots_e2m1_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint thread_index [[thread_index_in_threadgroup]]) {
    alignas(16) threadgroup half a_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    alignas(16) threadgroup half b_stage[nk_cross_tile_apple10_k][nk_cross_tile_apple10_k];
    nk_cross_widened_tile_apple10_<nk::e2m1x2_t>(a, b, c, arguments, group, thread_index, a_stage, b_stage);
}

#pragma endregion Kernels

constant uint nk_cross_scaled_tile_apple10_k = 32;

template <typename dtype_, uint block_size_, nk_cross_scale_metal_t scale_, nk_cross_metric_metal_t metric_>
void nk_cross_scaled_tile_apple10_(device uchar const *a, device uchar const *b, device float *c,
                                   device uchar const *a_scales, device uchar const *b_scales, float a_tensor,
                                   float b_tensor, constant nk_cross_arguments_metal_t &arguments, uint2 group,
                                   uint thread_index, threadgroup half (*a_stage)[block_size_],
                                   threadgroup half (*b_stage)[block_size_],
                                   threadgroup float const (*norms)[nk_cross_scaled_tile_apple10_k]) {
    constexpr uint side = nk_cross_scaled_tile_apple10_k, threads = 128;
    uint const first_row = arguments.row_start + group.y * side, first_column = group.x * side;
    if (arguments.upper_triangle && first_column + side <= first_row) return;
    constexpr auto descriptor = matmul2d_descriptor(side, side, block_size_, /*transpose_left=*/false,
                                                    /*transpose_right=*/true, /*relaxed_precision=*/false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    using stage_t = tensor<threadgroup half, extents<int32_t, block_size_, side>, tensor_inline>;
    stage_t a_tile(&a_stage[0][0], extents<int32_t, block_size_, side>()),
        b_tile(&b_stage[0][0], extents<int32_t, block_size_, side>());
    auto tile = multiply.template get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), float>();
    nk_cross_scaled_sum_metal_t sums[side * side / threads];
    for (uint16_t index = 0; index < tile.get_capacity(); ++index) sums[index] = {float2(0), 0};
    for (uint block = 0; block != arguments.depth / block_size_; ++block) {
        for (uint cell = thread_index; cell < side * block_size_; cell += threads) {
            uint const line = cell / block_size_, offset = cell % block_size_;
            uint const row = first_row + line, column = first_column + line;
            a_stage[line][offset] = row < arguments.row_end
                                        ? half(dtype_::load(a + row * arguments.a_stride, block * block_size_ + offset))
                                        : half(0);
            b_stage[line][offset] = column < arguments.column_count ? half(dtype_::load(b + column * arguments.b_stride,
                                                                                        block * block_size_ + offset))
                                                                    : half(0);
            if constexpr (scale_ == nk_cross_scale_e4m3_metal_k) {
                if (row < arguments.row_end)
                    a_stage[line][offset] *= nk::e4m3_t::widen(
                        ushort4(a_scales[row * arguments.a_scales_stride + block]))[0];
                if (column < arguments.column_count)
                    b_stage[line][offset] *= nk::e4m3_t::widen(
                        ushort4(b_scales[column * arguments.b_scales_stride + block]))[0];
            }
        }
        for (uint16_t index = 0; index < tile.get_capacity(); ++index)
            if (tile.is_valid_element(index)) tile[index] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        multiply.run(a_tile, b_tile, tile);
        for (uint16_t index = 0; index < tile.get_capacity(); ++index) {
            if (!tile.is_valid_element(index)) continue;
            auto const cell = tile.get_multidimensional_index(index);
            uint const column = first_column + uint(cell[0]), row = first_row + uint(cell[1]);
            if (row < arguments.row_end && column < arguments.column_count)
                nk_cross_scaled_block_add_metal_<scale_>(
                    sums[index], tile[index],
                    scale_ == nk_cross_scale_e4m3_metal_k ? 0x38 : a_scales[row * arguments.a_scales_stride + block],
                    scale_ == nk_cross_scale_e4m3_metal_k ? 0x38
                                                          : b_scales[column * arguments.b_scales_stride + block]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint const tensor_product = nk_cross_tensor_product_metal_(a_tensor, b_tensor);
    for (uint16_t index = 0; index < tile.get_capacity(); ++index) {
        if (!tile.is_valid_element(index)) continue;
        auto const cell = tile.get_multidimensional_index(index);
        uint const column = first_column + uint(cell[0]), row = first_row + uint(cell[1]);
        if (row >= arguments.row_end || column >= arguments.column_count || (arguments.upper_triangle && column < row))
            continue;
        float value = nk_cross_scaled_dot_metal_(sums[index], tensor_product);
        if constexpr (metric_ != nk_cross_dot_metal_k) {
            float const a_norm = norms[0][row - first_row], b_norm = norms[1][column - first_column];
            value = arguments.upper_triangle && row == column
                        ? 0
                        : nk_cross_distance_metal_<metric_>(value, a_norm, b_norm);
        }
        c[row * arguments.c_stride + column] = value;
    }
}

kernel void nk_dots_mxfp8e4m3_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_mxfp8e5m2_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_mxfp6e2m3_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_mxfp6e3m2_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_mxfp4_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][32], b_stage[nk_cross_scaled_tile_apple10_k][32];
    nk_cross_scaled_tile_apple10_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_nvfp4_apple10_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half a_stage[nk_cross_scaled_tile_apple10_k][16], b_stage[nk_cross_scaled_tile_apple10_k][16];
    nk_cross_scaled_tile_apple10_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_i4_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device int *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int8_t a_stage[64][64];
    alignas(128) threadgroup nk_i4_storage_apple10_t b_stage[64][nk_cross_int4_pitch_apple10_k];
    nk_cross_int4_tile_apple10_<nk::i4x2_t, int8_t, nk_i4_operand_apple10_t, nk_cross_dot_metal_k, 64>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_u4_apple10_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device uint *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uchar a_stage[64][64];
    alignas(128) threadgroup nk_u4_storage_apple10_t b_stage[64][nk_cross_int4_pitch_apple10_k];
    nk_cross_int4_tile_apple10_<nk::u4x2_t, uchar, nk_u4_operand_apple10_t, nk_cross_dot_metal_k, 64>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_i4_apple10_small_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device int *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int8_t a_stage[32][64];
    alignas(128) threadgroup nk_i4_storage_apple10_t b_stage[32][nk_cross_int4_pitch_apple10_k];
    nk_cross_int4_tile_apple10_<nk::i4x2_t, int8_t, nk_i4_operand_apple10_t, nk_cross_dot_metal_k, 32>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, nullptr);
}

kernel void nk_dots_u4_apple10_small_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                             device uint *c [[buffer(2)]],
                                             constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uchar a_stage[32][64];
    alignas(128) threadgroup nk_u4_storage_apple10_t b_stage[32][nk_cross_int4_pitch_apple10_k];
    nk_cross_int4_tile_apple10_<nk::u4x2_t, uchar, nk_u4_operand_apple10_t, nk_cross_dot_metal_k, 32>(
        a, b, c, arguments, group, thread_index, a_stage, b_stage, nullptr);
}
