/**
 *  @file include/numkong/dots/apple9.metal
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products for Apple GPUs of Metal family 9, on @c simdgroup_matrix products.
 *
 *  @sa include/numkong/dots/apple9.h, which embeds and launches this source after `metal.metal`
 *  @sa include/numkong/dots/apple10.metal, the M5 capability on the Neural Accelerators
 *
 *  Four simdgroups own a @b [64,64] output tile, each a @b [32,32] quadrant held as sixteen 8 × 8
 *  @c float accumulators, over steps of 32 dimensions staged into threadgroup tiles of @c half, or
 *  of @c bfloat for BF16. The 8-, 6- and 4-bit floats widen to @c half exactly on the way in, as
 *  `metal.metal` widens them, and every product of two staged values is exact in @c float, so only
 *  the sums round.
 */

/** Side of a threadgroup's output tile, dimensions per staged step, and threads per group. */
constant uint nk_cross_tile_apple9_k = 64, nk_cross_step_apple9_k = 32, nk_cross_threads_apple9_k = 128;
static_assert(nk_cross_tile_apple9_k == nk_cross_tile_metal_k, "unscaled kernels share the launch tile size");

constant uint nk_cross_scaled_tile_apple9_k = 32;

/** Staged values per line: one step, then 16 bytes that stagger consecutive lines across banks. */
constant uint nk_cross_pitch_apple9_k = nk_cross_step_apple9_k + 8;

/**
 *  @brief One @b [64,64] tile of C = A × Bᵀ, with B given row-major as @b [columns,depth], on
 *      @c simdgroup_matrix products accumulated in @c float.
 *
 *  Each thread streams the same aligned 16-byte chunks of the same lines through every step. A
 *  row's last chunk loads a byte at a time up to the depth, so no load passes a row's bytes, and
 *  its dimensions past the depth stage as zeros.
 */
template <typename dtype_, nk_cross_metric_metal_t metric_ = nk_cross_dot_metal_k>
void nk_cross_tile_apple9_(device uchar const *a, device uchar const *b, device float *c,
                           constant nk_cross_arguments_metal_t &arguments, uint2 group, uint thread_index,
                           uint simdgroup, uint lane,
                           threadgroup
                           typename dtype_::stage_t (*stage)[nk_cross_tile_apple9_k][nk_cross_pitch_apple9_k],
                           threadgroup float (*spills)[8][8],
                           threadgroup float const (*norms)[nk_cross_tile_metal_k] = nullptr) {
    using stage_t = typename dtype_::stage_t;
    constexpr uint side = nk_cross_tile_apple9_k, step = nk_cross_step_apple9_k, pitch = nk_cross_pitch_apple9_k;
    constexpr uint chunk_dimensions = 16 / sizeof(typename dtype_::raw_t) * dtype_::dimensions_per_value;
    constexpr uint chunks_per_line = step / chunk_dimensions;
    constexpr uint chunks_per_thread = 2 * side * chunks_per_line / nk_cross_threads_apple9_k;
    uint const first_row = arguments.rows_begin + group.y * side, first_column = group.x * side;
    if (arguments.upper_triangle && first_column + side <= first_row) return; // below the diagonal, group-uniform

    device uint4 const *lines[chunks_per_thread];
    uint line_dimensions[chunks_per_thread];
    threadgroup stage_t *stage_lines[chunks_per_thread];
    for (uint index = 0; index != chunks_per_thread; ++index) {
        uint const chunk_index = thread_index + index * nk_cross_threads_apple9_k;
        uint const operand = chunk_index / (side * chunks_per_line), line = chunk_index / chunks_per_line % side;
        uint const row = first_row + line, column = first_column + line;
        line_dimensions[index] = chunk_index % chunks_per_line * chunk_dimensions;
        stage_lines[index] = &stage[operand][line][line_dimensions[index]];
        device uchar const *start = operand ? b + column * arguments.b_stride : a + row * arguments.a_stride;
        bool const present = operand ? column < arguments.column_count : row < arguments.rows_end;
        lines[index] = present ? (device uint4 const *)start : nullptr;
    }

    uint const quadrant_row = simdgroup / 2 * 32, quadrant_column = simdgroup % 2 * 32;
    threadgroup stage_t const(*a_quadrant)[pitch] = stage[0] + quadrant_row;
    threadgroup stage_t const(*b_quadrant)[pitch] = stage[1] + quadrant_column;
    simdgroup_matrix<float, 8, 8> sums[4][4];
    for (uint row_step = 0; row_step != 4; ++row_step)
        for (uint column_step = 0; column_step != 4; ++column_step)
            sums[row_step][column_step] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

    for (uint depth_start = 0; depth_start < arguments.depth; depth_start += step) {
        for (uint index = 0; index != chunks_per_thread; ++index) {
            uint const dimension = depth_start + line_dimensions[index];
            uint4 chunk = uint4(0);
            if (lines[index] && dimension + chunk_dimensions <= arguments.depth)
                chunk = lines[index][dimension / chunk_dimensions];
            else if (lines[index] && dimension < arguments.depth)
                chunk = nk_tail_load_<dtype_>((device uchar const *)(lines[index] + dimension / chunk_dimensions),
                                              arguments.depth - dimension);
            for (uint quad = 0; quad != chunk_dimensions / 4; ++quad) {
                bool4 const inside = uint4(dimension + 4 * quad) + uint4(0, 1, 2, 3) < uint4(arguments.depth);
                *(threadgroup vec<stage_t, 4> *)(stage_lines[index] + 4 * quad) = select(
                    vec<stage_t, 4>(0), dtype_::widen(nk_chunk_codes_<dtype_>(chunk, quad)), inside);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 0; offset != step; offset += 8) {
            simdgroup_matrix<stage_t, 8, 8> a_fragments[4], b_fragments[4];
            for (uint row_step = 0; row_step != 4; ++row_step)
                simdgroup_load(a_fragments[row_step], &a_quadrant[8 * row_step][offset], pitch);
            for (uint column_step = 0; column_step != 4; ++column_step)
                simdgroup_load(b_fragments[column_step], &b_quadrant[8 * column_step][offset], pitch, ulong2(0, 0),
                               true);
            for (uint row_step = 0; row_step != 4; ++row_step)
                for (uint column_step = 0; column_step != 4; ++column_step)
                    simdgroup_multiply_accumulate(sums[row_step][column_step], a_fragments[row_step],
                                                  b_fragments[column_step], sums[row_step][column_step]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fragments pass through the simdgroup's spill so each cell is stored only inside the output.
    threadgroup float (*spill)[8] = spills[simdgroup];
    for (uint row_step = 0; row_step != 4; ++row_step) {
        for (uint column_step = 0; column_step != 4; ++column_step) {
            simdgroup_store(sums[row_step][column_step], &spill[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint cell = lane; cell < 64; cell += 32) {
                uint const row = first_row + quadrant_row + 8 * row_step + cell / 8;
                uint const column = first_column + quadrant_column + 8 * column_step + cell % 8;
                if (row >= arguments.rows_end || column >= arguments.column_count) continue;
                if (arguments.upper_triangle && column < row) continue;
                float value = spill[cell / 8][cell % 8];
                if (metric_ != nk_cross_dot_metal_k)
                    value = arguments.upper_triangle && row == column
                                ? 0.0f
                                : nk_cross_distance_metal_<metric_>(value, norms[0][row - first_row],
                                                                    norms[1][column - first_column]);
                c[row * arguments.c_stride + column] = value;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#pragma region Kernels

kernel void nk_dots_f16_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]],
                                       uint simdgroup [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[2][nk_cross_tile_apple9_k][nk_cross_pitch_apple9_k];
    threadgroup float spills[4][8][8];
    nk_cross_tile_apple9_<nk::f16_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_bf16_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup bfloat stage[2][nk_cross_tile_apple9_k][nk_cross_pitch_apple9_k];
    threadgroup float spills[4][8][8];
    nk_cross_tile_apple9_<nk::bf16_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e4m3_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[2][nk_cross_tile_apple9_k][nk_cross_pitch_apple9_k];
    threadgroup float spills[4][8][8];
    nk_cross_tile_apple9_<nk::e4m3_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e5m2_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[2][nk_cross_tile_apple9_k][nk_cross_pitch_apple9_k];
    threadgroup float spills[4][8][8];
    nk_cross_tile_apple9_<nk::e5m2_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e3m2_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[2][nk_cross_tile_apple9_k][nk_cross_pitch_apple9_k];
    threadgroup float spills[4][8][8];
    nk_cross_tile_apple9_<nk::e3m2_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e2m3_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[2][nk_cross_tile_apple9_k][nk_cross_pitch_apple9_k];
    threadgroup float spills[4][8][8];
    nk_cross_tile_apple9_<nk::e2m3_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e2m1_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[2][nk_cross_tile_apple9_k][nk_cross_pitch_apple9_k];
    threadgroup float spills[4][8][8];
    nk_cross_tile_apple9_<nk::e2m1x2_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}

#pragma endregion Kernels

template <typename dtype_, uint block_size_, nk_cross_scale_metal_t scale_, nk_cross_metric_metal_t metric_>
void nk_cross_scaled_tile_apple9_(device uchar const *a, device uchar const *b, device float *c,
                                  device uchar const *a_scales, device uchar const *b_scales, float a_tensor,
                                  float b_tensor, constant nk_cross_arguments_metal_t &arguments, uint2 group,
                                  uint thread_index, uint simdgroup, uint lane,
                                  threadgroup half (*stage)[nk_cross_scaled_tile_apple9_k][block_size_ + 8],
                                  threadgroup float (*spills)[8][8],
                                  threadgroup float const (*norms)[nk_cross_scaled_tile_apple9_k]) {
    constexpr uint side = nk_cross_scaled_tile_apple9_k, pitch = block_size_ + 8;
    uint const first_row = arguments.rows_begin + group.y * side, first_column = group.x * side;
    if (arguments.upper_triangle && first_column + side <= first_row) return;
    uint const quadrant_row = simdgroup / 2 * 16, quadrant_column = simdgroup % 2 * 16;
    nk_cross_scaled_sum_metal_t sums[2][2][2] = {};
    threadgroup float (*spill)[8] = spills[simdgroup];
    for (uint block = 0; block != arguments.depth / block_size_; ++block) {
        for (uint cell = thread_index; cell < side * block_size_; cell += nk_cross_threads_apple9_k) {
            uint const line = cell / block_size_, offset = cell % block_size_;
            uint const row = first_row + line, column = first_column + line;
            uint const dimension = block * block_size_ + offset;
            stage[0][line][offset] = row < arguments.rows_end
                                         ? half(dtype_::load(a + row * arguments.a_stride, dimension))
                                         : half(0);
            stage[1][line][offset] = column < arguments.column_count
                                         ? half(dtype_::load(b + column * arguments.b_stride, dimension))
                                         : half(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> dots[2][2];
        for (uint fragment_index = 0; fragment_index != 4; ++fragment_index) {
            uint const row_step = fragment_index / 2, column_step = fragment_index % 2;
            dots[row_step][column_step] = make_filled_simdgroup_matrix<float, 8, 8>(0);
        }
        for (uint offset = 0; offset != block_size_; offset += 8) {
            simdgroup_matrix<half, 8, 8> a_fragments[2], b_fragments[2];
            for (uint row_step = 0; row_step != 2; ++row_step)
                simdgroup_load(a_fragments[row_step], &stage[0][quadrant_row + 8 * row_step][offset], pitch);
            for (uint column_step = 0; column_step != 2; ++column_step)
                simdgroup_load(b_fragments[column_step], &stage[1][quadrant_column + 8 * column_step][offset], pitch,
                               ulong2(0), true);
            for (uint fragment_index = 0; fragment_index != 4; ++fragment_index) {
                uint const row_step = fragment_index / 2, column_step = fragment_index % 2;
                simdgroup_multiply_accumulate(dots[row_step][column_step], a_fragments[row_step],
                                              b_fragments[column_step], dots[row_step][column_step]);
            }
        }
        for (uint fragment_index = 0; fragment_index != 4; ++fragment_index) {
            uint const row_step = fragment_index / 2, column_step = fragment_index % 2;
            simdgroup_store(dots[row_step][column_step], &spill[0][0], 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint part = 0; part != 2; ++part) {
                uint const cell = lane + part * 32;
                uint const row = first_row + quadrant_row + 8 * row_step + cell / 8;
                uint const column = first_column + quadrant_column + 8 * column_step + cell % 8;
                if (row >= arguments.rows_end || column >= arguments.column_count) continue;
                nk_cross_scaled_block_add_metal_<scale_>(sums[row_step][column_step][part], spill[cell / 8][cell % 8],
                                                         a_scales[row * arguments.a_scales_stride + block],
                                                         b_scales[column * arguments.b_scales_stride + block]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint const tensor_product = nk_cross_tensor_product_metal_(a_tensor, b_tensor);
    for (uint output = 0; output != 8; ++output) {
        uint const row_step = output / 4, column_step = output / 2 % 2, part = output % 2;
        uint const cell = lane + part * 32;
        uint const row = first_row + quadrant_row + 8 * row_step + cell / 8;
        uint const column = first_column + quadrant_column + 8 * column_step + cell % 8;
        if (row >= arguments.rows_end || column >= arguments.column_count || (arguments.upper_triangle && column < row))
            continue;
        float value = nk_cross_scaled_dot_metal_(sums[row_step][column_step][part], tensor_product);
        if (metric_ != nk_cross_dot_metal_k) {
            float const a_norm = norms[0][row - first_row], b_norm = norms[1][column - first_column];
            value = arguments.upper_triangle && row == column
                        ? 0.0f
                        : nk_cross_distance_metal_<metric_>(value, a_norm, b_norm);
        }
        c[row * arguments.c_stride + column] = value;
    }
}

kernel void nk_dots_mxfp8e4m3_apple9_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half stage[2][nk_cross_scaled_tile_apple9_k][32 + 8];
    threadgroup float spills[4][8][8];
    nk_cross_scaled_tile_apple9_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, simdgroup, lane, stage, spills,
        nullptr);
}

kernel void nk_dots_mxfp8e5m2_apple9_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half stage[2][nk_cross_scaled_tile_apple9_k][32 + 8];
    threadgroup float spills[4][8][8];
    nk_cross_scaled_tile_apple9_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, simdgroup, lane, stage, spills,
        nullptr);
}

kernel void nk_dots_mxfp6e2m3_apple9_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half stage[2][nk_cross_scaled_tile_apple9_k][32 + 8];
    threadgroup float spills[4][8][8];
    nk_cross_scaled_tile_apple9_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, simdgroup, lane, stage, spills,
        nullptr);
}

kernel void nk_dots_mxfp6e3m2_apple9_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half stage[2][nk_cross_scaled_tile_apple9_k][32 + 8];
    threadgroup float spills[4][8][8];
    nk_cross_scaled_tile_apple9_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, simdgroup, lane, stage, spills,
        nullptr);
}

kernel void nk_dots_mxfp4_apple9_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half stage[2][nk_cross_scaled_tile_apple9_k][32 + 8];
    threadgroup float spills[4][8][8];
    nk_cross_scaled_tile_apple9_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, simdgroup, lane, stage, spills,
        nullptr);
}

kernel void nk_dots_nvfp4_apple9_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup half stage[2][nk_cross_scaled_tile_apple9_k][16 + 8];
    threadgroup float spills[4][8][8];
    nk_cross_scaled_tile_apple9_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, simdgroup, lane, stage, spills,
        nullptr);
}
