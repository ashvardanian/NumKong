/**
 *  @file include/numkong/dots/apple9.metal
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products for Apple GPUs of Metal family 9, on @c simdgroup_matrix products.
 *
 *  @sa include/numkong/dots/apple9.h, which embeds and launches this source after `simt.metal`
 *  @sa include/numkong/dots/apple10.metal, the M5 capability on the Neural Accelerators
 *
 *  Four simdgroups own a @b [64,64] output tile, each a @b [32,32] quadrant held as sixteen 8 × 8
 *  @c float accumulators, over steps of 32 dimensions staged into threadgroup tiles of @c half, or
 *  of @c bfloat for BF16. The 8-, 6- and 4-bit floats widen to @c half exactly on the way in, as
 *  `simt.metal` widens them, and every product of two staged values is exact in @c float, so only
 *  the sums round.
 */

/** Side of a threadgroup's output tile, dimensions per staged step, and threads per group. */
constant uint nk_cross_tile_apple9_k = 64, nk_cross_step_apple9_k = 32, nk_cross_threads_apple9_k = 128;
static_assert(nk_cross_tile_apple9_k == nk_cross_tile_metal_k, "the C launch core grids every capability by one side");

/** Staged values per line: one step, then 16 bytes that stagger consecutive lines across banks. */
constant uint nk_cross_pitch_apple9_k = nk_cross_step_apple9_k + 8;

/** Staged values of both operands, each @b [64,pitch]: A's rows, then B's columns. */
constant uint nk_cross_stage_cells_apple9_k = 2 * nk_cross_tile_apple9_k * nk_cross_pitch_apple9_k;

/**
 *  @brief One @b [64,64] tile of C = A × Bᵀ, with B given row-major as @b [columns,depth], on
 *      @c simdgroup_matrix products accumulated in @c float.
 *
 *  Each thread streams the same aligned 16-byte chunks of the same lines through every step. A
 *  row's last chunk loads a byte at a time up to the depth, so no load passes a row's bytes, and
 *  its dimensions past the depth stage as zeros.
 */
template <typename dtype_>
void nk_cross_tile_apple9_(device uchar const *a, device uchar const *b, device float *c,
                           constant nk_cross_arguments_metal_t &arguments, uint2 group, uint thread_index,
                           uint simdgroup, uint lane, threadgroup typename dtype_::stage_t *stage,
                           threadgroup float *spills) {
    using stage_t = typename dtype_::stage_t;
    constexpr uint side = nk_cross_tile_apple9_k, step = nk_cross_step_apple9_k, pitch = nk_cross_pitch_apple9_k;
    constexpr uint chunk_dimensions = 16 / sizeof(typename dtype_::raw_t) * dtype_::dimensions_per_value;
    constexpr uint chunks_per_line = step / chunk_dimensions;
    constexpr uint chunks_per_thread = 2 * side * chunks_per_line / nk_cross_threads_apple9_k;
    uint const first_row = arguments.row_start + group.y * side, first_column = group.x * side;
    if (arguments.upper_triangle && first_column + side <= first_row) return; // below the diagonal, group-uniform

    device uint4 const *lines[chunks_per_thread];
    uint line_dimensions[chunks_per_thread], stage_offsets[chunks_per_thread];
    for (uint index = 0; index != chunks_per_thread; ++index) {
        uint const chunk_index = thread_index + index * nk_cross_threads_apple9_k;
        uint const operand = chunk_index / (side * chunks_per_line), line = chunk_index / chunks_per_line % side;
        uint const row = first_row + line, column = first_column + line;
        line_dimensions[index] = chunk_index % chunks_per_line * chunk_dimensions;
        stage_offsets[index] = (operand * side + line) * pitch + line_dimensions[index];
        device uchar const *start = operand ? b + column * arguments.b_stride : a + row * arguments.a_stride;
        bool const present = operand ? column < arguments.column_count : row < arguments.row_end;
        lines[index] = present ? (device uint4 const *)start : nullptr;
    }

    uint const quadrant_row = simdgroup / 2 * 32, quadrant_column = simdgroup % 2 * 32;
    threadgroup stage_t const *a_quadrant = stage + quadrant_row * pitch;
    threadgroup stage_t const *b_quadrant = stage + (side + quadrant_column) * pitch;
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
                *(threadgroup vec<stage_t, 4> *)(stage + stage_offsets[index] + 4 * quad) = select(
                    vec<stage_t, 4>(0), dtype_::widen(nk_chunk_codes_<dtype_>(chunk, quad)), inside);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 0; offset != step; offset += 8) {
            simdgroup_matrix<stage_t, 8, 8> a_fragments[4], b_fragments[4];
            for (uint row_step = 0; row_step != 4; ++row_step)
                simdgroup_load(a_fragments[row_step], a_quadrant + 8 * row_step * pitch + offset, pitch);
            for (uint column_step = 0; column_step != 4; ++column_step)
                simdgroup_load(b_fragments[column_step], b_quadrant + 8 * column_step * pitch + offset, pitch,
                               ulong2(0, 0), true);
            for (uint row_step = 0; row_step != 4; ++row_step)
                for (uint column_step = 0; column_step != 4; ++column_step)
                    simdgroup_multiply_accumulate(sums[row_step][column_step], a_fragments[row_step],
                                                  b_fragments[column_step], sums[row_step][column_step]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fragments pass through the simdgroup's spill so each cell is stored only inside the output.
    threadgroup float *spill = spills + simdgroup * 64;
    for (uint row_step = 0; row_step != 4; ++row_step) {
        for (uint column_step = 0; column_step != 4; ++column_step) {
            simdgroup_store(sums[row_step][column_step], spill, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint cell = lane; cell < 64; cell += 32) {
                uint const row = first_row + quadrant_row + 8 * row_step + cell / 8;
                uint const column = first_column + quadrant_column + 8 * column_step + cell % 8;
                if (row >= arguments.row_end || column >= arguments.column_count) continue;
                if (arguments.upper_triangle && column < row) continue;
                c[row * arguments.c_stride + column] = spill[cell];
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
    alignas(16) threadgroup half stage[nk_cross_stage_cells_apple9_k];
    threadgroup float spills[4 * 64];
    nk_cross_tile_apple9_<nk::f16_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_bf16_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup bfloat stage[nk_cross_stage_cells_apple9_k];
    threadgroup float spills[4 * 64];
    nk_cross_tile_apple9_<nk::bf16_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e4m3_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[nk_cross_stage_cells_apple9_k];
    threadgroup float spills[4 * 64];
    nk_cross_tile_apple9_<nk::e4m3_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e5m2_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[nk_cross_stage_cells_apple9_k];
    threadgroup float spills[4 * 64];
    nk_cross_tile_apple9_<nk::e5m2_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e3m2_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[nk_cross_stage_cells_apple9_k];
    threadgroup float spills[4 * 64];
    nk_cross_tile_apple9_<nk::e3m2_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e2m3_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[nk_cross_stage_cells_apple9_k];
    threadgroup float spills[4 * 64];
    nk_cross_tile_apple9_<nk::e2m3_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}
kernel void nk_dots_e2m1_apple9_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]],
                                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    alignas(16) threadgroup half stage[nk_cross_stage_cells_apple9_k];
    threadgroup float spills[4 * 64];
    nk_cross_tile_apple9_<nk::e2m1x2_t>(a, b, c, arguments, group, thread_index, simdgroup, lane, stage, spills);
}

#pragma endregion Kernels
