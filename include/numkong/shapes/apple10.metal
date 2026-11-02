/**
 *  @file include/numkong/shapes/apple10.metal
 *  @author Ash Vardanian
 *  @date October 24, 2026
 *  @brief Chamfer, Hausdorff and superposed RMSD of F16 clouds for Apple GPUs of Metal family 10,
 *      on the Neural Accelerators.
 *
 *  @sa include/numkong/shapes/apple10.h, which embeds this source after `metal.metal`
 *  @sa include/numkong/shapes/blackwell.cuh, the CUDA sibling
 *
 *  Superposed RMSD multiplies frames as one product, C = A × Bᵀ, over their points. A holds the
 *  x, y and z rows of 21 row frames as rows 0-20, 21-41 and 42-62, and B those of 21 column
 *  frames, so each pair's 3 × 3 cross-covariance lands in one 3 × 3 lattice of C. @c matmul2d
 *  sums it in F32 over 64-point slabs staged in threadgroup memory, and every thread solves
 *  Theobald's quaternion polynomial for its pairs, folding into Ga + Gb − 2λ the way the CPU
 *  capabilities fold it.
 *
 *  Chamfer and Hausdorff fold the norms into the depth, so the products are squared distances,
 *  [a, ‖a‖²_high, ‖a‖²_low, 1, 1] · [−2b, 1, 1, ‖b‖²_high, ‖b‖²_low] over 7 of 16 depth values.
 *  One product compares 64 outer points with 64 inner points under one flip, and the flips take
 *  turns, as all 8 copies of the inner points would not fit beside the products in threadgroup
 *  memory. The expansion only searches: two threads share each outer point, each keeping per flip
 *  the least expanded value of its half, the group of 8 inner points holding it and the least of
 *  all other groups, and the pair recomputes that group in the difference form of the CPU
 *  capabilities, rescanning every inner point when another group comes within the expansion's
 *  rounding. The reverse direction runs as its own pass with the clouds swapped, and both clouds
 *  scale by the one power of two that keeps their norms within the range of F16.
 */
#include <metal_tensor>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

/** Outer points and inner points one Chamfer product compares, and the depth they fill. */
constant uint nk_shapes_points_apple10_k = 64;
constant uint nk_shapes_depth_apple10_k = 16;

/** Inner points of one group, the unit the expansion's search narrows the nearest point to. */
constant uint nk_shapes_group_points_apple10_k = 8;

/** Frames per side of an RMSD tile, whose three axes fill 63 of the 64 operand rows. */
constant uint nk_rmsds_frames_apple10_k = 21;

/** Points of one RMSD slab, the depth of one product. */
constant uint nk_rmsds_slab_points_apple10_k = 64;

#pragma region RMSD

/** One threadgroup's tile of 21 row and 21 column frames: every slab of their points staged as
 *  operand rows, each frame's axes in rows f, 21 + f and 42 + f, multiplied and accumulated, then
 *  each pair folded from its 3 × 3 lattice of the product. Every frame holds the points of the
 *  first row frame, as the launcher checks. */
kernel void nk_rmsds_packed_f16_apple10_kernel_(
    device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
    device uchar *results [[buffer(2)]], device float *rotations [[buffer(3)]],
    constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint side = nk_shapes_points_apple10_k, frames = nk_rmsds_frames_apple10_k, threads = 128;
    constexpr uint chunks_per_side = side * 8, thread_chunks = chunks_per_side / threads;
    // The two operands' slabs, then the product over them once the last slab is multiplied
    threadgroup float scratch[side * side];
    threadgroup float chunk_norms[2][chunks_per_side];
    threadgroup float norms[2][frames];
    threadgroup ulong packs[2][frames];
    threadgroup half(*stages)[side][side] = (threadgroup half(*)[side][side])scratch;
    threadgroup float (*products)[side] = (threadgroup float (*)[side])scratch;

    ulong const column_tiles = nk::divide_round_up(arguments.columns, ulong(frames));
    ulong const tile = ulong(group.y) * groups.x + group.x;
    if (tile >= nk::divide_round_up(arguments.rows, ulong(frames)) * column_tiles) return;
    ulong const row_begin = tile / column_tiles * frames, column_begin = tile % column_tiles * frames;
    uint const points = nk_shapes_packed_header_f16_metal_(nk_shapes_pack_metal_(a_table[0], arguments))->points;
    uint const axis_stride = nk::round_up_to_multiple(points * 2, nk_shapes_alignment_metal_k) / 2;
    uint const header_bytes = nk::round_up_to_multiple((uint)sizeof(nk_shapes_packed_header_f32_metal_t),
                                                       nk_shapes_alignment_metal_k);
    if (thread_index < 2 * frames) {
        uint const operand = thread_index / frames, frame = thread_index % frames;
        ulong const index = (operand ? column_begin : row_begin) + frame;
        bool const inside = index < (operand ? arguments.columns : arguments.rows);
        packs[operand][frame] = inside ? ulong(nk_shapes_pack_metal_((operand ? b_table : a_table)[index], arguments))
                                       : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto descriptor = matmul2d_descriptor(side, side, side, /*transpose_left=*/false,
                                                    /*transpose_right=*/true, /*relaxed_precision=*/false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    using stage_t = tensor<threadgroup half, extents<int32_t, side, side>, tensor_inline>;
    stage_t a_tile(&stages[0][0][0], extents<int32_t, side, side>()),
        b_tile(&stages[1][0][0], extents<int32_t, side, side>());
    auto product = multiply.template get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), float>();
#pragma unroll
    for (uint16_t index = 0; index < product.get_capacity(); ++index)
        if (product.is_valid_element(index)) product[index] = 0;

    // Each thread stages the same chunks of every slab, so its partial squares sum one row's chunk
    float partials[2][thread_chunks] = {};
    for (uint slab = 0; slab * nk_rmsds_slab_points_apple10_k < points; ++slab) {
        for (uint operand = 0; operand != 2; ++operand)
            for (uint slot = 0; slot != thread_chunks; ++slot) {
                uint const chunk_index = thread_index + slot * threads, row = chunk_index / 8, chunk = chunk_index % 8;
                uint const axis = row / frames, frame = row % frames;
                ulong const pack = axis < 3 ? packs[operand][frame] : 0;
                device half const *values = (device half const *)(reinterpret_cast<device uchar const *>(pack) +
                                                                  header_bytes) +
                                            axis * axis_stride;
                for (uint element = 0; element != 8; ++element) {
                    uint const point = slab * nk_rmsds_slab_points_apple10_k + chunk * 8 + element;
                    half const value = pack && point < points ? values[point] : half(0);
                    stages[operand][row][chunk * 8 + element] = value;
                    partials[operand][slot] += float(value) * float(value);
                }
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        multiply.run(a_tile, b_tile, product);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    tensor<threadgroup float, extents<int32_t, side, side>, tensor_inline> products_tile(
        &products[0][0], extents<int32_t, side, side>());
    product.store(products_tile);
    for (uint operand = 0; operand != 2; ++operand)
        for (uint slot = 0; slot != thread_chunks; ++slot)
            chunk_norms[operand][thread_index + slot * threads] = partials[operand][slot];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every frame's squared norm sums its 3 rows of 8 chunk columns in one order
    if (thread_index < 2 * frames) {
        uint const operand = thread_index / frames, frame = thread_index % frames;
        float norm = 0;
        for (uint axis = 0; axis != 3; ++axis)
            for (uint chunk = 0; chunk != 8; ++chunk) norm += chunk_norms[operand][(axis * frames + frame) * 8 + chunk];
        norms[operand][frame] = norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pair = thread_index; pair < frames * frames; pair += threads) {
        uint const row = pair / frames, column = pair % frames;
        if (row_begin + row >= arguments.rows || column_begin + column >= arguments.columns) continue;
        float cross_covariance[9];
        for (uint row_axis = 0; row_axis != 3; ++row_axis)
            for (uint column_axis = 0; column_axis != 3; ++column_axis)
                cross_covariance[row_axis * 3 + column_axis] =
                    products[row_axis * frames + row][column_axis * frames + column];
        float const residual = nk_rmsds_folded_f32_metal_(cross_covariance, norms[0][row] + norms[1][column]);
        device float *results_row = (device float *)(results + (row_begin + row) * arguments.results_stride);
        results_row[column_begin + column] = points ? precise::sqrt(precise::divide(residual, float(points))) : 0.0f;
    }
}

#pragma endregion RMSD

#pragma region Chamfer and Hausdorff

/** The power of two both clouds of a pair divide by, so that every squared norm stays below 2¹⁴ and
 *  every doubled coordinate below 2⁸ in F16: 2^−k for k = max(0, ⌊log₂ r⌋ − 6). */
inline float nk_shapes_scale_apple10_(float largest_radius) {
    int const exponent = int((as_type<uint>(largest_radius) >> 23) & 0xFF) - 127 - 6;
    return exponent > 0 ? as_type<float>(uint(127 - exponent) << 23) : 1.0f;
}

/** One packed point scaled by @p scale and rounded into F16, with its squared norm summed in F32
 *  from the rounded coordinates and split into two F16 halves, or zeros past the cloud. */
inline void nk_shapes_point_f16_apple10_(device half const *axes, uint axis_stride, uint point, uint count, float scale,
                                         thread half *coordinates, thread float &norm, thread half &norm_high,
                                         thread half &norm_low) {
#pragma clang fp contract(off) reassociate(off)
    norm = 0;
    for (uint axis = 0; axis != 3; ++axis) {
        coordinates[axis] = point < count ? half(float(axes[axis * axis_stride + point]) * scale) : half(0);
        norm += float(coordinates[axis]) * float(coordinates[axis]);
    }
    norm_high = half(norm), norm_low = half(norm - float(norm_high));
}

/** Per flip, adds the mean of one direction's nearest distances to @p means and raises @p maxima to
 *  the largest of them, over the threadgroup, in every thread. Each slab of outer points ends in
 *  reductions read from threadgroup memory, so past @p limit all threads leave together, stopping
 *  with per-flip lower bounds and returning true. */
template <bool hausdorff_>
inline bool nk_shapes_direction_f16_apple10_(device uchar const *outer_packed, device uchar const *inner_packed,
                                             float scale, float limit, uint thread_index, uint simdgroup, uint lane,
                                             threadgroup half (*outer_stage)[nk_shapes_depth_apple10_k],
                                             threadgroup half (*inner_stage)[nk_shapes_depth_apple10_k],
                                             threadgroup float (*products)[nk_shapes_points_apple10_k],
                                             threadgroup float (*partials)[16], thread float *means,
                                             thread float *maxima) {
#pragma clang fp contract(off) reassociate(off)
    constexpr uint side = nk_shapes_points_apple10_k, depth = nk_shapes_depth_apple10_k;
    constexpr uint group_points = nk_shapes_group_points_apple10_k;
    uint const outer_count = nk_shapes_packed_header_f16_metal_(outer_packed)->points;
    uint const inner_count = nk_shapes_packed_header_f16_metal_(inner_packed)->points;
    uint outer_stride, inner_stride;
    device half const *outer_axes = nk_shapes_packed_axes_f16_metal_(outer_packed, outer_stride);
    device half const *inner_axes = nk_shapes_packed_axes_f16_metal_(inner_packed, inner_stride);
    float const inner_reach = nk_shapes_packed_header_f16_metal_(inner_packed)->max_radius * scale;
    // Two threads share each outer row, the first scanning columns 0-31, the second 32-63
    uint const row = thread_index / 2, half_index = thread_index % 2;

    constexpr auto descriptor = matmul2d_descriptor(side, side, depth, /*transpose_left=*/false,
                                                    /*transpose_right=*/true, /*relaxed_precision=*/false);
    matmul2d<descriptor, execution_simdgroups<4>> multiply;
    using stage_t = tensor<threadgroup half, extents<int32_t, depth, side>, tensor_inline>;
    stage_t outer_tile(&outer_stage[0][0], extents<int32_t, depth, side>()),
        inner_tile(&inner_stage[0][0], extents<int32_t, depth, side>());
    tensor<threadgroup float, extents<int32_t, side, side>, tensor_inline> products_tile(
        &products[0][0], extents<int32_t, side, side>());
    auto product =
        multiply.template get_destination_cooperative_tensor<decltype(outer_tile), decltype(inner_tile), float>();

    float const count = float(outer_count);
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool stopped = false;
    for (uint outer_first = 0; outer_first < outer_count && !stopped; outer_first += side) {
        float flip_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0}, flip_maxima[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        half coordinates[3], norm_high, norm_low;
        float norm;
        if (thread_index < side) {
            nk_shapes_point_f16_apple10_(outer_axes, outer_stride, outer_first + thread_index, outer_count, scale,
                                         coordinates, norm, norm_high, norm_low);
            half const row_values[depth] = {coordinates[0], coordinates[1], coordinates[2], norm_high, norm_low, 1, 1};
            for (uint value = 0; value != depth; ++value) outer_stage[thread_index][value] = row_values[value];
        }
        // Each thread's tolerance follows the norm of the row it scans, not the one it staged
        nk_shapes_point_f16_apple10_(outer_axes, outer_stride, outer_first + row, outer_count, scale, coordinates, norm,
                                     norm_high, norm_low);
        // Rounding of the 7-term expansion and the split norms, with room for the accelerators
        float const tolerance = 0x1p-19f * (norm + inner_reach * inner_reach);
        float least[8], others[8];
        uint groups[8];
        for (uint flip = 0; flip != 8; ++flip) least[flip] = INFINITY, others[flip] = INFINITY, groups[flip] = 0;

        for (uint inner_first = 0; inner_first < inner_count; inner_first += side) {
            half inner_coordinates[3], inner_high, inner_low;
            float inner_norm;
            if (thread_index < side)
                nk_shapes_point_f16_apple10_(inner_axes, inner_stride, inner_first + thread_index, inner_count, scale,
                                             inner_coordinates, inner_norm, inner_high, inner_low);
            uint const valid = min(inner_count - inner_first, side);
            for (uint flip = 0; flip != 8; ++flip) {
                if (thread_index < side) {
                    // A flip negates an inner axis, and doubling a half is exact
                    half const column_values[depth] = {
                        (flip & 1) ? half(2) * inner_coordinates[0] : half(-2) * inner_coordinates[0],
                        (flip & 2) ? half(2) * inner_coordinates[1] : half(-2) * inner_coordinates[1],
                        (flip & 4) ? half(2) * inner_coordinates[2] : half(-2) * inner_coordinates[2],
                        1,
                        1,
                        inner_high,
                        inner_low};
                    for (uint value = 0; value != depth; ++value)
                        inner_stage[thread_index][value] = column_values[value];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                multiply.run(outer_tile, inner_tile, product);
                product.store(products_tile);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint local = 0; local != side / 2 / group_points; ++local) {
                    uint const first_column = half_index * (side / 2) + local * group_points;
                    float group_least = INFINITY;
                    for (uint point = 0; point != group_points; ++point)
                        if (first_column + point < valid)
                            group_least = fmin(fmax(products[row][first_column + point], 0.0f), group_least);
                    bool const below = group_least < least[flip];
                    others[flip] = fmin(below ? least[flip] : group_least, others[flip]);
                    groups[flip] = below ? (inner_first + first_column) / group_points : groups[flip];
                    least[flip] = below ? group_least : least[flip];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // The two halves of a row meet: the losing half's least becomes one of the other groups
        for (uint flip = 0; flip != 8; ++flip) {
            float const partner_least = simd_shuffle_xor(least[flip], 1);
            float const partner_others = simd_shuffle_xor(others[flip], 1);
            uint const partner_group = simd_shuffle_xor(groups[flip], 1);
            bool const kept = least[flip] < partner_least || (least[flip] == partner_least && half_index == 0);
            others[flip] = fmin(fmin(others[flip], partner_others), kept ? partner_least : least[flip]);
            groups[flip] = kept ? groups[flip] : partner_group;
            least[flip] = kept ? least[flip] : partner_least;
        }
        uint const point = outer_first + row;
        if (point < outer_count) {
            float const outer_x = outer_axes[point], outer_y = outer_axes[outer_stride + point],
                        outer_z = outer_axes[2 * outer_stride + point];
            // Each half finishes four flips, in the difference form of the CPU capabilities
            for (uint flip = half_index * 4; flip != half_index * 4 + 4; ++flip) {
                bool const rescan = others[flip] <= least[flip] + 2 * tolerance;
                uint const begin = rescan ? 0 : groups[flip] * group_points;
                uint const end = rescan ? inner_count : min(begin + group_points, inner_count);
                float const distance = precise::sqrt(nk_shapes_nearest_square_f16_metal_(
                    outer_x, outer_y, outer_z, inner_axes, inner_stride, begin, end, flip));
                flip_sums[flip] += distance, flip_maxima[flip] = fmax(distance, flip_maxima[flip]);
            }
        }

        for (uint flip = 0; flip != 8; ++flip) {
            float const simdgroup_sum = simd_sum(flip_sums[flip]), simdgroup_maximum = simd_max(flip_maxima[flip]);
            if (lane == 0) partials[simdgroup][flip] = simdgroup_sum, partials[simdgroup][8 + flip] = simdgroup_maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float reached = INFINITY;
        for (uint flip = 0; flip != 8; ++flip) {
            sums[flip] += ((partials[0][flip] + partials[1][flip]) + partials[2][flip]) + partials[3][flip];
            maxima[flip] = fmax(fmax(maxima[flip], fmax(partials[0][8 + flip], partials[1][8 + flip])),
                                fmax(partials[2][8 + flip], partials[3][8 + flip]));
            float const mean = (means[flip] + precise::divide(sums[flip], count)) / 2;
            reached = fmin(hausdorff_ ? maxima[flip] : mean, reached);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stopped = reached > limit;
    }
    for (uint flip = 0; flip != 8; ++flip) means[flip] += precise::divide(sums[flip], count);
    return stopped;
}

/** One threadgroup's pair of a many-to-many launch, numbered row-major. */
template <bool hausdorff_>
inline void nk_shapes_packed_f16_apple10_(device ulong const *a_table, device ulong const *b_table,
                                          device uchar *results, device float *rotations,
                                          constant nk_shapes_arguments_metal_t &arguments, ulong pair,
                                          uint thread_index, uint simdgroup, uint lane,
                                          threadgroup half (*outer_stage)[nk_shapes_depth_apple10_k],
                                          threadgroup half (*inner_stage)[nk_shapes_depth_apple10_k],
                                          threadgroup float (*products)[nk_shapes_points_apple10_k],
                                          threadgroup float (*partials)[16]) {
#pragma clang fp contract(off) reassociate(off)
    if (pair >= arguments.rows * arguments.columns) return;
    ulong const row = pair / arguments.columns, column = pair % arguments.columns;
    device uchar const *row_packed = nk_shapes_pack_metal_(a_table[row], arguments);
    device uchar const *column_packed = nk_shapes_pack_metal_(b_table[column], arguments);
    device nk_shapes_packed_header_f32_metal_t const *row_header = nk_shapes_packed_header_f16_metal_(row_packed);
    device nk_shapes_packed_header_f32_metal_t const *column_header = nk_shapes_packed_header_f16_metal_(column_packed);
    uint const rows = row_header->points, columns = column_header->points;
    float const gap = nk_hausdorff_bound_packed_f16_metal_(row_packed, column_packed);
    float flip_distances[8];
    if (!rows || !columns)
        for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = rows == columns ? 0 : INFINITY;
    else if (hausdorff_ && gap > arguments.limit)
        for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = gap;
    else {
        float const scale = nk_shapes_scale_apple10_(fmax(row_header->max_radius, column_header->max_radius));
        float means[8] = {0, 0, 0, 0, 0, 0, 0, 0}, maxima[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (!nk_shapes_direction_f16_apple10_<hausdorff_>(row_packed, column_packed, scale, arguments.limit,
                                                          thread_index, simdgroup, lane, outer_stage, inner_stage,
                                                          products, partials, means, maxima))
            nk_shapes_direction_f16_apple10_<hausdorff_>(column_packed, row_packed, scale, arguments.limit,
                                                         thread_index, simdgroup, lane, outer_stage, inner_stage,
                                                         products, partials, means, maxima);
        for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = hausdorff_ ? maxima[flip] : means[flip] / 2;
    }
    if (thread_index == 0)
        nk_shapes_write_best_flip_metal_(results, rotations, arguments, row, column, row_packed, column_packed,
                                         flip_distances);
}

kernel void nk_chamfers_packed_f16_apple10_kernel_(
    device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
    device uchar *results [[buffer(2)]], device float *rotations [[buffer(3)]],
    constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half outer_stage[nk_shapes_points_apple10_k][nk_shapes_depth_apple10_k];
    threadgroup half inner_stage[nk_shapes_points_apple10_k][nk_shapes_depth_apple10_k];
    threadgroup float products[nk_shapes_points_apple10_k][nk_shapes_points_apple10_k];
    threadgroup float partials[4][16];
    nk_shapes_packed_f16_apple10_<false>(a_table, b_table, results, rotations, arguments,
                                         ulong(group.y) * groups.x + group.x, thread_index, simdgroup, lane,
                                         outer_stage, inner_stage, products, partials);
}

kernel void nk_hausdorffs_packed_f16_apple10_kernel_(
    device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
    device uchar *results [[buffer(2)]], device float *rotations [[buffer(3)]],
    constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half outer_stage[nk_shapes_points_apple10_k][nk_shapes_depth_apple10_k];
    threadgroup half inner_stage[nk_shapes_points_apple10_k][nk_shapes_depth_apple10_k];
    threadgroup float products[nk_shapes_points_apple10_k][nk_shapes_points_apple10_k];
    threadgroup float partials[4][16];
    nk_shapes_packed_f16_apple10_<true>(a_table, b_table, results, rotations, arguments,
                                        ulong(group.y) * groups.x + group.x, thread_index, simdgroup, lane, outer_stage,
                                        inner_stage, products, partials);
}

#pragma endregion Chamfer and Hausdorff
