/**
 *  @file include/numkong/shapes/metal.metal
 *  @author Ash Vardanian
 *  @date October 24, 2026
 *  @brief Chamfer and Hausdorff distances between F16 point clouds, on the SIMT cores that every
 *      Apple GPU has.
 *
 *  @sa include/numkong/shapes/metal.h, which embeds and launches this source
 *  @sa include/numkong/shapes/cuda.cuh, the CUDA sibling
 *
 *  A pack is read where it lies: its header, then the x, y and z arrays of @c half, each on a
 *  64-byte boundary. A simdgroup scores one pair, its lanes splitting each cloud's points, and
 *  every square is formed in the serial order with contraction off, so each nearest square equals
 *  the serial one; only the sums over points run in another order.
 *
 *  The simdgroup bodies take packs by device address, so a kernel compiled with this source ahead
 *  of its own, like a graph walk scoring one query against many candidates, scores pairs as the
 *  many-to-many kernels do. Those kernels read the caller's tables of host addresses and translate
 *  each through the allocations the launcher resolved, as the host address of a shared buffer is
 *  never the address the GPU reads it at.
 */

/** Byte boundary of a pack's points and of each of its axes, as @c nk_shapes_alignment_k_. */
constant uint nk_shapes_alignment_metal_k = 64;

/** Most allocations the packs of one many-to-many launch may span. */
constant uint nk_shapes_blocks_metal_k = 64;

/** The opening of every pack of @c f16 points, laid out as @c nk_shapes_packed_header_f32_t. */
struct nk_shapes_packed_header_f32_metal_t {
    uint points;
    float centroid[3];
    float frame[4];
    float scale;
    float spreads[3];
    float max_radius;
};

/** One allocation the launcher resolved: the host bytes it spans and its GPU address. */
struct nk_shapes_block_metal_t {
    ulong host_begin, host_end, device_begin;
};

/** The launch record of the many-to-many kernels, laid out as the launchers' record of the same
 *  name. */
struct nk_shapes_arguments_metal_t {
    ulong rows, columns, results_stride;
    uint blocks, rotations;
    nk_shapes_block_metal_t block_list[nk_shapes_blocks_metal_k];
};

#pragma region Packs

/** The header of a pack of F16 points. */
inline device nk_shapes_packed_header_f32_metal_t const *nk_shapes_packed_header_f16_metal_(
    device uchar const *packed) {
    return (device nk_shapes_packed_header_f32_metal_t const *)packed;
}

/** The x array of a pack of F16 points, with the elements from each axis to the next. */
inline device half const *nk_shapes_packed_axes_f16_metal_(device uchar const *packed, thread uint &axis_stride) {
    uint const points = nk_shapes_packed_header_f16_metal_(packed)->points;
    axis_stride = nk::round_up_to_multiple(points * 2, nk_shapes_alignment_metal_k) / 2;
    return (device half const *)(packed + nk::round_up_to_multiple((uint)sizeof(nk_shapes_packed_header_f32_metal_t),
                                                                   nk_shapes_alignment_metal_k));
}

/** The device address of the pack at host address @p host , through the allocations the launcher
 *  resolved, every pack lying in one of them. */
inline device uchar const *nk_shapes_pack_metal_(ulong host, constant nk_shapes_arguments_metal_t &arguments) {
    uint block = 0;
    while (host - arguments.block_list[block].host_begin >=
           arguments.block_list[block].host_end - arguments.block_list[block].host_begin)
        ++block;
    return reinterpret_cast<device uchar const *>(host - arguments.block_list[block].host_begin +
                                                  arguments.block_list[block].device_begin);
}

/** The row-major rotation of the unit quaternion (x, y, z, w), as
 *  @c nk_quaternion_to_mat3x3_f32_serial_ forms it. */
inline void nk_quaternion_to_mat3x3_f32_metal_(device float const *quaternion, thread float *matrix) {
#pragma clang fp contract(off) reassociate(off)
    float const w = quaternion[3], x = quaternion[0], y = quaternion[1], z = quaternion[2];
    float const xx = x * x, yy = y * y, zz = z * z, xz = x * z, xy = x * y, yz = y * z;
    float const wx = w * x, wy = w * y, wz = w * z;
    matrix[0] = 1 - 2 * (yy + zz), matrix[1] = 2 * (xy - wz), matrix[2] = 2 * (xz + wy);
    matrix[3] = 2 * (xy + wz), matrix[4] = 1 - 2 * (xx + zz), matrix[5] = 2 * (yz - wx);
    matrix[6] = 2 * (xz - wy), matrix[7] = 2 * (yz + wx), matrix[8] = 1 - 2 * (xx + yy);
}

/** Writes the best of the 8 flip distances and, when the launch asks, its rotation 𝑄ₐᵀ × S × 𝑄ᵦ,
 *  or the identity for an empty cloud, as @c nk_shapes_best_flip_f32_serial_ does. */
inline void nk_shapes_write_best_flip_metal_(device uchar *results, device float *rotations,
                                             constant nk_shapes_arguments_metal_t &arguments, ulong row, ulong column,
                                             device uchar const *row_packed, device uchar const *column_packed,
                                             thread float const *flip_distances) {
#pragma clang fp contract(off) reassociate(off)
    uint best_flip = 0;
    for (uint flip = 1; flip != 8; ++flip)
        if (flip_distances[flip] < flip_distances[best_flip]) best_flip = flip;
    device float *results_row = (device float *)(results + row * arguments.results_stride);
    results_row[column] = flip_distances[best_flip];
    if (!arguments.rotations) return;
    bool const clouds = nk_shapes_packed_header_f16_metal_(row_packed)->points &&
                        nk_shapes_packed_header_f16_metal_(column_packed)->points;
    device float *rotation = rotations + (row * arguments.columns + column) * 9;
    float row_axes[9], column_axes[9], inner_column[9];
    nk_quaternion_to_mat3x3_f32_metal_(nk_shapes_packed_header_f16_metal_(row_packed)->frame, row_axes);
    nk_quaternion_to_mat3x3_f32_metal_(nk_shapes_packed_header_f16_metal_(column_packed)->frame, column_axes);
    float const signs[3] = {(best_flip & 1) ? -1.0f : 1.0f, (best_flip & 2) ? -1.0f : 1.0f,
                            (best_flip & 4) ? -1.0f : 1.0f};
    // The serial S × 𝑄ᵦ sums two exact zeros into each product, which leaves it unrounded
    for (uint cell = 0; cell != 9; ++cell) inner_column[cell] = signs[cell / 3] * column_axes[cell];
    for (uint output_row = 0; output_row != 3; ++output_row)
        for (uint output_column = 0; output_column != 3; ++output_column)
            rotation[output_row * 3 + output_column] =
                !clouds ? (output_row == output_column ? 1.0f : 0.0f)
                        : row_axes[output_row] * inner_column[output_column] +
                              row_axes[3 + output_row] * inner_column[3 + output_column] +
                              row_axes[6 + output_row] * inner_column[6 + output_column];
}

#pragma endregion Packs

#pragma region Chamfer and Hausdorff

/** The least squared distance from the outer point to the inner points under each of the 8 axis
 *  flips, widened to F32 and rounded in the serial order, so each equals the serial one. */
inline void nk_shapes_nearest_squares_f16_metal_(float outer_x, float outer_y, float outer_z,
                                                 device half const *inner_axes, uint inner_stride, uint inner_count,
                                                 thread float *nearest) {
#pragma clang fp contract(off) reassociate(off)
    for (uint flip = 0; flip != 8; ++flip) nearest[flip] = INFINITY;
    for (uint index = 0; index != inner_count; ++index) {
        float const inner_x = inner_axes[index], inner_y = inner_axes[inner_stride + index],
                    inner_z = inner_axes[2 * inner_stride + index];
        // A flip negates an inner axis, turning its difference into a sum
        float const minus_x = outer_x - inner_x, plus_x = outer_x + inner_x;
        float const minus_y = outer_y - inner_y, plus_y = outer_y + inner_y;
        float const minus_z = outer_z - inner_z, plus_z = outer_z + inner_z;
        float const minus_xx = minus_x * minus_x, plus_xx = plus_x * plus_x;
        float const minus_yy = minus_y * minus_y, plus_yy = plus_y * plus_y;
        float const minus_zz = minus_z * minus_z, plus_zz = plus_z * plus_z;
        float const planar[4] = {minus_xx + minus_yy, plus_xx + minus_yy, minus_xx + plus_yy, plus_xx + plus_yy};
        for (uint flip = 0; flip != 8; ++flip)
            nearest[flip] = fmin(planar[flip & 3] + ((flip & 4) ? plus_zz : minus_zz), nearest[flip]);
    }
}

/** The least squared distance under @p flip from the outer point to the inner points from @p begin
 *  to @p end , rounded as @c nk_shapes_nearest_squares_f16_metal_ rounds each. */
inline float nk_shapes_nearest_square_f16_metal_(float outer_x, float outer_y, float outer_z,
                                                 device half const *inner_axes, uint inner_stride, uint begin, uint end,
                                                 uint flip) {
#pragma clang fp contract(off) reassociate(off)
    float least = INFINITY;
    for (uint index = begin; index < end; ++index) {
        float const inner_x = inner_axes[index], inner_y = inner_axes[inner_stride + index],
                    inner_z = inner_axes[2 * inner_stride + index];
        float const x = (flip & 1) ? outer_x + inner_x : outer_x - inner_x;
        float const y = (flip & 2) ? outer_y + inner_y : outer_y - inner_y;
        float const z = (flip & 4) ? outer_z + inner_z : outer_z - inner_z;
        least = fmin((x * x + y * y) + z * z, least);
    }
    return least;
}

/** A stride coprime with @p count, so stepping by it visits each of the @p count points once and
 *  leaves the runs of neighbors a molecule's point order holds, reaching near points sooner. */
inline uint nk_shapes_visit_stride_metal_(uint count) {
    uint stride = (count * 5 / 8) | 1;
    while (true) {
        uint first = stride, second = count;
        while (second != 0) {
            uint const remainder = first % second;
            first = second, second = remainder;
        }
        if (first == 1) return stride;
        stride += 2;
    }
}

/** @c nk_shapes_nearest_squares_f16_metal_ for Hausdorff, stopping once no flip's nearest square
 *  can raise its @p farthest . A flip that stopped early holds an upper bound of its nearest
 *  square, which already lies below @p farthest , so the largest nearest square keeps its bits. */
inline void nk_shapes_nearest_squares_below_f16_metal_(float outer_x, float outer_y, float outer_z,
                                                       device half const *inner_axes, uint inner_stride,
                                                       uint inner_count, uint visit_stride,
                                                       thread float const *farthest, thread float *nearest) {
#pragma clang fp contract(off) reassociate(off)
    for (uint flip = 0; flip != 8; ++flip) nearest[flip] = INFINITY;
    uint index = 0;
    for (uint visit = 0; visit != inner_count; ++visit) {
        float const inner_x = inner_axes[index], inner_y = inner_axes[inner_stride + index],
                    inner_z = inner_axes[2 * inner_stride + index];
        float const minus_x = outer_x - inner_x, plus_x = outer_x + inner_x;
        float const minus_y = outer_y - inner_y, plus_y = outer_y + inner_y;
        float const minus_z = outer_z - inner_z, plus_z = outer_z + inner_z;
        float const minus_xx = minus_x * minus_x, plus_xx = plus_x * plus_x;
        float const minus_yy = minus_y * minus_y, plus_yy = plus_y * plus_y;
        float const minus_zz = minus_z * minus_z, plus_zz = plus_z * plus_z;
        float const planar[4] = {minus_xx + minus_yy, plus_xx + minus_yy, minus_xx + plus_yy, plus_xx + plus_yy};
        for (uint flip = 0; flip != 8; ++flip)
            nearest[flip] = fmin(planar[flip & 3] + ((flip & 4) ? plus_zz : minus_zz), nearest[flip]);
        index += visit_stride, index = index >= inner_count ? index - inner_count : index;
        if ((visit & 7) == 7) {
            bool settled = true;
            for (uint flip = 0; flip != 8; ++flip) settled = settled && nearest[flip] <= farthest[flip];
            if (settled) return;
        }
    }
}

/** A lower bound of the Hausdorff distance between two packs of F16 points from their headers
 *  alone: the gap between their largest radii. */
inline float nk_hausdorff_bound_packed_f16_metal_(device uchar const *a_packed, device uchar const *b_packed) {
    return fabs(nk_shapes_packed_header_f16_metal_(a_packed)->max_radius -
                nk_shapes_packed_header_f16_metal_(b_packed)->max_radius);
}

/** The Chamfer distance between two packs of F16 points on a whole simdgroup, as
 *  @c nk_chamfer_warp_packed_f16_cuda_ computes it. Every lane calls with the same packs and its
 *  own @p lane , the lanes split each cloud's points, and the simdgroup sums their nearest
 *  distances. Distances up to @p limit are exact; past it, the simdgroup may stop early and return
 *  a lower bound that also lies past @p limit . */
inline float nk_chamfer_simdgroup_packed_f16_metal_(device uchar const *a_packed, device uchar const *b_packed,
                                                    float limit, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    uint const a_count = nk_shapes_packed_header_f16_metal_(a_packed)->points;
    uint const b_count = nk_shapes_packed_header_f16_metal_(b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : INFINITY;
    uint a_stride, b_stride;
    device half const *a_axes = nk_shapes_packed_axes_f16_metal_(a_packed, a_stride);
    device half const *b_axes = nk_shapes_packed_axes_f16_metal_(b_packed, b_stride);
    // Per flip, the finished directions' mean distances: F/n after one, F/n + B/m after both
    float means[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint direction = 0; direction != 2; ++direction) {
        device half const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        uint const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        uint const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        float const count = float(outer_count);
        float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint first = 0; first < outer_count; first += 32) {
            uint const point = first + lane;
            float distances[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (point < outer_count) {
                nk_shapes_nearest_squares_f16_metal_(outer_axes[point], outer_axes[outer_stride + point],
                                                     outer_axes[2 * outer_stride + point], inner_axes, inner_stride,
                                                     inner_count, distances);
                for (uint flip = 0; flip != 8; ++flip) distances[flip] = precise::sqrt(distances[flip]);
            }
            bool exceeded = true;
            for (uint flip = 0; flip != 8; ++flip) {
                sums[flip] += simd_sum(distances[flip]);
                exceeded &= sums[flip] > (2 * limit - means[flip]) * count;
            }
            if (!exceeded) continue;
            float partial = INFINITY;
            for (uint flip = 0; flip != 8; ++flip)
                partial = fmin((means[flip] + precise::divide(sums[flip], count)) / 2, partial);
            if (partial > limit) return partial;
        }
        for (uint flip = 0; flip != 8; ++flip) means[flip] += precise::divide(sums[flip], count);
    }
    float best = means[0] / 2;
    for (uint flip = 1; flip != 8; ++flip) best = fmin(means[flip] / 2, best);
    return best;
}

/** The Hausdorff distance between two packs of F16 points on a whole simdgroup, as
 *  @c nk_hausdorff_warp_packed_f16_cuda_ computes it. Every lane calls with the same packs and its
 *  own @p lane , the lanes split each cloud's points, and the simdgroup keeps the farthest nearest
 *  square. Distances up to @p limit are exact; past it, the simdgroup may return early, even before
 *  reading a point, with a lower bound that also lies past @p limit . */
inline float nk_hausdorff_simdgroup_packed_f16_metal_(device uchar const *a_packed, device uchar const *b_packed,
                                                      float limit, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    uint const a_count = nk_shapes_packed_header_f16_metal_(a_packed)->points;
    uint const b_count = nk_shapes_packed_header_f16_metal_(b_packed)->points;
    if (!a_count || !b_count) return a_count == b_count ? 0 : INFINITY;
    float const bound = nk_hausdorff_bound_packed_f16_metal_(a_packed, b_packed);
    if (bound > limit) return bound;
    uint a_stride, b_stride;
    device half const *a_axes = nk_shapes_packed_axes_f16_metal_(a_packed, a_stride);
    device half const *b_axes = nk_shapes_packed_axes_f16_metal_(b_packed, b_stride);
    float const limit_squared = limit * limit;
    uint const a_visits = nk_shapes_visit_stride_metal_(a_count), b_visits = nk_shapes_visit_stride_metal_(b_count);
    float farthest[8] = {0, 0, 0, 0, 0, 0, 0, 0}, nearest[8];
    for (uint direction = 0; direction != 2; ++direction) {
        device half const *outer_axes = direction ? b_axes : a_axes, *inner_axes = direction ? a_axes : b_axes;
        uint const outer_stride = direction ? b_stride : a_stride, inner_stride = direction ? a_stride : b_stride;
        uint const outer_count = direction ? b_count : a_count, inner_count = direction ? a_count : b_count;
        uint const inner_visits = direction ? a_visits : b_visits;
        for (uint first = 0; first < outer_count; first += 32) {
            uint const point = first + lane;
            if (point < outer_count) {
                nk_shapes_nearest_squares_below_f16_metal_(outer_axes[point], outer_axes[outer_stride + point],
                                                           outer_axes[2 * outer_stride + point], inner_axes,
                                                           inner_stride, inner_count, inner_visits, farthest, nearest);
                for (uint flip = 0; flip != 8; ++flip) farthest[flip] = fmax(nearest[flip], farthest[flip]);
            }
            float least = INFINITY;
            for (uint flip = 0; flip != 8; ++flip) {
                farthest[flip] = simd_max(farthest[flip]);
                least = fmin(farthest[flip], least);
            }
            if (least > limit_squared && precise::sqrt(least) > limit) return precise::sqrt(least);
        }
    }
    float least = farthest[0];
    for (uint flip = 1; flip != 8; ++flip) least = fmin(farthest[flip], least);
    return precise::sqrt(least);
}

/** Per flip, the sum of one direction's nearest distances over a whole simdgroup, in every lane. */
inline void nk_shapes_direction_simdgroup_f16_metal_(device uchar const *outer_packed, device uchar const *inner_packed,
                                                     uint lane, thread float *sums) {
#pragma clang fp contract(off) reassociate(off)
    uint const outer_count = nk_shapes_packed_header_f16_metal_(outer_packed)->points;
    uint const inner_count = nk_shapes_packed_header_f16_metal_(inner_packed)->points;
    uint outer_stride, inner_stride;
    device half const *outer_axes = nk_shapes_packed_axes_f16_metal_(outer_packed, outer_stride);
    device half const *inner_axes = nk_shapes_packed_axes_f16_metal_(inner_packed, inner_stride);
    for (uint flip = 0; flip != 8; ++flip) sums[flip] = 0;
    for (uint point = lane; point < outer_count; point += 32) {
        float nearest[8];
        nk_shapes_nearest_squares_f16_metal_(outer_axes[point], outer_axes[outer_stride + point],
                                             outer_axes[2 * outer_stride + point], inner_axes, inner_stride,
                                             inner_count, nearest);
        for (uint flip = 0; flip != 8; ++flip) sums[flip] += precise::sqrt(nearest[flip]);
    }
    for (uint flip = 0; flip != 8; ++flip) sums[flip] = simd_sum(sums[flip]);
}

/** Raises each flip's @p farthest to the largest nearest square of one direction over a whole
 *  simdgroup, in every lane. The direction before it only tightens the break, as both directions
 *  end in one maximum. */
inline void nk_shapes_farthest_simdgroup_f16_metal_(device uchar const *outer_packed, device uchar const *inner_packed,
                                                    uint lane, thread float *farthest) {
#pragma clang fp contract(off) reassociate(off)
    uint const outer_count = nk_shapes_packed_header_f16_metal_(outer_packed)->points;
    uint const inner_count = nk_shapes_packed_header_f16_metal_(inner_packed)->points;
    uint outer_stride, inner_stride;
    device half const *outer_axes = nk_shapes_packed_axes_f16_metal_(outer_packed, outer_stride);
    device half const *inner_axes = nk_shapes_packed_axes_f16_metal_(inner_packed, inner_stride);
    uint const inner_visits = nk_shapes_visit_stride_metal_(inner_count);
    for (uint first = 0; first < outer_count; first += 32) {
        uint const point = first + lane;
        if (point < outer_count) {
            float nearest[8];
            nk_shapes_nearest_squares_below_f16_metal_(outer_axes[point], outer_axes[outer_stride + point],
                                                       outer_axes[2 * outer_stride + point], inner_axes, inner_stride,
                                                       inner_count, inner_visits, farthest, nearest);
            for (uint flip = 0; flip != 8; ++flip) farthest[flip] = fmax(nearest[flip], farthest[flip]);
        }
        for (uint flip = 0; flip != 8; ++flip) farthest[flip] = simd_max(farthest[flip]);
    }
}

/** Every flip's Chamfer or Hausdorff distance of one pair on a whole simdgroup, or the empty-cloud
 *  convention, in every lane. */
template <bool hausdorff_>
inline void nk_shapes_flips_simdgroup_f16_metal_(device uchar const *row_packed, device uchar const *column_packed,
                                                 uint lane, thread float *flip_distances) {
#pragma clang fp contract(off) reassociate(off)
    uint const rows = nk_shapes_packed_header_f16_metal_(row_packed)->points;
    uint const columns = nk_shapes_packed_header_f16_metal_(column_packed)->points;
    if (!rows || !columns) {
        for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = rows == columns ? 0 : INFINITY;
        return;
    }
    if constexpr (hausdorff_) {
        float farthest[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        nk_shapes_farthest_simdgroup_f16_metal_(row_packed, column_packed, lane, farthest);
        nk_shapes_farthest_simdgroup_f16_metal_(column_packed, row_packed, lane, farthest);
        for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = precise::sqrt(farthest[flip]);
    }
    else {
        float forward_sums[8], backward_sums[8];
        nk_shapes_direction_simdgroup_f16_metal_(row_packed, column_packed, lane, forward_sums);
        nk_shapes_direction_simdgroup_f16_metal_(column_packed, row_packed, lane, backward_sums);
        for (uint flip = 0; flip != 8; ++flip)
            flip_distances[flip] = (precise::divide(forward_sums[flip], float(rows)) +
                                    precise::divide(backward_sums[flip], float(columns))) /
                                   2;
    }
}

/** One simdgroup's pair of a many-to-many launch, numbered row-major. */
template <bool hausdorff_>
inline void nk_shapes_packed_f16_metal_(device ulong const *a_table, device ulong const *b_table, device uchar *results,
                                        device float *rotations, constant nk_shapes_arguments_metal_t &arguments,
                                        ulong pair, uint lane) {
    if (pair >= arguments.rows * arguments.columns) return;
    ulong const row = pair / arguments.columns, column = pair % arguments.columns;
    device uchar const *row_packed = nk_shapes_pack_metal_(a_table[row], arguments);
    device uchar const *column_packed = nk_shapes_pack_metal_(b_table[column], arguments);
    float flip_distances[8];
    nk_shapes_flips_simdgroup_f16_metal_<hausdorff_>(row_packed, column_packed, lane, flip_distances);
    if (lane == 0)
        nk_shapes_write_best_flip_metal_(results, rotations, arguments, row, column, row_packed, column_packed,
                                         flip_distances);
}

kernel void nk_chamfers_packed_f16_metal_kernel_(
    device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
    device uchar *results [[buffer(2)]], device float *rotations [[buffer(3)]],
    constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong const pair = (ulong(group.y) * groups.x + group.x) * 4 + simdgroup;
    nk_shapes_packed_f16_metal_<false>(a_table, b_table, results, rotations, arguments, pair, lane);
}

kernel void nk_hausdorffs_packed_f16_metal_kernel_(
    device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
    device uchar *results [[buffer(2)]], device float *rotations [[buffer(3)]],
    constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong const pair = (ulong(group.y) * groups.x + group.x) * 4 + simdgroup;
    nk_shapes_packed_f16_metal_<true>(a_table, b_table, results, rotations, arguments, pair, lane);
}

#pragma endregion Chamfer and Hausdorff

/*  The McAdams SVD of @c nk_svd3x3_f32_serial_ , step for step, with the square roots of this
 *  GPU's own instructions. */
#pragma region SVD F32

inline float nk_sum_two_products_f32_metal_(float left_0, float right_0, float left_1, float right_1) {
#pragma clang fp contract(off) reassociate(off)
    return left_0 * right_0 + left_1 * right_1;
}

inline float nk_sum_three_products_f32_metal_(float left_0, float right_0, float left_1, float right_1, float left_2,
                                              float right_2) {
#pragma clang fp contract(off) reassociate(off)
    return left_0 * right_0 + left_1 * right_1 + left_2 * right_2;
}

inline void nk_approximate_givens_quaternion_f32_metal_(float a11, float a12, float a22, thread float &cos_half,
                                                        thread float &sin_half) {
#pragma clang fp contract(off) reassociate(off)
    cos_half = 2 * (a11 - a22), sin_half = a12;
    bool const use_givens = 5.828427124f * sin_half * sin_half < cos_half * cos_half;
    float const w = rsqrt(nk_sum_two_products_f32_metal_(cos_half, cos_half, sin_half, sin_half));
    cos_half = use_givens ? w * cos_half : 0.923879532f;
    sin_half = use_givens ? w * sin_half : 0.3826834323f;
}

inline void nk_jacobi_conjugation_f32_metal_(uint index_x, uint index_y, uint index_z, thread float *s,
                                             thread float *quaternion) {
#pragma clang fp contract(off) reassociate(off)
    // `s` holds s11, s21, s22, s31, s32, s33 in that order
    float cos_half, sin_half;
    nk_approximate_givens_quaternion_f32_metal_(s[0], s[1], s[2], cos_half, sin_half);
    float const scale = nk_sum_two_products_f32_metal_(cos_half, cos_half, sin_half, sin_half);
    float const cos_theta = nk_sum_two_products_f32_metal_(cos_half, cos_half, -sin_half, sin_half) / scale;
    float const sin_theta = (2 * sin_half * cos_half) / scale;
    float const s11 = s[0], s21 = s[1], s22 = s[2], s31 = s[3], s32 = s[4], s33 = s[5];
    float const left_11 = nk_sum_two_products_f32_metal_(cos_theta, s11, sin_theta, s21);
    float const left_12 = nk_sum_two_products_f32_metal_(cos_theta, s21, sin_theta, s22);
    float const left_21 = nk_sum_two_products_f32_metal_(-sin_theta, s11, cos_theta, s21);
    float const left_22 = nk_sum_two_products_f32_metal_(-sin_theta, s21, cos_theta, s22);
    float const new_11 = nk_sum_two_products_f32_metal_(cos_theta, left_11, sin_theta, left_12);
    float const new_21 = nk_sum_two_products_f32_metal_(cos_theta, left_21, sin_theta, left_22);
    float const new_22 = nk_sum_two_products_f32_metal_(-sin_theta, left_21, cos_theta, left_22);
    float const new_31 = nk_sum_two_products_f32_metal_(cos_theta, s31, sin_theta, s32);
    float const new_32 = nk_sum_two_products_f32_metal_(-sin_theta, s31, cos_theta, s32);
    float quaternion_sines[3];
    quaternion_sines[0] = quaternion[0] * sin_half;
    quaternion_sines[1] = quaternion[1] * sin_half;
    quaternion_sines[2] = quaternion[2] * sin_half;
    sin_half *= quaternion[3];
    quaternion[0] *= cos_half, quaternion[1] *= cos_half;
    quaternion[2] *= cos_half, quaternion[3] *= cos_half;
    quaternion[index_z] += sin_half, quaternion[3] -= quaternion_sines[index_z];
    quaternion[index_x] += quaternion_sines[index_y], quaternion[index_y] -= quaternion_sines[index_x];
    // The next sweep pivots on the following pair, so the matrix cycles its axes
    s[0] = new_22, s[1] = new_32, s[2] = s33, s[3] = new_21, s[4] = new_31, s[5] = new_11;
}

inline void nk_quaternion_to_mat3x3_f32_metal_(thread float const *quaternion, thread float *matrix) {
#pragma clang fp contract(off) reassociate(off)
    float const w = quaternion[3], x = quaternion[0], y = quaternion[1], z = quaternion[2];
    float const xx = x * x, yy = y * y, zz = z * z, xz = x * z, xy = x * y, yz = y * z;
    float const wx = w * x, wy = w * y, wz = w * z;
    matrix[0] = 1 - 2 * (yy + zz), matrix[1] = 2 * (xy - wz), matrix[2] = 2 * (xz + wy);
    matrix[3] = 2 * (xy + wz), matrix[4] = 1 - 2 * (xx + zz), matrix[5] = 2 * (yz - wx);
    matrix[6] = 2 * (xz - wy), matrix[7] = 2 * (yz + wx), matrix[8] = 1 - 2 * (xx + yy);
}

inline void nk_qr_givens_quaternion_f32_metal_(float a1, float a2, thread float &cos_half, thread float &sin_half) {
#pragma clang fp contract(off) reassociate(off)
    float const epsilon = 1e-6f;
    float const squares = a1 * a1 + a2 * a2;
    float rho = squares * rsqrt(squares);
    rho = squares > epsilon ? rho : 0;
    sin_half = rho > epsilon ? a2 : 0;
    cos_half = fabs(a1) + (rho > epsilon ? rho : epsilon);
    if (a1 < 0) {
        float const swapped = sin_half;
        sin_half = cos_half, cos_half = swapped;
    }
    float const w = rsqrt(cos_half * cos_half + sin_half * sin_half);
    cos_half *= w, sin_half *= w;
}

/** Swaps columns @p first and @p second of both 3 × 3 matrices, negating the column moved to
 *  @p second . */
inline void nk_negating_swap_columns_f32_metal_(thread float *b, thread float *v, uint first, uint second) {
    for (uint row = 0; row != 9; row += 3) {
        float const b_first = b[row + first], v_first = v[row + first];
        b[row + first] = b[row + second], b[row + second] = -b_first;
        v[row + first] = v[row + second], v[row + second] = -v_first;
    }
}

inline void nk_sort_singular_values_f32_metal_(thread float *b, thread float *v) {
#pragma clang fp contract(off) reassociate(off)
    float norm_0 = b[0] * b[0] + b[3] * b[3] + b[6] * b[6];
    float norm_1 = b[1] * b[1] + b[4] * b[4] + b[7] * b[7];
    float norm_2 = b[2] * b[2] + b[5] * b[5] + b[8] * b[8];
    if (norm_0 < norm_1) {
        nk_negating_swap_columns_f32_metal_(b, v, 0, 1);
        float const kept = norm_0;
        norm_0 = norm_1, norm_1 = kept;
    }
    if (norm_0 < norm_2) {
        nk_negating_swap_columns_f32_metal_(b, v, 0, 2);
        float const kept = norm_0;
        norm_0 = norm_2, norm_2 = kept;
    }
    if (norm_1 < norm_2) nk_negating_swap_columns_f32_metal_(b, v, 1, 2);
}

inline void nk_qr_orthogonal_factor_f32_metal_(thread float const *input, thread float *q) {
#pragma clang fp contract(off) reassociate(off)
    float cos_half_1, sin_half_1, cos_half_2, sin_half_2, cos_half_3, sin_half_3;
    nk_qr_givens_quaternion_f32_metal_(input[0], input[3], cos_half_1, sin_half_1);
    float cos_theta = 1 - 2 * sin_half_1 * sin_half_1;
    float sin_theta = 2 * cos_half_1 * sin_half_1;
    float const rotated_1 = cos_theta * input[1] + sin_theta * input[4];
    float const rotated_4 = -sin_theta * input[1] + cos_theta * input[4];
    float const rotated_7 = input[7];
    nk_qr_givens_quaternion_f32_metal_(input[0] * cos_theta + sin_theta * input[3], input[6], cos_half_2, sin_half_2);
    cos_theta = 1 - 2 * sin_half_2 * sin_half_2;
    sin_theta = 2 * cos_half_2 * sin_half_2;
    float const remaining_7 = -sin_theta * rotated_1 + cos_theta * rotated_7;
    nk_qr_givens_quaternion_f32_metal_(rotated_4, remaining_7, cos_half_3, sin_half_3);
    float const sin_half_1_squared = sin_half_1 * sin_half_1;
    float const sin_half_2_squared = sin_half_2 * sin_half_2;
    float const sin_half_3_squared = sin_half_3 * sin_half_3;
    q[0] = (-1 + 2 * sin_half_1_squared) * (-1 + 2 * sin_half_2_squared);
    q[1] = 4 * cos_half_2 * cos_half_3 * (-1 + 2 * sin_half_1_squared) * sin_half_2 * sin_half_3 +
           2 * cos_half_1 * sin_half_1 * (-1 + 2 * sin_half_3_squared);
    q[2] = 4 * cos_half_1 * cos_half_3 * sin_half_1 * sin_half_3 -
           2 * cos_half_2 * (-1 + 2 * sin_half_1_squared) * sin_half_2 * (-1 + 2 * sin_half_3_squared);
    q[3] = 2 * cos_half_1 * sin_half_1 * (1 - 2 * sin_half_2_squared);
    q[4] = -8 * cos_half_1 * cos_half_2 * cos_half_3 * sin_half_1 * sin_half_2 * sin_half_3 +
           (-1 + 2 * sin_half_1_squared) * (-1 + 2 * sin_half_3_squared);
    q[5] = -2 * cos_half_3 * sin_half_3 + 4 * sin_half_1 *
                                              (cos_half_3 * sin_half_1 * sin_half_3 +
                                               cos_half_1 * cos_half_2 * sin_half_2 * (-1 + 2 * sin_half_3_squared));
    q[6] = 2 * cos_half_2 * sin_half_2;
    q[7] = 2 * cos_half_3 * (1 - 2 * sin_half_2_squared) * sin_half_3;
    q[8] = (-1 + 2 * sin_half_2_squared) * (-1 + 2 * sin_half_3_squared);
}

/** The left factor, the singular values on the diagonal positions, and the right factor of the
 *  row-major @p a . */
inline void nk_svd3x3_f32_metal_(thread float const *a, thread float *svd_left, thread float *svd_diagonal,
                                 thread float *svd_right) {
#pragma clang fp contract(off) reassociate(off)
    float ata[9], quaternion[4] = {0, 0, 0, 1}, product[9];
    for (uint row = 0; row != 3; ++row)
        for (uint column = row; column != 3; ++column)
            ata[row * 3 + column] = nk_sum_three_products_f32_metal_(a[row], a[column], a[3 + row], a[3 + column],
                                                                     a[6 + row], a[6 + column]);
    float s[6] = {ata[0], ata[1], ata[4], ata[2], ata[5], ata[8]};
    for (uint iteration = 0; iteration != 16; ++iteration) {
        nk_jacobi_conjugation_f32_metal_(0, 1, 2, s, quaternion);
        nk_jacobi_conjugation_f32_metal_(1, 2, 0, s, quaternion);
        nk_jacobi_conjugation_f32_metal_(2, 0, 1, s, quaternion);
    }
    float const norm = rsqrt(
        nk_sum_two_products_f32_metal_(quaternion[0], quaternion[0], quaternion[1], quaternion[1]) +
        nk_sum_two_products_f32_metal_(quaternion[2], quaternion[2], quaternion[3], quaternion[3]));
    quaternion[0] *= norm, quaternion[1] *= norm, quaternion[2] *= norm, quaternion[3] *= norm;
    nk_quaternion_to_mat3x3_f32_metal_(quaternion, svd_right);
    for (uint row = 0; row != 3; ++row)
        for (uint column = 0; column != 3; ++column)
            product[row * 3 + column] = nk_sum_three_products_f32_metal_(a[row * 3], svd_right[column], a[row * 3 + 1],
                                                                         svd_right[3 + column], a[row * 3 + 2],
                                                                         svd_right[6 + column]);
    nk_sort_singular_values_f32_metal_(product, svd_right);
    float const squared_0 = product[0] * product[0] + product[3] * product[3] + product[6] * product[6];
    float const squared_1 = product[1] * product[1] + product[4] * product[4] + product[7] * product[7];
    float const squared_2 = product[2] * product[2] + product[5] * product[5] + product[8] * product[8];
    nk_qr_orthogonal_factor_f32_metal_(product, svd_left);
    svd_diagonal[0] = sqrt(squared_0), svd_diagonal[4] = sqrt(squared_1), svd_diagonal[8] = sqrt(squared_2);
}

inline float nk_det3x3_f32_metal_(thread float const *m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

inline void nk_rotation_from_svd_f32_metal_(thread float const *svd_left, thread float const *svd_right,
                                            thread float *rotation) {
    for (uint row = 0; row != 3; ++row)
        for (uint column = 0; column != 3; ++column)
            rotation[row * 3 + column] = nk_sum_three_products_f32_metal_(
                svd_right[row * 3], svd_left[column * 3], svd_right[row * 3 + 1], svd_left[column * 3 + 1],
                svd_right[row * 3 + 2], svd_left[column * 3 + 2]);
}

#pragma endregion SVD F32
