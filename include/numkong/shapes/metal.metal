/**
 *  @file include/numkong/shapes/metal.metal
 *  @author Ash Vardanian
 *  @date October 24, 2026
 *  @brief Chamfer, Hausdorff, superposed RMSD and exact transport of F16 point clouds, on the SIMT
 *      cores that every Apple GPU has.
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
 *
 *  Transport runs the serial network simplex on one simdgroup per pair, whose lanes price arcs and
 *  whose first lane pivots while the rest wait at no cost. These GPUs have no F64, so the state F64
 *  holds on other capabilities is 64-bit fixed point here, each pair's in its slot of the caller's
 *  workspace, where every sum is exact and a negative reduced cost is never rounding.
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
    float limit;
    uint largest;
    ulong pair_begin, pair_end, slot_bytes;
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

/** The first of the 8 flips holding the least distance, as @c nk_shapes_best_flip_f32_serial_
 *  picks. */
inline uint nk_shapes_best_flip_metal_(thread float const *flip_distances) {
    uint best_flip = 0;
    for (uint flip = 1; flip != 8; ++flip)
        if (flip_distances[flip] < flip_distances[best_flip]) best_flip = flip;
    return best_flip;
}

/** Writes the best of the 8 flip distances and, when the launch asks, its rotation 𝑄ₐᵀ × S × 𝑄ᵦ,
 *  or the identity for an empty cloud, as @c nk_shapes_best_flip_f32_serial_ does. */
inline void nk_shapes_write_best_flip_metal_(device uchar *results, device float *rotations,
                                             constant nk_shapes_arguments_metal_t &arguments, ulong row, ulong column,
                                             device uchar const *row_packed, device uchar const *column_packed,
                                             thread float const *flip_distances) {
#pragma clang fp contract(off) reassociate(off)
    uint const best_flip = nk_shapes_best_flip_metal_(flip_distances);
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
 *  alone: the gap between their largest radii, less 8 roundings of those radii, as each is stored 3
 *  off and the gap and the computed distance add 5. */
inline float nk_hausdorff_bound_packed_f16_metal_(device uchar const *a_packed, device uchar const *b_packed) {
#pragma clang fp contract(off) reassociate(off)
    float const a_radius = nk_shapes_packed_header_f16_metal_(a_packed)->max_radius;
    float const b_radius = nk_shapes_packed_header_f16_metal_(b_packed)->max_radius;
    float const slack = 8 * 0x1p-24f * (a_radius + b_radius);
    float const gap = fabs(a_radius - b_radius) - slack;
    return gap > 0 ? gap : 0;
}

/** Adds each flip's mean nearest distance of one direction to its @p means over a whole simdgroup,
 *  in every lane. Past @p limit , which only reductions decide so every lane leaves together, it
 *  stops with per-flip lower bounds and returns true. */
inline bool nk_shapes_direction_simdgroup_f16_metal_(device uchar const *outer_packed, device uchar const *inner_packed,
                                                     float limit, uint lane, thread float *means) {
#pragma clang fp contract(off) reassociate(off)
    uint const outer_count = nk_shapes_packed_header_f16_metal_(outer_packed)->points;
    uint const inner_count = nk_shapes_packed_header_f16_metal_(inner_packed)->points;
    uint outer_stride, inner_stride;
    device half const *outer_axes = nk_shapes_packed_axes_f16_metal_(outer_packed, outer_stride);
    device half const *inner_axes = nk_shapes_packed_axes_f16_metal_(inner_packed, inner_stride);
    float const count = float(outer_count);
    float sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool stopped = false;
    for (uint first = 0; first < outer_count && !stopped; first += 32) {
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
        stopped = partial > limit;
    }
    for (uint flip = 0; flip != 8; ++flip) means[flip] += precise::divide(sums[flip], count);
    return stopped;
}

/** Raises each flip's @p farthest to the largest nearest square of one direction over a whole
 *  simdgroup, in every lane. The direction before it only tightens the break, as both directions
 *  end in one maximum. Past @p limit , which only reductions decide so every lane leaves together,
 *  it stops with per-flip lower bounds and returns true. */
inline bool nk_shapes_farthest_simdgroup_f16_metal_(device uchar const *outer_packed, device uchar const *inner_packed,
                                                    float limit, uint lane, thread float *farthest) {
#pragma clang fp contract(off) reassociate(off)
    uint const outer_count = nk_shapes_packed_header_f16_metal_(outer_packed)->points;
    uint const inner_count = nk_shapes_packed_header_f16_metal_(inner_packed)->points;
    uint outer_stride, inner_stride;
    device half const *outer_axes = nk_shapes_packed_axes_f16_metal_(outer_packed, outer_stride);
    device half const *inner_axes = nk_shapes_packed_axes_f16_metal_(inner_packed, inner_stride);
    uint const inner_visits = nk_shapes_visit_stride_metal_(inner_count);
    float const limit_squared = limit * limit;
    for (uint first = 0; first < outer_count; first += 32) {
        uint const point = first + lane;
        if (point < outer_count) {
            float nearest[8];
            nk_shapes_nearest_squares_below_f16_metal_(outer_axes[point], outer_axes[outer_stride + point],
                                                       outer_axes[2 * outer_stride + point], inner_axes, inner_stride,
                                                       inner_count, inner_visits, farthest, nearest);
            for (uint flip = 0; flip != 8; ++flip) farthest[flip] = fmax(nearest[flip], farthest[flip]);
        }
        float least = INFINITY;
        for (uint flip = 0; flip != 8; ++flip) {
            farthest[flip] = simd_max(farthest[flip]);
            least = fmin(farthest[flip], least);
        }
        if (least > limit_squared && precise::sqrt(least) > limit) return true;
    }
    return false;
}

/** Every flip's Chamfer or Hausdorff distance of one pair on a whole simdgroup, or the empty-cloud
 *  convention, in every lane. Distances up to @p limit are exact; past it, the simdgroup may stop
 *  early, even before reading a point, leaving per-flip lower bounds whose least lies past it. */
template <bool hausdorff_>
inline void nk_shapes_flips_simdgroup_f16_metal_(device uchar const *row_packed, device uchar const *column_packed,
                                                 float limit, uint lane, thread float *flip_distances) {
#pragma clang fp contract(off) reassociate(off)
    uint const rows = nk_shapes_packed_header_f16_metal_(row_packed)->points;
    uint const columns = nk_shapes_packed_header_f16_metal_(column_packed)->points;
    if (!rows || !columns) {
        for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = rows == columns ? 0 : INFINITY;
        return;
    }
    if constexpr (hausdorff_) {
        float const gap = nk_hausdorff_bound_packed_f16_metal_(row_packed, column_packed);
        if (gap > limit) {
            for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = gap;
            return;
        }
        float farthest[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (!nk_shapes_farthest_simdgroup_f16_metal_(row_packed, column_packed, limit, lane, farthest))
            nk_shapes_farthest_simdgroup_f16_metal_(column_packed, row_packed, limit, lane, farthest);
        for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = precise::sqrt(farthest[flip]);
    }
    else {
        float means[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (!nk_shapes_direction_simdgroup_f16_metal_(row_packed, column_packed, limit, lane, means))
            nk_shapes_direction_simdgroup_f16_metal_(column_packed, row_packed, limit, lane, means);
        for (uint flip = 0; flip != 8; ++flip) flip_distances[flip] = means[flip] / 2;
    }
}

/** The Chamfer distance between two packs of F16 points on a whole simdgroup, as
 *  @c nk_chamfer_warp_packed_f16_cuda_ computes it. Every lane calls with the same packs and its
 *  own @p lane , the lanes split each cloud's points, and the simdgroup sums their nearest
 *  distances. Distances up to @p limit are exact; past it, the simdgroup may stop early and return
 *  a lower bound that also lies past @p limit . */
inline float nk_chamfer_simdgroup_packed_f16_metal_(device uchar const *a_packed, device uchar const *b_packed,
                                                    float limit, uint lane) {
    float flip_distances[8];
    nk_shapes_flips_simdgroup_f16_metal_<false>(a_packed, b_packed, limit, lane, flip_distances);
    return flip_distances[nk_shapes_best_flip_metal_(flip_distances)];
}

/** The Hausdorff distance between two packs of F16 points on a whole simdgroup, as
 *  @c nk_hausdorff_warp_packed_f16_cuda_ computes it. Every lane calls with the same packs and its
 *  own @p lane , the lanes split each cloud's points, and the simdgroup keeps the farthest nearest
 *  square. Distances up to @p limit are exact; past it, the simdgroup may return early, even before
 *  reading a point, with a lower bound that also lies past @p limit . */
inline float nk_hausdorff_simdgroup_packed_f16_metal_(device uchar const *a_packed, device uchar const *b_packed,
                                                      float limit, uint lane) {
    float flip_distances[8];
    nk_shapes_flips_simdgroup_f16_metal_<true>(a_packed, b_packed, limit, lane, flip_distances);
    return flip_distances[nk_shapes_best_flip_metal_(flip_distances)];
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
    nk_shapes_flips_simdgroup_f16_metal_<hausdorff_>(row_packed, column_packed, arguments.limit, lane, flip_distances);
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

#pragma region RMSD

/** Σ‖aᵢ − R × bᵢ‖² at the best rotation, folded into Ga + Gb − 2λ and clamped at zero, as
 *  @c nk_rmsds_folded_f32_serial_ computes it from the row-major M = Σ aᵢ × bᵢᵀ. */
inline float nk_rmsds_folded_f32_metal_(thread float const *cross_covariance, float norms_squared) {
    float const upper = norms_squared / 2;
    if (!(upper > 0)) return 0;
    float const inverse = 1 / upper;
    float const xx = cross_covariance[0] * inverse, xy = cross_covariance[1] * inverse;
    float const xz = cross_covariance[2] * inverse, yx = cross_covariance[3] * inverse;
    float const yy = cross_covariance[4] * inverse, yz = cross_covariance[5] * inverse;
    float const zx = cross_covariance[6] * inverse, zy = cross_covariance[7] * inverse;
    float const zz = cross_covariance[8] * inverse;
    float const xx2 = xx * xx, yy2 = yy * yy, zz2 = zz * zz, xy2 = xy * xy, yz2 = yz * yz;
    float const xz2 = xz * xz, yx2 = yx * yx, zy2 = zy * zy, zx2 = zx * zx;
    float const xz_plus_zx = xz + zx, yz_plus_zy = yz + zy, xy_plus_yx = xy + yx;
    float const yz_minus_zy = yz - zy, xz_minus_zx = xz - zx, xy_minus_yx = xy - yx;
    float const xx_plus_yy = xx + yy, xx_minus_yy = xx - yy;
    float const off_diagonal = xy2 + xz2 - yx2 - zx2;
    float const quadratic = -2 * (xx2 + yy2 + zz2 + xy2 + yx2 + xz2 + zx2 + yz2 + zy2);
    float const linear = 8 * (xx * yz * zy + yy * zx * xz + zz * xy * yx - xx * yy * zz - yz * zx * xy - zy * yx * xz);
    float const twice_minor = 2 * (yz * zy - yy * zz);
    float const squares = yy2 + zz2 - xx2 + yz2 + zy2;
    float const constant_term = off_diagonal * off_diagonal + (squares + twice_minor) * (squares - twice_minor) +
                                (-xz_plus_zx * yz_minus_zy + xy_minus_yx * (xx_minus_yy - zz)) *
                                    (-xz_minus_zx * yz_plus_zy + xy_minus_yx * (xx_minus_yy + zz)) +
                                (-xz_plus_zx * yz_plus_zy - xy_plus_yx * (xx_plus_yy - zz)) *
                                    (-xz_minus_zx * yz_minus_zy - xy_plus_yx * (xx_plus_yy + zz)) +
                                (xy_plus_yx * yz_plus_zy + xz_plus_zx * (xx_minus_yy + zz)) *
                                    (-xy_minus_yx * yz_minus_zy + xz_plus_zx * (xx_plus_yy + zz)) +
                                (xy_plus_yx * yz_minus_zy + xz_minus_zx * (xx_minus_yy - zz)) *
                                    (-xy_minus_yx * yz_plus_zy + xz_minus_zx * (xx_plus_yy - zz));
    float const k00 = xx + yy + zz, k01 = yz_minus_zy, k02 = -xz_minus_zx, k03 = xy_minus_yx;
    float const k11 = xx - yy - zz, k12 = xy_plus_yx, k13 = xz_plus_zx;
    float const k22 = -xx + yy - zz, k23 = yz_plus_zy, k33 = -xx - yy + zz;
    // λ never falls below the largest diagonal entry of K, which stops a rounded step running away
    float const floor_bound = fmax(fmax(k00, k11), fmax(k22, k33));
    // ‖M‖ sets the scale of K's eigenvalues, so the adjugate's rounding noise is that cubed
    float const frobenius = precise::sqrt(-quadratic / 2),
                noise = 64 * 0x1p-24f * 8 * frobenius * frobenius * frobenius;
    float eigenvalue = 1, slope = 0;
    for (uint iteration = 0; iteration != 50; ++iteration) {
        float const squared = eigenvalue * eigenvalue;
        float const cubic = (squared + quadratic) * eigenvalue;
        float const shifted = cubic + linear;
        float const derivative = 2 * squared * eigenvalue + cubic + shifted;
        // A repeated largest root leaves no slope, and the iterate already sits on it
        if (!(derivative > 0)) break;
        slope = derivative;
        float const step = (shifted * eigenvalue + constant_term) / derivative;
        eigenvalue -= step;
        if (eigenvalue < floor_bound) {
            eigenvalue = floor_bound;
            break;
        }
        if (!(fabs(step) > 1e-7f)) break;
    }
    // Newton lands within ε of a root as steep as an isotropic or planar frame's, so it stays
    bool refined = slope > 4 * frobenius * frobenius * frobenius;
    for (uint refinement = refined ? 2 : 0; refinement != 2; ++refinement) {
        float const a00 = k00 - eigenvalue, a11 = k11 - eigenvalue, a22 = k22 - eigenvalue, a33 = k33 - eigenvalue;
        float const s0 = a00 * a11 - k01 * k01, s1 = a00 * k12 - k01 * k02, s2 = a00 * k13 - k01 * k03;
        float const s3 = k01 * k12 - a11 * k02, s4 = k01 * k13 - a11 * k03, s5 = k02 * k13 - k12 * k03;
        float const c0 = k02 * k13 - k03 * k12, c1 = k02 * k23 - k03 * a22, c2 = k02 * a33 - k03 * k23;
        float const c3 = k12 * k23 - k13 * a22, c4 = k12 * a33 - k13 * k23, c5 = a22 * a33 - k23 * k23;
        float const adjugate[4][4] = {{a11 * c5 - k12 * c4 + k13 * c3, -k01 * c5 + k02 * c4 - k03 * c3,
                                       k13 * s5 - k23 * s4 + a33 * s3, -k12 * s5 + a22 * s4 - k23 * s3},
                                      {-k01 * c5 + k12 * c2 - k13 * c1, a00 * c5 - k02 * c2 + k03 * c1,
                                       -k03 * s5 + k23 * s2 - a33 * s1, k02 * s5 - a22 * s2 + k23 * s1},
                                      {k01 * c4 - a11 * c2 + k13 * c0, -a00 * c4 + k01 * c2 - k03 * c0,
                                       k03 * s4 - k13 * s2 + a33 * s0, -k02 * s4 + k12 * s2 - k23 * s0},
                                      {-k01 * c3 + a11 * c1 - k12 * c0, a00 * c3 - k01 * c1 + k02 * c0,
                                       -k03 * s3 + k13 * s1 - k23 * s0, k02 * s3 - k12 * s1 + a22 * s0}};
        // The eigenvalue nearest λ dominates the adjugate, with either sign once λ lands below it
        uint column = 0;
        for (uint candidate = 1; candidate != 4; ++candidate)
            if (fabs(adjugate[candidate][candidate]) > fabs(adjugate[column][column])) column = candidate;
        if (!(fabs(adjugate[column][column]) > noise)) break;
        thread float const *vector = adjugate[column];
        float const image[4] = {k00 * vector[0] + k01 * vector[1] + k02 * vector[2] + k03 * vector[3],
                                k01 * vector[0] + k11 * vector[1] + k12 * vector[2] + k13 * vector[3],
                                k02 * vector[0] + k12 * vector[1] + k22 * vector[2] + k23 * vector[3],
                                k03 * vector[0] + k13 * vector[1] + k23 * vector[2] + k33 * vector[3]};
        float const length = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2] +
                             vector[3] * vector[3];
        float const quotient =
            (vector[0] * image[0] + vector[1] * image[1] + vector[2] * image[2] + vector[3] * image[3]) / length;
        // λ lies between the diagonal bound and the iterate, so a quotient outside is noise
        if (!(quotient >= floor_bound && quotient <= eigenvalue)) break;
        eigenvalue = quotient, refined = true;
    }
    float residual = 2 * upper * (1 - eigenvalue);
    // Exactly linear frames repeat the root, so they fold over the Kabsch SVD rotation instead
    if (!refined) {
        float const transposed[9] = {cross_covariance[0], cross_covariance[3], cross_covariance[6],
                                     cross_covariance[1], cross_covariance[4], cross_covariance[7],
                                     cross_covariance[2], cross_covariance[5], cross_covariance[8]};
        float svd_left[9], svd_diagonal[9], svd_right[9], rotation[9], trace = 0;
        nk_svd3x3_f32_metal_(transposed, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f32_metal_(svd_left, svd_right, rotation);
        if (nk_det3x3_f32_metal_(rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f32_metal_(svd_left, svd_right, rotation);
        }
        for (uint cell = 0; cell != 9; ++cell) trace += rotation[cell] * cross_covariance[cell];
        residual = norms_squared - 2 * trace;
    }
    return fmax(residual, 0.0f);
}

/** The superposed RMSD of two packs of F16 points on a whole simdgroup, in every lane. Every lane
 *  calls with the same packs and its own @p lane , the lanes split the points, and the simdgroup
 *  sums the moments in F32 for @c nk_rmsds_folded_f32_metal_ to fold. */
inline float nk_rmsds_pair_f16_metal_(device uchar const *row_packed, device uchar const *column_packed, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    uint const count = nk_shapes_packed_header_f16_metal_(row_packed)->points;
    if (!count) return 0;
    uint axis_stride;
    device half const *a_axes = nk_shapes_packed_axes_f16_metal_(row_packed, axis_stride);
    device half const *b_axes = nk_shapes_packed_axes_f16_metal_(column_packed, axis_stride);
    float cross_covariance[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, norms_squared = 0;
    for (uint point = lane; point < count; point += 32) {
        float const a[3] = {a_axes[point], a_axes[axis_stride + point], a_axes[2 * axis_stride + point]};
        float const b[3] = {b_axes[point], b_axes[axis_stride + point], b_axes[2 * axis_stride + point]};
        for (uint row = 0; row != 3; ++row)
            for (uint column = 0; column != 3; ++column) cross_covariance[row * 3 + column] += a[row] * b[column];
        for (uint axis = 0; axis != 3; ++axis) norms_squared += a[axis] * a[axis] + b[axis] * b[axis];
    }
    for (uint cell = 0; cell != 9; ++cell) cross_covariance[cell] = simd_sum(cross_covariance[cell]);
    float const residual = nk_rmsds_folded_f32_metal_(cross_covariance, simd_sum(norms_squared));
    return precise::sqrt(precise::divide(residual, float(count)));
}

kernel void nk_rmsds_packed_f16_metal_kernel_(
    device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
    device uchar *results [[buffer(2)]], device float *rotations [[buffer(3)]],
    constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    ulong const pair = (ulong(group.y) * groups.x + group.x) * 4 + simdgroup;
    if (pair >= arguments.rows * arguments.columns) return;
    ulong const row = pair / arguments.columns, column = pair % arguments.columns;
    float const distance = nk_rmsds_pair_f16_metal_(nk_shapes_pack_metal_(a_table[row], arguments),
                                                    nk_shapes_pack_metal_(b_table[column], arguments), lane);
    device float *results_row = (device float *)(results + row * arguments.results_stride);
    if (lane == 0) results_row[column] = distance;
}

#pragma endregion RMSD

#pragma region Transport

/** A tree link that leads nowhere. */
constant uint nk_shapes_no_node_metal_k = 0xFFFFFFFFu;

/** One simdgroup's transport in its slot of the caller's workspace, between clouds of up to
 *  @c points_stride points: the potentials, the spanning tree over row then column nodes as the
 *  serial solver keeps it, the rotated column points, and the gradient that replaces the distances
 *  as arc costs in Gromov-Wasserstein. Potentials are exact in 64-bit fixed point, on a scale that
 *  reads each arc's F32 cost down to 2⁻⁴⁰ of the largest. */
struct nk_shapes_network_metal_t {
    device long *potentials;
    device uint *parents, *flows, *depths, *first_children, *next_siblings, *previous_siblings;
    device float *rotated_points;
    device long const *gradient;
    device half const *row_axes, *column_axes;
    uint row_stride, column_stride, points_stride, rows, columns, gradient_shift;
    float cost_scale;
};

/** The network of one pair over the slot at @p slot , for clouds of up to @p largest points. */
inline nk_shapes_network_metal_t nk_shapes_network_f16_metal_(device uchar *slot, uint largest,
                                                              device uchar const *row_packed,
                                                              device uchar const *column_packed) {
    nk_shapes_network_metal_t network;
    uint const nodes = 2 * largest;
    network.potentials = (device long *)slot;
    device uint *links = (device uint *)(network.potentials + nodes);
    network.parents = links, network.flows = links + nodes, network.depths = links + 2 * nodes;
    network.first_children = links + 3 * nodes, network.next_siblings = links + 4 * nodes;
    network.previous_siblings = links + 5 * nodes;
    network.rotated_points = (device float *)(links + 6 * nodes);
    network.gradient = nullptr, network.gradient_shift = 0, network.points_stride = largest;
    network.row_axes = nk_shapes_packed_axes_f16_metal_(row_packed, network.row_stride);
    network.column_axes = nk_shapes_packed_axes_f16_metal_(column_packed, network.column_stride);
    network.rows = nk_shapes_packed_header_f16_metal_(row_packed)->points;
    network.columns = nk_shapes_packed_header_f16_metal_(column_packed)->points;
    float const radii = nk_shapes_packed_header_f16_metal_(row_packed)->max_radius +
                        nk_shapes_packed_header_f16_metal_(column_packed)->max_radius;
    // 2⁴⁰ over the power of two above radii², whose product with an F32 cost stays exact
    uint const exponent = (as_type<uint>(radii * radii) >> 23) & 0xFF;
    network.cost_scale = as_type<float>(min(293u - exponent, 254u) << 23);
    return network;
}

/** Appends @p child to the children of @p parent in the simplex tree. */
inline void nk_shapes_tree_attach_f16_metal_(thread nk_shapes_network_metal_t const &network, uint parent, uint child) {
    uint const after = network.first_children[parent];
    network.next_siblings[child] = after, network.previous_siblings[child] = nk_shapes_no_node_metal_k;
    if (after != nk_shapes_no_node_metal_k) network.previous_siblings[after] = child;
    network.first_children[parent] = child;
}

/** Removes @p child from the children of @p parent in the simplex tree. */
inline void nk_shapes_tree_detach_f16_metal_(thread nk_shapes_network_metal_t const &network, uint parent, uint child) {
    uint const before = network.previous_siblings[child], after = network.next_siblings[child];
    if (before != nk_shapes_no_node_metal_k) network.next_siblings[before] = after;
    else network.first_children[parent] = after;
    if (after != nk_shapes_no_node_metal_k) network.previous_siblings[after] = before;
}

/** Seeds the simplex tree with the northwest-corner plan rooted at the first row point, as
 *  @c nk_shapes_northwest_corner_serial_ does. Runs on one lane. */
inline void nk_shapes_tree_northwest_f16_metal_(thread nk_shapes_network_metal_t const &network) {
    uint const row_nodes = network.rows, column_nodes = network.columns;
    for (uint node = 0; node != row_nodes + column_nodes; ++node)
        network.first_children[node] = nk_shapes_no_node_metal_k;
    network.parents[0] = nk_shapes_no_node_metal_k, network.flows[0] = 0, network.depths[0] = 0;
    uint row = 0, column = 0, supply = column_nodes, demand = row_nodes, newest = row_nodes;
    for (;;) {
        uint const shipped = min(supply, demand);
        uint const above = newest < row_nodes ? row_nodes + column : row;
        network.parents[newest] = above, network.flows[newest] = shipped;
        network.depths[newest] = network.depths[above] + 1;
        nk_shapes_tree_attach_f16_metal_(network, above, newest);
        supply -= shipped, demand -= shipped;
        if (row + 1 == row_nodes && column + 1 == column_nodes) return;
        if (!supply && row + 1 != row_nodes) ++row, supply = column_nodes, newest = row;
        else ++column, demand = row_nodes, newest = row_nodes + column;
    }
}

/** An arc's squared distance between a row point and a rotated column point, as
 *  @c nk_shapes_point_cost_f32_serial_ rounds it. */
inline float nk_shapes_point_cost_f16_metal_(thread nk_shapes_network_metal_t const &network, uint row, uint column) {
#pragma clang fp contract(off) reassociate(off)
    uint const stride = network.points_stride;
    float const delta_x = float(network.row_axes[row]) - network.rotated_points[column];
    float const delta_y = float(network.row_axes[network.row_stride + row]) - network.rotated_points[stride + column];
    float const delta_z = float(network.row_axes[2 * network.row_stride + row]) -
                          network.rotated_points[2 * stride + column];
    return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
}

/** An arc's fixed-point cost: its squared distance on the pair's scale, or its negated gradient. */
template <bool gradient_>
inline long nk_shapes_arc_cost_f16_metal_(thread nk_shapes_network_metal_t const &network, uint row, uint column) {
    if constexpr (gradient_) return -(network.gradient[row * network.columns + column] >> network.gradient_shift);
    else return long(rint(nk_shapes_point_cost_f16_metal_(network, row, column) * network.cost_scale));
}

/** The simplex tree's potentials walked from its root. Runs on one lane. */
template <bool gradient_>
inline void nk_shapes_tree_potentials_f16_metal_(thread nk_shapes_network_metal_t const &network) {
    uint const row_nodes = network.rows;
    uint node = 0;
    network.potentials[0] = 0;
    for (;;) {
        if (network.first_children[node] != nk_shapes_no_node_metal_k) node = network.first_children[node];
        else {
            while (node && network.next_siblings[node] == nk_shapes_no_node_metal_k) node = network.parents[node];
            if (!node) return;
            node = network.next_siblings[node];
        }
        uint const above = network.parents[node];
        if (node < row_nodes)
            network.potentials[node] = network.potentials[above] +
                                       nk_shapes_arc_cost_f16_metal_<gradient_>(network, node, above - row_nodes);
        else
            network.potentials[node] = network.potentials[above] -
                                       nk_shapes_arc_cost_f16_metal_<gradient_>(network, above, node - row_nodes);
    }
}

/** One pivot of the serial network simplex: the entering arc closes a cycle through the join of its
 *  ends, Cunningham's rule picks the last blocking arc around it, the flow shifts, and the subtree
 *  cut off hangs from the entering arc with updated depths and potentials. Runs on one lane. */
inline void nk_shapes_tree_pivot_f16_metal_(thread nk_shapes_network_metal_t const &network, uint entering_row,
                                            uint entering_column, long entering_cost) {
    uint const no_node = nk_shapes_no_node_metal_k, row_nodes = network.rows;
    device uint *parent = network.parents, *flow = network.flows, *depth = network.depths;
    uint const row_node = entering_row, column_node = row_nodes + entering_column;
    uint row_side = row_node, column_side = column_node;
    while (row_side != column_side) {
        if (depth[row_side] > depth[column_side]) row_side = parent[row_side];
        else if (depth[column_side] > depth[row_side]) column_side = parent[column_side];
        else row_side = parent[row_side], column_side = parent[column_side];
    }
    uint const join = row_side;
    uint row_side_flow = no_node, row_side_leaving = no_node;
    uint column_side_flow = no_node, column_side_leaving = no_node;
    for (uint node = row_node; node != join; node = parent[node])
        if (node < row_nodes && flow[node] < row_side_flow) row_side_flow = flow[node], row_side_leaving = node;
    for (uint node = column_node; node != join; node = parent[node])
        if (node >= row_nodes && flow[node] <= column_side_flow)
            column_side_flow = flow[node], column_side_leaving = node;
    bool const leaves_column_side = column_side_flow <= row_side_flow;
    uint const delta = leaves_column_side ? column_side_flow : row_side_flow;
    uint const leaving = leaves_column_side ? column_side_leaving : row_side_leaving;
    for (uint node = row_node; node != join; node = parent[node])
        flow[node] = node < row_nodes ? flow[node] - delta : flow[node] + delta;
    for (uint node = column_node; node != join; node = parent[node])
        flow[node] = node < row_nodes ? flow[node] + delta : flow[node] - delta;
    uint const moved_root = leaves_column_side ? column_node : row_node;
    uint const anchor = leaves_column_side ? row_node : column_node;
    long const shift = leaves_column_side ? -entering_cost : entering_cost;
    uint node = moved_root, new_parent = anchor, carried_flow = delta;
    for (;;) {
        uint const old_parent = parent[node], old_flow = flow[node];
        nk_shapes_tree_detach_f16_metal_(network, old_parent, node);
        parent[node] = new_parent, flow[node] = carried_flow;
        nk_shapes_tree_attach_f16_metal_(network, new_parent, node);
        if (node == leaving) break;
        new_parent = node, carried_flow = old_flow, node = old_parent;
    }
    depth[moved_root] = depth[anchor] + 1, network.potentials[moved_root] += shift;
    node = moved_root;
    for (;;) {
        if (network.first_children[node] != no_node) node = network.first_children[node];
        else {
            while (node != moved_root && network.next_siblings[node] == no_node) node = parent[node];
            if (node == moved_root) break;
            node = network.next_siblings[node];
        }
        depth[node] = depth[parent[node]] + 1, network.potentials[node] += shift;
    }
}

/** The least of a whole simdgroup's @p value , through its halves as reductions take no 64-bit
 *  type, leaving in @p index the least index among the lanes holding it. */
inline long nk_shapes_simdgroup_argmin_metal_(long value, thread uint &index) {
    int const high = int(value >> 32);
    int const least_high = simd_min(high);
    uint const least_low = simd_min(high == least_high ? uint(value) : 0xFFFFFFFFu);
    long const least = (long(least_high) << 32) | long(least_low);
    index = simd_min(value == least ? index : nk_shapes_no_node_metal_k);
    return least;
}

/** Moves @p row and @p column by @p offset arcs in row-major order, wrapping after the last row. */
inline void nk_shapes_arc_advance_f16_metal_(thread nk_shapes_network_metal_t const &network, uint offset,
                                             thread uint &row, thread uint &column) {
    column += offset;
    while (column >= network.columns) column -= network.columns, row = row + 1 == network.rows ? 0 : row + 1;
}

/** The exact plan by the serial network simplex from the tree in @p network : the lanes price each
 *  block of √(n × m) arcs from the cursor, taking its first most negative reduced cost, and one
 *  lane pivots. Fixed-point sums are exact, so any negative reduced cost is real and no tolerance
 *  stops it, while the scale leaves the serial tolerance to the rounding of each cost. */
template <bool gradient_>
inline void nk_shapes_simplex_f16_metal_(thread nk_shapes_network_metal_t const &network, uint lane) {
    uint const rows = network.rows, columns = network.columns;
    ulong const arcs = ulong(rows) * columns;
    uint block = 10, cursor_row = 0, cursor_column = 0;
    while (ulong(block) * block < arcs) ++block;
    if (lane == 0) nk_shapes_tree_potentials_f16_metal_<gradient_>(network);
    simdgroup_barrier(mem_flags::mem_device);
    for (;;) {
        long entering_cost = 0;
        uint entering_row = 0, entering_column = 0;
        for (ulong scanned = 0; scanned < arcs && entering_cost == 0;) {
            uint const count = uint(min(arcs - scanned, ulong(block)));
            long least = 0;
            uint index = nk_shapes_no_node_metal_k;
            for (uint offset = lane; offset < count; offset += 32) {
                uint row = cursor_row, column = cursor_column;
                nk_shapes_arc_advance_f16_metal_(network, offset, row, column);
                long const reduced = nk_shapes_arc_cost_f16_metal_<gradient_>(network, row, column) -
                                     network.potentials[row] + network.potentials[rows + column];
                if (reduced < least) least = reduced, index = offset;
            }
            least = nk_shapes_simdgroup_argmin_metal_(least, index);
            if (least < 0) {
                entering_cost = least, entering_row = cursor_row, entering_column = cursor_column;
                nk_shapes_arc_advance_f16_metal_(network, index, entering_row, entering_column);
            }
            nk_shapes_arc_advance_f16_metal_(network, count, cursor_row, cursor_column);
            scanned += count;
        }
        if (entering_cost == 0) return;
        if (lane == 0) nk_shapes_tree_pivot_f16_metal_(network, entering_row, entering_column, entering_cost);
        simdgroup_barrier(mem_flags::mem_device);
    }
}

/** The tree's arcs as row, column and flow arrays @p plan_stride apart, as
 *  @c nk_shapes_tree_arcs_serial_ lists them. Runs on one lane. */
inline void nk_shapes_tree_plan_f16_metal_(thread nk_shapes_network_metal_t const &network, device uint *plan,
                                           uint plan_stride) {
    uint const rows = network.rows, nodes = rows + network.columns;
    for (uint node = 1; node != nodes; ++node) {
        uint const above = network.parents[node];
        plan[node - 1] = node < rows ? node : above;
        plan[plan_stride + node - 1] = (node < rows ? above : node) - rows;
        plan[2 * plan_stride + node - 1] = network.flows[node];
    }
}

/** The plan's mean cost and cross-covariance, summed in the serial order. Runs on one lane. */
inline float nk_shapes_tree_moments_f16_metal_(thread nk_shapes_network_metal_t const &network,
                                               thread float *cross_covariance) {
#pragma clang fp contract(off) reassociate(off)
    uint const rows = network.rows, columns = network.columns;
    float const mass = float(rows) * float(columns);
    float cost = 0;
    for (uint cell = 0; cell != 9; ++cell) cross_covariance[cell] = 0;
    for (uint node = 1; node != rows + columns; ++node) {
        uint const above = network.parents[node], flow = network.flows[node];
        uint const row = node < rows ? node : above, column = (node < rows ? above : node) - rows;
        cost += float(flow) * nk_shapes_point_cost_f16_metal_(network, row, column);
        if (!flow) continue;
        float const weight = float(flow) / mass;
        for (uint column_axis = 0; column_axis != 3; ++column_axis) {
            float const weighted = weight * float(network.column_axes[column_axis * network.column_stride + column]);
            for (uint row_axis = 0; row_axis != 3; ++row_axis)
                cross_covariance[column_axis * 3 + row_axis] +=
                    weighted * float(network.row_axes[row_axis * network.row_stride + row]);
        }
    }
    return cost / mass;
}

/** One flip of Wasserstein-Procrustes: exact transport alternating with the Kabsch fit of its plan
 *  while the mean cost falls, from the axis signs of @p flip . Returns the least cost, in every
 *  lane, and its @p flip_rotation of the column points. A plan beating @p parity_best replaces it
 *  in @p parity_plan unless that is null. */
inline float nk_wassersteinprocrustes_flip_f16_metal_(thread nk_shapes_network_metal_t const &network, uint flip,
                                                      uint lane, thread float *flip_rotation, device uint *parity_plan,
                                                      uint plan_stride, thread float &parity_best) {
#pragma clang fp contract(off) reassociate(off)
    float current[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    current[0] = (flip & 1) ? -1 : 1, current[4] = (flip & 2) ? -1 : 1, current[8] = (flip & 4) ? -1 : 1;
    float const flip_determinant = current[0] * current[4] * current[8];
    float best = INFINITY;
    for (uint cell = 0; cell != 9; ++cell) flip_rotation[cell] = current[cell];
    if (lane == 0) nk_shapes_tree_northwest_f16_metal_(network);
    for (;;) {
        for (uint column = lane; column < network.columns; column += 32) {
            float const x = network.column_axes[column], y = network.column_axes[network.column_stride + column],
                        z = network.column_axes[2 * network.column_stride + column];
            for (uint axis = 0; axis != 3; ++axis)
                network.rotated_points[axis * network.points_stride + column] = current[axis * 3] * x +
                                                                                current[axis * 3 + 1] * y +
                                                                                current[axis * 3 + 2] * z;
        }
        simdgroup_barrier(mem_flags::mem_device);
        nk_shapes_simplex_f16_metal_<false>(network, lane);
        float cost = 0, cross_covariance[9];
        if (lane == 0) cost = nk_shapes_tree_moments_f16_metal_(network, cross_covariance);
        cost = simd_broadcast_first(cost);
        if (!(cost < best)) return best;
        best = cost;
        for (uint cell = 0; cell != 9; ++cell) flip_rotation[cell] = current[cell];
        if (cost < parity_best) {
            parity_best = cost;
            if (parity_plan && lane == 0) nk_shapes_tree_plan_f16_metal_(network, parity_plan, plan_stride);
        }
        if (lane == 0) {
            float svd_left[9], svd_diagonal[9], svd_right[9];
            nk_svd3x3_f32_metal_(cross_covariance, svd_left, svd_diagonal, svd_right);
            nk_rotation_from_svd_f32_metal_(svd_left, svd_right, current);
            // Each start stays among the rotations, or among the reflections, it began with
            if (nk_det3x3_f32_metal_(current) * flip_determinant < 0) {
                svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
                nk_rotation_from_svd_f32_metal_(svd_left, svd_right, current);
            }
        }
        for (uint cell = 0; cell != 9; ++cell) current[cell] = simd_broadcast_first(current[cell]);
    }
}

/** The least mean cost over the 8 flips, with its @p rotation of the column points, in every lane.
 *  Keeps the best proper and improper plans for Gromov-Wasserstein unless those are null. */
inline float nk_wassersteinprocrustes_pair_f16_metal_(thread nk_shapes_network_metal_t const &network, uint lane,
                                                      thread float *rotation, device uint *proper_plan,
                                                      device uint *improper_plan, uint plan_stride) {
    float best = INFINITY, best_proper = INFINITY, best_improper = INFINITY, flip_rotation[9];
    for (uint flip = 0; flip != 8; ++flip) {
        bool const proper = !(popcount(flip) & 1);
        float const flip_best = nk_wassersteinprocrustes_flip_f16_metal_(
            network, flip, lane, flip_rotation, proper ? proper_plan : improper_plan, plan_stride,
            proper ? best_proper : best_improper);
        if (!(flip_best < best)) continue;
        best = flip_best;
        for (uint cell = 0; cell != 9; ++cell) rotation[cell] = flip_rotation[cell];
    }
    return best;
}

/** Whether the canonical order solves the column cloud first, by points, spreads, radius and then
 *  packed bytes, as @c nk_shapes_column_first_f16_serial_ decides. */
inline bool nk_shapes_column_first_f16_metal_(device uchar const *row_packed, device uchar const *column_packed) {
    device nk_shapes_packed_header_f32_metal_t const *row_header = nk_shapes_packed_header_f16_metal_(row_packed);
    device nk_shapes_packed_header_f32_metal_t const *column_header = nk_shapes_packed_header_f16_metal_(column_packed);
    if (row_header->points != column_header->points) return column_header->points < row_header->points;
    for (uint axis = 0; axis != 3; ++axis)
        if (row_header->spreads[axis] != column_header->spreads[axis])
            return column_header->spreads[axis] < row_header->spreads[axis];
    if (row_header->max_radius != column_header->max_radius) return column_header->max_radius < row_header->max_radius;
    uint axis_stride;
    device half const *row_axes = nk_shapes_packed_axes_f16_metal_(row_packed, axis_stride);
    device half const *column_axes = nk_shapes_packed_axes_f16_metal_(column_packed, axis_stride);
    device uchar const *row_bytes = (device uchar const *)row_axes, *column_bytes = (device uchar const *)column_axes;
    uint const axis_bytes = axis_stride * 2, points_bytes = row_header->points * 2;
    for (uint axis_offset = 0; axis_offset != 3 * axis_bytes; axis_offset += axis_bytes)
        for (uint index = axis_offset; index != axis_offset + points_bytes; ++index)
            if (row_bytes[index] != column_bytes[index]) return column_bytes[index] < row_bytes[index];
    return false;
}

/** 𝑄ₐᵀ × @p inner × 𝑄ᵦ, as @c nk_shapes_rotation_f32_serial_ composes it. */
inline void nk_shapes_rotation_f16_metal_(device float const *row_frame, thread float const *inner,
                                          device float const *column_frame, thread float *rotation) {
#pragma clang fp contract(off) reassociate(off)
    float row_axes[9], column_axes[9], inner_column[9];
    nk_quaternion_to_mat3x3_f32_metal_(row_frame, row_axes);
    nk_quaternion_to_mat3x3_f32_metal_(column_frame, column_axes);
    for (uint row = 0; row != 3; ++row)
        for (uint column = 0; column != 3; ++column)
            inner_column[row * 3 + column] = inner[row * 3] * column_axes[column] +
                                             inner[row * 3 + 1] * column_axes[3 + column] +
                                             inner[row * 3 + 2] * column_axes[6 + column];
    for (uint row = 0; row != 3; ++row)
        for (uint column = 0; column != 3; ++column)
            rotation[row * 3 + column] = row_axes[row] * inner_column[column] +
                                         row_axes[3 + row] * inner_column[3 + column] +
                                         row_axes[6 + row] * inner_column[6 + column];
}

/** One transport pair of a launch, opened in canonical order, as both kernels score it. */
struct nk_shapes_transport_pair_metal_t {
    device uchar const *row_packed, *column_packed;
    ulong row, column, pair;
    uint rows, columns;
    bool swapped;
};

/** The simdgroup's pair of this launch, numbered row-major from @c pair_begin ; false past its
 *  end. */
inline bool nk_shapes_transport_pair_metal_(device ulong const *a_table, device ulong const *b_table,
                                            constant nk_shapes_arguments_metal_t &arguments, uint2 group, uint2 groups,
                                            uint simdgroup, thread nk_shapes_transport_pair_metal_t &opened) {
    opened.pair = arguments.pair_begin + (ulong(group.y) * groups.x + group.x) * 4 + simdgroup;
    if (opened.pair >= arguments.pair_end) return false;
    opened.row = opened.pair / arguments.columns, opened.column = opened.pair % arguments.columns;
    device uchar const *a_packed = nk_shapes_pack_metal_(a_table[opened.row], arguments);
    device uchar const *b_packed = nk_shapes_pack_metal_(b_table[opened.column], arguments);
    opened.swapped = nk_shapes_column_first_f16_metal_(a_packed, b_packed);
    opened.row_packed = opened.swapped ? b_packed : a_packed;
    opened.column_packed = opened.swapped ? a_packed : b_packed;
    opened.rows = nk_shapes_packed_header_f16_metal_(opened.row_packed)->points;
    opened.columns = nk_shapes_packed_header_f16_metal_(opened.column_packed)->points;
    return true;
}

/** Writes a pair's distance and its rotation, back in the caller's order. Runs on one lane. */
inline void nk_shapes_write_transport_metal_(device uchar *results, device float *rotations,
                                             constant nk_shapes_arguments_metal_t &arguments,
                                             thread nk_shapes_transport_pair_metal_t const &opened, float distance,
                                             thread float const *inner) {
#pragma clang fp contract(off) reassociate(off)
    device float *results_row = (device float *)(results + opened.row * arguments.results_stride);
    results_row[opened.column] = distance;
    if (!arguments.rotations) return;
    float pair_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (opened.rows && opened.columns)
        nk_shapes_rotation_f16_metal_(nk_shapes_packed_header_f16_metal_(opened.row_packed)->frame, inner,
                                      nk_shapes_packed_header_f16_metal_(opened.column_packed)->frame, pair_rotation);
    device float *pair_output = rotations + opened.pair * 9;
    for (uint cell = 0; cell != 9; ++cell)
        pair_output[cell] = pair_rotation[opened.swapped ? cell % 3 * 3 + cell / 3 : cell];
}

kernel void nk_wassersteinprocrustes_packed_f16_metal_kernel_(
    device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
    device uchar *results [[buffer(2)]], device float *rotations [[buffer(3)]],
    constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]], device uchar *workspace [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    nk_shapes_transport_pair_metal_t opened;
    if (!nk_shapes_transport_pair_metal_(a_table, b_table, arguments, group, groups, simdgroup, opened)) return;
    float inner[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float distance = opened.rows == opened.columns ? 0 : INFINITY;
    if (opened.rows && opened.columns) {
        device uchar *slot = workspace + (opened.pair - arguments.pair_begin) * arguments.slot_bytes;
        nk_shapes_network_metal_t const network = nk_shapes_network_f16_metal_(slot, arguments.largest,
                                                                               opened.row_packed, opened.column_packed);
        distance = precise::sqrt(nk_wassersteinprocrustes_pair_f16_metal_(network, lane, inner, nullptr, nullptr, 0));
    }
    if (lane == 0) nk_shapes_write_transport_metal_(results, rotations, arguments, opened, distance, inner);
}

#pragma endregion Transport

/*  Gromov-Wasserstein in fixed point: distances within each cloud round to 𝐵 bits, so the gradient
 *  Σ flow × Dx × Dy over a plan is an exact 62-bit integer. A Frank-Wolfe step mixes gradients in
 *  32-bit fractions, and the transport over them shifts down to the bits the simplex sums. The
 *  line search reads exact differences of scores in F32, as only the step comes out of it. */
#pragma region Gromov Wasserstein

/** One simdgroup's Gromov-Wasserstein matrices in its slot: the plan's gradient, a vertex's
 *  gradient and the sparse product by Dy it passes through, the distances within each cloud, the
 *  dense plan, and the best proper and improper plans as row, column and flow arrays
 *  @c plan_stride apart. */
struct nk_shapes_gromov_metal_t {
    device long *gradient, *vertex_gradient, *partial_gradient;
    device uint *row_distances, *column_distances;
    device float *plan;
    device uint *proper_plan, *improper_plan;
    uint plan_stride, rows, columns, gradient_shift;
    float distance_scale;
};

/** The matrices of one pair past the transport state of @p slot , with the fixed-point scales of
 *  its clouds. */
inline nk_shapes_gromov_metal_t nk_shapes_gromov_f16_metal_(device uchar *slot, uint largest,
                                                            thread nk_shapes_network_metal_t &network,
                                                            device uchar const *row_packed,
                                                            device uchar const *column_packed) {
    nk_shapes_gromov_metal_t gromov;
    uint const nodes = 2 * largest, cells_capacity = largest * largest;
    uint const rows = network.rows, columns = network.columns, cells = rows * columns;
    uint const network_bytes = nk::round_up_to_multiple(nodes * (8 + 6 * 4) + 3 * largest * 4, 8u);
    gromov.gradient = (device long *)(slot + network_bytes);
    gromov.vertex_gradient = gromov.gradient + cells_capacity;
    gromov.partial_gradient = gromov.gradient + 2 * cells_capacity;
    gromov.row_distances = (device uint *)(gromov.gradient + 3 * cells_capacity);
    gromov.column_distances = gromov.row_distances + cells_capacity;
    gromov.plan = (device float *)(gromov.column_distances + cells_capacity);
    gromov.proper_plan = (device uint *)(gromov.plan + cells_capacity);
    gromov.improper_plan = gromov.proper_plan + 3 * nodes;
    gromov.plan_stride = nodes, gromov.rows = rows, gromov.columns = columns;
    // Sums over the n², m² or n × m cells of squared 𝐵-bit distances then fit 62 bits
    uint const larger = max(rows, columns), squared = larger * larger, cell_bits = 32 - clz(max(squared - 1, 1u));
    uint const distance_bits = min((62 - cell_bits) / 2, 24u), mass_bits = 32 - clz(max(cells - 1, 1u));
    uint const solver_bits = min(42u, 62 - cell_bits), gradient_bits = 2 * distance_bits + mass_bits;
    float const reach = 2 * fmax(nk_shapes_packed_header_f16_metal_(row_packed)->max_radius,
                                 nk_shapes_packed_header_f16_metal_(column_packed)->max_radius);
    int const reach_exponent = int((as_type<uint>(reach) >> 23) & 0xFF) - 126;
    gromov.distance_scale = as_type<float>(uint(clamp(127 + int(distance_bits) - reach_exponent, 1, 254)) << 23);
    gromov.gradient_shift = gradient_bits > solver_bits ? gradient_bits - solver_bits : 0;
    network.gradient = gromov.gradient, network.gradient_shift = gromov.gradient_shift;
    return gromov;
}

/** The distances within a cloud in fixed point, on one lane's rows each. */
inline void nk_shapes_distances_f16_metal_(device half const *axes, uint stride, uint count, float distance_scale,
                                           device uint *distances, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    for (uint first = lane; first < count; first += 32)
        for (uint second = 0; second != count; ++second) {
            float const delta_x = float(axes[first]) - float(axes[second]);
            float const delta_y = float(axes[stride + first]) - float(axes[stride + second]);
            float const delta_z = float(axes[2 * stride + first]) - float(axes[2 * stride + second]);
            float const squares = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
            distances[first * count + second] = uint(rint(precise::sqrt(squares) * distance_scale));
        }
}

/** Σ Dᵢⱼ² / count² of one cloud's distances. Runs on one lane. */
inline long nk_shapes_distance_spread_f16_metal_(device uint const *distances, uint count) {
    long spread = 0;
    for (uint cell = 0; cell != count * count; ++cell) spread += long(distances[cell]) * long(distances[cell]);
    return spread / (long(count) * long(count));
}

/** Dx × V × Dy of the plan @p arcs into @p target : each lane's columns of the sparse product by
 *  Dy, then of the dense one by Dx, in sums of flow × Dx × Dy. */
inline void nk_shapes_distance_gradient_f16_metal_(thread nk_shapes_gromov_metal_t const &gromov,
                                                   device uint const *arcs, device long *target, uint lane) {
    uint const rows = gromov.rows, columns = gromov.columns, arcs_count = rows + columns - 1;
    uint const plan_stride = gromov.plan_stride;
    device long *partial = gromov.partial_gradient;
    for (uint column = lane; column < columns; column += 32) {
        for (uint row = 0; row != rows; ++row) partial[row * columns + column] = 0;
        for (uint arc = 0; arc != arcs_count; ++arc)
            partial[arcs[arc] * columns + column] +=
                long(arcs[2 * plan_stride + arc]) *
                long(gromov.column_distances[arcs[plan_stride + arc] * columns + column]);
    }
    simdgroup_barrier(mem_flags::mem_device);
    for (uint column = lane; column < columns; column += 32)
        for (uint row = 0; row != rows; ++row) {
            long sum = 0;
            for (uint other = 0; other != rows; ++other)
                sum += long(gromov.row_distances[row * rows + other]) * partial[other * columns + column];
            target[row * columns + column] = sum;
        }
    simdgroup_barrier(mem_flags::mem_device);
}

/** The inner product of @p matrix with the plan @p arcs , in flows over the simplex's fixed point.
 *  Runs on one lane. */
inline long nk_shapes_plan_score_f16_metal_(thread nk_shapes_gromov_metal_t const &gromov, device long const *matrix,
                                            device uint const *arcs) {
    uint const plan_stride = gromov.plan_stride;
    long score = 0;
    for (uint arc = 0; arc != gromov.rows + gromov.columns - 1; ++arc)
        score += long(arcs[2 * plan_stride + arc]) *
                 (matrix[arcs[arc] * gromov.columns + arcs[plan_stride + arc]] >> gromov.gradient_shift);
    return score;
}

/** Loads the dense plan and gradient from the arcs of @p start_plan , returning ⟨G, P⟩ on the
 *  first lane. */
inline long nk_shapes_start_plan_f16_metal_(thread nk_shapes_gromov_metal_t const &gromov,
                                            device uint const *start_plan, uint lane) {
    uint const plan_stride = gromov.plan_stride, columns = gromov.columns, cells = gromov.rows * columns;
    nk_shapes_distance_gradient_f16_metal_(gromov, start_plan, gromov.gradient, lane);
    long score = 0;
    if (lane == 0) {
        for (uint cell = 0; cell != cells; ++cell) gromov.plan[cell] = 0;
        for (uint arc = 0; arc != gromov.rows + columns - 1; ++arc)
            gromov.plan[start_plan[arc] * columns + start_plan[plan_stride + arc]] =
                float(start_plan[2 * plan_stride + arc]) / float(cells);
        score = nk_shapes_plan_score_f16_metal_(gromov, gromov.gradient, start_plan);
    }
    simdgroup_barrier(mem_flags::mem_device);
    return score;
}

/** Loads the product plan, whose gradient is the outer product of the two matrices' row sums,
 *  returning ⟨G, P⟩ on the lane that runs it. Runs on one lane. */
inline long nk_shapes_product_plan_f16_metal_(thread nk_shapes_gromov_metal_t const &gromov) {
    uint const rows = gromov.rows, columns = gromov.columns;
    float const weight = 1.0f / float(rows * columns);
    long score = 0;
    for (uint column = 0; column != columns; ++column) {
        long column_sum = 0;
        for (uint other = 0; other != columns; ++other)
            column_sum += long(gromov.column_distances[other * columns + column]);
        gromov.partial_gradient[column] = column_sum;
    }
    for (uint row = 0; row != rows; ++row) {
        long row_sum = 0;
        for (uint other = 0; other != rows; ++other) row_sum += long(gromov.row_distances[row * rows + other]);
        for (uint column = 0; column != columns; ++column) {
            long const cell = row_sum * gromov.partial_gradient[column];
            gromov.plan[row * columns + column] = weight, gromov.gradient[row * columns + column] = cell;
            score += cell >> gromov.gradient_shift;
        }
    }
    return score;
}

/** @p value × @p fraction / 2³², with @p fraction at most 2³². */
inline ulong nk_shapes_mix_metal_(ulong value, ulong fraction) {
    return (value >> 32) * fraction + (((value & 0xFFFFFFFFul) * fraction) >> 32);
}

/** Moves the gradient, the dense plan and ⟨G, P⟩ a Frank-Wolfe step toward @p arcs , a step of
 *  @p step_fraction / 2³². ⟨V, G⟩ = ⟨P, Gᵥ⟩, as Dx and Dy are symmetric, so the score needs no sum
 *  over the cells. */
inline long nk_shapes_plan_step_f16_metal_(thread nk_shapes_gromov_metal_t const &gromov, device uint const *arcs,
                                           ulong step_fraction, long plan_score, long vertex_score,
                                           long vertex_curvature, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    uint const columns = gromov.columns, cells = gromov.rows * columns, plan_stride = gromov.plan_stride;
    ulong const keep_fraction = (ulong(1) << 32) - step_fraction;
    float const keep = float(keep_fraction) * 0x1p-32f, step = float(step_fraction) * 0x1p-32f;
    for (uint cell = lane; cell < cells; cell += 32) {
        gromov.gradient[cell] = long(nk_shapes_mix_metal_(ulong(gromov.gradient[cell]), keep_fraction) +
                                     nk_shapes_mix_metal_(ulong(gromov.vertex_gradient[cell]), step_fraction));
        gromov.plan[cell] *= keep;
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (lane == 0) {
        for (uint arc = 0; arc != gromov.rows + columns - 1; ++arc)
            gromov.plan[arcs[arc] * columns + arcs[plan_stride + arc]] += step * (float(arcs[2 * plan_stride + arc]) /
                                                                                  float(cells));
        ulong const towards = nk_shapes_mix_metal_(ulong(plan_score), keep_fraction) +
                              nk_shapes_mix_metal_(ulong(vertex_score), step_fraction);
        ulong const away = nk_shapes_mix_metal_(ulong(vertex_score), keep_fraction) +
                           nk_shapes_mix_metal_(ulong(vertex_curvature), step_fraction);
        plan_score = long(nk_shapes_mix_metal_(towards, keep_fraction) + nk_shapes_mix_metal_(away, step_fraction));
    }
    simdgroup_barrier(mem_flags::mem_device);
    return plan_score;
}

/** Frank-Wolfe from the start in @p gromov : up to 32 steps toward exact transport vertices with
 *  closed-form line searches, as the serial solver takes them. Returns ⟨G, P⟩ on the first lane. */
inline long nk_shapes_frank_wolfe_f16_metal_(thread nk_shapes_network_metal_t const &network,
                                             thread nk_shapes_gromov_metal_t const &gromov, long plan_score,
                                             uint lane) {
#pragma clang fp contract(off) reassociate(off)
    device uint *plan_vertex = gromov.proper_plan;
    if (lane == 0) nk_shapes_tree_northwest_f16_metal_(network);
    for (uint step_index = 0; step_index != 32; ++step_index) {
        nk_shapes_simplex_f16_metal_<true>(network, lane);
        if (lane == 0) nk_shapes_tree_plan_f16_metal_(network, plan_vertex, gromov.plan_stride);
        simdgroup_barrier(mem_flags::mem_device);
        nk_shapes_distance_gradient_f16_metal_(gromov, plan_vertex, gromov.vertex_gradient, lane);
        long vertex_score = 0, vertex_curvature = 0;
        if (lane == 0) {
            vertex_score = nk_shapes_plan_score_f16_metal_(gromov, gromov.gradient, plan_vertex);
            vertex_curvature = nk_shapes_plan_score_f16_metal_(gromov, gromov.vertex_gradient, plan_vertex);
        }
        float const slope = simd_broadcast_first(float(vertex_score - plan_score));
        float const curvature = simd_broadcast_first(float(vertex_curvature - 2 * vertex_score + plan_score));
        float const quadratic = -2 * curvature, linear = -4 * slope;
        float step = quadratic + linear < 0 ? 1 : 0;
        if (quadratic > 0) step = clamp(-linear / (2 * quadratic), 0.0f, 1.0f);
        ulong const step_fraction = ulong(step * 0x1p32f);
        if (!step_fraction) break;
        plan_score = nk_shapes_plan_step_f16_metal_(gromov, plan_vertex, step_fraction, plan_score, vertex_score,
                                                    vertex_curvature, lane);
    }
    return plan_score;
}

/** The orthogonal Procrustes fit of the dense plan, with the determinant's sign moved into V to fit
 *  over O(3) as the serial fit takes it. Left in @p rotation in every lane. */
inline void nk_shapes_plan_fit_f16_metal_(thread nk_shapes_network_metal_t const &network,
                                          thread nk_shapes_gromov_metal_t const &gromov, uint lane,
                                          thread float *rotation) {
#pragma clang fp contract(off) reassociate(off)
    uint const rows = gromov.rows, columns = gromov.columns;
    float cross_covariance[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint column = lane; column < columns; column += 32) {
        float const column_point[3] = {network.column_axes[column], network.column_axes[network.column_stride + column],
                                       network.column_axes[2 * network.column_stride + column]};
        for (uint row = 0; row != rows; ++row) {
            float const row_point[3] = {network.row_axes[row], network.row_axes[network.row_stride + row],
                                        network.row_axes[2 * network.row_stride + row]};
            float const weight = gromov.plan[row * columns + column];
            for (uint column_axis = 0; column_axis != 3; ++column_axis) {
                float const weighted = weight * column_point[column_axis];
                for (uint row_axis = 0; row_axis != 3; ++row_axis)
                    cross_covariance[column_axis * 3 + row_axis] += weighted * row_point[row_axis];
            }
        }
    }
    for (uint cell = 0; cell != 9; ++cell) cross_covariance[cell] = simd_sum(cross_covariance[cell]);
    float svd_left[9], svd_diagonal[9], svd_right[9];
    nk_svd3x3_f32_metal_(cross_covariance, svd_left, svd_diagonal, svd_right);
    // The McAdams factors are proper, so det H's sign moves into V to fit over O(3)
    if (nk_det3x3_f32_metal_(cross_covariance) < 0)
        svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
    nk_rotation_from_svd_f32_metal_(svd_left, svd_right, rotation);
}

/** Gromov-Wasserstein of one canonically ordered pair: Frank-Wolfe from the best proper and
 *  improper Wasserstein-Procrustes plans and the product plan. Returns the least square loss and
 *  the fit of its plan, in every lane. */
inline float nk_gromovwasserstein_pair_f16_metal_(thread nk_shapes_network_metal_t const &network,
                                                  thread nk_shapes_gromov_metal_t const &gromov, uint lane,
                                                  thread float *rotation) {
#pragma clang fp contract(off) reassociate(off)
    uint const rows = gromov.rows, columns = gromov.columns, cells = rows * columns;
    float const distance_scale = gromov.distance_scale;
    nk_wassersteinprocrustes_pair_f16_metal_(network, lane, rotation, gromov.proper_plan, gromov.improper_plan,
                                             gromov.plan_stride);
    simdgroup_barrier(mem_flags::mem_device);
    nk_shapes_distances_f16_metal_(network.row_axes, network.row_stride, rows, distance_scale, gromov.row_distances,
                                   lane);
    nk_shapes_distances_f16_metal_(network.column_axes, network.column_stride, columns, distance_scale,
                                   gromov.column_distances, lane);
    simdgroup_barrier(mem_flags::mem_device);
    long spreads = 0;
    if (lane == 0)
        spreads = nk_shapes_distance_spread_f16_metal_(gromov.row_distances, rows) +
                  nk_shapes_distance_spread_f16_metal_(gromov.column_distances, columns);
    float best = INFINITY;
    for (uint start = 0; start != 3; ++start) {
        long plan_score = 0;
        if (start != 2)
            plan_score = nk_shapes_start_plan_f16_metal_(gromov, start ? gromov.improper_plan : gromov.proper_plan,
                                                         lane);
        else if (lane == 0) plan_score = nk_shapes_product_plan_f16_metal_(gromov);
        simdgroup_barrier(mem_flags::mem_device);
        plan_score = nk_shapes_frank_wolfe_f16_metal_(network, gromov, plan_score, lane);
        float loss = 0;
        if (lane == 0) {
            long const mean_score = ((plan_score / long(cells)) << gromov.gradient_shift) / long(cells);
            loss = float(spreads - 2 * mean_score) / (distance_scale * distance_scale);
        }
        loss = simd_broadcast_first(loss);
        if (!(loss < best)) continue;
        best = loss;
        nk_shapes_plan_fit_f16_metal_(network, gromov, lane, rotation);
    }
    return best;
}

kernel void nk_gromovwassersteins_packed_f16_metal_kernel_(
    device ulong const *a_table [[buffer(0)]], device ulong const *b_table [[buffer(1)]],
    device uchar *results [[buffer(2)]], device float *rotations [[buffer(3)]],
    constant nk_shapes_arguments_metal_t &arguments [[buffer(4)]], device uchar *workspace [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    nk_shapes_transport_pair_metal_t opened;
    if (!nk_shapes_transport_pair_metal_(a_table, b_table, arguments, group, groups, simdgroup, opened)) return;
    float inner[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float distance = opened.rows == opened.columns ? 0 : INFINITY;
    if (opened.rows && opened.columns) {
        device uchar *slot = workspace + (opened.pair - arguments.pair_begin) * arguments.slot_bytes;
        nk_shapes_network_metal_t network = nk_shapes_network_f16_metal_(slot, arguments.largest, opened.row_packed,
                                                                         opened.column_packed);
        nk_shapes_gromov_metal_t const gromov = nk_shapes_gromov_f16_metal_(slot, arguments.largest, network,
                                                                            opened.row_packed, opened.column_packed);
        distance = precise::sqrt(fmax(nk_gromovwasserstein_pair_f16_metal_(network, gromov, lane, inner), 0.0f)) / 2;
    }
    if (lane == 0) nk_shapes_write_transport_metal_(results, rotations, arguments, opened, distance, inner);
}

#pragma endregion Gromov Wasserstein
