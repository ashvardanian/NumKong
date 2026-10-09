/**
 *  @file include/numkong/dots/metal.metal
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products on the SIMT cores of every Apple GPU, from Metal family 7 on.
 *
 *  @sa include/numkong/dots/metal.h, which embeds and launches this source
 *  @sa include/numkong/dots/simt.cuh, the CUDA and ROCm sibling
 *
 *  The baseline every Apple capability stands on, needing only Metal 3.1: no @c matmul2d and no
 *  @c simdgroup_matrix. Each of 256 threads owns a 4 × 4 grid of one 64 × 64 output tile, over
 *  16-deep slabs of A and B widened into threadgroup memory, accumulating in @c float, or exactly
 *  in @c int for integer codes, as the serial backends do; Apple GPUs have no F64. The pack and
 *  the launch records live here too, for the other sources to build on, while the dtype classes
 *  they read live in `types.metal`.
 *
 *  Block-scaled tiles multiply each block's sum by its two scales and add it into a compensated
 *  @c float pair carrying its own exponent, so blocks of any spread round once, and the tensor
 *  scales multiply in the epilogue. Integer distances form ab − d² and a + b − 2d in 64-bit
 *  integers before a @c float tail, as the CPU finalizers do.
 */

/** The four codes from dimension 4 × @p index of a 16-byte chunk of a row: its 16-bit halves, its
 *  bytes, or the nibbles of its bytes, the high nibble first. */
template <typename dtype_>
inline ushort4 nk_chunk_codes_(uint4 chunk, uint index) {
    if (sizeof(typename dtype_::raw_t) == 2) return as_type<ushort4>(index ? chunk.zw : chunk.xy);
    uchar4 const bytes = as_type<uchar4>(chunk[index / dtype_::dimensions_per_value]);
    if (dtype_::dimensions_per_value == 1) return ushort4(bytes);
    uchar2 const pair = index & 1 ? bytes.zw : bytes.xy;
    return ushort4(pair.x >> 4, pair.x & 15, pair.y >> 4, pair.y & 15);
}

/** The bytes at @p row holding its last @p dimensions, at most a chunk of them, read one at a time
 *  so no load passes the row's end, and zero after. */
template <typename dtype_>
inline uint4 nk_tail_load_(device uchar const *row, uint dimensions) {
    uint const count = nk::divide_round_up(dimensions * sizeof(typename dtype_::raw_t), dtype_::dimensions_per_value);
    uint4 words = uint4(0);
    for (uint byte = 0; byte != count; ++byte) words[byte / 4] |= uint(row[byte]) << (byte % 4 * 8);
    return words;
}

/** The launch record of a pack, laid out as the C launcher's record of the same name. */
struct nk_cross_pack_arguments_metal_t {
    ulong b_stride, depth_bytes, row_bytes, capability;
    uint column_count, depth, depth_padded_values, columns_begin, columns_end;
    ulong scales_stride;
    uint tensor_present;
};

/** The 64-byte header every packed buffer opens with, laid out as the C header of the same name. */
struct nk_cross_packed_buffer_header_t {
    uint column_count, depth_dimensions, depth_padded_values, scales_stride;
    float tensor_scale;
    uint norms_offset, reserved[8];
    ulong capability;
};

static_assert(sizeof(nk_cross_pack_arguments_metal_t) == 72, "mirrors the C record in dots/metal.h");
static_assert(sizeof(nk_cross_packed_buffer_header_t) == 64, "mirrors the C header in dots/serial.h");

/** Packs one column: the row zero-padded to its padded depth, then its sum of squares among the
 *  norms after every row. B's stride need not be value-aligned, so values come back from bytes. */
template <typename dtype_>
void nk_cross_pack_metal_(device uchar const *b, device uchar *b_packed,
                          constant nk_cross_pack_arguments_metal_t &arguments, uint column_first, uint lane) {
    using norm_t = typename dtype_::norm_t;
    constexpr uint value_bytes = sizeof(typename dtype_::raw_t);
    if (arguments.columns_begin == 0 && column_first == 0 && lane == 0) {
        device nk_cross_packed_buffer_header_t *header = (device nk_cross_packed_buffer_header_t *)b_packed;
        header->column_count = arguments.column_count;
        header->depth_dimensions = arguments.depth;
        header->depth_padded_values = arguments.depth_padded_values;
        header->scales_stride = 0;
        header->tensor_scale = 1;
        header->norms_offset = (uint)(sizeof(nk_cross_packed_buffer_header_t) +
                                      arguments.column_count * arguments.row_bytes);
        header->capability = arguments.capability;
        for (uint reserved = 0; reserved != 8; ++reserved) header->reserved[reserved] = 0;
    }
    uint const column = arguments.columns_begin + column_first;
    if (column >= arguments.columns_end) return;
    device uchar *rows = b_packed + sizeof(nk_cross_packed_buffer_header_t);
    device uchar const *source = b + column * arguments.b_stride;
    device uchar *destination = rows + column * arguments.row_bytes;
    for (ulong byte = lane; byte < arguments.row_bytes; byte += 32)
        destination[byte] = byte < arguments.depth_bytes ? source[byte] : 0;

    norm_t norm = 0;
    uint const values = arguments.depth / dtype_::dimensions_per_value;
    for (uint index = lane; index < values; index += 32) {
        uint bits = 0;
        for (uint byte = 0; byte != value_bytes; ++byte) bits |= (uint)source[index * value_bytes + byte] << (8 * byte);
        norm += dtype_::squares(bits);
    }
    norm = simd_sum(norm);
    device norm_t *norms = (device norm_t *)(rows + arguments.column_count * arguments.row_bytes);
    if (lane == 0) norms[column] = norm;
}

/** Side of the output tile a threadgroup owns, depth of one staged slab, and threads per group. */
constant uint nk_cross_tile_metal_k = 64, nk_cross_slab_metal_k = 16, nk_cross_threads_metal_k = 256;

/** The launch record of every capability's tile, A and B strides in bytes and C's in results. */
struct nk_cross_arguments_metal_t {
    ulong a_stride, b_stride, c_stride;
    uint rows_begin, rows_end, column_count, depth;

    /** Nonzero keeps only cells on and above the diagonal, the symmetric kernels' output. */
    uint upper_triangle;
    ulong a_scales_stride, b_scales_stride;
    uint a_tensor_present, b_tensor_present;
};

static_assert(sizeof(nk_cross_arguments_metal_t) == 72, "mirrors the C record in dots/metal.h");

enum nk_cross_metric_metal_t { nk_cross_dot_metal_k, nk_cross_angular_metal_k, nk_cross_euclidean_metal_k };

template <nk_cross_metric_metal_t metric_>
float nk_cross_subnormal_distance_metal_(float dot, float a_norm, float b_norm);

/** A distance from its dot and squared norms: a NaN dot gives NaN, and angular is 0 for two
 *  zero norms and 1 for one zero norm or a zero dot. Bit tests stand in for the comparisons
 *  fast math drops. */
template <nk_cross_metric_metal_t metric_>
float nk_cross_distance_metal_(float dot, float a_norm, float b_norm) {
    uint const dot_bits = as_type<uint>(dot), a_bits = as_type<uint>(a_norm), b_bits = as_type<uint>(b_norm);
    if ((dot_bits & 0x7fffffff) > 0x7f800000) return dot;
    if (metric_ == nk_cross_angular_metal_k) {
        bool const a_zero = (a_bits & 0x7fffffff) == 0, b_zero = (b_bits & 0x7fffffff) == 0;
        if (a_zero && b_zero) return 0.0f;
        if (a_zero || b_zero || (dot_bits & 0x7fffffff) == 0) return 1.0f;
    }
    if (((dot_bits & 0x7f800000) == 0 && (dot_bits & 0x7fffff)) ||
        ((a_bits & 0x7f800000) == 0 && (a_bits & 0x7fffff)) || ((b_bits & 0x7f800000) == 0 && (b_bits & 0x7fffff)))
        return nk_cross_subnormal_distance_metal_<metric_>(dot, a_norm, b_norm);
    if (metric_ == nk_cross_angular_metal_k) {
        float const distance = 1.0f - dot * rsqrt(a_norm) * rsqrt(b_norm);
        return distance > 0 ? distance : 0.0f;
    }
    float const squared = a_norm + b_norm - 2.0f * dot;
    return squared > 0 ? sqrt(squared) : 0.0f;
}

/** A distance from an exact integer dot and squared norms, as @c nk_cross_integer_metric_simt_ in
 *  `dots/simt.cuh`: ab − d² and a + b − 2d stay exact in 64 bits, so equal rows are exactly 0 apart
 *  and only the correctly rounded F32 tail rounds. */
template <nk_cross_metric_metal_t metric_>
float nk_cross_distance_metal_(long dot, uint a_norm, uint b_norm) {
    if (metric_ == nk_cross_euclidean_metal_k) {
        long const distance_squared = long(a_norm) + long(b_norm) - 2 * dot;
        return distance_squared > 0 ? precise::sqrt(float(distance_squared)) : 0.0f;
    }
    ulong const product = ulong(a_norm) * b_norm;
    if (product == 0) return (a_norm | b_norm) != 0 ? 1.0f : 0.0f;
    float const root = precise::sqrt(float(product));
    if (dot <= 0) return 1.0f - precise::divide(float(dot), root);
    ulong const dot_squared = ulong(dot) * ulong(dot);
    return precise::divide(float(product - dot_squared), float(product) + float(dot) * root);
}

template <typename dtype_, uint side_ = nk_cross_tile_metal_k>
void nk_cross_norms_metal_(device uchar const *a, device uchar const *b, constant nk_cross_arguments_metal_t &arguments,
                           uint2 group, uint thread_index, threadgroup typename dtype_::norm_t (*norms)[side_]) {
    if (thread_index < side_) {
        uint const row = arguments.rows_begin + group.y * side_ + thread_index;
        uint const column = group.x * side_ + thread_index;
        typename dtype_::norm_t a_norm = 0, b_norm = 0;
        for (uint depth = 0; depth < arguments.depth; ++depth) {
            if (row < arguments.rows_end) {
                auto const value = dtype_::load(a + row * arguments.a_stride, depth);
                a_norm += value * value;
            }
            if (arguments.upper_triangle && column < arguments.column_count) {
                auto const value = dtype_::load(b + column * arguments.b_stride, depth);
                b_norm += value * value;
            }
        }
        if (!arguments.upper_triangle && column < arguments.column_count)
            b_norm = ((device
                       typename dtype_::norm_t const *)(b + arguments.column_count * arguments.b_stride))[column];
        norms[0][thread_index] = a_norm;
        norms[1][thread_index] = b_norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/** One @b [64,64] tile of C = A × Bᵀ, with B given row-major as @b [columns,depth], on the SIMT
 *  cores: each of 256 threads owns a 4 × 4 grid of outputs strided by 16, so threadgroup reads
 *  broadcast along one axis and sweep the banks along the other. */
template <typename dtype_, typename result_type_, nk_cross_metric_metal_t metric_ = nk_cross_dot_metal_k>
void nk_cross_tile_metal_(device uchar const *a, device uchar const *b, device result_type_ *c,
                          constant nk_cross_arguments_metal_t &arguments, uint2 group, uint thread_index,
                          threadgroup typename dtype_::dot_result_t (*a_slab)[nk_cross_tile_metal_k + 1],
                          threadgroup typename dtype_::dot_result_t (*b_slab)[nk_cross_tile_metal_k + 1],
                          threadgroup typename dtype_::norm_t const (*norms)[nk_cross_tile_metal_k] = nullptr) {
    using accumulator_t = typename dtype_::dot_result_t;
    constexpr uint side = nk_cross_tile_metal_k, slab_depth = nk_cross_slab_metal_k;
    uint const first_row = arguments.rows_begin + group.y * side, first_column = group.x * side;
    if (arguments.upper_triangle && first_column + side <= first_row) return; // below the diagonal, group-uniform
    uint const thread_column = thread_index & 15, thread_row = thread_index >> 4;

    accumulator_t sums[4][4];
    for (uint row_step = 0; row_step != 4; ++row_step)
        for (uint column_step = 0; column_step != 4; ++column_step) sums[row_step][column_step] = 0;

    for (uint slab = 0; slab < arguments.depth; slab += slab_depth) {
        for (uint element = thread_index; element < side * slab_depth; element += nk_cross_threads_metal_k) {
            uint const line = element / slab_depth, offset = element % slab_depth, index = slab + offset;
            uint const row = first_row + line, column = first_column + line;
            bool const inside = index < arguments.depth;
            a_slab[offset][line] = inside && row < arguments.rows_end
                                       ? dtype_::load(a + row * arguments.a_stride, index)
                                       : accumulator_t(0);
            b_slab[offset][line] = inside && column < arguments.column_count
                                       ? dtype_::load(b + column * arguments.b_stride, index)
                                       : accumulator_t(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 0; offset != slab_depth; ++offset) {
            accumulator_t a_values[4], b_values[4];
            for (uint step = 0; step != 4; ++step)
                a_values[step] = a_slab[offset][thread_row + 16 * step],
                b_values[step] = b_slab[offset][thread_column + 16 * step];
            for (uint row_step = 0; row_step != 4; ++row_step)
                for (uint column_step = 0; column_step != 4; ++column_step)
                    sums[row_step][column_step] += a_values[row_step] * b_values[column_step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint row_step = 0; row_step != 4; ++row_step) {
        uint const row = first_row + thread_row + 16 * row_step;
        if (row >= arguments.rows_end) continue;
        for (uint column_step = 0; column_step != 4; ++column_step) {
            uint const column = first_column + thread_column + 16 * column_step;
            if (column >= arguments.column_count || (arguments.upper_triangle && column < row)) continue;
            result_type_ value = (result_type_)sums[row_step][column_step];
            if (metric_ != nk_cross_dot_metal_k)
                value = arguments.upper_triangle && row == column
                            ? 0.0f
                            : nk_cross_distance_metal_<metric_>(sums[row_step][column_step], norms[0][row - first_row],
                                                                norms[1][column - first_column]);
            c[row * arguments.c_stride + column] = value;
        }
    }
}

#pragma region Kernels

kernel void nk_dots_pack_i8_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                          constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                          uint simdgroup [[simdgroup_index_in_threadgroup]],
                                          uint simdgroups [[simdgroups_per_threadgroup]],
                                          uint group [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::i8_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_u8_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                          constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                          uint simdgroup [[simdgroup_index_in_threadgroup]],
                                          uint simdgroups [[simdgroups_per_threadgroup]],
                                          uint group [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::u8_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_i4_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                          constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                          uint simdgroup [[simdgroup_index_in_threadgroup]],
                                          uint simdgroups [[simdgroups_per_threadgroup]],
                                          uint group [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::i4x2_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_u4_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                          constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                          uint simdgroup [[simdgroup_index_in_threadgroup]],
                                          uint simdgroups [[simdgroups_per_threadgroup]],
                                          uint group [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::u4x2_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_f16_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                           constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                           uint simdgroup [[simdgroup_index_in_threadgroup]],
                                           uint simdgroups [[simdgroups_per_threadgroup]],
                                           uint group [[threadgroup_position_in_grid]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::f16_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_bf16_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                            constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                                            uint simdgroups [[simdgroups_per_threadgroup]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::bf16_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_e4m3_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                            constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                                            uint simdgroups [[simdgroups_per_threadgroup]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::e4m3_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_e5m2_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                            constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                                            uint simdgroups [[simdgroups_per_threadgroup]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::e5m2_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_e3m2_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                            constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                                            uint simdgroups [[simdgroups_per_threadgroup]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::e3m2_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_e2m3_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                            constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                                            uint simdgroups [[simdgroups_per_threadgroup]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::e2m3_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}
kernel void nk_dots_pack_e2m1_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *b_packed [[buffer(1)]],
                                            constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                                            uint simdgroups [[simdgroups_per_threadgroup]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_pack_metal_<nk::e2m1x2_t>(b, b_packed, arguments, group * simdgroups + simdgroup, lane);
}

kernel void nk_dots_i8_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                     device int *c [[buffer(2)]],
                                     constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::i8_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_u8_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                     device uint *c [[buffer(2)]],
                                     constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::u8_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_i4_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                     device int *c [[buffer(2)]],
                                     constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::i4x2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_u4_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                     device uint *c [[buffer(2)]],
                                     constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::u4x2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_f16_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                      device float *c [[buffer(2)]],
                                      constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::f16_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_bf16_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::bf16_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e4m3_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e4m3_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e5m2_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e5m2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e3m2_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e3m2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e2m3_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e2m3_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e2m1_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_tile_metal_<nk::e2m1x2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}

#pragma endregion Kernels

#pragma clang fp contract(off) reassociate(off)

struct nk_cross_scaled_sum_metal_t {
    float2 sum;
    int exponent;
};

float nk_cross_float_mantissa_metal_(float value, thread int &exponent) {
    uint const bits = as_type<uint>(value), fraction = bits & 0x7fffff, biased = (bits >> 23) & 255;
    if (biased == 255 || (biased == 0 && fraction == 0)) {
        exponent = 0;
        return value;
    }
    if (biased) {
        exponent = int(biased) - 127;
        return as_type<float>((bits & 0x807fffff) | 0x3f800000);
    }
    int shift;
    float const mantissa = frexp(float(fraction), shift);
    exponent = shift - 149;
    return bits & 0x80000000 ? -mantissa : mantissa;
}

void nk_cross_scaled_add_metal_(thread nk_cross_scaled_sum_metal_t &state, float value, int exponent) {
    if (!isfinite(value) || !isfinite(state.sum.x)) {
        state.sum = float2(state.sum.x + value, 0);
        state.exponent = 0;
        return;
    }
    if (value == 0) return;
    int shift;
    value = frexp(value, shift);
    exponent += shift;
    if (all(state.sum == float2(0))) {
        state.sum = float2(value, 0);
        state.exponent = exponent;
        return;
    }
    int const common = max(state.exponent, exponent);
    float const high = ldexp(state.sum.x, state.exponent - common);
    float const low = ldexp(state.sum.y, state.exponent - common);
    value = ldexp(value, exponent - common);
    float const sum = high + value, split = sum - high;
    float const error = (high - (sum - split)) + (value - split) + low;
    float const rounded = sum + error;
    state.sum = float2(rounded, error - (rounded - sum));
    state.exponent = common;
}

uint nk_cross_scaled_bits_metal_(float2 sum, int exponent) {
    if (!isfinite(sum.x) || all(sum == float2(0))) return as_type<uint>(sum.x);
    uint const sign = as_type<uint>(sum.x) & 0x80000000;
    if (sign) sum = -sum;
    int shift;
    float const high = frexp(sum.x, shift);
    sum = float2(high, ldexp(sum.y, -shift));
    exponent += shift;
    if (exponent >= -125) return as_type<uint>(ldexp(sum.x + sum.y, exponent)) | sign;
    if (exponent < -149) return sign;
    float const scaled = ldexp(sum.x, exponent + 149), error = ldexp(sum.y, exponent + 149);
    float rounded = rint(scaled);
    if (abs(scaled - rounded) == 0.5f && error != 0) rounded = error > 0 ? ceil(scaled) : floor(scaled);
    return sign | uint(rounded);
}

uint nk_cross_tensor_product_metal_(float a, float b) {
    int a_exponent, b_exponent;
    float const a_mantissa = nk_cross_float_mantissa_metal_(a, a_exponent);
    float const b_mantissa = nk_cross_float_mantissa_metal_(b, b_exponent);
    float const product = a_mantissa * b_mantissa;
    return nk_cross_scaled_bits_metal_(float2(product, isfinite(product) ? fma(a_mantissa, b_mantissa, -product) : 0),
                                       a_exponent + b_exponent);
}

float nk_cross_scaled_dot_metal_(nk_cross_scaled_sum_metal_t state, uint tensor_product) {
    uint const dot_bits = nk_cross_scaled_bits_metal_(state.sum, state.exponent);
    if (tensor_product == 0x3f800000) return as_type<float>(dot_bits);
    int dot_exponent, tensor_exponent;
    float const dot = nk_cross_float_mantissa_metal_(as_type<float>(dot_bits), dot_exponent);
    float const tensor = nk_cross_float_mantissa_metal_(as_type<float>(tensor_product), tensor_exponent);
    float const product = dot * tensor;
    return as_type<float>(nk_cross_scaled_bits_metal_(
        float2(product, isfinite(product) ? fma(dot, tensor, -product) : 0), dot_exponent + tensor_exponent));
}

template <nk_cross_metric_metal_t metric_>
float nk_cross_subnormal_distance_metal_(float dot, float a_norm, float b_norm) {
    uint const a_bits = as_type<uint>(a_norm), b_bits = as_type<uint>(b_norm);
    if (metric_ == nk_cross_angular_metal_k) {
        int a_exponent, b_exponent, dot_exponent;
        float const a_mantissa = nk_cross_float_mantissa_metal_(a_norm, a_exponent);
        float const b_mantissa = nk_cross_float_mantissa_metal_(b_norm, b_exponent);
        float const a_inverse = (a_bits & 0x7f800000) == 0 ? ldexp(rsqrt(ldexp(a_mantissa, a_exponent + 24)), 12)
                                                           : rsqrt(a_norm);
        float const b_inverse = (b_bits & 0x7f800000) == 0 ? ldexp(rsqrt(ldexp(b_mantissa, b_exponent + 24)), 12)
                                                           : rsqrt(b_norm);
        nk_cross_scaled_sum_metal_t state;
        state.sum = float2(nk_cross_float_mantissa_metal_(dot, dot_exponent), 0);
        state.exponent = dot_exponent;
        float const normalized = nk_cross_scaled_dot_metal_(state,
                                                            nk_cross_tensor_product_metal_(a_inverse, b_inverse));
        float const distance = 1 - normalized;
        return distance > 0 ? distance : 0;
    }
    int a_exponent, b_exponent;
    float const a_mantissa = nk_cross_float_mantissa_metal_(a_norm, a_exponent);
    float const b_mantissa = nk_cross_float_mantissa_metal_(b_norm, b_exponent);
    nk_cross_scaled_sum_metal_t state = {float2(0), 0};
    nk_cross_scaled_add_metal_(state, a_mantissa, a_exponent);
    nk_cross_scaled_add_metal_(state, b_mantissa, b_exponent);
    float const rounded_norms = as_type<float>(nk_cross_scaled_bits_metal_(state.sum, state.exponent));
    float const doubled_dot = as_type<float>(nk_cross_tensor_product_metal_(dot, 2));
    state = {float2(0), 0};
    float const rounded_mantissa = nk_cross_float_mantissa_metal_(rounded_norms, a_exponent);
    float const doubled_mantissa = nk_cross_float_mantissa_metal_(doubled_dot, b_exponent);
    nk_cross_scaled_add_metal_(state, rounded_mantissa, a_exponent);
    nk_cross_scaled_add_metal_(state, -doubled_mantissa, b_exponent);
    uint const squared_bits = nk_cross_scaled_bits_metal_(state.sum, state.exponent);
    float const squared = as_type<float>(squared_bits);
    if ((squared_bits & 0x80000000) || (squared_bits & 0x7fffffff) == 0 || isnan(squared)) return 0;
    if ((squared_bits & 0x7f800000) != 0) return sqrt(squared);
    int exponent;
    float const mantissa = nk_cross_float_mantissa_metal_(squared, exponent);
    return ldexp(sqrt(ldexp(mantissa, exponent & 1)), exponent >> 1);
}

enum nk_cross_scale_metal_t { nk_cross_scale_e4m3_metal_k, nk_cross_scale_e8m0_metal_k };

template <nk_cross_scale_metal_t scale_>
float nk_cross_block_scale_metal_(uchar a, uchar b, thread int &exponent) {
    exponent = 0;
    if (scale_ == nk_cross_scale_e4m3_metal_k)
        return float(nk::e4m3_t::widen(ushort4(a))[0]) * float(nk::e4m3_t::widen(ushort4(b))[0]);
    exponent = int(a) + int(b) - 254;
    return a == 255 || b == 255 ? NAN : a == 0 || b == 0 ? 0.0f : 1.0f;
}

template <typename dtype_, uint block_size_, nk_cross_scale_metal_t scale_>
float nk_cross_scaled_norm_metal_(device uchar const *row, device uchar const *scales, uint depth, float tensor) {
    nk_cross_scaled_sum_metal_t state = {float2(0), 0};
    for (uint block = 0; block != depth / block_size_; ++block) {
        float2 squares = float2(0);
        for (uint dimension = 0; dimension != block_size_; ++dimension) {
            float const value = dtype_::load(row, block * block_size_ + dimension);
            float const term = value * value, sum = squares.x + term, split = sum - squares.x;
            float const error = (squares.x - (sum - split)) + (term - split) + squares.y;
            squares = float2(sum + error, error - ((sum + error) - sum));
        }
        int exponent;
        float const scale = nk_cross_block_scale_metal_<scale_>(scales[block], scales[block], exponent);
        float const product = squares.x * scale;
        nk_cross_scaled_add_metal_(state, product, exponent);
        if (isfinite(product)) nk_cross_scaled_add_metal_(state, fma(squares.x, scale, -product), exponent);
        nk_cross_scaled_add_metal_(state, squares.y * scale, exponent);
    }
    int exponent;
    float const mantissa = nk_cross_float_mantissa_metal_(tensor, exponent);
    for (uint step = 0; step != 2; ++step) {
        float const product = state.sum.x * mantissa;
        float const error = isfinite(product) ? fma(state.sum.x, mantissa, -product) + state.sum.y * mantissa : 0;
        float const rounded = product + error;
        state.sum = float2(rounded, error - (rounded - product));
    }
    state.exponent += 2 * exponent;
    return as_type<float>(nk_cross_scaled_bits_metal_(state.sum, state.exponent));
}

template <typename dtype_, uint block_size_, nk_cross_scale_metal_t scale_>
void nk_cross_scaled_pack_metal_(device uchar const *b, device uchar *packed, device uchar const *scales,
                                 device float const *tensor_pointer,
                                 constant nk_cross_pack_arguments_metal_t &arguments, uint column_first, uint lane) {
    uint const blocks = arguments.depth / block_size_;
    uint const stride = nk::round_up_to_multiple(blocks, 16u);
    float const tensor = arguments.tensor_present ? *tensor_pointer : 1;
    if (arguments.columns_begin == 0 && column_first == 0 && lane == 0) {
        device nk_cross_packed_buffer_header_t *header = (device nk_cross_packed_buffer_header_t *)packed;
        header->column_count = arguments.column_count;
        header->depth_dimensions = arguments.depth;
        header->depth_padded_values = arguments.depth_padded_values;
        header->scales_stride = stride;
        header->tensor_scale = tensor;
        header->norms_offset = (uint)(sizeof(nk_cross_packed_buffer_header_t) +
                                      arguments.column_count * (arguments.row_bytes + stride));
        header->capability = arguments.capability;
        for (uint index = 0; index != 8; ++index) header->reserved[index] = 0;
    }
    uint const column = arguments.columns_begin + column_first;
    if (column >= arguments.columns_end) return;
    device uchar *rows = packed + sizeof(nk_cross_packed_buffer_header_t);
    device uchar const *source = b + column * arguments.b_stride;
    for (ulong byte = lane; byte < arguments.row_bytes; byte += 32)
        rows[column * arguments.row_bytes + byte] = byte < arguments.depth_bytes ? source[byte] : 0;
    device uchar *packed_scales = rows + arguments.column_count * arguments.row_bytes;
    for (uint byte = lane; byte < stride; byte += 32)
        packed_scales[column * stride + byte] = byte < blocks ? scales[column * arguments.scales_stride + byte] : 0;
    if (lane == 0) {
        device float *norms = (device float *)(packed_scales + arguments.column_count * stride);
        norms[column] = nk_cross_scaled_norm_metal_<dtype_, block_size_, scale_>(
            source, scales + column * arguments.scales_stride, arguments.depth, tensor);
    }
}

template <typename dtype_, uint block_size_, nk_cross_scale_metal_t scale_, uint side_ = nk_cross_tile_metal_k>
void nk_cross_scaled_norms_metal_(device uchar const *a, device uchar const *b, device uchar const *a_scales,
                                  device uchar const *b_scales, float a_tensor, float b_tensor,
                                  constant nk_cross_arguments_metal_t &arguments, uint2 group, uint thread_index,
                                  threadgroup float (*norms)[side_]) {
    if (thread_index < side_) {
        uint const row = arguments.rows_begin + group.y * side_ + thread_index;
        uint const column = group.x * side_ + thread_index;
        norms[0][thread_index] = row < arguments.rows_end
                                     ? nk_cross_scaled_norm_metal_<dtype_, block_size_, scale_>(
                                           a + row * arguments.a_stride, a_scales + row * arguments.a_scales_stride,
                                           arguments.depth, a_tensor)
                                     : 0;
        uint const blocks = arguments.depth / block_size_, stride = nk::round_up_to_multiple(blocks, 16u);
        norms[1][thread_index] =
            column >= arguments.column_count ? 0
            : arguments.upper_triangle
                ? nk_cross_scaled_norm_metal_<dtype_, block_size_, scale_>(
                      b + column * arguments.b_stride, b_scales + column * arguments.b_scales_stride, arguments.depth,
                      b_tensor)
                : ((device float const *)(b + arguments.column_count * (arguments.b_stride + stride)))[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <nk_cross_scale_metal_t scale_>
void nk_cross_scaled_block_add_metal_(thread nk_cross_scaled_sum_metal_t &state, float dot, uchar a_scale,
                                      uchar b_scale) {
#pragma clang fp contract(off) reassociate(off)
    int exponent;
    float const scale = nk_cross_block_scale_metal_<scale_>(a_scale, b_scale, exponent);
    float const product = dot * scale;
    if (scale_ == nk_cross_scale_e4m3_metal_k) {
        // NVFP4 block products fit in 20 significant bits and cannot overflow F32.
        float const sum = state.sum.x + product, split = sum - state.sum.x;
        float const error = (state.sum.x - (sum - split)) + (product - split) + state.sum.y;
        float const rounded = sum + error;
        state.sum = isfinite(product) && isfinite(state.sum.x) ? float2(rounded, error - (rounded - sum))
                                                               : float2(state.sum.x + product, 0);
    }
    else nk_cross_scaled_add_metal_(state, product, exponent);
}

template <typename dtype_, uint block_size_, nk_cross_scale_metal_t scale_, nk_cross_metric_metal_t metric_>
void nk_cross_scaled_tile_metal_(device uchar const *a, device uchar const *b, device float *c,
                                 device uchar const *a_scales, device uchar const *b_scales, float a_tensor,
                                 float b_tensor, constant nk_cross_arguments_metal_t &arguments, uint2 group,
                                 uint thread_index, threadgroup float (*a_slab)[nk_cross_tile_metal_k + 1],
                                 threadgroup float (*b_slab)[nk_cross_tile_metal_k + 1],
                                 threadgroup float const (*norms)[nk_cross_tile_metal_k]) {
    constexpr uint side = nk_cross_tile_metal_k;
    uint const first_row = arguments.rows_begin + group.y * side, first_column = group.x * side;
    if (arguments.upper_triangle && first_column + side <= first_row) return;
    uint const thread_column = thread_index & 15, thread_row = thread_index >> 4;
    nk_cross_scaled_sum_metal_t sums[4][4] = {};
    for (uint block = 0; block != arguments.depth / block_size_; ++block) {
        float dots[4][4] = {};
        for (uint slab = 0; slab != block_size_; slab += nk_cross_slab_metal_k) {
            for (uint element = thread_index; element < side * nk_cross_slab_metal_k;
                 element += nk_cross_threads_metal_k) {
                uint const line = element / nk_cross_slab_metal_k, offset = element % nk_cross_slab_metal_k;
                uint const row = first_row + line, column = first_column + line,
                           dimension = block * block_size_ + slab + offset;
                a_slab[offset][line] = row < arguments.rows_end ? dtype_::load(a + row * arguments.a_stride, dimension)
                                                                : 0;
                b_slab[offset][line] = column < arguments.column_count
                                           ? dtype_::load(b + column * arguments.b_stride, dimension)
                                           : 0;
                if (scale_ == nk_cross_scale_e4m3_metal_k) {
                    if (row < arguments.rows_end)
                        a_slab[offset][line] *= float(
                            nk::e4m3_t::widen(ushort4(a_scales[row * arguments.a_scales_stride + block]))[0]);
                    if (column < arguments.column_count)
                        b_slab[offset][line] *= float(
                            nk::e4m3_t::widen(ushort4(b_scales[column * arguments.b_scales_stride + block]))[0]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint offset = 0; offset != nk_cross_slab_metal_k; ++offset)
                for (uint row_step = 0; row_step != 4; ++row_step)
                    for (uint column_step = 0; column_step != 4; ++column_step)
                        dots[row_step][column_step] += a_slab[offset][thread_row + 16 * row_step] *
                                                       b_slab[offset][thread_column + 16 * column_step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint row_step = 0; row_step != 4; ++row_step)
            for (uint column_step = 0; column_step != 4; ++column_step) {
                uint const row = first_row + thread_row + 16 * row_step,
                           column = first_column + thread_column + 16 * column_step;
                if (row < arguments.rows_end && column < arguments.column_count)
                    nk_cross_scaled_block_add_metal_<scale_>(
                        sums[row_step][column_step], dots[row_step][column_step],
                        scale_ == nk_cross_scale_e4m3_metal_k ? 0x38
                                                              : a_scales[row * arguments.a_scales_stride + block],
                        scale_ == nk_cross_scale_e4m3_metal_k ? 0x38
                                                              : b_scales[column * arguments.b_scales_stride + block]);
            }
    }
    uint const tensor_product = nk_cross_tensor_product_metal_(a_tensor, b_tensor);
    for (uint output = 0; output != 16; ++output) {
        uint const row_step = output / 4, column_step = output % 4;
        uint const row = first_row + thread_row + 16 * row_step,
                   column = first_column + thread_column + 16 * column_step;
        if (row >= arguments.rows_end || column >= arguments.column_count || (arguments.upper_triangle && column < row))
            continue;
        float value = nk_cross_scaled_dot_metal_(sums[row_step][column_step], tensor_product);
        if (metric_ != nk_cross_dot_metal_k) {
            float const a_norm = norms[0][row - first_row], b_norm = norms[1][column - first_column];
            value = arguments.upper_triangle && row == column
                        ? 0.0f
                        : nk_cross_distance_metal_<metric_>(value, a_norm, b_norm);
        }
        c[row * arguments.c_stride + column] = value;
    }
}

kernel void nk_dots_pack_mxfp8e4m3_metal_kernel_(
    device uchar const *b [[buffer(0)]], device uchar *packed [[buffer(1)]],
    constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]], device uchar const *scales [[buffer(3)]],
    device float const *tensor [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint simdgroups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_scaled_pack_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k>(b, packed, scales, tensor, arguments,
                                                                             group * simdgroups + simdgroup, lane);
}

kernel void nk_dots_pack_mxfp8e5m2_metal_kernel_(
    device uchar const *b [[buffer(0)]], device uchar *packed [[buffer(1)]],
    constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]], device uchar const *scales [[buffer(3)]],
    device float const *tensor [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint simdgroups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_scaled_pack_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k>(b, packed, scales, tensor, arguments,
                                                                             group * simdgroups + simdgroup, lane);
}

kernel void nk_dots_pack_mxfp6e2m3_metal_kernel_(
    device uchar const *b [[buffer(0)]], device uchar *packed [[buffer(1)]],
    constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]], device uchar const *scales [[buffer(3)]],
    device float const *tensor [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint simdgroups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_scaled_pack_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k>(b, packed, scales, tensor, arguments,
                                                                             group * simdgroups + simdgroup, lane);
}

kernel void nk_dots_pack_mxfp6e3m2_metal_kernel_(
    device uchar const *b [[buffer(0)]], device uchar *packed [[buffer(1)]],
    constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]], device uchar const *scales [[buffer(3)]],
    device float const *tensor [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint simdgroups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_scaled_pack_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k>(b, packed, scales, tensor, arguments,
                                                                             group * simdgroups + simdgroup, lane);
}

kernel void nk_dots_pack_mxfp4_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *packed [[buffer(1)]],
                                             constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                             device uchar const *scales [[buffer(3)]],
                                             device float const *tensor [[buffer(4)]],
                                             uint group [[threadgroup_position_in_grid]],
                                             uint simdgroup [[simdgroup_index_in_threadgroup]],
                                             uint simdgroups [[simdgroups_per_threadgroup]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_scaled_pack_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k>(b, packed, scales, tensor, arguments,
                                                                               group * simdgroups + simdgroup, lane);
}

kernel void nk_dots_pack_nvfp4_metal_kernel_(device uchar const *b [[buffer(0)]], device uchar *packed [[buffer(1)]],
                                             constant nk_cross_pack_arguments_metal_t &arguments [[buffer(2)]],
                                             device uchar const *scales [[buffer(3)]],
                                             device float const *tensor [[buffer(4)]],
                                             uint group [[threadgroup_position_in_grid]],
                                             uint simdgroup [[simdgroup_index_in_threadgroup]],
                                             uint simdgroups [[simdgroups_per_threadgroup]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    nk_cross_scaled_pack_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k>(b, packed, scales, tensor, arguments,
                                                                               group * simdgroups + simdgroup, lane);
}

kernel void nk_dots_mxfp8e4m3_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, nullptr);
}

kernel void nk_dots_mxfp8e5m2_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, nullptr);
}

kernel void nk_dots_mxfp6e2m3_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, nullptr);
}

kernel void nk_dots_mxfp6e3m2_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, nullptr);
}

kernel void nk_dots_mxfp4_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, nullptr);
}

kernel void nk_dots_nvfp4_metal_kernel_(
    device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]], device float *c [[buffer(2)]],
    constant nk_cross_arguments_metal_t &arguments [[buffer(3)]], device uchar const *a_scales [[buffer(4)]],
    device uchar const *b_scales [[buffer(5)]], device float const *a_tensor_pointer [[buffer(6)]],
    device float const *b_tensor_pointer [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    float const a_tensor = arguments.a_tensor_present ? *a_tensor_pointer : 1;
    float const b_tensor = arguments.b_tensor_present ? *b_tensor_pointer : 1;
    threadgroup float a_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1],
        b_slab[nk_cross_slab_metal_k][nk_cross_tile_metal_k + 1];
    nk_cross_scaled_tile_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k, nk_cross_dot_metal_k>(
        a, b, c, a_scales, b_scales, a_tensor, b_tensor, arguments, group, thread_index, a_slab, b_slab, nullptr);
}
