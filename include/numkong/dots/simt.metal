/**
 *  @file include/numkong/dots/simt.metal
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products on the SIMT cores of every Apple GPU, from Metal family 7 on.
 *
 *  @sa include/numkong/dots/simt.h, which embeds and launches this source
 *  @sa include/numkong/dots/simt.cuh, the CUDA and ROCm sibling
 *
 *  The baseline every Apple capability stands on, needing only Metal 3.1: no @c matmul2d and no
 *  @c simdgroup_matrix. Each of 256 threads owns a 4 × 4 grid of one 64 × 64 output tile, over
 *  16-deep slabs of A and B widened into threadgroup memory, accumulating in @c float, or exactly
 *  in @c int for integer codes, as the serial backends do; Apple GPUs have no F64. The dtype
 *  classes, the pack and the launch records live here too, for the other sources to build on.
 */
#include <metal_stdlib>

using namespace metal;

/** One dimension's code of a row of nibble pairs, the high nibble first as in @c nk_i4x2_t, so
 *  dimensions 2k and 2k + 1 share byte k. */
inline ushort nk_b4_load_(device uchar const *row, uint index) {
    return (ushort)((row[index >> 1] >> (~index & 1) * 4) & 15);
}

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
    uint const count = (dimensions * sizeof(typename dtype_::raw_t) + dtype_::dimensions_per_value - 1) /
                       dtype_::dimensions_per_value;
    uint4 words = uint4(0);
    for (uint byte = 0; byte != count; ++byte) words[byte / 4] |= uint(row[byte]) << (byte % 4 * 8);
    return words;
}

/** The square of one stored value of a widened format, both nibbles of a packed pair when two
 *  dimensions share a byte. Every widened format places its sign, exponent and mantissa bits where
 *  @c half keeps them and rescales by the difference of the biases, a power of two; subnormal codes
 *  land on @c half subnormals, which Apple GPUs keep, so every code widens exactly. */
template <typename dtype_>
inline float nk_widened_squares_(uint bits) {
    float4 const values = dtype_::dimensions_per_value == 2
                              ? float4(dtype_::widen(ushort4(bits & 0xF, (bits >> 4) & 0xF, 0, 0)))
                              : float4(dtype_::widen(ushort4(bits & 0xFF, 0, 0, 0)));
    return values.x * values.x + values.y * values.y;
}

/** One dimension of a widened format's row as @c float. */
template <typename dtype_>
inline float nk_widened_load_(device uchar const *row, uint index) {
    ushort const code = dtype_::dimensions_per_value == 2 ? nk_b4_load_(row, index) : (ushort)row[index];
    return (float)dtype_::widen(ushort4(code, 0, 0, 0)).x;
}

/**
 *  How the kernels read each dtype, named as in `types.hpp`: its stored @c raw_t, the dimensions
 *  one value holds, the @c dot_result_t the @c metal tile sums it in, and the @c norm_t of its
 *  packed squares. A float also names the @c stage_t it multiplies in and widens four codes into it.
 */
namespace nk {

struct i8_t {
    using raw_t = int8_t;
    using dot_result_t = int;
    using norm_t = uint;
    static constant constexpr uint dimensions_per_value = 1;
    static int load(device uchar const *row, uint index) { return (int)(char)row[index]; }
    static uint squares(uint bits) { return (uint)((int)(char)bits * (int)(char)bits); }
};

struct u8_t {
    using raw_t = uchar;
    using dot_result_t = uint;
    using norm_t = uint;
    static constant constexpr uint dimensions_per_value = 1;
    static uint load(device uchar const *row, uint index) { return row[index]; }
    static uint squares(uint bits) { return bits * bits; }
};

struct i4x2_t {
    using raw_t = uchar;
    using dot_result_t = int;
    using norm_t = uint;
    static constant constexpr uint dimensions_per_value = 2;
    static int load(device uchar const *row, uint index) { return ((int)nk_b4_load_(row, index) ^ 8) - 8; }
    static uint squares(uint bits) {
        int const low = ((int)(bits & 15) ^ 8) - 8, high = ((int)((bits >> 4) & 15) ^ 8) - 8;
        return (uint)(low * low + high * high);
    }
};

struct u4x2_t {
    using raw_t = uchar;
    using dot_result_t = uint;
    using norm_t = uint;
    static constant constexpr uint dimensions_per_value = 2;
    static uint load(device uchar const *row, uint index) { return nk_b4_load_(row, index); }
    static uint squares(uint bits) { return (bits & 15) * (bits & 15) + ((bits >> 4) & 15) * ((bits >> 4) & 15); }
};

struct f16_t {
    using raw_t = half;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static half4 widen(ushort4 codes) { return as_type<half4>(codes); }
    static float load(device uchar const *row, uint index) { return (float)((device half const *)row)[index]; }
    static float squares(uint bits) {
        float const value = (float)as_type<half>((ushort)bits);
        return value * value;
    }
};

struct bf16_t {
    using raw_t = bfloat;
    using stage_t = bfloat;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static bfloat4 widen(ushort4 codes) { return as_type<bfloat4>(codes); }
    static float load(device uchar const *row, uint index) { return (float)((device bfloat const *)row)[index]; }
    static float squares(uint bits) {
        float const value = as_type<float>(bits << 16);
        return value * value;
    }
};

/** E4M3FN: bias 7, and the two all-ones codes are NaN rather than infinities. */
struct e4m3_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static half4 widen(ushort4 codes) {
        ushort4 const bits = ((codes & ushort(0x7F)) << ushort(7)) | ((codes & ushort(0x80)) << ushort(8));
        return select(as_type<half4>(bits) * half(256), half4(NAN), (codes & ushort(0x7F)) == ushort(0x7F));
    }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e4m3_t>(row, index); }
    static float squares(uint bits) { return nk_widened_squares_<e4m3_t>(bits); }
};

/** E5M2 is the top byte of an IEEE half, infinities and NaNs included. */
struct e5m2_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static half4 widen(ushort4 codes) { return as_type<half4>(codes << ushort(8)); }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e5m2_t>(row, index); }
    static float squares(uint bits) { return nk_widened_squares_<e5m2_t>(bits); }
};

/** E3M2, OCP FP6 in the low six bits of a byte: bias 3, no infinities or NaNs. */
struct e3m2_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static half4 widen(ushort4 codes) {
        ushort4 const bits = ((codes & ushort(0x1F)) << ushort(8)) | ((codes & ushort(0x20)) << ushort(10));
        return as_type<half4>(bits) * half(4096);
    }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e3m2_t>(row, index); }
    static float squares(uint bits) { return nk_widened_squares_<e3m2_t>(bits); }
};

/** E2M3, OCP FP6 in the low six bits of a byte: bias 1, no infinities or NaNs. */
struct e2m3_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 1;
    static half4 widen(ushort4 codes) {
        ushort4 const bits = ((codes & ushort(0x1F)) << ushort(7)) | ((codes & ushort(0x20)) << ushort(10));
        return as_type<half4>(bits) * half(16384);
    }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e2m3_t>(row, index); }
    static float squares(uint bits) { return nk_widened_squares_<e2m3_t>(bits); }
};

/** E2M1, OCP FP4, two to a byte with the high nibble first: bias 1, eight magnitudes up to 6. */
struct e2m1x2_t {
    using raw_t = uchar;
    using stage_t = half;
    using dot_result_t = float;
    using norm_t = float;
    static constant constexpr uint dimensions_per_value = 2;
    static half4 widen(ushort4 codes) {
        ushort4 const bits = ((codes & ushort(0x7)) << ushort(9)) | ((codes & ushort(0x8)) << ushort(12));
        return as_type<half4>(bits) * half(16384);
    }
    static float load(device uchar const *row, uint index) { return nk_widened_load_<e2m1x2_t>(row, index); }
    static float squares(uint bits) { return nk_widened_squares_<e2m1x2_t>(bits); }
};

} // namespace nk

/** The launch record of a pack, laid out as the C launcher's record of the same name. */
struct nk_cross_pack_arguments_metal_t {
    ulong b_stride_bytes, depth_bytes, row_bytes, capability;
    uint column_count, depth, depth_padded_values, columns_begin, columns_end;
};

/** The 64-byte header every packed buffer opens with, laid out as the C header of the same name. */
struct nk_cross_packed_buffer_header_t {
    uint column_count, depth_dimensions, depth_padded_values, reserved[11];
    ulong capability;
};

static_assert(sizeof(nk_cross_pack_arguments_metal_t) == 56, "mirrors the C record in dots/simt.h");
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
        header->capability = arguments.capability;
        for (uint reserved = 0; reserved != 11; ++reserved) header->reserved[reserved] = 0;
    }
    uint const column = arguments.columns_begin + column_first;
    if (column >= arguments.columns_end) return;
    device uchar *rows = b_packed + sizeof(nk_cross_packed_buffer_header_t);
    device uchar const *source = b + column * arguments.b_stride_bytes;
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

/** Threadgroup cells per staged operand: one extra column breaks the 64-element stride that would
 *  put a slab's stores on the same banks. */
constant uint nk_cross_slab_cells_metal_k = nk_cross_slab_metal_k * (nk_cross_tile_metal_k + 1);

/** The launch record of every capability's tile, A and B strides in bytes and C's in results. */
struct nk_cross_arguments_metal_t {
    ulong a_stride, b_stride, c_stride;
    uint row_start, row_end, column_count, depth;

    /** Nonzero keeps only cells on and above the diagonal, the symmetric kernels' output. */
    uint upper_triangle;
};

static_assert(sizeof(nk_cross_arguments_metal_t) == 48, "mirrors the C record in dots/simt.h");

/** One @b [64,64] tile of C = A × Bᵀ, with B given row-major as @b [columns,depth], on the SIMT
 *  cores: each of 256 threads owns a 4 × 4 grid of outputs strided by 16, so threadgroup reads
 *  broadcast along one axis and sweep the banks along the other. */
template <typename dtype_, typename result_type_>
void nk_cross_tile_metal_(device uchar const *a, device uchar const *b, device result_type_ *c,
                          constant nk_cross_arguments_metal_t &arguments, uint2 group, uint thread_index,
                          threadgroup typename dtype_::dot_result_t *a_slab,
                          threadgroup typename dtype_::dot_result_t *b_slab) {
    using accumulator_t = typename dtype_::dot_result_t;
    constexpr uint side = nk_cross_tile_metal_k, slab_depth = nk_cross_slab_metal_k, pitch = side + 1;
    uint const first_row = arguments.row_start + group.y * side, first_column = group.x * side;
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
            a_slab[offset * pitch + line] = inside && row < arguments.row_end
                                                ? dtype_::load(a + row * arguments.a_stride, index)
                                                : accumulator_t(0);
            b_slab[offset * pitch + line] = inside && column < arguments.column_count
                                                ? dtype_::load(b + column * arguments.b_stride, index)
                                                : accumulator_t(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 0; offset != slab_depth; ++offset) {
            accumulator_t a_values[4], b_values[4];
            for (uint step = 0; step != 4; ++step)
                a_values[step] = a_slab[offset * pitch + thread_row + 16 * step],
                b_values[step] = b_slab[offset * pitch + thread_column + 16 * step];
            for (uint row_step = 0; row_step != 4; ++row_step)
                for (uint column_step = 0; column_step != 4; ++column_step)
                    sums[row_step][column_step] += a_values[row_step] * b_values[column_step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint row_step = 0; row_step != 4; ++row_step) {
        uint const row = first_row + thread_row + 16 * row_step;
        if (row >= arguments.row_end) continue;
        for (uint column_step = 0; column_step != 4; ++column_step) {
            uint const column = first_column + thread_column + 16 * column_step;
            if (column >= arguments.column_count || (arguments.upper_triangle && column < row)) continue;
            c[row * arguments.c_stride + column] = (result_type_)sums[row_step][column_step];
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
    threadgroup int a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::i8_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_u8_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                     device uint *c [[buffer(2)]],
                                     constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::u8_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_i4_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                     device int *c [[buffer(2)]],
                                     constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup int a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::i4x2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_u4_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                     device uint *c [[buffer(2)]],
                                     constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::u4x2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_f16_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                      device float *c [[buffer(2)]],
                                      constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::f16_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_bf16_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::bf16_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e4m3_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::e4m3_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e5m2_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::e5m2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e3m2_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::e3m2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e2m3_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::e2m3_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}
kernel void nk_dots_e2m1_metal_kernel_(device uchar const *a [[buffer(0)]], device uchar const *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant nk_cross_arguments_metal_t &arguments [[buffer(3)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float a_slab[nk_cross_slab_cells_metal_k], b_slab[nk_cross_slab_cells_metal_k];
    nk_cross_tile_metal_<nk::e2m1x2_t>(a, b, c, arguments, group, thread_index, a_slab, b_slab);
}

#pragma endregion Kernels
