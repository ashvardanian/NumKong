/**
 *  @file include/numkong/attention/metal.metal
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Ragged attention and rotary position embeddings on the SIMT cores of every Apple GPU.
 *
 *  @sa include/numkong/attention/metal.h, which embeds and launches this source
 *  @sa include/numkong/attention/cuda.cuh, the CUDA sibling
 *
 *  The pack keeps the serial header and directory, then each segment's K planes of every head and
 *  its V planes, rows of raw elements. An attention threadgroup takes a segment and a query head,
 *  and the rows of it the task window holds: four simdgroups share each row of a segment of up to
 *  four queries, other rows get a simdgroup each up to depth 256, and deeper heads make the serial
 *  kernel's two sweeps. The tiled @c apple9 and @c apple10 kernels build on the online softmax
 *  here, over 32-row tiles cut at multiples of 32 within each segment, so every window computes a
 *  row alike. I8 scores stay exact and their weights quantize to U8, as in the serial kernel.
 *
 *  NVFP4 planes keep each element times its block scale, exact in F16, and run the F16 path with
 *  both tensor scales in the score multiplier and the output normalization. MX planes keep each
 *  element rebased by its plane's largest finite block exponent less 31, exact in BF16, with the
 *  plane-exponent table past them, and run the BF16 path, each query row rebasing by its own. Rows
 *  on raw planes, rows past the window and rows whose multiplier is not a normal F32 take the exact
 *  path instead: a simdgroup sweeps their keys twice, adding each block's sum times its scales into
 *  a compensated pair, as the block-scaled dots tiles do.
 *
 *  The BF16 gradients recompute each weight from its row's log-sum-exp in the CUDA kernels' two
 *  passes over segments × K and V heads: a simdgroup per key sums dK and dV over the query rows of
 *  every head of its group, then a simdgroup per query row sums dQ over its keys. A threadgroup
 *  shares eight staged rows of the other side at a time and keeps its sums in registers up to
 *  depth 256, and every gradient sums in one order whatever the window. The tiled @c apple9 and
 *  @c apple10 gradients are FlashAttention-2's over 32-key and 32-row tiles, 128 gradient columns
 *  at a time, P, dS and dO entering their products in BF16.
 */

/** The launch record of a RoPE kernel, laid out as @c nk_attention_rope_arguments_metal_t. */
struct nk_attention_rope_arguments_metal_t {
    ulong rows, heads, depth, x_stride, y_stride;
};

static_assert(sizeof(nk_attention_rope_arguments_metal_t) == 40, "mirrors the C record in attention/metal.h");

/** Rotates every head of every row as @c nk_attention_rope_f32_serial does, a simdgroup each. */
template <typename element_type_>
inline void nk_attention_rope_metal_(device uchar const *x, device float const *cosine, device float const *sine,
                                     device uchar *y, constant nk_attention_rope_arguments_metal_t &a, uint group,
                                     uint groups, uint simdgroup, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    uint const half_depth = uint(a.depth / 2);
    ulong const heads = a.rows * a.heads, head_bytes = a.depth * sizeof(typename element_type_::raw_t);
    for (ulong head = ulong(group) * 8 + simdgroup; head < heads; head += ulong(groups) * 8) {
        ulong const row = head / a.heads, first = (head - row * a.heads) * head_bytes;
        device uchar const *input = x + row * a.x_stride + first;
        device uchar *output = y + row * a.y_stride + first;
        device float const *cosine_row = cosine + row * half_depth, *sine_row = sine + row * half_depth;
        for (uint pair = lane; pair < half_depth; pair += 32) {
            float const low = element_type_::load(input, pair), high = element_type_::load(input, pair + half_depth);
            float const c = cosine_row[pair], s = sine_row[pair];
            element_type_::store(output, pair, low * c - high * s);
            element_type_::store(output, pair + half_depth, low * s + high * c);
        }
    }
}

kernel void nk_attention_rope_f32_metal_kernel_(device uchar const *x [[buffer(0)]],
                                                device float const *cosine [[buffer(1)]],
                                                device float const *sine [[buffer(2)]], device uchar *y [[buffer(3)]],
                                                constant nk_attention_rope_arguments_metal_t &a [[buffer(4)]],
                                                uint group [[threadgroup_position_in_grid]],
                                                uint groups [[threadgroups_per_grid]],
                                                uint simdgroup [[simdgroup_index_in_threadgroup]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_rope_metal_<nk::f32_t>(x, cosine, sine, y, a, group, groups, simdgroup, lane);
}

kernel void nk_attention_rope_bf16_metal_kernel_(device uchar const *x [[buffer(0)]],
                                                 device float const *cosine [[buffer(1)]],
                                                 device float const *sine [[buffer(2)]], device uchar *y [[buffer(3)]],
                                                 constant nk_attention_rope_arguments_metal_t &a [[buffer(4)]],
                                                 uint group [[threadgroup_position_in_grid]],
                                                 uint groups [[threadgroups_per_grid]],
                                                 uint simdgroup [[simdgroup_index_in_threadgroup]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_rope_metal_<nk::bf16_t>(x, cosine, sine, y, a, group, groups, simdgroup, lane);
}

kernel void nk_attention_rope_e4m3_metal_kernel_(device uchar const *x [[buffer(0)]],
                                                 device float const *cosine [[buffer(1)]],
                                                 device float const *sine [[buffer(2)]], device uchar *y [[buffer(3)]],
                                                 constant nk_attention_rope_arguments_metal_t &a [[buffer(4)]],
                                                 uint group [[threadgroup_position_in_grid]],
                                                 uint groups [[threadgroups_per_grid]],
                                                 uint simdgroup [[simdgroup_index_in_threadgroup]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_rope_metal_<nk::e4m3_t>(x, cosine, sine, y, a, group, groups, simdgroup, lane);
}

/** The packed header, laid out as @c nk_attention_packed_header_t. */
struct nk_attention_packed_header_metal_t {
    uint key_value_head_count, depth, segments;
    float key_tensor_scale, value_tensor_scale;
    uint reserved[9];
    ulong capability;
};

/** The launch record of the pack kernels, laid out as @c nk_attention_pack_arguments_metal_t. */
struct nk_attention_pack_arguments_metal_t {
    ulong key_value_head_count, depth, head_bytes, segments, key_stride, value_stride, tasks_begin, tasks_end;
    ulong key_bytes, value_bytes, packed_bytes, lengths_bytes, capability;
    ulong code_bytes, key_scales_stride, value_scales_stride, key_scales_bytes, value_scales_bytes;
    uint key_tensor_present, value_tensor_present;
};

static_assert(sizeof(nk_attention_pack_arguments_metal_t) == 152, "mirrors the C record in attention/metal.h");

/** The launch record of the attention kernels, laid out as @c nk_attention_arguments_metal_t. */
struct nk_attention_arguments_metal_t {
    ulong heads, kv_heads, depth, query_tokens, query_stride, output_stride, keys_before, keys_after;
    ulong tasks_begin, tasks_end, packed_bytes, offsets_bytes, log_sum_exp_bytes, capability;
    float scale2;
    uint query_tensor_present;
    ulong query_scales_stride;
};

static_assert(sizeof(nk_attention_arguments_metal_t) == 128, "mirrors the C record in attention/metal.h");

/** The launch record of the backward kernels, laid out as the C record of the same name: the
 *  attention record, its window over segments × K and V heads, then the gradients' own fields. */
struct nk_attention_backward_arguments_metal_t {
    nk_attention_arguments_metal_t attention;
    ulong query_gradient_stride, key_value_gradient_stride, key_value_gradient_bytes;
    float scale;
};

static_assert(sizeof(nk_attention_backward_arguments_metal_t) == 160, "mirrors the C record in attention/metal.h");

/** The widest span of block exponents an MX plane or query row rebases across, a raw plane's
 *  entry, and the binades a rebase leaves above the largest block, all as `serial.h` sets them. */
constant int nk_attention_plane_window_metal_k = 89, nk_attention_raw_plane_metal_k = -128,
             nk_attention_headroom_metal_k = 31;

/** Bytes from the start of a pack of @p segments segments to its payload: the header and the
 *  directory, padded as @c nk_attention_pack_directory_size_serial_ pads it. */
inline ulong nk_attention_payload_offset_metal_(ulong segments) {
    return 64 + nk::round_up_to_multiple(segments * 8 + (2 * segments + 1) * 4, 64ul);
}

/** Keys of @p segment at pack time: its length, or its slot when the pack got no lengths. */
inline uint nk_attention_key_count_metal_(device uint const *key_offsets, device uint const *key_lengths,
                                          constant nk_attention_pack_arguments_metal_t &a, ulong segment) {
    return a.lengths_bytes ? key_lengths[segment] : key_offsets[segment + 1] - key_offsets[segment];
}

/** Writes the header, recording the packing capability, and the directory, one simdgroup walking
 *  the segments 32 at a time, and returns the keys of every segment. */
inline ulong nk_attention_directory_metal_(device uint const *key_offsets, device uint const *key_lengths,
                                           device uchar *packed, constant nk_attention_pack_arguments_metal_t &a,
                                           uint lane) {
    ulong const payload_offset = nk_attention_payload_offset_metal_(a.segments);
    if (payload_offset > a.packed_bytes) return 0;
    for (ulong byte = lane; byte < payload_offset; byte += 32) packed[byte] = 0;
    simdgroup_barrier(mem_flags::mem_device);
    device nk_attention_packed_header_metal_t *header = (device nk_attention_packed_header_metal_t *)packed;
    if (!lane) {
        header->key_value_head_count = uint(a.key_value_head_count), header->depth = uint(a.depth);
        header->segments = uint(a.segments), header->capability = a.capability;
    }
    device ulong *payload_offsets = (device ulong *)(packed + 64);
    device uint *offsets_copy = (device uint *)(payload_offsets + a.segments);
    device uint *lengths_copy = offsets_copy + a.segments + 1;
    ulong const unit_bytes = 2 * a.key_value_head_count * a.head_bytes;
    uint running = 0;
    for (ulong chunk = 0; chunk < a.segments; chunk += 32) {
        ulong const segment = chunk + lane;
        uint const length = segment < a.segments ? nk_attention_key_count_metal_(key_offsets, key_lengths, a, segment)
                                                 : 0;
        uint const before = running + simd_prefix_exclusive_sum(length);
        if (segment < a.segments) {
            payload_offsets[segment] = ulong(before) * unit_bytes;
            offsets_copy[segment] = key_offsets[segment], lengths_copy[segment] = length;
        }
        running += simd_sum(length);
    }
    if (!lane) offsets_copy[a.segments] = key_offsets[a.segments];
    return running;
}

kernel void nk_attention_directory_metal_kernel_(device uint const *key_offsets [[buffer(2)]],
                                                 device uint const *key_lengths [[buffer(3)]],
                                                 device uchar *packed [[buffer(4)]],
                                                 constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_directory_metal_(key_offsets, key_lengths, packed, a, lane);
}

/** Keys of the segments from @p first up to @p end at pack time, which the threadgroup's 256
 *  threads sum together, each in @p partials once per simdgroup. */
inline ulong nk_attention_keys_between_metal_(device uint const *key_offsets, device uint const *key_lengths,
                                              constant nk_attention_pack_arguments_metal_t &a, ulong first, ulong end,
                                              uint thread_index, uint simdgroup, threadgroup uint *partials) {
    uint passed = 0;
    for (ulong other = first + thread_index; other < end; other += 256)
        passed += nk_attention_key_count_metal_(key_offsets, key_lengths, a, other);
    passed = simd_sum(passed);
    if (simd_is_first()) partials[simdgroup] = passed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong keys = 0;
    for (uint other = 0; other < 8; ++other) keys += partials[other];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return keys;
}

/** Copies the K and V planes of every task of the window over segments × K and V heads, a
 *  threadgroup per task, at offsets the key counts before it give, so windows run in any order. */
kernel void nk_attention_pack_metal_kernel_(
    device uchar const *keys [[buffer(0)]], device uchar const *values [[buffer(1)]],
    device uint const *key_offsets [[buffer(2)]], device uint const *key_lengths [[buffer(3)]],
    device uchar *packed [[buffer(4)]], constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partials[8];
    ulong const payload_offset = nk_attention_payload_offset_metal_(a.segments);
    ulong const row_bytes = a.key_value_head_count * a.head_bytes;
    ulong segment_first = 0, positions_before = 0;
    for (ulong task = a.tasks_begin + group; task < a.tasks_end; task += groups) {
        ulong const segment = task / a.key_value_head_count, head = task % a.key_value_head_count;
        positions_before += nk_attention_keys_between_metal_(key_offsets, key_lengths, a, segment_first, segment,
                                                             thread_index, simdgroup, partials);
        segment_first = segment;

        ulong const length = nk_attention_key_count_metal_(key_offsets, key_lengths, a, segment);
        ulong const start = key_offsets[segment], plane_bytes = length * a.head_bytes;
        ulong const keys_offset = payload_offset + 2 * positions_before * row_bytes + head * plane_bytes;
        if (!length || keys_offset > a.packed_bytes ||
            (a.key_value_head_count + 1) * plane_bytes > a.packed_bytes - keys_offset)
            continue;
        if (row_bytes > a.key_bytes || row_bytes > a.value_bytes ||
            (a.key_stride && start + length - 1 > (a.key_bytes - row_bytes) / a.key_stride) ||
            (a.value_stride && start + length - 1 > (a.value_bytes - row_bytes) / a.value_stride))
            continue;
        device uchar *keys_plane = packed + keys_offset;
        device uchar *values_plane = keys_plane + a.key_value_head_count * plane_bytes;
        for (ulong byte = thread_index; byte < plane_bytes; byte += 256) {
            ulong const position = byte / a.head_bytes, element = byte - position * a.head_bytes;
            keys_plane[byte] = keys[(start + position) * a.key_stride + head * a.head_bytes + element];
            values_plane[byte] = values[(start + position) * a.value_stride + head * a.head_bytes + element];
        }
    }
}

/** The plane-exponent entry of UE8M0 codes whose finite ones span @p low to @p high: their largest
 *  exponent, 0 when none is finite, or the raw entry past the window, as `serial.h` writes it. */
inline int nk_attention_plane_entry_metal_(uint low, uint high) {
    if (low > high) return 0;
    return int(high - low) > nk_attention_plane_window_metal_k ? nk_attention_raw_plane_metal_k : int(high) - 127;
}

/** The code of a block scale of one: 0x38 in UE4M3 and 127 in UE8M0. */
template <nk_cross_scale_metal_t scale_>
constexpr uchar nk_attention_unit_scale_metal_() {
    return scale_ == nk_cross_scale_e4m3_metal_k ? 0x38 : 127;
}

/** Element @p index of a row of @p element_type_ codes times its block scale, one of @p scales per
 *  @p block_size_ codes, over 2 to the power of @p base, as @c nk_attention_load_mxfp4_serial_
 *  decodes it, exactly while the result stays a normal F32. */
template <typename element_type_, uint block_size_, nk_cross_scale_metal_t scale_>
inline float nk_attention_load_scaled_metal_(device uchar const *codes, device uchar const *scales, uint index,
                                             int base) {
    int exponent;
    float const scale = nk_cross_block_scale_metal_<scale_>(scales[index / block_size_],
                                                            nk_attention_unit_scale_metal_<scale_>(), exponent);
    return ldexp(float(element_type_::load(codes, index)) * scale, exponent - base);
}

/** Packs @p positions rows of one head's codes and scales, @p codes_stride and @p scales_stride
 *  bytes apart, into @p plane and returns its plane-exponent entry, the threadgroup together, as
 *  @c nk_attention_pack_plane_mxfp4_serial_ does: exact F16 products for NVFP4, BF16 elements under
 *  the plane's base for MX, or each row's raw codes and scale codes past the window. */
template <typename element_type_, uint block_size_, nk_cross_scale_metal_t scale_>
inline int nk_attention_pack_plane_metal_(device uchar const *codes, ulong codes_stride, device uchar const *scales,
                                          ulong scales_stride, ulong positions,
                                          constant nk_attention_pack_arguments_metal_t &a, device uchar *plane,
                                          uint thread_index, uint simdgroup, threadgroup uint *partials) {
    using plane_t = conditional_t<scale_ == nk_cross_scale_e4m3_metal_k, nk::f16_t, nk::bf16_t>;
    ulong const blocks = a.depth / block_size_;
    int entry = 0;
    if (scale_ == nk_cross_scale_e8m0_metal_k) {
        uint low = 255, high = 0;
        for (ulong cell = thread_index; cell < positions * blocks; cell += 256) {
            uint const code = scales[cell / blocks * scales_stride + cell % blocks];
            if (code != 0 && code != 255) low = min(low, code), high = max(high, code);
        }
        low = simd_min(low), high = simd_max(high);
        if (simd_is_first()) partials[simdgroup] = low, partials[8 + simdgroup] = high;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint other = 0; other < 8; ++other) low = min(low, partials[other]), high = max(high, partials[8 + other]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        entry = nk_attention_plane_entry_metal_(low, high);
    }
    if (entry == nk_attention_raw_plane_metal_k) {
        for (ulong byte = thread_index; byte < positions * a.head_bytes; byte += 256) {
            ulong const position = byte / a.head_bytes, offset = byte - position * a.head_bytes;
            plane[byte] = offset < a.code_bytes            ? codes[position * codes_stride + offset]
                          : offset < a.code_bytes + blocks ? scales[position * scales_stride + offset - a.code_bytes]
                                                           : 0;
        }
        return entry;
    }
    int const base = scale_ == nk_cross_scale_e8m0_metal_k ? entry - nk_attention_headroom_metal_k : 0;
    for (ulong cell = thread_index; cell < positions * a.depth; cell += 256) {
        ulong const position = cell / a.depth;
        uint const channel = uint(cell - position * a.depth);
        plane_t::store(plane + position * a.head_bytes, channel,
                       nk_attention_load_scaled_metal_<element_type_, block_size_, scale_>(
                           codes + position * codes_stride, scales + position * scales_stride, channel, base));
    }
    return entry;
}

/** @c nk_attention_directory_metal_ for block-scaled planes: NVFP4 packs keep both tensor scales
 *  in the header, and MX packs zero the plane-exponent table's padding. */
template <nk_cross_scale_metal_t scale_>
inline void nk_attention_directory_scaled_metal_(device uint const *key_offsets, device uint const *key_lengths,
                                                 device uchar *packed, constant nk_attention_pack_arguments_metal_t &a,
                                                 device float const *key_tensor_scale,
                                                 device float const *value_tensor_scale, uint lane) {
    ulong const keys = nk_attention_directory_metal_(key_offsets, key_lengths, packed, a, lane);
    ulong const payload_offset = nk_attention_payload_offset_metal_(a.segments);
    if (payload_offset > a.packed_bytes) return;
    if (scale_ == nk_cross_scale_e4m3_metal_k) {
        device nk_attention_packed_header_metal_t *header = (device nk_attention_packed_header_metal_t *)packed;
        if (!lane) {
            header->key_tensor_scale = a.key_tensor_present ? *key_tensor_scale : 1;
            header->value_tensor_scale = a.value_tensor_present ? *value_tensor_scale : 1;
        }
        return;
    }
    ulong const plane_exponents = payload_offset + 2 * keys * a.key_value_head_count * a.head_bytes;
    ulong const entries = 2 * a.segments * a.key_value_head_count;
    for (ulong entry = entries + lane; entry < nk::round_up_to_multiple(entries, 64ul); entry += 32)
        if (plane_exponents + entry < a.packed_bytes) packed[plane_exponents + entry] = 0;
}

/** Packs the K and V planes of every task of the window from block-scaled codes, a threadgroup per
 *  task, where @c nk_attention_pack_metal_kernel_ places plain ones, each through
 *  @c nk_attention_pack_plane_metal_, and MX planes record their entries in the plane-exponent
 *  table past every segment's planes, each segment's K entries first. */
template <typename element_type_, uint block_size_, nk_cross_scale_metal_t scale_>
inline void nk_attention_pack_scaled_metal_(device uchar const *keys, device uchar const *values,
                                            device uint const *key_offsets, device uint const *key_lengths,
                                            device uchar *packed, constant nk_attention_pack_arguments_metal_t &a,
                                            device uchar const *key_scales, device uchar const *value_scales,
                                            uint group, uint groups, uint thread_index, uint simdgroup,
                                            threadgroup uint *partials) {
    constexpr bool rebased = scale_ == nk_cross_scale_e8m0_metal_k;
    ulong const payload_offset = nk_attention_payload_offset_metal_(a.segments);
    ulong const row_bytes = a.key_value_head_count * a.head_bytes, blocks = a.depth / block_size_;
    ulong const codes_row_bytes = a.key_value_head_count * a.code_bytes,
                scales_row_bytes = a.key_value_head_count * blocks;
    ulong const plane_exponents = rebased ? payload_offset + 2 * row_bytes *
                                                                 nk_attention_keys_between_metal_(
                                                                     key_offsets, key_lengths, a, 0, a.segments,
                                                                     thread_index, simdgroup, partials)
                                          : 0;
    ulong segment_first = 0, positions_before = 0;
    for (ulong task = a.tasks_begin + group; task < a.tasks_end; task += groups) {
        ulong const segment = task / a.key_value_head_count, head = task % a.key_value_head_count;
        positions_before += nk_attention_keys_between_metal_(key_offsets, key_lengths, a, segment_first, segment,
                                                             thread_index, simdgroup, partials);
        segment_first = segment;

        ulong const length = nk_attention_key_count_metal_(key_offsets, key_lengths, a, segment);
        ulong const start = key_offsets[segment], plane_bytes = length * a.head_bytes, last = start + length - 1;
        ulong const keys_offset = payload_offset + 2 * positions_before * row_bytes + head * plane_bytes;
        if (keys_offset > a.packed_bytes || (a.key_value_head_count + 1) * plane_bytes > a.packed_bytes - keys_offset)
            continue;
        if (length &&
            (codes_row_bytes > a.key_bytes || codes_row_bytes > a.value_bytes ||
             scales_row_bytes > a.key_scales_bytes || scales_row_bytes > a.value_scales_bytes ||
             (a.key_stride && last > (a.key_bytes - codes_row_bytes) / a.key_stride) ||
             (a.value_stride && last > (a.value_bytes - codes_row_bytes) / a.value_stride) ||
             (a.key_scales_stride && last > (a.key_scales_bytes - scales_row_bytes) / a.key_scales_stride) ||
             (a.value_scales_stride && last > (a.value_scales_bytes - scales_row_bytes) / a.value_scales_stride)))
            continue;
        device uchar *keys_plane = packed + keys_offset;
        int const key_exponent = nk_attention_pack_plane_metal_<element_type_, block_size_, scale_>(
            keys + start * a.key_stride + head * a.code_bytes, a.key_stride,
            key_scales + start * a.key_scales_stride + head * blocks, a.key_scales_stride, length, a, keys_plane,
            thread_index, simdgroup, partials);
        int const value_exponent = nk_attention_pack_plane_metal_<element_type_, block_size_, scale_>(
            values + start * a.value_stride + head * a.code_bytes, a.value_stride,
            value_scales + start * a.value_scales_stride + head * blocks, a.value_scales_stride, length, a,
            keys_plane + a.key_value_head_count * plane_bytes, thread_index, simdgroup, partials);
        ulong const entry = plane_exponents + 2 * segment * a.key_value_head_count + head;
        if (rebased && !thread_index && entry + a.key_value_head_count < a.packed_bytes)
            packed[entry] = uchar(key_exponent), packed[entry + a.key_value_head_count] = uchar(value_exponent);
    }
}

kernel void nk_attention_directory_nvfp4_metal_kernel_(device uint const *key_offsets [[buffer(2)]],
                                                       device uint const *key_lengths [[buffer(3)]],
                                                       device uchar *packed [[buffer(4)]],
                                                       constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
                                                       device float const *key_tensor_scale [[buffer(8)]],
                                                       device float const *value_tensor_scale [[buffer(9)]],
                                                       uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_directory_scaled_metal_<nk_cross_scale_e4m3_metal_k>(key_offsets, key_lengths, packed, a,
                                                                      key_tensor_scale, value_tensor_scale, lane);
}

kernel void nk_attention_directory_mxfp4_metal_kernel_(device uint const *key_offsets [[buffer(2)]],
                                                       device uint const *key_lengths [[buffer(3)]],
                                                       device uchar *packed [[buffer(4)]],
                                                       constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
                                                       uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_directory_scaled_metal_<nk_cross_scale_e8m0_metal_k>(key_offsets, key_lengths, packed, a, nullptr,
                                                                      nullptr, lane);
}

kernel void nk_attention_directory_mxfp6e2m3_metal_kernel_(device uint const *key_offsets [[buffer(2)]],
                                                           device uint const *key_lengths [[buffer(3)]],
                                                           device uchar *packed [[buffer(4)]],
                                                           constant nk_attention_pack_arguments_metal_t &a
                                                           [[buffer(5)]],
                                                           uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_directory_scaled_metal_<nk_cross_scale_e8m0_metal_k>(key_offsets, key_lengths, packed, a, nullptr,
                                                                      nullptr, lane);
}

kernel void nk_attention_directory_mxfp6e3m2_metal_kernel_(device uint const *key_offsets [[buffer(2)]],
                                                           device uint const *key_lengths [[buffer(3)]],
                                                           device uchar *packed [[buffer(4)]],
                                                           constant nk_attention_pack_arguments_metal_t &a
                                                           [[buffer(5)]],
                                                           uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_directory_scaled_metal_<nk_cross_scale_e8m0_metal_k>(key_offsets, key_lengths, packed, a, nullptr,
                                                                      nullptr, lane);
}

kernel void nk_attention_directory_mxfp8e4m3_metal_kernel_(device uint const *key_offsets [[buffer(2)]],
                                                           device uint const *key_lengths [[buffer(3)]],
                                                           device uchar *packed [[buffer(4)]],
                                                           constant nk_attention_pack_arguments_metal_t &a
                                                           [[buffer(5)]],
                                                           uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_directory_scaled_metal_<nk_cross_scale_e8m0_metal_k>(key_offsets, key_lengths, packed, a, nullptr,
                                                                      nullptr, lane);
}

kernel void nk_attention_directory_mxfp8e5m2_metal_kernel_(device uint const *key_offsets [[buffer(2)]],
                                                           device uint const *key_lengths [[buffer(3)]],
                                                           device uchar *packed [[buffer(4)]],
                                                           constant nk_attention_pack_arguments_metal_t &a
                                                           [[buffer(5)]],
                                                           uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_directory_scaled_metal_<nk_cross_scale_e8m0_metal_k>(key_offsets, key_lengths, packed, a, nullptr,
                                                                      nullptr, lane);
}

kernel void nk_attention_pack_nvfp4_metal_kernel_(
    device uchar const *keys [[buffer(0)]], device uchar const *values [[buffer(1)]],
    device uint const *key_offsets [[buffer(2)]], device uint const *key_lengths [[buffer(3)]],
    device uchar *packed [[buffer(4)]], constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
    device uchar const *key_scales [[buffer(6)]], device uchar const *value_scales [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partials[16];
    nk_attention_pack_scaled_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k>(
        keys, values, key_offsets, key_lengths, packed, a, key_scales, value_scales, group, groups, thread_index,
        simdgroup, partials);
}

kernel void nk_attention_pack_mxfp4_metal_kernel_(
    device uchar const *keys [[buffer(0)]], device uchar const *values [[buffer(1)]],
    device uint const *key_offsets [[buffer(2)]], device uint const *key_lengths [[buffer(3)]],
    device uchar *packed [[buffer(4)]], constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
    device uchar const *key_scales [[buffer(6)]], device uchar const *value_scales [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partials[16];
    nk_attention_pack_scaled_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k>(
        keys, values, key_offsets, key_lengths, packed, a, key_scales, value_scales, group, groups, thread_index,
        simdgroup, partials);
}

kernel void nk_attention_pack_mxfp6e2m3_metal_kernel_(
    device uchar const *keys [[buffer(0)]], device uchar const *values [[buffer(1)]],
    device uint const *key_offsets [[buffer(2)]], device uint const *key_lengths [[buffer(3)]],
    device uchar *packed [[buffer(4)]], constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
    device uchar const *key_scales [[buffer(6)]], device uchar const *value_scales [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partials[16];
    nk_attention_pack_scaled_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k>(
        keys, values, key_offsets, key_lengths, packed, a, key_scales, value_scales, group, groups, thread_index,
        simdgroup, partials);
}

kernel void nk_attention_pack_mxfp6e3m2_metal_kernel_(
    device uchar const *keys [[buffer(0)]], device uchar const *values [[buffer(1)]],
    device uint const *key_offsets [[buffer(2)]], device uint const *key_lengths [[buffer(3)]],
    device uchar *packed [[buffer(4)]], constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
    device uchar const *key_scales [[buffer(6)]], device uchar const *value_scales [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partials[16];
    nk_attention_pack_scaled_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k>(
        keys, values, key_offsets, key_lengths, packed, a, key_scales, value_scales, group, groups, thread_index,
        simdgroup, partials);
}

kernel void nk_attention_pack_mxfp8e4m3_metal_kernel_(
    device uchar const *keys [[buffer(0)]], device uchar const *values [[buffer(1)]],
    device uint const *key_offsets [[buffer(2)]], device uint const *key_lengths [[buffer(3)]],
    device uchar *packed [[buffer(4)]], constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
    device uchar const *key_scales [[buffer(6)]], device uchar const *value_scales [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partials[16];
    nk_attention_pack_scaled_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k>(
        keys, values, key_offsets, key_lengths, packed, a, key_scales, value_scales, group, groups, thread_index,
        simdgroup, partials);
}

kernel void nk_attention_pack_mxfp8e5m2_metal_kernel_(
    device uchar const *keys [[buffer(0)]], device uchar const *values [[buffer(1)]],
    device uint const *key_offsets [[buffer(2)]], device uint const *key_lengths [[buffer(3)]],
    device uchar *packed [[buffer(4)]], constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
    device uchar const *key_scales [[buffer(6)]], device uchar const *value_scales [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partials[16];
    nk_attention_pack_scaled_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k>(
        keys, values, key_offsets, key_lengths, packed, a, key_scales, value_scales, group, groups, thread_index,
        simdgroup, partials);
}

/** Whether @p element_type_ quantizes its softmax weights to U8, as the serial I8 kernel does. */
template <typename element_type_>
constexpr bool nk_attention_quantized_metal_() {
    return is_same<element_type_, nk::i8_t>::value;
}

/** Softmax weight of a probability of one: 255 for U8 weights, one for floats. */
template <typename element_type_>
constexpr float nk_attention_unit_metal_() {
    return nk_attention_quantized_metal_<element_type_>() ? 255.0f : 1.0f;
}

/** The natural log-sum-exp of a row whose base-2 scores peak at @p maximum, given the @p sum of its
 *  weights relative to it, each @p unit for a probability of one, or −∞ for a row without keys. */
inline float nk_attention_log_sum_exp_metal_(float maximum, float sum, float unit) {
    return sum > 0 ? (maximum + log2(sum / unit)) * M_LN2_F : -INFINITY;
}

/** One query row of a plain launch: read as the planes store their elements, under the launch's
 *  multiplier, normalized by the weight sum alone and never on the exact path. */
template <typename element_type_>
struct nk_attention_query_metal {
    device uchar const *codes;
    float multiplier;
    int base;
    static constant constexpr bool exact = false;
    auto load(uint channel) const { return element_type_::load(codes, channel); }
    float normalization(float sum) const { return sum > 0 ? precise::divide(1.0f, sum) : 0; }
    float output(float value) const { return value; }
};

/** The queries of a block-scaled launch, or one row of them: rows of @p element_type_ codes under
 *  one block scale per @p block_size_ codes, the base a row's elements rebase by, its score
 *  multiplier, the factor and power of two its output takes with the normalization, its K and V
 *  plane entries, whether it takes the exact path, and where the plane-exponent table lies. */
template <typename element_type_, uint block_size_, nk_cross_scale_metal_t scale_>
struct nk_attention_scaled_query_metal {
    device uchar const *codes, *scales, *plane_exponents;
    ulong plane_exponents_count;
    float multiplier, factor;
    int base, exponent, key_exponent, value_exponent;
    bool exact;
    float load(uint channel) const {
        return nk_attention_load_scaled_metal_<element_type_, block_size_, scale_>(codes, scales, channel, base);
    }
    float normalization(float sum) const { return sum > 0 ? precise::divide(factor, sum) : 0; }
    float output(float value) const { return ldexp(value, exponent); }
};

/** Query row @p row of head @p head of a plain launch. */
template <typename element_type_>
inline nk_attention_query_metal<element_type_> nk_attention_query_row_metal_(device uchar const *queries,
                                                                             constant nk_attention_arguments_metal_t &a,
                                                                             ulong row, ulong head) {
    nk_attention_query_metal<element_type_> query;
    query.codes = queries + row * a.query_stride + head * a.depth * sizeof(typename element_type_::raw_t);
    query.multiplier = a.scale2, query.base = 0;
    return query;
}

/** Query row @p row of head @p head of a block-scaled launch: its codes and block scales, under the
 *  launch's multiplier and factor until @c nk_attention_rebase_metal_ fills its own. */
template <typename element_type_, typename code_type_, uint block_size_, nk_cross_scale_metal_t scale_>
inline nk_attention_scaled_query_metal<code_type_, block_size_, scale_> nk_attention_query_row_metal_(
    nk_attention_scaled_query_metal<code_type_, block_size_, scale_> query, constant nk_attention_arguments_metal_t &a,
    ulong row, ulong head) {
    query.codes += row * a.query_stride + head * a.depth / code_type_::dimensions_per_value;
    query.scales += row * a.query_scales_stride + head * (a.depth / block_size_);
    return query;
}

/** The dot product of @p query and @p key, shared by the 32 lanes of a simdgroup and returned in
 *  each: exact in I32 for integer codes, like the serial kernel's, and in F32 for floats. */
template <typename element_type_, typename query_type_>
inline float nk_attention_score_metal_(thread query_type_ const &query, device uchar const *key, uint depth,
                                       uint lane) {
    typename element_type_::dot_result_t sum = 0;
    for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
        sum += query.load(channel) * element_type_::load(key, channel);
    return float(simd_sum(sum));
}

/** One threadgroup's share of a task window: the query rows of one segment and head that the window
 *  holds, and the segment, K and V head and planes they read. */
struct nk_attention_work_metal_t {
    ulong query_first, queries, length, row_begin, row_end, segment, key_value_head;
    long first_position;
    device uchar const *keys, *values;
};

/** The pack's segment count, or 0 when its header disagrees with the launch or its directory runs
 *  past the buffers. */
inline ulong nk_attention_segments_metal_(device uchar const *packed, constant nk_attention_arguments_metal_t &a) {
    device nk_attention_packed_header_metal_t const *header = (device nk_attention_packed_header_metal_t const *)packed;
    if (header->capability != a.capability || header->depth != a.depth || header->key_value_head_count != a.kv_heads)
        return 0;
    ulong const segments = header->segments;
    if (nk_attention_payload_offset_metal_(segments) > a.packed_bytes || (segments + 1) * 4 > a.offsets_bytes) return 0;
    return segments;
}

/** Fills the keys, first position and planes of @p work for @p key_value_head of @p segment, whose
 *  queries it already holds, returning false when the planes run past the pack. */
template <typename element_type_>
inline bool nk_attention_planes_metal_(device uchar const *packed, constant nk_attention_arguments_metal_t &a,
                                       ulong segments, ulong segment, ulong key_value_head,
                                       thread nk_attention_work_metal_t &work) {
    ulong const payload_offset = nk_attention_payload_offset_metal_(segments);
    device ulong const *payload_offsets = (device ulong const *)(packed + 64);
    device uint const *lengths = (device uint const *)(payload_offsets + segments) + segments + 1;
    ulong const length = lengths[segment], row_bytes = a.depth * sizeof(typename element_type_::raw_t);
    ulong const payload_bytes = a.packed_bytes - payload_offset;
    if (length > payload_bytes / row_bytes / a.kv_heads / 2) return false;
    ulong const plane_bytes = length * row_bytes;
    if (payload_offsets[segment] > payload_bytes ||
        2 * a.kv_heads * plane_bytes > payload_bytes - payload_offsets[segment])
        return false;
    work.length = length, work.first_position = long(length) - long(work.queries);
    work.segment = segment, work.key_value_head = key_value_head;
    work.keys = packed + payload_offset + payload_offsets[segment] + key_value_head * plane_bytes;
    work.values = work.keys + a.kv_heads * plane_bytes;
    return true;
}

/** Fills @p work for @p pair, segment × heads + head, returning false when the window holds none of
 *  its rows, its queries run past the rows the call addresses, or its planes run past the pack. */
template <typename element_type_>
inline bool nk_attention_work_metal_(device uchar const *packed, device uint const *query_offsets,
                                     constant nk_attention_arguments_metal_t &a, ulong segments, ulong pair,
                                     thread nk_attention_work_metal_t &work) {
    ulong const segment = pair / a.heads, head = pair % a.heads;
    ulong const tasks_begin = max(a.tasks_begin, ulong(query_offsets[0]) * a.heads);
    ulong const tasks_end = min(a.tasks_end, ulong(query_offsets[segments]) * a.heads);
    ulong const query_first = query_offsets[segment], query_end = query_offsets[segment + 1];
    if (tasks_begin >= tasks_end || query_first > query_end || query_end > a.query_tokens) return false;
    ulong const token_first = (tasks_begin + a.heads - 1 - head) / a.heads;
    ulong const token_end = min((tasks_end + a.heads - 1 - head) / a.heads, query_end);
    work.row_begin = token_first > query_first ? token_first - query_first : 0;
    work.row_end = token_end > query_first ? token_end - query_first : 0;
    work.query_first = query_first, work.queries = query_end - query_first;
    return work.row_begin < work.row_end &&
           nk_attention_planes_metal_<element_type_>(packed, a, segments, segment, head / (a.heads / a.kv_heads), work);
}

/** Stores the log-sum-exp of query token @p row and @p head, when the launch takes them. */
inline void nk_attention_store_log_sum_exp_metal_(device float *log_sum_exp, constant nk_attention_arguments_metal_t &a,
                                                  ulong row, ulong head, float value) {
    ulong const index = row * a.heads + head;
    if (index < a.log_sum_exp_bytes / 4) log_sum_exp[index] = value;
}

/** Writes the keys of a segment of @p length that the band shows to the row at signed @p position,
 *  as @c nk_diagonal_band_row_range_ does. */
inline void nk_attention_row_range_metal_(constant nk_attention_arguments_metal_t &a, long position, ulong length,
                                          thread ulong &begin, thread ulong &end) {
    if (position < 0) {
        ulong const lag = ulong(-position);
        begin = 0;
        end = a.keys_after < lag ? 0 : min(a.keys_after - lag + 1, length);
        return;
    }
    ulong const lead = ulong(position);
    begin = min(lead > a.keys_before ? lead - a.keys_before : 0, length);
    end = lead < length && a.keys_after < length - lead - 1 ? lead + a.keys_after + 1 : length;
}

/** Writes the keys the rows at @p first to @p last see together, as both ends of a row's keys only
 *  grow with the row: from the first row seeing any to the end of the last row's. */
inline void nk_attention_tile_keys_metal_(constant nk_attention_arguments_metal_t &a, long first, long last,
                                          ulong length, thread ulong &begin, thread ulong &end) {
    long const seeing = first < 0 && a.keys_after < ulong(-first) ? -long(a.keys_after) : first;
    ulong unused;
    nk_attention_row_range_metal_(a, seeing, length, begin, unused);
    nk_attention_row_range_metal_(a, last, length, unused, end);
}

/** The score of a key against a query a simdgroup holds in registers, 8 channels per lane. */
template <typename element_type_>
inline float nk_attention_cached_score_metal_(thread float const *query, device uchar const *keys, uint depth,
                                              uint lane) {
#pragma clang fp contract(off) reassociate(off)
    float sum = 0;
#pragma unroll
    for (uint index = 0; index < 8; ++index) {
        uint const channel = lane + index * 32;
        if (channel < depth) sum = fma(query[index], float(element_type_::load(keys, channel)), sum);
    }
    return simd_sum(sum);
}

/** The queries of a block-scaled launch over @p packed: NVFP4 ones fold the query and key tensor
 *  scales into the multiplier and keep the value one as the factor, and MX ones find the
 *  plane-exponent table past the planes, left empty for a pack the launch does not match. */
template <typename element_type_, uint block_size_, nk_cross_scale_metal_t scale_>
inline nk_attention_scaled_query_metal<element_type_, block_size_, scale_> nk_attention_scaled_queries_metal_(
    device uchar const *codes, device uchar const *scales, device float const *tensor_scale, device uchar const *packed,
    constant nk_attention_arguments_metal_t &a) {
    nk_attention_scaled_query_metal<element_type_, block_size_, scale_> queries;
    queries.codes = codes, queries.scales = scales, queries.plane_exponents = packed;
    queries.plane_exponents_count = 0, queries.multiplier = a.scale2, queries.factor = 1;
    queries.base = 0, queries.exponent = 0, queries.key_exponent = 0, queries.value_exponent = 0;
    queries.exact = false;
    ulong const segments = nk_attention_segments_metal_(packed, a);
    if (!segments) return queries;
    if (scale_ == nk_cross_scale_e4m3_metal_k) {
        device nk_attention_packed_header_metal_t const *header =
            (device nk_attention_packed_header_metal_t const *)packed;
        int query_exponent, key_exponent;
        float const query_mantissa = nk_cross_float_mantissa_metal_(a.query_tensor_present ? *tensor_scale : 1,
                                                                    query_exponent);
        float const key_mantissa = nk_cross_float_mantissa_metal_(header->key_tensor_scale, key_exponent);
        queries.multiplier = ldexp(a.scale2 * (query_mantissa * key_mantissa), query_exponent + key_exponent);
        queries.factor = header->value_tensor_scale;
        return queries;
    }
    ulong const payload_offset = nk_attention_payload_offset_metal_(segments), row_bytes = a.depth * sizeof(bfloat);
    device ulong const *payload_offsets = (device ulong const *)(packed + 64);
    device uint const *lengths = (device uint const *)(payload_offsets + segments) + segments + 1;
    ulong const payload_bytes = a.packed_bytes - payload_offset, last = payload_offsets[segments - 1];
    if (last > payload_bytes || lengths[segments - 1] > (payload_bytes - last) / row_bytes / a.kv_heads / 2)
        return queries;
    ulong const offset = payload_offset + last + 2 * a.kv_heads * lengths[segments - 1] * row_bytes;
    if (2 * a.kv_heads * segments > a.packed_bytes - offset) return queries;
    queries.plane_exponents = packed + offset, queries.plane_exponents_count = 2 * a.kv_heads * segments;
    return queries;
}

/** @p scale2 times 2 to the power of @p exponent when both are normal F32, or zero, which sends a
 *  row to the exact path. */
inline float nk_attention_rebased_multiplier_metal_(float scale2, int exponent) {
    uint const bits = as_type<uint>(scale2);
    int const biased = int((bits >> 23) & 255), rebased = biased + exponent;
    if (biased == 0 || biased == 255 || rebased < 1 || rebased > 254) return 0;
    return as_type<float>((bits & 0x807FFFFFu) | (uint(rebased) << 23));
}

/** Plain rows keep the launch's multiplier, as their planes hold their own elements. */
template <typename element_type_>
inline void nk_attention_rebase_metal_(thread nk_attention_query_metal<element_type_> &,
                                       constant nk_attention_arguments_metal_t &,
                                       thread nk_attention_work_metal_t const &) {}

/** Rebases an MX query row by its own largest finite block exponent less 31, as its planes rebase,
 *  folding both bases into its multiplier and its V plane's into its output power: past the window,
 *  on a raw plane, or under a multiplier that is not a normal F32, it takes the exact path. NVFP4
 *  rows keep what their launch set. */
template <typename element_type_, uint block_size_, nk_cross_scale_metal_t scale_>
inline void nk_attention_rebase_metal_(
    thread nk_attention_scaled_query_metal<element_type_, block_size_, scale_> &query,
    constant nk_attention_arguments_metal_t &a, thread nk_attention_work_metal_t const &work) {
    if (scale_ != nk_cross_scale_e8m0_metal_k) return;
    int const raw = nk_attention_raw_plane_metal_k;
    ulong const entry = 2 * work.segment * a.kv_heads + work.key_value_head;
    query.key_exponent = entry < query.plane_exponents_count ? int(as_type<char>(query.plane_exponents[entry])) : raw;
    query.value_exponent = entry + a.kv_heads < query.plane_exponents_count
                               ? int(as_type<char>(query.plane_exponents[entry + a.kv_heads]))
                               : raw;
    uint low = 255, high = 0;
    for (ulong block = 0; block < a.depth / block_size_; ++block) {
        uint const code = query.scales[block];
        if (code != 0 && code != 255) low = min(low, code), high = max(high, code);
    }
    int const query_exponent = nk_attention_plane_entry_metal_(low, high);
    query.base = query_exponent - nk_attention_headroom_metal_k;
    query.exponent = query.value_exponent - nk_attention_headroom_metal_k;
    query.multiplier = nk_attention_rebased_multiplier_metal_(
        a.scale2, query.base + query.key_exponent - nk_attention_headroom_metal_k);
    query.exact = query_exponent == raw || query.key_exponent == raw || query.value_exponent == raw ||
                  query.multiplier == 0;
}

/** The base-2 score of a block-scaled query row against @p key on the exact path, shared by the 32
 *  lanes of a simdgroup: each block's sum of codes times rebased BF16 elements, or raw codes, added
 *  times both scales into a compensated pair, then rounded with the multiplier. */
template <typename element_type_, uint block_size_, nk_cross_scale_metal_t scale_>
inline float nk_attention_exact_score_metal_(
    thread nk_attention_scaled_query_metal<element_type_, block_size_, scale_> const &query, device uchar const *key,
    constant nk_attention_arguments_metal_t &a, uint lane) {
    bool const raw = query.key_exponent == nk_attention_raw_plane_metal_k;
    ulong const code_bytes = a.depth / element_type_::dimensions_per_value;
    nk_cross_scaled_sum_metal_t state = {float2(0), 0};
    for (uint block = 0; block < a.depth / block_size_; ++block) {
        uint const channel = block * block_size_ + lane;
        float product = 0;
        if (lane < block_size_)
            product = float(element_type_::load(query.codes, channel)) *
                      (raw ? float(element_type_::load(key, channel)) : nk::bf16_t::load(key, channel));
        nk_cross_scaled_block_add_metal_<scale_>(
            state, simd_sum(product), query.scales[block],
            raw ? key[code_bytes + block] : nk_attention_unit_scale_metal_<scale_>());
    }
    if (!raw) state.exponent += query.key_exponent - nk_attention_headroom_metal_k;
    return nk_cross_scaled_dot_metal_(state, as_type<uint>(a.scale2));
}

/** Plain rows never take the exact path. */
template <typename element_type_>
inline void nk_attention_exact_row_metal_(thread nk_attention_query_metal<element_type_> const &,
                                          thread nk_attention_work_metal_t const &,
                                          constant nk_attention_arguments_metal_t &, ulong, ulong, device uchar *,
                                          device float *, uint) {}

/** The exact path of query row @p local of head @p head: a simdgroup sweeps its keys twice, as the
 *  serial kernel does, scoring each through @c nk_attention_exact_score_metal_, and sums the
 *  weighted values straight into the output row, decoding raw V rows at base zero. */
template <typename element_type_, uint block_size_, nk_cross_scale_metal_t scale_>
inline void nk_attention_exact_row_metal_(
    thread nk_attention_scaled_query_metal<element_type_, block_size_, scale_> const &query,
    thread nk_attention_work_metal_t const &work, constant nk_attention_arguments_metal_t &a, ulong local, ulong head,
    device uchar *output, device float *log_sum_exp, uint lane) {
#pragma clang fp contract(off) reassociate(off)
    bool const raw = query.value_exponent == nk_attention_raw_plane_metal_k;
    uint const depth = uint(a.depth);
    ulong const row = work.query_first + local, row_bytes = a.depth * sizeof(bfloat);
    ulong const code_bytes = a.depth / element_type_::dimensions_per_value;
    device float *result = (device float *)(output + row * a.output_stride) + head * a.depth;
    ulong begin, end;
    nk_attention_row_range_metal_(a, work.first_position + long(local), work.length, begin, end);
    float maximum = -INFINITY;
    for (ulong position = begin; position < end; ++position)
        maximum = max(maximum, nk_attention_exact_score_metal_(query, work.keys + position * row_bytes, a, lane));
    for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
        result[channel] = 0;
    float weights_sum = 0;
    for (ulong position = begin; position < end; ++position) {
        float const weight = nk_exp2_metal_(
            nk_attention_exact_score_metal_(query, work.keys + position * row_bytes, a, lane) - maximum);
        weights_sum += weight;
        device uchar const *value_row = work.values + position * row_bytes;
        for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32) {
            float const value = raw ? nk_attention_load_scaled_metal_<element_type_, block_size_, scale_>(
                                          value_row, value_row + code_bytes, channel, 0)
                                    : nk::bf16_t::load(value_row, channel);
            result[channel] = fma(weight, value, result[channel]);
        }
    }
    float const normalization = query.normalization(weights_sum);
    for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
        result[channel] = ldexp(result[channel] * normalization, raw ? 0 : query.exponent);
    if (!lane)
        nk_attention_store_log_sum_exp_metal_(log_sum_exp, a, row, head,
                                              nk_attention_log_sum_exp_metal_(maximum, weights_sum, 1));
}

/** Which part of a row a pass of the rows kernel computes: all of it, or one partition's maximum
 *  and weighted values for @c nk_attention_merge_metal_ to combine. */
enum nk_attention_phase_metal_t {
    nk_attention_complete_metal_k,
    nk_attention_partition_metal_k,
    nk_attention_values_metal_k,
};

/**
 *  @brief Attends the work's rows in the window over panels of 256 keys: a simdgroup per row, or
 *      four sharing each row when @p subgroup_count_ is 4, every depth up to 256.
 *
 *  The split passes write each of @p partitions key ranges of a row to its slot of @p split, the
 *  maximum, the weight sum and the weighted values. U8 weights need the whole row's maximum first,
 *  so their partitions write maxima in one pass and values in another. Rows on the exact path
 *  sweep alone in a complete pass, and the merge takes them after a split.
 */
template <typename element_type_, uint subgroup_count_,
          nk_attention_phase_metal_t phase_ = nk_attention_complete_metal_k, typename queries_type_>
inline void nk_attention_rows_metal_(queries_type_ queries, device uchar *output, device float *log_sum_exp,
                                     thread nk_attention_work_metal_t const &work,
                                     constant nk_attention_arguments_metal_t &a, ulong head, uint row_group,
                                     uint row_groups, uint warp, uint lane, threadgroup float *scratch,
                                     device float *split = nullptr, uint partition = 0, uint partitions = 1) {
    constexpr bool cooperative = subgroup_count_ > 1, quantized = nk_attention_quantized_metal_<element_type_>();
    constexpr nk_attention_phase_metal_t phase = phase_;
    uint const depth = uint(a.depth);
    ulong const row_bytes = a.depth * sizeof(typename element_type_::raw_t);
    uint const subgroup = cooperative ? warp : 0;
    scratch += cooperative ? 0 : warp * 256;
    threadgroup float *maxima = scratch + 256, *sums = maxima + 4, *partials = sums + 4;
    for (ulong local = work.row_begin + (cooperative ? row_group : row_group * 4 + warp); local < work.row_end;
         local += cooperative ? row_groups : row_groups * 4) {
        ulong const row = work.query_first + local;
        auto query = nk_attention_query_row_metal_<element_type_>(queries, a, row, head);
        nk_attention_rebase_metal_(query, a, work);
        if (query.exact) {
            if (phase == nk_attention_complete_metal_k && !subgroup)
                nk_attention_exact_row_metal_(query, work, a, local, head, output, log_sum_exp, lane);
            continue;
        }
        ulong begin, end;
        nk_attention_row_range_metal_(a, work.first_position + long(local), work.length, begin, end);
        device float *split_row = split;
        if constexpr (phase != nk_attention_complete_metal_k) {
            ulong const width = nk::divide_round_up(end - begin, ulong(partitions));
            begin = min(end, begin + partition * width);
            end = min(end, begin + width);
            split_row += (row * a.heads + head) * partitions * (a.depth + 2);
        }
        float query_values[8] = {0};
        if constexpr (!cooperative) {
#pragma unroll
            for (uint index = 0; index < 8; ++index) {
                uint const channel = lane + index * 32;
                if (channel < depth) query_values[index] = float(query.load(channel));
            }
        }
        float maximum = -INFINITY, sum = 0, values[8] = {0};
        if constexpr (quantized && phase == nk_attention_values_metal_k) {
            for (uint part = 0; part < partitions; ++part) maximum = max(maximum, split_row[part * (a.depth + 2)]);
        }
        else if constexpr (quantized) {
            for (ulong key = begin + subgroup; key < end; key += subgroup_count_)
                maximum = max(
                    maximum, nk_attention_score_metal_<element_type_>(query, work.keys + key * row_bytes, depth, lane) *
                                 query.multiplier);
            if constexpr (cooperative) {
                if (!lane) maxima[warp] = maximum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                maximum = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
            }
        }
        if constexpr (quantized && phase == nk_attention_partition_metal_k) {
            if (!warp && !lane) split_row[partition * (a.depth + 2)] = maximum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }
        for (ulong first = begin; first < end; first += 256) {
            ulong const count = min(256ul, end - first);
            float panel_maximum = maximum;
            for (ulong key = subgroup; key < count; key += subgroup_count_) {
                device uchar const *key_row = work.keys + (first + key) * row_bytes;
                float const score = (cooperative ? nk_attention_score_metal_<element_type_>(query, key_row, depth, lane)
                                                 : nk_attention_cached_score_metal_<element_type_>(
                                                       query_values, key_row, depth, lane)) *
                                    query.multiplier;
                if (!lane) scratch[key] = score;
                panel_maximum = max(panel_maximum, score);
            }
            if constexpr (cooperative) {
                if (!lane) maxima[warp] = panel_maximum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                panel_maximum = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
            }
            else simdgroup_barrier(mem_flags::mem_threadgroup);
            float const correction = maximum == panel_maximum ? 1 : nk_exp2_metal_(maximum - panel_maximum);
            maximum = panel_maximum, sum *= correction;
#pragma unroll
            for (uint index = 0; index < 8; ++index) values[index] *= correction;
            for (ulong key = subgroup; key < count; key += subgroup_count_) {
                float weight = nk_exp2_metal_(scratch[key] - maximum);
                if (quantized) weight = floor(weight * 255.0f + 0.5f);
                sum += weight;
                device uchar const *value_row = work.values + (first + key) * row_bytes;
#pragma unroll
                for (uint index = 0; index < 8; ++index) {
                    uint const channel = lane + index * 32;
                    if (channel < depth)
                        values[index] = fma(weight, float(element_type_::load(value_row, channel)), values[index]);
                }
            }
            if constexpr (cooperative) threadgroup_barrier(mem_flags::mem_threadgroup);
            else simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        constexpr float unit = nk_attention_unit_metal_<element_type_>();
        if constexpr (cooperative) {
            if (!lane) sums[warp] = sum;
            for (uint channel = lane; channel < depth; channel += 32)
                partials[warp * 256 + channel] = values[channel / 32];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!warp) {
                float const total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
                if constexpr (phase == nk_attention_complete_metal_k) {
                    float const inverse = query.normalization(total);
                    device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
                    for (uint channel = lane; channel < depth; channel += 32)
                        destination[channel] = query.output(((partials[channel] + partials[256 + channel]) +
                                                             (partials[512 + channel] + partials[768 + channel])) *
                                                            inverse);
                    if (!lane)
                        nk_attention_store_log_sum_exp_metal_(log_sum_exp, a, row, head,
                                                              nk_attention_log_sum_exp_metal_(maximum, total, unit));
                }
                else {
                    device float *destination = split_row + partition * (a.depth + 2) + 2;
                    if (!lane) {
                        if constexpr (!quantized) destination[-2] = maximum;
                        destination[-1] = total;
                    }
                    for (uint channel = lane; channel < depth; channel += 32)
                        destination[channel] = (partials[channel] + partials[256 + channel]) +
                                               (partials[512 + channel] + partials[768 + channel]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        else {
            float const inverse = query.normalization(sum);
            device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
#pragma unroll
            for (uint index = 0; index < 8; ++index) {
                uint const channel = lane + index * 32;
                if (channel < depth) destination[channel] = query.output(values[index] * inverse);
            }
            if (!lane)
                nk_attention_store_log_sum_exp_metal_(log_sum_exp, a, row, head,
                                                      nk_attention_log_sum_exp_metal_(maximum, sum, unit));
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

/** One split pass over every pair of the window, a partition of its key ranges per threadgroup
 *  along z, for rows of the decode path. */
template <typename element_type_, nk_attention_phase_metal_t phase_, typename queries_type_>
inline void nk_attention_split_metal_(queries_type_ queries, device uchar const *packed,
                                      device uint const *query_offsets, constant nk_attention_arguments_metal_t &a,
                                      device float *split, uint3 group, uint3 groups, uint warp, uint lane,
                                      threadgroup float *scratch) {
    ulong const segments = nk_attention_segments_metal_(packed, a);
    for (ulong pair = group.x; pair < segments * a.heads; pair += groups.x) {
        nk_attention_work_metal_t work;
        if (!nk_attention_work_metal_<element_type_>(packed, query_offsets, a, segments, pair, work)) continue;
        nk_attention_rows_metal_<element_type_, 4, phase_>(queries, nullptr, nullptr, work, a, pair % a.heads, group.y,
                                                           groups.y, warp, lane, scratch, split, group.z, groups.z);
    }
}

/** Combines the partitions of every row of the window that the split passes wrote, into its output
 *  row and log-sum-exp; U8 partitions already share the row's maximum, and rows on the exact path,
 *  which the split passes skip, sweep their keys here. */
template <typename element_type_, typename queries_type_>
inline void nk_attention_merge_metal_(queries_type_ queries, device uchar const *packed, device uchar *output,
                                      device uint const *query_offsets, constant nk_attention_arguments_metal_t &a,
                                      device float *log_sum_exp, device float const *split, uint2 group, uint2 groups,
                                      uint lane) {
    constexpr uint partitions = 16;
    constexpr bool quantized = nk_attention_quantized_metal_<element_type_>();
    ulong const segments = nk_attention_segments_metal_(packed, a);
    for (ulong pair = group.x; pair < segments * a.heads; pair += groups.x) {
        ulong const head = pair % a.heads;
        nk_attention_work_metal_t work;
        if (!nk_attention_work_metal_<element_type_>(packed, query_offsets, a, segments, pair, work)) continue;
        for (ulong local = work.row_begin + group.y; local < work.row_end; local += groups.y) {
            ulong const row = work.query_first + local;
            auto query = nk_attention_query_row_metal_<element_type_>(queries, a, row, head);
            nk_attention_rebase_metal_(query, a, work);
            if (query.exact) {
                nk_attention_exact_row_metal_(query, work, a, local, head, output, log_sum_exp, lane);
                continue;
            }
            device float const *parts = split + (row * a.heads + head) * partitions * (a.depth + 2);
            float const local_maximum = lane < partitions ? parts[lane * (a.depth + 2)] : -INFINITY;
            float const local_sum = lane < partitions ? parts[lane * (a.depth + 2) + 1] : 0;
            float const maximum = simd_max(local_maximum);
            float const correction = quantized || local_maximum == maximum ? 1
                                     : local_sum > 0                       ? nk_exp2_metal_(local_maximum - maximum)
                                                                           : 0;
            float const sum = simd_sum(local_sum * correction);
            float const inverse = query.normalization(sum);
            device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
            for (uint first = 0; first < a.depth; first += 32) {
                uint const channel = first + lane;
                float value = 0;
                for (uint part = 0; part < partitions; ++part) {
                    float const factor = simd_shuffle(correction, part);
                    if (channel < a.depth) value = fma(parts[part * (a.depth + 2) + 2 + channel], factor, value);
                }
                if (channel < a.depth) destination[channel] = query.output(value * inverse);
            }
            if (!lane)
                nk_attention_store_log_sum_exp_metal_(
                    log_sum_exp, a, row, head,
                    nk_attention_log_sum_exp_metal_(maximum, sum, nk_attention_unit_metal_<element_type_>()));
        }
    }
}

/** The whole @c metal attention: shared rows for decode segments, a simdgroup per row to depth
 *  256, and past it, or for U8 weights, the serial two sweeps. */
template <typename element_type_, typename queries_type_>
inline void nk_attention_fallback_metal_(queries_type_ queries, device uchar const *packed, device uchar *output,
                                         device uint const *query_offsets, constant nk_attention_arguments_metal_t &a,
                                         device float *log_sum_exp, uint2 group, uint2 groups, uint warp, uint lane,
                                         threadgroup float *scratch) {
#pragma clang fp contract(off) reassociate(off)
    constexpr bool quantized = nk_attention_quantized_metal_<element_type_>();
    ulong const segments = nk_attention_segments_metal_(packed, a);
    uint const depth = uint(a.depth);
    ulong const row_bytes = a.depth * sizeof(typename element_type_::raw_t);
    for (ulong pair = group.x; pair < segments * a.heads; pair += groups.x) {
        ulong const head = pair % a.heads;
        nk_attention_work_metal_t work;
        if (!nk_attention_work_metal_<element_type_>(packed, query_offsets, a, segments, pair, work)) continue;
        if (a.depth <= 256 && work.queries <= 4) {
            nk_attention_rows_metal_<element_type_, 4>(queries, output, log_sum_exp, work, a, head, group.y, groups.y,
                                                       warp, lane, scratch);
            continue;
        }
        if (a.depth <= 256 && !quantized) {
            nk_attention_rows_metal_<element_type_, 1>(queries, output, log_sum_exp, work, a, head, group.y, groups.y,
                                                       warp, lane, scratch);
            continue;
        }
        for (ulong local = work.row_begin + ulong(group.y) * 4 + warp; local < work.row_end;
             local += ulong(groups.y) * 4) {
            ulong const row = work.query_first + local;
            auto query = nk_attention_query_row_metal_<element_type_>(queries, a, row, head);
            nk_attention_rebase_metal_(query, a, work);
            if (query.exact) {
                nk_attention_exact_row_metal_(query, work, a, local, head, output, log_sum_exp, lane);
                continue;
            }
            device float *result = (device float *)(output + row * a.output_stride) + head * a.depth;
            ulong begin, end;
            nk_attention_row_range_metal_(a, work.first_position + long(local), work.length, begin, end);
            float maximum = -INFINITY;
            for (ulong position = begin; position < end; ++position)
                maximum = max(maximum, nk_attention_score_metal_<element_type_>(query, work.keys + position * row_bytes,
                                                                                depth, lane) *
                                           query.multiplier);
            for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                result[channel] = 0;
            float weights_sum = 0;
            for (ulong position = begin; position < end; ++position) {
                float const score = nk_attention_score_metal_<element_type_>(query, work.keys + position * row_bytes,
                                                                             depth, lane);
                float weight = nk_exp2_metal_(score * query.multiplier - maximum);
                if (quantized) weight = floor(weight * 255.0f + 0.5f);
                weights_sum += weight;
                device uchar const *value_row = work.values + position * row_bytes;
                for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                    result[channel] = fma(weight, float(element_type_::load(value_row, channel)), result[channel]);
            }
            float const inverse = query.normalization(weights_sum);
            for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                result[channel] = query.output(result[channel] * inverse);
            if (!lane)
                nk_attention_store_log_sum_exp_metal_(
                    log_sum_exp, a, row, head,
                    nk_attention_log_sum_exp_metal_(maximum, weights_sum, nk_attention_unit_metal_<element_type_>()));
        }
    }
}

kernel void nk_attention_packed_bf16_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::bf16_t>(queries, packed, output, query_offsets, a, log_sum_exp, group, groups,
                                             warp, lane, scratch);
}

kernel void nk_attention_packed_f16_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::f16_t>(queries, packed, output, query_offsets, a, log_sum_exp, group, groups, warp,
                                            lane, scratch);
}

kernel void nk_attention_packed_e4m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::e4m3_t>(queries, packed, output, query_offsets, a, log_sum_exp, group, groups,
                                             warp, lane, scratch);
}

kernel void nk_attention_packed_i8_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::i8_t>(queries, packed, output, query_offsets, a, log_sum_exp, group, groups, warp,
                                           lane, scratch);
}

kernel void nk_attention_packed_nvfp4_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], device float const *query_tensor_scale [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::f16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k>(
            queries, query_scales, query_tensor_scale, packed, a),
        packed, output, query_offsets, a, log_sum_exp, group, groups, warp, lane, scratch);
}

kernel void nk_attention_packed_mxfp4_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales,
                                                                                          nullptr, packed, a),
        packed, output, query_offsets, a, log_sum_exp, group, groups, warp, lane, scratch);
}

kernel void nk_attention_packed_mxfp6e2m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, a, log_sum_exp, group, groups, warp, lane, scratch);
}

kernel void nk_attention_packed_mxfp6e3m2_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, a, log_sum_exp, group, groups, warp, lane, scratch);
}

kernel void nk_attention_packed_mxfp8e4m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, a, log_sum_exp, group, groups, warp, lane, scratch);
}

kernel void nk_attention_packed_mxfp8e5m2_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device uchar const *query_scales [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_fallback_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, a, log_sum_exp, group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_bf16_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::bf16_t, nk_attention_partition_metal_k>(queries, packed, query_offsets, a, split,
                                                                          group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_f16_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::f16_t, nk_attention_partition_metal_k>(queries, packed, query_offsets, a, split,
                                                                         group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_e4m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::e4m3_t, nk_attention_partition_metal_k>(queries, packed, query_offsets, a, split,
                                                                          group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_i8_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::i8_t, nk_attention_partition_metal_k>(queries, packed, query_offsets, a, split, group,
                                                                        groups, warp, lane, scratch);
}

kernel void nk_attention_weights_i8_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::i8_t, nk_attention_values_metal_k>(queries, packed, query_offsets, a, split, group,
                                                                     groups, warp, lane, scratch);
}

kernel void nk_attention_split_nvfp4_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    device float const *query_tensor_scale [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::f16_t, nk_attention_partition_metal_k>(
        nk_attention_scaled_queries_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k>(
            queries, query_scales, query_tensor_scale, packed, a),
        packed, query_offsets, a, split, group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_mxfp4_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::bf16_t, nk_attention_partition_metal_k>(
        nk_attention_scaled_queries_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales,
                                                                                          nullptr, packed, a),
        packed, query_offsets, a, split, group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_mxfp6e2m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::bf16_t, nk_attention_partition_metal_k>(
        nk_attention_scaled_queries_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, query_offsets, a, split, group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_mxfp6e3m2_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::bf16_t, nk_attention_partition_metal_k>(
        nk_attention_scaled_queries_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, query_offsets, a, split, group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_mxfp8e4m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::bf16_t, nk_attention_partition_metal_k>(
        nk_attention_scaled_queries_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, query_offsets, a, split, group, groups, warp, lane, scratch);
}

kernel void nk_attention_split_mxfp8e5m2_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[1288];
    nk_attention_split_metal_<nk::bf16_t, nk_attention_partition_metal_k>(
        nk_attention_scaled_queries_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, query_offsets, a, split, group, groups, warp, lane, scratch);
}

kernel void nk_attention_merge_bf16_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::bf16_t>(queries, packed, output, query_offsets, a, log_sum_exp, split, group, groups,
                                          lane);
}

kernel void nk_attention_merge_f16_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::f16_t>(queries, packed, output, query_offsets, a, log_sum_exp, split, group, groups,
                                         lane);
}

kernel void nk_attention_merge_e4m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::e4m3_t>(queries, packed, output, query_offsets, a, log_sum_exp, split, group, groups,
                                          lane);
}

kernel void nk_attention_merge_i8_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::i8_t>(queries, packed, output, query_offsets, a, log_sum_exp, split, group, groups,
                                        lane);
}

kernel void nk_attention_merge_nvfp4_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    device float const *query_tensor_scale [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::f16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m1x2_t, 16, nk_cross_scale_e4m3_metal_k>(
            queries, query_scales, query_tensor_scale, packed, a),
        packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

kernel void nk_attention_merge_mxfp4_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m1x2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales,
                                                                                          nullptr, packed, a),
        packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

kernel void nk_attention_merge_mxfp6e2m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e2m3_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

kernel void nk_attention_merge_mxfp6e3m2_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e3m2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

kernel void nk_attention_merge_mxfp8e4m3_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e4m3_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

kernel void nk_attention_merge_mxfp8e5m2_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_arguments_metal_t &a [[buffer(4)]], device float *log_sum_exp [[buffer(5)]],
    device float const *split [[buffer(6)]], device uchar const *query_scales [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::bf16_t>(
        nk_attention_scaled_queries_metal_<nk::e5m2_t, 32, nk_cross_scale_e8m0_metal_k>(queries, query_scales, nullptr,
                                                                                        packed, a),
        packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

/** Stages a 32-row tile of the work's queries, each row under its base in @p bases, and a 64-key
 *  panel of its keys, @p step_ channels from @p depth_first, into @p left and @p right, with zeros
 *  past every edge. */
template <typename stage_type_, uint step_, typename element_type_, typename queries_type_>
inline void nk_attention_stage_qk_metal_(queries_type_ queries, thread nk_attention_work_metal_t const &work,
                                         constant nk_attention_arguments_metal_t &a, ulong head, ulong row_first,
                                         ulong key_first, ulong depth_first, threadgroup stage_type_ *left,
                                         threadgroup stage_type_ *right, threadgroup int const *bases,
                                         uint thread_index) {
    constexpr uint step = step_;
    ulong const row_bytes = a.depth * sizeof(typename element_type_::raw_t);
    for (uint cell = thread_index; cell < (32 + 64) * step; cell += 128) {
        uint const row = cell / step, channel = cell % step;
        float value = 0;
        if (row < 32) {
            ulong const local = row_first + row;
            if (local < work.queries && depth_first + channel < a.depth) {
                auto query = nk_attention_query_row_metal_<element_type_>(queries, a, work.query_first + local, head);
                query.base = bases[row];
                value = float(query.load(depth_first + channel));
            }
            left[cell] = stage_type_(value);
        }
        else {
            ulong const position = key_first + row - 32;
            if (position < work.length && depth_first + channel < a.depth)
                value = float(element_type_::load(work.keys + position * row_bytes, depth_first + channel));
            right[cell - 32 * step] = stage_type_(value);
        }
    }
}

/** Stages a 64-key panel of the work's values, 32 channels from @p depth_first, into @p right. */
template <typename stage_type_, typename element_type_>
inline void nk_attention_stage_v_metal_(thread nk_attention_work_metal_t const &work,
                                        constant nk_attention_arguments_metal_t &a, ulong key_first, ulong depth_first,
                                        threadgroup stage_type_ *right, uint thread_index) {
    ulong const row_bytes = a.depth * sizeof(typename element_type_::raw_t);
    for (uint cell = thread_index; cell < 64 * 32; cell += 128) {
        uint const position = cell / 32, channel = cell % 32;
        float const value = key_first + position < work.length && depth_first + channel < a.depth
                                ? float(element_type_::load(work.values + (key_first + position) * row_bytes,
                                                            depth_first + channel))
                                : 0;
        right[cell] = stage_type_(value);
    }
}

/** Folds a panel of scores, each row under its multiplier from @p multipliers, into each tile row's
 *  running maximum, recording its correction, over the keys the band shows to the row. */
inline void nk_attention_maximum_metal_(threadgroup float const *scores, threadgroup float *maximum,
                                        threadgroup float *corrections, threadgroup float const *multipliers,
                                        thread nk_attention_work_metal_t const &work,
                                        constant nk_attention_arguments_metal_t &a, ulong row_first, ulong key_first,
                                        uint thread_index) {
    uint const row = thread_index / 4, lane = thread_index % 4;
    ulong begin, end;
    nk_attention_row_range_metal_(a, work.first_position + long(row_first + row), work.length, begin, end);
    float value = -INFINITY;
    for (uint column = lane; column < 64; column += 4)
        if (row_first + row < work.queries && key_first + column >= begin && key_first + column < end)
            value = max(value, scores[row * 64 + column] * multipliers[row]);
    value = max(value, simd_shuffle_xor(value, 1));
    value = max(value, simd_shuffle_xor(value, 2));
    if (!lane) {
        float const previous = maximum[row], next = max(previous, value);
        corrections[row] = previous == next ? 1 : nk_exp2_metal_(previous - next);
        maximum[row] = next;
    }
}

/** Turns a panel of scores, each row under its multiplier from @p multipliers, into staged weights,
 *  U8 ones offset by −128 into @c int8_t, and folds their sum into each tile row's running sum. */
template <typename stage_type_>
inline void nk_attention_weights_metal_(threadgroup float const *scores, threadgroup stage_type_ *weights,
                                        threadgroup float const *maximum, threadgroup float const *corrections,
                                        threadgroup float *sums, threadgroup float const *multipliers,
                                        thread nk_attention_work_metal_t const &work,
                                        constant nk_attention_arguments_metal_t &a, ulong row_first, ulong key_first,
                                        uint thread_index) {
#pragma clang fp contract(off) reassociate(off)
    uint const row = thread_index / 4, lane = thread_index % 4;
    ulong begin, end;
    nk_attention_row_range_metal_(a, work.first_position + long(row_first + row), work.length, begin, end);
    float sum = 0;
    for (uint column = lane; column < 64; column += 4) {
        float const probability = row_first + row < work.queries && key_first + column >= begin &&
                                          key_first + column < end
                                      ? nk_exp2_metal_(scores[row * 64 + column] * multipliers[row] - maximum[row])
                                      : 0;
        if (sizeof(stage_type_) == 1) {
            int const weight = int(floor(probability * 255.0f + 0.5f));
            weights[row * 64 + column] = stage_type_(weight - 128), sum += float(weight);
        }
        else {
            stage_type_ const weight = stage_type_(probability);
            weights[row * 64 + column] = weight, sum += float(weight);
        }
    }
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    if (!lane) sums[row] = sums[row] * corrections[row] + sum;
}

/** Where a tile keeps its output rows between panels: in registers up to depth 128, or in the
 *  output rows themselves past it. */
enum nk_attention_accumulation_metal_t {
    nk_attention_accumulate_device_metal_k,
    nk_attention_accumulate_registers_metal_k,
};

/**
 *  @brief The tiled attention @c apple9 and @c apple10 share: Q · K and P · V of 32-row tiles
 *      and 64-key panels on @p operations_type_ matrices, with the online softmax between them.
 *
 *  Tiles are cut at multiples of 32 rows within each segment and staged whole, so a window's rows
 *  match one call over the grid bit for bit; only the window's rows are stored. U8 weights take the
 *  maximum over every panel first, and decode segments of up to four queries take the rows kernel.
 *  Each row's multiplier and base land in @p multipliers and @p bases, and rows on the exact path
 *  sweep alone after their tile, overwriting what it stored for them.
 */
template <typename stage_type_, typename operations_type_, typename element_type_,
          nk_attention_accumulation_metal_t accumulation_, typename queries_type_>
inline void nk_attention_matrix_metal_(queries_type_ queries, device uchar const *packed, device uchar *output,
                                       device uint const *query_offsets, device float *log_sum_exp,
                                       constant nk_attention_arguments_metal_t &a, uint2 group, uint2 groups,
                                       uint thread_index, uint warp, threadgroup stage_type_ *left,
                                       threadgroup stage_type_ *right, threadgroup float *scores,
                                       threadgroup float *maximum, threadgroup float *corrections,
                                       threadgroup float *sums, threadgroup float *multipliers,
                                       threadgroup int *bases) {
    using stage_t = stage_type_;
    using operations_t = operations_type_;
    constexpr bool resident = accumulation_ == nk_attention_accumulate_registers_metal_k;
    ulong const segments = nk_attention_segments_metal_(packed, a);
    for (ulong pair = group.x; pair < segments * a.heads; pair += groups.x) {
        ulong const head = pair % a.heads;
        nk_attention_work_metal_t work;
        if (!nk_attention_work_metal_<element_type_>(packed, query_offsets, a, segments, pair, work)) continue;
        if (work.queries <= 4 && a.depth <= 256) {
            nk_attention_rows_metal_<element_type_, 4>(queries, output, log_sum_exp, work, a, head, group.y, groups.y,
                                                       warp, thread_index % 32, scores);
            continue;
        }
        for (ulong row_first = (work.row_begin / 32 + group.y) * 32; row_first < work.row_end;
             row_first += ulong(groups.y) * 32) {
            ulong key_begin, key_end;
            nk_attention_tile_keys_metal_(a, work.first_position + long(row_first),
                                          work.first_position + long(min(row_first + 32, work.queries) - 1),
                                          work.length, key_begin, key_end);
            key_begin = key_begin < key_end ? key_begin & ~63ul : key_end;
            if (thread_index < 32) {
                maximum[thread_index] = -INFINITY, sums[thread_index] = 0, corrections[thread_index] = 1;
                multipliers[thread_index] = 0, bases[thread_index] = 0;
                if (row_first + thread_index < work.queries) {
                    auto query = nk_attention_query_row_metal_<element_type_>(
                        queries, a, work.query_first + row_first + thread_index, head);
                    nk_attention_rebase_metal_(query, a, work);
                    multipliers[thread_index] = query.multiplier, bases[thread_index] = query.base;
                }
            }
            // Rows of one pair share the output's factor and power of two
            auto tile_query = nk_attention_query_row_metal_<element_type_>(queries, a, work.query_first + row_first,
                                                                           head);
            nk_attention_rebase_metal_(tile_query, a, work);
            float accumulation[resident ? 32 : 1] = {};
            if constexpr (!resident)
                for (ulong cell = thread_index; cell < 32 * a.depth; cell += 128) {
                    ulong const local = row_first + cell / a.depth, channel = cell % a.depth,
                                row = work.query_first + local;
                    device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
                    if (local >= work.row_begin && local < work.row_end) destination[channel] = 0;
                }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            if constexpr (sizeof(stage_t) == 1) {
                for (ulong key_first = key_begin; key_first < key_end; key_first += 64) {
                    operations_t::template qk<stage_t, element_type_>(queries, work, a, head, row_first, key_first,
                                                                      left, right, scores, bases, thread_index, warp);
                    nk_attention_maximum_metal_(scores, maximum, corrections, multipliers, work, a, row_first,
                                                key_first, thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (thread_index < 32) corrections[thread_index] = 1;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (ulong key_first = key_begin; key_first < key_end; key_first += 64) {
                operations_t::template qk<stage_t, element_type_>(queries, work, a, head, row_first, key_first, left,
                                                                  right, scores, bases, thread_index, warp);
                if constexpr (sizeof(stage_t) != 1) {
                    nk_attention_maximum_metal_(scores, maximum, corrections, multipliers, work, a, row_first,
                                                key_first, thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                nk_attention_weights_metal_(scores, left, maximum, corrections, sums, multipliers, work, a, row_first,
                                            key_first, thread_index);
                threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
                for (ulong depth_first = 0; depth_first < (resident ? 128 : a.depth); depth_first += 32) {
                    if (depth_first >= a.depth) continue;
                    nk_attention_stage_v_metal_<stage_t, element_type_>(work, a, key_first, depth_first, right,
                                                                        thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    operations_t::template pv<stage_t>(left, right, scores, thread_index, warp);
#pragma unroll
                    for (uint index = 0; index < 8; ++index) {
                        uint const cell = thread_index + index * 128;
                        ulong const local = row_first + cell / 32, channel = depth_first + cell % 32;
                        ulong const row = work.query_first + local;
                        if constexpr (resident) {
                            uint const slot = uint(depth_first / 32) * 8 + index;
                            accumulation[slot] = accumulation[slot] * corrections[cell / 32] + scores[cell];
                        }
                        else if (local >= work.row_begin && local < work.row_end && channel < a.depth) {
                            device float *destination = (device float *)(output + row * a.output_stride) +
                                                        head * a.depth;
                            destination[channel] = destination[channel] * corrections[cell / 32] + scores[cell];
                        }
                    }
                    threadgroup_barrier(resident ? mem_flags::mem_threadgroup
                                                 : mem_flags::mem_threadgroup | mem_flags::mem_device);
                }
            }
            if constexpr (resident) {
#pragma unroll
                for (uint slot = 0; slot < 32; ++slot) {
                    uint const cell = thread_index + slot % 8 * 128;
                    ulong const local = row_first + cell / 32, row = work.query_first + local;
                    ulong const channel = slot / 8 * 32 + cell % 32;
                    float const sum = sums[cell / 32];
                    if (local < work.row_begin || local >= work.row_end || channel >= a.depth) continue;
                    device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
                    destination[channel] = tile_query.output(accumulation[slot] * tile_query.normalization(sum));
                }
            }
            else
                for (ulong cell = thread_index; cell < 32 * a.depth; cell += 128) {
                    ulong const local = row_first + cell / a.depth, channel = cell % a.depth,
                                row = work.query_first + local;
                    float const sum = sums[cell / a.depth];
                    device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
                    if (local >= work.row_begin && local < work.row_end)
                        destination[channel] = tile_query.output(destination[channel] * tile_query.normalization(sum));
                }
            if (thread_index < 32 && row_first + thread_index >= work.row_begin &&
                row_first + thread_index < work.row_end)
                nk_attention_store_log_sum_exp_metal_(
                    log_sum_exp, a, work.query_first + row_first + thread_index, head,
                    nk_attention_log_sum_exp_metal_(maximum[thread_index], sums[thread_index],
                                                    nk_attention_unit_metal_<element_type_>()));
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            for (ulong local = row_first + warp; local < min(row_first + 32, work.row_end); local += 4) {
                if (local < work.row_begin) continue;
                auto query = nk_attention_query_row_metal_<element_type_>(queries, a, work.query_first + local, head);
                nk_attention_rebase_metal_(query, a, work);
                if (query.exact)
                    nk_attention_exact_row_metal_(query, work, a, local, head, output, log_sum_exp, thread_index % 32);
            }
        }
    }
}

/** Fills @p work with every query row of @p task, segment × K and V heads + K and V head, and that
 *  head's planes, returning false when the queries run past the rows the call addresses or the
 *  planes run past the pack. */
template <typename element_type_>
inline bool nk_attention_backward_work_metal_(device uchar const *packed, device uint const *query_offsets,
                                              constant nk_attention_arguments_metal_t &a, ulong segments, ulong task,
                                              thread nk_attention_work_metal_t &work) {
    ulong const segment = task / a.kv_heads;
    ulong const query_first = query_offsets[segment], query_end = query_offsets[segment + 1];
    if (query_first > query_end || query_end > a.query_tokens) return false;
    work.query_first = query_first, work.queries = query_end - query_first;
    return nk_attention_planes_metal_<element_type_>(packed, a, segments, segment, task % a.kv_heads, work);
}

/** Writes the rows of @p work whose band shows a key from @p key_begin to @p key_end: from the
 *  first whose keys end past @p key_begin to the last whose keys begin before @p key_end. */
inline void nk_attention_panel_rows_metal_(constant nk_attention_arguments_metal_t &a,
                                           thread nk_attention_work_metal_t const &work, ulong key_begin, ulong key_end,
                                           thread ulong &row_begin, thread ulong &row_end) {
    // Bands past the segment's keys and queries together show the same keys
    long const reach = long(work.length + work.queries), rows = long(work.queries);
    long const before = long(min(a.keys_before, ulong(reach))), after = long(min(a.keys_after, ulong(reach)));
    row_begin = ulong(min(max(long(key_begin) - after - work.first_position, 0l), rows));
    row_end = ulong(min(max(long(key_end) + before - work.first_position, 0l), rows));
}

/** D = dO · O of the query row and head at @p offset, shared by the 32 lanes of a simdgroup and
 *  returned in each. */
inline float nk_attention_backward_row_dot_metal_(device uchar const *output, device uchar const *output_gradient,
                                                  ulong offset, uint depth, uint lane) {
    device float const *output_row = (device float const *)(output + offset);
    device float const *gradient_row = (device float const *)(output_gradient + offset);
    float sum = 0;
    for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
        sum = fma(gradient_row[channel], output_row[channel], sum);
    return simd_sum(sum);
}

/** What the key pass keeps of each of the eight folded query rows it stages: where its Q and dO
 *  lie, its base-2 log-sum-exp and D, and the keys the band shows it. */
struct nk_attention_backward_rows_metal_t {
    ulong query_offset[8], output_offset[8];
    float log_sum_exp2[8], dots[8];
    uint key_begin[8], key_end[8];
};

/**
 *  @brief The key and value gradients of every task of the window: a simdgroup per key and eight
 *      keys per threadgroup along y, each summing over the folded query rows, row × group + head.
 *
 *  The threadgroup stages the folded rows its keys' band shows eight at a time with their
 *  statistics, then each simdgroup takes P = 2^(score₂ − lse₂) and dS = P · (dO · V − D) · scale,
 *  and adds P · dO to dV and dS · Q to dK. Up to depth 256 it stages Q and dO too and sums in
 *  registers, while deeper heads read the rows and sum straight into the gradient rows.
 */
template <nk_attention_accumulation_metal_t accumulation_>
inline void nk_attention_backward_keys_bf16_metal_(device uchar const *queries, device uchar const *packed,
                                                   device uchar const *output, device uint const *query_offsets,
                                                   constant nk_attention_backward_arguments_metal_t &b,
                                                   device float const *log_sum_exp, device uchar const *output_gradient,
                                                   device uchar *key_gradient, device uchar *value_gradient,
                                                   uint2 group, uint2 groups, uint warp, uint lane,
                                                   threadgroup float *queries_tile, threadgroup float *gradients_tile,
                                                   threadgroup nk_attention_backward_rows_metal_t &rows) {
#pragma clang fp contract(off) reassociate(off)
    constexpr bool resident = accumulation_ == nk_attention_accumulate_registers_metal_k;
    constant nk_attention_arguments_metal_t &a = b.attention;
    ulong const segments = nk_attention_segments_metal_(packed, a), tasks_end = min(a.tasks_end, segments * a.kv_heads);
    ulong const group_size = a.heads / a.kv_heads, row_bytes = a.depth * sizeof(bfloat);
    ulong const gradient_bytes = b.key_value_gradient_bytes, gradient_stride = b.key_value_gradient_stride;
    ulong const gradient_row_bytes = a.kv_heads * a.depth * sizeof(float);
    device uint const *key_offsets = (device uint const *)(packed + 64 + segments * 8);
    uint const depth = uint(a.depth);
    for (ulong task = a.tasks_begin + group.x; task < tasks_end; task += groups.x) {
        nk_attention_work_metal_t work;
        if (!nk_attention_backward_work_metal_<nk::bf16_t>(packed, query_offsets, a, segments, task, work)) continue;
        ulong const key_first = key_offsets[task / a.kv_heads], key_value_head = task % a.kv_heads;
        if (!work.length || gradient_row_bytes > gradient_bytes ||
            (gradient_stride && key_first + work.length - 1 > (gradient_bytes - gradient_row_bytes) / gradient_stride))
            continue;
        for (ulong block = ulong(group.y) * 8; block < work.length; block += ulong(groups.y) * 8) {
            ulong const position = block + warp;
            ulong const gradient_offset = (key_first + position) * gradient_stride +
                                          key_value_head * a.depth * sizeof(float);
            bool const live = position < work.length;
            device uchar const *key = work.keys + position * row_bytes, *value = work.values + position * row_bytes;
            device float *key_gradient_row = (device float *)(key_gradient + gradient_offset);
            device float *value_gradient_row = (device float *)(value_gradient + gradient_offset);
            float key_cached[8] = {0}, value_cached[8] = {0}, key_sums[8] = {0}, value_sums[8] = {0};
            if constexpr (resident) {
#pragma unroll
                for (uint index = 0; index < 8; ++index) {
                    uint const channel = lane + index * 32;
                    if (live && channel < depth) {
                        key_cached[index] = nk::bf16_t::load(key, channel);
                        value_cached[index] = nk::bf16_t::load(value, channel);
                    }
                }
            }
            else if (live)
                for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                    key_gradient_row[channel] = 0, value_gradient_row[channel] = 0;

            ulong row_begin, row_end;
            nk_attention_panel_rows_metal_(a, work, block, min(block + 8, work.length), row_begin, row_end);
            for (ulong first = row_begin * group_size; first < row_end * group_size; first += 8) {
                uint const count = uint(min(8ul, row_end * group_size - first));
                if (warp < count) {
                    ulong const entry = first + warp, row = entry / group_size, token = work.query_first + row;
                    ulong const head = key_value_head * group_size + entry % group_size;
                    ulong const query_offset = token * a.query_stride + head * row_bytes;
                    ulong const output_offset = token * a.output_stride + head * a.depth * sizeof(float);
                    device float const *gradient = (device float const *)(output_gradient + output_offset);
                    float const dot = nk_attention_backward_row_dot_metal_(output, output_gradient, output_offset,
                                                                           depth, lane);
                    if constexpr (resident)
                        for (uint channel = lane; channel < depth; channel += 32) {
                            queries_tile[warp * 256 + channel] = nk::bf16_t::load(queries + query_offset, channel);
                            gradients_tile[warp * 256 + channel] = gradient[channel];
                        }
                    if (!lane) {
                        ulong begin, end;
                        nk_attention_row_range_metal_(a, work.first_position + long(row), work.length, begin, end);
                        rows.query_offset[warp] = query_offset, rows.output_offset[warp] = output_offset;
                        rows.log_sum_exp2[warp] = log_sum_exp[token * a.heads + head] * M_LOG2E_F;
                        rows.dots[warp] = dot, rows.key_begin[warp] = uint(begin), rows.key_end[warp] = uint(end);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint staged = 0; live && staged < count; ++staged) {
                    if (position < rows.key_begin[staged] || position >= rows.key_end[staged]) continue;
                    threadgroup float const *query_tile = queries_tile + staged * 256;
                    threadgroup float const *gradient_tile = gradients_tile + staged * 256;
                    device uchar const *query = queries + rows.query_offset[staged];
                    device float const *gradient = (device float const *)(output_gradient + rows.output_offset[staged]);
                    float2 dots = 0;
                    if constexpr (resident) {
#pragma unroll
                        for (uint index = 0; index < 8; ++index) {
                            uint const channel = lane + index * 32;
                            if (channel < depth)
                                dots = fma(float2(query_tile[channel], gradient_tile[channel]),
                                           float2(key_cached[index], value_cached[index]), dots);
                        }
                    }
                    else
                        for (uint channel = lane; channel < depth;
                             channel = depth - channel <= 32 ? depth : channel + 32)
                            dots = fma(float2(nk::bf16_t::load(query, channel), gradient[channel]),
                                       float2(nk::bf16_t::load(key, channel), nk::bf16_t::load(value, channel)), dots);
                    dots = simd_sum(dots);
                    float const weight = nk_exp2_metal_(dots.x * a.scale2 - rows.log_sum_exp2[staged]);
                    float const score_gradient = weight * (dots.y - rows.dots[staged]) * b.scale;
                    if constexpr (resident) {
#pragma unroll
                        for (uint index = 0; index < 8; ++index) {
                            uint const channel = lane + index * 32;
                            if (channel < depth) {
                                value_sums[index] = fma(weight, gradient_tile[channel], value_sums[index]);
                                key_sums[index] = fma(score_gradient, query_tile[channel], key_sums[index]);
                            }
                        }
                    }
                    else
                        for (uint channel = lane; channel < depth;
                             channel = depth - channel <= 32 ? depth : channel + 32) {
                            float const query_value = nk::bf16_t::load(query, channel);
                            value_gradient_row[channel] = fma(weight, gradient[channel], value_gradient_row[channel]);
                            key_gradient_row[channel] = fma(score_gradient, query_value, key_gradient_row[channel]);
                        }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if constexpr (resident) {
#pragma unroll
                for (uint index = 0; index < 8; ++index) {
                    uint const channel = lane + index * 32;
                    if (live && channel < depth) {
                        key_gradient_row[channel] = key_sums[index];
                        value_gradient_row[channel] = value_sums[index];
                    }
                }
            }
        }
    }
}

/**
 *  @brief The query gradients of every task of the window: a simdgroup per folded query row,
 *      eight per threadgroup along y, each summing dS · K over its keys.
 *
 *  Up to depth 256 the threadgroup stages the K and V rows its rows' band shows eight at a time and
 *  sums in registers, while deeper heads read the rows and sum straight into the gradient row.
 */
template <nk_attention_accumulation_metal_t accumulation_>
inline void nk_attention_backward_queries_bf16_metal_(device uchar const *queries, device uchar const *packed,
                                                      device uchar const *output, device uint const *query_offsets,
                                                      constant nk_attention_backward_arguments_metal_t &b,
                                                      device float const *log_sum_exp,
                                                      device uchar const *output_gradient, device uchar *query_gradient,
                                                      uint2 group, uint2 groups, uint warp, uint lane,
                                                      threadgroup float *keys_tile, threadgroup float *values_tile) {
#pragma clang fp contract(off) reassociate(off)
    constexpr bool resident = accumulation_ == nk_attention_accumulate_registers_metal_k;
    constant nk_attention_arguments_metal_t &a = b.attention;
    ulong const segments = nk_attention_segments_metal_(packed, a), tasks_end = min(a.tasks_end, segments * a.kv_heads);
    ulong const group_size = a.heads / a.kv_heads, row_bytes = a.depth * sizeof(bfloat);
    uint const depth = uint(a.depth);
    for (ulong task = a.tasks_begin + group.x; task < tasks_end; task += groups.x) {
        nk_attention_work_metal_t work;
        if (!nk_attention_backward_work_metal_<nk::bf16_t>(packed, query_offsets, a, segments, task, work)) continue;
        ulong const entries = work.queries * group_size, key_value_head = task % a.kv_heads;
        for (ulong block = ulong(group.y) * 8; block < entries; block += ulong(groups.y) * 8) {
            ulong const entry = block + warp, row = entry / group_size, token = work.query_first + row;
            ulong const head = key_value_head * group_size + entry % group_size;
            ulong const output_offset = token * a.output_stride + head * a.depth * sizeof(float);
            bool const live = entry < entries;
            device uchar const *query = queries + token * a.query_stride + head * row_bytes;
            device float const *gradient = (device float const *)(output_gradient + output_offset);
            device float *query_gradient_row = (device float *)(query_gradient + token * b.query_gradient_stride) +
                                               head * a.depth;
            ulong begin = 0, end = 0;
            float dot = 0, log_sum_exp2 = 0;
            if (live) {
                nk_attention_row_range_metal_(a, work.first_position + long(row), work.length, begin, end);
                dot = nk_attention_backward_row_dot_metal_(output, output_gradient, output_offset, depth, lane);
                log_sum_exp2 = log_sum_exp[token * a.heads + head] * M_LOG2E_F;
            }
            if constexpr (resident) {
                float query_cached[8] = {0}, gradient_cached[8] = {0}, sums[8] = {0};
#pragma unroll
                for (uint index = 0; index < 8; ++index) {
                    uint const channel = lane + index * 32;
                    if (live && channel < depth) {
                        query_cached[index] = nk::bf16_t::load(query, channel);
                        gradient_cached[index] = gradient[channel];
                    }
                }
                ulong panel_begin, panel_end;
                nk_attention_tile_keys_metal_(a, work.first_position + long(block / group_size),
                                              work.first_position + long((min(block + 8, entries) - 1) / group_size),
                                              work.length, panel_begin, panel_end);
                for (ulong panel = panel_begin; panel < panel_end; panel += 8) {
                    device uchar const *key = work.keys + (panel + warp) * row_bytes;
                    device uchar const *value = work.values + (panel + warp) * row_bytes;
                    for (uint channel = lane; panel + warp < panel_end && channel < depth; channel += 32) {
                        keys_tile[warp * 256 + channel] = nk::bf16_t::load(key, channel);
                        values_tile[warp * 256 + channel] = nk::bf16_t::load(value, channel);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint staged = 0; live && staged < 8; ++staged) {
                        if (panel + staged < begin || panel + staged >= end) continue;
                        threadgroup float const *key_tile = keys_tile + staged * 256;
                        threadgroup float const *value_tile = values_tile + staged * 256;
                        float2 dots = 0;
#pragma unroll
                        for (uint index = 0; index < 8; ++index) {
                            uint const channel = lane + index * 32;
                            if (channel < depth)
                                dots = fma(float2(query_cached[index], gradient_cached[index]),
                                           float2(key_tile[channel], value_tile[channel]), dots);
                        }
                        dots = simd_sum(dots);
                        float const weight = nk_exp2_metal_(dots.x * a.scale2 - log_sum_exp2);
                        float const score_gradient = weight * (dots.y - dot) * b.scale;
#pragma unroll
                        for (uint index = 0; index < 8; ++index) {
                            uint const channel = lane + index * 32;
                            if (channel < depth) sums[index] = fma(score_gradient, key_tile[channel], sums[index]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
#pragma unroll
                for (uint index = 0; index < 8; ++index) {
                    uint const channel = lane + index * 32;
                    if (live && channel < depth) query_gradient_row[channel] = sums[index];
                }
            }
            else if (live) {
                for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                    query_gradient_row[channel] = 0;
                for (ulong position = begin; position < end; ++position) {
                    device uchar const *key = work.keys + position * row_bytes;
                    device uchar const *value = work.values + position * row_bytes;
                    float2 dots = 0;
                    for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                        dots = fma(float2(nk::bf16_t::load(query, channel), gradient[channel]),
                                   float2(nk::bf16_t::load(key, channel), nk::bf16_t::load(value, channel)), dots);
                    dots = simd_sum(dots);
                    float const weight = nk_exp2_metal_(dots.x * a.scale2 - log_sum_exp2);
                    float const score_gradient = weight * (dots.y - dot) * b.scale;
                    for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                        query_gradient_row[channel] = fma(score_gradient, nk::bf16_t::load(key, channel),
                                                          query_gradient_row[channel]);
                }
            }
        }
    }
}

/** Which gradients a tiled pass sums: those of a tile's keys and values, or of its queries. */
enum nk_attention_backward_pass_metal_t {
    nk_attention_backward_keys_metal_k,
    nk_attention_backward_queries_metal_k,
};

/** What a gradients tile keeps of each of its 32 folded rows: its base-2 log-sum-exp, D = dO · O,
 *  and the keys the band shows it, none for rows past the task's. */
struct nk_attention_backward_panel_metal_t {
    float log_sum_exp2[32], dots[32];
    uint key_begin[32], key_end[32];
};

/** Fills @p panel for the 32 folded rows of the task from @p first, four threads summing each D. */
inline void nk_attention_backward_panel_metal_(device uchar const *output, device uchar const *output_gradient,
                                               device float const *log_sum_exp,
                                               thread nk_attention_work_metal_t const &work,
                                               constant nk_attention_arguments_metal_t &a, ulong key_value_head,
                                               ulong first, threadgroup nk_attention_backward_panel_metal_t &panel,
                                               uint thread_index) {
#pragma clang fp contract(off) reassociate(off)
    ulong const group_size = a.heads / a.kv_heads;
    uint const slot = thread_index / 4, part = thread_index % 4;
    ulong const entry = first + slot, row = entry / group_size, token = work.query_first + row;
    ulong const head = key_value_head * group_size + entry % group_size;
    bool const live = entry < work.queries * group_size;
    ulong const offset = token * a.output_stride + head * a.depth * sizeof(float);
    float dot = 0;
    for (ulong channel = part; live && channel < a.depth; channel += 4)
        dot = fma(nk::f32_t::load(output_gradient + offset, uint(channel)),
                  nk::f32_t::load(output + offset, uint(channel)), dot);
    dot += simd_shuffle_xor(dot, 1);
    dot += simd_shuffle_xor(dot, 2);
    if (part) return;
    ulong begin = 0, end = 0;
    if (live) nk_attention_row_range_metal_(a, work.first_position + long(row), work.length, begin, end);
    panel.log_sum_exp2[slot] = live ? log_sum_exp[token * a.heads + head] * M_LOG2E_F : 0;
    panel.dots[slot] = dot, panel.key_begin[slot] = uint(begin), panel.key_end[slot] = uint(end);
}

/** Stages @p width channels from @p depth_first of the task's 32 folded rows from @p first into
 *  @p query_tile and @p gradient_tile, output gradients rounded to BF16, zeros past every edge. */
inline void nk_attention_backward_stage_rows_metal_(device uchar const *queries, device uchar const *output_gradient,
                                                    thread nk_attention_work_metal_t const &work,
                                                    constant nk_attention_arguments_metal_t &a, ulong key_value_head,
                                                    ulong first, ulong depth_first, uint width,
                                                    threadgroup bfloat *query_tile, threadgroup bfloat *gradient_tile,
                                                    uint thread_index) {
    ulong const group_size = a.heads / a.kv_heads;
    for (uint cell = thread_index; cell < 32 * width; cell += 128) {
        uint const slot = cell / width, channel = uint(depth_first) + cell % width;
        ulong const entry = first + slot, token = work.query_first + entry / group_size;
        ulong const head = key_value_head * group_size + entry % group_size;
        bool const live = entry < work.queries * group_size && channel < a.depth;
        query_tile[cell] = bfloat(
            live ? nk::bf16_t::load(queries + token * a.query_stride + head * a.depth * sizeof(bfloat), channel) : 0);
        gradient_tile[cell] = bfloat(
            live ? nk::f32_t::load(output_gradient + token * a.output_stride + head * a.depth * sizeof(float), channel)
                 : 0);
    }
}

/** Stages @p width channels from @p depth_first of 32 rows of @p plane, the task's K or V, from key
 *  @p first into @p tile, zeros past every edge. */
inline void nk_attention_backward_stage_plane_metal_(device uchar const *plane,
                                                     thread nk_attention_work_metal_t const &work,
                                                     constant nk_attention_arguments_metal_t &a, ulong first,
                                                     ulong depth_first, uint width, threadgroup bfloat *tile,
                                                     uint thread_index) {
    ulong const row_bytes = a.depth * sizeof(bfloat);
    for (uint cell = thread_index; cell < 32 * width; cell += 128) {
        ulong const position = first + cell / width;
        uint const channel = uint(depth_first) + cell % width;
        tile[cell] = bfloat(
            position < work.length && channel < a.depth ? nk::bf16_t::load(plane + position * row_bytes, channel) : 0);
    }
}

/** Turns a tile's scores S and weight gradients dP, its 32 folded rows by 32 keys from
 *  @p key_first, into BF16 weights and score gradients, zero outside each row's band and laid
 *  out by keys for the key pass: P = 2^(S · scale₂ − lse₂) and dS = P · (dP − D) · scale. */
template <nk_attention_backward_pass_metal_t pass_>
inline void nk_attention_backward_weights_metal_(threadgroup float const *scores,
                                                 threadgroup float const *weight_gradients, threadgroup bfloat *weights,
                                                 threadgroup bfloat *score_gradients,
                                                 threadgroup nk_attention_backward_panel_metal_t const &panel,
                                                 constant nk_attention_backward_arguments_metal_t &b, ulong key_first,
                                                 uint thread_index) {
#pragma clang fp contract(off) reassociate(off)
    for (uint cell = thread_index; cell < 32 * 32; cell += 128) {
        uint const row = cell / 32, key = cell % 32;
        ulong const position = key_first + key;
        float weight = 0, score_gradient = 0;
        if (position >= panel.key_begin[row] && position < panel.key_end[row]) {
            weight = nk_exp2_metal_(scores[cell] * b.attention.scale2 - panel.log_sum_exp2[row]);
            score_gradient = weight * (weight_gradients[cell] - panel.dots[row]) * b.scale;
        }
        uint const slot = pass_ == nk_attention_backward_keys_metal_k ? key * 32 + row : cell;
        weights[slot] = bfloat(weight), score_gradients[slot] = bfloat(score_gradient);
    }
}

/** Writes the 32 rows of 128 columns from @p slice that @p tile holds to the gradient rows from
 *  @p gradient, @p stride bytes apart, skipping rows from @p rows on and columns past @p depth,
 *  then waits for the threadgroup. */
inline void nk_attention_backward_store_keys_metal_(threadgroup float const *tile, device uchar *gradient, ulong stride,
                                                    ulong rows, ulong slice, ulong depth, uint thread_index) {
    for (uint cell = thread_index; cell < 32 * 128; cell += 128) {
        ulong const position = cell / 128, channel = slice + cell % 128;
        if (position < rows && channel < depth)
            nk::f32_t::store(gradient + position * stride, uint(channel), tile[cell]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/**
 *  @brief The key and value gradients @c apple9 and @c apple10 share: per tile of 32 keys, over the
 *      folded rows that see it 32 at a time, S = Q · Kᵀ and dP = dO · Vᵀ on @p operations_type_
 *      matrices, then dV += Pᵀ · dO and dK += dSᵀ · Q.
 *
 *  Gradients sum 128 columns at a time in registers, deeper heads recomputing S and dP per slice,
 *  and every tile sums its rows in one order, so windows match one call bit for bit. Probabilities,
 *  score gradients and output gradients enter the products in BF16.
 */
template <typename operations_type_>
inline void nk_attention_backward_keys_matrix_metal_(
    device uchar const *queries, device uchar const *packed, device uchar const *output,
    device uint const *query_offsets, constant nk_attention_backward_arguments_metal_t &b,
    device float const *log_sum_exp, device uchar const *output_gradient, device uchar *key_gradient,
    device uchar *value_gradient, uint2 group, uint2 groups, uint thread_index, uint warp, threadgroup bfloat *stage,
    threadgroup float *scores, threadgroup float *weight_gradients, threadgroup bfloat *weights,
    threadgroup bfloat *score_gradients, threadgroup nk_attention_backward_panel_metal_t &panel) {
    using operations_t = operations_type_;
    constant nk_attention_arguments_metal_t &a = b.attention;
    ulong const segments = nk_attention_segments_metal_(packed, a), tasks_end = min(a.tasks_end, segments * a.kv_heads);
    ulong const group_size = a.heads / a.kv_heads;
    ulong const gradient_bytes = b.key_value_gradient_bytes, gradient_stride = b.key_value_gradient_stride;
    ulong const gradient_row_bytes = a.kv_heads * a.depth * sizeof(float);
    device uint const *key_offsets = (device uint const *)(packed + 64 + segments * 8);
    threadgroup float *gradient_tile = (threadgroup float *)stage;
    for (ulong task = a.tasks_begin + group.x; task < tasks_end; task += groups.x) {
        nk_attention_work_metal_t work;
        if (!nk_attention_backward_work_metal_<nk::bf16_t>(packed, query_offsets, a, segments, task, work)) continue;
        ulong const key_first = key_offsets[task / a.kv_heads], key_value_head = task % a.kv_heads;
        if (!work.length || gradient_row_bytes > gradient_bytes ||
            (gradient_stride && key_first + work.length - 1 > (gradient_bytes - gradient_row_bytes) / gradient_stride))
            continue;
        for (ulong tile = ulong(group.y) * 32; tile < work.length; tile += ulong(groups.y) * 32) {
            ulong row_begin, row_end;
            nk_attention_panel_rows_metal_(a, work, tile, min(tile + 32, work.length), row_begin, row_end);
            for (ulong slice = 0; slice < a.depth; slice += 128) {
                typename operations_t::gradient_t key_sums, value_sums;
                operations_t::zero(key_sums), operations_t::zero(value_sums);
                for (ulong first = row_begin * group_size; first < row_end * group_size; first += 32) {
                    nk_attention_backward_panel_metal_(output, output_gradient, log_sum_exp, work, a, key_value_head,
                                                       first, panel, thread_index);
                    operations_t::scores(queries, output_gradient, work, a, key_value_head, first, tile, stage, scores,
                                         weight_gradients, thread_index, warp);
                    nk_attention_backward_weights_metal_<nk_attention_backward_keys_metal_k>(
                        scores, weight_gradients, weights, score_gradients, panel, b, tile, thread_index);
                    nk_attention_backward_stage_rows_metal_(queries, output_gradient, work, a, key_value_head, first,
                                                            slice, 128, stage, stage + 4096, thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    operations_t::accumulate(value_sums, weights, stage + 4096, warp);
                    operations_t::accumulate(key_sums, score_gradients, stage, warp);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                ulong const gradient_offset = (key_first + tile) * gradient_stride +
                                              key_value_head * a.depth * sizeof(float);
                operations_t::store(value_sums, gradient_tile, thread_index, warp);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                nk_attention_backward_store_keys_metal_(gradient_tile, value_gradient + gradient_offset,
                                                        gradient_stride, work.length - tile, slice, a.depth,
                                                        thread_index);
                operations_t::store(key_sums, gradient_tile, thread_index, warp);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                nk_attention_backward_store_keys_metal_(gradient_tile, key_gradient + gradient_offset, gradient_stride,
                                                        work.length - tile, slice, a.depth, thread_index);
            }
        }
    }
}

/**
 *  @brief The query gradients @c apple9 and @c apple10 share: per tile of 32 folded rows, over the
 *      keys they see 32 at a time, S and dP as for the key gradients, then dQ += dS · K.
 *
 *  Gradients sum 128 columns at a time in registers, deeper heads recomputing S and dP per slice.
 */
template <typename operations_type_>
inline void nk_attention_backward_queries_matrix_metal_(
    device uchar const *queries, device uchar const *packed, device uchar const *output,
    device uint const *query_offsets, constant nk_attention_backward_arguments_metal_t &b,
    device float const *log_sum_exp, device uchar const *output_gradient, device uchar *query_gradient, uint2 group,
    uint2 groups, uint thread_index, uint warp, threadgroup bfloat *stage, threadgroup float *scores,
    threadgroup float *weight_gradients, threadgroup bfloat *weights, threadgroup bfloat *score_gradients,
    threadgroup nk_attention_backward_panel_metal_t &panel) {
    using operations_t = operations_type_;
    constant nk_attention_arguments_metal_t &a = b.attention;
    ulong const segments = nk_attention_segments_metal_(packed, a), tasks_end = min(a.tasks_end, segments * a.kv_heads);
    ulong const group_size = a.heads / a.kv_heads;
    threadgroup float *gradient_tile = (threadgroup float *)stage;
    for (ulong task = a.tasks_begin + group.x; task < tasks_end; task += groups.x) {
        nk_attention_work_metal_t work;
        if (!nk_attention_backward_work_metal_<nk::bf16_t>(packed, query_offsets, a, segments, task, work)) continue;
        ulong const entries = work.queries * group_size, key_value_head = task % a.kv_heads;
        for (ulong tile = ulong(group.y) * 32; tile < entries; tile += ulong(groups.y) * 32) {
            ulong key_begin, key_end;
            nk_attention_tile_keys_metal_(a, work.first_position + long(tile / group_size),
                                          work.first_position + long((min(tile + 32, entries) - 1) / group_size),
                                          work.length, key_begin, key_end);
            nk_attention_backward_panel_metal_(output, output_gradient, log_sum_exp, work, a, key_value_head, tile,
                                               panel, thread_index);
            for (ulong slice = 0; slice < a.depth; slice += 128) {
                typename operations_t::gradient_t query_sums;
                operations_t::zero(query_sums);
                for (ulong first = key_begin; first < key_end; first += 32) {
                    operations_t::scores(queries, output_gradient, work, a, key_value_head, tile, first, stage, scores,
                                         weight_gradients, thread_index, warp);
                    nk_attention_backward_weights_metal_<nk_attention_backward_queries_metal_k>(
                        scores, weight_gradients, weights, score_gradients, panel, b, first, thread_index);
                    nk_attention_backward_stage_plane_metal_(work.keys, work, a, first, slice, 128, stage,
                                                             thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    operations_t::accumulate(query_sums, score_gradients, stage, warp);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                operations_t::store(query_sums, gradient_tile, thread_index, warp);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint cell = thread_index; cell < 32 * 128; cell += 128) {
                    ulong const entry = tile + cell / 128, channel = slice + cell % 128;
                    if (entry >= entries || channel >= a.depth) continue;
                    ulong const token = work.query_first + entry / group_size;
                    ulong const head = key_value_head * group_size + entry % group_size;
                    nk::f32_t::store(query_gradient + token * b.query_gradient_stride + head * a.depth * sizeof(float),
                                     uint(channel), gradient_tile[cell]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

kernel void nk_attention_backward_keys_bf16_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar const *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_backward_arguments_metal_t &b [[buffer(4)]], device float const *log_sum_exp [[buffer(5)]],
    device uchar const *output_gradient [[buffer(6)]], device uchar *key_gradient [[buffer(8)]],
    device uchar *value_gradient [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]], uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float queries_tile[8 * 256], gradients_tile[8 * 256];
    threadgroup nk_attention_backward_rows_metal_t rows;
    if (b.attention.depth <= 256)
        nk_attention_backward_keys_bf16_metal_<nk_attention_accumulate_registers_metal_k>(
            queries, packed, output, query_offsets, b, log_sum_exp, output_gradient, key_gradient, value_gradient,
            group, groups, warp, lane, queries_tile, gradients_tile, rows);
    else
        nk_attention_backward_keys_bf16_metal_<nk_attention_accumulate_device_metal_k>(
            queries, packed, output, query_offsets, b, log_sum_exp, output_gradient, key_gradient, value_gradient,
            group, groups, warp, lane, queries_tile, gradients_tile, rows);
}

kernel void nk_attention_backward_queries_bf16_metal_kernel_(
    device uchar const *queries [[buffer(0)]], device uchar const *packed [[buffer(1)]],
    device uchar const *output [[buffer(2)]], device uint const *query_offsets [[buffer(3)]],
    constant nk_attention_backward_arguments_metal_t &b [[buffer(4)]], device float const *log_sum_exp [[buffer(5)]],
    device uchar const *output_gradient [[buffer(6)]], device uchar *query_gradient [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint warp [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float keys_tile[8 * 256], values_tile[8 * 256];
    if (b.attention.depth <= 256)
        nk_attention_backward_queries_bf16_metal_<nk_attention_accumulate_registers_metal_k>(
            queries, packed, output, query_offsets, b, log_sum_exp, output_gradient, query_gradient, group, groups,
            warp, lane, keys_tile, values_tile);
    else
        nk_attention_backward_queries_bf16_metal_<nk_attention_accumulate_device_metal_k>(
            queries, packed, output, query_offsets, b, log_sum_exp, output_gradient, query_gradient, group, groups,
            warp, lane, keys_tile, values_tile);
}
