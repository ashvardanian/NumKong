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
 */

/** The launch record of a RoPE kernel, laid out as @c nk_attention_rope_arguments_metal_t. */
struct nk_attention_rope_arguments_metal_t {
    ulong rows, heads, depth, x_stride, y_stride;
};

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
};

/** The launch record of the attention kernels, laid out as @c nk_attention_arguments_metal_t. */
struct nk_attention_arguments_metal_t {
    ulong heads, kv_heads, depth, query_stride, output_stride, keys_before, keys_after, tasks_begin, tasks_end;
    ulong query_bytes, packed_bytes, output_bytes, offsets_bytes, log_sum_exp_bytes;
    ulong capability;
    float scale2;
};

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
 *  the segments 32 at a time. */
kernel void nk_attention_directory_metal_kernel_(device uint const *key_offsets [[buffer(2)]],
                                                 device uint const *key_lengths [[buffer(3)]],
                                                 device uchar *packed [[buffer(4)]],
                                                 constant nk_attention_pack_arguments_metal_t &a [[buffer(5)]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    ulong const payload_offset = nk_attention_payload_offset_metal_(a.segments);
    if (payload_offset > a.packed_bytes) return;
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
        uint passed = 0;
        for (ulong other = segment_first + thread_index; other < segment; other += 256)
            passed += nk_attention_key_count_metal_(key_offsets, key_lengths, a, other);
        passed = simd_sum(passed);
        if (simd_is_first()) partials[simdgroup] = passed;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint other = 0; other < 8; ++other) positions_before += partials[other];
        threadgroup_barrier(mem_flags::mem_threadgroup);
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

/** The dot product of @p query and @p key, shared by the 32 lanes of a simdgroup and returned in
 *  each: exact in I32 for integer codes, like the serial kernel's, and in F32 for floats. */
template <typename element_type_>
inline float nk_attention_score_metal_(device uchar const *query, device uchar const *key, uint depth, uint lane) {
    typename element_type_::dot_result_t sum = 0;
    for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
        sum += element_type_::load(query, channel) * element_type_::load(key, channel);
    return float(simd_sum(sum));
}

/** One threadgroup's share of a task window: the query rows of one segment and head that the window
 *  holds, and that head's K and V planes. */
struct nk_attention_work_metal_t {
    ulong query_first, queries, length, row_begin, row_end;
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

/** Fills @p work for @p pair, segment × heads + head, returning false when the window holds none of
 *  its rows or its planes run past the pack. */
template <typename element_type_>
inline bool nk_attention_work_metal_(device uchar const *packed, device uint const *query_offsets,
                                     constant nk_attention_arguments_metal_t &a, ulong segments, ulong pair,
                                     thread nk_attention_work_metal_t &work) {
    ulong const segment = pair / a.heads, head = pair % a.heads;
    ulong const tasks_begin = max(a.tasks_begin, ulong(query_offsets[0]) * a.heads);
    ulong const tasks_end = min(a.tasks_end, ulong(query_offsets[segments]) * a.heads);
    ulong const query_first = query_offsets[segment], query_end = query_offsets[segment + 1];
    if (tasks_begin >= tasks_end || query_first > query_end) return false;
    ulong const token_first = (tasks_begin + a.heads - 1 - head) / a.heads;
    ulong const token_end = min((tasks_end + a.heads - 1 - head) / a.heads, query_end);
    work.row_begin = token_first > query_first ? token_first - query_first : 0;
    work.row_end = token_end > query_first ? token_end - query_first : 0;
    if (work.row_begin >= work.row_end) return false;

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
    work.query_first = query_first, work.queries = query_end - query_first, work.length = length;
    work.first_position = long(length) - long(work.queries);
    work.keys = packed + payload_offset + payload_offsets[segment] + head / (a.heads / a.kv_heads) * plane_bytes;
    work.values = work.keys + a.kv_heads * plane_bytes;
    return true;
}

/** Whether query token @p row has a whole query row and output row inside their buffers. */
template <typename element_type_>
inline bool nk_attention_row_valid_metal_(constant nk_attention_arguments_metal_t &a, ulong row) {
    ulong const query_bytes = a.heads * a.depth * sizeof(typename element_type_::raw_t),
                output_bytes = a.heads * a.depth * 4;
    return query_bytes <= a.query_bytes && output_bytes <= a.output_bytes &&
           (!a.query_stride || row <= (a.query_bytes - query_bytes) / a.query_stride) &&
           (!a.output_stride || row <= (a.output_bytes - output_bytes) / a.output_stride);
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
 *  so their partitions write maxima in one pass and values in another.
 */
template <typename element_type_, uint subgroup_count_,
          nk_attention_phase_metal_t phase_ = nk_attention_complete_metal_k>
inline void nk_attention_rows_metal_(device uchar const *queries, device uchar *output, device float *log_sum_exp,
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
        if (!nk_attention_row_valid_metal_<element_type_>(a, row)) continue;
        device uchar const *query = queries + row * a.query_stride + head * row_bytes;
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
                if (channel < depth) query_values[index] = float(element_type_::load(query, channel));
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
                                 a.scale2);
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
                                    a.scale2;
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
                    float const inverse = total > 0 ? precise::divide(1.0f, total) : 0;
                    device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
                    for (uint channel = lane; channel < depth; channel += 32)
                        destination[channel] = ((partials[channel] + partials[256 + channel]) +
                                                (partials[512 + channel] + partials[768 + channel])) *
                                               inverse;
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
            float const inverse = sum > 0 ? precise::divide(1.0f, sum) : 0;
            device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
#pragma unroll
            for (uint index = 0; index < 8; ++index) {
                uint const channel = lane + index * 32;
                if (channel < depth) destination[channel] = values[index] * inverse;
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
template <typename element_type_, nk_attention_phase_metal_t phase_>
inline void nk_attention_split_metal_(device uchar const *queries, device uchar const *packed,
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
 *  row and log-sum-exp; U8 partitions already share the row's maximum. */
template <typename element_type_>
inline void nk_attention_merge_metal_(device uchar const *packed, device uchar *output,
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
            if (!nk_attention_row_valid_metal_<element_type_>(a, row)) continue;
            device float const *parts = split + (row * a.heads + head) * partitions * (a.depth + 2);
            float const local_maximum = lane < partitions ? parts[lane * (a.depth + 2)] : -INFINITY;
            float const local_sum = lane < partitions ? parts[lane * (a.depth + 2) + 1] : 0;
            float const maximum = simd_max(local_maximum);
            float const correction = quantized || local_maximum == maximum ? 1
                                     : local_sum > 0                       ? nk_exp2_metal_(local_maximum - maximum)
                                                                           : 0;
            float const sum = simd_sum(local_sum * correction);
            float const inverse = sum > 0 ? precise::divide(1.0f, sum) : 0;
            device float *destination = (device float *)(output + row * a.output_stride) + head * a.depth;
            for (uint first = 0; first < a.depth; first += 32) {
                uint const channel = first + lane;
                float value = 0;
                for (uint part = 0; part < partitions; ++part) {
                    float const factor = simd_shuffle(correction, part);
                    if (channel < a.depth) value = fma(parts[part * (a.depth + 2) + 2 + channel], factor, value);
                }
                if (channel < a.depth) destination[channel] = value * inverse;
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
template <typename element_type_>
inline void nk_attention_fallback_metal_(device uchar const *queries, device uchar const *packed, device uchar *output,
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
            if (!nk_attention_row_valid_metal_<element_type_>(a, row)) continue;
            device uchar const *query = queries + row * a.query_stride + head * row_bytes;
            device float *result = (device float *)(output + row * a.output_stride) + head * a.depth;
            ulong begin, end;
            nk_attention_row_range_metal_(a, work.first_position + long(local), work.length, begin, end);
            float maximum = -INFINITY;
            for (ulong position = begin; position < end; ++position)
                maximum = max(maximum, nk_attention_score_metal_<element_type_>(query, work.keys + position * row_bytes,
                                                                                depth, lane) *
                                           a.scale2);
            for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                result[channel] = 0;
            float weights_sum = 0;
            for (ulong position = begin; position < end; ++position) {
                float const score = nk_attention_score_metal_<element_type_>(query, work.keys + position * row_bytes,
                                                                             depth, lane);
                float weight = nk_exp2_metal_(score * a.scale2 - maximum);
                if (quantized) weight = floor(weight * 255.0f + 0.5f);
                weights_sum += weight;
                device uchar const *value_row = work.values + position * row_bytes;
                for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                    result[channel] = fma(weight, float(element_type_::load(value_row, channel)), result[channel]);
            }
            float const inverse = weights_sum > 0 ? precise::divide(1.0f, weights_sum) : 0;
            for (uint channel = lane; channel < depth; channel = depth - channel <= 32 ? depth : channel + 32)
                result[channel] *= inverse;
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

kernel void nk_attention_merge_bf16_metal_kernel_(
    device uchar const *packed [[buffer(1)]], device uchar *output [[buffer(2)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *log_sum_exp [[buffer(5)]], device float const *split [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::bf16_t>(packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

kernel void nk_attention_merge_f16_metal_kernel_(
    device uchar const *packed [[buffer(1)]], device uchar *output [[buffer(2)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *log_sum_exp [[buffer(5)]], device float const *split [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::f16_t>(packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

kernel void nk_attention_merge_e4m3_metal_kernel_(
    device uchar const *packed [[buffer(1)]], device uchar *output [[buffer(2)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *log_sum_exp [[buffer(5)]], device float const *split [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::e4m3_t>(packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

kernel void nk_attention_merge_i8_metal_kernel_(
    device uchar const *packed [[buffer(1)]], device uchar *output [[buffer(2)]],
    device uint const *query_offsets [[buffer(3)]], constant nk_attention_arguments_metal_t &a [[buffer(4)]],
    device float *log_sum_exp [[buffer(5)]], device float const *split [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    nk_attention_merge_metal_<nk::i8_t>(packed, output, query_offsets, a, log_sum_exp, split, group, groups, lane);
}

/** Stages a 32-row tile of the work's queries and a 64-key panel of its keys, @p step_ channels
 *  from @p depth_first, into @p left and @p right, zeros past every edge. */
template <typename stage_type_, uint step_, typename element_type_>
inline void nk_attention_stage_qk_metal_(device uchar const *queries, thread nk_attention_work_metal_t const &work,
                                         constant nk_attention_arguments_metal_t &a, ulong head, ulong row_first,
                                         ulong key_first, ulong depth_first, threadgroup stage_type_ *left,
                                         threadgroup stage_type_ *right, uint thread_index) {
    constexpr uint step = step_;
    ulong const row_bytes = a.depth * sizeof(typename element_type_::raw_t);
    for (uint cell = thread_index; cell < (32 + 64) * step; cell += 128) {
        uint const row = cell / step, channel = cell % step;
        float value = 0;
        if (row < 32) {
            ulong const local = row_first + row;
            if (local < work.queries && depth_first + channel < a.depth &&
                nk_attention_row_valid_metal_<element_type_>(a, work.query_first + local))
                value = float(element_type_::load(
                    queries + (work.query_first + local) * a.query_stride + head * row_bytes, depth_first + channel));
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

/** Folds a panel of scores into each tile row's running maximum, recording its correction, over
 *  the keys the band shows to the row. */
inline void nk_attention_maximum_metal_(threadgroup float const *scores, threadgroup float *maximum,
                                        threadgroup float *corrections, thread nk_attention_work_metal_t const &work,
                                        constant nk_attention_arguments_metal_t &a, ulong row_first, ulong key_first,
                                        uint thread_index) {
    uint const row = thread_index / 4, lane = thread_index % 4;
    ulong begin, end;
    nk_attention_row_range_metal_(a, work.first_position + long(row_first + row), work.length, begin, end);
    float value = -INFINITY;
    for (uint column = lane; column < 64; column += 4)
        if (row_first + row < work.queries && key_first + column >= begin && key_first + column < end)
            value = max(value, scores[row * 64 + column] * a.scale2);
    value = max(value, simd_shuffle_xor(value, 1));
    value = max(value, simd_shuffle_xor(value, 2));
    if (!lane) {
        float const previous = maximum[row], next = max(previous, value);
        corrections[row] = previous == next ? 1 : nk_exp2_metal_(previous - next);
        maximum[row] = next;
    }
}

/** Turns a panel of scores into staged weights, U8 ones offset by −128 into @c int8_t, and folds
 *  their sum into each tile row's running sum. */
template <typename stage_type_>
inline void nk_attention_weights_metal_(threadgroup float const *scores, threadgroup stage_type_ *weights,
                                        threadgroup float const *maximum, threadgroup float const *corrections,
                                        threadgroup float *sums, thread nk_attention_work_metal_t const &work,
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
                                      ? nk_exp2_metal_(scores[row * 64 + column] * a.scale2 - maximum[row])
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
 */
template <typename stage_type_, typename operations_type_, typename element_type_,
          nk_attention_accumulation_metal_t accumulation_>
inline void nk_attention_matrix_metal_(device uchar const *queries, device uchar const *packed, device uchar *output,
                                       device uint const *query_offsets, device float *log_sum_exp,
                                       constant nk_attention_arguments_metal_t &a, uint2 group, uint2 groups,
                                       uint thread_index, uint warp, threadgroup stage_type_ *left,
                                       threadgroup stage_type_ *right, threadgroup float *scores,
                                       threadgroup float *maximum, threadgroup float *corrections,
                                       threadgroup float *sums) {
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
            if (thread_index < 32)
                maximum[thread_index] = -INFINITY, sums[thread_index] = 0, corrections[thread_index] = 1;
            float accumulation[resident ? 32 : 1] = {};
            if constexpr (!resident)
                for (ulong cell = thread_index; cell < 32 * a.depth; cell += 128) {
                    ulong const local = row_first + cell / a.depth, channel = cell % a.depth,
                                row = work.query_first + local;
                    if (local >= work.row_begin && local < work.row_end &&
                        nk_attention_row_valid_metal_<element_type_>(a, row))
                        ((device float *)(output + row * a.output_stride))[head * a.depth + channel] = 0;
                }
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            if constexpr (sizeof(stage_t) == 1) {
                for (ulong key_first = key_begin; key_first < key_end; key_first += 64) {
                    operations_t::template qk<stage_t, element_type_>(queries, work, a, head, row_first, key_first,
                                                                      left, right, scores, thread_index, warp);
                    nk_attention_maximum_metal_(scores, maximum, corrections, work, a, row_first, key_first,
                                                thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (thread_index < 32) corrections[thread_index] = 1;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (ulong key_first = key_begin; key_first < key_end; key_first += 64) {
                operations_t::template qk<stage_t, element_type_>(queries, work, a, head, row_first, key_first, left,
                                                                  right, scores, thread_index, warp);
                if constexpr (sizeof(stage_t) != 1) {
                    nk_attention_maximum_metal_(scores, maximum, corrections, work, a, row_first, key_first,
                                                thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                nk_attention_weights_metal_(scores, left, maximum, corrections, sums, work, a, row_first, key_first,
                                            thread_index);
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
                        else if (local >= work.row_begin && local < work.row_end && channel < a.depth &&
                                 nk_attention_row_valid_metal_<element_type_>(a, row)) {
                            device float &value =
                                ((device float *)(output + row * a.output_stride))[head * a.depth + channel];
                            value = value * corrections[cell / 32] + scores[cell];
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
                    if (local < work.row_begin || local >= work.row_end || channel >= a.depth ||
                        !nk_attention_row_valid_metal_<element_type_>(a, row))
                        continue;
                    ((device float *)(output + row * a.output_stride))[head * a.depth + channel] =
                        accumulation[slot] * (sum > 0 ? precise::divide(1.0f, sum) : 0);
                }
            }
            else
                for (ulong cell = thread_index; cell < 32 * a.depth; cell += 128) {
                    ulong const local = row_first + cell / a.depth, channel = cell % a.depth,
                                row = work.query_first + local;
                    float const sum = sums[cell / a.depth];
                    if (local >= work.row_begin && local < work.row_end &&
                        nk_attention_row_valid_metal_<element_type_>(a, row))
                        ((device float *)(output + row * a.output_stride))[head * a.depth + channel] *=
                            sum > 0 ? precise::divide(1.0f, sum) : 0;
                }
            if (thread_index < 32 && row_first + thread_index >= work.row_begin &&
                row_first + thread_index < work.row_end)
                nk_attention_store_log_sum_exp_metal_(
                    log_sum_exp, a, work.query_first + row_first + thread_index, head,
                    nk_attention_log_sum_exp_metal_(maximum[thread_index], sums[thread_index],
                                                    nk_attention_unit_metal_<element_type_>()));
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
    }
}
