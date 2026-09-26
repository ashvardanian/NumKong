/**
 *  @file javascript/numkong.c
 *  @author Ash Vardanian
 *  @date October 18, 2023
 *  @brief JavaScript bindings for NumKong.
 *
 *  @see NodeJS docs: https://nodejs.org/api/n-api.html
 */

#include <string.h> // `strlen` function

#include <node_api.h> // `napi_*` functions — N-API v6+ for BigInt (Node ≥ 10.20)

#include <numkong/numkong.h> // `nk_*` functions — must be first to bring `_GNU_SOURCE`

#include "parallel.h" // `nk_parallel_for_tiles`, tile sizes

#pragma region Helpers

/** Parses a dtype string, e.g. "f32", "f16", "bf16", into an nk_dtype_t enum value. */
static nk_dtype_t parse_dtype_string(char const *str) { return nk_dtype_named(str, strlen(str)); }

/** Validates that the N-API TypedArray type is compatible with the claimed dtype. */
static int is_compatible_napi_type(napi_typedarray_type napi_type, nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return napi_type == napi_float64_array;
    case nk_f32_k: return napi_type == napi_float32_array;
    case nk_f16_k:
    case nk_bf16_k: return napi_type == napi_uint16_array;
    case nk_e4m3_k:
    case nk_e5m2_k:
    case nk_e2m3_k:
    case nk_e3m2_k:
    case nk_u8_k:
    case nk_u1_k: return napi_type == napi_uint8_array;
    case nk_i8_k: return napi_type == napi_int8_array;
    case nk_i16_k: return napi_type == napi_int16_array;
    case nk_u16_k: return napi_type == napi_uint16_array;
    case nk_i32_k: return napi_type == napi_int32_array;
    case nk_u32_k: return napi_type == napi_uint32_array;
    case nk_i64_k: return napi_type == napi_bigint64_array;
    case nk_u64_k: return napi_type == napi_biguint64_array;
    default: return 0;
    }
}

/**
 *  @brief Converts an nk_scalar_buffer_t result to a JavaScript number.
 *  @param[in] env N-API environment.
 *  @param[in] result The scalar buffer containing the result.
 *  @param[in] out_dtype The dtype of the value stored in the buffer.
 *  @return napi_value containing the result as a JavaScript Number, or NULL on error.
 */
static napi_value nk_scalar_buffer_to_js_number(napi_env env, nk_scalar_buffer_t const *result, nk_dtype_t out_dtype) {
    // i64/u64 must return BigInt since they may exceed Number.MAX_SAFE_INTEGER
    if (out_dtype == nk_i64_k) {
        napi_value js_result;
        if (napi_create_bigint_int64(env, result->i64, &js_result) != napi_ok) return NULL;
        return js_result;
    }
    if (out_dtype == nk_u64_k) {
        napi_value js_result;
        if (napi_create_bigint_uint64(env, result->u64, &js_result) != napi_ok) return NULL;
        return js_result;
    }
    nk_f64c_t result_c;
    nk_scalar_buffer_to_f64c(result, out_dtype, &result_c);
    double result_f64 = result_c.real;
    napi_value js_result;
    if (napi_create_double(env, result_f64, &js_result) != napi_ok) return NULL;
    return js_result;
}

/** Returns the byte width for a given dtype. */
static inline size_t dtype_byte_width(nk_dtype_t dtype) { return nk_dtype_bits(dtype) / NUMKONG_BITS_PER_BYTE; }

/** Returns the N-API typed array type for a given output dtype. */
static inline napi_typedarray_type napi_type_for_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return napi_float64_array;
    case nk_f32_k: return napi_float32_array;
    case nk_i32_k: return napi_int32_array;
    case nk_u32_k: return napi_uint32_array;
    default: return napi_float32_array;
    }
}

#pragma endregion Helpers

#pragma region Distance API

/** Core distance computation — resolves dtype, dispatches kernel, converts result. */
static napi_value dense(napi_env env, napi_callback_info info, nk_kernel_kind_t kernel_kind, nk_dtype_t dtype) {
    size_t argc = 3;
    napi_value args[3];
    napi_status status;

    // Get callback info and ensure the argument count is correct (2 or 3 args)
    status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (status != napi_ok || argc < 2 || argc > 3) {
        napi_throw_error(env, NULL, "Expected 2 or 3 arguments: (a, b[, dtype])");
        return NULL;
    }

    // Obtain the typed arrays from the arguments
    void *data_a, *data_b;
    size_t length_a, length_b;
    napi_typedarray_type type_a, type_b;
    napi_status status_a, status_b;
    status_a = napi_get_typedarray_info(env, args[0], &type_a, &length_a, &data_a, NULL, NULL);
    status_b = napi_get_typedarray_info(env, args[1], &type_b, &length_b, &data_b, NULL, NULL);
    if (status_a != napi_ok || status_b != napi_ok || type_a != type_b || length_a != length_b) {
        napi_throw_error(env, NULL, "Both arguments must be typed arrays of matching types and dimensionality");
        return NULL;
    }

    // When dtype is unknown, try to resolve from optional 3rd argument or auto-detect
    if (dtype == nk_dtype_unknown_k) {
        if (argc == 3) {
            // Parse explicit dtype string from 3rd argument
            char dtype_str[16];
            size_t str_len;
            if (napi_get_value_string_utf8(env, args[2], dtype_str, sizeof(dtype_str), &str_len) != napi_ok) {
                napi_throw_error(env, NULL, "Third argument must be a dtype string");
                return NULL;
            }
            dtype = parse_dtype_string(dtype_str);
            if (dtype == nk_dtype_unknown_k) {
                napi_throw_error(env, NULL, "Unsupported dtype string");
                return NULL;
            }
            if (!is_compatible_napi_type(type_a, dtype)) {
                napi_throw_error(env, NULL, "TypedArray type is not compatible with the specified dtype");
                return NULL;
            }
        }
        else {
            // Auto-detect from N-API TypedArray type (backward-compatible 4-type whitelist)
            if (type_a != napi_float64_array && type_a != napi_float32_array && type_a != napi_int8_array &&
                type_a != napi_uint8_array) {
                napi_throw_error( //
                    env, NULL,
                    "Only f64, f32, i8, u8 arrays are auto-detected; " //
                    "pass dtype string as 3rd argument for other types");
                return NULL;
            }
            switch (type_a) {
            case napi_float64_array: dtype = nk_f64_k; break;
            case napi_float32_array: dtype = nk_f32_k; break;
            case napi_int8_array: dtype = nk_i8_k; break;
            case napi_uint8_array: dtype = nk_u8_k; break;
            default: break;
            }
        }
    }

    nk_metric_dense_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_cpu_find_kernel_punned(kernel_kind, dtype, (nk_kernel_punned_t *)&metric, &capability);
    if (!metric || !capability) {
        napi_throw_error(env, NULL, "Unsupported dtype for given metric");
        return NULL;
    }

    nk_dtype_t out_dtype = nk_kernel_output_dtype(kernel_kind, dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported output dtype for given metric/input combination");
        return NULL;
    }

    // Typed-array lengths count storage values; kernels take logical dimensions.
    size_t dimensions = length_a * nk_dimensions_per_value(dtype);

    nk_scalar_buffer_t result;
    metric(data_a, data_b, dimensions, &result);

    return nk_scalar_buffer_to_js_number(env, &result, out_dtype);
}

/** N-API entry for inner product, dot. */
napi_value api_ip(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_dot_k, nk_dtype_unknown_k);
}

/** N-API entry for angular distance. */
napi_value api_angular(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_angular_k, nk_dtype_unknown_k);
}

/** N-API entry for squared Euclidean distance. */
napi_value api_sqeuclidean(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_sqeuclidean_k, nk_dtype_unknown_k);
}

/** N-API entry for Euclidean distance. */
napi_value api_euclidean(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_euclidean_k, nk_dtype_unknown_k);
}

/** N-API entry for Kullback-Leibler divergence. */
napi_value api_kld(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_kld_k, nk_dtype_unknown_k);
}

/** N-API entry for Jensen-Shannon distance. */
napi_value api_jsd(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_jsd_k, nk_dtype_unknown_k);
}

/** N-API entry for Hamming distance. */
napi_value api_hamming(napi_env env, napi_callback_info info) { return dense(env, info, nk_kernel_hamming_k, nk_u1_k); }

/** N-API entry for Jaccard distance. */
napi_value api_jaccard(napi_env env, napi_callback_info info) { return dense(env, info, nk_kernel_jaccard_k, nk_u1_k); }

#pragma endregion Distance API

#pragma region Capabilities API

/**
 *  @brief Returns the CPU capabilities this machine executes.
 *  @return BigInt bitmask of nk_capability_t flags.
 *
 *  Describes the machine only. A capability reported here whose kernels were not compiled in will
 *  never run — see @b api_capabilities_enabled().
 */
napi_value api_capabilities_detected(napi_env env, napi_callback_info info) {
    napi_value result;
    napi_create_bigint_uint64(env, (uint64_t)nk_cpu_capabilities_detected(), &result);
    return result;
}

/**
 *  @brief Returns the CPU capabilities whose kernels were compiled into this binary.
 *  @return BigInt bitmask of nk_capability_t flags.
 */
napi_value api_capabilities_compiled(napi_env env, napi_callback_info info) {
    napi_value result;
    napi_create_bigint_uint64(env, (uint64_t)nk_cpu_capabilities_compiled(), &result);
    return result;
}

/**
 *  @brief Returns the CPU capabilities dispatch uses.
 *  @return BigInt bitmask of nk_capability_t flags, detected and compiled unless narrowed.
 *
 *  This is the mask every kernel lookup walks.
 */
napi_value api_capabilities_enabled(napi_env env, napi_callback_info info) {
    napi_value result;
    napi_create_bigint_uint64(env, (uint64_t)nk_cpu_capabilities_enabled(), &result);
    return result;
}

/**
 *  @brief Makes the BigInt mask argument the enabled set, clamped to detected and compiled.
 *  @return BigInt bitmask of the enabled set that took effect, always with the serial fallback.
 */
napi_value api_capabilities_enable(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc < 1) {
        napi_throw_error(env, NULL, "Expected 1 argument: a BigInt capability mask");
        return NULL;
    }
    uint64_t wanted;
    bool lossless;
    if (napi_get_value_bigint_uint64(env, args[0], &wanted, &lossless) != napi_ok) {
        napi_throw_error(env, NULL, "Capability mask must be a BigInt");
        return NULL;
    }
    napi_value result;
    napi_create_bigint_uint64(env, (uint64_t)nk_cpu_capabilities_enable((nk_capability_t)wanted), &result);
    return result;
}

/** Exports @c Capability, mapping every CPU tier's name to its BigInt bit, and no GPU tiers. */
static napi_status export_capability_names(napi_env env, napi_value exports) {
    napi_value names;
    napi_status status = napi_create_object(env, &names);
    if (status != napi_ok) return status;
    for (unsigned shift = 0; shift != 64; ++shift) {
        nk_capability_t const bit = (nk_capability_t)1 << shift;
        char name[NUMKONG_CAPABILITIES_NAME_CAPACITY];
        if ((bit & nk_cap_devices_k) || !nk_name_capabilities(bit, name, sizeof(name))) continue;
        napi_value value;
        if ((status = napi_create_bigint_uint64(env, (uint64_t)bit, &value)) != napi_ok ||
            (status = napi_set_named_property(env, names, name, value)) != napi_ok)
            return status;
    }
    return napi_set_named_property(env, exports, "Capability", names);
}

#pragma endregion Capabilities API

#pragma region Cast API

/** Converts a single value from a narrow type to f32. Reads uint32 bits, returns double. */
static napi_value cast_to_f32(napi_env env, napi_callback_info info, nk_dtype_t src_dtype) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (argc != 1) {
        napi_throw_error(env, NULL, "Expected 1 argument");
        return NULL;
    }

    uint32_t bits;
    if (napi_get_value_uint32(env, args[0], &bits) != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a number");
        return NULL;
    }

    nk_f32_t f32_val;
    nk_cast(&bits, src_dtype, 1, &f32_val, nk_f32_k);

    napi_value result;
    napi_create_double(env, (double)f32_val, &result);
    return result;
}

/** Converts a single f32 value to a narrow type. Reads double, returns uint32 bits. */
static napi_value cast_from_f32(napi_env env, napi_callback_info info, nk_dtype_t dst_dtype) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (argc != 1) {
        napi_throw_error(env, NULL, "Expected 1 argument");
        return NULL;
    }

    double f32_dbl;
    if (napi_get_value_double(env, args[0], &f32_dbl) != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a number");
        return NULL;
    }

    nk_f32_t f32_val = (nk_f32_t)f32_dbl;
    uint32_t bits = 0;
    nk_cast(&f32_val, nk_f32_k, 1, &bits, dst_dtype);

    napi_value result;
    napi_create_uint32(env, bits, &result);
    return result;
}

/** N-API entry for scalar f16-to-f32 conversion. */
napi_value api_cast_f16_to_f32(napi_env env, napi_callback_info info) { return cast_to_f32(env, info, nk_f16_k); }

/** N-API entry for scalar f32-to-f16 conversion. */
napi_value api_cast_f32_to_f16(napi_env env, napi_callback_info info) { return cast_from_f32(env, info, nk_f16_k); }

/** N-API entry for scalar bf16-to-f32 conversion. */
napi_value api_cast_bf16_to_f32(napi_env env, napi_callback_info info) { return cast_to_f32(env, info, nk_bf16_k); }

/** N-API entry for scalar f32-to-bf16 conversion. */
napi_value api_cast_f32_to_bf16(napi_env env, napi_callback_info info) { return cast_from_f32(env, info, nk_bf16_k); }

/** N-API entry for scalar e4m3-to-f32 conversion. */
napi_value api_cast_e4m3_to_f32(napi_env env, napi_callback_info info) { return cast_to_f32(env, info, nk_e4m3_k); }

/** N-API entry for scalar f32-to-e4m3 conversion. */
napi_value api_cast_f32_to_e4m3(napi_env env, napi_callback_info info) { return cast_from_f32(env, info, nk_e4m3_k); }

/** N-API entry for scalar e5m2-to-f32 conversion. */
napi_value api_cast_e5m2_to_f32(napi_env env, napi_callback_info info) { return cast_to_f32(env, info, nk_e5m2_k); }

/** N-API entry for scalar f32-to-e5m2 conversion. */
napi_value api_cast_f32_to_e5m2(napi_env env, napi_callback_info info) { return cast_from_f32(env, info, nk_e5m2_k); }

/**
 *  @brief Buffer casting function using nk_cast.
 *
 *  @code{.ts}
 *  (source: TypedArray, sourceType: string, destination: TypedArray, destinationType: string)
 *  @endcode
 *
 *  @param[in] env N-API environment.
 *  @param[in] info Callback info, the 4 positional arguments above.
 *  @return null, modifies the destination buffer in place.
 */
napi_value api_cast(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc != 4) {
        napi_throw_error(env, NULL, "cast requires 4 arguments: (src, srcType, dst, dstType)");
        return NULL;
    }

    // Get source and destination arrays
    void *src_data, *dst_data;
    size_t src_len, dst_len;
    napi_typedarray_type src_type, dst_type;

    napi_get_typedarray_info(env, args[0], &src_type, &src_len, &src_data, NULL, NULL);
    napi_get_typedarray_info(env, args[2], &dst_type, &dst_len, &dst_data, NULL, NULL);

    // Get dtype strings
    char src_dtype_str[16], dst_dtype_str[16];
    size_t str_len;
    napi_get_value_string_utf8(env, args[1], src_dtype_str, sizeof(src_dtype_str), &str_len);
    napi_get_value_string_utf8(env, args[3], dst_dtype_str, sizeof(dst_dtype_str), &str_len);

    // Map dtype strings to nk_dtype_t
    nk_dtype_t src_dtype = parse_dtype_string(src_dtype_str);
    nk_dtype_t dst_dtype = parse_dtype_string(dst_dtype_str);

    if (src_dtype == nk_dtype_unknown_k || dst_dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported dtype string");
        return NULL;
    }

    // Perform conversion using nk_cast
    nk_cast(src_data, src_dtype, src_len, dst_data, dst_dtype);

    return NULL; // Modifies dst_data in place
}

#pragma endregion Cast API

#pragma region Packed API

/** Query packed buffer byte count, dotsPackedSize(width, depth, dtype) → number. */
static napi_value api_dots_pack_size(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (argc != 3) {
        napi_throw_error(env, NULL, "dotsPackedSize requires 3 arguments: (width, depth, dtype)");
        return NULL;
    }

    uint32_t width, depth;
    napi_get_value_uint32(env, args[0], &width);
    napi_get_value_uint32(env, args[1], &depth);

    char dtype_str[16];
    size_t str_len;
    napi_get_value_string_utf8(env, args[2], dtype_str, sizeof(dtype_str), &str_len);
    nk_dtype_t dtype = parse_dtype_string(dtype_str);
    if (dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported dtype string");
        return NULL;
    }

    nk_dots_pack_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_cpu_find_kernel_punned(nk_kernel_dots_pack_size_k, dtype, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn) {
        napi_throw_error(env, NULL, "dots_pack_size not available for this dtype");
        return NULL;
    }

    nk_size_t byte_count = size_fn((nk_size_t)width, (nk_size_t)depth);

    napi_value result;
    napi_create_double(env, (double)byte_count, &result);
    return result;
}

/** Packs matrix B, returning a packed ArrayBuffer for dotsPacked, angularsPacked, or
 *  euclideansPacked:
 *  dotsPack(data, width, depth, strideBytes, dtype) → { buffer, width, depth, byteLength }. */
static napi_value api_dots_pack(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (argc != 5) {
        napi_throw_error(env, NULL, "dotsPack requires 5 arguments: (data, width, depth, strideBytes, dtype)");
        return NULL;
    }

    void *data;
    size_t data_len;
    napi_typedarray_type arr_type;
    napi_get_typedarray_info(env, args[0], &arr_type, &data_len, &data, NULL, NULL);

    uint32_t width, depth, stride_bytes;
    napi_get_value_uint32(env, args[1], &width);
    napi_get_value_uint32(env, args[2], &depth);
    napi_get_value_uint32(env, args[3], &stride_bytes);

    char dtype_str[16];
    size_t str_len;
    napi_get_value_string_utf8(env, args[4], dtype_str, sizeof(dtype_str), &str_len);
    nk_dtype_t dtype = parse_dtype_string(dtype_str);
    if (dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported dtype string");
        return NULL;
    }

    // Get packed size
    nk_dots_pack_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_cpu_find_kernel_punned(nk_kernel_dots_pack_size_k, dtype, (nk_kernel_punned_t *)&size_fn, &cap);
    if (!size_fn) {
        napi_throw_error(env, NULL, "dots_pack_size not available for this dtype");
        return NULL;
    }
    nk_size_t packed_byte_count = size_fn((nk_size_t)width, (nk_size_t)depth);

    // Allocate V8-managed ArrayBuffer for packed data
    void *packed_data = NULL;
    napi_value arraybuffer;
    if (napi_create_arraybuffer(env, packed_byte_count, &packed_data, &arraybuffer) != napi_ok) {
        napi_throw_error(env, NULL, "Failed to allocate packed buffer");
        return NULL;
    }

    // Pack
    nk_dots_pack_punned_t pack_fn = NULL;
    cap = nk_cap_serial_k;
    nk_cpu_find_kernel_punned(nk_kernel_dots_pack_k, dtype, (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!pack_fn) {
        napi_throw_error(env, NULL, "dots_pack not available for this dtype");
        return NULL;
    }
    pack_fn(data, (nk_size_t)width, (nk_size_t)depth, (nk_size_t)stride_bytes, packed_data, (nk_size_t)0,
            (nk_size_t)width);

    // Return object { buffer, width, depth, byteLength }
    napi_value result_obj;
    napi_create_object(env, &result_obj);

    napi_value js_width, js_depth, js_byte_length;
    napi_create_uint32(env, width, &js_width);
    napi_create_uint32(env, depth, &js_depth);
    napi_create_double(env, (double)packed_byte_count, &js_byte_length);

    napi_set_named_property(env, result_obj, "buffer", arraybuffer);
    napi_set_named_property(env, result_obj, "width", js_width);
    napi_set_named_property(env, result_obj, "depth", js_depth);
    napi_set_named_property(env, result_obj, "byteLength", js_byte_length);

    return result_obj;
}

/** One tile of rows of C = A × Bᵀ with B pre-packed. */
typedef struct packed_task_t {
    nk_dots_packed_punned_t kernel;
    char const *a;
    void const *b_packed;
    char *c;
    nk_size_t rows;
    nk_size_t columns;
    nk_size_t depth;
    nk_size_t a_stride_bytes;
    nk_size_t c_stride_bytes;
} packed_task_t;

static void packed_tile_(nk_size_t tile_index, void *context) {
    packed_task_t const *task = (packed_task_t const *)context;
    nk_size_t const row = tile_index * NUMKONG_PARALLEL_PACKED_TILE;
    nk_size_t const chunk = (row + NUMKONG_PARALLEL_PACKED_TILE <= task->rows) ? NUMKONG_PARALLEL_PACKED_TILE
                                                                               : (task->rows - row);
    task->kernel(task->a + row * task->a_stride_bytes, task->b_packed, task->c + row * task->c_stride_bytes, chunk,
                 task->columns, task->depth, task->a_stride_bytes, task->c_stride_bytes);
}

/**
 *  @brief Shared dispatcher for packed operations, dots, angulars and euclideans.
 *
 *  @code{.ts}
 *  (a: TypedArray, packed: ArrayBuffer, result: TypedArray, height: number, width: number,
 *      depth: number, aStride: number, resultStride: number, dtype: string, threads?: number)
 *  @endcode
 */
static napi_value api_packed_common(napi_env env, napi_callback_info info, nk_kernel_kind_t kernel_kind) {
    size_t argc = 10;
    napi_value args[10];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (argc < 9 || argc > 10) {
        napi_throw_error(env, NULL, "Packed operation requires 9-10 arguments (last is optional threads)");
        return NULL;
    }

    // arg[0]: TypedArray a
    void *a_data;
    size_t a_len;
    napi_typedarray_type a_type;
    napi_get_typedarray_info(env, args[0], &a_type, &a_len, &a_data, NULL, NULL);

    // arg[1]: ArrayBuffer packed
    void *packed_data;
    size_t packed_len;
    napi_get_arraybuffer_info(env, args[1], &packed_data, &packed_len);

    // arg[2]: TypedArray result
    void *result_data;
    size_t result_len;
    napi_typedarray_type result_type;
    napi_get_typedarray_info(env, args[2], &result_type, &result_len, &result_data, NULL, NULL);

    // args[3..7]: height, width, depth, aStride, resultStride
    uint32_t height, width, depth, a_stride, result_stride;
    napi_get_value_uint32(env, args[3], &height);
    napi_get_value_uint32(env, args[4], &width);
    napi_get_value_uint32(env, args[5], &depth);
    napi_get_value_uint32(env, args[6], &a_stride);
    napi_get_value_uint32(env, args[7], &result_stride);

    // arg[8]: dtype string
    char dtype_str[16];
    size_t str_len;
    napi_get_value_string_utf8(env, args[8], dtype_str, sizeof(dtype_str), &str_len);
    nk_dtype_t dtype = parse_dtype_string(dtype_str);
    if (dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported dtype string");
        return NULL;
    }

    nk_dots_packed_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_cpu_find_kernel_punned(kernel_kind, dtype, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel) {
        napi_throw_error(env, NULL, "Packed kernel not available for this dtype");
        return NULL;
    }

    uint32_t threads = 1;
    if (argc == 10) napi_get_value_uint32(env, args[9], &threads);

    packed_task_t task;
    task.kernel = kernel;
    task.a = a_data;
    task.b_packed = packed_data;
    task.c = result_data;
    task.rows = height;
    task.columns = width;
    task.depth = depth;
    task.a_stride_bytes = a_stride;
    task.c_stride_bytes = result_stride;
    nk_parallel_for_tiles(nk_size_divide_round_up_(height, NUMKONG_PARALLEL_PACKED_TILE), threads, packed_tile_, &task);
    return NULL;
}

static napi_value api_dots_packed(napi_env env, napi_callback_info info) {
    return api_packed_common(env, info, nk_kernel_dots_packed_k);
}
static napi_value api_angulars_packed(napi_env env, napi_callback_info info) {
    return api_packed_common(env, info, nk_kernel_angulars_packed_k);
}
static napi_value api_euclideans_packed(napi_env env, napi_callback_info info) {
    return api_packed_common(env, info, nk_kernel_euclideans_packed_k);
}

/** One tile of rows of C = A × Aᵀ. */
typedef struct symmetric_task_t {
    nk_dots_symmetric_punned_t kernel;
    void const *vectors;
    void *result;
    nk_size_t vectors_count;
    nk_size_t depth;
    nk_size_t stride_bytes;
    nk_size_t result_stride_bytes;
    nk_size_t row_start;
    nk_size_t row_end;
} symmetric_task_t;

static void symmetric_tile_(nk_size_t tile_index, void *context) {
    symmetric_task_t const *task = (symmetric_task_t const *)context;
    nk_size_t const tile_start = task->row_start + tile_index * NUMKONG_PARALLEL_SYMMETRIC_TILE;
    nk_size_t const tile_rows = (tile_start + NUMKONG_PARALLEL_SYMMETRIC_TILE <= task->row_end)
                                    ? NUMKONG_PARALLEL_SYMMETRIC_TILE
                                    : (task->row_end - tile_start);
    task->kernel(task->vectors, task->vectors_count, task->depth, task->stride_bytes, task->result,
                 task->result_stride_bytes, tile_start, tile_rows);
}

/**
 *  @brief Shared dispatcher for symmetric operations, dots, angulars and euclideans.
 *
 *  @code{.ts}
 *  (vectors: TypedArray, result: TypedArray, nVectors: number, depth: number,
 *      vectorsStride: number, resultStride: number, rowStart: number, rowCount: number,
 *      dtype: string, threads?: number)
 *  @endcode
 */
static napi_value api_symmetric_common(napi_env env, napi_callback_info info, nk_kernel_kind_t kernel_kind) {
    size_t argc = 10;
    napi_value args[10];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (argc < 9 || argc > 10) {
        napi_throw_error(env, NULL, "Symmetric operation requires 9-10 arguments (last is optional threads)");
        return NULL;
    }

    // arg[0]: TypedArray vectors
    void *vectors_data;
    size_t vectors_len;
    napi_typedarray_type vectors_type;
    napi_get_typedarray_info(env, args[0], &vectors_type, &vectors_len, &vectors_data, NULL, NULL);

    // arg[1]: TypedArray result
    void *result_data;
    size_t result_len;
    napi_typedarray_type result_type;
    napi_get_typedarray_info(env, args[1], &result_type, &result_len, &result_data, NULL, NULL);

    // args[2..7]: nVectors, depth, vectorsStride, resultStride, rowStart, rowCount
    uint32_t n_vectors, depth, vectors_stride, result_stride, row_start, row_count;
    napi_get_value_uint32(env, args[2], &n_vectors);
    napi_get_value_uint32(env, args[3], &depth);
    napi_get_value_uint32(env, args[4], &vectors_stride);
    napi_get_value_uint32(env, args[5], &result_stride);
    napi_get_value_uint32(env, args[6], &row_start);
    napi_get_value_uint32(env, args[7], &row_count);

    // arg[8]: dtype string
    char dtype_str[16];
    size_t str_len;
    napi_get_value_string_utf8(env, args[8], dtype_str, sizeof(dtype_str), &str_len);
    nk_dtype_t dtype = parse_dtype_string(dtype_str);
    if (dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported dtype string");
        return NULL;
    }

    nk_dots_symmetric_punned_t kernel = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_cpu_find_kernel_punned(kernel_kind, dtype, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel) {
        napi_throw_error(env, NULL, "Symmetric kernel not available for this dtype");
        return NULL;
    }

    uint32_t threads = 1;
    if (argc == 10) napi_get_value_uint32(env, args[9], &threads);

    symmetric_task_t task;
    task.kernel = kernel;
    task.vectors = vectors_data;
    task.result = result_data;
    task.vectors_count = n_vectors;
    task.depth = depth;
    task.stride_bytes = vectors_stride;
    task.result_stride_bytes = result_stride;
    task.row_start = row_start;
    // Widen before the sum so two `uint32_t` row bounds cannot wrap.
    task.row_end = (nk_size_t)row_start + row_count;
    nk_parallel_for_tiles(nk_size_divide_round_up_(row_count, NUMKONG_PARALLEL_SYMMETRIC_TILE), threads,
                          symmetric_tile_, &task);

    return NULL;
}

static napi_value api_dots_symmetric(napi_env env, napi_callback_info info) {
    return api_symmetric_common(env, info, nk_kernel_dots_symmetric_k);
}
static napi_value api_angulars_symmetric(napi_env env, napi_callback_info info) {
    return api_symmetric_common(env, info, nk_kernel_angulars_symmetric_k);
}
static napi_value api_euclideans_symmetric(napi_env env, napi_callback_info info) {
    return api_symmetric_common(env, info, nk_kernel_euclideans_symmetric_k);
}

#pragma endregion Packed API

#pragma region Module Init

/** Registers a C function as a named JavaScript export. */
static napi_status export_function(napi_env env, napi_value exports, char const *name, napi_callback func) {
    napi_value fn;
    napi_status status = napi_create_function(env, name, NAPI_AUTO_LENGTH, func, NULL, &fn);
    if (status != napi_ok) return status;
    return napi_set_named_property(env, exports, name, fn);
}

/** Module initialization — exports all functions, detects CPU capabilities. */
napi_value Init(napi_env env, napi_value exports) {
    if (export_function(env, exports, "dot", api_ip) != napi_ok ||
        export_function(env, exports, "inner", api_ip) != napi_ok ||
        export_function(env, exports, "sqeuclidean", api_sqeuclidean) != napi_ok ||
        export_function(env, exports, "euclidean", api_euclidean) != napi_ok ||
        export_function(env, exports, "angular", api_angular) != napi_ok ||
        export_function(env, exports, "hamming", api_hamming) != napi_ok ||
        export_function(env, exports, "jaccard", api_jaccard) != napi_ok ||
        export_function(env, exports, "kullbackleibler", api_kld) != napi_ok ||
        export_function(env, exports, "jensenshannon", api_jsd) != napi_ok ||
        export_function(env, exports, "capabilitiesDetected", api_capabilities_detected) != napi_ok ||
        export_function(env, exports, "capabilitiesCompiled", api_capabilities_compiled) != napi_ok ||
        export_function(env, exports, "capabilitiesEnabled", api_capabilities_enabled) != napi_ok ||
        export_function(env, exports, "capabilitiesEnable", api_capabilities_enable) != napi_ok ||
        export_capability_names(env, exports) != napi_ok ||
        export_function(env, exports, "castF16ToF32", api_cast_f16_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToF16", api_cast_f32_to_f16) != napi_ok ||
        export_function(env, exports, "castBF16ToF32", api_cast_bf16_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToBF16", api_cast_f32_to_bf16) != napi_ok ||
        export_function(env, exports, "castE4M3ToF32", api_cast_e4m3_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToE4M3", api_cast_f32_to_e4m3) != napi_ok ||
        export_function(env, exports, "castE5M2ToF32", api_cast_e5m2_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToE5M2", api_cast_f32_to_e5m2) != napi_ok ||
        export_function(env, exports, "cast", api_cast) != napi_ok ||
        export_function(env, exports, "dotsPackedSize", api_dots_pack_size) != napi_ok ||
        export_function(env, exports, "dotsPack", api_dots_pack) != napi_ok ||
        export_function(env, exports, "dotsPacked", api_dots_packed) != napi_ok ||
        export_function(env, exports, "angularsPacked", api_angulars_packed) != napi_ok ||
        export_function(env, exports, "euclideansPacked", api_euclideans_packed) != napi_ok ||
        export_function(env, exports, "dotsSymmetric", api_dots_symmetric) != napi_ok ||
        export_function(env, exports, "angularsSymmetric", api_angulars_symmetric) != napi_ok ||
        export_function(env, exports, "euclideansSymmetric", api_euclideans_symmetric) != napi_ok) {
        return NULL;
    }
    nk_cpu_configure_thread(nk_cpu_capabilities_enabled());
    return exports;
}

#pragma endregion Module Init

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
