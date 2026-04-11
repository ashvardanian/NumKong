/**
 *  @brief JavaScript bindings for NumKong.
 *  @file javascript/numkong.c
 *  @author Ash Vardanian
 *  @date October 18, 2023
 *
 *  @see NodeJS docs: https://nodejs.org/api/n-api.html
 */

#include <string.h> // `strcmp` function

#if defined(NK_USE_OPENMP)
#include <omp.h>
#endif

#include <node_api.h> // `napi_*` functions — N-API v6+ for BigInt (Node ≥ 10.20)

#include <numkong/numkong.h> // `nk_*` functions — must be first to bring `_GNU_SOURCE`

#define NK_PARALLEL_PACKED_TILE    64
#define NK_PARALLEL_SYMMETRIC_TILE 32

/** @brief Global variable that caches the CPU capabilities, and is computed just once, when the module is loaded. */
nk_capability_t static_capabilities = nk_cap_serial_k;

#pragma region Helpers

/** @brief Parses a dtype string (e.g. "f32", "f16", "bf16") into a nk_dtype_t enum value. */
static nk_dtype_t parse_dtype_string(const char *str) {
    if (strcmp(str, "f64") == 0) return nk_f64_k;
    else if (strcmp(str, "f32") == 0) return nk_f32_k;
    else if (strcmp(str, "f16") == 0) return nk_f16_k;
    else if (strcmp(str, "bf16") == 0) return nk_bf16_k;
    else if (strcmp(str, "e4m3") == 0) return nk_e4m3_k;
    else if (strcmp(str, "e5m2") == 0) return nk_e5m2_k;
    else if (strcmp(str, "e2m3") == 0) return nk_e2m3_k;
    else if (strcmp(str, "e3m2") == 0) return nk_e3m2_k;
    else if (strcmp(str, "i8") == 0) return nk_i8_k;
    else if (strcmp(str, "u8") == 0) return nk_u8_k;
    else if (strcmp(str, "i16") == 0) return nk_i16_k;
    else if (strcmp(str, "u16") == 0) return nk_u16_k;
    else if (strcmp(str, "i64") == 0) return nk_i64_k;
    else if (strcmp(str, "u64") == 0) return nk_u64_k;
    else if (strcmp(str, "u1") == 0) return nk_u1_k;
    return nk_dtype_unknown_k;
}

/** @brief Validates that the N-API TypedArray type is compatible with the claimed dtype. */
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
    default: return 0;
    }
}

/**
 *  @brief Converts an nk_scalar_buffer_t result to a JavaScript number.
 *  @param env N-API environment.
 *  @param result The scalar buffer containing the result.
 *  @param out_dtype The dtype of the value stored in the buffer.
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

/** @brief Returns the byte width for a given dtype. */
static inline size_t dtype_byte_width(nk_dtype_t dtype) { return nk_dtype_bits(dtype) / NK_BITS_PER_BYTE; }

/** @brief Returns the N-API typed array type for a given output dtype. */
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

/** @brief Core distance computation — resolves dtype, dispatches kernel, converts result. */
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
                napi_throw_error(
                    env, NULL,
                    "Only f64, f32, i8, u8 arrays are auto-detected; pass dtype string as 3rd argument " "for other " "types");
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
    nk_find_kernel_punned(kernel_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&metric, &capability);
    if (!metric || !capability) {
        napi_throw_error(env, NULL, "Unsupported dtype for given metric");
        return NULL;
    }

    nk_dtype_t out_dtype = nk_kernel_output_dtype(kernel_kind, dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported output dtype for given metric/input combination");
        return NULL;
    }

    // Adjust dimensions for sub-byte packed types (e.g. Uint8Array with u1 dtype → bits)
    nk_size_t to_bits = nk_dtype_bits(dtype);
    size_t dimensions = (to_bits && to_bits < NK_BITS_PER_BYTE) ? length_a * NK_BITS_PER_BYTE / to_bits : length_a;

    nk_scalar_buffer_t result;
    metric(data_a, data_b, dimensions, &result);

    return nk_scalar_buffer_to_js_number(env, &result, out_dtype);
}

/** @brief N-API entry for inner product (dot).  */
napi_value api_ip(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_dot_k, nk_dtype_unknown_k);
}
/** @brief N-API entry for angular distance.  */
napi_value api_angular(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_angular_k, nk_dtype_unknown_k);
}
/** @brief N-API entry for squared Euclidean distance.  */
napi_value api_sqeuclidean(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_sqeuclidean_k, nk_dtype_unknown_k);
}
/** @brief N-API entry for Euclidean distance.  */
napi_value api_euclidean(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_euclidean_k, nk_dtype_unknown_k);
}
/** @brief N-API entry for Kullback-Leibler divergence.  */
napi_value api_kld(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_kld_k, nk_dtype_unknown_k);
}
/** @brief N-API entry for Jensen-Shannon distance.  */
napi_value api_jsd(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_jsd_k, nk_dtype_unknown_k);
}
/** @brief N-API entry for Hamming distance.  */
napi_value api_hamming(napi_env env, napi_callback_info info) { return dense(env, info, nk_kernel_hamming_k, nk_u1_k); }
/** @brief N-API entry for Jaccard distance.  */
napi_value api_jaccard(napi_env env, napi_callback_info info) { return dense(env, info, nk_kernel_jaccard_k, nk_u1_k); }

#pragma endregion Distance API

#pragma region Capabilities API

/**
 *  @brief Returns the runtime-detected SIMD capabilities as a bitmask.
 *  @return BigInt bitmask of nk_capability_t flags (33 flags from NEON to SME2P1)
 *
 *  This function exposes the cached capability bitmask to JavaScript users,
 *  allowing them to query what SIMD extensions are available at runtime.
 *  The capabilities are detected once at module load time and cached in static_capabilities.
 */
napi_value api_get_capabilities(napi_env env, napi_callback_info info) {
    napi_value result;
    // Use cached capabilities from module load (static_capabilities set in Init())
    napi_create_bigint_uint64(env, (uint64_t)static_capabilities, &result);
    return result;
}

#pragma endregion Capabilities API

#pragma region Cast API

/** @brief Converts a single value from a narrow type to f32. Reads uint32 bits, returns double. */
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

/** @brief Converts a single f32 value to a narrow type. Reads double, returns uint32 bits. */
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

/** @brief N-API entry for scalar f16-to-f32 conversion.  */
napi_value api_cast_f16_to_f32(napi_env env, napi_callback_info info) { return cast_to_f32(env, info, nk_f16_k); }
/** @brief N-API entry for scalar f32-to-f16 conversion.  */
napi_value api_cast_f32_to_f16(napi_env env, napi_callback_info info) { return cast_from_f32(env, info, nk_f16_k); }
/** @brief N-API entry for scalar bf16-to-f32 conversion.  */
napi_value api_cast_bf16_to_f32(napi_env env, napi_callback_info info) { return cast_to_f32(env, info, nk_bf16_k); }
/** @brief N-API entry for scalar f32-to-bf16 conversion.  */
napi_value api_cast_f32_to_bf16(napi_env env, napi_callback_info info) { return cast_from_f32(env, info, nk_bf16_k); }
/** @brief N-API entry for scalar e4m3-to-f32 conversion.  */
napi_value api_cast_e4m3_to_f32(napi_env env, napi_callback_info info) { return cast_to_f32(env, info, nk_e4m3_k); }
/** @brief N-API entry for scalar f32-to-e4m3 conversion.  */
napi_value api_cast_f32_to_e4m3(napi_env env, napi_callback_info info) { return cast_from_f32(env, info, nk_e4m3_k); }
/** @brief N-API entry for scalar e5m2-to-f32 conversion.  */
napi_value api_cast_e5m2_to_f32(napi_env env, napi_callback_info info) { return cast_to_f32(env, info, nk_e5m2_k); }
/** @brief N-API entry for scalar f32-to-e5m2 conversion.  */
napi_value api_cast_f32_to_e5m2(napi_env env, napi_callback_info info) { return cast_from_f32(env, info, nk_e5m2_k); }

/**
 *  @brief Buffer casting function using nk_cast.
 *  @param env N-API environment
 *  @param info Callback info containing 4 arguments:
 *              - src: source TypedArray
 *              - srcType: source dtype string
 *              - dst: destination TypedArray
 *              - dstType: destination dtype string
 *  @return null (modifies dst in place)
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

/** @brief Query packed buffer byte count: dotsPackedSize(width, depth, dtype) → number */
static napi_value api_dots_packed_size(napi_env env, napi_callback_info info) {
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

    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_size_k, dtype, static_capabilities, (nk_kernel_punned_t *)&size_fn,
                          &cap);
    if (!size_fn) {
        napi_throw_error(env, NULL, "dots_packed_size not available for this dtype");
        return NULL;
    }

    nk_size_t byte_count = size_fn((nk_size_t)width, (nk_size_t)depth);

    napi_value result;
    napi_create_double(env, (double)byte_count, &result);
    return result;
}

/** @brief Pack B matrix: dotsPack(data, width, depth, strideBytes, dtype) → { buffer, width, depth, byteLength } */
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
    nk_dots_packed_size_punned_t size_fn = NULL;
    nk_capability_t cap = nk_cap_serial_k;
    nk_find_kernel_punned(nk_kernel_dots_packed_size_k, dtype, static_capabilities, (nk_kernel_punned_t *)&size_fn,
                          &cap);
    if (!size_fn) {
        napi_throw_error(env, NULL, "dots_packed_size not available for this dtype");
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
    nk_find_kernel_punned(nk_kernel_dots_pack_k, dtype, static_capabilities, (nk_kernel_punned_t *)&pack_fn, &cap);
    if (!pack_fn) {
        napi_throw_error(env, NULL, "dots_pack not available for this dtype");
        return NULL;
    }
    pack_fn(data, (nk_size_t)width, (nk_size_t)depth, (nk_size_t)stride_bytes, packed_data);

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

/**
 *  @brief Shared dispatcher for packed operations (dots, angulars, euclideans).
 *  Args: TypedArray a, ArrayBuffer packed, TypedArray result, numbers height/width/depth/aStride/resultStride, string
 * dtype
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
    nk_find_kernel_punned(kernel_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel) {
        napi_throw_error(env, NULL, "Packed kernel not available for this dtype");
        return NULL;
    }

    uint32_t threads = 1;
    if (argc == 10) napi_get_value_uint32(env, args[9], &threads);

#if defined(NK_USE_OPENMP)
    if (threads == 0) threads = (uint32_t)omp_get_max_threads();
    omp_set_num_threads((int)threads);
#endif

    nk_size_t const tile_count = nk_size_divide_round_up_(height, NK_PARALLEL_PACKED_TILE);
#pragma omp parallel for schedule(dynamic, 1) if (threads > 1)
    for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        nk_size_t row = tile_idx * NK_PARALLEL_PACKED_TILE;
        nk_size_t chunk = (row + NK_PARALLEL_PACKED_TILE <= height) ? NK_PARALLEL_PACKED_TILE : (height - row);
        kernel((char const *)a_data + row * a_stride, packed_data, (char *)result_data + row * result_stride, chunk,
               (nk_size_t)width, (nk_size_t)depth, (nk_size_t)a_stride, (nk_size_t)result_stride);
    }
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

/**
 *  @brief Shared dispatcher for symmetric operations (dots, angulars, euclideans).
 *  Args: TypedArray vectors, TypedArray result, numbers nVectors/depth/vectorsStride/resultStride/rowStart/rowCount,
 * string dtype
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
    nk_find_kernel_punned(kernel_kind, dtype, static_capabilities, (nk_kernel_punned_t *)&kernel, &cap);
    if (!kernel) {
        napi_throw_error(env, NULL, "Symmetric kernel not available for this dtype");
        return NULL;
    }

    uint32_t threads = 1;
    if (argc == 10) napi_get_value_uint32(env, args[9], &threads);

#if defined(NK_USE_OPENMP)
    if (threads == 0) threads = (uint32_t)omp_get_max_threads();
    omp_set_num_threads((int)threads);
#endif

    nk_size_t const tile_count = nk_size_divide_round_up_(row_count, NK_PARALLEL_SYMMETRIC_TILE);
#pragma omp parallel for schedule(dynamic, 1) if (threads > 1)
    for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        nk_size_t tile_start = (nk_size_t)row_start + tile_idx * NK_PARALLEL_SYMMETRIC_TILE;
        nk_size_t tile_rows = (tile_start + NK_PARALLEL_SYMMETRIC_TILE <= (nk_size_t)row_start + row_count)
                                  ? NK_PARALLEL_SYMMETRIC_TILE
                                  : ((nk_size_t)row_start + row_count - tile_start);
        kernel(vectors_data, (nk_size_t)n_vectors, (nk_size_t)depth, (nk_size_t)vectors_stride, result_data,
               (nk_size_t)result_stride, tile_start, tile_rows);
    }

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

/** @brief Registers a C function as a named JavaScript export. */
static napi_status export_function(napi_env env, napi_value exports, char const *name, napi_callback func) {
    napi_value fn;
    napi_status status = napi_create_function(env, name, NAPI_AUTO_LENGTH, func, NULL, &fn);
    if (status != napi_ok) return status;
    return napi_set_named_property(env, exports, name, fn);
}

/** @brief Module initialization — exports all functions, detects CPU capabilities.  */
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
        export_function(env, exports, "getCapabilities", api_get_capabilities) != napi_ok ||
        export_function(env, exports, "castF16ToF32", api_cast_f16_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToF16", api_cast_f32_to_f16) != napi_ok ||
        export_function(env, exports, "castBF16ToF32", api_cast_bf16_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToBF16", api_cast_f32_to_bf16) != napi_ok ||
        export_function(env, exports, "castE4M3ToF32", api_cast_e4m3_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToE4M3", api_cast_f32_to_e4m3) != napi_ok ||
        export_function(env, exports, "castE5M2ToF32", api_cast_e5m2_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToE5M2", api_cast_f32_to_e5m2) != napi_ok ||
        export_function(env, exports, "cast", api_cast) != napi_ok ||
        export_function(env, exports, "dotsPackedSize", api_dots_packed_size) != napi_ok ||
        export_function(env, exports, "dotsPack", api_dots_pack) != napi_ok ||
        export_function(env, exports, "dotsPacked", api_dots_packed) != napi_ok ||
        export_function(env, exports, "angularsPacked", api_angulars_packed) != napi_ok ||
        export_function(env, exports, "euclideansPacked", api_euclideans_packed) != napi_ok ||
        export_function(env, exports, "dotsSymmetric", api_dots_symmetric) != napi_ok ||
        export_function(env, exports, "angularsSymmetric", api_angulars_symmetric) != napi_ok ||
        export_function(env, exports, "euclideansSymmetric", api_euclideans_symmetric) != napi_ok) {
        return NULL;
    }
    static_capabilities = nk_capabilities();
    return exports;
}

#pragma endregion Module Init

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
