/**
 *  @brief Runtime dispatch library for NumKong.
 *  @file c/numkong.c
 *  @author Ash Vardanian
 *  @date March 13, 2024
 */
#include "dispatch.h"

/*  MemorySanitizer cannot track initialization through SIMD intrinsics (SVE, NEON, SSE, AVX),
 *  causing false-positive "use-of-uninitialized-value" reports. We unpoison results after dispatch.
 */
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define nk_unpoison_(ptr, size) __msan_unpoison((ptr), (size))
#endif
#endif
#ifndef nk_unpoison_
#define nk_unpoison_(ptr, size) nk_unused_(ptr), nk_unused_(size)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// WASM capability detection for standalone Emscripten builds.
// EM_JS embeds JavaScript probes for runtime SIMD detection. It only works in
// standalone builds — Pyodide side modules cannot use EM_JS (the linker fails
// with undefined ___em_js__* symbols). Pyodide builds define NK_PYODIDE_SIDE_MODULE
// and fall through to compile-time detection in capabilities.h instead.
#if defined(__EMSCRIPTEN__) && NK_RUNTIME_DISPATCH && !defined(NK_PYODIDE_SIDE_MODULE)
#include <emscripten.h>

// EM_JS expands to an empty-parameter-list declaration `()` and a trailing `;`,
// which trigger `-Wstrict-prototypes` and `-Wextra-semi` under Clang/Emscripten.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#pragma clang diagnostic ignored "-Wextra-semi"
#endif

EM_JS(int, nk_has_v128, (), {
    var test = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
        0x02, 0x01, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x0b
    ]);
    try {
        return WebAssembly.validate(test) ? 1 : 0;
    }
    catch (e) {
        return 0;
    }
});

EM_JS(int, nk_has_relaxed, (), {
    var test = new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x01, 0x60, 0x03,
        0x7b, 0x7b, 0x7b, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x09, 0x01, 0x07,
        0x00, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0xfd, 0xaf, 0x01, 0x0b
    ]);
    try {
        return WebAssembly.validate(test) ? 1 : 0;
    }
    catch (e) {
        return 0;
    }
});

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif // __EMSCRIPTEN__ && NK_RUNTIME_DISPATCH && !NK_PYODIDE_SIDE_MODULE

/**
 *  @brief Fill memory with 0xFF - produces NaN for floats, -1 for signed integers, and MAX for unsigned.
 *  Avoids libc dependency on memset.
 */
NK_HELPER_INLINE void nk_fill_error_(void *ptr, nk_size_t bytes) {
    nk_u8_t *p = (nk_u8_t *)ptr;
    while (bytes--) *p++ = 0xFF;
}

void nk_error_dense_(void const *a, void const *b, nk_size_t n, void *d) {
    nk_unused_(a);
    nk_unused_(b);
    nk_unused_(n);
    nk_fill_error_(d, sizeof(nk_fmax_t));
}

void nk_error_sparse_intersect_(void const *a, void const *b, nk_size_t a_length, nk_size_t b_length, void *result,
                                nk_size_t *count) {
    nk_unused_(a);
    nk_unused_(b);
    nk_unused_(a_length);
    nk_unused_(b_length);
    nk_unused_(result);
    if (count) *count = 0;
}

void nk_error_sparse_dot_(void const *a, void const *b, void const *a_weights, void const *b_weights,
                          nk_size_t a_length, nk_size_t b_length, void *product) {
    nk_unused_(a);
    nk_unused_(b);
    nk_unused_(a_weights);
    nk_unused_(b_weights);
    nk_unused_(a_length);
    nk_unused_(b_length);
    nk_fill_error_(product, sizeof(nk_fmax_t));
}

void nk_error_curved_(void const *a, void const *b, void const *c, nk_size_t n, void *result) {
    nk_unused_(a);
    nk_unused_(b);
    nk_unused_(c);
    nk_unused_(n);
    nk_fill_error_(result, sizeof(nk_fmax_t));
}

void nk_error_geospatial_(void const *a_lats, void const *a_lons, void const *b_lats, void const *b_lons, nk_size_t n,
                          void *results) {
    nk_unused_(a_lats);
    nk_unused_(a_lons);
    nk_unused_(b_lats);
    nk_unused_(b_lons);
    nk_unused_(n);
    nk_fill_error_(results, sizeof(nk_fmax_t));
}

void nk_error_each_fma_(void const *a, void const *b, void const *c, nk_size_t n, void const *alpha, void const *beta,
                        void *result) {
    nk_unused_(a);
    nk_unused_(b);
    nk_unused_(c);
    nk_unused_(alpha);
    nk_unused_(beta);
    nk_fill_error_(result, n * sizeof(nk_fmax_t));
}
void nk_error_each_swiglu_(void const *gate, void const *up, void *y, nk_size_t rows, nk_size_t cols,
                           nk_size_t gate_row_stride, nk_size_t up_row_stride, nk_size_t y_row_stride,
                           nk_f32_t input_scale) {
    nk_unused_(gate), nk_unused_(up), nk_unused_(rows), nk_unused_(cols), nk_unused_(gate_row_stride),
        nk_unused_(up_row_stride), nk_unused_(y_row_stride), nk_unused_(input_scale);
    nk_fill_error_(y, sizeof(nk_fmax_t));
}

void nk_error_each_blend_(void const *a, void const *b, nk_size_t n, void const *alpha, void const *beta,
                          void *result) {
    nk_unused_(a);
    nk_unused_(b);
    nk_unused_(alpha);
    nk_unused_(beta);
    nk_fill_error_(result, n * sizeof(nk_fmax_t));
}

void nk_error_each_scale_(void const *a, nk_size_t n, void const *alpha, void const *beta, void *result) {
    nk_unused_(a);
    nk_unused_(alpha);
    nk_unused_(beta);
    nk_fill_error_(result, n * sizeof(nk_fmax_t));
}

void nk_error_each_sum_(void const *a, void const *b, nk_size_t n, void *y) {
    nk_unused_(a);
    nk_unused_(b);
    nk_fill_error_(y, n * sizeof(nk_fmax_t));
}

void nk_error_trig_(void const *x, nk_size_t n, void *y) {
    nk_unused_(x);
    nk_fill_error_(y, n * sizeof(nk_fmax_t));
}
void nk_error_trig_rope_(void const *x, void *y, void const *cos, void const *sin, nk_size_t rows, nk_size_t heads,
                         nk_size_t half_dim, nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t input_scale) {
    nk_unused_(x), nk_unused_(cos), nk_unused_(sin), nk_unused_(rows), nk_unused_(heads), nk_unused_(half_dim),
        nk_unused_(x_row_stride), nk_unused_(y_row_stride), nk_unused_(input_scale);
    nk_fill_error_(y, sizeof(nk_fmax_t));
}

void nk_error_mesh_(void const *a, void const *b, nk_size_t n, void *a_centroid, void *b_centroid, void *rotation,
                    void *scale, void *result) {
    nk_unused_(a);
    nk_unused_(b);
    nk_unused_(n);
    if (a_centroid) nk_fill_error_(a_centroid, 3 * sizeof(nk_fmax_t));
    if (b_centroid) nk_fill_error_(b_centroid, 3 * sizeof(nk_fmax_t));
    if (rotation) nk_fill_error_(rotation, 9 * sizeof(nk_fmax_t));
    if (scale) nk_fill_error_(scale, sizeof(nk_fmax_t));
    nk_fill_error_(result, sizeof(nk_fmax_t));
}

void nk_error_reduce_moments_(void const *data, nk_size_t count, nk_size_t stride_bytes, void *sum_ptr,
                              void *sumsq_ptr) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(sum_ptr), nk_unused_(sumsq_ptr);
    nk_fill_error_(sum_ptr, sizeof(nk_fmax_t));
    nk_fill_error_(sumsq_ptr, sizeof(nk_fmax_t));
}

void nk_error_reduce_minmax_(void const *data, nk_size_t count, nk_size_t stride_bytes, void *min_value,
                             nk_size_t *min_index, void *max_value, nk_size_t *max_index) {
    nk_unused_(data), nk_unused_(count), nk_unused_(stride_bytes), nk_unused_(min_value), nk_unused_(min_index),
        nk_unused_(max_value), nk_unused_(max_index);
    nk_fill_error_(min_value, sizeof(nk_fmax_t));
    nk_fill_error_(min_index, sizeof(nk_size_t));
    nk_fill_error_(max_value, sizeof(nk_fmax_t));
    nk_fill_error_(max_index, sizeof(nk_size_t));
}
void nk_error_reduce_rmsnorm_(void const *x, void const *gamma, void *y, nk_size_t rows, nk_size_t groups,
                              nk_size_t cols, nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t eps,
                              nk_f32_t input_scale) {
    nk_unused_(x), nk_unused_(gamma), nk_unused_(rows), nk_unused_(groups), nk_unused_(cols), nk_unused_(x_row_stride),
        nk_unused_(y_row_stride), nk_unused_(eps), nk_unused_(input_scale);
    nk_fill_error_(y, sizeof(nk_fmax_t));
}

nk_size_t nk_error_pack_size_(nk_size_t n, nk_size_t k) {
    nk_unused_(n);
    nk_unused_(k);
    return 0;
}

void nk_error_packed_shape_(void const *packed, nk_size_t *n, nk_size_t *k) {
    nk_unused_(packed);
    if (n) *n = 0;
    if (k) *k = 0;
}

void nk_error_pack_(void const *b, nk_size_t n, nk_size_t k, nk_size_t b_stride, void *b_packed) {
    nk_unused_(b);
    nk_unused_(n);
    nk_unused_(k);
    nk_unused_(b_stride);
    nk_unused_(b_packed);
}

void nk_error_dots_(void const *a, void const *b_packed, void *c, nk_size_t m, nk_size_t n, nk_size_t k,
                    nk_size_t a_stride, nk_size_t c_stride) {
    nk_unused_(a);
    nk_unused_(b_packed);
    nk_unused_(k);
    nk_unused_(a_stride);
    for (nk_size_t row = 0; row < m; ++row) nk_fill_error_((nk_u8_t *)c + row * c_stride, n * sizeof(nk_fmax_t));
}

void nk_error_dots_symmetric_(void const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, void *result,
                              nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_unused_(vectors);
    nk_unused_(depth);
    nk_unused_(stride);
    nk_unused_(row_start);
    nk_unused_(row_count);
    for (nk_size_t row = 0; row < n_vectors; ++row)
        nk_fill_error_((nk_u8_t *)result + row * result_stride, n_vectors * sizeof(nk_fmax_t));
}

nk_size_t nk_error_attention_pack_size_(nk_size_t num_kv_heads, nk_size_t head_dim, nk_u32_t const *segment_lengths,
                                        nk_size_t segment_count) {
    nk_unused_(num_kv_heads), nk_unused_(head_dim), nk_unused_(segment_lengths), nk_unused_(segment_count);
    return 0;
}

void nk_error_attention_packed_shape_(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                      nk_size_t *segments) {
    nk_unused_(key_value_packed);
    if (heads) *heads = 0;
    if (depth) *depth = 0;
    if (segments) *segments = 0;
}

void nk_error_attention_pack_(void const *k, void const *v, nk_size_t num_kv_heads, nk_size_t head_dim,
                              nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
                              nk_size_t k_stride, nk_size_t v_stride, void *key_value_packed, nk_size_t begin,
                              nk_size_t end) {
    nk_unused_(k), nk_unused_(v), nk_unused_(num_kv_heads), nk_unused_(head_dim), nk_unused_(segment_offsets);
    nk_unused_(segment_lengths), nk_unused_(segment_count), nk_unused_(k_stride), nk_unused_(v_stride);
    nk_unused_(key_value_packed), nk_unused_(begin), nk_unused_(end);
}

void nk_error_attention_packed_(void const *q, void const *key_value_packed, void *output, nk_size_t num_heads,
                                nk_size_t num_kv_heads, nk_size_t head_dim, nk_u32_t const *query_offsets,
                                nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale, nk_size_t begin,
                                nk_size_t end) {
    // The ragged output extent (query_offsets[segments] rows) lives behind the backend-private
    // packed header, so unlike the dense error handlers this one cannot poison the output.
    nk_unused_(q), nk_unused_(key_value_packed), nk_unused_(output), nk_unused_(num_heads), nk_unused_(num_kv_heads);
    nk_unused_(head_dim), nk_unused_(query_offsets), nk_unused_(q_stride), nk_unused_(o_stride), nk_unused_(scale);
    nk_unused_(begin), nk_unused_(end);
}

// Global dispatch table - 64-byte aligned for cache performance
// Type defined in dispatch.h, made non-static for access from dtype files
NK_ALIGN64 nk_implementations_t nk_dispatch_table;

// Direct dispatch macros using central dispatch table (no lazy initialization)
#define nk_dispatch_dense_(name, extension, input_type, output_type)                                        \
    NK_API_RUNTIME void nk_##name##_##extension(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, nk_##output_type##_t *results) {               \
        nk_dispatch_table.name##_##extension(a, b, n, (void *)results);                                     \
        nk_unpoison_((void *)results, sizeof(nk_##output_type##_t));                                        \
    }

#define nk_dispatch_sparse_(name, extension, type)                                                                  \
    NK_API_RUNTIME void nk_##name##_##extension(nk_##type##_t const *a, nk_##type##_t const *b, nk_size_t a_length, \
                                                nk_size_t b_length, nk_##type##_t *result, nk_size_t *count) {      \
        nk_dispatch_table.name##_##extension(a, b, a_length, b_length, (void *)result, count);                      \
        nk_unpoison_(count, sizeof(nk_size_t));                                                                     \
        nk_unpoison_((void *)result, (*count) * sizeof(nk_##type##_t));                                             \
    }

#define nk_dispatch_sparse_dot_(name, index_type, weight_type, output_type)                                \
    NK_API_RUNTIME void nk_##name##_##index_type##weight_type(                                             \
        nk_##index_type##_t const *a, nk_##index_type##_t const *b, nk_##weight_type##_t const *a_weights, \
        nk_##weight_type##_t const *b_weights, nk_size_t a_length, nk_size_t b_length,                     \
        nk_##output_type##_t *product) {                                                                   \
        nk_dispatch_table.name##_##index_type##weight_type(a, b, a_weights, b_weights, a_length, b_length, \
                                                           (void *)product);                               \
        nk_unpoison_((void *)product, sizeof(nk_##output_type##_t));                                       \
    }

#define nk_dispatch_curved_(name, extension, output_type)                                                 \
    NK_API_RUNTIME void nk_##name##_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b, \
                                                nk_##extension##_t const *c, nk_size_t n,                 \
                                                nk_##output_type##_t *result) {                           \
        nk_dispatch_table.name##_##extension(a, b, c, n, (void *)result);                                 \
        nk_unpoison_((void *)result, sizeof(nk_##output_type##_t));                                       \
    }

#define nk_dispatch_geospatial_(name, extension, output_type)                                                       \
    NK_API_RUNTIME void nk_##name##_##extension(nk_##extension##_t const *a_lats, nk_##extension##_t const *a_lons, \
                                                nk_##extension##_t const *b_lats, nk_##extension##_t const *b_lons, \
                                                nk_size_t n, nk_##output_type##_t *results) {                       \
        nk_dispatch_table.name##_##extension(a_lats, a_lons, b_lats, b_lons, n, (void *)results);                   \
        nk_unpoison_((void *)results, sizeof(nk_##output_type##_t));                                                \
    }

#define nk_dispatch_each_fma_(extension, scalar_type)                                                        \
    NK_API_RUNTIME void nk_each_fma_##extension(                                                             \
        nk_##extension##_t const *a, nk_##extension##_t const *b, nk_##extension##_t const *c, nk_size_t n,  \
        nk_##scalar_type##_t const *alpha, nk_##scalar_type##_t const *beta, nk_##extension##_t *result) {   \
        nk_dispatch_table.each_fma_##extension(a, b, c, n, (void const *)alpha, (void const *)beta, result); \
        nk_unpoison_((void *)result, n * sizeof(nk_##extension##_t));                                        \
    }
#define nk_dispatch_each_swiglu_(extension, data_type)                                                      \
    NK_API_RUNTIME void nk_each_swiglu_##extension(                                                         \
        data_type const *gate, data_type const *up, data_type *y, nk_size_t rows, nk_size_t cols,           \
        nk_size_t gate_row_stride, nk_size_t up_row_stride, nk_size_t y_row_stride, nk_f32_t input_scale) { \
        ((nk_each_swiglu_punned_t)nk_dispatch_table.each_swiglu_##extension)(                               \
            gate, up, y, rows, cols, gate_row_stride, up_row_stride, y_row_stride, input_scale);            \
    }

#define nk_dispatch_each_blend_(extension, scalar_type)                                                           \
    NK_API_RUNTIME void nk_each_blend_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b,       \
                                                  nk_size_t n, nk_##scalar_type##_t const *alpha,                 \
                                                  nk_##scalar_type##_t const *beta, nk_##extension##_t *result) { \
        nk_dispatch_table.each_blend_##extension(a, b, n, (void const *)alpha, (void const *)beta, result);       \
        nk_unpoison_((void *)result, n * sizeof(nk_##extension##_t));                                             \
    }

#define nk_dispatch_each_scale_(extension, scalar_type)                                                                \
    NK_API_RUNTIME void nk_each_scale_##extension(nk_##extension##_t const *a, nk_size_t n,                            \
                                                  nk_##scalar_type##_t const *alpha, nk_##scalar_type##_t const *beta, \
                                                  nk_##extension##_t *result) {                                        \
        nk_dispatch_table.each_scale_##extension(a, n, (void const *)alpha, (void const *)beta, result);               \
        nk_unpoison_((void *)result, n * sizeof(nk_##extension##_t));                                                  \
    }

#define nk_dispatch_each_sum_(extension)                                                                               \
    NK_API_RUNTIME void nk_each_sum_##extension(nk_##extension##_t const *a, nk_##extension##_t const *b, nk_size_t n, \
                                                nk_##extension##_t *result) {                                          \
        nk_dispatch_table.each_sum_##extension(a, b, n, result);                                                       \
        nk_unpoison_((void *)result, n * sizeof(nk_##extension##_t));                                                  \
    }

#define nk_dispatch_trig_(name, extension)                                                          \
    NK_API_RUNTIME void nk_trig_##name##_##extension(nk_##extension##_t const *inputs, nk_size_t n, \
                                                     nk_##extension##_t *outputs) {                 \
        nk_dispatch_table.trig_##name##_##extension(inputs, n, outputs);                            \
        nk_unpoison_((void *)outputs, n * sizeof(nk_##extension##_t));                              \
    }
#define nk_dispatch_trig_rope_(extension, data_type)                                                                 \
    NK_API_RUNTIME void nk_trig_rope_##extension(                                                                    \
        data_type const *x, data_type *y, nk_f32_t const *cos, nk_f32_t const *sin, nk_size_t rows, nk_size_t heads, \
        nk_size_t half_dim, nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t input_scale) {                  \
        ((nk_kernel_trig_rope_punned_t)nk_dispatch_table.trig_rope_##extension)(                                     \
            x, y, cos, sin, rows, heads, half_dim, x_row_stride, y_row_stride, input_scale);                         \
    }

#define nk_dispatch_mesh_(name, extension, transform_type, metric_type)                                             \
    NK_API_RUNTIME void nk_##name##_##extension(                                                                    \
        nk_##extension##_t const *a, nk_##extension##_t const *b, nk_size_t n, nk_##transform_type##_t *a_centroid, \
        nk_##transform_type##_t *b_centroid, nk_##transform_type##_t *rotation, nk_##transform_type##_t *scale,     \
        nk_##metric_type##_t *result) {                                                                             \
        nk_dispatch_table.name##_##extension(a, b, n, (void *)a_centroid, (void *)b_centroid, (void *)rotation,     \
                                             (void *)scale, (void *)result);                                        \
        if (a_centroid) nk_unpoison_((void *)a_centroid, 3 * sizeof(nk_##transform_type##_t));                      \
        if (b_centroid) nk_unpoison_((void *)b_centroid, 3 * sizeof(nk_##transform_type##_t));                      \
        if (rotation) nk_unpoison_((void *)rotation, 9 * sizeof(nk_##transform_type##_t));                          \
        if (scale) nk_unpoison_((void *)scale, sizeof(nk_##transform_type##_t));                                    \
        nk_unpoison_((void *)result, sizeof(nk_##metric_type##_t));                                                 \
    }

#define nk_dispatch_reduce_moments_(extension, data_type, sum_type, sumsq_type)                                        \
    NK_API_RUNTIME void nk_reduce_moments_##extension(data_type const *data, nk_size_t count, nk_size_t stride_bytes,  \
                                                      sum_type *sum_ptr, sumsq_type *sumsq_ptr) {                      \
        ((nk_reduce_moments_punned_t)nk_dispatch_table.reduce_moments_##extension)(data, count, stride_bytes, sum_ptr, \
                                                                                   sumsq_ptr);                         \
        nk_unpoison_((void *)sum_ptr, sizeof(sum_type));                                                               \
        nk_unpoison_((void *)sumsq_ptr, sizeof(sumsq_type));                                                           \
    }

#define nk_dispatch_reduce_minmax_(extension, data_type, minmax_type)                                                  \
    NK_API_RUNTIME void nk_reduce_minmax_##extension(data_type const *data, nk_size_t count, nk_size_t stride_bytes,   \
                                                     minmax_type *min_value, nk_size_t *min_index,                     \
                                                     minmax_type *max_value, nk_size_t *max_index) {                   \
        ((nk_reduce_minmax_punned_t)nk_dispatch_table.reduce_minmax_##extension)(data, count, stride_bytes, min_value, \
                                                                                 min_index, max_value, max_index);     \
        nk_unpoison_((void *)min_value, sizeof(minmax_type));                                                          \
        nk_unpoison_(min_index, sizeof(nk_size_t));                                                                    \
        nk_unpoison_((void *)max_value, sizeof(minmax_type));                                                          \
        nk_unpoison_(max_index, sizeof(nk_size_t));                                                                    \
    }
#define nk_dispatch_reduce_rmsnorm_(extension, data_type)                                                          \
    NK_API_RUNTIME void nk_reduce_rmsnorm_##extension(                                                             \
        data_type const *x, nk_f32_t const *gamma, data_type *y, nk_size_t rows, nk_size_t groups, nk_size_t cols, \
        nk_size_t x_row_stride, nk_size_t y_row_stride, nk_f32_t eps, nk_f32_t input_scale) {                      \
        ((nk_reduce_rmsnorm_punned_t)nk_dispatch_table.reduce_rmsnorm_##extension)(                                \
            x, gamma, y, rows, groups, cols, x_row_stride, y_row_stride, eps, input_scale);                        \
    }

#define nk_dispatch_cross_pack_size_(api_name, name, input_type, accum_type)              \
    NK_API_RUNTIME nk_size_t nk_##api_name##_pack_size_##name(nk_size_t n, nk_size_t k) { \
        return nk_dispatch_table.api_name##_pack_size_##name(n, k);                       \
    }

#define nk_dispatch_cross_packed_shape_(api_name, name, input_type, accum_type)                               \
    NK_API_RUNTIME void nk_##api_name##_packed_shape_##name(void const *packed, nk_size_t *n, nk_size_t *k) { \
        nk_dispatch_table.api_name##_packed_shape_##name(packed, n, k);                                       \
    }

#define nk_dispatch_cross_pack_(api_name, name, input_type, accum_type)                                     \
    NK_API_RUNTIME void nk_##api_name##_pack_##name(nk_##input_type##_t const *b, nk_size_t n, nk_size_t k, \
                                                    nk_size_t b_stride, void *b_packed) {                   \
        nk_dispatch_table.api_name##_pack_##name(b, n, k, b_stride, b_packed);                              \
    }

#define nk_dispatch_dots_pack_(name, input_type)                                                         \
    NK_API_RUNTIME void nk_dots_pack_##name(nk_##input_type##_t const *b, nk_size_t n, nk_size_t k,      \
                                            nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, \
                                            nk_size_t columns_end) {                                     \
        nk_dispatch_table.dots_pack_##name(b, n, k, b_stride, b_packed, columns_begin, columns_end);     \
    }

#define nk_dispatch_cross_packed_(api_name, name, input_type, accum_type, output_type)                                \
    NK_API_RUNTIME void nk_##api_name##_packed_##name(nk_##input_type##_t const *a, void const *b_packed,             \
                                                      nk_##output_type##_t *c, nk_size_t m, nk_size_t n, nk_size_t k, \
                                                      nk_size_t a_stride, nk_size_t c_stride) {                       \
        nk_dispatch_table.api_name##_packed_##name(a, b_packed, c, m, n, k, a_stride, c_stride);                      \
        nk_unpoison_((void *)c, m * c_stride);                                                                        \
    }

#define nk_dispatch_cross_symmetric_(api_name, name, input_type, output_type)                                   \
    NK_API_RUNTIME void nk_##api_name##_symmetric_##name(                                                       \
        nk_##input_type##_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,             \
        nk_##output_type##_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {      \
        nk_dispatch_table.api_name##_symmetric_##name(vectors, n_vectors, depth, stride, result, result_stride, \
                                                      row_start, row_count);                                    \
        nk_unpoison_((void *)result, row_count * result_stride);                                                \
    }

#define nk_dispatch_maxsim_packed_(name, output_type)                                                           \
    NK_API_RUNTIME void nk_maxsim_packed_##name(void const *q_packed, void const *d_packed, nk_size_t n_q,      \
                                                nk_size_t n_d, nk_size_t depth, nk_##output_type##_t *result) { \
        nk_dispatch_table.maxsim_packed_##name(q_packed, d_packed, n_q, n_d, depth, (void *)result);            \
        nk_unpoison_((void *)result, sizeof(nk_##output_type##_t));                                             \
    }

#define nk_dispatch_attention_pack_size_(name)                                                                         \
    NK_API_RUNTIME nk_size_t nk_attention_pack_size_##name(nk_size_t num_kv_heads, nk_size_t head_dim,                 \
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count) { \
        return nk_dispatch_table.attention_pack_size_##name(num_kv_heads, head_dim, segment_lengths, segment_count);   \
    }

#define nk_dispatch_attention_packed_shape_(name)                                                                \
    NK_API_RUNTIME void nk_attention_packed_shape_##name(void const *packed, nk_size_t *heads, nk_size_t *depth, \
                                                         nk_size_t *segments) {                                  \
        nk_dispatch_table.attention_packed_shape_##name(packed, heads, depth, segments);                         \
    }

#define nk_dispatch_attention_pack_(name)                                                                              \
    NK_API_RUNTIME void nk_attention_pack_##name(                                                                      \
        nk_##name##_t const *k, nk_##name##_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim,                    \
        nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride, \
        nk_size_t v_stride, void *key_value_packed, nk_size_t begin, nk_size_t end) {                                  \
        nk_dispatch_table.attention_pack_##name(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths,        \
                                                segment_count, k_stride, v_stride, key_value_packed, begin, end);      \
    }

#define nk_dispatch_attention_packed_(name)                                                                         \
    NK_API_RUNTIME void nk_attention_packed_##name(                                                                 \
        nk_##name##_t const *q, void const *key_value_packed, nk_f32_t *output, nk_size_t num_heads,                \
        nk_size_t num_kv_heads, nk_size_t head_dim, nk_u32_t const *query_offsets, nk_size_t q_stride,              \
        nk_size_t o_stride, nk_f32_t scale, nk_size_t begin, nk_size_t end) {                                       \
        nk_dispatch_table.attention_packed_##name((void const *)q, key_value_packed, (void *)output, num_heads,     \
                                                  num_kv_heads, head_dim, query_offsets, q_stride, o_stride, scale, \
                                                  begin, end);                                                      \
    }

// Dot products
nk_dispatch_dense_(dot, f64c, f64c, f64c)
nk_dispatch_dense_(dot, f32c, f32c, f64c)
nk_dispatch_dense_(dot, bf16c, bf16c, f32c)
nk_dispatch_dense_(dot, f16c, f16c, f32c)
nk_dispatch_dense_(dot, f64, f64, f64)
nk_dispatch_dense_(dot, f32, f32, f64)
nk_dispatch_dense_(dot, bf16, bf16, f32)
nk_dispatch_dense_(dot, f16, f16, f32)
nk_dispatch_dense_(dot, e5m2, e5m2, f32)
nk_dispatch_dense_(dot, e4m3, e4m3, f32)
nk_dispatch_dense_(dot, e3m2, e3m2, f32)
nk_dispatch_dense_(dot, e2m3, e2m3, f32)
nk_dispatch_dense_(dot, i8, i8, i32)
nk_dispatch_dense_(dot, i4, i4x2, i32)
nk_dispatch_dense_(dot, u8, u8, u32)
nk_dispatch_dense_(dot, u4, u4x2, u32)
nk_dispatch_dense_(dot, u1, u1x8, u32)
nk_dispatch_dense_(vdot, f64c, f64c, f64c)
nk_dispatch_dense_(vdot, f32c, f32c, f64c)
nk_dispatch_dense_(vdot, bf16c, bf16c, f32c)
nk_dispatch_dense_(vdot, f16c, f16c, f32c)

// Spatial distances
nk_dispatch_dense_(angular, f64, f64, f64)
nk_dispatch_dense_(angular, f32, f32, f64)
nk_dispatch_dense_(angular, bf16, bf16, f32)
nk_dispatch_dense_(angular, f16, f16, f32)
nk_dispatch_dense_(angular, e5m2, e5m2, f32)
nk_dispatch_dense_(angular, e4m3, e4m3, f32)
nk_dispatch_dense_(angular, e3m2, e3m2, f32)
nk_dispatch_dense_(angular, e2m3, e2m3, f32)
nk_dispatch_dense_(angular, i8, i8, f32)
nk_dispatch_dense_(angular, i4, i4x2, f32)
nk_dispatch_dense_(angular, u8, u8, f32)
nk_dispatch_dense_(angular, u4, u4x2, f32)
nk_dispatch_dense_(euclidean, f64, f64, f64)
nk_dispatch_dense_(euclidean, f32, f32, f64)
nk_dispatch_dense_(euclidean, bf16, bf16, f32)
nk_dispatch_dense_(euclidean, f16, f16, f32)
nk_dispatch_dense_(euclidean, e5m2, e5m2, f32)
nk_dispatch_dense_(euclidean, e4m3, e4m3, f32)
nk_dispatch_dense_(euclidean, e3m2, e3m2, f32)
nk_dispatch_dense_(euclidean, e2m3, e2m3, f32)
nk_dispatch_dense_(euclidean, i8, i8, f32)
nk_dispatch_dense_(euclidean, i4, i4x2, f32)
nk_dispatch_dense_(euclidean, u8, u8, f32)
nk_dispatch_dense_(euclidean, u4, u4x2, f32)
nk_dispatch_dense_(sqeuclidean, f64, f64, f64)
nk_dispatch_dense_(sqeuclidean, f32, f32, f64)
nk_dispatch_dense_(sqeuclidean, bf16, bf16, f32)
nk_dispatch_dense_(sqeuclidean, f16, f16, f32)
nk_dispatch_dense_(sqeuclidean, e5m2, e5m2, f32)
nk_dispatch_dense_(sqeuclidean, e4m3, e4m3, f32)
nk_dispatch_dense_(sqeuclidean, e3m2, e3m2, f32)
nk_dispatch_dense_(sqeuclidean, e2m3, e2m3, f32)
nk_dispatch_dense_(sqeuclidean, i8, i8, u32)
nk_dispatch_dense_(sqeuclidean, i4, i4x2, u32)
nk_dispatch_dense_(sqeuclidean, u8, u8, u32)
nk_dispatch_dense_(sqeuclidean, u4, u4x2, u32)

// Binary distances
nk_dispatch_dense_(hamming, u8, u8, u32)
nk_dispatch_dense_(hamming, u1, u1x8, u32)
nk_dispatch_dense_(jaccard, u32, u32, f32)
nk_dispatch_dense_(jaccard, u16, u16, f32)
nk_dispatch_dense_(jaccard, u1, u1x8, f32)

// Curved spaces
nk_dispatch_curved_(bilinear, f64c, f64c)
nk_dispatch_curved_(bilinear, f32c, f64c)
nk_dispatch_curved_(bilinear, bf16c, f32c)
nk_dispatch_curved_(bilinear, f16c, f32c)
nk_dispatch_curved_(bilinear, f64, f64)
nk_dispatch_curved_(bilinear, f32, f64)
nk_dispatch_curved_(bilinear, bf16, f32)
nk_dispatch_curved_(bilinear, f16, f32)
nk_dispatch_curved_(mahalanobis, f64, f64)
nk_dispatch_curved_(mahalanobis, f32, f64)
nk_dispatch_curved_(mahalanobis, bf16, f32)
nk_dispatch_curved_(mahalanobis, f16, f32)

// Geospatial distances
nk_dispatch_geospatial_(haversine, f64, f64)
nk_dispatch_geospatial_(haversine, f32, f32)
nk_dispatch_geospatial_(vincenty, f64, f64)
nk_dispatch_geospatial_(vincenty, f32, f32)

// Probability distributions
nk_dispatch_dense_(kld, f64, f64, f64)
nk_dispatch_dense_(kld, f32, f32, f64)
nk_dispatch_dense_(kld, bf16, bf16, f32)
nk_dispatch_dense_(kld, f16, f16, f32)
nk_dispatch_dense_(jsd, f64, f64, f64)
nk_dispatch_dense_(jsd, f32, f32, f64)
nk_dispatch_dense_(jsd, bf16, bf16, f32)
nk_dispatch_dense_(jsd, f16, f16, f32)

// Mesh alignment (RMSD, Kabsch, Umeyama)
nk_dispatch_mesh_(rmsd, f64, f64, f64)
nk_dispatch_mesh_(rmsd, f32, f32, f64)
nk_dispatch_mesh_(rmsd, bf16, f32, f32)
nk_dispatch_mesh_(rmsd, f16, f32, f32)
nk_dispatch_mesh_(kabsch, f64, f64, f64)
nk_dispatch_mesh_(kabsch, f32, f32, f64)
nk_dispatch_mesh_(kabsch, bf16, f32, f32)
nk_dispatch_mesh_(kabsch, f16, f32, f32)
nk_dispatch_mesh_(umeyama, f64, f64, f64)
nk_dispatch_mesh_(umeyama, f32, f32, f64)
nk_dispatch_mesh_(umeyama, bf16, f32, f32)
nk_dispatch_mesh_(umeyama, f16, f32, f32)

// Sparse sets
nk_dispatch_sparse_(sparse_intersect, u64, u64)
nk_dispatch_sparse_(sparse_intersect, u32, u32)
nk_dispatch_sparse_(sparse_intersect, u16, u16)
nk_dispatch_sparse_dot_(sparse_dot, u32, f32, f64)
nk_dispatch_sparse_dot_(sparse_dot, u16, bf16, f32)

// Element-wise operations
nk_dispatch_each_scale_(f64c, f64c)
nk_dispatch_each_scale_(f32c, f32c)
nk_dispatch_each_scale_(f64, f64)
nk_dispatch_each_scale_(f32, f32)
nk_dispatch_each_scale_(bf16, f32)
nk_dispatch_each_scale_(f16, f32)
nk_dispatch_each_scale_(e5m2, f32)
nk_dispatch_each_scale_(e4m3, f32)
nk_dispatch_each_scale_(e3m2, f32)
nk_dispatch_each_scale_(e2m3, f32)
nk_dispatch_each_scale_(i64, f64)
nk_dispatch_each_scale_(i32, f64)
nk_dispatch_each_scale_(i16, f32)
nk_dispatch_each_scale_(i8, f32)
nk_dispatch_each_scale_(u64, f64)
nk_dispatch_each_scale_(u32, f64)
nk_dispatch_each_scale_(u16, f32)
nk_dispatch_each_scale_(u8, f32)
nk_dispatch_each_sum_(f64c)
nk_dispatch_each_sum_(f32c)
nk_dispatch_each_sum_(f64)
nk_dispatch_each_sum_(f32)
nk_dispatch_each_sum_(bf16)
nk_dispatch_each_sum_(f16)
nk_dispatch_each_sum_(e5m2)
nk_dispatch_each_sum_(e4m3)
nk_dispatch_each_sum_(e3m2)
nk_dispatch_each_sum_(e2m3)
nk_dispatch_each_sum_(i64)
nk_dispatch_each_sum_(i32)
nk_dispatch_each_sum_(i16)
nk_dispatch_each_sum_(i8)
nk_dispatch_each_sum_(u64)
nk_dispatch_each_sum_(u32)
nk_dispatch_each_sum_(u16)
nk_dispatch_each_sum_(u8)
nk_dispatch_each_blend_(f64c, f64c)
nk_dispatch_each_blend_(f32c, f32c)
nk_dispatch_each_blend_(f64, f64)
nk_dispatch_each_blend_(f32, f32)
nk_dispatch_each_blend_(bf16, f32)
nk_dispatch_each_blend_(f16, f32)
nk_dispatch_each_blend_(e5m2, f32)
nk_dispatch_each_blend_(e4m3, f32)
nk_dispatch_each_blend_(e3m2, f32)
nk_dispatch_each_blend_(e2m3, f32)
nk_dispatch_each_blend_(i64, f64)
nk_dispatch_each_blend_(i32, f64)
nk_dispatch_each_blend_(i16, f32)
nk_dispatch_each_blend_(i8, f32)
nk_dispatch_each_blend_(u64, f64)
nk_dispatch_each_blend_(u32, f64)
nk_dispatch_each_blend_(u16, f32)
nk_dispatch_each_blend_(u8, f32)
nk_dispatch_each_fma_(f64c, f64c)
nk_dispatch_each_swiglu_(f32, nk_f32_t) nk_dispatch_each_swiglu_(bf16, nk_bf16_t) nk_dispatch_each_fma_(f32c, f32c)
nk_dispatch_each_fma_(f64, f64)
nk_dispatch_each_fma_(f32, f32)
nk_dispatch_each_fma_(bf16, f32)
nk_dispatch_each_fma_(f16, f32)
nk_dispatch_each_fma_(e5m2, f32)
nk_dispatch_each_fma_(e4m3, f32)
nk_dispatch_each_fma_(e3m2, f32)
nk_dispatch_each_fma_(e2m3, f32)
nk_dispatch_each_fma_(i64, f64)
nk_dispatch_each_fma_(i32, f64)
nk_dispatch_each_fma_(i16, f32)
nk_dispatch_each_fma_(i8, f32)
nk_dispatch_each_fma_(u64, f64)
nk_dispatch_each_fma_(u32, f64)
nk_dispatch_each_fma_(u16, f32)
nk_dispatch_each_fma_(u8, f32)

// Trigonometry functions
nk_dispatch_trig_(sin, f64)
nk_dispatch_trig_rope_(f32, nk_f32_t) nk_dispatch_trig_rope_(bf16, nk_bf16_t) nk_dispatch_trig_(sin, f32)
nk_dispatch_trig_(sin, f16)
nk_dispatch_trig_(cos, f64)
nk_dispatch_trig_(cos, f32)
nk_dispatch_trig_(cos, f16)
nk_dispatch_trig_(atan, f64)
nk_dispatch_trig_(atan, f32)
nk_dispatch_trig_(atan, f16)

// Horizontal reductions: moments (sum + sum-of-squares)
nk_dispatch_reduce_moments_(f64, nk_f64_t, nk_f64_t, nk_f64_t)
nk_dispatch_reduce_moments_(f32, nk_f32_t, nk_f64_t, nk_f64_t)
nk_dispatch_reduce_moments_(bf16, nk_bf16_t, nk_f32_t, nk_f32_t)
nk_dispatch_reduce_moments_(f16, nk_f16_t, nk_f32_t, nk_f32_t)
nk_dispatch_reduce_moments_(e5m2, nk_e5m2_t, nk_f32_t, nk_f32_t)
nk_dispatch_reduce_moments_(e4m3, nk_e4m3_t, nk_f32_t, nk_f32_t)
nk_dispatch_reduce_moments_(e3m2, nk_e3m2_t, nk_f32_t, nk_f32_t)
nk_dispatch_reduce_moments_(e2m3, nk_e2m3_t, nk_f32_t, nk_f32_t)
nk_dispatch_reduce_moments_(i64, nk_i64_t, nk_i64_t, nk_u64_t)
nk_dispatch_reduce_moments_(i32, nk_i32_t, nk_i64_t, nk_u64_t)
nk_dispatch_reduce_moments_(i16, nk_i16_t, nk_i64_t, nk_u64_t)
nk_dispatch_reduce_moments_(i8, nk_i8_t, nk_i64_t, nk_u64_t)
nk_dispatch_reduce_moments_(i4, nk_i4x2_t, nk_i64_t, nk_u64_t)
nk_dispatch_reduce_moments_(u64, nk_u64_t, nk_u64_t, nk_u64_t)
nk_dispatch_reduce_moments_(u32, nk_u32_t, nk_u64_t, nk_u64_t)
nk_dispatch_reduce_moments_(u16, nk_u16_t, nk_u64_t, nk_u64_t)
nk_dispatch_reduce_moments_(u8, nk_u8_t, nk_u64_t, nk_u64_t)
nk_dispatch_reduce_moments_(u4, nk_u4x2_t, nk_u64_t, nk_u64_t)
nk_dispatch_reduce_moments_(u1, nk_u1x8_t, nk_u64_t, nk_u64_t)

// Fused transformer nonlinearities: grouped RMSNorm
nk_dispatch_reduce_rmsnorm_(e4m3, nk_e4m3_t)

    // Fused transformer nonlinearities: SwiGLU / SiLU
    nk_dispatch_each_swiglu_(e4m3, nk_e4m3_t)

    // Fused transformer nonlinearities: RoPE (NeoX split-half)
    nk_dispatch_trig_rope_(e4m3, nk_e4m3_t)

    // Horizontal reductions: minmax (min + max with indices)
    nk_dispatch_reduce_minmax_(f64, nk_f64_t, nk_f64_t)
nk_dispatch_reduce_rmsnorm_(f32, nk_f32_t) nk_dispatch_reduce_rmsnorm_(bf16, nk_bf16_t)
    nk_dispatch_reduce_minmax_(f32, nk_f32_t, nk_f32_t)
nk_dispatch_reduce_minmax_(bf16, nk_bf16_t, nk_bf16_t)
nk_dispatch_reduce_minmax_(f16, nk_f16_t, nk_f16_t)
nk_dispatch_reduce_minmax_(e5m2, nk_e5m2_t, nk_e5m2_t)
nk_dispatch_reduce_minmax_(e4m3, nk_e4m3_t, nk_e4m3_t)
nk_dispatch_reduce_minmax_(e3m2, nk_e3m2_t, nk_e3m2_t)
nk_dispatch_reduce_minmax_(e2m3, nk_e2m3_t, nk_e2m3_t)
nk_dispatch_reduce_minmax_(i64, nk_i64_t, nk_i64_t)
nk_dispatch_reduce_minmax_(i32, nk_i32_t, nk_i32_t)
nk_dispatch_reduce_minmax_(i16, nk_i16_t, nk_i16_t)
nk_dispatch_reduce_minmax_(i8, nk_i8_t, nk_i8_t)
nk_dispatch_reduce_minmax_(i4, nk_i4x2_t, nk_i8_t)
nk_dispatch_reduce_minmax_(u64, nk_u64_t, nk_u64_t)
nk_dispatch_reduce_minmax_(u32, nk_u32_t, nk_u32_t)
nk_dispatch_reduce_minmax_(u16, nk_u16_t, nk_u16_t)
nk_dispatch_reduce_minmax_(u8, nk_u8_t, nk_u8_t)
nk_dispatch_reduce_minmax_(u4, nk_u4x2_t, nk_u8_t)
nk_dispatch_reduce_minmax_(u1, nk_u1x8_t, nk_u8_t)

// Dots packed sizes
nk_dispatch_cross_pack_size_(dots, f64, f64, f64)
nk_dispatch_cross_pack_size_(dots, f32, f32, f32)
nk_dispatch_cross_pack_size_(dots, bf16, bf16, f32)
nk_dispatch_cross_pack_size_(dots, f16, f16, f32)
nk_dispatch_cross_pack_size_(dots, e5m2, e5m2, f32)
nk_dispatch_cross_pack_size_(dots, e4m3, e4m3, f32)
nk_dispatch_cross_pack_size_(dots, e3m2, e3m2, f32)
nk_dispatch_cross_pack_size_(dots, e2m3, e2m3, f32)
nk_dispatch_cross_pack_size_(dots, i8, i8, i32)
nk_dispatch_cross_pack_size_(dots, i4, i4x2, i32)
nk_dispatch_cross_pack_size_(dots, u8, u8, u32)
nk_dispatch_cross_pack_size_(dots, u4, u4x2, u32)
nk_dispatch_cross_pack_size_(dots, u1, u1x8, u32)

nk_dispatch_cross_packed_shape_(dots, f64, f64, f64) nk_dispatch_cross_packed_shape_(dots, f32, f32, f32)
    nk_dispatch_cross_packed_shape_(dots, bf16, bf16, f32) nk_dispatch_cross_packed_shape_(dots, f16, f16, f32)
        nk_dispatch_cross_packed_shape_(dots, e5m2, e5m2, f32) nk_dispatch_cross_packed_shape_(dots, e4m3, e4m3, f32)
            nk_dispatch_cross_packed_shape_(dots, e3m2, e3m2, f32)
                nk_dispatch_cross_packed_shape_(dots, e2m3, e2m3, f32)
                    nk_dispatch_cross_packed_shape_(dots, i8, i8, i32)
                        nk_dispatch_cross_packed_shape_(dots, i4, i4x2, i32)
                            nk_dispatch_cross_packed_shape_(dots, u8, u8, u32)
                                nk_dispatch_cross_packed_shape_(dots, u4, u4x2, u32)
                                    nk_dispatch_cross_packed_shape_(dots, u1, u1x8, u32)

    // Dots packing
    nk_dispatch_dots_pack_(f64, f64) nk_dispatch_dots_pack_(f32, f32) nk_dispatch_dots_pack_(bf16, bf16)
        nk_dispatch_dots_pack_(f16, f16) nk_dispatch_dots_pack_(e5m2, e5m2) nk_dispatch_dots_pack_(e4m3, e4m3)
            nk_dispatch_dots_pack_(e3m2, e3m2) nk_dispatch_dots_pack_(e2m3, e2m3) nk_dispatch_dots_pack_(i8, i8)
                nk_dispatch_dots_pack_(i4, i4x2) nk_dispatch_dots_pack_(u8, u8) nk_dispatch_dots_pack_(u4, u4x2)
                    nk_dispatch_dots_pack_(u1, u1x8)

    // Dots packed
    nk_dispatch_cross_packed_(dots, f64, f64, f64, f64)
nk_dispatch_cross_packed_(dots, f32, f32, f32, f64)
nk_dispatch_cross_packed_(dots, bf16, bf16, f32, f32)
nk_dispatch_cross_packed_(dots, f16, f16, f32, f32)
nk_dispatch_cross_packed_(dots, e5m2, e5m2, f32, f32)
nk_dispatch_cross_packed_(dots, e4m3, e4m3, f32, f32)
nk_dispatch_cross_packed_(dots, e3m2, e3m2, f32, f32)
nk_dispatch_cross_packed_(dots, e2m3, e2m3, f32, f32)
nk_dispatch_cross_packed_(dots, i8, i8, i32, i32)
nk_dispatch_cross_packed_(dots, i4, i4x2, i32, i32)
nk_dispatch_cross_packed_(dots, u8, u8, u32, u32)
nk_dispatch_cross_packed_(dots, u4, u4x2, u32, u32)
nk_dispatch_cross_packed_(dots, u1, u1x8, u32, u32)

// Dots symmetric
nk_dispatch_cross_symmetric_(dots, f64, f64, f64)
nk_dispatch_cross_symmetric_(dots, f32, f32, f64)
nk_dispatch_cross_symmetric_(dots, bf16, bf16, f32)
nk_dispatch_cross_symmetric_(dots, f16, f16, f32)
nk_dispatch_cross_symmetric_(dots, e5m2, e5m2, f32)
nk_dispatch_cross_symmetric_(dots, e4m3, e4m3, f32)
nk_dispatch_cross_symmetric_(dots, e3m2, e3m2, f32)
nk_dispatch_cross_symmetric_(dots, e2m3, e2m3, f32)
nk_dispatch_cross_symmetric_(dots, i8, i8, i32)
nk_dispatch_cross_symmetric_(dots, i4, i4x2, i32)
nk_dispatch_cross_symmetric_(dots, u8, u8, u32)
nk_dispatch_cross_symmetric_(dots, u4, u4x2, u32)
nk_dispatch_cross_symmetric_(dots, u1, u1x8, u32)

// Sets packed
nk_dispatch_cross_packed_(hammings, u1, u1x8, u32, u32)
nk_dispatch_cross_packed_(jaccards, u1, u1x8, f32, f32)

// Sets symmetric
nk_dispatch_cross_symmetric_(hammings, u1, u1x8, u32)
nk_dispatch_cross_symmetric_(jaccards, u1, u1x8, f32)

// Angulars packed
nk_dispatch_cross_packed_(angulars, f64, f64, f64, f64)
nk_dispatch_cross_packed_(angulars, f32, f32, f32, f64)
nk_dispatch_cross_packed_(angulars, bf16, bf16, f32, f32)
nk_dispatch_cross_packed_(angulars, f16, f16, f32, f32)
nk_dispatch_cross_packed_(angulars, e5m2, e5m2, f32, f32)
nk_dispatch_cross_packed_(angulars, e4m3, e4m3, f32, f32)
nk_dispatch_cross_packed_(angulars, e3m2, e3m2, f32, f32)
nk_dispatch_cross_packed_(angulars, e2m3, e2m3, f32, f32)
nk_dispatch_cross_packed_(angulars, i8, i8, i32, f32)
nk_dispatch_cross_packed_(angulars, i4, i4x2, i32, f32)
nk_dispatch_cross_packed_(angulars, u8, u8, u32, f32)
nk_dispatch_cross_packed_(angulars, u4, u4x2, u32, f32)

// Angulars symmetric
nk_dispatch_cross_symmetric_(angulars, f64, f64, f64)
nk_dispatch_cross_symmetric_(angulars, f32, f32, f64)
nk_dispatch_cross_symmetric_(angulars, bf16, bf16, f32)
nk_dispatch_cross_symmetric_(angulars, f16, f16, f32)
nk_dispatch_cross_symmetric_(angulars, e5m2, e5m2, f32)
nk_dispatch_cross_symmetric_(angulars, e4m3, e4m3, f32)
nk_dispatch_cross_symmetric_(angulars, e3m2, e3m2, f32)
nk_dispatch_cross_symmetric_(angulars, e2m3, e2m3, f32)
nk_dispatch_cross_symmetric_(angulars, i8, i8, f32)
nk_dispatch_cross_symmetric_(angulars, i4, i4x2, f32)
nk_dispatch_cross_symmetric_(angulars, u8, u8, f32)
nk_dispatch_cross_symmetric_(angulars, u4, u4x2, f32)

// Euclideans packed
nk_dispatch_cross_packed_(euclideans, f64, f64, f64, f64)
nk_dispatch_cross_packed_(euclideans, f32, f32, f32, f64)
nk_dispatch_cross_packed_(euclideans, bf16, bf16, f32, f32)
nk_dispatch_cross_packed_(euclideans, f16, f16, f32, f32)
nk_dispatch_cross_packed_(euclideans, e5m2, e5m2, f32, f32)
nk_dispatch_cross_packed_(euclideans, e4m3, e4m3, f32, f32)
nk_dispatch_cross_packed_(euclideans, e3m2, e3m2, f32, f32)
nk_dispatch_cross_packed_(euclideans, e2m3, e2m3, f32, f32)
nk_dispatch_cross_packed_(euclideans, i8, i8, i32, f32)
nk_dispatch_cross_packed_(euclideans, i4, i4x2, i32, f32)
nk_dispatch_cross_packed_(euclideans, u8, u8, u32, f32)
nk_dispatch_cross_packed_(euclideans, u4, u4x2, u32, f32)

// Euclideans symmetric
nk_dispatch_cross_symmetric_(euclideans, f64, f64, f64)
nk_dispatch_cross_symmetric_(euclideans, f32, f32, f64)
nk_dispatch_cross_symmetric_(euclideans, bf16, bf16, f32)
nk_dispatch_cross_symmetric_(euclideans, f16, f16, f32)
nk_dispatch_cross_symmetric_(euclideans, e5m2, e5m2, f32)
nk_dispatch_cross_symmetric_(euclideans, e4m3, e4m3, f32)
nk_dispatch_cross_symmetric_(euclideans, e3m2, e3m2, f32)
nk_dispatch_cross_symmetric_(euclideans, e2m3, e2m3, f32)
nk_dispatch_cross_symmetric_(euclideans, i8, i8, f32)
nk_dispatch_cross_symmetric_(euclideans, i4, i4x2, f32)
nk_dispatch_cross_symmetric_(euclideans, u8, u8, f32)
nk_dispatch_cross_symmetric_(euclideans, u4, u4x2, f32)

// MaxSim packed sizes
nk_dispatch_cross_pack_size_(maxsim, f32, f32, f32)
nk_dispatch_cross_pack_size_(maxsim, bf16, bf16, f32)
nk_dispatch_cross_pack_size_(maxsim, f16, f16, f32)

nk_dispatch_cross_packed_shape_(maxsim, f32, f32, f32) nk_dispatch_cross_packed_shape_(maxsim, bf16, bf16, f32)
    nk_dispatch_cross_packed_shape_(maxsim, f16, f16, f32)

    // MaxSim packing
    nk_dispatch_cross_pack_(maxsim, f32, f32, f32)
nk_dispatch_cross_pack_(maxsim, bf16, bf16, f32)
nk_dispatch_cross_pack_(maxsim, f16, f16, f32)

// MaxSim packed scoring
nk_dispatch_maxsim_packed_(f32, f64)
nk_dispatch_maxsim_packed_(bf16, f32)
nk_dispatch_maxsim_packed_(f16, f32)

// Attention packed KV sizes
nk_dispatch_attention_pack_size_(bf16)
nk_dispatch_attention_pack_size_(e4m3)
nk_dispatch_attention_pack_size_(i8)

nk_dispatch_attention_packed_shape_(bf16) nk_dispatch_attention_packed_shape_(e4m3)
    nk_dispatch_attention_packed_shape_(i8)

    // Attention KV packing
    nk_dispatch_attention_pack_(bf16)
nk_dispatch_attention_pack_(e4m3)
nk_dispatch_attention_pack_(i8)

// Attention computation
nk_dispatch_attention_packed_(bf16)
nk_dispatch_attention_packed_(e4m3)
nk_dispatch_attention_packed_(i8)

NK_API_RUNTIME int nk_uses_runtime_dispatch(void) { return 1; }
NK_API_RUNTIME int nk_configure_thread(nk_capability_t c) { return nk_configure_thread_(c); }

NK_API_RUNTIME void nk_cast(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    nk_dispatch_table.cast(from, from_type, n, to, to_type);
}

NK_API_RUNTIME void nk_cast_block_scaled(                                                     //
    void const *from, void const *from_scales,                                                //
    nk_scalar_buffer_t const *from_tensor_scale, nk_block_scaled_format_t const *from_format, //
    void *to, void *to_scales,                                                                //
    nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format,           //
    nk_size_t count) {
    nk_dispatch_table.cast_block_scaled(from, from_scales, from_tensor_scale, from_format, to, to_scales,
                                        to_tensor_scale, to_format, count);
}

// Forward declarations for dtype-specific dispatch initialization functions
void nk_dispatch_f64c_init_(nk_capability_t caps);
void nk_dispatch_f32c_init_(nk_capability_t caps);
void nk_dispatch_bf16c_init_(nk_capability_t caps);
void nk_dispatch_f16c_init_(nk_capability_t caps);
void nk_dispatch_f64_init_(nk_capability_t caps);
void nk_dispatch_f32_init_(nk_capability_t caps);
void nk_dispatch_bf16_init_(nk_capability_t caps);
void nk_dispatch_f16_init_(nk_capability_t caps);
void nk_dispatch_e5m2_init_(nk_capability_t caps);
void nk_dispatch_e4m3_init_(nk_capability_t caps);
void nk_dispatch_e3m2_init_(nk_capability_t caps);
void nk_dispatch_e2m3_init_(nk_capability_t caps);
void nk_dispatch_i64_init_(nk_capability_t caps);
void nk_dispatch_i32_init_(nk_capability_t caps);
void nk_dispatch_i16_init_(nk_capability_t caps);
void nk_dispatch_i8_init_(nk_capability_t caps);
void nk_dispatch_i4_init_(nk_capability_t caps);
void nk_dispatch_u64_init_(nk_capability_t caps);
void nk_dispatch_u32_init_(nk_capability_t caps);
void nk_dispatch_u16_init_(nk_capability_t caps);
void nk_dispatch_u8_init_(nk_capability_t caps);
void nk_dispatch_u4_init_(nk_capability_t caps);
void nk_dispatch_u1_init_(nk_capability_t caps);
void nk_dispatch_cast_init_(nk_capability_t caps);
void nk_dispatch_math_init_(nk_capability_t caps);

NK_HELPER_INLINE void nk_dispatch_table_build_(nk_capability_t caps) {
    nk_dispatch_table.enabled = caps;
    nk_dispatch_f64c_init_(caps);
    nk_dispatch_f32c_init_(caps);
    nk_dispatch_bf16c_init_(caps);
    nk_dispatch_f16c_init_(caps);
    nk_dispatch_f64_init_(caps);
    nk_dispatch_f32_init_(caps);
    nk_dispatch_bf16_init_(caps);
    nk_dispatch_f16_init_(caps);
    nk_dispatch_e5m2_init_(caps);
    nk_dispatch_e4m3_init_(caps);
    nk_dispatch_e3m2_init_(caps);
    nk_dispatch_e2m3_init_(caps);
    nk_dispatch_i64_init_(caps);
    nk_dispatch_i32_init_(caps);
    nk_dispatch_i16_init_(caps);
    nk_dispatch_i8_init_(caps);
    nk_dispatch_i4_init_(caps);
    nk_dispatch_u64_init_(caps);
    nk_dispatch_u32_init_(caps);
    nk_dispatch_u16_init_(caps);
    nk_dispatch_u8_init_(caps);
    nk_dispatch_u4_init_(caps);
    nk_dispatch_u1_init_(caps);
    nk_dispatch_cast_init_(caps);
    nk_dispatch_math_init_(caps);
}

/**
 *  @brief  Points dispatch at everything available. Idempotent, and safe to call from anywhere.
 *
 *  Runs from the library constructor and again from every entry point that needs the table, so a
 *  host that never runs constructors still gets a usable one. A built table always retains
 *  @b nk_cap_serial_k, so a zero @b enabled on the zero-initialized global means "not built yet"
 *  and no separate sentinel is needed.
 */
NK_HELPER_INLINE void nk_initialize_(void) {
    if (nk_dispatch_table.enabled != 0) return;
    nk_dispatch_table_build_(nk_capabilities_available());
}

/**
 *  @brief  The capabilities this CPU can execute, probed once.
 *
 *  Probed once because the probe is not repeatable, not merely because it is slow: on Arm64 it
 *  swaps the process-wide `SIGILL` disposition and `siglongjmp`s through a global buffer to test
 *  whether `MRS` traps. The answer cannot change mid-process anyway.
 */
NK_API_RUNTIME nk_capability_t nk_capabilities_detected(void) {
    static nk_capability_t cached = nk_cap_any_k;
    if (cached == nk_cap_any_k) cached = nk_capabilities_detected_();
    return cached;
}
NK_API_RUNTIME nk_capability_t nk_capabilities_compiled(void) { return nk_capabilities_compiled_(); }
NK_API_RUNTIME nk_capability_t nk_capabilities_available(void) {
    return nk_capabilities_detected() & nk_capabilities_compiled_();
}
NK_API_RUNTIME nk_capability_t nk_capabilities_enabled(void) {
    nk_initialize_();
    return nk_dispatch_table.enabled;
}

NK_API_RUNTIME void nk_capabilities_restrict(nk_capability_t caps) {
    nk_initialize_();
    // Clamped to `available`, and the serial fallback is never dropped, so dispatch can never be
    // pointed at a kernel that was not compiled in or that this CPU cannot execute.
    nk_dispatch_table_build_((caps & nk_capabilities_available()) | nk_cap_serial_k);
}
NK_API_RUNTIME void nk_capabilities_enable(nk_capability_t caps) {
    nk_capabilities_restrict(nk_capabilities_enabled() | caps);
}
NK_API_RUNTIME void nk_capabilities_disable(nk_capability_t caps) {
    nk_capabilities_restrict(nk_capabilities_enabled() & ~caps);
}

NK_API_RUNTIME void nk_find_kernel_punned( //
    nk_kernel_kind_t kind,                 //
    nk_dtype_t dtype,                      //
    nk_kernel_punned_t *kernel_output,     //
    nk_capability_t *capability_output) {

    // The enabled mask is the one the table was built from, so the kernel found here and the one
    // the direct entry points call are always the same. Initializing here also keeps the lookup
    // correct for hosts that never run library constructors.
    nk_initialize_();
    nk_capability_t const viable = nk_dispatch_table.enabled;

    // Modern compilers abso-freaking-lutely love optimizing-out my logic!
    // Just marking the variables as `volatile` is not enough, so we have
    // to add inline assembly to further discourage them!
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif

    nk_kernel_punned_t *m = kernel_output;
    nk_capability_t *c = capability_output;

    switch (dtype) {

    case nk_f64c_k: nk_dispatch_f64c_find_(viable, kind, m, c); return;
    case nk_f32c_k: nk_dispatch_f32c_find_(viable, kind, m, c); return;
    case nk_bf16c_k: nk_dispatch_bf16c_find_(viable, kind, m, c); return;
    case nk_f16c_k: nk_dispatch_f16c_find_(viable, kind, m, c); return;

    case nk_f64_k: nk_dispatch_f64_find_(viable, kind, m, c); return;
    case nk_f32_k: nk_dispatch_f32_find_(viable, kind, m, c); return;
    case nk_bf16_k: nk_dispatch_bf16_find_(viable, kind, m, c); return;
    case nk_f16_k: nk_dispatch_f16_find_(viable, kind, m, c); return;

    case nk_e5m2_k: nk_dispatch_e5m2_find_(viable, kind, m, c); return;
    case nk_e4m3_k: nk_dispatch_e4m3_find_(viable, kind, m, c); return;
    case nk_e3m2_k: nk_dispatch_e3m2_find_(viable, kind, m, c); return;
    case nk_e2m3_k: nk_dispatch_e2m3_find_(viable, kind, m, c); return;

    case nk_i64_k: nk_dispatch_i64_find_(viable, kind, m, c); return;
    case nk_i32_k: nk_dispatch_i32_find_(viable, kind, m, c); return;
    case nk_i16_k: nk_dispatch_i16_find_(viable, kind, m, c); return;
    case nk_i8_k: nk_dispatch_i8_find_(viable, kind, m, c); return;
    case nk_i4_k: nk_dispatch_i4_find_(viable, kind, m, c); return;

    case nk_u64_k: nk_dispatch_u64_find_(viable, kind, m, c); return;
    case nk_u32_k: nk_dispatch_u32_find_(viable, kind, m, c); return;
    case nk_u16_k: nk_dispatch_u16_find_(viable, kind, m, c); return;
    case nk_u8_k: nk_dispatch_u8_find_(viable, kind, m, c); return;
    case nk_u4_k: nk_dispatch_u4_find_(viable, kind, m, c); return;
    case nk_u1_k: nk_dispatch_u1_find_(viable, kind, m, c); return;

    case nk_dtype_unknown_k: nk_dispatch_cast_find_(viable, kind, m, c); return;
    default: break;
    }

    // Replace with zeros if no suitable implementation was found
    *m = (nk_kernel_punned_t)0;
    *c = (nk_capability_t)0;

    // Modern compilers abso-freaking-lutely love optimizing-out my logic!
    // Just marking the variables as `volatile` is not enough, so we have
    // to add inline assembly to further discourage them!
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

// Auto-initialization for dynamic libraries - ensures dispatch table is populated on library load
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor)) static void nk_auto_init(void) { nk_initialize_(); }
#elif defined(_MSC_VER)
static void nk_auto_init(void);
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) static void (*nk_auto_init_ptr)(void) = nk_auto_init;
static void nk_auto_init(void) { nk_initialize_(); }
#ifdef _WIN32
#include <windows.h>
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    nk_unused_(hinstDLL);
    nk_unused_(lpReserved);
    if (fdwReason == DLL_PROCESS_ATTACH) nk_auto_init();
    return TRUE;
}
#endif
#endif

#ifdef __cplusplus
}
#endif
