/**
 *  @file c/numkong.c
 *  @author Ash Vardanian
 *  @date March 13, 2024
 *  @brief The NumKong library's capability queries, its unified memory and stream dispatch, and the
 *      finder over every family's finder.
 */
#include <stdlib.h> // `malloc`, `free`

#include <stdatomic.h> // `atomic_load`, `atomic_store`

#include "numkong/numkong.h"

#include "dispatch.h" // `nk_capability_group_of_`

#ifdef __cplusplus
extern "C" {
#endif

NUMKONG_API nk_status_t nk_find_kernel_punned(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    switch (kind) {
    case nk_kernel_dot_k:
    case nk_kernel_vdot_k: return nk_dot_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_angular_k:
    case nk_kernel_euclidean_k:
    case nk_kernel_sqeuclidean_k: return nk_spatial_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_hamming_k:
    case nk_kernel_jaccard_k: return nk_set_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_bilinear_k:
    case nk_kernel_mahalanobis_k: return nk_curved_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_haversine_k:
    case nk_kernel_vincenty_k: return nk_geospatial_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_kld_k:
    case nk_kernel_jsd_k: return nk_probability_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_rmsd_k:
    case nk_kernel_kabsch_k:
    case nk_kernel_umeyama_k: return nk_mesh_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_sparse_dot_k:
    case nk_kernel_sparse_intersect_k: return nk_sparse_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_each_scale_k:
    case nk_kernel_each_sum_k:
    case nk_kernel_each_blend_k:
    case nk_kernel_each_fma_k:
    case nk_kernel_each_swiglu_k:
    case nk_kernel_each_rmsnorm_k: return nk_each_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_trig_sin_k:
    case nk_kernel_trig_cos_k:
    case nk_kernel_trig_atan_k: return nk_trigonometry_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_reduce_moments_k:
    case nk_kernel_reduce_minmax_k: return nk_reduce_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_dots_pack_size_k:
    case nk_kernel_dots_pack_k:
    case nk_kernel_dots_packed_k:
    case nk_kernel_dots_packed_shape_k:
    case nk_kernel_dots_symmetric_k: return nk_dots_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_hammings_packed_k:
    case nk_kernel_hammings_symmetric_k:
    case nk_kernel_jaccards_packed_k:
    case nk_kernel_jaccards_symmetric_k: return nk_sets_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_angulars_packed_k:
    case nk_kernel_angulars_symmetric_k:
    case nk_kernel_euclideans_packed_k:
    case nk_kernel_euclideans_symmetric_k:
        return nk_spatials_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_maxsim_pack_size_k:
    case nk_kernel_maxsim_pack_k:
    case nk_kernel_maxsim_packed_k:
    case nk_kernel_maxsim_packed_shape_k: return nk_maxsim_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_attention_pack_size_k:
    case nk_kernel_attention_pack_k:
    case nk_kernel_attention_bidirectional_packed_k:
    case nk_kernel_attention_causal_packed_k:
    case nk_kernel_attention_packed_shape_k:
    case nk_kernel_attention_rope_k: return nk_attention_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_cast_k:
    default: *kernel = NUMKONG_NULL, *capability = 0; return nk_missing_kernel_k;
    }
}

/** The CPU's capabilities, probed on the first query: the C++ wrappers ask on every call, and a
 *  query can be a system call. Racing threads probe the same CPU and store the same word, never
 *  zero once filled, as it always holds @c nk_cap_serial_k. The GPU queries keep nothing, as their
 *  runtimes answer from their own state. */
static _Atomic nk_capability_t nk_cpu_detected_;

NUMKONG_API nk_status_t nk_cpu_capabilities_detected(nk_capability_t *capabilities) {
    nk_capability_t detected = atomic_load(&nk_cpu_detected_);
    if (!detected) atomic_store(&nk_cpu_detected_, detected = nk_cpu_capabilities_detected_());
    *capabilities = detected;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_cpu_capabilities_compiled(nk_capability_t *capabilities) {
    *capabilities = nk_cpu_capabilities_compiled_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_cpu_capabilities_enabled(nk_capability_t *capabilities) {
    nk_cpu_capabilities_detected(capabilities);
    *capabilities &= nk_cpu_capabilities_compiled_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_cpu_configure_thread(nk_capability_t capabilities) {
    return nk_cpu_configure_thread_(capabilities);
}

NUMKONG_API nk_size_t nk_capabilities_name(nk_capability_t capabilities, char *buffer, nk_size_t capacity) {
    return nk_capabilities_name_(capabilities, buffer, capacity);
}

NUMKONG_API char const *nk_status_name(nk_status_t status) { return nk_status_name_(status); }

/*  With CUDA kernels in the library, `c/cuda/cuda.cu` counts, probes and opens streams on the
 *  devices instead. */
#if !NUMKONG_ARCH_CUDA_
NUMKONG_API nk_status_t nk_cuda_count_devices(nk_size_t *count) {
    *count = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_cuda_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_cuda_capabilities_detected_(ordinal, capabilities);
}
NUMKONG_API nk_status_t nk_cuda_stream_init(nk_size_t ordinal, void **stream) {
    return nk_cuda_stream_init_(ordinal, stream);
}
NUMKONG_API nk_status_t nk_cuda_stream_free(void *stream) { return nk_cuda_stream_free_(stream); }
#endif

NUMKONG_API nk_status_t nk_cuda_capabilities_compiled(nk_capability_t *capabilities) {
    *capabilities = nk_cuda_capabilities_compiled_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_cuda_capabilities_enabled(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_status_t const status = nk_cuda_capabilities_detected(ordinal, capabilities);
    *capabilities &= nk_cuda_capabilities_compiled_();
    return status;
}

/*  With ROCm kernels in the library, `c/rocm/rocm.hip` counts, probes and opens streams on the
 *  devices instead. */
#if !NUMKONG_ARCH_ROCM_
NUMKONG_API nk_status_t nk_rocm_count_devices(nk_size_t *count) {
    *count = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_rocm_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_rocm_capabilities_detected_(ordinal, capabilities);
}
NUMKONG_API nk_status_t nk_rocm_stream_init(nk_size_t ordinal, void **stream) {
    return nk_rocm_stream_init_(ordinal, stream);
}
NUMKONG_API nk_status_t nk_rocm_stream_free(void *stream) { return nk_rocm_stream_free_(stream); }
#endif

NUMKONG_API nk_status_t nk_rocm_capabilities_compiled(nk_capability_t *capabilities) {
    *capabilities = nk_rocm_capabilities_compiled_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rocm_capabilities_enabled(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_status_t const status = nk_rocm_capabilities_detected(ordinal, capabilities);
    *capabilities &= nk_rocm_capabilities_compiled_();
    return status;
}

/*  With Metal kernels in the library, `c/metal/metal.c` counts, probes and opens streams on the
 *  devices instead. */
#if !NUMKONG_WITH_METAL
NUMKONG_API nk_status_t nk_metal_count_devices(nk_size_t *count) {
    *count = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_metal_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    return nk_metal_capabilities_detected_(ordinal, capabilities);
}
NUMKONG_API nk_status_t nk_metal_stream_init(nk_size_t ordinal, void **stream) {
    return nk_metal_stream_init_(ordinal, stream);
}
NUMKONG_API nk_status_t nk_metal_stream_free(void *stream) { return nk_metal_stream_free_(stream); }
#endif

NUMKONG_API nk_status_t nk_metal_capabilities_compiled(nk_capability_t *capabilities) {
    *capabilities = nk_metal_capabilities_compiled_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_metal_capabilities_enabled(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_status_t const status = nk_metal_capabilities_detected(ordinal, capabilities);
    *capabilities &= nk_metal_capabilities_compiled_();
    return status;
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_serial(nk_size_t bytes, void **pointer, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    *pointer = NUMKONG_NULL;
    if (!bytes) return nk_success_k;
    unsigned char *const allocation = bytes <= NUMKONG_SIZE_MAX - 64 ? (unsigned char *)malloc(bytes + 64)
                                                                     : (unsigned char *)NUMKONG_NULL;
    if (!allocation) return nk_bad_alloc_k;
    // `malloc` aligns to less than 64 bytes, so the byte before each block keeps its distance back
    nk_size_t const shift = 64 - ((nk_size_t)allocation & 63);
    allocation[shift - 1] = (unsigned char)shift;
    *pointer = allocation + shift;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_memory_free_unified_serial(void *pointer, nk_size_t bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_unused_(bytes);
    unsigned char *const block = (unsigned char *)pointer;
    if (block) free(block - block[-1]);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_stream_synchronize_serial(void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_best(nk_size_t bytes, void **pointer, nk_capability_t capabilities,
                                                        void *stream) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_memory_allocate_unified_serial(bytes, pointer, stream);
#if NUMKONG_ARCH_CUDA_
    case nk_capability_group_cuda_k: return nk_memory_allocate_unified_cuda(bytes, pointer, stream);
#endif
#if NUMKONG_ARCH_ROCM_
    case nk_capability_group_rocm_k: return nk_memory_allocate_unified_rocm(bytes, pointer, stream);
#endif
#if NUMKONG_WITH_METAL
    case nk_capability_group_metal_k: return nk_memory_allocate_unified_metal(bytes, pointer, stream);
#endif
    default: *pointer = NUMKONG_NULL; return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_memory_free_unified_best(void *pointer, nk_size_t bytes, nk_capability_t capabilities,
                                                    void *stream) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_memory_free_unified_serial(pointer, bytes, stream);
#if NUMKONG_ARCH_CUDA_
    case nk_capability_group_cuda_k: return nk_memory_free_unified_cuda(pointer, bytes, stream);
#endif
#if NUMKONG_ARCH_ROCM_
    case nk_capability_group_rocm_k: return nk_memory_free_unified_rocm(pointer, bytes, stream);
#endif
#if NUMKONG_WITH_METAL
    case nk_capability_group_metal_k: return nk_memory_free_unified_metal(pointer, bytes, stream);
#endif
    default: return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_stream_synchronize_best(nk_capability_t capabilities, void *stream) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_stream_synchronize_serial(stream);
#if NUMKONG_ARCH_CUDA_
    case nk_capability_group_cuda_k: return nk_stream_synchronize_cuda(stream);
#endif
#if NUMKONG_ARCH_ROCM_
    case nk_capability_group_rocm_k: return nk_stream_synchronize_rocm(stream);
#endif
#if NUMKONG_WITH_METAL
    case nk_capability_group_metal_k: return nk_stream_synchronize_metal(stream);
#endif
    default: return nk_missing_gpu_k;
    }
}

#ifdef __cplusplus
}
#endif
