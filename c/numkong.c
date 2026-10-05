/**
 *  @file c/numkong.c
 *  @author Ash Vardanian
 *  @date March 13, 2024
 *  @brief The NumKong library's capability queries, its unified memory and stream dispatch, and the
 *      finder over every family's finder.
 */
#include "numkong/numkong.h"

#include "dispatch.h" // `nk_capability_group_of_`, `nk_capabilities_runnable_`

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

nk_capability_t nk_capabilities_runnable_[nk_capability_groups_k] = {nk_cap_serial_k, ~0ull, ~0ull, ~0ull};

/** Widens the CPU's runnable capabilities to what it detects and this library holds, once the AMX
 *  tile state they may need is granted, as Linux grants it to the whole process. A call before
 *  this runs, from another library's constructor, gets the serial kernels. */
#if !defined(_MSC_VER)
__attribute__((constructor))
#endif
static void nk_capabilities_widen_(void) {
    nk_capability_t const runnable = nk_cpu_capabilities_detected_() & nk_cpu_capabilities_compiled_();
#if NUMKONG_ARCH_X8664_
    nk_unused_(nk_cpu_configure_thread_x86_(runnable));
#endif
    nk_capabilities_runnable_[nk_capability_group_cpu_k] = runnable;
}

#if defined(_MSC_VER)
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) void (*nk_capabilities_widen_pointer_)(void) = nk_capabilities_widen_;
#endif

NUMKONG_API nk_status_t nk_cpu_capabilities_detected(nk_capability_t *capabilities) {
    *capabilities = nk_cpu_capabilities_detected_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_cpu_capabilities_compiled(nk_capability_t *capabilities) {
    *capabilities = nk_cpu_capabilities_compiled_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_cpu_capabilities_enabled(nk_capability_t *capabilities) {
    *capabilities = nk_capabilities_runnable_[nk_capability_group_cpu_k];
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_cpu_configure_thread(nk_capability_t capabilities) {
    return nk_cpu_configure_thread_(capabilities);
}

NUMKONG_API nk_size_t nk_capabilities_name(nk_capability_t capabilities, char *buffer, nk_size_t capacity) {
    return nk_capabilities_name_(capabilities, buffer, capacity);
}

NUMKONG_API char const *nk_status_name(nk_status_t status) { return nk_status_name_(status); }

/*  With CUDA kernels in the library, `c/target/cuda.cu` counts, probes and opens streams on the
 *  devices instead. */
#if !NUMKONG_ARCH_CUDA_
NUMKONG_API nk_status_t nk_cuda_count_devices(nk_size_t *count) {
    *count = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_cuda_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_cuda_stream_init(nk_size_t ordinal, void **stream) {
    nk_unused_(ordinal);
    *stream = NUMKONG_NULL;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_cuda_stream_free(void *stream) {
    nk_unused_(stream);
    return nk_missing_gpu_k;
}
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

/*  With ROCm kernels in the library, `c/target/rocm.hip` counts, probes and opens streams on the
 *  devices instead. */
#if !NUMKONG_ARCH_ROCM_
NUMKONG_API nk_status_t nk_rocm_count_devices(nk_size_t *count) {
    *count = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_rocm_capabilities_detected(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_rocm_stream_init(nk_size_t ordinal, void **stream) {
    nk_unused_(ordinal);
    *stream = NUMKONG_NULL;
    return nk_missing_gpu_k;
}
NUMKONG_API nk_status_t nk_rocm_stream_free(void *stream) {
    nk_unused_(stream);
    return nk_missing_gpu_k;
}
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

/*  With Metal kernels in the library, `c/target/metal.c` counts, probes and opens streams on the
 *  devices instead. */
#if !NUMKONG_ARCH_METAL_
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

NUMKONG_API nk_status_t nk_stream_synchronize_serial(void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_allocator_init_unified_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_allocator_init_unified_serial(allocator);
#if NUMKONG_ARCH_CUDA_
    case nk_capability_group_cuda_k: return nk_allocator_init_unified_cuda(allocator);
#endif
#if NUMKONG_ARCH_ROCM_
    case nk_capability_group_rocm_k: return nk_allocator_init_unified_rocm(allocator);
#endif
#if NUMKONG_ARCH_METAL_
    case nk_capability_group_metal_k: return nk_allocator_init_unified_metal(allocator);
#endif
    default: return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_allocator_init_device_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_missing_kernel_k;
#if NUMKONG_ARCH_CUDA_
    case nk_capability_group_cuda_k: return nk_allocator_init_device_cuda(allocator);
#endif
#if NUMKONG_ARCH_ROCM_
    case nk_capability_group_rocm_k: return nk_allocator_init_device_rocm(allocator);
#endif
#if NUMKONG_ARCH_METAL_
    case nk_capability_group_metal_k: return nk_allocator_init_unified_metal(allocator);
#endif
    default: return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_allocator_init_pinned_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_missing_kernel_k;
#if NUMKONG_ARCH_CUDA_
    case nk_capability_group_cuda_k: return nk_allocator_init_pinned_cuda(allocator);
#endif
#if NUMKONG_ARCH_ROCM_
    case nk_capability_group_rocm_k: return nk_allocator_init_pinned_rocm(allocator);
#endif
#if NUMKONG_ARCH_METAL_
    case nk_capability_group_metal_k: return nk_allocator_init_unified_metal(allocator);
#endif
    default: return nk_missing_gpu_k;
    }
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
#if NUMKONG_ARCH_METAL_
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
#if NUMKONG_ARCH_METAL_
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
#if NUMKONG_ARCH_METAL_
    case nk_capability_group_metal_k: return nk_stream_synchronize_metal(stream);
#endif
    default: return nk_missing_gpu_k;
    }
}

#ifdef __cplusplus
}
#endif
