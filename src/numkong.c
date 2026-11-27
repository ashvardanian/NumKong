/**
 *  @file src/numkong.c
 *  @author Ash Vardanian
 *  @date March 13, 2024
 *  @brief The NumKong library's capability queries, its unified memory and stream dispatch, and the
 *      finder over every family's finder.
 */
#include "numkong/numkong.h"

#include "numkong/cpu.h" // `nk_capabilities_detected_cpu_`, `nk_capabilities_enable_amx_`

#include "dispatch.h" // `nk_capability_group_of_`, `nk_capabilities_enabled_cpu_`

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
    case nk_kernel_sqeuclideans3d_k:
    case nk_kernel_haversines_pack_size_k:
    case nk_kernel_haversines_pack_k:
    case nk_kernel_haversines_packed_k:
    case nk_kernel_vincentys_pack_size_k:
    case nk_kernel_vincentys_pack_k:
    case nk_kernel_vincentys_packed_k:
    case nk_kernel_terrain_pack_size_k:
    case nk_kernel_terrain_pack_k:
    case nk_kernel_terrain_heights_packed_k:
    case nk_kernel_clearances_packed_k: return nk_geospatial_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_kld_k:
    case nk_kernel_jsd_k: return nk_probability_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_sparse_dot_k:
    case nk_kernel_sparse_intersect_k: return nk_sparse_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_each_scale_k:
    case nk_kernel_each_sum_k:
    case nk_kernel_each_blend_k:
    case nk_kernel_each_fma_k:
    case nk_kernel_each_swiglu_k:
    case nk_kernel_each_rmsnorm_k:
    case nk_kernel_each_rmscast_k: return nk_each_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_trig_sin_k:
    case nk_kernel_trig_cos_k:
    case nk_kernel_trig_atan_k: return nk_trigonometry_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_reduce_moments_k:
    case nk_kernel_reduce_minmax_k: return nk_reduce_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_dots_pack_size_k:
    case nk_kernel_dots_pack_k:
    case nk_kernel_dots_packed_k:
    case nk_kernel_dots_packed_shape_k:
    case nk_kernel_dots_packed_workspace_size_k:
    case nk_kernel_dots_symmetric_workspace_size_k:
    case nk_kernel_dots_symmetric_k: return nk_dots_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_hammings_packed_k:
    case nk_kernel_hammings_symmetric_k:
    case nk_kernel_jaccards_packed_k:
    case nk_kernel_hammings_packed_workspace_size_k:
    case nk_kernel_hammings_symmetric_workspace_size_k:
    case nk_kernel_jaccards_packed_workspace_size_k:
    case nk_kernel_jaccards_symmetric_workspace_size_k:
    case nk_kernel_jaccards_symmetric_k: return nk_sets_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_angulars_packed_k:
    case nk_kernel_angulars_symmetric_k:
    case nk_kernel_euclideans_packed_k:
    case nk_kernel_euclideans_symmetric_k:
    case nk_kernel_angulars_packed_workspace_size_k:
    case nk_kernel_angulars_symmetric_workspace_size_k:
    case nk_kernel_euclideans_packed_workspace_size_k:
    case nk_kernel_euclideans_symmetric_workspace_size_k:
        return nk_spatials_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_maxsim_pack_size_k:
    case nk_kernel_maxsim_pack_k:
    case nk_kernel_maxsim_packed_k:
    case nk_kernel_maxsim_packed_shape_k: return nk_maxsim_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_shapes_pack_size_k:
    case nk_kernel_shapes_pack_k:
    case nk_kernel_shapes_pack_normalized_k:
    case nk_kernel_shapes_packed_shape_k:
    case nk_kernel_shapes_unpack_k:
    case nk_kernel_chamfers_workspace_size_k:
    case nk_kernel_chamfers_packed_k:
    case nk_kernel_wassersteinprocrustes_workspace_size_k:
    case nk_kernel_wassersteinprocrustes_packed_k:
    case nk_kernel_gromovwassersteins_workspace_size_k:
    case nk_kernel_gromovwassersteins_packed_k:
    case nk_kernel_hausdorffs_packed_k:
    case nk_kernel_rmsds_packed_k: return nk_shapes_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_attention_pack_size_k:
    case nk_kernel_attention_pack_k:
    case nk_kernel_attention_packed_k:
    case nk_kernel_attention_packed_gradients_k:
    case nk_kernel_attention_packed_workspace_size_k:
    case nk_kernel_attention_packed_gradients_workspace_size_k:
    case nk_kernel_attention_packed_shape_k:
    case nk_kernel_attention_rope_k: return nk_attention_find_kernel(kind, dtype, capabilities, kernel, capability);
    case nk_kernel_cast_k:
    default: *kernel = NUMKONG_NULL, *capability = 0; return nk_missing_kernel_k;
    }
}

/*  Every pick clamps a CPU mask to it, so a caller may pass `nk_cap_cpus_k` itself. A call before
 *  the load-time fill, from another constructor, gets the serial kernels. */
nk_capability_t nk_capabilities_enabled_cpu_ = nk_cap_serial_k;

/** Fills the CPU's enabled capabilities with what it detects and this library compiled, once the
 *  AMX tile state they may need is granted, as Linux grants it to the whole process, and before
 *  the constructors of default priority in a statically linked image. */
#if !defined(_MSC_VER)
__attribute__((constructor(101)))
#endif
static void nk_capabilities_enable_cpu_(void) {
    nk_capabilities_enabled_cpu_ = nk_capabilities_enable_amx_(nk_capabilities_detected_cpu_() &
                                                               nk_capabilities_compiled_cpu_mask());
}

/*  MSVC has no constructor attribute, but runs every pointer in this section as the library loads,
 *  and `/include` keeps `/OPT:REF` from dropping the unreferenced pointer. */
#if defined(_MSC_VER)
#pragma section(".CRT$XCU", read)
#pragma comment(linker, "/include:nk_capabilities_enable_cpu_pointer_")
__declspec(allocate(".CRT$XCU")) void (*nk_capabilities_enable_cpu_pointer_)(void) = nk_capabilities_enable_cpu_;
#endif

NUMKONG_API nk_status_t nk_capabilities_detected_cpu(nk_capability_t *capabilities) {
    *capabilities = nk_capabilities_detected_cpu_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_capabilities_compiled_cpu(nk_capability_t *capabilities) {
    *capabilities = nk_capabilities_compiled_cpu_mask();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_capabilities_enabled_cpu(nk_capability_t *capabilities) {
    *capabilities = nk_capabilities_enabled_cpu_;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_configure_thread_cpu(nk_capability_t capabilities) {
    return nk_configure_thread_cpu_(capabilities);
}

NUMKONG_API nk_size_t nk_capabilities_name(nk_capability_t capabilities, char *buffer, nk_size_t capacity) {
    return nk_capabilities_name_(capabilities, buffer, capacity);
}

NUMKONG_API char const *nk_status_name(nk_status_t status) { return nk_status_name_(status); }

/*  Each vendor's other producers come from `src/capabilities/<vendor>.*` when CMake compiles
 *  its kernels, or from `src/missing/<vendor>.c`, answering `nk_missing_gpu_k`, when not. */
NUMKONG_API nk_status_t nk_capabilities_enabled_cuda(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_capability_t compiled = 0;
    nk_status_t const status = nk_capabilities_detected_cuda(ordinal, capabilities);
    nk_unused_(nk_capabilities_compiled_cuda(ordinal, &compiled));
    *capabilities &= compiled;
    return status;
}

NUMKONG_API nk_status_t nk_capabilities_enabled_rocm(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_capability_t compiled = 0;
    nk_status_t const status = nk_capabilities_detected_rocm(ordinal, capabilities);
    nk_unused_(nk_capabilities_compiled_rocm(ordinal, &compiled));
    *capabilities &= compiled;
    return status;
}

NUMKONG_API nk_status_t nk_capabilities_enabled_metal(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_capability_t compiled = 0;
    nk_status_t const status = nk_capabilities_detected_metal(ordinal, capabilities);
    nk_unused_(nk_capabilities_compiled_metal(ordinal, &compiled));
    *capabilities &= compiled;
    return status;
}

NUMKONG_API nk_status_t nk_stream_synchronize_serial(nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_allocator_init_unified_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_allocator_init_unified_serial(allocator);
    case nk_capability_group_cuda_k: return nk_allocator_init_unified_cuda(allocator);
    case nk_capability_group_rocm_k: return nk_allocator_init_unified_rocm(allocator);
    case nk_capability_group_metal_k: return nk_allocator_init_unified_metal(allocator);
    default: return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_allocator_init_device_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_missing_kernel_k;
    case nk_capability_group_cuda_k: return nk_allocator_init_device_cuda(allocator);
    case nk_capability_group_rocm_k: return nk_allocator_init_device_rocm(allocator);
    case nk_capability_group_metal_k: return nk_allocator_init_unified_metal(allocator);
    default: return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_allocator_init_pinned_best(nk_allocator_t *allocator, nk_capability_t capabilities) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_missing_kernel_k;
    case nk_capability_group_cuda_k: return nk_allocator_init_pinned_cuda(allocator);
    case nk_capability_group_rocm_k: return nk_allocator_init_pinned_rocm(allocator);
    case nk_capability_group_metal_k: return nk_allocator_init_unified_metal(allocator);
    default: return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_memory_allocate_unified_best(nk_size_t bytes, void **pointer, nk_capability_t capabilities,
                                                        nk_stream_t stream) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_memory_allocate_unified_serial(bytes, pointer, stream);
    case nk_capability_group_cuda_k: return nk_memory_allocate_unified_cuda(bytes, pointer, stream);
    case nk_capability_group_rocm_k: return nk_memory_allocate_unified_rocm(bytes, pointer, stream);
    case nk_capability_group_metal_k: return nk_memory_allocate_unified_metal(bytes, pointer, stream);
    default: *pointer = NUMKONG_NULL; return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_memory_free_unified_best(void *pointer, nk_size_t bytes, nk_capability_t capabilities,
                                                    nk_stream_t stream) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_memory_free_unified_serial(pointer, bytes, stream);
    case nk_capability_group_cuda_k: return nk_memory_free_unified_cuda(pointer, bytes, stream);
    case nk_capability_group_rocm_k: return nk_memory_free_unified_rocm(pointer, bytes, stream);
    case nk_capability_group_metal_k: return nk_memory_free_unified_metal(pointer, bytes, stream);
    default: return nk_missing_gpu_k;
    }
}

NUMKONG_API nk_status_t nk_stream_synchronize_best(nk_capability_t capabilities, nk_stream_t stream) {
    switch (nk_capability_group_of_(capabilities)) {
    case nk_capability_group_cpu_k: return nk_stream_synchronize_serial(stream);
    case nk_capability_group_cuda_k: return nk_stream_synchronize_cuda(stream);
    case nk_capability_group_rocm_k: return nk_stream_synchronize_rocm(stream);
    case nk_capability_group_metal_k: return nk_stream_synchronize_metal(stream);
    default: return nk_missing_gpu_k;
    }
}

#ifdef __cplusplus
}
#endif
