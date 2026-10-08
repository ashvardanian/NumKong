/**
 *  @file include/numkong/attention/blackwellultra.cuh
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief Ragged attention for the NVIDIA compute capability 10.3 parts, B300 and GB300.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/blackwell.cuh
 *
 *  The Blackwell kernels, whose softmax warps find each row's maximum as they load S from tensor
 *  memory: `tcgen05.ld.red`, which only Blackwell Ultra has, returns the extreme of the 32 columns
 *  it loads beside them, where Blackwell reduces them after the load.
 */
#ifndef NUMKONG_ATTENTION_BLACKWELLULTRA_CUH
#define NUMKONG_ATTENTION_BLACKWELLULTRA_CUH

#if NUMKONG_ARCH_CUDA_
#if NUMKONG_TARGET_BLACKWELLULTRA

#include "numkong/attention/blackwell.cuh" // `nk_define_attention_tma_blackwell_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

/** The largest of 32 consecutive F32 columns of this thread's lane, or their smallest when
 *  @p smallest, which `tcgen05.ld.red` reduces as it loads them. */
NUMKONG_DEVICE nk_f32_t nk_tmem_load_extreme_x32_blackwellultra_(nk_u32_t address, int smallest) {
    nk_u32_t values[32];
    nk_f32_t extreme;
    if (smallest)
        asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x32.min.f32 "                                                //
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "                        //
                     "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];\n" //
                     "tcgen05.wait::ld.sync.aligned;\n"
                     : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]), "=r"(values[4]),
                       "=r"(values[5]), "=r"(values[6]), "=r"(values[7]), "=r"(values[8]), "=r"(values[9]),
                       "=r"(values[10]), "=r"(values[11]), "=r"(values[12]), "=r"(values[13]), "=r"(values[14]),
                       "=r"(values[15]), "=r"(values[16]), "=r"(values[17]), "=r"(values[18]), "=r"(values[19]),
                       "=r"(values[20]), "=r"(values[21]), "=r"(values[22]), "=r"(values[23]), "=r"(values[24]),
                       "=r"(values[25]), "=r"(values[26]), "=r"(values[27]), "=r"(values[28]), "=r"(values[29]),
                       "=r"(values[30]), "=r"(values[31]), "=f"(extreme)
                     : "r"(address)
                     : "memory");
    else
        asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 "                                                //
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "                        //
                     "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];\n" //
                     "tcgen05.wait::ld.sync.aligned;\n"
                     : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]), "=r"(values[4]),
                       "=r"(values[5]), "=r"(values[6]), "=r"(values[7]), "=r"(values[8]), "=r"(values[9]),
                       "=r"(values[10]), "=r"(values[11]), "=r"(values[12]), "=r"(values[13]), "=r"(values[14]),
                       "=r"(values[15]), "=r"(values[16]), "=r"(values[17]), "=r"(values[18]), "=r"(values[19]),
                       "=r"(values[20]), "=r"(values[21]), "=r"(values[22]), "=r"(values[23]), "=r"(values[24]),
                       "=r"(values[25]), "=r"(values[26]), "=r"(values[27]), "=r"(values[28]), "=r"(values[29]),
                       "=r"(values[30]), "=r"(values[31]), "=f"(extreme)
                     : "r"(address)
                     : "memory");
    return extreme;
}

#pragma endregion Instructions

#pragma region BF16

nk_define_attention_pack_size_simt_(bf16, blackwellultra, 2)
nk_define_attention_packed_shape_cuda_(bf16, blackwellultra)
nk_define_attention_pack_blackwell_(bf16, blackwellultra, bf16)
nk_define_attention_tma_blackwell_(bf16, blackwellultra, bf16, nk_attention_float_threads_blackwell_k,
                                   nk_mma_f16_blackwell_, nk_mma_f16_tmem_blackwell_,
                                   nk_attention_weights_bf16_blackwell_, /*format=*/1)

/** Both BF16 backward kernels, on the Blackwell tensor-core passes when the launch gives them
 *  shared memory and the @c cuda kernels' loops when it does not. */
static __global__ void __launch_bounds__(nk_attention_backward_threads_blackwell_k, 1)
    nk_attention_backward_keys_bf16_blackwellultra_kernel_(nk_attention_backward_arguments_t arguments) {
    if (nk_dynamic_shared_bytes_ampere_()) nk_attention_backward_keys_blackwell_(&arguments);
    else nk_attention_backward_keys_cuda_(nk_bf16_k, &arguments);
}
static __global__ void __launch_bounds__(nk_attention_backward_threads_blackwell_k, 1)
    nk_attention_backward_queries_bf16_blackwellultra_kernel_(nk_attention_backward_arguments_t arguments) {
    if (nk_dynamic_shared_bytes_ampere_()) nk_attention_backward_queries_blackwell_(&arguments);
    else nk_attention_backward_queries_cuda_(nk_bf16_k, &arguments);
}
nk_define_attention_backward_cuda_(bf16, blackwellultra, bf16, nk_attention_backward_launch_blackwell_)

#pragma endregion BF16

#pragma region F16

nk_define_attention_pack_size_simt_(f16, blackwellultra, 2)
nk_define_attention_packed_shape_cuda_(f16, blackwellultra)
nk_define_attention_pack_blackwell_(f16, blackwellultra, f16)
nk_define_attention_tma_blackwell_(f16, blackwellultra, f16, nk_attention_float_threads_blackwell_k,
                                   nk_mma_f16_blackwell_, nk_mma_f16_tmem_blackwell_,
                                   nk_attention_weights_f16_blackwell_, /*format=*/0)

#pragma endregion F16

#pragma region E4M3

nk_define_attention_pack_size_simt_(e4m3, blackwellultra, 1)
nk_define_attention_packed_shape_cuda_(e4m3, blackwellultra)
nk_define_attention_pack_blackwell_(e4m3, blackwellultra, e4m3)
nk_define_attention_tma_blackwell_(e4m3, blackwellultra, e4m3, nk_attention_float_threads_blackwell_k,
                                   nk_mma_f8f6f4_blackwell_, nk_mma_f8f6f4_tmem_blackwell_,
                                   nk_attention_weights_e4m3_blackwell_, /*format=*/0)

#pragma endregion E4M3

#pragma region I8

nk_define_attention_pack_size_simt_(i8, blackwellultra, 1)
nk_define_attention_packed_shape_cuda_(i8, blackwellultra)
nk_define_attention_pack_blackwell_(i8, blackwellultra, i8)
nk_define_attention_tma_blackwell_(i8, blackwellultra, i8, nk_attention_integer_threads_blackwell_k,
                                   nk_mma_f16_blackwell_, nk_mma_f16_tmem_blackwell_,
                                   nk_attention_weights_u8_blackwell_, /*format=*/0)

#pragma endregion I8

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_BLACKWELLULTRA
#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_ATTENTION_BLACKWELLULTRA_CUH
