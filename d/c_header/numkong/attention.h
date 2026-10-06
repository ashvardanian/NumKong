/**
 *  @brief SIMD-accelerated Transformer Attention.
 *  @file include/numkong/attention.h
 *  @author Ash Vardanian
 *  @date January 11, 2026
 *
 *  Contains following kernels:
 *
 *  - Numerically stable SoftMax
 *  - End-to-end Attention
 *
 *  For dtypes:
 *
 *  - 32-bit floating point numbers → 32-bit floats
 *  - 16-bit floating point numbers → 32-bit floats
 *  - 16-bit brain-floating point numbers → 32-bit floats
 *  - 8-bit floating point numbers → 32-bit floats
 *  - 8-bit integers → 32-bit integer
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SSVE, SME
 *  - x86: Haswell, Ice Lake, Skylake, Genoa, Sapphire
 *
 *  @section usage Usage and Benefits
 *
 *
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NK_ATTENTION_H
#define NK_ATTENTION_H

#include "numkong/types.h"
#include "numkong/dots.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#endif // NK_ATTENTION_H
