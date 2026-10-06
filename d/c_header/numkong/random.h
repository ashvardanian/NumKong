/**
 *  @brief SIMD-accelerated Pseudo-Random Number Generators.
 *  @file include/numkong/random.h
 *  @author Ash Vardanian
 *  @date January 11, 2026
 *
 *  Implements following statistical distributions
 *
 *  - Uniform Distribution
 *  - Gaussian (Normal) Distribution
 *
 *  For dtypes:
 *
 *  - 64-bit floating point numbers
 *  - 32-bit floating point numbers
 *  - 16-bit floating point numbers
 *  - 16-bit brain-floating point numbers
 *  - 8-bit floating point numbers
 *  - 8-bit integers
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SSVE
 *  - x86: Haswell, Ice Lake, Skylake, Genoa
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
#ifndef NK_RANDOM_H
#define NK_RANDOM_H

#include "numkong/types.h"
#include "numkong/cast.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#endif // NK_RANDOM_H
