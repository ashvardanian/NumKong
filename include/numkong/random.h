/**
 *  @file include/numkong/random.h
 *  @author Ash Vardanian
 *  @date January 11, 2026
 *  @brief SIMD-accelerated Pseudo-Random Number Generators.
 *
 *  Implements following statistical distributions
 *
 *  - Uniform Distribution
 *  - Gaussian / Normal Distribution
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
 *  @section random_usage Usage and Benefits
 *
 *  @section random_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NUMKONG_RANDOM_H
#define NUMKONG_RANDOM_H

#include "numkong/types.h"
#include "numkong/cast.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#endif // NUMKONG_RANDOM_H
