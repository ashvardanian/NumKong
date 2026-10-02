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

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "numkong/types.h"
#include "numkong/cast.h"

#if defined(__cplusplus)
extern "C" {
#endif // defined(__cplusplus)

/** @brief Stateful generator used by the language bindings and native callers. */
typedef struct nk_random_generator_t {
    uint64_t state;
} nk_random_generator_t;

/** @brief Initialize a generator from an explicit seed. */
NK_PUBLIC void nk_random_seed(nk_random_generator_t *generator, uint64_t seed) {
    generator->state = seed;
}

/** @brief Advance the generator and return a deterministic 64-bit value. */
NK_PUBLIC uint64_t nk_random_next_u64(nk_random_generator_t *generator) {
    uint64_t value = (generator->state += UINT64_C(0x9E3779B97F4A7C15));
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

/** @brief Return a uniformly distributed double in [0, 1). */
NK_PUBLIC double nk_random_uniform_f64(nk_random_generator_t *generator) {
    return (double)(nk_random_next_u64(generator) >> 11) * 0x1.0p-53;
}

/** @brief Return a uniformly distributed float in [0, 1). */
NK_PUBLIC float nk_random_uniform_f32(nk_random_generator_t *generator) {
    return (float)((double)(nk_random_next_u64(generator) >> 40) * 0x1.0p-24);
}

/** @brief Return a uniformly distributed integer in [0, bound). */
NK_PUBLIC uint64_t nk_random_bounded_u64(nk_random_generator_t *generator, uint64_t bound) {
    if (!bound) return nk_random_next_u64(generator);
    uint64_t const threshold = (uint64_t)(0 - bound) % bound;
    uint64_t value;
    do { value = nk_random_next_u64(generator); } while (value < threshold);
    return value % bound;
}

/** @brief Return a standard normal sample using the Box-Muller transform. */
NK_PUBLIC double nk_random_standard_normal_f64(nk_random_generator_t *generator) {
    double const first = ((double)(nk_random_next_u64(generator) >> 11) + 1.0) * 0x1.0p-53;
    double const second = nk_random_uniform_f64(generator);
    double const radius = sqrt(-2.0 * log(first));
    double const angle = 2.0 * 3.14159265358979323846264338327950288 * second;
    return radius * cos(angle);
}

/** @brief Return a normal sample with the requested location and scale. */
NK_PUBLIC double nk_random_normal_f64(nk_random_generator_t *generator, double location, double scale) {
    return location + scale * nk_random_standard_normal_f64(generator);
}

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#endif // NK_RANDOM_H
