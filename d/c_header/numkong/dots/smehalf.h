/**
 *  @brief SIMD-accelerated Batched Dot Products for SME FP16.
 *  @file include/numkong/dots/smehalf.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  Implements hierarchical accumulation for E2M3 and E3M2 dot products:
 *  convert to FP16, accumulate products into FP16 ZA tiles, periodically
 *  widen to FP32 to preserve precision.
 *
 *  E2M3 and E3M2 are ideal for FP16 intermediate accumulation because their
 *  limited dynamic range (max ±7.5 and ±28) guarantees no single product
 *  can overflow FP16. The only constraint is mantissa precision:
 *
 *      Format   Max Product   FP16 Headroom   Safe Accumulations
 *      E2M3     7.5² = 56     11 - 7 bits     2^4 = 16 products
 *      E3M2     28² = 784     11 - 5 bits     2^6 = 64 products
 *
 *  The implementation uses FMOPA (f16→f32) for high throughput, widening
 *  to FP32 every 16 (E2M3) or 64 (E3M2) depth steps.
 *
 *  Requires: SME with FEAT_SME_F16F16 for native FP16 outer products.
 *
 *  @sa include/numkong/dots.h
 */
#ifndef NK_DOTS_SMEHALF_H
#define NK_DOTS_SMEHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_SMEHALF

#if defined(__cplusplus)
extern "C" {
#endif

// TODO: Implement E2M3/E3M2 dot products with FP16 intermediate accumulation
//   - nk_dots_e2m3_smehalf: widen every 16 products
//   - nk_dots_e3m2_smehalf: widen every 64 products

#if defined(__cplusplus)
}
#endif

#endif // NK_TARGET_SMEHALF
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_SMEHALF_H
