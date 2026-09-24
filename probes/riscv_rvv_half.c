/**
 *  @file probes/riscv_rvv_half.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for RVV Zvfh, half-precision vectors.
 */
#if !defined(__riscv_zvfh)
#error "Feature not available"
#endif
#include <riscv_vector.h>
int main(void) {
    size_t vl = __riscv_vsetvl_e16m1(4);
    vfloat16m1_t a = __riscv_vfmv_v_f_f16m1((_Float16)1.0f, vl);
    vfloat32m2_t wide = __riscv_vfwcvt_f_f_v_f32m2(a, vl);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m2_f32m1(wide, __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1)),
                                                        vl);
    float result = __riscv_vfmv_f_s_f32m1_f32(sum);
    return result > 0.0f ? 0 : 1;
}
