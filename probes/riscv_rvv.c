/**
 *  @file probes/riscv_rvv.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for RVV 1.0, the RISC-V Vector Extension.
 */
#if !defined(__riscv_v)
#error "Feature not available"
#endif
#include <riscv_vector.h>
int main(void) {
    size_t vl = __riscv_vsetvl_e32m1(4);
    vfloat32m1_t a = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t b = __riscv_vfmv_v_f_f32m1(2.0f, vl);
    vfloat32m1_t c = __riscv_vfadd_vv_f32m1(a, b, vl);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(c, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
    float result = __riscv_vfmv_f_s_f32m1_f32(sum);
    return result > 0.0f ? 0 : 1;
}
