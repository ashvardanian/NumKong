/**
 *  @file probes/riscv_rvv_bf16.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for RVV Zvfbfwma, BF16 widening fused multiply-add.
 */
#if !defined(__riscv_zvfbfwma)
#error "Feature not available"
#endif
#include <riscv_vector.h>
int main(void) {
    size_t vl = __riscv_vsetvl_e16m1(4);
    vuint16m1_t raw = __riscv_vmv_v_x_u16m1(0x3F80, vl); /* bf16 1.0 */
    vbfloat16m1_t a = __riscv_vreinterpret_v_u16m1_bf16m1(raw);
    size_t vl32 = __riscv_vsetvl_e32m2(4);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl32);
    acc = __riscv_vfwmaccbf16_vv_f32m2(acc, a, a, vl);
    vfloat32m1_t sum = __riscv_vfredusum_vs_f32m2_f32m1(acc, __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1)),
                                                        vl32);
    float result = __riscv_vfmv_f_s_f32m1_f32(sum);
    return result > 0.0f ? 0 : 1;
}
