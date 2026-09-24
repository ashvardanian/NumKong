/**
 *  @file probes/riscv_rvv_bb.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief NumKong ISA probe for RVV Zvbb, basic bit manipulation.
 */
#if !defined(__riscv_zvbb)
#error "Feature not available"
#endif
#include <riscv_vector.h>
int main(void) {
    size_t vl = __riscv_vsetvl_e8m1(4);
    vuint8m1_t a = __riscv_vmv_v_x_u8m1(0xFF, vl);
    /* vcpop.v — per-element popcount, the key Zvbb instruction */
    vuint8m1_t popcnt = __riscv_vcpop_v_u8m1(a, vl);
    vuint8m1_t sum = __riscv_vredsum_vs_u8m1_u8m1(popcnt, __riscv_vmv_v_x_u8m1(0, 1), vl);
    unsigned char result = __riscv_vmv_x_s_u8m1_u8(sum);
    /* Each lane is 0xFF → popcount 8, sum of 4 lanes = 32 */
    return result == 32 ? 0 : 1;
}
