/**
 *  @file probes/graniteamx.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the Granite Rapids AMX kernels, AMX-TILE plus AMX-FP16.
 */
#define NUMKONG_HEADER_ONLY       1
#define NUMKONG_TARGET_GRANITEAMX 1
#include "numkong/types.h"
#include "numkong/capabilities.h"    // `syscall`, `SYS_arch_prctl`
#include "numkong/dots/graniteamx.h" // `nk_dots_symmetric_f16_graniteamx`

int main(void) {
#if NUMKONG_OS_LINUX_
    syscall(SYS_arch_prctl, 0x1023, 18); // ARCH_REQ_XCOMP_PERM for XTILEDATA, as `nk_cpu_configure_thread` asks
#endif
    nk_f16_t vectors[16 * 32] = {0};
    nk_f32_t gram[16 * 16];
    nk_dots_symmetric_f16_graniteamx(vectors, 16, 32, 32 * sizeof(nk_f16_t), gram, 16 * sizeof(nk_f32_t), 0, 16, 0, 0,
                                     0);
    return gram[0] != 0;
}
