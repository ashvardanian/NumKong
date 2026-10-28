/**
 *  @file probes/sapphireamx.c
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *  @brief Whether the toolchain builds the Sapphire Rapids AMX kernels, AMX-TILE plus AMX-INT8.
 */
#define NUMKONG_HEADER_ONLY        1
#define NUMKONG_TARGET_SAPPHIREAMX 1
#include "numkong/types.h"
#include "numkong/capabilities.h"     // `syscall`, `SYS_arch_prctl`
#include "numkong/dots/sapphireamx.h" // `nk_dots_symmetric_i8_sapphireamx`
#include "numkong/attention/sapphireamx.h"

int main(void) {
#if NUMKONG_OS_LINUX_
    syscall(SYS_arch_prctl, 0x1023, 18); // ARCH_REQ_XCOMP_PERM for XTILEDATA, as `nk_cpu_configure_thread` asks
#endif
    nk_i8_t vectors[16 * 64];
    nk_i32_t gram[16 * 16];
    for (int i = 0; i != 16 * 64; ++i) vectors[i] = 1;
    nk_dots_symmetric_i8_sapphireamx(vectors, 16, 64, 64, gram, 16 * sizeof(nk_i32_t), 0, 16, 0, 0, 0);
    return gram[0] != 64;
}
