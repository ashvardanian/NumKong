# cmake/nk_arm_isa_probes.cmake — Arm ISA compiler-capability probes
#
# Detect which ISA extensions the compiler can emit.
# Probe source lives in probes/arm_*.c — shared with setup.py and build.rs.

# Apple Clang 21+ recognises M-series CPUs under -mcpu=native and advertises
# SME/SME2/BF16 feature macros correctly. Older Apple Clang didn't, which is
# why earlier code hardcoded -mcpu=apple-m4 — but on a pre-M4 runner that
# hardcode over-reports capabilities and compiles in bfdot/SME kernels that
# SIGILL at runtime on hardware lacking those features.
if (APPLE)
    set(nk_native_flags_ "-mcpu=native")
else ()
    set(nk_native_flags_ "-march=native")
endif ()
include(cmake/nk_isa_probe.cmake)

nk_isa_probes_begin_()

# NEON base probes — MSVC /arch:armv8.X sets __ARM_ARCH but not __ARM_FEATURE_*
# macros; the probes accept __ARM_ARCH fallback (e.g., __ARM_ARCH >= 804 for dotprod).
# FEAT_AdvSIMD (baseline ARM64)
nk_isa_probe_(nk_target_neon "" "-march=armv8-a+simd" "probes/arm_neon.c")
# FEAT_FP16: optional from ARMv8.2, mandatory at ARMv9.0 with AdvSIMD
nk_isa_probe_(nk_target_neonhalf "/arch:armv8.2" "-march=armv8.2-a+simd+fp16" "probes/arm_neon_half.c")
# FEAT_DotProd: optional from ARMv8.1, mandatory at ARMv8.4 with AdvSIMD
nk_isa_probe_(nk_target_neonsdot "/arch:armv8.4" "-march=armv8.2-a+dotprod" "probes/arm_neon_sdot.c")
# FEAT_BF16: optional from ARMv8.2, mandatory at ARMv8.6 with FP
nk_isa_probe_(nk_target_neonbfdot "/arch:armv8.6" "-march=armv8.6-a+simd+bf16" "probes/arm_neon_bfdot.c")
# FEAT_FHM: optional from ARMv8.1, mandatory at ARMv8.4 with FP16
nk_isa_probe_(nk_target_neonfhm "/arch:armv8.4" "-march=armv8.2-a+simd+fp16+fp16fml" "probes/arm_neon_fhm.c")

# SVE probes
nk_isa_probe_(nk_target_sve "" "-march=armv8.2-a+sve" "probes/arm_sve.c")
nk_isa_probe_(nk_target_svehalf "" "-march=armv8.2-a+sve+fp16" "probes/arm_sve_half.c")
nk_isa_probe_(nk_target_svebfdot "" "-march=armv8.2-a+sve+bf16" "probes/arm_sve_bfdot.c")
nk_isa_probe_(nk_target_svesdot "" "-march=armv8.2-a+sve+dotprod" "probes/arm_sve_sdot.c")
nk_isa_probe_(nk_target_sve2 "" "-march=armv8.2-a+sve2" "probes/arm_sve2.c")
nk_isa_probe_(nk_target_sve2p1 "" "-march=armv8.2-a+sve2p1" "probes/arm_sve2p1.c")

# NEON FP8
nk_isa_probe_(nk_target_neonfp8 "" "-march=armv8-a+simd+fp8dot4" "probes/arm_neonfp8.c")

# SME probes
nk_isa_probe_(nk_target_sme "" "-march=armv8-a+sme" "probes/arm_sme.c")
nk_isa_probe_(nk_target_sme2 "" "-march=armv8-a+sme2" "probes/arm_sme2.c")
nk_isa_probe_(nk_target_sme2p1 "" "-march=armv8-a+sme2p1" "probes/arm_sme2p1.c")
nk_isa_probe_(nk_target_smef64 "" "-march=armv8-a+sme+sme-f64f64" "probes/arm_sme_f64.c")
nk_isa_probe_(nk_target_smehalf "" "-march=armv8-a+sme+sme-f16f16" "probes/arm_sme_half.c")
nk_isa_probe_(nk_target_smebf16 "" "-march=armv8-a+sme2+sme-b16b16" "probes/arm_sme_bf16.c")
nk_isa_probe_(nk_target_smebi32 "" "-march=armv8-a+sme2" "probes/arm_sme_bi32.c")
nk_isa_probe_(nk_target_smelut2 "" "-march=armv8-a+sme2+sme-lutv2" "probes/arm_sme_lut2.c")
nk_isa_probe_(nk_target_smefa64 "" "-march=armv8-a+sme+sme-fa64" "probes/arm_sme_fa64.c")

nk_isa_probes_end_()

nk_build_isa_defs_(
    arm
    "Arm"
    "NEON;NEONHALF;NEONSDOT;NEONBFDOT;NEONFHM;SVE;SVEHALF;SVEBFDOT;SVESDOT;SVE2;SVE2P1;NEONFP8;SME;SME2;SME2P1;SMEF64;SMEHALF;SMEBF16;SMEBI32;SMELUT2;SMEFA64"
)
