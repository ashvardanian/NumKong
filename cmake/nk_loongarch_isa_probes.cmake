# cmake/nk_loongarch_isa_probes.cmake — LoongArch ISA compiler-capability probes
#
# Detect which ISA extensions the compiler can emit.
# Probe source lives in probes/loongarch_*.c — shared with setup.py and build.rs.

set(nk_native_flags_ "-march=native")
include(cmake/nk_isa_probe.cmake)

nk_instruction_set_probe_(nk_target_loongsonasx "" "-mlasx" "probes/loongarch_lasx.c")

nk_build_instruction_set_definitions_("LoongArch" "LOONGSONASX")
