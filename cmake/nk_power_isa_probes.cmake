# cmake/nk_power_isa_probes.cmake — Power ISA compiler-capability probes
#
# Detect which ISA extensions the compiler can emit.
# Probe source lives in probes/power_*.c — shared with setup.py and build.rs.

set(nk_native_flags_ "-mcpu=native")
include(cmake/nk_isa_probe.cmake)

nk_instruction_set_probe_(nk_target_powervsx "" "-mcpu=power9 -mvsx" "probes/power_vsx.c")

nk_build_instruction_set_definitions_("Power" "POWERVSX")
