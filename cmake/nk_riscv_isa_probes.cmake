# cmake/nk_riscv_isa_probes.cmake — RISC-V ISA compiler-capability probes
#
# Detect which ISA extensions the compiler can emit.
# RISC-V has no -march=native, so native falls back to compile result.
# Probe source lives in probes/riscv_*.c — shared with setup.py and build.rs.

set(nk_native_flags_ "")
include(cmake/nk_isa_probe.cmake)

nk_instruction_set_probe_(nk_target_rvv "" "-march=rv64gcv" "probes/riscv_rvv.c")
nk_instruction_set_probe_(nk_target_rvvhalf "" "-march=rv64gcv_zvfh" "probes/riscv_rvv_half.c")
nk_instruction_set_probe_(nk_target_rvvbf16 "" "-march=rv64gcv_zvfbfwma" "probes/riscv_rvv_bf16.c")
nk_instruction_set_probe_(nk_target_rvvbb "" "-march=rv64gcv_zvbb" "probes/riscv_rvv_bb.c")

nk_build_instruction_set_definitions_("RISC-V" "RVV;RVVHALF;RVVBF16;RVVBB")
