# cmake/nk_x86_isa_probes.cmake — x86 ISA compiler-capability probes
#
# Detect which ISA extensions the compiler can emit.
# Probe source lives in probes/x86_*.c — shared with setup.py and build.rs.

set(nk_native_flags_ "-march=native")
include(cmake/nk_isa_probe.cmake)

nk_isa_probes_begin_()

nk_isa_probe_(nk_target_haswell "/arch:AVX2" "-mavx2 -mfma -mf16c" "probes/x86_haswell.c")

nk_isa_probe_(nk_target_skylake "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl" "probes/x86_skylake.c")

nk_isa_probe_(nk_target_icelake "/arch:AVX512" "-mavx512vnni -mavx512vl" "probes/x86_icelake.c")

nk_isa_probe_(nk_target_genoa "/arch:AVX512" "-mavx512bf16 -mavx512vl" "probes/x86_genoa.c")

nk_isa_probe_(nk_target_sapphire "/arch:AVX512" "-mavx512fp16 -mavx512vl" "probes/x86_sapphire.c")

nk_isa_probe_(nk_target_sapphireamx "/arch:AVX512" "-mamx-tile -mamx-int8" "probes/x86_sapphireamx.c")

nk_isa_probe_(nk_target_graniteamx "/arch:AVX512" "-mamx-tile -mamx-fp16" "probes/x86_graniteamx.c")

nk_isa_probe_(nk_target_diamond "/arch:AVX10.2" "-mavx10.2-512" "probes/x86_diamond.c")

nk_isa_probe_(
    nk_target_diamondamx "/arch:AVX10.2" "-mamx-tile -mamx-fp8 -mamx-avx512 -mavx10.2-512" "probes/x86_diamondamx.c"
)

nk_isa_probe_(nk_target_turin "/arch:AVX512" "-mavx512vp2intersect" "probes/x86_turin.c")

nk_isa_probe_(nk_target_alder "/arch:AVX2" "-mavxvnni" "probes/x86_alder.c")

nk_isa_probe_(nk_target_sierra "/arch:AVX2" "-mavxvnniint8" "probes/x86_sierra.c")

nk_isa_probes_end_()

nk_build_isa_defs_(
    x86 "x86" "HASWELL;SKYLAKE;ICELAKE;GENOA;SAPPHIRE;SAPPHIREAMX;GRANITEAMX;DIAMOND;DIAMONDAMX;TURIN;ALDER;SIERRA"
)
