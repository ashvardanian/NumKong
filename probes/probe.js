#!/usr/bin/env node
/**
 * NumKong ISA probe script for Node.js / node-gyp builds.
 *
 * Try-compiles each probe .c file from probes/ to determine which ISA
 * extensions the current compiler supports. Writes results to
 * build/nk_probes.h as #define NK_TARGET_FOO 1/0.
 *
 * Usage: node scripts/probe_isa.js
 * Called automatically via package.json "preinstall" hook.
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const os = require("os");

const cc = process.env.CC || (process.platform === "win32" ? "cl.exe" : "cc");
const isWin = process.platform === "win32";

/** Try to compile a probe file. Returns true if compilation succeeds. */
function probeIsa(probeFile, flags) {
    const tmpObj = path.join(
        os.tmpdir(),
        `nk_probe_${path.basename(probeFile, ".c")}${isWin ? ".obj" : ".o"}`,
    );
    try {
        const cmd = isWin
            ? `"${cc}" /c ${flags.join(" ")} "${probeFile}" /Fo"${tmpObj}" /nologo`
            : `"${cc}" -c ${flags.join(" ")} "${probeFile}" -o "${tmpObj}" 2>/dev/null`;
        execSync(cmd, { stdio: "pipe", timeout: 30000 });
        return true;
    } catch {
        return false;
    } finally {
        try {
            fs.unlinkSync(tmpObj);
        } catch { }
    }
}

// Probe table: [define, probeFile, gccFlags, msvcFlags]
// x86 probes: GCC flags are minimal — each implies its prerequisites.
// E.g., -mavx512vnni implies -mavx512f; -mavxvnni implies -mavx2.
const PROBES = [
    // x86
    ["NK_TARGET_HASWELL", "probes/x86_haswell.c", ["-mavx2", "-mfma", "-mf16c"], ["/arch:AVX2"]],
    ["NK_TARGET_SKYLAKE", "probes/x86_skylake.c", ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl"], ["/arch:AVX512"]],
    ["NK_TARGET_ICELAKE", "probes/x86_icelake.c", ["-mavx512vnni", "-mavx512vl"], ["/arch:AVX512"]],
    ["NK_TARGET_GENOA", "probes/x86_genoa.c", ["-mavx512bf16", "-mavx512vl"], ["/arch:AVX512"]],
    ["NK_TARGET_SAPPHIRE", "probes/x86_sapphire.c", ["-mavx512fp16", "-mavx512vl"], ["/arch:AVX512"]],
    ["NK_TARGET_SAPPHIREAMX", "probes/x86_sapphireamx.c", ["-mamx-tile", "-mamx-int8"], ["/arch:AVX512"]],
    ["NK_TARGET_GRANITEAMX", "probes/x86_graniteamx.c", ["-mamx-tile", "-mamx-fp16"], ["/arch:AVX512"]],
    ["NK_TARGET_DIAMOND", "probes/x86_diamond.c", ["-mavx10.2-512"], ["/arch:AVX10.2"]],
    ["NK_TARGET_TURIN", "probes/x86_turin.c", ["-mavx512vp2intersect"], ["/arch:AVX512"]],
    ["NK_TARGET_ALDER", "probes/x86_alder.c", ["-mavxvnni"], ["/arch:AVX2"]],
    ["NK_TARGET_SIERRA", "probes/x86_sierra.c", ["-mavxvnniint8"], ["/arch:AVX2"]],
    // ARM NEON base probes — msvc_flags are empty because MSVC does not define
    // __ARM_FEATURE_* macros via /arch: flags. For MSVC header-only builds,
    // types.h infers features from __ARM_ARCH level instead.
    ["NK_TARGET_NEON", "probes/arm_neon.c", ["-march=armv8-a+simd"], []], // FEAT_AdvSIMD
    ["NK_TARGET_NEONHALF", "probes/arm_neon_half.c", ["-march=armv8.2-a+simd+fp16"], ["/arch:armv8.2"]], // FEAT_FP16
    ["NK_TARGET_NEONSDOT", "probes/arm_neon_sdot.c", ["-march=armv8.2-a+dotprod"], ["/arch:armv8.4"]], // FEAT_DotProd
    ["NK_TARGET_NEONBFDOT", "probes/arm_neon_bfdot.c", ["-march=armv8.6-a+simd+bf16"], ["/arch:armv8.6"]], // FEAT_BF16
    ["NK_TARGET_NEONFHM", "probes/arm_neon_fhm.c", ["-march=armv8.2-a+simd+fp16+fp16fml"], ["/arch:armv8.4"]], // FEAT_FHM
    // ARM SVE/SME
    ["NK_TARGET_SVE", "probes/arm_sve.c", ["-march=armv8.2-a+sve"], []],
    ["NK_TARGET_SVEHALF", "probes/arm_sve_half.c", ["-march=armv8.2-a+sve+fp16"], []],
    ["NK_TARGET_SVEBFDOT", "probes/arm_sve_bfdot.c", ["-march=armv8.2-a+sve+bf16"], []],
    ["NK_TARGET_SVESDOT", "probes/arm_sve_sdot.c", ["-march=armv8.2-a+sve+dotprod"], []],
    ["NK_TARGET_SVE2", "probes/arm_sve2.c", ["-march=armv8.2-a+sve2"], []],
    ["NK_TARGET_SVE2P1", "probes/arm_sve2p1.c", ["-march=armv8.2-a+sve2p1"], []],
    ["NK_TARGET_NEONFP8", "probes/arm_neonfp8.c", ["-march=armv8-a+simd+fp8dot4"], []],
    ["NK_TARGET_SME", "probes/arm_sme.c", ["-march=armv8-a+sme"], []],
    ["NK_TARGET_SME2", "probes/arm_sme2.c", ["-march=armv8-a+sme2"], []],
    ["NK_TARGET_SME2P1", "probes/arm_sme2p1.c", ["-march=armv8-a+sme2p1"], []],
    ["NK_TARGET_SMEF64", "probes/arm_sme_f64.c", ["-march=armv8-a+sme+sme-f64f64"], []],
    ["NK_TARGET_SMEHALF", "probes/arm_sme_half.c", ["-march=armv8-a+sme+sme-f16f16"], []],
    ["NK_TARGET_SMEBF16", "probes/arm_sme_bf16.c", ["-march=armv8-a+sme2+sme-b16b16"], []],
    ["NK_TARGET_SMEBI32", "probes/arm_sme_bi32.c", ["-march=armv8-a+sme2"], []],
    ["NK_TARGET_SMELUT2", "probes/arm_sme_lut2.c", ["-march=armv8-a+sme2+sme-lutv2"], []],
    ["NK_TARGET_SMEFA64", "probes/arm_sme_fa64.c", ["-march=armv8-a+sme+sme-fa64"], []],
    // RISC-V
    ["NK_TARGET_RVV", "probes/riscv_rvv.c", ["-march=rv64gcv"], []],
    ["NK_TARGET_RVVHALF", "probes/riscv_rvv_half.c", ["-march=rv64gcv_zvfh"], []],
    ["NK_TARGET_RVVBF16", "probes/riscv_rvv_bf16.c", ["-march=rv64gcv_zvfbfwma"], []],
    ["NK_TARGET_RVVBB", "probes/riscv_rvv_bb.c", ["-march=rv64gcv_zvbb"], []],
    // LoongArch
    ["NK_TARGET_LOONGSONASX", "probes/loongarch_lasx.c", ["-mlasx"], []],
    // Power
    ["NK_TARGET_POWERVSX", "probes/power_vsx.c", ["-mcpu=power9", "-mvsx"], []],
    // WASM
    ["NK_TARGET_V128RELAXED", "probes/wasm_v128relaxed.c", ["-mrelaxed-simd"], []],
];

function main() {
    const pkgRoot = path.join(__dirname, "..");
    fs.mkdirSync(pkgRoot, { recursive: true });

    const arch = process.arch; // 'x64', 'arm64', etc.
    const lines = [
        "/* Auto-generated by scripts/probe_isa.js — do not edit */",
        `/* Compiler: ${cc}, Platform: ${process.platform}, Arch: ${arch} */`,
        "",
    ];

    let enabled = 0;
    for (const [define, probeFile, gccFlags, msvcFlags] of PROBES) {
        const flags = isWin ? msvcFlags : gccFlags;
        const supported = probeIsa(path.join(pkgRoot, probeFile), flags);
        lines.push(`#define ${define} ${supported ? 1 : 0}`);
        if (supported) {
            enabled++;
            console.log(`[NumKong] Probe ${define}: supported`);
        }
    }

    lines.push("");
    const header = lines.join("\n");
    const outPath = path.join(pkgRoot, "nk_probes.h");
    fs.writeFileSync(outPath, header);
    console.log(
        `[NumKong] Wrote ${outPath} (${enabled} ISAs enabled out of ${PROBES.length})`,
    );
}

main();
