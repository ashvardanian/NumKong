"""
NumKong build configuration.

This file configures wheels compilation for NumKong CPython bindings.
The architecture detection uses environment variable overrides (set via cibuildwheel)
to support cross-compilation scenarios like building ARM64 wheels on x64 hosts.
"""

from __future__ import annotations

import glob
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__lib_name__ = "numkong"
__version__ = Path("VERSION").read_text().strip()

if sys.platform == "darwin":
    _bad_dev_dir = os.environ.get("DEVELOPER_DIR")
    if _bad_dev_dir and (_bad_dev_dir == "public" or not Path(_bad_dev_dir).exists()):
        print(f"[NumKong] Ignoring invalid DEVELOPER_DIR={_bad_dev_dir!r}")
        os.environ.pop("DEVELOPER_DIR", None)


def is_64bit_x86() -> bool:
    """Detect x86-64 architecture with environment override support."""
    override = os.environ.get("NK_TARGET_X8664_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("x86_64", "x64", "amd64")) and (sys.maxsize > 2**32)


def is_64bit_arm() -> bool:
    """Detect ARM64 architecture with environment override support."""
    override = os.environ.get("NK_TARGET_ARM64_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("arm64", "aarch64")) and (sys.maxsize > 2**32)


def is_64bit_riscv() -> bool:
    """Detect RISC-V 64-bit architecture with environment override support."""
    override = os.environ.get("NK_TARGET_RISCV64_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("riscv64",)) and (sys.maxsize > 2**32)


def is_64bit_loongarch() -> bool:
    """Detect LoongArch 64-bit architecture with environment override support."""
    override = os.environ.get("NK_TARGET_LOONGARCH64_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("loongarch64",)) and (sys.maxsize > 2**32)


def is_64bit_power() -> bool:
    """Detect Power 64-bit architecture with environment override support."""
    override = os.environ.get("NK_TARGET_POWER64_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("ppc64le", "ppc64", "powerpc64le", "powerpc64")) and (sys.maxsize > 2**32)


def is_wasm() -> bool:
    """Detect WASM/Emscripten target."""
    host = os.environ.get("_PYTHON_HOST_PLATFORM", "")
    return "emscripten" in host or "wasm" in host


def detect_cc():
    """Detect the C compiler."""
    if sys.platform == "win32":
        return ("cl.exe", True)
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    return (cc.split()[0], False)


def cross_target_flags() -> list[str]:
    """Return --target flags when cross-compiling on macOS."""
    if sys.platform != "darwin":
        return []
    host = platform.machine().lower()
    if is_64bit_x86() and host in ("arm64", "aarch64"):
        return ["--target=x86_64-apple-darwin"]
    if is_64bit_arm() and host in ("x86_64", "amd64", "x64"):
        return ["--target=arm64-apple-darwin"]
    return []


def probe_isa(cc, probe_file, flags, is_msvc=False):
    """Try to compile a probe .c file. Returns True if compiler supports this ISA."""
    with tempfile.NamedTemporaryFile(suffix=".obj" if is_msvc else ".o", delete=False) as tmp:
        obj_path = tmp.name
    try:
        prefix = [cc, "/c"] if is_msvc else [cc, "-c"]
        out_flag = ["/Fo" + obj_path] if is_msvc else ["-o", obj_path]
        extra = [] if is_msvc else cross_target_flags()
        return subprocess.run(prefix + extra + flags + [probe_file] + out_flag, capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(obj_path)
        except OSError:
            pass


ProbeTable = list[tuple[str, str, list[str], list[str]]]

# Probe table: (NK_TARGET_NAME, probe_file, gcc_flags, msvc_flags)
# The probe files contain #error guards for unsupported OS/runtime combinations,
# so we do not need per-platform override logic.
# x86 probes: GCC flags are minimal — each implies its prerequisites.
# E.g., -mavx512vnni implies -mavx512f; -mavxvnni implies -mavx2.
PROBE_TABLE_X86: ProbeTable = [
    ("HASWELL", "probes/x86_haswell.c", ["-mavx2", "-mfma", "-mf16c"], ["/arch:AVX2"]),
    ("SKYLAKE", "probes/x86_skylake.c", ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl"], ["/arch:AVX512"]),
    ("ICELAKE", "probes/x86_icelake.c", ["-mavx512vnni", "-mavx512vl"], ["/arch:AVX512"]),
    ("GENOA", "probes/x86_genoa.c", ["-mavx512bf16", "-mavx512vl"], ["/arch:AVX512"]),
    ("SAPPHIRE", "probes/x86_sapphire.c", ["-mavx512fp16", "-mavx512vl"], ["/arch:AVX512"]),
    ("SAPPHIREAMX", "probes/x86_sapphireamx.c", ["-mamx-tile", "-mamx-int8"], ["/arch:AVX512"]),
    ("GRANITEAMX", "probes/x86_graniteamx.c", ["-mamx-tile", "-mamx-fp16"], ["/arch:AVX512"]),
    ("DIAMOND", "probes/x86_diamond.c", ["-mavx10.2-512"], ["/arch:AVX10.2"]),
    ("TURIN", "probes/x86_turin.c", ["-mavx512vp2intersect"], ["/arch:AVX512"]),
    ("ALDER", "probes/x86_alder.c", ["-mavxvnni"], ["/arch:AVX2"]),
    ("SIERRA", "probes/x86_sierra.c", ["-mavxvnniint8"], ["/arch:AVX2"]),
]

# ARM probes: msvc_flags are empty because MSVC does not define __ARM_FEATURE_*
# macros via /arch: flags. For MSVC header-only builds, types.h infers features
# from __ARM_ARCH level instead. SVE/SME probes also have #error guards for _WIN32.
PROBE_TABLE_ARM: ProbeTable = [
    # FEAT_AdvSIMD
    ("NEON", "probes/arm_neon.c", ["-march=armv8-a+simd"], []),
    ("NEONHALF", "probes/arm_neon_half.c", ["-march=armv8.2-a+simd+fp16"], ["/arch:armv8.2"]),  # FEAT_FP16
    ("NEONSDOT", "probes/arm_neon_sdot.c", ["-march=armv8.2-a+dotprod"], ["/arch:armv8.4"]),  # FEAT_DotProd
    ("NEONBFDOT", "probes/arm_neon_bfdot.c", ["-march=armv8.6-a+simd+bf16"], ["/arch:armv8.6"]),  # FEAT_BF16
    ("NEONFHM", "probes/arm_neon_fhm.c", ["-march=armv8.2-a+simd+fp16+fp16fml"], ["/arch:armv8.4"]),  # FEAT_FHM
    ("NEONFP8", "probes/arm_neonfp8.c", ["-march=armv8-a+simd+fp8dot4"], []),
    ("SVE", "probes/arm_sve.c", ["-march=armv8.2-a+sve"], []),
    ("SVEHALF", "probes/arm_sve_half.c", ["-march=armv8.2-a+sve+fp16"], []),
    ("SVEBFDOT", "probes/arm_sve_bfdot.c", ["-march=armv8.2-a+sve+bf16"], []),
    ("SVESDOT", "probes/arm_sve_sdot.c", ["-march=armv8.2-a+sve+dotprod"], []),
    ("SVE2", "probes/arm_sve2.c", ["-march=armv8.2-a+sve2"], []),
    ("SVE2P1", "probes/arm_sve2p1.c", ["-march=armv8.2-a+sve2p1"], []),
    ("SME", "probes/arm_sme.c", ["-march=armv8-a+sme"], []),
    ("SME2", "probes/arm_sme2.c", ["-march=armv8-a+sme2"], []),
    ("SME2P1", "probes/arm_sme2p1.c", ["-march=armv8-a+sme2p1"], []),
    ("SMEF64", "probes/arm_sme_f64.c", ["-march=armv8-a+sme+sme-f64f64"], []),
    ("SMEHALF", "probes/arm_sme_half.c", ["-march=armv8-a+sme+sme-f16f16"], []),
    ("SMEBF16", "probes/arm_sme_bf16.c", ["-march=armv8-a+sme2+b16b16"], []),
    ("SMEBI32", "probes/arm_sme_bi32.c", ["-march=armv8-a+sme2+sme-i16i32"], []),
    ("SMELUT2", "probes/arm_sme_lut2.c", ["-march=armv8-a+sme2+lut"], []),
    ("SMEFA64", "probes/arm_sme_fa64.c", ["-march=armv8-a+sme+sme-fa64"], []),
]

PROBE_TABLE_RISCV: ProbeTable = [
    ("RVV", "probes/riscv_rvv.c", ["-march=rv64gcv"], []),
    ("RVVHALF", "probes/riscv_rvv_half.c", ["-march=rv64gcv_zvfh"], []),
    ("RVVBF16", "probes/riscv_rvv_bf16.c", ["-march=rv64gcv_zvfbfwma"], []),
    ("RVVBB", "probes/riscv_rvv_bb.c", ["-march=rv64gcv_zvbb"], []),
]

PROBE_TABLE_LOONGARCH: ProbeTable = [
    ("LOONGSONASX", "probes/loongarch_lasx.c", ["-mlasx"], []),
]

PROBE_TABLE_POWER: ProbeTable = [
    ("POWERVSX", "probes/power_vsx.c", ["-mcpu=power9", "-mvsx"], []),
]

PROBE_TABLE_WASM: ProbeTable = [
    ("V128RELAXED", "probes/wasm_v128relaxed.c", ["-mrelaxed-simd"], []),
]


def probe_all_isas() -> list[tuple[str, str]]:
    """Probe all ISAs relevant to the current architecture. Returns macro list."""
    cc, is_msvc = detect_cc()
    macros: list[tuple[str, str]] = []

    tables: list[tuple[bool, ProbeTable]] = [
        (is_64bit_x86(), PROBE_TABLE_X86),
        (is_64bit_arm(), PROBE_TABLE_ARM),
        (is_64bit_riscv(), PROBE_TABLE_RISCV),
        (is_64bit_loongarch(), PROBE_TABLE_LOONGARCH),
        (is_64bit_power(), PROBE_TABLE_POWER),
        (is_wasm(), PROBE_TABLE_WASM),
    ]

    for arch_match, table in tables:
        for name, probe_file, gcc_flags, msvc_flags in table:
            if arch_match and os.path.isfile(probe_file):
                # Allow env-var override: NK_TARGET_FOO=1/true forces on, =0/false forces off
                env_val = os.environ.get(f"NK_TARGET_{name}", "").lower()
                if env_val in ("1", "true"):
                    macros.append((f"NK_TARGET_{name}", "1"))
                    print(f"[NumKong] NK_TARGET_{name}: force-enabled via environment")
                    continue
                if env_val in ("0", "false"):
                    macros.append((f"NK_TARGET_{name}", "0"))
                    print(f"[NumKong] NK_TARGET_{name}: force-disabled via environment")
                    continue
                flags = msvc_flags if is_msvc else gcc_flags
                ok = probe_isa(cc, probe_file, flags, is_msvc)
                macros.append((f"NK_TARGET_{name}", "1" if ok else "0"))
                if ok:
                    print(f"[NumKong] Probe NK_TARGET_{name}: supported")
                else:
                    print(f"[NumKong] Probe NK_TARGET_{name}: not supported")
            else:
                macros.append((f"NK_TARGET_{name}", "0"))

    return macros


def linux_settings() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Build settings for Linux."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-fopenmp",
        "-fdiagnostics-color=always",
        "-fvisibility=default",
        "-fPIC",
        "-w",  # Hush warnings
    ]
    # On RISC-V, GCC needs `-march` with the V extension for vector types to be
    # available at translation-unit scope (`#pragma GCC target` only affects
    # codegen, not type declarations).
    # Keep the module-wide baseline portable (`rv64gcv`) so wheels can import on
    # weaker emulated CPUs. Richer kernels still compile through the explicit
    # NK_TARGET_* defines below plus per-function target attributes.
    if is_64bit_riscv():
        compile_args.append("-march=rv64gcv")
    link_args = [
        "-shared",
        "-fopenmp",
        "-lm",  # Add vectorized `logf` implementation from the `glibc`
    ]
    macros: list[tuple[str, str]] = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_USE_OPENMP", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
    ]
    macros.extend(probe_all_isas())
    return compile_args, link_args, macros


def darwin_settings() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Build settings for macOS."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-Xpreprocessor",
        "-fopenmp",
        "-w",  # Hush warnings
    ]
    link_args: list[str] = ["-lomp"]
    macros: list[tuple[str, str]] = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_USE_OPENMP", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
    ]
    macros.extend(probe_all_isas())
    return compile_args, link_args, macros


def freebsd_settings() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Build settings for FreeBSD."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-fopenmp",
        "-fdiagnostics-color=always",
        "-fvisibility=default",
        "-fPIC",
        "-w",  # Hush warnings
    ]
    link_args = [
        "-shared",
        "-fopenmp",
        "-lm",  # Math library
    ]
    macros: list[tuple[str, str]] = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_USE_OPENMP", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
    ]
    macros.extend(probe_all_isas())
    return compile_args, link_args, macros


def windows_settings() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Build settings for Windows."""
    compile_args = [
        "/std:c11",
        "/O2",
        "/openmp",
        # Dealing with MinGW linking errors
        # https://cibuildwheel.readthedocs.io/en/stable/faq/#windows-importerror-dll-load-failed-the-specific-module-could-not-be-found
        "/d2FH4-",
        "/w",
    ]
    link_args: list[str] = []
    macros: list[tuple[str, str]] = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_USE_OPENMP", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
    ]
    macros.extend(probe_all_isas())
    # MSVC requires architecture-specific macros for winnt.h
    if is_64bit_arm():
        macros.append(("_ARM64_", "1"))
    elif is_64bit_x86():
        macros.append(("_AMD64_", "1"))

    return compile_args, link_args, macros


def emscripten_settings() -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Build settings for Emscripten/Pyodide (WASM)."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-w",
    ]
    link_args: list[str] = []
    # Dynamic dispatch is needed for the Python bindings (nk_find_kernel_punned).
    # The EM_JS runtime probes in c/numkong.c are guarded by NK_DYNAMIC_DISPATCH
    # and __EMSCRIPTEN__; when building as a Pyodide side module, we define
    # NK_PYODIDE_SIDE_MODULE to replace them with conservative stubs (serial only).
    macros: list[tuple[str, str]] = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_PYODIDE_SIDE_MODULE", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
    ]
    # Probing handles everything: non-WASM probes fail (wrong headers),
    # WASM V128RELAXED probe succeeds if emcc supports relaxed SIMD.
    macros.extend(probe_all_isas())
    return compile_args, link_args, macros


# pyodide-build sets _PYTHON_HOST_PLATFORM to "emscripten-wasm32" during cross-compilation.
# sys.platform remains "darwin" or "linux" on the host, so we check this env var first.
_host_platform = os.environ.get("_PYTHON_HOST_PLATFORM", "")
if "emscripten" in _host_platform:
    compile_args, link_args, macros = emscripten_settings()
elif sys.platform == "linux":
    compile_args, link_args, macros = linux_settings()
elif sys.platform.startswith("freebsd"):
    # FreeBSD platform strings can be "freebsd11", "freebsd12", etc.
    compile_args, link_args, macros = freebsd_settings()
elif sys.platform == "darwin":
    compile_args, link_args, macros = darwin_settings()
elif sys.platform == "win32":
    compile_args, link_args, macros = windows_settings()
else:
    # Default to minimal settings for unknown platforms
    compile_args, link_args, macros = [], [], []


def _is_editable_install() -> bool:
    if "develop" in sys.argv or ("install" in sys.argv and "-e" in sys.argv):
        return True
    return any(Path(p, f"{__lib_name__}.egg-link").exists() for p in sys.path)


SETUP_KWARGS = (
    {
        "packages": ["numkong"],
        "package_dir": {"numkong": "python/numkong"},
        "package_data": {"numkong": ["__init__.pyi", "py.typed"]},
    }
    if not _is_editable_install()
    else {}
)

if _is_editable_install():
    print("[NumKong] Editable install detected - skipping bundled type stubs.")


# Use glob to find all dispatch files
base_sources = [
    "python/numkong.c",
    "python/tensor.c",
    "python/matrix.c",
    "python/types.c",
    "python/distance.c",
    "python/each.c",
    "python/mesh.c",
    "python/maxsim.c",
    "python/numpy_interop.c",
    "c/numkong.c",
]

dispatch_sources = sorted(glob.glob("c/dispatch_*.c"))

ext_modules = [
    Extension(
        "numkong",
        sources=base_sources + dispatch_sources,
        include_dirs=["include", "python"],
        language="c",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros,
    )
]


class ParallelBuildExt(build_ext):
    def initialize_options(self):
        super().initialize_options()
        # In Docker containers (e.g. cibuildwheel), `os.cpu_count()` returns the
        # *host* core count, not the container's allocated vCPUs.  Launching dozens
        # of heavy SIMD compilation jobs in parallel OOMs the container (exit 143).
        self.parallel = int(os.environ.get("NK_BUILD_PARALLEL", min(os.cpu_count() or 1, 4)))


setup(
    name=__lib_name__,
    cmdclass={"build_ext": ParallelBuildExt},
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/NumKong",
    description="Portable mixed-precision math, linear-algebra, & retrieval library with 2000+ SIMD kernels for x86, Arm, RISC-V, LoongArch, Power, & WebAssembly",
    long_description=(
        Path("python/README.md").read_text(encoding="utf8") + "\n\n" + Path("README.md").read_text(encoding="utf8")
    ),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Programming Language :: C",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    ext_modules=ext_modules,
    zip_safe=False,
    include_package_data=True,
    **SETUP_KWARGS,
)
