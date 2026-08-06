{
    "variables": {
        "openssl_fips": ""
    },
    "targets": [
        {
            "target_name": "numkong",
            "sources": [
                "javascript/numkong.c",
                "c/numkong.c",
                "c/parallel.c",
                "c/dispatch_f64.c",
                "c/dispatch_f32.c",
                "c/dispatch_f16.c",
                "c/dispatch_bf16.c",
                "c/dispatch_i8.c",
                "c/dispatch_u8.c",
                "c/dispatch_u1.c",
                "c/dispatch_e4m3.c",
                "c/dispatch_e5m2.c",
                "c/dispatch_other.c",
                "c/dispatch_f64c.c",
                "c/dispatch_f32c.c",
                "c/dispatch_f16c.c",
                "c/dispatch_bf16c.c",
                "c/dispatch_i16.c",
                "c/dispatch_i32.c",
                "c/dispatch_i64.c",
                "c/dispatch_u16.c",
                "c/dispatch_u32.c",
                "c/dispatch_u64.c",
                "c/dispatch_i4.c",
                "c/dispatch_u4.c",
                "c/dispatch_e2m3.c",
                "c/dispatch_e3m2.c",
            ],
            "include_dirs": [
                "include",
                "c"
            ],
            "defines": [
                "NK_NATIVE_F16=0",
                "NK_NATIVE_BF16=0",
                "NK_DYNAMIC_DISPATCH=1"
            ],
            "cflags": [
                "-std=c11",
                "-O3",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-Wno-cast-function-type",
                "-Wno-switch",
                "-Wno-psabi",
                "-include",
                "<(module_root_dir)/nk_probes.h",
            ],
            "msvs_settings": {
                "VCCLCompilerTool": {
                    "ForcedIncludeFiles": [
                        "<(module_root_dir)/nk_probes.h"
                    ],
                    "AdditionalOptions": [
                        "/Zc:preprocessor"
                    ],
                },
            },
            "conditions": [
                # Only this branch gets OpenMP; macOS and Windows use their own pools.
                [
                    "OS!='mac' and OS!='win'",
                    {
                        "cflags": [
                            "-fopenmp"
                        ],
                        "ldflags": [
                            "-fopenmp"
                        ]
                    }
                ],
                # Pin TU baseline to each arch's ABI floor; SIMD kernels use per-function pragmas.
                # Keep per-arch table in sync with cmake/nk_compiler_flags.cmake, build.rs, setup.py.
                [
                    "OS!='win' and target_arch=='arm64'",
                    {
                        "cflags": [
                            "-march=armv8-a"
                        ]
                    }
                ],
                [
                    "OS!='win' and target_arch=='x64'",
                    {
                        "cflags": [
                            "-march=x86-64"
                        ]
                    }
                ],
                [
                    "OS!='win' and target_arch=='riscv64'",
                    {
                        "cflags": [
                            "-march=rv64gc"
                        ]
                    }
                ],
                [
                    "OS!='win' and target_arch=='ppc64'",
                    {
                        "cflags": [
                            "-mcpu=power8"
                        ]
                    }
                ],
                [
                    "OS!='win' and target_arch=='loong64'",
                    {
                        "cflags": [
                            "-march=loongarch64",
                            "-mlasx"
                        ]
                    }
                ],
                # Forbid auto-vectorization so serial fallbacks don't get silently
                # promoted to NEON/SSE2/VSX. SIMD kernels use explicit intrinsics
                # and per-function `target` pragmas; unaffected. MSVC has no
                # command-line vectorizer toggle.
                [
                    "OS!='win'",
                    {
                        "cflags": [
                            "-fno-tree-vectorize",
                            "-fno-tree-slp-vectorize"
                        ]
                    }
                ],
                [
                    "OS=='mac'",
                    {
                        "xcode_settings": {
                            "MACOSX_DEPLOYMENT_TARGET": "11.0"
                        }
                    }
                ],
                # MSVC: no per-function target pragma; these match defaults.
                [
                    "OS=='win' and target_arch=='arm64'",
                    {
                        "defines": [
                            "_ARM64_"
                        ],
                        "msvs_settings": {
                            "VCCLCompilerTool": {
                                "AdditionalOptions": [
                                    "/arch:armv8.0"
                                ]
                            }
                        }
                    }
                ],
                [
                    "OS=='win' and target_arch=='x64'",
                    {
                        "defines": [
                            "_AMD64_"
                        ],
                        "msvs_settings": {
                            "VCCLCompilerTool": {
                                "AdditionalOptions": [
                                    "/arch:SSE2"
                                ]
                            }
                        }
                    }
                ],
            ],
        }
    ],
}