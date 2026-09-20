{
    "variables": {
        "openssl_fips": "",
        # `NK_MARCH_NATIVE=1` opts into a host-tuned, non-portable build, as in CMakeLists.txt.
        "nk_march_native%": "<!(node -p \"['1','true','TRUE'].includes(process.env.NK_MARCH_NATIVE||'')?1:0\")"
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
                # Keep per-arch table in sync with CMakeLists.txt, build.rs, setup.py.
                # macOS is excluded: `-arch` already pins the slice, and a per-arch `-march=`
                # conflicts with the other slice of a universal build.
                [
                    "nk_march_native==0 and OS!='win' and OS!='mac' and target_arch=='arm64'",
                    {
                        "cflags": [
                            "-march=armv8-a"
                        ]
                    }
                ],
                [
                    "nk_march_native==0 and OS!='win' and OS!='mac' and target_arch=='x64'",
                    {
                        "cflags": [
                            "-march=x86-64"
                        ]
                    }
                ],
                [
                    "nk_march_native==0 and OS!='win' and OS!='mac' and target_arch=='riscv64'",
                    {
                        "cflags": [
                            "-march=rv64gc"
                        ]
                    }
                ],
                [
                    "nk_march_native==0 and OS!='win' and OS!='mac' and target_arch=='ppc64'",
                    {
                        "cflags": [
                            "-mcpu=power8"
                        ]
                    }
                ],
                [
                    "nk_march_native==0 and OS!='win' and OS!='mac' and target_arch=='loong64'",
                    {
                        "cflags": [
                            "-march=loongarch64",
                            "-mlasx"
                        ]
                    }
                ],
                [
                    "nk_march_native==1 and OS!='win' and OS!='mac'",
                    {
                        "cflags": [
                            "-march=native"
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
                # gyp ignores `cflags` on the mac flavor, so every flag above must be
                # repeated here or the addon builds unoptimized and without the probes.
                [
                    "OS=='mac'",
                    {
                        "xcode_settings": {
                            "MACOSX_DEPLOYMENT_TARGET": "11.0",
                            "OTHER_CFLAGS": [
                                "-std=c11",
                                "-O3",
                                "-fno-tree-vectorize",
                                "-fno-tree-slp-vectorize",
                                "-Wno-unknown-pragmas",
                                "-Wno-cast-function-type",
                                "-Wno-switch",
                                "-include",
                                "<(module_root_dir)/nk_probes.h"
                            ]
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