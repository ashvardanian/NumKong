{
    "variables": {
        "openssl_fips": "",
        # `NUMKONG_TARGET_ARCH=native` opts into a host-tuned, non-portable build, as in CMakeLists.txt.
        "nk_march_native%": "<!(node -p \"process.env.NUMKONG_TARGET_ARCH==='native'?1:0\")"
    },
    "targets": [
        {
            "target_name": "numkong",
            "sources": [
                "javascript/numkong.c",
                "c/numkong.c",
                "c/parallel.c",
                "c/dispatch/attention.c",
                "c/dispatch/cast.c",
                "c/dispatch/curved.c",
                "c/dispatch/dot.c",
                "c/dispatch/dots.c",
                "c/dispatch/each.c",
                "c/dispatch/geospatial.c",
                "c/dispatch/maxsim.c",
                "c/dispatch/mesh.c",
                "c/dispatch/probability.c",
                "c/dispatch/reduce.c",
                "c/dispatch/scalar.c",
                "c/dispatch/set.c",
                "c/dispatch/sets.c",
                "c/dispatch/sparse.c",
                "c/dispatch/spatial.c",
                "c/dispatch/spatials.c",
                "c/dispatch/trigonometry.c",
                "c/cpu/serial.c",
                "c/cpu/haswell.c",
                "c/cpu/alder.c",
                "c/cpu/sierra.c",
                "c/cpu/skylake.c",
                "c/cpu/icelake.c",
                "c/cpu/genoa.c",
                "c/cpu/turin.c",
                "c/cpu/sapphire.c",
                "c/cpu/diamond.c",
                "c/cpu/sapphireamx.c",
                "c/cpu/graniteamx.c",
                "c/cpu/diamondamx.c",
                "c/cpu/neon.c",
                "c/cpu/neonhalf.c",
                "c/cpu/neonbfdot.c",
                "c/cpu/neonfhm.c",
                "c/cpu/neonsdot.c",
                "c/cpu/neonfp8.c",
                "c/cpu/sve.c",
                "c/cpu/svehalf.c",
                "c/cpu/svesdot.c",
                "c/cpu/svebfdot.c",
                "c/cpu/sve2.c",
                "c/cpu/sme.c",
                "c/cpu/smef64.c",
                "c/cpu/smebi32.c",
                "c/cpu/rvv.c",
                "c/cpu/rvvhalf.c",
                "c/cpu/rvvbf16.c",
                "c/cpu/rvvbb.c",
                "c/cpu/v128.c",
                "c/cpu/v128relaxed.c",
                "c/cpu/powervsx.c",
                "c/cpu/loongsonasx.c",
            ],
            "include_dirs": [
                "include",
                "c"
            ],
            "defines": [
                "NUMKONG_NATIVE_F16=0",
                "NUMKONG_NATIVE_BF16=0"
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
                        "/Zc:preprocessor",
                        "/std:c11",
                        "/experimental:c11atomics"
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