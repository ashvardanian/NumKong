# NumKong GYP include for downstream native addons.
#
# Usage in your binding.gyp:
#
#   {
#     "includes": ["<!(node -p \"require.resolve('numkong/numkong.gypi')\")"],
#     "targets": [{
#       "target_name": "my_addon",
#       "dependencies": ["numkong_lib"],
#       "sources": ["my_addon.c"],
#     }]
#   }
#
{
    "variables": {
        "numkong_root%": "<!(node -e \"try{console.log(require('path').dirname(require.resolve('numkong/package.json')))}catch{console.log('numkong')}\")",
    },
    "targets": [
        {
            "target_name": "numkong_lib",
            "type": "static_library",
            "actions": [
                {
                    "action_name": "numkong_probe",
                    "inputs": ["<(numkong_root)/probes/probe.js"],
                    "outputs": ["<!(node -e \"console.log(require('path').resolve('<(numkong_root)','nk_probes.h'))\")"],
                    "action": ["node", "<(numkong_root)/probes/probe.js"],
                    "message": "Probing ISA capabilities for NumKong",
                },
            ],
            "sources": [
                "<(numkong_root)/c/numkong.c",
                "<(numkong_root)/c/dispatch/attention.c",
                "<(numkong_root)/c/dispatch/cast.c",
                "<(numkong_root)/c/dispatch/curved.c",
                "<(numkong_root)/c/dispatch/dot.c",
                "<(numkong_root)/c/dispatch/dots.c",
                "<(numkong_root)/c/dispatch/each.c",
                "<(numkong_root)/c/dispatch/geospatial.c",
                "<(numkong_root)/c/dispatch/maxsim.c",
                "<(numkong_root)/c/dispatch/mesh.c",
                "<(numkong_root)/c/dispatch/probability.c",
                "<(numkong_root)/c/dispatch/reduce.c",
                "<(numkong_root)/c/dispatch/scalar.c",
                "<(numkong_root)/c/dispatch/set.c",
                "<(numkong_root)/c/dispatch/sets.c",
                "<(numkong_root)/c/dispatch/sparse.c",
                "<(numkong_root)/c/dispatch/spatial.c",
                "<(numkong_root)/c/dispatch/spatials.c",
                "<(numkong_root)/c/dispatch/trigonometry.c",
                "<(numkong_root)/c/cpu/serial.c",
                "<(numkong_root)/c/cpu/haswell.c",
                "<(numkong_root)/c/cpu/alder.c",
                "<(numkong_root)/c/cpu/sierra.c",
                "<(numkong_root)/c/cpu/skylake.c",
                "<(numkong_root)/c/cpu/icelake.c",
                "<(numkong_root)/c/cpu/genoa.c",
                "<(numkong_root)/c/cpu/turin.c",
                "<(numkong_root)/c/cpu/sapphire.c",
                "<(numkong_root)/c/cpu/diamond.c",
                "<(numkong_root)/c/cpu/sapphireamx.c",
                "<(numkong_root)/c/cpu/graniteamx.c",
                "<(numkong_root)/c/cpu/diamondamx.c",
                "<(numkong_root)/c/cpu/neon.c",
                "<(numkong_root)/c/cpu/neonhalf.c",
                "<(numkong_root)/c/cpu/neonbfdot.c",
                "<(numkong_root)/c/cpu/neonfhm.c",
                "<(numkong_root)/c/cpu/neonsdot.c",
                "<(numkong_root)/c/cpu/neonfp8.c",
                "<(numkong_root)/c/cpu/sve.c",
                "<(numkong_root)/c/cpu/svehalf.c",
                "<(numkong_root)/c/cpu/svesdot.c",
                "<(numkong_root)/c/cpu/svebfdot.c",
                "<(numkong_root)/c/cpu/sve2.c",
                "<(numkong_root)/c/cpu/sme.c",
                "<(numkong_root)/c/cpu/smef64.c",
                "<(numkong_root)/c/cpu/smebi32.c",
                "<(numkong_root)/c/cpu/rvv.c",
                "<(numkong_root)/c/cpu/rvvhalf.c",
                "<(numkong_root)/c/cpu/rvvbf16.c",
                "<(numkong_root)/c/cpu/rvvbb.c",
                "<(numkong_root)/c/cpu/v128.c",
                "<(numkong_root)/c/cpu/v128relaxed.c",
                "<(numkong_root)/c/cpu/powervsx.c",
                "<(numkong_root)/c/cpu/loongsonasx.c",
            ],
            "include_dirs": [
                "<(numkong_root)/include",
                "<(numkong_root)/c",
            ],
            "defines": [
                "NUMKONG_NATIVE_F16=0",
                "NUMKONG_NATIVE_BF16=0",
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
                "<!(node -e \"console.log(require('path').resolve('<(numkong_root)','nk_probes.h'))\")",
            ],
            "msvs_settings": {
                "VCCLCompilerTool": {
                    "ForcedIncludeFiles": [
                        "<!(node -e \"console.log(require('path').resolve('<(numkong_root)','nk_probes.h'))\")",
                    ],
                    "AdditionalOptions": [
                        "/Zc:preprocessor",
                        "/std:c11",
                        "/experimental:c11atomics"
                    ],
                },
            },
            "conditions": [
                [
                    "OS=='mac'",
                    {
                        "xcode_settings": {
                            "MACOSX_DEPLOYMENT_TARGET": "11.0",
                            "OTHER_CFLAGS": [
                                "-std=c11",
                                "-O3",
                                "-include",
                                "<!(node -e \"console.log(require('path').resolve('<(numkong_root)','nk_probes.h'))\")",
                            ],
                        },
                    },
                ],
            ],
            "direct_dependent_settings": {
                "include_dirs": [
                    "<(numkong_root)/include",
                ],
            },
        },
    ],
}
