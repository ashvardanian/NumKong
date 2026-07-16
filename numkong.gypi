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
                "<(numkong_root)/c/dispatch_f64.c",
                "<(numkong_root)/c/dispatch_f32.c",
                "<(numkong_root)/c/dispatch_f16.c",
                "<(numkong_root)/c/dispatch_bf16.c",
                "<(numkong_root)/c/dispatch_i8.c",
                "<(numkong_root)/c/dispatch_u8.c",
                "<(numkong_root)/c/dispatch_u1.c",
                "<(numkong_root)/c/dispatch_e4m3.c",
                "<(numkong_root)/c/dispatch_e5m2.c",
                "<(numkong_root)/c/dispatch_other.c",
                "<(numkong_root)/c/dispatch_f64c.c",
                "<(numkong_root)/c/dispatch_f32c.c",
                "<(numkong_root)/c/dispatch_f16c.c",
                "<(numkong_root)/c/dispatch_bf16c.c",
                "<(numkong_root)/c/dispatch_i16.c",
                "<(numkong_root)/c/dispatch_i32.c",
                "<(numkong_root)/c/dispatch_i64.c",
                "<(numkong_root)/c/dispatch_u16.c",
                "<(numkong_root)/c/dispatch_u32.c",
                "<(numkong_root)/c/dispatch_u64.c",
                "<(numkong_root)/c/dispatch_i4.c",
                "<(numkong_root)/c/dispatch_u4.c",
                "<(numkong_root)/c/dispatch_e2m3.c",
                "<(numkong_root)/c/dispatch_e3m2.c",
            ],
            "include_dirs": [
                "<(numkong_root)/include",
            ],
            "defines": [
                "NK_NATIVE_F16=0",
                "NK_NATIVE_BF16=0",
                "NK_RUNTIME_DISPATCH=1",
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
                        "/Zc:preprocessor"
                    ],
                },
            },
            "conditions": [
                [
                    "OS=='mac'",
                    {
                        "xcode_settings": {
                            "MACOSX_DEPLOYMENT_TARGET": "11.0",
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
