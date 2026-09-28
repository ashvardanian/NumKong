// swift-tools-version:6.4

import PackageDescription

// SwiftPM runs no compiler probes and always builds C with the toolchain's own clang, so the capabilities
// every unit and dispatch list agree on are fixed here; `types.h` turns off those of other architectures.
let capabilities = [
    "HASWELL", "ALDER", "SIERRA", "SKYLAKE", "ICELAKE", "GENOA", "TURIN", "SAPPHIRE", "DIAMOND",
    "SAPPHIREAMX", "GRANITEAMX", "DIAMONDAMX",
    "NEON", "NEONHALF", "NEONBFDOT", "NEONFHM", "NEONSDOT", "NEONFP8",
    "RVV", "RVVHALF", "RVVBF16", "RVVBB", "V128", "V128RELAXED", "POWERVSX", "LOONGSONASX",
]
// Apple's clang modules refuse `<arm_sve.h>`, which the SME kernels build on too.
let capabilitiesBeyondApple = ["SVE", "SVEHALF", "SVESDOT", "SVEBFDOT", "SVE2", "SME", "SMEF64", "SMEBI32"]
let applePlatforms: [Platform] = [.macOS, .iOS, .tvOS, .watchOS, .visionOS]
let otherPlatforms: [Platform] = [.linux, .android, .windows, .wasi]
let capabilitySettings: [CSetting] =
    capabilities.map { .define("NUMKONG_TARGET_\($0)", to: "1") }
    + capabilitiesBeyondApple.map { .define("NUMKONG_TARGET_\($0)", to: "1", .when(platforms: otherPlatforms)) }
    + capabilitiesBeyondApple.map { .define("NUMKONG_TARGET_\($0)", to: "0", .when(platforms: applePlatforms)) }

let package = Package(
    name: "NumKong",
    // SPM has no `.linux` platform constant — Linux is supported and tested in CI; it simply
    // ignores the `platforms` array on non-Apple hosts.
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .tvOS(.v15),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "NumKong", targets: ["CNumKong", "NumKong"]),
        // Exposed so USearch can link the C layer without the Swift wrapper.
        .library(name: "CNumKong", targets: ["CNumKong"]),
    ],
    targets: [
        .testTarget(
            name: "Test",
            dependencies: ["NumKong"],
            path: "test/swift",
            cSettings: [
                .define("NUMKONG_NATIVE_F16", to: "0"),
                .define("NUMKONG_NATIVE_BF16", to: "0"),
            ]
        ),
        .testTarget(
            name: "Bench",
            dependencies: ["NumKong"],
            path: "bench/swift"
        ),
        .target(
            name: "NumKong",
            dependencies: ["CNumKong"],
            path: "swift",
            exclude: ["README.md"],
            cSettings: [
                .define("NUMKONG_NATIVE_F16", to: "0"),
                .define("NUMKONG_NATIVE_BF16", to: "0"),
            ]
        ),
        // `path` must contain every entry in `sources` — SPM silently resolves zero sources for
        // paths that escape it, so the root is the target directory and `sources` selects.
        .target(
            name: "CNumKong",
            path: ".",
            // The Metal kernels travel embedded in their C headers, so SPM must not build them
            // on its own.
            exclude: [
                "include/numkong/dots/simt.metal", "include/numkong/dots/apple9.metal",
                "include/numkong/dots/apple10.metal",
            ],
            sources: [
                "c/numkong.c",
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
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("c"),
                .define("NUMKONG_NATIVE_F16", to: "0"),
                .define("NUMKONG_NATIVE_BF16", to: "0"),
            ] + capabilitySettings
        ),
    ]
)
