// swift-tools-version:6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.
// 6.1 is the minimum version required for the package access level.

import PackageDescription

let package = Package(
    name: "NumKong",
    // SPM has no `.linux` platform constant — Linux is supported and tested in CI;
    // it simply ignores the `platforms` array on non-Apple hosts.
    platforms: [
        .macOS(.v11),
        .iOS(.v14),
        .tvOS(.v14),
        .watchOS(.v7),
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
                .define("NK_RUNTIME_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
        .testTarget(
            name: "Bench",
            dependencies: ["NumKong", "CNumKong"],
            path: "bench/swift",
            cSettings: [
                .define("NK_RUNTIME_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
        .target(
            name: "NumKong",
            dependencies: ["CNumKong"],
            path: "swift",
            exclude: ["README.md"],
            cSettings: [
                .define("NK_RUNTIME_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
        // No `.unsafeFlags` here — SPM forbids depending on targets with unsafeFlags
        // when pulled as a remote package dependency.  `-Wno-psabi` is GCC-only anyway
        // (ARM Linux ABI notes); CMake handles it in the GNU-only branch.
        .target(
            name: "CNumKong",
            path: "include",
            exclude: ["README.md"],
            // The dispatch sources live in `c/` and are listed here rather than given their own
            // target: a C target with no sources of its own emits no object file, which SPM
            // refuses to link transitively (swiftlang/swift-package-manager#5706).
            sources: [
                "../c/dispatch_bf16.c",
                "../c/dispatch_bf16c.c",
                "../c/dispatch_e2m3.c",
                "../c/dispatch_e3m2.c",
                "../c/dispatch_e4m3.c",
                "../c/dispatch_e5m2.c",
                "../c/dispatch_f16.c",
                "../c/dispatch_f16c.c",
                "../c/dispatch_f32.c",
                "../c/dispatch_f32c.c",
                "../c/dispatch_f64.c",
                "../c/dispatch_f64c.c",
                "../c/dispatch_i16.c",
                "../c/dispatch_i32.c",
                "../c/dispatch_i4.c",
                "../c/dispatch_i64.c",
                "../c/dispatch_i8.c",
                "../c/dispatch_other.c",
                "../c/dispatch_u1.c",
                "../c/dispatch_u16.c",
                "../c/dispatch_u32.c",
                "../c/dispatch_u4.c",
                "../c/dispatch_u64.c",
                "../c/dispatch_u8.c",
                "../c/numkong.c",
            ],
            // `include/` is the module header root, so the `module.modulemap` umbrella and the
            // `#include "numkong/<...>.h"` chain resolve exactly as in the CMake/Rust/Python
            // builds (`-I include`). `c/dispatch.h` is quoted, so it resolves next to its own
            // translation units.
            publicHeadersPath: ".",
            cSettings: [
                .define("NK_RUNTIME_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
    ]
)
