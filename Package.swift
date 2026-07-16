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
        // We need to expose the underlying `CNumKongDispatch` target to simplify
        // linking for USearch
        .library(name: "NumKong", targets: ["NumKong"]),
        .library(name: "CNumKongDispatch", targets: ["CNumKongDispatch"]),
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
            dependencies: ["CNumKong", "CNumKongDispatch"],
            path: "swift",
            exclude: ["README.md"],
            cSettings: [
                .define("NK_RUNTIME_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
            ]
        ),
        .target(
            name: "CNumKong",
            path: "include",
            publicHeadersPath: ".",
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
            name: "CNumKongDispatch",
            dependencies: ["CNumKong"],
            path: "c",
            // dispatch.h is internal to this target — expose it so SPM doesn't
            // look for a non-existent `include/` subdirectory.
            publicHeadersPath: ".",
            cSettings: [
                .define("NK_RUNTIME_DISPATCH", to: "1"),
                .define("NK_NATIVE_F16", to: "0"),
                .define("NK_NATIVE_BF16", to: "0"),
                .headerSearchPath("../include"),
            ]
        ),
    ]
)
