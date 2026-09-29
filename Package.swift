// swift-tools-version:6.4

import PackageDescription

// CMake builds and probes the C library, as for every other binding, and SwiftPM links what it
// produced: an XCFramework on Apple platforms and an artifact bundle elsewhere. A checkout points
// `NUMKONG_SWIFT_ARTIFACT` at the one `cmake --build --preset swift` wrote, relative to this
// directory; everyone else downloads the release's, whose checksums the release workflow writes.
let release = "https://github.com/ashvardanian/NumKong/releases/download/v7.8.2"
let appleChecksum = "0000000000000000000000000000000000000000000000000000000000000000"
let portableChecksum = "0000000000000000000000000000000000000000000000000000000000000000"

let library: [Target]
let libraryDependencies: [Target.Dependency]
if let artifact = Context.environment["NUMKONG_SWIFT_ARTIFACT"] {
    library = [.binaryTarget(name: "CNumKong", path: artifact)]
    libraryDependencies = ["CNumKong"]
} else {
    library = [
        .binaryTarget(name: "CNumKongApple", url: "\(release)/CNumKong.xcframework.zip", checksum: appleChecksum),
        .binaryTarget(name: "CNumKongPortable", url: "\(release)/CNumKong.artifactbundle.zip", checksum: portableChecksum),
    ]
    libraryDependencies = [
        .target(name: "CNumKongApple", condition: .when(platforms: [.macOS, .macCatalyst, .iOS, .tvOS, .watchOS, .visionOS])),
        .target(name: "CNumKongPortable", condition: .when(platforms: [.linux, .android, .windows, .wasi])),
    ]
}

let package = Package(
    name: "NumKong",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .tvOS(.v15),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "NumKong", targets: ["NumKong"])
    ],
    targets: library + [
        .target(
            name: "NumKong",
            dependencies: libraryDependencies,
            path: "swift",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "Test",
            dependencies: ["NumKong"],
            path: "test/swift"
        ),
        .testTarget(
            name: "Bench",
            dependencies: ["NumKong"],
            path: "bench/swift"
        ),
    ]
)
