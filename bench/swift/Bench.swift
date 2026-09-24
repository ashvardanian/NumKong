//
//  bench/swift/Bench.swift
//  Swift Testing benchmarks for NumKong.
//
//  Runs on an iPad, iPhone or Mac through Xcode, or anywhere SwiftPM runs:
//
//  ```sh
//  xcodebuild test -scheme NumKong-Package -destination 'platform=iOS,name=...' -only-testing Bench
//  swift test -c release --filter Bench
//  ```
//
//  Environment variables, matching C++ nk_bench:
//      NK_DENSE_DIMENSIONS  — pairwise vector length, defaulting to 1536
//      NK_MATRIX_HEIGHT     — GEMM M / dataset rows, defaulting to 1024
//      NK_MATRIX_WIDTH      — GEMM N / query rows, defaulting to 128
//      NK_MATRIX_DEPTH      — GEMM K / vector dims, defaulting to 1536
//
//  `xcodebuild` forwards them to the tests only with a `TEST_RUNNER_` prefix.
//
//  - Author: Ash Vardanian
//  - Date: March 14, 2026
//

import Foundation
import NumKong
import Testing

private let environment = ProcessInfo.processInfo.environment

private func env(_ key: String, default d: Int) -> Int {
    environment[key].flatMap { Int($0) } ?? d
}

private let denseDims = env("NK_DENSE_DIMENSIONS", default: 1536)
private let matrixHeight = env("NK_MATRIX_HEIGHT", default: 1024)
private let matrixWidth = env("NK_MATRIX_WIDTH", default: 128)
private let matrixDepth = env("NK_MATRIX_DEPTH", default: 1536)
private let pairwiseReps = 10_000

/// One benchmark: `prepare` builds its inputs once and returns the call to time.
struct Workload: Sendable, CustomTestStringConvertible {
    let testDescription: String
    let prepare: @Sendable () throws -> () -> Void
}

/// Draws `count` values uniformly from [-1, 1] and converts each to the benchmarked type.
private func uniform<T: SendableMetatype>(_ convert: @escaping @Sendable (Float32) -> T) -> @Sendable (Int) -> [T] {
    { count in (0..<count).map { _ in convert(Float32.random(in: -1...1)) } }
}

/// Draws `count` random bytes and reinterprets each as the benchmarked type.
private func bytes<T: SendableMetatype>(_ convert: @escaping @Sendable (UInt8) -> T) -> @Sendable (Int) -> [T] {
    { count in (0..<count).map { _ in convert(UInt8.random(in: .min ... .max)) } }
}

private func pairwise<T: SendableMetatype>(
    _ name: String, _ width: Int, _ make: @escaping @Sendable (Int) -> [T],
    _ op: @escaping @Sendable ([T], [T]) -> Any?
) -> Workload {
    Workload(testDescription: name) {
        let a = make(width)
        let b = make(width)
        return { for _ in 0..<pairwiseReps { _ = op(a, b) } }
    }
}

private func packed<T: NumKongDotsMatrixElement & SendableMetatype>(
    _ name: String, _ depth: Int, _ make: @escaping @Sendable (Int) -> [T],
    _ op: @escaping @Sendable (Tensor<T>, PackedMatrix<T>) throws -> Any
) -> Workload {
    Workload(testDescription: "\(name) packed") {
        let a = try Tensor<T>.fromArray(make(matrixHeight * depth), rows: matrixHeight, cols: depth)
        let b = try Tensor<T>.fromArray(make(matrixWidth * depth), rows: matrixWidth, cols: depth)
        let packed = try b.packForDots()
        return { _ = try! op(a, packed) }
    }
}

private func symmetric<T: SendableMetatype>(
    _ name: String, _ depth: Int, _ make: @escaping @Sendable (Int) -> [T],
    _ op: @escaping @Sendable (Tensor<T>) throws -> Any
) -> Workload {
    Workload(testDescription: "\(name) symmetric") {
        let a = try Tensor<T>.fromArray(make(matrixHeight * depth), rows: matrixHeight, cols: depth)
        return { _ = try! op(a) }
    }
}

/// Every spatial kernel on one type: pairwise, packed and symmetric.
private func spatial<T: NumKongSpatial & NumKongSpatialsMatrixElement & SendableMetatype>(
    _ name: String, _ make: @escaping @Sendable (Int) -> [T]
) -> [Workload] {
    [
        pairwise("\(name) dot", denseDims, make) { $0.dot($1) },
        pairwise("\(name) angular", denseDims, make) { $0.angular($1) },
        pairwise("\(name) euclidean", denseDims, make) { $0.euclidean($1) },
        pairwise("\(name) sqeuclidean", denseDims, make) { $0.sqeuclidean($1) },
        packed("\(name) dots", matrixDepth, make) { try $0.dotsPacked($1) },
        packed("\(name) angulars", matrixDepth, make) { try $0.angularsPacked($1) },
        packed("\(name) euclideans", matrixDepth, make) { try $0.euclideansPacked($1) },
        symmetric("\(name) dots", matrixDepth, make) { try $0.dotsSymmetric() },
        symmetric("\(name) angulars", matrixDepth, make) { try $0.angularsSymmetric() },
        symmetric("\(name) euclideans", matrixDepth, make) { try $0.euclideansSymmetric() },
    ]
}

/// Every binary kernel on one bit-packed type, whose values each hold eight dimensions.
private func binary<T: NumKongDot & NumKongHamming & NumKongJaccard & NumKongSetsMatrixElement & SendableMetatype>(
    _ name: String, _ make: @escaping @Sendable (Int) -> [T]
) -> [Workload] {
    let width = denseDims / 8
    let depth = matrixDepth / 8
    return [
        pairwise("\(name) dot", width, make) { $0.dot($1) },
        pairwise("\(name) hamming", width, make) { $0.hamming($1) },
        pairwise("\(name) jaccard", width, make) { $0.jaccard($1) },
        packed("\(name) dots", depth, make) { try $0.dotsPacked($1) },
        packed("\(name) hammings", depth, make) { try $0.hammingsPacked($1) },
        packed("\(name) jaccards", depth, make) { try $0.jaccardsPacked($1) },
        symmetric("\(name) dots", depth, make) { try $0.dotsSymmetric() },
        symmetric("\(name) hammings", depth, make) { try $0.hammingsSymmetric() },
        symmetric("\(name) jaccards", depth, make) { try $0.jaccardsSymmetric() },
    ]
}

let workloads: [Workload] = {
    var families = [
        spatial("Float64", uniform { Float64($0) }),
        spatial("Float32", uniform { $0 }),
        spatial("BFloat16", uniform { BFloat16(float: $0) }),
        spatial("Int8", bytes { Int8(bitPattern: $0) }),
        spatial("UInt8", bytes { $0 }),
        spatial("E4M3", uniform { E4M3(float: $0) }),
        spatial("E5M2", uniform { E5M2(float: $0) }),
        spatial("E2M3", uniform { E2M3(float: $0) }),
        spatial("E3M2", uniform { E3M2(float: $0) }),
        binary("U1x8", bytes { U1x8($0) }),
    ]
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    families.append(spatial("Float16", uniform { Float16($0) }))
    #endif
    return families.flatMap { $0 }
}()

@Suite(.serialized)
struct Bench {
    @Test func configuration() {
        print("Capabilities: \(String(Capabilities.available, radix: 2))")
        print("Dense dimensions: \(denseDims)")
        print("Matrix: \(matrixHeight)×\(matrixDepth) × \(matrixWidth)×\(matrixDepth)")
    }

    /// Reports the fastest of ten runs, the one least disturbed by the rest of the system.
    @Test(arguments: workloads, [false, true])
    func run(_ workload: Workload, serial: Bool) throws {
        let call = try workload.prepare()
        if serial { Capabilities.restrict(Capabilities.serial) }
        defer { Capabilities.restrict(Capabilities.available) }
        let clock = ContinuousClock()
        let fastest = (0..<10).map { _ in clock.measure(call) }.min()!
        print("\(workload.testDescription)\(serial ? ", serial" : ""): \(fastest)")
    }
}
