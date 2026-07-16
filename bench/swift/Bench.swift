//  Bench.swift
//  NumKong
//
//  XCTest performance benchmarks for NumKong.
//  Run on-device (iPad, iPhone, Mac) via:
//      xcodebuild test -scheme NumKong -destination 'platform=iOS,name=...' -only-testing Bench
//  Or locally:
//      swift test --filter Bench
//
//  Environment variables (matching C++ nk_bench):
//      NK_DENSE_DIMENSIONS  — pairwise vector length  (default 1536)
//      NK_MATRIX_HEIGHT     — GEMM M / dataset rows   (default 1024)
//      NK_MATRIX_WIDTH      — GEMM N / query rows     (default 128)
//      NK_MATRIX_DEPTH      — GEMM K / vector dims    (default 1536)

#if canImport(Darwin)

import CNumKong
import NumKong
import XCTest

// MARK: - Configuration

private func env(_ key: String, default d: Int) -> Int {
    if let v = ProcessInfo.processInfo.environment[key], let i = Int(v) { return i }
    return d
}

private let denseDims = env("NK_DENSE_DIMENSIONS", default: 1536)
private let matrixHeight = env("NK_MATRIX_HEIGHT", default: 1024)
private let matrixWidth = env("NK_MATRIX_WIDTH", default: 128)
private let matrixDepth = env("NK_MATRIX_DEPTH", default: 1536)
private let pairwiseReps = 10_000

// MARK: - Random Generators

private func randomFloat64(_ n: Int) -> [Float64] {
    (0..<n).map { _ in Float64.random(in: -1...1) }
}

private func randomFloat32(_ n: Int) -> [Float32] {
    (0..<n).map { _ in Float32.random(in: -1...1) }
}

#if !arch(x86_64)
@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
private func randomFloat16(_ n: Int) -> [Float16] {
    (0..<n).map { _ in Float16.random(in: -1...1) }
}
#endif  // !arch(x86_64)

private func randomInt8(_ n: Int) -> [Int8] {
    (0..<n).map { _ in Int8.random(in: .min ... .max) }
}

private func randomUInt8(_ n: Int) -> [UInt8] {
    (0..<n).map { _ in UInt8.random(in: .min ... .max) }
}

private func randomBFloat16(_ n: Int) -> [BFloat16] {
    (0..<n).map { _ in BFloat16(float: Float32.random(in: -1...1)) }
}

private func randomE4M3(_ n: Int) -> [E4M3] {
    (0..<n).map { _ in E4M3(float: Float32.random(in: -1...1)) }
}

private func randomE5M2(_ n: Int) -> [E5M2] {
    (0..<n).map { _ in E5M2(float: Float32.random(in: -1...1)) }
}

private func randomE2M3(_ n: Int) -> [E2M3] {
    (0..<n).map { _ in E2M3(float: Float32.random(in: -1...1)) }
}

private func randomE3M2(_ n: Int) -> [E3M2] {
    (0..<n).map { _ in E3M2(float: Float32.random(in: -1...1)) }
}

private func randomU1x8(_ n: Int) -> [U1x8] {
    (0..<n).map { _ in U1x8(UInt8.random(in: .min ... .max)) }
}

// MARK: - Capability Detection

private func capabilityNames(_ caps: nk_capability_t) -> String {
    var names: [String] = []
    if caps & 1 != 0 { names.append("serial") }
    if caps & (1 << 1) != 0 { names.append("neon") }
    if caps & (1 << 2) != 0 { names.append("haswell") }
    if caps & (1 << 3) != 0 { names.append("skylake") }
    if caps & (1 << 4) != 0 { names.append("neon_fp16") }
    if caps & (1 << 5) != 0 { names.append("neon_sdot") }
    if caps & (1 << 6) != 0 { names.append("neon_fhm") }
    if caps & (1 << 7) != 0 { names.append("icelake") }
    if caps & (1 << 8) != 0 { names.append("genoa") }
    if caps & (1 << 9) != 0 { names.append("neon_bf16") }
    if caps & (1 << 10) != 0 { names.append("sve") }
    if caps & (1 << 11) != 0 { names.append("sve_fp16") }
    if caps & (1 << 12) != 0 { names.append("sve_sdot") }
    if caps & (1 << 17) != 0 { names.append("sapphire") }
    if caps & (1 << 24) != 0 { names.append("sme") }
    if caps & (1 << 25) != 0 { names.append("sme2") }
    return names.joined(separator: ", ")
}

// MARK: - Serial Dispatch Helper

private func withSerialDispatch(_ body: () -> Void) {
    nk_capabilities_restrict(1)  // nk_cap_serial_k
    defer { nk_capabilities_restrict(nk_capabilities_available()) }
    body()
}

// MARK: - Pairwise Benchmark Helpers

extension XCTestCase {
    /// Benchmarks a pairwise operation over `pairwiseReps` iterations, with optional serial-only dispatch.
    func benchPairwise<T>(_ make: (Int) -> [T], _ op: @escaping ([T], [T]) -> Any?, serial: Bool = false) {
        let dims = T.self == U1x8.self ? denseDims / 8 : denseDims
        let a = make(dims)
        let b = make(dims)
        let body = { self.measure(metrics: [XCTClockMetric()]) { for _ in 0..<pairwiseReps { _ = op(a, b) } } }
        if serial { withSerialDispatch(body) } else { body() }
    }

    /// Benchmarks a packed matrix operation, with optional serial-only dispatch.
    func benchPacked<T: NumKongDotsMatrixElement>(
        _ make: (Int) -> [T], serial: Bool = false, _ op: @escaping (Tensor<T>, PackedMatrix<T>) throws -> Any
    ) throws {
        let depth = T.self == U1x8.self ? matrixDepth / 8 : matrixDepth
        let a = try Tensor<T>.fromArray(make(matrixHeight * depth), rows: matrixHeight, cols: depth)
        let b = try Tensor<T>.fromArray(make(matrixWidth * depth), rows: matrixWidth, cols: depth)
        let packed = try b.packForDots()
        let body = { self.measure(metrics: [XCTClockMetric()]) { _ = try! op(a, packed) } }
        if serial { withSerialDispatch(body) } else { body() }
    }

    /// Benchmarks a symmetric matrix operation, with optional serial-only dispatch.
    func benchSymmetric<T>(_ make: (Int) -> [T], serial: Bool = false, _ op: @escaping (Tensor<T>) throws -> Any) throws
    {
        let depth = T.self == U1x8.self ? matrixDepth / 8 : matrixDepth
        let a = try Tensor<T>.fromArray(make(matrixHeight * depth), rows: matrixHeight, cols: depth)
        let body = { self.measure(metrics: [XCTClockMetric()]) { _ = try! op(a) } }
        if serial { withSerialDispatch(body) } else { body() }
    }
}

// MARK: - Info

final class BenchInfo: XCTestCase {
    func testCapabilities() {
        let caps = nk_capabilities_available()
        let usesRuntimeDispatch = nk_uses_runtime_dispatch()
        print("Capabilities: \(caps) [\(capabilityNames(caps))]")
        print("Runtime dispatch: \(usesRuntimeDispatch != 0 ? "YES" : "NO")")
        print("Dense dimensions: \(denseDims)")
        print("Matrix: \(matrixHeight)×\(matrixDepth) × \(matrixWidth)×\(matrixDepth)")
    }
}

// MARK: - Pairwise: Dot

final class BenchDot: XCTestCase {
    func testFloat64() { benchPairwise(randomFloat64) { $0.dot($1) } }
    func testFloat64Serial() { benchPairwise(randomFloat64, { $0.dot($1) }, serial: true) }
    func testFloat32() { benchPairwise(randomFloat32) { $0.dot($1) } }
    func testFloat32Serial() { benchPairwise(randomFloat32, { $0.dot($1) }, serial: true) }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() { benchPairwise(randomFloat16) { $0.dot($1) } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() { benchPairwise(randomFloat16, { $0.dot($1) }, serial: true) }
    #endif  // !arch(x86_64)
    func testBFloat16() { benchPairwise(randomBFloat16) { $0.dot($1) } }
    func testBFloat16Serial() { benchPairwise(randomBFloat16, { $0.dot($1) }, serial: true) }
    func testInt8() { benchPairwise(randomInt8) { $0.dot($1) } }
    func testInt8Serial() { benchPairwise(randomInt8, { $0.dot($1) }, serial: true) }
    func testUInt8() { benchPairwise(randomUInt8) { $0.dot($1) } }
    func testUInt8Serial() { benchPairwise(randomUInt8, { $0.dot($1) }, serial: true) }
    func testE4M3() { benchPairwise(randomE4M3) { $0.dot($1) } }
    func testE4M3Serial() { benchPairwise(randomE4M3, { $0.dot($1) }, serial: true) }
    func testE5M2() { benchPairwise(randomE5M2) { $0.dot($1) } }
    func testE5M2Serial() { benchPairwise(randomE5M2, { $0.dot($1) }, serial: true) }
    func testE2M3() { benchPairwise(randomE2M3) { $0.dot($1) } }
    func testE2M3Serial() { benchPairwise(randomE2M3, { $0.dot($1) }, serial: true) }
    func testE3M2() { benchPairwise(randomE3M2) { $0.dot($1) } }
    func testE3M2Serial() { benchPairwise(randomE3M2, { $0.dot($1) }, serial: true) }
    func testU1x8() { benchPairwise(randomU1x8) { $0.dot($1) } }
    func testU1x8Serial() { benchPairwise(randomU1x8, { $0.dot($1) }, serial: true) }
}

// MARK: - Pairwise: Angular

final class BenchAngular: XCTestCase {
    func testFloat64() { benchPairwise(randomFloat64) { $0.angular($1) } }
    func testFloat64Serial() { benchPairwise(randomFloat64, { $0.angular($1) }, serial: true) }
    func testFloat32() { benchPairwise(randomFloat32) { $0.angular($1) } }
    func testFloat32Serial() { benchPairwise(randomFloat32, { $0.angular($1) }, serial: true) }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() { benchPairwise(randomFloat16) { $0.angular($1) } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() { benchPairwise(randomFloat16, { $0.angular($1) }, serial: true) }
    #endif  // !arch(x86_64)
    func testBFloat16() { benchPairwise(randomBFloat16) { $0.angular($1) } }
    func testBFloat16Serial() { benchPairwise(randomBFloat16, { $0.angular($1) }, serial: true) }
    func testInt8() { benchPairwise(randomInt8) { $0.angular($1) } }
    func testInt8Serial() { benchPairwise(randomInt8, { $0.angular($1) }, serial: true) }
    func testUInt8() { benchPairwise(randomUInt8) { $0.angular($1) } }
    func testUInt8Serial() { benchPairwise(randomUInt8, { $0.angular($1) }, serial: true) }
    func testE4M3() { benchPairwise(randomE4M3) { $0.angular($1) } }
    func testE4M3Serial() { benchPairwise(randomE4M3, { $0.angular($1) }, serial: true) }
    func testE5M2() { benchPairwise(randomE5M2) { $0.angular($1) } }
    func testE5M2Serial() { benchPairwise(randomE5M2, { $0.angular($1) }, serial: true) }
    func testE2M3() { benchPairwise(randomE2M3) { $0.angular($1) } }
    func testE2M3Serial() { benchPairwise(randomE2M3, { $0.angular($1) }, serial: true) }
    func testE3M2() { benchPairwise(randomE3M2) { $0.angular($1) } }
    func testE3M2Serial() { benchPairwise(randomE3M2, { $0.angular($1) }, serial: true) }
}

// MARK: - Pairwise: Euclidean

final class BenchEuclidean: XCTestCase {
    func testFloat64() { benchPairwise(randomFloat64) { $0.euclidean($1) } }
    func testFloat64Serial() { benchPairwise(randomFloat64, { $0.euclidean($1) }, serial: true) }
    func testFloat32() { benchPairwise(randomFloat32) { $0.euclidean($1) } }
    func testFloat32Serial() { benchPairwise(randomFloat32, { $0.euclidean($1) }, serial: true) }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() { benchPairwise(randomFloat16) { $0.euclidean($1) } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() { benchPairwise(randomFloat16, { $0.euclidean($1) }, serial: true) }
    #endif  // !arch(x86_64)
    func testBFloat16() { benchPairwise(randomBFloat16) { $0.euclidean($1) } }
    func testBFloat16Serial() { benchPairwise(randomBFloat16, { $0.euclidean($1) }, serial: true) }
    func testInt8() { benchPairwise(randomInt8) { $0.euclidean($1) } }
    func testInt8Serial() { benchPairwise(randomInt8, { $0.euclidean($1) }, serial: true) }
    func testUInt8() { benchPairwise(randomUInt8) { $0.euclidean($1) } }
    func testUInt8Serial() { benchPairwise(randomUInt8, { $0.euclidean($1) }, serial: true) }
    func testE4M3() { benchPairwise(randomE4M3) { $0.euclidean($1) } }
    func testE4M3Serial() { benchPairwise(randomE4M3, { $0.euclidean($1) }, serial: true) }
    func testE5M2() { benchPairwise(randomE5M2) { $0.euclidean($1) } }
    func testE5M2Serial() { benchPairwise(randomE5M2, { $0.euclidean($1) }, serial: true) }
    func testE2M3() { benchPairwise(randomE2M3) { $0.euclidean($1) } }
    func testE2M3Serial() { benchPairwise(randomE2M3, { $0.euclidean($1) }, serial: true) }
    func testE3M2() { benchPairwise(randomE3M2) { $0.euclidean($1) } }
    func testE3M2Serial() { benchPairwise(randomE3M2, { $0.euclidean($1) }, serial: true) }
}

// MARK: - Pairwise: Squared Euclidean

final class BenchSqEuclidean: XCTestCase {
    func testFloat64() { benchPairwise(randomFloat64) { $0.sqeuclidean($1) } }
    func testFloat64Serial() { benchPairwise(randomFloat64, { $0.sqeuclidean($1) }, serial: true) }
    func testFloat32() { benchPairwise(randomFloat32) { $0.sqeuclidean($1) } }
    func testFloat32Serial() { benchPairwise(randomFloat32, { $0.sqeuclidean($1) }, serial: true) }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() { benchPairwise(randomFloat16) { $0.sqeuclidean($1) } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() { benchPairwise(randomFloat16, { $0.sqeuclidean($1) }, serial: true) }
    #endif  // !arch(x86_64)
    func testBFloat16() { benchPairwise(randomBFloat16) { $0.sqeuclidean($1) } }
    func testBFloat16Serial() { benchPairwise(randomBFloat16, { $0.sqeuclidean($1) }, serial: true) }
    func testInt8() { benchPairwise(randomInt8) { $0.sqeuclidean($1) } }
    func testInt8Serial() { benchPairwise(randomInt8, { $0.sqeuclidean($1) }, serial: true) }
    func testUInt8() { benchPairwise(randomUInt8) { $0.sqeuclidean($1) } }
    func testUInt8Serial() { benchPairwise(randomUInt8, { $0.sqeuclidean($1) }, serial: true) }
    func testE4M3() { benchPairwise(randomE4M3) { $0.sqeuclidean($1) } }
    func testE4M3Serial() { benchPairwise(randomE4M3, { $0.sqeuclidean($1) }, serial: true) }
    func testE5M2() { benchPairwise(randomE5M2) { $0.sqeuclidean($1) } }
    func testE5M2Serial() { benchPairwise(randomE5M2, { $0.sqeuclidean($1) }, serial: true) }
    func testE2M3() { benchPairwise(randomE2M3) { $0.sqeuclidean($1) } }
    func testE2M3Serial() { benchPairwise(randomE2M3, { $0.sqeuclidean($1) }, serial: true) }
    func testE3M2() { benchPairwise(randomE3M2) { $0.sqeuclidean($1) } }
    func testE3M2Serial() { benchPairwise(randomE3M2, { $0.sqeuclidean($1) }, serial: true) }
}

// MARK: - Pairwise: Hamming (binary)

final class BenchHamming: XCTestCase {
    func testU1x8() { benchPairwise(randomU1x8) { $0.hamming($1) } }
    func testU1x8Serial() { benchPairwise(randomU1x8, { $0.hamming($1) }, serial: true) }
}

// MARK: - Pairwise: Jaccard (binary)

final class BenchJaccard: XCTestCase {
    func testU1x8() { benchPairwise(randomU1x8) { $0.jaccard($1) } }
    func testU1x8Serial() { benchPairwise(randomU1x8, { $0.jaccard($1) }, serial: true) }
}

// MARK: - Packed: Dots

final class BenchDotsPacked: XCTestCase {
    func testFloat64() throws { try benchPacked(randomFloat64) { try $0.dotsPacked($1) } }
    func testFloat64Serial() throws { try benchPacked(randomFloat64, serial: true) { try $0.dotsPacked($1) } }
    func testFloat32() throws { try benchPacked(randomFloat32) { try $0.dotsPacked($1) } }
    func testFloat32Serial() throws { try benchPacked(randomFloat32, serial: true) { try $0.dotsPacked($1) } }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() throws { try benchPacked(randomFloat16) { try $0.dotsPacked($1) } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() throws { try benchPacked(randomFloat16, serial: true) { try $0.dotsPacked($1) } }
    #endif  // !arch(x86_64)
    func testBFloat16() throws { try benchPacked(randomBFloat16) { try $0.dotsPacked($1) } }
    func testBFloat16Serial() throws { try benchPacked(randomBFloat16, serial: true) { try $0.dotsPacked($1) } }
    func testInt8() throws { try benchPacked(randomInt8) { try $0.dotsPacked($1) } }
    func testInt8Serial() throws { try benchPacked(randomInt8, serial: true) { try $0.dotsPacked($1) } }
    func testUInt8() throws { try benchPacked(randomUInt8) { try $0.dotsPacked($1) } }
    func testUInt8Serial() throws { try benchPacked(randomUInt8, serial: true) { try $0.dotsPacked($1) } }
    func testE4M3() throws { try benchPacked(randomE4M3) { try $0.dotsPacked($1) } }
    func testE4M3Serial() throws { try benchPacked(randomE4M3, serial: true) { try $0.dotsPacked($1) } }
    func testE5M2() throws { try benchPacked(randomE5M2) { try $0.dotsPacked($1) } }
    func testE5M2Serial() throws { try benchPacked(randomE5M2, serial: true) { try $0.dotsPacked($1) } }
    func testE2M3() throws { try benchPacked(randomE2M3) { try $0.dotsPacked($1) } }
    func testE2M3Serial() throws { try benchPacked(randomE2M3, serial: true) { try $0.dotsPacked($1) } }
    func testE3M2() throws { try benchPacked(randomE3M2) { try $0.dotsPacked($1) } }
    func testE3M2Serial() throws { try benchPacked(randomE3M2, serial: true) { try $0.dotsPacked($1) } }
    func testU1x8() throws { try benchPacked(randomU1x8) { try $0.dotsPacked($1) } }
    func testU1x8Serial() throws { try benchPacked(randomU1x8, serial: true) { try $0.dotsPacked($1) } }
}

// MARK: - Packed: Angulars

final class BenchAngularsPacked: XCTestCase {
    func testFloat64() throws { try benchPacked(randomFloat64) { try $0.angularsPacked($1) } }
    func testFloat64Serial() throws { try benchPacked(randomFloat64, serial: true) { try $0.angularsPacked($1) } }
    func testFloat32() throws { try benchPacked(randomFloat32) { try $0.angularsPacked($1) } }
    func testFloat32Serial() throws { try benchPacked(randomFloat32, serial: true) { try $0.angularsPacked($1) } }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() throws { try benchPacked(randomFloat16) { try $0.angularsPacked($1) } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() throws { try benchPacked(randomFloat16, serial: true) { try $0.angularsPacked($1) } }
    #endif  // !arch(x86_64)
    func testBFloat16() throws { try benchPacked(randomBFloat16) { try $0.angularsPacked($1) } }
    func testBFloat16Serial() throws { try benchPacked(randomBFloat16, serial: true) { try $0.angularsPacked($1) } }
    func testInt8() throws { try benchPacked(randomInt8) { try $0.angularsPacked($1) } }
    func testInt8Serial() throws { try benchPacked(randomInt8, serial: true) { try $0.angularsPacked($1) } }
    func testUInt8() throws { try benchPacked(randomUInt8) { try $0.angularsPacked($1) } }
    func testUInt8Serial() throws { try benchPacked(randomUInt8, serial: true) { try $0.angularsPacked($1) } }
    func testE4M3() throws { try benchPacked(randomE4M3) { try $0.angularsPacked($1) } }
    func testE4M3Serial() throws { try benchPacked(randomE4M3, serial: true) { try $0.angularsPacked($1) } }
    func testE5M2() throws { try benchPacked(randomE5M2) { try $0.angularsPacked($1) } }
    func testE5M2Serial() throws { try benchPacked(randomE5M2, serial: true) { try $0.angularsPacked($1) } }
    func testE2M3() throws { try benchPacked(randomE2M3) { try $0.angularsPacked($1) } }
    func testE2M3Serial() throws { try benchPacked(randomE2M3, serial: true) { try $0.angularsPacked($1) } }
    func testE3M2() throws { try benchPacked(randomE3M2) { try $0.angularsPacked($1) } }
    func testE3M2Serial() throws { try benchPacked(randomE3M2, serial: true) { try $0.angularsPacked($1) } }
}

// MARK: - Packed: Euclideans

final class BenchEuclideansPacked: XCTestCase {
    func testFloat64() throws { try benchPacked(randomFloat64) { try $0.euclideansPacked($1) } }
    func testFloat64Serial() throws { try benchPacked(randomFloat64, serial: true) { try $0.euclideansPacked($1) } }
    func testFloat32() throws { try benchPacked(randomFloat32) { try $0.euclideansPacked($1) } }
    func testFloat32Serial() throws { try benchPacked(randomFloat32, serial: true) { try $0.euclideansPacked($1) } }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() throws { try benchPacked(randomFloat16) { try $0.euclideansPacked($1) } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() throws { try benchPacked(randomFloat16, serial: true) { try $0.euclideansPacked($1) } }
    #endif  // !arch(x86_64)
    func testBFloat16() throws { try benchPacked(randomBFloat16) { try $0.euclideansPacked($1) } }
    func testBFloat16Serial() throws { try benchPacked(randomBFloat16, serial: true) { try $0.euclideansPacked($1) } }
    func testInt8() throws { try benchPacked(randomInt8) { try $0.euclideansPacked($1) } }
    func testInt8Serial() throws { try benchPacked(randomInt8, serial: true) { try $0.euclideansPacked($1) } }
    func testUInt8() throws { try benchPacked(randomUInt8) { try $0.euclideansPacked($1) } }
    func testUInt8Serial() throws { try benchPacked(randomUInt8, serial: true) { try $0.euclideansPacked($1) } }
    func testE4M3() throws { try benchPacked(randomE4M3) { try $0.euclideansPacked($1) } }
    func testE4M3Serial() throws { try benchPacked(randomE4M3, serial: true) { try $0.euclideansPacked($1) } }
    func testE5M2() throws { try benchPacked(randomE5M2) { try $0.euclideansPacked($1) } }
    func testE5M2Serial() throws { try benchPacked(randomE5M2, serial: true) { try $0.euclideansPacked($1) } }
    func testE2M3() throws { try benchPacked(randomE2M3) { try $0.euclideansPacked($1) } }
    func testE2M3Serial() throws { try benchPacked(randomE2M3, serial: true) { try $0.euclideansPacked($1) } }
    func testE3M2() throws { try benchPacked(randomE3M2) { try $0.euclideansPacked($1) } }
    func testE3M2Serial() throws { try benchPacked(randomE3M2, serial: true) { try $0.euclideansPacked($1) } }
}

// MARK: - Packed: Hammings (binary)

final class BenchHammingsPacked: XCTestCase {
    func testU1x8() throws { try benchPacked(randomU1x8) { try $0.hammingsPacked($1) } }
    func testU1x8Serial() throws { try benchPacked(randomU1x8, serial: true) { try $0.hammingsPacked($1) } }
}

// MARK: - Packed: Jaccards (binary)

final class BenchJaccardsPacked: XCTestCase {
    func testU1x8() throws { try benchPacked(randomU1x8) { try $0.jaccardsPacked($1) } }
    func testU1x8Serial() throws { try benchPacked(randomU1x8, serial: true) { try $0.jaccardsPacked($1) } }
}

// MARK: - Symmetric: Dots

final class BenchDotsSymmetric: XCTestCase {
    func testFloat64() throws { try benchSymmetric(randomFloat64) { try $0.dotsSymmetric() } }
    func testFloat64Serial() throws { try benchSymmetric(randomFloat64, serial: true) { try $0.dotsSymmetric() } }
    func testFloat32() throws { try benchSymmetric(randomFloat32) { try $0.dotsSymmetric() } }
    func testFloat32Serial() throws { try benchSymmetric(randomFloat32, serial: true) { try $0.dotsSymmetric() } }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() throws { try benchSymmetric(randomFloat16) { try $0.dotsSymmetric() } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() throws { try benchSymmetric(randomFloat16, serial: true) { try $0.dotsSymmetric() } }
    #endif  // !arch(x86_64)
    func testBFloat16() throws { try benchSymmetric(randomBFloat16) { try $0.dotsSymmetric() } }
    func testBFloat16Serial() throws { try benchSymmetric(randomBFloat16, serial: true) { try $0.dotsSymmetric() } }
    func testInt8() throws { try benchSymmetric(randomInt8) { try $0.dotsSymmetric() } }
    func testInt8Serial() throws { try benchSymmetric(randomInt8, serial: true) { try $0.dotsSymmetric() } }
    func testUInt8() throws { try benchSymmetric(randomUInt8) { try $0.dotsSymmetric() } }
    func testUInt8Serial() throws { try benchSymmetric(randomUInt8, serial: true) { try $0.dotsSymmetric() } }
    func testE4M3() throws { try benchSymmetric(randomE4M3) { try $0.dotsSymmetric() } }
    func testE4M3Serial() throws { try benchSymmetric(randomE4M3, serial: true) { try $0.dotsSymmetric() } }
    func testE5M2() throws { try benchSymmetric(randomE5M2) { try $0.dotsSymmetric() } }
    func testE5M2Serial() throws { try benchSymmetric(randomE5M2, serial: true) { try $0.dotsSymmetric() } }
    func testE2M3() throws { try benchSymmetric(randomE2M3) { try $0.dotsSymmetric() } }
    func testE2M3Serial() throws { try benchSymmetric(randomE2M3, serial: true) { try $0.dotsSymmetric() } }
    func testE3M2() throws { try benchSymmetric(randomE3M2) { try $0.dotsSymmetric() } }
    func testE3M2Serial() throws { try benchSymmetric(randomE3M2, serial: true) { try $0.dotsSymmetric() } }
    func testU1x8() throws { try benchSymmetric(randomU1x8) { try $0.dotsSymmetric() } }
    func testU1x8Serial() throws { try benchSymmetric(randomU1x8, serial: true) { try $0.dotsSymmetric() } }
}

// MARK: - Symmetric: Angulars

final class BenchAngularsSymmetric: XCTestCase {
    func testFloat64() throws { try benchSymmetric(randomFloat64) { try $0.angularsSymmetric() } }
    func testFloat64Serial() throws { try benchSymmetric(randomFloat64, serial: true) { try $0.angularsSymmetric() } }
    func testFloat32() throws { try benchSymmetric(randomFloat32) { try $0.angularsSymmetric() } }
    func testFloat32Serial() throws { try benchSymmetric(randomFloat32, serial: true) { try $0.angularsSymmetric() } }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() throws { try benchSymmetric(randomFloat16) { try $0.angularsSymmetric() } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() throws {
        try benchSymmetric(randomFloat16, serial: true) { try $0.angularsSymmetric() }
    }
    #endif  // !arch(x86_64)
    func testBFloat16() throws { try benchSymmetric(randomBFloat16) { try $0.angularsSymmetric() } }
    func testBFloat16Serial() throws { try benchSymmetric(randomBFloat16, serial: true) { try $0.angularsSymmetric() } }
    func testInt8() throws { try benchSymmetric(randomInt8) { try $0.angularsSymmetric() } }
    func testInt8Serial() throws { try benchSymmetric(randomInt8, serial: true) { try $0.angularsSymmetric() } }
    func testUInt8() throws { try benchSymmetric(randomUInt8) { try $0.angularsSymmetric() } }
    func testUInt8Serial() throws { try benchSymmetric(randomUInt8, serial: true) { try $0.angularsSymmetric() } }
    func testE4M3() throws { try benchSymmetric(randomE4M3) { try $0.angularsSymmetric() } }
    func testE4M3Serial() throws { try benchSymmetric(randomE4M3, serial: true) { try $0.angularsSymmetric() } }
    func testE5M2() throws { try benchSymmetric(randomE5M2) { try $0.angularsSymmetric() } }
    func testE5M2Serial() throws { try benchSymmetric(randomE5M2, serial: true) { try $0.angularsSymmetric() } }
    func testE2M3() throws { try benchSymmetric(randomE2M3) { try $0.angularsSymmetric() } }
    func testE2M3Serial() throws { try benchSymmetric(randomE2M3, serial: true) { try $0.angularsSymmetric() } }
    func testE3M2() throws { try benchSymmetric(randomE3M2) { try $0.angularsSymmetric() } }
    func testE3M2Serial() throws { try benchSymmetric(randomE3M2, serial: true) { try $0.angularsSymmetric() } }
}

// MARK: - Symmetric: Euclideans

final class BenchEuclideansSymmetric: XCTestCase {
    func testFloat64() throws { try benchSymmetric(randomFloat64) { try $0.euclideansSymmetric() } }
    func testFloat64Serial() throws { try benchSymmetric(randomFloat64, serial: true) { try $0.euclideansSymmetric() } }
    func testFloat32() throws { try benchSymmetric(randomFloat32) { try $0.euclideansSymmetric() } }
    func testFloat32Serial() throws { try benchSymmetric(randomFloat32, serial: true) { try $0.euclideansSymmetric() } }
    #if !arch(x86_64)
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16() throws { try benchSymmetric(randomFloat16) { try $0.euclideansSymmetric() } }
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    func testFloat16Serial() throws {
        try benchSymmetric(randomFloat16, serial: true) { try $0.euclideansSymmetric() }
    }
    #endif  // !arch(x86_64)
    func testBFloat16() throws { try benchSymmetric(randomBFloat16) { try $0.euclideansSymmetric() } }
    func testBFloat16Serial() throws {
        try benchSymmetric(randomBFloat16, serial: true) { try $0.euclideansSymmetric() }
    }
    func testInt8() throws { try benchSymmetric(randomInt8) { try $0.euclideansSymmetric() } }
    func testInt8Serial() throws { try benchSymmetric(randomInt8, serial: true) { try $0.euclideansSymmetric() } }
    func testUInt8() throws { try benchSymmetric(randomUInt8) { try $0.euclideansSymmetric() } }
    func testUInt8Serial() throws { try benchSymmetric(randomUInt8, serial: true) { try $0.euclideansSymmetric() } }
    func testE4M3() throws { try benchSymmetric(randomE4M3) { try $0.euclideansSymmetric() } }
    func testE4M3Serial() throws { try benchSymmetric(randomE4M3, serial: true) { try $0.euclideansSymmetric() } }
    func testE5M2() throws { try benchSymmetric(randomE5M2) { try $0.euclideansSymmetric() } }
    func testE5M2Serial() throws { try benchSymmetric(randomE5M2, serial: true) { try $0.euclideansSymmetric() } }
    func testE2M3() throws { try benchSymmetric(randomE2M3) { try $0.euclideansSymmetric() } }
    func testE2M3Serial() throws { try benchSymmetric(randomE2M3, serial: true) { try $0.euclideansSymmetric() } }
    func testE3M2() throws { try benchSymmetric(randomE3M2) { try $0.euclideansSymmetric() } }
    func testE3M2Serial() throws { try benchSymmetric(randomE3M2, serial: true) { try $0.euclideansSymmetric() } }
}

// MARK: - Symmetric: Hammings (binary)

final class BenchHammingsSymmetric: XCTestCase {
    func testU1x8() throws { try benchSymmetric(randomU1x8) { try $0.hammingsSymmetric() } }
    func testU1x8Serial() throws { try benchSymmetric(randomU1x8, serial: true) { try $0.hammingsSymmetric() } }
}

// MARK: - Symmetric: Jaccards (binary)

final class BenchJaccardsSymmetric: XCTestCase {
    func testU1x8() throws { try benchSymmetric(randomU1x8) { try $0.jaccardsSymmetric() } }
    func testU1x8Serial() throws { try benchSymmetric(randomU1x8, serial: true) { try $0.jaccardsSymmetric() } }
}

#endif  // canImport(Darwin)
