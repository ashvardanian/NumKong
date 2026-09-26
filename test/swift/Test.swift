//
//  test/swift/Test.swift
//  Swift Testing correctness tests for the NumKong Swift bindings.
//
//  - Author: Ash Vardanian
//  - Date: March 14, 2026
//

import NumKong
import Testing

/// One kernel call on fixed inputs, and the value it must land within `tolerance` of.
struct Kernel: Sendable, CustomTestStringConvertible {
    let testDescription: String
    let expected: Double
    let tolerance: Double
    let run: @Sendable () -> Double?

    init(
        _ name: String, _ expected: Double, within tolerance: Double = 0.01,
        _ run: @escaping @Sendable () -> Double?
    ) {
        self.testDescription = name
        self.expected = expected
        self.tolerance = tolerance
        self.run = run
    }
}

/// New York then London, as latitude and longitude in radians.
let newYorkLondon: [Float64] = [40.7128, -74.0060, 51.5074, -0.1278].map { $0 * .pi / 180 }

/// Calls a buffer-pointer geodesic on one coordinate pair, the path the sequence overloads bypass.
func viaBuffers<T: BinaryFloatingPoint>(
    _ kernel: (
        UnsafeBufferPointer<T>, UnsafeBufferPointer<T>, UnsafeBufferPointer<T>, UnsafeBufferPointer<T>,
        UnsafeMutableBufferPointer<T>
    ) -> Bool,
    _ coordinates: [T]
) -> Double? {
    var result: [T] = [0]
    let succeeded = coordinates.withUnsafeBufferPointer { c in
        result.withUnsafeMutableBufferPointer { r in
            kernel(
                .init(rebasing: c[0..<1]), .init(rebasing: c[1..<2]), .init(rebasing: c[2..<3]),
                .init(rebasing: c[3..<4]), r)
        }
    }
    return succeeded ? Double(result[0]) : nil
}

let kernels: [Kernel] = {
    let c64 = newYorkLondon
    let c32 = newYorkLondon.map(Float32.init)
    var kernels = [
        Kernel("Int8 angular", 0) { [Int8](arrayLiteral: 3, 97, 127).angular([3, 97, 127]).map(Double.init) },
        Kernel("Float32 angular", 0) { [Float32](arrayLiteral: 1, 2, 3).angular([1, 2, 3]) },
        Kernel("Float64 angular", 0) { [Float64](arrayLiteral: 1, 2, 3).angular([1, 2, 3]) },
        Kernel("Int8 dot", 32, within: 0) { [Int8](arrayLiteral: 1, 2, 3).dot([4, 5, 6]).map(Double.init) },
        Kernel("Float32 dot", 32) { [Float32](arrayLiteral: 1, 2, 3).dot([4, 5, 6]) },
        Kernel("Float64 dot", 32) { [Float64](arrayLiteral: 1, 2, 3).dot([4, 5, 6]) },
        Kernel("Int8 euclidean", 27.0.squareRoot()) {
            [Int8](arrayLiteral: 1, 2, 3).euclidean([4, 5, 6]).map(Double.init)
        },
        Kernel("Float32 euclidean", 27.0.squareRoot()) { [Float32](arrayLiteral: 1, 2, 3).euclidean([4, 5, 6]) },
        Kernel("Float64 euclidean", 27.0.squareRoot()) { [Float64](arrayLiteral: 1, 2, 3).euclidean([4, 5, 6]) },
        Kernel("Int8 sqeuclidean", 27, within: 0) {
            [Int8](arrayLiteral: 1, 2, 3).sqeuclidean([4, 5, 6]).map(Double.init)
        },
        Kernel("Float32 sqeuclidean", 27) { [Float32](arrayLiteral: 1, 2, 3).sqeuclidean([4, 5, 6]) },
        Kernel("Float64 sqeuclidean", 27) { [Float64](arrayLiteral: 1, 2, 3).sqeuclidean([4, 5, 6]) },
        Kernel("E4M3 dot", 32, within: 1.5) {
            [E4M3(float: 1), E4M3(float: 2), E4M3(float: 3)]
                .dot([E4M3(float: 4), E4M3(float: 5), E4M3(float: 6)]).map(Double.init)
        },
        // Lanes (1, 2, 3, 4) . (4, 5, 6, 7) = 4 + 10 + 18 + 28
        Kernel("I4x2 dot", 60, within: 0) {
            [I4x2(first: 1, second: 2), I4x2(first: 3, second: 4)]
                .dot([I4x2(first: 4, second: 5), I4x2(first: 6, second: 7)]).map(Double.init)
        },
        // Lanes (15, 15, 0, 1) . (15, 1, 15, 15) = 225 + 15 + 0 + 15
        Kernel("U4x2 dot", 255, within: 0) {
            [U4x2(first: 15, second: 15), U4x2(first: 0, second: 1)]
                .dot([U4x2(first: 15, second: 1), U4x2(first: 15, second: 15)]).map(Double.init)
        },
        Kernel("I4x2 angular", 0) {
            let a = [I4x2(first: -5, second: 3), I4x2(first: 1, second: 7)]
            return a.angular(a).map(Double.init)
        },
        Kernel("U1x8 hamming", 16, within: 0) {
            [U1x8(0xFF), U1x8(0xFF)].hamming([U1x8(0x00), U1x8(0x00)]).map(Double.init)
        },
        Kernel("U1x8 jaccard", 0) { [U1x8(0xFF)].jaccard([U1x8(0xFF)]).map(Double.init) },
        Kernel("Float64 haversine buffers", 5_539_000, within: 5_000) {
            viaBuffers(Float64.haversine(aLat:aLon:bLat:bLon:result:), c64)
        },
        Kernel("Float32 haversine buffers", 5_539_000, within: 5_000) {
            viaBuffers(Float32.haversine(aLat:aLon:bLat:bLon:result:), c32)
        },
        Kernel("Float64 vincenty buffers", 5_570_000, within: 20_000) {
            viaBuffers(Float64.vincenty(aLat:aLon:bLat:bLon:result:), c64)
        },
        Kernel("Float32 vincenty buffers", 5_570_000, within: 50_000) {
            viaBuffers(Float32.vincenty(aLat:aLon:bLat:bLon:result:), c32)
        },
        Kernel("Float64 haversine", 5_539_000, within: 5_000) {
            haversine(aLat: [c64[0]], aLon: [c64[1]], bLat: [c64[2]], bLon: [c64[3]])?.first
        },
        Kernel("Float32 haversine", 5_539_000, within: 5_000) {
            haversine(aLat: [c32[0]], aLon: [c32[1]], bLat: [c32[2]], bLon: [c32[3]])?.first.map(Double.init)
        },
        Kernel("Float64 vincenty", 5_570_000, within: 20_000) {
            vincenty(aLat: [c64[0]], aLon: [c64[1]], bLat: [c64[2]], bLon: [c64[3]])?.first
        },
        Kernel("Float32 vincenty", 5_570_000, within: 50_000) {
            vincenty(aLat: [c32[0]], aLon: [c32[1]], bLat: [c32[2]], bLon: [c32[3]])?.first.map(Double.init)
        },
    ]
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    kernels += [
        Kernel("Float16 angular", 0) { [Float16](arrayLiteral: 1, 2, 3).angular([1, 2, 3]).map(Double.init) },
        Kernel("Float16 dot", 32) { [Float16](arrayLiteral: 1, 2, 3).dot([4, 5, 6]).map(Double.init) },
        Kernel("Float16 euclidean", 27.0.squareRoot()) {
            [Float16](arrayLiteral: 1, 2, 3).euclidean([4, 5, 6]).map(Double.init)
        },
        Kernel("Float16 sqeuclidean", 27) {
            [Float16](arrayLiteral: 1, 2, 3).sqeuclidean([4, 5, 6]).map(Double.init)
        },
    ]
    #endif
    return kernels
}()

@Test func capabilities() {
    print("Capabilities: \(Capabilities.enabled)")
    let names: [(Capabilities, String)] = [
        (.serial, "serial"), (.neon, "neon"), (.haswell, "haswell"), (.skylake, "skylake"),
        (.neonHalf, "neonhalf"), (.neonSDot, "neonsdot"), (.neonFhm, "neonfhm"), (.icelake, "icelake"),
        (.genoa, "genoa"), (.neonBfDot, "neonbfdot"), (.sve, "sve"), (.sveHalf, "svehalf"),
        (.sveSDot, "svesdot"), (.alder, "alder"), (.sveBfDot, "svebfdot"), (.sve2, "sve2"),
        (.v128Relaxed, "v128relaxed"), (.sapphire, "sapphire"), (.sapphireAmx, "sapphireamx"), (.rvv, "rvv"),
        (.rvvHalf, "rvvhalf"), (.rvvBf16, "rvvbf16"), (.graniteAmx, "graniteamx"), (.turin, "turin"),
        (.sme, "sme"), (.sme2, "sme2"), (.smeF64, "smef64"), (.smeFa64, "smefa64"),
        (.sve2p1, "sve2p1"), (.sme2p1, "sme2p1"), (.smeHalf, "smehalf"), (.smeBf16, "smebf16"),
        (.smeLut2, "smelut2"), (.rvvBB, "rvvbb"), (.sierra, "sierra"), (.smeBi32, "smebi32"),
        (.loongsonAsx, "loongsonasx"), (.powerVsx, "powervsx"), (.diamond, "diamond"), (.neonFp8, "neonfp8"),
        (.diamondAmx, "diamondamx"), (.v128, "v128"),
    ]
    for (capability, name) in names { #expect(capability.description == name) }
}

@Test(arguments: kernels)
func kernel(_ kernel: Kernel) throws {
    let result = try #require(kernel.run())
    #expect(abs(result - kernel.expected) <= kernel.tolerance)
}

@Test func bfloat16Roundtrip() {
    #expect(BFloat16(float: 1.5).float == 1.5)
}

@Test func i4x2Lanes() {
    let x = I4x2(first: 7, second: -8)
    #expect(x.bitPattern == 0x78)
    #expect((x.first, x.second) == (7, -8))
    #expect((I4x2(bitPattern: 0xFF).first, I4x2(bitPattern: 0xFF).second) == (-1, -1))
}

@Test func u4x2Lanes() {
    let x = U4x2(first: 1, second: 15)
    #expect(x.bitPattern == 0x1F)
    #expect((x.first, x.second) == (1, 15))
}

@Test func u1x8Lanes() {
    let x = U1x8(0b10110011)
    #expect(x.bitPattern == 0b10110011)
    #expect(x.popcount == 5)
}

@Test func tensorFromArray() throws {
    let t = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6], rows: 2, cols: 3)
    #expect((t.rows, t.cols, t.count) == (2, 3, 6))
    #expect([t[0, 0], t[0, 2], t[1, 0], t[1, 2]] == [1, 3, 4, 6])
}

@Test func tensorZeros() throws {
    let t = try Tensor<Float32>.zeros(rows: 3, cols: 4)
    #expect((t.rows, t.cols) == (3, 4))
    #expect((0..<3).allSatisfy { t.row($0).allSatisfy { $0 == 0 } })
}

@Test func tensorRejectsEmpty() {
    #expect(throws: NumKongMatrixError.invalidDimensions) { try Tensor<Float32>.zeros(rows: 0, cols: 4) }
}

@Test func tensorResizeWithinCapacity() throws {
    let t = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6, 7, 8], rows: 2, cols: 4)
    #expect(t.tryResize(rows: 2, cols: 2))
    #expect((t.rows, t.cols, t.count, t.capacity) == (2, 2, 4, 8))
    #expect((t[0, 0], t[1, 1]) == (1, 4))  // storage never moves within capacity
    #expect(!t.tryResize(rows: 3, cols: 4))  // 12 > 8, left unchanged
    #expect((t.rows, t.cols) == (2, 2))
}

@Test func tensorReserveAndClear() throws {
    let t = try Tensor<Float32>.fromArray([1, 2, 3, 4], rows: 1, cols: 4)
    t.reserve(64)
    #expect(t.capacity >= 64)
    #expect(t.count == 4)  // reserve grows capacity, it does not reshape
    #expect((t[0, 0], t[0, 3]) == (1, 4))
    #expect(t.tryResize(rows: 8, cols: 8))
    #expect(t.count == 64)
    t.clear()
    #expect(t.count == 0)
    #expect(t.capacity >= 64)
}

@Test func tensorDotsPacked() throws {
    let a = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6], rows: 2, cols: 3)
    let b = try Tensor<Float32>.fromArray([7, 8, 9, 1, 0, 1], rows: 2, cols: 3)
    let result = try a.dotsPacked(b.packForDots())
    #expect((result.rows, result.cols) == (2, 2))
    #expect([result[0, 0], result[0, 1], result[1, 0], result[1, 1]] == [50, 4, 122, 10])
}

@Test func tensorAngularsPacked() throws {
    let a = try Tensor<Float32>.fromArray([1, 0, 0, 0, 1, 0], rows: 2, cols: 3)
    let result = try a.angularsPacked(a.packForDots())
    #expect(abs(result[0, 0]) <= 0.01)
    #expect(abs(result[0, 1] - 1) <= 0.05)
    #expect(abs(result[1, 0] - 1) <= 0.05)
    #expect(abs(result[1, 1]) <= 0.01)
}

@Test func hammingsPacked() throws {
    let a = try Tensor<U1x8>.fromArray([U1x8(0xFF), U1x8(0x00), U1x8(0x00), U1x8(0xFF)], rows: 2, cols: 2)
    let b = try Tensor<U1x8>.fromArray([U1x8(0xFF), U1x8(0xFF)], rows: 1, cols: 2)
    let result = try a.hammingsPacked(b.packForDots())
    #expect((result.rows, result.cols) == (2, 1))
    #expect([result[0, 0], result[1, 0]] == [8, 8])  // each row misses one full byte of all-ones
}

@Test func hammingsSymmetric() throws {
    let t = try Tensor<U1x8>.fromArray([U1x8(0xFF), U1x8(0x00), U1x8(0x0F)], rows: 3, cols: 1)
    let result = try t.hammingsSymmetric()
    #expect((result.rows, result.cols) == (3, 3))
    #expect([result[0, 0], result[1, 1], result[2, 2], result[0, 1]] == [0, 0, 0, 8])
}

@Test func jaccardsPacked() throws {
    let a = try Tensor<U1x8>.fromArray([U1x8(0xFF), U1x8(0x00)], rows: 1, cols: 2)
    let result = try a.jaccardsPacked(a.packForDots())
    #expect((result.rows, result.cols) == (1, 1))
    #expect(abs(result[0, 0]) <= 0.01)  // identical sets, and NaN fails this too
}

@Test func jaccardsSymmetric() throws {
    let t = try Tensor<U1x8>.fromArray([U1x8(0xFF), U1x8(0xFF), U1x8(0x00)], rows: 3, cols: 1)
    let result = try t.jaccardsSymmetric()
    #expect((result.rows, result.cols) == (3, 3))
    #expect(abs(result[0, 1]) <= 0.01)
}

@Test func maxSimFloat32() throws {
    let t = try Tensor<Float32>.fromArray([1, 0, 0, 0, 0, 1, 0, 0], rows: 2, cols: 4)
    #expect(try t.maxSimPack().score(t.maxSimPack()).isFinite)
}

@Test func maxSimBFloat16() throws {
    let t = try Tensor<BFloat16>.fromArray([1, 0, 0, 0, 0, 1, 0, 0].map { BFloat16(float: $0) }, rows: 2, cols: 4)
    #expect(try t.maxSimPack().score(t.maxSimPack()).isFinite)
}
