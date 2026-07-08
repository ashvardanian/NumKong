//  Test.swift
//  NumKong
//
//  Created by Ash Vardanian on March 14, 2026.
//

import NumKong
import XCTest

class NumKongTests: XCTestCase {
    override class func setUp() {
        print("Capabilities: \(Capabilities.available)")
    }

    func testAngularInt8() throws {
        let a: [Int8] = [3, 97, 127]
        let b: [Int8] = [3, 97, 127]
        let result = try XCTUnwrap(a.angular(b))
        XCTAssertEqual(result, 0.00012027938, accuracy: 0.01)
    }

    #if !arch(x86_64)
    func testAngularFloat16() throws {
        let a: [Float16] = [1.0, 2.0, 3.0]
        let b: [Float16] = [1.0, 2.0, 3.0]
        let result = try XCTUnwrap(a.angular(b))
        XCTAssertEqual(result, 0.004930496, accuracy: 0.01)
    }
    #endif

    func testAngularFloat32() throws {
        let a: [Float32] = [1.0, 2.0, 3.0]
        let b: [Float32] = [1.0, 2.0, 3.0]
        let result = try XCTUnwrap(a.angular(b))
        XCTAssertEqual(result, 0.004930496, accuracy: 0.01)
    }

    func testAngularFloat64() throws {
        let a: [Float64] = [1.0, 2.0, 3.0]
        let b: [Float64] = [1.0, 2.0, 3.0]
        let result = try XCTUnwrap(a.angular(b))
        XCTAssertEqual(result, 0.004930496, accuracy: 0.01)
    }

    func testInnerInt8() throws {
        let a: [Int8] = [1, 2, 3]
        let b: [Int8] = [4, 5, 6]
        let result = try XCTUnwrap(a.dot(b))
        XCTAssertEqual(result, 32)
    }

    #if !arch(x86_64)
    func testDotFloat16() throws {
        let a: [Float16] = [1.0, 2.0, 3.0]
        let b: [Float16] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.dot(b))
        XCTAssertEqual(result, 32.0, accuracy: 0.01)
    }
    #endif

    func testDotFloat32() throws {
        let a: [Float32] = [1.0, 2.0, 3.0]
        let b: [Float32] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.dot(b))
        XCTAssertEqual(result, 32.0, accuracy: 0.01)
    }

    func testDotFloat64() throws {
        let a: [Float64] = [1.0, 2.0, 3.0]
        let b: [Float64] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.dot(b))
        XCTAssertEqual(result, 32.0, accuracy: 0.01)
    }

    func testEuclideanInt8() throws {
        let a: [Int8] = [1, 2, 3]
        let b: [Int8] = [4, 5, 6]
        let result = try XCTUnwrap(a.euclidean(b))
        XCTAssertEqual(result, 5.196152422706632, accuracy: 0.01)
    }

    #if !arch(x86_64)
    func testEuclideanFloat16() throws {
        let a: [Float16] = [1.0, 2.0, 3.0]
        let b: [Float16] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.euclidean(b))
        XCTAssertEqual(result, 5.196152422706632, accuracy: 0.01)
    }
    #endif

    func testEuclideanFloat32() throws {
        let a: [Float32] = [1.0, 2.0, 3.0]
        let b: [Float32] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.euclidean(b))
        XCTAssertEqual(result, 5.196152422706632, accuracy: 0.01)
    }

    func testEuclideanFloat64() throws {
        let a: [Float64] = [1.0, 2.0, 3.0]
        let b: [Float64] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.euclidean(b))
        XCTAssertEqual(result, 5.196152422706632, accuracy: 0.01)
    }

    func testSqEuclideanInt8() throws {
        let a: [Int8] = [1, 2, 3]
        let b: [Int8] = [4, 5, 6]
        let result = try XCTUnwrap(a.sqeuclidean(b))
        XCTAssertEqual(result, 27)
    }

    #if !arch(x86_64)
    func testSqEuclideanFloat16() throws {
        let a: [Float16] = [1.0, 2.0, 3.0]
        let b: [Float16] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.sqeuclidean(b))
        XCTAssertEqual(result, 27.0, accuracy: 0.01)
    }
    #endif

    func testSqEuclideanFloat32() throws {
        let a: [Float32] = [1.0, 2.0, 3.0]
        let b: [Float32] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.sqeuclidean(b))
        XCTAssertEqual(result, 27.0, accuracy: 0.01)
    }

    func testSqEuclideanFloat64() throws {
        let a: [Float64] = [1.0, 2.0, 3.0]
        let b: [Float64] = [4.0, 5.0, 6.0]
        let result = try XCTUnwrap(a.sqeuclidean(b))
        XCTAssertEqual(result, 27.0, accuracy: 0.01)
    }

    // MARK: - Geospatial Tests

    func testHaversineFloat64() throws {
        let aLat: [Float64] = [40.7128 * .pi / 180.0]
        let aLon: [Float64] = [-74.0060 * .pi / 180.0]
        let bLat: [Float64] = [51.5074 * .pi / 180.0]
        let bLon: [Float64] = [-0.1278 * .pi / 180.0]
        var result: [Float64] = [0]

        let success = aLat.withUnsafeBufferPointer { aLatPtr in
            aLon.withUnsafeBufferPointer { aLonPtr in
                bLat.withUnsafeBufferPointer { bLatPtr in
                    bLon.withUnsafeBufferPointer { bLonPtr in
                        result.withUnsafeMutableBufferPointer { resultPtr in
                            Float64.haversine(
                                aLat: aLatPtr,
                                aLon: aLonPtr,
                                bLat: bLatPtr,
                                bLon: bLonPtr,
                                result: resultPtr
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(success)
        XCTAssertEqual(result[0], 5_539_000, accuracy: 5000)
    }

    func testHaversineFloat32() throws {
        let aLat: [Float32] = [40.7128 * .pi / 180.0]
        let aLon: [Float32] = [-74.0060 * .pi / 180.0]
        let bLat: [Float32] = [51.5074 * .pi / 180.0]
        let bLon: [Float32] = [-0.1278 * .pi / 180.0]
        var result: [Float32] = [0]

        let success = aLat.withUnsafeBufferPointer { aLatPtr in
            aLon.withUnsafeBufferPointer { aLonPtr in
                bLat.withUnsafeBufferPointer { bLatPtr in
                    bLon.withUnsafeBufferPointer { bLonPtr in
                        result.withUnsafeMutableBufferPointer { resultPtr in
                            Float32.haversine(
                                aLat: aLatPtr,
                                aLon: aLonPtr,
                                bLat: bLatPtr,
                                bLon: bLonPtr,
                                result: resultPtr
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(success)
        XCTAssertEqual(result[0], 5_539_000, accuracy: 5000)
    }

    func testVincentyFloat64() throws {
        let aLat: [Float64] = [40.7128 * .pi / 180.0]
        let aLon: [Float64] = [-74.0060 * .pi / 180.0]
        let bLat: [Float64] = [51.5074 * .pi / 180.0]
        let bLon: [Float64] = [-0.1278 * .pi / 180.0]
        var result: [Float64] = [0]

        let success = aLat.withUnsafeBufferPointer { aLatPtr in
            aLon.withUnsafeBufferPointer { aLonPtr in
                bLat.withUnsafeBufferPointer { bLatPtr in
                    bLon.withUnsafeBufferPointer { bLonPtr in
                        result.withUnsafeMutableBufferPointer { resultPtr in
                            Float64.vincenty(
                                aLat: aLatPtr,
                                aLon: aLonPtr,
                                bLat: bLatPtr,
                                bLon: bLonPtr,
                                result: resultPtr
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(success)
        XCTAssertEqual(result[0], 5_570_000, accuracy: 20000)
    }

    func testVincentyFloat32() throws {
        let aLat: [Float32] = [40.7128 * .pi / 180.0]
        let aLon: [Float32] = [-74.0060 * .pi / 180.0]
        let bLat: [Float32] = [51.5074 * .pi / 180.0]
        let bLon: [Float32] = [-0.1278 * .pi / 180.0]
        var result: [Float32] = [0]

        let success = aLat.withUnsafeBufferPointer { aLatPtr in
            aLon.withUnsafeBufferPointer { aLonPtr in
                bLat.withUnsafeBufferPointer { bLatPtr in
                    bLon.withUnsafeBufferPointer { bLonPtr in
                        result.withUnsafeMutableBufferPointer { resultPtr in
                            Float32.vincenty(
                                aLat: aLatPtr,
                                aLon: aLonPtr,
                                bLat: bLatPtr,
                                bLon: bLonPtr,
                                result: resultPtr
                            )
                        }
                    }
                }
            }
        }
        XCTAssertTrue(success)
        XCTAssertEqual(result[0], 5_570_000, accuracy: 50000)
    }

    // MARK: - Geospatial Free Function Tests

    func testHaversineFreeFloat64() throws {
        let aLat: [Float64] = [40.7128 * .pi / 180.0]
        let aLon: [Float64] = [-74.0060 * .pi / 180.0]
        let bLat: [Float64] = [51.5074 * .pi / 180.0]
        let bLon: [Float64] = [-0.1278 * .pi / 180.0]
        let result = try XCTUnwrap(haversine(aLat: aLat, aLon: aLon, bLat: bLat, bLon: bLon))
        XCTAssertEqual(result[0], 5_539_000, accuracy: 5000)
    }

    func testHaversineFreeFloat32() throws {
        let aLat: [Float32] = [40.7128 * .pi / 180.0]
        let aLon: [Float32] = [-74.0060 * .pi / 180.0]
        let bLat: [Float32] = [51.5074 * .pi / 180.0]
        let bLon: [Float32] = [-0.1278 * .pi / 180.0]
        let result: [Float32]? = haversine(aLat: aLat, aLon: aLon, bLat: bLat, bLon: bLon)
        let unwrapped = try XCTUnwrap(result)
        XCTAssertEqual(unwrapped[0], 5_539_000, accuracy: 5000)
    }

    func testVincentyFreeFloat64() throws {
        let aLat: [Float64] = [40.7128 * .pi / 180.0]
        let aLon: [Float64] = [-74.0060 * .pi / 180.0]
        let bLat: [Float64] = [51.5074 * .pi / 180.0]
        let bLon: [Float64] = [-0.1278 * .pi / 180.0]
        let result = try XCTUnwrap(vincenty(aLat: aLat, aLon: aLon, bLat: bLat, bLon: bLon))
        XCTAssertEqual(result[0], 5_570_000, accuracy: 20000)
    }

    func testVincentyFreeFloat32() throws {
        let aLat: [Float32] = [40.7128 * .pi / 180.0]
        let aLon: [Float32] = [-74.0060 * .pi / 180.0]
        let bLat: [Float32] = [51.5074 * .pi / 180.0]
        let bLon: [Float32] = [-0.1278 * .pi / 180.0]
        let result: [Float32]? = vincenty(aLat: aLat, aLon: aLon, bLat: bLat, bLon: bLon)
        let unwrapped = try XCTUnwrap(result)
        XCTAssertEqual(unwrapped[0], 5_570_000, accuracy: 50000)
    }

    // MARK: - New Low-Precision Types

    func testBFloat16Roundtrip() throws {
        let x = BFloat16(float: 1.5)
        XCTAssertEqual(x.float, 1.5, accuracy: 0.02)
    }

    func testE4M3Dot() throws {
        let a: [E4M3] = [E4M3(float: 1), E4M3(float: 2), E4M3(float: 3)]
        let b: [E4M3] = [E4M3(float: 4), E4M3(float: 5), E4M3(float: 6)]
        let result = try XCTUnwrap(a.dot(b))
        XCTAssertEqual(result, 32.0, accuracy: 1.5)
    }

    // MARK: - Packed Matrix APIs

    func testDotsPackedFloat32() throws {
        let a: [Float32] = [
            1, 2, 3,
            4, 5, 6,
        ]  // 2x3
        let b: [Float32] = [
            7, 8, 9,
            1, 0, 1,
        ]  // 2x3
        var c = Array(repeating: Float64(0), count: 4)  // 2x2, Float32 dots accumulate into Float64

        try a.withUnsafeBufferPointer { aPtr in
            try b.withUnsafeBufferPointer { bPtr in
                try c.withUnsafeMutableBufferPointer { cPtr in
                    let aMatrix = MatrixView(baseAddress: aPtr.baseAddress!, rows: 2, cols: 3)
                    let bMatrix = MatrixView(baseAddress: bPtr.baseAddress!, rows: 2, cols: 3)
                    var cMatrix = MatrixSpan(baseAddress: cPtr.baseAddress!, rows: 2, cols: 2)
                    let packed = try PackedMatrix<Float32>(packing: bMatrix)
                    try dots_packed(aMatrix, packed, &cMatrix)
                }
            }
        }

        XCTAssertEqual(c[0], 50, accuracy: 0.01)  // [1,2,3] · [7,8,9]
        XCTAssertEqual(c[1], 4, accuracy: 0.01)  // [1,2,3] · [1,0,1]
        XCTAssertEqual(c[2], 122, accuracy: 0.01)  // [4,5,6] · [7,8,9]
        XCTAssertEqual(c[3], 10, accuracy: 0.01)  // [4,5,6] · [1,0,1]
    }

    func testAngularsPackedFloat32() throws {
        let a: [Float32] = [
            1, 0, 0,
            0, 1, 0,
        ]  // 2x3
        let b: [Float32] = [
            1, 0, 0,
            0, 1, 0,
        ]  // 2x3
        var out = Array(repeating: Float64(0), count: 4)  // 2x2, Float32 spatial kernels output Float64

        try a.withUnsafeBufferPointer { aPtr in
            try b.withUnsafeBufferPointer { bPtr in
                try out.withUnsafeMutableBufferPointer { outPtr in
                    let aMatrix = MatrixView(baseAddress: aPtr.baseAddress!, rows: 2, cols: 3)
                    let bMatrix = MatrixView(baseAddress: bPtr.baseAddress!, rows: 2, cols: 3)
                    var outMatrix = MatrixSpan(baseAddress: outPtr.baseAddress!, rows: 2, cols: 2)
                    let packed = try PackedMatrix<Float32>(packing: bMatrix)
                    try angulars_packed(aMatrix, packed, &outMatrix)
                }
            }
        }

        XCTAssertEqual(out[0], 0, accuracy: 0.01)
        XCTAssertEqual(out[1], 1, accuracy: 0.05)
        XCTAssertEqual(out[2], 1, accuracy: 0.05)
        XCTAssertEqual(out[3], 0, accuracy: 0.01)
    }

    // MARK: - Owning Tensor Tests

    func testTensorFromArray() throws {
        let t = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6], rows: 2, cols: 3)
        XCTAssertEqual(t.rows, 2)
        XCTAssertEqual(t.cols, 3)
        XCTAssertEqual(t.count, 6)
        XCTAssertEqual(t[0, 0], 1)
        XCTAssertEqual(t[0, 2], 3)
        XCTAssertEqual(t[1, 0], 4)
        XCTAssertEqual(t[1, 2], 6)
    }

    func testTensorZeros() throws {
        let t = try Tensor<Float32>.zeros(rows: 3, cols: 4)
        XCTAssertEqual(t.rows, 3)
        XCTAssertEqual(t.cols, 4)
        for r in 0..<3 {
            for c in 0..<4 {
                XCTAssertEqual(t[r, c], 0)
            }
        }
    }

    func testTensorResizeWithinCapacity() throws {
        let t = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6, 7, 8], rows: 2, cols: 4)  // capacity 8
        XCTAssertEqual(t.capacity, 8)
        XCTAssertTrue(t.tryResize(rows: 2, cols: 2))
        XCTAssertEqual(t.rows, 2)
        XCTAssertEqual(t.cols, 2)
        XCTAssertEqual(t.count, 4)
        XCTAssertEqual(t.capacity, 8)  // capacity is unchanged by resize
        XCTAssertEqual(t[0, 0], 1)  // storage never moves within capacity
        XCTAssertEqual(t[1, 1], 4)  // flat index 1*2 + 1 = 3
        XCTAssertFalse(t.tryResize(rows: 3, cols: 4))  // 12 > 8, left unchanged
        XCTAssertEqual(t.rows, 2)
        XCTAssertEqual(t.cols, 2)
    }

    func testTensorReserveAndClear() throws {
        let t = try Tensor<Float32>.fromArray([1, 2, 3, 4], rows: 1, cols: 4)
        t.reserve(64)
        XCTAssertGreaterThanOrEqual(t.capacity, 64)
        XCTAssertEqual(t.count, 4)  // reserve grows capacity, it does not reshape
        XCTAssertEqual(t[0, 0], 1)  // contents preserved across the reallocation
        XCTAssertEqual(t[0, 3], 4)
        XCTAssertTrue(t.tryResize(rows: 8, cols: 8))  // the grown capacity admits a larger shape
        XCTAssertEqual(t.count, 64)
        t.clear()
        XCTAssertEqual(t.count, 0)
        XCTAssertGreaterThanOrEqual(t.capacity, 64)  // capacity retained
    }

    func testTensorDotsPacked_Float32() throws {
        let a = try Tensor<Float32>.fromArray([1, 2, 3, 4, 5, 6], rows: 2, cols: 3)
        let b = try Tensor<Float32>.fromArray([7, 8, 9, 1, 0, 1], rows: 2, cols: 3)
        let packed = try b.packForDots()
        let result = try a.dotsPacked(packed)
        XCTAssertEqual(result.rows, 2)
        XCTAssertEqual(result.cols, 2)
        XCTAssertEqual(result[0, 0], 50, accuracy: 0.01)
        XCTAssertEqual(result[0, 1], 4, accuracy: 0.01)
        XCTAssertEqual(result[1, 0], 122, accuracy: 0.01)
        XCTAssertEqual(result[1, 1], 10, accuracy: 0.01)
    }

    func testTensorAngularsPacked_Float32() throws {
        let a = try Tensor<Float32>.fromArray([1, 0, 0, 0, 1, 0], rows: 2, cols: 3)
        let b = try Tensor<Float32>.fromArray([1, 0, 0, 0, 1, 0], rows: 2, cols: 3)
        let packed = try b.packForDots()
        let result = try a.angularsPacked(packed)
        XCTAssertEqual(result[0, 0], 0, accuracy: 0.01)
        XCTAssertEqual(result[0, 1], 1, accuracy: 0.05)
        XCTAssertEqual(result[1, 0], 1, accuracy: 0.05)
        XCTAssertEqual(result[1, 1], 0, accuracy: 0.01)
    }

    // MARK: - U1x8 and Binary Metric Tests

    func testU1x8Roundtrip() throws {
        let x = U1x8(0b10110011)
        XCTAssertEqual(x.bitPattern, 0b10110011)
        XCTAssertEqual(x.popcount, 5)
    }

    func testHammingU1x8Scalar() throws {
        // All bits set vs none set: hamming distance = number of bits
        let a: [U1x8] = [U1x8(0xFF), U1x8(0xFF)]
        let b: [U1x8] = [U1x8(0x00), U1x8(0x00)]
        let result = try XCTUnwrap(a.hamming(b))
        XCTAssertEqual(result, 16)  // 16 bits differ
    }

    func testJaccardU1x8Scalar() throws {
        let a: [U1x8] = [U1x8(0xFF)]
        let b: [U1x8] = [U1x8(0xFF)]
        let result = try XCTUnwrap(a.jaccard(b))
        XCTAssertTrue(result.isFinite)
        XCTAssertEqual(result, 0, accuracy: 0.01)  // identical sets => 0 distance
    }

    func testHammingsPackedU1x8() throws {
        // 2 vectors of 16 bits each (2 U1x8 elements per row)
        let a = try Tensor<U1x8>.fromArray(
            [
                U1x8(0xFF), U1x8(0x00),
                U1x8(0x00), U1x8(0xFF),
            ], rows: 2, cols: 2)
        let b = try Tensor<U1x8>.fromArray(
            [
                U1x8(0xFF), U1x8(0xFF),
            ], rows: 1, cols: 2)
        let packed = try b.packForDots()
        let result = try a.hammingsPacked(packed)
        XCTAssertEqual(result.rows, 2)
        XCTAssertEqual(result.cols, 1)
        // a[0] = 0xFF00 vs b[0] = 0xFFFF: 8 bits differ
        XCTAssertEqual(result[0, 0], 8)
        // a[1] = 0x00FF vs b[0] = 0xFFFF: 8 bits differ
        XCTAssertEqual(result[1, 0], 8)
    }

    func testHammingsSymmetricU1x8() throws {
        // 3 vectors of 8 bits each (1 U1x8 element per row)
        let t = try Tensor<U1x8>.fromArray(
            [
                U1x8(0xFF),
                U1x8(0x00),
                U1x8(0x0F),
            ], rows: 3, cols: 1)
        let result = try t.hammingsSymmetric()
        XCTAssertEqual(result.rows, 3)
        XCTAssertEqual(result.cols, 3)
        // Diagonal should be 0
        XCTAssertEqual(result[0, 0], 0)
        XCTAssertEqual(result[1, 1], 0)
        XCTAssertEqual(result[2, 2], 0)
        // 0xFF vs 0x00 = 8 bits
        XCTAssertEqual(result[0, 1], 8)
    }

    func testJaccardsPackedU1x8() throws {
        let a = try Tensor<U1x8>.fromArray(
            [
                U1x8(0xFF), U1x8(0x00),
            ], rows: 1, cols: 2)
        let b = try Tensor<U1x8>.fromArray(
            [
                U1x8(0xFF), U1x8(0x00),
            ], rows: 1, cols: 2)
        let packed = try b.packForDots()
        let result = try a.jaccardsPacked(packed)
        XCTAssertEqual(result.rows, 1)
        XCTAssertEqual(result.cols, 1)
        XCTAssertTrue(result[0, 0].isFinite)
        XCTAssertEqual(result[0, 0], 0, accuracy: 0.01)  // identical
    }

    func testJaccardsSymmetricU1x8() throws {
        let t = try Tensor<U1x8>.fromArray(
            [
                U1x8(0xFF),
                U1x8(0xFF),
                U1x8(0x00),
            ], rows: 3, cols: 1)
        let result = try t.jaccardsSymmetric()
        XCTAssertEqual(result.rows, 3)
        XCTAssertEqual(result.cols, 3)
        // Identical vectors
        XCTAssertEqual(result[0, 1], 0, accuracy: 0.01)
    }

    // MARK: - MaxSim Tests

    func testMaxSimPack_Float32() throws {
        // 2 vectors of dimension 4
        let q = try Tensor<Float32>.fromArray([1, 0, 0, 0, 0, 1, 0, 0], rows: 2, cols: 4)
        let d = try Tensor<Float32>.fromArray([1, 0, 0, 0, 0, 1, 0, 0], rows: 2, cols: 4)
        let qPacked = try q.maxSimPack()
        let dPacked = try d.maxSimPack()
        let score = qPacked.score(dPacked)
        XCTAssertTrue(score.isFinite)
    }

    func testMaxSimPack_BFloat16() throws {
        let q = try Tensor<BFloat16>.fromArray(
            [
                BFloat16(float: 1), BFloat16(float: 0), BFloat16(float: 0), BFloat16(float: 0),
                BFloat16(float: 0), BFloat16(float: 1), BFloat16(float: 0), BFloat16(float: 0),
            ], rows: 2, cols: 4)
        let d = try Tensor<BFloat16>.fromArray(
            [
                BFloat16(float: 1), BFloat16(float: 0), BFloat16(float: 0), BFloat16(float: 0),
                BFloat16(float: 0), BFloat16(float: 1), BFloat16(float: 0), BFloat16(float: 0),
            ], rows: 2, cols: 4)
        let qPacked = try q.maxSimPack()
        let dPacked = try d.maxSimPack()
        let score = qPacked.score(dPacked)
        XCTAssertTrue(score.isFinite)
    }
}
