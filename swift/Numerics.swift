//  Numerics.swift
//  NumKong
//
//  Created by Ash Vardanian on March 14, 2026.
//

import CNumKong

// MARK: - Shared Helpers: Sequence

/// Logical dimensions held by `values` storage values of `dtype`, mirroring `nk_dimensions_per_value`.
@usableFromInline
internal func valuesToDimensions(_ values: Int, _ dtype: nk_dtype_t) -> Int {
    values * Int(nk_dimensions_per_value(dtype))
}

/// Storage values holding `dimensions` logical dimensions of `dtype`, a multiple of the values per byte.
@usableFromInline
internal func dimensionsToValues(_ dimensions: Int, _ dtype: nk_dtype_t) -> Int {
    dimensions / Int(nk_dimensions_per_value(dtype))
}

@usableFromInline
package func _nkWithDensePair<A: Sequence, B: Sequence, T, R>(
    _ a: A,
    _ b: B,
    _ body: (UnsafePointer<T>, UnsafePointer<T>, Int) -> R
) -> R?
where A.Element == T, B.Element == T {
    let lhs = Array(a)
    let rhs = Array(b)
    guard !lhs.isEmpty && lhs.count == rhs.count else { return nil }
    return lhs.withUnsafeBufferPointer { lhsPtr in
        rhs.withUnsafeBufferPointer { rhsPtr in
            body(lhsPtr.baseAddress!, rhsPtr.baseAddress!, lhs.count)
        }
    }
}

@usableFromInline
package func _nkWithDensePairRebound<A: Sequence, B: Sequence, E, T, R>(
    _ a: A,
    _ b: B,
    to: T.Type,
    _ body: (UnsafePointer<T>, UnsafePointer<T>, Int) -> R
) -> R?
where A.Element == E, B.Element == E {
    _nkWithDensePair(a, b) { lhsPtr, rhsPtr, count in
        let lhsRebound = UnsafeRawPointer(lhsPtr).assumingMemoryBound(to: T.self)
        let rhsRebound = UnsafeRawPointer(rhsPtr).assumingMemoryBound(to: T.self)
        return body(lhsRebound, rhsRebound, count)
    }
}

@usableFromInline
package func _nkF32ToBf16Bits(_ value: Float32) -> UInt16 {
    var src = value
    var dst: nk_bf16_t = 0
    nk_f32_to_bf16(&src, &dst)
    return UInt16(dst)
}

@usableFromInline
package func _nkBf16BitsToF32(_ bits: UInt16) -> Float32 {
    var src = nk_bf16_t(bits)
    var dst: Float32 = 0
    nk_bf16_to_f32(&src, &dst)
    return dst
}

@usableFromInline
package func _nkF32ToE4M3Bits(_ value: Float32) -> UInt8 {
    var src = value
    var dst: nk_e4m3_t = 0
    nk_f32_to_e4m3(&src, &dst)
    return UInt8(dst)
}

@usableFromInline
package func _nkE4M3BitsToF32(_ bits: UInt8) -> Float32 {
    var src = nk_e4m3_t(bits)
    var dst: Float32 = 0
    nk_e4m3_to_f32(&src, &dst)
    return dst
}

@usableFromInline
package func _nkF32ToE5M2Bits(_ value: Float32) -> UInt8 {
    var src = value
    var dst: nk_e5m2_t = 0
    nk_f32_to_e5m2(&src, &dst)
    return UInt8(dst)
}

@usableFromInline
package func _nkE5M2BitsToF32(_ bits: UInt8) -> Float32 {
    var src = nk_e5m2_t(bits)
    var dst: Float32 = 0
    nk_e5m2_to_f32(&src, &dst)
    return dst
}

@usableFromInline
package func _nkF32ToE2M3Bits(_ value: Float32) -> UInt8 {
    var src = value
    var dst: nk_e2m3_t = 0
    nk_f32_to_e2m3(&src, &dst)
    return UInt8(dst)
}

@usableFromInline
package func _nkE2M3BitsToF32(_ bits: UInt8) -> Float32 {
    var src = nk_e2m3_t(bits)
    var dst: Float32 = 0
    nk_e2m3_to_f32(&src, &dst)
    return dst
}

@usableFromInline
package func _nkF32ToE3M2Bits(_ value: Float32) -> UInt8 {
    var src = value
    var dst: nk_e3m2_t = 0
    nk_f32_to_e3m2(&src, &dst)
    return UInt8(dst)
}

@usableFromInline
package func _nkE3M2BitsToF32(_ bits: UInt8) -> Float32 {
    var src = nk_e3m2_t(bits)
    var dst: Float32 = 0
    nk_e3m2_to_f32(&src, &dst)
    return dst
}

@usableFromInline
package func _nkWithGeoQuad<A: Sequence, B: Sequence, C: Sequence, D: Sequence, T>(
    _ a: A, _ b: B, _ c: C, _ d: D,
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafeMutablePointer<T>, Int) ->
        Void
) -> [T]?
where A.Element == T, B.Element == T, C.Element == T, D.Element == T, T: BinaryFloatingPoint {
    let aArr = Array(a)
    let bArr = Array(b)
    let cArr = Array(c)
    let dArr = Array(d)
    let n = aArr.count
    guard n > 0 && n == bArr.count && n == cArr.count && n == dArr.count else { return nil }
    var result = [T](repeating: 0, count: n)
    aArr.withUnsafeBufferPointer { ap in
        bArr.withUnsafeBufferPointer { bp in
            cArr.withUnsafeBufferPointer { cp in
                dArr.withUnsafeBufferPointer { dp in
                    result.withUnsafeMutableBufferPointer { rp in
                        body(ap.baseAddress!, bp.baseAddress!, cp.baseAddress!, dp.baseAddress!, rp.baseAddress!, n)
                    }
                }
            }
        }
    }
    return result
}

// MARK: - Low-Precision Storage Types

/// Brain floating-point: 16-bit storage with 8-bit exponent, used in ML inference.
@frozen
public struct BFloat16: Equatable, Hashable, Sendable {
    public var bitPattern: UInt16

    @inlinable
    public init(bitPattern: UInt16) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToBf16Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkBf16BitsToF32(bitPattern) }
}

/// FP8 format with 5-bit exponent and 2-bit mantissa (FP8 E5M2), used in gradient storage.
@frozen
public struct E5M2: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8

    @inlinable
    public init(bitPattern: UInt8) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToE5M2Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkE5M2BitsToF32(bitPattern) }
}

/// FP8 format with 4-bit exponent and 3-bit mantissa (FP8 E4M3), used in transformer inference.
@frozen
public struct E4M3: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8

    @inlinable
    public init(bitPattern: UInt8) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToE4M3Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkE4M3BitsToF32(bitPattern) }
}

/// MX format with 3-bit exponent and 2-bit mantissa (MX E3M2).
@frozen
public struct E3M2: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8

    @inlinable
    public init(bitPattern: UInt8) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToE3M2Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkE3M2BitsToF32(bitPattern) }
}

/// MX format with 2-bit exponent and 3-bit mantissa (MX E2M3).
@frozen
public struct E2M3: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8

    @inlinable
    public init(bitPattern: UInt8) {
        self.bitPattern = bitPattern
    }

    @inlinable
    public init(float: Float32) {
        self.bitPattern = _nkF32ToE2M3Bits(float)
    }

    @inlinable
    public var float: Float32 { _nkE2M3BitsToF32(bitPattern) }
}

// MARK: - String Descriptions

private func _hexPad(_ value: UInt16, width: Int) -> String {
    let s = String(value, radix: 16, uppercase: false)
    return String(repeating: "0", count: max(0, width - s.count)) + s
}

private func _hexPad(_ value: UInt8, width: Int) -> String {
    let s = String(value, radix: 16, uppercase: false)
    return String(repeating: "0", count: max(0, width - s.count)) + s
}

extension BFloat16: CustomStringConvertible {
    public var description: String { "\(float) [0x\(_hexPad(bitPattern, width: 4))]" }
}

extension E5M2: CustomStringConvertible {
    public var description: String { "\(float) [0x\(_hexPad(bitPattern, width: 2))]" }
}

extension E4M3: CustomStringConvertible {
    public var description: String { "\(float) [0x\(_hexPad(bitPattern, width: 2))]" }
}

extension E3M2: CustomStringConvertible {
    public var description: String { "\(float) [0x\(_hexPad(bitPattern, width: 2))]" }
}

extension E2M3: CustomStringConvertible {
    public var description: String { "\(float) [0x\(_hexPad(bitPattern, width: 2))]" }
}

// MARK: - Float Literals

extension BFloat16: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init(float: Float32(value))
    }
}

extension E5M2: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init(float: Float32(value))
    }
}

extension E4M3: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init(float: Float32(value))
    }
}

extension E3M2: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init(float: Float32(value))
    }
}

extension E2M3: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init(float: Float32(value))
    }
}

// MARK: - Nibble Storage Types

/// Packed pair of 4-bit signed integers, `[high nibble : low nibble]`, each in -8...7.
@frozen
public struct I4x2: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8
    @inlinable public init(_ bits: UInt8) { self.bitPattern = bits }
    @inlinable public init(bitPattern: UInt8) { self.bitPattern = bitPattern }
    /// Packs two lanes, `first` into the high nibble, keeping the low four bits of each.
    @inlinable public init(first: Int8, second: Int8) {
        self.bitPattern = (UInt8(bitPattern: first) << 4) | (UInt8(bitPattern: second) & 0x0F)
    }
    /// The first lane, in the high nibble, sign-extended.
    @inlinable public var first: Int8 { Int8(bitPattern: bitPattern) >> 4 }
    /// The second lane, in the low nibble, sign-extended.
    @inlinable public var second: Int8 { Int8(bitPattern: bitPattern << 4) >> 4 }
}

/// Packed pair of 4-bit unsigned integers, `[high nibble : low nibble]`, each in 0...15.
@frozen
public struct U4x2: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8
    @inlinable public init(_ bits: UInt8) { self.bitPattern = bits }
    @inlinable public init(bitPattern: UInt8) { self.bitPattern = bitPattern }
    /// Packs two lanes, `first` into the high nibble, keeping the low four bits of each.
    @inlinable public init(first: UInt8, second: UInt8) { self.bitPattern = (first << 4) | (second & 0x0F) }
    /// The first lane, in the high nibble.
    @inlinable public var first: UInt8 { bitPattern >> 4 }
    /// The second lane, in the low nibble.
    @inlinable public var second: UInt8 { bitPattern & 0x0F }
}

/// Packed pair of 4-bit E2M1 floats, `[high nibble : low nibble]`, each of 1 sign, 2 exponent and 1 mantissa bits.
@frozen
public struct E2M1x2: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8
    @inlinable public init(_ bits: UInt8) { self.bitPattern = bits }
    @inlinable public init(bitPattern: UInt8) { self.bitPattern = bitPattern }
    /// Packs two lane codes, `first` into the high nibble, keeping the low four bits of each.
    @inlinable public init(first: UInt8, second: UInt8) { self.bitPattern = (first << 4) | (second & 0x0F) }
    /// The first lane code, in the high nibble.
    @inlinable public var first: UInt8 { bitPattern >> 4 }
    /// The second lane code, in the low nibble.
    @inlinable public var second: UInt8 { bitPattern & 0x0F }
}

extension I4x2: CustomStringConvertible {
    public var description: String { "(\(first), \(second)) [0x\(_hexPad(bitPattern, width: 2))]" }
}

extension U4x2: CustomStringConvertible {
    public var description: String { "(\(first), \(second)) [0x\(_hexPad(bitPattern, width: 2))]" }
}

extension E2M1x2: CustomStringConvertible {
    public var description: String { "E2M1x2(0x\(_hexPad(bitPattern, width: 2)))" }
}

// MARK: - Binary Storage Type

/// Packed 8-bit binary vector element for Hamming and Jaccard distance computations.
@frozen
public struct U1x8: Equatable, Hashable, Sendable {
    public var bitPattern: UInt8
    @inlinable public init(_ bits: UInt8) { self.bitPattern = bits }
    @inlinable public init(bitPattern: UInt8) { self.bitPattern = bitPattern }
    @inlinable public var popcount: Int { bitPattern.nonzeroBitCount }
}

extension U1x8: CustomStringConvertible {
    public var description: String {
        let binary = String(bitPattern, radix: 2)
        let padded = String(repeating: "0", count: max(0, 8 - binary.count)) + binary
        return "0b\(padded) [0x\(_hexPad(bitPattern, width: 2))]"
    }
}
