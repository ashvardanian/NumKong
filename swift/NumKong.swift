//
//  swift/NumKong.swift
//  Geospatial distance protocols, capability detection, and thread configuration.
//
//  - Author: Ash Vardanian
//  - Date: March 14, 2026
//

import CNumKong

// MARK: - Geospatial Protocols

/// A type that can compute SIMD-accelerated great-circle Haversine distances.
public protocol NumKongHaversine: BinaryFloatingPoint {
    static func haversine(
        aLat: UnsafeBufferPointer<Self>,
        aLon: UnsafeBufferPointer<Self>,
        bLat: UnsafeBufferPointer<Self>,
        bLon: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>
    ) -> Bool

    static func haversine<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Self]?
    where A.Element == Self, B.Element == Self, C.Element == Self, D.Element == Self
}

/// A type that can compute SIMD-accelerated Vincenty ellipsoidal geodesic distances.
public protocol NumKongVincenty: BinaryFloatingPoint {
    static func vincenty(
        aLat: UnsafeBufferPointer<Self>,
        aLon: UnsafeBufferPointer<Self>,
        bLat: UnsafeBufferPointer<Self>,
        bLon: UnsafeBufferPointer<Self>,
        result: UnsafeMutableBufferPointer<Self>
    ) -> Bool

    static func vincenty<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Self]?
    where A.Element == Self, B.Element == Self, C.Element == Self, D.Element == Self
}

/// Convenience alias for types supporting both Haversine and Vincenty geospatial distances.
public typealias NumKongGeospatial = NumKongHaversine & NumKongVincenty

extension Float64: NumKongHaversine {
    @inlinable @inline(__always)
    public static func haversine(
        aLat: UnsafeBufferPointer<Float64>,
        aLon: UnsafeBufferPointer<Float64>,
        bLat: UnsafeBufferPointer<Float64>,
        bLon: UnsafeBufferPointer<Float64>,
        result: UnsafeMutableBufferPointer<Float64>
    ) -> Bool {
        let n = aLat.count
        guard
            n > 0 && n == aLon.count && n == bLat.count && n == bLon.count
                && n == result.count
        else {
            return false
        }
        nk_haversine_f64(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            nk_size_t(n),
            result.baseAddress!
        )
        return true
    }
}

extension Float64: NumKongVincenty {
    @inlinable @inline(__always)
    public static func vincenty(
        aLat: UnsafeBufferPointer<Float64>,
        aLon: UnsafeBufferPointer<Float64>,
        bLat: UnsafeBufferPointer<Float64>,
        bLon: UnsafeBufferPointer<Float64>,
        result: UnsafeMutableBufferPointer<Float64>
    ) -> Bool {
        let n = aLat.count
        guard
            n > 0 && n == aLon.count && n == bLat.count && n == bLon.count
                && n == result.count
        else {
            return false
        }
        nk_vincenty_f64(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            nk_size_t(n),
            result.baseAddress!
        )
        return true
    }
}

extension Float32: NumKongHaversine {
    @inlinable @inline(__always)
    public static func haversine(
        aLat: UnsafeBufferPointer<Float32>,
        aLon: UnsafeBufferPointer<Float32>,
        bLat: UnsafeBufferPointer<Float32>,
        bLon: UnsafeBufferPointer<Float32>,
        result: UnsafeMutableBufferPointer<Float32>
    ) -> Bool {
        let n = aLat.count
        guard
            n > 0 && n == aLon.count && n == bLat.count && n == bLon.count
                && n == result.count
        else {
            return false
        }
        nk_haversine_f32(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            nk_size_t(n),
            result.baseAddress!
        )
        return true
    }
}

extension Float32: NumKongVincenty {
    @inlinable @inline(__always)
    public static func vincenty(
        aLat: UnsafeBufferPointer<Float32>,
        aLon: UnsafeBufferPointer<Float32>,
        bLat: UnsafeBufferPointer<Float32>,
        bLon: UnsafeBufferPointer<Float32>,
        result: UnsafeMutableBufferPointer<Float32>
    ) -> Bool {
        let n = aLat.count
        guard
            n > 0 && n == aLon.count && n == bLat.count && n == bLon.count
                && n == result.count
        else {
            return false
        }
        nk_vincenty_f32(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            nk_size_t(n),
            result.baseAddress!
        )
        return true
    }
}

// MARK: - Geospatial Sequence Extensions

extension Float64 {
    @inlinable @inline(__always)
    public static func haversine<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Float64]?
    where A.Element == Float64, B.Element == Float64, C.Element == Float64, D.Element == Float64 {
        _nkWithGeoQuad(aLat, aLon, bLat, bLon) { a, b, c, d, r, n in
            nk_haversine_f64(a, b, c, d, nk_size_t(n), r)
        }
    }

    @inlinable @inline(__always)
    public static func vincenty<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Float64]?
    where A.Element == Float64, B.Element == Float64, C.Element == Float64, D.Element == Float64 {
        _nkWithGeoQuad(aLat, aLon, bLat, bLon) { a, b, c, d, r, n in
            nk_vincenty_f64(a, b, c, d, nk_size_t(n), r)
        }
    }
}

extension Float32 {
    @inlinable @inline(__always)
    public static func haversine<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Float32]?
    where A.Element == Float32, B.Element == Float32, C.Element == Float32, D.Element == Float32 {
        _nkWithGeoQuad(aLat, aLon, bLat, bLon) { a, b, c, d, r, n in
            nk_haversine_f32(a, b, c, d, nk_size_t(n), r)
        }
    }

    @inlinable @inline(__always)
    public static func vincenty<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Float32]?
    where A.Element == Float32, B.Element == Float32, C.Element == Float32, D.Element == Float32 {
        _nkWithGeoQuad(aLat, aLon, bLat, bLon) { a, b, c, d, r, n in
            nk_vincenty_f32(a, b, c, d, nk_size_t(n), r)
        }
    }
}

// MARK: - Geospatial Functions

@inlinable @inline(__always)
public func haversine<A: Sequence>(
    aLat: A, aLon: A, bLat: A, bLon: A
) -> [Float64]?
where A.Element == Float64 {
    Float64.haversine(aLat: aLat, aLon: aLon, bLat: bLat, bLon: bLon)
}

@inlinable @inline(__always)
public func haversine<A: Sequence>(
    aLat: A, aLon: A, bLat: A, bLon: A
) -> [Float32]?
where A.Element == Float32 {
    Float32.haversine(aLat: aLat, aLon: aLon, bLat: bLat, bLon: bLon)
}

@inlinable @inline(__always)
public func vincenty<A: Sequence>(
    aLat: A, aLon: A, bLat: A, bLon: A
) -> [Float64]?
where A.Element == Float64 {
    Float64.vincenty(aLat: aLat, aLon: aLon, bLat: bLat, bLon: bLon)
}

@inlinable @inline(__always)
public func vincenty<A: Sequence>(
    aLat: A, aLon: A, bLat: A, bLon: A
) -> [Float32]?
where A.Element == Float32 {
    Float32.vincenty(aLat: aLat, aLon: aLon, bLat: bLat, bLon: bLon)
}

// MARK: - Capabilities

/// A set of CPU SIMD tiers, reported along two independent axes and the set dispatch uses.
///
/// Prefer ``enabled`` unless you specifically mean one of the raw axes: ``detected`` describes
/// the CPU and says nothing about whether a kernel was compiled into this binary, so selecting on
/// it alone claims hardware support for code that may not exist here.
public struct Capabilities: OptionSet, Sendable, CustomStringConvertible {
    public let rawValue: UInt64
    public init(rawValue: UInt64) { self.rawValue = rawValue }

    public static let serial = Capabilities(rawValue: 1 << 0)
    public static let neon = Capabilities(rawValue: 1 << 1)
    public static let haswell = Capabilities(rawValue: 1 << 2)
    public static let skylake = Capabilities(rawValue: 1 << 3)
    public static let neonHalf = Capabilities(rawValue: 1 << 4)
    public static let neonSDot = Capabilities(rawValue: 1 << 5)
    public static let neonFhm = Capabilities(rawValue: 1 << 6)
    public static let icelake = Capabilities(rawValue: 1 << 7)
    public static let genoa = Capabilities(rawValue: 1 << 8)
    public static let neonBfDot = Capabilities(rawValue: 1 << 9)
    public static let sve = Capabilities(rawValue: 1 << 10)
    public static let sveHalf = Capabilities(rawValue: 1 << 11)
    public static let sveSDot = Capabilities(rawValue: 1 << 12)
    public static let alder = Capabilities(rawValue: 1 << 13)
    public static let sveBfDot = Capabilities(rawValue: 1 << 14)
    public static let sve2 = Capabilities(rawValue: 1 << 15)
    public static let v128Relaxed = Capabilities(rawValue: 1 << 16)
    public static let sapphire = Capabilities(rawValue: 1 << 17)
    public static let sapphireAmx = Capabilities(rawValue: 1 << 18)
    public static let rvv = Capabilities(rawValue: 1 << 19)
    public static let rvvHalf = Capabilities(rawValue: 1 << 20)
    public static let rvvBf16 = Capabilities(rawValue: 1 << 21)
    public static let graniteAmx = Capabilities(rawValue: 1 << 22)
    public static let turin = Capabilities(rawValue: 1 << 23)
    public static let sme = Capabilities(rawValue: 1 << 24)
    public static let sme2 = Capabilities(rawValue: 1 << 25)
    public static let smeF64 = Capabilities(rawValue: 1 << 26)
    public static let smeFa64 = Capabilities(rawValue: 1 << 27)
    public static let sve2p1 = Capabilities(rawValue: 1 << 28)
    public static let sme2p1 = Capabilities(rawValue: 1 << 29)
    public static let smeHalf = Capabilities(rawValue: 1 << 30)
    public static let smeBf16 = Capabilities(rawValue: 1 << 31)
    public static let smeLut2 = Capabilities(rawValue: 1 << 32)
    public static let rvvBB = Capabilities(rawValue: 1 << 33)
    public static let sierra = Capabilities(rawValue: 1 << 34)
    public static let smeBi32 = Capabilities(rawValue: 1 << 35)
    public static let loongsonAsx = Capabilities(rawValue: 1 << 36)
    public static let powerVsx = Capabilities(rawValue: 1 << 37)
    public static let diamond = Capabilities(rawValue: 1 << 38)
    public static let neonFp8 = Capabilities(rawValue: 1 << 39)
    public static let diamondAmx = Capabilities(rawValue: 1 << 40)
    public static let v128 = Capabilities(rawValue: 1 << 41)

    /// What this CPU supports, from CPUID or HWCAP.
    public static var detected: Capabilities { Capabilities(rawValue: UInt64(nk_cpu_capabilities_detected())) }

    /// What this binary contains, as decided by the ISA probes at build time.
    public static var compiled: Capabilities { Capabilities(rawValue: UInt64(nk_cpu_capabilities_compiled())) }

    /// What dispatch uses: ``detected`` and ``compiled`` at once, unless narrowed by
    /// ``enable(_:)``. Always contains ``serial``.
    public static var enabled: Capabilities { Capabilities(rawValue: UInt64(nk_cpu_capabilities_enabled())) }

    /// Makes `wanted` the ``enabled`` set, clamped to ``detected`` and ``compiled`` and keeping
    /// ``serial``.
    /// - Returns: The set that took effect.
    @discardableResult
    public static func enable(_ wanted: Capabilities) -> Capabilities {
        Capabilities(rawValue: UInt64(nk_cpu_capabilities_enable(nk_capability_t(wanted.rawValue))))
    }

    /// Configures the current thread for `capabilities`, usually ``enabled``, e.g. AMX tile state
    /// on x86. Must be called once per thread before using AMX operations.
    /// - Returns: `true` on success.
    @discardableResult
    public static func configureThread(_ capabilities: Capabilities) -> Bool {
        nk_cpu_configure_thread(nk_capability_t(capabilities.rawValue)) != 0
    }

    /// The tier names, comma-separated, like "serial,haswell".
    public var description: String {
        String(unsafeUninitializedCapacity: Int(NUMKONG_CAPABILITIES_NAME_CAPACITY)) { names in
            names.withMemoryRebound(to: CChar.self) {
                Int(nk_name_capabilities(nk_capability_t(rawValue), $0.baseAddress, nk_size_t($0.count)))
            }
        }
    }
}
