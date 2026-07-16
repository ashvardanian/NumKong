//  NumKong.swift
//  NumKong
//
//  Created by Ash Vardanian on March 14, 2026.
//

import CNumKong

// MARK: - Geospatial Protocols

/// A type that can compute SIMD-accelerated Haversine (great-circle) distances.
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

/// A type that can compute SIMD-accelerated Vincenty (ellipsoidal) geodesic distances.
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
            UInt64(n),
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
            UInt64(n),
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
            UInt64(n),
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
            UInt64(n),
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
            nk_haversine_f64(a, b, c, d, UInt64(n), r)
        }
    }

    @inlinable @inline(__always)
    public static func vincenty<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Float64]?
    where A.Element == Float64, B.Element == Float64, C.Element == Float64, D.Element == Float64 {
        _nkWithGeoQuad(aLat, aLon, bLat, bLon) { a, b, c, d, r, n in
            nk_vincenty_f64(a, b, c, d, UInt64(n), r)
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
            nk_haversine_f32(a, b, c, d, UInt64(n), r)
        }
    }

    @inlinable @inline(__always)
    public static func vincenty<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Float32]?
    where A.Element == Float32, B.Element == Float32, C.Element == Float32, D.Element == Float32 {
        _nkWithGeoQuad(aLat, aLon, bLat, bLon) { a, b, c, d, r, n in
            nk_vincenty_f32(a, b, c, d, UInt64(n), r)
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

/// Bitmasks of SIMD instruction sets, along two independent axes and the sets derived from them.
///
/// Prefer ``available`` unless you specifically mean one of the raw axes: ``detected`` describes
/// the CPU and says nothing about whether a kernel was compiled into this binary, so selecting on
/// it alone claims hardware support for code that may not exist here.
public enum Capabilities {
    /// What this CPU supports, from CPUID or HWCAP.
    public static var detected: UInt64 { nk_capabilities_detected() }

    /// What this binary contains, as decided by the ISA probes at build time.
    public static var compiled: UInt64 { nk_capabilities_compiled() }

    /// What can actually execute here: ``detected`` intersected with ``compiled``.
    public static var available: UInt64 { nk_capabilities_available() }

    /// What dispatch is currently restricted to, a subset of ``available``.
    public static var enabled: UInt64 { nk_capabilities_enabled() }

    /// Whether `capability` can actually execute here, i.e. whether it is in ``available``.
    /// False both when this CPU lacks the feature and when its kernels were not compiled in.
    public static func has(_ capability: UInt64) -> Bool { available & capability != 0 }

    /// Restricts dispatch to `capabilities`, clamped to ``available``; serial is always kept.
    public static func restrict(_ capabilities: UInt64) { nk_capabilities_restrict(capabilities) }

    /// Adds `capabilities` to ``enabled``. Anything not in ``available`` is ignored.
    public static func enable(_ capabilities: UInt64) { nk_capabilities_enable(capabilities) }

    /// Removes `capabilities` from ``enabled``. The serial fallback cannot be removed.
    public static func disable(_ capabilities: UInt64) { nk_capabilities_disable(capabilities) }

    /// Configures the current thread for the capabilities that can run here, e.g. AMX tile
    /// state on x86. Must be called once per thread before using AMX operations.
    /// - Returns: `true` on success.
    @discardableResult
    public static func configureThread() -> Bool { nk_configure_thread(available) != 0 }

    public static let serial: UInt64 = 1 << 0
    public static let neon: UInt64 = 1 << 1
    public static let haswell: UInt64 = 1 << 2
    public static let skylake: UInt64 = 1 << 3
    public static let neonHalf: UInt64 = 1 << 4
    public static let neonSDot: UInt64 = 1 << 5
    public static let neonFhm: UInt64 = 1 << 6
    public static let icelake: UInt64 = 1 << 7
    public static let genoa: UInt64 = 1 << 8
    public static let neonBfDot: UInt64 = 1 << 9
    public static let sve: UInt64 = 1 << 10
    public static let sveHalf: UInt64 = 1 << 11
    public static let sveSDot: UInt64 = 1 << 12
    public static let alder: UInt64 = 1 << 13
    public static let sveBfDot: UInt64 = 1 << 14
    public static let sve2: UInt64 = 1 << 15
    public static let v128Relaxed: UInt64 = 1 << 16
    public static let sapphire: UInt64 = 1 << 17
    public static let sapphireAmx: UInt64 = 1 << 18
    public static let rvv: UInt64 = 1 << 19
    public static let rvvHalf: UInt64 = 1 << 20
    public static let rvvBf16: UInt64 = 1 << 21
    public static let graniteAmx: UInt64 = 1 << 22
    public static let turin: UInt64 = 1 << 23
    public static let sme: UInt64 = 1 << 24
    public static let sme2: UInt64 = 1 << 25
    public static let smeF64: UInt64 = 1 << 26
    public static let smeFa64: UInt64 = 1 << 27
    public static let sve2p1: UInt64 = 1 << 28
    public static let sme2p1: UInt64 = 1 << 29
    public static let smeHalf: UInt64 = 1 << 30
    public static let smeBf16: UInt64 = 1 << 31
    public static let smeLut2: UInt64 = 1 << 32
    public static let rvvBB: UInt64 = 1 << 33
    public static let sierra: UInt64 = 1 << 34
    public static let smeBi32: UInt64 = 1 << 35
    public static let loongsonAsx: UInt64 = 1 << 36
    public static let powerVsx: UInt64 = 1 << 37
    public static let diamond: UInt64 = 1 << 38
    public static let neonFp8: UInt64 = 1 << 39
    public static let diamondAmx: UInt64 = 1 << 40
}
