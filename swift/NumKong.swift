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
        return nk_haversine_f64_best(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            nk_size_t(n),
            result.baseAddress!,
            Capabilities.cpus.native,
            nil
        ) == nk_success_k
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
        return nk_vincenty_f64_best(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            nk_size_t(n),
            result.baseAddress!,
            Capabilities.cpus.native,
            nil
        ) == nk_success_k
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
        return nk_haversine_f32_best(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            nk_size_t(n),
            result.baseAddress!,
            Capabilities.cpus.native,
            nil
        ) == nk_success_k
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
        return nk_vincenty_f32_best(
            aLat.baseAddress!,
            aLon.baseAddress!,
            bLat.baseAddress!,
            bLon.baseAddress!,
            nk_size_t(n),
            result.baseAddress!,
            Capabilities.cpus.native,
            nil
        ) == nk_success_k
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
            nk_haversine_f64_best(a, b, c, d, nk_size_t(n), r, Capabilities.cpus.native, nil)
        }
    }

    @inlinable @inline(__always)
    public static func vincenty<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Float64]?
    where A.Element == Float64, B.Element == Float64, C.Element == Float64, D.Element == Float64 {
        _nkWithGeoQuad(aLat, aLon, bLat, bLon) { a, b, c, d, r, n in
            nk_vincenty_f64_best(a, b, c, d, nk_size_t(n), r, Capabilities.cpus.native, nil)
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
            nk_haversine_f32_best(a, b, c, d, nk_size_t(n), r, Capabilities.cpus.native, nil)
        }
    }

    @inlinable @inline(__always)
    public static func vincenty<A: Sequence, B: Sequence, C: Sequence, D: Sequence>(
        aLat: A, aLon: B, bLat: C, bLon: D
    ) -> [Float32]?
    where A.Element == Float32, B.Element == Float32, C.Element == Float32, D.Element == Float32 {
        _nkWithGeoQuad(aLat, aLon, bLat, bLon) { a, b, c, d, r, n in
            nk_vincenty_f32_best(a, b, c, d, nk_size_t(n), r, Capabilities.cpus.native, nil)
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

// MARK: - Capabilities and Devices

/// A set of capabilities of a CPU or a GPU, as a ``Device`` reports them.
public struct Capabilities: OptionSet, Sendable, CustomStringConvertible {
    public let rawValue: UInt64
    public init(rawValue: UInt64) { self.rawValue = rawValue }

    /// The C API's mask: `UInt` on Linux arm64, where `nk_capability_t` is `unsigned long`.
    @usableFromInline var native: nk_capability_t { nk_capability_t(rawValue) }

    public static let serial = Capabilities(rawValue: 1 << 0)
    public static let neon = Capabilities(rawValue: 1 << 1)
    public static let neonHalf = Capabilities(rawValue: 1 << 2)
    public static let neonBFDot = Capabilities(rawValue: 1 << 3)
    public static let neonFHM = Capabilities(rawValue: 1 << 4)
    public static let neonSDot = Capabilities(rawValue: 1 << 5)
    public static let neonFP8 = Capabilities(rawValue: 1 << 6)
    public static let sve = Capabilities(rawValue: 1 << 7)
    public static let sveHalf = Capabilities(rawValue: 1 << 8)
    public static let sveSDot = Capabilities(rawValue: 1 << 9)
    public static let sveBFDot = Capabilities(rawValue: 1 << 10)
    public static let sve2 = Capabilities(rawValue: 1 << 11)
    public static let sme = Capabilities(rawValue: 1 << 12)
    public static let smeF64 = Capabilities(rawValue: 1 << 13)
    public static let smeBI32 = Capabilities(rawValue: 1 << 14)
    public static let haswell = Capabilities(rawValue: 1 << 15)
    public static let alder = Capabilities(rawValue: 1 << 16)
    public static let sierra = Capabilities(rawValue: 1 << 17)
    public static let skylake = Capabilities(rawValue: 1 << 18)
    public static let iceLake = Capabilities(rawValue: 1 << 19)
    public static let genoa = Capabilities(rawValue: 1 << 20)
    public static let turin = Capabilities(rawValue: 1 << 21)
    public static let sapphire = Capabilities(rawValue: 1 << 22)
    public static let diamond = Capabilities(rawValue: 1 << 23)
    public static let sapphireAMX = Capabilities(rawValue: 1 << 24)
    public static let graniteAMX = Capabilities(rawValue: 1 << 25)
    public static let diamondAMX = Capabilities(rawValue: 1 << 26)
    public static let rvv = Capabilities(rawValue: 1 << 27)
    public static let rvvBF16 = Capabilities(rawValue: 1 << 28)
    public static let rvvHalf = Capabilities(rawValue: 1 << 29)
    public static let rvvBB = Capabilities(rawValue: 1 << 30)
    public static let v128 = Capabilities(rawValue: 1 << 31)
    public static let v128Relaxed = Capabilities(rawValue: 1 << 32)
    public static let powerVSX = Capabilities(rawValue: 1 << 33)
    public static let loongsonASX = Capabilities(rawValue: 1 << 34)

    public static let cuda = Capabilities(rawValue: 1 << 48)
    public static let ampere = Capabilities(rawValue: 1 << 49)
    public static let ada = Capabilities(rawValue: 1 << 50)
    public static let hopper = Capabilities(rawValue: 1 << 51)
    public static let blackwell = Capabilities(rawValue: 1 << 52)
    public static let blackwellRTX = Capabilities(rawValue: 1 << 53)
    public static let rocm = Capabilities(rawValue: 1 << 56)
    public static let cdna4 = Capabilities(rawValue: 1 << 57)
    public static let cdna5 = Capabilities(rawValue: 1 << 58)
    public static let metal = Capabilities(rawValue: 1 << 60)
    public static let apple9 = Capabilities(rawValue: 1 << 61)
    public static let apple10 = Capabilities(rawValue: 1 << 62)

    /// Every CPU capability, the bits below the first GPU vendor's, which every CPU kernel call
    /// passes, as the library clamps it to ``Device/capabilitiesEnabled``.
    public static let cpus = Capabilities(rawValue: (1 << 48) - 1)
    /// Every GPU capability.
    public static let gpus: Capabilities = [
        .cuda, .ampere, .ada, .hopper, .blackwell, .blackwellRTX, .rocm, .cdna4, .cdna5, .metal, .apple9, .apple10,
    ]
    /// Every capability.
    public static let any = Capabilities(rawValue: .max)

    /// The capability names, comma-separated, like "serial,haswell".
    public var description: String {
        String(unsafeUninitializedCapacity: Int(NUMKONG_CAPABILITIES_NAME_CAPACITY)) { names in
            names.withMemoryRebound(to: CChar.self) {
                Int(nk_capabilities_name(native, $0.baseAddress, nk_size_t($0.count)))
            }
        }
    }
}

/// Which runtime a device belongs to, as the `nk_<kind>_*` C functions name it.
public enum DeviceKind: Sendable {
    case cpu, cuda, rocm, metal
}

/// Why a call failed, one case per failing `nk_status_t`, which `nk_status_name` spells.
public enum Status: Int32, Sendable {
    case badAllocation = -10
    case unexpectedDimensions = -15
    case missingGpu = -16
    case deviceCodeMismatch = -17
    case deviceMemoryMismatch = -18
    case missingKernel = -19
    case misaligned = -20
    case packMismatch = -21
    case missingLibrary = -22

    /// The C spelling, like "unexpected_dimensions".
    public var name: String { String(cString: nk_status_name(nk_status_t(rawValue: rawValue))) }
}

/// One failed call: a status every caller can branch on, and the detail the status cannot
/// carry.
public struct Error: Swift.Error, CustomStringConvertible, Equatable, Sendable {
    /// The failure identity, shared with the C library.
    public let status: Status
    /// This binding's detail when it rejected the call itself, else empty.
    public let message: String

    public var description: String {
        message.isEmpty ? "numkong: \(status.name)" : "numkong: \(status.name): \(message)"
    }
}

/// Builds the failure this binding reports for a fault it caught before the boundary.
@usableFromInline
func fail(_ status: Status, _ message: String) -> Error { Error(status: status, message: message) }

/// One device NumKong can run kernels on: the host CPU, or a GPU by its runtime's own ordinal, the
/// one `cudaSetDevice` or `hipSetDevice` takes, or the position in Metal's device list.
///
/// Prefer ``capabilitiesEnabled`` unless you specifically mean one of the raw axes:
/// ``capabilitiesDetected`` describes the device and says nothing about whether a kernel was
/// compiled into this binary, so selecting on it alone claims support for code that may not exist.
public struct Device: Sendable, Equatable {
    public let kind: DeviceKind
    public let ordinal: Int

    /// The host CPU, which every build has.
    public static let cpu = Device(kind: .cpu, unchecked: 0)

    private init(kind: DeviceKind, unchecked ordinal: Int) {
        self.kind = kind
        self.ordinal = ordinal
    }

    /// Device `ordinal` of `kind`.
    /// - Throws: ``Error`` past the last device of `kind`.
    public init(kind: DeviceKind, ordinal: Int) throws {
        guard ordinal >= 0, ordinal < (try Device.count(kind)) else {
            throw fail(.missingGpu, "no \(kind) device \(ordinal)")
        }
        self.init(kind: kind, unchecked: ordinal)
    }

    /// How many devices of `kind` the process sees: one CPU, or the GPUs its runtime counts.
    /// - Throws: ``Error`` without a GPU of `kind`.
    public static func count(_ kind: DeviceKind) throws -> Int {
        var count: nk_size_t = 1
        switch kind {
        case .cpu: break
        case .cuda: try _nkCheck(nk_cuda_count_devices(&count))
        case .rocm: try _nkCheck(nk_rocm_count_devices(&count))
        case .metal: try _nkCheck(nk_metal_count_devices(&count))
        }
        return Int(count)
    }

    /// What this device runs, whether or not this binary holds kernels for it.
    public var capabilitiesDetected: Capabilities {
        get throws {
            var mask: nk_capability_t = 0
            let device = nk_size_t(ordinal)
            switch kind {
            case .cpu: try _nkCheck(nk_cpu_capabilities_detected(&mask))
            case .cuda: try _nkCheck(nk_cuda_capabilities_detected(device, &mask))
            case .rocm: try _nkCheck(nk_rocm_capabilities_detected(device, &mask))
            case .metal: try _nkCheck(nk_metal_capabilities_detected(device, &mask))
            }
            return Capabilities(rawValue: UInt64(mask))
        }
    }

    /// What this binary holds kernels for on devices of this kind, whether or not this one runs.
    public var capabilitiesCompiled: Capabilities {
        var mask: nk_capability_t = 0
        switch kind {
        case .cpu: _ = nk_cpu_capabilities_compiled(&mask)
        case .cuda: _ = nk_cuda_capabilities_compiled(&mask)
        case .rocm: _ = nk_rocm_capabilities_compiled(&mask)
        case .metal: _ = nk_metal_capabilities_compiled(&mask)
        }
        return Capabilities(rawValue: UInt64(mask))
    }

    /// What this device's kernel calls run within: ``capabilitiesDetected`` and
    /// ``capabilitiesCompiled`` at once. On the CPU the library settles it as it loads, and it always
    /// contains ``Capabilities/serial``.
    public var capabilitiesEnabled: Capabilities {
        get throws {
            var mask: nk_capability_t = 0
            let device = nk_size_t(ordinal)
            switch kind {
            case .cpu: try _nkCheck(nk_cpu_capabilities_enabled(&mask))
            case .cuda: try _nkCheck(nk_cuda_capabilities_enabled(device, &mask))
            case .rocm: try _nkCheck(nk_rocm_capabilities_enabled(device, &mask))
            case .metal: try _nkCheck(nk_metal_capabilities_enabled(device, &mask))
            }
            return Capabilities(rawValue: UInt64(mask))
        }
    }

    /// Configures the current thread for `capabilities`, usually ``capabilitiesEnabled``, e.g. AMX
    /// tile state on x86. Must be called once per thread before using AMX operations.
    /// - Throws: ``Error`` on a GPU, which has no thread state to configure.
    public func configureThread(_ capabilities: Capabilities) throws {
        guard kind == .cpu else { throw fail(.missingKernel, "GPUs have no thread state to configure") }
        try _nkCheck(nk_cpu_configure_thread(capabilities.native))
    }

}

/// Throws the ``Error`` a C call reports, if any.
@usableFromInline
func _nkCheck(_ status: nk_status_t) throws(Error) {
    guard status != nk_success_k else { return }
    throw Error(status: Status(rawValue: status.rawValue) ?? .deviceCodeMismatch, message: "")
}
