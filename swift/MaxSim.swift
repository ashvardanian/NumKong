//
//  swift/MaxSim.swift
//  Protocol and conformances for late-interaction MaxSim scoring over packed matrices.
//
//  - Author: Ash Vardanian
//  - Date: March 14, 2026
//

import CNumKong

// MARK: - MaxSim Protocol

/// Element type that supports late-interaction MaxSim scoring with packed representations.
public protocol NumKongMaxSimElement {
    associatedtype MaxSimOutput
    static func _nk_maxsim_pack_size(_ vectors: Int, _ depth: Int) throws -> Int
    static func _nk_maxsim_packed_shape(_ packed: UnsafeRawPointer, _ vectors: inout Int, _ depth: inout Int) throws
    static func _nk_maxsim_pack(
        _ vectorsData: UnsafePointer<Self>, _ vectorsCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer) throws
    static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<MaxSimOutput>) throws
}

// MARK: - MaxSimPackedMatrix

/// Owns a packed representation of multi-vector embeddings for MaxSim scoring.
public final class MaxSimPackedMatrix<Element: NumKongMaxSimElement>: @unchecked Sendable {
    public let vectors: Int
    public let depth: Int
    public let byteCount: Int

    @usableFromInline
    let rawPointer: UnsafeMutableRawPointer

    @usableFromInline
    init(vectors: Int, depth: Int, byteCount: Int, rawPointer: UnsafeMutableRawPointer) {
        self.vectors = vectors
        self.depth = depth
        self.byteCount = byteCount
        self.rawPointer = rawPointer
    }

    deinit {
        rawPointer.deallocate()
    }

    /// Packs a matrix view into the MaxSim-optimized layout.
    public convenience init(packing matrix: MatrixView<Element>) throws {
        guard matrix.rows > 0 && matrix.cols > 0 else {
            throw NumKongMatrixError.invalidDimensions
        }
        let bytes = try Element._nk_maxsim_pack_size(matrix.rows, matrix.cols)
        guard bytes > 0 else { throw NumKongMatrixError.packedBufferTooSmall }
        let ptr = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 64)
        self.init(vectors: matrix.rows, depth: matrix.cols, byteCount: bytes, rawPointer: ptr)
        try Element._nk_maxsim_pack(matrix.baseAddress, matrix.rows, matrix.cols, matrix.rowStrideBytes, ptr)
    }

    /// Computes the MaxSim score between this query and a document's packed matrix.
    public func score(_ document: MaxSimPackedMatrix<Element>) throws -> Element.MaxSimOutput {
        let ptr = UnsafeMutablePointer<Element.MaxSimOutput>.allocate(capacity: 1)
        let raw = UnsafeMutableRawPointer(ptr)
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: MemoryLayout<Element.MaxSimOutput>.size)
        defer { ptr.deallocate() }
        try Element._nk_maxsim_packed(
            UnsafeRawPointer(rawPointer),
            UnsafeRawPointer(document.rawPointer),
            vectors,
            document.vectors,
            depth,
            ptr
        )
        return ptr.pointee
    }

    /// Reads the packed shape, __[vectors,depth]__, from the buffer's self-describing header.
    public var shape: (vectors: Int, depth: Int) {
        get throws {
            var v = 0
            var d = 0
            try Element._nk_maxsim_packed_shape(UnsafeRawPointer(rawPointer), &v, &d)
            return (v, d)
        }
    }
}

// MARK: - Float32 MaxSim Conformance

extension Float32: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float64

    public static func _nk_maxsim_pack_size(_ vectors: Int, _ depth: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(
            nk_maxsim_pack_size_f32_best(nk_size_t(vectors), nk_size_t(depth), Capabilities.enabled.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_maxsim_packed_shape(_ packed: UnsafeRawPointer, _ vectors: inout Int, _ depth: inout Int)
        throws
    {
        var v: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_maxsim_packed_shape_f32_best(packed, &v, &d, Capabilities.enabled.native, nil))
        vectors = Int(v)
        depth = Int(d)
    }

    public static func _nk_maxsim_pack(
        _ vectorsData: UnsafePointer<Float32>, _ vectorsCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) throws {
        try _nkCheck(
            nk_maxsim_pack_f32_best(
                vectorsData, nk_size_t(vectorsCount), nk_size_t(depth), nk_size_t(stride), packed,
                Capabilities.enabled.native, nil))
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float64>
    ) throws {
        try _nkCheck(
            nk_maxsim_packed_f32_best(
                queryPacked, docPacked, nk_size_t(queryCount), nk_size_t(docCount), nk_size_t(depth), result,
                Capabilities.enabled.native, nil))
    }
}

// MARK: - BFloat16 MaxSim Conformance

extension BFloat16: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float32

    public static func _nk_maxsim_pack_size(_ vectors: Int, _ depth: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(
            nk_maxsim_pack_size_bf16_best(nk_size_t(vectors), nk_size_t(depth), Capabilities.enabled.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_maxsim_packed_shape(_ packed: UnsafeRawPointer, _ vectors: inout Int, _ depth: inout Int)
        throws
    {
        var v: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_maxsim_packed_shape_bf16_best(packed, &v, &d, Capabilities.enabled.native, nil))
        vectors = Int(v)
        depth = Int(d)
    }

    public static func _nk_maxsim_pack(
        _ vectorsData: UnsafePointer<BFloat16>, _ vectorsCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(vectorsData).assumingMemoryBound(to: nk_bf16_t.self)
        try _nkCheck(
            nk_maxsim_pack_bf16_best(
                cPtr, nk_size_t(vectorsCount), nk_size_t(depth), nk_size_t(stride), packed,
                Capabilities.enabled.native, nil))
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float32>
    ) throws {
        try _nkCheck(
            nk_maxsim_packed_bf16_best(
                queryPacked, docPacked, nk_size_t(queryCount), nk_size_t(docCount), nk_size_t(depth), result,
                Capabilities.enabled.native, nil))
    }
}

// MARK: - Float16 MaxSim Conformance

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
extension Float16: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float32

    public static func _nk_maxsim_pack_size(_ vectors: Int, _ depth: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(
            nk_maxsim_pack_size_f16_best(nk_size_t(vectors), nk_size_t(depth), Capabilities.enabled.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_maxsim_packed_shape(_ packed: UnsafeRawPointer, _ vectors: inout Int, _ depth: inout Int)
        throws
    {
        var v: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_maxsim_packed_shape_f16_best(packed, &v, &d, Capabilities.enabled.native, nil))
        vectors = Int(v)
        depth = Int(d)
    }

    public static func _nk_maxsim_pack(
        _ vectorsData: UnsafePointer<Float16>, _ vectorsCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(vectorsData).assumingMemoryBound(to: nk_f16_t.self)
        try _nkCheck(
            nk_maxsim_pack_f16_best(
                cPtr, nk_size_t(vectorsCount), nk_size_t(depth), nk_size_t(stride), packed,
                Capabilities.enabled.native, nil))
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float32>
    ) throws {
        try _nkCheck(
            nk_maxsim_packed_f16_best(
                queryPacked, docPacked, nk_size_t(queryCount), nk_size_t(docCount), nk_size_t(depth), result,
                Capabilities.enabled.native, nil))
    }
}
#endif
