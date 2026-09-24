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
    static func _nk_maxsim_pack_size(_ vectors: Int, _ depth: Int) -> Int
    static func _nk_maxsim_packed_shape(_ packed: UnsafeRawPointer, _ vectors: inout Int, _ depth: inout Int)
    static func _nk_maxsim_pack(
        _ vectorsData: UnsafePointer<Self>, _ vectorsCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer)
    static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<MaxSimOutput>)
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
        let bytes = Element._nk_maxsim_pack_size(matrix.rows, matrix.cols)
        guard bytes > 0 else { throw NumKongMatrixError.packedBufferTooSmall }
        let ptr = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 64)
        Element._nk_maxsim_pack(matrix.baseAddress, matrix.rows, matrix.cols, matrix.rowStrideBytes, ptr)
        self.init(vectors: matrix.rows, depth: matrix.cols, byteCount: bytes, rawPointer: ptr)
    }

    /// Computes the MaxSim score between this query and a document's packed matrix.
    public func score(_ document: MaxSimPackedMatrix<Element>) -> Element.MaxSimOutput {
        let ptr = UnsafeMutablePointer<Element.MaxSimOutput>.allocate(capacity: 1)
        let raw = UnsafeMutableRawPointer(ptr)
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: MemoryLayout<Element.MaxSimOutput>.size)
        defer { ptr.deallocate() }
        Element._nk_maxsim_packed(
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
        var v = 0
        var d = 0
        Element._nk_maxsim_packed_shape(UnsafeRawPointer(rawPointer), &v, &d)
        return (v, d)
    }
}

// MARK: - Float32 MaxSim Conformance

extension Float32: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float64

    public static func _nk_maxsim_pack_size(_ vectors: Int, _ depth: Int) -> Int {
        Int(nk_maxsim_pack_size_f32(nk_size_t(vectors), nk_size_t(depth)))
    }

    public static func _nk_maxsim_packed_shape(_ packed: UnsafeRawPointer, _ vectors: inout Int, _ depth: inout Int) {
        var v: nk_size_t = 0
        var d: nk_size_t = 0
        nk_maxsim_packed_shape_f32(packed, &v, &d)
        vectors = Int(v)
        depth = Int(d)
    }

    public static func _nk_maxsim_pack(
        _ vectorsData: UnsafePointer<Float32>, _ vectorsCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) {
        nk_maxsim_pack_f32(vectorsData, nk_size_t(vectorsCount), nk_size_t(depth), nk_size_t(stride), packed)
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float64>
    ) {
        nk_maxsim_packed_f32(
            queryPacked, docPacked, nk_size_t(queryCount), nk_size_t(docCount), nk_size_t(depth), result)
    }
}

// MARK: - BFloat16 MaxSim Conformance

extension BFloat16: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float32

    public static func _nk_maxsim_pack_size(_ vectors: Int, _ depth: Int) -> Int {
        Int(nk_maxsim_pack_size_bf16(nk_size_t(vectors), nk_size_t(depth)))
    }

    public static func _nk_maxsim_packed_shape(_ packed: UnsafeRawPointer, _ vectors: inout Int, _ depth: inout Int) {
        var v: nk_size_t = 0
        var d: nk_size_t = 0
        nk_maxsim_packed_shape_bf16(packed, &v, &d)
        vectors = Int(v)
        depth = Int(d)
    }

    public static func _nk_maxsim_pack(
        _ vectorsData: UnsafePointer<BFloat16>, _ vectorsCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(vectorsData).assumingMemoryBound(to: nk_bf16_t.self)
        nk_maxsim_pack_bf16(cPtr, nk_size_t(vectorsCount), nk_size_t(depth), nk_size_t(stride), packed)
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float32>
    ) {
        nk_maxsim_packed_bf16(
            queryPacked, docPacked, nk_size_t(queryCount), nk_size_t(docCount), nk_size_t(depth), result)
    }
}

// MARK: - Float16 MaxSim Conformance

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
extension Float16: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float32

    public static func _nk_maxsim_pack_size(_ vectors: Int, _ depth: Int) -> Int {
        Int(nk_maxsim_pack_size_f16(nk_size_t(vectors), nk_size_t(depth)))
    }

    public static func _nk_maxsim_packed_shape(_ packed: UnsafeRawPointer, _ vectors: inout Int, _ depth: inout Int) {
        var v: nk_size_t = 0
        var d: nk_size_t = 0
        nk_maxsim_packed_shape_f16(packed, &v, &d)
        vectors = Int(v)
        depth = Int(d)
    }

    public static func _nk_maxsim_pack(
        _ vectorsData: UnsafePointer<Float16>, _ vectorsCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(vectorsData).assumingMemoryBound(to: nk_f16_t.self)
        nk_maxsim_pack_f16(cPtr, nk_size_t(vectorsCount), nk_size_t(depth), nk_size_t(stride), packed)
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float32>
    ) {
        nk_maxsim_packed_f16(
            queryPacked, docPacked, nk_size_t(queryCount), nk_size_t(docCount), nk_size_t(depth), result)
    }
}
#endif
