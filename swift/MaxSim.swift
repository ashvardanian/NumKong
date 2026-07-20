//  MaxSim.swift
//  NumKong
//
//  Created by Ash Vardanian on March 14, 2026.
//

import CNumKong

// MARK: - MaxSim Protocol

/// Element type that supports MaxSim (late-interaction) scoring with packed representations.
public protocol NumKongMaxSimElement {
    associatedtype MaxSimOutput
    static func _nk_maxsim_pack_size(_ vectorCount: Int, _ depth: Int) -> Int
    static func _nk_maxsim_pack(
        _ vectors: UnsafePointer<Self>, _ vectorCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer)
    static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<MaxSimOutput>)
}

// MARK: - MaxSimPackedMatrix

/// Owns a packed representation of multi-vector embeddings for MaxSim scoring.
public final class MaxSimPackedMatrix<Element: NumKongMaxSimElement>: @unchecked Sendable {
    public let vectorCount: Int
    public let depth: Int
    public let byteCount: Int

    @usableFromInline
    let rawPointer: UnsafeMutableRawPointer

    @usableFromInline
    init(vectorCount: Int, depth: Int, byteCount: Int, rawPointer: UnsafeMutableRawPointer) {
        self.vectorCount = vectorCount
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
        self.init(vectorCount: matrix.rows, depth: matrix.cols, byteCount: bytes, rawPointer: ptr)
    }

    /// Computes the MaxSim score between this (query) and a document's packed matrix.
    public func score(_ document: MaxSimPackedMatrix<Element>) -> Element.MaxSimOutput {
        let ptr = UnsafeMutablePointer<Element.MaxSimOutput>.allocate(capacity: 1)
        let raw = UnsafeMutableRawPointer(ptr)
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: MemoryLayout<Element.MaxSimOutput>.size)
        defer { ptr.deallocate() }
        Element._nk_maxsim_packed(
            UnsafeRawPointer(rawPointer),
            UnsafeRawPointer(document.rawPointer),
            vectorCount,
            document.vectorCount,
            depth,
            ptr
        )
        return ptr.pointee
    }
}

// MARK: - Float32 MaxSim Conformance

extension Float32: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float64

    public static func _nk_maxsim_pack_size(_ vectorCount: Int, _ depth: Int) -> Int {
        Int(nk_maxsim_pack_size_f32(UInt64(vectorCount), UInt64(depth)))
    }

    public static func _nk_maxsim_pack(
        _ vectors: UnsafePointer<Float32>, _ vectorCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) {
        nk_maxsim_pack_f32(vectors, UInt64(vectorCount), UInt64(depth), UInt64(stride), packed)
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float64>
    ) {
        nk_maxsim_packed_f32(queryPacked, docPacked, UInt64(queryCount), UInt64(docCount), UInt64(depth), result)
    }
}

// MARK: - BFloat16 MaxSim Conformance

extension BFloat16: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float32

    public static func _nk_maxsim_pack_size(_ vectorCount: Int, _ depth: Int) -> Int {
        Int(nk_maxsim_pack_size_bf16(UInt64(vectorCount), UInt64(depth)))
    }

    public static func _nk_maxsim_pack(
        _ vectors: UnsafePointer<BFloat16>, _ vectorCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_bf16_t.self)
        nk_maxsim_pack_bf16(cPtr, UInt64(vectorCount), UInt64(depth), UInt64(stride), packed)
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float32>
    ) {
        nk_maxsim_packed_bf16(queryPacked, docPacked, UInt64(queryCount), UInt64(docCount), UInt64(depth), result)
    }
}

// MARK: - Float16 MaxSim Conformance

#if !arch(x86_64)
@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: NumKongMaxSimElement {
    public typealias MaxSimOutput = Float32

    public static func _nk_maxsim_pack_size(_ vectorCount: Int, _ depth: Int) -> Int {
        Int(nk_maxsim_pack_size_f16(UInt64(vectorCount), UInt64(depth)))
    }

    public static func _nk_maxsim_pack(
        _ vectors: UnsafePointer<Float16>, _ vectorCount: Int, _ depth: Int, _ stride: Int,
        _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_f16_t.self)
        nk_maxsim_pack_f16(cPtr, UInt64(vectorCount), UInt64(depth), UInt64(stride), packed)
    }

    public static func _nk_maxsim_packed(
        _ queryPacked: UnsafeRawPointer, _ docPacked: UnsafeRawPointer, _ queryCount: Int, _ docCount: Int,
        _ depth: Int, _ result: UnsafeMutablePointer<Float32>
    ) {
        nk_maxsim_packed_f16(queryPacked, docPacked, UInt64(queryCount), UInt64(docCount), UInt64(depth), result)
    }
}
#endif
