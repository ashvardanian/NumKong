//  Matrix.swift
//  NumKong
//
//  Created by Ash Vardanian on March 14, 2026.
//

import CNumKong

/// Errors thrown by matrix operations.
public enum NumKongMatrixError: Error {
    /// The matrix has zero or negative rows/cols.
    case invalidDimensions
    /// The row stride is smaller than `cols * MemoryLayout<Element>.stride`.
    case invalidStride
    /// The output matrix shape does not match the expected dimensions.
    case outputShapeMismatch
    /// The vector depth (cols) of the two matrices differs.
    case depthMismatch
    /// The requested row window exceeds matrix bounds.
    case rowWindowOutOfBounds
    /// The packed buffer size returned by the kernel is zero.
    case packedBufferTooSmall
}

/// Non-owning, immutable view over a row-major matrix stored in contiguous memory.
public struct MatrixView<Element> {
    public let baseAddress: UnsafePointer<Element>
    public let rows: Int
    public let cols: Int
    public let rowStrideBytes: Int

    @inlinable
    public init(
        baseAddress: UnsafePointer<Element>,
        rows: Int,
        cols: Int,
        rowStrideBytes: Int? = nil
    ) {
        self.baseAddress = baseAddress
        self.rows = rows
        self.cols = cols
        self.rowStrideBytes = rowStrideBytes ?? cols * MemoryLayout<Element>.stride
    }
}

/// Non-owning, mutable view over a row-major matrix stored in contiguous memory.
public struct MatrixSpan<Element> {
    public let baseAddress: UnsafeMutablePointer<Element>
    public let rows: Int
    public let cols: Int
    public let rowStrideBytes: Int

    @inlinable
    public init(
        baseAddress: UnsafeMutablePointer<Element>,
        rows: Int,
        cols: Int,
        rowStrideBytes: Int? = nil
    ) {
        self.baseAddress = baseAddress
        self.rows = rows
        self.cols = cols
        self.rowStrideBytes = rowStrideBytes ?? cols * MemoryLayout<Element>.stride
    }
}

// MARK: - Deprecated Aliases

@available(*, deprecated, renamed: "MatrixView")
public typealias Matrix = MatrixView

@available(*, deprecated, renamed: "MatrixSpan")
public typealias MutableMatrix = MatrixSpan

// MARK: - PackedMatrix

/// Owns a kernel-optimized packed copy of a matrix for batch distance computations.
public final class PackedMatrix<Element>: @unchecked Sendable {
    public let rows: Int
    public let cols: Int
    public let byteCount: Int

    @usableFromInline
    let rawPointer: UnsafeMutableRawPointer

    @usableFromInline
    init(rows: Int, cols: Int, byteCount: Int, rawPointer: UnsafeMutableRawPointer) {
        self.rows = rows
        self.cols = cols
        self.byteCount = byteCount
        self.rawPointer = rawPointer
    }

    deinit {
        rawPointer.deallocate()
    }

    /// Provides read-only access to the underlying packed bytes.
    @inlinable
    public var rawBuffer: UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: rawPointer, count: byteCount)
    }
}

// MARK: - Validation

@usableFromInline
func _nkValidateMatrixView<Element>(_ matrix: MatrixView<Element>) throws {
    guard matrix.rows > 0 && matrix.cols > 0 else {
        throw NumKongMatrixError.invalidDimensions
    }
    let minStride = matrix.cols * MemoryLayout<Element>.stride
    guard matrix.rowStrideBytes >= minStride else {
        throw NumKongMatrixError.invalidStride
    }
}

@usableFromInline
func _nkValidateMatrixSpan<Element>(_ matrix: MatrixSpan<Element>) throws {
    guard matrix.rows > 0 && matrix.cols > 0 else {
        throw NumKongMatrixError.invalidDimensions
    }
    let minStride = matrix.cols * MemoryLayout<Element>.stride
    guard matrix.rowStrideBytes >= minStride else {
        throw NumKongMatrixError.invalidStride
    }
}

// MARK: - Dots Protocol

/// Element type that supports batch dot-product matrix operations.
public protocol NumKongDotsMatrixElement {
    associatedtype DotsOutput

    static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int
    static func _nk_dots_pack(
        _ b: UnsafePointer<Self>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer)
    static func _nk_dots_packed(
        _ a: UnsafePointer<Self>,
        _ bPacked: UnsafeRawPointer,
        _ c: UnsafeMutablePointer<DotsOutput>,
        _ m: Int,
        _ n: Int,
        _ k: Int,
        _ aStride: Int,
        _ cStride: Int
    )
    static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<DotsOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowStart: Int,
        _ rowCount: Int
    )
}

// MARK: - Spatials Protocol

/// Element type that supports batch angular and Euclidean matrix operations.
public protocol NumKongSpatialsMatrixElement: NumKongDotsMatrixElement {
    associatedtype SpatialOutput

    static func _nk_angulars_packed(
        _ a: UnsafePointer<Self>,
        _ bPacked: UnsafeRawPointer,
        _ result: UnsafeMutablePointer<SpatialOutput>,
        _ rows: Int,
        _ cols: Int,
        _ depth: Int,
        _ aStride: Int,
        _ rStride: Int
    )
    static func _nk_euclideans_packed(
        _ a: UnsafePointer<Self>,
        _ bPacked: UnsafeRawPointer,
        _ result: UnsafeMutablePointer<SpatialOutput>,
        _ rows: Int,
        _ cols: Int,
        _ depth: Int,
        _ aStride: Int,
        _ rStride: Int
    )
    static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<SpatialOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowStart: Int,
        _ rowCount: Int
    )
    static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<SpatialOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowStart: Int,
        _ rowCount: Int
    )
}

// MARK: - Sets Protocol

/// Element type that supports batch Hamming and Jaccard matrix operations.
public protocol NumKongSetsMatrixElement: NumKongDotsMatrixElement {
    associatedtype HammingOutput
    associatedtype JaccardOutput

    static func _nk_hammings_packed(
        _ a: UnsafePointer<Self>,
        _ bPacked: UnsafeRawPointer,
        _ result: UnsafeMutablePointer<HammingOutput>,
        _ rows: Int,
        _ cols: Int,
        _ depth: Int,
        _ aStride: Int,
        _ rStride: Int
    )
    static func _nk_hammings_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<HammingOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowStart: Int,
        _ rowCount: Int
    )
    static func _nk_jaccards_packed(
        _ a: UnsafePointer<Self>,
        _ bPacked: UnsafeRawPointer,
        _ result: UnsafeMutablePointer<JaccardOutput>,
        _ rows: Int,
        _ cols: Int,
        _ depth: Int,
        _ aStride: Int,
        _ rStride: Int
    )
    static func _nk_jaccards_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<JaccardOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowStart: Int,
        _ rowCount: Int
    )
}

// MARK: - PackedMatrix Packing

extension PackedMatrix where Element: NumKongDotsMatrixElement {
    /// Packs the given matrix view into a kernel-optimized layout for batch dot products.
    public convenience init(packing matrix: MatrixView<Element>) throws {
        try _nkValidateMatrixView(matrix)
        let bytes = Element._nk_dots_pack_size(matrix.rows, matrix.cols)
        guard bytes > 0 else { throw NumKongMatrixError.packedBufferTooSmall }
        let ptr = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 64)
        Element._nk_dots_pack(matrix.baseAddress, matrix.rows, matrix.cols, matrix.rowStrideBytes, ptr)
        self.init(rows: matrix.rows, cols: matrix.cols, byteCount: bytes, rawPointer: ptr)
    }
}

// MARK: - Dots Functions

/// Computes dot products between each row of `a` and every row packed in `bPacked`.
@inlinable
public func dots_packed<Element: NumKongDotsMatrixElement>(
    _ a: MatrixView<Element>,
    _ bPacked: PackedMatrix<Element>,
    _ result: inout MatrixSpan<Element.DotsOutput>
) throws {
    try _nkValidateMatrixView(a)
    try _nkValidateMatrixSpan(result)

    guard a.cols == bPacked.cols else { throw NumKongMatrixError.depthMismatch }
    guard result.rows == a.rows && result.cols == bPacked.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    Element._nk_dots_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.cols,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes the symmetric dot-product matrix for all row pairs in `vectors`.
@inlinable
public func dots_symmetric<Element: NumKongDotsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.DotsOutput>,
    rowStart: Int = 0,
    rowCount: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.cols == vectors.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    let count = rowCount ?? (vectors.rows - rowStart)
    guard rowStart >= 0 && count >= 0 && rowStart + count <= vectors.rows else {
        throw NumKongMatrixError.rowWindowOutOfBounds
    }

    Element._nk_dots_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.cols,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowStart,
        count
    )
}

// MARK: - Spatials Functions

/// Computes angular distances between each row of `a` and every row packed in `bPacked`.
@inlinable
public func angulars_packed<Element: NumKongSpatialsMatrixElement>(
    _ a: MatrixView<Element>,
    _ bPacked: PackedMatrix<Element>,
    _ result: inout MatrixSpan<Element.SpatialOutput>
) throws {
    try _nkValidateMatrixView(a)
    try _nkValidateMatrixSpan(result)

    guard a.cols == bPacked.cols else { throw NumKongMatrixError.depthMismatch }
    guard result.rows == a.rows && result.cols == bPacked.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    Element._nk_angulars_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.cols,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes Euclidean distances between each row of `a` and every row packed in `bPacked`.
@inlinable
public func euclideans_packed<Element: NumKongSpatialsMatrixElement>(
    _ a: MatrixView<Element>,
    _ bPacked: PackedMatrix<Element>,
    _ result: inout MatrixSpan<Element.SpatialOutput>
) throws {
    try _nkValidateMatrixView(a)
    try _nkValidateMatrixSpan(result)

    guard a.cols == bPacked.cols else { throw NumKongMatrixError.depthMismatch }
    guard result.rows == a.rows && result.cols == bPacked.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    Element._nk_euclideans_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.cols,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes the symmetric angular-distance matrix for all row pairs in `vectors`.
@inlinable
public func angulars_symmetric<Element: NumKongSpatialsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.SpatialOutput>,
    rowStart: Int = 0,
    rowCount: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.cols == vectors.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    let count = rowCount ?? (vectors.rows - rowStart)
    guard rowStart >= 0 && count >= 0 && rowStart + count <= vectors.rows else {
        throw NumKongMatrixError.rowWindowOutOfBounds
    }

    Element._nk_angulars_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.cols,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowStart,
        count
    )
}

/// Computes the symmetric Euclidean-distance matrix for all row pairs in `vectors`.
@inlinable
public func euclideans_symmetric<Element: NumKongSpatialsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.SpatialOutput>,
    rowStart: Int = 0,
    rowCount: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.cols == vectors.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    let count = rowCount ?? (vectors.rows - rowStart)
    guard rowStart >= 0 && count >= 0 && rowStart + count <= vectors.rows else {
        throw NumKongMatrixError.rowWindowOutOfBounds
    }

    Element._nk_euclideans_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.cols,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowStart,
        count
    )
}

// MARK: - Sets Functions

/// Computes Hamming distances between each row of `a` and every row packed in `bPacked`.
@inlinable
public func hammings_packed<Element: NumKongSetsMatrixElement>(
    _ a: MatrixView<Element>,
    _ bPacked: PackedMatrix<Element>,
    _ result: inout MatrixSpan<Element.HammingOutput>
) throws {
    try _nkValidateMatrixView(a)
    try _nkValidateMatrixSpan(result)

    guard a.cols == bPacked.cols else { throw NumKongMatrixError.depthMismatch }
    guard result.rows == a.rows && result.cols == bPacked.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    Element._nk_hammings_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.cols,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes the symmetric Hamming-distance matrix for all row pairs in `vectors`.
@inlinable
public func hammings_symmetric<Element: NumKongSetsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.HammingOutput>,
    rowStart: Int = 0,
    rowCount: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.cols == vectors.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    let count = rowCount ?? (vectors.rows - rowStart)
    guard rowStart >= 0 && count >= 0 && rowStart + count <= vectors.rows else {
        throw NumKongMatrixError.rowWindowOutOfBounds
    }

    Element._nk_hammings_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.cols,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowStart,
        count
    )
}

/// Computes Jaccard distances between each row of `a` and every row packed in `bPacked`.
@inlinable
public func jaccards_packed<Element: NumKongSetsMatrixElement>(
    _ a: MatrixView<Element>,
    _ bPacked: PackedMatrix<Element>,
    _ result: inout MatrixSpan<Element.JaccardOutput>
) throws {
    try _nkValidateMatrixView(a)
    try _nkValidateMatrixSpan(result)

    guard a.cols == bPacked.cols else { throw NumKongMatrixError.depthMismatch }
    guard result.rows == a.rows && result.cols == bPacked.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    Element._nk_jaccards_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.cols,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes the symmetric Jaccard-distance matrix for all row pairs in `vectors`.
@inlinable
public func jaccards_symmetric<Element: NumKongSetsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.JaccardOutput>,
    rowStart: Int = 0,
    rowCount: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.cols == vectors.rows else {
        throw NumKongMatrixError.outputShapeMismatch
    }

    let count = rowCount ?? (vectors.rows - rowStart)
    guard rowStart >= 0 && count >= 0 && rowStart + count <= vectors.rows else {
        throw NumKongMatrixError.rowWindowOutOfBounds
    }

    Element._nk_jaccards_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.cols,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowStart,
        count
    )
}

// MARK: - Kernel Bindings: Float32

extension Float32: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float64

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_f32(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<Float32>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        nk_dots_pack_f32(b, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<Float32>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float64>, _ m: Int,
        _ n: Int, _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        nk_dots_packed_f32(a, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Float32>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_dots_symmetric_f32(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension Float32: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float64

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<Float32>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float64>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        nk_angulars_packed_f32(
            a, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<Float32>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float64>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        nk_euclideans_packed_f32(
            a, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Float32>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_angulars_symmetric_f32(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Float32>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_euclideans_symmetric_f32(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

// MARK: - Kernel Bindings: Float64

extension Float64: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float64

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_f64(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<Float64>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        nk_dots_pack_f64(b, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<Float64>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float64>, _ m: Int,
        _ n: Int, _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        nk_dots_packed_f64(a, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Float64>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_dots_symmetric_f64(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension Float64: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float64

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<Float64>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float64>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        nk_angulars_packed_f64(
            a, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<Float64>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float64>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        nk_euclideans_packed_f64(
            a, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Float64>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_angulars_symmetric_f64(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Float64>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_euclideans_symmetric_f64(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

// MARK: - Kernel Bindings: Int8

extension Int8: NumKongDotsMatrixElement {
    public typealias DotsOutput = Int32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_i8(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<Int8>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        nk_dots_pack_i8(b, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<Int8>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Int32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        nk_dots_packed_i8(a, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Int8>, _ result: UnsafeMutablePointer<Int32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_dots_symmetric_i8(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension Int8: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<Int8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        nk_angulars_packed_i8(
            a, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<Int8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        nk_euclideans_packed_i8(
            a, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Int8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_angulars_symmetric_i8(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Int8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_euclideans_symmetric_i8(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

// MARK: - Kernel Bindings: UInt8

extension UInt8: NumKongDotsMatrixElement {
    public typealias DotsOutput = UInt32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_u8(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<UInt8>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        nk_dots_pack_u8(b, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<UInt8>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<UInt32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        nk_dots_packed_u8(a, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<UInt8>, _ result: UnsafeMutablePointer<UInt32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_dots_symmetric_u8(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension UInt8: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<UInt8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        nk_angulars_packed_u8(
            a, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<UInt8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        nk_euclideans_packed_u8(
            a, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<UInt8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_angulars_symmetric_u8(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<UInt8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        nk_euclideans_symmetric_u8(
            vectors, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

// MARK: - Kernel Bindings: Float16

#if !arch(x86_64)
@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_f16(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<Float16>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_f16_t.self)
        nk_dots_pack_f16(cPtr, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<Float16>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int,
        _ n: Int, _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_f16_t.self)
        nk_dots_packed_f16(cPtr, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Float16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_f16_t.self)
        nk_dots_symmetric_f16(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<Float16>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_f16_t.self)
        nk_angulars_packed_f16(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<Float16>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_f16_t.self)
        nk_euclideans_packed_f16(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Float16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_f16_t.self)
        nk_angulars_symmetric_f16(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Float16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_f16_t.self)
        nk_euclideans_symmetric_f16(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}
#endif

// MARK: - Kernel Bindings: Minifloats

extension BFloat16: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_bf16(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<BFloat16>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_bf16_t.self)
        nk_dots_pack_bf16(cPtr, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<BFloat16>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int,
        _ n: Int, _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_bf16_t.self)
        nk_dots_packed_bf16(cPtr, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<BFloat16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_bf16_t.self)
        nk_dots_symmetric_bf16(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension BFloat16: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<BFloat16>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_bf16_t.self)
        nk_angulars_packed_bf16(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<BFloat16>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_bf16_t.self)
        nk_euclideans_packed_bf16(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<BFloat16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_bf16_t.self)
        nk_angulars_symmetric_bf16(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<BFloat16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_bf16_t.self)
        nk_euclideans_symmetric_bf16(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension E4M3: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_e4m3(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E4M3>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e4m3_t.self)
        nk_dots_pack_e4m3(cPtr, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E4M3>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e4m3_t.self)
        nk_dots_packed_e4m3(cPtr, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E4M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e4m3_t.self)
        nk_dots_symmetric_e4m3(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension E4M3: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E4M3>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e4m3_t.self)
        nk_angulars_packed_e4m3(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E4M3>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e4m3_t.self)
        nk_euclideans_packed_e4m3(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E4M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e4m3_t.self)
        nk_angulars_symmetric_e4m3(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E4M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e4m3_t.self)
        nk_euclideans_symmetric_e4m3(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension E5M2: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_e5m2(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E5M2>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e5m2_t.self)
        nk_dots_pack_e5m2(cPtr, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E5M2>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e5m2_t.self)
        nk_dots_packed_e5m2(cPtr, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E5M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e5m2_t.self)
        nk_dots_symmetric_e5m2(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension E5M2: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E5M2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e5m2_t.self)
        nk_angulars_packed_e5m2(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E5M2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e5m2_t.self)
        nk_euclideans_packed_e5m2(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E5M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e5m2_t.self)
        nk_angulars_symmetric_e5m2(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E5M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e5m2_t.self)
        nk_euclideans_symmetric_e5m2(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension E2M3: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_e2m3(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E2M3>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e2m3_t.self)
        nk_dots_pack_e2m3(cPtr, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E2M3>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m3_t.self)
        nk_dots_packed_e2m3(cPtr, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E2M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m3_t.self)
        nk_dots_symmetric_e2m3(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension E2M3: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E2M3>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m3_t.self)
        nk_angulars_packed_e2m3(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E2M3>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m3_t.self)
        nk_euclideans_packed_e2m3(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E2M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m3_t.self)
        nk_angulars_symmetric_e2m3(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E2M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m3_t.self)
        nk_euclideans_symmetric_e2m3(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension E3M2: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_e3m2(UInt64(n), UInt64(k)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E3M2>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e3m2_t.self)
        nk_dots_pack_e3m2(cPtr, UInt64(n), UInt64(k), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E3M2>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e3m2_t.self)
        nk_dots_packed_e3m2(cPtr, bPacked, c, UInt64(m), UInt64(n), UInt64(k), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E3M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e3m2_t.self)
        nk_dots_symmetric_e3m2(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

extension E3M2: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E3M2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e3m2_t.self)
        nk_angulars_packed_e3m2(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E3M2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e3m2_t.self)
        nk_euclideans_packed_e3m2(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E3M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e3m2_t.self)
        nk_angulars_symmetric_e3m2(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E3M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e3m2_t.self)
        nk_euclideans_symmetric_e3m2(
            cPtr, UInt64(nVectors), UInt64(depth), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

// MARK: - Kernel Bindings: U1x8 Dots

extension U1x8: NumKongDotsMatrixElement {
    public typealias DotsOutput = UInt32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) -> Int {
        Int(nk_dots_pack_size_u1(UInt64(n), UInt64(k * 8)))
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<U1x8>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_u1x8_t.self)
        nk_dots_pack_u1(cPtr, UInt64(n), UInt64(k * 8), UInt64(bStride), packed)
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<U1x8>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<UInt32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u1x8_t.self)
        nk_dots_packed_u1(cPtr, bPacked, c, UInt64(m), UInt64(n), UInt64(k * 8), UInt64(aStride), UInt64(cStride))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<U1x8>, _ result: UnsafeMutablePointer<UInt32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u1x8_t.self)
        nk_dots_symmetric_u1(
            cPtr, UInt64(nVectors), UInt64(depth * 8), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}

// MARK: - Kernel Bindings: U1x8 Sets

extension U1x8: NumKongSetsMatrixElement {
    public typealias HammingOutput = UInt32
    public typealias JaccardOutput = Float32

    public static func _nk_hammings_packed(
        _ a: UnsafePointer<U1x8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<UInt32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u1x8_t.self)
        nk_hammings_packed_u1(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth * 8), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_hammings_symmetric(
        _ vectors: UnsafePointer<U1x8>, _ result: UnsafeMutablePointer<UInt32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u1x8_t.self)
        nk_hammings_symmetric_u1(
            cPtr, UInt64(nVectors), UInt64(depth * 8), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }

    public static func _nk_jaccards_packed(
        _ a: UnsafePointer<U1x8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ cols: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u1x8_t.self)
        nk_jaccards_packed_u1(
            cPtr, bPacked, result, UInt64(rows), UInt64(cols), UInt64(depth * 8), UInt64(aStride), UInt64(rStride))
    }

    public static func _nk_jaccards_symmetric(
        _ vectors: UnsafePointer<U1x8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowStart: Int, _ rowCount: Int
    ) {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u1x8_t.self)
        nk_jaccards_symmetric_u1(
            cPtr, UInt64(nVectors), UInt64(depth * 8), UInt64(stride), result, UInt64(resultStride), UInt64(rowStart),
            UInt64(rowCount))
    }
}
