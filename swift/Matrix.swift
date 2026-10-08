//
//  swift/Matrix.swift
//  Views, spans, and protocols for row-major matrices and packed kernel right-hand sides.
//
//  - Author: Ash Vardanian
//  - Date: March 14, 2026
//

import CNumKong

/// Non-owning, immutable view over a row-major matrix stored in contiguous memory.
public struct MatrixView<Element> {
    public let baseAddress: UnsafePointer<Element>
    public let rows: Int
    public let columns: Int
    public let rowStrideBytes: Int

    @inlinable
    public init(
        baseAddress: UnsafePointer<Element>,
        rows: Int,
        columns: Int,
        rowStrideBytes: Int? = nil
    ) {
        self.baseAddress = baseAddress
        self.rows = rows
        self.columns = columns
        self.rowStrideBytes = rowStrideBytes ?? columns * MemoryLayout<Element>.stride
    }
}

/// Non-owning, mutable view over a row-major matrix stored in contiguous memory.
public struct MatrixSpan<Element> {
    public let baseAddress: UnsafeMutablePointer<Element>
    public let rows: Int
    public let columns: Int
    public let rowStrideBytes: Int

    @inlinable
    public init(
        baseAddress: UnsafeMutablePointer<Element>,
        rows: Int,
        columns: Int,
        rowStrideBytes: Int? = nil
    ) {
        self.baseAddress = baseAddress
        self.rows = rows
        self.columns = columns
        self.rowStrideBytes = rowStrideBytes ?? columns * MemoryLayout<Element>.stride
    }
}

// MARK: - PackedMatrix

/// Owns a kernel-optimized packed copy of a matrix for batch distance computations.
public final class PackedMatrix<Element>: @unchecked Sendable {
    public let rows: Int
    public let columns: Int
    public let byteCount: Int

    @usableFromInline
    let rawPointer: UnsafeMutableRawPointer

    @usableFromInline
    init(rows: Int, columns: Int, byteCount: Int, rawPointer: UnsafeMutableRawPointer) {
        self.rows = rows
        self.columns = columns
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
    guard matrix.rows > 0 && matrix.columns > 0 else {
        throw fail(.unexpectedDimensions, "rows and columns must be positive")
    }
    let minStride = matrix.columns * MemoryLayout<Element>.stride
    guard matrix.rowStrideBytes >= minStride else {
        throw fail(.unexpectedDimensions, "the row stride is shorter than a row")
    }
}

@usableFromInline
func _nkValidateMatrixSpan<Element>(_ matrix: MatrixSpan<Element>) throws {
    guard matrix.rows > 0 && matrix.columns > 0 else {
        throw fail(.unexpectedDimensions, "rows and columns must be positive")
    }
    let minStride = matrix.columns * MemoryLayout<Element>.stride
    guard matrix.rowStrideBytes >= minStride else {
        throw fail(.unexpectedDimensions, "the row stride is shorter than a row")
    }
}

// MARK: - Dots Protocol

/// Element type that supports batch dot-product matrix operations.
public protocol NumKongDotsMatrixElement {
    associatedtype DotsOutput

    static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int
    static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int) throws
    static func _nk_dots_pack(
        _ b: UnsafePointer<Self>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer) throws
    static func _nk_dots_packed(
        _ a: UnsafePointer<Self>,
        _ bPacked: UnsafeRawPointer,
        _ c: UnsafeMutablePointer<DotsOutput>,
        _ m: Int,
        _ n: Int,
        _ k: Int,
        _ aStride: Int,
        _ cStride: Int
    ) throws
    static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<DotsOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowsBegin: Int,
        _ rowsEnd: Int
    ) throws
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
        _ columns: Int,
        _ depth: Int,
        _ aStride: Int,
        _ rStride: Int
    ) throws
    static func _nk_euclideans_packed(
        _ a: UnsafePointer<Self>,
        _ bPacked: UnsafeRawPointer,
        _ result: UnsafeMutablePointer<SpatialOutput>,
        _ rows: Int,
        _ columns: Int,
        _ depth: Int,
        _ aStride: Int,
        _ rStride: Int
    ) throws
    static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<SpatialOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowsBegin: Int,
        _ rowsEnd: Int
    ) throws
    static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<SpatialOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowsBegin: Int,
        _ rowsEnd: Int
    ) throws
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
        _ columns: Int,
        _ depth: Int,
        _ aStride: Int,
        _ rStride: Int
    ) throws
    static func _nk_hammings_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<HammingOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowsBegin: Int,
        _ rowsEnd: Int
    ) throws
    static func _nk_jaccards_packed(
        _ a: UnsafePointer<Self>,
        _ bPacked: UnsafeRawPointer,
        _ result: UnsafeMutablePointer<JaccardOutput>,
        _ rows: Int,
        _ columns: Int,
        _ depth: Int,
        _ aStride: Int,
        _ rStride: Int
    ) throws
    static func _nk_jaccards_symmetric(
        _ vectors: UnsafePointer<Self>,
        _ result: UnsafeMutablePointer<JaccardOutput>,
        _ nVectors: Int,
        _ depth: Int,
        _ stride: Int,
        _ resultStride: Int,
        _ rowsBegin: Int,
        _ rowsEnd: Int
    ) throws
}

// MARK: - PackedMatrix Packing

extension PackedMatrix where Element: NumKongDotsMatrixElement {
    /// Packs the given matrix view into a kernel-optimized layout for batch dot products.
    public convenience init(packing matrix: MatrixView<Element>) throws {
        try _nkValidateMatrixView(matrix)
        let bytes = try Element._nk_dots_pack_size(matrix.rows, matrix.columns)
        guard bytes > 0 else { throw fail(.missingKernel, "no capability sized a pack") }
        let ptr = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 64)
        self.init(rows: matrix.rows, columns: matrix.columns, byteCount: bytes, rawPointer: ptr)
        try Element._nk_dots_pack(matrix.baseAddress, matrix.rows, matrix.columns, matrix.rowStrideBytes, ptr)
    }

    /// Reads the packed matrix shape, __[rows,columns]__, from the buffer's self-describing header.
    public var shape: (rows: Int, columns: Int) {
        get throws {
            var r = 0
            var c = 0
            try Element._nk_dots_packed_shape(UnsafeRawPointer(rawPointer), &r, &c)
            return (r, c)
        }
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

    guard a.columns == bPacked.columns else { throw fail(.unexpectedDimensions, "the operands differ in depth") }
    guard result.rows == a.rows && result.columns == bPacked.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    try Element._nk_dots_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.columns,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes the symmetric dot-product matrix for all row pairs in `vectors`.
@inlinable
public func dots_symmetric<Element: NumKongDotsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.DotsOutput>,
    rowsBegin: Int = 0,
    rowsEnd: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.columns == vectors.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    let resolvedRowsEnd = rowsEnd ?? vectors.rows
    guard rowsBegin >= 0 && rowsBegin <= resolvedRowsEnd && resolvedRowsEnd <= vectors.rows else {
        throw fail(.unexpectedDimensions, "the row window exceeds the matrix")
    }

    try Element._nk_dots_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.columns,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowsBegin,
        resolvedRowsEnd
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

    guard a.columns == bPacked.columns else { throw fail(.unexpectedDimensions, "the operands differ in depth") }
    guard result.rows == a.rows && result.columns == bPacked.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    try Element._nk_angulars_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.columns,
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

    guard a.columns == bPacked.columns else { throw fail(.unexpectedDimensions, "the operands differ in depth") }
    guard result.rows == a.rows && result.columns == bPacked.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    try Element._nk_euclideans_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.columns,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes the symmetric angular-distance matrix for all row pairs in `vectors`.
@inlinable
public func angulars_symmetric<Element: NumKongSpatialsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.SpatialOutput>,
    rowsBegin: Int = 0,
    rowsEnd: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.columns == vectors.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    let resolvedRowsEnd = rowsEnd ?? vectors.rows
    guard rowsBegin >= 0 && rowsBegin <= resolvedRowsEnd && resolvedRowsEnd <= vectors.rows else {
        throw fail(.unexpectedDimensions, "the row window exceeds the matrix")
    }

    try Element._nk_angulars_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.columns,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowsBegin,
        resolvedRowsEnd
    )
}

/// Computes the symmetric Euclidean-distance matrix for all row pairs in `vectors`.
@inlinable
public func euclideans_symmetric<Element: NumKongSpatialsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.SpatialOutput>,
    rowsBegin: Int = 0,
    rowsEnd: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.columns == vectors.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    let resolvedRowsEnd = rowsEnd ?? vectors.rows
    guard rowsBegin >= 0 && rowsBegin <= resolvedRowsEnd && resolvedRowsEnd <= vectors.rows else {
        throw fail(.unexpectedDimensions, "the row window exceeds the matrix")
    }

    try Element._nk_euclideans_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.columns,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowsBegin,
        resolvedRowsEnd
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

    guard a.columns == bPacked.columns else { throw fail(.unexpectedDimensions, "the operands differ in depth") }
    guard result.rows == a.rows && result.columns == bPacked.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    try Element._nk_hammings_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.columns,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes the symmetric Hamming-distance matrix for all row pairs in `vectors`.
@inlinable
public func hammings_symmetric<Element: NumKongSetsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.HammingOutput>,
    rowsBegin: Int = 0,
    rowsEnd: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.columns == vectors.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    let resolvedRowsEnd = rowsEnd ?? vectors.rows
    guard rowsBegin >= 0 && rowsBegin <= resolvedRowsEnd && resolvedRowsEnd <= vectors.rows else {
        throw fail(.unexpectedDimensions, "the row window exceeds the matrix")
    }

    try Element._nk_hammings_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.columns,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowsBegin,
        resolvedRowsEnd
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

    guard a.columns == bPacked.columns else { throw fail(.unexpectedDimensions, "the operands differ in depth") }
    guard result.rows == a.rows && result.columns == bPacked.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    try Element._nk_jaccards_packed(
        a.baseAddress,
        UnsafeRawPointer(bPacked.rawPointer),
        result.baseAddress,
        a.rows,
        bPacked.rows,
        a.columns,
        a.rowStrideBytes,
        result.rowStrideBytes
    )
}

/// Computes the symmetric Jaccard-distance matrix for all row pairs in `vectors`.
@inlinable
public func jaccards_symmetric<Element: NumKongSetsMatrixElement>(
    _ vectors: MatrixView<Element>,
    _ result: inout MatrixSpan<Element.JaccardOutput>,
    rowsBegin: Int = 0,
    rowsEnd: Int? = nil
) throws {
    try _nkValidateMatrixView(vectors)
    try _nkValidateMatrixSpan(result)

    guard result.rows == vectors.rows && result.columns == vectors.rows else {
        throw fail(.unexpectedDimensions, "the result shape does not match the operands")
    }

    let resolvedRowsEnd = rowsEnd ?? vectors.rows
    guard rowsBegin >= 0 && rowsBegin <= resolvedRowsEnd && resolvedRowsEnd <= vectors.rows else {
        throw fail(.unexpectedDimensions, "the row window exceeds the matrix")
    }

    try Element._nk_jaccards_symmetric(
        vectors.baseAddress,
        result.baseAddress,
        vectors.rows,
        vectors.columns,
        vectors.rowStrideBytes,
        result.rowStrideBytes,
        rowsBegin,
        resolvedRowsEnd
    )
}

// MARK: - Kernel Bindings: Float64

extension Float64: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float64

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_f64_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_f64_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<Float64>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        try _nkCheck(
            nk_dots_pack_f64_best(
                b,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<Float64>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float64>, _ m: Int,
        _ n: Int, _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        try _nkCheck(
            nk_dots_packed_f64_best(
                a,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Float64>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_dots_symmetric_f64_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension Float64: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float64

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<Float64>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float64>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        try _nkCheck(
            nk_angulars_packed_f64_best(
                a,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<Float64>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float64>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        try _nkCheck(
            nk_euclideans_packed_f64_best(
                a,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Float64>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_angulars_symmetric_f64_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Float64>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_euclideans_symmetric_f64_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: Float32

extension Float32: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float64

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_f32_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_f32_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<Float32>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        try _nkCheck(
            nk_dots_pack_f32_best(
                b,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<Float32>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float64>, _ m: Int,
        _ n: Int, _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        try _nkCheck(
            nk_dots_packed_f32_best(
                a,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Float32>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_dots_symmetric_f32_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension Float32: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float64

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<Float32>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float64>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        try _nkCheck(
            nk_angulars_packed_f32_best(
                a,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<Float32>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float64>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        try _nkCheck(
            nk_euclideans_packed_f32_best(
                a,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Float32>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_angulars_symmetric_f32_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Float32>, _ result: UnsafeMutablePointer<Float64>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_euclideans_symmetric_f32_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: BFloat16

extension BFloat16: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_bf16_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_bf16_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<BFloat16>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_bf16_t.self)
        try _nkCheck(
            nk_dots_pack_bf16_best(
                cPtr,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<BFloat16>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int,
        _ n: Int, _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_bf16_t.self)
        try _nkCheck(
            nk_dots_packed_bf16_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<BFloat16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_bf16_t.self)
        try _nkCheck(
            nk_dots_symmetric_bf16_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension BFloat16: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<BFloat16>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_bf16_t.self)
        try _nkCheck(
            nk_angulars_packed_bf16_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<BFloat16>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_bf16_t.self)
        try _nkCheck(
            nk_euclideans_packed_bf16_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<BFloat16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_bf16_t.self)
        try _nkCheck(
            nk_angulars_symmetric_bf16_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<BFloat16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_bf16_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_bf16_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: Float16

#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
extension Float16: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_f16_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_f16_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<Float16>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_f16_t.self)
        try _nkCheck(
            nk_dots_pack_f16_best(
                cPtr,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<Float16>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int,
        _ n: Int, _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_f16_t.self)
        try _nkCheck(
            nk_dots_packed_f16_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Float16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_f16_t.self)
        try _nkCheck(
            nk_dots_symmetric_f16_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension Float16: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<Float16>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_f16_t.self)
        try _nkCheck(
            nk_angulars_packed_f16_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<Float16>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_f16_t.self)
        try _nkCheck(
            nk_euclideans_packed_f16_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Float16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_f16_t.self)
        try _nkCheck(
            nk_angulars_symmetric_f16_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Float16>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_f16_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_f16_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}
#endif

// MARK: - Kernel Bindings: E5M2

extension E5M2: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_e5m2_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_e5m2_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E5M2>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e5m2_t.self)
        try _nkCheck(
            nk_dots_pack_e5m2_best(
                cPtr,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E5M2>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e5m2_t.self)
        try _nkCheck(
            nk_dots_packed_e5m2_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E5M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e5m2_t.self)
        try _nkCheck(
            nk_dots_symmetric_e5m2_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension E5M2: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E5M2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e5m2_t.self)
        try _nkCheck(
            nk_angulars_packed_e5m2_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E5M2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e5m2_t.self)
        try _nkCheck(
            nk_euclideans_packed_e5m2_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E5M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e5m2_t.self)
        try _nkCheck(
            nk_angulars_symmetric_e5m2_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E5M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e5m2_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_e5m2_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: E4M3

extension E4M3: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_e4m3_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_e4m3_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E4M3>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e4m3_t.self)
        try _nkCheck(
            nk_dots_pack_e4m3_best(
                cPtr,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E4M3>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e4m3_t.self)
        try _nkCheck(
            nk_dots_packed_e4m3_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E4M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e4m3_t.self)
        try _nkCheck(
            nk_dots_symmetric_e4m3_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension E4M3: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E4M3>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e4m3_t.self)
        try _nkCheck(
            nk_angulars_packed_e4m3_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E4M3>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e4m3_t.self)
        try _nkCheck(
            nk_euclideans_packed_e4m3_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E4M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e4m3_t.self)
        try _nkCheck(
            nk_angulars_symmetric_e4m3_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E4M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e4m3_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_e4m3_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: E3M2

extension E3M2: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_e3m2_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_e3m2_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E3M2>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e3m2_t.self)
        try _nkCheck(
            nk_dots_pack_e3m2_best(
                cPtr,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E3M2>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e3m2_t.self)
        try _nkCheck(
            nk_dots_packed_e3m2_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E3M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e3m2_t.self)
        try _nkCheck(
            nk_dots_symmetric_e3m2_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension E3M2: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E3M2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e3m2_t.self)
        try _nkCheck(
            nk_angulars_packed_e3m2_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E3M2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e3m2_t.self)
        try _nkCheck(
            nk_euclideans_packed_e3m2_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E3M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e3m2_t.self)
        try _nkCheck(
            nk_angulars_symmetric_e3m2_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E3M2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e3m2_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_e3m2_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: E2M3

extension E2M3: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_e2m3_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_e2m3_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E2M3>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e2m3_t.self)
        try _nkCheck(
            nk_dots_pack_e2m3_best(
                cPtr,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E2M3>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m3_t.self)
        try _nkCheck(
            nk_dots_packed_e2m3_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E2M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m3_t.self)
        try _nkCheck(
            nk_dots_symmetric_e2m3_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension E2M3: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E2M3>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m3_t.self)
        try _nkCheck(
            nk_angulars_packed_e2m3_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E2M3>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m3_t.self)
        try _nkCheck(
            nk_euclideans_packed_e2m3_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E2M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m3_t.self)
        try _nkCheck(
            nk_angulars_symmetric_e2m3_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E2M3>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m3_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_e2m3_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: E2M1x2

extension E2M1x2: NumKongDotsMatrixElement {
    public typealias DotsOutput = Float32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(
            nk_dots_pack_size_e2m1_best(
                nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_e2m1_k)), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_e2m1_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = dimensionsToValues(Int(d), nk_e2m1_k)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<E2M1x2>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_e2m1x2_t.self)
        try _nkCheck(
            nk_dots_pack_e2m1_best(
                cPtr,
                nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_e2m1_k)), nk_size_t(bStride), packed, 0,
                nk_size_t(n),
                Capabilities.cpus.native, nil
            ))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<E2M1x2>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Float32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m1x2_t.self)
        try _nkCheck(
            nk_dots_packed_e2m1_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_e2m1_k)),
                nk_size_t(aStride),
                nk_size_t(cStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<E2M1x2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m1x2_t.self)
        try _nkCheck(
            nk_dots_symmetric_e2m1_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_e2m1_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension E2M1x2: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<E2M1x2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m1x2_t.self)
        try _nkCheck(
            nk_angulars_packed_e2m1_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns),
                nk_size_t(valuesToDimensions(depth, nk_e2m1_k)),
                nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<E2M1x2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_e2m1x2_t.self)
        try _nkCheck(
            nk_euclideans_packed_e2m1_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns),
                nk_size_t(valuesToDimensions(depth, nk_e2m1_k)),
                nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<E2M1x2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m1x2_t.self)
        try _nkCheck(
            nk_angulars_symmetric_e2m1_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_e2m1_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<E2M1x2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_e2m1x2_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_e2m1_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_e2m1_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: Int8

extension Int8: NumKongDotsMatrixElement {
    public typealias DotsOutput = Int32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_i8_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_i8_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<Int8>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        try _nkCheck(
            nk_dots_pack_i8_best(
                b,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<Int8>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Int32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        try _nkCheck(
            nk_dots_packed_i8_best(
                a,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<Int8>, _ result: UnsafeMutablePointer<Int32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_dots_symmetric_i8_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension Int8: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<Int8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        try _nkCheck(
            nk_angulars_packed_i8_best(
                a,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<Int8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        try _nkCheck(
            nk_euclideans_packed_i8_best(
                a,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<Int8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_angulars_symmetric_i8_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<Int8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_euclideans_symmetric_i8_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: I4x2

extension I4x2: NumKongDotsMatrixElement {
    public typealias DotsOutput = Int32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(
            nk_dots_pack_size_i4_best(
                nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_i4_k)), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_i4_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = dimensionsToValues(Int(d), nk_i4_k)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<I4x2>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_i4x2_t.self)
        try _nkCheck(
            nk_dots_pack_i4_best(
                cPtr,
                nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_i4_k)), nk_size_t(bStride), packed, 0,
                nk_size_t(n), Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<I4x2>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<Int32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_i4x2_t.self)
        try _nkCheck(
            nk_dots_packed_i4_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_i4_k)),
                nk_size_t(aStride),
                nk_size_t(cStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<I4x2>, _ result: UnsafeMutablePointer<Int32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_i4x2_t.self)
        try _nkCheck(
            nk_dots_symmetric_i4_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_i4_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension I4x2: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<I4x2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_i4x2_t.self)
        try _nkCheck(
            nk_angulars_packed_i4_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(valuesToDimensions(depth, nk_i4_k)),
                nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<I4x2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_i4x2_t.self)
        try _nkCheck(
            nk_euclideans_packed_i4_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(valuesToDimensions(depth, nk_i4_k)),
                nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<I4x2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_i4x2_t.self)
        try _nkCheck(
            nk_angulars_symmetric_i4_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_i4_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<I4x2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_i4x2_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_i4_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_i4_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: UInt8

extension UInt8: NumKongDotsMatrixElement {
    public typealias DotsOutput = UInt32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(nk_dots_pack_size_u8_best(nk_size_t(n), nk_size_t(k), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_u8_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = Int(d)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<UInt8>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        try _nkCheck(
            nk_dots_pack_u8_best(
                b,
                nk_size_t(n), nk_size_t(k), nk_size_t(bStride), packed, 0, nk_size_t(n),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<UInt8>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<UInt32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        try _nkCheck(
            nk_dots_packed_u8_best(
                a,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(k), nk_size_t(aStride), nk_size_t(cStride),
                Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<UInt8>, _ result: UnsafeMutablePointer<UInt32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_dots_symmetric_u8_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension UInt8: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<UInt8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        try _nkCheck(
            nk_angulars_packed_u8_best(
                a,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<UInt8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        try _nkCheck(
            nk_euclideans_packed_u8_best(
                a,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(depth), nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<UInt8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_angulars_symmetric_u8_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<UInt8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        try _nkCheck(
            nk_euclideans_symmetric_u8_best(
                vectors,
                nk_size_t(nVectors), nk_size_t(depth), nk_size_t(stride), result, nk_size_t(resultStride),
                nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: U4x2

extension U4x2: NumKongDotsMatrixElement {
    public typealias DotsOutput = UInt32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(
            nk_dots_pack_size_u4_best(
                nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_u4_k)), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_u4_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = dimensionsToValues(Int(d), nk_u4_k)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<U4x2>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_u4x2_t.self)
        try _nkCheck(
            nk_dots_pack_u4_best(
                cPtr,
                nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_u4_k)), nk_size_t(bStride), packed, 0,
                nk_size_t(n), Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<U4x2>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<UInt32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u4x2_t.self)
        try _nkCheck(
            nk_dots_packed_u4_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_u4_k)),
                nk_size_t(aStride),
                nk_size_t(cStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<U4x2>, _ result: UnsafeMutablePointer<UInt32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u4x2_t.self)
        try _nkCheck(
            nk_dots_symmetric_u4_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_u4_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

extension U4x2: NumKongSpatialsMatrixElement {
    public typealias SpatialOutput = Float32

    public static func _nk_angulars_packed(
        _ a: UnsafePointer<U4x2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u4x2_t.self)
        try _nkCheck(
            nk_angulars_packed_u4_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(valuesToDimensions(depth, nk_u4_k)),
                nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_packed(
        _ a: UnsafePointer<U4x2>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u4x2_t.self)
        try _nkCheck(
            nk_euclideans_packed_u4_best(
                cPtr,
                bPacked, result, nk_size_t(rows), nk_size_t(columns), nk_size_t(valuesToDimensions(depth, nk_u4_k)),
                nk_size_t(aStride),
                nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_angulars_symmetric(
        _ vectors: UnsafePointer<U4x2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u4x2_t.self)
        try _nkCheck(
            nk_angulars_symmetric_u4_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_u4_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_euclideans_symmetric(
        _ vectors: UnsafePointer<U4x2>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u4x2_t.self)
        try _nkCheck(
            nk_euclideans_symmetric_u4_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_u4_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: U1x8 Dots

extension U1x8: NumKongDotsMatrixElement {
    public typealias DotsOutput = UInt32

    public static func _nk_dots_pack_size(_ n: Int, _ k: Int) throws -> Int {
        var bytes: nk_size_t = 0
        try _nkCheck(
            nk_dots_pack_size_u1_best(
                nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_u1_k)), Capabilities.cpus.native, &bytes))
        return Int(bytes)
    }

    public static func _nk_dots_packed_shape(_ packed: UnsafeRawPointer, _ columns: inout Int, _ depth: inout Int)
        throws
    {
        var w: nk_size_t = 0
        var d: nk_size_t = 0
        try _nkCheck(nk_dots_packed_shape_u1_best(packed, &w, &d, Capabilities.cpus.native, nil))
        columns = Int(w)
        depth = dimensionsToValues(Int(d), nk_u1_k)
    }

    public static func _nk_dots_pack(
        _ b: UnsafePointer<U1x8>, _ n: Int, _ k: Int, _ bStride: Int, _ packed: UnsafeMutableRawPointer
    ) throws {
        let cPtr = UnsafeRawPointer(b).assumingMemoryBound(to: nk_u1x8_t.self)
        try _nkCheck(
            nk_dots_pack_u1_best(
                cPtr,
                nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_u1_k)), nk_size_t(bStride), packed, 0,
                nk_size_t(n), Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_packed(
        _ a: UnsafePointer<U1x8>, _ bPacked: UnsafeRawPointer, _ c: UnsafeMutablePointer<UInt32>, _ m: Int, _ n: Int,
        _ k: Int, _ aStride: Int, _ cStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u1x8_t.self)
        try _nkCheck(
            nk_dots_packed_u1_best(
                cPtr,
                bPacked, c, nk_size_t(m), nk_size_t(n), nk_size_t(valuesToDimensions(k, nk_u1_k)),
                nk_size_t(aStride),
                nk_size_t(cStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_dots_symmetric(
        _ vectors: UnsafePointer<U1x8>, _ result: UnsafeMutablePointer<UInt32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u1x8_t.self)
        try _nkCheck(
            nk_dots_symmetric_u1_best(
                cPtr,
                nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_u1_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}

// MARK: - Kernel Bindings: U1x8 Sets

extension U1x8: NumKongSetsMatrixElement {
    public typealias HammingOutput = UInt32
    public typealias JaccardOutput = Float32

    public static func _nk_hammings_packed(
        _ a: UnsafePointer<U1x8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<UInt32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u1x8_t.self)
        try _nkCheck(
            nk_hammings_packed_u1_best(
                cPtr, bPacked, result, nk_size_t(rows), nk_size_t(columns),
                nk_size_t(valuesToDimensions(depth, nk_u1_k)),
                nk_size_t(aStride), nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_hammings_symmetric(
        _ vectors: UnsafePointer<U1x8>, _ result: UnsafeMutablePointer<UInt32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u1x8_t.self)
        try _nkCheck(
            nk_hammings_symmetric_u1_best(
                cPtr, nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_u1_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }

    public static func _nk_jaccards_packed(
        _ a: UnsafePointer<U1x8>, _ bPacked: UnsafeRawPointer, _ result: UnsafeMutablePointer<Float32>, _ rows: Int,
        _ columns: Int, _ depth: Int, _ aStride: Int, _ rStride: Int
    ) throws {
        let cPtr = UnsafeRawPointer(a).assumingMemoryBound(to: nk_u1x8_t.self)
        try _nkCheck(
            nk_jaccards_packed_u1_best(
                cPtr, bPacked, result, nk_size_t(rows), nk_size_t(columns),
                nk_size_t(valuesToDimensions(depth, nk_u1_k)),
                nk_size_t(aStride), nk_size_t(rStride), Capabilities.cpus.native, nil))
    }

    public static func _nk_jaccards_symmetric(
        _ vectors: UnsafePointer<U1x8>, _ result: UnsafeMutablePointer<Float32>, _ nVectors: Int, _ depth: Int,
        _ stride: Int, _ resultStride: Int, _ rowsBegin: Int, _ rowsEnd: Int
    ) throws {
        let cPtr = UnsafeRawPointer(vectors).assumingMemoryBound(to: nk_u1x8_t.self)
        try _nkCheck(
            nk_jaccards_symmetric_u1_best(
                cPtr, nk_size_t(nVectors), nk_size_t(valuesToDimensions(depth, nk_u1_k)), nk_size_t(stride), result,
                nk_size_t(resultStride), nk_size_t(rowsBegin), nk_size_t(rowsEnd), Capabilities.cpus.native, nil))
    }
}
