//  Tensor.swift
//  NumKong
//
//  Created by Ash Vardanian on March 14, 2026.
//

import CNumKong

// MARK: - Owning Tensor

/// Owning, row-major dense matrix that deallocates its memory on `deinit`.
public final class Tensor<Element>: @unchecked Sendable {
    @usableFromInline
    var rawPointer: UnsafeMutablePointer<Element>
    public private(set) var rows: Int
    public private(set) var cols: Int
    /// Allocated element capacity (>= `count`) — the ceiling `tryResize` honors.
    public private(set) var capacity: Int
    public var count: Int { rows * cols }
    public var shape: (rows: Int, cols: Int) { (rows, cols) }

    @usableFromInline
    init(rawPointer: UnsafeMutablePointer<Element>, rows: Int, cols: Int, capacity: Int? = nil) {
        self.rawPointer = rawPointer
        self.rows = rows
        self.cols = cols
        self.capacity = capacity ?? (rows * cols)
    }

    deinit {
        rawPointer.deallocate()
    }

    /// Creates a tensor by copying elements from a flat array into owned memory.
    public static func fromArray(_ data: [Element], rows: Int, cols: Int) throws -> Tensor {
        guard rows > 0 && cols > 0 else { throw NumKongMatrixError.invalidDimensions }
        guard data.count == rows * cols else { throw NumKongMatrixError.outputShapeMismatch }
        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: rows * cols)
        data.withUnsafeBufferPointer { src in
            ptr.initialize(from: src.baseAddress!, count: rows * cols)
        }
        return Tensor(rawPointer: ptr, rows: rows, cols: cols)
    }

    /// Creates a tensor filled with a repeating value.
    public static func full(rows: Int, cols: Int, value: Element) throws -> Tensor {
        guard rows > 0 && cols > 0 else { throw NumKongMatrixError.invalidDimensions }
        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: rows * cols)
        ptr.initialize(repeating: value, count: rows * cols)
        return Tensor(rawPointer: ptr, rows: rows, cols: cols)
    }

    @inlinable
    /// Returns a non-owning immutable view of this tensor's storage.
    public func view() -> MatrixView<Element> {
        MatrixView(baseAddress: UnsafePointer(rawPointer), rows: rows, cols: cols)
    }

    @inlinable
    /// Returns a non-owning mutable view of this tensor's storage.
    public func span() -> MatrixSpan<Element> {
        MatrixSpan(baseAddress: rawPointer, rows: rows, cols: cols)
    }

    @inlinable
    public subscript(row: Int, col: Int) -> Element {
        get { rawPointer[row * cols + col] }
        set { rawPointer[row * cols + col] = newValue }
    }

    @inlinable
    /// Returns a buffer pointer to the elements of row `i`.
    public func row(_ i: Int) -> UnsafeBufferPointer<Element> {
        UnsafeBufferPointer(start: UnsafePointer(rawPointer) + i * cols, count: cols)
    }

    /// Resizes within the allocated `capacity` without moving storage, so existing `view()`,
    /// `span()`, and `row()` results stay valid.
    ///
    /// - Returns: `false` (leaving the tensor unchanged) if either extent is negative or
    ///   `rows * cols` exceeds `capacity`; call `reserve(_:)` first to grow.
    @discardableResult
    public func tryResize(rows newRows: Int, cols newCols: Int) -> Bool {
        guard newRows >= 0, newCols >= 0, newRows * newCols <= capacity else { return false }
        rows = newRows
        cols = newCols
        return true
    }

    /// Grows `capacity` to at least `minimumCapacity`, reallocating and copying the live elements.
    /// A no-op when already large enough.
    ///
    /// - Warning: This may move storage, INVALIDATING any outstanding `view()`, `span()`, or
    ///   `row()` pointers obtained before the call.
    public func reserve(_ minimumCapacity: Int) {
        guard minimumCapacity > capacity else { return }
        let grown = UnsafeMutablePointer<Element>.allocate(capacity: minimumCapacity)
        grown.initialize(from: rawPointer, count: count)
        rawPointer.deallocate()
        rawPointer = grown
        capacity = minimumCapacity
    }

    /// Resets to an empty 0×0 tensor while keeping the allocated `capacity`.
    public func clear() {
        rows = 0
        cols = 0
    }
}

// MARK: - Zero-initialized Tensor

extension Tensor {
    @usableFromInline
    static func _zeroInitialized(rows: Int, cols: Int) throws -> Tensor {
        guard rows > 0 && cols > 0 else { throw NumKongMatrixError.invalidDimensions }
        let count = rows * cols
        let ptr = UnsafeMutablePointer<Element>.allocate(capacity: count)
        let raw = UnsafeMutableRawPointer(ptr)
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: count * MemoryLayout<Element>.stride)
        return Tensor(rawPointer: ptr, rows: rows, cols: cols)
    }
}

extension Tensor where Element: ExpressibleByIntegerLiteral {
    /// Creates a zero-filled tensor.
    public static func zeros(rows: Int, cols: Int) throws -> Tensor {
        try full(rows: rows, cols: cols, value: 0)
    }
}

// MARK: - Dots Extensions

extension Tensor where Element: NumKongDotsMatrixElement {
    /// Packs this tensor into a kernel-optimized layout for batch dot products.
    public func packForDots() throws -> PackedMatrix<Element> {
        try PackedMatrix<Element>(packing: view())
    }

    /// Computes dot products between this tensor's rows and a packed matrix, returning an owned result.
    public func dotsPacked(_ packed: PackedMatrix<Element>) throws -> Tensor<Element.DotsOutput> {
        let result = try Tensor<Element.DotsOutput>._zeroInitialized(rows: rows, cols: packed.rows)
        var rSpan = result.span()
        try dots_packed(view(), packed, &rSpan)
        return result
    }

    /// Computes the symmetric dot-product matrix for all row pairs, returning an owned result.
    public func dotsSymmetric(rowStart: Int = 0, rowCount: Int? = nil) throws -> Tensor<Element.DotsOutput> {
        let result = try Tensor<Element.DotsOutput>._zeroInitialized(rows: rows, cols: rows)
        var rSpan = result.span()
        try dots_symmetric(view(), &rSpan, rowStart: rowStart, rowCount: rowCount)
        return result
    }
}

// MARK: - Spatials Extensions

extension Tensor where Element: NumKongSpatialsMatrixElement {
    /// Computes angular distances between this tensor's rows and a packed matrix.
    public func angularsPacked(_ packed: PackedMatrix<Element>) throws -> Tensor<Element.SpatialOutput> {
        let result = try Tensor<Element.SpatialOutput>._zeroInitialized(rows: rows, cols: packed.rows)
        var rSpan = result.span()
        try angulars_packed(view(), packed, &rSpan)
        return result
    }

    /// Computes Euclidean distances between this tensor's rows and a packed matrix.
    public func euclideansPacked(_ packed: PackedMatrix<Element>) throws -> Tensor<Element.SpatialOutput> {
        let result = try Tensor<Element.SpatialOutput>._zeroInitialized(rows: rows, cols: packed.rows)
        var rSpan = result.span()
        try euclideans_packed(view(), packed, &rSpan)
        return result
    }

    /// Computes the symmetric angular-distance matrix for all row pairs.
    public func angularsSymmetric(rowStart: Int = 0, rowCount: Int? = nil) throws -> Tensor<Element.SpatialOutput> {
        let result = try Tensor<Element.SpatialOutput>._zeroInitialized(rows: rows, cols: rows)
        var rSpan = result.span()
        try angulars_symmetric(view(), &rSpan, rowStart: rowStart, rowCount: rowCount)
        return result
    }

    /// Computes the symmetric Euclidean-distance matrix for all row pairs.
    public func euclideansSymmetric(rowStart: Int = 0, rowCount: Int? = nil) throws -> Tensor<Element.SpatialOutput> {
        let result = try Tensor<Element.SpatialOutput>._zeroInitialized(rows: rows, cols: rows)
        var rSpan = result.span()
        try euclideans_symmetric(view(), &rSpan, rowStart: rowStart, rowCount: rowCount)
        return result
    }
}

// MARK: - Sets Extensions

extension Tensor where Element: NumKongSetsMatrixElement {
    /// Computes Hamming distances between this tensor's rows and a packed matrix.
    public func hammingsPacked(_ packed: PackedMatrix<Element>) throws -> Tensor<Element.HammingOutput> {
        let result = try Tensor<Element.HammingOutput>._zeroInitialized(rows: rows, cols: packed.rows)
        var rSpan = result.span()
        try NumKong.hammings_packed(view(), packed, &rSpan)
        return result
    }

    /// Computes the symmetric Hamming-distance matrix for all row pairs.
    public func hammingsSymmetric(rowStart: Int = 0, rowCount: Int? = nil) throws -> Tensor<Element.HammingOutput> {
        let result = try Tensor<Element.HammingOutput>._zeroInitialized(rows: rows, cols: rows)
        var rSpan = result.span()
        try NumKong.hammings_symmetric(view(), &rSpan, rowStart: rowStart, rowCount: rowCount)
        return result
    }

    /// Computes Jaccard distances between this tensor's rows and a packed matrix.
    public func jaccardsPacked(_ packed: PackedMatrix<Element>) throws -> Tensor<Element.JaccardOutput> {
        let result = try Tensor<Element.JaccardOutput>._zeroInitialized(rows: rows, cols: packed.rows)
        var rSpan = result.span()
        try NumKong.jaccards_packed(view(), packed, &rSpan)
        return result
    }

    /// Computes the symmetric Jaccard-distance matrix for all row pairs.
    public func jaccardsSymmetric(rowStart: Int = 0, rowCount: Int? = nil) throws -> Tensor<Element.JaccardOutput> {
        let result = try Tensor<Element.JaccardOutput>._zeroInitialized(rows: rows, cols: rows)
        var rSpan = result.span()
        try NumKong.jaccards_symmetric(view(), &rSpan, rowStart: rowStart, rowCount: rowCount)
        return result
    }
}

// MARK: - MaxSim Extensions

extension Tensor where Element: NumKongMaxSimElement {
    /// Packs this tensor into a MaxSim-optimized layout for late-interaction scoring.
    public func maxSimPack() throws -> MaxSimPackedMatrix<Element> {
        try MaxSimPackedMatrix<Element>(packing: view())
    }
}

// MARK: - PackedMatrix convenience from Tensor

extension PackedMatrix where Element: NumKongDotsMatrixElement {
    /// Packs a tensor's storage into a kernel-optimized layout for batch dot products.
    public convenience init(packing tensor: Tensor<Element>) throws {
        try self.init(packing: tensor.view())
    }
}
