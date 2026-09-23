//  Scalar.swift
//  NumKong
//
//  Created by Ash Vardanian on March 14, 2026.
//

import CNumKong

// MARK: - Spatial Protocols

/// A type that can compute SIMD-accelerated dot products with output type widening.
public protocol NumKongDot {
    associatedtype DotOutput
    static func dot<A, B>(_ a: A, _ b: B) -> DotOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// A type that can compute SIMD-accelerated angular (cosine) distances.
public protocol NumKongAngular {
    associatedtype AngularOutput
    static func angular<A, B>(_ a: A, _ b: B) -> AngularOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// A type that can compute SIMD-accelerated squared Euclidean distances.
public protocol NumKongSqEuclidean {
    associatedtype SqEuclideanOutput
    static func sqeuclidean<A, B>(_ a: A, _ b: B) -> SqEuclideanOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// A type that can compute SIMD-accelerated Euclidean distances.
public protocol NumKongEuclidean {
    associatedtype EuclideanOutput
    static func euclidean<A, B>(_ a: A, _ b: B) -> EuclideanOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// Convenience alias for types supporting all four spatial distance metrics.
public typealias NumKongSpatial = NumKongDot & NumKongAngular & NumKongEuclidean & NumKongSqEuclidean

// MARK: - Floating-Point Scalars

extension Float64: NumKongDot {
    public typealias DotOutput = Float64

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_dot_f64(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float64: NumKongAngular {
    public typealias AngularOutput = Float64

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_angular_f64(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float64: NumKongEuclidean {
    public typealias EuclideanOutput = Float64

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_euclidean_f64(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float64: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = Float64

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float64, B.Element == Float64 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_sqeuclidean_f64(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float32: NumKongDot {
    public typealias DotOutput = Float64

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_dot_f32(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float32: NumKongAngular {
    public typealias AngularOutput = Float64

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_angular_f32(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float32: NumKongEuclidean {
    public typealias EuclideanOutput = Float64

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_euclidean_f32(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Float32: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = Float64

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float64?
    where A: Sequence, B: Sequence, A.Element == Float32, B.Element == Float32 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float64 = 0
            nk_sqeuclidean_f32(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension BFloat16: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == BFloat16, B.Element == BFloat16 {
        _nkWithDensePairRebound(a, b, to: nk_bf16_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_bf16(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension BFloat16: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == BFloat16, B.Element == BFloat16 {
        _nkWithDensePairRebound(a, b, to: nk_bf16_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_bf16(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension BFloat16: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == BFloat16, B.Element == BFloat16 {
        _nkWithDensePairRebound(a, b, to: nk_bf16_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_bf16(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension BFloat16: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == BFloat16, B.Element == BFloat16 {
        _nkWithDensePairRebound(a, b, to: nk_bf16_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_bf16(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

// Float16 is a type-level absence on x86_64 — the compiler rejects it before
// @available runtime checks apply, so we need a compile-time arch guard.
// See: https://github.com/unum-cloud/USearch/pull/739
#if !arch(x86_64)
@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_f16_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_f16_t.self)
            var result: Float32 = 0
            nk_dot_f16(aPtr, bPtr, UInt64(n), &result)
            return result
        }
    }
}

@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_f16_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_f16_t.self)
            var result: Float32 = 0
            nk_angular_f16(aPtr, bPtr, UInt64(n), &result)
            return result
        }
    }
}

@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_f16_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_f16_t.self)
            var result: Float32 = 0
            nk_euclidean_f16(aPtr, bPtr, UInt64(n), &result)
            return result
        }
    }
}

@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Float16, B.Element == Float16 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_f16_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_f16_t.self)
            var result: Float32 = 0
            nk_sqeuclidean_f16(aPtr, bPtr, UInt64(n), &result)
            return result
        }
    }
}
#endif  // !arch(x86_64)

extension E5M2: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E5M2, B.Element == E5M2 {
        _nkWithDensePairRebound(a, b, to: nk_e5m2_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_e5m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E5M2: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E5M2, B.Element == E5M2 {
        _nkWithDensePairRebound(a, b, to: nk_e5m2_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_e5m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E5M2: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E5M2, B.Element == E5M2 {
        _nkWithDensePairRebound(a, b, to: nk_e5m2_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_e5m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E5M2: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E5M2, B.Element == E5M2 {
        _nkWithDensePairRebound(a, b, to: nk_e5m2_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_e5m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E4M3: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E4M3, B.Element == E4M3 {
        _nkWithDensePairRebound(a, b, to: nk_e4m3_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_e4m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E4M3: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E4M3, B.Element == E4M3 {
        _nkWithDensePairRebound(a, b, to: nk_e4m3_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_e4m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E4M3: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E4M3, B.Element == E4M3 {
        _nkWithDensePairRebound(a, b, to: nk_e4m3_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_e4m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E4M3: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E4M3, B.Element == E4M3 {
        _nkWithDensePairRebound(a, b, to: nk_e4m3_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_e4m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E3M2: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E3M2, B.Element == E3M2 {
        _nkWithDensePairRebound(a, b, to: nk_e3m2_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_e3m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E3M2: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E3M2, B.Element == E3M2 {
        _nkWithDensePairRebound(a, b, to: nk_e3m2_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_e3m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E3M2: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E3M2, B.Element == E3M2 {
        _nkWithDensePairRebound(a, b, to: nk_e3m2_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_e3m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E3M2: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E3M2, B.Element == E3M2 {
        _nkWithDensePairRebound(a, b, to: nk_e3m2_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_e3m2(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E2M3: NumKongDot {
    public typealias DotOutput = Float32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E2M3, B.Element == E2M3 {
        _nkWithDensePairRebound(a, b, to: nk_e2m3_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_dot_e2m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E2M3: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E2M3, B.Element == E2M3 {
        _nkWithDensePairRebound(a, b, to: nk_e2m3_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_e2m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E2M3: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E2M3, B.Element == E2M3 {
        _nkWithDensePairRebound(a, b, to: nk_e2m3_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_e2m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension E2M3: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == E2M3, B.Element == E2M3 {
        _nkWithDensePairRebound(a, b, to: nk_e2m3_t.self) { ap, bp, n in
            var result: Float32 = 0
            nk_sqeuclidean_e2m3(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

// MARK: - Signed Integer Scalars

extension Int8: NumKongDot {
    public typealias DotOutput = Int32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Int32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Int32 = 0
            nk_dot_i8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Int8: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_i8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Int8: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_i8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension Int8: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = UInt32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == Int8, B.Element == Int8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: UInt32 = 0
            nk_sqeuclidean_i8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension I4x2: NumKongDot {
    public typealias DotOutput = Int32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> Int32?
    where A: Sequence, B: Sequence, A.Element == I4x2, B.Element == I4x2 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_i4x2_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_i4x2_t.self)
            var result: Int32 = 0
            nk_dot_i4(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_i4_k)), &result)
            return result
        }
    }
}

extension I4x2: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == I4x2, B.Element == I4x2 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_i4x2_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_i4x2_t.self)
            var result: Float32 = 0
            nk_angular_i4(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_i4_k)), &result)
            return result
        }
    }
}

extension I4x2: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == I4x2, B.Element == I4x2 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_i4x2_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_i4x2_t.self)
            var result: Float32 = 0
            nk_euclidean_i4(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_i4_k)), &result)
            return result
        }
    }
}

extension I4x2: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = UInt32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == I4x2, B.Element == I4x2 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_i4x2_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_i4x2_t.self)
            var result: UInt32 = 0
            nk_sqeuclidean_i4(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_i4_k)), &result)
            return result
        }
    }
}

// MARK: - Unsigned Integer Scalars

extension UInt8: NumKongDot {
    public typealias DotOutput = UInt32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == UInt8, B.Element == UInt8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: UInt32 = 0
            nk_dot_u8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension UInt8: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == UInt8, B.Element == UInt8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float32 = 0
            nk_angular_u8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension UInt8: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == UInt8, B.Element == UInt8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: Float32 = 0
            nk_euclidean_u8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension UInt8: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = UInt32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == UInt8, B.Element == UInt8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            var result: UInt32 = 0
            nk_sqeuclidean_u8(ap, bp, UInt64(n), &result)
            return result
        }
    }
}

extension U4x2: NumKongDot {
    public typealias DotOutput = UInt32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == U4x2, B.Element == U4x2 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_u4x2_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_u4x2_t.self)
            var result: UInt32 = 0
            nk_dot_u4(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_u4_k)), &result)
            return result
        }
    }
}

extension U4x2: NumKongAngular {
    public typealias AngularOutput = Float32

    @inlinable @inline(__always)
    public static func angular<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == U4x2, B.Element == U4x2 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_u4x2_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_u4x2_t.self)
            var result: Float32 = 0
            nk_angular_u4(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_u4_k)), &result)
            return result
        }
    }
}

extension U4x2: NumKongEuclidean {
    public typealias EuclideanOutput = Float32

    @inlinable @inline(__always)
    public static func euclidean<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == U4x2, B.Element == U4x2 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_u4x2_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_u4x2_t.self)
            var result: Float32 = 0
            nk_euclidean_u4(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_u4_k)), &result)
            return result
        }
    }
}

extension U4x2: NumKongSqEuclidean {
    public typealias SqEuclideanOutput = UInt32

    @inlinable @inline(__always)
    public static func sqeuclidean<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == U4x2, B.Element == U4x2 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_u4x2_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_u4x2_t.self)
            var result: UInt32 = 0
            nk_sqeuclidean_u4(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_u4_k)), &result)
            return result
        }
    }
}

extension U1x8: NumKongDot {
    public typealias DotOutput = UInt32

    @inlinable @inline(__always)
    public static func dot<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == U1x8, B.Element == U1x8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_u1x8_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_u1x8_t.self)
            var result: UInt32 = 0
            nk_dot_u1(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_u1_k)), &result)
            return result
        }
    }
}

// MARK: - Set Protocols

/// A type that can compute SIMD-accelerated Hamming distances over bit sets.
public protocol NumKongHamming {
    associatedtype HammingOutput
    static func hamming<A, B>(_ a: A, _ b: B) -> HammingOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

/// A type that can compute SIMD-accelerated Jaccard distances over bit sets.
public protocol NumKongJaccard {
    associatedtype JaccardOutput
    static func jaccard<A, B>(_ a: A, _ b: B) -> JaccardOutput?
    where A: Sequence, B: Sequence, A.Element == Self, B.Element == Self
}

// MARK: - U1x8 Sets

extension U1x8: NumKongHamming {
    @inlinable @inline(__always)
    public static func hamming<A, B>(_ a: A, _ b: B) -> UInt32?
    where A: Sequence, B: Sequence, A.Element == U1x8, B.Element == U1x8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_u1x8_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_u1x8_t.self)
            var result: UInt32 = 0
            nk_hamming_u1(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_u1_k)), &result)
            return result
        }
    }
}

extension U1x8: NumKongJaccard {
    @inlinable @inline(__always)
    public static func jaccard<A, B>(_ a: A, _ b: B) -> Float32?
    where A: Sequence, B: Sequence, A.Element == U1x8, B.Element == U1x8 {
        _nkWithDensePair(a, b) { ap, bp, n in
            let aPtr = UnsafeRawPointer(ap).assumingMemoryBound(to: nk_u1x8_t.self)
            let bPtr = UnsafeRawPointer(bp).assumingMemoryBound(to: nk_u1x8_t.self)
            var result: Float32 = 0
            nk_jaccard_u1(aPtr, bPtr, UInt64(valuesToDimensions(n, nk_u1_k)), &result)
            return result
        }
    }
}

// MARK: - Collection Extensions

extension RandomAccessCollection where Element: NumKongDot {
    /// Computes the SIMD-accelerated dot product with another sequence.
    @inlinable @inline(__always)
    public func dot<B>(_ b: B) -> Element.DotOutput?
    where B: Sequence, B.Element == Element {
        Element.dot(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongAngular {
    /// Computes the SIMD-accelerated angular (cosine) distance to another sequence.
    @inlinable @inline(__always)
    public func angular<B>(_ b: B) -> Element.AngularOutput?
    where B: Sequence, B.Element == Element {
        Element.angular(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongEuclidean {
    /// Computes the SIMD-accelerated Euclidean distance to another sequence.
    @inlinable @inline(__always)
    public func euclidean<B>(_ b: B) -> Element.EuclideanOutput?
    where B: Sequence, B.Element == Element {
        Element.euclidean(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongSqEuclidean {
    /// Computes the SIMD-accelerated squared Euclidean distance to another sequence.
    @inlinable @inline(__always)
    public func sqeuclidean<B>(_ b: B) -> Element.SqEuclideanOutput?
    where B: Sequence, B.Element == Element {
        Element.sqeuclidean(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongHamming {
    /// Computes the SIMD-accelerated Hamming distance to another sequence.
    @inlinable @inline(__always)
    public func hamming<B>(_ b: B) -> Element.HammingOutput?
    where B: Sequence, B.Element == Element {
        Element.hamming(self, b)
    }
}

extension RandomAccessCollection where Element: NumKongJaccard {
    /// Computes the SIMD-accelerated Jaccard distance to another sequence.
    @inlinable @inline(__always)
    public func jaccard<B>(_ b: B) -> Element.JaccardOutput?
    where B: Sequence, B.Element == Element {
        Element.jaccard(self, b)
    }
}
