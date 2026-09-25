// Package numkong provides SIMD-accelerated similarity measures and numeric kernels.
//
// # Operations
//
//   - Dot products: [DotF64], [DotF32], [DotI8], [DotU8]
//   - Angular distances: [AngularF64], [AngularF32], [AngularI8], [AngularU8]
//   - Euclidean distances: [EuclideanF64], [EuclideanF32], [EuclideanI8], [EuclideanU8]
//   - Squared Euclidean: [SqEuclideanF64], [SqEuclideanF32], [SqEuclideanI8], [SqEuclideanU8]
//   - Set similarities: [HammingU8], [HammingU1], [JaccardU1], [JaccardU16], [JaccardU32]
//   - Divergences: [KullbackLeiblerF64], [KullbackLeiblerF32], [JensenShannonF64],
//     [JensenShannonF32]
//   - Geospatial distances: [HaversineF64], [HaversineF32], [VincentyF64], [VincentyF32]
//
// # Batch Operations
//
// Packed kernels take a right-hand side packed once into a [DotsPackedMatrix]:
//
//   - Packed dot products: [DotsPackedF64], [DotsPackedF32], [DotsPackedI8], [DotsPackedU8]
//   - Packed angular: [AngularsPackedF64], [AngularsPackedF32], [AngularsPackedI8],
//     [AngularsPackedU8]
//   - Packed Euclidean: [EuclideansPackedF64], [EuclideansPackedF32], [EuclideansPackedI8],
//     [EuclideansPackedU8]
//   - Packed binary distances: [HammingsPackedU1], [JaccardsPackedU1]
//   - Late interaction: [MaxSimF32] over a [MaxSimPackedMatrix]
//
// Symmetric kernels compare every pair within one set and write only the upper triangle.
//
//   - Symmetric dot: [DotsSymmetricF64], [DotsSymmetricF32], [DotsSymmetricI8], [DotsSymmetricU8]
//   - Symmetric angular: [AngularsSymmetricF64], [AngularsSymmetricF32], [AngularsSymmetricI8],
//     [AngularsSymmetricU8]
//   - Symmetric Euclidean: [EuclideansSymmetricF64], [EuclideansSymmetricF32],
//     [EuclideansSymmetricI8], [EuclideansSymmetricU8]
//   - Symmetric binary distances: [HammingsSymmetricU1], [JaccardsSymmetricU1]
//
// # Output Types
//
// Outputs are widened to prevent overflow: float32 inputs produce float64, int8 inputs produce
// int32 or float32, and uint8 inputs produce uint32 or float32.
//
// # Binary Vectors
//
// Binary vectors pack 8 dimensions per byte, least significant bit first. [DimensionsPerValue] and
// [DimensionsToValues] convert a dimension count into stored values for a dtype name such as "u1".
//
// # Threads
//
// [ConfigureThread] pins the goroutine to an OS thread and configures its SIMD state. A
// [WorkerPool] keeps such threads alive, and the WithPool variants split a batch across them.
//
// # Errors
//
// Every function panics on invalid inputs such as mismatched lengths or short slices. Scalar
// functions return zero for empty inputs.
//
// File: golang/numkong.go
// Author: Ash Vardanian
package numkong

// #cgo CFLAGS: -O3 -I../include -DNUMKONG_NATIVE_F16=0 -DNUMKONG_NATIVE_BF16=0
// #cgo LDFLAGS: -O3 -L. -lm
import "C"
