// Angular, Euclidean and squared Euclidean distances of float and integer slices.
//
// File: golang/spatial.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"

// region Angular Cosine Distance

// AngularF64 computes the angular distance between two float64 vectors.
// Both slices must have the same length.
func AngularF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_angular_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// AngularF32 computes the angular distance between two float32 vectors.
// Both slices must have the same length.
func AngularF32(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_angular_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// AngularI8 computes the angular distance between two int8 vectors.
// Both slices must have the same length.
func AngularI8(a, b []int8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_angular_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// AngularU8 computes the angular distance between two uint8 vectors.
// Both slices must have the same length.
func AngularU8(a, b []uint8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_angular_u8((*C.nk_u8_t)(&a[0]), (*C.nk_u8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// endregion

// region Euclidean L2 Distance

// EuclideanF64 computes the Euclidean distance between two float64 vectors.
// Both slices must have the same length.
func EuclideanF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_euclidean_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// EuclideanF32 computes the Euclidean distance between two float32 vectors.
// Both slices must have the same length.
func EuclideanF32(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_euclidean_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// EuclideanI8 computes the Euclidean distance between two int8 vectors.
// Both slices must have the same length.
func EuclideanI8(a, b []int8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_euclidean_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// EuclideanU8 computes the Euclidean distance between two uint8 vectors.
// Both slices must have the same length.
func EuclideanU8(a, b []uint8) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_euclidean_u8((*C.nk_u8_t)(&a[0]), (*C.nk_u8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// endregion

// region Squared Euclidean L2sq Distance

// SqEuclideanF64 computes the squared Euclidean distance between two float64 vectors.
// Both slices must have the same length.
func SqEuclideanF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_sqeuclidean_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// SqEuclideanF32 computes the squared Euclidean distance between two float32 vectors.
// Both slices must have the same length.
func SqEuclideanF32(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_sqeuclidean_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// SqEuclideanI8 computes the squared Euclidean distance between two int8 vectors.
// Both slices must have the same length.
func SqEuclideanI8(a, b []int8) uint32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_u32_t
	C.nk_sqeuclidean_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return uint32(result)
}

// SqEuclideanU8 computes the squared Euclidean distance between two uint8 vectors.
// Both slices must have the same length.
func SqEuclideanU8(a, b []uint8) uint32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_u32_t
	C.nk_sqeuclidean_u8((*C.nk_u8_t)(&a[0]), (*C.nk_u8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return uint32(result)
}

// endregion
