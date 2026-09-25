// Dot products of float and integer slices, backed by the dispatched nk_dot family.
//
// File: golang/dot.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"

// DotF64 computes the dot product of two float64 vectors.
// Both slices must have the same length.
func DotF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_dot_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// DotF32 computes the dot product of two float32 vectors.
// Both slices must have the same length.
func DotF32(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_dot_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// DotI8 computes the dot product of two int8 vectors.
// Both slices must have the same length.
func DotI8(a, b []int8) int32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_i32_t
	C.nk_dot_i8((*C.nk_i8_t)(&a[0]), (*C.nk_i8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return int32(result)
}

// DotU8 computes the dot product of two uint8 vectors.
// Both slices must have the same length.
func DotU8(a, b []uint8) uint32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_u32_t
	C.nk_dot_u8((*C.nk_u8_t)(&a[0]), (*C.nk_u8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return uint32(result)
}
