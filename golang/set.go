// File: golang/set.go
// Author: Ash Vardanian

package numkong

/*
#cgo CFLAGS: -O3 -I../include
#cgo LDFLAGS: -O3 -L. -lm
#define NK_NATIVE_F16 (0)
#define NK_NATIVE_BF16 (0)
#include "numkong/numkong.h"
*/
import "C"

// HammingU8 counts the positions at which two uint8 vectors differ.
// Both slices must have the same length.
func HammingU8(a, b []uint8) uint32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_u32_t
	C.nk_hamming_u8((*C.nk_u8_t)(&a[0]), (*C.nk_u8_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return uint32(result)
}

// HammingU1 computes the Hamming distance between two binary vectors of depth dimensions, a
// multiple of 8, and both slices hold at least [DimensionsToValues]("u1", depth) bytes.
func HammingU1(a, b []byte, depth int) uint32 {
	validateDimensions("u1", depth)
	values := DimensionsToValues("u1", depth)
	if len(a) < values || len(b) < values {
		panic("slices too short for the given number of bits")
	}
	if depth == 0 {
		return 0
	}
	var result C.nk_u32_t
	C.nk_hamming_u1((*C.nk_u1x8_t)(&a[0]), (*C.nk_u1x8_t)(&b[0]), C.nk_size_t(depth), &result)
	return uint32(result)
}

// JaccardU1 computes the Jaccard distance between two binary vectors of depth dimensions, a
// multiple of 8, and both slices hold at least [DimensionsToValues]("u1", depth) bytes.
func JaccardU1(a, b []byte, depth int) float32 {
	validateDimensions("u1", depth)
	values := DimensionsToValues("u1", depth)
	if len(a) < values || len(b) < values {
		panic("slices too short for the given number of bits")
	}
	if depth == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_jaccard_u1((*C.nk_u1x8_t)(&a[0]), (*C.nk_u1x8_t)(&b[0]), C.nk_size_t(depth), &result)
	return float32(result)
}

// JaccardU16 computes the Jaccard distance between two uint16 set-hash vectors.
// Both slices must have the same length.
func JaccardU16(a, b []uint16) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_jaccard_u16((*C.nk_u16_t)(&a[0]), (*C.nk_u16_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}

// JaccardU32 computes the Jaccard distance between two uint32 set-hash vectors.
// Both slices must have the same length.
func JaccardU32(a, b []uint32) float32 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f32_t
	C.nk_jaccard_u32((*C.nk_u32_t)(&a[0]), (*C.nk_u32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float32(result)
}
