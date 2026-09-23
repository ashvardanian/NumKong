package numkong

/*
#cgo CFLAGS: -O3 -I../include
#cgo LDFLAGS: -O3 -L. -lm
#define NK_NATIVE_F16 (0)
#define NK_NATIVE_BF16 (0)
#include "numkong/numkong.h"
*/
import "C"

// HammingU8 computes the Hamming distance between two uint8 vectors.
// Returns uint32 (number of positions where the values differ).
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

// HammingU1 computes the Hamming distance between two packed binary vectors.
// `depth` counts dimensions, a multiple of 8, the values per byte.
func HammingU1(a, b []byte, depth int) uint32 {
	validateDimensions(C.nk_u1_k, depth)
	values := dimensionsToValues(C.nk_u1_k, depth)
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

// JaccardU1 computes the Jaccard distance between two packed binary vectors.
// `depth` counts dimensions, a multiple of 8, the values per byte.
func JaccardU1(a, b []byte, depth int) float32 {
	validateDimensions(C.nk_u1_k, depth)
	values := dimensionsToValues(C.nk_u1_k, depth)
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
