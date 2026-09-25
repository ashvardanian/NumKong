// DType names such as "u1", and how many logical dimensions each stores per value.
//
// File: golang/types.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"

// nativeDType maps a [DotsPackedMatrix.DType] name such as "u1" onto the C enumeration.
func nativeDType(dtype string) C.nk_dtype_t {
	switch dtype {
	case "f64":
		return C.nk_f64_k
	case "f32":
		return C.nk_f32_k
	case "i8":
		return C.nk_i8_k
	case "u8":
		return C.nk_u8_k
	case "u1":
		return C.nk_u1_k
	}
	panic("unknown dtype " + dtype)
}

// DimensionsPerValue returns how many logical dimensions of dtype share one stored value, such as 8
// for "u1". It panics on any name other than "f64", "f32", "i8", "u8" and "u1".
func DimensionsPerValue(dtype string) int {
	return int(C.nk_dimensions_per_value(nativeDType(dtype)))
}

// DimensionsToValues returns how many stored values hold dimensions logical dimensions of dtype.
// The dimensions must be a multiple of [DimensionsPerValue] for dtype.
func DimensionsToValues(dtype string, dimensions int) int {
	return dimensions / DimensionsPerValue(dtype)
}

// validateDimensions panics unless dimensions is a multiple of [DimensionsPerValue] for dtype.
func validateDimensions(dtype string, dimensions int) {
	if dimensions%DimensionsPerValue(dtype) != 0 {
		panic("dimension count must be a multiple of DimensionsPerValue(dtype)")
	}
}
