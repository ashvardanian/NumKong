package numkong

/*
#cgo CFLAGS: -O3 -I../include
#cgo LDFLAGS: -O3 -L. -lm
#define NK_NATIVE_F16 (0)
#define NK_NATIVE_BF16 (0)
#include "numkong/numkong.h"
*/
import "C"
import "unsafe"

// PackedMatrix holds a pre-packed right-hand-side matrix for batch operations.
// Construct with NewPackedMatrixF64, NewPackedMatrixF32, NewPackedMatrixI8,
// NewPackedMatrixU8, or NewPackedMatrixU1. The struct carries the packed buffer,
// dimensions, and scalar type so compute functions can validate.
type PackedMatrix struct {
	data  []byte
	width int
	depth int
	dtype string // "f64", "f32", "i8", "u8", "u1"
}

func (p PackedMatrix) Width() int    { return p.width }
func (p PackedMatrix) Depth() int    { return p.depth }
func (p PackedMatrix) Dtype() string { return p.dtype }
func (p PackedMatrix) Bytes() []byte { return p.data }

// Shape reads the packed matrix dimensions (width, depth) back from the buffer's
// self-describing header via nk_dots_packed_shape_<dtype>.
func (p PackedMatrix) Shape() (width, depth int) {
	if len(p.data) == 0 {
		return 0, 0
	}
	var w, d C.nk_size_t
	blob := unsafe.Pointer(&p.data[0])
	switch p.dtype {
	case "f64":
		C.nk_dots_packed_shape_f64(blob, &w, &d)
	case "f32":
		C.nk_dots_packed_shape_f32(blob, &w, &d)
	case "i8":
		C.nk_dots_packed_shape_i8(blob, &w, &d)
	case "u8":
		C.nk_dots_packed_shape_u8(blob, &w, &d)
	case "u1":
		C.nk_dots_packed_shape_u1(blob, &w, &d)
	}
	return int(w), int(d)
}

// NewPackedMatrixF64 packs a B matrix (width × depth f64 values) for batch operations.
// b must have capacity >= width × depth.
func NewPackedMatrixF64(b []float64, width, depth int) PackedMatrix {
	if len(b) < width*depth {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_f64(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_f64(
		(*C.nk_f64_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(depth*8),
		unsafe.Pointer(&data[0]))
	return PackedMatrix{data: data, width: width, depth: depth, dtype: "f64"}
}

// NewPackedMatrixF32 packs a B matrix (width × depth f32 values) for batch operations.
// b must have capacity >= width × depth.
func NewPackedMatrixF32(b []float32, width, depth int) PackedMatrix {
	if len(b) < width*depth {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_f32(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_f32(
		(*C.nk_f32_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		unsafe.Pointer(&data[0]))
	return PackedMatrix{data: data, width: width, depth: depth, dtype: "f32"}
}

// NewPackedMatrixI8 packs a B matrix (width × depth i8 values) for batch operations.
// b must have capacity >= width × depth.
func NewPackedMatrixI8(b []int8, width, depth int) PackedMatrix {
	if len(b) < width*depth {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_i8(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_i8(
		(*C.nk_i8_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(depth),
		unsafe.Pointer(&data[0]))
	return PackedMatrix{data: data, width: width, depth: depth, dtype: "i8"}
}

// NewPackedMatrixU8 packs a B matrix (width × depth u8 values) for batch operations.
// b must have capacity >= width × depth.
func NewPackedMatrixU8(b []uint8, width, depth int) PackedMatrix {
	if len(b) < width*depth {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_u8(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_u8(
		(*C.nk_u8_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(depth),
		unsafe.Pointer(&data[0]))
	return PackedMatrix{data: data, width: width, depth: depth, dtype: "u8"}
}

// NewPackedMatrixU1 packs a B matrix (width vectors, depth bits each) for batch operations.
// b is packed bits (ceil(depth/8) bytes per vector), depth is number of bits.
// b must have capacity >= width × ceil(depth/8).
func NewPackedMatrixU1(b []byte, width, depth int) PackedMatrix {
	bytesPerVec := (depth + 7) / 8
	if len(b) < width*bytesPerVec {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_u1(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_u1(
		(*C.nk_u1x8_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(bytesPerVec),
		unsafe.Pointer(&data[0]))
	return PackedMatrix{data: data, width: width, depth: depth, dtype: "u1"}
}
