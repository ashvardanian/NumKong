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

// region Packing

// DotsPackedMatrix holds a matrix packed once for the Dots, Angulars, Euclideans, Hammings and Jaccards Packed kernels.
// Construct it with the constructor of the element type, such as [NewDotsPackedMatrixF32].
type DotsPackedMatrix struct {
	data  []byte
	width int
	depth int
	dtype string // "f64", "f32", "i8", "u8", "u1"
}

// Width returns the number of packed vectors.
func (p DotsPackedMatrix) Width() int { return p.width }

// Depth returns the number of dimensions per packed vector.
func (p DotsPackedMatrix) Depth() int { return p.depth }

// DType returns the element type the matrix was packed from: "f64", "f32", "i8", "u8" or "u1".
func (p DotsPackedMatrix) DType() string { return p.dtype }

// Bytes returns the packed buffer.
func (p DotsPackedMatrix) Bytes() []byte { return p.data }

// Shape reads the width and depth back from the header of the packed buffer.
func (p DotsPackedMatrix) Shape() (width, depth int) {
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

// NewDotsPackedMatrixF64 packs width float64 vectors of depth dimensions for the Packed kernels.
// The slice b holds at least width * depth values.
func NewDotsPackedMatrixF64(b []float64, width, depth int) DotsPackedMatrix {
	if len(b) < width*depth {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_f64(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_f64(
		(*C.nk_f64_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(depth*8),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(width))
	return DotsPackedMatrix{data: data, width: width, depth: depth, dtype: "f64"}
}

// NewDotsPackedMatrixF32 packs width float32 vectors of depth dimensions for the Packed kernels.
// The slice b holds at least width * depth values.
func NewDotsPackedMatrixF32(b []float32, width, depth int) DotsPackedMatrix {
	if len(b) < width*depth {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_f32(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_f32(
		(*C.nk_f32_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(width))
	return DotsPackedMatrix{data: data, width: width, depth: depth, dtype: "f32"}
}

// NewDotsPackedMatrixI8 packs width int8 vectors of depth dimensions for the Packed kernels.
// The slice b holds at least width * depth values.
func NewDotsPackedMatrixI8(b []int8, width, depth int) DotsPackedMatrix {
	if len(b) < width*depth {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_i8(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_i8(
		(*C.nk_i8_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(depth),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(width))
	return DotsPackedMatrix{data: data, width: width, depth: depth, dtype: "i8"}
}

// NewDotsPackedMatrixU8 packs width uint8 vectors of depth dimensions for the Packed kernels.
// The slice b holds at least width * depth values.
func NewDotsPackedMatrixU8(b []uint8, width, depth int) DotsPackedMatrix {
	if len(b) < width*depth {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_u8(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_u8(
		(*C.nk_u8_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(depth),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(width))
	return DotsPackedMatrix{data: data, width: width, depth: depth, dtype: "u8"}
}

// NewDotsPackedMatrixU1 packs width binary vectors of depth dimensions for the Packed kernels.
// The depth is a multiple of 8, and b holds at least width * [DimensionsToValues]("u1", depth) bytes.
func NewDotsPackedMatrixU1(b []byte, width, depth int) DotsPackedMatrix {
	validateDimensions("u1", depth)
	bytesPerVec := DimensionsToValues("u1", depth)
	if len(b) < width*bytesPerVec {
		panic("input slice too short for the given width and depth")
	}
	size := int(C.nk_dots_pack_size_u1(C.nk_size_t(width), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_dots_pack_u1(
		(*C.nk_u1x8_t)(&b[0]),
		C.nk_size_t(width), C.nk_size_t(depth),
		C.nk_size_t(bytesPerVec),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(width))
	return DotsPackedMatrix{data: data, width: width, depth: depth, dtype: "u1"}
}

// endregion

// region Packed Batch Operations (Dots)

// DotsPackedF64 computes the dot product of each of height float64 rows of a with every packed row of b.
// Each row has b.Depth() dimensions, and c holds at least height * b.Width() entries.
func DotsPackedF64(a []float64, b DotsPackedMatrix, c []float64, height int) {
	if b.DType() != "f64" {
		panic("DotsPackedMatrix dtype must be f64")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(c) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_dots_packed_f64(
		(*C.nk_f64_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&c[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*8),
		C.nk_size_t(b.width*8))
}

// DotsPackedF32 computes the dot product of each of height float32 rows of a with every packed row of b.
// Each row has b.Depth() dimensions, and c holds at least height * b.Width() entries.
func DotsPackedF32(a []float32, b DotsPackedMatrix, c []float64, height int) {
	if b.DType() != "f32" {
		panic("DotsPackedMatrix dtype must be f32")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(c) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_dots_packed_f32(
		(*C.nk_f32_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&c[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*4),
		C.nk_size_t(b.width*8))
}

// DotsPackedI8 computes the dot product of each of height int8 rows of a with every packed row of b.
// Each row has b.Depth() dimensions, and c holds at least height * b.Width() entries.
func DotsPackedI8(a []int8, b DotsPackedMatrix, c []int32, height int) {
	if b.DType() != "i8" {
		panic("DotsPackedMatrix dtype must be i8")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(c) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_dots_packed_i8(
		(*C.nk_i8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_i32_t)(&c[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.width*4))
}

// DotsPackedU8 computes the dot product of each of height uint8 rows of a with every packed row of b.
// Each row has b.Depth() dimensions, and c holds at least height * b.Width() entries.
func DotsPackedU8(a []uint8, b DotsPackedMatrix, c []uint32, height int) {
	if b.DType() != "u8" {
		panic("DotsPackedMatrix dtype must be u8")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(c) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_dots_packed_u8(
		(*C.nk_u8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_u32_t)(&c[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.width*4))
}

// endregion

// region Symmetric Operations (Dots)

// DotsSymmetricF64 computes the dot product between every pair of nVectors float64 vectors of depth dimensions.
// The vectors are stored row-major, and only entries with row <= column are written into result, which holds at least nVectors * nVectors entries.
func DotsSymmetricF64(vectors []float64, nVectors, depth int, result []float64) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	dotsSymmetricF64(vectors, nVectors, depth, result, 0, nVectors)
}

func dotsSymmetricF64(vectors []float64, nVectors, depth int, result []float64, rowStart, rowCount int) {
	C.nk_dots_symmetric_f64(
		(*C.nk_f64_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*8),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// DotsSymmetricF32 computes the dot product between every pair of nVectors float32 vectors of depth dimensions.
// The vectors are stored row-major, and only entries with row <= column are written into result, which holds at least nVectors * nVectors entries.
func DotsSymmetricF32(vectors []float32, nVectors, depth int, result []float64) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	dotsSymmetricF32(vectors, nVectors, depth, result, 0, nVectors)
}

func dotsSymmetricF32(vectors []float32, nVectors, depth int, result []float64, rowStart, rowCount int) {
	C.nk_dots_symmetric_f32(
		(*C.nk_f32_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// DotsSymmetricI8 computes the dot product between every pair of nVectors int8 vectors of depth dimensions.
// The vectors are stored row-major, and only entries with row <= column are written into result, which holds at least nVectors * nVectors entries.
func DotsSymmetricI8(vectors []int8, nVectors, depth int, result []int32) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	dotsSymmetricI8(vectors, nVectors, depth, result, 0, nVectors)
}

func dotsSymmetricI8(vectors []int8, nVectors, depth int, result []int32, rowStart, rowCount int) {
	C.nk_dots_symmetric_i8(
		(*C.nk_i8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_i32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// DotsSymmetricU8 computes the dot product between every pair of nVectors uint8 vectors of depth dimensions.
// The vectors are stored row-major, and only entries with row <= column are written into result, which holds at least nVectors * nVectors entries.
func DotsSymmetricU8(vectors []uint8, nVectors, depth int, result []uint32) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	dotsSymmetricU8(vectors, nVectors, depth, result, 0, nVectors)
}

func dotsSymmetricU8(vectors []uint8, nVectors, depth int, result []uint32, rowStart, rowCount int) {
	C.nk_dots_symmetric_u8(
		(*C.nk_u8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_u32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// endregion
