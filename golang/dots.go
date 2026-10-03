// Dot products of many rows against a matrix packed once, or within one set of rows.
//
// File: golang/dots.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"
import "unsafe"

// region Packing

// DotsPackedMatrix holds a matrix packed once for the Dots, Angulars, Euclideans, Hammings and
// Jaccards Packed kernels. Construct it with the constructor of the element type, such as
// [NewDotsPackedMatrixF32].
type DotsPackedMatrix struct {
	data    []byte
	columns int
	depth   int
	dtype   string // "f64", "f32", "i8", "u8", "u1"
}

// Columns returns the number of packed vectors.
func (p DotsPackedMatrix) Columns() int { return p.columns }

// Depth returns the number of dimensions per packed vector.
func (p DotsPackedMatrix) Depth() int { return p.depth }

// DType returns the element type the matrix was packed from: "f64", "f32", "i8", "u8" or "u1".
func (p DotsPackedMatrix) DType() string { return p.dtype }

// Bytes returns the packed buffer.
func (p DotsPackedMatrix) Bytes() []byte { return p.data }

// Shape reads the columns and depth back from the header of the packed buffer.
func (p DotsPackedMatrix) Shape() (columns, depth int) {
	if len(p.data) == 0 {
		return 0, 0
	}
	var w, d C.nk_size_t
	blob := unsafe.Pointer(&p.data[0])
	switch p.dtype {
	case "f64":
		check(C.nk_dots_packed_shape_f64_best(blob, &w, &d, C.nk_cap_cpus_k, nil))
	case "f32":
		check(C.nk_dots_packed_shape_f32_best(blob, &w, &d, C.nk_cap_cpus_k, nil))
	case "i8":
		check(C.nk_dots_packed_shape_i8_best(blob, &w, &d, C.nk_cap_cpus_k, nil))
	case "u8":
		check(C.nk_dots_packed_shape_u8_best(blob, &w, &d, C.nk_cap_cpus_k, nil))
	case "u1":
		check(C.nk_dots_packed_shape_u1_best(blob, &w, &d, C.nk_cap_cpus_k, nil))
	}
	return int(w), int(d)
}

// NewDotsPackedMatrixF64 packs columns float64 vectors of depth dimensions for the Packed kernels.
// The slice b holds at least columns * depth values.
func NewDotsPackedMatrixF64(b []float64, columns, depth int) DotsPackedMatrix {
	if len(b) < columns*depth {
		panic("input slice too short for the given columns and depth")
	}
	var size C.nk_size_t
	check(C.nk_dots_pack_size_f64_best(C.nk_size_t(columns), C.nk_size_t(depth), C.nk_cap_cpus_k, &size))
	data := make([]byte, size)
	check(C.nk_dots_pack_f64_best(
		(*C.nk_f64_t)(&b[0]),
		C.nk_size_t(columns), C.nk_size_t(depth),
		C.nk_size_t(depth*8),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(columns), C.nk_cap_cpus_k, nil))
	return DotsPackedMatrix{data: data, columns: columns, depth: depth, dtype: "f64"}
}

// NewDotsPackedMatrixF32 packs columns float32 vectors of depth dimensions for the Packed kernels.
// The slice b holds at least columns * depth values.
func NewDotsPackedMatrixF32(b []float32, columns, depth int) DotsPackedMatrix {
	if len(b) < columns*depth {
		panic("input slice too short for the given columns and depth")
	}
	var size C.nk_size_t
	check(C.nk_dots_pack_size_f32_best(C.nk_size_t(columns), C.nk_size_t(depth), C.nk_cap_cpus_k, &size))
	data := make([]byte, size)
	check(C.nk_dots_pack_f32_best(
		(*C.nk_f32_t)(&b[0]),
		C.nk_size_t(columns), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(columns), C.nk_cap_cpus_k, nil))
	return DotsPackedMatrix{data: data, columns: columns, depth: depth, dtype: "f32"}
}

// NewDotsPackedMatrixI8 packs columns int8 vectors of depth dimensions for the Packed kernels. The
// slice b holds at least columns * depth values.
func NewDotsPackedMatrixI8(b []int8, columns, depth int) DotsPackedMatrix {
	if len(b) < columns*depth {
		panic("input slice too short for the given columns and depth")
	}
	var size C.nk_size_t
	check(C.nk_dots_pack_size_i8_best(C.nk_size_t(columns), C.nk_size_t(depth), C.nk_cap_cpus_k, &size))
	data := make([]byte, size)
	check(C.nk_dots_pack_i8_best(
		(*C.nk_i8_t)(&b[0]),
		C.nk_size_t(columns), C.nk_size_t(depth),
		C.nk_size_t(depth),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(columns), C.nk_cap_cpus_k, nil))
	return DotsPackedMatrix{data: data, columns: columns, depth: depth, dtype: "i8"}
}

// NewDotsPackedMatrixU8 packs columns uint8 vectors of depth dimensions for the Packed kernels. The
// slice b holds at least columns * depth values.
func NewDotsPackedMatrixU8(b []uint8, columns, depth int) DotsPackedMatrix {
	if len(b) < columns*depth {
		panic("input slice too short for the given columns and depth")
	}
	var size C.nk_size_t
	check(C.nk_dots_pack_size_u8_best(C.nk_size_t(columns), C.nk_size_t(depth), C.nk_cap_cpus_k, &size))
	data := make([]byte, size)
	check(C.nk_dots_pack_u8_best(
		(*C.nk_u8_t)(&b[0]),
		C.nk_size_t(columns), C.nk_size_t(depth),
		C.nk_size_t(depth),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(columns), C.nk_cap_cpus_k, nil))
	return DotsPackedMatrix{data: data, columns: columns, depth: depth, dtype: "u8"}
}

// NewDotsPackedMatrixU1 packs columns binary vectors of depth dimensions for the Packed kernels.
// The depth is a multiple of 8, and b holds at least columns * [DimensionsToValues]("u1", depth)
// bytes.
func NewDotsPackedMatrixU1(b []byte, columns, depth int) DotsPackedMatrix {
	validateDimensions("u1", depth)
	bytesPerVec := DimensionsToValues("u1", depth)
	if len(b) < columns*bytesPerVec {
		panic("input slice too short for the given columns and depth")
	}
	var size C.nk_size_t
	check(C.nk_dots_pack_size_u1_best(C.nk_size_t(columns), C.nk_size_t(depth), C.nk_cap_cpus_k, &size))
	data := make([]byte, size)
	check(C.nk_dots_pack_u1_best(
		(*C.nk_u1x8_t)(&b[0]),
		C.nk_size_t(columns), C.nk_size_t(depth),
		C.nk_size_t(bytesPerVec),
		unsafe.Pointer(&data[0]), C.nk_size_t(0), C.nk_size_t(columns), C.nk_cap_cpus_k, nil))
	return DotsPackedMatrix{data: data, columns: columns, depth: depth, dtype: "u1"}
}

// endregion

// region Packed Dots

// DotsPackedF64 computes the dot product of each of rows float64 rows of a with every packed row
// of b. Each row has [DotsPackedMatrix.Depth] dimensions, and c holds at least rows *
// [DotsPackedMatrix.Columns] entries.
func DotsPackedF64(a []float64, b DotsPackedMatrix, c []float64, rows int) {
	if b.DType() != "f64" {
		panic("DotsPackedMatrix dtype must be f64")
	}
	if len(a) < rows*b.depth {
		panic("input slice too short for the given rows and depth")
	}
	if len(c) < rows*b.columns {
		panic("output slice too short for the given rows and columns")
	}
	check(C.nk_dots_packed_f64_best(
		(*C.nk_f64_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&c[0]),
		C.nk_size_t(rows), C.nk_size_t(b.columns), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*8),
		C.nk_size_t(b.columns*8), C.nk_cap_cpus_k, nil))
}

// DotsPackedF32 computes the dot product of each of rows float32 rows of a with every packed row
// of b. Each row has [DotsPackedMatrix.Depth] dimensions, and c holds at least rows *
// [DotsPackedMatrix.Columns] entries.
func DotsPackedF32(a []float32, b DotsPackedMatrix, c []float64, rows int) {
	if b.DType() != "f32" {
		panic("DotsPackedMatrix dtype must be f32")
	}
	if len(a) < rows*b.depth {
		panic("input slice too short for the given rows and depth")
	}
	if len(c) < rows*b.columns {
		panic("output slice too short for the given rows and columns")
	}
	check(C.nk_dots_packed_f32_best(
		(*C.nk_f32_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&c[0]),
		C.nk_size_t(rows), C.nk_size_t(b.columns), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*4),
		C.nk_size_t(b.columns*8), C.nk_cap_cpus_k, nil))
}

// DotsPackedI8 computes the dot product of each of rows int8 rows of a with every packed row of
// b. Each row has [DotsPackedMatrix.Depth] dimensions, and c holds at least rows *
// [DotsPackedMatrix.Columns] entries.
func DotsPackedI8(a []int8, b DotsPackedMatrix, c []int32, rows int) {
	if b.DType() != "i8" {
		panic("DotsPackedMatrix dtype must be i8")
	}
	if len(a) < rows*b.depth {
		panic("input slice too short for the given rows and depth")
	}
	if len(c) < rows*b.columns {
		panic("output slice too short for the given rows and columns")
	}
	check(C.nk_dots_packed_i8_best(
		(*C.nk_i8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_i32_t)(&c[0]),
		C.nk_size_t(rows), C.nk_size_t(b.columns), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.columns*4), C.nk_cap_cpus_k, nil))
}

// DotsPackedU8 computes the dot product of each of rows uint8 rows of a with every packed row of
// b. Each row has [DotsPackedMatrix.Depth] dimensions, and c holds at least rows *
// [DotsPackedMatrix.Columns] entries.
func DotsPackedU8(a []uint8, b DotsPackedMatrix, c []uint32, rows int) {
	if b.DType() != "u8" {
		panic("DotsPackedMatrix dtype must be u8")
	}
	if len(a) < rows*b.depth {
		panic("input slice too short for the given rows and depth")
	}
	if len(c) < rows*b.columns {
		panic("output slice too short for the given rows and columns")
	}
	check(C.nk_dots_packed_u8_best(
		(*C.nk_u8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_u32_t)(&c[0]),
		C.nk_size_t(rows), C.nk_size_t(b.columns), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.columns*4), C.nk_cap_cpus_k, nil))
}

// endregion

// region Symmetric Dots

// DotsSymmetricF64 computes the dot product between every pair of nVectors float64 vectors of depth
// dimensions. The vectors are stored row-major, and only entries with row <= column are written
// into result, which holds at least nVectors * nVectors entries.
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
	check(C.nk_dots_symmetric_f64_best(
		(*C.nk_f64_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*8),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount), C.nk_cap_cpus_k, nil))
}

// DotsSymmetricF32 computes the dot product between every pair of nVectors float32 vectors of depth
// dimensions. The vectors are stored row-major, and only entries with row <= column are written
// into result, which holds at least nVectors * nVectors entries.
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
	check(C.nk_dots_symmetric_f32_best(
		(*C.nk_f32_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount), C.nk_cap_cpus_k, nil))
}

// DotsSymmetricI8 computes the dot product between every pair of nVectors int8 vectors of depth
// dimensions. The vectors are stored row-major, and only entries with row <= column are written
// into result, which holds at least nVectors * nVectors entries.
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
	check(C.nk_dots_symmetric_i8_best(
		(*C.nk_i8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_i32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount), C.nk_cap_cpus_k, nil))
}

// DotsSymmetricU8 computes the dot product between every pair of nVectors uint8 vectors of depth
// dimensions. The vectors are stored row-major, and only entries with row <= column are written
// into result, which holds at least nVectors * nVectors entries.
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
	check(C.nk_dots_symmetric_u8_best(
		(*C.nk_u8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_u32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount), C.nk_cap_cpus_k, nil))
}

// endregion
