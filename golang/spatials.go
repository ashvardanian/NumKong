// File: golang/spatials.go
// Author: Ash Vardanian

package numkong

/*
#cgo CFLAGS: -O3 -I../include
#cgo LDFLAGS: -O3 -L. -lm
#define NUMKONG_NATIVE_F16 (0)
#define NUMKONG_NATIVE_BF16 (0)
#include "numkong/numkong.h"
*/
import "C"
import "unsafe"

// region Packed Angulars / Euclideans

// AngularsPackedF64 computes the angular distance from each of height float64 rows of a to every
// packed row of b. Each row has [DotsPackedMatrix.Depth] dimensions, and result holds at least
// height * [DotsPackedMatrix.Width] entries.
func AngularsPackedF64(a []float64, b DotsPackedMatrix, result []float64, height int) {
	if b.DType() != "f64" {
		panic("DotsPackedMatrix dtype must be f64")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_angulars_packed_f64(
		(*C.nk_f64_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*8),
		C.nk_size_t(b.width*8))
}

// AngularsPackedF32 computes the angular distance from each of height float32 rows of a to every
// packed row of b. Each row has [DotsPackedMatrix.Depth] dimensions, and result holds at least
// height * [DotsPackedMatrix.Width] entries.
func AngularsPackedF32(a []float32, b DotsPackedMatrix, result []float64, height int) {
	if b.DType() != "f32" {
		panic("DotsPackedMatrix dtype must be f32")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_angulars_packed_f32(
		(*C.nk_f32_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*4),
		C.nk_size_t(b.width*8))
}

// AngularsPackedI8 computes the angular distance from each of height int8 rows of a to every packed
// row of b. Each row has [DotsPackedMatrix.Depth] dimensions, and result holds at least height *
// [DotsPackedMatrix.Width] entries.
func AngularsPackedI8(a []int8, b DotsPackedMatrix, result []float32, height int) {
	if b.DType() != "i8" {
		panic("DotsPackedMatrix dtype must be i8")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_angulars_packed_i8(
		(*C.nk_i8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.width*4))
}

// AngularsPackedU8 computes the angular distance from each of height uint8 rows of a to every
// packed row of b. Each row has [DotsPackedMatrix.Depth] dimensions, and result holds at least
// height * [DotsPackedMatrix.Width] entries.
func AngularsPackedU8(a []uint8, b DotsPackedMatrix, result []float32, height int) {
	if b.DType() != "u8" {
		panic("DotsPackedMatrix dtype must be u8")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_angulars_packed_u8(
		(*C.nk_u8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.width*4))
}

// EuclideansPackedF64 computes the Euclidean distance from each of height float64 rows of a to
// every packed row of b. Each row has [DotsPackedMatrix.Depth] dimensions, and result holds at
// least height * [DotsPackedMatrix.Width] entries.
func EuclideansPackedF64(a []float64, b DotsPackedMatrix, result []float64, height int) {
	if b.DType() != "f64" {
		panic("DotsPackedMatrix dtype must be f64")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_euclideans_packed_f64(
		(*C.nk_f64_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*8),
		C.nk_size_t(b.width*8))
}

// EuclideansPackedF32 computes the Euclidean distance from each of height float32 rows of a to
// every packed row of b. Each row has [DotsPackedMatrix.Depth] dimensions, and result holds at
// least height * [DotsPackedMatrix.Width] entries.
func EuclideansPackedF32(a []float32, b DotsPackedMatrix, result []float64, height int) {
	if b.DType() != "f32" {
		panic("DotsPackedMatrix dtype must be f32")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_euclideans_packed_f32(
		(*C.nk_f32_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth*4),
		C.nk_size_t(b.width*8))
}

// EuclideansPackedI8 computes the Euclidean distance from each of height int8 rows of a to every
// packed row of b. Each row has [DotsPackedMatrix.Depth] dimensions, and result holds at least
// height * [DotsPackedMatrix.Width] entries.
func EuclideansPackedI8(a []int8, b DotsPackedMatrix, result []float32, height int) {
	if b.DType() != "i8" {
		panic("DotsPackedMatrix dtype must be i8")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_euclideans_packed_i8(
		(*C.nk_i8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.width*4))
}

// EuclideansPackedU8 computes the Euclidean distance from each of height uint8 rows of a to every
// packed row of b. Each row has [DotsPackedMatrix.Depth] dimensions, and result holds at least
// height * [DotsPackedMatrix.Width] entries.
func EuclideansPackedU8(a []uint8, b DotsPackedMatrix, result []float32, height int) {
	if b.DType() != "u8" {
		panic("DotsPackedMatrix dtype must be u8")
	}
	if len(a) < height*b.depth {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*b.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_euclideans_packed_u8(
		(*C.nk_u8_t)(&a[0]),
		unsafe.Pointer(&b.data[0]),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(b.width), C.nk_size_t(b.depth),
		C.nk_size_t(b.depth),
		C.nk_size_t(b.width*4))
}

// endregion

// region Symmetric Angulars / Euclideans

// AngularsSymmetricF64 computes the angular distance between every pair of nVectors row-major
// float64 vectors of depth dimensions, writing only entries with row <= column into result, which
// holds at least nVectors * nVectors entries.
func AngularsSymmetricF64(vectors []float64, nVectors, depth int, result []float64) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	angularsSymmetricF64(vectors, nVectors, depth, result, 0, nVectors)
}

func angularsSymmetricF64(vectors []float64, nVectors, depth int, result []float64, rowStart, rowCount int) {
	C.nk_angulars_symmetric_f64(
		(*C.nk_f64_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*8),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// AngularsSymmetricF32 computes the angular distance between every pair of nVectors row-major
// float32 vectors of depth dimensions, writing only entries with row <= column into result, which
// holds at least nVectors * nVectors entries.
func AngularsSymmetricF32(vectors []float32, nVectors, depth int, result []float64) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	angularsSymmetricF32(vectors, nVectors, depth, result, 0, nVectors)
}

func angularsSymmetricF32(vectors []float32, nVectors, depth int, result []float64, rowStart, rowCount int) {
	C.nk_angulars_symmetric_f32(
		(*C.nk_f32_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// AngularsSymmetricI8 computes the angular distance between every pair of nVectors row-major int8
// vectors of depth dimensions, writing only entries with row <= column into result, which holds at
// least nVectors * nVectors entries.
func AngularsSymmetricI8(vectors []int8, nVectors, depth int, result []float32) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	angularsSymmetricI8(vectors, nVectors, depth, result, 0, nVectors)
}

func angularsSymmetricI8(vectors []int8, nVectors, depth int, result []float32, rowStart, rowCount int) {
	C.nk_angulars_symmetric_i8(
		(*C.nk_i8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// AngularsSymmetricU8 computes the angular distance between every pair of nVectors row-major uint8
// vectors of depth dimensions, writing only entries with row <= column into result, which holds at
// least nVectors * nVectors entries.
func AngularsSymmetricU8(vectors []uint8, nVectors, depth int, result []float32) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	angularsSymmetricU8(vectors, nVectors, depth, result, 0, nVectors)
}

func angularsSymmetricU8(vectors []uint8, nVectors, depth int, result []float32, rowStart, rowCount int) {
	C.nk_angulars_symmetric_u8(
		(*C.nk_u8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// EuclideansSymmetricF64 computes the Euclidean distance between every pair of nVectors row-major
// float64 vectors of depth dimensions, writing only entries with row <= column into result, which
// holds at least nVectors * nVectors entries.
func EuclideansSymmetricF64(vectors []float64, nVectors, depth int, result []float64) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	euclideansSymmetricF64(vectors, nVectors, depth, result, 0, nVectors)
}

func euclideansSymmetricF64(vectors []float64, nVectors, depth int, result []float64, rowStart, rowCount int) {
	C.nk_euclideans_symmetric_f64(
		(*C.nk_f64_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*8),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// EuclideansSymmetricF32 computes the Euclidean distance between every pair of nVectors row-major
// float32 vectors of depth dimensions, writing only entries with row <= column into result, which
// holds at least nVectors * nVectors entries.
func EuclideansSymmetricF32(vectors []float32, nVectors, depth int, result []float64) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	euclideansSymmetricF32(vectors, nVectors, depth, result, 0, nVectors)
}

func euclideansSymmetricF32(vectors []float32, nVectors, depth int, result []float64, rowStart, rowCount int) {
	C.nk_euclideans_symmetric_f32(
		(*C.nk_f32_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		(*C.nk_f64_t)(&result[0]),
		C.nk_size_t(nVectors*8),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// EuclideansSymmetricI8 computes the Euclidean distance between every pair of nVectors row-major
// int8 vectors of depth dimensions, writing only entries with row <= column into result, which
// holds at least nVectors * nVectors entries.
func EuclideansSymmetricI8(vectors []int8, nVectors, depth int, result []float32) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	euclideansSymmetricI8(vectors, nVectors, depth, result, 0, nVectors)
}

func euclideansSymmetricI8(vectors []int8, nVectors, depth int, result []float32, rowStart, rowCount int) {
	C.nk_euclideans_symmetric_i8(
		(*C.nk_i8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// EuclideansSymmetricU8 computes the Euclidean distance between every pair of nVectors row-major
// uint8 vectors of depth dimensions, writing only entries with row <= column into result, which
// holds at least nVectors * nVectors entries.
func EuclideansSymmetricU8(vectors []uint8, nVectors, depth int, result []float32) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	euclideansSymmetricU8(vectors, nVectors, depth, result, 0, nVectors)
}

func euclideansSymmetricU8(vectors []uint8, nVectors, depth int, result []float32, rowStart, rowCount int) {
	C.nk_euclideans_symmetric_u8(
		(*C.nk_u8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(depth),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// endregion
