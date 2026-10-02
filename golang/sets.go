// Hamming and Jaccard distances of many bit vectors, against a packed matrix or within one set.
//
// File: golang/sets.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"
import "unsafe"

// HammingsPackedU1 computes the Hamming distance from each of rows binary vectors to every packed
// query. Each vector has [DotsPackedMatrix.Depth] dimensions, and result holds at least rows *
// [DotsPackedMatrix.Columns] entries.
func HammingsPackedU1(vectors []byte, query DotsPackedMatrix, result []uint32, rows int) {
	if query.DType() != "u1" {
		panic("DotsPackedMatrix dtype must be u1")
	}
	bytesPerVec := DimensionsToValues("u1", query.depth)
	if len(vectors) < rows*bytesPerVec {
		panic("input slice too short for the given rows and depth")
	}
	if len(result) < rows*query.columns {
		panic("output slice too short for the given rows and columns")
	}
	check(C.nk_hammings_packed_u1_best(
		(*C.nk_u1x8_t)(&vectors[0]),
		unsafe.Pointer(&query.data[0]),
		(*C.nk_u32_t)(&result[0]),
		C.nk_size_t(rows), C.nk_size_t(query.columns), C.nk_size_t(query.depth),
		C.nk_size_t(bytesPerVec),
		C.nk_size_t(query.columns*4), capabilities(), nil))
}

// HammingsSymmetricU1 computes the Hamming distance between every pair of nVectors binary vectors
// of depth dimensions. The depth is a multiple of 8, and only entries with row <= column are
// written into result, which holds at least nVectors * nVectors entries.
func HammingsSymmetricU1(vectors []byte, nVectors, depth int, result []uint32) {
	validateDimensions("u1", depth)
	bytesPerVec := DimensionsToValues("u1", depth)
	if len(vectors) < nVectors*bytesPerVec {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	hammingsSymmetricU1(vectors, nVectors, depth, result, 0, nVectors)
}

func hammingsSymmetricU1(vectors []byte, nVectors, depth int, result []uint32, rowStart, rowCount int) {
	bytesPerVec := DimensionsToValues("u1", depth)
	check(C.nk_hammings_symmetric_u1_best(
		(*C.nk_u1x8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(bytesPerVec),
		(*C.nk_u32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount), capabilities(), nil))
}

// JaccardsPackedU1 computes the Jaccard distance from each of rows binary vectors to every packed
// query. Each vector has [DotsPackedMatrix.Depth] dimensions, and result holds at least rows *
// [DotsPackedMatrix.Columns] entries.
func JaccardsPackedU1(vectors []byte, query DotsPackedMatrix, result []float32, rows int) {
	if query.DType() != "u1" {
		panic("DotsPackedMatrix dtype must be u1")
	}
	bytesPerVec := DimensionsToValues("u1", query.depth)
	if len(vectors) < rows*bytesPerVec {
		panic("input slice too short for the given rows and depth")
	}
	if len(result) < rows*query.columns {
		panic("output slice too short for the given rows and columns")
	}
	check(C.nk_jaccards_packed_u1_best(
		(*C.nk_u1x8_t)(&vectors[0]),
		unsafe.Pointer(&query.data[0]),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(rows), C.nk_size_t(query.columns), C.nk_size_t(query.depth),
		C.nk_size_t(bytesPerVec),
		C.nk_size_t(query.columns*4), capabilities(), nil))
}

// JaccardsSymmetricU1 computes the Jaccard distance between every pair of nVectors binary vectors
// of depth dimensions. The depth is a multiple of 8, and only entries with row <= column are
// written into result, which holds at least nVectors * nVectors entries.
func JaccardsSymmetricU1(vectors []byte, nVectors, depth int, result []float32) {
	validateDimensions("u1", depth)
	bytesPerVec := DimensionsToValues("u1", depth)
	if len(vectors) < nVectors*bytesPerVec {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	jaccardsSymmetricU1(vectors, nVectors, depth, result, 0, nVectors)
}

func jaccardsSymmetricU1(vectors []byte, nVectors, depth int, result []float32, rowStart, rowCount int) {
	bytesPerVec := DimensionsToValues("u1", depth)
	check(C.nk_jaccards_symmetric_u1_best(
		(*C.nk_u1x8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(bytesPerVec),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount), capabilities(), nil))
}
