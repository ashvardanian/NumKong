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

// HammingsPackedU1 computes Hamming distances for binary vectors where queries are pre-packed.
// vectors: height binary vectors of query.Depth() dimensions each.
// result must have capacity >= height × query.Width().
func HammingsPackedU1(vectors []byte, query DotsPackedMatrix, result []uint32, height int) {
	if query.Dtype() != "u1" {
		panic("DotsPackedMatrix dtype must be u1")
	}
	bytesPerVec := dimensionsToValues(C.nk_u1_k, query.depth)
	if len(vectors) < height*bytesPerVec {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*query.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_hammings_packed_u1(
		(*C.nk_u1x8_t)(&vectors[0]),
		unsafe.Pointer(&query.data[0]),
		(*C.nk_u32_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(query.width), C.nk_size_t(query.depth),
		C.nk_size_t(bytesPerVec),
		C.nk_size_t(query.width*4))
}

// HammingsSymmetricU1 computes the Hamming distance matrix for a set of binary vectors.
// vectors: nVectors binary vectors of depth dimensions each.
// `depth` counts dimensions, a multiple of 8, the values per byte.
// result must have capacity >= nVectors × nVectors.
func HammingsSymmetricU1(vectors []byte, nVectors, depth int, result []uint32) {
	validateDimensions(C.nk_u1_k, depth)
	bytesPerVec := dimensionsToValues(C.nk_u1_k, depth)
	if len(vectors) < nVectors*bytesPerVec {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	hammingsSymmetricU1(vectors, nVectors, depth, result, 0, nVectors)
}

func hammingsSymmetricU1(vectors []byte, nVectors, depth int, result []uint32, rowStart, rowCount int) {
	bytesPerVec := dimensionsToValues(C.nk_u1_k, depth)
	C.nk_hammings_symmetric_u1(
		(*C.nk_u1x8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(bytesPerVec),
		(*C.nk_u32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}

// JaccardsPackedU1 computes Jaccard distances for binary vectors where queries are pre-packed.
// vectors: height binary vectors of query.Depth() dimensions each.
// result must have capacity >= height × query.Width().
func JaccardsPackedU1(vectors []byte, query DotsPackedMatrix, result []float32, height int) {
	if query.Dtype() != "u1" {
		panic("DotsPackedMatrix dtype must be u1")
	}
	bytesPerVec := dimensionsToValues(C.nk_u1_k, query.depth)
	if len(vectors) < height*bytesPerVec {
		panic("input slice too short for the given height and depth")
	}
	if len(result) < height*query.width {
		panic("output slice too short for the given height and width")
	}
	C.nk_jaccards_packed_u1(
		(*C.nk_u1x8_t)(&vectors[0]),
		unsafe.Pointer(&query.data[0]),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(height), C.nk_size_t(query.width), C.nk_size_t(query.depth),
		C.nk_size_t(bytesPerVec),
		C.nk_size_t(query.width*4))
}

// JaccardsSymmetricU1 computes the Jaccard distance matrix for a set of binary vectors.
// vectors: nVectors binary vectors of depth dimensions each.
// `depth` counts dimensions, a multiple of 8, the values per byte.
// result must have capacity >= nVectors × nVectors.
func JaccardsSymmetricU1(vectors []byte, nVectors, depth int, result []float32) {
	validateDimensions(C.nk_u1_k, depth)
	bytesPerVec := dimensionsToValues(C.nk_u1_k, depth)
	if len(vectors) < nVectors*bytesPerVec {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	jaccardsSymmetricU1(vectors, nVectors, depth, result, 0, nVectors)
}

func jaccardsSymmetricU1(vectors []byte, nVectors, depth int, result []float32, rowStart, rowCount int) {
	bytesPerVec := dimensionsToValues(C.nk_u1_k, depth)
	C.nk_jaccards_symmetric_u1(
		(*C.nk_u1x8_t)(&vectors[0]),
		C.nk_size_t(nVectors), C.nk_size_t(depth),
		C.nk_size_t(bytesPerVec),
		(*C.nk_f32_t)(&result[0]),
		C.nk_size_t(nVectors*4),
		C.nk_size_t(rowStart), C.nk_size_t(rowCount))
}
