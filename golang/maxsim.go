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

// MaxSimPackedMatrix holds a set of vectors packed once for [MaxSimF32].
// Construct it with [NewMaxSimPackedMatrixF32].
type MaxSimPackedMatrix struct {
	data    []byte
	vectors int
	depth   int
}

// Vectors returns the number of packed vectors.
func (p MaxSimPackedMatrix) Vectors() int { return p.vectors }

// Depth returns the number of dimensions per packed vector.
func (p MaxSimPackedMatrix) Depth() int { return p.depth }

// Bytes returns the packed buffer.
func (p MaxSimPackedMatrix) Bytes() []byte { return p.data }

// Shape reads the vector count and depth back from the header of the packed buffer.
func (p MaxSimPackedMatrix) Shape() (vectors, depth int) {
	if len(p.data) == 0 {
		return 0, 0
	}
	var v, d C.nk_size_t
	C.nk_maxsim_packed_shape_f32(unsafe.Pointer(&p.data[0]), &v, &d)
	return int(v), int(d)
}

// NewMaxSimPackedMatrixF32 packs vectorsCount float32 vectors of depth dimensions for [MaxSimF32].
// The slice vectorsData holds at least vectorsCount * depth values.
func NewMaxSimPackedMatrixF32(vectorsData []float32, vectorsCount, depth int) MaxSimPackedMatrix {
	if len(vectorsData) < vectorsCount*depth {
		panic("input slice too short for the given vectorsCount and depth")
	}
	size := int(C.nk_maxsim_pack_size_f32(C.nk_size_t(vectorsCount), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_maxsim_pack_f32(
		(*C.nk_f32_t)(&vectorsData[0]),
		C.nk_size_t(vectorsCount), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		unsafe.Pointer(&data[0]))
	return MaxSimPackedMatrix{data: data, vectors: vectorsCount, depth: depth}
}

// MaxSimF32 sums, over the query vectors, the angular distance from each to its nearest document vector.
// Both matrices must have the same depth.
func MaxSimF32(query, document MaxSimPackedMatrix) float64 {
	if query.depth != document.depth {
		panic("query and document must have the same depth")
	}
	var result C.nk_f64_t
	C.nk_maxsim_packed_f32(
		unsafe.Pointer(&query.data[0]),
		unsafe.Pointer(&document.data[0]),
		C.nk_size_t(query.vectors), C.nk_size_t(document.vectors), C.nk_size_t(query.depth),
		&result)
	return float64(result)
}
