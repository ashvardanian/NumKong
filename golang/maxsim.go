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

// MaxSimPackedMatrix holds a pre-packed set of vectors for MaxSim computation.
// Construct with NewMaxSimPackedMatrixF32.
type MaxSimPackedMatrix struct {
	data    []byte
	vectors int
	depth   int
}

func (p MaxSimPackedMatrix) Vectors() int  { return p.vectors }
func (p MaxSimPackedMatrix) Depth() int    { return p.depth }
func (p MaxSimPackedMatrix) Bytes() []byte { return p.data }

// Shape reads the packed vector-set dimensions (vectors, depth) back from the
// buffer's self-describing header via nk_maxsim_packed_shape_f32.
func (p MaxSimPackedMatrix) Shape() (vectors, depth int) {
	if len(p.data) == 0 {
		return 0, 0
	}
	var v, d C.nk_size_t
	C.nk_maxsim_packed_shape_f32(unsafe.Pointer(&p.data[0]), &v, &d)
	return int(v), int(d)
}

// NewMaxSimPackedMatrixF32 packs vectors for MaxSim computation.
// vectors must have capacity >= vectorCount × depth.
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

// MaxSimF32 computes MaxSim (ColBERT late interaction) between pre-packed queries and documents.
// Returns the MaxSim score as float64 (widened).
// query and document must have the same depth.
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
