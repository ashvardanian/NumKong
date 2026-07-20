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

// MaxSimPacked holds a pre-packed set of vectors for MaxSim computation.
// Construct with NewMaxSimPackedF32.
type MaxSimPacked struct {
	data        []byte
	vectorCount int
	depth       int
}

func (p MaxSimPacked) VectorCount() int { return p.vectorCount }
func (p MaxSimPacked) Depth() int       { return p.depth }
func (p MaxSimPacked) Bytes() []byte    { return p.data }

// NewMaxSimPackedF32 packs vectors for MaxSim computation.
// vectors must have capacity >= vectorCount × depth.
func NewMaxSimPackedF32(vectors []float32, vectorCount, depth int) MaxSimPacked {
	if len(vectors) < vectorCount*depth {
		panic("input slice too short for the given vectorCount and depth")
	}
	size := int(C.nk_maxsim_pack_size_f32(C.nk_size_t(vectorCount), C.nk_size_t(depth)))
	data := make([]byte, size)
	C.nk_maxsim_pack_f32(
		(*C.nk_f32_t)(&vectors[0]),
		C.nk_size_t(vectorCount), C.nk_size_t(depth),
		C.nk_size_t(depth*4),
		unsafe.Pointer(&data[0]))
	return MaxSimPacked{data: data, vectorCount: vectorCount, depth: depth}
}

// MaxSimF32 computes MaxSim (ColBERT late interaction) between pre-packed queries and documents.
// Returns the MaxSim score as float64 (widened).
// query and document must have the same depth.
func MaxSimF32(query, document MaxSimPacked) float64 {
	if query.depth != document.depth {
		panic("query and document must have the same depth")
	}
	var result C.nk_f64_t
	C.nk_maxsim_packed_f32(
		unsafe.Pointer(&query.data[0]),
		unsafe.Pointer(&document.data[0]),
		C.nk_size_t(query.vectorCount), C.nk_size_t(document.vectorCount), C.nk_size_t(query.depth),
		&result)
	return float64(result)
}
