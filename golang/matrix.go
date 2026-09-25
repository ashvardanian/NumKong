// File: golang/matrix.go
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
import (
	"runtime"
	"sync"
)

// WorkerPool runs batch kernels on goroutines pinned to OS threads, each configuring its SIMD state
// once. Create it with [NewWorkerPool], reuse it across calls, and release it with
// [WorkerPool.Close].
type WorkerPool struct {
	tasks []chan func()
	done  sync.WaitGroup
}

// NewWorkerPool starts n pinned workers, or GOMAXPROCS workers when n is not positive.
func NewWorkerPool(n int) *WorkerPool {
	if n <= 0 {
		n = runtime.GOMAXPROCS(0)
	}
	p := &WorkerPool{tasks: make([]chan func(), n)}
	for i := range p.tasks {
		p.tasks[i] = make(chan func())
		p.done.Add(1)
		go func(ch chan func()) {
			defer p.done.Done()
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			C.nk_configure_thread(C.nk_capability_t(C.nk_capabilities_available()))
			for fn := range ch {
				fn()
			}
		}(p.tasks[i])
	}
	return p
}

// Size returns the number of workers in the pool.
func (p *WorkerPool) Size() int { return len(p.tasks) }

// Close stops every worker and waits for it to exit.
func (p *WorkerPool) Close() {
	for _, ch := range p.tasks {
		close(ch)
	}
	p.done.Wait()
}

// run splits [0, totalRows) across pool workers and blocks until all complete.
func (p *WorkerPool) run(totalRows int, fn func(lo, hi int)) {
	workers := min(len(p.tasks), totalRows)
	if workers <= 0 {
		fn(0, totalRows)
		return
	}
	perWorker := divideRoundUp(totalRows, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*perWorker, min((w+1)*perWorker, totalRows)
		if lo >= hi {
			break
		}
		wg.Add(1)
		p.tasks[w] <- func() {
			defer wg.Done()
			fn(lo, hi)
		}
	}
	wg.Wait()
}

// region DotsPackedMatrix WithPool methods

// DotsF64WithPool is [DotsPackedF64] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) DotsF64WithPool(a []float64, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		DotsPackedF64(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// DotsF32WithPool is [DotsPackedF32] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) DotsF32WithPool(a []float32, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		DotsPackedF32(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// DotsI8WithPool is [DotsPackedI8] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) DotsI8WithPool(a []int8, c []int32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		DotsPackedI8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// DotsU8WithPool is [DotsPackedU8] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) DotsU8WithPool(a []uint8, c []uint32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		DotsPackedU8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// AngularsF64WithPool is [AngularsPackedF64] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) AngularsF64WithPool(a []float64, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		AngularsPackedF64(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// AngularsF32WithPool is [AngularsPackedF32] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) AngularsF32WithPool(a []float32, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		AngularsPackedF32(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// AngularsI8WithPool is [AngularsPackedI8] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) AngularsI8WithPool(a []int8, c []float32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		AngularsPackedI8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// AngularsU8WithPool is [AngularsPackedU8] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) AngularsU8WithPool(a []uint8, c []float32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		AngularsPackedU8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// EuclideansF64WithPool is [EuclideansPackedF64] with the height rows of a split across pool, each
// of [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) EuclideansF64WithPool(a []float64, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		EuclideansPackedF64(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// EuclideansF32WithPool is [EuclideansPackedF32] with the height rows of a split across pool, each
// of [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) EuclideansF32WithPool(a []float32, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		EuclideansPackedF32(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// EuclideansI8WithPool is [EuclideansPackedI8] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) EuclideansI8WithPool(a []int8, c []float32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		EuclideansPackedI8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// EuclideansU8WithPool is [EuclideansPackedU8] with the height rows of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) EuclideansU8WithPool(a []uint8, c []float32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		EuclideansPackedU8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// HammingsU1WithPool is [HammingsPackedU1] with the height vectors of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) HammingsU1WithPool(vectors []byte, c []uint32, height int, pool *WorkerPool) {
	bytesPerVec := DimensionsToValues("u1", pm.depth)
	pool.run(height, func(lo, hi int) {
		HammingsPackedU1(vectors[lo*bytesPerVec:hi*bytesPerVec], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// JaccardsU1WithPool is [JaccardsPackedU1] with the height vectors of a split across pool, each of
// [DotsPackedMatrix.Depth] dimensions, while c holds at least height * [DotsPackedMatrix.Width]
// entries.
func (pm DotsPackedMatrix) JaccardsU1WithPool(vectors []byte, c []float32, height int, pool *WorkerPool) {
	bytesPerVec := DimensionsToValues("u1", pm.depth)
	pool.run(height, func(lo, hi int) {
		JaccardsPackedU1(vectors[lo*bytesPerVec:hi*bytesPerVec], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// endregion

// region Symmetric WithPool functions

// DotsSymmetricF64WithPool is [DotsSymmetricF64] with the nVectors rows split across pool, the
// vectors stored row-major; only entries with row <= column are written into result, which holds at
// least nVectors * nVectors entries.
func DotsSymmetricF64WithPool(vectors []float64, nVectors, depth int, result []float64, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		dotsSymmetricF64(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// DotsSymmetricF32WithPool is [DotsSymmetricF32] with the nVectors rows split across pool, the
// vectors stored row-major; only entries with row <= column are written into result, which holds at
// least nVectors * nVectors entries.
func DotsSymmetricF32WithPool(vectors []float32, nVectors, depth int, result []float64, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		dotsSymmetricF32(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// DotsSymmetricI8WithPool is [DotsSymmetricI8] with the nVectors rows split across pool, the
// vectors stored row-major; only entries with row <= column are written into result, which holds at
// least nVectors * nVectors entries.
func DotsSymmetricI8WithPool(vectors []int8, nVectors, depth int, result []int32, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		dotsSymmetricI8(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// DotsSymmetricU8WithPool is [DotsSymmetricU8] with the nVectors rows split across pool, the
// vectors stored row-major; only entries with row <= column are written into result, which holds at
// least nVectors * nVectors entries.
func DotsSymmetricU8WithPool(vectors []uint8, nVectors, depth int, result []uint32, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		dotsSymmetricU8(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// AngularsSymmetricF64WithPool is [AngularsSymmetricF64] with the nVectors rows split across pool,
// the vectors stored row-major; only entries with row <= column are written into result, which
// holds at least nVectors * nVectors entries.
func AngularsSymmetricF64WithPool(vectors []float64, nVectors, depth int, result []float64, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		angularsSymmetricF64(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// AngularsSymmetricF32WithPool is [AngularsSymmetricF32] with the nVectors rows split across pool,
// the vectors stored row-major; only entries with row <= column are written into result, which
// holds at least nVectors * nVectors entries.
func AngularsSymmetricF32WithPool(vectors []float32, nVectors, depth int, result []float64, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		angularsSymmetricF32(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// AngularsSymmetricI8WithPool is [AngularsSymmetricI8] with the nVectors rows split across pool,
// the vectors stored row-major; only entries with row <= column are written into result, which
// holds at least nVectors * nVectors entries.
func AngularsSymmetricI8WithPool(vectors []int8, nVectors, depth int, result []float32, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		angularsSymmetricI8(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// AngularsSymmetricU8WithPool is [AngularsSymmetricU8] with the nVectors rows split across pool,
// the vectors stored row-major; only entries with row <= column are written into result, which
// holds at least nVectors * nVectors entries.
func AngularsSymmetricU8WithPool(vectors []uint8, nVectors, depth int, result []float32, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		angularsSymmetricU8(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// EuclideansSymmetricF64WithPool is [EuclideansSymmetricF64] with the nVectors rows split across
// pool, the vectors stored row-major; only entries with row <= column are written into result,
// which holds at least nVectors * nVectors entries.
func EuclideansSymmetricF64WithPool(vectors []float64, nVectors, depth int, result []float64, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		euclideansSymmetricF64(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// EuclideansSymmetricF32WithPool is [EuclideansSymmetricF32] with the nVectors rows split across
// pool, the vectors stored row-major; only entries with row <= column are written into result,
// which holds at least nVectors * nVectors entries.
func EuclideansSymmetricF32WithPool(vectors []float32, nVectors, depth int, result []float64, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		euclideansSymmetricF32(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// EuclideansSymmetricI8WithPool is [EuclideansSymmetricI8] with the nVectors rows split across
// pool, the vectors stored row-major; only entries with row <= column are written into result,
// which holds at least nVectors * nVectors entries.
func EuclideansSymmetricI8WithPool(vectors []int8, nVectors, depth int, result []float32, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		euclideansSymmetricI8(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// EuclideansSymmetricU8WithPool is [EuclideansSymmetricU8] with the nVectors rows split across
// pool, the vectors stored row-major; only entries with row <= column are written into result,
// which holds at least nVectors * nVectors entries.
func EuclideansSymmetricU8WithPool(vectors []uint8, nVectors, depth int, result []float32, pool *WorkerPool) {
	if len(vectors) < nVectors*depth {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		euclideansSymmetricU8(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// HammingsSymmetricU1WithPool is [HammingsSymmetricU1] with the nVectors rows split across pool.
// The depth is a multiple of 8, and only entries with row <= column are written into result, which
// holds at least nVectors * nVectors entries.
func HammingsSymmetricU1WithPool(vectors []byte, nVectors, depth int, result []uint32, pool *WorkerPool) {
	validateDimensions("u1", depth)
	bytesPerVec := DimensionsToValues("u1", depth)
	if len(vectors) < nVectors*bytesPerVec {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		hammingsSymmetricU1(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// JaccardsSymmetricU1WithPool is [JaccardsSymmetricU1] with the nVectors rows split across pool.
// The depth is a multiple of 8, and only entries with row <= column are written into result, which
// holds at least nVectors * nVectors entries.
func JaccardsSymmetricU1WithPool(vectors []byte, nVectors, depth int, result []float32, pool *WorkerPool) {
	validateDimensions("u1", depth)
	bytesPerVec := DimensionsToValues("u1", depth)
	if len(vectors) < nVectors*bytesPerVec {
		panic("input slice too short for the given nVectors and depth")
	}
	if len(result) < nVectors*nVectors {
		panic("result slice too short for nVectors × nVectors")
	}
	pool.run(nVectors, func(lo, hi int) {
		jaccardsSymmetricU1(vectors, nVectors, depth, result, lo, hi-lo)
	})
}

// endregion
