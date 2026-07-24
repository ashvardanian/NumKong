package numkong

/*
#cgo CFLAGS: -O3 -I../include
#cgo LDFLAGS: -O3 -L. -lm
#define NK_NATIVE_F16 (0)
#define NK_NATIVE_BF16 (0)
#include "numkong/numkong.h"
*/
import "C"
import (
	"runtime"
	"sync"
)

// WorkerPool is a pool of goroutines pinned to OS threads with pre-configured
// SIMD state. Create with NewWorkerPool, reuse across batch calls, and Close
// when done. Each worker locks its OS thread and configures SIMD state once at
// startup — the state persists for the pool's lifetime.
type WorkerPool struct {
	tasks []chan func()
	done  sync.WaitGroup
}

// NewWorkerPool creates a pool of n worker goroutines, each pinned to an OS
// thread with SIMD state configured. If n <= 0, defaults to GOMAXPROCS.
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

// Close shuts down all workers and waits for them to finish.
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
	perWorker := (totalRows + workers - 1) / workers
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

// region PackedMatrix WithPool methods

// DotsF64WithPool computes A × Bᵀ in parallel using the pool. Output type: f64.
func (pm PackedMatrix) DotsF64WithPool(a []float64, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		DotsPackedF64(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// DotsF32WithPool computes A × Bᵀ in parallel using the pool. Output type: f64.
func (pm PackedMatrix) DotsF32WithPool(a []float32, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		DotsPackedF32(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// DotsI8WithPool computes A × Bᵀ in parallel using the pool. Output type: i32.
func (pm PackedMatrix) DotsI8WithPool(a []int8, c []int32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		DotsPackedI8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// DotsU8WithPool computes A × Bᵀ in parallel using the pool. Output type: u32.
func (pm PackedMatrix) DotsU8WithPool(a []uint8, c []uint32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		DotsPackedU8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// AngularsF64WithPool computes angular distances in parallel using the pool.
func (pm PackedMatrix) AngularsF64WithPool(a []float64, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		AngularsPackedF64(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// AngularsF32WithPool computes angular distances in parallel using the pool.
func (pm PackedMatrix) AngularsF32WithPool(a []float32, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		AngularsPackedF32(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// AngularsI8WithPool computes angular distances in parallel using the pool.
func (pm PackedMatrix) AngularsI8WithPool(a []int8, c []float32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		AngularsPackedI8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// AngularsU8WithPool computes angular distances in parallel using the pool.
func (pm PackedMatrix) AngularsU8WithPool(a []uint8, c []float32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		AngularsPackedU8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// EuclideansF64WithPool computes Euclidean distances in parallel using the pool.
func (pm PackedMatrix) EuclideansF64WithPool(a []float64, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		EuclideansPackedF64(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// EuclideansF32WithPool computes Euclidean distances in parallel using the pool.
func (pm PackedMatrix) EuclideansF32WithPool(a []float32, c []float64, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		EuclideansPackedF32(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// EuclideansI8WithPool computes Euclidean distances in parallel using the pool.
func (pm PackedMatrix) EuclideansI8WithPool(a []int8, c []float32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		EuclideansPackedI8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// EuclideansU8WithPool computes Euclidean distances in parallel using the pool.
func (pm PackedMatrix) EuclideansU8WithPool(a []uint8, c []float32, height int, pool *WorkerPool) {
	pool.run(height, func(lo, hi int) {
		EuclideansPackedU8(a[lo*pm.depth:hi*pm.depth], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// HammingsU1WithPool computes Hamming distances for binary vectors in parallel.
func (pm PackedMatrix) HammingsU1WithPool(vectors []byte, c []uint32, height int, pool *WorkerPool) {
	bytesPerVec := (pm.depth + 7) / 8
	pool.run(height, func(lo, hi int) {
		HammingsPackedU1(vectors[lo*bytesPerVec:hi*bytesPerVec], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// JaccardsU1WithPool computes Jaccard distances for binary vectors in parallel.
func (pm PackedMatrix) JaccardsU1WithPool(vectors []byte, c []float32, height int, pool *WorkerPool) {
	bytesPerVec := (pm.depth + 7) / 8
	pool.run(height, func(lo, hi int) {
		JaccardsPackedU1(vectors[lo*bytesPerVec:hi*bytesPerVec], pm, c[lo*pm.width:hi*pm.width], hi-lo)
	})
}

// endregion

// region Symmetric WithPool functions

// DotsSymmetricF64WithPool computes the Gram matrix in parallel using the pool.
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

// DotsSymmetricF32WithPool computes the Gram matrix in parallel using the pool.
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

// DotsSymmetricI8WithPool computes the Gram matrix in parallel using the pool.
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

// DotsSymmetricU8WithPool computes the Gram matrix in parallel using the pool.
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

// AngularsSymmetricF64WithPool computes the angular distance matrix in parallel.
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

// AngularsSymmetricF32WithPool computes the angular distance matrix in parallel.
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

// AngularsSymmetricI8WithPool computes the angular distance matrix in parallel.
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

// AngularsSymmetricU8WithPool computes the angular distance matrix in parallel.
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

// EuclideansSymmetricF64WithPool computes the Euclidean distance matrix in parallel.
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

// EuclideansSymmetricF32WithPool computes the Euclidean distance matrix in parallel.
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

// EuclideansSymmetricI8WithPool computes the Euclidean distance matrix in parallel.
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

// EuclideansSymmetricU8WithPool computes the Euclidean distance matrix in parallel.
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

// HammingsSymmetricU1WithPool computes the Hamming distance matrix in parallel.
func HammingsSymmetricU1WithPool(vectors []byte, nVectors, depth int, result []uint32, pool *WorkerPool) {
	bytesPerVec := (depth + 7) / 8
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

// JaccardsSymmetricU1WithPool computes the Jaccard distance matrix in parallel.
func JaccardsSymmetricU1WithPool(vectors []byte, nVectors, depth int, result []float32, pool *WorkerPool) {
	bytesPerVec := (depth + 7) / 8
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
