// Batch kernels split across the goroutine pool against their single-threaded results.
//
// File: golang/matrix_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestWorkerPool(t *testing.T) {
	pool := numkong.NewWorkerPool(4)
	if pool.Size() != 4 {
		t.Errorf("Expected pool size 4, got %d", pool.Size())
	}
	pool.Close()
}

func TestWorkerPoolDefault(t *testing.T) {
	pool := numkong.NewWorkerPool(0)
	if pool.Size() <= 0 {
		t.Errorf("Expected positive pool size, got %d", pool.Size())
	}
	pool.Close()
}

func TestPackedDotsF32WithPool(t *testing.T) {
	height, width, depth := 8, 3, 3
	a := make([]float32, height*depth)
	for i := range a {
		a[i] = float32(i%7) + 1
	}
	b := []float32{7, 8, 9, 10, 11, 12, 1, 0, 1}
	bPacked := numkong.NewDotsPackedMatrixF32(b, width, depth)

	// Single-threaded reference
	ref := make([]float64, height*width)
	numkong.DotsPackedF32(a, bPacked, ref, height)

	// Pool-based
	pool := numkong.NewWorkerPool(4)
	defer pool.Close()
	got := make([]float64, height*width)
	bPacked.DotsF32WithPool(a, got, height, pool)

	for i := range ref {
		if math.Abs(got[i]-ref[i]) > 1e-6 {
			t.Errorf("DotsF32WithPool[%d]: expected %v, got %v", i, ref[i], got[i])
		}
	}
}

func TestPackedAngularsF32WithPool(t *testing.T) {
	height, width, depth := 6, 2, 3
	a := make([]float32, height*depth)
	for i := range a {
		a[i] = float32(i%5) + 1
	}
	b := []float32{1, 0, 0, 0, 1, 0}
	bPacked := numkong.NewDotsPackedMatrixF32(b, width, depth)

	ref := make([]float64, height*width)
	numkong.AngularsPackedF32(a, bPacked, ref, height)

	pool := numkong.NewWorkerPool(3)
	defer pool.Close()
	got := make([]float64, height*width)
	bPacked.AngularsF32WithPool(a, got, height, pool)

	for i := range ref {
		if math.Abs(got[i]-ref[i]) > 0.01 {
			t.Errorf("AngularsF32WithPool[%d]: expected %v, got %v", i, ref[i], got[i])
		}
	}
}

func TestSymmetricDotsF32WithPool(t *testing.T) {
	n, depth := 6, 3
	vectors := make([]float32, n*depth)
	for i := range vectors {
		vectors[i] = float32(i%7) + 1
	}

	ref := make([]float64, n*n)
	numkong.DotsSymmetricF32(vectors, n, depth, ref)

	pool := numkong.NewWorkerPool(3)
	defer pool.Close()
	got := make([]float64, n*n)
	numkong.DotsSymmetricF32WithPool(vectors, n, depth, got, pool)

	// Only upper triangle is defined for symmetric operations
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			r, g := ref[i*n+j], got[i*n+j]
			if math.Abs(g-r) > 0.01 {
				t.Errorf("DotsSymmetricF32WithPool[%d][%d]: expected %v, got %v", i, j, r, g)
			}
		}
	}
}

func TestSymmetricAngularsF64WithPool(t *testing.T) {
	n, depth := 4, 3
	vectors := []float64{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}

	ref := make([]float64, n*n)
	numkong.AngularsSymmetricF64(vectors, n, depth, ref)

	pool := numkong.NewWorkerPool(2)
	defer pool.Close()
	got := make([]float64, n*n)
	numkong.AngularsSymmetricF64WithPool(vectors, n, depth, got, pool)

	// Only upper triangle is defined for symmetric operations
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			r, g := ref[i*n+j], got[i*n+j]
			if math.Abs(g-r) > 0.01 {
				t.Errorf("AngularsSymmetricF64WithPool[%d][%d]: expected %v, got %v", i, j, r, g)
			}
		}
	}
}

func TestPoolEdgeCases(t *testing.T) {
	t.Run("height less than pool size", func(t *testing.T) {
		pool := numkong.NewWorkerPool(8)
		defer pool.Close()
		height, width, depth := 2, 2, 3
		a := []float32{1, 2, 3, 4, 5, 6}
		b := []float32{7, 8, 9, 10, 11, 12}
		bPacked := numkong.NewDotsPackedMatrixF32(b, width, depth)

		ref := make([]float64, height*width)
		numkong.DotsPackedF32(a, bPacked, ref, height)

		got := make([]float64, height*width)
		bPacked.DotsF32WithPool(a, got, height, pool)

		for i := range ref {
			if math.Abs(got[i]-ref[i]) > 1e-6 {
				t.Errorf("[%d]: expected %v, got %v", i, ref[i], got[i])
			}
		}
	})

	t.Run("height equals 1", func(t *testing.T) {
		pool := numkong.NewWorkerPool(4)
		defer pool.Close()
		height, width, depth := 1, 2, 3
		a := []float32{1, 2, 3}
		b := []float32{7, 8, 9, 10, 11, 12}
		bPacked := numkong.NewDotsPackedMatrixF32(b, width, depth)

		ref := make([]float64, height*width)
		numkong.DotsPackedF32(a, bPacked, ref, height)

		got := make([]float64, height*width)
		bPacked.DotsF32WithPool(a, got, height, pool)

		for i := range ref {
			if math.Abs(got[i]-ref[i]) > 1e-6 {
				t.Errorf("[%d]: expected %v, got %v", i, ref[i], got[i])
			}
		}
	})
}
