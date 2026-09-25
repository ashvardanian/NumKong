// Packed and symmetric angular and Euclidean distances against pairwise ones.
//
// File: golang/spatials_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestAngularsPackedF64(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{7, 8, 9, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixF64(b, width, depth)

	result := make([]float64, height*width)
	numkong.AngularsPackedF64(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.AngularF64(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(got-expected) > 0.01 {
				t.Errorf("AngularsPackedF64[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestAngularsPackedF32(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixF32(b, width, depth)

	result := make([]float64, height*width)
	numkong.AngularsPackedF32(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.AngularF32(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(got-expected) > 0.01 {
				t.Errorf("AngularsPackedF32[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestAngularsPackedI8(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []int8{1, 2, 3, 4, 5, 6}
	b := []int8{7, 8, 9, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixI8(b, width, depth)

	result := make([]float32, height*width)
	numkong.AngularsPackedI8(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.AngularI8(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(float64(got-expected)) > 0.01 {
				t.Errorf("AngularsPackedI8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestAngularsPackedU8(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []uint8{1, 2, 3, 4, 5, 6}
	b := []uint8{7, 8, 9, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixU8(b, width, depth)

	result := make([]float32, height*width)
	numkong.AngularsPackedU8(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.AngularU8(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(float64(got-expected)) > 0.01 {
				t.Errorf("AngularsPackedU8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestEuclideansPackedF64(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{7, 8, 9, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixF64(b, width, depth)

	result := make([]float64, height*width)
	numkong.EuclideansPackedF64(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.EuclideanF64(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(got-expected) > 0.01 {
				t.Errorf("EuclideansPackedF64[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestEuclideansPackedF32(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixF32(b, width, depth)

	result := make([]float64, height*width)
	numkong.EuclideansPackedF32(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.EuclideanF32(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(got-expected) > 0.01 {
				t.Errorf("EuclideansPackedF32[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestEuclideansPackedI8(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []int8{1, 2, 3, 4, 5, 6}
	b := []int8{7, 8, 9, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixI8(b, width, depth)

	result := make([]float32, height*width)
	numkong.EuclideansPackedI8(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.EuclideanI8(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(float64(got-expected)) > 0.01 {
				t.Errorf("EuclideansPackedI8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestEuclideansPackedU8(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []uint8{1, 2, 3, 4, 5, 6}
	b := []uint8{7, 8, 9, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixU8(b, width, depth)

	result := make([]float32, height*width)
	numkong.EuclideansPackedU8(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.EuclideanU8(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(float64(got-expected)) > 0.01 {
				t.Errorf("EuclideansPackedU8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestAngularsSymmetricF64(t *testing.T) {
	n, depth := 3, 3
	vectors := []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float64, n*n)
	numkong.AngularsSymmetricF64(vectors, n, depth, result)

	// Diagonal should be 0, the distance to self
	for i := 0; i < n; i++ {
		if math.Abs(result[i*n+i]) > 0.01 {
			t.Errorf("AngularsSymmetricF64 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
	// Upper triangle off-diagonal should be 1, as the unit vectors are orthogonal
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if math.Abs(result[i*n+j]-1.0) > 0.01 {
				t.Errorf("AngularsSymmetricF64[%d][%d]: expected ~1, got %v", i, j, result[i*n+j])
			}
		}
	}
}

func TestAngularsSymmetricF32(t *testing.T) {
	n, depth := 3, 3
	vectors := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float64, n*n)
	numkong.AngularsSymmetricF32(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		if math.Abs(result[i*n+i]) > 0.01 {
			t.Errorf("AngularsSymmetricF32 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
}

func TestAngularsSymmetricI8(t *testing.T) {
	n, depth := 3, 3
	vectors := []int8{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float32, n*n)
	numkong.AngularsSymmetricI8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		if math.Abs(float64(result[i*n+i])) > 0.01 {
			t.Errorf("AngularsSymmetricI8 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
}

func TestAngularsSymmetricU8(t *testing.T) {
	n, depth := 3, 3
	vectors := []uint8{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float32, n*n)
	numkong.AngularsSymmetricU8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		if math.Abs(float64(result[i*n+i])) > 0.01 {
			t.Errorf("AngularsSymmetricU8 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
}

func TestEuclideansSymmetricF64(t *testing.T) {
	n, depth := 3, 3
	vectors := []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float64, n*n)
	numkong.EuclideansSymmetricF64(vectors, n, depth, result)

	// Diagonal should be 0
	for i := 0; i < n; i++ {
		if math.Abs(result[i*n+i]) > 0.01 {
			t.Errorf("EuclideansSymmetricF64 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
	// Upper triangle off-diagonal: sqrt(2) ≈ 1.414
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			expected := math.Sqrt(2)
			if math.Abs(result[i*n+j]-expected) > 0.01 {
				t.Errorf("EuclideansSymmetricF64[%d][%d]: expected %v, got %v", i, j, expected, result[i*n+j])
			}
		}
	}
}

func TestEuclideansSymmetricF32(t *testing.T) {
	n, depth := 3, 3
	vectors := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float64, n*n)
	numkong.EuclideansSymmetricF32(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		if math.Abs(result[i*n+i]) > 0.01 {
			t.Errorf("EuclideansSymmetricF32 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
}

func TestEuclideansSymmetricI8(t *testing.T) {
	n, depth := 3, 3
	vectors := []int8{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float32, n*n)
	numkong.EuclideansSymmetricI8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		if math.Abs(float64(result[i*n+i])) > 0.01 {
			t.Errorf("EuclideansSymmetricI8 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
}

func TestEuclideansSymmetricU8(t *testing.T) {
	n, depth := 3, 3
	vectors := []uint8{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float32, n*n)
	numkong.EuclideansSymmetricU8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		if math.Abs(float64(result[i*n+i])) > 0.01 {
			t.Errorf("EuclideansSymmetricU8 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
}
