// Packed and symmetric dot products against pairwise ones, and the short inputs packing rejects.
//
// File: golang/dots_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestPackedConstructorValidation(t *testing.T) {
	t.Run("F64 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		numkong.NewDotsPackedMatrixF64([]float64{1, 2, 3}, 2, 3) // needs 6, got 3
	})

	t.Run("F32 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		numkong.NewDotsPackedMatrixF32([]float32{1, 2}, 2, 2) // needs 4, got 2
	})

	t.Run("I8 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		numkong.NewDotsPackedMatrixI8([]int8{1}, 2, 2) // needs 4, got 1
	})

	t.Run("U8 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		numkong.NewDotsPackedMatrixU8([]uint8{1}, 2, 2) // needs 4, got 1
	})

	t.Run("U1 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		numkong.NewDotsPackedMatrixU1([]byte{0xFF}, 2, 16) // needs 4, got 1
	})

	t.Run("U1 partial byte", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for a depth that is not a multiple of 8")
			}
		}()
		numkong.NewDotsPackedMatrixU1([]byte{0xFF, 0xFF}, 2, 7)
	})
}

func TestDotsPackedF64(t *testing.T) {
	// A: 2×3 matrix, B: 3×3 matrix → C: 2×3 result
	height, width, depth := 2, 3, 3
	a := []float64{1, 2, 3, 4, 5, 6}             // 2 rows of depth 3
	b := []float64{7, 8, 9, 10, 11, 12, 1, 0, 1} // 3 rows of depth 3

	bPacked := numkong.NewDotsPackedMatrixF64(b, width, depth)

	c := make([]float64, height*width)
	numkong.DotsPackedF64(a, bPacked, c, height)

	// Verify against scalar dot products
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.DotF64(aVec, bVec)
			got := c[i*width+j]
			if math.Abs(got-expected) > 1e-6 {
				t.Errorf("DotsPackedF64[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestDotsPackedF32(t *testing.T) {
	height, width, depth := 2, 3, 3
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 10, 11, 12, 1, 0, 1}

	bPacked := numkong.NewDotsPackedMatrixF32(b, width, depth)

	c := make([]float64, height*width)
	numkong.DotsPackedF32(a, bPacked, c, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.DotF32(aVec, bVec)
			got := c[i*width+j]
			if math.Abs(got-expected) > 0.01 {
				t.Errorf("DotsPackedF32[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestDotsPackedI8(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []int8{1, 2, 3, 4, 5, 6}
	b := []int8{7, 8, 9, 10, 11, 12}

	bPacked := numkong.NewDotsPackedMatrixI8(b, width, depth)

	c := make([]int32, height*width)
	numkong.DotsPackedI8(a, bPacked, c, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.DotI8(aVec, bVec)
			got := c[i*width+j]
			if got != expected {
				t.Errorf("DotsPackedI8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestDotsPackedU8(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []uint8{1, 2, 3, 4, 5, 6}
	b := []uint8{7, 8, 9, 10, 11, 12}

	bPacked := numkong.NewDotsPackedMatrixU8(b, width, depth)

	c := make([]uint32, height*width)
	numkong.DotsPackedU8(a, bPacked, c, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := numkong.DotU8(aVec, bVec)
			got := c[i*width+j]
			if got != expected {
				t.Errorf("DotsPackedU8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestDotsSymmetricF64(t *testing.T) {
	n, depth := 3, 3
	vectors := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	result := make([]float64, n*n)
	numkong.DotsSymmetricF64(vectors, n, depth, result)

	// Only upper triangle is filled (i <= j)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			aVec := vectors[i*depth : (i+1)*depth]
			bVec := vectors[j*depth : (j+1)*depth]
			expected := numkong.DotF64(aVec, bVec)
			got := result[i*n+j]
			if math.Abs(got-expected) > 1e-6 {
				t.Errorf("DotsSymmetricF64[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestDotsSymmetricF32(t *testing.T) {
	n, depth := 3, 3
	vectors := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	result := make([]float64, n*n)
	numkong.DotsSymmetricF32(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			aVec := vectors[i*depth : (i+1)*depth]
			bVec := vectors[j*depth : (j+1)*depth]
			expected := numkong.DotF32(aVec, bVec)
			got := result[i*n+j]
			if math.Abs(got-expected) > 0.01 {
				t.Errorf("DotsSymmetricF32[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestDotsSymmetricI8(t *testing.T) {
	n, depth := 3, 3
	vectors := []int8{1, 2, 3, 4, 5, 6, 7, 8, 9}
	result := make([]int32, n*n)
	numkong.DotsSymmetricI8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			aVec := vectors[i*depth : (i+1)*depth]
			bVec := vectors[j*depth : (j+1)*depth]
			expected := numkong.DotI8(aVec, bVec)
			got := result[i*n+j]
			if got != expected {
				t.Errorf("DotsSymmetricI8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestDotsSymmetricU8(t *testing.T) {
	n, depth := 3, 3
	vectors := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9}
	result := make([]uint32, n*n)
	numkong.DotsSymmetricU8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			aVec := vectors[i*depth : (i+1)*depth]
			bVec := vectors[j*depth : (j+1)*depth]
			expected := numkong.DotU8(aVec, bVec)
			got := result[i*n+j]
			if got != expected {
				t.Errorf("DotsSymmetricU8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}
