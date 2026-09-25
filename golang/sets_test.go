// Packed and symmetric Hamming and Jaccard distances of bit vectors against known counts.
//
// File: golang/sets_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestHammingsSymmetricU1(t *testing.T) {
	// 3 binary vectors, 8 bits each (1 byte per vector)
	n, depth := 3, 8
	vectors := []byte{0xFF, 0x00, 0x0F} // all ones, all zeros, half
	result := make([]uint32, n*n)
	numkong.HammingsSymmetricU1(vectors, n, depth, result)

	// Diagonal should be 0
	for i := 0; i < n; i++ {
		if result[i*n+i] != 0 {
			t.Errorf("HammingsSymmetricU1 diagonal[%d]: expected 0, got %v", i, result[i*n+i])
		}
	}
	// 0xFF vs 0x00: 8 bits differ
	if result[0*n+1] != 8 {
		t.Errorf("HammingsSymmetricU1[0][1]: expected 8, got %v", result[0*n+1])
	}
	// 0xFF vs 0x0F: 4 bits differ
	if result[0*n+2] != 4 {
		t.Errorf("HammingsSymmetricU1[0][2]: expected 4, got %v", result[0*n+2])
	}
}

func TestHammingsPackedU1(t *testing.T) {
	rows, cols, depth := 2, 2, 8
	v := []byte{0xFF, 0x0F} // 2 row vectors
	b := []byte{0x00, 0x0F} // 2 column vectors to pack

	bPacked := numkong.NewDotsPackedMatrixU1(b, cols, depth)

	result := make([]uint32, rows*cols)
	numkong.HammingsPackedU1(v, bPacked, result, rows)

	// v[0]=0xFF vs b[0]=0x00: 8 bits differ
	if result[0*cols+0] != 8 {
		t.Errorf("HammingsPackedU1[0][0]: expected 8, got %v", result[0*cols+0])
	}
	// v[0]=0xFF vs b[1]=0x0F: 4 bits differ
	if result[0*cols+1] != 4 {
		t.Errorf("HammingsPackedU1[0][1]: expected 4, got %v", result[0*cols+1])
	}
	// v[1]=0x0F vs b[0]=0x00: 4 bits differ
	if result[1*cols+0] != 4 {
		t.Errorf("HammingsPackedU1[1][0]: expected 4, got %v", result[1*cols+0])
	}
	// v[1]=0x0F vs b[1]=0x0F: 0 bits differ
	if result[1*cols+1] != 0 {
		t.Errorf("HammingsPackedU1[1][1]: expected 0, got %v", result[1*cols+1])
	}
}

func TestJaccardsSymmetricU1(t *testing.T) {
	n, depth := 3, 8
	vectors := []byte{0xFF, 0x00, 0x0F}
	result := make([]float32, n*n)
	numkong.JaccardsSymmetricU1(vectors, n, depth, result)

	// Diagonal should be 0
	for i := 0; i < n; i++ {
		if math.Abs(float64(result[i*n+i])) > 0.01 {
			t.Errorf("JaccardsSymmetricU1 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
	// 0xFF vs 0x00: no intersection within the whole union, so the Jaccard distance is 1
	if math.Abs(float64(result[0*n+1])-1.0) > 0.01 {
		t.Errorf("JaccardsSymmetricU1[0][1]: expected ~1, got %v", result[0*n+1])
	}
}

func TestJaccardsPackedU1(t *testing.T) {
	rows, cols, depth := 2, 2, 8
	v := []byte{0xFF, 0x0F}
	b := []byte{0x00, 0xFF}

	bPacked := numkong.NewDotsPackedMatrixU1(b, cols, depth)

	result := make([]float32, rows*cols)
	numkong.JaccardsPackedU1(v, bPacked, result, rows)

	// v[0]=0xFF vs b[1]=0xFF: identical → Jaccard distance = 0
	if math.Abs(float64(result[0*cols+1])) > 0.01 {
		t.Errorf("JaccardsPackedU1[0][1]: expected ~0, got %v", result[0*cols+1])
	}
	// v[0]=0xFF vs b[0]=0x00: Jaccard distance = 1
	if math.Abs(float64(result[0*cols+0])-1.0) > 0.01 {
		t.Errorf("JaccardsPackedU1[0][0]: expected ~1, got %v", result[0*cols+0])
	}
}
