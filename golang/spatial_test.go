// Angular, Euclidean and squared Euclidean distances of short slices against known values.
//
// File: golang/spatial_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestAngularI8(t *testing.T) {
	a := []int8{1, 0}
	b := []int8{0, 1}

	result := numkong.AngularI8(a, b)
	expected := float32(1.0) // Angular distance of orthogonal vectors is 1
	if math.Abs(float64(result-expected)) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularF32(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}

	result := numkong.AngularF32(a, b)
	expected := float64(1.0) // Angular distance of orthogonal vectors is 1
	if math.Abs(result-expected) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularF64(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{0, 1}

	result := numkong.AngularF64(a, b)
	expected := float64(1.0)
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularIdentical(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{1, 2, 3}

	result := numkong.AngularF32(a, b)
	// Cosine distance of identical vectors is 0
	if math.Abs(result) > 0.01 {
		t.Errorf("Expected ~0, got %v", result)
	}
}

func TestAngularU8(t *testing.T) {
	a := []uint8{1, 0}
	b := []uint8{0, 1}

	result := numkong.AngularU8(a, b)
	expected := float32(1.0)
	if math.Abs(float64(result-expected)) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := numkong.EuclideanF64(a, b)
	// sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
	expected := math.Sqrt(27)
	if math.Abs(result-expected) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := numkong.EuclideanF32(a, b)
	expected := math.Sqrt(27)
	if math.Abs(result-expected) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := numkong.EuclideanI8(a, b)
	expected := float32(math.Sqrt(27))
	if math.Abs(float64(result-expected)) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanU8(t *testing.T) {
	a := []uint8{1, 2, 3}
	b := []uint8{4, 5, 6}
	result := numkong.EuclideanU8(a, b)
	expected := float32(math.Sqrt(27))
	if math.Abs(float64(result-expected)) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := numkong.SqEuclideanF64(a, b)
	expected := float64(27) // (4-1)^2 + (5-2)^2 + (6-3)^2 = 27
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := numkong.SqEuclideanF32(a, b)
	expected := float64(27)
	if math.Abs(result-expected) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := numkong.SqEuclideanI8(a, b)
	expected := uint32(27)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanU8(t *testing.T) {
	a := []uint8{1, 2, 3}
	b := []uint8{4, 5, 6}
	result := numkong.SqEuclideanU8(a, b)
	expected := uint32(27)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestVectorLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	a := []int8{1, 0}
	b := []int8{0}
	_ = numkong.AngularI8(a, b) // This should panic
}
