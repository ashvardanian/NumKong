// Dot products of short slices against hand-computed sums, and of empty slices.
//
// File: golang/dot_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestDotF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := numkong.DotF64(a, b)
	expected := float64(32) // 1*4 + 2*5 + 3*6 = 32
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestDotF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := numkong.DotF32(a, b)
	expected := float64(32)
	if math.Abs(result-expected) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestDotI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := numkong.DotI8(a, b)
	expected := int32(32)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestDotU8(t *testing.T) {
	a := []uint8{1, 2, 3}
	b := []uint8{4, 5, 6}
	result := numkong.DotU8(a, b)
	expected := uint32(32)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEmptyVectors(t *testing.T) {
	a := []float32{}
	b := []float32{}
	result := numkong.DotF32(a, b)
	if result != 0 {
		t.Errorf("Expected 0 for empty vectors, got %v", result)
	}
}
