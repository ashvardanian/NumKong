// Hamming and Jaccard distances over bit vectors, bytes and set-hash signatures.
//
// File: golang/set_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestHammingU8(t *testing.T) {
	a := []uint8{1, 2, 3, 4}
	b := []uint8{1, 0, 3, 5}
	result := numkong.HammingU8(a, b)
	// Positions 1 and 3 differ (values 2!=0 and 4!=5)
	expected := uint32(2)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestHammingU8Identical(t *testing.T) {
	a := []uint8{10, 20, 30}
	b := []uint8{10, 20, 30}
	result := numkong.HammingU8(a, b)
	if result != 0 {
		t.Errorf("Expected 0 for identical vectors, got %v", result)
	}
}

func TestHammingU1(t *testing.T) {
	// 0xFF = 11111111, 0x0F = 00001111 → 4 bits differ
	a := []byte{0xFF}
	b := []byte{0x0F}
	result := numkong.HammingU1(a, b, 8)
	if result != 4 {
		t.Errorf("Expected 4, got %v", result)
	}

	// All same → 0
	result = numkong.HammingU1(a, a, 8)
	if result != 0 {
		t.Errorf("Expected 0, got %v", result)
	}

	// All different → 8
	c := []byte{0x00}
	result = numkong.HammingU1(a, c, 8)
	if result != 8 {
		t.Errorf("Expected 8, got %v", result)
	}
}

func TestJaccardU1(t *testing.T) {
	// 0xFF vs 0x0F: intersection=4, union=8 → Jaccard distance = 1 - 4/8 = 0.5
	a := []byte{0xFF}
	b := []byte{0x0F}
	result := numkong.JaccardU1(a, b, 8)
	if math.Abs(float64(result)-0.5) > 0.01 {
		t.Errorf("Expected ~0.5, got %v", result)
	}

	// Identical → 0
	result = numkong.JaccardU1(a, a, 8)
	if math.Abs(float64(result)) > 0.01 {
		t.Errorf("Expected ~0 for identical, got %v", result)
	}

	// 0xFF vs 0x00: intersection=0, union=8 → Jaccard distance = 1
	c := []byte{0x00}
	result = numkong.JaccardU1(a, c, 8)
	if math.Abs(float64(result)-1.0) > 0.01 {
		t.Errorf("Expected ~1, got %v", result)
	}
}

func TestJaccardU16(t *testing.T) {
	// Identical vectors → Jaccard distance = 0
	a := []uint16{1, 2, 3, 4}
	result := numkong.JaccardU16(a, a)
	if math.Abs(float64(result)) > 0.01 {
		t.Errorf("Expected ~0 for identical, got %v", result)
	}

	// Completely different → Jaccard distance = 1
	b := []uint16{5, 6, 7, 8}
	result = numkong.JaccardU16(a, b)
	if math.Abs(float64(result)-1.0) > 0.01 {
		t.Errorf("Expected ~1, got %v", result)
	}
}

func TestJaccardU32(t *testing.T) {
	// Identical vectors → Jaccard distance = 0
	a := []uint32{10, 20, 30, 40}
	result := numkong.JaccardU32(a, a)
	if math.Abs(float64(result)) > 0.01 {
		t.Errorf("Expected ~0 for identical, got %v", result)
	}

	// Completely different → Jaccard distance = 1
	b := []uint32{50, 60, 70, 80}
	result = numkong.JaccardU32(a, b)
	if math.Abs(float64(result)-1.0) > 0.01 {
		t.Errorf("Expected ~1, got %v", result)
	}
}
