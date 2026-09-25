// Kullback-Leibler and Jensen-Shannon divergences of identical and differing distributions.
//
// File: golang/probability_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestKullbackLeiblerF64(t *testing.T) {
	// KLD of identical distributions = 0
	a := []float64{0.25, 0.25, 0.25, 0.25}
	result := numkong.KullbackLeiblerF64(a, a)
	if math.Abs(result) > 1e-6 {
		t.Errorf("Expected ~0 for identical distributions, got %v", result)
	}

	// KLD is non-negative
	b := []float64{0.1, 0.2, 0.3, 0.4}
	result = numkong.KullbackLeiblerF64(a, b)
	if result < -1e-6 {
		t.Errorf("Expected non-negative KLD, got %v", result)
	}
}

func TestKullbackLeiblerF32(t *testing.T) {
	// KLD of identical distributions = 0
	a := []float32{0.25, 0.25, 0.25, 0.25}
	result := numkong.KullbackLeiblerF32(a, a)
	if math.Abs(result) > 1e-3 {
		t.Errorf("Expected ~0 for identical distributions, got %v", result)
	}

	// KLD is non-negative
	b := []float32{0.1, 0.2, 0.3, 0.4}
	result = numkong.KullbackLeiblerF32(a, b)
	if result < -1e-3 {
		t.Errorf("Expected non-negative KLD, got %v", result)
	}
}

func TestJensenShannonF64(t *testing.T) {
	// JSD of identical distributions = 0
	a := []float64{0.25, 0.25, 0.25, 0.25}
	result := numkong.JensenShannonF64(a, a)
	if math.Abs(result) > 1e-6 {
		t.Errorf("Expected ~0 for identical distributions, got %v", result)
	}

	// JSD is symmetric
	b := []float64{0.1, 0.2, 0.3, 0.4}
	ab := numkong.JensenShannonF64(a, b)
	ba := numkong.JensenShannonF64(b, a)
	if math.Abs(ab-ba) > 1e-6 {
		t.Errorf("JSD should be symmetric: JSD(a,b)=%v, JSD(b,a)=%v", ab, ba)
	}

	// JSD is non-negative
	if ab < -1e-6 {
		t.Errorf("Expected non-negative JSD, got %v", ab)
	}
}

func TestJensenShannonF32(t *testing.T) {
	// JSD of identical distributions = 0
	a := []float32{0.25, 0.25, 0.25, 0.25}
	result := numkong.JensenShannonF32(a, a)
	if math.Abs(result) > 1e-3 {
		t.Errorf("Expected ~0 for identical distributions, got %v", result)
	}

	// JSD is symmetric
	b := []float32{0.1, 0.2, 0.3, 0.4}
	ab := numkong.JensenShannonF32(a, b)
	ba := numkong.JensenShannonF32(b, a)
	if math.Abs(ab-ba) > 1e-3 {
		t.Errorf("JSD should be symmetric: JSD(a,b)=%v, JSD(b,a)=%v", ab, ba)
	}

	// JSD is non-negative
	if ab < -1e-3 {
		t.Errorf("Expected non-negative JSD, got %v", ab)
	}
}
