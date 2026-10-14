// Haversine and Vincenty distances between known cities, and mismatched or empty slices.
//
// File: golang/geospatial_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestHaversinesF64(t *testing.T) {
	// New York City to London
	// NYC: 40.7128° N, 74.0060° W
	// London: 51.5074° N, 0.1278° W
	degToRad := math.Pi / 180.0
	aLatitudes := []float64{40.7128 * degToRad}
	aLongitudes := []float64{-74.0060 * degToRad}
	bLatitudes := []float64{51.5074 * degToRad}
	bLongitudes := []float64{-0.1278 * degToRad}
	results := make([]float64, 1)

	numkong.HaversinesF64(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results)
	// Expected: ~5,570 km on the mean sphere
	expected := 5570000.0
	if math.Abs(results[0]-expected) > 5000 {
		t.Errorf("Expected ~%v m, got %v m", expected, results[0])
	}
}

func TestVincentysF64(t *testing.T) {
	degToRad := math.Pi / 180.0
	aLatitudes := []float64{40.7128 * degToRad}
	aLongitudes := []float64{-74.0060 * degToRad}
	bLatitudes := []float64{51.5074 * degToRad}
	bLongitudes := []float64{-0.1278 * degToRad}
	results := make([]float64, 1)

	numkong.VincentysF64(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results)
	// Vincenty is more accurate, expected: ~5,585 km
	expected := 5585000.0
	if math.Abs(results[0]-expected) > 20000 {
		t.Errorf("Expected ~%v m, got %v m", expected, results[0])
	}
}

func TestGeospatialLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for mismatched lengths")
		}
	}()

	aLatitudes := []float64{1.0, 2.0}
	aLongitudes := []float64{1.0}
	bLatitudes := []float64{1.0}
	bLongitudes := []float64{1.0}
	results := make([]float64, 1)

	numkong.HaversinesF64(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results)
}

func TestGeospatialEmpty(t *testing.T) {
	aLatitudes := []float64{}
	aLongitudes := []float64{}
	bLatitudes := []float64{}
	bLongitudes := []float64{}
	results := []float64{}
	// Empty inputs should not panic
	numkong.HaversinesF64(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results)
	numkong.VincentysF64(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results)
}
