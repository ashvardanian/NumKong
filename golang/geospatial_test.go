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

func TestHaversineF64(t *testing.T) {
	// New York City to London
	// NYC: 40.7128° N, 74.0060° W
	// London: 51.5074° N, 0.1278° W
	degToRad := math.Pi / 180.0
	aLat := []float64{40.7128 * degToRad}
	aLon := []float64{-74.0060 * degToRad}
	bLat := []float64{51.5074 * degToRad}
	bLon := []float64{-0.1278 * degToRad}
	result := make([]float64, 1)

	numkong.HaversineF64(aLat, aLon, bLat, bLon, result)
	// Expected: ~5,539 km
	expected := 5539000.0
	if math.Abs(result[0]-expected) > 5000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

func TestHaversineF32(t *testing.T) {
	degToRad := float32(math.Pi / 180.0)
	aLat := []float32{40.7128 * degToRad}
	aLon := []float32{-74.0060 * degToRad}
	bLat := []float32{51.5074 * degToRad}
	bLon := []float32{-0.1278 * degToRad}
	result := make([]float32, 1)

	numkong.HaversineF32(aLat, aLon, bLat, bLon, result)
	expected := float32(5539000)
	if math.Abs(float64(result[0]-expected)) > 5000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

func TestVincentyF64(t *testing.T) {
	degToRad := math.Pi / 180.0
	aLat := []float64{40.7128 * degToRad}
	aLon := []float64{-74.0060 * degToRad}
	bLat := []float64{51.5074 * degToRad}
	bLon := []float64{-0.1278 * degToRad}
	result := make([]float64, 1)

	numkong.VincentyF64(aLat, aLon, bLat, bLon, result)
	// Vincenty is more accurate, expected: ~5,570 km
	expected := 5570000.0
	if math.Abs(result[0]-expected) > 20000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

func TestVincentyF32(t *testing.T) {
	degToRad := float32(math.Pi / 180.0)
	aLat := []float32{40.7128 * degToRad}
	aLon := []float32{-74.0060 * degToRad}
	bLat := []float32{51.5074 * degToRad}
	bLon := []float32{-0.1278 * degToRad}
	result := make([]float32, 1)

	numkong.VincentyF32(aLat, aLon, bLat, bLon, result)
	expected := float32(5570000)
	if math.Abs(float64(result[0]-expected)) > 20000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

func TestGeospatialLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for mismatched lengths")
		}
	}()

	aLat := []float64{1.0, 2.0}
	aLon := []float64{1.0}
	bLat := []float64{1.0}
	bLon := []float64{1.0}
	result := make([]float64, 1)

	numkong.HaversineF64(aLat, aLon, bLat, bLon, result)
}

func TestGeospatialEmpty(t *testing.T) {
	aLat := []float64{}
	aLon := []float64{}
	bLat := []float64{}
	bLon := []float64{}
	result := []float64{}
	// Empty inputs should not panic
	numkong.HaversineF64(aLat, aLon, bLat, bLon, result)
	numkong.VincentyF64(aLat, aLon, bLat, bLon, result)
}
