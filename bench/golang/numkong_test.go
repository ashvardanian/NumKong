// Benchmarks the angular distance of float32 vectors against a pure Go cosine distance.
//
// File: bench/golang/numkong_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"math"
	"math/rand/v2"
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func cosineDistance(a, b []float32) float32 {
	var dotProduct float32
	var normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	sim := dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1 - sim
}

func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	return vec
}

func BenchmarkCosineDistanceNative(b *testing.B) {
	first, second := generateRandomVector(1536), generateRandomVector(1536)
	for range b.N {
		cosineDistance(first, second)
	}
}

func BenchmarkAngularDistanceSIMD(b *testing.B) {
	first, second := generateRandomVector(1536), generateRandomVector(1536)
	for range b.N {
		numkong.AngularF32(first, second)
	}
}
