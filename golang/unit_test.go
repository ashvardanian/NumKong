package numkong

import (
	"math"
	"testing"
)

// region Dot Product Tests

func TestDotF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := DotF64(a, b)
	expected := float64(32) // 1*4 + 2*5 + 3*6 = 32
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestDotF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := DotF32(a, b)
	expected := float64(32)
	if math.Abs(result-expected) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestDotI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := DotI8(a, b)
	expected := int32(32)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestDotU8(t *testing.T) {
	a := []uint8{1, 2, 3}
	b := []uint8{4, 5, 6}
	result := DotU8(a, b)
	expected := uint32(32)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// region Angular (Cosine) Distance Tests

func TestAngularI8(t *testing.T) {
	a := []int8{1, 0}
	b := []int8{0, 1}

	result := AngularI8(a, b)
	expected := float32(1.0) // Angular distance of orthogonal vectors is 1
	if math.Abs(float64(result-expected)) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularF32(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}

	result := AngularF32(a, b)
	expected := float64(1.0) // Angular distance of orthogonal vectors is 1
	if math.Abs(result-expected) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularF64(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{0, 1}

	result := AngularF64(a, b)
	expected := float64(1.0)
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestAngularIdentical(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{1, 2, 3}

	result := AngularF32(a, b)
	// Cosine distance of identical vectors is 0
	if math.Abs(result) > 0.01 {
		t.Errorf("Expected ~0, got %v", result)
	}
}

func TestAngularU8(t *testing.T) {
	a := []uint8{1, 0}
	b := []uint8{0, 1}

	result := AngularU8(a, b)
	expected := float32(1.0)
	if math.Abs(float64(result-expected)) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// region Euclidean Distance Tests

func TestEuclideanF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := EuclideanF64(a, b)
	// sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
	expected := math.Sqrt(27)
	if math.Abs(result-expected) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := EuclideanF32(a, b)
	expected := math.Sqrt(27)
	if math.Abs(result-expected) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := EuclideanI8(a, b)
	expected := float32(math.Sqrt(27))
	if math.Abs(float64(result-expected)) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestEuclideanU8(t *testing.T) {
	a := []uint8{1, 2, 3}
	b := []uint8{4, 5, 6}
	result := EuclideanU8(a, b)
	expected := float32(math.Sqrt(27))
	if math.Abs(float64(result-expected)) > 0.01 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// region Squared Euclidean Distance Tests

func TestSqEuclideanF64(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	result := SqEuclideanF64(a, b)
	expected := float64(27) // (4-1)^2 + (5-2)^2 + (6-3)^2 = 27
	if math.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanF32(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	result := SqEuclideanF32(a, b)
	expected := float64(27)
	if math.Abs(result-expected) > 1e-3 {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanI8(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	result := SqEuclideanI8(a, b)
	expected := uint32(27)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestSqEuclideanU8(t *testing.T) {
	a := []uint8{1, 2, 3}
	b := []uint8{4, 5, 6}
	result := SqEuclideanU8(a, b)
	expected := uint32(27)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

// region Hamming Distance Tests

func TestHammingU8(t *testing.T) {
	a := []uint8{1, 2, 3, 4}
	b := []uint8{1, 0, 3, 5}
	result := HammingU8(a, b)
	// Positions 1 and 3 differ (values 2!=0 and 4!=5)
	expected := uint32(2)
	if result != expected {
		t.Errorf("Expected %v, got %v", expected, result)
	}
}

func TestHammingU8Identical(t *testing.T) {
	a := []uint8{10, 20, 30}
	b := []uint8{10, 20, 30}
	result := HammingU8(a, b)
	if result != 0 {
		t.Errorf("Expected 0 for identical vectors, got %v", result)
	}
}

// region Scalar Set Tests (HammingU1, JaccardU1, JaccardU16, JaccardU32)

func TestHammingU1(t *testing.T) {
	// 0xFF = 11111111, 0x0F = 00001111 → 4 bits differ
	a := []byte{0xFF}
	b := []byte{0x0F}
	result := HammingU1(a, b, 8)
	if result != 4 {
		t.Errorf("Expected 4, got %v", result)
	}

	// All same → 0
	result = HammingU1(a, a, 8)
	if result != 0 {
		t.Errorf("Expected 0, got %v", result)
	}

	// All different → 8
	c := []byte{0x00}
	result = HammingU1(a, c, 8)
	if result != 8 {
		t.Errorf("Expected 8, got %v", result)
	}
}

func TestJaccardU1(t *testing.T) {
	// 0xFF vs 0x0F: intersection=4, union=8 → Jaccard distance = 1 - 4/8 = 0.5
	a := []byte{0xFF}
	b := []byte{0x0F}
	result := JaccardU1(a, b, 8)
	if math.Abs(float64(result)-0.5) > 0.01 {
		t.Errorf("Expected ~0.5, got %v", result)
	}

	// Identical → 0
	result = JaccardU1(a, a, 8)
	if math.Abs(float64(result)) > 0.01 {
		t.Errorf("Expected ~0 for identical, got %v", result)
	}

	// 0xFF vs 0x00: intersection=0, union=8 → Jaccard distance = 1
	c := []byte{0x00}
	result = JaccardU1(a, c, 8)
	if math.Abs(float64(result)-1.0) > 0.01 {
		t.Errorf("Expected ~1, got %v", result)
	}
}

func TestJaccardU16(t *testing.T) {
	// Identical vectors → Jaccard distance = 0
	a := []uint16{1, 2, 3, 4}
	result := JaccardU16(a, a)
	if math.Abs(float64(result)) > 0.01 {
		t.Errorf("Expected ~0 for identical, got %v", result)
	}

	// Completely different → Jaccard distance = 1
	b := []uint16{5, 6, 7, 8}
	result = JaccardU16(a, b)
	if math.Abs(float64(result)-1.0) > 0.01 {
		t.Errorf("Expected ~1, got %v", result)
	}
}

func TestJaccardU32(t *testing.T) {
	// Identical vectors → Jaccard distance = 0
	a := []uint32{10, 20, 30, 40}
	result := JaccardU32(a, a)
	if math.Abs(float64(result)) > 0.01 {
		t.Errorf("Expected ~0 for identical, got %v", result)
	}

	// Completely different → Jaccard distance = 1
	b := []uint32{50, 60, 70, 80}
	result = JaccardU32(a, b)
	if math.Abs(float64(result)-1.0) > 0.01 {
		t.Errorf("Expected ~1, got %v", result)
	}
}

// region Probability Metrics Tests

func TestKullbackLeiblerF64(t *testing.T) {
	// KLD of identical distributions = 0
	a := []float64{0.25, 0.25, 0.25, 0.25}
	result := KullbackLeiblerF64(a, a)
	if math.Abs(result) > 1e-6 {
		t.Errorf("Expected ~0 for identical distributions, got %v", result)
	}

	// KLD is non-negative
	b := []float64{0.1, 0.2, 0.3, 0.4}
	result = KullbackLeiblerF64(a, b)
	if result < -1e-6 {
		t.Errorf("Expected non-negative KLD, got %v", result)
	}
}

func TestKullbackLeiblerF32(t *testing.T) {
	// KLD of identical distributions = 0
	a := []float32{0.25, 0.25, 0.25, 0.25}
	result := KullbackLeiblerF32(a, a)
	if math.Abs(result) > 1e-3 {
		t.Errorf("Expected ~0 for identical distributions, got %v", result)
	}

	// KLD is non-negative
	b := []float32{0.1, 0.2, 0.3, 0.4}
	result = KullbackLeiblerF32(a, b)
	if result < -1e-3 {
		t.Errorf("Expected non-negative KLD, got %v", result)
	}
}

func TestJensenShannonF64(t *testing.T) {
	// JSD of identical distributions = 0
	a := []float64{0.25, 0.25, 0.25, 0.25}
	result := JensenShannonF64(a, a)
	if math.Abs(result) > 1e-6 {
		t.Errorf("Expected ~0 for identical distributions, got %v", result)
	}

	// JSD is symmetric
	b := []float64{0.1, 0.2, 0.3, 0.4}
	ab := JensenShannonF64(a, b)
	ba := JensenShannonF64(b, a)
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
	result := JensenShannonF32(a, a)
	if math.Abs(result) > 1e-3 {
		t.Errorf("Expected ~0 for identical distributions, got %v", result)
	}

	// JSD is symmetric
	b := []float32{0.1, 0.2, 0.3, 0.4}
	ab := JensenShannonF32(a, b)
	ba := JensenShannonF32(b, a)
	if math.Abs(ab-ba) > 1e-3 {
		t.Errorf("JSD should be symmetric: JSD(a,b)=%v, JSD(b,a)=%v", ab, ba)
	}

	// JSD is non-negative
	if ab < -1e-3 {
		t.Errorf("Expected non-negative JSD, got %v", ab)
	}
}

// region Geospatial Tests

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

	HaversineF64(aLat, aLon, bLat, bLon, result)
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

	HaversineF32(aLat, aLon, bLat, bLon, result)
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

	VincentyF64(aLat, aLon, bLat, bLon, result)
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

	VincentyF32(aLat, aLon, bLat, bLon, result)
	expected := float32(5570000)
	if math.Abs(float64(result[0]-expected)) > 20000 {
		t.Errorf("Expected ~%v m, got %v m", expected, result[0])
	}
}

// region Error Handling Tests

func TestVectorLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	a := []int8{1, 0}
	b := []int8{0}
	_ = AngularI8(a, b) // This should panic
}

func TestEmptyVectors(t *testing.T) {
	a := []float32{}
	b := []float32{}
	result := DotF32(a, b)
	if result != 0 {
		t.Errorf("Expected 0 for empty vectors, got %v", result)
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

	HaversineF64(aLat, aLon, bLat, bLon, result)
}

func TestGeospatialEmpty(t *testing.T) {
	aLat := []float64{}
	aLon := []float64{}
	bLat := []float64{}
	bLon := []float64{}
	result := []float64{}
	// Empty inputs should not panic
	HaversineF64(aLat, aLon, bLat, bLon, result)
	VincentyF64(aLat, aLon, bLat, bLon, result)
}

func TestPackedConstructorValidation(t *testing.T) {
	t.Run("F64 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		NewPackedMatrixF64([]float64{1, 2, 3}, 2, 3) // needs 6, got 3
	})

	t.Run("F32 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		NewPackedMatrixF32([]float32{1, 2}, 2, 2) // needs 4, got 2
	})

	t.Run("I8 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		NewPackedMatrixI8([]int8{1}, 2, 2) // needs 4, got 1
	})

	t.Run("U8 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		NewPackedMatrixU8([]uint8{1}, 2, 2) // needs 4, got 1
	})

	t.Run("U1 too short", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Expected panic for short input")
			}
		}()
		NewPackedMatrixU1([]byte{0xFF}, 2, 16) // needs 4, got 1
	})
}

func TestMaxSimConstructorValidation(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for short input")
		}
	}()
	NewMaxSimPackedMatrixF32([]float32{1, 2, 3}, 2, 4) // needs 8, got 3
}

func TestMaxSimDepthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for depth mismatch")
		}
	}()
	q := NewMaxSimPackedMatrixF32(make([]float32, 8), 1, 8)
	d := NewMaxSimPackedMatrixF32(make([]float32, 16), 1, 16)
	MaxSimF32(q, d)
}

// region Packed Batch Operations Tests

func TestDotsPackedF64(t *testing.T) {
	// A: 2×3 matrix, B: 3×3 matrix → C: 2×3 result
	height, width, depth := 2, 3, 3
	a := []float64{1, 2, 3, 4, 5, 6}             // 2 rows of depth 3
	b := []float64{7, 8, 9, 10, 11, 12, 1, 0, 1} // 3 rows of depth 3

	bPacked := NewPackedMatrixF64(b, width, depth)

	c := make([]float64, height*width)
	DotsPackedF64(a, bPacked, c, height)

	// Verify against scalar dot products
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := DotF64(aVec, bVec)
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

	bPacked := NewPackedMatrixF32(b, width, depth)

	c := make([]float64, height*width)
	DotsPackedF32(a, bPacked, c, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := DotF32(aVec, bVec)
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

	bPacked := NewPackedMatrixI8(b, width, depth)

	c := make([]int32, height*width)
	DotsPackedI8(a, bPacked, c, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := DotI8(aVec, bVec)
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

	bPacked := NewPackedMatrixU8(b, width, depth)

	c := make([]uint32, height*width)
	DotsPackedU8(a, bPacked, c, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := DotU8(aVec, bVec)
			got := c[i*width+j]
			if got != expected {
				t.Errorf("DotsPackedU8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

// region Packed Angulars/Euclideans Tests

func TestAngularsPackedF64(t *testing.T) {
	height, width, depth := 2, 2, 3
	a := []float64{1, 2, 3, 4, 5, 6}
	b := []float64{7, 8, 9, 1, 0, 1}

	bPacked := NewPackedMatrixF64(b, width, depth)

	result := make([]float64, height*width)
	AngularsPackedF64(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := AngularF64(aVec, bVec)
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

	bPacked := NewPackedMatrixF32(b, width, depth)

	result := make([]float64, height*width)
	AngularsPackedF32(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := AngularF32(aVec, bVec)
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

	bPacked := NewPackedMatrixI8(b, width, depth)

	result := make([]float32, height*width)
	AngularsPackedI8(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := AngularI8(aVec, bVec)
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

	bPacked := NewPackedMatrixU8(b, width, depth)

	result := make([]float32, height*width)
	AngularsPackedU8(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := AngularU8(aVec, bVec)
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

	bPacked := NewPackedMatrixF64(b, width, depth)

	result := make([]float64, height*width)
	EuclideansPackedF64(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := EuclideanF64(aVec, bVec)
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

	bPacked := NewPackedMatrixF32(b, width, depth)

	result := make([]float64, height*width)
	EuclideansPackedF32(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := EuclideanF32(aVec, bVec)
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

	bPacked := NewPackedMatrixI8(b, width, depth)

	result := make([]float32, height*width)
	EuclideansPackedI8(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := EuclideanI8(aVec, bVec)
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

	bPacked := NewPackedMatrixU8(b, width, depth)

	result := make([]float32, height*width)
	EuclideansPackedU8(a, bPacked, result, height)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			aVec := a[i*depth : (i+1)*depth]
			bVec := b[j*depth : (j+1)*depth]
			expected := EuclideanU8(aVec, bVec)
			got := result[i*width+j]
			if math.Abs(float64(got-expected)) > 0.01 {
				t.Errorf("EuclideansPackedU8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

// region Symmetric Operations Tests

func TestDotsSymmetricF64(t *testing.T) {
	n, depth := 3, 3
	vectors := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	result := make([]float64, n*n)
	DotsSymmetricF64(vectors, n, depth, result)

	// Only upper triangle is filled (i <= j)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			aVec := vectors[i*depth : (i+1)*depth]
			bVec := vectors[j*depth : (j+1)*depth]
			expected := DotF64(aVec, bVec)
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
	DotsSymmetricF32(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			aVec := vectors[i*depth : (i+1)*depth]
			bVec := vectors[j*depth : (j+1)*depth]
			expected := DotF32(aVec, bVec)
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
	DotsSymmetricI8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			aVec := vectors[i*depth : (i+1)*depth]
			bVec := vectors[j*depth : (j+1)*depth]
			expected := DotI8(aVec, bVec)
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
	DotsSymmetricU8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			aVec := vectors[i*depth : (i+1)*depth]
			bVec := vectors[j*depth : (j+1)*depth]
			expected := DotU8(aVec, bVec)
			got := result[i*n+j]
			if got != expected {
				t.Errorf("DotsSymmetricU8[%d][%d]: expected %v, got %v", i, j, expected, got)
			}
		}
	}
}

func TestAngularsSymmetricF64(t *testing.T) {
	n, depth := 3, 3
	vectors := []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
	result := make([]float64, n*n)
	AngularsSymmetricF64(vectors, n, depth, result)

	// Diagonal should be 0 (distance to self)
	for i := 0; i < n; i++ {
		if math.Abs(result[i*n+i]) > 0.01 {
			t.Errorf("AngularsSymmetricF64 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
	// Upper triangle off-diagonal should be 1 (orthogonal unit vectors)
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
	AngularsSymmetricF32(vectors, n, depth, result)

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
	AngularsSymmetricI8(vectors, n, depth, result)

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
	AngularsSymmetricU8(vectors, n, depth, result)

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
	EuclideansSymmetricF64(vectors, n, depth, result)

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
	EuclideansSymmetricF32(vectors, n, depth, result)

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
	EuclideansSymmetricI8(vectors, n, depth, result)

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
	EuclideansSymmetricU8(vectors, n, depth, result)

	for i := 0; i < n; i++ {
		if math.Abs(float64(result[i*n+i])) > 0.01 {
			t.Errorf("EuclideansSymmetricU8 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
}

// region Hamming/Jaccard U1 Tests

func TestHammingsSymmetricU1(t *testing.T) {
	// 3 binary vectors, 8 bits each (1 byte per vector)
	n, depth := 3, 8
	vectors := []byte{0xFF, 0x00, 0x0F} // all ones, all zeros, half
	result := make([]uint32, n*n)
	HammingsSymmetricU1(vectors, n, depth, result)

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

	bPacked := NewPackedMatrixU1(b, cols, depth)

	result := make([]uint32, rows*cols)
	HammingsPackedU1(v, bPacked, result, rows)

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
	JaccardsSymmetricU1(vectors, n, depth, result)

	// Diagonal should be 0
	for i := 0; i < n; i++ {
		if math.Abs(float64(result[i*n+i])) > 0.01 {
			t.Errorf("JaccardsSymmetricU1 diagonal[%d]: expected ~0, got %v", i, result[i*n+i])
		}
	}
	// 0xFF vs 0x00: Jaccard distance = 1 (no intersection, all union)
	if math.Abs(float64(result[0*n+1])-1.0) > 0.01 {
		t.Errorf("JaccardsSymmetricU1[0][1]: expected ~1, got %v", result[0*n+1])
	}
}

func TestJaccardsPackedU1(t *testing.T) {
	rows, cols, depth := 2, 2, 8
	v := []byte{0xFF, 0x0F}
	b := []byte{0x00, 0xFF}

	bPacked := NewPackedMatrixU1(b, cols, depth)

	result := make([]float32, rows*cols)
	JaccardsPackedU1(v, bPacked, result, rows)

	// v[0]=0xFF vs b[1]=0xFF: identical → Jaccard distance = 0
	if math.Abs(float64(result[0*cols+1])) > 0.01 {
		t.Errorf("JaccardsPackedU1[0][1]: expected ~0, got %v", result[0*cols+1])
	}
	// v[0]=0xFF vs b[0]=0x00: Jaccard distance = 1
	if math.Abs(float64(result[0*cols+0])-1.0) > 0.01 {
		t.Errorf("JaccardsPackedU1[0][0]: expected ~1, got %v", result[0*cols+0])
	}
}

// region WorkerPool Tests

func TestWorkerPool(t *testing.T) {
	pool := NewWorkerPool(4)
	if pool.Size() != 4 {
		t.Errorf("Expected pool size 4, got %d", pool.Size())
	}
	pool.Close()
}

func TestWorkerPoolDefault(t *testing.T) {
	pool := NewWorkerPool(0)
	if pool.Size() <= 0 {
		t.Errorf("Expected positive pool size, got %d", pool.Size())
	}
	pool.Close()
}

func TestPackedDotsF32WithPool(t *testing.T) {
	height, width, depth := 8, 3, 3
	a := make([]float32, height*depth)
	for i := range a {
		a[i] = float32(i%7) + 1
	}
	b := []float32{7, 8, 9, 10, 11, 12, 1, 0, 1}
	bPacked := NewPackedMatrixF32(b, width, depth)

	// Single-threaded reference
	ref := make([]float64, height*width)
	DotsPackedF32(a, bPacked, ref, height)

	// Pool-based
	pool := NewWorkerPool(4)
	defer pool.Close()
	got := make([]float64, height*width)
	bPacked.DotsF32WithPool(a, got, height, pool)

	for i := range ref {
		if math.Abs(got[i]-ref[i]) > 1e-6 {
			t.Errorf("DotsF32WithPool[%d]: expected %v, got %v", i, ref[i], got[i])
		}
	}
}

func TestPackedAngularsF32WithPool(t *testing.T) {
	height, width, depth := 6, 2, 3
	a := make([]float32, height*depth)
	for i := range a {
		a[i] = float32(i%5) + 1
	}
	b := []float32{1, 0, 0, 0, 1, 0}
	bPacked := NewPackedMatrixF32(b, width, depth)

	ref := make([]float64, height*width)
	AngularsPackedF32(a, bPacked, ref, height)

	pool := NewWorkerPool(3)
	defer pool.Close()
	got := make([]float64, height*width)
	bPacked.AngularsF32WithPool(a, got, height, pool)

	for i := range ref {
		if math.Abs(got[i]-ref[i]) > 0.01 {
			t.Errorf("AngularsF32WithPool[%d]: expected %v, got %v", i, ref[i], got[i])
		}
	}
}

func TestSymmetricDotsF32WithPool(t *testing.T) {
	n, depth := 6, 3
	vectors := make([]float32, n*depth)
	for i := range vectors {
		vectors[i] = float32(i%7) + 1
	}

	ref := make([]float64, n*n)
	DotsSymmetricF32(vectors, n, depth, ref)

	pool := NewWorkerPool(3)
	defer pool.Close()
	got := make([]float64, n*n)
	DotsSymmetricF32WithPool(vectors, n, depth, got, pool)

	// Only upper triangle is defined for symmetric operations
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			r, g := ref[i*n+j], got[i*n+j]
			if math.Abs(g-r) > 0.01 {
				t.Errorf("DotsSymmetricF32WithPool[%d][%d]: expected %v, got %v", i, j, r, g)
			}
		}
	}
}

func TestSymmetricAngularsF64WithPool(t *testing.T) {
	n, depth := 4, 3
	vectors := []float64{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}

	ref := make([]float64, n*n)
	AngularsSymmetricF64(vectors, n, depth, ref)

	pool := NewWorkerPool(2)
	defer pool.Close()
	got := make([]float64, n*n)
	AngularsSymmetricF64WithPool(vectors, n, depth, got, pool)

	// Only upper triangle is defined for symmetric operations
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			r, g := ref[i*n+j], got[i*n+j]
			if math.Abs(g-r) > 0.01 {
				t.Errorf("AngularsSymmetricF64WithPool[%d][%d]: expected %v, got %v", i, j, r, g)
			}
		}
	}
}

func TestPoolEdgeCases(t *testing.T) {
	t.Run("height less than pool size", func(t *testing.T) {
		pool := NewWorkerPool(8)
		defer pool.Close()
		height, width, depth := 2, 2, 3
		a := []float32{1, 2, 3, 4, 5, 6}
		b := []float32{7, 8, 9, 10, 11, 12}
		bPacked := NewPackedMatrixF32(b, width, depth)

		ref := make([]float64, height*width)
		DotsPackedF32(a, bPacked, ref, height)

		got := make([]float64, height*width)
		bPacked.DotsF32WithPool(a, got, height, pool)

		for i := range ref {
			if math.Abs(got[i]-ref[i]) > 1e-6 {
				t.Errorf("[%d]: expected %v, got %v", i, ref[i], got[i])
			}
		}
	})

	t.Run("height equals 1", func(t *testing.T) {
		pool := NewWorkerPool(4)
		defer pool.Close()
		height, width, depth := 1, 2, 3
		a := []float32{1, 2, 3}
		b := []float32{7, 8, 9, 10, 11, 12}
		bPacked := NewPackedMatrixF32(b, width, depth)

		ref := make([]float64, height*width)
		DotsPackedF32(a, bPacked, ref, height)

		got := make([]float64, height*width)
		bPacked.DotsF32WithPool(a, got, height, pool)

		for i := range ref {
			if math.Abs(got[i]-ref[i]) > 1e-6 {
				t.Errorf("[%d]: expected %v, got %v", i, ref[i], got[i])
			}
		}
	})
}

// region MaxSim Tests

func TestMaxSimF32(t *testing.T) {
	// MaxSim computes sum of angular distances (1 - max_cosine) for each query.
	// With perfectly matching vectors, angular distance = 0.
	depth := 128
	queryCount, documentCount := 2, 3
	query := make([]float32, queryCount*depth)
	document := make([]float32, documentCount*depth)

	// q[0] along dim 0, q[1] along dim 1
	query[0] = 1.0
	query[depth+1] = 1.0
	// d[0] along dim 0 (perfect match for q[0]), d[1] along dim 2, d[2] along dim 1 (match for q[1])
	document[0] = 1.0
	document[depth+2] = 1.0
	document[2*depth+1] = 1.0

	qPacked := NewMaxSimPackedMatrixF32(query, queryCount, depth)
	dPacked := NewMaxSimPackedMatrixF32(document, documentCount, depth)

	result := MaxSimF32(qPacked, dPacked)
	// Both queries have perfect matches → angular distance ≈ 0
	if result < 0 || result > 0.1 {
		t.Errorf("MaxSimF32: expected ~0 for matching vectors, got %v", result)
	}
}

func TestMaxSimF32NonNegative(t *testing.T) {
	queryCount, documentCount, depth := 3, 4, 8
	query := make([]float32, queryCount*depth)
	document := make([]float32, documentCount*depth)
	for i := range query {
		query[i] = float32(i) * 0.1
	}
	for i := range document {
		document[i] = float32(i) * 0.05
	}

	qPacked := NewMaxSimPackedMatrixF32(query, queryCount, depth)
	dPacked := NewMaxSimPackedMatrixF32(document, documentCount, depth)

	result := MaxSimF32(qPacked, dPacked)
	if result < 0 {
		t.Errorf("MaxSimF32: expected non-negative result, got %v", result)
	}
}
