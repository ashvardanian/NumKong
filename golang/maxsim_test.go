// MaxSim scores of matching and arbitrary tokens, and the short or mismatched inputs it rejects.
//
// File: golang/maxsim_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestMaxSimConstructorValidation(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for short input")
		}
	}()
	numkong.NewMaxSimPackedMatrixF32([]float32{1, 2, 3}, 2, 4) // needs 8, got 3
}

func TestMaxSimDepthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for depth mismatch")
		}
	}()
	q := numkong.NewMaxSimPackedMatrixF32(make([]float32, 8), 1, 8)
	d := numkong.NewMaxSimPackedMatrixF32(make([]float32, 16), 1, 16)
	numkong.MaxSimF32(q, d)
}

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
	// d[0] along dim 0 matches q[0], d[1] along dim 2 matches neither, d[2] along dim 1 matches q[1]
	document[0] = 1.0
	document[depth+2] = 1.0
	document[2*depth+1] = 1.0

	qPacked := numkong.NewMaxSimPackedMatrixF32(query, queryCount, depth)
	dPacked := numkong.NewMaxSimPackedMatrixF32(document, documentCount, depth)

	result := numkong.MaxSimF32(qPacked, dPacked)
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

	qPacked := numkong.NewMaxSimPackedMatrixF32(query, queryCount, depth)
	dPacked := numkong.NewMaxSimPackedMatrixF32(document, documentCount, depth)

	result := numkong.MaxSimF32(qPacked, dPacked)
	if result < 0 {
		t.Errorf("MaxSimF32: expected non-negative result, got %v", result)
	}
}
