// How many logical dimensions each dtype stores per value.
//
// File: golang/types_test.go
// Author: Ash Vardanian

package numkong_test

import (
	"testing"

	numkong "github.com/ashvardanian/NumKong/golang"
)

func TestDimensionsPerValue(t *testing.T) {
	for dtype, expected := range map[string]int{"f64": 1, "f32": 1, "i8": 1, "u8": 1, "u1": 8} {
		if got := numkong.DimensionsPerValue(dtype); got != expected {
			t.Errorf("DimensionsPerValue(%q): expected %d, got %d", dtype, expected, got)
		}
	}
	if got := numkong.DimensionsToValues("u1", 64); got != 8 {
		t.Errorf("DimensionsToValues(\"u1\", 64): expected 8, got %d", got)
	}
}
