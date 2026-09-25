// Kullback-Leibler and Jensen-Shannon divergences of float64 and float32 distributions.
//
// File: golang/probability.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"

// KullbackLeiblerF64 computes the Kullback-Leibler divergence between two float64 distributions.
// Both slices must have the same length.
func KullbackLeiblerF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_kld_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// KullbackLeiblerF32 computes the Kullback-Leibler divergence between two float32 distributions.
// Both slices must have the same length.
func KullbackLeiblerF32(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_kld_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// JensenShannonF64 computes the Jensen-Shannon distance between two float64 distributions.
// Both slices must have the same length.
func JensenShannonF64(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_jsd_f64((*C.nk_f64_t)(&a[0]), (*C.nk_f64_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}

// JensenShannonF32 computes the Jensen-Shannon distance between two float32 distributions.
// Both slices must have the same length.
func JensenShannonF32(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("both vectors must have the same length")
	}
	if len(a) == 0 {
		return 0
	}
	var result C.nk_f64_t
	C.nk_jsd_f32((*C.nk_f32_t)(&a[0]), (*C.nk_f32_t)(&b[0]), C.nk_size_t(len(a)), &result)
	return float64(result)
}
