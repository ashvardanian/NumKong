// Haversine and Vincenty distances in meters from every point of one set to every point of another.
//
// File: golang/geospatial.go
// Author: Ash Vardanian

package numkong

// #include "numkong/numkong.h"
import "C"
import "unsafe"

// HaversinesF64 writes the great-circle distance in meters from each a point to each b point,
// with float64 coordinates in radians, into results, row-major with len(bLatitudes) columns.
// Each latitude slice matches its longitude slice, and results holds len(aLatitudes) *
// len(bLatitudes) entries.
func HaversinesF64(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results []float64) {
	columns := checkGeodesics(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results)
	if len(results) == 0 {
		return
	}
	check(C.nk_haversines_packed_f64_best(
		packHaversinesF64(aLatitudes, aLongitudes), packHaversinesF64(bLatitudes, bLongitudes),
		(*C.nk_f64_t)(&results[0]), C.nk_size_t(columns*8), C.nk_cap_cpus_k, nil))
}

// VincentysF64 writes the ellipsoidal geodesic distance in meters from each a point to each b
// point, with float64 coordinates in radians, into results, row-major with len(bLatitudes)
// columns. Each latitude slice matches its longitude slice, and results holds len(aLatitudes) *
// len(bLatitudes) entries.
func VincentysF64(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results []float64) {
	columns := checkGeodesics(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results)
	if len(results) == 0 {
		return
	}
	check(C.nk_vincentys_packed_f64_best(
		packVincentysF64(aLatitudes, aLongitudes), packVincentysF64(bLatitudes, bLongitudes),
		(*C.nk_f64_t)(&results[0]), C.nk_size_t(columns*8), C.nk_cap_cpus_k, nil))
}

// checkGeodesics panics unless each latitude slice matches its longitude slice and results holds
// one entry per pair, and returns the number of columns.
func checkGeodesics(aLatitudes, aLongitudes, bLatitudes, bLongitudes, results []float64) int {
	rows, columns := len(aLatitudes), len(bLatitudes)
	if rows != len(aLongitudes) || columns != len(bLongitudes) || rows*columns != len(results) {
		panic("each latitude slice must match its longitude slice, and results must hold rows × columns entries")
	}
	return columns
}

func packHaversinesF64(latitudes, longitudes []float64) unsafe.Pointer {
	var size C.nk_size_t
	check(C.nk_haversines_pack_size_f64_best(C.nk_size_t(len(latitudes)), C.nk_cap_cpus_k, &size))
	packed := alignedBytes(size)
	check(C.nk_haversines_pack_f64_best(
		(*C.nk_f64_t)(&latitudes[0]), (*C.nk_f64_t)(&longitudes[0]), C.nk_size_t(len(latitudes)),
		packed, C.nk_cap_cpus_k, nil))
	return packed
}

func packVincentysF64(latitudes, longitudes []float64) unsafe.Pointer {
	var size C.nk_size_t
	check(C.nk_vincentys_pack_size_f64_best(C.nk_size_t(len(latitudes)), C.nk_cap_cpus_k, &size))
	packed := alignedBytes(size)
	check(C.nk_vincentys_pack_f64_best(
		(*C.nk_f64_t)(&latitudes[0]), (*C.nk_f64_t)(&longitudes[0]), C.nk_size_t(len(latitudes)),
		packed, C.nk_cap_cpus_k, nil))
	return packed
}

// alignedBytes allocates size bytes starting on an nk_default_alignment_k boundary, as packs need.
func alignedBytes(size C.nk_size_t) unsafe.Pointer {
	alignment := int(C.nk_default_alignment_k)
	data := make([]byte, int(size)+alignment-1)
	misalignment := int(uintptr(unsafe.Pointer(&data[0])) % uintptr(alignment))
	return unsafe.Pointer(&data[(alignment-misalignment)%alignment])
}
