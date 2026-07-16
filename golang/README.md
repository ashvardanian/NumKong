# NumKong for Go

NumKong's Go binding gives you native SIMD-accelerated kernels without building a custom cGo shim.
It covers dot products, dense distances, geospatial helpers, packed matrix kernels, symmetric self-similarity, binary set metrics, probability metrics, and MaxSim late interaction through a slice-based API that fits naturally into Go's numeric idioms.

## Quickstart

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}

	fmt.Println(nk.DotF32(a, b))     // 32 (returned as float64)
	fmt.Println(nk.AngularF32(a, b)) // cosine distance (returned as float64)
}
```

## Highlights

__Slice-based API.__
Plain Go slices are the input model.
__Widened outputs.__
`int8` and `float32` storage widens into safer return types.
__Packed matrix kernels.__
GEMM-like batch workloads with pack-once-reuse-many semantics via `PackedMatrix`.
__Symmetric self-similarity.__
SYRK-like kernels that skip duplicate `(i, j)` and `(j, i)` work.
__MaxSim late interaction.__
ColBERT-style scoring with pre-packed queries and documents via `MaxSimPacked`.
__Binary set metrics.__
Hamming and Jaccard on packed bit vectors and set-hash vectors.
__Probability metrics.__
Kullback-Leibler divergence and Jensen-Shannon distance.
__Geospatial included.__
`Haversine*` and `Vincenty*` are part of the same package.
__Capability bits exposed.__
You can inspect the runtime SIMD surface from Go.

## Ecosystem Comparison

| Feature                      | NumKong                                                                                  | [GoNum](https://github.com/gonum/gonum)                  |
| ---------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Operation families           | dots, distances, binary, probability, geospatial, MaxSim                                 | dots, distances, some statistics                         |
| Precision                    | BFloat16 through sub-byte; automatic widening; Kahan summation; 0 ULP in Float32/Float64 | Float64 only; standard accuracy                          |
| Runtime SIMD dispatch        | auto-selects best ISA per-thread at runtime across x86, ARM, RISC-V                      | no runtime dispatch; some hand-written assembly routines |
| Packed matrix, GEMM-like     | pack once, reuse across query batches via `PackedMatrix`                                 | `mat.Dense.Mul` — no persistent packing                  |
| Symmetric kernels, SYRK-like | skips duplicate pairs, up to 2x speedup for self-distance                                | no duplicate-pair skipping                               |
| Memory model                 | slice-based, caller-owned; cGo zero-copy pointer passing                                 | allocates internally in many functions                   |
| Host-side parallelism        | reusable `WorkerPool` for packed and symmetric batch ops                                 | partial — gonum/optimize has some parallel support       |

## Installation

The Go binding compiles the C library from headers at `go build` time via cGo.
No pre-compiled shared library is required — just a C compiler.

Import the subpackage from the root module:

    import nk "github.com/ashvardanian/NumKong/golang"

The module path is `github.com/ashvardanian/NumKong`.
The Go binding lives under `github.com/ashvardanian/NumKong/golang`.

CGO must be enabled (the default).
Any C11-capable compiler works: GCC, Clang, or MSVC.

## Dot Products

Dot products cover `float64`, `float32`, `int8`, and `uint8`.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	a64 := []float64{1, 2, 3}
	b64 := []float64{4, 5, 6}
	fmt.Println(nk.DotF64(a64, b64))

	a32 := []float32{1, 2, 3}
	b32 := []float32{4, 5, 6}
	fmt.Println(nk.DotF32(a32, b32)) // widened to float64

	a8 := []int8{1, 2, 3, 4}
	b8 := []int8{4, 3, 2, 1}
	fmt.Println(nk.DotI8(a8, b8)) // widened to int32

	au := []uint8{1, 2, 3, 4}
	bu := []uint8{4, 3, 2, 1}
	fmt.Println(nk.DotU8(au, bu)) // widened to uint32
}
```

`DotF32` returns `float64`.
`DotI8` returns `int32`.
`DotU8` returns `uint32`.
Those widened outputs are deliberate.

## Dense Distances

The dense distance family includes squared Euclidean, Euclidean, and angular distance.
Each metric supports `float64`, `float32`, `int8`, and `uint8`.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}

	fmt.Println(nk.SqEuclideanF64(a, b))
	fmt.Println(nk.EuclideanF64(a, b))
	fmt.Println(nk.AngularF64(a, b))
}
```

The `int8` and `uint8` families widen their outputs.
`AngularF32` and `EuclideanF32` return `float64`.
`AngularI8`, `AngularU8`, `EuclideanI8`, `EuclideanU8` return `float32`.
`SqEuclideanI8` and `SqEuclideanU8` return `uint32`.

## Binary Metrics

Hamming and Jaccard distances for binary and set-hash vectors.

```go
// Byte-level Hamming distance
a := []uint8{1, 2, 3, 4}
b := []uint8{1, 0, 3, 5}
dist := nk.HammingU8(a, b) // 2

// Bit-level Hamming distance (packed binary vectors)
x := []byte{0xFF}
y := []byte{0x0F}
bits := nk.HammingU1(x, y, 8) // 4

// Bit-level Jaccard distance
jd := nk.JaccardU1(x, y, 8) // 0.5

// Set-hash Jaccard (MinHash-style)
h16a := []uint16{1, 2, 3, 4}
h16b := []uint16{3, 4, 5, 6}
nk.JaccardU16(h16a, h16b)

h32a := []uint32{10, 20, 30}
h32b := []uint32{30, 40, 50}
nk.JaccardU32(h32a, h32b)
```

## Probability Metrics

Kullback-Leibler divergence and Jensen-Shannon distance for probability distributions.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	p := []float64{0.25, 0.25, 0.25, 0.25}
	q := []float64{0.1, 0.2, 0.3, 0.4}

	fmt.Println(nk.KullbackLeiblerF64(p, q)) // KL divergence
	fmt.Println(nk.JensenShannonF64(p, q))   // JS distance (symmetric)
}
```

`KullbackLeiblerF64` and `JensenShannonF64` take `[]float64` and return `float64`.
`KullbackLeiblerF32` and `JensenShannonF32` take `[]float32` and return `float64` (widened).

## Geospatial Metrics

The Go package also exposes Haversine and Vincenty helpers.
Inputs are in radians.
Outputs are written into caller-owned slices.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	// Statue of Liberty (40.6892°N, 74.0445°W) → Big Ben (51.5007°N, 0.1246°W)
	libertyLat := []float64{0.7101605100}
	libertyLon := []float64{-1.2923203180}
	bigBenLat := []float64{0.8988567821}
	bigBenLon := []float64{-0.0021746802}
	distance := make([]float64, 1)

	nk.VincentyF64(libertyLat, libertyLon, bigBenLat, bigBenLon, distance)  // ≈ 5,589,857 m (ellipsoidal, baseline)
	nk.HaversineF64(libertyLat, libertyLon, bigBenLat, bigBenLon, distance) // ≈ 5,543,723 m (spherical, ~46 km less)

	fmt.Println(distance[0])
}
```

The output slice is caller-owned.
That keeps allocation behavior explicit and predictable.

## Packed Matrix Kernels for GEMM-Like Workloads

Packed kernels are the main batch-throughput path.
The `PackedMatrix` struct wraps the packed buffer with its dimensions and dtype, providing type safety.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	height, width, depth := 4, 8, 16
	a := make([]float32, height*depth) // 4 query vectors of dimension 16
	b := make([]float32, width*depth)  // 8 database vectors of dimension 16

	// Fill with sample data
	for i := range a { a[i] = float32(i % 7) }
	for i := range b { b[i] = float32(i % 5) }

	// Pack the right-hand side (once, reuse across batches)
	bPacked := nk.NewPackedMatrixF32(b, width, depth)

	// Compute A × Bᵀ
	c := make([]float64, height*width)
	nk.DotsPackedF32(a, bPacked, c, height)

	// Angular and Euclidean distances use the same PackedMatrix
	angDist := make([]float64, height*width)
	nk.AngularsPackedF32(a, bPacked, angDist, height)

	eucDist := make([]float64, height*width)
	nk.EuclideansPackedF32(a, bPacked, eucDist, height)

	fmt.Println(c[:width]) // first row of the result matrix
}
```

`DotsPackedF64` takes `[]float64` and produces `[]float64`.
`DotsPackedF32` takes `[]float32` and produces `[]float64` (widened).
`DotsPackedI8` takes `[]int8` and produces `[]int32` (widened).
`DotsPackedU8` takes `[]uint8` and produces `[]uint32` (widened).
The same widening pattern applies to `AngularsPacked*` and `EuclideansPacked*` variants.

## Symmetric Kernels for SYRK-Like Workloads

Symmetric kernels compute self-similarity or self-distance matrices.
They skip duplicate `(i, j)` and `(j, i)` pairs, filling only the upper triangle and mirroring the result.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	n, depth := 4, 8
	vectors := make([]float32, n*depth)
	for i := range vectors { vectors[i] = float32(i % 5) }

	// Gram matrix: all-pairs dot products
	gram := make([]float64, n*n)
	nk.DotsSymmetricF32(vectors, n, depth, gram)

	// Angular distance matrix
	angDist := make([]float64, n*n)
	nk.AngularsSymmetricF32(vectors, n, depth, angDist)

	// Euclidean distance matrix
	eucDist := make([]float64, n*n)
	nk.EuclideansSymmetricF32(vectors, n, depth, eucDist)

	fmt.Println("gram[0]:", gram[0])
	fmt.Println("angular[0,1]:", angDist[1])
	fmt.Println("euclidean[0,1]:", eucDist[1])
}
```

Available symmetric variants: `DotsSymmetric{F64,F32,I8,U8}`, `AngularsSymmetric{F64,F32,I8,U8}`, `EuclideansSymmetric{F64,F32,I8,U8}`.

## Binary Packed and Symmetric Kernels

Binary vectors use `[]byte` storage where `depth` is the number of bits.
Packing uses `NewPackedMatrixU1`.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	n, depth := 4, 64 // 4 vectors of 64 bits each
	bytesPerVec := (depth + 7) / 8
	vectors := make([]byte, n*bytesPerVec)
	for i := range vectors { vectors[i] = byte(i * 37) }

	// Pack for batch queries
	cols := 2
	queryVectors := vectors[:cols*bytesPerVec]
	queryPacked := nk.NewPackedMatrixU1(queryVectors, cols, depth)

	// Hamming distances: n database vectors × cols query vectors
	hammingResult := make([]uint32, n*cols)
	nk.HammingsPackedU1(vectors, queryPacked, hammingResult, n)

	// Jaccard distances
	jaccardResult := make([]float32, n*cols)
	nk.JaccardsPackedU1(vectors, queryPacked, jaccardResult, n)

	// Symmetric Hamming distance matrix
	hammingSym := make([]uint32, n*n)
	nk.HammingsSymmetricU1(vectors, n, depth, hammingSym)

	// Symmetric Jaccard distance matrix
	jaccardSym := make([]float32, n*n)
	nk.JaccardsSymmetricU1(vectors, n, depth, jaccardSym)

	fmt.Println("hamming packed:", hammingResult[:cols])
	fmt.Println("jaccard symmetric[0,1]:", jaccardSym[1])
}
```

## MaxSim and ColBERT-Style Late Interaction

MaxSim is the late-interaction primitive used by systems such as [ColBERT](https://arxiv.org/abs/2004.12832).
It computes the sum of per-query-token maximum cosine similarities across document tokens.
The result is an angular distance: `sum(1 - max_cosine)`.

The `MaxSimPacked` struct wraps packed vectors with their metadata.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	queryTokens, docTokens, depth := 4, 8, 16

	queries := make([]float32, queryTokens*depth)
	docs := make([]float32, docTokens*depth)
	for i := range queries { queries[i] = float32(i%5) + 1 }
	for i := range docs { docs[i] = float32(i%3) + 1 }

	// Pack both sides
	qPacked := nk.NewMaxSimPackedF32(queries, queryTokens, depth)
	dPacked := nk.NewMaxSimPackedF32(docs, docTokens, depth)

	// Compute MaxSim score
	score := nk.MaxSimF32(qPacked, dPacked)
	fmt.Println("MaxSim score:", score)
}
```

## Parallel Batch Processing

`WorkerPool` provides pre-pinned goroutines with pre-configured SIMD state.
Create once, reuse across batch calls, close when done.
The pool amortizes `ConfigureThread` + `LockOSThread` cost across all batch operations.

```go
package main

import (
	"fmt"

	nk "github.com/ashvardanian/NumKong/golang"
)

func main() {
	width, depth, totalQueries := 1024, 128, 10000

	db := make([]float32, width*depth)
	queries := make([]float32, totalQueries*depth)
	for i := range db { db[i] = float32(i%7) * 0.1 }
	for i := range queries { queries[i] = float32(i%11) * 0.1 }

	dbPacked := nk.NewPackedMatrixF32(db, width, depth)

	// Create a reusable pool (defaults to GOMAXPROCS workers)
	pool := nk.NewWorkerPool(8)
	defer pool.Close()

	// Packed batch operations dispatch to the pool
	results := make([]float64, totalQueries*width)
	dbPacked.DotsF32WithPool(queries, results, totalQueries, pool)

	// Angular and Euclidean distances use the same pool
	angResults := make([]float64, totalQueries*width)
	dbPacked.AngularsF32WithPool(queries, angResults, totalQueries, pool)

	fmt.Println("first result row:", results[:width])
}
```

Symmetric operations also support pool-based parallelism:

```go
n, depth := 1000, 128
vectors := make([]float32, n*depth)
gram := make([]float64, n*n)

pool := nk.NewWorkerPool(0) // 0 = GOMAXPROCS
defer pool.Close()

nk.DotsSymmetricF32WithPool(vectors, n, depth, gram, pool)
```

For one-off parallel work without a pool, you can still use `ConfigureThread` directly:

```go
go func() {
	defer nk.ConfigureThread()() // lock thread + configure SIMD; defer unlocks
	nk.DotsPackedF32(queries, dbPacked, results, height)
}()
```

## Thread Configuration and Capabilities

`ConfigureThread` pins the current goroutine to an OS thread via `runtime.LockOSThread`, enables CPU-specific acceleration features such as Intel AMX, then returns an unlock function.
Goroutines can migrate between OS threads, so thread-local state (AMX tiles) would be lost without pinning.

```go
defer nk.ConfigureThread()()                // auto-detect, lock thread, defer unlock
defer nk.ConfigureThreadWith(caps)()        // explicit capability mask variant
caps := nk.CapabilitiesAvailable()          // what can actually run here
```

Idiomatic usage is `defer nk.ConfigureThread()()` — the first `()` calls `ConfigureThread` (which locks the thread and returns the unlock function), the second `()` is deferred and calls the unlock function when the surrounding function returns.

`ConfigureThreadWith` lets you narrow the enabled feature set.
The package also exposes capability bit constants, like `CapSerial`, `CapNeon`, `CapHaswell`, `CapSkylake`, `CapSapphire`, `CapSapphireAmx`, and `CapSme`.
These are useful for logging the active platform or gating optional benchmark paths.

## cGo Integration Notes

This package is a cGo wrapper over the C library.
That means a few rules matter:

- Input slices must have matching lengths where the API expects paired vectors.
- Length mismatches and insufficient slice capacity panic uniformly across all functions.
- Empty slices return zero for scalar outputs rather than crashing.
- The slice backing arrays remain owned by Go.
- `PackedMatrix` and `MaxSimPacked` structs own their packed buffers and carry dimensions and dtype metadata.
- Constructors validate that input slices are large enough for the given dimensions.
- Batch functions validate both input and output slice sizes.
- Symmetric output matrices must be `n × n` in size.

### Memory Safety

Go automatically pins slice backing arrays for the duration of each cGo call.
No `runtime.Pinner` or manual pinning is needed from the caller.

`PackedMatrix` and `MaxSimPacked` hold strong Go references to their `[]byte` buffers.
This keeps the packed data alive for garbage collection as long as the struct is reachable.

Memory footprint in Go is easiest to think about in two layers.
The slice header is the ordinary Go slice header.
The payload is the backing array you already own.
NumKong does not wrap those slices in extra heap-owning tensor objects in this binding.
