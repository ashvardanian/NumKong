# Native random array sampling — 2026-11-07

NumKong's native `Generator` is compared with `numpy.random.default_rng(137)` (PCG64), using NumPy 2.5.2 on both machines. NumKong reports version 7.8.5 from the PR source tree; the JSON files record SHA-256 hashes of the actual RNG sources. Both architectures used identical source files. Final clang-format changed only whitespace in `python/random.c`; the JSON files retain the measured hashes and record the formatted hash separately.

Each row is the median of 21 single-threaded Python calls after three warmups. Array-producing calls include allocation on both paths; NumKong uses `array_factory=np.empty` and fills that allocation directly. Calls alternate backend order. `shuffle` runs in place and was measured separately for each backend.

Outputs contain 1,000,000 values, except sparse `choice`, which selects 10,000 distinct indices from 10,000,000 candidates. The exact parameters are in [the benchmark script](../../bench_random.py). Generators produce different sequences; correctness is checked with bounds, state replay, CDF, moment, frequency, and buffer-guard tests.

Environments: Apple M4 Max, macOS 27.0.1, CPython 3.12.7; Intel Xeon 2.80 GHz on GCP n2-standard-2, Linux x86-64, CPython 3.12.16.

All times below are milliseconds; lower is better.

| Workload | M4 NumKong | M4 NumPy | Xeon NumKong | Xeon NumPy |
| --- | ---: | ---: | ---: | ---: |
| `random_f32` | 0.521 | 1.667 | 1.974 | 2.655 |
| `random_f64` | 0.684 | 3.085 | 2.219 | 3.718 |
| `uniform_f64` | 0.735 | 3.258 | 2.196 | 4.721 |
| `normal_f64` | 1.957 | 4.062 | 4.739 | 13.576 |
| `standard_normal_f32` | 1.675 | 2.681 | 4.692 | 11.831 |
| `standard_normal_f64` | 1.859 | 3.971 | 4.789 | 12.211 |
| `integers_u8` | 0.182 | 0.473 | 0.227 | 1.129 |
| `integers_i32` | 1.377 | 1.921 | 3.012 | 3.224 |
| `laplace_f64` | 5.675 | 11.964 | 14.222 | 21.278 |
| `beta_f64` | 18.191 | 18.731 | 40.971 | 49.489 |
| `poisson_small` | 19.514 | 23.774 | 27.996 | 36.698 |
| `poisson_large` | 21.874 | 24.674 | 45.935 | 53.201 |
| `choice_sparse` | 0.056 | 0.097 | 0.122 | 0.208 |
| `shuffle_i64` | 5.265 | 5.343 | 17.629 | 18.389 |

Uniform, normal, and byte-array fills show the largest gains. Beta and shuffle are close to parity on M4 Max. These results describe the listed workloads on these CPUs; array size, parameters, another NumPy engine, and concurrent use can change the ranking.

Reproduce with `python bench/bench_random.py --size 1000000 --repeats 21`. Raw results: [Apple M4 Max](apple-m4-max.json), [Intel Xeon](intel-xeon.json).

Validation: 93 RNG tests passed on both platforms. Native headers also passed C99/C++20 compilation and a two-translation-unit link check; the C99 run passed AddressSanitizer and UndefinedBehaviorSanitizer. The implementation uses scalar kernels and host `libm`.
