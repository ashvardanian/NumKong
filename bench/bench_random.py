"""Compare native Generator array calls with NumPy's default PCG64 Generator."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numkong as nk
import numpy as np


def benchmark_uniform_f32(size, repeats):
    native = nk.Generator(137, array_factory=np.empty)
    reference = np.random.default_rng(137)

    def native_uniform():
        return native.uniform(-1.0, 1.0, size=size, dtype=np.float32)

    def numpy_cast():
        return reference.uniform(-1.0, 1.0, size=size).astype(np.float32)

    def numpy_inplace():
        values = reference.random(size, dtype=np.float32)
        np.multiply(values, np.float32(2.0), out=values)
        np.add(values, np.float32(-1.0), out=values)
        return values

    calls = [("numkong", native_uniform), ("numpy_cast", numpy_cast), ("numpy_inplace", numpy_inplace)]
    samples = {name: [] for name, _ in calls}
    for _, fill in calls:
        values = fill()
        assert values.shape == (size,) and values.dtype == np.float32
        assert np.all((values >= -1.0) & (values <= 1.0))
    for iteration in range(repeats + 3):
        offset = iteration % len(calls)
        for name, fill in calls[offset:] + calls[:offset]:
            start = time.perf_counter_ns()
            values = fill()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if iteration >= 3:
                samples[name].append(elapsed)
            del values
    native_ms = statistics.median(samples["numkong"])
    results = []
    for name, baseline in (
        ("numpy_cast", "uniform(-1, 1, size).astype(float32)"),
        ("numpy_inplace", "random(size, dtype=float32), multiply(out=), add(out=)"),
    ):
        numpy_ms = statistics.median(samples[name])
        results.append(
            {
                "workload": "uniform_f32_" + name.removeprefix("numpy_"),
                "numpy_baseline": baseline,
                "numkong_ms": native_ms,
                "numpy_ms": numpy_ms,
                "numpy_ms_over_numkong_ms": numpy_ms / native_ms,
            }
        )
    return results


def benchmark(size, repeats):
    native = nk.Generator(137, array_factory=np.empty)
    reference = np.random.default_rng(137)
    results = []
    cases = [
        ("random_f32", "random", {"size": size, "dtype": np.float32}),
        ("random_f64", "random", {"size": size}),
        ("uniform_f64", "uniform", {"low": -1.0, "high": 1.0, "size": size}),
        ("normal_f64", "normal", {"loc": 1.0, "scale": 2.0, "size": size}),
        ("standard_normal_f32", "standard_normal", {"size": size, "dtype": np.float32}),
        ("standard_normal_f64", "standard_normal", {"size": size}),
        ("integers_u8", "integers", {"low": 0, "high": 256, "size": size, "dtype": np.uint8}),
        ("integers_i32", "integers", {"low": -1000, "high": 1000, "size": size, "dtype": np.int32}),
        ("laplace_f64", "laplace", {"loc": 0.0, "scale": 1.0, "size": size}),
        ("beta_f64", "beta", {"a": 2.0, "b": 3.0, "size": size}),
        ("poisson_small", "poisson", {"lam": 2.0, "size": size}),
        ("poisson_large", "poisson", {"lam": 50.0, "size": size}),
        ("choice_sparse", "choice", {"a": size * 10, "size": size // 100, "replace": False}),
    ]
    for name, method, kwargs in cases:
        samples = {"numkong": [], "numpy": []}
        calls = [("numkong", getattr(native, method)), ("numpy", getattr(reference, method))]
        for iteration in range(repeats + 3):
            for label, call in calls[:: -1 if iteration % 2 else 1]:
                start = time.perf_counter_ns()
                result = call(**kwargs)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if iteration >= 3:
                    samples[label].append(elapsed)
                del result
        row = {"workload": name, **{label + "_ms": statistics.median(times) for label, times in samples.items()}}
        row["numpy_ms_over_numkong_ms"] = row["numpy_ms"] / row["numkong_ms"]
        results.append(row)
    for label, generator in (("numkong", native), ("numpy", reference)):
        values = np.arange(size, dtype=np.int64)
        samples = []
        for iteration in range(repeats + 3):
            start = time.perf_counter_ns()
            generator.shuffle(values)
            if iteration >= 3:
                samples.append((time.perf_counter_ns() - start) / 1e6)
        if label == "numkong":
            row = {"workload": "shuffle_i64"}
        row[label + "_ms"] = statistics.median(samples)
    row["numpy_ms_over_numkong_ms"] = row["numpy_ms"] / row["numkong_ms"]
    results.append(row)
    results.extend(benchmark_uniform_f32(size, repeats))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--uniform-f32-only",
        action="store_true",
        help="Measure bounded float32 uniform against cast and in-place NumPy routes",
    )
    args = parser.parse_args()
    if args.size < 100 or args.repeats < 1:
        parser.error("size must be >= 100 and repeats must be >= 1")
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    elif Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numkong": nk.__version__,
        "cpu": cpu,
        "platform": platform.platform(),
        "size": args.size,
        "repeats": args.repeats,
        "boundary": "Python call with allocation; native array_factory=numpy.empty; both single-threaded",
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ("include/numkong/random.h", "include/numkong/random/ziggurat.h", "python/random.c")
        },
        "results": (
            benchmark_uniform_f32(args.size, args.repeats)
            if args.uniform_f32_only
            else benchmark(args.size, args.repeats)
        ),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
