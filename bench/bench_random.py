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
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path)
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
        "results": benchmark(args.size, args.repeats),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
