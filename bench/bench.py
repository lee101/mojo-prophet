"""Honest same-process benchmarks against upstream Prophet and NumPy."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_prophet import Prophet as MojoProphet  # noqa: E402
from mojo_prophet._lib import addr, lib  # noqa: E402
from prophet import Prophet as PythonProphet  # noqa: E402
import prophet  # noqa: E402


def timeit(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def benchmark(name, ours, reference, repeat=5):
    actual = ours()
    expected = reference()
    if not np.allclose(actual, expected, atol=2e-10, rtol=1e-12):
        raise AssertionError(f"{name} benchmark inputs do not produce parity")
    ours_time = timeit(ours, repeat)
    reference_time = timeit(reference, repeat)
    return name, ours_time, reference_time


def main():
    MojoProphet.fourier_series(pd.Series(pd.date_range("2020-01-01", periods=2)), 7, 3)
    rows = []

    dates = pd.Series(pd.date_range("2000-01-01", periods=250_000, freq="h"))
    rows.append(
        benchmark(
            "Fourier series (250k, order 10)",
            lambda: MojoProphet.fourier_series(dates, 365.25, 10),
            lambda: PythonProphet.fourier_series(dates, 365.25, 10),
        )
    )

    rng = np.random.default_rng(0)
    t = np.linspace(0, 2, 300_000)
    changepoints = np.linspace(0.05, 0.8, 25)
    deltas = rng.normal(0, 0.03, 25)
    rows.append(
        benchmark(
            "Piecewise linear (300k, 25 cp)",
            lambda: MojoProphet.piecewise_linear(t, deltas, 0.7, 0.2, changepoints),
            lambda: PythonProphet.piecewise_linear(t, deltas, 0.7, 0.2, changepoints),
        )
    )

    cap = np.full_like(t, 10.0)
    rows.append(
        benchmark(
            "Piecewise logistic (300k, 25 cp)",
            lambda: MojoProphet.piecewise_logistic(t, cap, deltas, 1.7, 0.2, changepoints),
            lambda: PythonProphet.piecewise_logistic(t, cap, deltas, 1.7, 0.2, changepoints),
        )
    )

    n = 100_000
    dates_fit = pd.date_range("2000-01-01", periods=n, freq="h")
    day = np.arange(n) / 24
    y = 5 + 0.01 * day + 2 * np.sin(2 * np.pi * day / 7)
    frame = pd.DataFrame({"ds": dates_fit, "y": y})

    model = MojoProphet(
        n_changepoints=25,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        uncertainty_samples=0,
    ).fit(frame)
    design, metadata = model._design(frame)
    scale = np.max(np.abs(y))
    y_scaled = np.ascontiguousarray(y / scale)
    penalties = np.array(
        [0.0 if np.isinf(m["prior_scale"]) else 1 / m["prior_scale"] ** 2 for m in metadata]
    )
    penalties[:2] = 1e-12

    def mojo_solve():
        coef = np.empty(design.shape[1])
        work = np.empty((design.shape[1], design.shape[1]))
        chunks = min(32, os.cpu_count() or 1)
        scratch = np.empty(chunks * (design.shape[1] ** 2 + design.shape[1]))
        prediction = np.empty(len(y))
        ok = lib().mop_ridge_fit_parallel(
            addr(design),
            addr(y_scaled),
            addr(penalties),
            addr(coef),
            addr(work),
            addr(scratch),
            design.shape[0],
            design.shape[1],
            chunks,
        )
        if not ok:
            raise RuntimeError("Mojo ridge solve failed")
        if not lib().mop_matvec(
            addr(design), addr(coef), addr(prediction), design.shape[0], design.shape[1]
        ):
            raise RuntimeError("Mojo matrix-vector kernel rejected benchmark arguments")
        return prediction * scale

    def numpy_solve():
        coef = np.linalg.solve(
            design.T @ design + np.diag(penalties),
            design.T @ y_scaled,
        )
        return design @ coef * scale

    rows.append(
        benchmark(
            "Ridge solve+predict (100k, 55 cols)",
            mojo_solve,
            numpy_solve,
            repeat=3,
        )
    )

    cpu = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in cpuinfo
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        cpu = platform.processor() or platform.machine()
    print(f"Machine: {cpu} | {platform.platform()}")
    print(f"Versions: Prophet {prophet.__version__} | NumPy {np.__version__}")
    print("Timing: best of 5 runs; ridge solve+predict best of 3")
    print()
    print("| case | mojo-prophet | reference | ratio |")
    print("|---|---:|---:|---:|")
    for name, ours_time, ref_time in rows:
        label = "faster" if ours_time < ref_time else "slower"
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | {ref_time * 1e3:.2f} ms "
            f"| {ref_time / ours_time:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
