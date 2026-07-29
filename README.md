# mojo-prophet

`mojo-prophet` is a standalone Mojo port of the compute-heavy core of
[Prophet](https://github.com/facebook/prophet). It provides fast Fourier
seasonality generation, piecewise trend evaluation, and deterministic fitting
for additive trend/seasonality models behind a Python API with Prophet's core
names and signatures.

This is a useful covered subset, not a replacement for every Prophet feature.
Import `Prophet` from `mojo_prophet` when the deterministic additive model is
appropriate.

## Coverage

Covered:

- `Prophet(...)`, `fit`, `predict`, and `make_future_dataframe`
- linear and flat trend fitting with automatic or explicit changepoints
- automatic daily, weekly, and yearly Fourier seasonalities
- custom and conditional seasonalities through `add_seasonality`
- supplied holiday dataframes, including lower and upper windows
- standardized additive regressors through `add_regressor`
- point forecasts, component columns, and residual-normal intervals
- upstream-compatible `fourier_series`, `make_seasonality_features`,
  `piecewise_linear`, `piecewise_logistic`, and `flat_trend`

Not covered:

- Stan MAP optimization, MCMC, or posterior predictive intervals
- logistic growth fitting, although logistic trend evaluation is accelerated
- multiplicative seasonalities or regressors
- country holiday generation, plotting, diagnostics, cross-validation, and
  serialization helpers
- floors, caps, and saturating-minimum models

The fitting distinction matters. Upstream Prophet uses a Laplace prior for
changepoint deltas and performs inference in Stan. This port uses deterministic
penalized least squares with an L2 changepoint penalty. The two agree on the
covered feature and trend equations, but a fitted noisy series with flexible
changepoints will not generally produce identical parameters.

## Install

The repository pins the tested Mojo nightly and includes Prophet 1.3.0 as its
parity-test dependency.

```bash
pixi install
pixi run build
pixi run test
```

The activation environment puts `python/` on `PYTHONPATH`. The build creates
`dist/libmojo-prophet.so`.

## Usage

```python
import numpy as np
import pandas as pd
from mojo_prophet import Prophet

dates = pd.date_range("2022-01-01", periods=730, freq="D")
day = np.arange(len(dates))
history = pd.DataFrame({
    "ds": dates,
    "y": 20 + 0.01 * day + 2 * np.sin(2 * np.pi * day / 7),
})

model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=False,
    daily_seasonality=False,
    uncertainty_samples=0,
).fit(history)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
print(forecast[["ds", "trend", "weekly", "yhat"]].tail())
```

Custom seasonalities use Prophet's signature:

```python
model = Prophet(weekly_seasonality=False)
model.add_seasonality(
    name="monthly",
    period=30.5,
    fourier_order=5,
    prior_scale=10.0,
)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 machine running
Linux 6.8.0. Times are the best of five runs, except the solve benchmark,
which uses three. The exact kernel references are Prophet 1.3.0; the ridge
reference is NumPy 2.5.1 using the same prebuilt row-major design matrix.

| case | mojo-prophet | reference | ratio |
|---|---:|---:|---:|
| Fourier series (250k, order 10) | 107.80 ms | 196.60 ms | 1.82x faster |
| Piecewise linear (300k, 25 cp) | 8.96 ms | 147.25 ms | 16.43x faster |
| Piecewise logistic (300k, 25 cp) | 17.56 ms | 43.80 ms | 2.49x faster |
| Ridge solve+predict (100k, 55 cols) | 36.93 ms | 65.38 ms | 1.77x faster |

Large ridge fits split the rows into CPU chunks, form thread-private Gram
matrices without atomics, and reduce those matrices before Cholesky
factorization. Smaller fits remain serial to avoid thread-launch overhead. The
trend kernels avoid Prophet's large temporary arrays and are substantially
faster on this workload.

No GPU path is shipped.

## How it works

All Mojo kernels live in one compilation unit and are exported through a C ABI.
The Python layer owns every allocation as a C-contiguous NumPy `float64` array
and passes its address and dimensions through `ctypes`; Mojo never owns Python
memory and performs no cross-FFI allocation.

Feature and design matrices are row-major. Fourier pairs are stored
`sin(1), cos(1), sin(2), cos(2), ...`, matching Prophet exactly. Linear trends
use hinge columns `max(0, t - changepoint)`. Fitting forms `X.T @ X` and
`X.T @ y` in SIMD row chunks for large inputs, adds per-column prior penalties,
solves the symmetric system with an in-place Cholesky factorization, and
evaluates large forecasts with a parallel SIMD matrix-vector kernel. SIMD loops
use the target's native width and scalar tails, so dimensions need not be
multiples of the vector width.

## License

MIT
