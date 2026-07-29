import numpy as np
import pandas as pd
import pytest

from mojo_prophet import Prophet
from mojo_prophet._lib import addr, f64, lib
from mojo_prophet.forecaster import _ridge_chunks

upstream = pytest.importorskip("prophet")
UpstreamProphet = upstream.Prophet


@pytest.mark.parametrize(
    ("period", "order"),
    [(7.0, 3), (365.25, 10), (1.0, 4)],
)
def test_fourier_series_matches_upstream(period, order):
    dates = pd.Series(pd.date_range("2019-11-03", periods=100, freq="3h"))
    actual = Prophet.fourier_series(dates, period, order)
    expected = UpstreamProphet.fourier_series(dates, period, order)
    assert np.allclose(actual, expected, atol=2e-11, rtol=0)


def test_fourier_series_timezone_matches_upstream():
    dates = pd.Series(pd.date_range("2024-01-01", periods=40, freq="6h", tz="US/Pacific"))
    actual = Prophet.fourier_series(dates, 7, 3)
    expected = UpstreamProphet.fourier_series(dates, 7, 3)
    assert np.allclose(actual, expected, atol=2e-11, rtol=0)


def test_make_seasonality_features_matches_upstream():
    dates = pd.Series(pd.date_range("2020-01-01", periods=20))
    actual = Prophet.make_seasonality_features(dates, 30.5, 5, "monthly")
    expected = UpstreamProphet.make_seasonality_features(dates, 30.5, 5, "monthly")
    pd.testing.assert_index_equal(actual.columns, expected.columns)
    assert np.allclose(actual, expected, atol=2e-11, rtol=0)


@pytest.mark.parametrize("size", [77, 101])
def test_piecewise_linear_matches_upstream(size):
    t = np.linspace(-0.2, 2.0, size)
    changepoints = np.array([0.0, 0.4, 1.2, 1.8])
    deltas = np.array([0.1, 0.3, -0.7, 0.2])
    actual = Prophet.piecewise_linear(t, deltas, 2.0, 0.5, changepoints)
    expected = UpstreamProphet.piecewise_linear(t, deltas, 2.0, 0.5, changepoints)
    assert np.allclose(actual, expected, atol=2e-12, rtol=0)


def test_piecewise_logistic_matches_upstream():
    t = np.linspace(-0.2, 2.0, 201)
    cap = np.linspace(8.0, 12.0, len(t))
    changepoints = np.array([0.0, 0.4, 1.2, 1.8])
    deltas = np.array([0.1, 0.3, -0.7, 0.2])
    actual = Prophet.piecewise_logistic(t, cap, deltas, 2.0, 0.5, changepoints)
    expected = UpstreamProphet.piecewise_logistic(t, cap, deltas, 2.0, 0.5, changepoints)
    assert np.allclose(actual, expected, atol=2e-12, rtol=0)


def test_flat_trend_matches_upstream():
    t = np.linspace(0, 1, 17)
    assert np.array_equal(Prophet.flat_trend(t, 3.25), UpstreamProphet.flat_trend(t, 3.25))


def test_contiguous_float64_input_stays_zero_copy():
    values = np.arange(17, dtype=np.float64)
    assert f64(values) is values


def test_noncontiguous_and_float32_inputs_are_copied_to_contiguous_float64():
    values = np.arange(20, dtype=np.float32)[::2]
    converted = f64(values)
    assert converted.dtype == np.float64
    assert converted.flags.c_contiguous
    assert not np.shares_memory(converted, values)


def test_addr_rejects_buffers_with_unsafe_dtype_or_strides():
    with pytest.raises(ValueError, match="C-contiguous float64"):
        addr(np.arange(4, dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous float64"):
        addr(np.arange(8, dtype=np.float64)[::2])


def test_empty_kernel_inputs_return_without_null_pointer_construction():
    assert Prophet.fourier_series(pd.Series([], dtype="datetime64[ns]"), 7, 3).shape == (0, 6)
    assert Prophet.piecewise_linear(np.array([]), np.array([]), 2, 1, np.array([])).size == 0
    assert Prophet.piecewise_logistic(
        np.array([]), np.array([]), np.array([]), 2, 1, np.array([])
    ).size == 0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: Prophet.piecewise_linear(
                np.arange(3), np.ones(2), 1, 0, np.ones(1)
            ),
            "same length",
        ),
        (
            lambda: Prophet.piecewise_logistic(
                np.arange(3), np.ones(2), np.array([]), 1, 0, np.array([])
            ),
            "cap and t",
        ),
    ],
)
def test_trend_kernels_reject_length_mismatches_before_ffi(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_exported_kernels_reject_invalid_pointers_and_dimensions():
    assert lib().mop_fourier_series(0, 0, 1, 7.0, 3) == 0
    assert lib().mop_matvec(0, 0, 0, 1, 3) == 0
    assert lib().mop_component(0, 0, 0, 1, 3, 2, 2) == 0


@pytest.mark.parametrize("n", [31, 142_858])
def test_matvec_simd_tail_on_both_parallel_sides(n):
    rng = np.random.default_rng(12)
    d = 7
    x = rng.normal(size=(n, d))
    coef = rng.normal(size=d)
    actual = np.empty(n)
    assert lib().mop_matvec(addr(x), addr(coef), addr(actual), n, d) == 1
    assert np.allclose(actual, x @ coef, atol=2e-12, rtol=1e-12)


def test_parallel_ridge_simd_tail_matches_numpy():
    rng = np.random.default_rng(21)
    n, d, chunks = 5_003, 7, 8
    x = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    penalty = np.linspace(0.1, 0.7, d)
    coef = np.empty(d)
    work = np.empty((d, d))
    scratch = np.empty(chunks * (d * d + d))
    ok = lib().mop_ridge_fit_parallel(
        addr(x),
        addr(y),
        addr(penalty),
        addr(coef),
        addr(work),
        addr(scratch),
        n,
        d,
        chunks,
    )
    expected = np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ y)
    assert ok == 1
    assert np.allclose(coef, expected, atol=2e-11, rtol=2e-11)


def test_parallel_ridge_threshold():
    d = 7
    boundary = (2_000_000 + d * (d + 1) // 2 - 1) // (d * (d + 1) // 2)
    assert _ridge_chunks(boundary - 1, d) == 0
    assert _ridge_chunks(boundary, d) > 0


def synthetic_frame(n=500):
    dates = pd.date_range("2020-01-01", periods=n)
    day = np.arange(n, dtype=np.float64)
    y = (
        10.0
        + 0.03 * day
        + 2.0 * np.sin(2 * np.pi * day / 7)
        + 0.5 * np.cos(2 * np.pi * day / 30.5)
    )
    return pd.DataFrame({"ds": dates, "y": y})


def test_additive_fit_recovers_known_signal():
    frame = synthetic_frame()
    model = Prophet(
        n_changepoints=0,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,
    ).add_seasonality("monthly", 30.5, 3)
    forecast = model.fit(frame).predict(frame)
    assert np.sqrt(np.mean((forecast["yhat"] - frame["y"]) ** 2)) < 1e-4
    assert np.max(np.abs(forecast["trend"] - (10 + 0.03 * np.arange(len(frame))))) < 1e-4
    assert np.allclose(
        forecast["yhat"],
        forecast["trend"] + forecast["additive_terms"],
        atol=1e-12,
    )


def test_fit_matches_numpy_penalized_reference():
    frame = synthetic_frame(200)
    model = Prophet(
        n_changepoints=0,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=0,
    ).fit(frame)
    t = np.linspace(0.0, 1.0, len(frame))
    fourier = Prophet.fourier_series(frame["ds"], 7, 3)
    design = np.column_stack([np.ones(len(frame)), t, fourier])
    scale = np.max(np.abs(frame["y"]))
    penalty = np.diag([1e-12, 1e-12] + [1 / 10.0**2] * 6)
    expected_coef = np.linalg.solve(design.T @ design + penalty, design.T @ (frame["y"] / scale))
    expected = design @ expected_coef * scale
    assert np.allclose(model.predict(frame)["yhat"], expected, atol=1e-10)


def test_explicit_changepoint_fit():
    n = 300
    dates = pd.date_range("2023-01-01", periods=n)
    day = np.arange(n)
    y = 4 + 0.02 * day + 0.06 * np.maximum(0, day - 120)
    frame = pd.DataFrame({"ds": dates, "y": y})
    model = Prophet(
        changepoints=[dates[120]],
        changepoint_prior_scale=1e6,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    ).fit(frame)
    assert np.max(np.abs(model.predict(frame)["yhat"] - y)) < 1e-8
    assert model.params["delta"].shape == (1, 1)


def test_conditional_seasonality():
    n = 140
    dates = pd.date_range("2023-01-01", periods=n)
    active = np.arange(n) < 70
    y = 5 + active * np.sin(2 * np.pi * np.arange(n) / 7)
    frame = pd.DataFrame({"ds": dates, "y": y, "active": active})
    model = Prophet(
        n_changepoints=0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    )
    model.add_seasonality("conditional", 7, 3, condition_name="active").fit(frame)
    forecast = model.predict(frame)
    assert np.max(np.abs(forecast["conditional"][~active])) == 0
    assert np.sqrt(np.mean((forecast["yhat"] - y) ** 2)) < 2e-4


def test_extra_regressor_and_standardization():
    n = 120
    rng = np.random.default_rng(4)
    x = rng.normal(10, 3, n)
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2023-01-01", periods=n),
            "x": x,
            "y": 2 + 1.5 * x,
        }
    )
    model = Prophet(
        growth="flat",
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    ).add_regressor("x", prior_scale=1e6)
    forecast = model.fit(frame).predict(frame)
    assert np.max(np.abs(forecast["yhat"] - frame["y"])) < 1e-8
    assert model.extra_regressors["x"]["mu"] == pytest.approx(x.mean())


def test_holiday_effect():
    dates = pd.date_range("2022-01-01", periods=100)
    holidays = pd.DataFrame(
        {
            "holiday": ["launch"],
            "ds": [dates[50]],
            "lower_window": [-1],
            "upper_window": [1],
        }
    )
    y = np.full(100, 3.0)
    y[49:52] += 4
    frame = pd.DataFrame({"ds": dates, "y": y})
    model = Prophet(
        growth="flat",
        holidays=holidays,
        holidays_prior_scale=1e6,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    ).fit(frame)
    forecast = model.predict(frame)
    assert np.max(np.abs(forecast["yhat"] - y)) < 1e-8
    assert np.allclose(forecast.loc[49:51, "launch"], 4.0, atol=1e-8)


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (pd.date_range("2020-01-01", periods=731, freq="D"), "yearly"),
        (pd.date_range("2024-01-01", periods=30, freq="D"), "weekly"),
        (pd.date_range("2024-01-01", periods=73, freq="h"), "daily"),
    ],
)
def test_automatic_seasonalities_are_enabled_when_history_supports_them(
    dates, expected
):
    model = Prophet(n_changepoints=0, uncertainty_samples=0)
    model.fit(pd.DataFrame({"ds": dates, "y": np.arange(len(dates))}))
    assert expected in model.seasonalities


def test_automatic_changepoints_are_created_in_configured_range():
    frame = synthetic_frame(100)
    model = Prophet(
        n_changepoints=5,
        changepoint_range=0.8,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    ).fit(frame)
    assert len(model.changepoints) == 5
    assert model.changepoints.max() <= frame["ds"].iloc[79]


def test_make_future_dataframe():
    frame = synthetic_frame(30)
    model = Prophet(
        n_changepoints=0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    ).fit(frame)
    full = model.make_future_dataframe(5)
    future = model.make_future_dataframe(5, include_history=False)
    assert len(full) == 35
    assert len(future) == 5
    assert future["ds"].iloc[0] == frame["ds"].iloc[-1] + pd.Timedelta(days=1)


def test_prediction_intervals_contain_point_forecast():
    frame = synthetic_frame(80)
    forecast = Prophet(
        n_changepoints=0,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    ).fit(frame).predict(frame)
    assert (forecast["yhat_lower"] <= forecast["yhat"]).all()
    assert (forecast["yhat"] <= forecast["yhat_upper"]).all()


@pytest.mark.parametrize("bad_y", [[1.0, np.inf], [1.0, "not numeric"]])
def test_fit_rejects_nonfinite_or_nonnumeric_y(bad_y):
    frame = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=2), "y": bad_y}
    )
    with pytest.raises(ValueError, match="y must"):
        Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
        ).fit(frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mcmc_samples": 10},
        {"growth": "logistic"},
        {"seasonality_mode": "multiplicative"},
    ],
)
def test_unsupported_fit_modes_are_explicit(kwargs):
    with pytest.raises(NotImplementedError):
        Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            **kwargs,
        ).fit(synthetic_frame(20))
