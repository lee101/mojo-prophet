"""A deterministic additive subset of Prophet accelerated by Mojo."""

from __future__ import annotations

from collections import OrderedDict
from datetime import timedelta
import math
import os
from statistics import NormalDist
from typing import Literal, SupportsFloat

import numpy as np
import pandas as pd

from ._lib import addr, f64, lib

_PARALLEL_RIDGE_MIN_UPDATES = 2_000_000
_RIDGE_CHUNKS = min(32, os.cpu_count() or 1)


def _ridge_chunks(n: int, d: int) -> int:
    if n * d * (d + 1) // 2 < _PARALLEL_RIDGE_MIN_UPDATES:
        return 0
    return min(_RIDGE_CHUNKS, n)


class Prophet:
    """Prophet-compatible additive trend and seasonality forecaster.

    Fitting uses penalized least squares rather than Prophet's Stan backend.
    The covered model is deterministic and intended for fast additive fits.
    """

    def __init__(
        self,
        growth: Literal["linear", "logistic", "flat"] = "linear",
        changepoints: pd.Series | list[pd.Timestamp] | None = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: Literal["auto"] | int = "auto",
        weekly_seasonality: Literal["auto"] | int = "auto",
        daily_seasonality: Literal["auto"] | int = "auto",
        holidays: pd.DataFrame | None = None,
        seasonality_mode: Literal["additive", "multiplicative"] = "additive",
        seasonality_prior_scale: SupportsFloat = 10.0,
        holidays_prior_scale: SupportsFloat = 10.0,
        changepoint_prior_scale: SupportsFloat = 0.05,
        mcmc_samples: int = 0,
        interval_width: float = 0.80,
        uncertainty_samples: int = 1000,
        stan_backend: str | None = None,
        scaling: Literal["absmax", "minmax"] = "absmax",
        holidays_mode: Literal["additive", "multiplicative"] | None = None,
    ) -> None:
        if growth not in ("linear", "logistic", "flat"):
            raise ValueError('Parameter "growth" should be "linear", "logistic" or "flat".')
        if not 0 <= changepoint_range <= 1:
            raise ValueError('Parameter "changepoint_range" must be in [0, 1]')
        if seasonality_mode not in ("additive", "multiplicative"):
            raise ValueError('seasonality_mode must be "additive" or "multiplicative"')
        if scaling not in ("absmax", "minmax"):
            raise ValueError("scaling must be one of 'absmax' or 'minmax'")
        if int(n_changepoints) < 0:
            raise ValueError("n_changepoints must be >= 0")
        if not 0 < float(interval_width) < 1:
            raise ValueError("interval_width must be between 0 and 1")
        for name, value in (
            ("seasonality_prior_scale", seasonality_prior_scale),
            ("holidays_prior_scale", holidays_prior_scale),
            ("changepoint_prior_scale", changepoint_prior_scale),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        self.growth = growth
        self.changepoints = (
            pd.Series(pd.to_datetime(changepoints), name="ds")
            if changepoints is not None
            else None
        )
        self.specified_changepoints = changepoints is not None
        self.n_changepoints = (
            len(self.changepoints) if self.specified_changepoints else int(n_changepoints)
        )
        self.changepoint_range = float(changepoint_range)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays.copy() if holidays is not None else None
        self.seasonality_mode = seasonality_mode
        self.holidays_mode = holidays_mode or seasonality_mode
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.mcmc_samples = int(mcmc_samples)
        self.interval_width = float(interval_width)
        self.uncertainty_samples = int(uncertainty_samples)
        self.stan_backend = stan_backend
        self.scaling = scaling
        self.seasonalities: OrderedDict[str, dict] = OrderedDict()
        self.extra_regressors: OrderedDict[str, dict] = OrderedDict()
        self.history: pd.DataFrame | None = None
        self.history_dates: pd.Series | None = None
        self.params: dict[str, np.ndarray] = {}

    @staticmethod
    def _days(dates: pd.Series) -> np.ndarray:
        dates = pd.to_datetime(dates)
        epoch = pd.Timestamp("1970-01-01", tz=dates.dt.tz)
        return f64((dates - epoch).dt.total_seconds().to_numpy() / 86400.0)

    @staticmethod
    def fourier_series(
        dates: pd.Series, period: float, series_order: int
    ) -> np.ndarray:
        if not isinstance(series_order, (int, np.integer)) or series_order < 1:
            raise ValueError("series_order must be >= 1")
        if not math.isfinite(period) or period <= 0:
            raise ValueError("period must be a finite positive number")
        days = Prophet._days(pd.Series(dates))
        result = np.empty((len(days), 2 * series_order), dtype=np.float64)
        if len(days) and not lib().mop_fourier_series(
            addr(days), addr(result), len(days), float(period), int(series_order)
        ):
            raise RuntimeError("Mojo Fourier kernel rejected its arguments")
        return result

    @classmethod
    def make_seasonality_features(
        cls,
        dates: pd.Series,
        period: float,
        series_order: int,
        prefix: str,
    ) -> pd.DataFrame:
        values = cls.fourier_series(dates, period, series_order)
        columns = [f"{prefix}_delim_{i + 1}" for i in range(values.shape[1])]
        return pd.DataFrame(values, columns=columns)

    @staticmethod
    def piecewise_linear(
        t: np.ndarray,
        deltas: np.ndarray,
        k: float,
        m: float,
        changepoint_ts: np.ndarray,
    ) -> np.ndarray:
        t = f64(t)
        deltas = f64(deltas)
        changepoint_ts = f64(changepoint_ts)
        if t.ndim != 1 or deltas.ndim != 1 or changepoint_ts.ndim != 1:
            raise ValueError("t, deltas, and changepoint_ts must be one-dimensional")
        if deltas.size != changepoint_ts.size:
            raise ValueError("deltas and changepoint_ts must have the same length")
        result = np.empty(t.shape, dtype=np.float64)
        if t.size and not lib().mop_piecewise_linear(
            addr(t),
            addr(deltas),
            addr(changepoint_ts),
            addr(result),
            t.size,
            deltas.size,
            float(k),
            float(m),
        ):
            raise RuntimeError("Mojo piecewise-linear kernel rejected its arguments")
        return result

    @staticmethod
    def piecewise_logistic(
        t: np.ndarray,
        cap: np.ndarray | pd.Series,
        deltas: np.ndarray,
        k: float,
        m: float,
        changepoint_ts: np.ndarray,
    ) -> np.ndarray:
        t = f64(t)
        cap = f64(cap)
        deltas = f64(deltas)
        changepoint_ts = f64(changepoint_ts)
        if any(value.ndim != 1 for value in (t, cap, deltas, changepoint_ts)):
            raise ValueError("t, cap, deltas, and changepoint_ts must be one-dimensional")
        if cap.size != t.size:
            raise ValueError("cap and t must have the same length")
        if deltas.size != changepoint_ts.size:
            raise ValueError("deltas and changepoint_ts must have the same length")
        result = np.empty(t.shape, dtype=np.float64)
        gamma = np.empty(max(1, deltas.size), dtype=np.float64)
        if t.size and not lib().mop_piecewise_logistic(
            addr(t),
            addr(cap),
            addr(deltas),
            addr(changepoint_ts),
            addr(result),
            addr(gamma),
            t.size,
            deltas.size,
            float(k),
            float(m),
        ):
            raise RuntimeError("Mojo piecewise-logistic kernel rejected its arguments")
        return result

    @staticmethod
    def flat_trend(t: np.ndarray, m: float) -> np.ndarray:
        return np.full_like(np.asarray(t, dtype=np.float64), float(m))

    def _validate_name(self, name: str) -> None:
        if "_delim_" in name:
            raise ValueError('Name cannot contain "_delim_"')
        if name in self.seasonalities or name in self.extra_regressors:
            raise ValueError(f"Name {name!r} already used.")

    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: float | None = None,
        mode: Literal["additive", "multiplicative"] | None = None,
        condition_name: str | None = None,
    ) -> "Prophet":
        if self.history is not None:
            raise Exception("Seasonality must be added prior to model fitting.")
        if name not in ("daily", "weekly", "yearly"):
            self._validate_name(name)
        if fourier_order <= 0:
            raise ValueError("Fourier Order must be > 0")
        ps = self.seasonality_prior_scale if prior_scale is None else float(prior_scale)
        if ps <= 0:
            raise ValueError("Prior scale must be > 0")
        mode = mode or self.seasonality_mode
        if mode not in ("additive", "multiplicative"):
            raise ValueError('mode must be "additive" or "multiplicative"')
        self.seasonalities[name] = {
            "period": float(period),
            "fourier_order": int(fourier_order),
            "prior_scale": ps,
            "mode": mode,
            "condition_name": condition_name,
        }
        return self

    def add_regressor(
        self,
        name: str,
        prior_scale: float | None = None,
        standardize: Literal["auto"] | bool = "auto",
        mode: Literal["additive", "multiplicative"] | None = None,
    ) -> "Prophet":
        if self.history is not None:
            raise Exception("Regressors must be added prior to model fitting.")
        self._validate_name(name)
        ps = self.holidays_prior_scale if prior_scale is None else float(prior_scale)
        mode = mode or self.seasonality_mode
        if ps <= 0:
            raise ValueError("Prior scale must be > 0")
        if mode not in ("additive", "multiplicative"):
            raise ValueError("mode must be 'additive' or 'multiplicative'")
        self.extra_regressors[name] = {
            "prior_scale": ps,
            "standardize": standardize,
            "mu": 0.0,
            "std": 1.0,
            "mode": mode,
        }
        return self

    def _auto_seasonalities(self, dates: pd.Series) -> None:
        span = dates.max() - dates.min()
        diffs = dates.sort_values().diff().dropna()
        min_dt = diffs[diffs > pd.Timedelta(0)].min()
        rules = (
            ("yearly", self.yearly_seasonality, 365.25, 10, span < pd.Timedelta(days=730)),
            (
                "weekly",
                self.weekly_seasonality,
                7.0,
                3,
                span < pd.Timedelta(weeks=2) or min_dt >= pd.Timedelta(weeks=1),
            ),
            (
                "daily",
                self.daily_seasonality,
                1.0,
                4,
                span < pd.Timedelta(days=2) or min_dt >= pd.Timedelta(days=1),
            ),
        )
        for name, setting, period, default, disabled in rules:
            if name in self.seasonalities:
                continue
            order = 0 if setting == "auto" and disabled else default if setting is True or setting == "auto" else int(setting)
            if order:
                self.add_seasonality(name, period, order)

    def _set_changepoints(self, dates: pd.Series) -> None:
        if self.changepoints is None:
            hist_size = int(np.floor(len(dates) * self.changepoint_range))
            count = min(self.n_changepoints, max(0, hist_size - 1))
            if count:
                indexes = np.linspace(0, hist_size - 1, count + 1).round().astype(int)[1:]
                self.changepoints = dates.iloc[indexes].reset_index(drop=True)
            else:
                self.changepoints = pd.Series(pd.to_datetime([]), name="ds")
            self.n_changepoints = count
        elif len(self.changepoints):
            if self.changepoints.min() < dates.min() or self.changepoints.max() > dates.max():
                raise ValueError("Changepoints must fall within training data.")
        self.changepoints_t = (
            f64((self.changepoints - self.start) / self.t_scale)
            if len(self.changepoints)
            else np.empty(0, dtype=np.float64)
        )

    def _holiday_features(self, dates: pd.Series) -> tuple[list[np.ndarray], list[dict]]:
        arrays: list[np.ndarray] = []
        metadata: list[dict] = []
        if self.holidays is None or self.holidays.empty:
            return arrays, metadata
        holidays = self.holidays.copy()
        if not {"ds", "holiday"} <= set(holidays):
            raise ValueError('holidays must have "ds" and "holiday" columns.')
        holidays["ds"] = pd.to_datetime(holidays["ds"])
        for name in sorted(holidays["holiday"].unique()):
            group = holidays[holidays["holiday"] == name]
            if name in self.seasonalities or name in self.extra_regressors:
                raise ValueError(f"Holiday name {name!r} is already used by another component")
            prior_scale = float(
                group.get(
                    "prior_scale", pd.Series([self.holidays_prior_scale])
                ).iloc[0]
            )
            if not math.isfinite(prior_scale) or prior_scale <= 0:
                raise ValueError(f"Holiday {name!r} prior_scale must be finite and positive")
            offsets = set()
            for row in group.itertuples():
                low = int(getattr(row, "lower_window", 0))
                high = int(getattr(row, "upper_window", 0))
                for offset in range(low, high + 1):
                    offsets.add(offset)
            for offset in sorted(offsets):
                active = {
                    day + timedelta(days=offset)
                    for day in group["ds"].dt.normalize().tolist()
                }
                arrays.append(dates.dt.normalize().isin(active).to_numpy(dtype=np.float64)[:, None])
                metadata.append(
                    {
                        "name": name,
                        "kind": "holiday",
                        "prior_scale": prior_scale,
                    }
                )
        return arrays, metadata

    def _design(self, frame: pd.DataFrame, fitting: bool = False) -> tuple[np.ndarray, list[dict]]:
        t = f64((frame["ds"] - self.start) / self.t_scale)
        columns = [np.ones((len(frame), 1), dtype=np.float64)]
        metadata = [{"name": "intercept", "kind": "intercept", "prior_scale": np.inf}]
        if self.growth == "linear":
            columns.append(t[:, None])
            metadata.append({"name": "trend", "kind": "trend", "prior_scale": np.inf})
            for i, cp in enumerate(self.changepoints_t):
                columns.append(np.maximum(0.0, t - cp)[:, None])
                metadata.append(
                    {
                        "name": "trend",
                        "kind": "changepoint",
                        "prior_scale": self.changepoint_prior_scale,
                        "index": i,
                    }
                )
        for name, props in self.seasonalities.items():
            values = self.fourier_series(
                frame["ds"], props["period"], props["fourier_order"]
            )
            condition = props["condition_name"]
            if condition is not None:
                if condition not in frame:
                    raise ValueError(f"Condition {condition!r} missing from dataframe")
                values[~frame[condition].astype(bool).to_numpy()] = 0.0
            columns.append(values)
            metadata.extend(
                {
                    "name": name,
                    "kind": "seasonality",
                    "prior_scale": props["prior_scale"],
                }
                for _ in range(values.shape[1])
            )
        holiday_arrays, holiday_meta = self._holiday_features(frame["ds"])
        columns.extend(holiday_arrays)
        metadata.extend(holiday_meta)
        for name, props in self.extra_regressors.items():
            if name not in frame:
                raise ValueError(f"Regressor {name!r} missing from dataframe")
            try:
                values = frame[name].to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Regressor {name!r} must be numeric") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"Regressor {name!r} must contain only finite values")
            if fitting:
                standardize = props["standardize"]
                if standardize == "auto":
                    standardize = set(np.unique(values)) != {0.0, 1.0}
                if standardize:
                    props["mu"] = float(values.mean())
                    props["std"] = float(values.std(ddof=1))
                    if props["std"] == 0:
                        raise ValueError(f"Regressor {name!r} is constant")
            values = (values - props["mu"]) / props["std"]
            columns.append(values[:, None])
            metadata.append(
                {"name": name, "kind": "regressor", "prior_scale": props["prior_scale"]}
            )
        return np.ascontiguousarray(np.hstack(columns)), metadata

    def fit(self, df: pd.DataFrame, **kwargs) -> "Prophet":
        if self.history is not None:
            raise Exception("Prophet object can only be fit once. Instantiate a new object.")
        if self.mcmc_samples:
            raise NotImplementedError("mcmc_samples is outside the additive deterministic subset")
        if self.growth == "logistic":
            raise NotImplementedError("logistic trend evaluation is covered, logistic fitting is not")
        if self.seasonality_mode != "additive" or self.holidays_mode != "additive":
            raise NotImplementedError("only additive mode is covered")
        if any(p["mode"] != "additive" for p in self.seasonalities.values()) or any(
            p["mode"] != "additive" for p in self.extra_regressors.values()
        ):
            raise NotImplementedError("multiplicative components are not covered")
        if not isinstance(df, pd.DataFrame) or not {"ds", "y"} <= set(df):
            raise ValueError("Dataframe must have columns 'ds' and 'y'.")
        history = df.copy()
        history["ds"] = pd.to_datetime(history["ds"])
        history = history.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
        if len(history) < 2:
            raise ValueError("Dataframe has less than 2 non-NaN rows.")
        try:
            y_values = history["y"].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Column y must be numeric.") from exc
        if not np.isfinite(y_values).all():
            raise ValueError("Column y must contain only finite values.")
        self.start = history["ds"].min()
        self.t_scale = history["ds"].max() - self.start
        if self.t_scale <= pd.Timedelta(0):
            raise ValueError("Dataframe has less than 2 unique timestamps.")
        self._auto_seasonalities(history["ds"])
        self._set_changepoints(history["ds"])
        self.history_dates = history["ds"].copy()
        self.history = history
        design, metadata = self._design(history, fitting=True)
        y = f64(y_values)
        if self.scaling == "absmax":
            self.y_min = 0.0
            self.y_scale = float(np.max(np.abs(y))) or 1.0
        else:
            self.y_min = float(y.min())
            self.y_scale = float(y.max() - y.min()) or 1.0
        y_scaled = np.ascontiguousarray((y - self.y_min) / self.y_scale)
        penalties = np.array(
            [0.0 if np.isinf(item["prior_scale"]) else 1.0 / item["prior_scale"] ** 2 for item in metadata],
            dtype=np.float64,
        )
        penalties[: 2 if self.growth == "linear" else 1] = 1e-12
        coef = np.empty(design.shape[1], dtype=np.float64)
        work = np.empty((design.shape[1], design.shape[1]), dtype=np.float64)
        n, d = design.shape
        chunks = _ridge_chunks(n, d)
        if chunks:
            scratch = np.empty(chunks * (d * d + d), dtype=np.float64)
            ok = lib().mop_ridge_fit_parallel(
                addr(design),
                addr(y_scaled),
                addr(penalties),
                addr(coef),
                addr(work),
                addr(scratch),
                n,
                d,
                chunks,
            )
        else:
            ok = lib().mop_ridge_fit(
                addr(design),
                addr(y_scaled),
                addr(penalties),
                addr(coef),
                addr(work),
                n,
                d,
            )
        if not ok:
            coef = np.linalg.lstsq(
                np.vstack([design, np.diag(np.sqrt(penalties))]),
                np.concatenate([y_scaled, np.zeros(design.shape[1])]),
                rcond=None,
            )[0]
        fitted_scaled = np.empty(len(history), dtype=np.float64)
        if not lib().mop_matvec(
            addr(design), addr(coef), addr(fitted_scaled), len(history), design.shape[1]
        ):
            raise RuntimeError("Mojo matrix-vector kernel rejected its arguments")
        residual = y_scaled - fitted_scaled
        self.sigma_obs = float(np.sqrt(np.dot(residual, residual) / max(1, len(y) - design.shape[1])))
        self._coef = coef
        self._metadata = metadata
        delta = np.array([coef[i] for i, item in enumerate(metadata) if item["kind"] == "changepoint"])
        self.params = {
            "k": np.array([[coef[1] if self.growth == "linear" else 0.0]]),
            "m": np.array([[coef[0]]]),
            "delta": delta[None, :],
            "beta": np.array(
                [[coef[i] for i, item in enumerate(metadata) if item["kind"] not in ("intercept", "trend", "changepoint")]]
            ),
            "sigma_obs": np.array([[self.sigma_obs]]),
        }
        return self

    def _component(self, design: np.ndarray, name: str) -> np.ndarray:
        indexes = [i for i, item in enumerate(self._metadata) if item["name"] == name]
        if not indexes:
            return np.zeros(design.shape[0])
        result = np.empty(design.shape[0], dtype=np.float64)
        if not lib().mop_component(
            addr(design),
            addr(self._coef),
            addr(result),
            design.shape[0],
            design.shape[1],
            indexes[0],
            len(indexes),
        ):
            raise RuntimeError("Mojo component kernel rejected its arguments")
        return result * self.y_scale

    def predict(self, df: pd.DataFrame | None = None, vectorized: bool = True) -> pd.DataFrame:
        if self.history is None:
            raise Exception("Model has not been fit.")
        frame = self.history.copy() if df is None else df.copy()
        if frame.empty:
            raise ValueError("Dataframe has no rows.")
        if "ds" not in frame:
            raise ValueError("Dataframe must have column ds.")
        frame["ds"] = pd.to_datetime(frame["ds"])
        design, _ = self._design(frame)
        scaled = np.empty(len(frame), dtype=np.float64)
        if not lib().mop_matvec(
            addr(design), addr(self._coef), addr(scaled), len(frame), design.shape[1]
        ):
            raise RuntimeError("Mojo matrix-vector kernel rejected its arguments")
        forecast = pd.DataFrame({"ds": frame["ds"].to_numpy()})
        trend = self._component(design, "intercept") + self._component(design, "trend") + self.y_min
        forecast["trend"] = trend
        additive = np.zeros(len(frame))
        component_names = list(self.seasonalities)
        component_names += sorted({m["name"] for m in self._metadata if m["kind"] == "holiday"})
        component_names += list(self.extra_regressors)
        for name in component_names:
            values = self._component(design, name)
            forecast[name] = values
            additive += values
        forecast["additive_terms"] = additive
        forecast["multiplicative_terms"] = 0.0
        forecast["yhat"] = scaled * self.y_scale + self.y_min
        if self.uncertainty_samples:
            quantile = NormalDist().inv_cdf((1.0 + self.interval_width) / 2.0)
            radius = quantile * self.sigma_obs * self.y_scale
            forecast["yhat_lower"] = forecast["yhat"] - radius
            forecast["yhat_upper"] = forecast["yhat"] + radius
        return forecast

    def make_future_dataframe(
        self, periods: int, freq: str | None = "D", include_history: bool = True
    ) -> pd.DataFrame:
        if self.history_dates is None:
            raise Exception("Model has not been fit.")
        if freq is None:
            freq = pd.infer_freq(self.history_dates.tail(5))
            if freq is None:
                raise Exception("Unable to infer `freq`")
        last = self.history_dates.max()
        future = pd.date_range(start=last, periods=periods + 1, freq=freq)
        future = future[future > last][:periods]
        dates = (
            np.concatenate((self.history_dates.to_numpy(), future.to_numpy()))
            if include_history
            else future
        )
        return pd.DataFrame({"ds": dates})
