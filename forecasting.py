from datetime import datetime, timedelta, timezone

import numpy as np
try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except (ImportError, Exception):
    _HAS_SKLEARN = False

HORIZON = 24
LOOKBACK = 168
MIN_HISTORY = 504


def features(values, t, start):
    previous_day = values[t - 24:t]
    weekly_samples = values[t - 168:t:24]
    timestamp = start + timedelta(hours=int(t))

    calendar = [
        np.sin(2 * np.pi * timestamp.hour / 24),
        np.cos(2 * np.pi * timestamp.hour / 24),
        np.sin(2 * np.pi * timestamp.weekday() / 7),
        np.cos(2 * np.pi * timestamp.weekday() / 7),
    ]

    summary = [
        values[t - 24:t].mean(),
        values[t - 168:t].mean(),
        values[t - 24:t].std(),
    ]

    return np.concatenate(
        [previous_day, weekly_samples, calendar, summary]
    )


def make_windows(values, start):
    starts = np.arange(LOOKBACK, len(values) - HORIZON + 1)

    x = np.array([features(values, t, start) for t in starts])
    y = np.array([values[t:t + HORIZON] for t in starts])

    return starts, x, y


def split_masks(starts, cutoff):
    # A training target must finish before the test period.
    training = starts + HORIZON <= cutoff

    # Non-overlapping 24-hour test targets.
    testing = (
        (starts >= cutoff)
        & ((starts - cutoff) % HORIZON == 0)
    )

    return training, testing


class _NumpyRidgePipeline:
    def __init__(self, alpha=10.0):
        self.alpha = float(alpha)
        self.mean_x = None
        self.scale_x = None
        self.mean_y = None
        self.coef_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_x = x.mean(axis=0)
        self.scale_x = x.std(axis=0)
        self.scale_x[self.scale_x == 0.0] = 1.0

        x_scaled = (x - self.mean_x) / self.scale_x
        self.mean_y = y.mean(axis=0)
        y_centered = y - self.mean_y

        n_features = x.shape[1]
        a = x_scaled.T @ x_scaled + self.alpha * np.eye(n_features)
        b = x_scaled.T @ y_centered
        self.coef_ = np.linalg.solve(a, b)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        x_scaled = (x - self.mean_x) / self.scale_x
        return x_scaled @ self.coef_ + self.mean_y


def new_model():
    if _HAS_SKLEARN:
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        )
    return _NumpyRidgePipeline(alpha=10.0)


def wape(actual, predicted):
    denominator = float(np.sum(actual))
    if denominator == 0:
        return None

    return round(
        100 * float(np.sum(np.abs(actual - predicted))) / denominator,
        2,
    )


def forecast(values, start: datetime):
    values = np.asarray(values, dtype=float)

    if len(values) < MIN_HISTORY:
        raise ValueError(
            f"At least {MIN_HISTORY} consecutive hourly observations are required."
        )

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Activity counts must be finite and nonnegative.")

    starts, x, y = make_windows(values, start)
    cutoff = int(len(values) * 0.75)
    training, testing = split_masks(starts, cutoff)

    if training.sum() < 48 or testing.sum() < 3:
        raise ValueError("Insufficient independent training or test windows.")

    model = new_model()
    model.fit(x[training], y[training])

    predicted = np.maximum(model.predict(x[testing]), 0)
    actual = y[testing]

    # Same hours from the previous week.
    baseline = np.array([
        values[t - 168:t - 168 + HORIZON]
        for t in starts[testing]
    ])

    previous_totals = np.array([
        values[t - 24:t].sum()
        for t in starts[testing]
    ])

    direction_accuracy = float(np.mean(
        (predicted.sum(axis=1) > previous_totals)
        == (actual.sum(axis=1) > previous_totals)
    ))

    model_wape = wape(actual, predicted)
    baseline_wape = wape(actual, baseline)

    # Evaluation is complete before fitting the operational forecast.
    # This refit does not alter the reported holdout measurements.
    model.fit(x, y)

    future = np.maximum(
        model.predict(
            features(values, len(values), start).reshape(1, -1)
        )[0],
        0,
    )

    previous_24 = float(values[-24:].sum())
    preceding_24 = float(values[-48:-24].sum())
    forecast_total = float(future.sum())

    growth = (
        100 * (previous_24 / preceding_24 - 1)
        if preceding_24 > 0
        else None
    )

    forecast_growth = (
        100 * (forecast_total / previous_24 - 1)
        if previous_24 > 0
        else None
    )

    return {
        "model": "standardized-ridge-direct-24h-v1",
        "forecast": [round(float(value), 2) for value in future],
        "previous_24h": round(previous_24, 2),
        "forecast_24h": round(forecast_total, 2),
        "observed_growth_pct": (
            round(growth, 2) if growth is not None else None
        ),
        "forecast_growth_pct": (
            round(forecast_growth, 2)
            if forecast_growth is not None
            else None
        ),
        "evaluation": {
            "model_wape_pct": model_wape,
            "baseline_wape_pct": baseline_wape,
            "direction_accuracy_pct": round(direction_accuracy * 100, 2),
            "test_windows": int(testing.sum()),
            "test_start": (
                start + timedelta(hours=int(starts[testing][0]))
            ).isoformat() + "Z",
            "test_end_exclusive": (
                start
                + timedelta(hours=int(starts[testing][-1]) + HORIZON)
            ).isoformat() + "Z",
            "beats_baseline": (
                model_wape < baseline_wape
                if model_wape is not None and baseline_wape is not None
                else None
            ),
        },
    }


def adaptive_forecast(values, start: datetime = None, horizon: int = 24):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 3:
        raise ValueError("At least 3 observations are required to forecast.")

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Activity counts must be finite and nonnegative.")

    if start is None:
        start = datetime.now(timezone.utc).replace(
            tzinfo=None, minute=0, second=0, microsecond=0
        ) - timedelta(hours=n)

    # If long enough for standard Ridge direct model and horizon is 24, use it
    if n >= MIN_HISTORY and horizon == HORIZON:
        try:
            return forecast(values, start)
        except ValueError:
            pass  # Fall through to adaptive model

    # Check all zero
    if float(values.sum()) == 0:
        return {
            "model": "zero-activity",
            "forecast": [0.0] * horizon,
            "previous_24h": 0.0,
            "forecast_24h": 0.0,
            "observed_growth_pct": None,
            "forecast_growth_pct": None,
            "evaluation": {
                "model_wape_pct": None,
                "baseline_wape_pct": None,
                "direction_accuracy_pct": 100.0,
                "test_windows": 1,
                "test_start": start.isoformat() + "Z",
                "test_end_exclusive": (start + timedelta(hours=n)).isoformat() + "Z",
                "beats_baseline": None,
            },
        }

    # Trend calculation using linear fit on recent values
    t = np.arange(n)
    trend_window = min(n, 168)
    t_recent = t[-trend_window:]
    v_recent = values[-trend_window:]

    slope, intercept = np.polyfit(t_recent, v_recent, 1)

    # Hourly seasonality if series is long enough (>= 24)
    seasonal_pattern = np.zeros(24)
    if n >= 24:
        detrended = values - (slope * t + intercept)
        hour_indices = np.array(
            [(start + timedelta(hours=int(i))).hour for i in range(n)]
        )
        for h in range(24):
            mask = hour_indices == h
            if mask.any():
                seasonal_pattern[h] = float(np.mean(detrended[mask]))
        seasonal_pattern -= seasonal_pattern.mean()

    # Dynamic Ridge Auto-Regressive model when n >= 8
    def fit_and_predict(train_series, train_start, steps_ahead):
        m = len(train_series)
        if m < 8:
            cur_t = np.arange(m)
            cur_slope, cur_intercept = np.polyfit(cur_t, train_series, 1)
            future_steps = np.arange(steps_ahead)
            damped_slope = cur_slope * (0.92 ** future_steps)
            pred = train_series[-1] + np.cumsum(damped_slope)
            return np.maximum(0.0, pred)

        lookback = min(12, m // 3)
        x_list, y_list = [], []
        for i in range(lookback, m):
            feat = [
                train_series[i - 1],
                train_series[i - 2] if i >= 2 else train_series[i - 1],
                train_series[i - 3] if i >= 3 else train_series[i - 1],
                float(np.mean(train_series[max(0, i - 6) : i])),
                float(np.std(train_series[max(0, i - 6) : i])) if i >= 6 else 0.0,
                i / m,
            ]
            if n >= 24:
                ts = train_start + timedelta(hours=int(i))
                feat.append(np.sin(2 * np.pi * ts.hour / 24))
                feat.append(np.cos(2 * np.pi * ts.hour / 24))
            x_list.append(feat)
            y_list.append(train_series[i])

        x_arr = np.array(x_list)
        y_arr = np.array(y_list)
        model = new_model()
        model.fit(x_arr, y_arr)

        current_history = list(train_series)
        preds = []
        for _ in range(steps_ahead):
            curr_len = len(current_history)
            feat = [
                current_history[-1],
                current_history[-2],
                current_history[-3],
                float(np.mean(current_history[-6:])),
                float(np.std(current_history[-6:])),
                curr_len / m,
            ]
            if n >= 24:
                ts = train_start + timedelta(hours=int(curr_len))
                feat.append(np.sin(2 * np.pi * ts.hour / 24))
                feat.append(np.cos(2 * np.pi * ts.hour / 24))
            pred_val = float(model.predict(np.array([feat]))[0])
            if n >= 24:
                ts = train_start + timedelta(hours=int(curr_len))
                pred_val = 0.7 * pred_val + 0.3 * (
                    pred_val + seasonal_pattern[ts.hour]
                )
            pred_val = max(0.0, pred_val)
            preds.append(pred_val)
            current_history.append(pred_val)

        return np.array(preds)

    # Holdout validation (test on last test_len points)
    test_len = min(horizon, max(3, n // 5))
    if n - test_len >= 8:
        train_data = values[:-test_len]
        test_actual = values[-test_len:]
        test_pred = fit_and_predict(train_data, start, test_len)

        baseline_pred = np.full(test_len, train_data[-1])

        model_w = wape(test_actual, test_pred)
        baseline_w = wape(test_actual, baseline_pred)

        if test_len > 1:
            actual_dir = np.diff(test_actual) > 0
            pred_dir = np.diff(test_pred) > 0
            dir_acc = float(np.mean(actual_dir == pred_dir) * 100)
        else:
            dir_acc = (
                100.0
                if (test_pred[0] > train_data[-1])
                == (test_actual[0] > train_data[-1])
                else 0.0
            )

        test_start_dt = start + timedelta(hours=n - test_len)
        test_end_dt = start + timedelta(hours=n)
    else:
        model_w = None
        baseline_w = None
        dir_acc = 100.0
        test_start_dt = start
        test_end_dt = start + timedelta(hours=n)

    future = fit_and_predict(values, start, horizon)

    comp_window = min(horizon, n)
    previous_comp = float(values[-comp_window:].sum())
    forecast_comp = float(future[:comp_window].sum())
    preceding_window = (
        min(comp_window, n - comp_window) if n >= comp_window * 2 else None
    )

    if preceding_window and preceding_window > 0:
        preceding_sum = float(
            values[-comp_window - preceding_window : -comp_window].sum()
        )
        observed_growth = (
            100 * (previous_comp / preceding_sum - 1)
            if preceding_sum > 0
            else None
        )
    else:
        observed_growth = None

    forecast_growth = (
        100 * (forecast_comp / previous_comp - 1)
        if previous_comp > 0
        else None
    )

    return {
        "model": "adaptive-trend-seasonal-ridge-v1",
        "forecast": [round(float(v), 2) for v in future],
        "previous_24h": round(previous_comp, 2),
        "forecast_24h": round(forecast_comp, 2),
        "observed_growth_pct": (
            round(observed_growth, 2) if observed_growth is not None else None
        ),
        "forecast_growth_pct": (
            round(forecast_growth, 2) if forecast_growth is not None else None
        ),
        "evaluation": {
            "model_wape_pct": model_w,
            "baseline_wape_pct": baseline_w,
            "direction_accuracy_pct": round(dir_acc, 2),
            "test_windows": 1,
            "test_start": test_start_dt.isoformat() + "Z",
            "test_end_exclusive": test_end_dt.isoformat() + "Z",
            "beats_baseline": (
                model_w < baseline_w
                if (model_w is not None and baseline_w is not None)
                else None
            ),
        },
    }

