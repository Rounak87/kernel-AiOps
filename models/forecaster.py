"""
Saturation Forecaster for Time-to-Failure (TTS) prediction.

Once an anomaly is detected and the root-cause metric is isolated by the Causal Engine,
the Forecaster models the trajectory of that metric to answer:
    "At the current rate of degradation, how many seconds until catastrophic saturation?"

Methodology:
    1. Fetches recent time-series telemetry for the degraded metric from InfluxDB.
    2. Fits ordinary least squares (OLS) linear trend regression over the incident window.
    3. Calculates degradation slope (rate of change per second) and R^2 confidence.
    4. Computes Time-to-Saturation (TTS): seconds until the metric breaches its critical threshold.
    5. Generates a projection curve for the next N seconds.
"""

import logging
import numpy as np
from dataclasses import dataclass, field

from pipeline.db import fetch_history

log = logging.getLogger(__name__)

# Default operational saturation thresholds (in metric-native units)
DEFAULT_CRITICAL_THRESHOLDS: dict[str, float] = {
    "avg_stall_ns":         100_000_000.0,  # 100ms scheduler run-queue stall
    "avg_lock_wait_ns":      50_000_000.0,  # 50ms lock contention wait
    "avg_syscall_ns":        20_000_000.0,  # 20ms average syscall duration
    "major_faults":                  50.0,  # 50 major page faults/sec (active swap thrashing)
    "involuntary_switches":         500.0,  # 500 preemptions/sec (CPU starvation)
    "lock_wait_count":               20.0,  # 20 contention events/sec
}

MIN_FORECAST_SAMPLES = 15
DEFAULT_INCIDENT_WINDOW = "30d"
DEFAULT_HORIZON_SECONDS = 60


@dataclass
class ForecastResult:
    service: str
    metric: str
    current_value: float
    threshold: float
    slope_per_sec: float
    r_squared: float
    trend_direction: str  # "RISING", "FALLING", "STABLE"
    tts_seconds: float | None
    is_saturated: bool
    forecast_curve: list[float] = field(default_factory=list)
    summary: str = ""


class SaturationForecaster:
    """
    Extrapolates degraded kernel metrics to estimate Time-To-Saturation (TTS).
    """

    def __init__(
        self,
        custom_thresholds: dict[str, float] | None = None,
        incident_window: str = DEFAULT_INCIDENT_WINDOW,
        horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    ):
        self.thresholds = dict(DEFAULT_CRITICAL_THRESHOLDS)
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
        self.window = incident_window
        self.horizon = horizon_seconds

    def forecast(
        self,
        service: str,
        metric: str,
        threshold: float | None = None,
    ) -> ForecastResult:
        """
        Estimate time-to-saturation for a specific metric on a given service.
        """
        target_threshold = threshold if threshold is not None else self.thresholds.get(metric, 100.0)

        history = fetch_history(service, metric, window=self.window)
        if len(history) < MIN_FORECAST_SAMPLES:
            return ForecastResult(
                service=service,
                metric=metric,
                current_value=0.0,
                threshold=target_threshold,
                slope_per_sec=0.0,
                r_squared=0.0,
                trend_direction="STABLE",
                tts_seconds=None,
                is_saturated=False,
                summary=f"Insufficient history ({len(history)} samples) to forecast.",
            )

        y = np.array(history, dtype=float)
        n = len(y)
        x = np.arange(n, dtype=float)

        current_val = float(y[-1])
        is_saturated = current_val >= target_threshold

        # Fit linear regression: y = slope * x + intercept
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-9:
            slope = 0.0
            intercept = y_mean
            r_squared = 0.0
        else:
            slope = float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
            intercept = float(y_mean - slope * x_mean)
            y_pred = slope * x + intercept
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0
            r_squared = max(0.0, min(1.0, r_squared))

        # Trend direction
        if slope > 1e-5:
            trend_direction = "RISING"
        elif slope < -1e-5:
            trend_direction = "FALLING"
        else:
            trend_direction = "STABLE"

        # Calculate Time-to-Saturation (TTS)
        tts_seconds: float | None = None
        if is_saturated:
            tts_seconds = 0.0
        elif slope > 0:
            remaining = target_threshold - current_val
            tts_seconds = round(remaining / slope, 1)

        # Generate forward projection curve for the next N seconds
        future_x = np.arange(n, n + self.horizon, dtype=float)
        projected = slope * future_x + intercept
        forecast_curve = [round(max(0.0, float(v)), 2) for v in projected]

        # Human-readable engineering summary
        if is_saturated:
            summary = (
                f"CRITICAL: {metric} ({current_val:.1f}) is ALREADY SATURATED "
                f"above critical threshold ({target_threshold:.1f})."
            )
        elif tts_seconds is not None:
            summary = (
                f"WARNING: {metric} is RISING (+{slope:.2f}/sec, R^2={r_squared:.2f}). "
                f"Projected to breach threshold ({target_threshold:.1f}) in {tts_seconds} seconds."
            )
        else:
            summary = (
                f"HEALTHY: {metric} is {trend_direction.lower()} "
                f"(current={current_val:.1f}, threshold={target_threshold:.1f}). No saturation imminent."
            )

        return ForecastResult(
            service=service,
            metric=metric,
            current_value=current_val,
            threshold=target_threshold,
            slope_per_sec=round(slope, 4),
            r_squared=round(r_squared, 4),
            trend_direction=trend_direction,
            tts_seconds=tts_seconds,
            is_saturated=is_saturated,
            forecast_curve=forecast_curve,
            summary=summary,
        )
