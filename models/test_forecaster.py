"""
Smoke test for the Saturation Forecaster against live InfluxDB telemetry.
Run with: python -m models.test_forecaster
"""
from models.forecaster import SaturationForecaster

def main():
    forecaster = SaturationForecaster()

    test_cases = [
        ("multipathd", "avg_stall_ns"),
        ("multipathd", "avg_lock_wait_ns"),
        ("jbd2/nvme0n1p1-", "avg_syscall_ns"),
        ("ml-engine", "avg_lock_wait_ns"),
        ("chronyd", "avg_stall_ns"),
    ]

    print("=" * 75)
    print("       SATURATION FORECASTER & TIME-TO-FAILURE (TTS) ENGINE")
    print("=" * 75)

    for svc, metric in test_cases:
        res = forecaster.forecast(svc, metric)
        print(f"\nService:     [{svc}]")
        print(f"Metric:      {metric}")
        print(f"Current Val: {res.current_value:.1f} / Threshold: {res.threshold:.1f}")
        print(f"Trend:       {res.trend_direction} (slope={res.slope_per_sec:.2f}/s, R^2={res.r_squared:.3f})")
        print(f"TTS:         {f'{res.tts_seconds} seconds' if res.tts_seconds is not None else 'No saturation risk'}")
        print(f"Summary:     {res.summary}")
        print("-" * 75)

if __name__ == "__main__":
    main()
