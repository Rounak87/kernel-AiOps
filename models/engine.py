"""
Unified ML Analytics Engine.

Orchestrates the Tri-Model pipeline:
    1. AnomalyDetector (Isolation Forest) - Real-time trigger
    2. CausalEngine (Granger Causality)   - Root-cause DAG isolation
    3. SaturationForecaster (OLS Trend)   - Time-to-Saturation (TTS) projection
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from models.anomaly_detector import AnomalyDetector, ANOMALY_THRESHOLD
from models.causal_engine import CausalEngine, CausalAnalysisResult
from models.forecaster import SaturationForecaster, ForecastResult
from pipeline.db import list_active_services

log = logging.getLogger(__name__)


@dataclass
class IncidentDiagnosis:
    service: str
    timestamp: str
    anomaly_score: float
    is_anomalous: bool
    root_cause_metric: str | None = None
    time_to_saturation_seconds: float | None = None
    is_saturated: bool = False
    causal_edges_count: int = 0
    causal_summary: str = ""
    forecast_summary: str = ""
    status: str = "HEALTHY"  # "HEALTHY", "WARNING", "CRITICAL"


class MLAnalyticsEngine:
    """
    Unified entry point executing concurrent tri-model analysis.
    """

    def __init__(self, anomaly_threshold: float = ANOMALY_THRESHOLD):
        self.detector = AnomalyDetector()
        self.causal = CausalEngine()
        self.forecaster = SaturationForecaster()
        self.threshold = anomaly_threshold

    def diagnose(self, service: str) -> IncidentDiagnosis:
        """
        Run full tri-model pipeline on a single service.
        """
        ts = datetime.now(timezone.utc).isoformat()
        score = self.detector.score(service)

        if score is None:
            return IncidentDiagnosis(
                service=service,
                timestamp=ts,
                anomaly_score=0.0,
                is_anomalous=False,
                status="UNKNOWN",
            )

        is_anomalous = score >= self.threshold

        if not is_anomalous:
            return IncidentDiagnosis(
                service=service,
                timestamp=ts,
                anomaly_score=score,
                is_anomalous=False,
                status="HEALTHY",
            )

        # Step 2: Causal DAG Analysis to find root-cause metric
        causal_res: CausalAnalysisResult = self.causal.analyze(service)
        root_cause = causal_res.root_cause

        # Step 3: Saturation Forecasting on the root cause (or fallback metric)
        target_metric = root_cause or "avg_stall_ns"
        forecast_res: ForecastResult = self.forecaster.forecast(service, target_metric)

        status = "CRITICAL" if forecast_res.is_saturated else "WARNING"

        return IncidentDiagnosis(
            service=service,
            timestamp=ts,
            anomaly_score=score,
            is_anomalous=True,
            root_cause_metric=root_cause,
            time_to_saturation_seconds=forecast_res.tts_seconds,
            is_saturated=forecast_res.is_saturated,
            causal_edges_count=len(causal_res.edges),
            causal_summary=causal_res.summary,
            forecast_summary=forecast_res.summary,
            status=status,
        )

    def diagnose_all_active(self, window: str = "30d") -> list[IncidentDiagnosis]:
        """
        Scans all active services, detects anomalies, and generates structured diagnoses.
        """
        services = list_active_services(window=window)
        diagnoses = []
        for svc in services:
            d = self.diagnose(svc)
            diagnoses.append(d)
        return diagnoses


if __name__ == "__main__":
    print("=" * 80)
    print("      TRI-MODEL ML ANALYTICS ENGINE: END-TO-END DIAGNOSIS")
    print("=" * 80)

    engine = MLAnalyticsEngine()
    print("Scanning active services across InfluxDB baseline...")
    diagnoses = engine.diagnose_all_active()

    anomalous = [d for d in diagnoses if d.is_anomalous]
    print(f"\nScanned: {len(diagnoses)} services | Anomalous Incidents: {len(anomalous)}")
    print("-" * 80)

    for inc in anomalous:
        print(f"\n[INCIDENT ALERT] Service: {inc.service} (Status: {inc.status})")
        print(f"  Anomaly Score:      {inc.anomaly_score:.4f} (Threshold: {engine.threshold})")
        print(f"  Root Cause Metric:  {inc.root_cause_metric or 'Under analysis'}")
        print(f"  Causal Edges:       {inc.causal_edges_count} directed interactions")
        print(f"  TTS to Saturation:  {inc.time_to_saturation_seconds}s (Saturated: {inc.is_saturated})")
        print(f"  Diagnosis:          {inc.causal_summary}")
        print(f"  Forecast:           {inc.forecast_summary}")
        print("-" * 80)
