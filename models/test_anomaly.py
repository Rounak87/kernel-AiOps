"""
Quick smoke test for the anomaly detector against live InfluxDB data.
Run with: python -m models.test_anomaly
"""
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from models.anomaly_detector import AnomalyDetector, ANOMALY_THRESHOLD
from pipeline.db import list_active_services

detector = AnomalyDetector()
services = list_active_services(window="7d")
print(f"Services available for scoring: {len(services)}")
print()

results = []
for svc in services:
    score = detector.score(svc)
    if score is not None:
        results.append((svc, score))

results.sort(key=lambda x: -x[1])

print(f"{'SERVICE':<24} {'ANOMALY SCORE':<16} STATUS")
print("-" * 58)
for svc, score in results:
    status = "*** ANOMALY ***" if score >= ANOMALY_THRESHOLD else "normal"
    print(f"{svc:<24} {score:<16.4f} {status}")

print()
print(f"Scored: {len(results)} / {len(services)} services")
anomalies = [(s, sc) for s, sc in results if sc >= ANOMALY_THRESHOLD]
print(f"Anomalies detected: {len(anomalies)} (threshold >= {ANOMALY_THRESHOLD})")
