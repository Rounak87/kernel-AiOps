"""
Isolation Forest anomaly detector.

For each active service, maintains a rolling behavioral baseline and scores
incoming metric vectors. Scores above ANOMALY_THRESHOLD trigger the causal
analysis pipeline.

Feature vector (6 dimensions per service per second):
    avg_stall_ns          - scheduler run-queue wait latency
    involuntary_switches  - kernel preemptions (involuntary context switches)
    avg_syscall_ns        - average syscall execution duration
    major_faults          - major page faults (disk I/O required)
    avg_lock_wait_ns      - mutex/futex contention wait time
    lock_wait_count       - number of lock contention events
"""

import logging
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from pipeline.db import fetch_history, fetch_latest

log = logging.getLogger(__name__)

ANOMALY_THRESHOLD    = 0.85
TRAINING_WINDOW      = "30d"
SCORE_WINDOW         = "30d"
MIN_TRAINING_SAMPLES = 50

FEATURES = [
    "avg_stall_ns",
    "involuntary_switches",
    "avg_syscall_ns",
    "major_faults",
    "avg_lock_wait_ns",
    "lock_wait_count",
]


@dataclass
class AnomalyEvent:
    service: str
    score: float
    feature_vector: dict[str, float]


class AnomalyDetector:
    """
    Per-service Isolation Forest anomaly detector.

    Each service gets its own model trained on its historical baseline.
    The score returned is normalized to [0, 1] where values above
    ANOMALY_THRESHOLD indicate significant deviation from the baseline.
    """

    def __init__(self, contamination: float = 0.05):
        self._models: dict[str, IsolationForest] = {}
        self._contamination = contamination

    def _build_training_matrix(self, service: str) -> np.ndarray | None:
        """
        Fetch historical data for all features and assemble into a
        (n_samples, n_features) matrix for model training.
        """
        columns = {}
        for feat in FEATURES:
            values = fetch_history(service, feat, window=TRAINING_WINDOW)
            if len(values) < MIN_TRAINING_SAMPLES:
                log.debug("service=%s feature=%s insufficient samples=%d", service, feat, len(values))
                return None
            columns[feat] = values

        # Align all feature series to the shortest length
        min_len = min(len(v) for v in columns.values())
        matrix = np.array([columns[f][:min_len] for f in FEATURES], dtype=float).T
        return matrix

    def train(self, service: str) -> bool:
        """
        Train an Isolation Forest model on the service's historical baseline.
        Returns True if training succeeded, False if insufficient data.
        """
        X = self._build_training_matrix(service)
        if X is None:
            return False

        model = IsolationForest(
            n_estimators=200,
            contamination=self._contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        self._models[service] = model
        log.info("trained anomaly model for service=%s samples=%d", service, X.shape[0])
        return True

    def score(self, service: str) -> float | None:
        """
        Score the current metric snapshot for a service.

        Returns an anomaly score in [0, 1]. Returns None if the service has
        no trained model or current metrics are unavailable.
        """
        if service not in self._models:
            trained = self.train(service)
            if not trained:
                return None

        current = fetch_latest(service, window=SCORE_WINDOW)
        if not current:
            log.debug("no data found for service=%s within window=%s", service, SCORE_WINDOW)
            return None

        vector = np.array([[current.get(f, 0.0) for f in FEATURES]])
        model = self._models[service]

        # IsolationForest returns raw scores in [-0.5, 0.5].
        # Negate and shift to [0, 1] so higher = more anomalous.
        raw = model.score_samples(vector)[0]
        normalized = float(np.clip(-raw + 0.5, 0.0, 1.0))
        return round(normalized, 4)

    def evaluate_all(self, services: list[str]) -> list[AnomalyEvent]:
        """
        Score all given services and return anomaly events for those that
        exceed the detection threshold.
        """
        events = []
        for svc in services:
            s = self.score(svc)
            if s is None:
                continue

            log.debug("service=%s anomaly_score=%.3f", svc, s)

            if s >= ANOMALY_THRESHOLD:
                current = fetch_latest(svc, window=SCORE_WINDOW) or {}
                events.append(AnomalyEvent(
                    service=svc,
                    score=round(s, 4),
                    feature_vector=current,
                ))
                log.warning("ANOMALY service=%s score=%.3f", svc, s)

        return events
