"""
Granger Causality Engine for Root Cause Analysis (RCA).

When the Anomaly Detector flags a service, the Causal Engine inspects the
time-series telemetry of that service across its 6 metric dimensions to answer:
    "Which metric was the initiating root cause, and which metrics are just downstream symptoms?"

Methodology:
    1. Fetches time-aligned metric histories from InfluxDB for the target service.
    2. Filters out zero-variance (flat) series.
    3. Runs pairwise Granger Causality tests (vector autoregression F-tests) across all metric pairs.
    4. Identifies directed causal edges where p-value < significance_level (default 0.05).
    5. Builds a directed causal graph (DAG) and computes Net Causal Influence (out-degree - in-degree)
       to pinpoint the primary root-cause metric.
"""

import logging
import warnings
import numpy as np
from dataclasses import dataclass, field
from statsmodels.tsa.stattools import grangercausalitytests

from pipeline.db import fetch_history
from models.anomaly_detector import FEATURES

log = logging.getLogger(__name__)

# Suppress statsmodels convergence / future warnings for clean logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SIGNIFICANCE_THRESHOLD = 0.05
DEFAULT_MAX_LAG        = 3
MIN_SAMPLES_REQUIRED   = 30
DEFAULT_WINDOW         = "7d"


@dataclass
class CausalEdge:
    source: str        # The initiating metric (cause)
    target: str        # The impacted metric (symptom)
    p_value: float     # Statistical significance (lower is stronger)
    best_lag: int      # Time-lag in seconds (e.g. 1s, 2s)
    confidence: float  # 1.0 - p_value


@dataclass
class CausalAnalysisResult:
    service: str
    root_cause: str | None
    edges: list[CausalEdge] = field(default_factory=list)
    adjacency: dict[str, list[str]] = field(default_factory=dict)
    active_metrics: list[str] = field(default_factory=list)
    summary: str = ""


class CausalEngine:
    """
    Evaluates time-lagged causal interactions across kernel telemetry metrics.
    """

    def __init__(
        self,
        significance_threshold: float = SIGNIFICANCE_THRESHOLD,
        max_lag: int = DEFAULT_MAX_LAG,
        window: str = DEFAULT_WINDOW,
    ):
        self.significance = significance_threshold
        self.max_lag = max_lag
        self.window = window

    def _fetch_aligned_series(self, service: str) -> dict[str, np.ndarray]:
        """
        Pull all 6 metrics from InfluxDB for the target service and
        align them to equal length arrays. Filter out constant / zero-variance series.
        """
        raw_series: dict[str, list[float]] = {}
        for feat in FEATURES:
            data = fetch_history(service, feat, window=self.window)
            if len(data) >= MIN_SAMPLES_REQUIRED:
                raw_series[feat] = data

        if not raw_series:
            return {}

        min_len = min(len(s) for s in raw_series.values())
        if min_len < MIN_SAMPLES_REQUIRED:
            return {}

        aligned: dict[str, np.ndarray] = {}
        for feat, vals in raw_series.items():
            arr = np.array(vals[:min_len], dtype=float)
            # Check for non-zero variance (constant series cannot cause anything)
            if np.var(arr) > 1e-9:
                aligned[feat] = arr
            else:
                log.debug("Metric %s for %s has zero variance (constant). Skipping.", feat, service)

        return aligned

    def analyze(self, service: str) -> CausalAnalysisResult:
        """
        Execute Granger causality tests across all active metric pairs for the given service.
        """
        series = self._fetch_aligned_series(service)
        active_metrics = list(series.keys())

        if len(active_metrics) < 2:
            return CausalAnalysisResult(
                service=service,
                root_cause=None,
                active_metrics=active_metrics,
                summary=f"Insufficient active metric dimensions ({len(active_metrics)}) with variance to test causality.",
            )

        confirmed_edges: list[CausalEdge] = []
        adjacency: dict[str, list[str]] = {m: [] for m in active_metrics}
        in_degree: dict[str, int] = {m: 0 for m in active_metrics}
        out_degree: dict[str, int] = {m: 0 for m in active_metrics}

        # Pairwise Granger tests: test if candidate_cause -> candidate_effect
        for cause in active_metrics:
            for effect in active_metrics:
                if cause == effect:
                    continue

                # grangercausalitytests expects 2D array [effect, cause]
                # to test whether column 1 (cause) Granger-causes column 0 (effect)
                data_matrix = np.column_stack([series[effect], series[cause]])

                try:
                    gc_res = grangercausalitytests(data_matrix, maxlag=self.max_lag)
                except Exception as e:
                    log.debug("Granger test failed for %s -> %s: %s", cause, effect, e)
                    continue

                # Find best lag with lowest p-value
                best_lag = 1
                best_p = 1.0
                for lag in range(1, self.max_lag + 1):
                    # ssr_ftest: (F_stat, p_value, df_denom, df_num)
                    p_val = gc_res[lag][0]["ssr_ftest"][1]
                    if p_val < best_p:
                        best_p = p_val
                        best_lag = lag

                if best_p < self.significance:
                    edge = CausalEdge(
                        source=cause,
                        target=effect,
                        p_value=round(best_p, 5),
                        best_lag=best_lag,
                        confidence=round(1.0 - best_p, 4),
                    )
                    confirmed_edges.append(edge)
                    adjacency[cause].append(effect)
                    out_degree[cause] += 1
                    in_degree[effect] += 1

        # Determine Root Cause
        # The true root cause is the metric with maximum net causal power (out_degree - in_degree)
        root_cause = None
        if confirmed_edges:
            net_influence = {m: out_degree[m] - in_degree[m] for m in active_metrics}
            # Sort by highest net influence, then highest out_degree
            sorted_candidates = sorted(
                active_metrics,
                key=lambda m: (net_influence[m], out_degree[m]),
                reverse=True,
            )
            top_candidate = sorted_candidates[0]
            if out_degree[top_candidate] > 0:
                root_cause = top_candidate

        # Build human-readable summary
        if root_cause:
            targets = adjacency.get(root_cause, [])
            targets_str = ", ".join(targets) if targets else "downstream subsystems"
            summary = (
                f"Root cause identified: '{root_cause}' with {out_degree[root_cause]} outgoing causal links. "
                f"Statistically drives degradation in: [{targets_str}]."
            )
        else:
            summary = "No statistically significant directed causal edges detected among active metrics."

        return CausalAnalysisResult(
            service=service,
            root_cause=root_cause,
            edges=confirmed_edges,
            adjacency=adjacency,
            active_metrics=active_metrics,
            summary=summary,
        )
