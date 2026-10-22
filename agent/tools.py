"""
Diagnostic tool implementations for the LLM agent.

Each function is exposed as a native Gemini function-calling tool.
Tools 1 and 2 query live data from InfluxDB.
Tools 3 and 4 query structured config and log files.
"""

import os
import re
import yaml
import json
import logging
from pathlib import Path
from pipeline.db import fetch_history, list_active_services

log = logging.getLogger(__name__)

_AGENT_DIR = Path(__file__).parent
_TOPOLOGY_PATH    = _AGENT_DIR / "topology.yaml"
_DEPLOYMENTS_PATH = _AGENT_DIR / "deployments.yaml"
_LOGS_DIR         = _AGENT_DIR / "logs"

with open(_TOPOLOGY_PATH)    as f: _TOPOLOGY    = yaml.safe_load(f)
with open(_DEPLOYMENTS_PATH) as f: _DEPLOYMENTS = yaml.safe_load(f)


def query_metrics_history(service: str, metric: str, window: str = "30m") -> str:
    """
    Return recent time-series values for a specific kernel metric on a service.
    Window accepts InfluxDB duration strings: '5m', '30m', '1h', '7d'.
    """
    values = fetch_history(service, metric, window=window)
    if not values and window != "30d":
        values = fetch_history(service, metric, window="30d")
        if values:
            window = "30d"
    if not values:
        return json.dumps({"error": f"No data for service={service} metric={metric} window={window}"})
    summary = {
        "service": service,
        "metric": metric,
        "window": window,
        "count": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(sum(values) / len(values), 2),
        "last_value": round(values[-1], 2),
        "sample_tail": [round(v, 2) for v in values[-10:]],
    }
    return json.dumps(summary)


def get_service_topology(service: str) -> str:
    """
    Return upstream and downstream service dependencies for a given service.
    """
    entry = _TOPOLOGY.get("services", {}).get(service)
    if not entry:
        services = list_active_services(window="30d")
        return json.dumps({
            "error": f"No topology entry for '{service}'.",
            "known_services": services[:20],
        })
    return json.dumps({
        "service": service,
        "type": entry.get("type"),
        "description": entry.get("description"),
        "upstream": entry.get("upstream", []),
        "downstream": entry.get("downstream", []),
    })


def get_recent_deployments(service: str, hours: int = 24) -> str:
    """
    Return deployment events for a service in the last N hours.
    """
    service_key = service.split("/")[0]
    records = _DEPLOYMENTS.get("deployments", {}).get(service_key, [])
    if not records:
        return json.dumps({
            "service": service,
            "deployments": [],
            "note": "No deployment records found for this service in the last 24h.",
        })
    return json.dumps({
        "service": service,
        "deployments": records,
        "count": len(records),
    })


def search_logs(service: str, pattern: str) -> str:
    """
    Search the service's log file for lines matching a keyword or pattern.
    Returns up to 10 matching lines with timestamps.
    """
    service_key = service.split("/")[0].replace("-", "-")
    log_candidates = [
        _LOGS_DIR / f"{service_key}.log",
        _LOGS_DIR / f"{service}.log",
    ]

    log_file = None
    for candidate in log_candidates:
        if candidate.exists():
            log_file = candidate
            break

    if log_file is None:
        return json.dumps({
            "service": service,
            "pattern": pattern,
            "matches": [],
            "note": f"No log file found for service '{service}'.",
        })

    try:
        regex = re.compile(pattern, re.IGNORECASE)
        lines = log_file.read_text(encoding="utf-8").splitlines()
        matches = [l for l in lines if regex.search(l)][:10]
    except re.error as e:
        matches = [l for l in lines if pattern.lower() in l.lower()][:10]

    return json.dumps({
        "service": service,
        "pattern": pattern,
        "match_count": len(matches),
        "matches": matches,
    })


TOOL_FUNCTIONS = {
    "query_metrics_history": query_metrics_history,
    "get_service_topology":  get_service_topology,
    "get_recent_deployments": get_recent_deployments,
    "search_logs":           search_logs,
}
