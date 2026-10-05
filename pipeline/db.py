from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone
import time

INFLUX_URL   = "http://localhost:8086"
INFLUX_TOKEN = "aiops-token-2024"
INFLUX_ORG   = "aiops"
INFLUX_BUCKET = "kernel_metrics"

_client    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
_write_api = _client.write_api(write_options=SYNCHRONOUS)
_query_api = _client.query_api()


def write_metric_batch(node: str, metrics: list[dict]) -> int:
    """
    Write a batch of per-process kernel metric records to InfluxDB.

    Each record must contain at minimum: service, timestamp, and one or more
    numeric fields (avg_stall_ns, lock_wait_ns, etc.).

    Returns the number of points written.
    """
    points = []

    for m in metrics:
        service = m.get("service")
        if not service:
            continue

        ts = datetime.fromtimestamp(m.get("timestamp", time.time()), tz=timezone.utc)

        point = (
            Point("kernel_metrics")
            .tag("service",  service)
            .tag("node",     node)
            .tag("process",  m.get("process", "unknown"))
            .field("avg_stall_ns",          int(m.get("avg_stall_ns", 0)))
            .field("involuntary_switches",   int(m.get("involuntary_switches", 0)))
            .field("voluntary_switches",     int(m.get("voluntary_switches", 0)))
            .field("avg_syscall_ns",         int(m.get("avg_syscall_ns", 0)))
            .field("syscall_count",          int(m.get("syscall_count", 0)))
            .field("major_faults",           int(m.get("major_faults", 0)))
            .field("minor_faults",           int(m.get("minor_faults", 0)))
            .field("avg_lock_wait_ns",       int(m.get("avg_lock_wait_ns", 0)))
            .field("lock_wait_count",        int(m.get("lock_wait_count", 0)))
            .time(ts)
        )
        points.append(point)

    if points:
        _write_api.write(bucket=INFLUX_BUCKET, record=points)

    return len(points)


def fetch_latest(service: str) -> dict[str, float]:
    """
    Return the most recent field values for a service.
    Used by the anomaly detector to build the current feature vector.
    """
    fields = [
        "avg_stall_ns", "involuntary_switches", "avg_syscall_ns",
        "major_faults", "avg_lock_wait_ns", "lock_wait_count"
    ]

    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30s)
      |> filter(fn: (r) => r["service"] == "{service}")
      |> filter(fn: (r) => {" or ".join([f'r["_field"] == "{f}"' for f in fields])})
      |> last()
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """

    result = {}
    tables = _query_api.query(query)
    for table in tables:
        for record in table.records:
            for field in fields:
                val = record.values.get(field)
                if val is not None:
                    result[field] = float(val)

    return result


def fetch_history(service: str, field: str, window: str = "5m") -> list[float]:
    """
    Return a time-ordered list of values for a specific metric.
    Used by the causal engine and forecaster.

    Args:
        service: service name tag
        field:   metric field name (e.g. "avg_stall_ns")
        window:  InfluxDB duration string (e.g. "5m", "15m", "1h")
    """
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{window})
      |> filter(fn: (r) => r["service"] == "{service}")
      |> filter(fn: (r) => r["_field"] == "{field}")
      |> sort(columns: ["_time"])
    """

    values = []
    tables = _query_api.query(query)
    for table in tables:
        for record in table.records:
            v = record.get_value()
            if v is not None:
                values.append(float(v))

    return values


def fetch_multi_field_history(service: str, fields: list[str], window: str = "5m") -> dict[str, list[float]]:
    """
    Return a dict of {field: [values]} for multiple metrics simultaneously.
    Used by the Granger causality engine which needs all metrics in one pass.
    """
    field_filter = " or ".join([f'r["_field"] == "{f}"' for f in fields])

    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{window})
      |> filter(fn: (r) => r["service"] == "{service}")
      |> filter(fn: (r) => {field_filter})
      |> sort(columns: ["_time"])
    """

    result: dict[str, list[float]] = {f: [] for f in fields}
    tables = _query_api.query(query)
    for table in tables:
        field_name = table.columns[1].default_value if table.columns else None
        for record in table.records:
            fname = record.get_field()
            v = record.get_value()
            if fname in result and v is not None:
                result[fname].append(float(v))

    return result


def list_active_services(window: str = "1m") -> list[str]:
    """
    Return all service names that have reported metrics in the last window.
    """
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{window})
      |> filter(fn: (r) => r["_field"] == "avg_stall_ns")
      |> keep(columns: ["service"])
      |> distinct(column: "service")
    """

    services = []
    tables = _query_api.query(query)
    for table in tables:
        for record in table.records:
            svc = record.get_value()
            if svc:
                services.append(svc)

    return services
