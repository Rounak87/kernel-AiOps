"""
Telemetry inspection CLI utility.

Queries InfluxDB and displays the latest per-service kernel telemetry
(scheduler stalls, involuntary context switches, syscall duration, lock contention).
"""
from pipeline.db import list_active_services, fetch_latest


def inspect_live_metrics(window: str = "10m") -> None:
    services = list_active_services(window=window)
    if not services:
        print(f"No active services detected in the last {window}.")
        return

    print("=" * 88)
    print(f"{'SERVICE':<24} {'STALL (ms)':<14} {'INVOL SW':<12} {'SYSCALL (ms)':<16} {'LOCK WAIT (ms)':<16}")
    print("-" * 88)

    for s in sorted(services):
        m = fetch_latest(s, window=window)
        if not m:
            continue
        stall_ms = m.get("avg_stall_ns", 0) / 1e6
        invol = m.get("involuntary_switches", 0)
        syscall_ms = m.get("avg_syscall_ns", 0) / 1e6
        lock_ms = m.get("avg_lock_wait_ns", 0) / 1e6

        print(f"{s:<24} {stall_ms:<14.2f} {invol:<12.0f} {syscall_ms:<16.2f} {lock_ms:<16.2f}")

    print("=" * 88)
    print(f"Total reporting services: {len(services)}")


if __name__ == "__main__":
    inspect_live_metrics()
