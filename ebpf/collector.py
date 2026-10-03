from bcc import BPF
import time
import json
import argparse
import requests
import os
import signal
import sys
from datetime import datetime

BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct proc_metrics {
    u64 total_stall_ns;
    u64 stall_count;
    u64 involuntary_switches;
    u64 voluntary_switches;
    u64 major_faults;
    u64 minor_faults;
    u64 total_syscall_ns;
    u64 syscall_count;
    u64 total_lock_wait_ns;
    u64 lock_wait_count;
    char comm[16];
};

BPF_HASH(metrics, u32, struct proc_metrics);
BPF_HASH(sched_start, u32, u64);
BPF_HASH(syscall_start, u64, u64);
BPF_HASH(lock_start, u64, u64);

static inline struct proc_metrics *get_metrics(u32 pid) {
    struct proc_metrics *m = metrics.lookup(&pid);
    if (!m) {
        struct proc_metrics zero = {};
        bpf_get_current_comm(&zero.comm, sizeof(zero.comm));
        metrics.update(&pid, &zero);
        m = metrics.lookup(&pid);
    }
    return m;
}

TRACEPOINT_PROBE(sched, sched_switch) {
    u64 ts = bpf_ktime_get_ns();
    u32 prev_pid = args->prev_pid;
    u32 next_pid = args->next_pid;

    if (prev_pid > 0) {
        sched_start.update(&prev_pid, &ts);
        struct proc_metrics *m = get_metrics(prev_pid);
        if (m) {
            if (args->prev_state != 0) {
                m->involuntary_switches++;
            } else {
                m->voluntary_switches++;
            }
        }
    }

    if (next_pid > 0) {
        u64 *start_ts = sched_start.lookup(&next_pid);
        if (start_ts) {
            u64 delta = ts - *start_ts;
            if (delta < 10000000000ULL) {
                struct proc_metrics *m = get_metrics(next_pid);
                if (m) {
                    m->total_stall_ns += delta;
                    m->stall_count++;
                }
            }
            sched_start.delete(&next_pid);
        }
    }

    return 0;
}

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    u64 tid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    syscall_start.update(&tid, &ts);
    return 0;
}

TRACEPOINT_PROBE(raw_syscalls, sys_exit) {
    u64 tid = bpf_get_current_pid_tgid();
    u64 *start_ts = syscall_start.lookup(&tid);
    if (!start_ts) return 0;

    u64 delta = bpf_ktime_get_ns() - *start_ts;
    syscall_start.delete(&tid);

    if (delta < 1000 || delta > 5000000000ULL) return 0;

    u32 pid = tid >> 32;
    struct proc_metrics *m = get_metrics(pid);
    if (m) {
        m->total_syscall_ns += delta;
        m->syscall_count++;
    }
    return 0;
}

TRACEPOINT_PROBE(exceptions, page_fault_user) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct proc_metrics *m = get_metrics(pid);
    if (m) {
        if (args->error_code & 1) {
            m->minor_faults++;
        } else {
            m->major_faults++;
        }
    }
    return 0;
}

int trace_futex_enter(struct pt_regs *ctx) {
    u64 tid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    lock_start.update(&tid, &ts);
    return 0;
}

int trace_futex_return(struct pt_regs *ctx) {
    u64 tid = bpf_get_current_pid_tgid();
    u64 *start_ts = lock_start.lookup(&tid);
    if (!start_ts) return 0;

    u64 delta = bpf_ktime_get_ns() - *start_ts;
    lock_start.delete(&tid);

    if (delta < 1000 || delta > 5000000000ULL) return 0;

    u32 pid = tid >> 32;
    struct proc_metrics *m = get_metrics(pid);
    if (m) {
        m->total_lock_wait_ns += delta;
        m->lock_wait_count++;
    }
    return 0;
}
"""

SERVICE_MAP = {
    "python3": "ml-engine",
    "node": "api-gateway",
    "postgres": "database",
    "redis-server": "cache",
    "nginx": "load-balancer",
    "java": "order-service",
    "stress-ng": "stress-test",
    "stress": "stress-test",
    "sysbench": "benchmark",
}

EXCLUDE_PROCESSES = {"multipathd", "systemd", "kworker", "swapper", "ksoftirqd", "kthreadd"}


def get_service_name(comm):
    comm = comm.strip()
    if comm in EXCLUDE_PROCESSES:
        return None
    if comm in SERVICE_MAP:
        return SERVICE_MAP[comm]
    for key, service in SERVICE_MAP.items():
        if comm.startswith(key):
            return service
    return comm


def format_ns(ns):
    if ns < 1000:
        return f"{ns}ns"
    elif ns < 1_000_000:
        return f"{ns / 1000:.1f}us"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f}ms"
    else:
        return f"{ns / 1_000_000_000:.2f}s"


def print_header():
    print("=" * 120)
    print(f"  kernel-aiops | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | kernel {os.uname().release}")
    print("=" * 120)
    print(f"{'PID':<8} {'SERVICE':<20} {'SCHED STALL':<14} {'INVOL SW':<10} "
          f"{'SYSCALL LAT':<14} {'SYSCALLS':<10} {'MAJ FAULT':<10} "
          f"{'LOCK WAIT':<14} {'LOCKS':<8}")
    print("-" * 120)


def collect_and_send(bpf_obj, endpoint=None):
    metrics_table = bpf_obj["metrics"]
    records = []

    for pid, m in metrics_table.items():
        pid_val = pid.value
        if pid_val < 100:
            continue

        comm = m.comm.decode("utf-8", errors="replace").strip("\x00")
        if not comm:
            continue

        service = get_service_name(comm)
        if service is None:
            continue

        avg_stall = m.total_stall_ns / max(m.stall_count, 1)
        avg_syscall = m.total_syscall_ns / max(m.syscall_count, 1)
        avg_lock_wait = m.total_lock_wait_ns / max(m.lock_wait_count, 1)

        record = {
            "pid": pid_val,
            "process": comm,
            "service": service,
            "timestamp": time.time(),
            "total_stall_ns": m.total_stall_ns,
            "avg_stall_ns": int(avg_stall),
            "stall_count": m.stall_count,
            "involuntary_switches": m.involuntary_switches,
            "voluntary_switches": m.voluntary_switches,
            "total_syscall_ns": m.total_syscall_ns,
            "avg_syscall_ns": int(avg_syscall),
            "syscall_count": m.syscall_count,
            "major_faults": m.major_faults,
            "minor_faults": m.minor_faults,
            "total_lock_wait_ns": m.total_lock_wait_ns,
            "avg_lock_wait_ns": int(avg_lock_wait),
            "lock_wait_count": m.lock_wait_count,
        }

        records.append(record)

        print(f"{pid_val:<8} {service:<20} {format_ns(int(avg_stall)):<14} "
              f"{m.involuntary_switches:<10} {format_ns(int(avg_syscall)):<14} "
              f"{m.syscall_count:<10} {m.major_faults:<10} "
              f"{format_ns(int(avg_lock_wait)):<14} {m.lock_wait_count:<8}")

    metrics_table.clear()

    if endpoint and records:
        try:
            resp = requests.post(
                endpoint,
                json={"node": os.uname().nodename, "timestamp": time.time(), "metrics": records},
                timeout=2
            )
            status = f"sent {len(records)} records -> {resp.status_code}"
        except requests.exceptions.RequestException as e:
            status = f"send failed: {str(e)[:60]}"
    else:
        status = f"{len(records)} processes"

    print("-" * 120)
    print(f"  {status}")
    print("=" * 120)

    return records


def main():
    parser = argparse.ArgumentParser(description="kernel-aiops eBPF collector")
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    print("loading eBPF program...")
    b = BPF(text=BPF_PROGRAM)

    try:
        b.attach_kprobe(event="do_futex", fn_name="trace_futex_enter")
        b.attach_kretprobe(event="do_futex", fn_name="trace_futex_return")
    except Exception as e:
        print(f"futex kprobe unavailable: {e}")

    print(f"collecting | interval={args.interval}s | endpoint={args.endpoint or 'local'}")

    def signal_handler(sig, frame):
        print("\nstopping collector")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        if not args.no_clear:
            os.system("clear")
        print_header()
        collect_and_send(b, endpoint=args.endpoint)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
