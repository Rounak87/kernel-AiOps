import subprocess
import threading
import time
import argparse
import os
import multiprocessing


def cpu_stress(duration=60):
    print(f"cpu stress | {multiprocessing.cpu_count() * 2} workers | {duration}s")

    def burn():
        end = time.time() + duration
        x = 0
        while time.time() < end:
            for _ in range(10000):
                x = (x * 7 + 3) % 1000000

    workers = [threading.Thread(target=burn, daemon=True) for _ in range(multiprocessing.cpu_count() * 2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print("cpu stress complete")


def lock_stress(duration=60):
    print(f"lock stress | 16 threads | 1 shared lock | {duration}s")

    lock = threading.Lock()
    counter = [0]

    def fighter():
        end = time.time() + duration
        while time.time() < end:
            with lock:
                for _ in range(1000):
                    counter[0] += 1
            time.sleep(0.0001)

    workers = [threading.Thread(target=fighter, daemon=True) for _ in range(16)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print(f"lock stress complete | acquisitions: {counter[0]}")


def memory_stress(duration=60):
    print(f"memory stress | 4 workers | 50MB blocks | {duration}s")

    def eater():
        end = time.time() + duration
        blocks = []
        while time.time() < end:
            try:
                block = bytearray(50 * 1024 * 1024)
                for i in range(0, len(block), 4096):
                    block[i] = 42
                blocks.append(block)
                if len(blocks) > 5:
                    del blocks[0]
                time.sleep(0.1)
            except MemoryError:
                blocks.clear()
                time.sleep(1)

    workers = [threading.Thread(target=eater, daemon=True) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print("memory stress complete")


def syscall_stress(duration=60):
    print(f"syscall stress | 8 workers | fsync I/O | {duration}s")

    def hammer():
        end = time.time() + duration
        tmpdir = "/tmp/aiops_stress"
        os.makedirs(tmpdir, exist_ok=True)
        path = f"{tmpdir}/stress_{threading.get_ident()}.tmp"
        while time.time() < end:
            try:
                with open(path, "w") as f:
                    f.write("x" * 4096)
                    f.flush()
                    os.fsync(f.fileno())
                with open(path, "r") as f:
                    _ = f.read()
            except Exception:
                pass
        try:
            os.remove(path)
        except Exception:
            pass

    workers = [threading.Thread(target=hammer, daemon=True) for _ in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print("syscall stress complete")


def main():
    parser = argparse.ArgumentParser(description="kernel-aiops stress generator")
    parser.add_argument("--scenario", choices=["cpu", "lock", "memory", "syscall", "all"], required=True)
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    print(f"scenario={args.scenario} duration={args.duration}s")
    time.sleep(1)

    scenarios = {
        "cpu": cpu_stress,
        "lock": lock_stress,
        "memory": memory_stress,
        "syscall": syscall_stress,
    }

    if args.scenario == "all":
        threads = [threading.Thread(target=fn, args=(args.duration,)) for fn in scenarios.values()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        scenarios[args.scenario](args.duration)


if __name__ == "__main__":
    main()
