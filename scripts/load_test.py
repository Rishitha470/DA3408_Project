"""Measure API throughput and latency (closed loop: N concurrent clients for D seconds).

    python scripts/load_test.py --url http://localhost:8000 --concurrency 1 4 8 --duration 10
"""
import argparse
import random
import statistics
import threading
import time

import requests

from simulate_traffic import TEST_SPLIT, load_rows, random_patient

# Real held-out rows (so load tests do not distort the drift window); generated ones as a fallback.
ROWS = load_rows() if TEST_SPLIT.exists() else None


def next_body():
    return random.choice(ROWS) if ROWS else random_patient(random)


def worker(url, stop_at, latencies, errors, lock):
    s = requests.Session()
    while time.time() < stop_at:
        t0 = time.perf_counter()
        try:
            ok = s.post(f"{url}/api/v1/predict", json=next_body(), timeout=10).status_code == 200
        except requests.RequestException:
            ok = False
        dt = (time.perf_counter() - t0) * 1000
        with lock:
            (latencies if ok else errors).append(dt)


def run(url, concurrency, duration):
    lat, err, lock = [], [], threading.Lock()
    stop_at = time.time() + duration
    threads = [threading.Thread(target=worker, args=(url, stop_at, lat, err, lock)) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lat.sort()
    pct = (lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else float("nan"))
    return {"concurrency": concurrency, "requests": len(lat), "errors": len(err),
            "rps": round(len(lat) / duration, 1), "p50_ms": round(pct(0.50), 1),
            "p95_ms": round(pct(0.95), 1), "p99_ms": round(pct(0.99), 1),
            "mean_ms": round(statistics.mean(lat), 1) if lat else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--duration", type=float, default=10)
    a = ap.parse_args()
    print(f"{'conc':>4} {'reqs':>6} {'err':>4} {'rps':>7} {'p50':>7} {'p95':>7} {'p99':>7}")
    for c in a.concurrency:
        r = run(a.url, c, a.duration)
        print(f"{r['concurrency']:>4} {r['requests']:>6} {r['errors']:>4} {r['rps']:>7} "
              f"{r['p50_ms']:>7} {r['p95_ms']:>7} {r['p99_ms']:>7}")
