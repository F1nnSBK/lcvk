import time
import os
import ctypes
import json
import sys

# PYTHONPATH fallback and PithosMIDB Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark import PithosMIDB

print("=== Pithos FFI Boundary Analysis ===")

# Retrieve centralized singleton instance (loads lib and initializes isolate thread)
db = PithosMIDB()
lib = db.lib
thread = db.thread

# 1. Warmup
print("Warming up FFI bridge...")
for _ in range(5000):
    lib.vdb_size(thread, b"warmup_dummy")

# 2. Benchmark ctypes FFI calls
print("Running 100,000 FFI calls...")
iterations = 100000
start_time = time.perf_counter()
for _ in range(iterations):
    lib.vdb_size(thread, b"bench_dummy")
end_time = time.perf_counter()

total_time_ms = (end_time - start_time) * 1000.0
avg_time_us = (total_time_ms * 1000.0) / iterations

print(f"Total time for {iterations} calls: {total_time_ms:.2f} ms")
print(f"Average FFI border crossing latency: {avg_time_us:.4f} us")

# Compare with pure Python function call overhead
def py_noop(x):
    return len(x)

start_py = time.perf_counter()
for _ in range(iterations):
    py_noop(b"bench_dummy")
end_py = time.perf_counter()
py_time_us = ((end_py - start_py) * 1e6) / iterations

print(f"Pure Python call overhead: {py_time_us:.4f} us")

# Export results
metrics = {
    "ffi_iterations": iterations,
    "total_ffi_time_ms": total_time_ms,
    "avg_ffi_latency_us": avg_time_us,
    "avg_python_overhead_us": py_time_us,
    "net_ffi_overhead_us": max(0.0, avg_time_us - py_time_us)
}

os.makedirs("temp/benchmark_data", exist_ok=True)
metrics_path = "temp/benchmark_data/ffi_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"FFI metrics saved to {metrics_path}")
