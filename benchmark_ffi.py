import time
import os
import ctypes
import json
import numpy as np

# Load native library
lib_path = "./libpithos.dylib"
if not os.path.exists(lib_path):
    # Try current directory first, then target/
    lib_path = "target/libpithos.dylib"

print(f"=== Pithos FFI Boundary Analysis ===")
print(f"Loading native library from: {lib_path}")

lib = ctypes.CDLL(lib_path)

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

lib.graal_create_isolate.argtypes = [
    ctypes.c_void_p, 
    ctypes.POINTER(ctypes.POINTER(GraalIsolate)), 
    ctypes.POINTER(ctypes.POINTER(GraalIsolateThread))
]
lib.graal_create_isolate.restype = ctypes.c_int

lib.vdb_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.vdb_size.restype = ctypes.c_longlong

isolate = ctypes.POINTER(GraalIsolate)()
thread = ctypes.POINTER(GraalIsolateThread)()

status = lib.graal_create_isolate(None, ctypes.byref(isolate), ctypes.byref(thread))
if status != 0:
    raise RuntimeError("Failed to create GraalVM isolate")

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

with open("ffi_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("FFI metrics saved to ffi_metrics.json")
