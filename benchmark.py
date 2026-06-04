from typing import Tuple
import os
import sys
import time
import ctypes
import struct
import numpy as np

# Configuration parameters for scale benchmark
NUM_RECORDS = 1_000_000
DIMENSION = 384
BYTES_PER_RECORD = 48
DB_FILE = "lunar_scale_test.bin"

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

class LcvkEngine:
    """Python OOP wrapper for the AOT-compiled Lunar Custom Vector Kernel (LCVK) native library."""
    
    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Native shared library not found at: {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        self.isolate = ctypes.POINTER(GraalIsolate)()
        self.thread = ctypes.POINTER(GraalIsolateThread)()
        
        # Configure ctypes C-API signatures
        self.lib.graal_create_isolate.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.POINTER(GraalIsolate)), 
            ctypes.POINTER(ctypes.POINTER(GraalIsolateThread))
        ]
        self.lib.graal_create_isolate.restype = ctypes.c_int
        
        self.lib.graal_tear_down_isolate.argtypes = [ctypes.c_void_p]
        self.lib.graal_tear_down_isolate.restype = ctypes.c_int

        self.lib.vdb_init.argtypes = [ctypes.c_void_p]
        self.lib.vdb_init.restype = ctypes.c_int

        self.lib.vdb_load_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.vdb_load_index.restype = ctypes.c_int

        self.lib.vdb_batch_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_batch_search.restype = ctypes.c_int

        self.lib.vdb_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_size.restype = ctypes.c_longlong

        # Instantiate isolate thread context
        status = self.lib.graal_create_isolate(None, ctypes.byref(self.isolate), ctypes.byref(self.thread))
        if status != 0:
            raise RuntimeError("Failed to allocate GraalVM isolate thread.")
            
        # Initialize internal DB coordinator
        status = self.lib.vdb_init(self.thread)
        if status != 0:
            raise RuntimeError("Failed to initialize LCVK DB engine.")

    def load_index(self, index_name: str, file_path: str) -> int:
        name_bytes = index_name.encode("utf-8")
        path_bytes = file_path.encode("utf-8")
        return self.lib.vdb_load_index(self.thread, name_bytes, path_bytes)

    def batch_search(self, index_name: str, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        num_queries = queries.shape[0]
        out_ids = np.zeros(num_queries * k, dtype=np.int64)
        out_distances = np.zeros(num_queries * k, dtype=np.int32)
        
        c_index_name = index_name.encode("utf-8")
        query_ptr = queries.ctypes.data_as(ctypes.c_void_p)
        ids_ptr = out_ids.ctypes.data_as(ctypes.c_void_p)
        dists_ptr = out_distances.ctypes.data_as(ctypes.c_void_p)
        
        status = self.lib.vdb_batch_search(
            self.thread, c_index_name, query_ptr, num_queries, k, ids_ptr, dists_ptr
        )
        if status != 0:
            raise RuntimeError(f"Search failed with code: {status}")
            
        return out_ids, out_distances

    def size(self, index_name: str) -> int:
        c_index = index_name.encode("utf-8")
        return self.lib.vdb_size(self.thread, c_index)

    def close(self):
        if self.thread:
            self.lib.graal_tear_down_isolate(self.thread)
            self.thread = None
            self.isolate = None

def generate_lcvk_file(file_path: str, num_records: int):
    # Binary PLAN magic header structure (64-byte aligned)
    magic = b"PLAN"
    planet_id = 1  # Moon
    header_data = struct.pack("<BQQ", planet_id, num_records, 1737400)
    padding = b"\x00" * 43
    header = magic + header_data + padding
    assert len(header) == 64
    
    # Stream high-entropy binary vectors
    vectors = np.random.randint(0, 256, size=(num_records, BYTES_PER_RECORD), dtype=np.uint8)
    
    with open(file_path, "wb") as f:
        f.write(header)
        f.write(vectors.tobytes())

def run_benchmark():
    # 1. Generate scale dataset
    generate_lcvk_file(DB_FILE, NUM_RECORDS)
    
    # 2. Resolve native library path
    so_paths = ["./build-output/liblunar_core.so", "./liblunar_core.so"]
    lib_path = None
    for p in so_paths:
        if os.path.exists(p):
            lib_path = p
            break
            
    if not lib_path:
        print("[Error] LCVK native library not found in search paths.", file=sys.stderr)
        sys.exit(1)
        
    engine = LcvkEngine(lib_path)
    
    # 3. Off-heap memory map via Panama FFM
    t_load_start = time.perf_counter()
    status = engine.load_index("lunar_index", DB_FILE)
    t_load_ms = (time.perf_counter() - t_load_start) * 1000.0
    
    if status != 0:
        print(f"[Error] Failed to load index. Code: {status}", file=sys.stderr)
        sys.exit(1)
        
    total_records = engine.size("lunar_index")
    db_size_mb = (total_records * BYTES_PER_RECORD) / (1024.0 * 1024.0)
    
    # 4. Prepare batch queries
    num_queries = 278
    k_neighbors = 100
    queries = np.random.randint(-2**63, 2**63 - 1, size=(num_queries, 6), dtype=np.int64)
    
    # 5. Execute parallel vector scan
    t_search_start = time.perf_counter()
    out_ids, out_dists = engine.batch_search("lunar_index", queries, k_neighbors)
    t_search_sec = time.perf_counter() - t_search_start
    t_search_ms = t_search_sec * 1000.0
    
    # 6. Compute metrics
    total_comparisons = total_records * num_queries
    throughput_mvps = (total_comparisons / t_search_sec) / 1e6
    avg_latency_us = (t_search_sec * 1e6) / num_queries
    
    # Arithmetic: 6 XORs + 6 POPCNTs per 384-bit Hamming comparison = 12 operations
    giga_ops_per_second = (total_comparisons * 12.0) / t_search_sec / 1e9
    
    # Bandwidth: Zero-copy mmap means we load the 48MB database exactly once into memory caches
    effective_bandwidth_gb_s = (total_records * BYTES_PER_RECORD) / t_search_sec / 1e9
    
    # 7. Print performance report
    print("\n" + "="*80)
    print("                 LCVK PLANETARY SCALE STRESS-TEST REPORT                ")
    print("========================================================================")
    print(f" Dataset Configuration")
    print(f"  - Vector Cardinality      : {total_records:,} tiles")
    print(f"  - Database Size on Disk   : {db_size_mb:.2f} MB")
    print(f"  - Vector Dimensionality   : {DIMENSION} bits (packed off-heap)")
    print(f" Execution Workload")
    print(f"  - Query Batch Size        : {num_queries} queries (L1-saturated)")
    print(f"  - Nearest Neighbors (K)   : {k_neighbors}")
    print(f"  - Total Comparisons       : {total_comparisons:,}")
    print("------------------------------------------------------------------------")
    print(f" Performance Metrics")
    print(f"  - Memory Mapping Latency  : {t_load_ms:12.4f} ms (Zero-Copy mmap)")
    print(f"  - Batch Scan Latency      : {t_search_ms:12.3f} ms")
    print(f"  - Mean Query Latency      : {avg_latency_us:12.2f} us / query")
    print(f"  - Vector Throughput       : {throughput_mvps:12.2f} million vectors / sec (MVPS)")
    print(f"  - Computational Intensity : {giga_ops_per_second:12.2f} Giga-Operations / sec (GOPS)")
    print(f"  - Effective Scan Bandwidth: {effective_bandwidth_gb_s:12.3f} GB/s")
    print("========================================================================\n")
    
    # 8. Teardown
    engine.close()
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

if __name__ == "__main__":
    run_benchmark()
