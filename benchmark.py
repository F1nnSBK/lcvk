from typing import Tuple
import os
import sys
import time
import ctypes
import struct
import argparse
import numpy as np

# Configuration parameters for scale benchmark
NUM_RECORDS = 10_000_000
DIMENSION = 384
BYTES_PER_RECORD = 64
DB_FILE = "lunar_scale_test.bin"

# Fixed reproducible target vector for semantic search verification
np.random.seed(42)
CAVE_VECTOR = np.random.randint(-2**63, 2**63 - 1, size=6, dtype=np.int64)

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

        self.lib.vdb_query_planetary_grid.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        ]
        self.lib.vdb_query_planetary_grid.restype = ctypes.c_longlong

        self.lib.vdb_set_chunk_size.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong
        ]
        self.lib.vdb_set_chunk_size.restype = ctypes.c_int

        self.lib.vdb_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_size.restype = ctypes.c_longlong

        self.lib.vdb_close.argtypes = [ctypes.c_void_p]
        self.lib.vdb_close.restype = ctypes.c_int

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

    def query_planetary_grid(self, index_name: str, queries: np.ndarray, families: np.ndarray, thresholds: np.ndarray, voting_mask: np.ndarray) -> int:
        c_index_name = index_name.encode("utf-8")
        query_ptr = queries.ctypes.data_as(ctypes.c_void_p)
        families_ptr = families.ctypes.data_as(ctypes.c_void_p)
        thresholds_ptr = thresholds.ctypes.data_as(ctypes.c_void_p)
        mask_ptr = voting_mask.ctypes.data_as(ctypes.c_void_p)
        num_queries = queries.shape[0]
        
        return self.lib.vdb_query_planetary_grid(
            self.thread, c_index_name, query_ptr, families_ptr, thresholds_ptr, num_queries, mask_ptr
        )

    def set_chunk_size(self, index_name: str, chunk_size: int) -> int:
        c_index_name = index_name.encode("utf-8")
        return self.lib.vdb_set_chunk_size(self.thread, c_index_name, chunk_size)

    def size(self, index_name: str) -> int:
        c_index = index_name.encode("utf-8")
        return self.lib.vdb_size(self.thread, c_index)

    def close(self):
        if self.thread:
            self.lib.vdb_close(self.thread)
            self.lib.graal_tear_down_isolate(self.thread)
            self.thread = None
            self.isolate = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="LCVK Scale Benchmark")
    parser.add_argument("--trace", action="store_true", help="Enable deep execution profiling with step-by-step timestamps")
    return parser.parse_args()

def format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds * 1e6:.2f} µs"

def print_performance_table(duration_p1: float, duration_p2: float, trace_data: dict = None):
    total_time = duration_p1 + duration_p2
    
    col1_title = "Pipeline Stage / Operational Step"
    col2_title = "Duration"
    col3_title = "Budget %"
    
    w1 = 54
    w2 = 13
    w3 = 11
    
    print("\n" + " " * 20 + "Pipeline Performance Summary")
    print(f"┌{'─' * w1}┬{'─' * w2}┬{'─' * w3}┐")
    print(f"│ {col1_title:<{w1-2}} │ {col2_title:>{w2-2}} │ {col3_title:>{w3-2}} │")
    print(f"├{'─' * w1}┼{'─' * w2}┼{'─' * w3}┤")
    
    def print_row(label, val, pct):
        print(f"│ {label:<{w1-2}} │ {val:>{w2-2}} │ {pct:>{w3-2}} │")
        
    print_row("Phase 1: Scale Dataset Setup (Total)", format_duration(duration_p1), f"{(duration_p1/total_time)*100:5.1f}%")
    if trace_data:
        for k, v in trace_data.items():
            if k.startswith("p1_"):
                name = "  └─ " + k[3:].replace("_", " ").title()
                print_row(name, format_duration(v), f"{(v/total_time)*100:5.1f}%")
                
    print_row("Phase 2: High-Performance Engine Operations (Total)", format_duration(duration_p2), f"{(duration_p2/total_time)*100:5.1f}%")
    if trace_data:
        for k, v in trace_data.items():
            if k.startswith("p2_"):
                name = "  └─ " + k[3:].replace("_", " ").title()
                print_row(name, format_duration(v), f"{(v/total_time)*100:5.1f}%")
                
    print(f"├{'─' * w1}┼{'─' * w2}┼{'─' * w3}┤")
    print_row("Total Pipeline Execution Time", format_duration(total_time), "100.0%")
    print(f"└{'─' * w1}┴{'─' * w2}┴{'─' * w3}┘\n")

def generate_lcvk_file(file_path: str, num_records: int):
    # Binary PLAN magic header structure (64-byte aligned)
    magic = b"PLAN"
    planet_id = 1  # Moon
    header_data = struct.pack("<BQQ", planet_id, num_records, 1737400)
    padding = b"\x00" * 43
    header = magic + header_data + padding
    assert len(header) == 64
    
    # Construct 64-byte cache-line aligned records:
    data = np.zeros((num_records, BYTES_PER_RECORD), dtype=np.uint8)
    
    # Set sequential IDs in the first 8 bytes of each record
    ids = np.arange(num_records, dtype=np.uint64)
    data[:, 0:8] = ids.view(np.uint8).reshape(-1, 8)
    
    # Set random binary vector data in the middle 48 bytes (offset 8 to 56)
    data[:, 8:56] = np.random.randint(0, 256, size=(num_records, 48), dtype=np.uint8)
    
    # Inject the CAVE_VECTOR at target IDs: 100, 50000, 99999
    cave_bytes = CAVE_VECTOR.view(np.uint8)
    target_ids = [100, 50000, 99999]
    for tid in target_ids:
        if tid < num_records:
            data[tid, 8:56] = cave_bytes
            
    # Offset 56 to 64 remains 0 (metadata)
    
    with open(file_path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())

def run_benchmark(trace_data=None):
    t_start_p1 = time.perf_counter()
    
    # 1. Generate scale dataset with semantic targets
    print(f"Generating scale database {DB_FILE} with {NUM_RECORDS:,} records...")
    t_gen_start = time.perf_counter()
    generate_lcvk_file(DB_FILE, NUM_RECORDS)
    if trace_data is not None:
        trace_data["p1_dataset_generation"] = time.perf_counter() - t_gen_start
        
    # 2. Resolve native library path
    t_lib_start = time.perf_counter()
    import platform
    if platform.system() == "Darwin":
        so_paths = [
            "./target/lunar_core.dylib",
            "./build-output/liblunar_core.dylib",
            "./liblunar_core.dylib",
            "./target/lunar_core.so",
            "./build-output/liblunar_core.so",
            "./liblunar_core.so"
        ]
    else:
        so_paths = [
            "./build-output/liblunar_core.so",
            "./liblunar_core.so",
            "./target/lunar_core.so",
            "./target/lunar_core.dylib",
            "./build-output/liblunar_core.dylib",
            "./liblunar_core.dylib"
        ]
    lib_path = None
    for p in so_paths:
        if os.path.exists(p):
            lib_path = p
            break
            
    if not lib_path:
        print("[Error] LCVK native library not found in search paths.", file=sys.stderr)
        sys.exit(1)
        
    engine = LcvkEngine(lib_path)
    if trace_data is not None:
        trace_data["p1_library_resolution"] = time.perf_counter() - t_lib_start
        
    duration_p1 = time.perf_counter() - t_start_p1
    
    t_start_p2 = time.perf_counter()
    
    # 3. Off-heap memory map via Panama FFM
    t_load_start = time.perf_counter()
    status = engine.load_index("lunar_index", DB_FILE)
    t_load_duration = time.perf_counter() - t_load_start
    t_load_ms = t_load_duration * 1000.0
    if trace_data is not None:
        trace_data["p2_native_mmap_load"] = t_load_duration
    
    if status != 0:
        print(f"[Error] Failed to load index. Code: {status}", file=sys.stderr)
        sys.exit(1)
        
    total_records = engine.size("lunar_index")
    db_size_mb = (total_records * BYTES_PER_RECORD) / (1024.0 * 1024.0)
    
    print(f"Index loaded successfully: {total_records:,} records ({db_size_mb:.2f} MB) in {t_load_ms:.4f} ms")

    # 4. Perform Semantic Reality-Check using Multi-Family Resonant Voting
    print("\nRunning Semantic Reality-Check...")
    voting_mask = np.zeros(total_records, dtype=np.uint8)
    
    # 8 queries matching the CAVE_VECTOR perfectly
    voting_queries = np.tile(CAVE_VECTOR, (8, 1)) # shape (8, 6)
    families = np.arange(8, dtype=np.int32)
    thresholds = np.zeros(8, dtype=np.int32)
    
    t_vote_start = time.perf_counter()
    resonant_count = engine.query_planetary_grid("lunar_index", voting_queries, families, thresholds, voting_mask)
    t_vote_duration = time.perf_counter() - t_vote_start
    t_vote_ms = t_vote_duration * 1000.0
    if trace_data is not None:
        trace_data["p2_semantic_resonance_voting"] = t_vote_duration
    
    print(f"Resonant voting scan completed in {t_vote_ms:.3f} ms. Found {resonant_count} resonant tiles.")
    
    # Assertions for semantic verification
    assert resonant_count == 3, f"Expected 3 resonant tiles, got {resonant_count}"
    
    target_ids = [100, 50000, 99999]
    for tid in target_ids:
        mask_val = voting_mask[tid]
        assert mask_val == 0xFF, f"Expected voting mask at ID {tid} to be 0xFF (perfect match for all 8 families), got {hex(mask_val)}"
        
    # Verify that only the target IDs had any family matches
    non_zero_indices = np.flatnonzero(voting_mask)
    assert set(non_zero_indices) == set(target_ids), f"Unexpected non-zero voting mask indices: {non_zero_indices.tolist()}"
    print("Semantic validation PASSED: Target tiles detected with 100% precision (0xFF voting mask).")

    # 5. Prepare batch queries for the parallel vector scan
    num_queries = 278
    k_neighbors = 100
    # Generate batch queries once to keep inputs identical across chunk sweeps
    queries = np.random.randint(-2**63, 2**63 - 1, size=(num_queries, 6), dtype=np.int64)
    
    # 6. Sweep chunk sizes to find the hardware sweet spot
    chunk_sizes = [1000, 5000, 10000, 20000, 50000]
    sweep_results = []
    
    print("\n" + "="*80)
    print("                 LCVK CHUNK-SIZE PERFORMANCE SWEEP REPORT               ")
    print("========================================================================")
    print(f" {'Chunk Size':<12} | {'Scan Latency':<16} | {'Avg Query Latency':<20} | {'Throughput':<18} ")
    print("------------------------------------------------------------------------")
    
    t_sweep_start = time.perf_counter()
    for chunk_size in chunk_sizes:
        # Set chunk size dynamically
        engine.set_chunk_size("lunar_index", chunk_size)
        
        # Execute parallel vector scan
        t_search_start = time.perf_counter()
        out_ids, out_dists = engine.batch_search("lunar_index", queries, k_neighbors)
        t_search_sec = time.perf_counter() - t_search_start
        t_search_ms = t_search_sec * 1000.0
        
        # Compute metrics
        total_comparisons = total_records * num_queries
        throughput_mvps = (total_comparisons / t_search_sec) / 1e6
        avg_latency_us = (t_search_sec * 1e6) / num_queries
        
        print(f" {chunk_size:<12,} | {t_search_ms:12.3f} ms | {avg_latency_us:16.2f} us | {throughput_mvps:12.2f} MVPS ")
        sweep_results.append((chunk_size, t_search_ms, avg_latency_us, throughput_mvps, t_search_sec))
        
    if trace_data is not None:
        trace_data["p2_batch_search_chunk_sweep"] = time.perf_counter() - t_sweep_start
        
    print("========================================================================\n")
    
    # Find the best chunk size based on highest MVPS
    best_run = max(sweep_results, key=lambda x: x[3])
    best_chunk, best_ms, best_lat, best_mvps, best_sec = best_run
    
    # 7. Print overall summary report for the best run
    total_comparisons = total_records * num_queries
    giga_ops_per_second = (total_comparisons * 12.0) / best_sec / 1e9
    effective_bandwidth_gb_s = (total_records * BYTES_PER_RECORD) / best_sec / 1e9
    
    print("="*80)
    print("                 LCVK PLANETARY SCALE STRESS-TEST REPORT (BEST RUN)     ")
    print("========================================================================")
    print(f" Dataset Configuration")
    print(f"  - Vector Cardinality      : {total_records:,} tiles")
    print(f"  - Database Size on Disk   : {db_size_mb:.2f} MB")
    print(f"  - Vector Dimensionality   : {DIMENSION} bits (packed off-heap)")
    print(f" Execution Workload")
    print(f"  - Query Batch Size        : {num_queries} queries (L1-saturated)")
    print(f"  - Nearest Neighbors (K)   : {k_neighbors}")
    print(f"  - Total Comparisons       : {total_comparisons:,}")
    print(f"  - Optimal Chunk Size      : {best_chunk:,}")
    print("------------------------------------------------------------------------")
    print(f" Performance Metrics")
    print(f"  - Memory Mapping Latency  : {t_load_ms:12.4f} ms (Zero-Copy mmap)")
    print(f"  - Batch Scan Latency      : {best_ms:12.3f} ms")
    print(f"  - Mean Query Latency      : {best_lat:12.2f} us / query")
    print(f"  - Vector Throughput       : {best_mvps:12.2f} million vectors / sec (MVPS)")
    print(f"  - Computational Intensity : {giga_ops_per_second:12.2f} Giga-Operations / sec (GOPS)")
    print(f"  - Effective Scan Bandwidth: {effective_bandwidth_gb_s:12.3f} GB/s")
    print("========================================================================\n")
    
    duration_p2 = time.perf_counter() - t_start_p2
    
    # 8. Teardown
    engine.close()
    
    # Save values to lcvk_metrics.json
    import json
    metrics_path = "lcvk_metrics.json"
    metrics_data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            pass
            
    metrics_data.update({
        "best_mvps": float(best_mvps)
    })
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        
    # Generate updated plots automatically
    try:
        from generate_graphics import create_throughput_plot
        create_throughput_plot()
        print("Updated throughput comparison plot successfully.")
    except Exception as e:
        print(f"Warning: Could not regenerate throughput plot ({e})")
        
    print_performance_table(duration_p1, duration_p2, trace_data)

if __name__ == "__main__":
    main_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_path)
    args = parse_arguments()
    trace_data = {} if args.trace else None
    run_benchmark(trace_data)
