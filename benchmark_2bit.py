import os
import sys
import time
import json
import ctypes
import contextlib
import numpy as np
import faiss
import platform

# ---------------------------------------------------------------------------
# Stderr suppression for GraalVM/Panama native warnings
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_stderr():
    sys.stderr.flush()
    err_fd = sys.stderr.fileno()
    saved = os.dup(err_fd)
    null_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(null_fd, err_fd)
    os.close(null_fd)
    try:
        yield
    finally:
        os.dup2(saved, err_fd)
        os.close(saved)


# ---------------------------------------------------------------------------
# PithosEngine (ctypes wrapper supporting V2 API)
# ---------------------------------------------------------------------------

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

class PithosEngine:
    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Native shared library not found at: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)
        self.isolate = ctypes.POINTER(GraalIsolate)()
        self.thread = ctypes.POINTER(GraalIsolateThread)()

        self.lib.graal_create_isolate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(GraalIsolate)),
            ctypes.POINTER(ctypes.POINTER(GraalIsolateThread)),
        ]
        self.lib.graal_create_isolate.restype = ctypes.c_int
        
        self.lib.vdb_init.argtypes = [ctypes.c_void_p]
        self.lib.vdb_init.restype = ctypes.c_int
        
        self.lib.vdb_load_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.vdb_load_index.restype = ctypes.c_int
        
        self.lib.vdb_load_index_with_weights.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int,
        ]
        self.lib.vdb_load_index_with_weights.restype = ctypes.c_int
        
        self.lib.vdb_batch_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.lib.vdb_batch_search.restype = ctypes.c_int
        
        self.lib.vdb_compile_index_file_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_byte, ctypes.c_longlong,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, # q_mode
        ]
        self.lib.vdb_compile_index_file_v2.restype = ctypes.c_int
        
        self.lib.vdb_set_energy_budget.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double,
        ]
        self.lib.vdb_set_energy_budget.restype = ctypes.c_int
        
        self.lib.vdb_close.argtypes = [ctypes.c_void_p]
        self.lib.vdb_close.restype = ctypes.c_int

        with suppress_stderr():
            status = self.lib.graal_create_isolate(
                None, ctypes.byref(self.isolate), ctypes.byref(self.thread)
            )
        if status != 0:
            raise RuntimeError("Failed to create GraalVM isolate.")
        with suppress_stderr():
            status = self.lib.vdb_init(self.thread)
        if status != 0:
            raise RuntimeError("Failed to initialize Pithos engine.")

    def compile_index_file_v2(self, path, planet_id, radius, dim, tiers, ids, vectors, q_mode):
        with suppress_stderr():
            return self.lib.vdb_compile_index_file_v2(
                self.thread, path.encode(), planet_id, radius, dim,
                tiers.ctypes.data_as(ctypes.c_void_p), len(tiers),
                ids.ctypes.data_as(ctypes.c_void_p),
                vectors.ctypes.data_as(ctypes.c_void_p), len(ids),
                q_mode
            )

    def load_index(self, name, path, weights=None, lora_dim=0):
        if weights is not None:
            with suppress_stderr():
                return self.lib.vdb_load_index_with_weights(
                    self.thread, name.encode(), path.encode(),
                    weights.ctypes.data_as(ctypes.c_void_p), lora_dim,
                )
        with suppress_stderr():
            return self.lib.vdb_load_index(self.thread, name.encode(), path.encode())

    def batch_search(self, name, queries, k):
        n = queries.shape[0]
        out_ids = np.zeros(n * k, dtype=np.int64)
        out_dists = np.zeros(n * k, dtype=np.int32)
        with suppress_stderr():
            status = self.lib.vdb_batch_search(
                self.thread, name.encode(),
                queries.ctypes.data_as(ctypes.c_void_p), n, k,
                out_ids.ctypes.data_as(ctypes.c_void_p),
                out_dists.ctypes.data_as(ctypes.c_void_p),
            )
        if status != 0:
            raise RuntimeError(f"batch_search failed: {status}")
        return out_ids.reshape(n, k), out_dists.reshape(n, k)

    def set_energy_budget(self, name, tau):
        with suppress_stderr():
            return self.lib.vdb_set_energy_budget(
                self.thread, name.encode(), ctypes.c_double(tau)
            )

    def close(self):
        if self.thread:
            self.lib.vdb_close(self.thread)
            self.thread = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lib_path():
    ext = "dylib" if platform.system() == "Darwin" else "so"
    for p in [f"./target/libpithos.{ext}", f"./build-output/libpithos.{ext}", f"./libpithos.{ext}"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Pithos native library not found.")


def compute_recall_at_k(gt_ids: np.ndarray, pred_ids: np.ndarray, k: int) -> float:
    recalls = []
    for i in range(gt_ids.shape[0]):
        gt_set = set(gt_ids[i, :k].tolist())
        pred_set = set(pred_ids[i, :k].tolist())
        recalls.append(len(gt_set & pred_set) / k)
    return float(np.mean(recalls))


def get_index_size_bytes(db_file):
    total_size = 0
    if os.path.exists(db_file):
        total_size += os.path.getsize(db_file)
    for ext in ["_ids.bin", "_metadata.bin"]:
        p = db_file + ext
        if os.path.exists(p):
            total_size += os.path.getsize(p)
    t = 0
    while True:
        p = f"{db_file}_tier_{t}.bin"
        if os.path.exists(p):
            total_size += os.path.getsize(p)
            t += 1
        else:
            break
    return total_size


def cleanup_index(db_file):
    for ext in ["", "_ids.bin", "_metadata.bin"]:
        p = db_file + ext
        if os.path.exists(p):
            os.remove(p)
    t = 0
    while True:
        p = f"{db_file}_tier_{t}.bin"
        if os.path.exists(p):
            os.remove(p)
            t += 1
        else:
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DIMENSION = 384
TIERS = np.array([64, 128, 256, 384], dtype=np.int32)
K_VALUES = [1, 10, 50, 100, 500, 1000, 2000, 5000]

def main():
    print("=" * 80)
    print("      PITHOS QUANTIZATION MODE COMPARISON: 1-BIT VS 2-BIT (TERNARY)")
    print("=" * 80)

    # 1. Load dataset
    queries_path = "temp/benchmark_data/queries.npy"
    db_path = "temp/benchmark_data/db_vectors_subset.npy"
    weights_path = "temp/benchmark_data/weights.npy"

    if not os.path.exists(queries_path) or not os.path.exists(db_path):
        print(f"[Error] Required dataset files not found at temp/benchmark_data/")
        sys.exit(1)

    queries = np.load(queries_path)
    db_vectors = np.load(db_path)
    num_records = db_vectors.shape[0]
    num_queries = queries.shape[0]
    max_k = max(K_VALUES)

    print(f"Database subset size: {num_records:,} vectors x {DIMENSION}D")
    print(f"Number of queries:    {num_queries}")
    print(f"Tiers:                {TIERS.tolist()}")

    # 2. FAISS Ground Truth (exact Float32 L2)
    print("\nComputing FAISS exact L2 ground truth...")
    faiss_index = faiss.IndexFlatL2(DIMENSION)
    faiss_index.add(db_vectors)
    t_faiss_start = time.perf_counter()
    gt_dists, gt_ids = faiss_index.search(queries, max_k)
    t_faiss = time.perf_counter() - t_faiss_start
    print(f"FAISS batch search finished in {t_faiss * 1000:.2f} ms")

    # Load engine
    lib_path = get_lib_path()
    print(f"Loading native library: {lib_path}")
    engine = PithosEngine(lib_path)

    # Load weights
    if os.path.exists(weights_path):
        weights = np.load(weights_path)
        print("Loaded LoRA weights for spectral energy evaluation.")
    else:
        q_mat, _ = np.linalg.qr(np.random.normal(size=(DIMENSION, DIMENSION)))
        weights = q_mat.astype(np.float32)
        print("Generated random orthogonal adapter weights.")

    ids = np.arange(num_records, dtype=np.int64)

    # Dictionary to collect results
    results = {
        "faiss": {
            "search_time_ms": round(t_faiss * 1000, 2),
            "latency_us": round((t_faiss * 1e6) / num_queries, 2),
            "throughput_mvps": round((num_queries * num_records) / (t_faiss * 1e6), 2)
        },
        "1bit": {},
        "2bit": {}
    }

    modes = [
        {"name": "1-Bit (Sign-only)", "q_mode": 0, "key": "1bit", "file": "temp/benchmark_data/pithos_1bit_test"},
        {"name": "2-Bit (Ternary / Noise Filter)", "q_mode": 1, "key": "2bit", "file": "temp/benchmark_data/pithos_2bit_test"}
    ]

    for mode in modes:
        print(f"\n--- Benchmarking {mode['name']} Mode ---")
        db_file = mode["file"]
        cleanup_index(db_file)

        # Ingestion
        t_ingest_start = time.perf_counter()
        status = engine.compile_index_file_v2(db_file, 1, 1737400, DIMENSION, TIERS, ids, db_vectors, mode["q_mode"])
        t_ingest = time.perf_counter() - t_ingest_start
        if status != 0:
            print(f"[Error] compilation failed with status: {status}")
            sys.exit(1)
        print(f"Index compiled in {t_ingest:.4f} seconds.")

        # Index File Size
        size_bytes = get_index_size_bytes(db_file)
        size_mb = size_bytes / (1024.0 * 1024.0)
        print(f"Total index size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

        # Load index
        status = engine.load_index("pithos_mode_idx", db_file, weights, DIMENSION)
        if status != 0:
            print(f"[Error] load_index failed with status: {status}")
            sys.exit(1)

        # Warmup search
        engine.batch_search("pithos_mode_idx", queries[:5], max_k)

        # Timed searches (run 5 times to get stable average)
        search_runs = 5
        search_durations = []
        for run in range(search_runs):
            t_start = time.perf_counter()
            pithos_ids, pithos_dists = engine.batch_search("pithos_mode_idx", queries, max_k)
            t_dur = time.perf_counter() - t_start
            search_durations.append(t_dur)

        t_search_avg = sum(search_durations) / len(search_durations)
        t_search_min = min(search_durations)
        print(f"Pithos search finished: Avg={t_search_avg * 1000:.2f} ms, Min={t_search_min * 1000:.2f} ms (across {search_runs} runs)")

        # Compute recall
        recalls = {}
        for k in K_VALUES:
            recalls[k] = compute_recall_at_k(gt_ids, pithos_ids, k)

        results[mode["key"]] = {
            "compile_time_sec": round(t_ingest, 4),
            "size_mb": round(size_mb, 2),
            "size_bytes": size_bytes,
            "search_time_ms": round(t_search_avg * 1000, 2),
            "latency_us": round((t_search_avg * 1e6) / num_queries, 2),
            "throughput_mvps": round((num_queries * num_records) / (t_search_avg * 1e6), 2),
            "recall": {str(k): round(recalls[k], 4) for k in K_VALUES}
        }

        # Cleanup
        engine.lib.vdb_drop_index(engine.thread, "pithos_mode_idx".encode())
        cleanup_index(db_file)

    engine.close()

    # 3. Print Comparison Table
    print("\n" + "=" * 80)
    print("                           SUMMARY OF RESULTS")
    print("=" * 80)
    print(f"{'Metric':<25} | {'FAISS':<12} | {'1-Bit (Sign)':<15} | {'2-Bit (Ternary)':<15}")
    print("-" * 80)
    
    r_1 = results["1bit"]
    r_2 = results["2bit"]
    rf = results["faiss"]

    print(f"{'Index Size (MB)':<25} | {'N/A':<12} | {r_1['size_mb']:<15.2f} | {r_2['size_mb']:<15.2f}")
    print(f"{'Compile Time (s)':<25} | {'N/A':<12} | {r_1['compile_time_sec']:<15.4f} | {r_2['compile_time_sec']:<15.4f}")
    print(f"{'Batch Search Time (ms)':<25} | {rf['search_time_ms']:<12.2f} | {r_1['search_time_ms']:<15.2f} | {r_2['search_time_ms']:<15.2f}")
    print(f"{'Query Latency (us)':<25} | {rf['latency_us']:<12.2f} | {r_1['latency_us']:<15.2f} | {r_2['latency_us']:<15.2f}")
    print(f"{'Throughput (MVPS)':<25} | {rf['throughput_mvps']:<12.2f} | {r_1['throughput_mvps']:<15.2f} | {r_2['throughput_mvps']:<15.2f}")
    print("-" * 80)
    for k in K_VALUES:
        k_str = str(k)
        print(f"{f'Recall@{k}':<25} | {1.0:<12.4f} | {r_1['recall'][k_str]:<15.4f} | {r_2['recall'][k_str]:<15.4f}")
    print("=" * 80)

    # 4. Save metrics to file
    out_metrics_path = "temp/benchmark_data/comparison_metrics.json"
    with open(out_metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {out_metrics_path}")

if __name__ == "__main__":
    main()
