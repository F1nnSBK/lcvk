import os
import sys
import time
import json
import ctypes
import contextlib
import numpy as np
import faiss
import platform

# PYTHONPATH fallback and PithosMIDB Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark import PithosMIDB


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
    engine = PithosMIDB()

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
        status = engine.compile_index_file(db_file, 1, 1737400, DIMENSION, TIERS, ids, db_vectors, mode["q_mode"])
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
