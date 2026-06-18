"""
Recall@K Benchmark -- Speed-Accuracy Trade-Off Analysis

Measures how many of FAISS Flat L2's exact Top-K neighbors Pithos recovers
after 1-bit binarization and Matryoshka cascading. This is the scientific
proof that the speedup does not come at unacceptable accuracy cost.

Ground truth: FAISS IndexFlatL2 (exact brute-force L2 in float32 space).
Challenger:   Pithos Host-Native (binary Hamming cascade via AOT Java 25).
"""

import os
import sys
import time
import json
import ctypes
import contextlib
import numpy as np
import faiss


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
# PithosEngine (minimal ctypes wrapper, same as benchmark.py)
# ---------------------------------------------------------------------------

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

class PithosEngine:
    def __init__(self, lib_path: str):
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
        self.lib.vdb_compile_index_file.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_byte, ctypes.c_longlong,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ]
        self.lib.vdb_compile_index_file.restype = ctypes.c_int
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

    def compile_index_file(self, path, planet_id, radius, dim, tiers, ids, vectors):
        with suppress_stderr():
            return self.lib.vdb_compile_index_file(
                self.thread, path.encode(), planet_id, radius, dim,
                tiers.ctypes.data_as(ctypes.c_void_p), len(tiers),
                ids.ctypes.data_as(ctypes.c_void_p),
                vectors.ctypes.data_as(ctypes.c_void_p), len(ids),
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
    import platform
    ext = "dylib" if platform.system() == "Darwin" else "so"
    for p in [f"./target/libpithos.{ext}", f"./build-output/libpithos.{ext}", f"./libpithos.{ext}"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Pithos native library not found.")


def compute_recall_at_k(gt_ids: np.ndarray, pred_ids: np.ndarray, k: int) -> float:
    """
    gt_ids:   (num_queries, K_gt) -- ground truth neighbor IDs from FAISS
    pred_ids: (num_queries, K_pred) -- predicted neighbor IDs from Pithos
    Returns mean Recall@K across all queries.
    """
    recalls = []
    for i in range(gt_ids.shape[0]):
        gt_set = set(gt_ids[i, :k].tolist())
        pred_set = set(pred_ids[i, :k].tolist())
        recalls.append(len(gt_set & pred_set) / k)
    return float(np.mean(recalls))


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
K_VALUES = [1, 10, 50, 100]
DB_FILE = "pithos_recall_test"


def main():
    print("=" * 72)
    print("    RECALL@K BENCHMARK -- Speed-Accuracy Trade-Off")
    print("=" * 72)

    # 1. Load real data
    if not os.path.exists("queries.npy") or not os.path.exists("db_vectors_subset.npy"):
        print("[Error] queries.npy or db_vectors_subset.npy not found.")
        print("        Run ingest_pipeline.py and query_generator.py first.")
        sys.exit(1)

    queries = np.load("queries.npy")
    db_vectors = np.load("db_vectors_subset.npy")  # 10k unique real embeddings

    num_records = db_vectors.shape[0]
    num_queries = queries.shape[0]
    max_k = max(K_VALUES)

    print(f"Database: {num_records:,} unique records x {DIMENSION}D (real lunar embeddings)")
    print(f"Queries:  {num_queries} (real DINOv3 lunar queries)")
    print(f"K values: {K_VALUES}")

    # 2. FAISS Ground Truth (exact L2)
    print("\nComputing FAISS Flat L2 ground truth...", end=" ", flush=True)
    faiss_index = faiss.IndexFlatL2(DIMENSION)
    faiss_index.add(db_vectors)
    t0 = time.perf_counter()
    gt_dists, gt_ids = faiss_index.search(queries, max_k)
    t_faiss = time.perf_counter() - t0
    print(f"done in {t_faiss * 1000:.1f} ms")

    # 3. Pithos search
    print("Compiling Pithos index...", end=" ", flush=True)
    lib_path = get_lib_path()
    engine = PithosEngine(lib_path)

    ids = np.arange(num_records, dtype=np.int64)
    status = engine.compile_index_file(DB_FILE, 1, 1737400, DIMENSION, TIERS, ids, db_vectors)
    if status != 0:
        print(f"\n[Error] compile failed: {status}")
        sys.exit(1)

    # Load with real LoRA weights for authentic SVD energy computation
    if os.path.exists("weights.npy"):
        weights = np.load("weights.npy")
        print("done (using real LoRA weights)")
    else:
        q_mat, _ = np.linalg.qr(np.random.normal(size=(DIMENSION, DIMENSION)))
        weights = q_mat.astype(np.float32)
        print("done (using QR mock weights)")

    status = engine.load_index("recall_idx", DB_FILE, weights, DIMENSION)
    if status != 0:
        print(f"[Error] load failed: {status}")
        sys.exit(1)

    # Warmup
    engine.batch_search("recall_idx", queries[:5], max_k)

    # Timed Pithos search
    print("Running Pithos batch search...", end=" ", flush=True)
    t0 = time.perf_counter()
    pithos_ids, pithos_dists = engine.batch_search("recall_idx", queries, max_k)
    t_pithos = time.perf_counter() - t0
    print(f"done in {t_pithos * 1000:.1f} ms")

    engine.close()
    cleanup_index(DB_FILE)

    # 4. Compute Recall@K
    print("\n" + "=" * 72)
    print("    RECALL@K RESULTS (Ground Truth: FAISS Flat L2)")
    print("=" * 72)

    header = f"{'K':>6}  {'Recall@K':>10}  {'FAISS (ms)':>12}  {'Pithos (ms)':>12}  {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    recall_results = []
    for k in K_VALUES:
        recall = compute_recall_at_k(gt_ids, pithos_ids, k)
        speedup = t_faiss / t_pithos if t_pithos > 0 else 0

        print(
            f"{k:>6}  {recall:>10.4f}  {t_faiss * 1000:>12.2f}  "
            f"{t_pithos * 1000:>12.2f}  {speedup:>9.1f}x"
        )

        recall_results.append({
            "k": k,
            "recall": round(recall, 4),
            "faiss_ms": round(t_faiss * 1000, 2),
            "pithos_ms": round(t_pithos * 1000, 2),
            "speedup": round(speedup, 1),
        })

    print("=" * 72)

    # Summary
    r10 = next((r for r in recall_results if r["k"] == 10), None)
    r100 = next((r for r in recall_results if r["k"] == 100), None)
    if r10 and r100:
        print(
            f"\nSummary: Recall@10 = {r10['recall']:.2%}, "
            f"Recall@100 = {r100['recall']:.2%}, "
            f"Speedup = {r100['speedup']:.1f}x"
        )

    # 5. Export
    metrics_path = "recall_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "dataset": "lunar_real_data",
            "num_records": num_records,
            "num_queries": num_queries,
            "dimension": DIMENSION,
            "tiers": TIERS.tolist(),
            "recall_results": recall_results,
        }, f, indent=2)
    print(f"Metrics exported to {metrics_path}")


if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)
    main()
