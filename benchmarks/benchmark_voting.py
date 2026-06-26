"""
Discipline 2: Application-Specific Resonant Voting Stress-Test

Compares Pithos's native Multi-Family Resonant Voting kernel
(vdb_query_planetary_grid) against an emulated FAISS equivalent
that must perform the same logical operation via batch search +
Python-side threshold filtering + manual bitmask aggregation.

This benchmark exposes the architectural gap between a general-purpose
index emulating voting and a purpose-built hardware-aligned filter kernel.
"""

import os
import sys
import time
import json
import ctypes
import contextlib
import numpy as np
import faiss


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
# PithosEngine (voting-capable subset)
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
        self.lib.vdb_compile_index_file.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_byte, ctypes.c_longlong,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ]
        self.lib.vdb_compile_index_file.restype = ctypes.c_int
        self.lib.vdb_query_planetary_grid.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ]
        self.lib.vdb_query_planetary_grid.restype = ctypes.c_longlong
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

    def query_planetary_grid(self, name, queries, families, thresholds, voting_mask):
        with suppress_stderr():
            return self.lib.vdb_query_planetary_grid(
                self.thread, name.encode(),
                queries.ctypes.data_as(ctypes.c_void_p),
                families.ctypes.data_as(ctypes.c_void_p),
                thresholds.ctypes.data_as(ctypes.c_void_p),
                queries.shape[0],
                voting_mask.ctypes.data_as(ctypes.c_void_p),
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
# FAISS Emulated Resonant Voting
# ---------------------------------------------------------------------------

def faiss_emulated_voting(db_vectors, queries, families, thresholds, num_records):
    """
    Emulates Pithos Multi-Family Resonant Voting using FAISS Flat L2.

    FAISS has no native bitmask voting, so we must:
    1. Compute all pairwise L2 distances via FAISS batch search
    2. Convert L2 thresholds to approximate Hamming equivalents
    3. Loop through results and manually OR family bits into a voting mask

    This is the best a general-purpose index can do to replicate Pithos voting.
    """
    dim = db_vectors.shape[1]
    num_queries = queries.shape[0]

    voting_mask = np.zeros(num_records, dtype=np.uint8)

    # Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(db_vectors)

    # For each query, search entire database and apply threshold filtering.
    # FAISS range_search is the fair equivalent (returns all vectors within radius).
    # We convert Hamming thresholds to approximate L2 radii.
    # For unit-norm vectors: L2^2 = 2 - 2*cos(theta), Hamming ~ D/2 * (1 - cos(theta))
    # So L2^2 ~ 2 * Hamming / (D/2) = 4 * Hamming / D
    for i in range(num_queries):
        hamming_threshold = thresholds[i]
        family_id = families[i]
        l2_radius_sq = 4.0 * float(hamming_threshold) / float(dim)

        # range_search returns (lims, D, I) where lims[i]:lims[i+1] are results for query i
        lims, dists, ids = index.range_search(queries[i:i+1], l2_radius_sq)
        start, end = int(lims[0]), int(lims[1])
        matched_ids = ids[start:end]

        # Apply family bitmask OR
        voting_mask[matched_ids] |= np.uint8(1 << family_id)

    return voting_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DIMENSION = 384
TIERS = np.array([64, 128, 256, 384], dtype=np.int32)
DB_FILE = "pithos_voting_test"
NUM_WARMUP_RUNS = 2
NUM_TIMED_RUNS = 5


def main():
    print("=" * 72)
    print("    DISCIPLINE 2: RESONANT VOTING STRESS-TEST")
    print("=" * 72)

    # 1. Load real data
    for f in ["temp/benchmark_data/queries.npy", "temp/benchmark_data/db_vectors_subset.npy", "temp/benchmark_data/families.npy", "temp/benchmark_data/thresholds.npy"]:
        if not os.path.exists(f):
            print(f"[Error] {f} not found. Run ingest_pipeline.py first.")
            sys.exit(1)

    queries = np.load("temp/benchmark_data/queries.npy")             # (278, 384)
    families = np.load("temp/benchmark_data/families.npy")            # (278,) int32, values 0-7
    thresholds = np.load("temp/benchmark_data/thresholds.npy")        # (278,) int32
    db_subset = np.load("temp/benchmark_data/db_vectors_subset.npy")  # (10000, 384)

    # Replicate to 100k with micro-noise for scale parity with throughput benchmarks
    np.random.seed(42)
    db_vectors = np.tile(db_subset, (10, 1))
    db_vectors = db_vectors + np.random.normal(0.0, 1e-5, db_vectors.shape).astype(np.float32)

    num_records = db_vectors.shape[0]
    num_queries = queries.shape[0]
    total_comparisons = num_records * num_queries

    print(f"Database:   {num_records:,} records x {DIMENSION}D")
    print(f"Queries:    {num_queries} (8 scientific criteria families)")
    print(f"Threshold:  {thresholds[0]} Hamming bits (F1-optimized)")
    print(f"Comparisons: {total_comparisons:,}")

    # ------------------------------------------------------------------
    # A. FAISS Emulated Resonant Voting
    # ------------------------------------------------------------------
    print("\nRunning FAISS Emulated Resonant Voting...", end=" ", flush=True)

    # Warmup
    for _ in range(NUM_WARMUP_RUNS):
        _ = faiss_emulated_voting(db_vectors, queries[:10], families[:10], thresholds[:10], num_records)

    # Timed runs
    faiss_times = []
    faiss_mask = None
    for _ in range(NUM_TIMED_RUNS):
        t0 = time.perf_counter()
        faiss_mask = faiss_emulated_voting(db_vectors, queries, families, thresholds, num_records)
        faiss_times.append(time.perf_counter() - t0)

    t_faiss_best = min(faiss_times)
    faiss_mvps = (total_comparisons / t_faiss_best) / 1e6
    faiss_resonant = int(np.count_nonzero(faiss_mask))
    print(f"done in {t_faiss_best * 1000:.2f} ms (best of {NUM_TIMED_RUNS})")

    # ------------------------------------------------------------------
    # B. Pithos Native Resonant Voting
    # ------------------------------------------------------------------
    print("Running Pithos Native Resonant Voting...", end=" ", flush=True)

    lib_path = get_lib_path()
    engine = PithosEngine(lib_path)

    ids = np.arange(num_records, dtype=np.int64)
    status = engine.compile_index_file(DB_FILE, 1, 1737400, DIMENSION, TIERS, ids, db_vectors)
    if status != 0:
        print(f"\n[Error] compile failed: {status}")
        sys.exit(1)

    if os.path.exists("temp/benchmark_data/weights.npy"):
        weights = np.load("temp/benchmark_data/weights.npy")
    else:
        q_mat, _ = np.linalg.qr(np.random.normal(size=(DIMENSION, DIMENSION)))
        weights = q_mat.astype(np.float32)

    status = engine.load_index("voting_idx", DB_FILE, weights, DIMENSION)
    if status != 0:
        print(f"[Error] load failed: {status}")
        sys.exit(1)

    # Warmup
    for _ in range(NUM_WARMUP_RUNS):
        warmup_mask = np.zeros(num_records, dtype=np.uint8)
        engine.query_planetary_grid("voting_idx", queries[:10], families[:10], thresholds[:10], warmup_mask)

    # Timed runs
    pithos_times = []
    pithos_resonant = 0
    for _ in range(NUM_TIMED_RUNS):
        pithos_mask = np.zeros(num_records, dtype=np.uint8)
        t0 = time.perf_counter()
        pithos_resonant = engine.query_planetary_grid(
            "voting_idx", queries, families, thresholds, pithos_mask
        )
        pithos_times.append(time.perf_counter() - t0)

    t_pithos_best = min(pithos_times)
    pithos_mvps = (total_comparisons / t_pithos_best) / 1e6
    print(f"done in {t_pithos_best * 1000:.2f} ms (best of {NUM_TIMED_RUNS})")

    engine.close()
    cleanup_index(DB_FILE)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    speedup = t_faiss_best / t_pithos_best if t_pithos_best > 0 else 0

    print("\n" + "=" * 72)
    print("    RESONANT VOTING RESULTS")
    print("=" * 72)

    w1, w2, w3, w4 = 34, 16, 18, 10
    print(f"{'Backend':<{w1}} | {'Total Time (ms)':>{w2}} | {'Throughput (MVPS)':>{w3}} | {'Speedup':>{w4}}")
    print("-" * (w1 + w2 + w3 + w4 + 9))
    print(
         f"{'FAISS Emulated Voting':<{w1}} | "
         f"{t_faiss_best * 1000:>{w2}.2f} | "
         f"{faiss_mvps:>{w3},.2f} | "
         f"{'1.0x':>{w4}}"
    )
    print(
         f"{'Pithos Native FFM Kernel':<{w1}} | "
         f"{t_pithos_best * 1000:>{w2}.2f} | "
         f"{pithos_mvps:>{w3},.2f} | "
         f"{speedup:>{w4 - 1}.1f}x"
    )
    print("=" * 72)

    print(f"\nFAISS resonant records: {faiss_resonant:,}")
    print(f"Pithos resonant records: {pithos_resonant:,}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    metrics = {
        "dataset": "lunar_real_data_100k",
        "num_records": num_records,
        "num_queries": num_queries,
        "num_families": 8,
        "threshold": int(thresholds[0]),
        "faiss_emulated": {
            "time_ms": round(t_faiss_best * 1000, 2),
            "mvps": round(faiss_mvps, 2),
            "resonant_records": faiss_resonant,
        },
        "pithos_native": {
            "time_ms": round(t_pithos_best * 1000, 2),
            "mvps": round(pithos_mvps, 2),
            "resonant_records": int(pithos_resonant),
        },
        "speedup": round(speedup, 1),
    }

    with open("temp/benchmark_data/voting_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics exported to temp/benchmark_data/voting_metrics.json")


if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)
    main()
