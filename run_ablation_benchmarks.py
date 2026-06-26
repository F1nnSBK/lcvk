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

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

class PithosEngineAblation:
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
        
        self.lib.vdb_compile_index_file_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_byte, ctypes.c_longlong,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]
        self.lib.vdb_compile_index_file_v2.restype = ctypes.c_int
        
        self.lib.vdb_close.argtypes = [ctypes.c_void_p]
        self.lib.vdb_close.restype = ctypes.c_int

        # LSM delta buffer API
        self.lib.vdb_create_delta_buffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.vdb_create_delta_buffer.restype = ctypes.c_int
        
        self.lib.vdb_insert.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong, ctypes.c_void_p]
        self.lib.vdb_insert.restype = ctypes.c_int
        
        self.lib.vdb_delete_from_delta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong]
        self.lib.vdb_delete_from_delta.restype = ctypes.c_int
        
        self.lib.vdb_delta_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_delta_size.restype = ctypes.c_longlong
        
        self.lib.vdb_needs_flush.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_needs_flush.restype = ctypes.c_int
        
        self.lib.vdb_search_merged.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.lib.vdb_search_merged.restype = ctypes.c_int
        
        self.lib.vdb_backup_delta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.vdb_backup_delta.restype = ctypes.c_int
        
        self.lib.vdb_restore_delta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.vdb_restore_delta.restype = ctypes.c_int

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

    def compile_index_file_v2(self, path, planet_id, radius, dim, tiers, ids, vectors, qmode):
        with suppress_stderr():
            return self.lib.vdb_compile_index_file_v2(
                self.thread, path.encode(), planet_id, radius, dim,
                tiers.ctypes.data_as(ctypes.c_void_p), len(tiers),
                ids.ctypes.data_as(ctypes.c_void_p),
                vectors.ctypes.data_as(ctypes.c_void_p), len(ids), qmode
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

    # LSM methods
    def create_delta_buffer(self, name, threshold):
        return self.lib.vdb_create_delta_buffer(self.thread, name.encode(), threshold)
        
    def insert(self, name, record_id, vector):
        return self.lib.vdb_insert(self.thread, name.encode(), record_id, vector.ctypes.data_as(ctypes.c_void_p))
        
    def delete_from_delta(self, name, record_id):
        return self.lib.vdb_delete_from_delta(self.thread, name.encode(), record_id)
        
    def delta_size(self, name):
        return self.lib.vdb_delta_size(self.thread, name.encode())
        
    def needs_flush(self, name):
        return self.lib.vdb_needs_flush(self.thread, name.encode())
        
    def search_merged(self, name, query, k):
        out_ids = np.zeros(k, dtype=np.int64)
        out_dists = np.zeros(k, dtype=np.int32)
        status = self.lib.vdb_search_merged(
            self.thread, name.encode(),
            query.ctypes.data_as(ctypes.c_void_p), k,
            out_ids.ctypes.data_as(ctypes.c_void_p),
            out_dists.ctypes.data_as(ctypes.c_void_p)
        )
        if status != 0:
            raise RuntimeError(f"search_merged failed: {status}")
        return out_ids, out_dists
        
    def backup_delta(self, name, path):
        return self.lib.vdb_backup_delta(self.thread, name.encode(), path.encode())
        
    def restore_delta(self, name, path, threshold):
        return self.lib.vdb_restore_delta(self.thread, name.encode(), path.encode(), threshold)

    def close(self):
        if self.thread:
            self.lib.vdb_close(self.thread)
            self.thread = None

def get_lib_path():
    import platform
    ext = "dylib" if platform.system() == "Darwin" else "so"
    for p in [f"./target/libpithos.{ext}", f"./build-output/libpithos.{ext}", f"./libpithos.{ext}"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Pithos native library not found.")

def cleanup_index(db_file):
    for ext in ["", "_ids.bin", "_metadata.bin", "_fp16.bin"]:
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

def main():
    lib_path = get_lib_path()
    engine = PithosEngineAblation(lib_path)

    # Load data (10k unique real embeddings, 278 queries)
    db_vectors = np.load("temp/benchmark_data/db_vectors_subset.npy") # shape (10000, 384)
    queries = np.load("temp/benchmark_data/queries.npy")             # shape (278, 384)

    # --------------------------------------------------------------------------
    # 1. Ablation Study: Dimensions-Adaptive SIMD Dispatch (D <= 32)
    # Compare SIMD Float-L2 (D=32) vs Hamming 1-Bit (D=33)
    # --------------------------------------------------------------------------
    print("\n--- 1. Dimensions-Adaptive SIMD Dispatch Benchmark ---")
    
    # 1A. SIMD Float-L2 (D=32)
    db_vectors_32 = db_vectors[:, :32].astype(np.float32)
    queries_32 = queries[:, :32].astype(np.float32)
    
    # FAISS ground truth for D=32
    faiss_32 = faiss.IndexFlatL2(32)
    faiss_32.add(db_vectors_32)
    _, gt_ids_32 = faiss_32.search(queries_32, 10)
    
    db_file_32 = "temp/ablation_d32"
    cleanup_index(db_file_32)
    tiers_32 = np.array([16, 32], dtype=np.int32)
    ids_32 = np.arange(len(db_vectors_32), dtype=np.int64)
    
    # Compile and load (QMode 0 = 1-Bit)
    engine.compile_index_file_v2(db_file_32, 1, 1737400, 32, tiers_32, ids_32, db_vectors_32, 0)
    
    # orthogonal weights for D=32
    q_mat_32, _ = np.linalg.qr(np.random.normal(size=(32, 32)))
    weights_32 = q_mat_32.astype(np.float32)
    engine.load_index("idx_32", db_file_32, weights_32, 32)
    
    # Warmup
    engine.batch_search("idx_32", queries_32[:5], 10)
    
    t0 = time.perf_counter()
    pred_ids_32, _ = engine.batch_search("idx_32", queries_32, 10)
    t_32 = (time.perf_counter() - t0) * 1000.0 / len(queries_32)
    
    recalls_32 = []
    for i in range(len(queries_32)):
        r = len(set(gt_ids_32[i]) & set(pred_ids_32[i])) / 10.0
        recalls_32.append(r)
    recall_32 = np.mean(recalls_32)
    
    print(f"SIMD Float-L2 (D=32): Recall@10 = {recall_32:.4f}, Latency = {t_32:.4f} ms")
    
    # 1B. Hamming 1-Bit (D=33) (acts as proxy for Hamming D=32)
    db_vectors_33 = db_vectors[:, :33].astype(np.float32)
    queries_33 = queries[:, :33].astype(np.float32)
    
    faiss_33 = faiss.IndexFlatL2(33)
    faiss_33.add(db_vectors_33)
    _, gt_ids_33 = faiss_33.search(queries_33, 10)
    
    db_file_33 = "temp/ablation_d33"
    cleanup_index(db_file_33)
    tiers_33 = np.array([16, 33], dtype=np.int32)
    ids_33 = np.arange(len(db_vectors_33), dtype=np.int64)
    
    engine.compile_index_file_v2(db_file_33, 1, 1737400, 33, tiers_33, ids_33, db_vectors_33, 0)
    
    q_mat_33, _ = np.linalg.qr(np.random.normal(size=(33, 33)))
    weights_33 = q_mat_33.astype(np.float32)
    engine.load_index("idx_33", db_file_33, weights_33, 33)
    
    engine.batch_search("idx_33", queries_33[:5], 10)
    
    t0 = time.perf_counter()
    pred_ids_33, _ = engine.batch_search("idx_33", queries_33, 10)
    t_33 = (time.perf_counter() - t0) * 1000.0 / len(queries_33)
    
    recalls_33 = []
    for i in range(len(queries_33)):
        r = len(set(gt_ids_33[i]) & set(pred_ids_33[i])) / 10.0
        recalls_33.append(r)
    recall_33 = np.mean(recalls_33)
    
    print(f"Hamming 1-Bit (D=33): Recall@10 = {recall_33:.4f}, Latency = {t_33:.4f} ms")
    
    # --------------------------------------------------------------------------
    # 2. QMODE_FLOAT_HYBRID vs QMode 0 (1-Bit) vs QMode 1 (2-Bit) (D=32)
    # --------------------------------------------------------------------------
    print("\n--- 2. QMODE_FLOAT_HYBRID Benchmark (D=32) ---")
    
    # 2A. QMode 1 (2-Bit) D=32
    db_file_32_q1 = "temp/ablation_d32_q1"
    cleanup_index(db_file_32_q1)
    engine.compile_index_file_v2(db_file_32, 1, 1737400, 32, tiers_32, ids_32, db_vectors_32, 1) # QMode=1
    
    engine.load_index("idx_32_q1", db_file_32, weights_32, 32)
    engine.batch_search("idx_32_q1", queries_32[:5], 10)
    pred_ids_32_q1, _ = engine.batch_search("idx_32_q1", queries_32, 10)
    recalls_32_q1 = []
    for i in range(len(queries_32)):
        r = len(set(gt_ids_32[i]) & set(pred_ids_32_q1[i])) / 10.0
        recalls_32_q1.append(r)
    recall_32_q1 = np.mean(recalls_32_q1)
    
    size_q1 = sum(os.path.getsize(db_file_32 + ext) for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin"] if os.path.exists(db_file_32 + ext))
    
    # 2B. QMode 2 (Float32 Bypass) D=32
    cleanup_index(db_file_32)
    engine.compile_index_file_v2(db_file_32, 1, 1737400, 32, tiers_32, ids_32, db_vectors_32, 2) # QMode=2
    
    engine.load_index("idx_32_q2", db_file_32, weights_32, 32)
    engine.batch_search("idx_32_q2", queries_32[:5], 10)
    pred_ids_32_q2, _ = engine.batch_search("idx_32_q2", queries_32, 10)
    recalls_32_q2 = []
    for i in range(len(queries_32)):
        r = len(set(gt_ids_32[i]) & set(pred_ids_32_q2[i])) / 10.0
        recalls_32_q2.append(r)
    recall_32_q2 = np.mean(recalls_32_q2)
    
    size_q2 = sum(os.path.getsize(db_file_32 + ext) for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin"] if os.path.exists(db_file_32 + ext))
    
    # Size for QMode 0
    cleanup_index(db_file_32)
    engine.compile_index_file_v2(db_file_32, 1, 1737400, 32, tiers_32, ids_32, db_vectors_32, 0)
    size_q0 = sum(os.path.getsize(db_file_32 + ext) for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin"] if os.path.exists(db_file_32 + ext))

    print(f"QMode 0 (1-Bit):   Recall@10 = {recall_32:.4f}, Dateigröße = {size_q0/1024:.2f} KB")
    print(f"QMode 1 (2-Bit):   Recall@10 = {recall_32_q1:.4f}, Dateigröße = {size_q1/1024:.2f} KB")
    print(f"QMode 2 (Float32): Recall@10 = {recall_32_q2:.4f}, Dateigröße = {size_q2/1024:.2f} KB")

    # --------------------------------------------------------------------------
    # 3. FP16 In-Engine Reranking vs Asymmetric Reranking (D=384)
    # --------------------------------------------------------------------------
    print("\n--- 3. FP16 In-Engine Reranking Benchmark (D=384) ---")
    
    db_file_384 = "temp/ablation_d384"
    cleanup_index(db_file_384)
    tiers_384 = np.array([64, 128, 256, 384], dtype=np.int32)
    ids_384 = np.arange(len(db_vectors), dtype=np.int64)
    
    # Compile index (automatically creates _fp16.bin sidecar)
    engine.compile_index_file_v2(db_file_384, 1, 1737400, 384, tiers_384, ids_384, db_vectors, 0)
    
    # FAISS ground truth for D=384
    faiss_384 = faiss.IndexFlatL2(384)
    faiss_384.add(db_vectors)
    _, gt_ids_384 = faiss_384.search(queries, 100) # K=100
    
    # Load weights
    if os.path.exists("temp/benchmark_data/weights.npy"):
        weights_384 = np.load("temp/benchmark_data/weights.npy")
    else:
        q_mat_384, _ = np.linalg.qr(np.random.normal(size=(384, 384)))
        weights_384 = q_mat_384.astype(np.float32)
        
    # 3A. FP16 Exakt (with FP16 sidecar present)
    engine.load_index("idx_384_fp16", db_file_384, weights_384, 384)
    engine.batch_search("idx_384_fp16", queries[:5], 100)
    
    t0 = time.perf_counter()
    pred_ids_fp16, _ = engine.batch_search("idx_384_fp16", queries, 10) # check recall@10
    t_fp16 = (time.perf_counter() - t0) * 1000.0 / len(queries)
    
    # Calculate recall@10 and recall@100
    r10_fp16 = np.mean([len(set(gt_ids_384[i, :10]) & set(pred_ids_fp16[i, :10])) / 10.0 for i in range(len(queries))])
    
    # Let's search with K=100 to get recall@100
    pred_ids_fp16_k100, _ = engine.batch_search("idx_384_fp16", queries, 100)
    r100_fp16 = np.mean([len(set(gt_ids_384[i, :100]) & set(pred_ids_fp16_k100[i, :100])) / 100.0 for i in range(len(queries))])
    
    print(f"FP16 Exakt: Recall@10 = {r10_fp16:.4f}, Recall@100 = {r100_fp16:.4f}, Latency = {t_fp16:.4f} ms")
    
    # 3B. Asymmetric (rename _fp16.bin to disable it)
    os.rename(db_file_384 + "_fp16.bin", db_file_384 + "_fp16.bin.bak")
    
    engine.load_index("idx_384_asym", db_file_384, weights_384, 384)
    engine.batch_search("idx_384_asym", queries[:5], 100)
    
    t0 = time.perf_counter()
    pred_ids_asym, _ = engine.batch_search("idx_384_asym", queries, 10)
    t_asym = (time.perf_counter() - t0) * 1000.0 / len(queries)
    
    r10_asym = np.mean([len(set(gt_ids_384[i, :10]) & set(pred_ids_asym[i, :10])) / 10.0 for i in range(len(queries))])
    
    pred_ids_asym_k100, _ = engine.batch_search("idx_384_asym", queries, 100)
    r100_asym = np.mean([len(set(gt_ids_384[i, :100]) & set(pred_ids_asym_k100[i, :100])) / 100.0 for i in range(len(queries))])
    
    os.rename(db_file_384 + "_fp16.bin.bak", db_file_384 + "_fp16.bin") # restore
    
    print(f"Asymmetric: Recall@10 = {r10_asym:.4f}, Recall@100 = {r100_asym:.4f}, Latency = {t_asym:.4f} ms")

    # --------------------------------------------------------------------------
    # 4. LSM Delta-Buffer Benchmark
    # --------------------------------------------------------------------------
    print("\n--- 4. LSM Delta-Buffer Benchmark ---")
    engine.load_index("lsm_idx", db_file_384, weights_384, 384)
    
    # Create delta buffer
    status = engine.create_delta_buffer("lsm_idx", 1000)
    print(f"Create Delta Buffer status: {status}")
    
    # Insert 1000 vectors
    t0 = time.perf_counter()
    for i in range(1000):
        # Insert raw vectors from db_vectors (or dummy ones)
        # Using row index as record ID
        vec = db_vectors[i].astype(np.float32)
        engine.insert("lsm_idx", 1000000 + i, vec)
    t_insert = (time.perf_counter() - t0) * 1000.0
    print(f"Inserted 1,000 vectors in {t_insert:.2f} ms (avg {t_insert/1000:.4f} ms per vector)")
    
    # Check size
    d_size = engine.delta_size("lsm_idx")
    print(f"Delta buffer size: {d_size}")
    
    # Search merged
    single_q = queries[0].astype(np.float32)
    t0 = time.perf_counter()
    # Perform 10 queries
    for q in queries[:10]:
        engine.search_merged("lsm_idx", q.astype(np.float32), 10)
    t_search = (time.perf_counter() - t0) * 1000.0 / 10
    print(f"search_merged (10k Base + 1k Delta): avg latency = {t_search:.4f} ms")
    
    # Delete from delta
    del_status = engine.delete_from_delta("lsm_idx", 1000000)
    print(f"Delete record status: {del_status}, new size: {engine.delta_size('lsm_idx')}")
    
    # Backup delta
    backup_file = "temp/delta_backup.bin"
    if os.path.exists(backup_file):
        os.remove(backup_file)
    t0 = time.perf_counter()
    bk_status = engine.backup_delta("lsm_idx", backup_file)
    t_backup = (time.perf_counter() - t0) * 1000.0
    print(f"backup_delta status: {bk_status} in {t_backup:.2f} ms (file size: {os.path.getsize(backup_file)/1024:.2f} KB)")
    
    # Restore delta
    res_status = engine.restore_delta("lsm_idx", backup_file, 1000)
    print(f"restore_delta status: {res_status}, restored size: {engine.delta_size('lsm_idx')}")
    
    # Clean up all ablation indexes
    cleanup_index(db_file_32)
    cleanup_index(db_file_33)
    cleanup_index(db_file_384)
    if os.path.exists(backup_file):
        os.remove(backup_file)
    
    # Export all these ablation metrics to a json file
    ablation_metrics = {
        "simd_recall_32": float(recall_32),
        "simd_latency_32": float(t_32),
        "hamming_recall_33": float(recall_33),
        "hamming_latency_33": float(t_33),
        
        "qmode0_recall": float(recall_32),
        "qmode0_size_kb": float(size_q0 / 1024.0),
        "qmode1_recall": float(recall_32_q1),
        "qmode1_size_kb": float(size_q1 / 1024.0),
        "qmode2_recall": float(recall_32_q2),
        "qmode2_size_kb": float(size_q2 / 1024.0),
        
        "fp16_r10": float(r10_fp16),
        "fp16_r100": float(r100_fp16),
        "fp16_latency": float(t_fp16),
        "asym_r10": float(r10_asym),
        "asym_r100": float(r100_asym),
        "asym_latency": float(t_asym),
        
        "lsm_insert_avg_ms": float(t_insert / 1000.0),
        "lsm_search_merged_ms": float(t_search),
        "lsm_backup_ms": float(t_backup)
    }
    
    with open("temp/benchmark_data/ablation_metrics.json", "w") as f:
        json.dump(ablation_metrics, f, indent=4)
        
    engine.close()

if __name__ == "__main__":
    main()
