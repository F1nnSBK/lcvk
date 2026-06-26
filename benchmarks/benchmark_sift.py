import os
import sys
import tarfile
import urllib.request
import numpy as np
import time
import json
import faiss
import ctypes
import contextlib
import struct

# Suppress native stderr warnings
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

# Pithos FFI Bridge Client
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

    def load_index(self, name, path):
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

    def close(self):
        if self.thread:
            self.lib.vdb_close(self.thread)
            self.thread = None


def read_fvecs(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    if len(data) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    d = struct.unpack('i', data[:4])[0]
    record_size = 4 + 4 * d
    n = len(data) // record_size
    fv = np.frombuffer(data, dtype=np.float32).reshape(n, d + 1)
    return fv[:, 1:].copy()


def read_ivecs(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    if len(data) == 0:
        return np.zeros((0, 0), dtype=np.int32)
    d = struct.unpack('i', data[:4])[0]
    record_size = 4 + 4 * d
    n = len(data) // record_size
    iv = np.frombuffer(data, dtype=np.int32).reshape(n, d + 1)
    return iv[:, 1:].copy()


def get_lib_path():
    import platform
    ext = "dylib" if platform.system() == "Darwin" else "so"
    for p in [f"./target/libpithos.{ext}", f"./build-output/libpithos.{ext}", f"./libpithos.{ext}"]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Pithos native library not found.")


def download_sift10k():
    tar_path = "siftsmall.tgz"
    url = "https://github.com/TileDB-Inc/TileDB-Vector-Search/releases/download/0.0.1/siftsmall.tgz"
    
    if not os.path.exists("siftsmall_base.fvecs") and not os.path.exists("siftsmall/siftsmall_base.fvecs"):
        print(f"Downloading SIFT10K dataset from: {url} ...")
        try:
            # Configure headers to look like a browser to prevent blockages
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(tar_path, 'wb') as out_file:
                out_file.write(response.read())
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall()
            os.remove(tar_path)
            print("SIFT10K downloaded and extracted successfully.")
        except Exception as e:
            print(f"[Warning] Failed to download SIFT10K: {e}. Generating synthetic SIFT-like data.")
            os.makedirs("siftsmall", exist_ok=True)
            np.random.seed(42)
            
            centers = np.random.normal(100.0, 50.0, size=(100, 128)).astype(np.float32)
            
            # Base vectors (10000 records)
            with open("siftsmall/siftsmall_base.fvecs", "wb") as f:
                for _ in range(10000):
                    c = centers[np.random.choice(100)]
                    v = c + np.random.normal(0.0, 15.0, size=128).astype(np.float32)
                    f.write(struct.pack('i', 128))
                    f.write(v.tobytes())
            
            # Query vectors (100 records)
            with open("siftsmall/siftsmall_query.fvecs", "wb") as f:
                for _ in range(100):
                    c = centers[np.random.choice(100)]
                    v = c + np.random.normal(0.0, 15.0, size=128).astype(np.float32)
                    f.write(struct.pack('i', 128))
                    f.write(v.tobytes())
            
            # Compute groundtruth
            base_clean = read_fvecs("siftsmall/siftsmall_base.fvecs")
            query_clean = read_fvecs("siftsmall/siftsmall_query.fvecs")
            index = faiss.IndexFlatL2(128)
            index.add(base_clean)
            _, gt_ids = index.search(query_clean, 100)
            
            with open("siftsmall/siftsmall_groundtruth.ivecs", "wb") as f:
                for row in gt_ids:
                    f.write(struct.pack('i', 100))
                    f.write(row.astype(np.int32).tobytes())
            print("Synthetic SIFT-like dataset created.")

def compute_recall_at_k(gt_ids: np.ndarray, pred_ids: np.ndarray, k: int) -> float:
    recalls = []
    for i in range(gt_ids.shape[0]):
        gt_set = set(gt_ids[i, :k].tolist())
        pred_set = set(pred_ids[i, :k].tolist())
        recalls.append(len(gt_set & pred_set) / k)
    return float(np.mean(recalls))

def main():
    download_sift10k()
    
    base_file = "siftsmall_base.fvecs" if os.path.exists("siftsmall_base.fvecs") else "siftsmall/siftsmall_base.fvecs"
    query_file = "siftsmall_query.fvecs" if os.path.exists("siftsmall_query.fvecs") else "siftsmall/siftsmall_query.fvecs"
    gt_file = "siftsmall_groundtruth.ivecs" if os.path.exists("siftsmall_groundtruth.ivecs") else "siftsmall/siftsmall_groundtruth.ivecs"
    
    # Read SIFT files
    base = read_fvecs(base_file)
    queries = read_fvecs(query_file)
    gt = read_ivecs(gt_file)
    
    print(f"SIFT10K Base vectors shape: {base.shape}")
    print(f"SIFT10K Query vectors shape: {queries.shape}")
    print(f"SIFT10K Ground truth shape: {gt.shape}")
    
    n_records = base.shape[0]
    dim = base.shape[1]
    
    # Run FAISS Flat L2 baseline
    index_faiss = faiss.IndexFlatL2(dim)
    index_faiss.add(base)
    
    start_faiss = time.perf_counter()
    faiss_dists, faiss_ids = index_faiss.search(queries, 100)
    end_faiss = time.perf_counter()
    faiss_time_ms = (end_faiss - start_faiss) * 1000.0
    
    # Initialize Pithos
    lib_path = get_lib_path()
    engine = PithosEngine(lib_path)
    
    db_file = "pithos_sift_temp"
    tiers = np.array([64, 128], dtype=np.int32)
    ids = np.arange(n_records, dtype=np.int64)
    
    # Compile Pithos index
    engine.compile_index_file(db_file, 1, 6371000, dim, tiers, ids, base)
    engine.load_index("sift_index", db_file)
    
    # Run Pithos search
    start_pithos = time.perf_counter()
    pithos_ids, pithos_dists = engine.batch_search("sift_index", queries, 100)
    end_pithos = time.perf_counter()
    pithos_time_ms = (end_pithos - start_pithos) * 1000.0
    
    # Compute recall metrics
    recalls = {}
    for k in [1, 10, 50, 100]:
        recalls[f"recall_{k}"] = compute_recall_at_k(gt, pithos_ids, k)
        
    speedup = faiss_time_ms / pithos_time_ms if pithos_time_ms > 0 else 1.0
    
    print("\n" + "=" * 50)
    print("    SIFT10K RECALL RESULTS (Ground Truth: FAISS)")
    print("=" * 50)
    for k in [1, 10, 50, 100]:
        print(f"Recall@{k:02d}: {recalls[f'recall_{k}']*100:.2f}%")
    print(f"FAISS Time:  {faiss_time_ms:.2f} ms")
    print(f"Pithos Time: {pithos_time_ms:.2f} ms")
    print(f"Speedup:     {speedup:.2f}x")
    print("=" * 50)
    
    # Clean up files
    engine.close()
    for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin"]:
        p = db_file + ext
        if os.path.exists(p):
            os.remove(p)
            
    # Export metrics
    sift_metrics = {
        "faiss_time_ms": faiss_time_ms,
        "pithos_time_ms": pithos_time_ms,
        "speedup": speedup,
        "recall_1": recalls["recall_1"],
        "recall_10": recalls["recall_10"],
        "recall_50": recalls["recall_50"],
        "recall_100": recalls["recall_100"]
    }
    with open("sift_metrics.json", "w") as f:
        json.dump(sift_metrics, f, indent=4)
        
if __name__ == "__main__":
    main()
