import os
import sys
import ctypes
import numpy as np

# ==============================================================================
# Pithos PySpark Integration Wrapper
# This script demonstrates how to execute the high-performance AOT-compiled
# vector search kernel inside Apache Spark executors on an NVIDIA DGX (Linux).
# ==============================================================================

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

class PithosEngine:
    """Python ctypes Wrapper to load liblunar_core.so within the Spark executor."""
    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Native shared library not found at: {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        self.isolate = ctypes.POINTER(GraalIsolate)()
        self.thread = ctypes.POINTER(GraalIsolateThread)()
        
        # C-API signatures matching org.pithos.CApi
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

        self.lib.vdb_load_index_with_weights.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int
        ]
        self.lib.vdb_load_index_with_weights.restype = ctypes.c_int

        self.lib.vdb_query_planetary_grid.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        ]
        self.lib.vdb_query_planetary_grid.restype = ctypes.c_longlong

        self.lib.vdb_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.vdb_size.restype = ctypes.c_longlong

        self.lib.vdb_close.argtypes = [ctypes.c_void_p]
        self.lib.vdb_close.restype = ctypes.c_int

        # Instantiate GraalVM isolate and database coordinator
        status = self.lib.graal_create_isolate(None, ctypes.byref(self.isolate), ctypes.byref(self.thread))
        if status != 0:
            raise RuntimeError("Failed to allocate GraalVM isolate thread.")
            
        status = self.lib.vdb_init(self.thread)
        if status != 0:
            raise RuntimeError("Failed to initialize Pithos DB engine.")

    def load_index(self, index_name: str, file_path: str, weights: np.ndarray = None, lora_dim: int = 0):
        name_bytes = index_name.encode("utf-8")
        path_bytes = file_path.encode("utf-8")
        if weights is not None:
            weights_ptr = weights.ctypes.data_as(ctypes.c_void_p)
            status = self.lib.vdb_load_index_with_weights(self.thread, name_bytes, path_bytes, weights_ptr, lora_dim)
        else:
            status = self.lib.vdb_load_index(self.thread, name_bytes, path_bytes)
            
        if status != 0:
            raise RuntimeError(f"Failed to memory-map index file {file_path}. Code: {status}")

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

    def size(self, index_name: str) -> int:
        c_index = index_name.encode("utf-8")
        return self.lib.vdb_size(self.thread, c_index)

    def close(self):
        if self.thread:
            self.lib.vdb_close(self.thread)
            self.lib.graal_tear_down_isolate(self.thread)
            self.thread = None
            self.isolate = None


def process_partition_with_pithos(partition_data):
    """
    Spark MapPartitions Function: Executed on each cluster worker thread.
    """
    lib_name = "libpithos.so"
    if not os.path.exists(lib_name):
        lib_name = "./build-output/libpithos.so"
        
    if not os.path.exists(lib_name):
        raise FileNotFoundError(f"Native shared library {lib_name} not found in worker directory.")
    
    engine = PithosEngine(lib_name)
    results = []
    
    try:
        for row in partition_data:
            partition_id = row["partition_id"]
            file_path = row["file_path"]
            
            # Load with mock weights (or dynamic weights)
            index_name = f"idx_{partition_id}"
            dimension = 384
            mock_weights = np.eye(dimension, dtype=np.float32)
            engine.load_index(index_name, file_path, mock_weights, dimension)
            total_records = engine.size(index_name)
            
            # Setup float queries (278 target queries)
            np.random.seed(42)
            CAVE_VECTOR = np.random.uniform(-1.0, 1.0, size=dimension).astype(np.float32)
            voting_queries = np.tile(CAVE_VECTOR, (8, 1)) # shape (8, 384)
            families = np.arange(8, dtype=np.int32)
            thresholds = np.zeros(8, dtype=np.int32) # Hamming tolerance
            
            voting_mask = np.zeros(total_records, dtype=np.uint8)
            
            # Scan!
            resonant_count = engine.query_planetary_grid(index_name, voting_queries, families, thresholds, voting_mask)
            resonant_indices = np.flatnonzero(voting_mask >= 7)
            
            for idx in resonant_indices:
                results.append({
                    "partition_id": partition_id,
                    "local_offset": int(idx),
                    "resonance_score": int(voting_mask[idx])
                })
                
    finally:
        engine.close()
        
    return iter(results)


def main():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("Pithos-Planetary-Scale-Search") \
        .getOrCreate()
        
    print("Spark Session initialized. Preparing partitioned grid...")
    
    grid_partitions = [
        {"partition_id": i, "file_path": f"/mnt/nvme/lunar_grid_part_{i}"}
        for i in range(64)
    ]
    
    dist_grid = spark.sparkContext.parallelize(grid_partitions, numSlices=64)
    results_rdd = dist_grid.mapPartitions(process_partition_with_pithos)
    
    print("Scanning grid on executors via Pithos AOT kernel...")
    resonant_candidates = results_rdd.collect()
    
    print(f"\nScan complete! Found {len(resonant_candidates)} resonant lunar target locations:")
    for cand in resonant_candidates[:20]:
        print(f" -> Part {cand['partition_id']}: Offset {cand['local_offset']} | Resonance: {cand['resonance_score']}/8")
        
    spark.stop()

if __name__ == "__main__":
    if "SPARK_ENV_LOADED" not in os.environ:
        print("Local test mode: Initializing PithosEngine standalone wrapper...")
        try:
            import platform
            if platform.system() == "Darwin":
                so_paths = [
                    "./target/libpithos.dylib",
                    "./build-output/libpithos.dylib",
                    "./libpithos.dylib",
                    "./target/libpithos.so",
                    "./build-output/libpithos.so",
                ]
            else:
                so_paths = [
                    "./build-output/libpithos.so",
                    "./libpithos.so",
                    "./target/libpithos.so",
                ]
            
            lib_path = None
            for p in so_paths:
                if os.path.exists(p):
                    try:
                        ctypes.CDLL(p)
                        lib_path = p
                        break
                    except Exception:
                        continue
            
            if lib_path:
                engine = PithosEngine(lib_path)
                print(f"Success! Native library {lib_path} successfully wrapped.")
                engine.close()
            else:
                print(f"Could not run standalone test: library not found.")
        except Exception as e:
            print(f"Error during wrapper initialization: {e}")
    else:
        main()
