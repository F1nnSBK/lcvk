import os
import sys
import ctypes
import numpy as np

# ==============================================================================
# LCVK PySpark Integration Wrapper
# This script demonstrates how to execute the high-performance AOT-compiled
# vector search kernel inside Apache Spark executors on an NVIDIA DGX (Linux).
# ==============================================================================

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

class LcvkEngine:
    """Python ctypes Wrapper to load liblunar_core.so within the Spark executor."""
    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Native shared library not found at: {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        self.isolate = ctypes.POINTER(GraalIsolate)()
        self.thread = ctypes.POINTER(GraalIsolateThread)()
        
        # C-API signatures matching org.lcvk.vectordb.CApi
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
            raise RuntimeError("Failed to initialize LCVK DB engine.")

    def load_index(self, index_name: str, file_path: str):
        name_bytes = index_name.encode("utf-8")
        path_bytes = file_path.encode("utf-8")
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


def process_partition_with_lcvk(partition_data):
    """
    Spark MapPartitions Function: Executed on each cluster worker thread.
    Each partition of the RDD represents a segment of the overall grid to scan.
    """
    # 1. Resolve liblunar_core.so path (Spark distributes files to the worker's working directory)
    lib_name = "liblunar_core.so"
    if not os.path.exists(lib_name):
        # Fallback to local build path if running in standalone test mode
        lib_name = "./build-output/liblunar_core.so"
        
    if not os.path.exists(lib_name):
        raise FileNotFoundError(f"Native shared library {lib_name} not found in worker directory.")
    
    # 2. Instantiate isolated native LCVK engine for this partition thread
    engine = LcvkEngine(lib_name)
    
    results = []
    
    try:
        # partition_data will yield file paths or partition configurations
        for row in partition_data:
            partition_id = row["partition_id"]
            file_path = row["file_path"] # Fast storage path (e.g. lustre mount, GPUDirect mount)
            
            # Map index zero-copy off-heap
            index_name = f"idx_{partition_id}"
            engine.load_index(index_name, file_path)
            total_records = engine.size(index_name)
            
            # Setup queries (278 target queries matching LCVK specifications)
            # In a real environment, broadcast variables should be used for queries
            np.random.seed(42)
            CAVE_VECTOR = np.random.randint(-2**63, 2**63 - 1, size=6, dtype=np.int64)
            voting_queries = np.tile(CAVE_VECTOR, (8, 1)) # shape (8, 6)
            families = np.arange(8, dtype=np.int32)
            thresholds = np.zeros(8, dtype=np.int32) # Hamming tolerance
            
            # Pre-allocate zero-copy target voting buffer
            voting_mask = np.zeros(total_records, dtype=np.uint8)
            
            # Scan!
            resonant_count = engine.query_planetary_grid(index_name, voting_queries, families, thresholds, voting_mask)
            
            # Find indices with high resonance (e.g. fully resonant target matching >= 7 families)
            resonant_indices = np.flatnonzero(voting_mask >= 7)
            
            for idx in resonant_indices:
                global_index_id = int(idx) # Local offset can be translated to global coordinate
                score = int(voting_mask[idx])
                results.append({
                    "partition_id": partition_id,
                    "local_offset": global_index_id,
                    "resonance_score": score
                })
                
    finally:
        engine.close()
        
    return iter(results)


def main():
    # Setup Spark Session (configure executor options for DGX high throughput)
    # The config options `--enable-native-access=ALL-UNNAMED` must be passed via spark-submit
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("Lunar-LCVK-Planetary-Scale-Search") \
        .getOrCreate()
        
    print("Spark Session initialized. Preparing partitioned grid...")
    
    # 1. Define files for the planetary grid partitions (e.g. 148 GB split into 64 partitions of 2.3 GB)
    # On a DGX, these paths point to highly parallel NVMe storage mounts (Lustre/GPUDirect/Weka)
    grid_partitions = [
        {"partition_id": i, "file_path": f"/mnt/nvme/lunar_grid_part_{i}.bin"}
        for i in range(64)
    ]
    
    # 2. Parallelize partition config across Spark cluster
    dist_grid = spark.sparkContext.parallelize(grid_partitions, numSlices=64)
    
    # 3. Execute the LCVK scan using mapPartitions (Map phase)
    # Spark sends the job to workers, which execute compiled C kernel on the raw data
    results_rdd = dist_grid.mapPartitions(process_partition_with_lcvk)
    
    # 4. Collect results (Reduce/Aggregate phase)
    print("Scanning grid on executors via LCVK AOT kernel...")
    resonant_candidates = results_rdd.collect()
    
    print(f"\nScan complete! Found {len(resonant_candidates)} resonant lunar target locations:")
    for cand in resonant_candidates[:20]:
        print(f" -> Part {cand['partition_id']}: Offset {cand['local_offset']} | Resonance: {cand['resonance_score']}/8")
        
    spark.stop()

if __name__ == "__main__":
    # If run locally without spark-submit, test basic wrapper initialization
    if "SPARK_ENV_LOADED" not in os.environ:
        print("Local test mode: Initializing LcvkEngine standalone wrapper...")
        try:
            import platform
            if platform.system() == "Darwin":
                so_paths = [
                    "./target/lunar_core.dylib",
                    "./build-output/liblunar_core.dylib",
                    "./liblunar_core.dylib",
                    "./target/lunar_core.so",
                    "./build-output/liblunar_core.so",
                ]
            else:
                so_paths = [
                    "./build-output/liblunar_core.so",
                    "./liblunar_core.so",
                    "./target/lunar_core.so",
                ]
            
            lib_path = None
            for p in so_paths:
                if os.path.exists(p):
                    try:
                        # Try loading it; skip if invalid format for current OS
                        ctypes.CDLL(p)
                        lib_path = p
                        break
                    except Exception:
                        continue
            
            if lib_path:
                engine = LcvkEngine(lib_path)
                print(f"Success! Native library {lib_path} successfully wrapped.")
                engine.close()
            else:
                print(f"Could not run standalone test: compatible library not found. Build locally or run ./build.sh.")
        except Exception as e:
            print(f"Error during wrapper initialization: {e}")
    else:
        main()
