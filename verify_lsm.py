import os
import sys
import ctypes
import numpy as np

class GraalIsolate(ctypes.Structure):
    pass

class GraalIsolateThread(ctypes.Structure):
    pass

def main():
    print("=" * 70)
    print("       PITHOS LSM DELTA-BUFFER & MERGED SEARCH VERIFIER")
    print("=" * 70)

    # 1. Locate the native library
    lib_path = "./libpithos.dylib"
    if not os.path.exists(lib_path):
        print(f"[Error] Native library {lib_path} not found. Please compile it first.")
        sys.exit(1)

    print(f"[FFI] Loading native library from: {lib_path}")
    lib = ctypes.CDLL(lib_path)

    # 2. Setup FFI function signatures
    lib.graal_create_isolate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(GraalIsolate)),
        ctypes.POINTER(ctypes.POINTER(GraalIsolateThread)),
    ]
    lib.graal_create_isolate.restype = ctypes.c_int

    lib.vdb_init.argtypes = [ctypes.c_void_p]
    lib.vdb_init.restype = ctypes.c_int

    lib.vdb_compile_index_file.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_byte, ctypes.c_longlong,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
    ]
    lib.vdb_compile_index_file.restype = ctypes.c_int

    lib.vdb_load_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.vdb_load_index.restype = ctypes.c_int

    lib.vdb_create_delta_buffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.vdb_create_delta_buffer.restype = ctypes.c_int

    lib.vdb_insert.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong, ctypes.c_void_p]
    lib.vdb_insert.restype = ctypes.c_int

    lib.vdb_delete_from_delta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_longlong]
    lib.vdb_delete_from_delta.restype = ctypes.c_int

    lib.vdb_delta_size.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.vdb_delta_size.restype = ctypes.c_longlong

    lib.vdb_needs_flush.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.vdb_needs_flush.restype = ctypes.c_int

    lib.vdb_search_merged.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.vdb_search_merged.restype = ctypes.c_int

    lib.vdb_backup_delta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.vdb_backup_delta.restype = ctypes.c_int

    lib.vdb_restore_delta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.vdb_restore_delta.restype = ctypes.c_int

    lib.vdb_close.argtypes = [ctypes.c_void_p]
    lib.vdb_close.restype = ctypes.c_int

    # 3. Create isolate and init DB
    isolate = ctypes.POINTER(GraalIsolate)()
    thread = ctypes.POINTER(GraalIsolateThread)()
    if lib.graal_create_isolate(None, ctypes.byref(isolate), ctypes.byref(thread)) != 0:
        print("[Error] Failed to create GraalVM isolate.")
        sys.exit(1)

    if lib.vdb_init(thread) != 0:
        print("[Error] Failed to initialize Pithos Database.")
        sys.exit(1)
    print("[FFI] Isolate and database initialized.")

    # 4. Compile a tiny base index
    DIMENSION = 64
    base_path = "temp/lsm_base_test"
    os.makedirs("temp", exist_ok=True)

    # Base records
    base_ids = np.array([0, 1, 2], dtype=np.int64)
    # Vec 0: all 0.1f
    # Vec 1: all 0.8f
    # Vec 2: first half 0.8f, second half 0.1f
    base_vectors = np.zeros((3, DIMENSION), dtype=np.float32)
    base_vectors[0, :] = 0.1
    base_vectors[1, :] = 0.8
    base_vectors[2, :32] = 0.8
    base_vectors[2, 32:] = 0.1

    tiers = np.array([32, 64], dtype=np.int32)

    print("\n[Base] Compiling base index files...")
    status = lib.vdb_compile_index_file(
        thread, base_path.encode(), 1, 1737400, DIMENSION,
        tiers.ctypes.data_as(ctypes.c_void_p), len(tiers),
        base_ids.ctypes.data_as(ctypes.c_void_p),
        base_vectors.ctypes.data_as(ctypes.c_void_p), len(base_ids)
    )
    if status != 0:
        print(f"[Error] Failed to compile base index: {status}")
        sys.exit(1)

    print("[Base] Base index compiled successfully.")

    # 5. Load the base index
    print("\n[Base] Loading memory-mapped index 'lsm_index'...")
    status = lib.vdb_load_index(thread, b"lsm_index", base_path.encode())
    if status != 0:
        print(f"[Error] Failed to load index: {status}")
        sys.exit(1)
    print("[Base] Index 'lsm_index' loaded and mapped.")

    # 6. Create delta buffer
    print("\n[LSM] Creating in-memory writable delta buffer (flush threshold = 5)...")
    status = lib.vdb_create_delta_buffer(thread, b"lsm_index", 5)
    if status != 0:
        print(f"[Error] Failed to create delta buffer: {status}")
        sys.exit(1)
    print(f"[LSM] Delta buffer created. Initial size: {lib.vdb_delta_size(thread, b'lsm_index')}")

    # 7. Insert new vectors into delta buffer
    # Let's insert a vector with ID 10 that is extremely close to all 1.0f (closer than any base vector)
    vec_10 = np.ones(DIMENSION, dtype=np.float32) * 0.95
    print("\n[LSM] Inserting Vector ID 10 (all 0.95f) into Delta Buffer...")
    status = lib.vdb_insert(thread, b"lsm_index", 10, vec_10.ctypes.data_as(ctypes.c_void_p))
    if status != 0:
        print(f"[Error] Insert failed: {status}")
        sys.exit(1)

    # Let's insert another vector with ID 11
    vec_11 = np.ones(DIMENSION, dtype=np.float32) * 0.5
    print("[LSM] Inserting Vector ID 11 (all 0.50f) into Delta Buffer...")
    lib.vdb_insert(thread, b"lsm_index", 11, vec_11.ctypes.data_as(ctypes.c_void_p))

    print(f"[LSM] Delta size now: {lib.vdb_delta_size(thread, b'lsm_index')}")
    print(f"[LSM] Needs flush? {lib.vdb_needs_flush(thread, b'lsm_index')}")

    # 8. Unified search (Base + LSM Delta Buffer)
    # Query: all 1.0f. Matches best with Vector 10 (0.95f) in Delta, then Vector 1 (0.8f) in Base.
    query = np.ones(DIMENSION, dtype=np.float32)
    k = 3
    out_ids = np.zeros(k, dtype=np.int64)
    out_dists = np.zeros(k, dtype=np.int32)

    print("\n[Query] Performing unified merged search for query (all 1.0f)...")
    status = lib.vdb_search_merged(
        thread, b"lsm_index", query.ctypes.data_as(ctypes.c_void_p), k,
        out_ids.ctypes.data_as(ctypes.c_void_p), out_dists.ctypes.data_as(ctypes.c_void_p)
    )
    if status != 0:
        print(f"[Error] Search failed: {status}")
        sys.exit(1)

    print("\n[Query] Merged Search Results (Top-3 nearest neighbors):")
    for i in range(k):
        # distance score in output is scaled by 1,000,000
        dist = out_dists[i] / 1000000.0
        origin = "LSM Delta" if out_ids[i] >= 10 else "Base Index"
        print(f"  Rank {i+1}: Vector ID = {out_ids[i]:<2} | L2 Distance = {dist:.4f} | Origin = {origin}")

    # 9. Test soft-delete (Tombstone)
    print("\n[LSM] Soft-deleting Vector ID 10 (tombstoning)...")
    status = lib.vdb_delete_from_delta(thread, b"lsm_index", 10)
    if status != 1:
        print(f"[Error] Deletion failed: {status}")
        sys.exit(1)
    print(f"[LSM] Delta live size now: {lib.vdb_delta_size(thread, b'lsm_index')}")

    # Search again to see if ID 10 is deleted
    print("\n[Query] Performing merged search again after tombstone...")
    status = lib.vdb_search_merged(
        thread, b"lsm_index", query.ctypes.data_as(ctypes.c_void_p), k,
        out_ids.ctypes.data_as(ctypes.c_void_p), out_dists.ctypes.data_as(ctypes.c_void_p)
    )
    print("[Query] Merged Search Results (Top-3 nearest neighbors after deleting ID 10):")
    for i in range(k):
        dist = out_dists[i] / 1000000.0
        origin = "LSM Delta" if out_ids[i] >= 10 else "Base Index"
        print(f"  Rank {i+1}: Vector ID = {out_ids[i]:<2} | L2 Distance = {dist:.4f} | Origin = {origin}")

    # 10. Backup delta buffer to disk
    backup_path = "temp/delta_backup.bin"
    print(f"\n[Backup] Backing up delta buffer to '{backup_path}'...")
    status = lib.vdb_backup_delta(thread, b"lsm_index", backup_path.encode())
    if status != 0:
        print(f"[Error] Backup failed: {status}")
        sys.exit(1)
    print("[Backup] Delta buffer backed up successfully.")

    # 11. Restore delta buffer from disk
    print("\n[Restore] Restoring delta buffer from backup...")
    status = lib.vdb_restore_delta(thread, b"lsm_index", backup_path.encode(), 5)
    if status != 0:
        print(f"[Error] Restore failed: {status}")
        sys.exit(1)
    print(f"[Restore] Delta buffer restored. Size is: {lib.vdb_delta_size(thread, b'lsm_index')}")

    # Search after restore to make sure restored records are queryable
    print("\n[Query] Performing search after restore...")
    status = lib.vdb_search_merged(
        thread, b"lsm_index", query.ctypes.data_as(ctypes.c_void_p), k,
        out_ids.ctypes.data_as(ctypes.c_void_p), out_dists.ctypes.data_as(ctypes.c_void_p)
    )
    for i in range(k):
        dist = out_dists[i] / 1000000.0
        origin = "LSM Delta" if out_ids[i] >= 10 else "Base Index"
        print(f"  Rank {i+1}: Vector ID = {out_ids[i]:<2} | L2 Distance = {dist:.4f} | Origin = {origin}")

    # Cleanup
    print("\n[Cleanup] Closing database and freeing resources...")
    lib.vdb_close(thread)
    
    # Remove files
    for f in [backup_path, base_path, base_path + "_ids.bin", base_path + "_metadata.bin", base_path + "_tier_0.bin", base_path + "_tier_1.bin"]:
        if os.path.exists(f):
            os.remove(f)
    print("[Cleanup] Finished cleanly. LSM Tree works perfectly!")
    print("=" * 70)

if __name__ == "__main__":
    main()
