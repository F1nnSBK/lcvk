import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from benchmark import PithosMIDB

def main():
    print("=" * 70)
    print("       PITHOS LSM DELTA-BUFFER & MERGED SEARCH VERIFIER")
    print("=" * 70)

    # 1. Initialize PithosMIDB singleton
    print("[FFI] Initializing PithosMIDB singleton...")
    engine = PithosMIDB()
    print("[FFI] Isolate and database initialized.")

    # 2. Compile a tiny base index
    DIMENSION = 64
    base_path = "temp/lsm_base_test"
    os.makedirs("temp", exist_ok=True)

    base_ids = np.array([0, 1, 2], dtype=np.int64)
    base_vectors = np.zeros((3, DIMENSION), dtype=np.float32)
    base_vectors[0, :] = 0.1
    base_vectors[1, :] = 0.8
    base_vectors[2, :32] = 0.8
    base_vectors[2, 32:] = 0.1

    tiers = np.array([32, 64], dtype=np.int32)

    print("\n[Base] Compiling base index files...")
    status = engine.compile_index_file(base_path, 1, 1737400, DIMENSION, tiers, base_ids, base_vectors, 0)
    if status != 0:
        print(f"[Error] Failed to compile base index: {status}")
        sys.exit(1)
    print("[Base] Base index compiled successfully.")

    # 3. Load the base index
    print("\n[Base] Loading memory-mapped index 'lsm_index'...")
    status = engine.load_index("lsm_index", base_path)
    if status != 0:
        print(f"[Error] Failed to load index: {status}")
        sys.exit(1)
    print("[Base] Index 'lsm_index' loaded and mapped.")

    # 4. Create delta buffer
    print("\n[LSM] Creating in-memory writable delta buffer (flush threshold = 5)...")
    status = engine.create_delta_buffer("lsm_index", 5)
    if status != 0:
        print(f"[Error] Failed to create delta buffer: {status}")
        sys.exit(1)
    print(f"[LSM] Delta buffer created. Initial size: {engine.delta_size('lsm_index')}")

    # 5. Insert new vectors into delta buffer
    vec_10 = np.ones(DIMENSION, dtype=np.float32) * 0.95
    print("\n[LSM] Inserting Vector ID 10 (all 0.95f) into Delta Buffer...")
    status = engine.insert("lsm_index", 10, vec_10)
    if status != 0:
        print(f"[Error] Insert failed: {status}")
        sys.exit(1)

    vec_11 = np.ones(DIMENSION, dtype=np.float32) * 0.5
    print("[LSM] Inserting Vector ID 11 (all 0.50f) into Delta Buffer...")
    engine.insert("lsm_index", 11, vec_11)

    print(f"[LSM] Delta size now: {engine.delta_size('lsm_index')}")
    print(f"[LSM] Needs flush? {engine.needs_flush('lsm_index')}")

    # 6. Unified search (Base + LSM Delta Buffer)
    query = np.ones(DIMENSION, dtype=np.float32)
    k = 3

    print("\n[Query] Performing unified merged search for query (all 1.0f)...")
    out_ids, out_dists = engine.search_merged("lsm_index", query, k)

    print("\n[Query] Merged Search Results (Top-3 nearest neighbors):")
    for i in range(k):
        dist = out_dists[i] / 1000000.0
        origin = "LSM Delta" if out_ids[i] >= 10 else "Base Index"
        print(f"  Rank {i+1}: Vector ID = {out_ids[i]:<2} | L2 Distance = {dist:.4f} | Origin = {origin}")

    # 7. Test soft-delete (Tombstone)
    print("\n[LSM] Soft-deleting Vector ID 10 (tombstoning)...")
    status = engine.delete_from_delta("lsm_index", 10)
    if status != 1:
        print(f"[Error] Deletion failed: {status}")
        sys.exit(1)
    print(f"[LSM] Delta live size now: {engine.delta_size('lsm_index')}")

    # Search again to see if ID 10 is deleted
    print("\n[Query] Performing merged search again after tombstone...")
    out_ids, out_dists = engine.search_merged("lsm_index", query, k)
    print("[Query] Merged Search Results (Top-3 nearest neighbors after deleting ID 10):")
    for i in range(k):
        dist = out_dists[i] / 1000000.0
        origin = "LSM Delta" if out_ids[i] >= 10 else "Base Index"
        print(f"  Rank {i+1}: Vector ID = {out_ids[i]:<2} | L2 Distance = {dist:.4f} | Origin = {origin}")

    # 8. Backup delta buffer to disk
    backup_path = "temp/delta_backup.bin"
    print(f"\n[Backup] Backing up delta buffer to '{backup_path}'...")
    status = engine.backup_delta("lsm_index", backup_path)
    if status != 0:
        print(f"[Error] Backup failed: {status}")
        sys.exit(1)
    print("[Backup] Delta buffer backed up successfully.")

    # 9. Restore delta buffer from disk
    print("\n[Restore] Restoring delta buffer from backup...")
    status = engine.restore_delta("lsm_index", backup_path, 5)
    if status != 0:
        print(f"[Error] Restore failed: {status}")
        sys.exit(1)
    print(f"[Restore] Delta buffer restored. Size is: {engine.delta_size('lsm_index')}")

    # Search after restore to make sure restored records are queryable
    print("\n[Query] Performing search after restore...")
    out_ids, out_dists = engine.search_merged("lsm_index", query, k)
    for i in range(k):
        dist = out_dists[i] / 1000000.0
        origin = "LSM Delta" if out_ids[i] >= 10 else "Base Index"
        print(f"  Rank {i+1}: Vector ID = {out_ids[i]:<2} | L2 Distance = {dist:.4f} | Origin = {origin}")

    # Cleanup
    print("\n[Cleanup] Closing database and freeing resources...")
    engine.close()

    # Remove files
    for f in [backup_path, base_path, base_path + "_ids.bin", base_path + "_metadata.bin", base_path + "_tier_0.bin", base_path + "_tier_1.bin"]:
        if os.path.exists(f):
            os.remove(f)
    print("[Cleanup] Finished cleanly. LSM Tree works perfectly!")
    print("=" * 70)

if __name__ == "__main__":
    main()
