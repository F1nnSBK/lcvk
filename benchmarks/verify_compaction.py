import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark import PithosMIDB, generate_hypersphere_vectors

def main():
    print("[Python Verify Compaction] Starting compaction verification...")
    
    engine = PithosMIDB()
    
    dimension = 64
    tiers = np.array([32, 64], dtype=np.int32)
    
    vectors1 = generate_hypersphere_vectors(50, dimension)
    ids1 = np.arange(50, dtype=np.int64)
    db1_path = "temp/db_comp_1"
    
    print(f"[Python Verify Compaction] Compiling {db1_path}...")
    status = engine.compile_index_file(db1_path, 1, 1000, dimension, tiers, ids1, vectors1)
    if status != 0:
        raise RuntimeError(f"Failed to compile {db1_path}, code: {status}")
        
    vectors2 = generate_hypersphere_vectors(50, dimension)
    ids2 = np.arange(50, 100, dtype=np.int64)
    db2_path = "temp/db_comp_2"
    
    print(f"[Python Verify Compaction] Compiling {db2_path}...")
    status = engine.compile_index_file(db2_path, 1, 1000, dimension, tiers, ids2, vectors2)
    if status != 0:
        raise RuntimeError(f"Failed to compile {db2_path}, code: {status}")
        
    db_compacted_path = "temp/db_compacted"
    print(f"[Python Verify Compaction] Compacting {db1_path} and {db2_path} into {db_compacted_path}...")
    status = engine.compact_indexes([db1_path, db2_path], db_compacted_path)
    if status != 0:
        raise RuntimeError(f"Compaction failed, code: {status}")
        
    print(f"[Python Verify Compaction] Loading compacted index...")
    status = engine.load_index("compacted_test", db_compacted_path)
    if status != 0:
        raise RuntimeError(f"Failed to load compacted index, code: {status}")
        
    info = engine.get_info("compacted_test")
    print(f"[Python Verify Compaction] Compacted Index Info: {info}")
    
    assert info["size"] == 100, f"Expected size 100, got {info['size']}"
    assert info["dimension"] == dimension, f"Expected dimension {dimension}, got {info['dimension']}"
    assert info["tiers_count"] == len(tiers), f"Expected tiers count {len(tiers)}, got {info['tiers_count']}"
    
    query = generate_hypersphere_vectors(1, dimension)
    ids, dists = engine.batch_search("compacted_test", query, 5)
    print(f"[Python Verify Compaction] KNN Search Results:\nIDs: {ids}\nDists: {dists}")
    assert ids.shape == (1, 5), f"Expected shape (1, 5), got {ids.shape}"
    
    engine.close()
    for base in [db1_path, db2_path, db_compacted_path]:
        for ext in ["", "_ids.bin", "_metadata.bin", "_tier_0.bin", "_tier_1.bin", "_fp16.bin"]:
            p = base + ext
            if os.path.exists(p):
                os.remove(p)
                
    print("[Python Verify Compaction] SUCCESS!")

if __name__ == "__main__":
    main()
